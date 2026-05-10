import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def find_origin_candle(df, start_idx, target_price, col='High', lookback=100):
    start_pos = df.index.get_loc(start_idx)
    end_pos = max(0, start_pos - lookback)
    sub_df = df.iloc[end_pos:start_pos+1]
    matches = sub_df[np.isclose(sub_df[col], target_price, atol=1e-5)]
    if len(matches) > 0:
        return matches.iloc[-1]
    return None

def main():
    print("Loading SMC dataset...")
    df = pd.read_csv('processed_smc_data.csv', index_col='Datetime')
    df.index = pd.to_datetime(df.index, utc=True)
    
    df_clean = df.dropna().copy()
    features = [
        'EMA_50', 'EMA_200', 'RSI_14', 'ATRr_14', 'MACDh_12_26_9', 'Distance_to_EMA50',
        'SMC_State', 'Dist_to_Must_Break', 'Dist_to_Must_Crash', 'SMC_In_Penalty_Box',
        'CHoCH_Extension_Distance', 'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos', 
        'Minute_Sin', 'Minute_Cos', 'Market_Volatility_Regime'
    ]
    X = df_clean[features]
    y = df_clean['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
    
    print("Training ML Model with tuned parameters...")
    # Using params typically found in Step 4
    best_params = {'n_estimators': 500, 'min_samples_split': 10, 'max_depth': 5}
    model = RandomForestClassifier(random_state=42, **best_params)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)

    df_test = df.loc[X_test.index].copy()
    df_test['Prob_0'] = probs[:, 0]
    df_test['Prob_1'] = probs[:, 1]

    print("Running Backtest with Dynamic Threshold and Limit Orders...")
    balance = 10000.0
    risk_pct = 0.01
    
    position = 0 
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    units = 0.0
    
    pending_order = 0
    pend_entry = 0.0
    pend_sl = 0.0
    pend_tp = 0.0
    
    equity_curve = []
    total_trades = 0
    winning_trades = 0
    gross_profit = 0.0
    gross_loss = 0.0
    
    last_event = 0
    buffer = 0.0005 # 5 pips
    
    # Dynamic Threshold tracking
    recent_outcomes = []
    base_threshold = 0.60
    
    for idx, row in df_test.iterrows():
        # Update last event
        if row['SMC_Event'] != 0:
            # Cancel pending orders on new structural events
            pending_order = 0
            
            # Dynamic Threshold Calculation
            dynamic_threshold = base_threshold
            if len(recent_outcomes) >= 10:
                recent_win_rate = recent_outcomes.count('Win') / len(recent_outcomes)
                if recent_win_rate < 0.40:
                    dynamic_threshold = min(0.95, base_threshold + 0.10)
            
            # Evaluate new setups
            if not row['SMC_In_Penalty_Box']:
                # LONG SETUP: CHoCH Down (-2) following BOS Up (1)
                if row['SMC_Event'] == -2 and last_event == 1 and row['Prob_1'] > dynamic_threshold:
                        origin_low_candle = find_origin_candle(df, idx, row['SMC_Must_Crash'], col='Low')
                        origin_high_candle = find_origin_candle(df, idx, row['SMC_Must_Break'], col='High')
                        if origin_low_candle is not None and origin_high_candle is not None:
                            pend_entry = origin_low_candle['High']
                            pend_tp = origin_high_candle['Low']
                            pend_sl = row['SMC_Must_Crash'] - buffer
                            if pend_tp > pend_entry > pend_sl:
                                pending_order = 1
                                
                # SHORT SETUP: CHoCH Up (2) following BOS Down (-1)
                elif row['SMC_Event'] == 2 and last_event == -1 and row['Prob_0'] > dynamic_threshold:
                        origin_high_candle = find_origin_candle(df, idx, row['SMC_Must_Break'], col='High')
                        origin_low_candle = find_origin_candle(df, idx, row['SMC_Must_Crash'], col='Low')
                        if origin_high_candle is not None and origin_low_candle is not None:
                            pend_entry = origin_high_candle['Low']
                            pend_tp = origin_low_candle['High']
                            pend_sl = row['SMC_Must_Break'] + buffer
                            if pend_tp < pend_entry < pend_sl:
                                pending_order = -1
            
            last_event = row['SMC_Event']
            
        # Process open positions
        trade_closed = False
        pnl = 0.0
        
        if position == 1:
            if row['Low'] <= sl_price:
                pnl = (sl_price - entry_price) * units
                trade_closed = True
            elif row['High'] >= tp_price:
                pnl = (tp_price - entry_price) * units
                trade_closed = True
                
        elif position == -1:
            if row['High'] >= sl_price:
                pnl = (entry_price - sl_price) * units
                trade_closed = True
            elif row['Low'] <= tp_price:
                pnl = (entry_price - tp_price) * units
                trade_closed = True
                
        if trade_closed:
            balance += pnl
            position = 0
            total_trades += 1
            if pnl > 0:
                winning_trades += 1
                gross_profit += pnl
                recent_outcomes.append('Win')
            else:
                gross_loss += abs(pnl)
                recent_outcomes.append('Loss')
                
            if len(recent_outcomes) > 10:
                recent_outcomes.pop(0)
                
        # Process pending limit orders
        if position == 0 and pending_order != 0:
            filled = False
            if pending_order == 1 and row['Low'] <= pend_entry:
                filled = True
                position = 1
            elif pending_order == -1 and row['High'] >= pend_entry:
                filled = True
                position = -1
                
            if filled:
                entry_price = pend_entry
                sl_price = pend_sl
                tp_price = pend_tp
                
                risk_amount = balance * risk_pct
                sl_dist = abs(entry_price - sl_price)
                units = risk_amount / sl_dist if sl_dist > 0 else 0
                pending_order = 0
                
        equity_curve.append(balance)

    df_test['Equity'] = equity_curve

    # Metrics Calculations
    net_profit = balance - 10000.0
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    print("\n" + "=" * 40)
    print(" " * 6 + "FINAL BACKTEST RESULTS (LIMIT ORDERS & ML)")
    print("=" * 40)
    print(f"Starting Capital:    $10000.00")
    print(f"Total Net Profit:    ${net_profit:.2f}")
    print(f"Final Balance:       ${balance:.2f}")
    print("-" * 40)
    print(f"Total Trades:        {total_trades}")
    print(f"Win Rate:            {win_rate:.2f}%")
    print(f"Profit Factor:       {profit_factor:.2f}")
    print("=" * 40 + "\n")

    print("Saving Equity Curve plot...")
    plt.figure(figsize=(12, 6))
    plt.plot(df_test.index, df_test['Equity'], label="Account Equity", color="gold", linewidth=1.5)
    plt.title("Step 8: Final SMC Optimized Equity Curve\n(Amu Khani Limit Logic + ML Filter + No Mondays)", fontsize=14, pad=15)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Account Balance (USD)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=10000, color='r', linestyle='-', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_filename = 'equity_curve_smc_optimized.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved successfully as '{plot_filename}'.")

if __name__ == "__main__":
    main()
