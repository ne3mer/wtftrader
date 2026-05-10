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
    
    # Train exclusively on valid setups
    df_signals = df.dropna(subset=['Target']).copy()
    
    features = [
        'EMA_50', 'EMA_200', 'RSI_14', 'ATRr_14', 'MACDh_12_26_9', 'Distance_to_EMA50',
        'SMC_State', 'Dist_to_Must_Break', 'Dist_to_Must_Crash', 'SMC_In_Penalty_Box',
        'CHoCH_Extension_Distance', 'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos', 
        'Minute_Sin', 'Minute_Cos', 'Market_Volatility_Regime', 'Is_Toxic_Window'
    ]
    X = df_signals[features]
    y = df_signals['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
    
    print("Training ML Model with tuned parameters...")
    # Using params typically found in Step 4
    best_params = {'n_estimators': 500, 'min_samples_split': 10, 'max_depth': 5}
    model = RandomForestClassifier(random_state=42, **best_params)
    model.fit(X_train, y_train)

    print("Running Backtest with ML > 50% Limit Orders...")
    
    # We must run backtest on the full timeline (including non-signals) that corresponds to the test period
    split_idx = X_test.index[0]
    df_test_full = df.loc[split_idx:].copy()
    
    # Predict probabilities for the entire test set
    probs = model.predict_proba(df_test_full[features])
    df_test_full['Prob_1'] = probs[:, 1] # Probability of Setup WIN

    balance = 10000.0
    risk_pct = 0.01
    
    position = 0 
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    units = 0.0
    
    pending_setup = 0
    extreme_level = 0.0
    structural_tp = 0.0
    swept = False
    sweep_extreme = 0.0
    last_event = 0
    
    equity_curve = []
    total_trades = 0
    winning_trades = 0
    gross_profit = 0.0
    gross_loss = 0.0
    last_event = 0
    
    for idx, row in df_test_full.iterrows():
        # Update last event
        if row['SMC_Event'] != 0:
            # Cancel pending setups on new structural events
            pending_setup = 0
            
            # Evaluate new setups
            if not row['SMC_In_Penalty_Box']:
                req_prob = 0.75 if row['Is_Toxic_Window'] == 1 else 0.50
                
                # LONG SETUP
                if row['SMC_Event'] == -2 and last_event == 1 and row['Prob_1'] > req_prob:
                    pending_setup = 1
                    extreme_level = row['SMC_Must_Crash']
                    structural_tp = row['SMC_Must_Break']
                    swept = False
                    sweep_extreme = float('inf')
                                
                # SHORT SETUP
                elif row['SMC_Event'] == 2 and last_event == -1 and row['Prob_1'] > req_prob:
                    pending_setup = -1
                    extreme_level = row['SMC_Must_Break']
                    structural_tp = row['SMC_Must_Crash']
                    swept = False
                    sweep_extreme = float('-inf')
            
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
            else:
                gross_loss += abs(pnl)
                
        # Process pending setups
        if position == 0 and pending_setup != 0:
            if pending_setup == 1:
                if not swept:
                    if row['Low'] < extreme_level:
                        swept = True
                        sweep_extreme = min(sweep_extreme, row['Low'])
                else:
                    sweep_extreme = min(sweep_extreme, row['Low'])
                    if row['Close'] > extreme_level:
                        # Filled!
                        position = 1
                        entry_price = row['Close']
                        sl_price = sweep_extreme - (1.0 * row['ATRr_14'])
                        tp_price = structural_tp
                        
                        risk_amount = balance * risk_pct
                        sl_dist = abs(entry_price - sl_price)
                        units = risk_amount / sl_dist if sl_dist > 0 else 0
                        pending_setup = 0
                        
            elif pending_setup == -1:
                if not swept:
                    if row['High'] > extreme_level:
                        swept = True
                        sweep_extreme = max(sweep_extreme, row['High'])
                else:
                    sweep_extreme = max(sweep_extreme, row['High'])
                    if row['Close'] < extreme_level:
                        # Filled!
                        position = -1
                        entry_price = row['Close']
                        sl_price = sweep_extreme + (1.0 * row['ATRr_14'])
                        tp_price = structural_tp
                        
                        risk_amount = balance * risk_pct
                        sl_dist = abs(entry_price - sl_price)
                        units = risk_amount / sl_dist if sl_dist > 0 else 0
                        pending_setup = 0
                
        equity_curve.append(balance)

    df_test_full['Equity'] = equity_curve

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
    plt.plot(df_test_full.index, df_test_full['Equity'], label="Account Equity", color="gold", linewidth=1.5)
    plt.title("Step 8: Final SMC Optimized Equity Curve\n(Amu Khani Sweep Logic + Setup Outcome ML)", fontsize=14, pad=15)
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
