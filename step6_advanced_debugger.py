import pandas as pd
import numpy as np

def find_origin_candle(df, start_idx, target_price, col='High', lookback=100):
    # Search backwards from start_idx for the candle that exactly matches target_price
    # This finds the exact origin candle of the structural extreme
    start_pos = df.index.get_loc(start_idx)
    end_pos = max(0, start_pos - lookback)
    
    sub_df = df.iloc[end_pos:start_pos+1]
    
    # We want the most recent match using np.isclose for float precision
    matches = sub_df[np.isclose(sub_df[col], target_price, atol=1e-5)]
    if len(matches) > 0:
        return matches.iloc[-1]
    return None

def main():
    print("Loading processed SMC data...")
    df = pd.read_csv('processed_smc_data.csv', index_col='Datetime')
    df.index = pd.to_datetime(df.index, utc=True)
    
    if df.empty:
        print("Dataset is empty.")
        return

    # Backtest variables
    balance = 10000.0
    risk_pct = 0.01
    
    position = 0 # 1 for Long, -1 for Short
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    units = 0.0
    
    pending_order = 0
    pend_entry = 0.0
    pend_sl = 0.0
    pend_tp = 0.0
    
    trade_log = []
    
    last_event = 0
    buffer = 0.0005 # 5 pips SL buffer
    
    print("Running Backtest and Failure Analysis...")
    for idx, row in df.iterrows():
        # Update last event
        if row['SMC_Event'] != 0:
            # Cancel pending orders on new structural events
            pending_order = 0
            
            # Evaluate new setups
            if not row['SMC_In_Penalty_Box']:
                # LONG SETUP: Event -2 (CHoCH Down) + Last was 1 (BOS Up)
                if row['SMC_Event'] == -2 and last_event == 1:
                    # Find origin candles
                    origin_low_candle = find_origin_candle(df, idx, row['SMC_Must_Crash'], col='Low')
                    origin_high_candle = find_origin_candle(df, idx, row['SMC_Must_Break'], col='High')
                    
                    if origin_low_candle is not None and origin_high_candle is not None:
                        pend_entry = origin_low_candle['High']
                        pend_tp = origin_high_candle['Low']
                        pend_sl = row['SMC_Must_Crash'] - buffer
                        
                        if pend_tp > pend_entry and pend_entry > pend_sl:
                            pending_order = 1
                            signal_data = row
                            
                # SHORT SETUP: Event 2 (CHoCH Up) + Last was -1 (BOS Down)
                elif row['SMC_Event'] == 2 and last_event == -1:
                    origin_high_candle = find_origin_candle(df, idx, row['SMC_Must_Break'], col='High')
                    origin_low_candle = find_origin_candle(df, idx, row['SMC_Must_Crash'], col='Low')
                    
                    if origin_high_candle is not None and origin_low_candle is not None:
                        pend_entry = origin_high_candle['Low']
                        pend_tp = origin_low_candle['High']
                        pend_sl = row['SMC_Must_Break'] + buffer
                        
                        if pend_tp < pend_entry and pend_entry < pend_sl:
                            pending_order = -1
                            signal_data = row
                            
            # Update last_event AFTER evaluating sequence
            last_event = row['SMC_Event']

        # Process open positions
        trade_closed = False
        pnl = 0.0
        
        if position == 1:
            if row['Low'] <= sl_price:
                pnl = (sl_price - entry_price) * units
                trade_closed = True
                outcome = 'Loss'
            elif row['High'] >= tp_price:
                pnl = (tp_price - entry_price) * units
                trade_closed = True
                outcome = 'Win'
                
        elif position == -1:
            if row['High'] >= sl_price:
                pnl = (entry_price - sl_price) * units
                trade_closed = True
                outcome = 'Loss'
            elif row['Low'] <= tp_price:
                pnl = (entry_price - tp_price) * units
                trade_closed = True
                outcome = 'Win'
                
        if trade_closed:
            balance += pnl
            trade_log.append({
                'Exit_Time': idx,
                'Type': 'Long' if position == 1 else 'Short',
                'Outcome': outcome,
                'PnL': pnl,
                'Signal_Time': signal_time,
                'DayOfWeek': signal_time.dayofweek,
                'Hour': signal_time.hour,
                'Dist_to_Must_Break': signal_dist_break,
                'Dist_to_Must_Crash': signal_dist_crash
            })
            position = 0
            
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
                
                signal_time = idx
                signal_dist_break = signal_data['Dist_to_Must_Break']
                signal_dist_crash = signal_data['Dist_to_Must_Crash']
                pending_order = 0
                
    # Analysis
    trades_df = pd.DataFrame(trade_log)
    if trades_df.empty:
        print("No trades executed.")
        return
        
    losses = trades_df[trades_df['Outcome'] == 'Loss']
    wins = trades_df[trades_df['Outcome'] == 'Win']
    
    total_trades = len(trades_df)
    win_rate = len(wins) / total_trades * 100
    
    print(f"Backtest Complete! Total Trades: {total_trades}, Win Rate: {win_rate:.2f}%")
    
    # Generate Report
    with open('amu_khani_bug_report.txt', 'w') as f:
        f.write("========================================\\n")
        f.write("      AMU KHANI SMC LOSS ANALYSIS       \\n")
        f.write("========================================\\n\\n")
        
        f.write(f"Total Trades: {total_trades}\\n")
        f.write(f"Total Losses: {len(losses)}\\n\\n")
        
        f.write("--- 1. Toxic Trading Hours & Days ---\\n")
        f.write("Losses by Day of Week (0=Mon, 4=Fri):\\n")
        f.write(losses['DayOfWeek'].value_counts().to_string() + "\\n\\n")
        
        f.write("Losses by Hour of Day:\\n")
        f.write(losses['Hour'].value_counts().head(5).to_string() + "\\n\\n")
        
        f.write("--- 2. Structural Distance Analysis ---\\n")
        f.write("Average Distance to Must_Break on Wins vs Losses:\\n")
        f.write(f"Wins:   {wins['Dist_to_Must_Break'].mean():.5f}\\n")
        f.write(f"Losses: {losses['Dist_to_Must_Break'].mean():.5f}\\n\\n")
        
        f.write("Average Distance to Must_Crash on Wins vs Losses:\\n")
        f.write(f"Wins:   {wins['Dist_to_Must_Crash'].mean():.5f}\\n")
        f.write(f"Losses: {losses['Dist_to_Must_Crash'].mean():.5f}\\n\\n")
        
        f.write("--- Conclusion & Recommended Filters ---\\n")
        if len(losses) > 0:
            worst_day = losses['DayOfWeek'].mode().iloc[0]
            worst_hour = losses['Hour'].mode().iloc[0]
            day_map = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday'}
            f.write(f"- Avoid trading on {day_map.get(worst_day, 'Weekends')}s, especially around Hour {worst_hour}.\\n")
            if losses['Dist_to_Must_Break'].mean() < wins['Dist_to_Must_Break'].mean():
                f.write("- Losses occur when price is significantly closer to structural extremes. Consider adding a minimum distance filter.\\n")
        
    print("Bug report saved to amu_khani_bug_report.txt")

if __name__ == "__main__":
    main()
