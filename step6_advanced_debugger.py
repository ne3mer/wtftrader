import pandas as pd

def main():
    print("Loading Processed SMC Dataset for Advanced Debugging...")
    df = pd.read_csv('processed_smc_data.csv', index_col='Datetime')
    df.index = pd.to_datetime(df.index, utc=True)

    # Filter to only rows that were filled and resolved (Target is not NaN)
    setups = df.dropna(subset=['Target']).copy()
    
    total_setups = len(setups)
    if total_setups == 0:
        print("No filled setups found in the dataset.")
        return
        
    wins = setups['Target'].sum()
    losses = total_setups - wins
    win_rate = (wins / total_setups) * 100
    
    print(f"\n--- MACRO OVERVIEW ---")
    print(f"Total Filled Setups: {total_setups}")
    print(f"Total Wins (2:1 RR): {int(wins)}")
    print(f"Total Losses (1:1 SL): {int(losses)}")
    print(f"Raw Strategy Win Rate: {win_rate:.2f}%")
    
    # Analyze by Hour
    print("\n--- PERFORMANCE BY HOUR (UTC) ---")
    hourly_stats = setups.groupby(setups.index.hour)['Target'].agg(['count', 'mean']).rename(columns={'count': 'Setups', 'mean': 'Win Rate'})
    hourly_stats['Win Rate'] = hourly_stats['Win Rate'] * 100
    print(hourly_stats.to_string(formatters={'Win Rate': '{:.2f}%'.format}))
    
    # Specifically isolate 12:00-13:00 Window
    london_ny_overlap = setups[setups.index.hour.isin([12, 13])]
    l_ny_count = len(london_ny_overlap)
    if l_ny_count > 0:
        l_ny_win_rate = (london_ny_overlap['Target'].sum() / l_ny_count) * 100
        print(f"\n--- 12:00-13:00 UTC OVERLAP ANALYSIS ---")
        print(f"Setups during overlap: {l_ny_count}")
        print(f"Win Rate during overlap: {l_ny_win_rate:.2f}%")
        if l_ny_win_rate < win_rate:
            print("Conclusion: The overlap is TOXIC compared to the baseline.")
        else:
            print("Conclusion: The overlap is HEALTHY/NEUTRAL compared to the baseline.")
    else:
        print("\n--- 12:00-13:00 UTC OVERLAP ANALYSIS ---")
        print("No setups found during the overlap window.")
        
    # Analyze by Volatility Regime
    print("\n--- VOLATILITY REGIME ANALYSIS ---")
    high_vol = setups[setups['Market_Volatility_Regime'] > 1.0]
    low_vol = setups[setups['Market_Volatility_Regime'] <= 1.0]
    
    if len(high_vol) > 0:
        hv_wr = (high_vol['Target'].sum() / len(high_vol)) * 100
        print(f"High Volatility (>50d Mean) Setups: {len(high_vol)}, Win Rate: {hv_wr:.2f}%")
    if len(low_vol) > 0:
        lv_wr = (low_vol['Target'].sum() / len(low_vol)) * 100
        print(f"Low Volatility (<=50d Mean) Setups: {len(low_vol)}, Win Rate: {lv_wr:.2f}%")

    # Output to report
    with open('amu_khani_bug_report.txt', 'w') as f:
        f.write("Amu Khani Setup Outcome Debugger (ATR RR Logic)\n")
        f.write("==================================================\n")
        f.write(f"Total Setups: {total_setups}\n")
        f.write(f"Win Rate: {win_rate:.2f}%\n\n")
        f.write("Performance by Hour:\n")
        f.write(hourly_stats.to_string(formatters={'Win Rate': '{:.2f}%'.format}))

if __name__ == "__main__":
    main()
