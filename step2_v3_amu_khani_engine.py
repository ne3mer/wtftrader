import yfinance as yf
import pandas as pd
import numpy as np

import pandas_ta as ta

class Extreme:
    def __init__(self, index, price):
        self.index = index
        self.price = price
        self.index_cross = index
        self.processed = False

    def copy(self):
        c = Extreme(self.index, self.price)
        c.index_cross = self.index_cross
        c.processed = self.processed
        return c

def recalc_level(arr, p, original, direction):
    # direction 1 for high, -1 for low
    if direction == -1:
        mn = float('inf')
        r = None
        for ext in arr:
            if ext.price < mn and ext.index > p.index:
                mn = ext.price
                r = ext
        return r.copy() if r is not None else (original.copy() if original else None)
    else:
        mx = -float('inf')
        r = None
        for ext in arr:
            if ext.price > mx and ext.index > p.index:
                mx = ext.price
                r = ext
        return r.copy() if r is not None else (original.copy() if original else None)

def main():
    symbol = "EURUSD=X"
    period = "730d"
    interval = "1h"
    
    print(f"Fetching {period} of {interval} data for {symbol}...")
    df = yf.Ticker(symbol).history(period=period, interval=interval)
    
    if df.empty:
        print("Failed to fetch data.")
        return

    # Ensure timezone is aware and uniform
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    print("Calculating technical indicators...")
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df['Distance_to_EMA50'] = df['Close'] - df['EMA_50']
    df['ATR_24h_Avg'] = df['ATRr_14'].rolling(window=24).mean()

    # Convert to fast numpy arrays for O(1) iteration
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    n = len(df)

    # State Variables - High
    wait_for_green_h = True
    wait_for_red_h = False
    wait_for_break_h = False
    break_level_h = closes[0]
    pick_h = closes[0]
    
    # State Variables - Low
    wait_for_red_l = True
    wait_for_green_l = False
    wait_for_break_l = False
    break_level_l = closes[0]
    low_l = closes[0]

    h = Extreme(0, pick_h)
    l = Extreme(0, low_l)

    confirmedhigh = []
    confirmedlow = []

    coming_high = None
    coming_low = None

    must_break = None
    must_crash = None

    state = 0
    last_bos_h = None
    last_bos = None

    last_event = 0
    chop_blocker = False
    penalty_box = False
    max_lookback = 100

    # Output arrays
    out_event = np.zeros(n, dtype=int)
    out_state = np.zeros(n, dtype=int)
    out_must_break = np.full(n, np.nan)
    out_must_crash = np.full(n, np.nan)
    out_in_penalty = np.zeros(n, dtype=bool)

    print("Running Amu Khani State Machine...")
    for i in range(n):
        open_val = opens[i]
        high_val = highs[i]
        low_val = lows[i]
        close_val = closes[i]

        new_h = False
        new_l = False
        event = 0

        # ==========================================
        # 1. HIGH DETECTION
        # ==========================================
        if close_val > open_val and high_val > pick_h:
            new_h = False
            wait_for_green_h = True
            h = Extreme(i, pick_h)
            
        if close_val > open_val and wait_for_green_h:
            pick_h = high_val
            wait_for_green_h = False
            wait_for_red_h = True
            break_level_h = low_val
            h = Extreme(i, pick_h)
        elif wait_for_red_h and close_val < open_val:
            wait_for_red_h = False
            last_pick = pick_h
            pick_h = max(high_val, pick_h)
            break_level_h = min(low_val, break_level_h)
            wait_for_break_h = True
            if pick_h > last_pick:
                if i > 0 and closes[i-1] > opens[i-1] and close_val < open_val:
                    break_level_h = min(low_val, lows[i-1])
                h = Extreme(i, pick_h)
        elif wait_for_break_h:
            if close_val < break_level_h:
                wait_for_green_h = True
                wait_for_red_h = False
                wait_for_break_h = False
                new_h = True
            else:
                last_pick = pick_h
                pick_h = max(high_val, pick_h)
                if pick_h > last_pick:
                    if i > 0 and closes[i-1] > opens[i-1] and close_val < open_val:
                        break_level_h = min(low_val, lows[i-1])
                    h = Extreme(i, pick_h)

        # ==========================================
        # 2. LOW DETECTION
        # ==========================================
        if close_val < open_val and low_val < low_l:
            new_l = False
            wait_for_red_l = True
            l = Extreme(i, low_l)
            
        if close_val < open_val and wait_for_red_l:
            low_l = low_val
            wait_for_red_l = False
            wait_for_green_l = True
            break_level_l = high_val
            l = Extreme(i, low_l)
        elif wait_for_green_l and close_val > open_val:
            wait_for_green_l = False
            last_low = low_l
            low_l = min(low_val, low_l)
            break_level_l = max(high_val, break_level_l)
            wait_for_break_l = True
            if low_l < last_low:
                if i > 0 and closes[i-1] < opens[i-1] and close_val > open_val:
                    break_level_l = max(high_val, highs[i-1])
                l = Extreme(i, low_l)
        elif wait_for_break_l:
            if close_val > break_level_l:
                wait_for_red_l = True
                wait_for_green_l = False
                wait_for_break_l = False
                new_l = True
            else:
                last_low = low_l
                low_l = min(low_val, low_l)
                if low_l < last_low:
                    if i > 0 and closes[i-1] < opens[i-1] and close_val > open_val:
                        break_level_l = max(high_val, highs[i-1])
                    l = Extreme(i, low_l)

        # ==========================================
        # 3. CONFIRMING EXTREMES
        # ==========================================
        if new_h:
            coming_high = h.copy()
            confirmedhigh.insert(0, h.copy()) # Index 0 is always newest
            new_h = False
        if new_l:
            coming_low = l.copy()
            confirmedlow.insert(0, l.copy())
            new_l = False

        # ==========================================
        # 4. MARKET STRUCTURE (BOS & CHOCH)
        # ==========================================
        if coming_high is not None:
            if must_break is None:
                must_break = coming_high.copy()
            else:
                highest_body = max(closes[coming_high.index], opens[coming_high.index])
                if coming_high.price > must_break.price and highest_body > must_break.price:
                    must_break = coming_high.copy()
                elif must_crash is not None:
                    if len(confirmedlow) > 0 and coming_high.price < must_crash.price and coming_high.price > confirmedlow[0].price:
                        must_break = coming_high.copy()

        if coming_low is not None:
            if must_crash is None:
                must_crash = coming_low.copy()
            else:
                lowest_body = min(closes[coming_low.index], opens[coming_low.index])
                if coming_low.price < must_crash.price and lowest_body < must_crash.price:
                    must_crash = coming_low.copy()
                elif must_break is not None:
                    if len(confirmedhigh) > 0 and coming_low.price > must_break.price and coming_low.price < confirmedhigh[0].price:
                        must_crash = coming_low.copy()

        # Check Must Break (BOS/CHoCH Up)
        if must_break is not None:
            if coming_low is not None and last_bos_h is not None:
                must_crash_tmp = recalc_level(confirmedlow, last_bos_h, must_crash, -1)
                if must_crash_tmp is not None and must_crash.index > last_bos_h.index and must_crash_tmp.index < last_bos_h.index_cross:
                    must_crash = must_crash_tmp.copy()
                    
            if close_val > must_break.price and not must_break.processed:
                must_break.processed = True
                event = 1 if state >= 0 else 2
                state = 1
                if event == 1:
                    last_bos_h = must_break.copy()
                    last_bos_h.index_cross = i
                    must_crash = recalc_level(confirmedlow, must_break, must_crash, -1)

        # Check Must Crash (BOS/CHoCH Down)
        if must_crash is not None:
            if coming_high is not None and last_bos is not None:
                must_break_tmp = recalc_level(confirmedhigh, last_bos, must_break, 1)
                if must_break_tmp is not None and must_break.index < last_bos.index and must_break_tmp.index < last_bos.index_cross:
                    must_break = must_break_tmp.copy()
                    
            if close_val < must_crash.price and not must_crash.processed:
                must_crash.processed = True
                event = -1 if state <= 0 else -2
                state = -1
                if event == -1:
                    last_bos = must_crash.copy()
                    last_bos.index_cross = i
                    must_break = recalc_level(confirmedhigh, must_crash, must_break, 1)
                if event == -2:
                    must_break = recalc_level(confirmedhigh, must_crash, must_break, 1)

        # ==========================================
        # 5. EVENT FILTER (Penalty Box)
        # ==========================================
        if event != 0:
            if event == 1 or event == -1:
                chop_blocker = False
            
            if event == 2 or event == -2:
                origin_index = must_break.index if event == 2 else must_crash.index
                bars_since_origin = i - origin_index
                lookback_ok = bars_since_origin <= max_lookback

                if chop_blocker:
                    penalty_box = True
                else:
                    if penalty_box:
                        penalty_box = False
                chop_blocker = True
            
            last_event = event

        # Save to Output Arrays
        out_event[i] = event
        out_state[i] = state
        out_must_break[i] = must_break.price if must_break else np.nan
        out_must_crash[i] = must_crash.price if must_crash else np.nan
        out_in_penalty[i] = penalty_box

    # Apply back to DataFrame
    df['SMC_Event'] = out_event
    df['SMC_State'] = out_state
    df['SMC_Must_Break'] = out_must_break
    df['SMC_Must_Crash'] = out_must_crash
    df['SMC_In_Penalty_Box'] = out_in_penalty
    
    df['Dist_to_Must_Break'] = df['SMC_Must_Break'] - df['Close']
    df['Dist_to_Must_Crash'] = df['Close'] - df['SMC_Must_Crash']
    
    # Calculate Bars Since Last Event
    last_event_bar = 0
    bars_since_event = np.zeros(n, dtype=int)
    for i in range(n):
        if out_event[i] != 0:
            last_event_bar = i
        bars_since_event[i] = i - last_event_bar
    df['Bars_Since_Event'] = bars_since_event

    # New SMC Features
    df['CHoCH_Extension_Distance'] = 0.0
    df.loc[df['SMC_Event'] == 2, 'CHoCH_Extension_Distance'] = df['Close'] - df['SMC_Must_Crash']
    df.loc[df['SMC_Event'] == -2, 'CHoCH_Extension_Distance'] = df['SMC_Must_Break'] - df['Close']
    
    # Cyclical Time Features
    df['Hour_Sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['Day_Sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['Day_Cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df['Minute_Sin'] = np.sin(2 * np.pi * df.index.minute / 60)
    df['Minute_Cos'] = np.cos(2 * np.pi * df.index.minute / 60)
    
    # Market Volatility Regime
    # 50 days of 1h data = 1200 bars
    df['ATR_50d_Mean'] = df['ATRr_14'].rolling(window=1200).mean()
    df['Market_Volatility_Regime'] = df['ATRr_14'] / df['ATR_50d_Mean']

    # ML Target
    df['Next_Close'] = df['Close'].shift(-1)
    df['Target'] = (df['Next_Close'] > df['Close']).astype(int)

    # Drop early records with no structural boundaries
    df = df.dropna(subset=['SMC_Must_Break', 'SMC_Must_Crash', 'ATR_50d_Mean'])

    print("Saving processed data to processed_smc_data.csv...")
    df.to_csv("processed_smc_data.csv")
    
    print("\nSample of structural events:")
    events_df = df[df['SMC_Event'] != 0].tail(10)
    print(events_df[['Close', 'SMC_Event', 'SMC_State', 'Dist_to_Must_Break', 'CHoCH_Extension_Distance', 'Hour_Sin', 'Market_Volatility_Regime']])

if __name__ == "__main__":
    main()
