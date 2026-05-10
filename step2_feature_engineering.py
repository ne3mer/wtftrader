import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta

def main():
    symbol = "EURUSD=X"
    period = "730d"
    interval = "1h"

    print(f"Fetching {period} of {interval} data for {symbol}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        print("Failed to fetch data.")
        quit()

    cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[cols_to_keep].copy()
    
    print("Calculating technical indicators...")
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    df['Distance_to_EMA50'] = df['Close'] - df['EMA_50']

    # --- SMC Features ---
    print("Calculating SMC features...")
    # 1. Swing High/Low (20 periods)
    df['Last_Swing_High'] = df['High'].rolling(window=20).max().shift(1)
    df['Last_Swing_Low'] = df['Low'].rolling(window=20).min().shift(1)

    # 2. Market Structure (BOS)
    df['BOS'] = 0
    df.loc[df['Close'] > df['Last_Swing_High'], 'BOS'] = 1
    df.loc[df['Close'] < df['Last_Swing_Low'], 'BOS'] = -1

    # 3. Fair Value Gaps (FVG)
    bullish_fvg = df['Low'] > df['High'].shift(2)
    bearish_fvg = df['High'] < df['Low'].shift(2)
    df['FVG_Present'] = 0
    df.loc[bullish_fvg, 'FVG_Present'] = 1
    df.loc[bearish_fvg, 'FVG_Present'] = -1

    # 4. Liquidity Grabs (Sweep)
    bullish_sweep = (df['Low'] < df['Last_Swing_Low']) & (df['Close'] > df['Last_Swing_Low'])
    bearish_sweep = (df['High'] > df['Last_Swing_High']) & (df['Close'] < df['Last_Swing_High'])
    df['Liquidity_Sweep'] = 0
    df.loc[bullish_sweep, 'Liquidity_Sweep'] = 1
    df.loc[bearish_sweep, 'Liquidity_Sweep'] = -1

    # 5. Body to Wick Ratio
    body_size = (df['Close'] - df['Open']).abs()
    total_size = df['High'] - df['Low']
    df['Body_to_Wick_Ratio'] = body_size / total_size.replace(0, np.nan)
    df['Body_to_Wick_Ratio'] = df['Body_to_Wick_Ratio'].fillna(0)

    # Drop NaNs
    df.dropna(inplace=True)

    print("\nColumns after feature engineering:")
    print(df.columns.tolist())

    print("\nLast 5 rows:")
    print(df[['Close', 'BOS', 'FVG_Present', 'Liquidity_Sweep', 'Body_to_Wick_Ratio']].tail(5))

if __name__ == "__main__":
    main()
