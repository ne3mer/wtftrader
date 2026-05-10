import yfinance as yf
import pandas as pd
import pandas_ta as ta

def main():
    # 1. Fetch the data
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
    
    # 2. Add technical indicators using pandas_ta
    print("Calculating technical indicators...")
    # EMA_50 and EMA_200
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    
    # RSI (length=14)
    df.ta.rsi(length=14, append=True)
    
    # ATR (length=14)
    df.ta.atr(length=14, append=True)
    
    # MACD (fast=12, slow=26, signal=9)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    # 3. Create custom feature
    # pandas_ta automatically names the EMA column 'EMA_50'
    df['Distance_to_EMA50'] = df['Close'] - df['EMA_50']

    # 4. Drop any rows with NaN values
    df.dropna(inplace=True)

    # 5. Print the names of all columns
    print("\nColumns after feature engineering:")
    print(df.columns.tolist())

    # 6. Print the last 5 rows of the DataFrame
    print("\nLast 5 rows:")
    print(df.tail(5))

if __name__ == "__main__":
    main()
