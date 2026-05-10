import yfinance as yf
import pandas as pd

def main():
    # Define symbol, period, and interval
    symbol = "EURUSD=X"
    period = "730d"  # yfinance supports max 730 days for 1h interval, which is roughly 2 years
    interval = "1h"

    print(f"Fetching {period} of {interval} data for {symbol}...")
    
    # Fetch historical data
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    # Check if data was fetched successfully
    if df.empty:
        print("Failed to fetch data.")
        quit()

    # Clean the data: keep specific columns and drop NaNs
    cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[cols_to_keep].copy()
    df.dropna(inplace=True)

    # Print the first 5 rows and the last 5 rows
    print("First 5 rows:")
    print(df.head(5))
    print("\nLast 5 rows:")
    print(df.tail(5))

if __name__ == "__main__":
    main()
