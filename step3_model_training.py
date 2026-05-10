import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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
    
    # Feature Engineering
    print("Calculating technical indicators...")
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    df['Distance_to_EMA50'] = df['Close'] - df['EMA_50']

    # 2. Create Target column
    # 1 if Next Close > Current Close, else 0
    df['Next_Close'] = df['Close'].shift(-1)
    df['Target'] = (df['Next_Close'] > df['Close']).astype(int)

    # 3. Drop NaNs (including the last row which has NaN for Next_Close)
    df.dropna(inplace=True)

    # 4. Define Features (X)
    features = ['EMA_50', 'EMA_200', 'RSI_14', 'ATRr_14', 'MACDh_12_26_9', 'Distance_to_EMA50']
    X = df[features]

    # 5. Define Target (y)
    y = df['Target']

    # 6. Split the data
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

    # 7. Initialize and train RandomForestClassifier
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 8. Make predictions and print metrics
    print("Making predictions...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy Score: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
