import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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

    # SMC Features
    df['Last_Swing_High'] = df['High'].rolling(window=20).max().shift(1)
    df['Last_Swing_Low'] = df['Low'].rolling(window=20).min().shift(1)
    df['BOS'] = 0
    df.loc[df['Close'] > df['Last_Swing_High'], 'BOS'] = 1
    df.loc[df['Close'] < df['Last_Swing_Low'], 'BOS'] = -1

    bullish_fvg = df['Low'] > df['High'].shift(2)
    bearish_fvg = df['High'] < df['Low'].shift(2)
    df['FVG_Present'] = 0
    df.loc[bullish_fvg, 'FVG_Present'] = 1
    df.loc[bearish_fvg, 'FVG_Present'] = -1

    bullish_sweep = (df['Low'] < df['Last_Swing_Low']) & (df['Close'] > df['Last_Swing_Low'])
    bearish_sweep = (df['High'] > df['Last_Swing_High']) & (df['Close'] < df['Last_Swing_High'])
    df['Liquidity_Sweep'] = 0
    df.loc[bullish_sweep, 'Liquidity_Sweep'] = 1
    df.loc[bearish_sweep, 'Liquidity_Sweep'] = -1

    body_size = (df['Close'] - df['Open']).abs()
    total_size = df['High'] - df['Low']
    df['Body_to_Wick_Ratio'] = body_size / total_size.replace(0, np.nan)
    df['Body_to_Wick_Ratio'] = df['Body_to_Wick_Ratio'].fillna(0)

    # Target
    df['Next_Close'] = df['Close'].shift(-1)
    df['Target'] = (df['Next_Close'] > df['Close']).astype(int)

    df = df.dropna()

    features = [
        'EMA_50', 'EMA_200', 'RSI_14', 'ATRr_14', 'MACDh_12_26_9', 'Distance_to_EMA50',
        'BOS', 'FVG_Present', 'Liquidity_Sweep', 'Body_to_Wick_Ratio'
    ]
    X = df[features]
    y = df['Target']

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Making predictions...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy Score: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
