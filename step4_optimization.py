import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

    # Create Target column
    df['Next_Close'] = df['Close'].shift(-1)
    df['Target'] = (df['Next_Close'] > df['Close']).astype(int)

    # Drop NaNs
    df.dropna(inplace=True)

    # Define Features (X) and Target (y)
    features = ['EMA_50', 'EMA_200', 'RSI_14', 'ATRr_14', 'MACDh_12_26_9', 'Distance_to_EMA50']
    X = df[features]
    y = df['Target']

    # Split the data to get a test set for final evaluation
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

    # 2. TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    # 3. RandomizedSearchCV
    param_distributions = {
        'n_estimators': [100, 200, 500],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10]
    }

    print("Running RandomizedSearchCV with TimeSeriesSplit... (this may take a minute)")
    rf = RandomForestClassifier(random_state=42)
    
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=10,
        cv=tscv,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1  # Use all available cores
    )

    random_search.fit(X_train, y_train)

    print(f"\nBest Parameters: {random_search.best_params_}")
    best_model = random_search.best_estimator_

    # 4. Use model.predict_proba()
    print("\nEvaluating on holdout test set...")
    probs = best_model.predict_proba(X_test)
    
    prob_0 = probs[:, 0]
    prob_1 = probs[:, 1]

    # 5. Apply Confidence Threshold
    strong_signals = []
    for p0, p1 in zip(prob_0, prob_1):
        if p1 > 0.60:
            strong_signals.append(1)
        elif p0 > 0.60:
            strong_signals.append(-1)
        else:
            strong_signals.append(0)
            
    test_results = pd.DataFrame({
        'Target': y_test,
        'Prob_0': prob_0,
        'Prob_1': prob_1,
        'Strong_Signal': strong_signals
    })

    # 6. Print the percentage of trades that meet this 60% confidence threshold
    total_samples = len(test_results)
    high_conf_trades = test_results[test_results['Strong_Signal'] != 0].copy()
    num_high_conf = len(high_conf_trades)
    
    selectivity_pct = (num_high_conf / total_samples) * 100
    print(f"\nTotal test samples: {total_samples}")
    print(f"Number of high-confidence trades (prob > 60%): {num_high_conf}")
    print(f"Percentage of trades meeting threshold: {selectivity_pct:.2f}%")

    # 7. Print the accuracy of ONLY those high-confidence trades
    if num_high_conf > 0:
        # Convert -1 signals to 0 to match the Target mapping (0 = Down/Flat, 1 = Up)
        high_conf_trades['Prediction'] = high_conf_trades['Strong_Signal'].map({1: 1, -1: 0})
        
        acc = accuracy_score(high_conf_trades['Target'], high_conf_trades['Prediction'])
        print(f"Accuracy of ONLY high-confidence trades: {acc:.4f}")
    else:
        print("No trades met the 60% confidence threshold.")

if __name__ == "__main__":
    main()
