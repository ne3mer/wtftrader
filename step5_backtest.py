import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    # 1. Fetch data and Feature Engineering
    symbol = "EURUSD=X"
    period = "730d"
    interval = "1h"

    print("Fetching data and calculating features...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        print("Failed to fetch data.")
        quit()

    cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[cols_to_keep].copy()
    
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    df['Distance_to_EMA50'] = df['Close'] - df['EMA_50']

    # Target
    df['Next_Close'] = df['Close'].shift(-1)
    df['Target'] = (df['Next_Close'] > df['Close']).astype(int)

    df.dropna(inplace=True)

    features = ['EMA_50', 'EMA_200', 'RSI_14', 'ATRr_14', 'MACDh_12_26_9', 'Distance_to_EMA50']
    X = df[features]
    y = df['Target']

    print("Splitting data and training the best model from Step 4...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

    # Train best model from Step 4
    best_params = {'n_estimators': 100, 'min_samples_split': 5, 'max_depth': 10}
    model = RandomForestClassifier(random_state=42, **best_params)
    model.fit(X_train, y_train)

    # 2. Predict probabilities on Test set
    print("Generating High Confidence Signals...")
    probs = model.predict_proba(X_test)
    prob_0 = probs[:, 0]
    prob_1 = probs[:, 1]

    strong_signals = []
    for p0, p1 in zip(prob_0, prob_1):
        if p1 > 0.60:
            strong_signals.append(1)
        elif p0 > 0.60:
            strong_signals.append(-1)
        else:
            strong_signals.append(0)

    # Prepare Backtesting DataFrame
    df_test = X_test.copy()
    df_test['Close'] = df.loc[X_test.index, 'Close']
    df_test['High'] = df.loc[X_test.index, 'High']
    df_test['Low'] = df.loc[X_test.index, 'Low']
    df_test['ATR'] = df.loc[X_test.index, 'ATRr_14']
    df_test['Strong_Signal'] = strong_signals

    # 3. Define Trading Logic
    print("Running Backtest Simulator...")
    balance = 10000.0
    UNITS = 10000  # 0.1 Lot
    
    position = 0
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    
    # Tracking variables
    equity_curve = []
    total_trades = 0
    winning_trades = 0
    peak_balance = balance
    max_drawdown = 0.0

    for index, row in df_test.iterrows():
        # Manage open position
        if position != 0:
            high = row['High']
            low = row['Low']
            
            # Check Long
            if position == 1:
                # Conservative: check SL first in case of big volatile candle
                if low <= sl_price:
                    pnl = (sl_price - entry_price) * UNITS
                    balance += pnl
                    position = 0
                    total_trades += 1
                    if pnl > 0: winning_trades += 1
                elif high >= tp_price:
                    pnl = (tp_price - entry_price) * UNITS
                    balance += pnl
                    position = 0
                    total_trades += 1
                    if pnl > 0: winning_trades += 1
                    
            # Check Short
            elif position == -1:
                if high >= sl_price:
                    pnl = (entry_price - sl_price) * UNITS
                    balance += pnl
                    position = 0
                    total_trades += 1
                    if pnl > 0: winning_trades += 1
                elif low <= tp_price:
                    pnl = (entry_price - tp_price) * UNITS
                    balance += pnl
                    position = 0
                    total_trades += 1
                    if pnl > 0: winning_trades += 1
                    
        # Open new position if flat
        if position == 0:
            signal = row['Strong_Signal']
            if signal == 1:
                position = 1
                entry_price = row['Close']
                atr = row['ATR']
                sl_price = entry_price - 1.5 * atr
                tp_price = entry_price + 2.0 * atr
            elif signal == -1:
                position = -1
                entry_price = row['Close']
                atr = row['ATR']
                sl_price = entry_price + 1.5 * atr
                tp_price = entry_price - 2.0 * atr

        # Track metrics
        equity_curve.append(balance)
        if balance > peak_balance:
            peak_balance = balance
        drawdown = (peak_balance - balance) / peak_balance * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    df_test['Equity'] = equity_curve

    # 5. Calculate and print metrics
    net_profit = balance - 10000.0
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    print("\n" + "=" * 40)
    print(" " * 12 + "BACKTEST RESULTS")
    print("=" * 40)
    print(f"Starting Capital:    $10000.00")
    print(f"Total Net Profit:    ${net_profit:.2f}")
    print(f"Final Balance:       ${balance:.2f}")
    print("-" * 40)
    print(f"Total Trades:        {total_trades}")
    print(f"Win Rate:            {win_rate:.2f}%")
    print(f"Max Drawdown:        {max_drawdown:.2f}%")
    print("=" * 40 + "\n")

    # 6. Plot the Equity Curve
    print("Saving Equity Curve plot...")
    plt.figure(figsize=(12, 6))
    plt.plot(df_test.index, df_test['Equity'], label="Account Equity", color="dodgerblue", linewidth=1.5)
    
    # Formatting
    plt.title("Step 5: Algorithmic Backtest Equity Curve\n(0.1 Lot | SL: 1.5 ATR | TP: 2.0 ATR)", fontsize=14, pad=15)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Account Balance (USD)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.fill_between(df_test.index, df_test['Equity'], 10000, where=(df_test['Equity'] >= 10000), interpolate=True, color='green', alpha=0.1)
    plt.fill_between(df_test.index, df_test['Equity'], 10000, where=(df_test['Equity'] < 10000), interpolate=True, color='red', alpha=0.1)
    plt.axhline(y=10000, color='r', linestyle='-', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plot_filename = 'equity_curve.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved successfully as '{plot_filename}'.")

if __name__ == "__main__":
    main()
