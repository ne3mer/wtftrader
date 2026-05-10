import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    print("Loading SMC dataset...")
    df = pd.read_csv('processed_smc_data.csv', index_col='Datetime', parse_dates=True)
    df = df.dropna()

    if df.empty:
        print("Failed to load or clean data.")
        quit()

    features = [
        'EMA_50', 'EMA_200', 'RSI_14', 'ATRr_14', 'MACDh_12_26_9', 'Distance_to_EMA50',
        'SMC_State', 'Dist_to_Must_Break', 'Dist_to_Must_Crash', 'SMC_In_Penalty_Box',
        'CHoCH_Extension_Distance', 'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos', 
        'Minute_Sin', 'Minute_Cos', 'Market_Volatility_Regime'
    ]
    X = df[features]
    y = df['Target']

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

    tscv = TimeSeriesSplit(n_splits=5)
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
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    print(f"\nBest Parameters: {random_search.best_params_}")
    best_model = random_search.best_estimator_

    print("\nEvaluating on holdout test set...")
    probs = best_model.predict_proba(X_test)
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
            
    test_results = pd.DataFrame({'Target': y_test, 'Prob_0': prob_0, 'Prob_1': prob_1, 'Strong_Signal': strong_signals})

    total_samples = len(test_results)
    high_conf_trades = test_results[test_results['Strong_Signal'] != 0].copy()
    num_high_conf = len(high_conf_trades)
    
    selectivity_pct = (num_high_conf / total_samples) * 100
    print(f"\nTotal test samples: {total_samples}")
    print(f"Number of high-confidence signals (prob > 60%): {num_high_conf}")
    print(f"Percentage of signals meeting threshold: {selectivity_pct:.2f}%")

    if num_high_conf > 0:
        high_conf_trades['Prediction'] = high_conf_trades['Strong_Signal'].map({1: 1, -1: 0})
        acc = accuracy_score(high_conf_trades['Target'], high_conf_trades['Prediction'])
        print(f"Accuracy of ONLY high-confidence signals: {acc:.4f}")
    else:
        print("No signals met the 60% confidence threshold.")

if __name__ == "__main__":
    main()
