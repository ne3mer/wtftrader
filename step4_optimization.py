import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    print("Loading SMC dataset...")
    df = pd.read_csv('processed_smc_data.csv', index_col='Datetime', parse_dates=True)
    df = df.dropna(subset=['Target']) # Train ONLY on valid labeled setups!

    if df.empty:
        print("Failed to load or clean data.")
        quit()

    features = [
        'EMA_50', 'EMA_200', 'RSI_14', 'ATRr_14', 'MACDh_12_26_9', 'Distance_to_EMA50',
        'SMC_State', 'Dist_to_Must_Break', 'Dist_to_Must_Crash', 'SMC_In_Penalty_Box',
        'CHoCH_Extension_Distance', 'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos', 
        'Minute_Sin', 'Minute_Cos', 'Market_Volatility_Regime', 'Is_Toxic_Window'
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

    test_results = pd.DataFrame({'Target': y_test, 'Prob_1': prob_1})
    
    # We predict taking the trade if ML probability of Win (Prob_1) > 0.50
    test_results['Prediction'] = (test_results['Prob_1'] > 0.50).astype(int)
    
    acc = accuracy_score(test_results['Target'], test_results['Prediction'])
    total_samples = len(test_results)
    
    print(f"\nTotal test setups: {total_samples}")
    print(f"Accuracy of setup predictions: {acc:.4f}")

if __name__ == "__main__":
    main()
