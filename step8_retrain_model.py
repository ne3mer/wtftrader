import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    print("Initializing Walk-Forward Retraining Sequence...")
    
    # 1. Load the latest available dataset
    print("Loading processed SMC dataset...")
    df = pd.read_csv('processed_smc_data.csv', index_col='Datetime')
    df.index = pd.to_datetime(df.index, utc=True)
    df_clean = df.dropna().copy()
    
    if df_clean.empty:
        print("Dataset is empty. Cannot retrain.")
        return

    features = [
        'EMA_50', 'EMA_200', 'RSI_14', 'ATRr_14', 'MACDh_12_26_9', 'Distance_to_EMA50',
        'SMC_State', 'Dist_to_Must_Break', 'Dist_to_Must_Crash', 'SMC_In_Penalty_Box',
        'CHoCH_Extension_Distance', 'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos', 
        'Minute_Sin', 'Minute_Cos', 'Market_Volatility_Regime'
    ]
    
    X = df_clean[features]
    y = df_clean['Target']
    
    print(f"Training on entire historical dataset ({len(X)} samples)...")
    
    # 2. Instantiate the model with the previously optimized hyperparameters
    best_params = {'n_estimators': 500, 'min_samples_split': 10, 'max_depth': 5}
    model = RandomForestClassifier(random_state=42, **best_params)
    
    # 3. Train on the FULL dataset
    model.fit(X, y)
    print("Model training complete.")
    
    # 4. Save the new model
    model_filename = 'final_smc_bot.pkl'
    joblib.dump(model, model_filename)
    
    print(f"Success: Updated model saved as '{model_filename}'.")
    print("The Amu Khani Bot is ready for live execution.")

if __name__ == "__main__":
    main()
