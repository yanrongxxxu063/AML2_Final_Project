import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import os

def create_lags(df, features, lag_months=1):
    df_lagged = df.copy()
    # Sort just in case
    df_lagged = df_lagged.sort_values(by=['Sub-Nbhood', 'Beds', 'Month'])
    
    for f in features:
        df_lagged[f'{f}_lag{lag_months}'] = df_lagged.groupby(['Sub-Nbhood', 'Beds'])[f].shift(lag_months)
    return df_lagged

def main():
    base_dir = Path(r"c:\Users\jesse\CUcourses\Applied Machine Learning 2\Group Project")
    data_file = base_dir / "Dataset" / "integrated_panel_with_regimes.csv"
    
    # Create an output directory for plots
    plot_dir = base_dir / "Plots" / "Phase4"
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    df['Month'] = pd.to_datetime(df['Month'])
    
    # Target Variable: Predict Next Month's Log Median Sale Price
    df = df.sort_values(by=['Sub-Nbhood', 'Beds', 'Month'])
    df['Target_Log_Price'] = df.groupby(['Sub-Nbhood', 'Beds'])['Median_Sale_Price'].shift(-1)
    df['Target_Log_Price'] = np.log(df['Target_Log_Price'])

    # Drop missing target values (the last month for each segment)
    df = df.dropna(subset=['Target_Log_Price'])
    
    # Encode categorical variables
    df['Regime'] = df['Regime'].astype('category')
    df['Sub-Nbhood'] = df['Sub-Nbhood'].astype('category')
    
    # Features
    base_macro_features = ['FEDFUNDS', 'UNRATE', 'PCE_YOY', 'MORTGAGE_SPREAD']
    sentiment_features = ['LM_Polarity', 'LM_Subjectivity', 'LM_Positive', 'LM_Negative']
    categorical_features = ['Beds', 'Sub-Nbhood']
    
    # Prepare dummy variables for categorical
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    cat_dummy_cols = [c for c in df.columns if c.startswith('Beds_') or c.startswith('Sub-Nbhood_')]
    
    # Base feature set
    X_base = base_macro_features + cat_dummy_cols
    
    # Sentiment-Aware feature set
    X_sentiment = X_base + sentiment_features
    
    # Regime-Aware feature set
    df = pd.get_dummies(df, columns=['Regime'], drop_first=True)
    regime_dummy_cols = [c for c in df.columns if c.startswith('Regime_')]
    X_regime_sentiment = X_base + sentiment_features + regime_dummy_cols
    
    # Time-based Train/Test Split (80/20)
    # Since we have panel data, we split by Time to prevent data leakage
    split_date = df['Month'].quantile(0.8)
    print(f"Splitting train/test at {split_date}")
    
    train_df = df[df['Month'] <= split_date].copy()
    test_df = df[df['Month'] > split_date].copy()
    
    # Drop NaNs created by lagging if any (none yet but safe)
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    def train_evaluate(model_name, features):
        print(f"\nTraining {model_name}...")
        X_train = train_df[features]
        y_train = train_df['Target_Log_Price']
        X_test = test_df[features]
        y_test = test_df['Target_Log_Price']
        
        # Random Forest Base Model
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
        rf.fit(X_train, y_train)
        
        preds = rf.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE : {mae:.4f}")
        print(f"  R2  : {r2:.4f}")
        
        return rf, rmse, mae, r2, list(X_train.columns), rf.feature_importances_

    # Train Baseline Model
    rf_base, rmse_b, mae_b, r2_b, cols_b, feat_imp_b = train_evaluate('Baseline (Macro Only)', X_base)
    
    # Train Sentiment Model
    rf_sent, rmse_s, mae_s, r2_s, cols_s, feat_imp_s = train_evaluate('Sentiment-Aware', X_sentiment)
    
    # Train Regime+Sentiment Model
    rf_full, rmse_f, mae_f, r2_f, cols_f, feat_imp_f = train_evaluate('Regime+Sentiment Aware', X_regime_sentiment)
    
    # Plot Evaluation Metrics
    metrics_df = pd.DataFrame({
        'Model': ['Baseline', 'Sentiment-Aware', 'Regime+Sentiment'],
        'RMSE': [rmse_b, rmse_s, rmse_f],
        'MAE': [mae_b, mae_s, mae_f],
        'R2': [r2_b, r2_s, r2_f]
    })
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.barplot(data=metrics_df, x='Model', y='RMSE', palette='Blues_r')
    plt.title('RMSE Comparison (Lower is Better)')
    plt.xticks(rotation=15)
    
    plt.subplot(1, 2, 2)
    sns.barplot(data=metrics_df, x='Model', y='R2', palette='Greens')
    plt.title('R2 Score Comparison (Higher is Better)')
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    plt.savefig(plot_dir / "model_comparison.png")
    plt.close()
    
    # Feature Importance for Full Model
    imp_df = pd.DataFrame({'Feature': cols_f, 'Importance': feat_imp_f})
    imp_df = imp_df.sort_values('Importance', ascending=False).head(20) # Top 20
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=imp_df, x='Importance', y='Feature', palette='magma')
    plt.title("Top 20 Feature Importances (Regime+Sentiment Model)")
    plt.tight_layout()
    plt.savefig(plot_dir / "feature_importance.png")
    plt.close()

    print("\nPhase 4 Complete. Plots saved to 'Plots/Phase4'.")

if __name__ == "__main__":
    main()
