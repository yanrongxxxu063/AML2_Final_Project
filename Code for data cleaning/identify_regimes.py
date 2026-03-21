import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
import os

def main():
    base_dir = Path(r"c:\Users\jesse\CUcourses\Applied Machine Learning 2\Group Project")
    data_file = base_dir / "Dataset" / "integrated_panel.csv"
    
    # Create an output directory for plots
    plot_dir = base_dir / "Plots" / "Phase2"
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    df['Month'] = pd.to_datetime(df['Month'])
    
    # We want to cluster on the monthly macro & sentiment features, not on the housing segments.
    # Therefore, we first drop duplicates on the Month level to get the unique macro/sentiment time series.
    features = ['FEDFUNDS', 'UNRATE', 'PCE_YOY', 'MORTGAGE_SPREAD', 'LM_Polarity', 'LM_Subjectivity']
    macro_df = df[['Month'] + features].drop_duplicates().sort_values('Month').reset_index(drop=True)
    print(f"Unique months for clustering: {len(macro_df)}")
    
    # EDA: Correlation Matrix
    corr = macro_df[features].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Macro and Sentiment Features")
    plt.tight_layout()
    plt.savefig(plot_dir / "correlation_matrix.png")
    plt.close()
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(macro_df[features])
    
    # Find optimal number of clusters using Silhouette Score and Elbow Method
    max_k = 8
    inertias = []
    sil_scores = []
    K_range = range(2, max_k + 1)
    
    print("Evaluating K-Means for K=2 to 8...")
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))
    
    # Plot Elbow and Silhouette
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(K_range, inertias, 'bo-', label='Inertia')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    ax2.plot(K_range, sil_scores, 'rs-', label='Silhouette Score')
    ax2.set_ylabel('Silhouette Score', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('Elbow Method and Silhouette Score for Optimal K')
    plt.tight_layout()
    plt.savefig(plot_dir / "kmeans_evaluation.png")
    plt.close()
    
    # Select K based on highest silhouette score
    optimal_k = K_range[np.argmax(sil_scores)]
    print(f"Optimal number of clusters selected: K={optimal_k}")
    
    # Fit final KMeans model
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    macro_df['Regime'] = kmeans.fit_predict(X_scaled)
    
    # Characterize the regimes
    regime_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
    regime_centers.index.name = 'Regime'
    print("\nRegime Cluster Centers:")
    print(regime_centers)
    
    # Count of months per regime
    print("\nMonths per Regime:")
    print(macro_df['Regime'].value_counts())
    
    # Plot Regimes over Time
    plt.figure(figsize=(14, 6))
    sns.scatterplot(data=macro_df, x='Month', y='FEDFUNDS', hue='Regime', palette='tab10', s=100)
    plt.plot(macro_df['Month'], macro_df['FEDFUNDS'], color='grey', alpha=0.5, zorder=1)
    plt.title("Economic Regimes over Time (Mapped against FEDFUNDS)")
    plt.ylabel("Federal Funds Rate")
    plt.xlabel("Year")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "regimes_over_time_fedfunds.png")
    plt.close()
    
    # Merge the Regime labels back into the full integrated panel
    df = pd.merge(df, macro_df[['Month', 'Regime']], on='Month', how='left')
    
    # Save the final dataset with Regimes
    output_file = base_dir / "Dataset" / "integrated_panel_with_regimes.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved final dataset with Regimes to {output_file}")

if __name__ == "__main__":
    main()
