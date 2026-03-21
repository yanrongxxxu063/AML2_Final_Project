import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from pathlib import Path
import os

def main():
    base_dir = Path(r"c:\Users\jesse\CUcourses\Applied Machine Learning 2\Group Project")
    data_file = base_dir / "Dataset" / "integrated_panel_with_regimes.csv"
    
    # Create an output directory for plots
    plot_dir = base_dir / "Plots" / "Phase3"
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    df['Month'] = pd.to_datetime(df['Month'])
    
    # Drop sub-neighborhoods with very few transactions to keep visuals cleaner
    top_neighborhoods = df['Sub-Nbhood'].value_counts().nlargest(5).index
    df_top = df[df['Sub-Nbhood'].isin(top_neighborhoods)].copy()
    
    # 1. Visualization: Boxplots of Median Sale Price across Regimes (Overall)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Regime', y='Median_Sale_Price', palette='Set2')
    plt.yscale('log')
    plt.title("Distribution of Median Sale Prices by Economic Regime (Log Scale)")
    plt.ylabel("Log(Median Sale Price)")
    plt.xlabel("Economic Regime")
    plt.tight_layout()
    plt.savefig(plot_dir / "price_by_regime_boxplot.png")
    plt.close()
    
    # 2. Visualization: Price by Regime separated by Bedrooms
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Beds', y='Median_Sale_Price', hue='Regime', palette='Set2')
    plt.yscale('log')
    plt.title("Median Sale Prices by Bedroom Count across Regimes (Log Scale)")
    plt.ylabel("Log(Median Sale Price)")
    plt.xlabel("Number of Bedrooms")
    plt.legend(title='Regime')
    plt.tight_layout()
    plt.savefig(plot_dir / "price_by_beds_and_regime.png")
    plt.close()

    # 3. Visualization: Price by Regime in Top 5 Neighborhoods
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=df_top, x='Sub-Nbhood', y='Median_Sale_Price', hue='Regime', palette='Set2')
    plt.yscale('log')
    plt.title("Median Sale Prices in Top 5 Neighborhoods across Regimes (Log Scale)")
    plt.ylabel("Log(Median Sale Price)")
    plt.xlabel("Sub-Neighborhood")
    plt.xticks(rotation=45)
    plt.legend(title='Regime')
    plt.tight_layout()
    plt.savefig(plot_dir / "price_by_neighborhood_and_regime.png")
    plt.close()

    # 4. Statistical Testing: ANOVA
    print("\nRunning ANOVA: Median_Sale_Price ~ C(Regime) + C(Beds) + C(Regime):C(Beds)")
    
    # Prepare data for OLS (handle valid values only)
    model_data = df.dropna(subset=['Median_Sale_Price', 'Regime', 'Beds']).copy()
    model_data['Log_Price'] = np.log(model_data['Median_Sale_Price'])
    model_data['Regime'] = model_data['Regime'].astype(str)
    model_data['Beds'] = model_data['Beds'].astype(str)
    
    # Fit OLS model with interaction terms
    model = ols('Log_Price ~ C(Regime) + C(Beds) + C(Regime):C(Beds)', data=model_data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    
    # Save the ANOVA results to CSV
    anova_file = base_dir / "Dataset" / "phase3_anova_results.csv"
    anova_table.to_csv(anova_file)
    print(f"\nSaved ANOVA results to {anova_file}")
    
    print("\nPhase 3 Analysis Complete. Check the 'Plots/Phase3' directory for visualizations.")

if __name__ == "__main__":
    main()
