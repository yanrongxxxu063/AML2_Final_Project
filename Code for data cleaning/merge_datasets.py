import pandas as pd
from pathlib import Path

def main():
    base_dir = Path(r"c:\Users\jesse\CUcourses\Applied Machine Learning 2\Group Project")
    
    # 1. Load Housing Data
    print("Loading housing data...")
    housing_file = base_dir / "Dataset" / "Section 2 -Housing Market Heterogeneity" / "cleaned_data.csv"
    housing_df = pd.read_csv(housing_file)
    
    # Clean up housing data for aggregation
    housing_df['Sale Date'] = pd.to_datetime(housing_df['Sale Date'], errors='coerce')
    housing_df = housing_df.dropna(subset=['Sale Date', 'Sale Price', 'Sub-Nbhood', 'Beds'])
    
    # Create month identifier (1st of the month) for merging
    housing_df['Month'] = housing_df['Sale Date'].dt.to_period('M').dt.to_timestamp()
    
    # Aggregate transaction-level data to Segment-level monthly data
    # Segment = Month + Sub-Nbhood + Beds
    segment_df = housing_df.groupby(['Month', 'Sub-Nbhood', 'Beds']).agg(
        Median_Sale_Price=('Sale Price', 'median'),
        Avg_Sale_Price=('Sale Price', 'mean'),
        Transaction_Count=('Sale Price', 'count')
    ).reset_index()
    print(f"Aggregated housing segments shape: {segment_df.shape}")
    
    # 2. Load Macroeconomic Data
    print("Loading macroeconomic data...")
    macro_file = base_dir / "Dataset" / "Section 1-Economic regime detection" / "macro_regimes_ready.csv"
    macro_df = pd.read_csv(macro_file)
    macro_df['Month'] = pd.to_datetime(macro_df['observation_date'])
    macro_df = macro_df.drop(columns=['observation_date'])
    
    # 3. Load Sentiment Data
    print("Loading sentiment data...")
    sentiment_file = base_dir / "Dataset" / "Section 3- Sentiment Analysis" / "fomc_sentiment_monthly.csv"
    sentiment_df = pd.read_csv(sentiment_file)
    sentiment_df['Month'] = pd.to_datetime(sentiment_df['month'])
    sentiment_df = sentiment_df.drop(columns=['month'])
    
    # 4. Integrate Datasets
    print("Merging datasets...")
    # Merge macro data into segment data
    merged_df = pd.merge(segment_df, macro_df, on='Month', how='inner')
    
    # Forward-fill sentiment data to handle months without FOMC meetings
    all_months = pd.DataFrame({'Month': pd.date_range(start=merged_df['Month'].min(), end=merged_df['Month'].max(), freq='MS')})
    sentiment_full = pd.merge(all_months, sentiment_df, on='Month', how='left').sort_values('Month').ffill()
    
    # Merge sentiment data into the panel
    final_df = pd.merge(merged_df, sentiment_full, on='Month', how='inner')
    
    # Display final structure
    print(f"Final Integrated Panel Shape: {final_df.shape}")
    print("Columns:", list(final_df.columns))
    
    # Save the integrated panel
    output_file = base_dir / "Dataset" / "integrated_panel.csv"
    final_df.to_csv(output_file, index=False)
    print(f"Saved integrated panel to {output_file}")

if __name__ == "__main__":
    main()
