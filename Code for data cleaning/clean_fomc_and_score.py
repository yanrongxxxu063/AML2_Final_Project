import pandas as pd
import re
import pysentiment2 as ps

def clean_text(text):
    if pd.isna(text):
        return ""
    # Strip URLs
    text = re.sub(r'http\S+|www\.\S+', '', str(text))
    # Strip HTML entities if any left
    text = re.sub(r'&\w+;', ' ', text)
    # Split by line and keep lines >= 40 chars
    lines = text.split('\n')
    valid_lines = [line.strip() for line in lines if len(line.strip()) >= 40]
    cleaned = ' '.join(valid_lines)
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def main():
    print("Loading raw FOMC data...")
    df = pd.read_csv(r"c:\Users\jesse\CUcourses\Applied Machine Learning 2\Group Project\Dataset\Section 3- Sentiment Analysis\fomc_raw.csv")
    
    print(f"Initial shape: {df.shape}")
    
    # Clean text
    print("Cleaning text...")
    df['text_clean'] = df['text'].apply(clean_text)
    
    # Calculate word count
    df['word_count'] = df['text_clean'].apply(lambda x: len(x.split()))
    
    # Filter documents < 100 words
    df = df[df['word_count'] >= 100].copy()
    print(f"Shape after dropping short docs: {df.shape}")
    
    # Sentiment scoring
    print("Scoring sentiment with Loughran and McDonald dictionary...")
    lm = ps.LM()
    
    def get_sentiment(text):
        tokens = lm.tokenize(text)
        score = lm.get_score(tokens)
        return pd.Series([score['Positive'], score['Negative'], score['Polarity'], score['Subjectivity']])
        
    df[['LM_Positive', 'LM_Negative', 'LM_Polarity', 'LM_Subjectivity']] = df['text_clean'].apply(get_sentiment)
    
    # Save cleaned documents
    out_cleaned = r"c:\Users\jesse\CUcourses\Applied Machine Learning 2\Group Project\Dataset\Section 3- Sentiment Analysis\fomc_cleaned.csv"
    df.to_csv(out_cleaned, index=False)
    print(f"Saved cleaned FOMC data to {out_cleaned}")
    
    # Aggregate to monthly level by meeting_date
    print("Aggregating to monthly level...")
    df['meeting_date'] = pd.to_datetime(df['meeting_date'])
    df['month'] = df['meeting_date'].dt.to_period('M')
    
    monthly_sentiment = df.groupby('month').agg({
        'LM_Polarity': 'mean',
        'LM_Subjectivity': 'mean',
        'LM_Positive': 'sum',
        'LM_Negative': 'sum'
    }).reset_index()
    monthly_sentiment['month'] = monthly_sentiment['month'].dt.to_timestamp()
    
    out_monthly = r"c:\Users\jesse\CUcourses\Applied Machine Learning 2\Group Project\Dataset\Section 3- Sentiment Analysis\fomc_sentiment_monthly.csv"
    monthly_sentiment.to_csv(out_monthly, index=False)
    print(f"Saved monthly sentiment to {out_monthly}")

if __name__ == "__main__":
    main()
