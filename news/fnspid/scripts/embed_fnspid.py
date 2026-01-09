"""
Generate FinBERT embeddings for cleaned FNSPID dataset.

Pipeline:
1. Load cleaned stock prices and news
2. Generate FinBERT embeddings per article
3. Aggregate embeddings per day (mean pooling)
4. Combine OHLCV + news embeddings into final dataset

Input: Cleaned CSV files from clean_fnspid.py
Output: Final dataset CSV with embeddings
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
from transformers import AutoTokenizer, AutoModel
import torch

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = Path(__file__).parent.parent / "data" / "fnspid"
PROCESSED_DIR = DATA_DIR / "processed"

FINBERT_MODEL = "ProsusAI/finbert"
EMBEDDING_DIM = 768
BATCH_SIZE = 32

# ============================================================================
# EMBEDDING GENERATION
# ============================================================================

def generate_embeddings(news_df):
    """Generate FinBERT embeddings for each article."""
    print("\n" + "="*60)
    print("Generating FinBERT Embeddings")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading model: {FINBERT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModel.from_pretrained(FINBERT_MODEL)
    model.to(device)
    model.eval()
    
    texts = news_df['text'].tolist()
    embeddings = []
    
    print(f"Processing {len(texts):,} articles in batches of {BATCH_SIZE}...")
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Generating embeddings"):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, 
                          truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        embeddings.append(batch_embeddings)
    
    all_embeddings = np.vstack(embeddings)
    embedding_cols = [f'embedding_{i}' for i in range(EMBEDDING_DIM)]
    news_df[embedding_cols] = pd.DataFrame(all_embeddings, index=news_df.index)
    
    print(f"✓ Generated {len(all_embeddings):,} embeddings")
    return news_df


def aggregate_embeddings_per_day(news_df):
    """Aggregate article embeddings per day per ticker (mean pooling)."""
    print("\n" + "="*60)
    print("Aggregating Embeddings Per Day")
    print("="*60)
    
    embedding_cols = [c for c in news_df.columns if c.startswith('embedding_')]
    news_df['date_only'] = pd.to_datetime(news_df['date']).dt.date
    
    aggregated = []
    for (ticker, date_only), group in tqdm(news_df.groupby(['ticker', 'date_only']),
                                           desc="Aggregating"):
        mean_embeddings = group[embedding_cols].mean().values
        aggregated.append({
            'ticker': ticker,
            'date': pd.to_datetime(date_only),
            'num_articles': len(group),
            **{f'news_embedding_{i}': mean_embeddings[i] for i in range(len(mean_embeddings))}
        })
    
    result = pd.DataFrame(aggregated)
    print(f"✓ Aggregated to {len(result):,} daily records")
    return result


# ============================================================================
# INTEGRATION
# ============================================================================

def combine_stock_and_news(stock_df, news_df):
    """Combine stock prices with aggregated news embeddings."""
    print("\n" + "="*60)
    print("Combining Stock and News Data")
    print("="*60)
    
    stock_df['date_only'] = pd.to_datetime(stock_df['date']).dt.date
    news_df['date_only'] = pd.to_datetime(news_df['date']).dt.date
    
    merged = stock_df.merge(news_df, on=['ticker', 'date_only'], how='left', suffixes=('', '_news'))
    
    # Fill missing embeddings with zeros
    embedding_cols = [c for c in merged.columns if c.startswith('news_embedding_')]
    merged[embedding_cols] = merged[embedding_cols].fillna(0)
    merged['num_articles'] = merged['num_articles'].fillna(0)
    
    merged = merged.drop(columns=['date_only', 'date_news'], errors='ignore')
    merged = merged.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    print(f"✓ Final dataset: {len(merged):,} records")
    print(f"  With news: {merged['num_articles'].gt(0).sum():,}")
    print(f"  Without news: {merged['num_articles'].eq(0).sum():,}")
    
    return merged


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main embedding pipeline."""
    print("="*60)
    print("FNSPID Dataset Embedding Generation")
    print("="*60)
    
    # Load cleaned data
    stock_file = PROCESSED_DIR / "stock_prices_cleaned.csv"
    news_file = PROCESSED_DIR / "news_cleaned.csv"
    
    if not stock_file.exists():
        raise FileNotFoundError(f"Cleaned stock file not found: {stock_file}\n"
                              f"Run clean_fnspid.py first!")
    
    if not news_file.exists():
        raise FileNotFoundError(f"Cleaned news file not found: {news_file}\n"
                              f"Run clean_fnspid.py first!")
    
    print(f"Loading cleaned data...")
    stock_df = pd.read_csv(stock_file)
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    
    news_df = pd.read_csv(news_file)
    news_df['date'] = pd.to_datetime(news_df['date'])
    
    print(f"  Stock records: {len(stock_df):,}")
    print(f"  News records: {len(news_df):,}")
    
    # Generate embeddings
    news_with_embeddings = generate_embeddings(news_df.copy())
    
    # Aggregate per day
    aggregated_news = aggregate_embeddings_per_day(news_with_embeddings)
    
    # Combine
    final_dataset = combine_stock_and_news(stock_df, aggregated_news)
    
    # Save
    final_output = PROCESSED_DIR / "fnspid_dataset.csv"
    final_dataset.to_csv(final_output, index=False)
    print(f"\n✓ Saved: {final_output}")
    print(f"  Shape: {final_dataset.shape}")
    
    print("\n" + "="*60)
    print("Embedding Generation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

