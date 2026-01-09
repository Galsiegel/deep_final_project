"""
Clean and filter FNSPID dataset.

Pipeline:
1. Load and clean stock prices (remove invalid data, convert to UTC)
2. Load and clean news data (remove invalid data, convert to UTC)
3. Filter by date range and top companies
4. Select top N articles per day per ticker

Output: Cleaned CSV files ready for embedding generation.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = Path(__file__).parent.parent / "data" / "fnspid"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

START_DATE = "2022-01-01"
END_DATE = "2022-12-31"
TOP_N_COMPANIES = 10
TOP_N_ARTICLES_PER_DAY = 5

# ============================================================================
# STOCK PRICE PROCESSING
# ============================================================================

def clean_stock_prices(df):
    """
    Clean stock price data (based on original repo's preprocessing):
    - Remove invalid price data (negative values, zeros)
    - Convert dates to UTC
    - Remove duplicates
    """
    df = df.copy()
    
    # Remove invalid OHLCV data
    price_cols = ['open', 'high', 'low', 'close', 'adj close']
    for col in price_cols:
        if col in df.columns:
            # Remove rows with invalid prices (negative or zero)
            df = df[df[col] > 0]
    
    # Remove rows where high < low or close outside [low, high] range
    if all(c in df.columns for c in ['high', 'low', 'close']):
        df = df[df['high'] >= df['low']]
        df = df[(df['close'] >= df['low']) & (df['close'] <= df['high'])]
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['ticker', 'date'], keep='first')
    
    return df


def load_and_clean_stock_prices():
    """Load stock prices from CSV files and clean them."""
    print("\n" + "="*60)
    print("Loading Stock Prices")
    print("="*60)
    
    stock_dir = RAW_DIR / "stock_prices"
    csv_files = [f for f in stock_dir.glob("**/*.csv") if '__MACOSX' not in str(f)]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {stock_dir}")
    
    print(f"Found {len(csv_files)} CSV files")
    
    dfs = []
    for csv_file in tqdm(csv_files, desc="Loading"):
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            df['ticker'] = csv_file.stem  # Extract ticker from filename
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Skipped {csv_file.name}: {e}")
    
    if not dfs:
        raise ValueError("No valid stock price data found!")
    
    stock_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(stock_df):,} records")
    
    # Standardize column names
    stock_df.columns = stock_df.columns.str.lower().str.strip()
    if 'date' in stock_df.columns:
        stock_df['date'] = pd.to_datetime(stock_df['date'], errors='coerce')
        # Convert to UTC (original repo's approach)
        if stock_df['date'].dt.tz is None:
            stock_df['date'] = stock_df['date'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='shift_forward')
        else:
            stock_df['date'] = stock_df['date'].dt.tz_convert('UTC')
    
    # Clean data
    stock_df = clean_stock_prices(stock_df)
    print(f"After cleaning: {len(stock_df):,} records")
    
    return stock_df


def identify_top_companies(stock_df):
    """Identify top N companies by average volume (proxy for liquidity)."""
    print("\n" + "="*60)
    print(f"Identifying Top {TOP_N_COMPANIES} Companies")
    print("="*60)
    
    # Use average volume as proxy for company importance
    company_metrics = stock_df.groupby('ticker')['volume'].mean().sort_values(ascending=False)
    top_companies = company_metrics.head(TOP_N_COMPANIES).index.tolist()
    
    print(f"Top {TOP_N_COMPANIES} companies:")
    for i, ticker in enumerate(top_companies, 1):
        print(f"  {i:2d}. {ticker}")
    
    return top_companies


def filter_stock_prices(stock_df, top_companies):
    """Filter stock prices by date range and top companies."""
    print("\n" + "="*60)
    print("Filtering Stock Prices")
    print("="*60)
    
    # Filter by date
    mask = (stock_df['date'] >= START_DATE) & (stock_df['date'] <= END_DATE)
    filtered = stock_df[mask].copy()
    print(f"After date filter: {len(filtered):,} records")
    
    # Filter by companies
    mask = filtered['ticker'].isin(top_companies)
    filtered = filtered[mask].copy()
    print(f"After company filter: {len(filtered):,} records")
    
    # Select OHLCV columns
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    if 'adj close' in filtered.columns:
        ohlcv_cols.append('adj close')
    
    cols = ['ticker', 'date'] + [c for c in ohlcv_cols if c in filtered.columns]
    filtered = filtered[cols].copy()
    
    filtered = filtered.sort_values(['ticker', 'date']).reset_index(drop=True)
    print(f"Final: {len(filtered):,} records, {filtered['ticker'].nunique()} tickers")
    
    return filtered


# ============================================================================
# NEWS PROCESSING
# ============================================================================

def clean_news_data(df):
    """
    Clean news data (based on original repo's preprocessing):
    - Remove invalid/redundant news
    - Convert dates to UTC
    - Remove duplicates
    """
    df = df.copy()
    
    # Remove rows with missing text
    text_col = 'text' if 'text' in df.columns else 'Article'
    if text_col in df.columns:
        df = df[df[text_col].notna()]
        df = df[df[text_col].astype(str).str.strip() != '']
    
    # Remove duplicates
    if 'date' in df.columns and 'ticker' in df.columns:
        df = df.drop_duplicates(subset=['ticker', 'date', text_col], keep='first')
    
    return df


def load_and_clean_news():
    """Load news data and clean it."""
    print("\n" + "="*60)
    print("Loading News Data")
    print("="*60)
    
    news_file = RAW_DIR / "nasdaq_exteral_data.csv"
    if not news_file.exists():
        raise FileNotFoundError(f"News file not found: {news_file}")
    
    print(f"File size: {news_file.stat().st_size / (1024**2):.1f} MB")
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252']
    news_df = None
    
    for encoding in encodings:
        try:
            # Try C engine first (faster)
            try:
                news_df = pd.read_csv(news_file, encoding=encoding, low_memory=False, 
                                     engine='c', on_bad_lines='skip')
            except:
                # Fall back to Python engine (more forgiving)
                news_df = pd.read_csv(news_file, encoding=encoding, engine='python',
                                     on_bad_lines='skip', quoting=1)
            print(f"✓ Loaded with {encoding} encoding")
            break
        except Exception as e:
            continue
    
    if news_df is None:
        raise ValueError("Could not load news file")
    
    print(f"Loaded {len(news_df):,} records")
    
    # Map columns to standard names
    col_mapping = {
        'Stock_symbol': 'ticker',
        'Date': 'date',
        'Article': 'text',
        'Article_title': 'title'
    }
    # Only rename columns that exist
    rename_dict = {k: v for k, v in col_mapping.items() if k in news_df.columns}
    news_df.rename(columns=rename_dict, inplace=True)
    
    # Combine title + article if both exist
    if 'title' in news_df.columns and 'text' in news_df.columns:
        news_df['text'] = news_df['title'].fillna('') + ' ' + news_df['text'].fillna('')
    elif 'title' in news_df.columns and 'text' not in news_df.columns:
        news_df['text'] = news_df['title']
    
    # Convert date to UTC
    if 'date' in news_df.columns:
        news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
        if news_df['date'].dt.tz is None:
            news_df['date'] = news_df['date'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='shift_forward')
        else:
            news_df['date'] = news_df['date'].dt.tz_convert('UTC')
    
    # Clean data
    news_df = clean_news_data(news_df)
    print(f"After cleaning: {len(news_df):,} records")
    
    return news_df


def filter_news(news_df, top_companies):
    """Filter news by date, companies, and select top N per day."""
    print("\n" + "="*60)
    print("Filtering News Articles")
    print("="*60)
    
    # Filter by date
    mask = (news_df['date'] >= START_DATE) & (news_df['date'] <= END_DATE)
    filtered = news_df[mask].copy()
    print(f"After date filter: {len(filtered):,} records")
    
    # Normalize tickers
    filtered = filtered[filtered['ticker'].notna()].copy()
    filtered['ticker'] = filtered['ticker'].astype(str).str.strip().str.upper()
    
    # Filter by companies
    top_companies_upper = [t.upper().strip() for t in top_companies]
    mask = filtered['ticker'].isin(top_companies_upper)
    filtered = filtered[mask].copy()
    print(f"After company filter: {len(filtered):,} records")
    
    if len(filtered) == 0:
        print("⚠ WARNING: No articles found for selected companies!")
        return pd.DataFrame()
    
    # Select top N articles per day per ticker
    filtered['date_only'] = filtered['date'].dt.date
    filtered = filtered.sort_values(['ticker', 'date'], ascending=[True, False])
    
    top_articles = []
    for (ticker, date_only), group in tqdm(filtered.groupby(['ticker', 'date_only']), 
                                           desc="Selecting top articles"):
        top_articles.append(group.head(TOP_N_ARTICLES_PER_DAY))
    
    filtered = pd.concat(top_articles, ignore_index=True)
    print(f"After selecting top {TOP_N_ARTICLES_PER_DAY} per day: {len(filtered):,} records")
    
    return filtered[['ticker', 'date', 'text']].copy()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main cleaning pipeline."""
    print("="*60)
    print("FNSPID Dataset Cleaning")
    print("="*60)
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Top {TOP_N_COMPANIES} companies")
    print(f"Top {TOP_N_ARTICLES_PER_DAY} articles per day")
    print("="*60)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Stock prices
    stock_df = load_and_clean_stock_prices()
    top_companies = identify_top_companies(stock_df)
    filtered_stock = filter_stock_prices(stock_df, top_companies)
    
    stock_output = PROCESSED_DIR / "stock_prices_cleaned.csv"
    filtered_stock.to_csv(stock_output, index=False)
    print(f"\n✓ Saved: {stock_output}")
    
    # 2. News
    news_df = load_and_clean_news()
    filtered_news = filter_news(news_df, top_companies)
    
    if len(filtered_news) == 0:
        print("\n⚠ No news articles found. Exiting.")
        return
    
    news_output = PROCESSED_DIR / "news_cleaned.csv"
    filtered_news.to_csv(news_output, index=False)
    print(f"\n✓ Saved: {news_output}")
    
    print("\n" + "="*60)
    print("Cleaning Complete!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - {stock_output}")
    print(f"  - {news_output}")
    print(f"\nNext: Run embed_fnspid.py to generate embeddings")


if __name__ == "__main__":
    main()

