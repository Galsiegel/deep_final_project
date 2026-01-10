"""
Data loading and preprocessing utilities for stock market data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from datetime import datetime, timedelta


class StockDataLoader:
    """
    Handles loading and preprocessing stock market data from CSV files.
    
    Implements strict temporal splits to prevent data leakage.
    """
    
    def __init__(self, data_dir: str = "data/2017_2019"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing stock CSV files
        """
        self.data_dir = Path(data_dir)
        self.data = None
        self.tickers = []
        
    def load_data(self, tickers: Optional[List[str]] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load stock data from CSV files.
        
        Args:
            tickers: List of ticker symbols to load. If None, loads all available.
            limit: Maximum number of tickers to load (for quick testing)
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume, Ticker
        """
        csv_files = sorted(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")
        
        # Filter by tickers if specified
        if tickers:
            csv_files = [f for f in csv_files if self._extract_ticker(f.name) in tickers]
        
        # Limit number of files if specified
        if limit:
            csv_files = csv_files[:limit]
        
        print(f"Loading {len(csv_files)} stock files...")
        
        all_data = []
        for csv_file in csv_files:
            try:
                ticker = self._extract_ticker(csv_file.name)
                df = pd.read_csv(csv_file)
                
                # Clean and standardize columns
                df = df.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # Parse dates and remove timezone info for consistency
                df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
                
                # Add ticker column
                df['ticker'] = ticker
                
                # Select relevant columns
                df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']]
                
                # Remove any rows with missing values
                df = df.dropna()
                
                # Sort by date
                df = df.sort_values('date').reset_index(drop=True)
                
                all_data.append(df)
                self.tickers.append(ticker)
                
            except Exception as e:
                print(f"ERROR loading {csv_file.name} ({ticker}): {e}")
                if 'df' in locals():
                    print(f"  Columns found: {df.columns.tolist()}")
                    print(f"  After rename attempt: {df.columns.tolist()}")
                continue  # Skip this file and continue with others
        
        self.data = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(self.data)} records for {len(self.tickers)} tickers")
        print(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        
        return self.data
    
    def _extract_ticker(self, filename: str) -> str:
        """Extract ticker symbol from filename (e.g., '2017_2019_AAPL_data.csv' -> 'AAPL')"""
        parts = filename.replace('.csv', '').split('_')
        # Format: YYYY_YYYY_TICKER_data.csv
        if len(parts) >= 3:
            return parts[2]
        return filename.replace('.csv', '')
    
    def temporal_split(
        self, 
        df: pd.DataFrame,
        train_end: str,
        val_start: str,
        val_end: str,
        test_start: str,
        lookback_days: int = 20
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform strict temporal split with gaps to prevent data leakage.
        
        Args:
            df: DataFrame with stock data
            train_end: End date for training (format: 'YYYY-MM-DD')
            val_start: Start date for validation (includes gap after train)
            val_end: End date for validation
            test_start: Start date for test (includes gap after val)
            lookback_days: Number of days in lookback window (used for gap)
            
        Returns:
            train_df, val_df, test_df
        """
        train_end_dt = pd.to_datetime(train_end)
        val_start_dt = pd.to_datetime(val_start)
        val_end_dt = pd.to_datetime(val_end)
        test_start_dt = pd.to_datetime(test_start)
        
        # Verify gaps
        gap1 = (val_start_dt - train_end_dt).days
        gap2 = (test_start_dt - val_end_dt).days
        
        print(f"\nTemporal Split Configuration:")
        print(f"  Train: up to {train_end}")
        print(f"  Gap 1: {gap1} days (recommended: >= {lookback_days})")
        print(f"  Val: {val_start} to {val_end}")
        print(f"  Gap 2: {gap2} days (recommended: >= {lookback_days})")
        print(f"  Test: from {test_start}")
        
        if gap1 < lookback_days:
            print(f"  ⚠️  Warning: Gap 1 ({gap1} days) < lookback window ({lookback_days} days)")
        if gap2 < lookback_days:
            print(f"  ⚠️  Warning: Gap 2 ({gap2} days) < lookback window ({lookback_days} days)")
        
        # Split data
        train_df = df[df['date'] <= train_end_dt].copy()
        val_df = df[(df['date'] >= val_start_dt) & (df['date'] <= val_end_dt)].copy()
        test_df = df[df['date'] >= test_start_dt].copy()
        
        print(f"\nSplit sizes:")
        print(f"  Train: {len(train_df)} records")
        print(f"  Val:   {len(val_df)} records")
        print(f"  Test:  {len(test_df)} records")
        
        return train_df, val_df, test_df
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns (percentage change) for each ticker.
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            DataFrame with added 'return' column
        """
        df = df.copy()
        df['return'] = df.groupby('ticker')['close'].pct_change() * 100
        return df
    
    def create_classification_targets(
        self, 
        df: pd.DataFrame, 
        thresholds: Dict[str, List[float]]
    ) -> pd.DataFrame:
        """
        Create classification targets based on next-day return.
        
        Args:
            df: DataFrame with 'return' column
            thresholds: Dict with keys '3class', '5class', '7class' and threshold lists
                       Example: {'5class': [-2, -0.5, 0.5, 2]}
                       
        Returns:
            DataFrame with added classification target columns
        """
        df = df.copy()
        
        # Shift returns to get next-day return
        df['next_day_return'] = df.groupby('ticker')['return'].shift(-1)
        
        # 3-class: Down, Neutral, Up
        if '3class' in thresholds:
            # Use a small epsilon for neutral class around zero
            epsilon = 0.01  # +/- 0.01%
            df['target_3class'] = pd.cut(
                df['next_day_return'],
                bins=[-np.inf, -epsilon, epsilon, np.inf],
                labels=[0, 1, 2],  # Down, Neutral, Up
                include_lowest=True
            )
        
        # 5-class: Large Down, Small Down, Neutral, Small Up, Large Up
        if '5class' in thresholds:
            bins = [-np.inf] + thresholds['5class'] + [np.inf]
            df['target_5class'] = pd.cut(
                df['next_day_return'],
                bins=bins,
                labels=[0, 1, 2, 3, 4],
                include_lowest=True
            )
        
        # 7-class
        if '7class' in thresholds:
            bins = [-np.inf] + thresholds['7class'] + [np.inf]
            df['target_7class'] = pd.cut(
                df['next_day_return'],
                bins=bins,
                labels=[0, 1, 2, 3, 4, 5, 6],
                include_lowest=True
            )
        
        return df
    
    def get_ticker_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Get data for a specific ticker."""
        return df[df['ticker'] == ticker].copy()


def get_default_split_dates() -> Dict[str, str]:
    """
    Get default temporal split dates for 2017-2019 dataset.
    
    Returns:
        Dictionary with split dates
    """
    return {
        'train_end': '2018-06-30',
        'val_start': '2018-07-21',  # 21-day gap
        'val_end': '2018-12-31',
        'test_start': '2019-01-21',  # 21-day gap
    }


if __name__ == "__main__":
    # Quick test
    loader = StockDataLoader()
    data = loader.load_data(limit=5)
    print(f"\nSample data:\n{data.head()}")
    print(f"\nTickers loaded: {loader.tickers}")

