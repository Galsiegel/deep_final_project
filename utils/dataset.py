"""
PyTorch Dataset for stock price sequences.

Creates sliding window sequences from stock data with proper normalization.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path


class StockSequenceDataset(Dataset):
    """
    Creates sequences of stock data for time series prediction.
    
    Each sample is:
        X: [lookback, num_features] - Historical OHLCV data
        y: scalar or class - Target (price or direction)
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        lookback: int,
        features: List[str],
        task: str,
        stride: int = 1,
        class_thresholds: Optional[List[float]] = None,
        scalers: Optional[Dict[str, StandardScaler]] = None,
        fit_scalers: bool = True,
        volume_transform: str = 'log'
    ):
        """
        Initialize dataset.
        
        Args:
            data: DataFrame with columns [date, open, high, low, close, volume, ticker]
            lookback: Number of days to look back
            features: List of feature names to use (e.g., ['open', 'high', 'low', 'close', 'volume'])
            task: 'regression' or 'classification'
            stride: Step size for sliding window
            class_thresholds: Thresholds for classification (e.g., [-2, -0.5, 0.5, 2])
            scalers: Pre-fitted scalers (for val/test sets)
            fit_scalers: Whether to fit new scalers (True for train, False for val/test)
            volume_transform: 'log' or 'none' for volume preprocessing
        """
        self.lookback = lookback
        self.features = features
        self.task = task
        self.stride = stride
        self.class_thresholds = class_thresholds
        self.volume_transform = volume_transform
        
        # Store scalers per ticker
        self.scalers = scalers if scalers is not None else {}
        
        # Process data
        self.sequences = []
        self.targets = []
        self.ticker_indices = []  # Track which ticker each sequence belongs to
        
        # Process each ticker separately
        for ticker in data['ticker'].unique():
            ticker_data = data[data['ticker'] == ticker].sort_values('date').reset_index(drop=True)
            
            # Need at least lookback + 1 days (to have a target)
            if len(ticker_data) < lookback + 1:
                continue
            
            # Apply volume transform
            ticker_data = ticker_data.copy()
            if volume_transform == 'log' and 'volume' in self.features:
                ticker_data['volume'] = np.log1p(ticker_data['volume'])  # log(1 + volume)
            
            # Normalize per ticker
            if fit_scalers:
                scaler = StandardScaler()
                ticker_data[features] = scaler.fit_transform(ticker_data[features])
                self.scalers[ticker] = scaler
            else:
                if ticker not in self.scalers:
                    raise ValueError(f"No scaler found for ticker {ticker}")
                ticker_data[features] = self.scalers[ticker].transform(ticker_data[features])
            
            # Create sequences
            # For classification, need i+1 to exist for next-day target
            max_idx = len(ticker_data) - 1 if task == 'classification' else len(ticker_data)
            
            for i in range(lookback, max_idx, stride):
                # Get sequence of historical data
                seq = ticker_data[features].iloc[i-lookback:i].values
                
                # Get target
                if task == 'regression':
                    # Predict normalized next-day close price
                    target = ticker_data['close'].iloc[i]
                    
                elif task == 'classification':
                    # Calculate NEXT-DAY % return and convert to class
                    # Today's close (end of input sequence)
                    today_close = ticker_data['close'].iloc[i]  # Already normalized
                    # Tomorrow's close (what we want to predict)
                    tomorrow_close = ticker_data['close'].iloc[i+1]  # Already normalized
                    
                    # Denormalize to calculate actual % return
                    # Get the close price index
                    close_idx = features.index('close')
                    scaler = self.scalers[ticker]
                    
                    # Denormalize using inverse transform
                    today_close_denorm = scaler.inverse_transform([[0]*close_idx + [today_close] + [0]*(len(features)-close_idx-1)])[0][close_idx]
                    tomorrow_close_denorm = scaler.inverse_transform([[0]*close_idx + [tomorrow_close] + [0]*(len(features)-close_idx-1)])[0][close_idx]
                    
                    # Calculate next-day return
                    pct_return = ((tomorrow_close_denorm - today_close_denorm) / today_close_denorm) * 100
                    
                    # Convert to class
                    target = self._return_to_class(pct_return)
                
                self.sequences.append(seq)
                self.targets.append(target)
                self.ticker_indices.append(ticker)
    
    def _return_to_class(self, pct_return: float) -> int:
        """Convert percentage return to class label."""
        if self.class_thresholds is None:
            raise ValueError("class_thresholds required for classification")
        
        # Find which bin the return falls into
        for i, threshold in enumerate(self.class_thresholds):
            if pct_return < threshold:
                return i
        return len(self.class_thresholds)  # Last class
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get one sample.
        
        Returns:
            X: [lookback, num_features] tensor
            y: scalar tensor (regression) or class index (classification)
        """
        X = torch.FloatTensor(self.sequences[idx])
        
        if self.task == 'regression':
            y = torch.FloatTensor([self.targets[idx]])
        else:  # classification
            y = torch.LongTensor([self.targets[idx]])[0]  # CrossEntropy expects class index
        
        return X, y
    
    def get_scalers(self) -> Dict[str, StandardScaler]:
        """Return fitted scalers for saving."""
        return self.scalers
    
    def get_class_weights(self) -> Optional[torch.Tensor]:
        """
        Calculate class weights for handling imbalance.
        Only for classification task.
        """
        if self.task != 'classification':
            return None
        
        # Count samples per class
        targets = np.array(self.targets)
        classes, counts = np.unique(targets, return_counts=True)
        
        # Calculate weights (inverse frequency)
        weights = len(targets) / (len(classes) * counts)
        
        return torch.FloatTensor(weights)


def save_scalers(scalers: Dict[str, StandardScaler], path: str):
    """Save scalers to file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(scalers, f)


def load_scalers(path: str) -> Dict[str, StandardScaler]:
    """Load scalers from file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    # Quick test
    print("Testing StockSequenceDataset...")
    
    from utils.data_utils import StockDataLoader, get_default_split_dates
    
    # Load data
    loader = StockDataLoader()
    data = loader.load_data(limit=2)
    data = loader.calculate_returns(data)
    
    # Create dataset
    dataset = StockSequenceDataset(
        data=data,
        lookback=10,
        features=['open', 'high', 'low', 'close', 'volume'],
        task='regression',
        fit_scalers=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample shape: X={dataset[0][0].shape}, y={dataset[0][1].shape}")
    print(f"Number of scalers: {len(dataset.get_scalers())}")

