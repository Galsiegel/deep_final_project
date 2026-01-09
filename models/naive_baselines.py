"""
Naive baseline prediction methods for stock price forecasting.

These simple methods establish a performance floor that neural networks must beat.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.linear_model import LinearRegression


class NaivePredictor:
    """Base class for naive prediction methods."""
    
    def __init__(self, name: str):
        self.name = name
        
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for next-day closing price.
        
        Args:
            df: DataFrame with stock data (must have 'close' column)
            
        Returns:
            DataFrame with added 'prediction' column
        """
        raise NotImplementedError


class PersistencePredictor(NaivePredictor):
    """
    Persistence (Last Value) Baseline.
    
    Predicts tomorrow's price will be the same as today's price.
    This is often surprisingly hard to beat in stock prediction!
    """
    
    def __init__(self):
        super().__init__("Persistence (t-1)")
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict tomorrow = today."""
        df = df.copy()
        df['prediction'] = df.groupby('ticker')['close'].shift(1)
        return df


class MovingAveragePredictor(NaivePredictor):
    """
    Moving Average Baseline.
    
    Predicts tomorrow's price as the average of the last N days.
    """
    
    def __init__(self, window: int = 5):
        super().__init__(f"Moving Average ({window}-day)")
        self.window = window
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict tomorrow = mean of last N days."""
        df = df.copy()
        df['prediction'] = df.groupby('ticker')['close'].transform(
            lambda x: x.shift(1).rolling(window=self.window, min_periods=1).mean()
        )
        return df


class LinearTrendPredictor(NaivePredictor):
    """
    Linear Trend Extrapolation Baseline.
    
    Fits a linear regression to the last N days and extrapolates to predict tomorrow.
    """
    
    def __init__(self, window: int = 5):
        super().__init__(f"Linear Trend ({window}-day)")
        self.window = window
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict tomorrow by extrapolating linear trend."""
        df = df.copy()
        predictions = []
        
        # Group by ticker to handle each stock separately
        for ticker, group in df.groupby('ticker'):
            ticker_predictions = []
            closes = group['close'].values
            
            for i in range(len(closes)):
                if i < self.window:
                    # Not enough history - use persistence
                    pred = closes[i-1] if i > 0 else closes[i]
                else:
                    # Fit linear regression to last N days
                    window_data = closes[i-self.window:i]
                    X = np.arange(self.window).reshape(-1, 1)
                    y = window_data
                    
                    try:
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        # Predict next point (extrapolate)
                        pred = model.predict([[self.window]])[0]
                        
                        # Sanity check: bound prediction to reasonable range
                        # (within 20% of recent average to avoid extreme outliers)
                        recent_avg = window_data.mean()
                        max_change = recent_avg * 0.2
                        pred = np.clip(pred, recent_avg - max_change, recent_avg + max_change)
                    except:
                        # Fall back to persistence if fitting fails
                        pred = closes[i-1]
                
                ticker_predictions.append(pred)
            
            predictions.extend(ticker_predictions)
        
        df['prediction'] = predictions
        return df


class RandomClassifier:
    """
    Random Classifier Baseline.
    
    Predicts each class with equal probability (chance-level performance).
    Only used for classification tasks.
    """
    
    def __init__(self, n_classes: int, seed: int = 42):
        self.name = f"Random Classifier ({n_classes}-class)"
        self.n_classes = n_classes
        self.seed = seed
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate random class predictions."""
        df = df.copy()
        np.random.seed(self.seed)
        df['prediction'] = np.random.randint(0, self.n_classes, size=len(df))
        return df


def predict_with_all_baselines(
    df: pd.DataFrame,
    methods: str = 'all'
) -> Dict[str, pd.DataFrame]:
    """
    Generate predictions using all naive baseline methods.
    
    Args:
        df: DataFrame with stock data
        methods: Which methods to use ('all', 'regression', 'classification')
        
    Returns:
        Dictionary mapping method name to DataFrame with predictions
    """
    results = {}
    
    if methods in ['all', 'regression']:
        # Regression baselines
        predictors = [
            PersistencePredictor(),
            MovingAveragePredictor(window=5),
            LinearTrendPredictor(window=5)
        ]
        
        for predictor in predictors:
            print(f"Running {predictor.name}...")
            results[predictor.name] = predictor.predict(df)
    
    if methods in ['all', 'classification']:
        # Classification baselines (random guessing)
        for n_classes in [3, 5, 7]:
            classifier = RandomClassifier(n_classes)
            print(f"Running {classifier.name}...")
            results[classifier.name] = classifier.predict(df)
    
    return results


def calculate_classification_from_regression(
    predictions: pd.DataFrame,
    thresholds: Dict[str, list]
) -> pd.DataFrame:
    """
    Convert regression predictions to classification predictions.
    
    Args:
        predictions: DataFrame with 'prediction' column (predicted prices)
        thresholds: Classification thresholds
        
    Returns:
        DataFrame with added classification prediction columns
    """
    df = predictions.copy()
    
    # Calculate predicted return
    df['predicted_return'] = ((df['prediction'] - df['close']) / df['close']) * 100
    
    # Convert to classes
    if '3class' in thresholds:
        df['pred_3class'] = pd.cut(
            df['predicted_return'],
            bins=[-np.inf, 0, 0, np.inf],
            labels=[0, 1, 2],
            include_lowest=True
        )
    
    if '5class' in thresholds:
        bins = [-np.inf] + thresholds['5class'] + [np.inf]
        df['pred_5class'] = pd.cut(
            df['predicted_return'],
            bins=bins,
            labels=[0, 1, 2, 3, 4],
            include_lowest=True
        )
    
    if '7class' in thresholds:
        bins = [-np.inf] + thresholds['7class'] + [np.inf]
        df['pred_7class'] = pd.cut(
            df['predicted_return'],
            bins=bins,
            labels=[0, 1, 2, 3, 4, 5, 6],
            include_lowest=True
        )
    
    return df


if __name__ == "__main__":
    # Quick test
    print("Testing naive predictors...")
    
    # Create sample data
    dates = pd.date_range('2019-01-01', periods=20)
    sample_data = pd.DataFrame({
        'date': dates,
        'close': [100 + i * 2 + np.random.randn() * 2 for i in range(20)],
        'ticker': 'TEST'
    })
    
    # Test each predictor
    predictors = [
        PersistencePredictor(),
        MovingAveragePredictor(5),
        LinearTrendPredictor(5)
    ]
    
    for pred in predictors:
        result = pred.predict(sample_data)
        print(f"\n{pred.name}:")
        print(result[['date', 'close', 'prediction']].tail())

