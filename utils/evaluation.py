"""
Evaluation metrics and comparison framework for stock predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import json


class RegressionMetrics:
    """Calculate regression metrics for stock price predictions."""
    
    @staticmethod
    def calculate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {
                'mse': np.nan,
                'mae': np.nan,
                'rmse': np.nan,
                'mape': np.nan,
                'directional_accuracy': np.nan
            }
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Directional Accuracy (did we predict up/down correctly?)
        # Compare if prediction moved in the same direction as actual
        # This requires having previous day's price
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        }


class ClassificationMetrics:
    """Calculate classification metrics for direction prediction."""
    
    @staticmethod
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: Actual class labels
            y_pred: Predicted class labels
            average: Averaging method for multi-class metrics
            
        Returns:
            Dictionary of metrics
        """
        # Remove NaN values
        mask = ~(pd.isna(y_true) | pd.isna(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {
                'accuracy': np.nan,
                'precision': np.nan,
                'recall': np.nan,
                'f1': np.nan
            }
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    @staticmethod
    def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Get confusion matrix."""
        mask = ~(pd.isna(y_true) | pd.isna(y_pred))
        return confusion_matrix(y_true[mask], y_pred[mask])
    
    @staticmethod
    def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Get detailed classification report."""
        mask = ~(pd.isna(y_true) | pd.isna(y_pred))
        return classification_report(y_true[mask], y_pred[mask])


class DirectionalAccuracy:
    """
    Calculate directional accuracy for regression predictions.
    
    Checks if the predicted direction (up/down) matches actual direction.
    """
    
    @staticmethod
    def calculate(
        actual_prices: np.ndarray,
        predicted_prices: np.ndarray,
        previous_prices: np.ndarray
    ) -> float:
        """
        Calculate directional accuracy.
        
        Args:
            actual_prices: Actual next-day prices
            predicted_prices: Predicted next-day prices
            previous_prices: Previous day (t-1) prices
            
        Returns:
            Directional accuracy (0-1)
        """
        # Remove NaN values
        mask = ~(np.isnan(actual_prices) | np.isnan(predicted_prices) | np.isnan(previous_prices))
        actual_prices = actual_prices[mask]
        predicted_prices = predicted_prices[mask]
        previous_prices = previous_prices[mask]
        
        if len(actual_prices) == 0:
            return np.nan
        
        # Calculate actual and predicted directions
        actual_direction = np.sign(actual_prices - previous_prices)
        predicted_direction = np.sign(predicted_prices - previous_prices)
        
        # Calculate accuracy
        correct = (actual_direction == predicted_direction).sum()
        total = len(actual_direction)
        
        return float(correct / total)


class BaselineEvaluator:
    """
    Comprehensive evaluation framework for baseline methods.
    """
    
    def __init__(self, classification_thresholds: Optional[Dict[str, List[float]]] = None):
        """
        Initialize evaluator.
        
        Args:
            classification_thresholds: Thresholds for classification tasks
        """
        self.classification_thresholds = classification_thresholds or {
            '5class': [-2, -0.5, 0.5, 2]
        }
        self.results = {}
    
    def evaluate_regression(
        self,
        df: pd.DataFrame,
        method_name: str,
        actual_col: str = 'close',
        pred_col: str = 'prediction'
    ) -> Dict[str, float]:
        """
        Evaluate regression predictions.
        
        Args:
            df: DataFrame with actual and predicted values
            method_name: Name of the prediction method
            actual_col: Column name for actual values
            pred_col: Column name for predictions
            
        Returns:
            Dictionary of metrics
        """
        # Get actual next-day prices
        df = df.copy()
        df['next_day_actual'] = df.groupby('ticker')[actual_col].shift(-1)
        
        # Get previous day prices for directional accuracy
        df['prev_day_actual'] = df.groupby('ticker')[actual_col].shift(1)
        
        # Calculate regression metrics
        metrics = RegressionMetrics.calculate(
            df['next_day_actual'].values,
            df[pred_col].values
        )
        
        # Calculate directional accuracy
        # Compare direction from yesterday to tomorrow (actual) vs yesterday to prediction
        dir_acc = DirectionalAccuracy.calculate(
            df['next_day_actual'].values,
            df[pred_col].values,
            df['prev_day_actual'].values  # Use previous day, not current day
        )
        metrics['directional_accuracy'] = dir_acc
        
        self.results[method_name] = {
            'type': 'regression',
            'metrics': metrics
        }
        
        return metrics
    
    def evaluate_classification(
        self,
        df: pd.DataFrame,
        method_name: str,
        n_classes: int,
        actual_col: str,
        pred_col: str
    ) -> Dict[str, float]:
        """
        Evaluate classification predictions.
        
        Args:
            df: DataFrame with actual and predicted class labels
            method_name: Name of the prediction method
            n_classes: Number of classes (3, 5, or 7)
            actual_col: Column name for actual labels
            pred_col: Column name for predicted labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = ClassificationMetrics.calculate(
            df[actual_col].values,
            df[pred_col].values
        )
        
        self.results[f"{method_name}_{n_classes}class"] = {
            'type': 'classification',
            'n_classes': n_classes,
            'metrics': metrics
        }
        
        return metrics
    
    def print_comparison_table(self):
        """Print a formatted comparison table of all results."""
        if not self.results:
            print("No results to display")
            return
        
        # Separate regression and classification results
        regression_results = {k: v for k, v in self.results.items() if v['type'] == 'regression'}
        classification_results = {k: v for k, v in self.results.items() if v['type'] == 'classification'}
        
        # Print regression results
        if regression_results:
            print("\n" + "="*80)
            print("REGRESSION RESULTS (Price Prediction)")
            print("="*80)
            print(f"{'Method':<25} {'MSE':>10} {'MAE':>10} {'RMSE':>10} {'Dir Acc':>10}")
            print("-"*80)
            
            for method_name, result in regression_results.items():
                m = result['metrics']
                print(f"{method_name:<25} {m['mse']:>10.4f} {m['mae']:>10.4f} {m['rmse']:>10.4f} {m['directional_accuracy']:>10.4f}")
        
        # Print classification results
        if classification_results:
            print("\n" + "="*80)
            print("CLASSIFICATION RESULTS (Direction Prediction)")
            print("="*80)
            print(f"{'Method':<30} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1':>12}")
            print("-"*80)
            
            for method_name, result in classification_results.items():
                m = result['metrics']
                print(f"{method_name:<30} {m['accuracy']:>12.4f} {m['precision']:>12.4f} {m['recall']:>12.4f} {m['f1']:>12.4f}")
        
        print("="*80 + "\n")
    
    def save_results(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        print(f"Results loaded from {filepath}")


if __name__ == "__main__":
    # Quick test
    print("Testing evaluation metrics...")
    
    # Test regression metrics
    y_true = np.array([100, 101, 102, 103, 104])
    y_pred = np.array([100.5, 100.8, 102.1, 103.5, 103.8])
    
    reg_metrics = RegressionMetrics.calculate(y_true, y_pred)
    print("\nRegression Metrics:")
    for k, v in reg_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test classification metrics
    y_true_cls = np.array([0, 1, 2, 1, 2, 0, 1])
    y_pred_cls = np.array([0, 1, 1, 1, 2, 0, 2])
    
    cls_metrics = ClassificationMetrics.calculate(y_true_cls, y_pred_cls)
    print("\nClassification Metrics:")
    for k, v in cls_metrics.items():
        print(f"  {k}: {v:.4f}")

