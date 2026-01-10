"""
Visualization utilities for stock predictions and model evaluation.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')


class StockVisualizer:
    """Create visualizations for stock predictions and model comparison."""
    
    def __init__(self, figsize: tuple = (12, 6), save_dir: str = "results/figures"):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size
            save_dir: Directory to save figures
        """
        self.figsize = figsize
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_price_history(
        self, 
        df: pd.DataFrame, 
        ticker: str,
        save_name: Optional[str] = None
    ):
        """
        Plot historical stock prices.
        
        Args:
            df: DataFrame with stock data
            ticker: Ticker symbol to plot
            save_name: Filename to save figure (if provided)
        """
        ticker_data = df[df['ticker'] == ticker].copy()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(ticker_data['date'], ticker_data['close'], label='Close Price', linewidth=2)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title(f'{ticker} Stock Price History', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            self._save_fig(save_name)
        else:
            plt.show()
        
        plt.close()
    
    def plot_predictions_vs_actual(
        self,
        df: pd.DataFrame,
        ticker: str,
        methods: List[str],
        actual_col: str = 'close',
        date_range: Optional[tuple] = None,
        save_name: Optional[str] = None
    ):
        """
        Plot predicted vs actual prices for multiple methods.
        
        Args:
            df: DataFrame with predictions
            ticker: Ticker symbol to plot
            methods: List of method names (column names with predictions)
            actual_col: Column name for actual prices
            date_range: Optional (start_date, end_date) tuple to zoom in
            save_name: Filename to save figure
        """
        ticker_data = df[df['ticker'] == ticker].copy()
        
        # Get next-day actual prices for comparison
        ticker_data['next_day_actual'] = ticker_data[actual_col].shift(-1)
        
        # Filter date range if specified
        if date_range:
            mask = (ticker_data['date'] >= date_range[0]) & (ticker_data['date'] <= date_range[1])
            ticker_data = ticker_data[mask]
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot actual prices
        ax.plot(ticker_data['date'], ticker_data['next_day_actual'], 
                label='Actual', linewidth=2.5, color='black', alpha=0.8)
        
        # Plot predictions from each method
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        for method, color in zip(methods, colors):
            if method in ticker_data.columns:
                ax.plot(ticker_data['date'], ticker_data[method],
                       label=method, linewidth=1.5, alpha=0.7, color=color, linestyle='--')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title(f'{ticker} - Predictions vs Actual', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            self._save_fig(save_name)
        else:
            plt.show()
        
        plt.close()
    
    def plot_prediction_errors(
        self,
        results: Dict[str, pd.DataFrame],
        ticker: str,
        save_name: Optional[str] = None
    ):
        """
        Plot prediction errors (residuals) for each method.
        
        Args:
            results: Dictionary mapping method name to DataFrame with predictions
            ticker: Ticker symbol to analyze
            save_name: Filename to save figure
        """
        fig, axes = plt.subplots(len(results), 1, figsize=(12, 4*len(results)))
        if len(results) == 1:
            axes = [axes]
        
        for ax, (method_name, df) in zip(axes, results.items()):
            ticker_data = df[df['ticker'] == ticker].copy()
            ticker_data['next_day_actual'] = ticker_data['close'].shift(-1)
            ticker_data['error'] = ticker_data['prediction'] - ticker_data['next_day_actual']
            
            # Remove NaN
            ticker_data = ticker_data.dropna(subset=['error'])
            
            ax.plot(ticker_data['date'], ticker_data['error'], alpha=0.7)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
            ax.set_xlabel('Date')
            ax.set_ylabel('Prediction Error ($)')
            ax.set_title(f'{method_name} - {ticker}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            self._save_fig(save_name)
        else:
            plt.show()
        
        plt.close()
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        metric_names: List[str] = ['mse', 'mae', 'rmse'],
        save_name: Optional[str] = None
    ):
        """
        Plot bar chart comparing metrics across methods.
        
        Args:
            metrics_dict: Dictionary mapping method name to metrics dict
            metric_names: List of metric names to plot
            save_name: Filename to save figure
        """
        n_metrics = len(metric_names)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        methods = list(metrics_dict.keys())
        
        for ax, metric_name in zip(axes, metric_names):
            values = [metrics_dict[method].get(metric_name, np.nan) for method in methods]
            
            bars = ax.bar(range(len(methods)), values, color=plt.cm.viridis(np.linspace(0, 1, len(methods))))
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.set_ylabel(metric_name.upper())
            ax.set_title(f'{metric_name.upper()} Comparison', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                if not np.isnan(value):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_name:
            self._save_fig(save_name)
        else:
            plt.show()
        
        plt.close()
    
    def plot_directional_accuracy(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        save_name: Optional[str] = None
    ):
        """
        Plot directional accuracy comparison.
        
        Args:
            metrics_dict: Dictionary mapping method name to metrics dict
            save_name: Filename to save figure
        """
        methods = list(metrics_dict.keys())
        dir_acc = [metrics_dict[method].get('directional_accuracy', np.nan) for method in methods]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(methods, dir_acc, color=plt.cm.RdYlGn(np.array(dir_acc)))
        ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Random Chance')
        ax.set_xlabel('Directional Accuracy', fontsize=12)
        ax.set_title('Directional Accuracy Comparison (Can We Predict Up/Down?)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, value in zip(bars, dir_acc):
            if not np.isnan(value):
                ax.text(value, bar.get_y() + bar.get_height()/2,
                       f' {value:.3f}', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            self._save_fig(save_name)
        else:
            plt.show()
        
        plt.close()
    
    def plot_returns_distribution(
        self,
        df: pd.DataFrame,
        save_name: Optional[str] = None
    ):
        """
        Plot distribution of daily returns.
        
        Args:
            df: DataFrame with 'return' column
            save_name: Filename to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        axes[0].hist(df['return'].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Daily Return (%)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Daily Returns', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot by ticker (top 10 tickers by volume)
        top_tickers = df.groupby('ticker')['volume'].mean().nlargest(10).index
        plot_data = df[df['ticker'].isin(top_tickers)]
        
        plot_data.boxplot(column='return', by='ticker', ax=axes[1])
        axes[1].set_xlabel('Ticker')
        axes[1].set_ylabel('Daily Return (%)')
        axes[1].set_title('Return Distribution by Ticker (Top 10)', fontweight='bold')
        plt.suptitle('')  # Remove default title
        
        plt.tight_layout()
        
        if save_name:
            self._save_fig(save_name)
        else:
            plt.show()
        
        plt.close()
    
    def plot_class_distribution(
        self,
        df: pd.DataFrame,
        class_column: str,
        class_labels: List[str],
        save_name: Optional[str] = None
    ):
        """
        Plot class distribution for classification targets.
        
        Args:
            df: DataFrame with classification targets
            class_column: Column name with class labels
            class_labels: Names for each class
            save_name: Filename to save figure
        """
        counts = df[class_column].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(range(len(counts)), counts.values, 
                     color=plt.cm.viridis(np.linspace(0, 1, len(counts))))
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(class_labels, rotation=45, ha='right')
        ax.set_ylabel('Count')
        ax.set_title(f'Class Distribution - {class_column}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        total = counts.sum()
        for bar, count in zip(bars, counts.values):
            pct = count / total * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_name:
            self._save_fig(save_name)
        else:
            plt.show()
        
        plt.close()
    
    def _save_fig(self, filename: str):
        """Save current figure to file."""
        filepath = self.save_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {filepath}")


if __name__ == "__main__":
    print("Visualization module ready.")
    print(f"Figures will be saved to: results/figures/")

