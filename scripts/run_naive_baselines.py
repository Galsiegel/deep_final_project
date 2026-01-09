"""
Phase 0: Run Naive Baseline Predictions

This script establishes baseline performance using simple prediction methods:
- Persistence (last value)
- Moving average
- Linear trend extrapolation

These baselines establish the performance floor that neural networks must beat.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from utils.data_utils import StockDataLoader, get_default_split_dates
from models.naive_baselines import (
    PersistencePredictor,
    MovingAveragePredictor,
    LinearTrendPredictor
)
from utils.evaluation import BaselineEvaluator, RegressionMetrics, DirectionalAccuracy
from utils.visualization import StockVisualizer
import argparse
import json
from datetime import datetime


def create_run_directory(base_dir: str = "results/phase0_runs") -> Path:
    """
    Create a timestamped directory for this run.
    
    Args:
        base_dir: Base directory for all runs
        
    Returns:
        Path to the created run directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "data").mkdir(exist_ok=True)
    
    return run_dir


def main(args):
    """Run naive baseline experiments."""
    
    print("="*80)
    print("PHASE 0: NAIVE BASELINE PREDICTIONS")
    print("="*80)
    
    # Create timestamped run directory
    if args.no_timestamp:
        # Legacy mode: use simple results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        figures_dir = results_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
    else:
        # Create timestamped run directory
        results_dir = create_run_directory("results/phase0_runs")
        figures_dir = results_dir / "figures"
        print(f"\nRun directory: {results_dir}")
        print(f"Results will be saved to: {results_dir.absolute()}")
    
    # Load data
    print("\n1. Loading stock data...")
    loader = StockDataLoader(args.data_dir)
    
    if args.tickers:
        data = loader.load_data(tickers=args.tickers.split(','))
    else:
        data = loader.load_data(limit=args.limit)
    
    # Calculate returns for classification targets
    print("\n2. Calculating returns...")
    data = loader.calculate_returns(data)
    
    # Create classification targets
    print("\n3. Creating classification targets...")
    classification_thresholds = {
        '5class': [-2, -0.5, 0.5, 2]
    }
    data = loader.create_classification_targets(data, classification_thresholds)
    
    # Temporal split
    print("\n4. Splitting data temporally...")
    split_dates = get_default_split_dates()
    
    train_df, val_df, test_df = loader.temporal_split(
        data,
        train_end=split_dates['train_end'],
        val_start=split_dates['val_start'],
        val_end=split_dates['val_end'],
        test_start=split_dates['test_start'],
        lookback_days=args.lookback
    )
    
    # Choose split for evaluation
    if args.split == 'train':
        eval_df = train_df
    elif args.split == 'val':
        eval_df = val_df
    else:
        eval_df = test_df
    
    print(f"\n5. Running naive baseline predictions on {args.split} set...")
    print(f"   Evaluating on {len(eval_df)} records")
    
    # Initialize predictors
    predictors = [
        PersistencePredictor(),
        MovingAveragePredictor(window=5),
        LinearTrendPredictor(window=5)
    ]
    
    # Initialize evaluator
    evaluator = BaselineEvaluator(classification_thresholds)
    
    # Store predictions for visualization
    all_predictions = {}
    
    # Run each predictor
    for predictor in predictors:
        print(f"\n   Running {predictor.name}...")
        
        # Generate predictions
        pred_df = predictor.predict(eval_df)
        all_predictions[predictor.name] = pred_df
        
        # Evaluate regression
        metrics = evaluator.evaluate_regression(
            pred_df,
            predictor.name,
            actual_col='close',
            pred_col='prediction'
        )
        
        print(f"   [OK] MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, "
              f"RMSE: {metrics['rmse']:.4f}, Dir Acc: {metrics['directional_accuracy']:.4f}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("6. RESULTS SUMMARY")
    evaluator.print_comparison_table()
    
    # Save results with metadata
    results_file = results_dir / f"naive_baselines_{args.split}.json"
    
    # Add run metadata
    run_metadata = {
        "timestamp": datetime.now().isoformat(),
        "split": args.split,
        "data_dir": args.data_dir,
        "tickers": args.tickers if args.tickers else f"all (limit={args.limit})",
        "num_tickers": len(loader.tickers),
        "num_records": len(eval_df),
        "lookback_days": args.lookback,
        "split_dates": get_default_split_dates()
    }
    
    # Save results with metadata
    results_with_metadata = {
        "metadata": run_metadata,
        "results": evaluator.results
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_with_metadata, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Save a copy of the configuration
    config_file = results_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Configuration saved to {config_file}")
    
    # Visualizations
    if args.visualize:
        print("\n7. Generating visualizations...")
        viz = StockVisualizer(save_dir=figures_dir)
        
        # Get sample ticker for visualization
        sample_ticker = loader.tickers[0] if loader.tickers else 'AAPL'
        print(f"   Using {sample_ticker} for sample visualizations...")
        
        # Plot 1: Price history
        print("   - Price history")
        viz.plot_price_history(
            eval_df,
            sample_ticker,
            save_name=f"price_history_{sample_ticker}_{args.split}.png"
        )
        
        # Plot 2: Predictions vs Actual
        print("   - Predictions vs Actual")
        
        # Prepare data with all predictions
        ticker_data = eval_df[eval_df['ticker'] == sample_ticker].copy()
        for method_name, pred_df in all_predictions.items():
            method_ticker_data = pred_df[pred_df['ticker'] == sample_ticker]
            ticker_data[method_name] = method_ticker_data['prediction'].values
        
        # Plot subset of dates for clarity (last 60 days)
        if len(ticker_data) > 60:
            start_date = ticker_data['date'].iloc[-60]
            end_date = ticker_data['date'].iloc[-1]
        else:
            start_date = ticker_data['date'].iloc[0]
            end_date = ticker_data['date'].iloc[-1]
        
        viz.plot_predictions_vs_actual(
            ticker_data,
            sample_ticker,
            methods=list(all_predictions.keys()),
            date_range=(start_date, end_date),
            save_name=f"predictions_vs_actual_{sample_ticker}_{args.split}.png"
        )
        
        # Plot 3: Metrics comparison
        print("   - Metrics comparison")
        metrics_dict = {
            name: result['metrics'] 
            for name, result in evaluator.results.items()
            if result['type'] == 'regression'
        }
        viz.plot_metrics_comparison(
            metrics_dict,
            metric_names=['mse', 'mae', 'rmse'],
            save_name=f"metrics_comparison_{args.split}.png"
        )
        
        # Plot 4: Directional accuracy
        print("   - Directional accuracy")
        viz.plot_directional_accuracy(
            metrics_dict,
            save_name=f"directional_accuracy_{args.split}.png"
        )
        
        # Plot 5: Returns distribution
        print("   - Returns distribution")
        viz.plot_returns_distribution(
            eval_df,
            save_name=f"returns_distribution_{args.split}.png"
        )
        
        # Plot 6: Class distribution
        print("   - Class distribution (5-class)")
        viz.plot_class_distribution(
            eval_df.dropna(subset=['target_5class']),
            'target_5class',
            class_labels=['Large Down', 'Small Down', 'Neutral', 'Small Up', 'Large Up'],
            save_name=f"class_distribution_5class_{args.split}.png"
        )
        
        print(f"\n   [OK] All figures saved to {results_dir / 'figures'}/")
    
    print("\n" + "="*80)
    print("PHASE 0 COMPLETE")
    print("="*80)
    print(f"\nRun Directory: {results_dir.absolute()}")
    print(f"   |-- config.json (run configuration)")
    print(f"   |-- naive_baselines_{args.split}.json (results)")
    if args.visualize:
        print(f"   +-- figures/ (6 visualizations)")
    print("\nKey Takeaways:")
    print("- These naive baselines establish the minimum performance threshold")
    print("- Neural networks (Phase 1) must significantly outperform these methods")
    print("- Pay special attention to directional accuracy - it's often hard to beat!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run naive baseline predictions")
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/2017_2019',
        help='Directory containing stock CSV files'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Which data split to evaluate on'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of tickers to load (for quick testing)'
    )
    
    parser.add_argument(
        '--tickers',
        type=str,
        default=None,
        help='Comma-separated list of tickers to use (e.g., "AAPL,GOOG,MSFT")'
    )
    
    parser.add_argument(
        '--lookback',
        type=int,
        default=20,
        help='Lookback window size (used for gap calculation)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=True,
        help='Generate visualizations'
    )
    
    parser.add_argument(
        '--no-visualize',
        dest='visualize',
        action='store_false',
        help='Skip visualizations'
    )
    
    parser.add_argument(
        '--no-timestamp',
        action='store_true',
        help='Use simple results/ directory instead of timestamped run directory'
    )
    
    args = parser.parse_args()
    
    main(args)

