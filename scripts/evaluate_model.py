"""
Evaluate trained model on test set.

Usage:
    # Regression
    python scripts/evaluate_model.py --run_dir results/phase1_runs/regression/run_XXX
    
    # Classification
    python scripts/evaluate_model.py --run_dir results/phase1_runs/classification/run_XXX
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
import yaml
import argparse
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from utils.data_utils import StockDataLoader
from utils.dataset import StockSequenceDataset, load_scalers
from models.regression_model import StockRegressionModel
from models.classification_model import StockClassificationModel
from utils.visualization import StockVisualizer


def evaluate_regression(model, test_loader, device, scalers, test_dataset):
    """Evaluate regression model."""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_ticker_indices = []
    
    with torch.no_grad():
        batch_start = 0
        for X, day_open, y in test_loader:
            X, day_open, y = X.to(device), day_open.to(device), y.to(device)
            outputs = model(X, day_open)
            
            batch_size = len(outputs)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
            # Track which ticker each sample belongs to
            for i in range(batch_size):
                sample_idx = batch_start + i
                if sample_idx < len(test_dataset.ticker_indices):
                    all_ticker_indices.append(test_dataset.ticker_indices[sample_idx])
            
            batch_start += batch_size
    
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Calculate metrics on normalized values
    mse_norm = np.mean((all_preds - all_targets) ** 2)
    mae_norm = np.mean(np.abs(all_preds - all_targets))
    rmse_norm = np.sqrt(mse_norm)
    
    # Inverse transform to actual prices
    all_preds_actual = np.zeros_like(all_preds)
    all_targets_actual = np.zeros_like(all_targets)
    
    for i, ticker in enumerate(all_ticker_indices):
        if ticker in scalers:
            scaler = scalers[ticker]
            
            # Inverse transform (close is the last feature in most cases, index -2 before volume)
            # We need to create a dummy array with all features, then extract close
            # For simplicity, we'll use the close feature index
            close_idx = 3  # close is typically the 4th feature (0:open, 1:high, 2:low, 3:close, 4:volume)
            
            # Create dummy feature array
            dummy = np.zeros((1, len(scaler.mean_)))
            
            # Set close value and inverse transform
            dummy[0, close_idx] = all_preds[i]
            pred_actual = scaler.inverse_transform(dummy)[0, close_idx]
            all_preds_actual[i] = pred_actual
            
            dummy[0, close_idx] = all_targets[i]
            target_actual = scaler.inverse_transform(dummy)[0, close_idx]
            all_targets_actual[i] = target_actual
    
    # Calculate metrics on actual prices
    mse_actual = np.mean((all_preds_actual - all_targets_actual) ** 2)
    mae_actual = np.mean(np.abs(all_preds_actual - all_targets_actual))
    rmse_actual = np.sqrt(mse_actual)
    
    # Directional accuracy (same for both normalized and actual)
    if len(all_preds) > 1:
        pred_changes = np.diff(all_preds)
        target_changes = np.diff(all_targets)
        correct = np.sign(pred_changes) == np.sign(target_changes)
        dir_acc = np.mean(correct)
    else:
        dir_acc = 0.0
    
    metrics = {
        'mse_normalized': float(mse_norm),
        'mae_normalized': float(mae_norm),
        'rmse_normalized': float(rmse_norm),
        'mse_actual': float(mse_actual),
        'mae_actual': float(mae_actual),
        'rmse_actual': float(rmse_actual),
        'directional_accuracy': float(dir_acc)
    }
    
    return metrics, all_preds, all_targets, all_preds_actual, all_targets_actual, all_ticker_indices


def evaluate_classification(model, test_loader, device, class_names):
    """Evaluate classification model."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, day_open, y in test_loader:
            X, day_open, y = X.to(device), day_open.to(device), y.to(device)
            outputs = model(X, day_open)
            pred_classes = torch.argmax(outputs, dim=1)
            
            all_preds.extend(pred_classes.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        mask = all_targets == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == i).mean()
            per_class_acc[class_name] = float(class_acc)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics, all_preds, all_targets


def main(args):
    print("="*80)
    print("EVALUATING MODEL ON TEST SET")
    print("="*80)
    
    run_dir = Path(args.run_dir)
    
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return
    
    # Load config
    with open(run_dir / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    task = config['task']
    print(f"\nTask: {task}")
    print(f"Run directory: {run_dir}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\n1. Loading test data...")
    loader = StockDataLoader(config['data']['data_dir'])
    
    if 'tickers' in config:
        data = loader.load_data(tickers=config['tickers'])
    elif 'limit' in config:
        data = loader.load_data(limit=config['limit'])
    else:
        data = loader.load_data()
    
    data = loader.calculate_returns(data)
    
    # Split data
    split_dates = config['data']
    train_df, val_df, test_df = loader.temporal_split(
        data,
        train_end=split_dates['train_end'],
        val_start=split_dates['val_start'],
        val_end=split_dates['val_end'],
        test_start=split_dates['test_start'],
        lookback_days=config['data']['lookback']
    )
    
    # Load scalers
    scalers = load_scalers(str(run_dir / 'scalers' / 'scalers.pkl'))
    
    # Create test dataset
    if task == 'regression':
        test_dataset = StockSequenceDataset(
            data=test_df,
            lookback=config['data']['lookback'],
            features=config['data']['features'],
            task='regression',
            stride=1,
            scalers=scalers,
            fit_scalers=False,
            volume_transform=config['data']['volume_transform']
        )
    else:  # classification
        test_dataset = StockSequenceDataset(
            data=test_df,
            lookback=config['data']['lookback'],
            features=config['data']['features'],
            task='classification',
            stride=1,
            class_thresholds=config['classification']['class_thresholds'],
            scalers=scalers,
            fit_scalers=False,
            volume_transform=config['data']['volume_transform']
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    print(f"  Test samples: {len(test_dataset)}")
    
    # Create model
    print("\n2. Loading model...")
    if task == 'regression':
        model = StockRegressionModel(
            input_size=len(config['data']['features']),
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout_gru=config['model']['dropout_gru'],
            dropout_fc=config['model']['dropout_fc'],
            fc_hidden_size=config['model']['fc_hidden_size']
        )
        checkpoint_name = 'best_val_loss.pth'
    else:  # classification
        model = StockClassificationModel(
            input_size=len(config['data']['features']),
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            num_classes=config['classification']['num_classes'],
            dropout_gru=config['model']['dropout_gru'],
            dropout_fc=config['model']['dropout_fc'],
            fc_hidden_size=config['model']['fc_hidden_size']
        )
        checkpoint_name = 'best_val_metric.pth'
    
    # Load checkpoint
    checkpoint_path = run_dir / 'checkpoints' / checkpoint_name
    if not checkpoint_path.exists():
        print(f"Warning: {checkpoint_name} not found, trying last.pth")
        checkpoint_path = run_dir / 'checkpoints' / 'last.pth'
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"  Loaded checkpoint: {checkpoint_name}")
    print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    
    # Evaluate
    print("\n3. Evaluating...")
    
    if task == 'regression':
        metrics, preds, targets, preds_actual, targets_actual, all_ticker_indices = evaluate_regression(
            model, test_loader, device, scalers, test_dataset
        )
        
        print("\n" + "="*80)
        print("REGRESSION RESULTS (Test Set)")
        print("="*80)
        print("\nActual Price Metrics:")
        print(f"  MSE:   {metrics['mse_actual']:.4f}")
        print(f"  MAE:   {metrics['mae_actual']:.4f}")
        print(f"  RMSE:  {metrics['rmse_actual']:.4f}")
        print("\nNormalized Metrics:")
        print(f"  MSE:   {metrics['mse_normalized']:.4f}")
        print(f"  MAE:   {metrics['mae_normalized']:.4f}")
        print(f"  RMSE:  {metrics['rmse_normalized']:.4f}")
        print("\nDirectional Accuracy: {:.2f}%".format(metrics['directional_accuracy'] * 100))
        
    else:  # classification
        class_names = config['classification']['class_names']
        metrics, preds, targets = evaluate_classification(model, test_loader, device, class_names)
        
        print("\n" + "="*80)
        print("CLASSIFICATION RESULTS (Test Set)")
        print("="*80)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        
        print("\nPer-class Accuracy:")
        for class_name, acc in metrics['per_class_accuracy'].items():
            print(f"  {class_name}: {acc:.4f}")
        
        print("\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print("     ", " ".join([f"{name[:6]:>6}" for name in class_names]))
        for i, row in enumerate(cm):
            print(f"{class_names[i][:6]:>6}", " ".join([f"{val:>6}" for val in row]))
    
    # Save results
    results_file = run_dir / 'test_results.json'
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Generate visualizations
    if task == 'regression':
        print("\n4. Generating visualizations...")
        figures_dir = run_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        viz = StockVisualizer(save_dir=str(figures_dir))
        
        # Prepare data for time series plot
        # Get a sample ticker and reconstruct the time series
        if args.ticker and args.ticker in scalers:
            sample_ticker = args.ticker
        else:
            sample_ticker = list(scalers.keys())[0] if scalers else None
            if args.ticker:
                print(f"   Warning: Ticker '{args.ticker}' not found in training data.")
                print(f"   Available tickers: {', '.join(list(scalers.keys())[:5])}...")
        
        if sample_ticker:
            print(f"   Using {sample_ticker} for sample visualizations...")
            
            # Get test data for this ticker
            ticker_test_df = test_df[test_df['ticker'] == sample_ticker].sort_values('date').reset_index(drop=True)
            
            # Get predictions for this ticker
            ticker_mask = np.array([t == sample_ticker for t in all_ticker_indices])
            ticker_preds_actual = preds_actual[ticker_mask]
            ticker_targets_actual = targets_actual[ticker_mask]
            
            # Create DataFrame for plotting
            # Note: predictions start after lookback period
            lookback = config['data']['lookback']
            plot_df = ticker_test_df.iloc[lookback:lookback+len(ticker_preds_actual)].copy()
            plot_df['prediction'] = ticker_preds_actual
            plot_df['actual'] = ticker_targets_actual
            
            # Plot 1: Time series - Predictions vs Actual
            print("   - Predictions vs Actual (time series)")
            import matplotlib.pyplot as plt
            
            # Plot last 60 days or all if less
            n_days = min(60, len(plot_df))
            plot_subset = plot_df.iloc[-n_days:]
            
            plt.figure(figsize=(14, 7))
            plt.plot(plot_subset['date'], plot_subset['actual'], 
                    label='Actual', linewidth=2.5, color='black', alpha=0.8)
            plt.plot(plot_subset['date'], plot_subset['prediction'],
                    label='GRU Prediction', linewidth=1.5, alpha=0.7, 
                    color='steelblue', linestyle='--')
            
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price ($)', fontsize=12)
            plt.title(f'{sample_ticker} - GRU Predictions vs Actual Prices', 
                     fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(figures_dir / f'predictions_vs_actual_{sample_ticker}_timeseries.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            print(f"     Saved to {figures_dir / f'predictions_vs_actual_{sample_ticker}_timeseries.png'}")
        
        # Plot 2: Prediction errors histogram
        print("   - Error distribution")
        errors = preds_actual - targets_actual
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Prediction Error ($)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors (GRU Model)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'error_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"     Saved to {figures_dir / 'error_distribution.png'}")
        
        # Plot 2: Predictions vs Actual scatter
        print("   - Predictions vs Actual scatter")
        plt.figure(figsize=(10, 10))
        plt.scatter(targets_actual, preds_actual, alpha=0.5, s=10)
        min_val = min(targets_actual.min(), preds_actual.min())
        max_val = max(targets_actual.max(), preds_actual.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.title('Predictions vs Actual Prices (GRU Model)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'predictions_vs_actual_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"     Saved to {figures_dir / 'predictions_vs_actual_scatter.png'}")
        
        # Plot 3: Metrics comparison bar chart
        print("   - Metrics comparison")
        plt.figure(figsize=(10, 6))
        metric_names = ['MAE', 'RMSE']
        gru_values = [metrics['mae_actual'], metrics['rmse_actual']]
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        plt.bar(x, gru_values, width, label='GRU Model', alpha=0.8)
        plt.xlabel('Metric')
        plt.ylabel('Value ($)')
        plt.title('Model Performance Metrics (Actual Prices)')
        plt.xticks(x, metric_names)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(figures_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"     Saved to {figures_dir / 'metrics_comparison.png'}")
        
        # Plot 4: Directional accuracy
        print("   - Directional accuracy")
        plt.figure(figsize=(8, 6))
        plt.barh(['GRU Model'], [metrics['directional_accuracy']], color='steelblue', alpha=0.8)
        plt.axvline(x=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Random Chance')
        plt.xlabel('Directional Accuracy')
        plt.title('Directional Accuracy: Can the Model Predict Up/Down?')
        plt.xlim([0, 1])
        plt.legend()
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value label
        plt.text(metrics['directional_accuracy'], 0, 
                f" {metrics['directional_accuracy']:.3f}", 
                va='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'directional_accuracy.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"     Saved to {figures_dir / 'directional_accuracy.png'}")
        
        print(f"\n   [OK] All figures saved to {figures_dir}/")
    
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    
    parser.add_argument(
        '--run_dir',
        type=str,
        required=True,
        help='Path to run directory (e.g., results/phase1_runs/regression/run_XXX)'
    )
    
    parser.add_argument(
        '--ticker',
        type=str,
        default=None,
        help='Ticker symbol to visualize (e.g., AAPL, MSFT). If not specified, uses first ticker alphabetically.'
    )
    
    args = parser.parse_args()
    
    main(args)

