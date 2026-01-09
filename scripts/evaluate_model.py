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


def evaluate_regression(model, test_loader, device, scalers):
    """Evaluate regression model."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Calculate metrics on normalized values
    mse = np.mean((all_preds - all_targets) ** 2)
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(mse)
    
    # Directional accuracy
    if len(all_preds) > 1:
        pred_changes = np.diff(all_preds)
        target_changes = np.diff(all_targets)
        correct = np.sign(pred_changes) == np.sign(target_changes)
        dir_acc = np.mean(correct)
    else:
        dir_acc = 0.0
    
    metrics = {
        'mse_normalized': float(mse),
        'mae_normalized': float(mae),
        'rmse_normalized': float(rmse),
        'directional_accuracy': float(dir_acc)
    }
    
    return metrics, all_preds, all_targets


def evaluate_classification(model, test_loader, device, class_names):
    """Evaluate classification model."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"  Loaded checkpoint: {checkpoint_name}")
    print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    
    # Evaluate
    print("\n3. Evaluating...")
    
    if task == 'regression':
        metrics, preds, targets = evaluate_regression(model, test_loader, device, scalers)
        
        print("\n" + "="*80)
        print("REGRESSION RESULTS (Test Set)")
        print("="*80)
        print(f"MSE (normalized):  {metrics['mse_normalized']:.4f}")
        print(f"MAE (normalized):  {metrics['mae_normalized']:.4f}")
        print(f"RMSE (normalized): {metrics['rmse_normalized']:.4f}")
        print(f"Directional Acc:   {metrics['directional_accuracy']:.4f}")
        
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
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    
    parser.add_argument(
        '--run_dir',
        type=str,
        required=True,
        help='Path to run directory (e.g., results/phase1_runs/regression/run_XXX)'
    )
    
    args = parser.parse_args()
    
    main(args)

