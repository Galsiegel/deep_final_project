"""
Train GRU regression model for stock price prediction.

Usage:
    python scripts/train_regression.py
    python scripts/train_regression.py --limit 10  # Quick test
    python scripts/train_regression.py --tickers "AAPL,GOOG"
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import argparse
from datetime import datetime
import json
import numpy as np

from models.data_utils import StockDataLoader, get_default_split_dates
from models.dataset import StockSequenceDataset, save_scalers
from models.regression_model import StockRegressionModel
from models.trainer import Trainer


def load_config(config_path: str = 'configs/regression.yaml') -> dict:
    """Load configuration from YAML files."""
    # Load base config
    with open('configs/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load task-specific config and merge
    with open(config_path, 'r') as f:
        task_config = yaml.safe_load(f)
    
    config.update(task_config)
    return config


def main(args):
    print("="*80)
    print("PHASE 1: TRAINING REGRESSION MODEL")
    print("="*80)
    
    # Load configuration
    config = load_config()
    
    # Override config with command line args if provided
    if args.tickers:
        config['tickers'] = args.tickers.split(',')
    if args.limit:
        config['limit'] = args.limit
    
    # Set random seed
    torch.manual_seed(config['seed'])
    
    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"results/phase1_runs/regression/run_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    
    # Save configuration
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("\n1. Loading data...")
    loader = StockDataLoader(config['data']['data_dir'])
    
    if 'tickers' in config:
        data = loader.load_data(tickers=config['tickers'])
    elif 'limit' in config:
        data = loader.load_data(limit=config['limit'])
    else:
        data = loader.load_data()
    
    # Calculate returns (needed for dataset)
    data = loader.calculate_returns(data)
    
    print("\n2. Splitting data temporally...")
    split_dates = config['data']
    train_df, val_df, test_df = loader.temporal_split(
        data,
        train_end=split_dates['train_end'],
        val_start=split_dates['val_start'],
        val_end=split_dates['val_end'],
        test_start=split_dates['test_start'],
        lookback_days=config['data']['lookback']
    )
    
    print("\n3. Creating datasets...")
    
    # Training dataset (fit scalers)
    train_dataset = StockSequenceDataset(
        data=train_df,
        lookback=config['data']['lookback'],
        features=config['data']['features'],
        task='regression',
        stride=config['data']['stride'],
        fit_scalers=True,
        volume_transform=config['data']['volume_transform']
    )
    
    # Validation dataset (use train scalers)
    val_dataset = StockSequenceDataset(
        data=val_df,
        lookback=config['data']['lookback'],
        features=config['data']['features'],
        task='regression',
        stride=config['data']['stride'],
        scalers=train_dataset.get_scalers(),
        fit_scalers=False,
        volume_transform=config['data']['volume_transform']
    )
    
    # Save scalers
    scalers_dir = run_dir / 'scalers'
    scalers_dir.mkdir(exist_ok=True)
    save_scalers(train_dataset.get_scalers(), str(scalers_dir / 'scalers.pkl'))
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0  # Use 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    print("\n4. Creating model...")
    model = StockRegressionModel(
        input_size=len(config['data']['features']),
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout_gru=config['model']['dropout_gru'],
        dropout_fc=config['model']['dropout_fc'],
        fc_hidden_size=config['model']['fc_hidden_size']
    )
    
    model = model.to(device)
    print(f"  Model parameters: {model.get_num_parameters():,}")
    
    # Save model architecture
    with open(run_dir / 'model_summary.txt', 'w') as f:
        f.write(str(model))
        f.write(f"\n\nTotal parameters: {model.get_num_parameters():,}")
    
    print("\n5. Setting up training...")
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config['training']['scheduler']['mode'],
        factor=config['training']['scheduler']['factor'],
        patience=config['training']['scheduler']['patience'],
        min_lr=config['training']['scheduler']['min_lr']
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        task='regression',
        run_dir=run_dir,
        grad_clip_norm=config['training']['grad_clip_norm'],
        early_stopping_patience=config['training']['early_stopping']['patience'],
        best_metric=config['regression']['best_metric'],
        best_metric_mode=config['regression']['best_metric_mode']
    )
    
    print("\n6. Training...")
    trainer.train(num_epochs=config['training']['epochs'])
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {run_dir}")
    print(f"\nTo view training curves:")
    print(f"  tensorboard --logdir {run_dir / 'tensorboard'}")
    print(f"\nCheckpoints:")
    print(f"  - best_val_loss.pth")
    print(f"  - best_val_metric.pth (best {config['regression']['best_metric']})")
    print(f"  - last.pth")
    
    # Optional: Evaluate on test set
    if args.eval_test:
        print("\n" + "="*80)
        print("EVALUATING ON TEST SET")
        print("="*80)
        
        from models.dataset import load_scalers
        
        # Create test dataset
        test_dataset = StockSequenceDataset(
            data=test_df,
            lookback=config['data']['lookback'],
            features=config['data']['features'],
            task='regression',
            stride=1,
            scalers=train_dataset.get_scalers(),
            fit_scalers=False,
            volume_transform=config['data']['volume_transform']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        print(f"\nTest samples: {len(test_dataset)}")
        
        # Load best model
        checkpoint_path = run_dir / 'checkpoints' / 'best_val_loss.pth'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
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
        
        # Calculate metrics
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
        
        print("\nTest Set Results:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Directional Accuracy: {dir_acc:.4f}")
        
        # Save results
        test_results = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'directional_accuracy': float(dir_acc)
        }
        
        with open(run_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\nTest results saved to: {run_dir / 'test_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train regression model")
    
    parser.add_argument(
        '--tickers',
        type=str,
        default=None,
        help='Comma-separated list of tickers (e.g., "AAPL,GOOG,MSFT")'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of tickers for quick testing'
    )
    
    parser.add_argument(
        '--eval-test',
        action='store_true',
        help='Evaluate on test set after training (default: False)'
    )
    
    args = parser.parse_args()
    
    main(args)

