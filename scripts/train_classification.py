"""
Train GRU classification model for stock direction prediction.

Usage:
    python scripts/train_classification.py
    python scripts/train_classification.py --limit 10  # Quick test
    python scripts/train_classification.py --tickers "AAPL,GOOG"
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
from models.classification_model import StockClassificationModel
from models.trainer import Trainer


def load_config(config_path: str = 'configs/classification.yaml') -> dict:
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
    print("PHASE 1: TRAINING CLASSIFICATION MODEL")
    print("="*80)
    
    # Load configuration
    config = load_config()
    
    # Override config with command line args
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
    run_dir = Path(f"results/phase1_runs/classification/run_{timestamp}")
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
    
    # Calculate returns
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
    
    # Training dataset
    train_dataset = StockSequenceDataset(
        data=train_df,
        lookback=config['data']['lookback'],
        features=config['data']['features'],
        task='classification',
        stride=config['data']['stride'],
        class_thresholds=config['classification']['class_thresholds'],
        fit_scalers=True,
        volume_transform=config['data']['volume_transform']
    )
    
    # Validation dataset
    val_dataset = StockSequenceDataset(
        data=val_df,
        lookback=config['data']['lookback'],
        features=config['data']['features'],
        task='classification',
        stride=config['data']['stride'],
        class_thresholds=config['classification']['class_thresholds'],
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
    
    # Check class distribution
    if config['classification']['use_class_weights']:
        class_weights = train_dataset.get_class_weights()
        print(f"  Class weights: {class_weights}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    print("\n4. Creating model...")
    model = StockClassificationModel(
        input_size=len(config['data']['features']),
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_classes=config['classification']['num_classes'],
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
    
    # Loss function with class weights
    if config['classification']['use_class_weights']:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print(f"  Using weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"  Using standard CrossEntropyLoss")
    
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
        task='classification',
        run_dir=run_dir,
        grad_clip_norm=config['training']['grad_clip_norm'],
        early_stopping_patience=config['training']['early_stopping']['patience'],
        best_metric=config['classification']['best_metric'],
        best_metric_mode=config['classification']['best_metric_mode']
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
    print(f"  - best_val_metric.pth (best {config['classification']['best_metric']})")
    print(f"  - last.pth")
    
    # Optional: Evaluate on test set
    if args.eval_test:
        print("\n" + "="*80)
        print("EVALUATING ON TEST SET")
        print("="*80)
        
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        # Create test dataset
        test_dataset = StockSequenceDataset(
            data=test_df,
            lookback=config['data']['lookback'],
            features=config['data']['features'],
            task='classification',
            stride=1,
            class_thresholds=config['classification']['class_thresholds'],
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
        checkpoint_path = run_dir / 'checkpoints' / 'best_val_metric.pth'
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
        
        print("\nTest Set Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Per-class accuracy
        class_names = config['classification']['class_names']
        print("\nPer-class Accuracy:")
        for i, class_name in enumerate(class_names):
            mask = all_targets == i
            if mask.sum() > 0:
                class_acc = (all_preds[mask] == i).mean()
                print(f"  {class_name}: {class_acc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        print("\nConfusion Matrix:")
        print("     ", " ".join([f"{name[:6]:>6}" for name in class_names]))
        for i, row in enumerate(cm):
            print(f"{class_names[i][:6]:>6}", " ".join([f"{val:>6}" for val in row]))
        
        # Save results
        test_results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': cm.tolist()
        }
        
        with open(run_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\nTest results saved to: {run_dir / 'test_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classification model")
    
    parser.add_argument(
        '--tickers',
        type=str,
        default=None,
        help='Comma-separated list of tickers'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of tickers for testing'
    )
    
    parser.add_argument(
        '--eval-test',
        action='store_true',
        help='Evaluate on test set after training (default: False)'
    )
    
    args = parser.parse_args()
    
    main(args)

