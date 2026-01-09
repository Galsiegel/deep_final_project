# Phase 1: GRU Baseline Models

Train GRU models for stock prediction (regression and classification).

## Quick Start

### Train Regression Model (Price Prediction)
```bash
# Train on all stocks
python scripts/train_regression.py

# Quick test with 5 stocks
python scripts/train_regression.py --limit 5

# Train + evaluate on test set (recommended)
python scripts/train_regression.py --limit 5 --eval-test

# Specific tickers
python scripts/train_regression.py --tickers "AAPL,GOOG,MSFT"
```

### Train Classification Model (Direction Prediction)
```bash
# Train on all stocks
python scripts/train_classification.py

# Quick test + evaluate
python scripts/train_classification.py --limit 5 --eval-test
```

### Evaluate on Test Set (Manual)
```bash
# If you didn't use --eval-test flag during training
python scripts/evaluate_model.py --run_dir results/phase1_runs/regression/run_YYYYMMDD_HHMMSS
```

## Model Architecture

```
Input: [batch, 10, 5]  # 10 days lookback, 5 features (OHLCV)
  ↓
GRU (2 layers, hidden=128, dropout=0.2)
  ↓
FC (128 → 64, ReLU, dropout=0.3)
  ↓
Output head:
  - Regression: FC (64 → 1) → price
  - Classification: FC (64 → 5) → class logits
```

## Configuration

All settings in `configs/`:
- `base.yaml` - Shared config (model, data, training)
- `regression.yaml` - Regression-specific
- `classification.yaml` - Classification-specific

Key hyperparameters:
- Lookback: 10 days
- Batch size: 64
- Learning rate: 0.001
- Gradient clipping: 1.0
- Early stopping: 15 epochs

## Output Structure

```
results/phase1_runs/
  ├── regression/run_YYYYMMDD_HHMMSS/
  │   ├── config.yaml
  │   ├── model_summary.txt
  │   ├── scalers/scalers.pkl
  │   ├── checkpoints/
  │   │   ├── best_val_loss.pth
  │   │   ├── best_val_metric.pth
  │   │   └── last.pth
  │   ├── tensorboard/
  │   ├── training_history.json
  │   └── test_results.json
  └── classification/run_YYYYMMDD_HHMMSS/
      └── ...
```

## View Training Progress

```bash
tensorboard --logdir results/phase1_runs/regression/run_YYYYMMDD_HHMMSS/tensorboard
```

## Success Criteria

Compare against Phase 0 naive baselines:
- **Regression:** MSE < 1.0, Dir Acc > 55%
- **Classification:** F1 > 0.50, Accuracy > 55%

