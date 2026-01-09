<!-- 715514cb-ab53-4a55-8191-e33d56afef99 a8326a0e-093b-40b8-92a9-d09d0a6d4483 -->
# Phase 1: GRU Baseline Model

## Overview

Build separate GRU models for:

1. **Regression:** Predict next-day closing price
2. **Classification:** Predict direction (5-class: Large Down, Small Down, Neutral, Small Up, Large Up)

Models train on stock OHLCV data only (no news embeddings yet).

## Architecture

### Shared Base Architecture

```
Input: [batch, seq_len=20, features=5]  # OHLCV
  ↓
GRU(input_size=5, hidden_size=128, num_layers=2, dropout=0.2)
  ↓
Take last hidden state: [batch, 128]
  ↓
Linear(128 → 64) + ReLU + Dropout(0.3)
  ↓
[batch, 64] → Task-specific head
```

### Task-Specific Heads

**Regression Model:**

```
[batch, 64] → Linear(64 → 1) → [batch, 1]
Output: Normalized next-day close price
Loss: MSE
```

**Classification Model:**

```
[batch, 64] → Linear(64 → 5) → [batch, 5]
Output: Logits for 5 classes
Loss: CrossEntropyLoss
```

## Implementation Details

### Data Processing

**Features:** OHLCV (Open, High, Low, Close, Volume)

**Normalization:** Per-ticker StandardScaler

- Each ticker normalized independently
- Fit on train set only, transform val/test
- Store scaler for each ticker for denormalization

**Sequence Creation:**

- Lookback window: 20 days
- Stride: 1 (maximum overlap for training data)
- Target: Next day's close price (t+1)
- No sequences crossing train/val/test boundaries

**Data Split:** Use same temporal splits as Phase 0

- Train: up to 2018-06-30
- Gap: 21 days
- Val: 2018-07-21 to 2018-12-31
- Gap: 21 days
- Test: from 2019-01-21

### Training Configuration

**Fixed Hyperparameters:**

```yaml
model:
  type: gru
  hidden_size: 128
  num_layers: 2
  dropout_gru: 0.2
  dropout_fc: 0.3
  fc_hidden_size: 64

data:
  lookback: 20
  features: [open, high, low, close, volume]
  normalization: per_ticker
  stride: 1
  batch_size: 64

training:
  epochs: 100
  learning_rate: 0.001
  optimizer: adam
  grad_clip_norm: 1.0
  
  scheduler:
    type: reduce_on_plateau
    mode: min
    factor: 0.5
    patience: 10
    min_lr: 1.0e-6
  
  early_stopping:
    patience: 15
    monitor: val_loss

regression:
  loss: mse
  
classification:
  loss: cross_entropy
  num_classes: 5
  class_thresholds: [-2.0, -0.5, 0.5, 2.0]  # % returns
```

### Checkpointing Strategy

Save 3 types per model:

1. **best_val_loss.pth** - Lowest validation loss
2. **best_val_metric.pth** - Best task-specific metric:

   - Regression: Best directional accuracy
   - Classification: Best accuracy

3. **last.pth** - Latest epoch (for resuming training)

Each checkpoint includes:

- Model state_dict
- Optimizer state_dict
- Scheduler state_dict
- Epoch number
- Metrics history
- Scalers (for denormalization)

## Files to Create

### Core Model Files

```
models/
  ├── gru_model.py              # Base GRU encoder
  ├── regression_model.py       # Regression wrapper
  ├── classification_model.py   # Classification wrapper
  ├── dataset.py                # PyTorch Dataset
  └── trainer.py                # Training engine
```

### Scripts

```
scripts/
  ├── train_regression.py       # Train regression model
  ├── train_classification.py   # Train classification model
  └── evaluate_model.py         # Evaluate on test set
```

### Configuration

```
configs/
  ├── base.yaml                 # Shared config
  ├── regression.yaml           # Regression-specific
  └── classification.yaml       # Classification-specific
```

## Metrics & Logging

### TensorBoard Logging

**Training metrics (per epoch):**

- Train loss, val loss
- Learning rate
- Gradient norms

**Regression metrics:**

- MSE, MAE, RMSE (normalized)
- MSE, MAE, RMSE (denormalized, actual $)
- Directional accuracy
- Sample predictions vs actual (plot)

**Classification metrics:**

- Accuracy, Precision, Recall, F1 (weighted)
- Per-class accuracy
- Confusion matrix (image)

### Output Structure

```
results/phase1_runs/
  └── run_YYYYMMDD_HHMMSS_regression/  # or _classification
      ├── config.yaml              # Full configuration
      ├── model_summary.txt        # Architecture details
      ├── scalers/                 # Per-ticker StandardScalers
      │   ├── AAPL_scaler.pkl
      │   ├── GOOG_scaler.pkl
      │   └── ...
      ├── checkpoints/
      │   ├── best_val_loss.pth
      │   ├── best_val_metric.pth
      │   └── last.pth
      ├── tensorboard/             # TensorBoard logs
      │   └── events.out.tfevents...
      ├── training_log.csv         # Epoch-by-epoch metrics
      └── final_results.json       # Test set evaluation
```

## Evaluation & Comparison

After training both models, compare against Phase 0 naive baselines:

```
Method                          MSE      MAE     RMSE    Dir Acc    F1
---------------------------------------------------------------------
Persistence (Phase 0)          1.13     0.83    1.06     48.3%      -
Moving Avg (Phase 0)           1.77     1.06    1.33     50.4%      -
Linear Trend (Phase 0)         1.29     0.89    1.14     53.2%      -
---------------------------------------------------------------------
GRU Regression (Phase 1)        ?        ?       ?        ?         -
GRU Classification (Phase 1)    -        -       -        -         ?
```

**Success Criteria:**

- Regression MSE < 1.0 (beat Persistence)
- Directional Accuracy > 55% (significantly beat baselines)
- Classification F1 > 0.50 (beat random)

## Implementation Steps

1. Create base GRU encoder (`gru_model.py`)
2. Implement PyTorch Dataset for sequences (`dataset.py`)
3. Build regression model wrapper (`regression_model.py`)
4. Build classification model wrapper (`classification_model.py`)
5. Implement trainer with TensorBoard logging (`trainer.py`)
6. Create configuration system (`configs/*.yaml`)
7. Write training scripts (`scripts/train_*.py`)
8. Write evaluation script (`scripts/evaluate_model.py`)
9. Train regression model
10. Train classification model
11. Evaluate both on test set
12. Compare against Phase 0 baselines

### To-dos

- [ ] Create base GRU encoder module
- [ ] Implement PyTorch Dataset for sequences
- [ ] Build regression model wrapper
- [ ] Build classification model wrapper
- [ ] Implement training loop with TensorBoard
- [ ] Create YAML configuration files
- [ ] Write training scripts
- [ ] Write evaluation script