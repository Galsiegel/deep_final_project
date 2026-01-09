<!-- 715514cb-ab53-4a55-8191-e33d56afef99 a8326a0e-093b-40b8-92a9-d09d0a6d4483 -->
# Stock Prediction Network Plan

## Overview

Implement a modular stock prediction system with GRU architecture supporting:

- Dual prediction modes: regression (price) and classification (direction)
- Baseline training on stock data only (OHLCV)
- Future embedding integration via two strategies: simple concatenation and separate branches

## Architecture Design

### Core Model Components

```
Input: [Date, OHLCV, Volume] → Lookback Window → GRU → Prediction Head
```

**Model Structure:**

- **Input Layer**: Processes sequences of stock features (lookback window)
- **GRU Backbone**: 1-3 layer GRU with configurable hidden size
- **Dual Output Heads**: 
  - Regression head: predicts next-day closing price
  - Classification head: predicts direction (up/down/neutral)

**Key Design Decisions:**

- Use GRU over LSTM for efficiency
- Modular architecture allowing easy embedding integration later
- Separate prediction heads for easy task comparison

### Data Pipeline

**Files to use:**

- `data/2017_2019/*.csv` - 198 stock tickers with OHLCV data (2017-2019)
- Format: Date, Open, High, Low, Close, Volume, Dividends, Stock Splits

**Preprocessing:**

1. Load and normalize OHLCV features (min-max or z-score per ticker)
2. Create sliding windows (lookback_window days)
3. Generate targets:

   - Regression: next day's closing price (raw value)
   - Classification: price movement direction with multiple granularity options:
     - 3-class: Down (<0%), Neutral (0%), Up (>0%)
     - 5-class: Large Down (<-2%), Small Down (-2% to -0.5%), Neutral (-0.5% to +0.5%), Small Up (+0.5% to +2%), Large Up (>+2%)
     - 7-class: Very Large Down (<-3%), Large Down (-3% to -1.5%), Small Down (-1.5% to -0.5%), Neutral (-0.5% to +0.5%), Small Up (+0.5% to +1.5%), Large Up (+1.5% to +3%), Very Large Up (>+3%)

4. Train/val/test split with strict temporal ordering to prevent data leakage:

   - Temporal split with gaps: Train → Gap (lookback days) → Val → Gap → Test
   - Example: Train (2017-01 to 2018-06), Gap (20 days), Val (2018-07 to 2018-12), Gap (20 days), Test (2019-01 to 2019-12)
   - No shuffling or random selection - maintains chronological order
   - Same date ranges applied across all tickers

## Implementation Phases

### Phase 0: Naive Baselines (Essential Reference Point)

Before building neural networks, establish simple baseline predictions to ensure our models provide real value.

**Naive methods to implement:**

1. **Persistence (Last Value)**: 

   - Prediction = yesterday's closing price
   - Simplest baseline - assumes no change

2. **Moving Average (5-day)**:

   - Prediction = mean of last 5 days' closing prices
   - Smooths out short-term fluctuations

3. **Linear Trend Extrapolation**:

   - Fit linear regression to last 5-10 days
   - Extrapolate to predict next day

4. **Random Classifier** (for classification only):

   - Predict each class with equal probability
   - Establishes chance-level performance

**Implementation:**

- Create `scripts/naive_baselines.py` - Compute all naive predictions
- Save results to `results/naive_baselines.json` for comparison
- Must use same train/val/test splits as neural models
- Compute all metrics (MSE, MAE, Accuracy, F1) for fair comparison

**Why this matters:** If GRU doesn't beat persistence baseline, the model isn't learning useful patterns!

### Phase 1: Baseline Model (Stock Only)

**Create:**

- `models/stock_predictor.py` - Main model architecture
- `models/data_loader.py` - Dataset class and data loading utilities
- `models/trainer.py` - Training loop, validation, and logging
- `models/config.py` - Hyperparameters and configuration
- `scripts/train_baseline.py` - Training script for baseline model

**Key Features:**

- Configurable GRU layers (1-3 layers, hidden_size=64/128/256)
- Dropout for regularization
- Both regression and classification training modes
- Checkpoint saving and loading
- Metrics: MSE/MAE for regression, Accuracy/F1 for classification

**Hyperparameters to tune:**

- Lookback window: 5, 10, 20, 30 days
- Hidden size: 64, 128, 256
- Number of GRU layers: 1, 2, 3
- Dropout: 0.1, 0.2, 0.3
- Learning rate: 1e-3, 1e-4
- Batch size: 32, 64, 128

### Phase 2: Embedding Integration (Future)

**Two integration strategies to implement:**

**(a) Simple Concatenation:**

```
Input: [OHLCV + NewsEmbedding] → GRU → Prediction
```

- Minimal code changes
- News embeddings concatenated with stock features at each timestep

**(b) Separate Branches:**

```
Stock: [OHLCV] → GRU → stock_features
News: [Embeddings] → Dense → news_features
Combined: [stock_features, news_features] → Prediction
```

- Stock branch can reuse baseline model weights
- News branch processes embeddings separately
- Fusion layer combines both modalities

**Files to modify/create:**

- Extend `models/stock_predictor.py` with embedding-aware models
- Update `models/data_loader.py` to load news embeddings
- Create `scripts/train_enhanced.py` for training with embeddings

## Evaluation & Comparison

**Metrics to track:**

- **Regression**: MSE, MAE, RMSE, directional accuracy
- **Classification**: Accuracy, Precision, Recall, F1-score
- **Financial**: Sharpe ratio (if implementing trading strategy)

**Comparison matrix:**

```
Model                    | MSE    | MAE   | RMSE  | Direction Acc | F1 (5-class)
-------------------------|--------|-------|-------|---------------|-------------
Persistence (t-1)        |   ?    |   ?   |   ?   |      ?        | ?
Moving Avg (5-day)       |   ?    |   ?   |   ?   |      ?        | ?
Linear Trend             |   ?    |   ?   |   ?   |      ?        | ?
Random Classifier        |   -    |   -   |   -   |      ?        | ?
-------------------------|--------|-------|-------|---------------|-------------
GRU Baseline (stock)     |   ?    |   ?   |   ?   |      ?        | ?
GRU Enhanced-Concat (a)  |   ?    |   ?   |   ?   |      ?        | ?
GRU Enhanced-Branch (b)  |   ?    |   ?   |   ?   |      ?        | ?
```

**Goal:** Neural models must significantly outperform naive baselines to justify complexity.

## Key Files

- `models/stock_predictor.py` - Neural network architectures
- `models/data_loader.py` - Data loading and preprocessing
- `models/trainer.py` - Training and evaluation logic
- `models/config.py` - Configuration management
- `scripts/train_baseline.py` - Baseline training script
- `results/` - Output directory for checkpoints and logs