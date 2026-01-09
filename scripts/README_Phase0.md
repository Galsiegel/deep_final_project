# Phase 0: Naive Baselines

Establish baseline performance using simple prediction methods (Persistence, Moving Average, Linear Trend).

## Quick Start

```bash
# Run on test set with all stocks
python scripts/run_naive_baselines.py

# Run on specific tickers
python scripts/run_naive_baselines.py --tickers "AAPL,GOOG,MSFT"

# Quick test
python scripts/run_naive_baselines.py --limit 5 --no-visualize
```

## Key Arguments

- `--split`: train/val/test (default: test)
- `--tickers`: Comma-separated ticker list
- `--limit`: Limit number of stocks for testing
- `--no-visualize`: Skip plots (faster)
- `--no-timestamp`: Use simple results/ dir instead of timestamped runs

## Output Structure

Each run creates a timestamped directory:

```
results/phase0_runs/run_YYYYMMDD_HHMMSS/
  ├── config.json                # Run configuration
  ├── naive_baselines_test.json  # Results + metadata
  └── figures/                   # 6 visualizations
```

## Metrics

- **MSE, MAE, RMSE**: Price prediction error (lower is better)
- **Directional Accuracy**: Predicted up/down correctly (higher is better)

## Example Results (AAPL)

```
Method                     MSE      MAE     Dir Acc
Persistence (t-1)         1.13     0.83      48%
Moving Average (5-day)    1.77     1.06      50%
Linear Trend (5-day)      1.29     0.89      53%
```

Neural networks (Phase 1+) must beat these baselines!
