# Stock Price Prediction with Temporal Fusion Transformer (TFT)

A deep learning project implementing a **Temporal Fusion Transformer** for financial time-series classification, combining technical indicators, earnings reports, and financial news sentiment analysis to predict short-term stock price movements.

## 📋 Overview

This project predicts stock price movements (Buy/Neutral/Sell) using a custom TFT architecture that processes:
- **Technical indicators**: OHLCV data with engineered features
- **Earnings calendar**: Quarterly earnings report dates
- **News sentiment**: Financial news embeddings with relevance scoring

The model achieves interpretable predictions through Variable Selection Networks (VSN) and directional focal loss, designed specifically for trading applications.

---

## 🏗️ Architecture

### Core Components

1. **Variable Selection Network (VSN)**
   - Automatically identifies relevant features at each time step
   - Uses Gated Residual Networks (GRN) for adaptive non-linear processing


2. **Stock-Specific TFT (`StockTFT`)**
   - Static stock embeddings for context preservation
   - Bidirectional LSTM for temporal encoding
   - Multi-head attention for global temporal relationships
   - Classifier head with GRN post-processing

3. **Custom Focal Loss**
   - Directional penalty to heavily penalize opposite-direction predictions
   - Configurable focus on hard samples (`gamma`) and trading accuracy (`dir_weight`)
   - Helps avoid catastrophic "Buy when Sell" mistakes

---

## 📊 Dataset Pipeline

### Raw Data Sources

The project orchestrates data from three streams:

| Data Type | Source API | Content |
|-----------|-----------|---------|
| **Technical** | Tiingo | Adjusted OHLCV prices |
| **Earnings** | Alpha Vantage | Quarterly earnings report dates |
| **News** | EODHD | Financial news articles with metadata |

### Feature Engineering

**Technical Features (13 dimensions):**
- Price patterns: `Gap`, `Body`, `UWick`, `LWick`
- Volume metrics: `LocalVolZ` (rolling z-score), `GlobalVolScale` (inflation-adjusted)
- Temporal encoding: `DOW_sin/cos`, `Month_sin/cos`, `DOM_sin/cos`
- Event flag: `ER_Flag` (earnings report indicator)

**News Features (769 dimensions per timestep):**
- FinBERT embeddings (768-dim) from top-weighted articles
- Net sentiment score (1-dim) from ProsusAI/finbert
- Relevance filtering via cross-encoder
- Weekend enrichment for Monday predictions

### Dataset Outputs

```
data/processed_data/
├── technical_ER_dataset.pt          # 13 features, Perfect Cube (all stocks × all dates)
└── technical_ER_news_dataset.pt     # 782 features, Top news-heavy stocks
```


---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate deep_final_project
```

### 2. API Key Configuration

Set up your API keys as environment variables:

```powershell
# Windows (PowerShell)
$env:TIINGO_API_KEY="your_tiingo_key"
$env:ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"
$env:EODHD_API_KEY="your_eodhd_key"
```

```bash
# Linux/Mac
export TIINGO_API_KEY="your_tiingo_key"
export ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"
export EODHD_API_KEY="your_eodhd_key"
```

**Get API Keys:**
- Tiingo: https://www.tiingo.com/
- Alpha Vantage: https://www.alphavantage.co/
- EODHD: https://eodhd.com/

### 3. Data Collection

```python
from src.data_loader import DataOrchestrator

# Initialize orchestrator
orchestrator = DataOrchestrator(
    api_keys={
        'tiingo': 'YOUR_KEY',
        'alpha_vantage': 'YOUR_KEY',
        'eodhd': 'YOUR_KEY'
    },
    fetch_range={'start': '2020-01-01', 'end': '2025-12-31'}
)

# Fetch data for your stock universe
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', ...]  # Up to 48 stocks recommended
orchestrator.sync_all(stocks)
```

### 4. Dataset Processing

```python
from src.data_processor import FeatureProcessor

processor = FeatureProcessor()

# Generate Technical + ER dataset (Perfect Cube)
processor.generate_tech_er_dataset(
    stocks=stocks,
    dates={'start': '2021-01-01', 'end': '2024-12-31'},
    M=60,           # Lookback window (60 trading days)
    T=5,            # Prediction horizon (5 days)
    multiplier=1.5  # ATR multiplier for labels
)

# Optional: Generate News-fused dataset
processor.generate_tech_er_news_dataset(
    stocks=stocks,
    dates={'start': '2021-01-01', 'end': '2024-12-31'},
    M=60,
    T=5,
    multiplier=1.5,
    news_threshold=20  # Top 20 stocks by news volume
)
```

### 5. Training

Open `notebooks/TFT.ipynb` and run all cells, or use the script directly:

```python
from src.TFT import StockTFT, CustomFocalLoss, MasterStockDataset, initialize_weights
import torch
from torch.utils.data import DataLoader

# Load dataset
ds = MasterStockDataset("data/processed_data/technical_ER_dataset.pt")

# Initialize model
model = StockTFT(
    input_dim=13,
    hidden_dim=64,
    num_layers=1,
    n_heads=8
).to(device)
model.apply(initialize_weights)

# Define loss and optimizer
criterion = CustomFocalLoss(
    alpha=ds.get_class_weights().to(device),
    gamma=1.6,
    dir_weight=0.6
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Train (see notebook for full training loop)
```

---

## 📁 Project Structure

```
deep_final_project/
├── src/
│   ├── data_loader.py        # Raw data orchestration
│   ├── data_processor.py     # Feature engineering & dataset generation
│   └── TFT.py                # Model architecture & loss functions
├── notebooks/
│   ├── TFT.ipynb             # Main training notebook
│   ├── TFT_data_loader.ipynb # Dataset exploration
│   └── final_simulation.ipynb # Backtesting & strategy evaluation
├── data/
│   ├── raw_data/
│   │   ├── technical_data/   # OHLCV CSVs
│   │   ├── earnings_data/    # Earnings dates
│   │   └── news_data/        # News JSONL files
│   └── processed_data/       # PyTorch .pt datasets
├── results/
│   └── training/             # Training runs with checkpoints & logs
├── environment.yml           # Conda environment specification
└── README.md
```

---

## 🎯 Training Configuration

Default hyperparameters (from `notebooks/TFT.ipynb`):

```python
TRAIN_CONFIG = {
    'epochs': 6,
    'lr': 1e-4,
    'patience': 15,
    'hidden_dim': 64,
    'num_layers': 1,
    'n_heads': 8,
    'gamma': 1.6,        # Focal loss focus parameter
    'dir_weight': 0.6    # Directional penalty weight
}
```



**Labeling Logic:**
- **Buy (2)**: Price exceeds `Open + (multiplier × ATR)` within T days
- **Sell (0)**: Price falls below `Open - (multiplier × ATR)` within T days
- **Neutral (1)**: Neither threshold is reached

---

## 📈 Results Interpretation

After training, the notebook generates:

1. **Loss & Accuracy Curves**: Track overfitting and convergence
2. **Feature Importance Plot**: VSN attention weights reveal which features drive predictions
3. **Test Set Evaluation**: Out-of-sample accuracy and label distribution analysis

Example output:
```
✅ FINAL TEST RESULTS for technical_ER_dataset_20260127_1919:
      | Accuracy: 46.67%
      | Loss:     0.44424
      | Dist:     Pred [0.0%/100.0%/0.0%] vs True [30.0%/46.7%/23.3%]
```

---

## 🔧 Extending the Model

### Adding More Features

Edit `src/data_processor.py`:
```python
self.feature_cols = [
    'Gap', 'UWick', 'LWick', 'Body', 'LocalVolZ', 'GlobalVolScale',
    'DOW_sin', 'DOW_cos', 'Month_sin', 'Month_cos', 'DOM_sin', 'DOM_cos',
    'ER_Flag',
    'YOUR_NEW_FEATURE'  # Add here
]
```


## 📚 Dependencies

**Core Libraries:**
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- Sentence-Transformers
- Pandas, NumPy, Matplotlib

**NLP Models:**
- `ProsusAI/finbert` (sentiment classification)
- `all-MiniLM-L6-v2` (semantic similarity)
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (relevance scoring)

See `environment.yml` for complete specifications.

---




## 🤝 Contributing

This is an academic research project. For questions or collaboration:
1. Check existing issues/TODOs in `TODOs.md`
2. Ensure reproducibility by documenting any parameter changes
3. Follow the existing code structure in `src/`

---

## 📄 License

This project is for educational and research purposes. 

**Disclaimer**: This model is for academic research only and should not be used for actual trading without proper risk management and further validation.

---

## 🙏 Acknowledgments

- **Temporal Fusion Transformer**: Inspired by [Lim et al. (2021)](https://arxiv.org/abs/1912.09363)
- **FinBERT**: ProsusAI and HKU teams for financial NLP models
- **APIs**: Tiingo, Alpha Vantage, and EODHD for financial data access

---

## 📞 Contact

For questions about this implementation, please open an issue in the repository.

---
