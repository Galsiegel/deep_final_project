# Stock Price Prediction with Temporal Fusion Transformer (TFT)

A dual-path TFT for financial time-series classification, combining technical indicators, FinBERT news embeddings, and stock identity to predict short-term price movements (Buy / Neutral / Sell).

---

## Architecture

Three parallel input streams merge before temporal processing:

| Path | Input | Module | Output |
|------|-------|--------|--------|
| **1 — VSN** | `x_tech [B,M,17]` (13 technical + 4 sentiment) | Variable Selection Network (17 GRNs + Softmax) | `[B,M,64]` |
| **2 — EmbeddingBranch** | `x_emb [B,M,768]` (FinBERT [CLS]) | 768→256→64, LayerNorm, GELU | `[B,M,64]` |
| **3 — Stock Embedding** | `sid [B]` (stock index) | Embed(256)→Proj(64), LayerNorm | `[B,M,64]` |

**Post-merge:** Concat `[B,M,192]` → LayerNorm + Linear → `[B,M,64]` → Bidirectional LSTM → Multi-Head Attention (8 heads) → GRN → Classifier → `[B,3]`

**Loss:** Custom Focal Loss with directional penalty (`gamma`, `dir_weight`) to avoid opposite-direction predictions.

---

## Features

| Category | Details |
|----------|---------|
| **Technical (13)** | Gap, Body, UWick, LWick, LocalVolZ, GlobalVolScale, DOW/Month/DOM sin/cos, ER_Flag |
| **Sentiment (4)** | Net_Sent, Positive, Negative, Neutral (EODHD / FinBERT) |
| **News Embedding (768)** | FinBERT [CLS] from top daily article per stock |

**Labeling:** Buy if price > `Open + ATR×1.5` within 5 days; Sell if < `Open - ATR×1.5`; else Neutral.

---

## Quick Start

```bash
conda env create -f environment.yml && conda activate deep_final_project
```

Set API keys (`TIINGO_API_KEY`, `ALPHA_VANTAGE_API_KEY`, `EODHD_API_KEY`), then:

```python
# 1. Fetch raw data
from src.data_loader import DataOrchestrator
orchestrator = DataOrchestrator(api_keys={...}, fetch_range={...})
orchestrator.sync_all(stocks)

# 2. Extract news features (GPU recommended)
from src.news_processor import NewsProcessor
NewsProcessor().process_all_tickers(stocks, "data/raw_data/news_data", "data/processed_data/daily_news_features.jsonl")

# 3. Generate dual-path dataset
from src.data_processor import FeatureProcessor
FeatureProcessor().generate_dual_path_dataset(stocks, dates, M=60, T=5, multiplier=1.5, news_jsonl_path="...")

# 4. Train
from src.TFT import StockTFT, CustomFocalLoss, MasterStockDataset
ds = MasterStockDataset("data/processed_data/dual_path_dataset.pt")
model = StockTFT(tech_input_dim=17, embedding_dim=768, num_stocks=ds.num_stocks, use_news=True).to(device)
```

Or simply open **`notebooks/TFT.ipynb`** and run all cells.

---

## Project Structure

```
src/
  data_loader.py          # Raw data orchestration (Tiingo, Alpha Vantage, EODHD)
  data_processor.py       # Feature engineering & dataset generation
  news_processor.py       # News → FinBERT embeddings + sentiment
  sentiment_loader.py     # Fast (ticker, date) lookup for news features
  TFT.py                  # Model, loss, dataset classes
notebooks/
  TFT.ipynb               # Training notebook
  TFT_data_loader.ipynb   # Dataset exploration & generation
  generate_sentiment_dataset.ipynb
  final_simulation.ipynb  # Backtesting
data/
  raw_data/               # OHLCV, earnings, news JSONLs
  processed_data/         # .pt datasets + daily_news_features.jsonl
```

---

## Configuration

```python
TRAIN_CONFIG = {
    'epochs': 6, 'lr': 1e-4, 'patience': 15,
    'hidden_dim': 64, 'num_layers': 1, 'n_heads': 8,
    'gamma': 1.6, 'dir_weight': 0.6, 'use_news': True
}
```

Set `use_news=False` to disable the EmbeddingBranch and fall back to a 2-path model (VSN + Stock Embedding).

---

## Dependencies

PyTorch ≥ 2.0, Transformers ≥ 4.30, Sentence-Transformers, Pandas, NumPy, Matplotlib. NLP models: `ProsusAI/finbert`, `all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2`. See `environment.yml`.

---

## Acknowledgments

- [Temporal Fusion Transformer — Lim et al. (2021)](https://arxiv.org/abs/1912.09363)
- [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert)
- Data APIs: Tiingo, Alpha Vantage, EODHD

*Academic research project — not intended for live trading.*
