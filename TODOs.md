**Project breakdown (for efficiency + parallel work)**
This file breaks the assignment into steps so we can work more efficiently and in parallel.

**Data**

1. Find or create a stock-market dataset with standard market features (OHLCV, etc.).
   This should be fairly easy—there are lots of APIs and ready-made datasets.
   
2. Create a news dataset linked to the stock market. Make sure it includes:
     tags / tickers
     publication date (and ideally time of day)
   This is harder and may require a paid API.
3. Build an enhanced dataset by adding embedding outputs to the relevant stock data (align by ticker + time window).
4. Bonus: add global / macro news as an additional signal.


**Embedding model**

1. Add the embedding model to the project — Done

**Stock prediction model**

1. Build a basic stock prediction model.
2. Train it on raw stock-market data only (no news).
3. Optimize it as much as possible and present results.
4. Retrain the model using the enhanced dataset (stock + embeddings).
