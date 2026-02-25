"""
src/sentiment_loader.py
Loads pre-computed daily news features (embedding + sentiment) from JSONL
and provides fast lookup by (ticker, date) for dataset generation.
"""

import json
import numpy as np
from pathlib import Path


EMBEDDING_DIM = 768
SENTIMENT_DIM = 4  # net, positive, negative, neutral
TOTAL_NEWS_DIM = EMBEDDING_DIM + SENTIMENT_DIM  # 772


class SentimentFeatureLoader:
    """
    Loads pre-computed news features and provides separate lookups for:
      - embedding: 768-dim FinBERT [CLS] vector of top article
      - sentiment: 4-dim [net, positive, negative, neutral]
    """

    def __init__(self, news_jsonl_path):
        self.embedding_lookup = {}   # (ticker, date) -> np.array[768]
        self.sentiment_lookup = {}   # (ticker, date) -> np.array[4]
        self._load_features(news_jsonl_path)

    def _load_features(self, path):
        path = Path(path)
        if not path.exists():
            print(f"Warning: {path} not found. News features will default to zeros.")
            return

        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ticker = row.get('ticker')
                date = row.get('date')
                features = row.get('features', [])

                if not ticker or not date or len(features) < TOTAL_NEWS_DIM:
                    continue

                key = (ticker, date)
                self.embedding_lookup[key] = np.array(
                    features[:EMBEDDING_DIM], dtype=np.float32
                )
                self.sentiment_lookup[key] = np.array(
                    features[EMBEDDING_DIM:TOTAL_NEWS_DIM], dtype=np.float32
                )
                count += 1

        print(f"Loaded {count} news feature entries from {path.name}")

    def get_embedding(self, ticker, date_str):
        """Returns 768-dim embedding vector, or zeros if not available."""
        return self.embedding_lookup.get(
            (ticker, date_str),
            np.zeros(EMBEDDING_DIM, dtype=np.float32)
        )

    def get_sentiment(self, ticker, date_str):
        """Returns 4-dim sentiment [net, pos, neg, neu], defaults to neutral."""
        return self.sentiment_lookup.get(
            (ticker, date_str),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        )

    def has_news(self, ticker, date_str):
        """Check if news data exists for this ticker/date."""
        return (ticker, date_str) in self.embedding_lookup

    def __len__(self):
        return len(self.embedding_lookup)
