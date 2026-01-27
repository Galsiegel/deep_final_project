"""
src/data_processor.py
The 'Chef': Processes raw technical, earnings, and news data into ML-ready datasets.
Saves two outputs:
1. Technical + ER Dataset (13 features, Perfect Cube)
2. Technical + ER + News Dataset (782 features, Top News-Heavy Stocks)
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime, timedelta
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder, SentenceTransformer, util

class FeatureProcessor:
    def __init__(self, base_dir="../data"):
        self.base_path = Path(base_dir)
        self.raw_tech_dir = self.base_path / "raw_data" / "technical_data"
        self.raw_er_dir = self.base_path / "raw_data" / "earnings_data"
        self.raw_news_dir = self.base_path / "raw_data" / "news_data"
        self.output_dir = self.base_path / "processed_data"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.feature_cols = [
            'Gap', 'UWick', 'LWick', 'Body', 'LocalVolZ', 'GlobalVolScale',
            'DOW_sin', 'DOW_cos', 'Month_sin', 'Month_cos', 'DOM_sin', 'DOM_cos',
            'ER_Flag'
        ]

    # --- 1. Internal Helpers ---

    def _get_inflation_factor(self, year):
        factors = {2020: 1.25, 2021: 1.20, 2022: 1.12, 2023: 1.07, 2024: 1.03, 2025: 1.00}
        return factors.get(year, 1.0)

    def _get_window_label(self, df, t, multiplier, target_days):
        """Labeling logic: 0=Sell, 1=Neutral, 2=Buy based on ATR bands."""
        try:
            anchor_open = df.iloc[t+1]['Open']
            atr = df.iloc[t]['ATR_Abs']
            buy_lvl, sell_lvl = anchor_open + (multiplier * atr), anchor_open - (multiplier * atr)

            for i in range(1, target_days + 1):
                idx = t + i
                if idx >= len(df): break
                h, l = df.iloc[idx]['High'], df.iloc[idx]['Low']
                if h >= buy_lvl: return 2
                if l <= sell_lvl: return 0
            return 1
        except: return 1

    def _calculate_global_vol_bounds(self, stocks):
        all_log_v = []
        for ticker in stocks:
            path = self.raw_tech_dir / f"{ticker}_RAW.csv"
            if not path.exists(): continue
            df = pd.read_csv(path)
            v = (df['Close'] * df['Volume'] * pd.to_datetime(df['Date']).dt.year.apply(self._get_inflation_factor)).replace(0, 1)
            all_log_v.append(np.log(v))
        if not all_log_v: return 0, 1
        combined = pd.concat(all_log_v)
        return combined.min(), combined.max()

    def _process_stock_features(self, ticker, v_min, v_max):
        p_path = self.raw_tech_dir / f"{ticker}_RAW.csv"
        e_path = self.raw_er_dir / f"{ticker}_ER.csv"

        if not p_path.exists():
            print(f"⚠️ Price data missing for {ticker}. Skipping...")
            return None

        df = pd.read_csv(p_path).dropna().reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'])

        er_dates = pd.to_datetime(pd.read_csv(e_path)['report_date']).dt.date.values if e_path.exists() else []
        df['ER_Flag'] = df['Date'].dt.date.isin(er_dates).astype(float)

        df['DOW'] = df['Date'].dt.dayofweek
        df['DOW_sin'], df['DOW_cos'] = np.sin(2*np.pi*df['DOW']/7), np.cos(2*np.pi*df['DOW']/7)
        df['Month_sin'] = np.sin(2*np.pi*(df['Date'].dt.month-1)/12)
        df['Month_cos'] = np.cos(2*np.pi*(df['Date'].dt.month-1)/12)
        dim = df['Date'].dt.days_in_month.astype(float)
        df['DOM_sin'] = np.sin(2*np.pi*(df['Date'].dt.day-1)/dim)
        df['DOM_cos'] = np.cos(2*np.pi*(df['Date'].dt.day-1)/dim)

        df['Gap'] = (df['Open'] / df['Close'].shift(1)) - 1
        df['Body'] = (df['Close'] / df['Open']) - 1
        df['UWick'] = (df['High'] / df[['Open', 'Close']].max(axis=1)) - 1
        df['LWick'] = (df['Low'] / df[['Open', 'Close']].min(axis=1)) - 1
        df['LogV'] = np.log((df['Close'] * df['Volume'] * df['Date'].dt.year.apply(self._get_inflation_factor)).replace(0, 1))
        df['GlobalVolScale'] = 2 * (df['LogV'] - v_min) / (v_max - v_min) - 1

        tr = pd.concat([(df['High']-df['Low']), (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
        df['ATR_Abs'] = tr.rolling(window=14).mean()

        return df.dropna().reset_index(drop=True)

    # --- 2. News Fusion Engine ---

    def _get_daily_news_vectors(self, ticker, df_tech, bi_encoder, cross_encoder, tokenizer, model, sentiment_pipe):
        """Aggregates news into 769-dim daily vectors using weights and relevance."""
        path = self.raw_news_dir / f"{ticker}_NEWS.jsonl"
        if not path.exists(): return {}

        with open(path, 'r', encoding='utf-8') as f:
            raw_data = [json.loads(line) for line in f if line.strip()]

        news_df = pd.DataFrame(raw_data)
        news_df['day'] = pd.to_datetime(news_df['date']).dt.strftime('%Y-%m-%d')
        daily_map = {}

        for day in sorted(news_df['day'].unique()):
            current_articles = news_df[news_df['day'] == day].to_dict('records')

            # Weekend Enrichment for Monday
            dt = pd.to_datetime(day)
            if dt.weekday() == 0:
                for d_off, penalty in [(1, 0.5), (2, 1.0)]:
                    prev_day = (dt - timedelta(days=d_off)).strftime('%Y-%m-%d')
                    prev_arts = news_df[news_df['day'] == prev_day].to_dict('records')
                    for a in prev_arts: a['weekend_penalty'] = penalty
                    current_articles.extend(prev_arts)

            if not current_articles: continue

            # Duplicate Filter & Relevance
            titles = [a['title'] for a in current_articles]
            embeddings = bi_encoder.encode(titles, convert_to_tensor=True, show_progress_bar=False)
            cosine_scores = util.cos_sim(embeddings, embeddings)

            kept_indices = []
            for i in range(len(current_articles)):
                if all(cosine_scores[i][k] < 0.9 for k in kept_indices): kept_indices.append(i)

            clean_articles = [current_articles[i] for i in kept_indices]
            passages = [f"{a['title']} {a.get('description', '')}" for a in clean_articles]
            rel_scores = cross_encoder.predict([(f"Financial news for {ticker}", p) for p in passages], show_progress_bar=False)

            valid_idx = [i for i, s in enumerate(rel_scores) if s >= -1.8]
            if not valid_idx: continue

            # Weighted Aggregation (Top 10)
            top_idx = valid_idx[:10]
            weights = np.exp(rel_scores[top_idx] - np.max(rel_scores[top_idx]))
            weights /= np.sum(weights)

            # FinBERT Embeddings + Sentiment
            inputs = tokenizer([clean_articles[i]['title'] for i in top_idx], return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                cls_emb = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()

            sent_outs = sentiment_pipe([clean_articles[i]['title'] for i in top_idx])
            net_sent = [next(item['score'] for item in res if item['label']=='positive') -
                        next(item['score'] for item in res if item['label']=='negative') for res in sent_outs]

            daily_map[day] = np.append(np.average(cls_emb, axis=0, weights=weights), np.average(net_sent, weights=weights))

        return daily_map

    # --- 3. Generation Methods ---

    def generate_tech_er_dataset(self, stocks, dates, M, T, multiplier):
        """Generates the Technical + ER dataset as a Perfect Cube."""
        stocks = sorted(stocks)
        v_min, v_max = self._calculate_global_vol_bounds(stocks)
        all_samples, active_stocks = [], []

        for ticker in tqdm(stocks, desc="Processing Technicals"):
            df = self._process_stock_features(ticker, v_min, v_max)
            if df is None: continue
            active_stocks.append(ticker)
            ticker_id = len(active_stocks) - 1

            for t in range(M - 1, len(df) - T):
                if df.iloc[t]['Date'] < pd.Timestamp(dates['start']) or df.iloc[t]['Date'] > pd.Timestamp(dates['end']): continue

                v_slice = df.iloc[t-(M-1):t+1]['LogV']
                df_win = df.iloc[t-(M-1):t+1].copy()
                df_win['LocalVolZ'] = ((df_win['LogV'] - v_slice.mean()) / (v_slice.std() + 1e-9)).clip(-3, 3) / 3.0

                all_samples.append({
                    'x': df_win[self.feature_cols].values.astype(np.float32),
                    'static_id': ticker_id,
                    'y': self._get_window_label(df, t, multiplier, T),
                    'metadata': {'ticker': ticker, 'date': df.iloc[t]['Date'].strftime('%Y-%m-%d')}
                })

        date_counts = Counter([s['metadata']['date'] for s in all_samples])
        valid_dates = {d for d, count in date_counts.items() if count == len(active_stocks)}
        cube = sorted([s for s in all_samples if s['metadata']['date'] in valid_dates], key=lambda x: (x['metadata']['date'], x['static_id']))
        torch.save(cube, self.output_dir / "technical_ER_dataset.pt")
        print(f"✅ Tech+ER Cube Saved: {len(cube)} samples.")

    def generate_tech_er_news_dataset(self, stocks, dates, M, T, multiplier, news_threshold):
        """Fuses Technical + ER + News NLP into a 782-dim master dataset."""
        news_counts = []
        for s in stocks:
            p = self.raw_news_dir / f"{s}_NEWS.jsonl"
            if p.exists():
                with open(p, 'rb') as f: news_counts.append((s, sum(1 for _ in f)))

        if not news_counts:
            print("⚠️ No news data found. Skipping fused dataset.")
            return

        top_stocks = sorted([x[0] for x in sorted(news_counts, key=lambda x: x[1], reverse=True)[:min(len(news_counts), news_threshold)]])
        v_min, v_max = self._calculate_global_vol_bounds(top_stocks)

        # Initialize NLP Models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bi_enc = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        cross_enc = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
        tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        model = AutoModel.from_pretrained('yiyanghkust/finbert-tone').to(device)
        sent_pipe = pipeline("text-classification", model="ProsusAI/finbert", device=(0 if device=="cuda" else -1), top_k=None)

        all_fused = []
        for ticker in tqdm(top_stocks, desc="Fusing News + Tech"):
            df_tech = self._process_stock_features(ticker, v_min, v_max)
            if df_tech is None: continue

            news_map = self._get_daily_news_vectors(ticker, df_tech, bi_enc, cross_enc, tokenizer, model, sent_pipe)
            ticker_id = top_stocks.index(ticker)

            for t in range(M - 1, len(df_tech) - T):
                if df_tech.iloc[t]['Date'] < pd.Timestamp(dates['start']) or df_tech.iloc[t]['Date'] > pd.Timestamp(dates['end']): continue

                v_slice = df_tech.iloc[t-(M-1):t+1]['LogV']
                df_win = df_tech.iloc[t-(M-1):t+1].copy()
                df_win['LocalVolZ'] = ((df_win['LogV'] - v_slice.mean()) / (v_slice.std() + 1e-9)).clip(-3, 3) / 3.0

                tech_x = df_win[self.feature_cols].values
                news_x = np.array([news_map.get(df_tech.iloc[i]['Date'].strftime('%Y-%m-%d'), np.zeros(769)) for i in range(t-(M-1), t+1)])

                all_fused.append({
                    'x': np.hstack([tech_x, news_x]).astype(np.float32),
                    'static_id': ticker_id,
                    'y': self._get_window_label(df_tech, t, multiplier, T),
                    'metadata': {'ticker': ticker, 'date': df_tech.iloc[t]['Date'].strftime('%Y-%m-%d')}
                })

        date_counts = Counter([s['metadata']['date'] for s in all_fused])
        valid_dates = {d for d, count in date_counts.items() if count == len(top_stocks)}
        cube = sorted([s for s in all_fused if s['metadata']['date'] in valid_dates], key=lambda x: (x['metadata']['date'], x['static_id']))
        torch.save(cube, self.output_dir / "technical_ER_news_dataset.pt")
        print(f"✅ Master Fused Dataset Saved: {len(cube)} samples.")