"""
src/data_loader.py
The 'Engine': Orchestrates the raw acquisition of data.
Saves files into a strict hierarchical raw_data structure with date-overlap checks.
"""

import time
import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from tqdm.auto import tqdm

class DataOrchestrator:
    def __init__(self, api_keys, fetch_range, base_dir="../data"):
        self.keys = api_keys
        self.start_date = fetch_range['start']
        self.end_date = fetch_range['end']

        # --- Directory Mapping ---
        self.base_path = Path(base_dir)
        self.technical_dir = self.base_path / "raw_data" / "technical_data"
        self.earnings_dir = self.base_path / "raw_data" / "earnings_data"
        self.news_dir = self.base_path / "raw_data" / "news_data"

        for d in [self.technical_dir, self.earnings_dir, self.news_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _check_date_overlap(self, ticker, data_type):
        """Checks existing files for date coverage and returns overlapping range or None."""
        path = None
        date_col = 'Date'

        if data_type == 'prices':
            path = self.technical_dir / f"{ticker}_RAW.csv"
        elif data_type == 'earnings':
            path = self.earnings_dir / f"{ticker}_ER.csv"
            date_col = 'report_date'
        elif data_type == 'news':
            path = self.news_dir / f"{ticker}_NEWS.jsonl"

        if not path or not path.exists():
            return None

        try:
            if data_type == 'news':
                # For jsonl, read first and last line to find dates
                with open(path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if not lines: return None
                    dates = [pd.to_datetime(json.loads(l)['date']).date() for l in [lines[0], lines[-1]]]
            else:
                df = pd.read_csv(path)
                if df.empty: return None
                dates = pd.to_datetime(df[date_col]).dt.date.tolist()

            req_start = pd.to_datetime(self.start_date).date()
            req_end = pd.to_datetime(self.end_date).date()

            # Find overlap within the requested range
            existing = [d for d in dates if req_start <= d <= req_end]
            if existing:
                return min(existing), max(existing)
        except:
            pass
        return None

    def fetch_prices(self, ticker):
        """Fetches technical data with overlap check."""
        overlap = self._check_date_overlap(ticker, 'prices')
        if overlap:
            print(f"prices data for {ticker} exists for {overlap[0]} - {overlap[1]}, skipping")
            return True

        print(f"Fetching {ticker} prices")
        path = self.technical_dir / f"{ticker}_RAW.csv"
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        params = {
            'startDate': self.start_date,
            'endDate': self.end_date,
            'token': self.keys.get('tiingo'),
            'format': 'json'
        }
        try:
            res = requests.get(url, params=params)
            if res.status_code == 200 and res.json():
                df = pd.DataFrame(res.json())
                df = df[['date', 'adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'adjVolume']]
                df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                df['Date'] = pd.to_datetime(df['Date']).dt.date
                df.to_csv(path, index=False)
                return True
        except Exception as e:
            print(f"Warning: Price Fetch Error [{ticker}]: {e}")
        return False

    def fetch_earnings(self, ticker):
        """Fetches earnings data with overlap check."""
        overlap = self._check_date_overlap(ticker, 'earnings')
        if overlap:
            print(f"earnings data for {ticker} exists for {overlap[0]} - {overlap[1]}, skipping")
            return True

        print(f"Fetching {ticker} earnings")
        path = self.earnings_dir / f"{ticker}_ER.csv"
        url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={self.keys.get('alpha_vantage')}"
        try:
            res = requests.get(url).json()
            if "quarterlyEarnings" in res:
                df = pd.DataFrame(res["quarterlyEarnings"])[['reportedDate']]
                df.columns = ['report_date']
                df.to_csv(path, index=False)
                return True
        except Exception as e:
            print(f"Warning: Earnings Fetch Error [{ticker}]: {e}")
        return False

    def fetch_news(self, ticker):
        """Fetches news data with overlap check."""
        overlap = self._check_date_overlap(ticker, 'news')
        if overlap:
            print(f"news data for {ticker} exists for {overlap[0]} - {overlap[1]}, skipping")
            return True

        print(f"Fetching {ticker} news")
        path = self.news_dir / f"{ticker}_NEWS.jsonl"
        cursor = datetime.strptime(self.end_date, "%Y-%m-%d").date()
        limit_date = datetime.strptime(self.start_date, "%Y-%m-%d").date()

        with open(path, "w", encoding='utf-8') as f:
            while cursor >= limit_date:
                window_start = max(cursor - timedelta(days=30), limit_date)
                url = (f"https://eodhd.com/api/news?s={ticker}.US&api_token={self.keys.get('eodhd')}"
                       f"&from={window_start}&to={cursor}&limit=1000&fmt=json")
                try:
                    res = requests.get(url)
                    if res.status_code == 429:
                        time.sleep(60); continue
                    if not res.ok: break

                    articles = res.json()
                    if not articles or not isinstance(articles, list):
                        cursor = window_start - timedelta(days=1); continue

                    for art in articles:
                        art['_meta_ticker'] = ticker
                        f.write(json.dumps(art) + "\n")

                    if len(articles) == 1000:
                        cursor = datetime.fromisoformat(articles[-1]['date']).date() - timedelta(days=1)
                    else:
                        cursor = window_start - timedelta(days=1)
                    time.sleep(0.2)
                except: break
        return True

    def sync_all(self, stocks_to_process, earnings_limit=24):
        """Main coordination loop."""
        e_count = 0
        has_tiingo = self.keys.get('tiingo') is not None
        has_av = self.keys.get('alpha_vantage') is not None
        has_eodhd = self.keys.get('eodhd') is not None

        for ticker in tqdm(stocks_to_process, desc="Syncing Raw Data"):
            if has_tiingo:
                self.fetch_prices(ticker)
                time.sleep(0.5)

            if has_av and e_count < earnings_limit:
                if self.fetch_earnings(ticker): e_count += 1

            if has_eodhd:
                self.fetch_news(ticker)

        print(f"\nSynchronization Complete. Data updated for {len(stocks_to_process)} tickers.")