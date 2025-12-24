# Given we will fetch big amounts of data, decide how to save the results.
# Dont forget to set limit

from tools.api_utils import load_api_key

# Load API key from Massive\API_KEYS file
MASSIVE_API_KEY = load_api_key()


from massive import RESTClient
# Use the variable, not the literal string
client = RESTClient(api_key=MASSIVE_API_KEY)

ticker = "AAPL"

# List Aggregates (Bars)
# Aggregates are OHLCV (Open, High, Low, Close, Volume) data bars
aggs = []
for a in client.list_aggs(
    ticker=ticker,           # Stock symbol (e.g., "AAPL" for Apple Inc.)
    multiplier=1,            # Number of timespan units per bar (1 = single unit bars)
    timespan="minutes",        # Time unit for each bar: "minute", "hour", "day", "week", "month"
    from_="2025-02-01",      # Start date for the data range (YYYY-MM-DD format)
    to="2025-02-02",         # End date for the data range (YYYY-MM-DD format)
    limit=5                  # Maximum number of bars to return (max 50000 per API docs)
):
    aggs.append(a)

print(aggs)

# # Get Last Trade
# trade = client.get_last_trade(ticker=ticker)
# print(trade)

# # List Trades
# trades = client.list_trades(ticker=ticker, timestamp="2025-01-04")
# for trade in trades:
#     print(trade)

# # Get Last Quote
# quote = client.get_last_quote(ticker=ticker)
# print(quote)

# # List Quotes
# quotes = client.list_quotes(ticker=ticker, timestamp="2022-01-04")
# for quote in quotes:
#     print(quote)