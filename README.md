# Deep Final Project
Neural Network to Predict Stock Prices

## Overview
This project uses financial news data from the Massive (formerly polygon.io) API to train a neural network for stock price prediction.

## Massive News API Setup

### Prerequisites
- Python 3.7 or higher
- A Massive API key (get one at https://massive.com or https://polygon.io)

### Installation

1. **Install required packages:**
   ```bash
   # Using conda (recommended)
   conda env create -f environment.yml
   conda activate deep_final_project
   ```

2. **Set up your API key:**
   
   Option A: Using environment variable (recommended)
   ```bash
   # On Windows (PowerShell)
   $env:MASSIVE_API_KEY="your_api_key_here"
   
   # On Windows (Command Prompt)
   set MASSIVE_API_KEY=your_api_key_here
   
   # On Linux/Mac
   export MASSIVE_API_KEY="your_api_key_here"
   ```
   
   Option B: Using a .env file
   - Create a file named `.env` in the project root
   - Add: `MASSIVE_API_KEY=your_api_key_here`
   - The `python-dotenv` package will automatically load it

   Option C: Pass directly in code
   ```python
   client = MassiveNewsClient(api_key="your_api_key_here")
   ```

### Quick Start

```python
from Massive.news_client import MassiveNewsClient

# Initialize the client
client = MassiveNewsClient()

# Get latest news for Apple
articles = client.get_news_by_ticker("AAPL", limit=10)

# Print article titles
for article in articles:
    print(article['title'])
```

### Running Examples

Try the example script to see the API in action:

```bash
python Massive/example_usage.py
```

This will demonstrate:
- Basic news fetching
- Advanced filtering by date ranges
- Article formatting
- Preparing text for embeddings
- Sentiment analysis

### API Client Features

The `MassiveNewsClient` class provides:

- **`get_news()`** - Full-featured method with all filtering options
- **`get_news_by_ticker()`** - Simple method for getting news by ticker
- **`get_all_news_pages()`** - Fetch all pages of results automatically
- **`format_article()`** - Format articles for display
- **`extract_text_for_embedding()`** - Prepare text for embedding networks

### Documentation

For detailed API documentation, see `Massive/news.md`.

### Next Steps: Connecting to Embedding Networks

The client includes a `extract_text_for_embedding()` method that combines article text into a format suitable for embedding models. This makes it easy to:

1. Fetch news articles
2. Extract text content
3. Generate embeddings
4. Use embeddings for neural network training

Example:
```python
client = MassiveNewsClient()
articles = client.get_news_by_ticker("AAPL", limit=100)

# Prepare text for embeddings
texts = [client.extract_text_for_embedding(article) for article in articles]

# Now pass 'texts' to your embedding model
# embeddings = your_embedding_model.encode(texts)
```
