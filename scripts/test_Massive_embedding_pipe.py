"""

This test will get a news output feom massive, and send it to embedding with
python predict.py --text_path test.txt --output_dir output/ --model_path models/classifier_model/finbert-sentiment

"""

import sys
import os
import json
from datetime import datetime

# Add parent directory to path to import Massive tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Massive.tools.api_utils import load_api_key
from massive import RESTClient
from transformers import AutoTokenizer, AutoModel
import torch

# Step 1: Setup Massive API client
print("Loading API key...")
MASSIVE_API_KEY = load_api_key()
client = RESTClient(api_key=MASSIVE_API_KEY)

# Step 2: Get news articles
print("\nFetching news from Massive API...")
ticker = "AAPL"  # Apple stock
limit = 5  # Get 5 articles

# Get news articles using list_ticker_news method
news_articles = []
print(f"Fetching news for {ticker}...")
try:
    for article in client.list_ticker_news(ticker=ticker, limit=limit):
        news_articles.append(article)
        title = article.title if hasattr(article, 'title') and article.title else 'No title'
        print(f"  [+] {title[:60]}...")
        # Stop when we have enough articles
        if len(news_articles) >= limit:
            break
except Exception as e:
    # Rate limiting or other errors - show what we got
    if "429" in str(e) or "rate limit" in str(e).lower():
        print(f"\n[!] Rate limit reached, but got {len(news_articles)} articles")
    else:
        print(f"\n[!] Error: {e}")
        if len(news_articles) == 0:
            import traceback
            traceback.print_exc()
            sys.exit(1)

print(f"\n[+] Fetched {len(news_articles)} articles")

# Step 3: Display article details
print("\n" + "="*60)
print("ARTICLE DETAILS")
print("="*60)
for i, article in enumerate(news_articles, 1):
    print(f"\nArticle {i}:")
    print(f"  Title: {article.title if hasattr(article, 'title') else 'N/A'}")
    desc = article.description if hasattr(article, 'description') and article.description else 'N/A'
    print(f"  Description: {desc[:100]}...")
    print(f"  Published: {article.published_utc if hasattr(article, 'published_utc') else 'N/A'}")
    print(f"  Tickers: {article.tickers if hasattr(article, 'tickers') else []}")
    print(f"  Author: {article.author if hasattr(article, 'author') else 'N/A'}")

print("\n" + "="*60)
print("[+] News fetching test complete!")
print("="*60)

# Step 4: Save articles to JSON
print("\n" + "="*60)
print("SAVING TO JSON")
print("="*60)

# Create data directory if it doesn't exist
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(data_dir, exist_ok=True)

# Convert articles to dictionaries for JSON serialization
articles_data = []
for article in news_articles:
    article_dict = {
        'title': article.title if hasattr(article, 'title') else None,
        'description': article.description if hasattr(article, 'description') else None,
        'published_utc': article.published_utc if hasattr(article, 'published_utc') else None,
        'tickers': article.tickers if hasattr(article, 'tickers') else [],
        'author': article.author if hasattr(article, 'author') else None,
        'article_url': article.article_url if hasattr(article, 'article_url') else None,
        'id': article.id if hasattr(article, 'id') else None,
    }
    articles_data.append(article_dict)

# Save to JSON file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json_filename = os.path.join(data_dir, f'news_{ticker}_{timestamp}.json')

with open(json_filename, 'w', encoding='utf-8') as f:
    json.dump({
        'ticker': ticker,
        'fetched_at': datetime.now().isoformat(),
        'count': len(articles_data),
        'articles': articles_data
    }, f, indent=2, ensure_ascii=False)

print(f"[+] Saved {len(articles_data)} articles to:")
print(f"    {json_filename}")

# Step 5: Show how to load and extract text for FinBERT
print("\n" + "="*60)
print("LOADING FROM JSON FOR FINBERT")
print("="*60)

# Load the JSON file
with open(json_filename, 'r', encoding='utf-8') as f:
    loaded_data = json.load(f)

# Extract text for FinBERT processing
texts_for_embedding = []
for article in loaded_data['articles']:
    # Combine title and description for embedding
    text = f"{article.get('title', '')} {article.get('description', '')}"
    texts_for_embedding.append(text.strip())

print(f"[+] Loaded {len(texts_for_embedding)} articles from JSON")
print(f"[+] Prepared {len(texts_for_embedding)} texts for FinBERT embedding")
print(f"[+] Example text length: {len(texts_for_embedding[0])} characters")

print("\n" + "="*60)
print("READY FOR FINBERT PROCESSING")
print("="*60)
print("Texts are ready in 'texts_for_embedding' list")
print("Uncomment FinBERT code below to generate embeddings!")
print("="*60)

# Step 6: Load FinBERT model and generate embeddings


print("\n" + "="*60)
print("GENERATING EMBEDDINGS WITH FINBERT")
print("="*60)

# Load FinBERT model
print("\nLoading FinBERT model...")
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print("FinBERT loaded!")

# Generate embeddings
print("\nGenerating embeddings...")
embeddings = []

for i, text in enumerate(texts_for_embedding):
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Use mean pooling of all tokens
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    
    embeddings.append(embedding)
    print(f"  Article {i+1}: Embedding shape {embedding.shape}")

# Save embeddings back to JSON (optional)
# Add embeddings to articles data
for i, article in enumerate(loaded_data['articles']):
    article['embedding'] = embeddings[i].tolist()  # Convert tensor to list

# Save updated JSON with embeddings
json_with_embeddings = json_filename.replace('.json', '_with_embeddings.json')
with open(json_with_embeddings, 'w', encoding='utf-8') as f:
    json.dump(loaded_data, f, indent=2, ensure_ascii=False)
print(f"\n[+] Saved embeddings to: {json_with_embeddings}")

# Display results
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Articles processed: {len(texts_for_embedding)}")
print(f"Embeddings generated: {len(embeddings)}")
print(f"Embedding dimension: {embeddings[0].shape[0]}")
print("\nDone!")
