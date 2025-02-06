import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def get_sentiment(text):
    response = requests.post('http://localhost:11434/api/generate', 
        json={
            "model": "mistral",
            "prompt": f"Analyze the sentiment of this text and respond with only POSITIVE, NEGATIVE, or NEUTRAL: {text}"
        })
    
    sentiment = response.json()['response'].strip()
    return sentiment

def analyze_crypto_sentiment(tweets):
    # Parallel processing for faster analysis
    with ThreadPoolExecutor(max_workers=4) as executor:
        sentiments = list(executor.map(get_sentiment, tweets))
    
    # Calculate metrics
    sentiment_df = pd.DataFrame({'sentiment': sentiments})
    sentiment_scores = {
        'POSITIVE': 1,
        'NEUTRAL': 0, 
        'NEGATIVE': -1
    }
    
    sentiment_df['score'] = sentiment_df['sentiment'].map(sentiment_scores)
    return sentiment_df['sentiment'].value_counts(), sentiment_df['score'].mean()