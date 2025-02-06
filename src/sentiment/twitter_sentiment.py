import tweepy
import praw
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

class TwitterSentiment:
   def __init__(self, bearer_token):
       self.client = tweepy.Client(bearer_token=bearer_token)
       
   async def get_sentiment(self, coin_symbol, hours=24):
       query = f"#{coin_symbol} OR ${coin_symbol} -is:retweet lang:en"
       end_time = datetime.utcnow()
       start_time = end_time - timedelta(hours=hours)
       
       tweets = self.client.search_recent_tweets(
           query=query,
           max_results=100,
           start_time=start_time,
           end_time=end_time
       )
       
       with ThreadPoolExecutor() as executor:
           sentiments = list(executor.map(self._analyze_sentiment, tweets.data))
           
       return self._calculate_metrics(sentiments)
   
   def _analyze_sentiment(self, tweet):
       response = requests.post('http://localhost:11434/api/generate',
           json={
               "model": "mistral",
               "prompt": f"Analyze sentiment (POSITIVE/NEGATIVE/NEUTRAL): {tweet.text}"
           })
       return response.json()['response'].strip()
   
   def _calculate_metrics(self, sentiments):
       df = pd.DataFrame({'sentiment': sentiments})
       sentiment_scores = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
       df['score'] = df['sentiment'].map(sentiment_scores)
       
       return {
           'sentiment_score': df['score'].mean(),
           'sentiment_counts': df['sentiment'].value_counts().to_dict(),
           'total_mentions': len(df)
       }
