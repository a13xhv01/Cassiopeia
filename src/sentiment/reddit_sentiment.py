import tweepy
import praw
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

class RedditSentiment:
   def __init__(self, client_id, client_secret):
       self.reddit = praw.Reddit(
           client_id=client_id,
           client_secret=client_secret,
           user_agent="crypto_sentiment_bot"
       )
       
   async def get_sentiment(self, coin_symbol):
       subreddit = self.reddit.subreddit(f"{coin_symbol}")
       posts = subreddit.hot(limit=50)
       
       with ThreadPoolExecutor() as executor:
           sentiments = list(executor.map(self._analyze_post, posts))
           
       return self._calculate_metrics(sentiments)
   
   def _analyze_post(self, post):
       text = f"{post.title} {post.selftext}"
       response = requests.post('http://localhost:11434/api/generate',
           json={
               "model": "mistral", 
               "prompt": f"Analyze sentiment (POSITIVE/NEGATIVE/NEUTRAL): {text}"
           })
       return {
           'sentiment': response.json()['response'].strip(),
           'score': post.score,
           'comments': post.num_comments
       }
   
   def _calculate_metrics(self, sentiments):
       df = pd.DataFrame(sentiments)
       sentiment_scores = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
       df['numeric_score'] = df['sentiment'].map(sentiment_scores)
       
       return {
           'sentiment_score': df['numeric_score'].mean(),
           'sentiment_counts': df['sentiment'].value_counts().to_dict(),
           'avg_post_score': df['score'].mean(),
           'avg_comments': df['comments'].mean(),
           'total_posts': len(df)
       }