import tweepy
import praw
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from .twitter_sentiment import TwitterSentiment
from .reddit_sentiment import RedditSentiment
from .news_sentiment import NewsSentiment

class SocialMediaSentiment:
   def __init__(self, twitter_bearer_token, tavily_api_key, reddit_client_id, reddit_client_secret):
       self.twitter = TwitterSentiment(twitter_bearer_token)
       self.reddit = RedditSentiment(reddit_client_id, reddit_client_secret)
       self.news = NewsSentiment(tavily_api_key)

   async def get_combined_sentiment(self, coin_symbol):
       twitter_metrics = await self.twitter.get_sentiment(coin_symbol)
       reddit_metrics = await self.reddit.get_sentiment(coin_symbol)
       news_metrics = await self.news.get_sentiment(coin_symbol)
       
       return {
           'twitter': twitter_metrics,
           'reddit': reddit_metrics,
           'news': news_metrics,
           'combined_score': (twitter_metrics['sentiment_score'] + 
                            reddit_metrics['sentiment_score'] +
                            news_metrics['sentiment_score']) / 3
       }

