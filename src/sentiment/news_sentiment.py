import aiohttp
from typing import Dict, List
import json

class NewsSentiment:
   def __init__(self, tavily_api_key: str):
       self.api_key = tavily_api_key
       self.tavily_url = "https://api.tavily.com/search"
       self.ollama_url = "http://localhost:11434/api/generate"

   async def get_sentiment(self, coin_symbol: str, max_results: int = 10) -> Dict:
       news = await self._fetch_news(coin_symbol, max_results)
       sentiments = await self._analyze_sentiments(news)
       return self._calculate_metrics(sentiments)

   async def _fetch_news(self, coin_symbol: str, max_results: int) -> List[Dict]:
       params = {
           "api_key": self.api_key,
           "query": f"{coin_symbol} cryptocurrency news",
           "search_depth": "moderate",
           "max_results": max_results,
           "include_domains": ["coindesk.com", "cointelegraph.com", "cryptonews.com"],
           "exclude_domains": ["youtube.com", "twitter.com", "reddit.com"]
       }

       async with aiohttp.ClientSession() as session:
           async with session.get(self.tavily_url, params=params) as response:
               data = await response.json()
               return data.get('results', [])

   async def _analyze_sentiments(self, news_articles: List[Dict]) -> List[Dict]:
       sentiments = []
       async with aiohttp.ClientSession() as session:
           for article in news_articles:
               text = f"{article['title']} {article.get('description', '')}"
               sentiment = await self._get_sentiment(session, text)
               sentiments.append({
                   'title': article['title'],
                   'url': article['url'],
                   'publish_date': article.get('publish_date'),
                   'sentiment': sentiment
               })
       return sentiments

   async def _get_sentiment(self, session: aiohttp.ClientSession, text: str) -> str:
       prompt = f"Analyze the sentiment of this crypto news (POSITIVE/NEGATIVE/NEUTRAL): {text}"
       payload = {"model": "mistral", "prompt": prompt}
       
       async with session.post(self.ollama_url, json=payload) as response:
           result = await response.json()
           return result['response'].strip()

   def _calculate_metrics(self, sentiments: List[Dict]) -> Dict:
       df = pd.DataFrame(sentiments)
       sentiment_scores = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
       df['score'] = df['sentiment'].map(sentiment_scores)

       return {
           'sentiment_score': df['score'].mean(),
           'sentiment_counts': df['sentiment'].value_counts().to_dict(),
           'total_articles': len(df),
           'articles': df.to_dict('records')
       }