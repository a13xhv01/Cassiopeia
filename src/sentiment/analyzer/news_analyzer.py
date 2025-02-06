from dataclasses import dataclass
from typing import List, Dict, Optional
import aiohttp
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from datetime import datetime, timedelta
import asyncio
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

@dataclass
class NewsArticle:
   title: str
   content: str
   url: str
   source: str
   published_at: datetime
   entities: Dict[str, List[str]]
   topics: List[str]
   sentiment_score: float

class CryptoNewsAnalyzer:
   def __init__(self, brave_api_key: str, cryptocompare_api_key: str):
       self.brave_api_key = brave_api_key
       self.cryptocompare_api_key = cryptocompare_api_key
       self.chrome_options = self._setup_chrome_options()
       nltk.download(['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'])

   def _setup_chrome_options(self) -> Options:
       options = Options()
       options.add_argument('--headless')
       options.add_argument('--no-sandbox')
       return options

   async def get_news_sentiment(self, coin_symbol: str, days: int = 1) -> Dict:
       articles = await self._fetch_all_news(coin_symbol, days)
       return self._aggregate_sentiments(articles)

   async def _fetch_all_news(self, coin_symbol: str, days: int) -> List[NewsArticle]:
       brave_articles = await self._fetch_brave_news(coin_symbol)
       cc_articles = await self._fetch_cryptocompare_news(coin_symbol)
       
       all_articles = brave_articles + cc_articles
       processed_articles = []

       for article in all_articles:
           content = await self._scrape_content(article['url'])
           processed = await self._process_article(article, content)
           processed_articles.append(processed)

       return processed_articles

   async def _fetch_brave_news(self, coin_symbol: str) -> List[Dict]:
       url = "https://api.search.brave.com/news"
       params = {
           "q": f"{coin_symbol} cryptocurrency",
           "count": 20,
           "token": self.brave_api_key
       }

       async with aiohttp.ClientSession() as session:
           async with session.get(url, params=params) as response:
               data = await response.json()
               return data.get('articles', [])

   async def _fetch_cryptocompare_news(self, coin_symbol: str) -> List[Dict]:
       url = "https://min-api.cryptocompare.com/data/v2/news/"
       params = {
           "api_key": self.cryptocompare_api_key,
           "categories": coin_symbol.lower()
       }

       async with aiohttp.ClientSession() as session:
           async with session.get(url, params=params) as response:
               data = await response.json()
               return data.get('Data', [])

   async def _scrape_content(self, url: str) -> str:
       with webdriver.Chrome(options=self.chrome_options) as driver:
           driver.get(url)
           return driver.find_element_by_tag_name('body').text

   async def _process_article(self, article: Dict, content: str) -> NewsArticle:
       sentiment = await self._analyze_sentiment(content)
       entities = self._extract_entities(content)
       topics = self._extract_topics(content)

       return NewsArticle(
           title=article.get('title', ''),
           content=content,
           url=article.get('url', ''),
           source=article.get('source', ''),
           published_at=self._parse_date(article.get('published', '')),
           entities=entities,
           topics=topics,
           sentiment_score=sentiment
       )

   async def _analyze_sentiment(self, text: str) -> float:
       async with aiohttp.ClientSession() as session:
           async with session.post('http://localhost:11434/api/generate', 
               json={
                   "model": "mistral",
                   "prompt": f"Rate the sentiment of this text on a scale of -1 to 1, return only the number: {text[:500]}"
               }) as response:
               result = await response.json()
               return float(result['response'].strip())

   def _extract_entities(self, text: str) -> Dict[str, List[str]]:
       sentences = sent_tokenize(text)
       entities = {'PERSON': [], 'ORGANIZATION': [], 'GPE': []}
       
       for sentence in sentences:
           chunked = ne_chunk(pos_tag(nltk.word_tokenize(sentence)))
           for chunk in chunked:
               if hasattr(chunk, 'label'):
                   if chunk.label() in entities:
                       entities[chunk.label()].append(' '.join([c[0] for c in chunk]))
       return entities

   def _extract_topics(self, text: str) -> List[str]:
       response = requests.post('http://localhost:11434/api/generate',
           json={
               "model": "mistral",
               "prompt": f"Extract 3-5 main topics from this text, return as comma-separated list: {text[:500]}"
           })
       return [topic.strip() for topic in response.json()['response'].split(',')]

   def _parse_date(self, date_str: str) -> datetime:
       return datetime.fromisoformat(date_str.replace('Z', '+00:00'))

   def _aggregate_sentiments(self, articles: List[NewsArticle]) -> Dict:
       df = pd.DataFrame([{
           'title': a.title,
           'sentiment': a.sentiment_score,
           'source': a.source,
           'published_at': a.published_at,
           'topics': a.topics,
           'entities': a.entities
       } for a in articles])

       return {
           'average_sentiment': df['sentiment'].mean(),
           'sentiment_stddev': df['sentiment'].std(),
           'sentiment_by_source': df.groupby('source')['sentiment'].mean().to_dict(),
           'top_topics': pd.Series([t for topics in df['topics'] for t in topics]).value_counts().head().to_dict(),
           'top_entities': {
               entity_type: pd.Series([e for entities in df['entities'] for e in entities.get(entity_type, [])])
                   .value_counts().head().to_dict()
               for entity_type in ['PERSON', 'ORGANIZATION', 'GPE']
           }
       }