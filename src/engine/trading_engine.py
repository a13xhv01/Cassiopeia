from dataclasses import dataclass
from typing import Dict, List, Optional
import asyncio
import aiohttp
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import torch
from prometheus_client import start_http_server, Counter, Gauge, Histogram
import logging
from datetime import datetime, timedelta
import redis
from fastapi import FastAPI, BackgroundTasks
import ray

@dataclass
class ExchangeConfig:
   exchange_id: str
   api_key: str
   api_secret: str
   symbols: List[str]
   timeframes: List[str]

@dataclass
class RiskLimits:
   max_position_size: float
   max_daily_drawdown: float
   max_trade_amount: float
   circuit_breaker_threshold: float

class ExchangeManager:
   def __init__(self, config: ExchangeConfig):
       self.exchange = getattr(ccxt, config.exchange_id)({
           'apiKey': config.api_key,
           'secret': config.api_secret,
           'enableRateLimit': True
       })
       self.symbols = config.symbols
       self.timeframes = config.timeframes
       
   async def fetch_ohlcv(self, symbol: str, timeframe: str) -> pd.DataFrame:
       try:
           ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe)
           df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
           df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
           return df
       except Exception as e:
           logging.error(f"Error fetching OHLCV data: {e}")
           raise

   async def place_order(self, symbol: str, side: str, amount: float, price: float) -> Dict:
       try:
           order = await self.exchange.create_order(
               symbol=symbol,
               type='limit',
               side=side,
               amount=amount,
               price=price
           )
           return order
       except Exception as e:
           logging.error(f"Error placing order: {e}")
           raise

class RiskManager:
   def __init__(self, limits: RiskLimits):
       self.limits = limits
       self.metrics = {
           'position_size': Gauge('position_size', 'Current position size'),
           'daily_pnl': Gauge('daily_pnl', 'Daily PnL'),
           'drawdown': Gauge('drawdown', 'Current drawdown')
       }
       
   def check_risk_limits(self, position_size: float, pnl: float) -> bool:
       if position_size > self.limits.max_position_size:
           logging.warning(f"Position size {position_size} exceeds limit")
           return False
           
       if pnl < -self.limits.max_daily_drawdown:
           logging.warning(f"Daily drawdown {pnl} exceeds limit")
           return False
           
       return True
       
   def update_metrics(self, position_size: float, pnl: float):
       self.metrics['position_size'].set(position_size)
       self.metrics['daily_pnl'].set(pnl)

class ModelServer:
   def __init__(self, model_path: str):
       self.model = torch.load(model_path)
       self.model.eval()
       self.redis_client = redis.Redis(host='localhost', port=6379)
       
   async def predict(self, state: Dict) -> np.ndarray:
       with torch.no_grad():
           state_tensor = torch.FloatTensor(state)
           action = self.model(state_tensor).numpy()
           return action
           
   def cache_prediction(self, state_hash: str, prediction: np.ndarray):
       self.redis_client.setex(
           f"pred:{state_hash}",
           timedelta(minutes=5),
           prediction.tobytes()
       )

class TradingEngine:
   def __init__(self,
                exchange: ExchangeManager,
                model_server: ModelServer,
                risk_manager: RiskManager):
       self.exchange = exchange
       self.model_server = model_server
       self.risk_manager = risk_manager
       self.metrics = {
           'trades': Counter('trades_total', 'Total number of trades'),
           'latency': Histogram('prediction_latency', 'Model prediction latency')
       }
       
   async def execute_trade(self, symbol: str, action: np.ndarray):
       position_size = action[1]
       
       if not self.risk_manager.check_risk_limits(position_size, self._calculate_pnl()):
           return
           
       current_price = await self.exchange.fetch_ticker(symbol)['last']
       
       if action[0] > 0:  # Buy
           order = await self.exchange.place_order(
               symbol=symbol,
               side='buy',
               amount=position_size,
               price=current_price
           )
       else:  # Sell
           order = await self.exchange.place_order(
               symbol=symbol,
               side='sell',
               amount=position_size,
               price=current_price
           )
           
       self.metrics['trades'].inc()
       return order

class DataProcessor:
   def __init__(self):
       self.features = {}
       
   async def process_market_data(self, ohlcv: pd.DataFrame) -> Dict:
       # Calculate technical indicators
       features = {
           'rsi': self._calculate_rsi(ohlcv['close']),
           'macd': self._calculate_macd(ohlcv['close']),
           'bb': self._calculate_bollinger_bands(ohlcv['close'])
       }
       return features
       
   def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
       delta = close.diff()
       gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
       loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
       rs = gain / loss
       return 100 - (100 / (1 + rs))

app = FastAPI()

@app.post("/predict")
async def predict(state: Dict, background_tasks: BackgroundTasks):
   model_server = ModelServer("model.pt")
   prediction = await model_server.predict(state)
   background_tasks.add_task(model_server.cache_prediction, str(hash(str(state))), prediction)
   return {"prediction": prediction.tolist()}

@app.get("/metrics")
async def metrics():
   return PrometheusMetrics.generate_latest()

def main():
   config = ExchangeConfig(
       exchange_id="binance",
       api_key="YOUR_API_KEY",
       api_secret="YOUR_API_SECRET",
       symbols=["BTC/USDT", "ETH/USDT"],
       timeframes=["1m", "5m", "15m", "1h"]
   )
   
   risk_limits = RiskLimits(
       max_position_size=1.0,
       max_daily_drawdown=0.1,
       max_trade_amount=100000,
       circuit_breaker_threshold=0.2
   )
   
   exchange = ExchangeManager(config)
   model_server = ModelServer("model.pt")
   risk_manager = RiskManager(risk_limits)
   engine = TradingEngine(exchange, model_server, risk_manager)
   
   start_http_server(8000)
   
   asyncio.run(app.run(host="0.0.0.0", port=8080))
