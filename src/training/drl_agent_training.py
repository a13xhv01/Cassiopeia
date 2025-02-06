from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import ray
from ray import tune
import ccxt
from datetime import datetime, timedelta

@dataclass
class TrainingConfig:
   timeframes: List[str]
   batch_size: int
   learning_rate: float
   num_epochs: int
   validation_interval: int
   num_workers: int
   checkpoint_dir: str

class DataLoader:
   def __init__(self, exchange_id: str, symbol: str):
       self.exchange = getattr(ccxt, exchange_id)()
       self.symbol = symbol
       
   async def fetch_historical_data(self, 
                                 timeframe: str,
                                 start_date: datetime,
                                 end_date: datetime) -> pd.DataFrame:
       ohlcv = await self.exchange.fetch_ohlcv(
           symbol=self.symbol,
           timeframe=timeframe,
           since=int(start_date.timestamp() * 1000),
           limit=1000
       )
       
       df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
       df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
       return df

class MultiTimeframeDataset(torch.utils.data.Dataset):
   def __init__(self, data: Dict[str, pd.DataFrame], lookback: int):
       self.data = data
       self.lookback = lookback
       self.timeframes = list(data.keys())
       
   def __len__(self) -> int:
       return min(len(df) - self.lookback for df in self.data.values())
       
   def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
       sample = {}
       for timeframe in self.timeframes:
           df = self.data[timeframe]
           window = df.iloc[idx:idx + self.lookback]
           sample[timeframe] = torch.FloatTensor(window.values)
       return sample

class Backtester:
   def __init__(self, env, agent, test_data: pd.DataFrame):
       self.env = env
       self.agent = agent
       self.test_data = test_data
       
   def run_backtest(self) -> Dict:
       state = self.env.reset()
       total_reward = 0
       trades = []
       
       while True:
           action = self.agent.act(torch.FloatTensor(state), evaluate=True)
           next_state, reward, done, info = self.env.step(action.numpy())
           
           trades.append({
               'timestamp': info['timestamp'],
               'action': action.numpy(),
               'reward': reward,
               'portfolio_value': info['portfolio_value']
           })
           
           total_reward += reward
           state = next_state
           
           if done:
               break
               
       return {
           'total_reward': total_reward,
           'trades': pd.DataFrame(trades),
           'sharpe_ratio': self._calculate_sharpe_ratio(trades),
           'max_drawdown': self._calculate_max_drawdown(trades)
       }
       
   def _calculate_sharpe_ratio(self, trades: List[Dict]) -> float:
       returns = pd.DataFrame(trades)['portfolio_value'].pct_change().dropna()
       return np.sqrt(252) * returns.mean() / returns.std()
       
   def _calculate_max_drawdown(self, trades: List[Dict]) -> float:
       portfolio_values = pd.DataFrame(trades)['portfolio_value']
       rolling_max = portfolio_values.expanding().max()
       drawdowns = (portfolio_values - rolling_max) / rolling_max
       return drawdowns.min()

class PaperTrader:
   def __init__(self, env, agent, exchange_id: str):
       self.env = env
       self.agent = agent
       self.exchange = getattr(ccxt, exchange_id)()
       
   async def execute_paper_trade(self, action: np.ndarray):
       current_price = await self.exchange.fetch_ticker(self.env.symbol)['last']
       position_size = action[1] * self.env.initial_balance
       
       return {
           'timestamp': datetime.utcnow(),
           'action': action,
           'price': current_price,
           'position_size': position_size,
           'portfolio_value': self.env._get_portfolio_value()
       }

class DistributedTrainer:
   def __init__(self, config: TrainingConfig):
       self.config = config
       ray.init(num_cpus=config.num_workers)
       
   def train(self, agent, env, train_data: Dict[str, pd.DataFrame]):
       trainer = tune.Trainer(
           backend="torch",
           num_workers=self.config.num_workers,
           checkpoint_freq=self.config.validation_interval,
           checkpoint_at_end=True,
           keep_checkpoints_num=5,
           checkpoint_score_attr="mean_reward",
           stop={"training_iteration": self.config.num_epochs},
           config={
               "env": env,
               "agent": agent,
               "train_data": train_data,
               "batch_size": self.config.batch_size,
               "learning_rate": self.config.learning_rate
           }
       )
       
       analysis = trainer.run()
       best_checkpoint = analysis.get_best_checkpoint(metric="mean_reward")
       return best_checkpoint

def train_worker(config: Dict):
   env = config["env"]
   agent = config["agent"]
   train_data = config["train_data"]
   
   dataset = MultiTimeframeDataset(train_data, env.lookback_window)
   dataloader = torch.utils.data.DataLoader(
       dataset,
       batch_size=config["batch_size"],
       shuffle=True
   )
   
   optimizer = torch.optim.Adam(agent.parameters(), lr=config["learning_rate"])
   
   for epoch in range(config["training_iteration"]):
       epoch_reward = 0
       for batch in dataloader:
           state = env.reset()
           done = False
           
           while not done:
               action = agent.act(torch.FloatTensor(state))
               next_state, reward, done, _ = env.step(action.numpy())
               
               agent.update(optimizer, state, action, reward, next_state, done)
               state = next_state
               epoch_reward += reward
               
       tune.report(mean_reward=epoch_reward)