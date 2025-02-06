from typing import Dict, List, Optional, Tuple
import gym
from gym import spaces
import numpy as np
import pandas as pd
from dataclasses import dataclass
import requests
from enum import Enum

@dataclass
class TradingState:
   price: float
   position: float
   cash: float
   technical_indicators: Dict[str, float]
   market_metrics: Dict[str, float]
   sentiment_scores: Dict[str, float]
   chain_metrics: Dict[str, float]

class Action(Enum):
   HOLD = 0
   BUY = 1
   SELL = 2

class CryptoTradingEnv(gym.Env):
   def __init__(self, 
                initial_balance: float = 100000.0,
                transaction_fee: float = 0.001,
                lookback_window: int = 60,
                position_sizing: List[float] = [0.25, 0.5, 0.75, 1.0]):
       
       self.initial_balance = initial_balance
       self.transaction_fee = transaction_fee
       self.lookback_window = lookback_window
       self.position_sizes = position_sizing
       
       # Action space: action_type (buy/sell/hold) × position_sizing
       self.action_space = spaces.MultiDiscrete([3, len(position_sizing)])
       
       # Observation space
       self.observation_space = spaces.Dict({
           'market_data': spaces.Box(low=-np.inf, high=np.inf, shape=(lookback_window, 10)),
           'account_state': spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
           'sentiment': spaces.Box(low=-1, high=1, shape=(3,)),
           'chain_metrics': spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
       })
       
       self.reset()

   def _get_llm_strategy(self, state: TradingState) -> str:
       prompt = self._create_strategy_prompt(state)
       response = requests.post('http://localhost:11434/api/generate',
           json={
               "model": "mistral",
               "prompt": prompt
           })
       return response.json()['response']

   def _create_strategy_prompt(self, state: TradingState) -> str:
       return f"""Current market state:
Price: {state.price}
Technical indicators: {state.technical_indicators}
Market metrics: {state.market_metrics}
Sentiment: {state.sentiment_scores}
Chain metrics: {state.chain_metrics}

Based on these metrics, suggest a trading action (BUY/SELL/HOLD) and position size (0.25/0.5/0.75/1.0)
Respond with action and size only, e.g. 'BUY 0.5'"""

   def _calculate_reward(self, 
                        action: Tuple[Action, float], 
                        old_value: float, 
                        new_value: float) -> float:
       # Sharpe ratio components
       returns = (new_value - old_value) / old_value
       risk_free_rate = 0.02 / 252  # Daily risk-free rate
       
       # Transaction cost penalty
       action_type, size = action
       if action_type != Action.HOLD:
           transaction_cost = self.transaction_fee * size * old_value
       else:
           transaction_cost = 0
           
       # Volatility penalty
       volatility = np.std(self.price_history[-30:]) / np.mean(self.price_history[-30:])
       vol_penalty = volatility * abs(returns)
       
       # Combined reward
       reward = returns - risk_free_rate - transaction_cost - vol_penalty
       return reward

   def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
       action_type, size_idx = action
       position_size = self.position_sizes[size_idx]
       
       old_portfolio_value = self._get_portfolio_value()
       
       # Execute trade
       if action_type == Action.BUY:
           self._execute_buy(position_size)
       elif action_type == Action.SELL:
           self._execute_sell(position_size)
           
       # Update state
       self._update_state()
       new_portfolio_value = self._get_portfolio_value()
       
       # Calculate reward
       reward = self._calculate_reward(
           (Action(action_type), position_size),
           old_portfolio_value,
           new_portfolio_value
       )
       
       # Check if episode is done
       done = self.current_step >= len(self.price_history) - 1 or \
              new_portfolio_value <= self.initial_balance * 0.5
              
       return self._get_observation(), reward, done, {}

   def _execute_buy(self, size: float):
       available_cash = self.cash * size
       shares = (available_cash * (1 - self.transaction_fee)) / self.current_price
       self.position += shares
       self.cash -= shares * self.current_price * (1 + self.transaction_fee)

   def _execute_sell(self, size: float):
       shares = self.position * size
       self.position -= shares
       self.cash += shares * self.current_price * (1 - self.transaction_fee)

   def _get_observation(self) -> Dict:
       return {
           'market_data': self.market_data_buffer,
           'account_state': np.array([
               self.cash,
               self.position,
               self._get_portfolio_value()
           ]),
           'sentiment': np.array([
               self.sentiment_scores['social'],
               self.sentiment_scores['news'],
               self.sentiment_scores['chain']
           ]),
           'chain_metrics': np.array([
               self.chain_metrics['whale_movements'],
               self.chain_metrics['exchange_flows'],
               self.chain_metrics['network_volume'],
               self.chain_metrics['gas_fees']
           ])
       }

   def _get_portfolio_value(self) -> float:
       return self.cash + self.position * self.current_price

   def reset(self) -> Dict:
       self.cash = self.initial_balance
       self.position = 0
       self.current_step = 0
       
       # Initialize buffers
       self.market_data_buffer = np.zeros((self.lookback_window, 10))
       self.price_history = []
       self.sentiment_scores = {'social': 0, 'news': 0, 'chain': 0}
       self.chain_metrics = {
           'whale_movements': 0,
           'exchange_flows': 0,
           'network_volume': 0,
           'gas_fees': 0
       }
       
       self._update_state()
       return self._get_observation()

   def _update_state(self):
       # Update market data, sentiment scores, and chain metrics
       # This would be implemented to fetch real data in production
       pass