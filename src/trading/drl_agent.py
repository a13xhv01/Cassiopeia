import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Dict
from collections import deque
import random

class StateEncoder(nn.Module):
   def __init__(self, lookback_window: int, feature_dim: int):
       super().__init__()
       
       self.cnn = nn.Sequential(
           nn.Conv1d(feature_dim, 32, kernel_size=3),
           nn.ReLU(),
           nn.Conv1d(32, 64, kernel_size=3),
           nn.ReLU(),
           nn.Conv1d(64, 128, kernel_size=3),
           nn.ReLU(),
           nn.AdaptiveAvgPool1d(1)
       )
       
       self.lstm = nn.LSTM(
           input_size=feature_dim,
           hidden_size=128,
           num_layers=2,
           batch_first=True
       )

   def forward(self, market_data: torch.Tensor) -> torch.Tensor:
       batch_size = market_data.size(0)
       
       # CNN path
       cnn_out = self.cnn(market_data.transpose(1, 2))
       cnn_out = cnn_out.view(batch_size, -1)
       
       # LSTM path
       lstm_out, _ = self.lstm(market_data)
       lstm_out = lstm_out[:, -1, :]
       
       # Combine
       return torch.cat([cnn_out, lstm_out], dim=1)

class SAC(nn.Module):
   def __init__(self, 
                state_dim: int,
                action_dim: int,
                hidden_dim: int = 256):
       super().__init__()
       
       self.actor = Actor(state_dim, action_dim, hidden_dim)
       self.critic1 = Critic(state_dim, action_dim, hidden_dim)
       self.critic2 = Critic(state_dim, action_dim, hidden_dim)
       self.value = Value(state_dim, hidden_dim)
       
       self.risk_manager = RiskManager()

   def act(self, state: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
       action_mean, action_std = self.actor(state)
       
       if evaluate:
           return action_mean
       
       dist = Normal(action_mean, action_std)
       action = dist.sample()
       
       # Apply risk management constraints
       action = self.risk_manager.adjust_action(state, action)
       
       return action

class Actor(nn.Module):
   def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
       super().__init__()
       
       self.net = nn.Sequential(
           nn.Linear(state_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, hidden_dim),
           nn.ReLU()
       )
       
       self.mean = nn.Linear(hidden_dim, action_dim)
       self.log_std = nn.Linear(hidden_dim, action_dim)

   def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
       x = self.net(state)
       mean = self.mean(x)
       log_std = self.log_std(x).clamp(-20, 2)
       return mean, log_std.exp()

class Critic(nn.Module):
   def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
       super().__init__()
       
       self.net = nn.Sequential(
           nn.Linear(state_dim + action_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, 1)
       )

   def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
       x = torch.cat([state, action], dim=1)
       return self.net(x)

class Value(nn.Module):
   def __init__(self, state_dim: int, hidden_dim: int):
       super().__init__()
       
       self.net = nn.Sequential(
           nn.Linear(state_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, hidden_dim),
           nn.ReLU(),
           nn.Linear(hidden_dim, 1)
       )

   def forward(self, state: torch.Tensor) -> torch.Tensor:
       return self.net(state)

class RiskManager:
   def __init__(self,
                max_position: float = 1.0,
                max_leverage: float = 2.0,
                stop_loss: float = 0.1):
       self.max_position = max_position
       self.max_leverage = max_leverage
       self.stop_loss = stop_loss

   def adjust_action(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
       portfolio_value = state[:, -1]  # Assuming last state feature is portfolio value
       
       # Position size constraints
       action = torch.clamp(action, -self.max_position, self.max_position)
       
       # Leverage constraints
       leverage = torch.abs(action) * portfolio_value
       leverage_mask = leverage <= (portfolio_value * self.max_leverage)
       action = action * leverage_mask.float()
       
       # Stop loss
       current_loss = (portfolio_value - state[:, -2]) / state[:, -2]  # Previous portfolio value
       stop_loss_mask = current_loss >= -self.stop_loss
       action = action * stop_loss_mask.float()
       
       return action

class ReplayBuffer:
   def __init__(self, capacity: int):
       self.buffer = deque(maxlen=capacity)

   def push(self, state: np.ndarray, action: np.ndarray, 
            reward: float, next_state: np.ndarray, done: bool):
       self.buffer.append((state, action, reward, next_state, done))

   def sample(self, batch_size: int) -> Tuple:
       batch = random.sample(self.buffer, batch_size)
       state, action, reward, next_state, done = zip(*batch)
       
       return (
           torch.FloatTensor(state),
           torch.FloatTensor(action),
           torch.FloatTensor(reward),
           torch.FloatTensor(next_state),
           torch.FloatTensor(done)
       )

   def __len__(self) -> int:
       return len(self.buffer)