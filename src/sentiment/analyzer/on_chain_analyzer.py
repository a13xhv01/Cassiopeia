from dataclasses import dataclass
from typing import Dict, List, Optional
import aiohttp
from web3 import Web3
import pandas as pd
from datetime import datetime, timedelta

@dataclass
class ChainMetrics:
   whale_movements: Dict[str, float]
   exchange_flows: Dict[str, float]
   network_metrics: Dict[str, float]
   timestamp: datetime

class OnChainAnalyzer:
   def __init__(self, web3_url: str, etherscan_api_key: str):
       self.w3 = Web3(Web3.HTTPProvider(web3_url))
       self.etherscan_key = etherscan_api_key
       self.whale_threshold = 100  # Minimum ETH for whale status
       self.known_exchanges = self._load_exchange_addresses()

   async def get_metrics(self, coin_symbol: str) -> ChainMetrics:
       whale_data = await self._analyze_whale_movements()
       exchange_data = await self._analyze_exchange_flows()
       network_data = await self._get_network_metrics()
       
       return ChainMetrics(
           whale_movements=whale_data,
           exchange_flows=exchange_data,
           network_metrics=network_data,
           timestamp=datetime.utcnow()
       )

   async def _analyze_whale_movements(self) -> Dict[str, float]:
       async with aiohttp.ClientSession() as session:
           url = f"https://api.etherscan.io/api"
           params = {
               'module': 'account',
               'action': 'tokentx',
               'apikey': self.etherscan_key,
               'startblock': self._get_24h_old_block(),
           }
           
           async with session.get(url, params=params) as response:
               transactions = await response.json()
               return self._process_whale_transactions(transactions['result'])

   def _process_whale_transactions(self, txs: List[Dict]) -> Dict[str, float]:
       df = pd.DataFrame(txs)
       whale_txs = df[df['value'].astype(float) > self.whale_threshold * 10**18]
       
       return {
           'total_whale_volume': whale_txs['value'].astype(float).sum() / 10**18,
           'unique_whales': len(whale_txs['from'].unique()),
           'avg_transaction_size': whale_txs['value'].astype(float).mean() / 10**18
       }

   async def _analyze_exchange_flows(self) -> Dict[str, float]:
       blocks = await self._get_recent_blocks(24)
       inflows = outflows = 0
       
       for block in blocks:
           transactions = await self._get_block_transactions(block)
           for tx in transactions:
               if tx['to'] in self.known_exchanges:
                   inflows += float(tx['value'])
               elif tx['from'] in self.known_exchanges:
                   outflows += float(tx['value'])
                   
       return {
           'exchange_inflows': inflows / 10**18,
           'exchange_outflows': outflows / 10**18,
           'net_flow': (inflows - outflows) / 10**18
       }

   async def _get_network_metrics(self) -> Dict[str, float]:
       latest_block = await self.w3.eth.get_block('latest')
       
       return {
           'gas_price': await self.w3.eth.gas_price / 10**9,
           'transaction_volume': latest_block.transactions.length,
           'difficulty': latest_block.difficulty,
           'block_time': self._calculate_block_time(latest_block)
       }

   def _load_exchange_addresses(self) -> List[str]:
       return [
           '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',  # Binance
           '0x2faf487a4414fe77e2327f0bf4ae2a264a776ad2',  # FTX
           '0x13f4EA83D0bd40E75C8222255bc855a974568Dd4',  # Kraken
       ]

   def _get_24h_old_block(self) -> int:
       current_block = self.w3.eth.block_number
       blocks_per_day = 6500  # Approximate
       return current_block - blocks_per_day

   async def _get_recent_blocks(self, hours: int) -> List[int]:
       current_block = self.w3.eth.block_number
       blocks_needed = int(hours * 6500 / 24)
       return range(current_block - blocks_needed, current_block)

   def _calculate_block_time(self, latest_block) -> float:
       previous_block = self.w3.eth.get_block(latest_block.number - 1)
       return latest_block.timestamp - previous_block.timestamp