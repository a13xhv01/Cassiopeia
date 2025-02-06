analyzer = OnChainAnalyzer(
    web3_url="YOUR_INFURA_URL",
    etherscan_api_key="YOUR_ETHERSCAN_KEY"
)
metrics = await analyzer.get_metrics("ETH")