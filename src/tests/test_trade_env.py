env = CryptoTradingEnv(
    initial_balance=100000.0,
    transaction_fee=0.001,
    lookback_window=60,
    position_sizing=[0.25, 0.5, 0.75, 1.0]
)

obs = env.reset()
action = env.action_space.sample()  # Replace with actual agent action
next_obs, reward, done, info = env.step(action)