config = TrainingConfig(
    timeframes=['1m', '5m', '15m', '1h'],
    batch_size=64,
    learning_rate=3e-4,
    num_epochs=1000,
    validation_interval=10,
    num_workers=4,
    checkpoint_dir="./checkpoints"
)

trainer = DistributedTrainer(config)
best_checkpoint = trainer.train(agent, env, train_data)

backtester = Backtester(env, agent, test_data)
results = backtester.run_backtest()

paper_trader = PaperTrader(env, agent, "binance")
trade_result = await paper_trader.execute_paper_trade(action)