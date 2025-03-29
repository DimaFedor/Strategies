import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
from strategies.atr_based import ATRBreakoutStrategy

# Тест ініціалізації стратегії
def test_initialization():
    price_data = pd.DataFrame({
        'timestamp': pd.date_range('2025-02-01', periods=10, freq='D'),
        'symbol': ['BTC'] * 10,
        'close': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'high': [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],
        'low': [0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9],
        'volume': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    })

    strategy = ATRBreakoutStrategy(price_data, atr_period=14, atr_mult=2.5, lookback=2, holding_period=3)

    assert strategy.atr_period == 14
    assert strategy.atr_mult == 2.5
    assert strategy.lookback == 2
    assert strategy.holding_period == 3
    assert strategy.price_data.equals(price_data)


# Тест генерації сигналів
def test_generate_signals():
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2025-02-01', '2025-02-01', '2025-02-02', '2025-02-02']),
        'symbol': ['BTCUSDT', 'ETHUSDT', 'BTCUSDT', 'ETHUSDT'],
        'close': [30000, 2000, 31000, 2100],
        'high': [30500, 2050, 31500, 2150],
        'low': [29500, 1950, 30500, 2050],
        'volume': [100, 50, 200, 100]
    })

    df.set_index(['timestamp', 'symbol'], inplace=True)

    strategy = ATRBreakoutStrategy(df)
    signals = strategy.generate_signals()

    assert signals is not None
    assert not signals.empty


# Тест запуску бектесту
def test_run_backtest():
    price_data = pd.DataFrame({
        'timestamp': pd.date_range('2025-02-01', periods=10, freq='D'),
        'symbol': ['BTC'] * 10,
        'close': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'high': [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],
        'low': [0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9],
        'volume': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    })

    price_data.set_index(['timestamp', 'symbol'], inplace=True)

    strategy = ATRBreakoutStrategy(price_data, atr_period=14, atr_mult=2.5, lookback=2, holding_period=3)
    portfolio_value = strategy.run_backtest()

    assert portfolio_value is not None
    assert len(portfolio_value) > 0


# Тест отримання метрик
def test_get_metrics():
    price_data = pd.DataFrame({
        'timestamp': pd.date_range('2025-02-01', periods=10, freq='D'),
        'symbol': ['BTC'] * 10,
        'close': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'high': [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],
        'low': [0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9],
        'volume': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    })

    price_data.set_index(['timestamp', 'symbol'], inplace=True)

    strategy = ATRBreakoutStrategy(price_data, atr_period=14, atr_mult=2.5, lookback=2, holding_period=3)

    strategy.run_backtest()
    metrics = strategy.get_metrics()

    assert "Total Return" in metrics
    assert "Sharpe Ratio" in metrics
    assert "Max Drawdown" in metrics
    assert "Win Rate" in metrics
    assert "Expectancy" in metrics
    assert metrics["Total Return"] is not None


if __name__ == "__main__":
    pytest.main()
