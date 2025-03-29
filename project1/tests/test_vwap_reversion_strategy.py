import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
from strategies.vwap_reversion import VWAPReversionStrategy

def generate_mock_data():
    date_range = pd.date_range(start='2025-02-01', periods=100, freq='1min')
    symbols = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT']

    # Generate the data such that all symbols have the same length of data
    data = {
        'timestamp': date_range.repeat(len(symbols)),  # Repeat the dates for each symbol
        'symbol': symbols * len(date_range),  # Repeat symbols for each time step
        'close': [i + 100 for i in range(len(date_range) * len(symbols))],
        'high': [i + 101 for i in range(len(date_range) * len(symbols))],
        'low': [i + 99 for i in range(len(date_range) * len(symbols))],
        'volume': [1000 + i * 10 for i in range(len(date_range) * len(symbols))],
    }

    df = pd.DataFrame(data)
    df.set_index(['timestamp', 'symbol'], inplace=True)

    return df



def test_initialization():
    price_data = generate_mock_data()

    strategy = VWAPReversionStrategy(price_data, threshold=0.01)

    assert strategy.threshold == 0.01
    assert strategy.price_data.equals(price_data)


# Тест генерації сигналів
def test_generate_signals():
    price_data = generate_mock_data()

    strategy = VWAPReversionStrategy(price_data, threshold=0.01)
    signals = strategy.generate_signals()

    assert signals is not None
    assert not signals.empty


# Тест запуску бектесту
def test_run_backtest():
    price_data = generate_mock_data()

    strategy = VWAPReversionStrategy(price_data, threshold=0.01)

    portfolio_value = strategy.run_backtest()

    assert portfolio_value is not None
    assert len(portfolio_value) > 0


# Тест отримання метрик
def test_get_metrics():
    price_data = generate_mock_data()

    strategy = VWAPReversionStrategy(price_data, threshold=0.01)

    strategy.run_backtest()
    metrics = strategy.get_metrics()

    assert "Total Return" in metrics
    assert "Sharpe Ratio" in metrics
    assert "Max Drawdown" in metrics
    assert "Win Rate" in metrics
    assert "Expectancy" in metrics

    assert isinstance(metrics["Total Return"], (float, int))
    assert isinstance(metrics["Sharpe Ratio"], (float, int))
    assert isinstance(metrics["Max Drawdown"], (float, int))
    assert isinstance(metrics["Win Rate"], (float, int))
    assert isinstance(metrics["Expectancy"], (float, int))


if __name__ == "__main__":
    pytest.main()
