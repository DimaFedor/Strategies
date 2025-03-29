import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
from strategies.sma_cross import SMACrossoverStrategy
import os
def get_file_path(file_name):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, '..', '..', 'project1', 'data', file_name)


def test_initialization():
    file_path = get_file_path('btc_1m_feb25.parquet')
    price_data = pd.read_parquet(file_path)

    strategy = SMACrossoverStrategy(price_data, fast_window=10, slow_window=30)

    assert strategy.fast_window == 10
    assert strategy.slow_window == 30
    assert strategy.price_data.equals(price_data)


# Тест генерації сигналів
def test_generate_signals():
    file_path = get_file_path('btc_1m_feb25.parquet')
    df = pd.read_parquet(file_path)

    print("Loaded DataFrame:")
    print(df.head())


    if 'close' not in df.columns or 'high' not in df.columns or 'low' not in df.columns:
        raise ValueError("Цінові дані відсутні або неправильно вказано стовпці")


    strategy = SMACrossoverStrategy(df, fast_window=10, slow_window=30)
    signals = strategy.generate_signals()

    print("Generated Signals DataFrame:")
    print(signals)

    # Перевірки
    assert signals is not None
    assert not signals.empty



def test_run_backtest():
    file_path = get_file_path('btc_1m_feb25.parquet')
    price_data = pd.read_parquet(file_path)

    strategy = SMACrossoverStrategy(price_data, fast_window=10, slow_window=30)

    portfolio_value = strategy.run_backtest()

    assert portfolio_value is not None
    assert len(portfolio_value) > 0


def test_get_metrics():
    file_path = get_file_path('btc_1m_feb25.parquet')
    price_data = pd.read_parquet(file_path)

    strategy = SMACrossoverStrategy(price_data, fast_window=10, slow_window=30)

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
