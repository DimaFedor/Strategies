import os
import pandas as pd
from strategies.vwap_reversion import VWAPReversionStrategy
from strategies.sma_cross import SMACrossoverStrategy
from strategies.atr_based import ATRBreakoutStrategy
from core.backtester import Backtester
from core.data_loader import get_data

def preprocess_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Передобробка цінових даних для подальшого використання у стратегіях.

    :param df: DataFrame з ціновими даними.
    :return: Відформатований DataFrame з індексом ['timestamp', 'symbol'].
    """
    required = {'timestamp', 'symbol', 'close'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Відсутні необхідні колонки: {missing}")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.groupby(['timestamp', 'symbol'], as_index=False).agg({
        'open': 'last',
        'high': 'last',
        'low': 'last',
        'close': 'last',
        'volume': 'last'
    })

    df.set_index(['timestamp', 'symbol'], inplace=True)
    df.sort_index(inplace=True)
    return df

def run_all_strategies(price_data: pd.DataFrame, exclude_symbols=None) -> pd.DataFrame:
    """
    Запускає кілька торгових стратегій та обчислює їх метрики.

    :param price_data: DataFrame з підготовленими ціновими даними.
    :param exclude_symbols: Список символів, які потрібно виключити з аналізу.
    :return: DataFrame зі зведеними метриками всіх стратегій.
    """
    # Використовуємо default список виключених символів, якщо не передано.
    exclude_symbols = exclude_symbols if exclude_symbols else []

    strategies = [
        ("atr_breakout", ATRBreakoutStrategy, {
            "atr_period": 14,
            "atr_mult": 4.5,
            "lookback": 80,
            "holding_period": 60
        }),
        ("vwap_reversion", VWAPReversionStrategy, {"threshold": 0.001, "exclude_symbols": exclude_symbols}),
        ("sma_cross", SMACrossoverStrategy, {"fast_window": 120, "slow_window": 450}),
    ]

    all_metrics = {}

    for name, strategy_cls, kwargs in strategies:
        backtester = Backtester(strategy_cls, price_data, name, **kwargs)
        metrics_df = backtester.run()
        all_metrics[name] = metrics_df

    return pd.concat(all_metrics, axis=1)

def main():
    """
    Основна функція для завантаження даних, запуску стратегій та збереження результатів.
    """
    data_path = "data/btc_1m_feb25.parquet"
    df_raw = get_data(
        start_date="2025-02-01",
        end_date="2025-02-28",
        filepath=data_path,
        top_n=100,
        force_download=False
    )

    price_data = preprocess_price_data(df_raw.reset_index())

    exclude_symbols = ["RVNBTC", "ONEBTC", "ZILBTC"]

    comparison = run_all_strategies(price_data, exclude_symbols=exclude_symbols)

    os.makedirs("results", exist_ok=True)
    comparison.to_csv("results/metrics_comparison.csv", index=False)

    print("Зведена таблиця метрик збережена: results/metrics_comparison.csv\n")
    print("Зведені метрики по стратегіях:")
    print(comparison)

if __name__ == "__main__":
    main()
