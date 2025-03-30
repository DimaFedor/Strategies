import pandas as pd
import vectorbt as vbt
from .base import StrategyBase

class SMACrossoverStrategy(StrategyBase):
    """
    Стратегія перетину ковзних середніх (SMA Crossover Strategy).

    Використовує дві ковзні середні (швидку та повільну) для генерації торгових сигналів.
    """
    def __init__(self, price_data: pd.DataFrame, fast_window=10, slow_window=50):
        """
        Ініціалізує параметри стратегії.

        :param price_data: Цінові дані у форматі DataFrame (очікується стовпець 'close').
        :param fast_window: Період швидкої ковзної середньої.
        :param slow_window: Період повільної ковзної середньої.
        """
        super().__init__(price_data)
        if fast_window >= slow_window:
            raise ValueError("fast_window має бути менше за slow_window")
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.signals = None
        self.portfolio = None

    def generate_signals(self) -> pd.DataFrame:
        """
        Генерує торгові сигнали на основі перетину ковзних середніх.

        :return: Серія сигналів входу (+1) та виходу (-1).
        """
        close = self.price_data['close'].unstack()

        valid_symbols = close.columns[close.notna().sum() >= self.slow_window]
        close = close[valid_symbols]

        fast_ma = vbt.MA.run(close, window=self.fast_window).ma
        slow_ma = vbt.MA.run(close, window=self.slow_window).ma

        if isinstance(fast_ma.columns, pd.MultiIndex):
            fast_ma.columns = fast_ma.columns.get_level_values(-1)
        if isinstance(slow_ma.columns, pd.MultiIndex):
            slow_ma.columns = slow_ma.columns.get_level_values(-1)

        common_cols = fast_ma.columns.intersection(slow_ma.columns)
        fast_ma = fast_ma[common_cols]
        slow_ma = slow_ma[common_cols]
        self.filtered_close = close[common_cols]

        entries = fast_ma > slow_ma
        exits = fast_ma < slow_ma

        self.signals = (entries, exits)
        return entries.astype(int) - exits.astype(int)

    def run_backtest(self):
        """
        Виконує бектест стратегії.

        :return: Серія значень портфеля в часі.
        """
        if self.signals is None:
            self.generate_signals()
        # ЯКЩО ПОТРІБНО ЗРОБИТИ СТАТИЧНИЙ init_cash просто розкоментуйте рядки
        # total_cash = 100  # Загальна сума
        # symbols = self.filtered_close.columns  # Всі активи
        # cash_per_asset = total_cash / len(symbols)  # Розподіл на кожен акт
        entries, exits = self.signals

        self.portfolio = vbt.Portfolio.from_signals(
            close=self.filtered_close,
            entries=entries,
            exits=exits,
            # init_cash=pd.Series(cash_per_asset, index=symbols),
            init_cash=10,  # По 10 одиниць для кожного активу, сумарний 1000
            fees=0.001,
            slippage=0.0011,
            freq="1min"
        )

        return self.portfolio.value()

    def get_metrics(self) -> dict:
        """
        Обчислює основні метрики стратегії.

        :return: Словник із метриками стратегії.
        """
        if self.portfolio is None:
            self.run_backtest()

        stats = self.portfolio.stats()

        return {
            "Total Return": stats.get("Total Return [%]", pd.NA).mean(),
            "Sharpe Ratio": stats.get("Sharpe Ratio", pd.NA).mean(),
            "Max Drawdown": stats.get("Max Drawdown [%]", pd.NA).mean(),
            "Win Rate": stats.get("Win Rate [%]", pd.NA).mean(),
            "Expectancy": stats.get("Expectancy", pd.NA).mean(),
        }
