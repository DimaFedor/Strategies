import pandas as pd
import vectorbt as vbt
from .base import StrategyBase

class VWAPReversionStrategy(StrategyBase):
    """
    Стратегія повернення до VWAP (VWAP Reversion Strategy).

    Використовує індикатор VWAP для виявлення точок входу та виходу на основі відхилення ціни від VWAP.
    """
    def __init__(self, price_data: pd.DataFrame, threshold=0.01, exclude_symbols=None):
        """
        Ініціалізує параметри стратегії.

        :param price_data: Цінові дані у форматі DataFrame (очікуються стовпці 'close' та 'volume').
        :param threshold: Граничне відхилення від VWAP для входу в позицію.
        :param exclude_symbols: Список символів (пар), які потрібно виключити з обчислень.
        """
        super().__init__(price_data)
        self.threshold = threshold
        self.exclude_symbols = exclude_symbols if exclude_symbols else []
        self.signals = None
        self.portfolio = None

    def generate_signals(self) -> pd.DataFrame:
        """
        Генерує торгові сигнали на основі відхилення від VWAP.

        :return: Серія сигналів входу (+1) та виходу (-1).
        """
        close = self.price_data['close'].unstack()
        volume = self.price_data['volume'].unstack()

        if self.exclude_symbols:
            close = close.drop(columns=self.exclude_symbols)
            volume = volume.drop(columns=self.exclude_symbols)

        entries = pd.DataFrame(False, index=close.index, columns=close.columns)
        exits = pd.DataFrame(False, index=close.index, columns=close.columns)

        for symbol in close.columns:
            close_sym = close[symbol]
            volume_sym = volume[symbol].replace(0, pd.NA)

            vwap = (close_sym * volume_sym).cumsum() / volume_sym.cumsum()
            vwap = vwap.ffill().bfill().infer_objects(copy=False)

            entry = (close_sym < vwap * (1 - self.threshold)).fillna(False)
            exit = (close_sym >= vwap).fillna(False)

            entries[symbol] = entry
            exits[symbol] = exit

        self.signals = (entries, exits)
        self.filtered_close = close
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

        def safe_get(key):
            value = stats.get(key, pd.NA)
            if isinstance(value, (pd.Series, pd.DataFrame)):
                return value.mean(skipna=True).mean()
            return value if pd.notna(value) else 0.0

        return {
            "Total Return": safe_get("Total Return [%]"),
            "Sharpe Ratio": safe_get("Sharpe Ratio"),
            "Max Drawdown": safe_get("Max Drawdown [%]"),
            "Win Rate": safe_get("Win Rate [%]"),
            "Expectancy": safe_get("Expectancy"),
        }
