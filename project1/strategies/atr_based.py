import pandas as pd
import vectorbt as vbt
from .base import StrategyBase

class ATRBreakoutStrategy(StrategyBase):
    """
    Стратегія прориву ATR (ATR Breakout Strategy).

    Використовує індикатор ATR для визначення рівня прориву ціни та виставлення стопів.
    """

    def __init__(self, price_data: pd.DataFrame, atr_period=14, atr_mult=2.5, lookback=20, holding_period=30, sma_period=50):
        """
        Ініціалізує параметри стратегії.

        :param price_data: Цінові дані у форматі DataFrame (очікуються стовпці 'close', 'high', 'low').
        :param atr_period: Період ATR для розрахунку волатильності.
        :param atr_mult: Множник ATR для визначення рівня стоп-лосу.
        :param lookback: Період для визначення рівня прориву.
        :param holding_period: Максимальна тривалість утримання позиції.
        :param sma_period: Період для обчислення SMA.
        """
        super().__init__(price_data)
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.lookback = lookback
        self.holding_period = holding_period
        self.sma_period = sma_period
        self.signals = None
        self.portfolio = None

    def generate_signals(self):
        """
        Генерує торгові сигнали на основі прориву ATR.

        :return: Серія сигналів входу (+1) та виходу (-1).
        """
        close = self.price_data['close'].unstack()
        high = self.price_data['high'].unstack()
        low = self.price_data['low'].unstack()

        sma = close.rolling(window=self.sma_period).mean()

        atr = vbt.ATR.run(high, low, close, window=self.atr_period).atr.reindex_like(close)

        breakout_level = close.shift(1).rolling(self.lookback).max()

        entries = (close > breakout_level).fillna(False)

        trailing_stop = sma - atr * self.atr_mult

        exits_by_stop = (close < trailing_stop).fillna(False)
        exits_by_time = entries.shift(self.holding_period).fillna(False).astype(bool)
        exits = exits_by_stop | exits_by_time

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
            init_cash=10,# По 10 одиниць для кожного активу, сумарний 1000
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
            val = stats.get(key, pd.NA)
            if isinstance(val, pd.Series):
                return val.mean(skipna=True)
            return val if pd.notna(val) else 0.0

        return {
            "Total Return": safe_get("Total Return [%]"),
            "Sharpe Ratio": safe_get("Sharpe Ratio"),
            "Max Drawdown": safe_get("Max Drawdown [%]"),
            "Win Rate": safe_get("Win Rate [%]"),
            "Expectancy": safe_get("Expectancy"),
        }
