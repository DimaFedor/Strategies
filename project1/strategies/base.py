from abc import ABC, abstractmethod
import pandas as pd

class StrategyBase(ABC):
    def __init__(self, price_data: pd.DataFrame):
        """
        price_data: DataFrame з MultiIndex (timestamp, symbol)
        та колонками: open, high, low, close, volume
        """
        self.price_data = price_data

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """Повертає DataFrame з сигналами: 1 – long, -1 – short, 0 – нейтрально"""
        pass

    @abstractmethod
    def run_backtest(self) -> pd.DataFrame:
        """Виконує бектест і повертає equity curve у вигляді DataFrame"""
        pass

    @abstractmethod
    def get_metrics(self) -> dict:
        """Повертає словник з метриками стратегії"""
        pass
