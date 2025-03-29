import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Backtester:
    """
    Клас для виконання тестування стратегій на основі наданих даних.
    Включає методи для запуску стратегії, збереження результатів, побудови графіків і метрик.
    Attributes:
        strategy: Об'єкт стратегії для тестування.
        strategy_name (str): Назва стратегії.
        results_dir (str): Директорія для збереження результатів тестування.
        run(self):
            Запускає тестування стратегії, генерує метрики, зберігає графіки та метрики в директорії результатів.
    """

    def __init__(self, strategy_cls, price_data: pd.DataFrame, strategy_name: str, **kwargs):
        """
        Ініціалізує клас Backtester.

        Параметри:
            strategy_cls: Клас стратегії для тестування.
            price_data (pd.DataFrame): Дані про ціни для тестування.
            strategy_name (str): Назва стратегії.
            **kwargs: Додаткові параметри для ініціалізації стратегії.
        """
        self.strategy = strategy_cls(price_data, **kwargs)
        self.strategy_name = strategy_name
        self.results_dir = f"results/{strategy_name}"
        os.makedirs(self.results_dir, exist_ok=True)

    def run(self):
        """
        Запускає тестування стратегії, генерує метрики та зберігає результати.

        Повертає:
            pd.DataFrame: Зведена таблиця з метриками стратегії.
        """
        print(f"[{self.strategy_name}] Running strategy...")

        # Запуск тестування стратегії
        equity_curve = self.strategy.run_backtest()
        metrics = self.strategy.get_metrics()

        # Збереження метрик у файл
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=["value"])
        metrics_df.to_csv(f"{self.results_dir}/metrics.csv")
        print(f"[{self.strategy_name}] Metrics saved.")

        # Побудова графіка кривої капіталу
        total_equity = equity_curve.sum(axis=1)
        plt.figure(figsize=(10, 5))
        total_equity.plot()
        plt.title(f'Equity Curve: {self.strategy_name}')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/equity_curve.png")
        plt.close()

        try:
            total_returns = self.strategy.portfolio.total_return() * 100
            symbols = total_returns.index.values
            values = total_returns.values

            shape = int(np.ceil(np.sqrt(len(values))))
            padded_values = np.pad(values, (0, shape ** 2 - len(values)), mode='constant', constant_values=np.nan)
            padded_symbols = np.pad(symbols, (0, shape ** 2 - len(symbols)), mode='constant', constant_values="")

            matrix_values = padded_values.reshape((shape, shape))
            matrix_symbols = padded_symbols.reshape((shape, shape))

            annotations = np.array([[f"{matrix_symbols[i, j]}\n{matrix_values[i, j]:.1f}"
                                     if matrix_symbols[i, j] else ""
                                     for j in range(shape)] for i in range(shape)])

            plt.figure(figsize=(12, 8))
            sns.heatmap(matrix_values, annot=annotations, fmt="", cmap="viridis", linewidths=0.5, linecolor='black')
            plt.title(f"Heatmap: Total Return by Symbol [{self.strategy_name}]")
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/heatmap.png")
            plt.close()
        except Exception as e:
            print(f"Не вдалося побудувати heatmap: {e}")

        print(f"[{self.strategy_name}] Backtest complete. Results saved in {self.results_dir}")
        return metrics_df.T
