import numpy as np
import pandas as pd
import vectorbt as vbt
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(portfolio: vbt.Portfolio) -> dict:
    """
    Обчислює основні метрики ефективності портфеля.
    :param portfolio: Об'єкт портфеля VectorBT
    :return: Словник з метриками
    """
    total_return = portfolio.total_return()
    sharpe_ratio = portfolio.sharpe_ratio()
    max_drawdown = portfolio.max_drawdown()
    winrate = portfolio.win_rate()
    expectancy = portfolio.expected_return()
    exposure_time = portfolio.exposure_time()

    return {
        'Total Return': total_return.values[0],
        'Sharpe Ratio': sharpe_ratio.values[0],
        'Max Drawdown': max_drawdown.values[0],
        'Win Rate': winrate.values[0],
        'Expectancy': expectancy.values[0],
        'Exposure Time': exposure_time.values[0]
    }


def plot_equity_curve(portfolio: vbt.Portfolio, save_path: str):
    """Будує графік equity curve і зберігає його."""
    portfolio.plot().write_image(save_path)


def plot_performance_heatmap(results: pd.DataFrame, save_path: str):
    """Будує heatmap по performance для всіх 100 пар і зберігає."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(results.pivot(index='strategy', columns='pair', values='Total Return'), cmap='coolwarm', annot=True)
    plt.title("Performance Heatmap")
    plt.savefig(save_path)
    plt.close()


def compare_metrics(metrics_dict: dict, save_path: str):
    """Порівнює метрики різних стратегій і зберігає графік."""
    df = pd.DataFrame(metrics_dict).T
    df.plot(kind='bar', figsize=(12, 6))
    plt.title("Comparison of Strategy Metrics")
    plt.ylabel("Value")
    plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(save_path)
    plt.close()
