import os
import hashlib
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zipfile
import io
import datetime
from tqdm import tqdm

def compute_checksum(df: pd.DataFrame) -> str:
    """
    Обчислює контрольну суму (MD5) для перевірки цілісності даних.
    """
    csv_data = df.to_csv(index=True)
    return hashlib.md5(csv_data.encode('utf-8')).hexdigest()

def save_data(df: pd.DataFrame, filepath: str):
    """
    Зберігає DataFrame у форматі Parquet разом із контрольної сумою.
    """
    table = pa.Table.from_pandas(df)
    pq.write_table(table, filepath, compression='snappy')
    checksum = compute_checksum(df)
    with open(filepath + ".md5", "w") as f:
        f.write(checksum)

def load_data(filepath: str) -> pd.DataFrame:
    """
    Завантажує дані з файлу Parquet і перевіряє їхню цілісність за допомогою контрольної суми.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_parquet(filepath)
    md5_path = filepath + ".md5"
    if not os.path.exists(md5_path):
        raise FileNotFoundError(f"Checksum not found: {md5_path}")
    with open(md5_path, "r") as f:
        saved_checksum = f.read().strip()
    current_checksum = compute_checksum(df)
    if current_checksum != saved_checksum:
        raise ValueError("Checksum mismatch.")
    return df

def get_top_btc_pairs(top_n: int = 2) -> list:
    """
    Отримує топові торговані пари BTC на Binance за обсягом.
    """
    url_info = "https://api.binance.com/api/v3/exchangeInfo"
    info = requests.get(url_info).json()
    symbols_info = info.get("symbols", [])
    btc_pairs = [
        s["symbol"] for s in symbols_info
        if s["status"] == "TRADING" and ("BTC" in (s["baseAsset"], s["quoteAsset"]))
    ]
    url_24h = "https://api.binance.com/api/v3/ticker/24hr"
    tickers = requests.get(url_24h).json()
    volume_sorted = sorted(
        ((t["symbol"], float(t["volume"])) for t in tickers if t["symbol"] in btc_pairs),
        key=lambda x: x[1],
        reverse=True
    )
    return [sym for sym, _ in volume_sorted[:top_n]]

def fetch_binance_1m_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Завантажує 1-хвилинні дані по символу з Binance за вибраний період.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    all_data = []
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        url = f"https://data.binance.vision/data/spot/daily/klines/{symbol}/1m/{symbol}-1m-{date_str}.zip"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f, header=None)
                    df.columns = [
                        "open_time", "open", "high", "low", "close", "volume",
                        "close_time", "quote_asset_volume", "number_of_trades",
                        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
                    ]
                    df["symbol"] = symbol
                    all_data.append(df)
        except:
            pass
        current += datetime.timedelta(days=1)
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


def download_data(start_date: str, end_date: str, top_n: int = 100) -> pd.DataFrame:
    """
    Завантажує дані для топових BTC-пар за заданий період.
    """
    symbols = get_top_btc_pairs(top_n)
    all_frames = []
    for sym in tqdm(symbols, desc="Downloading"):
        df = fetch_binance_1m_data(sym, start_date, end_date)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    df_all = pd.concat(all_frames, ignore_index=True)

    # Перевірка значень open_time
    print(f"Sample open_time values:\n{df_all['open_time'].head()}")
    print(f"Min open_time: {df_all['open_time'].min()}")
    print(f"Max open_time: {df_all['open_time'].max()}")

    # Для мікросекунд поділяємо на 1,000,000
    if df_all['open_time'].max() > 1e12:  # Якщо max open_time в мікросекундах (1e12 — це понад мільярд мікросекунд)
        df_all["timestamp"] = pd.to_datetime(df_all["open_time"] / 1e6, unit="s")  # Перетворюємо в секунди
    else:
        df_all["timestamp"] = pd.to_datetime(df_all["open_time"], unit="ms")  # Якщо все ж у мілісекундах

    # Виведення для перевірки
    print(f"Full sample of open_time and corresponding timestamp:\n{df_all[['open_time', 'timestamp']].head(20)}")

    # Фільтруємо необхідні стовпці та встановлюємо індекс
    df_all = df_all[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
    df_all.set_index(["timestamp", "symbol"], inplace=True)
    df_all.sort_index(inplace=True)

    return df_all


def get_data(start_date: str, end_date: str, filepath: str = "data/btc_data.parquet", top_n: int = 100,
             force_download: bool = False) -> pd.DataFrame:
    """
    Отримує дані: завантажує локально або завантажує з Binance, якщо необхідно.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not force_download and os.path.exists(filepath):
        try:
            return load_data(filepath)
        except:
            pass
    df = download_data(start_date, end_date, top_n)
    if df.empty:
        return df
    save_data(df, filepath)
    return df
