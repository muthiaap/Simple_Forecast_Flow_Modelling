import pandas as pd
import numpy as np

# Lag feature buat liat nilai masalalu
def add_lag_features(df: pd.DataFrame, lags=(1, 2, 3), target_col="Close"):
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_t-{lag}"] = df[target_col].shift(lag)
    return df

# Rolling nunjukin mean dan standard deviation
def add_rolling_features(df: pd.DataFrame, windows=(5, 10, 20), target_col="Close"):
    df = df.copy()
    for w in windows:
        df[f"SMA_{w}"] = df[target_col].rolling(window=w, min_periods=w).mean()
        df[f"STD_{w}"] = df[target_col].rolling(window=w, min_periods=w).std()
    return df

# RSI rata2 kenaikan atau penurunan nilai saham.
def add_rsi(df: pd.DataFrame, period: int = 14, target_col="Close"):
    df = df.copy()
    delta = df[target_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    avg_loss = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def add_returns(df: pd.DataFrame, target_col="Close"):
    df = df.copy()
    df["RET_1"] = df[target_col].pct_change(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["LOG_RET_1"] = np.log(df[target_col] / df[target_col].shift(1))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def preprocess_stock_data(df: pd.DataFrame, date_col="Date", target_col="Close"):
    """
    - Pastikan Date & sort kronologis
    - Hapus duplikat tanggal
    - Fitur: lag, SMA/STD, RSI, returns
    - Drop NA awal akibat rolling/lag
    """
    df = df.copy()

    if date_col not in df.columns:
        raise ValueError(f"Kolom '{date_col}' tidak ditemukan. Kolom tersedia: {list(df.columns)}")
    if target_col not in df.columns:
        raise ValueError(f"Kolom '{target_col}' tidak ditemukan. Kolom tersedia: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df = df.drop_duplicates(subset=[date_col], keep="last").reset_index(drop=True)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    df = add_lag_features(df, lags=(1, 2, 3), target_col=target_col)
    df = add_rolling_features(df, windows=(5, 10, 20), target_col=target_col)
    df = add_rsi(df, period=14, target_col=target_col)
    df = add_returns(df, target_col=target_col)

    df = df.dropna().reset_index(drop=True)
    return df