import os
from datetime import datetime
import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, start: str = "2020-01-01", end: str = None, save_csv: bool = False):
    end_str = end or datetime.today().strftime("%Y-%m-%d")
    print(f"🔄 Mengambil data {ticker} dari {start} sampai {end_str}...")
    df = yf.download(ticker, start=start, end=end)

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    # Flatten columns: ('Close','BBCA.JK') -> 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [(c[0] if isinstance(c, tuple) else c) for c in df.columns.to_list()]

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    if "Date" not in df.columns:
        date_like = [c for c in df.columns if str(c).lower() in ("date", "datetime", "index")]
        if date_like:
            df = df.rename(columns={date_like[0]: "Date"})

    # save the data
    if save_csv:
        os.makedirs("data", exist_ok=True)
        out = os.path.join("data", f"data_{ticker.replace('.', '_')}.csv")
        df.to_csv(out, index=False)
        print(f"Saved to: {out}")

    return df
