import os
import warnings
import json
import pandas as pd
import yfinance as yf
from datetime import datetime

from config import (
    NIFTY50_TICKERS, NIFTY50_INDEX,
    START_DATE, END_DATE, DATA_DIR, OHLCV_DIR,
    BULK_OHLCV_PARQUET, INDEX_PARQUET, CONSTITUENTS_JSON,
)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OHLCV_DIR, exist_ok=True)

# ── yfinance ticker fixes ─────────────────────────────────────────────────────
YF_SYMBOL_MAP: dict[str, str] = {
    "BAJAJ-AUTO":  "BAJAJ-AUTO",
    "M&M":         "M%26M",
}


def _to_yf(symbol: str) -> str:
    """Convert NSE symbol to yfinance ticker (appends .NS, handles special chars)."""
    sym = YF_SYMBOL_MAP.get(symbol, symbol.replace("&", "%26"))
    return f"{sym}.NS"


def _from_yf(yf_ticker: str) -> str:
    """Strip .NS suffix and reverse any encoding."""
    sym = yf_ticker.replace(".NS", "").replace("%26", "&")
    return sym


def download_universe_ohlcv(
    tickers: list[str] = NIFTY50_TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
) -> dict[str, pd.DataFrame]:
    """
    Download complete OHLCV data for all tickers:
    - Vectorized extraction from yfinance batch download.
    - Memory-efficient float32 / int64 typing.
    - Automatically adjusts prices for splits & dividends.
    - Saves individual {SYMBOL}.parquet files to data/ohlcv/.
    - Combines all valid stocks into data/nifty50_bulk_ohlcv.parquet.
    - Returns a dict {symbol: ohlcv_df}.
    """
    yf_tickers = [_to_yf(t) for t in tickers]
    ticker_to_symbol = dict(zip(yf_tickers, tickers))

    print(f"Downloading OHLCV for {len(yf_tickers)} stocks ({start} -> {end}) ...")
    raw = yf.download(
        tickers=yf_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
        group_by="ticker",
    )

    valid_dfs: dict[str, pd.DataFrame] = {}
    bulk_records = []

    for yf_tick, sym in ticker_to_symbol.items():
        try:
            if len(yf_tickers) == 1:
                df_stock = raw.copy()
            else:
                if isinstance(raw.columns, pd.MultiIndex):
                    if yf_tick in raw.columns.levels[0]:
                        df_stock = raw[yf_tick].copy()
                    elif yf_tick in raw.columns.levels[1]:
                        df_stock = raw.xs(yf_tick, axis=1, level=1).copy()
                    else:
                        continue
                else:
                    continue

            # Check if required columns exist
            cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df_stock.columns]
            df_stock = df_stock[cols].dropna(subset=["Close"]).copy()

            # Quality check: reject if >20% missing or empty
            if df_stock.empty or len(df_stock) < 100:
                print(f"  Skipping {sym}: insufficient data ({len(df_stock)} rows)")
                continue

            # Ensure datetime index & memory-optimize datatypes
            df_stock.index = pd.to_datetime(df_stock.index)
            df_stock.index.name = "Date"

            for p_col in ["Open", "High", "Low", "Close"]:
                if p_col in df_stock.columns:
                    df_stock[p_col] = df_stock[p_col].astype("float32")
            if "Volume" in df_stock.columns:
                df_stock["Volume"] = df_stock["Volume"].fillna(0).astype("int64")

            # 1. Save per-stock parquet
            stock_parquet_path = os.path.join(OHLCV_DIR, f"{sym}.parquet")
            df_stock.to_parquet(stock_parquet_path, engine="pyarrow", compression="snappy")
            valid_dfs[sym] = df_stock

            # 2. Append to bulk records
            df_bulk = df_stock.reset_index()
            df_bulk["Symbol"] = sym
            bulk_records.append(df_bulk)

        except Exception as e:
            print(f"  Error processing {sym}: {e}")

    # 3. Save combined bulk parquet
    if bulk_records:
        bulk_df = pd.concat(bulk_records, ignore_index=True)
        bulk_df.to_parquet(BULK_OHLCV_PARQUET, engine="pyarrow", compression="snappy", index=False)
        print(f"  Saved bulk OHLCV to {BULK_OHLCV_PARQUET}")

    print(f"  Successfully processed and cached {len(valid_dfs)} / {len(tickers)} stocks to {OHLCV_DIR}/")
    return valid_dfs


def download_index_ohlcv(
    ticker: str = NIFTY50_INDEX,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """Download and cache complete OHLCV for benchmark index."""
    print(f"Downloading index OHLCV {ticker} ...")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data for index {ticker}")

    # Flatten MultiIndex columns if returned by yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker in raw.columns.levels[0]:
            raw = raw[ticker].copy()
        elif ticker in raw.columns.levels[1]:
            raw = raw.xs(ticker, axis=1, level=1).copy()
        else:
            raw.columns = [c[0] for c in raw.columns]

    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    idx_df = raw[cols].copy()
    idx_df.index = pd.to_datetime(idx_df.index)
    idx_df.index.name = "Date"

    for p_col in ["Open", "High", "Low", "Close"]:
        if p_col in idx_df.columns:
            idx_df[p_col] = idx_df[p_col].astype("float32")
    if "Volume" in idx_df.columns:
        idx_df["Volume"] = idx_df["Volume"].fillna(0).astype("int64")

    idx_df.to_parquet(INDEX_PARQUET, engine="pyarrow", compression="snappy")
    print(f"  Saved index OHLCV to {INDEX_PARQUET}")
    return idx_df


def load_universe_ohlcv() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load all per-stock OHLCV DataFrames and index DataFrame from Parquet.
    If Parquets do not exist, triggers download.
    Returns (stocks_dict, index_df).
    """
    index_exists = os.path.exists(INDEX_PARQUET)
    stocks_present = [
        s for s in NIFTY50_TICKERS
        if os.path.exists(os.path.join(OHLCV_DIR, f"{s}.parquet"))
    ]

    if not index_exists or len(stocks_present) < len(NIFTY50_TICKERS) * 0.8:
        print("Parquet data missing or incomplete. Starting fresh download...")
        stocks_dict = download_universe_ohlcv()
        idx_df = download_index_ohlcv()
    else:
        print(f"Loading {len(stocks_present)} stocks from {OHLCV_DIR}/...")
        stocks_dict = {}
        for sym in stocks_present:
            p = os.path.join(OHLCV_DIR, f"{sym}.parquet")
            stocks_dict[sym] = pd.read_parquet(p, engine="pyarrow")
        idx_df = pd.read_parquet(INDEX_PARQUET, engine="pyarrow")

    return stocks_dict, idx_df


def load_or_download() -> tuple[pd.DataFrame, pd.Series]:
    """
    High-level backward-compatible loader for the strategy pipeline.
    Extracts aligned Close prices across universe and index.
    Returns (close_df, index_close_series).
    """
    stocks_dict, idx_df = load_universe_ohlcv()

    # Build aligned close price matrix
    close_series_list = []
    for sym, df in stocks_dict.items():
        s = df["Close"].rename(sym)
        close_series_list.append(s)

    close_df = pd.concat(close_series_list, axis=1)

    # Clean & align
    missing_frac = close_df.isna().mean()
    bad = missing_frac[missing_frac > 0.20].index.tolist()
    if bad:
        print(f"  Dropping {len(bad)} tickers with >20% missing data: {bad}")
    close_df = close_df.drop(columns=bad, errors="ignore")

    close_df = close_df.ffill().dropna()

    idx_close = idx_df["Close"].squeeze()
    idx_close.name = "NIFTY50"

    # Align dates
    common = close_df.index.intersection(idx_close.index)
    close_df = close_df.loc[common]
    idx_close = idx_close.loc[common]

    print(f"  Ready: {close_df.shape[1]} stocks x {close_df.shape[0]} trading days ({close_df.index[0].date()} -> {close_df.index[-1].date()})")
    return close_df, idx_close


if __name__ == "__main__":
    close, idx = load_or_download()
    print("Universe sample Close prices:")
    print(close.tail(3).iloc[:, :5])
    print("\nIndex sample Close prices:")
    print(idx.tail(3))
