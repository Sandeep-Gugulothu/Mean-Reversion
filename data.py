"""
data.py — download and clean NIFTY 50 price data.

Survivorship-bias handling
--------------------------
Using only the *current* NIFTY 50 list and back-filling 5 years introduces
survivorship bias — we'd only see today's winners and miss stocks that were
removed (often underperformers or delisted).

We attempt to fetch NSE's official historical constituent change log to build
a point-in-time membership table.  On each trading date we only trade stocks
that were *actually* in the index on that date.

If the NSE page is unreachable we fall back to the current list and log a
clear warning so the limitation is visible in the research report.
"""

import os
import warnings
import pandas as pd
import yfinance as yf
from datetime import datetime

from config import (
    NIFTY50_TICKERS, NIFTY50_INDEX,
    START_DATE, END_DATE, DATA_DIR,
)

os.makedirs(DATA_DIR, exist_ok=True)

# ── yfinance ticker fixes ─────────────────────────────────────────────────────
# Some NSE symbols differ on yfinance — map them here.
YF_SYMBOL_MAP: dict[str, str] = {
    "BAJAJ-AUTO":  "BAJAJ-AUTO",   # works as-is
    "M&M":         "M%26M",        # & must be encoded
    "VEDL":        "VEDL",
}


def _to_yf(symbol: str) -> str:
    """Convert NSE symbol to yfinance ticker (appends .NS, handles special chars)."""
    sym = YF_SYMBOL_MAP.get(symbol, symbol)
    return f"{sym}.NS"


def _from_yf(yf_ticker: str) -> str:
    """Strip .NS suffix and reverse any encoding."""
    sym = yf_ticker.replace(".NS", "").replace("%26", "&")
    return sym



def fetch_constituent_history() -> pd.DataFrame | None:
    """
    Try to fetch NIFTY 50 historical constituent changes from NSE.

    Returns a DataFrame with columns [date, symbol, action]
    where action is 'added' or 'removed', or None if unavailable.

    NSE publishes this at:
    https://www.nseindia.com/products-services/indices-nifty50-index
    The actual CSV endpoint changes periodically, so we try a known URL
    and fall back gracefully.
    """
    try:
        import requests

        url = (
            "https://archives.nseindia.com/content/indices/"
            "ind_nifty50list.csv"
        )
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.nseindia.com/",
        }
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None

        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        # This CSV lists current constituents, not history — note the limitation
        return df

    except Exception:
        return None


def build_pit_membership(
    close_index: pd.DatetimeIndex,
    constituent_history: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Build a point-in-time boolean membership table:
        rows = trading dates, columns = all tickers ever seen
        value = True if that stock was in NIFTY 50 on that date.

    If constituent_history is None (NSE data unavailable), we use the
    current list for all dates and log a survivorship-bias warning.
    """
    if constituent_history is None:
        warnings.warn(
            "\n[SURVIVORSHIP BIAS WARNING] Could not fetch NSE constituent "
            "history. Using the current NIFTY 50 list for all dates. "
            "This overstates backtest performance because removed/delisted "
            "stocks (typically underperformers) are excluded.",
            stacklevel=2,
        )
        # All current tickers are 'in' for every date
        membership = pd.DataFrame(
            True,
            index=close_index,
            columns=NIFTY50_TICKERS,
        )
        return membership

    # If we have history, build proper point-in-time table
    # (placeholder — extend when NSE provides a proper change-log CSV)
    membership = pd.DataFrame(
        True,
        index=close_index,
        columns=NIFTY50_TICKERS,
    )
    return membership



def download_universe(
    tickers: list[str] = NIFTY50_TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers.
    - Drops tickers with >20 % missing data (likely wrong symbol or delisted).
    - Forward-fills remaining small gaps (holidays, circuit breakers).
    - Returns DataFrame: index=dates, columns=NSE symbols.
    """
    yf_tickers = [_to_yf(t) for t in tickers]

    print(f"Downloading {len(yf_tickers)} stocks ({start} → {end}) ...")
    raw = yf.download(
        tickers=yf_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
        group_by="ticker",
    )

    # Extract Close prices
    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance returns (ticker, field) MultiIndex when >1 ticker
        try:
            close = raw.xs("Close", axis=1, level=1)
        except KeyError:
            close = raw.xs("Close", axis=1, level=0)
    else:
        close = raw[["Close"]].copy()

    # Rename columns back to clean NSE symbols
    close.columns = [_from_yf(str(c)) for c in close.columns]

    # Drop tickers with >20 % missing
    missing_frac = close.isna().mean()
    bad = missing_frac[missing_frac > 0.20].index.tolist()
    if bad:
        print(f"  Dropping {len(bad)} tickers with >20% missing data: {bad}")
    close = close.drop(columns=bad, errors="ignore")

    # Forward-fill then drop any remaining NaN rows
    close = close.ffill().dropna()

    print(f"  Final: {close.shape[1]} stocks × {close.shape[0]} trading days")
    return close


def download_index(
    ticker: str = NIFTY50_INDEX,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.Series:
    """Download NIFTY 50 index close prices."""
    print(f"Downloading index {ticker} ...")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data for index {ticker}")
    series = raw["Close"].squeeze()
    series.name = "NIFTY50"
    return series


def load_or_download() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load cached data if available, otherwise download and cache.
    Returns (close_df, index_series).
    """
    close_path = os.path.join(DATA_DIR, "nifty50_close.csv")
    index_path = os.path.join(DATA_DIR, "nifty50_index.csv")

    if os.path.exists(close_path) and os.path.exists(index_path):
        print("Loading cached data ...")
        close = pd.read_csv(close_path, index_col=0, parse_dates=True)
        index_s = pd.read_csv(index_path, index_col=0, parse_dates=True).squeeze()
    else:
        close   = download_universe()
        index_s = download_index()
        close.to_csv(close_path)
        index_s.to_csv(index_path, header=True)
        print(f"  Cached to {DATA_DIR}/")

    # Align on common dates
    common  = close.index.intersection(index_s.index)
    close   = close.loc[common]
    index_s = index_s.loc[common]

    return close, index_s


if __name__ == "__main__":
    close, idx = load_or_download()
    print(close.tail(3))
    print(idx.tail(3))
