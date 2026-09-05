import os
import sys
import json
import pytest
import pandas as pd
import yfinance as yf

# Ensure project root is on sys.path when running script directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    CONSTITUENTS_JSON, OHLCV_DIR,
    BULK_OHLCV_PARQUET, INDEX_PARQUET,
    NIFTY50_TICKERS, NIFTY50_INDEX,
)
from src.data import _to_yf, _from_yf


class TestConstituentsJSON:
    """Validate data/nifty50_constituents.json."""

    def test_json_file_exists(self):
        assert os.path.exists(CONSTITUENTS_JSON), f"Missing {CONSTITUENTS_JSON}"

    def test_json_structure_and_count(self):
        with open(CONSTITUENTS_JSON, "r") as f:
            data = json.load(f)

        assert "tickers" in data, "JSON must contain 'tickers' key"
        assert "count" in data, "JSON must contain 'count' key"
        tickers = data["tickers"]

        # Exactly 50 constituents in Nifty 50
        assert len(tickers) == 50, f"Expected exactly 50 tickers, got {len(tickers)}"
        assert data["count"] == 50, f"Expected count=50, got {data['count']}"

        # No empty strings or whitespace
        assert all(isinstance(t, str) and len(t.strip()) > 0 for t in tickers)

        # Unique tickers
        assert len(set(tickers)) == 50, "Constituent tickers must be unique"

        # Ensure index header is not included as a stock
        assert "NIFTY 50" not in tickers
        assert "^NSEI" not in tickers


class TestTickerFetchability:
    """Validate ticker translation and Yahoo Finance symbol resolution."""

    def test_ticker_mapping_rules(self):
        # M&M must be encoded as M%26M.NS
        assert _to_yf("M&M") == "M%26M.NS"
        assert _from_yf("M%26M.NS") == "M&M"

        # Standard ticker
        assert _to_yf("RELIANCE") == "RELIANCE.NS"
        assert _from_yf("RELIANCE.NS") == "RELIANCE"

    def test_all_symbols_fetchable(self):
        """Fetch 1 recent bar for all 50 tickers to verify symbols exist on Yahoo Finance."""
        yf_tickers = [_to_yf(t) for t in NIFTY50_TICKERS]
        print(f"\n   [Testing Yahoo Finance API] Fetching live quotes for {len(yf_tickers)} symbols...", flush=True)
        data = yf.download(yf_tickers, period="5d", progress=False, group_by="ticker", auto_adjust=True)
        assert not data.empty, "Yahoo Finance batch download returned empty DataFrame"

        missing = []
        for i, (sym, yf_t) in enumerate(zip(NIFTY50_TICKERS, yf_tickers), 1):
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    sub = data[yf_t] if yf_t in data.columns.levels[0] else data.xs(yf_t, axis=1, level=1)
                else:
                    sub = data
                if sub["Close"].dropna().empty:
                    missing.append(sym)
            except Exception:
                missing.append(sym)

            # Real-time streaming status line
            status = "OK" if sym not in missing else "FAIL"
            print(f"\r   [{i:02d}/50] Validating ticker: {sym:<12} [{status}]", end="", flush=True)

        print()  # New line after completion
        assert len(missing) == 0, f"Failed to fetch data for {len(missing)} tickers: {missing}"


class TestParquetDataIntegrity:
    """Validate stored Parquet files in data/ and data/ohlcv/."""

    def test_per_stock_parquet_files_exist(self):
        assert os.path.exists(OHLCV_DIR), f"Directory {OHLCV_DIR} missing"
        parquet_files = [f for f in os.listdir(OHLCV_DIR) if f.endswith(".parquet")]
        print(f"\n   Found {len(parquet_files)} parquet files in {OHLCV_DIR}/", flush=True)
        assert len(parquet_files) == 50, f"Expected 50 parquet files in {OHLCV_DIR}, found {len(parquet_files)}"

    def test_per_stock_parquet_schema_and_sanity(self):
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        print(f"\n   Checking schema & sanity across all 50 parquet files...", flush=True)

        for i, sym in enumerate(NIFTY50_TICKERS, 1):
            path = os.path.join(OHLCV_DIR, f"{sym}.parquet")
            assert os.path.exists(path), f"Missing {path}"

            df = pd.read_parquet(path)
            assert not df.empty, f"File {sym}.parquet is empty"
            assert required_cols.issubset(set(df.columns)), f"{sym}.parquet missing required columns: {required_cols - set(df.columns)}"

            # Index checks
            assert isinstance(df.index, pd.DatetimeIndex), f"{sym}.parquet index is not DatetimeIndex"
            assert df.index.is_monotonic_increasing, f"{sym}.parquet dates are not sorted chronologically"
            assert not df.index.has_duplicates, f"{sym}.parquet has duplicate dates"

            # Sanity checks: High >= Low, Close > 0, Volume >= 0
            assert (df["High"] >= df["Low"]).all(), f"{sym} has High < Low"
            assert (df["Close"] > 0).all(), f"{sym} has non-positive Close"
            assert (df["Volume"] >= 0).all(), f"{sym} has negative Volume"

            # Real-time streaming status line
            print(f"\r   [{i:02d}/50] Checking Parquet: {sym:<12} ({len(df):>4} bars) [VALID]", end="", flush=True)

        print()  # New line after completion

    def test_index_parquet(self):
        assert os.path.exists(INDEX_PARQUET), f"Missing {INDEX_PARQUET}"
        idx_df = pd.read_parquet(INDEX_PARQUET)
        assert not idx_df.empty
        assert "Close" in idx_df.columns
        assert isinstance(idx_df.index, pd.DatetimeIndex)
        assert (idx_df["Close"] > 0).all()
        print(f"\n   Index ^NSEI: {len(idx_df)} trading days verified ({idx_df.index[0].date()} to {idx_df.index[-1].date()}) - OK", flush=True)

    def test_bulk_parquet(self):
        assert os.path.exists(BULK_OHLCV_PARQUET), f"Missing {BULK_OHLCV_PARQUET}"
        bulk_df = pd.read_parquet(BULK_OHLCV_PARQUET)
        assert not bulk_df.empty
        assert "Symbol" in bulk_df.columns
        unique_syms = set(bulk_df["Symbol"].unique())
        assert unique_syms == set(NIFTY50_TICKERS)
        print(f"\n   Bulk OHLCV: {len(bulk_df):,} total rows across {len(unique_syms)} unique symbols - OK", flush=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s", "-W", "ignore::FutureWarning"]))
