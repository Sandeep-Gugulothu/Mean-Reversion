"""
config.py — all strategy parameters in one place.
"""

import os
import json
from datetime import datetime, timedelta

# Project root directory (one level up from src/)
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Output folders
DATA_DIR:    str = os.path.join(PROJECT_ROOT, "data")
PLOTS_DIR:   str = os.path.join(PROJECT_ROOT, "plots")
RESULTS_DIR: str = os.path.join(PROJECT_ROOT, "results")
OHLCV_DIR:   str = os.path.join(DATA_DIR, "ohlcv")

# Data files
CONSTITUENTS_JSON:  str = os.path.join(DATA_DIR, "nifty50_constituents.json")
BULK_OHLCV_PARQUET: str = os.path.join(DATA_DIR, "nifty50_bulk_ohlcv.parquet")
INDEX_PARQUET:      str = os.path.join(DATA_DIR, "nifty50_index.parquet")

# Universe (loaded from JSON)
def _load_tickers() -> list[str]:
    if not os.path.exists(CONSTITUENTS_JSON):
        raise FileNotFoundError(f"Missing constituents file: {CONSTITUENTS_JSON}. Please provide the JSON file.")
    with open(CONSTITUENTS_JSON, "r") as f:
        data = json.load(f)
    return data.get("tickers", [])

NIFTY50_TICKERS: list[str] = _load_tickers()

# yfinance ticker for NIFTY 50 index
NIFTY50_INDEX: str = "^NSEI"

# Date range (last 5 years)
END_DATE:   str = datetime.today().strftime("%Y-%m-%d")
START_DATE: str = (datetime.today() - timedelta(days=5 * 365 + 2)).strftime("%Y-%m-%d")

# Strategy parameters
ZSCORE_WINDOW:  int   = 20      # rolling window for z-score
BUY_THRESHOLD:  float = -2.0    # buy when z-score < this  (oversold)
SELL_THRESHOLD: float =  2.0    # short when z-score > this (overbought)
HOLDING_PERIOD: int   =  5      # fixed hold period in trading days
EXIT_Z_THRESHOLD: float = 0.5   # dynamic exit when |z| < this (mean reverted)

# Transaction costs (0.10% per leg = 10 bps, 20 bps round-trip)
COST_PER_LEG: float = 0.0010

# Portfolio risk parameters
MAX_ACTIVE_POSITIONS: int = 10   # maximum simultaneous active positions cap

# Benchmark & Evaluation
SHARPE_GOOD_THRESHOLD: float = 0.5
CAGR_GOOD_THRESHOLD:   float = 0.0   # CAGR > 0 %
