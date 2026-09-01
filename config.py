"""
config.py — all strategy parameters in one place.
"""

from datetime import datetime, timedelta

# Universe 
NIFTY50_TICKERS: list[str] = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
    "SUNPHARMA", "ULTRACEMCO", "BAJFINANCE", "WIPRO", "NESTLEIND",
    "POWERGRID", "NTPC", "TECHM", "HCLTECH", "ONGC",
    "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "ADANIENT", "ADANIPORTS",
    "COALINDIA", "DIVISLAB", "DRREDDY", "CIPLA", "EICHERMOT",
    "BAJAJFINSV", "BAJAJ-AUTO", "HEROMOTOCO", "BRITANNIA", "GRASIM",
    "INDUSINDBK", "M&M", "HINDALCO", "VEDL", "BPCL",
    "IOC", "SHREECEM", "APOLLOHOSP", "TATACONSUM", "UPL",
]

# yfinance ticker for NIFTY 50 index
NIFTY50_INDEX: str = "^NSEI"

# Date range (last 5 years)
END_DATE:   str = datetime.today().strftime("%Y-%m-%d")
START_DATE: str = (datetime.today() - timedelta(days=5 * 365 + 2)).strftime("%Y-%m-%d")

# Strategy parameters (per-stock)
ZSCORE_WINDOW:  int   = 20      # rolling window for z-score
BUY_THRESHOLD:  float = -2.0    # buy when z-score < this  (oversold)
SELL_THRESHOLD: float =  2.0    # short when z-score > this (overbought)
HOLDING_PERIOD: int   =  5      # hold each position for N trading days

# Transaction costs
# 0.00 % per leg applied on position *changes* (entry + exit) for testing deployment costs are more.
COST_PER_LEG: float = 0.000

# Sharpe / CAGR
SHARPE_GOOD_THRESHOLD: float = 0.5
CAGR_GOOD_THRESHOLD:   float = 0.0   # CAGR > 0 %

# Output folders
DATA_DIR:    str = "data"
PLOTS_DIR:   str = "plots"
RESULTS_DIR: str = "results"
