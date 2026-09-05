"""
signals.py — kept for reference / standalone analysis only.

In the per-stock pipeline the z-score and signals are computed
*inside* backtest.backtest_single() for each stock independently.
This module exposes the same helpers as stand-alone functions in case
you want to inspect z-scores or signals without running the full backtest.
"""

import numpy as np
import pandas as pd

from config import ZSCORE_WINDOW, BUY_THRESHOLD, SELL_THRESHOLD


def compute_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns: r_t = (P_t - P_{t-1}) / P_{t-1}"""
    return close.pct_change().iloc[1:]


def compute_zscore(returns: pd.DataFrame, window: int = ZSCORE_WINDOW) -> pd.DataFrame:
    """
    Rolling z-score of returns over *window* days.
    NaN for the first *window* rows (insufficient history).
    """
    roll_mean = returns.rolling(window=window, min_periods=window).mean()
    roll_std  = returns.rolling(window=window, min_periods=window).std()
    return (returns - roll_mean) / roll_std.replace(0, np.nan)


def generate_signals(
    zscore:         pd.DataFrame,
    buy_threshold:  float = BUY_THRESHOLD,
    sell_threshold: float = SELL_THRESHOLD,
) -> pd.DataFrame:
    """
    +1  → Buy   (z < buy_threshold,  stock oversold)
    -1  → Sell  (z > sell_threshold, stock overbought)
     0  → No signal
    """
    signals = pd.DataFrame(0, index=zscore.index, columns=zscore.columns)
    signals[zscore < buy_threshold]  =  1
    signals[zscore > sell_threshold] = -1
    return signals


if __name__ == "__main__":
    from data import load_or_download

    close, _ = load_or_download()
    ret = compute_returns(close)
    z   = compute_zscore(ret)
    sig = generate_signals(z)

    n_buy  = int((sig ==  1).sum().sum())
    n_sell = int((sig == -1).sum().sum())
    print(f"Buy signals: {n_buy:,}  |  Sell signals: {n_sell:,}")
