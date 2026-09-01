"""
backtest.py — per-stock mean reversion backtest engine.

Each stock is backtested independently:
  1. Compute daily returns from close prices.
  2. Compute rolling z-score (window = ZSCORE_WINDOW).
  3. Simulate a counter-based holding period:
       - Enter long  (+1) when z < BUY_THRESHOLD
       - Enter short (-1) when z > SELL_THRESHOLD
       - Hold for HOLDING_PERIOD days (new signal ignored while in position)
  4. Deduct transaction cost on every position *change*.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from config import (
    ZSCORE_WINDOW, BUY_THRESHOLD, SELL_THRESHOLD,
    HOLDING_PERIOD, COST_PER_LEG,
)


@dataclass
class StockResult:
    """Holds all metrics and series for a single stock backtest."""
    symbol:      str
    total_ret:   float          # total return (decimal)
    cagr:        float          # CAGR (decimal)
    sharpe:      float
    max_dd:      float          # max drawdown (decimal, negative)
    win_rate:    float          # fraction of active days that were profitable
    n_trades:    int
    net_ret:     pd.Series = field(repr=False)   # daily net returns
    equity:      pd.Series = field(repr=False)   # cumulative equity curve


# ── Core per-stock simulation ─────────────────────────────────────────────────

def backtest_single(
    prices:     pd.Series,
    window:     int   = ZSCORE_WINDOW,
    buy_z:      float = BUY_THRESHOLD,
    sell_z:     float = SELL_THRESHOLD,
    hold:       int   = HOLDING_PERIOD,
    cost:       float = COST_PER_LEG,
) -> StockResult:
    """
    Run the mean reversion strategy on a single stock's close-price series.

    Position logic (counter-based, faithful to the notebook):
      - When not in a position and z[t-1] < buy_z  → enter long,  hold N days.
      - When not in a position and z[t-1] > sell_z → enter short, hold N days.
      - A new signal is ignored while a position is active (counter > 0).

    Cost is deducted on every position change (|Δpos| * cost).
    """
    ret   = prices.pct_change().dropna()
    mu    = ret.rolling(window).mean()
    sigma = ret.rolling(window).std()
    z     = (ret - mu) / sigma

    n       = len(ret)
    pos     = np.zeros(n)
    counter = 0
    z_arr   = z.values

    for t in range(1, n):
        if counter > 0:
            pos[t] = pos[t - 1]
            counter -= 1
        else:
            pos[t] = 0
            if not np.isnan(z_arr[t - 1]):
                if z_arr[t - 1] < buy_z:
                    pos[t]  = 1
                    counter = hold
                elif z_arr[t - 1] > sell_z:
                    pos[t]  = -1
                    counter = hold

    gross      = pos * ret.values
    trade_cost = np.abs(np.diff(pos, prepend=0)) * cost
    net        = gross - trade_cost

    net_series = pd.Series(net, index=ret.index, name=prices.name)
    return _compute_metrics(net_series, symbol=str(prices.name))


def _compute_metrics(daily_ret: pd.Series, symbol: str) -> StockResult:
    """Compute all performance metrics from a daily net-return Series."""
    equity   = (1 + daily_ret).cumprod()
    total    = float(equity.iloc[-1] - 1)
    n_years  = len(daily_ret) / 252
    cagr     = float((1 + total) ** (1 / n_years) - 1) if n_years > 0 else 0.0

    std = float(daily_ret.std())
    sharpe = float(daily_ret.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    roll_max = equity.cummax()
    max_dd   = float(((equity - roll_max) / roll_max).min())

    active   = daily_ret[daily_ret != 0]
    win_rate = float((active > 0).mean()) if len(active) > 0 else 0.0

    # Count trades: number of position-entry events
    n_trades = int(
        (np.diff((daily_ret.values != 0).astype(int)) != 0).sum() / 2
    )

    return StockResult(
        symbol=symbol,
        total_ret=total,
        cagr=cagr,
        sharpe=sharpe,
        max_dd=max_dd,
        win_rate=win_rate,
        n_trades=n_trades,
        net_ret=daily_ret,
        equity=equity,
    )


def run_backtest(close: pd.DataFrame) -> dict[str, StockResult]:
    """
    Run the per-stock backtest across every column in *close*.

    Returns a dict  {symbol: StockResult}.
    """
    results: dict[str, StockResult] = {}
    for stock in close.columns:
        results[stock] = backtest_single(close[stock])
    print(f"  Backtested {len(results)} stocks.")
    return results


# ── Benchmark helper ──────────────────────────────────────────────────────────

def benchmark_metrics(index_close: pd.Series) -> dict:
    """
    Compute buy-and-hold metrics for the NIFTY 50 index.

    Returns a plain dict with keys:
        cagr, sharpe, total_ret, max_dd
    """
    bm_ret    = index_close.squeeze().pct_change().dropna()
    bm_equity = (1 + bm_ret).cumprod()
    bm_total  = float(bm_equity.iloc[-1] - 1)
    bm_nyears = len(bm_ret) / 252
    bm_cagr   = float((1 + bm_total) ** (1 / bm_nyears) - 1) if bm_nyears > 0 else 0.0

    std = float(bm_ret.std())
    bm_sharpe = float(bm_ret.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    roll_max = bm_equity.cummax()
    bm_max_dd = float(((bm_equity - roll_max) / roll_max).min())

    return dict(
        cagr=bm_cagr,
        sharpe=bm_sharpe,
        total_ret=bm_total,
        max_dd=bm_max_dd,
        daily=bm_ret,
        equity=bm_equity,
    )


if __name__ == "__main__":
    from data import load_or_download

    close, idx = load_or_download()
    results = run_backtest(close)
    bm = benchmark_metrics(idx)

    # Quick sanity check — print top 5 by Sharpe
    top5 = sorted(results.values(), key=lambda r: r.sharpe, reverse=True)[:5]
    for r in top5:
        print(f"  {r.symbol:<14} CAGR={r.cagr*100:+.1f}%  Sharpe={r.sharpe:.2f}")
    print(f"\n  Benchmark  CAGR={bm['cagr']*100:+.1f}%  Sharpe={bm['sharpe']:.2f}")
