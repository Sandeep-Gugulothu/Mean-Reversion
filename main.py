"""
main.py — run the full per-stock mean reversion research pipeline.

Steps
-----
1. Load / download NIFTY 50 price data
2. Run per-stock backtest
3. Build summary DataFrame
4. Compute benchmark metrics (NIFTY 50 Buy & Hold)
5. Generate all charts
6. Print overall conclusion & save metrics CSV
"""

import os
import pandas as pd

from config   import RESULTS_DIR, PLOTS_DIR, DATA_DIR
from data     import load_or_download
from backtest import run_backtest, benchmark_metrics
from plots    import generate_all_plots
from report   import print_conclusion

os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)


def build_summary(results: dict) -> pd.DataFrame:
    """
    Assemble the per-stock metrics into a single DataFrame,
    sorted by Sharpe ratio descending (mirrors notebook Section 4).
    """
    rows = {
        stock: {
            "Total Return (%)": round(r.total_ret * 100, 2),
            "CAGR (%)":         round(r.cagr      * 100, 2),
            "Sharpe Ratio":     round(r.sharpe,           3),
            "Max Drawdown (%)": round(r.max_dd    * 100, 2),
            "Win Rate (%)":     round(r.win_rate  * 100, 2),
            "Trades":           r.n_trades,
        }
        for stock, r in results.items()
    }
    summary = pd.DataFrame(rows).T
    summary.index.name = "Stock"
    summary = summary.sort_values("Sharpe Ratio", ascending=False)
    return summary


def main() -> None:

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 1 — Data")
    print("=" * 55)
    close, index_close = load_or_download()
    stocks = close.columns.tolist()
    print(f"  Stocks : {len(stocks)}  |  Days : {len(close)}")
    print(f"  Range  : {close.index[0].date()} → {close.index[-1].date()}")

    # ── 2. Per-stock backtest ─────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 2 — Per-Stock Backtest")
    print("=" * 55)
    results = run_backtest(close)

    # ── 3. Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 3 — Summary Table")
    print("=" * 55)
    summary = build_summary(results)
    print(summary.to_string())

    # ── 4. Benchmark metrics ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 4 — Benchmark (NIFTY 50 Buy & Hold)")
    print("=" * 55)
    bm = benchmark_metrics(index_close)
    print(f"  CAGR   : {bm['cagr']*100:.2f}%")
    print(f"  Sharpe : {bm['sharpe']:.3f}")

    # ── 5. Charts ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 5 — Generating Charts")
    print("=" * 55)
    generate_all_plots(results, summary, bm["cagr"], bm["sharpe"])

    # ── 6. Conclusion & CSV ───────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 6 — Conclusion")
    print("=" * 55)
    print_conclusion(summary, bm["cagr"], bm["sharpe"], stocks)

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  DONE")
    print("=" * 55)
    print(f"  Charts  →  {PLOTS_DIR}/")
    print(f"  Metrics →  {RESULTS_DIR}/per_stock_metrics.csv")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
