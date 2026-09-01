"""
report.py — print and save the overall conclusion (mirrors notebook Section 7).
"""

import os
import pandas as pd

from config import (
    RESULTS_DIR,
    SHARPE_GOOD_THRESHOLD,
    CAGR_GOOD_THRESHOLD,
)

os.makedirs(RESULTS_DIR, exist_ok=True)


def print_conclusion(
    summary:   pd.DataFrame,
    bm_cagr:   float,
    bm_sharpe: float,
    stocks:    list[str],
) -> None:
    """
    Print the overall conclusion table (mirrors notebook Section 7).
    Also saves per_stock_metrics.csv to RESULTS_DIR.
    """
    avg_cagr   = summary["CAGR (%)"].mean()
    avg_sharpe = summary["Sharpe Ratio"].mean()

    good     = summary[
        (summary["Sharpe Ratio"] > SHARPE_GOOD_THRESHOLD) &
        (summary["CAGR (%)"]     > CAGR_GOOD_THRESHOLD * 100)
    ]
    pct_good = len(good) / len(stocks) * 100

    print("\n" + "=" * 60)
    print("  OVERALL CONCLUSION")
    print("=" * 60)
    print(f"  {'Stocks analysed':<38} {len(stocks)}")
    print(f"  {'Avg CAGR (strategy)':<38} {avg_cagr:.2f}%")
    print(f"  {'Avg Sharpe (strategy)':<38} {avg_sharpe:.3f}")
    print(f"  {'Benchmark CAGR':<38} {bm_cagr*100:.2f}%")
    print(f"  {'Benchmark Sharpe':<38} {bm_sharpe:.3f}")
    print(
        f"  Stocks where strategy works well "
        f"(Sharpe>{SHARPE_GOOD_THRESHOLD} & CAGR>0): "
        f"{len(good)} ({pct_good:.0f}%)"
    )
    print()

    if avg_sharpe > bm_sharpe:
        print("On average, mean reversion BEATS the benchmark "
              "on a risk-adjusted basis.")
    else:
        print("On average, mean reversion TRAILS the benchmark "
              "on a risk-adjusted basis.")

    print()
    print(f"  Best stocks  : {summary.head(5).index.tolist()}")
    print(f"  Worst stocks : {summary.tail(5).index.tolist()}")
    print("=" * 60)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "per_stock_metrics.csv")
    # Drop the equity column (Series) before saving
    save_df = summary.drop(
        columns=[c for c in summary.columns if c == "equity"],
        errors="ignore",
    )
    save_df.to_csv(csv_path)
    print(f"\n  Metrics saved → {csv_path}")
