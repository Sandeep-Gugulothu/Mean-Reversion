"""
plots.py — generate all per-stock charts matching the notebook output.

Charts produced
---------------
1. per_stock_equity_curves.png  — equity curves, top-10 vs bottom-10 by Sharpe
2. per_stock_cagr.png           — CAGR bar chart, all stocks (sorted descending)
3. per_stock_sharpe.png         — Sharpe bar chart, all stocks (sorted descending)
4. per_stock_drawdown.png       — max-drawdown bar chart, top-5 stocks by Sharpe
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from config import PLOTS_DIR

os.makedirs(PLOTS_DIR, exist_ok=True)


def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Chart 1 — Equity curves: top-10 vs bottom-10 ────────────────────────────

def plot_equity_curves(results: dict, summary: pd.DataFrame) -> str:
    """
    Two-panel chart:
      Left  — top-10 stocks by Sharpe ratio
      Right — bottom-10 stocks by Sharpe ratio
    """
    top10    = summary.head(10).index.tolist()
    bottom10 = summary.tail(10).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(
        "Equity Curves — Mean Reversion Strategy per Stock",
        fontsize=13, fontweight="bold",
    )

    for stock in top10:
        results[stock].equity.plot(ax=axes[0], label=stock, linewidth=1)
    axes[0].set_title("Top 10 Stocks (by Sharpe)")
    axes[0].set_ylabel("Portfolio Value (₹1 invested)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for stock in bottom10:
        results[stock].equity.plot(ax=axes[1], label=stock, linewidth=1)
    axes[1].set_title("Bottom 10 Stocks (by Sharpe)")
    axes[1].set_ylabel("Portfolio Value (₹1 invested)")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    return _save(fig, "per_stock_equity_curves.png")


# ── Chart 2 — CAGR bar chart ──────────────────────────────────────────────────

def plot_cagr(summary: pd.DataFrame, bm_cagr: float) -> str:
    """
    Horizontal bar chart of per-stock CAGR (%) sorted descending.
    Orange dashed line shows NIFTY 50 buy-and-hold CAGR.
    Bars are steelblue for positive CAGR, crimson for negative.
    """
    cagr_series = summary["CAGR (%)"].sort_values(ascending=False)
    colors = ["steelblue" if v > 0 else "crimson" for v in cagr_series]

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.bar(cagr_series.index, cagr_series.values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(
        bm_cagr * 100,
        color="orange", linewidth=1.5, linestyle="--",
        label=f"NIFTY B&H CAGR ({bm_cagr*100:.1f}%)",
    )
    ax.set_xticks(range(len(cagr_series)))
    ax.set_xticklabels(cagr_series.index, rotation=90, fontsize=8)
    ax.set_title("Per-Stock CAGR — Mean Reversion Strategy")
    ax.set_ylabel("CAGR (%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return _save(fig, "per_stock_cagr.png")


# ── Chart 3 — Sharpe bar chart ────────────────────────────────────────────────

def plot_sharpe(summary: pd.DataFrame, bm_sharpe: float) -> str:
    """
    Bar chart of per-stock Sharpe ratio sorted descending.
    Colour coding:
      steelblue  → Sharpe > 0.5  (good)
      gold       → 0 < Sharpe ≤ 0.5
      crimson    → Sharpe ≤ 0
    Green dashed line at 0.5 threshold; orange dashed at benchmark Sharpe.
    """
    sharpe_series = summary["Sharpe Ratio"].sort_values(ascending=False)
    colors = [
        "steelblue" if v > 0.5 else ("gold" if v > 0 else "crimson")
        for v in sharpe_series
    ]

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.bar(sharpe_series.index, sharpe_series.values, color=colors)
    ax.axhline(0,   color="black", linewidth=0.8)
    ax.axhline(0.5, color="green", linewidth=1.2, linestyle="--",
               label="Sharpe = 0.5 (good threshold)")
    ax.axhline(
        bm_sharpe,
        color="orange", linewidth=1.5, linestyle="--",
        label=f"NIFTY B&H Sharpe ({bm_sharpe:.2f})",
    )
    ax.set_xticks(range(len(sharpe_series)))
    ax.set_xticklabels(sharpe_series.index, rotation=90, fontsize=8)
    ax.set_title("Per-Stock Sharpe Ratio — Mean Reversion Strategy")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return _save(fig, "per_stock_sharpe.png")


# ── Chart 4 — Drawdown: top-5 stocks ─────────────────────────────────────────

def plot_drawdown(results: dict, summary: pd.DataFrame) -> str:
    """
    Drawdown curves for the top-5 stocks by Sharpe ratio.
    """
    top5 = summary.head(5).index.tolist()

    fig, ax = plt.subplots(figsize=(14, 5))
    for stock in top5:
        equity   = results[stock].equity
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max * 100
        drawdown.plot(ax=ax, label=stock, linewidth=1.2)

    ax.set_title("Drawdown — Top 5 Stocks (by Sharpe)")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return _save(fig, "per_stock_drawdown.png")


# ── Master function ───────────────────────────────────────────────────────────

def generate_all_plots(
    results:   dict,
    summary:   pd.DataFrame,
    bm_cagr:   float,
    bm_sharpe: float,
) -> list[str]:
    """Generate and save all four charts. Returns list of file paths."""
    print("\nGenerating charts ...")
    paths = [
        plot_equity_curves(results, summary),
        plot_cagr(summary, bm_cagr),
        plot_sharpe(summary, bm_sharpe),
        plot_drawdown(results, summary),
    ]
    print(f"  All charts saved to: {PLOTS_DIR}/")
    return paths
