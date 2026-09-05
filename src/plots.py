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
import numpy as np
import pandas as pd

try:
    from .config import PLOTS_DIR
except ImportError:
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
    """Drawdown curves for top-5 stocks by Sharpe ratio."""
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


# ── REQUIRED DELIVERABLE CHARTS (Problem Statement) ──────────────────────────

def plot_equity_curve(
    port_res,
    bm_metrics: dict,
    alt_port_res=None,
    filename: str = "equity_curve.png",
) -> str:
    """
    Required Chart 1: Master Equity Curve
    Plots Strategy Portfolio Master Equity vs NIFTY 50 Benchmark Buy & Hold.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Base Strategy Equity
    port_equity = port_res.equity * 100
    ax.plot(
        port_equity.index, port_equity.values,
        label=f"Strategy Net (CAGR={port_res.cagr*100:.1f}%, Sharpe={port_res.sharpe:.2f}, MaxDD={port_res.max_dd*100:.1f}%)",
        color="royalblue", linewidth=2.0,
    )

    # Optional Alternative Mode (e.g. Fixed Hold vs Dynamic Exit)
    if alt_port_res is not None:
        alt_equity = alt_port_res.equity * 100
        ax.plot(
            alt_equity.index, alt_equity.values,
            label=f"{alt_port_res.name} (CAGR={alt_port_res.cagr*100:.1f}%, Sharpe={alt_port_res.sharpe:.2f})",
            color="teal", linewidth=1.5, linestyle="--",
        )

    # Benchmark Equity
    bm_equity = bm_metrics["equity"] * 100
    # Align dates
    bm_equity = bm_equity.reindex(port_equity.index).ffill()
    # Re-base to 100 at start
    bm_equity = bm_equity / bm_equity.iloc[0] * 100
    ax.plot(
        bm_equity.index, bm_equity.values,
        label=f"NIFTY 50 Benchmark (CAGR={bm_metrics['cagr']*100:.1f}%, Sharpe={bm_metrics['sharpe']:.2f}, MaxDD={bm_metrics['max_dd']*100:.1f}%)",
        color="darkorange", linewidth=1.8, linestyle=":",
    )

    ax.set_title("Master Equity Curve: Mean Reversion Strategy vs NIFTY 50 Benchmark", fontsize=12, fontweight="bold")
    ax.set_ylabel("Portfolio Value (Base = 100)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    return _save(fig, filename)


def plot_drawdown_chart(
    port_res,
    bm_metrics: dict,
    filename: str = "drawdown_chart.png",
) -> str:
    """
    Required Chart 2: Underwater Drawdown Chart
    Compares peak-to-trough drawdowns between Strategy and NIFTY 50.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    # Strategy Drawdown
    port_eq = port_res.equity
    port_dd = ((port_eq - port_eq.cummax()) / port_eq.cummax()) * 100

    # Benchmark Drawdown
    bm_eq = bm_metrics["equity"].reindex(port_eq.index).ffill()
    bm_dd = ((bm_eq - bm_eq.cummax()) / bm_eq.cummax()) * 100

    ax.plot(port_dd.index, port_dd.values, label=f"Strategy DD (Max: {port_res.max_dd*100:.1f}%)", color="royalblue", linewidth=1.5)
    ax.fill_between(port_dd.index, port_dd.values, 0, color="royalblue", alpha=0.2)

    ax.plot(bm_dd.index, bm_dd.values, label=f"NIFTY 50 DD (Max: {bm_metrics['max_dd']*100:.1f}%)", color="darkorange", linewidth=1.2, linestyle="--")
    ax.fill_between(bm_dd.index, bm_dd.values, 0, color="darkorange", alpha=0.1)

    ax.set_title("Underwater Drawdown: Strategy vs NIFTY 50 Benchmark", fontsize=12, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    return _save(fig, filename)


def plot_returns_distribution(
    port_res,
    filename: str = "returns_distribution.png",
) -> str:
    """
    Required Chart 3: Distribution of Returns
    Histogram of daily net returns with normal distribution fit, 95% VaR, skewness, and kurtosis.
    """
    import numpy as np
    from scipy.stats import norm, skew, kurtosis

    returns = port_res.net_ret.dropna() * 100  # in percentage
    mean_ret = returns.mean()
    std_ret = returns.std()
    ret_skew = skew(returns)
    ret_kurt = kurtosis(returns)  # excess kurtosis
    var_95 = np.percentile(returns, 5)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Histogram
    n, bins, patches = ax.hist(
        returns, bins=60, density=True, alpha=0.6,
        color="steelblue", edgecolor="white", label="Observed Daily Returns",
    )

    # Normal Distribution Fit
    x = np.linspace(returns.min(), returns.max(), 200)
    ax.plot(x, norm.pdf(x, mean_ret, std_ret), "r--", linewidth=1.8, label="Fitted Normal Distribution")

    # Metrics lines
    ax.axvline(mean_ret, color="darkgreen", linestyle="-", linewidth=1.5, label=f"Mean: {mean_ret:+.3f}%")
    ax.axvline(var_95, color="crimson", linestyle=":", linewidth=1.8, label=f"Daily 95% VaR: {var_95:.2f}%")

    stats_text = (
        f"Mean: {mean_ret:+.3f}%\n"
        f"Std Dev: {std_ret:.3f}%\n"
        f"Skewness: {ret_skew:+.2f}\n"
        f"Excess Kurtosis: {ret_kurt:.2f}\n"
        f"Daily 95% VaR: {var_95:.2f}%"
    )
    ax.text(
        0.03, 0.95, stats_text, transform=ax.transAxes,
        fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4),
    )

    ax.set_title("Distribution of Daily Net Returns", fontsize=12, fontweight="bold")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    return _save(fig, filename)


def plot_yearly_performance(
    port_net_ret: pd.Series,
    bm_ret: pd.Series,
    filename: str = "yearly_performance.png",
) -> str:
    """
    Required Chart 4: Yearly Performance Comparison
    Grouped bar chart comparing annual returns of Strategy Net vs Benchmark.
    """
    # Align
    df = pd.DataFrame({"Strategy": port_net_ret, "Benchmark": bm_ret}).dropna()
    df["Year"] = df.index.year

    # Annual compounded returns: prod(1 + r) - 1
    yearly = df.groupby("Year").apply(lambda g: (1 + g[["Strategy", "Benchmark"]]).prod() - 1) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(yearly))
    width = 0.35

    rects1 = ax.bar(x - width/2, yearly["Strategy"], width, label="Strategy (Net of Costs)", color="royalblue")
    rects2 = ax.bar(x + width/2, yearly["Benchmark"], width, label="NIFTY 50 Benchmark", color="darkorange")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Annual Return (%)")
    ax.set_title("Yearly Performance Comparison: Strategy vs NIFTY 50 Benchmark", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in yearly.index])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add data labels
    for rect in rects1:
        h = rect.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3 if h >= 0 else -10), textcoords="offset points",
                    ha="center", va=va, fontsize=8)

    for rect in rects2:
        h = rect.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3 if h >= 0 else -10), textcoords="offset points",
                    ha="center", va=va, fontsize=8)

    plt.tight_layout()
    return _save(fig, filename)


# ── Master function ───────────────────────────────────────────────────────────

def generate_all_plots(
    results:   dict,
    summary:   pd.DataFrame,
    bm_cagr:   float,
    bm_sharpe: float,
    port_res=None,
    bm_metrics: dict = None,
) -> list[str]:
    """Generate and save all per-stock and portfolio charts."""
    print("\nGenerating charts ...")
    paths = [
        plot_equity_curves(results, summary),
        plot_cagr(summary, bm_cagr),
        plot_sharpe(summary, bm_sharpe),
        plot_drawdown(results, summary),
    ]

    if port_res is not None and bm_metrics is not None:
        paths.append(plot_equity_curve(port_res, bm_metrics, filename="equity_curve.png"))
        paths.append(plot_drawdown_chart(port_res, bm_metrics, filename="drawdown_chart.png"))
        paths.append(plot_returns_distribution(port_res, filename="returns_distribution.png"))
        paths.append(plot_yearly_performance(port_res.net_ret, bm_metrics["daily"], filename="yearly_performance.png"))

    print(f"  All charts saved to: {PLOTS_DIR}/")
    return paths
