"""
research.py — Factor decay analysis and transaction cost sensitivity analysis.

Addresses core tasks:
1. Factor Decay Analysis:
   - Evaluates if mean-reversion alpha has improved, stayed stable, or weakened over time.
   - Evaluates yearly performance (2021 -> 2026) and multi-year sub-periods (In-Sample vs Out-of-Sample).
2. Transaction Cost Simulation & Breakeven:
   - Sweeps slippage & fees from 0 to 30 bps per leg.
   - Identifies the maximum allowable transaction cost before alpha decays to zero.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .config import RESULTS_DIR, PLOTS_DIR, COST_PER_LEG, HOLDING_PERIOD
    from .backtest import backtest_portfolio, backtest_event_driven, BacktestResult
except ImportError:
    from config import RESULTS_DIR, PLOTS_DIR, COST_PER_LEG, HOLDING_PERIOD
    from backtest import backtest_portfolio, backtest_event_driven, BacktestResult


def run_factor_decay_analysis(
    close_df: pd.DataFrame,
    cost: float = COST_PER_LEG,
    exit_mode: str = "fixed",
) -> pd.DataFrame:
    """
    Analyzes whether strategy performance has degraded over time.
    Calculates year-by-year CAGR, Sharpe Ratio, Max Drawdown, and Win Rate.
    """
    years = sorted(close_df.index.year.unique())
    yearly_records = []

    for y in years:
        sub_df = close_df[close_df.index.year == y]
        if len(sub_df) < 50:  # Skip years with insufficient trading days
            continue

        ed_res = backtest_event_driven(sub_df, hold=HOLDING_PERIOD, cost=cost)
        res = ed_res.backtest
        yearly_records.append({
            "Year": y,
            "Trading Days": len(sub_df),
            "Total Return (%)": round(res.total_ret * 100, 2),
            "CAGR (%)": round(res.cagr * 100, 2),
            "Sharpe Ratio": round(res.sharpe, 3),
            "Max Drawdown (%)": round(res.max_dd * 100, 2),
            "Win Rate (%)": round(res.win_rate * 100, 2),
            "Trades": res.n_trades,
        })

    decay_df = pd.DataFrame(yearly_records).set_index("Year")
    csv_path = os.path.join(RESULTS_DIR, "factor_decay.csv")
    decay_df.to_csv(csv_path)

    # Plot yearly Sharpe trend
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = decay_df.index.astype(str)
    sharpes = decay_df["Sharpe Ratio"]
    colors = ["teal" if s > 0 else "crimson" for s in sharpes]

    bars = ax.bar(x, sharpes, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Factor Decay Analysis: Annual Sharpe Ratio Over Time", fontsize=11, fontweight="bold")
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.set_xlabel("Year")
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3 if h >= 0 else -10), textcoords="offset points",
                    ha="center", va=va, fontsize=9, fontweight="bold")

    plot_path = os.path.join(PLOTS_DIR, "factor_decay.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Factor decay analysis saved -> {csv_path}")
    print(f"  Factor decay plot saved      -> {plot_path}")
    return decay_df


def run_cost_sensitivity_sweep(
    close_df: pd.DataFrame,
    exit_mode: str = "fixed",
) -> pd.DataFrame:
    """
    Sweeps transaction costs from 0 to 30 bps per leg (0 to 60 bps round-trip).
    Determines the breakeven cost limit for the mean reversion factor.
    """
    cost_levels_bps = [0, 2.5, 5, 7.5, 10, 15, 20, 25, 30]
    records = []

    for bps in cost_levels_bps:
        cost_decimal = (bps / 10000.0)  # e.g. 10 bps = 0.0010
        ed_res = backtest_event_driven(close_df, hold=HOLDING_PERIOD, cost=cost_decimal)
        res = ed_res.backtest
        records.append({
            "Cost per Leg (bps)": bps,
            "Round-Trip (bps)": bps * 2,
            "CAGR (%)": round(res.cagr * 100, 2),
            "Sharpe Ratio": round(res.sharpe, 3),
            "Max Drawdown (%)": round(res.max_dd * 100, 2),
            "Total Return (%)": round(res.total_ret * 100, 2),
        })

    cost_df = pd.DataFrame(records).set_index("Cost per Leg (bps)")
    csv_path = os.path.join(RESULTS_DIR, "cost_sensitivity.csv")
    cost_df.to_csv(csv_path)

    # Plot Cost Breakeven curve
    fig, ax1 = plt.subplots(figsize=(9, 4.5))

    ax1.plot(cost_df.index, cost_df["CAGR (%)"], "o-", color="royalblue", linewidth=2.0, label="CAGR (%)")
    ax1.set_xlabel("Cost per Leg (Basis Points)", fontsize=10)
    ax1.set_ylabel("CAGR (%)", color="royalblue", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="royalblue")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    ax2 = ax1.twinx()
    ax2.plot(cost_df.index, cost_df["Sharpe Ratio"], "s--", color="crimson", linewidth=2.0, label="Sharpe Ratio")
    ax2.set_ylabel("Sharpe Ratio", color="crimson", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax2.axhline(0, color="crimson", linestyle=":", alpha=0.5)

    # Mark standard benchmark cost of 10 bps
    ax1.axvline(10, color="darkorange", linestyle="--", linewidth=1.2, label="Base Cost (10 bps/leg)")

    fig.suptitle("Transaction Cost Sensitivity & Breakeven Curve", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, "cost_sensitivity.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Cost sensitivity analysis saved -> {csv_path}")
    print(f"  Cost curve plot saved            -> {plot_path}")
    return cost_df
