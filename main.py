"""
main.py — End-to-End Quantitative Research Pipeline for NIFTY 50 Mean Reversion.

Delivers:
1. Data Pipeline: 50 NIFTY constituents, 5-year OHLCV, adjusted prices.
2. Signal Generation: Rolling 20-day returns, z-scores, zero-lookahead execution lag.
3. Realistic Friction: 10 bps transaction cost per leg (20 bps round-trip).
4. Backtest Engines:
   - Per-stock individual backtests across all 50 stocks.
   - Unified Portfolio Engine (equal-weighted, gross leverage <= 100%).
5. Factor Decay Analysis: Year-by-year Sharpe & CAGR stability.
6. Cost Sensitivity Analysis: Breakeven cost curve.
7. Visualizations:
   - Required Chart 1: Master Equity Curve (Strategy vs NIFTY 50 Benchmark).
   - Required Chart 2: Underwater Drawdown Comparison.
   - Required Chart 3: Distribution of Daily Returns (VaR, Skewness, Kurtosis).
   - Required Chart 4: Yearly Performance Comparison Bar Chart.
   - Per-stock diagnostic charts.
8. Reports: Console summary, CSV exports, and survivorship bias disclosures.
"""

import os
import sys
import pandas as pd

# Add src to sys.path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.config import (
    RESULTS_DIR, PLOTS_DIR, DATA_DIR,
    ZSCORE_WINDOW, BUY_THRESHOLD, SELL_THRESHOLD,
    HOLDING_PERIOD, COST_PER_LEG, EXIT_Z_THRESHOLD,
)
from src.data import load_or_download
from src.backtest import run_backtest, backtest_portfolio, backtest_event_driven, benchmark_metrics
from src.plots import generate_all_plots
from src.report import print_conclusion
from src.research import run_factor_decay_analysis, run_cost_sensitivity_sweep

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def build_summary(results: dict) -> pd.DataFrame:
    """Assemble per-stock metrics into a single DataFrame sorted by Sharpe descending."""
    rows = {
        stock: {
            "Total Return (%)": round(r.total_ret * 100, 2),
            "CAGR (%)": round(r.cagr * 100, 2),
            "Sharpe Ratio": round(r.sharpe, 3),
            "Max Drawdown (%)": round(r.max_dd * 100, 2),
            "Win Rate (%)": round(r.win_rate * 100, 2),
            "Trades": r.n_trades,
        }
        for stock, r in results.items()
    }
    summary = pd.DataFrame(rows).T
    summary.index.name = "Stock"
    return summary.sort_values("Sharpe Ratio", ascending=False)


def main() -> None:
    print("\n" + "=" * 65)
    print("  NIFTY 50 MEAN REVERSION QUANTITATIVE RESEARCH PIPELINE")
    print("=" * 65)

    # ── 1. Data Collection & Preprocessing ─────────────────────────────────────
    print("\n" + "-" * 65)
    print("  STEP 1: Data Collection & Preprocessing")
    print("-" * 65)
    close, index_close = load_or_download()
    stocks = close.columns.tolist()
    print(f"  Constituents: {len(stocks)} stocks")
    print(f"  History     : {len(close)} trading days ({close.index[0].date()} to {close.index[-1].date()})")
    print(f"  Price Column: Adj Close (Corporate Action & Dividend Adjusted)")

    # ── 2. Per-Stock Backtesting ───────────────────────────────────────────────
    print("\n" + "-" * 65)
    print(f"  STEP 2: Per-Stock Backtesting (Lookback={ZSCORE_WINDOW}d, Hold={HOLDING_PERIOD}d, Cost={COST_PER_LEG*100:.2f}%/leg)")
    print("-" * 65)
    per_stock_results = run_backtest(close, hold=HOLDING_PERIOD, cost=COST_PER_LEG, exit_mode="fixed")
    summary = build_summary(per_stock_results)
    print(f"\nTop 5 Stocks by Sharpe Ratio:")
    print(summary.head(5).to_string())
    print(f"\nBottom 5 Stocks by Sharpe Ratio:")
    print(summary.tail(5).to_string())

    # ── 3. Benchmark Metrics ───────────────────────────────────────────────────
    print("\n" + "-" * 65)
    print("  STEP 3: Benchmark Comparison (NIFTY 50 Index Buy-and-Hold)")
    print("-" * 65)
    bm = benchmark_metrics(index_close)
    print(f"  NIFTY 50 Total Return : {bm['total_ret']*100:+.2f}%")
    print(f"  NIFTY 50 CAGR         : {bm['cagr']*100:+.2f}%")
    print(f"  NIFTY 50 Sharpe Ratio : {bm['sharpe']:.3f}")
    print(f"  NIFTY 50 Max Drawdown : {bm['max_dd']*100:.2f}%")

    # ── 4. Unified Portfolio Engine ───────────────────────────────────────────
    print("\n" + "-" * 65)
    print(f"  STEP 4: Unified Portfolio Engine (Buy-Only, 5-Day Hold / Z-Exit, Cash Ledger)")
    print("-" * 65)
    # Event-Driven Ledger (matching notebook)
    ed_res_gross = backtest_event_driven(close, hold=HOLDING_PERIOD, cost=0.0)
    ed_res_net = backtest_event_driven(close, hold=HOLDING_PERIOD, cost=COST_PER_LEG)

    port_gross = ed_res_gross.backtest
    port_fixed = ed_res_net.backtest

    port_comp = pd.DataFrame([
        {
            "Strategy": "Strategy (Gross, 0 bps)",
            "Total Return (%)": round(port_gross.total_ret * 100, 2),
            "CAGR (%)": round(port_gross.cagr * 100, 2),
            "Sharpe": round(port_gross.sharpe, 3),
            "Max DD (%)": round(port_gross.max_dd * 100, 2),
            "Win Rate (%)": round(port_gross.win_rate * 100, 2),
            "Total Trades": port_gross.n_trades,
        },
        {
            "Strategy": f"Strategy (Net, {int(COST_PER_LEG*10000)} bps/leg)",
            "Total Return (%)": round(port_fixed.total_ret * 100, 2),
            "CAGR (%)": round(port_fixed.cagr * 100, 2),
            "Sharpe": round(port_fixed.sharpe, 3),
            "Max DD (%)": round(port_fixed.max_dd * 100, 2),
            "Win Rate (%)": round(port_fixed.win_rate * 100, 2),
            "Total Trades": port_fixed.n_trades,
        },
        {
            "Strategy": "NIFTY 50 Index (Buy & Hold)",
            "Total Return (%)": round(bm["total_ret"] * 100, 2),
            "CAGR (%)": round(bm["cagr"] * 100, 2),
            "Sharpe": round(bm["sharpe"], 3),
            "Max DD (%)": round(bm["max_dd"] * 100, 2),
            "Win Rate (%)": round(float((bm["daily"] > 0).mean() * 100), 2),
            "Total Trades": 1,
        },
    ]).set_index("Strategy")
    print(port_comp.to_string())

    # Save portfolio comparison
    port_comp.to_csv(os.path.join(RESULTS_DIR, "portfolio_metrics.csv"))

    # ── 5. Factor Decay Analysis ──────────────────────────────────────────────
    print("\n" + "-" * 65)
    print("  STEP 5: Factor Decay Analysis (Calendar Year Stability)")
    print("-" * 65)
    decay_df = run_factor_decay_analysis(close, cost=COST_PER_LEG, exit_mode="fixed")
    print(decay_df[["CAGR (%)", "Sharpe Ratio", "Max Drawdown (%)", "Win Rate (%)", "Trades"]].to_string())

    # ── 6. Transaction Cost Simulation & Breakeven ────────────────────────────
    print("\n" + "-" * 65)
    print("  STEP 6: Transaction Cost Simulation & Breakeven Analysis")
    print("-" * 65)
    cost_df = run_cost_sensitivity_sweep(close, exit_mode="fixed")
    print(cost_df.to_string())

    # ── 7. Visualizations ─────────────────────────────────────────────────────
    print("\n" + "-" * 65)
    print("  STEP 7: Generating Required Visualizations")
    print("-" * 65)
    chart_paths = generate_all_plots(
        results=per_stock_results,
        summary=summary,
        bm_cagr=bm["cagr"],
        bm_sharpe=bm["sharpe"],
        port_res=port_fixed,
        bm_metrics=bm,
    )

    # ── 8. Conclusion & Institutional Disclosures ─────────────────────────────
    print("\n" + "-" * 65)
    print("  STEP 8: Research Conclusion & Limitations")
    print("-" * 65)
    print_conclusion(summary, bm["cagr"], bm["sharpe"], stocks)

    print("\n" + "=" * 65)
    print("  [RESEARCH COMPLETE] ALL DELIVERABLES GENERATED")
    print("=" * 65)
    print(f"  Visualizations saved to -> {PLOTS_DIR}/")
    print(f"    1. equity_curve.png         (Master Equity vs NIFTY 50)")
    print(f"    2. drawdown_chart.png        (Underwater Drawdown Comparison)")
    print(f"    3. returns_distribution.png  (Daily Returns VaR/Skew/Kurtosis)")
    print(f"    4. yearly_performance.png    (Annual Returns Comparison)")
    print(f"    5. factor_decay.png          (Yearly Sharpe Stability)")
    print(f"    6. cost_sensitivity.png       (Cost Breakeven Curve)")
    print(f"  Data & Metrics saved to -> {RESULTS_DIR}/")
    print(f"    - per_stock_metrics.csv")
    print(f"    - portfolio_metrics.csv")
    print(f"    - factor_decay.csv")
    print(f"    - cost_sensitivity.csv")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
