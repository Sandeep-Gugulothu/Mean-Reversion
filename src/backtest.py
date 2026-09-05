"""
backtest.py — quantitative backtesting engine for mean reversion.

Features:
  1. Zero lookahead bias: pos = target_position.shift(1).
  2. Realistic transaction costs: deducted on turnover |pos_t - pos_{t-1}|.
  3. Accurate trade counting: counts actual entry fills.
  4. Unified Portfolio Engine: aggregates signals across N stocks into ONE portfolio.
  5. Both fixed-holding and dynamic z-exit modes.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

try:
    from .config import (
        ZSCORE_WINDOW, BUY_THRESHOLD, SELL_THRESHOLD,
        HOLDING_PERIOD, COST_PER_LEG, EXIT_Z_THRESHOLD,
        MAX_ACTIVE_POSITIONS,
    )
    from .signals import (
        compute_returns, compute_zscore, generate_raw_signals,
        expand_positions_fixed_hold, expand_positions_dynamic_exit,
        expand_positions_zscore,
        apply_execution_lag,
    )
except ImportError:
    from config import (
        ZSCORE_WINDOW, BUY_THRESHOLD, SELL_THRESHOLD,
        HOLDING_PERIOD, COST_PER_LEG, EXIT_Z_THRESHOLD,
        MAX_ACTIVE_POSITIONS,
    )
    from signals import (
        compute_returns, compute_zscore, generate_raw_signals,
        expand_positions_fixed_hold, expand_positions_dynamic_exit,
        expand_positions_zscore,
        apply_execution_lag,
    )


@dataclass
class BacktestResult:
    """Holds performance metrics and series for a backtest (single stock or portfolio)."""
    name:        str
    total_ret:   float          # total return (decimal)
    cagr:        float          # CAGR (decimal)
    sharpe:      float          # Annualized Sharpe ratio
    max_dd:      float          # Max drawdown (decimal, negative)
    win_rate:    float          # Fraction of active return periods that were positive
    n_trades:    int            # Exact number of completed round-trip trades
    gross_ret:   pd.Series = field(repr=False)   # daily gross returns
    net_ret:     pd.Series = field(repr=False)   # daily net returns (after costs)
    equity:      pd.Series = field(repr=False)   # cumulative equity curve
    positions:   pd.Series = field(repr=False)   # position series (-1, 0, +1)


# Backwards-compatible alias for existing code
StockResult = BacktestResult


def _compute_metrics(
    daily_net: pd.Series,
    daily_gross: pd.Series,
    positions: pd.Series,
    name: str,
    n_trades: int | None = None,
) -> BacktestResult:
    """Compute institutional performance metrics from daily returns."""
    equity = (1 + daily_net).cumprod()
    total = float(equity.iloc[-1] - 1) if len(equity) > 0 else 0.0
    n_years = len(daily_net) / 252
    cagr = float((1 + total) ** (1 / n_years) - 1) if (n_years > 0 and total > -1.0) else -1.0

    std = float(daily_net.std())
    sharpe = float(daily_net.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    roll_max = equity.cummax()
    max_dd = float(((equity - roll_max) / roll_max).min()) if len(equity) > 0 else 0.0

    active = daily_net[daily_net != 0]
    win_rate = float((active > 0).mean()) if len(active) > 0 else 0.0

    # FIX (Feedback P1): Exact trade entries count
    if n_trades is None:
        entries = (positions.abs() > 0) & (positions.shift(1).fillna(0).abs() == 0)
        n_trades = int(entries.sum())

    return BacktestResult(
        name=name,
        total_ret=total,
        cagr=cagr,
        sharpe=sharpe,
        max_dd=max_dd,
        win_rate=win_rate,
        n_trades=n_trades,
        gross_ret=daily_gross,
        net_ret=daily_net,
        equity=equity,
        positions=positions,
    )


# ── Single-Stock Backtest ─────────────────────────────────────────────────────

def backtest_single(
    prices: pd.Series,
    window: int = ZSCORE_WINDOW,
    buy_z: float = BUY_THRESHOLD,
    sell_z: float = SELL_THRESHOLD,
    hold: int = HOLDING_PERIOD,
    cost: float = COST_PER_LEG,
    exit_mode: str = "zscore",   # "zscore" (pure Z-score), "fixed", or "dynamic"
    exit_z: float = EXIT_Z_THRESHOLD,
) -> BacktestResult:
    """
    Run the mean reversion strategy on a single stock's price series.

    Guarantees:
      - Explicit signal lag: pos = target_pos.shift(1) (No Lookahead Bias).
      - Accurate transaction costs: turnover * cost on position changes.
      - Exit modes:
          * "zscore": Pure Z-score exit (exits when |z| <= exit_z, no time limit).
          * "fixed": Fixed holding period of `hold` days.
          * "dynamic": Z-score exit with a max holding period cap.
    """
    ret = compute_returns(prices)
    z = compute_zscore(ret, window=window)

    if exit_mode == "zscore":
        target_pos = expand_positions_zscore(z, buy_threshold=buy_z, sell_threshold=sell_z, exit_z=exit_z)
    elif exit_mode == "dynamic":
        raw_sig = generate_raw_signals(z, buy_threshold=buy_z, sell_threshold=sell_z)
        target_pos = expand_positions_dynamic_exit(raw_sig, z, max_hold=hold, exit_z=exit_z)
    else:  # "fixed"
        raw_sig = generate_raw_signals(z, buy_threshold=buy_z, sell_threshold=sell_z)
        target_pos = expand_positions_fixed_hold(raw_sig, hold=hold)

    # Apply 1-day execution lag (Feedback P1: Zero lookahead)
    executed_pos = apply_execution_lag(target_pos)

    # Returns & turnover
    gross = executed_pos * ret
    turnover = executed_pos.diff().abs().fillna(executed_pos.abs().iloc[0])
    trade_costs = turnover * cost
    net = gross - trade_costs

    sym_name = str(prices.name) if prices.name else "Stock"
    return _compute_metrics(net, gross, executed_pos, name=sym_name)


def run_backtest(
    close_df: pd.DataFrame,
    window: int = ZSCORE_WINDOW,
    buy_z: float = BUY_THRESHOLD,
    sell_z: float = SELL_THRESHOLD,
    hold: int = HOLDING_PERIOD,
    cost: float = COST_PER_LEG,
    exit_mode: str = "zscore",
    exit_z: float = EXIT_Z_THRESHOLD,
) -> dict[str, BacktestResult]:
    """Run per-stock backtests across all columns in close_df."""
    results: dict[str, BacktestResult] = {}
    for stock in close_df.columns:
        results[stock] = backtest_single(
            close_df[stock],
            window=window,
            buy_z=buy_z,
            sell_z=sell_z,
            hold=hold,
            cost=cost,
            exit_mode=exit_mode,
            exit_z=exit_z,
        )
    print(f"  Backtested {len(results)} individual stocks (mode='{exit_mode}', cost={cost*100:.2f}%/leg).")
    return results


# ── Unified Portfolio Strategy ──────────────────────────────────

def backtest_portfolio(
    close_df: pd.DataFrame,
    window: int = ZSCORE_WINDOW,
    buy_z: float = BUY_THRESHOLD,
    sell_z: float = SELL_THRESHOLD,
    hold: int = HOLDING_PERIOD,
    cost: float = COST_PER_LEG,
    max_positions: int = MAX_ACTIVE_POSITIONS,
    exit_mode: str = "zscore",
    exit_z: float = EXIT_Z_THRESHOLD,
) -> BacktestResult:
    """
    AGGREGATE TO ONE UNIFIED PORTFOLIO:
    Instead of isolated single-stock theoretical bets, this runs a real deployable strategy:
      1. Calculate target positions for all stocks simultaneously (pure Z-score, fixed, or dynamic).
      2. Apply 1-day execution lag to all positions.
      3. At each date t, compute total active positions: N_t.
      4. Scale position weights so total portfolio exposure <= 100% (equal weight across active bets).
      5. Aggregate gross returns, deduct portfolio-level turnover costs, and build one Master Equity Curve.
    """
    returns_df = compute_returns(close_df)
    zscore_df = compute_zscore(returns_df, window=window)

    target_pos_dict = {}
    for col in close_df.columns:
        if exit_mode == "zscore":
            target_pos_dict[col] = expand_positions_zscore(
                zscore_df[col], buy_threshold=buy_z, sell_threshold=sell_z, exit_z=exit_z
            )
        elif exit_mode == "dynamic":
            raw_sig = generate_raw_signals(zscore_df[col], buy_z, sell_z)
            target_pos_dict[col] = expand_positions_dynamic_exit(
                raw_sig, zscore_df[col], max_hold=hold, exit_z=exit_z
            )
        else:  # "fixed"
            raw_sig = generate_raw_signals(zscore_df[col], buy_z, sell_z)
            target_pos_dict[col] = expand_positions_fixed_hold(
                raw_sig, hold=hold
            )

    target_pos_df = pd.DataFrame(target_pos_dict, index=returns_df.index)

    # Apply 1-day execution lag across the entire matrix (No lookahead)
    executed_pos_df = apply_execution_lag(target_pos_df)

    # Portfolio Allocation & Weighting
    # Total active absolute exposure per day
    active_counts = executed_pos_df.abs().sum(axis=1)
    
    # Weight per stock: if N active stocks, each gets 1/N weight.
    # If 0 active stocks, weight is 0 (100% cash).
    divisor = active_counts.clip(lower=1.0)
    weights_df = executed_pos_df.div(divisor, axis=0)

    # Portfolio gross return
    port_gross = (weights_df * returns_df).sum(axis=1)

    # Portfolio turnover & transaction costs
    weight_changes = weights_df.diff().abs().fillna(weights_df.abs().iloc[0])
    port_turnover = weight_changes.sum(axis=1)
    port_costs = port_turnover * cost
    port_net = port_gross - port_costs

    # Representative position indicator: average net portfolio exposure
    port_exposure = weights_df.sum(axis=1)

    # Count total entries across all individual stocks in portfolio
    stock_entries = (executed_pos_df.abs() > 0) & (executed_pos_df.shift(1).fillna(0).abs() == 0)
    total_trades = int(stock_entries.sum().sum())

    name_map = {
        "zscore": "Portfolio_PureZScore",
        "dynamic": "Portfolio_DynamicExit",
        "fixed": "Portfolio_FixedHold",
    }
    res_name = name_map.get(exit_mode, "Portfolio_MeanReversion")

    return _compute_metrics(
        port_net, port_gross, port_exposure, name=res_name, n_trades=total_trades
    )


# ── Event-Driven Cash Ledger Simulation ──────────────────────────────────────

@dataclass
class EventDrivenResult:
    """Output of the streaming day-by-day simulation."""
    backtest:     BacktestResult   # standard metrics (compatible with plots/report)
    daily_ledger: pd.DataFrame     # Date | Cash | Invested | Total Value | Holdings Count | Active Holdings
    trade_log:    pd.DataFrame     # Entry Date | Exit Date | Symbol | Hold Days | PnL % | Reason


def backtest_event_driven(
    close_df: pd.DataFrame,
    window: int = ZSCORE_WINDOW,
    buy_z: float = BUY_THRESHOLD,
    sell_z: float = SELL_THRESHOLD,
    hold: int = HOLDING_PERIOD,
    cost: float = COST_PER_LEG,
    initial_capital: float = 1_000_000.0,
    position_frac: float = 1 / 50,
) -> EventDrivenResult:
    """
    Streaming day-by-day event-driven simulation with a real cash ledger.

    Daily lifecycle (strictly no lookahead — signals use yesterday's close):
      1. Check exits: for each held position increment hold days;
         exit if hold_days >= `hold` OR yesterday's z-score > +sell_z.
         Proceeds (net of exit fee) returned to cash.
      2. Check entries: stocks where yesterday's z-score < buy_z and not
         already held. Allocate `position_frac` of total portfolio value
         from available cash. Deduct entry fee.
      3. Mark-to-market: value all positions at today's close.
         Record daily ledger snapshot.

    Returns EventDrivenResult with daily_ledger, trade_log, and a
    BacktestResult (equity curve + standard metrics) for pipeline compatibility.
    """
    returns_df = compute_returns(close_df)
    zscore_df  = compute_zscore(returns_df, window=window)

    # Align close prices to the return dates (drops day 0 where return cannot be computed)
    close_aligned = close_df.loc[returns_df.index]
    dates  = returns_df.index
    prices = close_aligned.values         # shape (T, N)
    z_vals = zscore_df.values             # shape (T, N)
    cols   = list(close_aligned.columns)
    col_idx = {sym: i for i, sym in enumerate(cols)}

    # ── State ─────────────────────────────────────────────────────────────────
    cash = initial_capital
    # holdings: symbol -> {shares, entry_price, entry_date, hold_days}
    holdings: dict[str, dict] = {}

    ledger_rows: list[dict] = []
    trade_rows:  list[dict] = []
    equity_vals: list[float] = []

    for t, date in enumerate(dates):
        price_row = prices[t]
        # Yesterday's z-score used for signal (zero lookahead)
        z_prev = z_vals[t - 1] if t > 0 else np.full(len(cols), np.nan)

        # ── 1. Exits ──────────────────────────────────────────────────────────
        to_exit = []
        for sym, pos in holdings.items():
            pos["hold_days"] += 1
            z_yesterday = z_prev[col_idx[sym]]
            time_exit  = pos["hold_days"] >= hold
            z_exit     = (not np.isnan(z_yesterday)) and (z_yesterday > sell_z)
            if time_exit or z_exit:
                to_exit.append((sym, "time" if time_exit else "z_exit"))

        for sym, reason in to_exit:
            pos        = holdings.pop(sym)
            exit_price = price_row[col_idx[sym]]
            proceeds   = pos["shares"] * exit_price
            fee        = proceeds * cost
            cash      += proceeds - fee
            pnl_pct    = (exit_price / pos["entry_price"] - 1) - 2 * cost
            trade_rows.append({
                "Entry Date":  pos["entry_date"],
                "Exit Date":   date,
                "Symbol":      sym,
                "Hold Days":   pos["hold_days"],
                "PnL %":       round(pnl_pct * 100, 4),
                "Reason":      reason,
            })

        # ── 2. Entries ────────────────────────────────────────────────────────
        for i, sym in enumerate(cols):
            if sym in holdings:
                continue
            z_yesterday = z_prev[i]
            if np.isnan(z_yesterday) or z_yesterday >= buy_z:
                continue
            
            open_slots = 50 - len(holdings)
            if open_slots <= 0 or cash <= 0:
                break
                
            alloc = cash / open_slots
            alloc = min(alloc, cash)
            if alloc < 500:  # Realistic order size floor
                continue
            fee    = alloc * cost
            shares = (alloc - fee) / price_row[i]
            cash  -= alloc
            holdings[sym] = {
                "shares":      shares,
                "entry_price": price_row[i],
                "entry_date":  date,
                "hold_days":   0,
            }

        # ── 3. Mark-to-Market ─────────────────────────────────────────────────
        invested_value = sum(
            holdings[s]["shares"] * price_row[col_idx[s]] for s in holdings
        )
        total_value = cash + invested_value
        equity_vals.append(total_value)

        ledger_rows.append({
            "Date":             date,
            "Cash (Rs.)": round(cash, 2),
            "Invested (Rs.)": round(invested_value, 2),
            "Total Value (Rs.)": round(total_value, 2),
            "Holdings Count":   len(holdings),
            "Active Holdings":  ",".join(sorted(holdings.keys())),
        })

    # ── Build output DataFrames ───────────────────────────────────────────────
    daily_ledger = pd.DataFrame(ledger_rows).set_index("Date")
    trade_log    = pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame(
        columns=["Entry Date", "Exit Date", "Symbol", "Hold Days", "PnL %", "Reason"]
    )

    equity_s  = pd.Series(equity_vals, index=dates, name="EventDriven")
    daily_ret = equity_s.pct_change().fillna(0.0)
    result    = _compute_metrics(
        daily_ret, daily_ret, daily_ret.clip(lower=0).clip(upper=1),
        name="Portfolio_EventDriven",
        n_trades=len(trade_log),
    )
    # Attach the real equity curve (absolute Rs.) instead of the normalised one
    result = BacktestResult(
        **{k: (equity_s if k == "equity" else getattr(result, k))
           for k in result.__dataclass_fields__}
    )

    print(
        f"  Event-driven sim: {len(trade_log)} trades | "
        f"Final equity: Rs.{equity_vals[-1]:,.0f} | "
        f"Sharpe: {result.sharpe:.3f}"
    )
    return EventDrivenResult(backtest=result, daily_ledger=daily_ledger, trade_log=trade_log)


# ── Benchmark Helper ──────────────────────────────────────────────────────────

def benchmark_metrics(index_close: pd.Series) -> dict:
    """Compute buy-and-hold metrics for the NIFTY 50 index."""
    bm_ret = index_close.squeeze().pct_change().dropna()
    bm_equity = (1 + bm_ret).cumprod()
    bm_total = float(bm_equity.iloc[-1] - 1)
    bm_nyears = len(bm_ret) / 252
    bm_cagr = float((1 + bm_total) ** (1 / bm_nyears) - 1) if bm_nyears > 0 else 0.0

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
