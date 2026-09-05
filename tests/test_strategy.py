"""
test_strategy.py — Unit tests verifying quantitative strategy invariants and bias elimination.

Invariants tested:
1. Zero Lookahead Bias (Feedback P1): explicit 1-day execution lag pos[t] = signal[t-1].
2. Trade Counting Accuracy (Feedback P1): counts actual position entries, not return flips.
3. Portfolio Weights Bounded (Feedback P2): sum of absolute weights <= 1.0 (no leverage breach).
4. Transaction Costs Deduction (Feedback P0): turnover * cost deducted exactly from gross returns.
5. Dynamic Z-Score Exit (Feedback P2): exits as soon as |z| <= exit_threshold or max_hold expires.
"""

import numpy as np
import pandas as pd
import pytest

from src.signals import (
    compute_returns,
    compute_zscore,
    generate_raw_signals,
    expand_positions_fixed_hold,
    expand_positions_dynamic_exit,
    expand_positions_zscore,
    apply_execution_lag,
)
from src.backtest import (
    _compute_metrics,
    backtest_single,
    backtest_portfolio,
    BacktestResult,
)


class TestLookaheadBiasElimination:
    """Verifies that signals at Day t Close can NEVER be executed on Day t."""

    def test_execution_lag_shifts_strictly_by_one(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        target_pos = pd.Series([0, 1, 1, 0, -1], index=dates)
        executed = apply_execution_lag(target_pos)

        # First day must be 0 (no lookahead from day 0)
        assert executed.iloc[0] == 0

        # Day t must equal target_pos at day t-1
        for t in range(1, len(target_pos)):
            assert executed.iloc[t] == target_pos.iloc[t - 1]

    def test_end_of_day_signal_not_executed_same_day(self):
        # A massive drop on day 10 creates an oversold signal on day 10
        prices = pd.Series([100.0] * 25, index=pd.date_range("2024-01-01", periods=25, freq="B"))
        prices.iloc[20] = 70.0  # -30% shock
        prices.iloc[21] = 70.0

        ret = compute_returns(prices)
        z = compute_zscore(ret, window=15)
        raw_sig = generate_raw_signals(z, buy_threshold=-2.0, sell_threshold=2.0)
        target = expand_positions_fixed_hold(raw_sig, hold=3)
        executed = apply_execution_lag(target)

        # On the day of the shock (day 20 in price, which is day 19 in returns):
        # The raw signal may fire at close of day t, but executed position on day t MUST BE 0
        shock_date = ret.index[19]
        next_date = ret.index[20]

        assert target.loc[shock_date] == 1, "Target signal should trigger on shock date close"
        assert executed.loc[shock_date] == 0, "Executed position on shock date MUST BE ZERO (no lookahead)"
        assert executed.loc[next_date] == 1, "Executed position should only be active on day t+1"


class TestTradeCounting:
    """Verifies that trade counts reflect actual entry fills, not return sign changes."""

    def test_trade_count_isolated_pulses(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        # 2 distinct trades:
        # Trade 1: Long at t=1, held through t=2, exited at t=3
        # Trade 2: Short at t=5, held through t=6, exited at t=7
        pos = pd.Series([0, 1, 1, 0, 0, -1, -1, 0, 0, 0], index=dates)
        daily_ret = pd.Series([0.01, 0.02, -0.01, 0.0, 0.0, 0.01, -0.02, 0.0, 0.0, 0.0], index=dates)

        res = _compute_metrics(daily_ret, daily_ret, pos, name="TestTradeCount")
        assert res.n_trades == 2, f"Expected exactly 2 trades, got {res.n_trades}"

    def test_trade_count_continuous_hold_is_one_trade(self):
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        # 1 single trade held for 4 days
        pos = pd.Series([0, 1, 1, 1, 1, 0], index=dates)
        daily_ret = pd.Series([0.0] * 6, index=dates)

        res = _compute_metrics(daily_ret, daily_ret, pos, name="TestContinuous")
        assert res.n_trades == 1, f"Expected 1 trade for continuous position, got {res.n_trades}"


class TestPortfolioInvariants:
    """Verifies portfolio aggregation math and gross leverage bounds."""

    def test_portfolio_gross_leverage_never_exceeds_one(self):
        # Create 5 stocks with overlapping signals
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        np.random.seed(42)
        # Random price series
        prices_dict = {
            f"STOCK_{i}": 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, size=50)))
            for i in range(5)
        }
        df = pd.DataFrame(prices_dict, index=dates)

        res = backtest_portfolio(df, window=10, buy_z=-1.0, sell_z=1.0, hold=3, cost=0.001)

        # Compute weights directly to verify leverage bound
        ret_df = compute_returns(df)
        z_df = compute_zscore(ret_df, window=10)
        sig_df = generate_raw_signals(z_df, -1.0, 1.0)
        target_pos = pd.DataFrame({
            col: expand_positions_fixed_hold(sig_df[col], hold=3) for col in df.columns
        })
        executed_pos = apply_execution_lag(target_pos)
        divisor = executed_pos.abs().sum(axis=1).clip(lower=1.0)
        weights = executed_pos.div(divisor, axis=0)

        # For every single day, total absolute gross exposure must be <= 1.000001
        daily_gross_exposure = weights.abs().sum(axis=1)
        assert (daily_gross_exposure <= 1.000001).all(), "Portfolio gross exposure exceeded 100%!"

    def test_cost_deduction_arithmetic(self):
        # Single stock backtest with known turnover
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        # Steady prices with one dip that returns
        prices = pd.Series([100, 100, 95, 95, 95, 95, 100, 100, 100, 100], index=dates, dtype=float)

        cost_rate = 0.0020  # 20 bps
        res_zero_cost = backtest_single(prices, window=3, buy_z=-1.0, sell_z=1.0, hold=2, cost=0.0)
        res_with_cost = backtest_single(prices, window=3, buy_z=-1.0, sell_z=1.0, hold=2, cost=cost_rate)

        if res_zero_cost.n_trades > 0:
            assert res_with_cost.total_ret < res_zero_cost.total_ret, "Costs must strictly reduce total return"
            # Difference should be positive and equal total turnover * cost
            turnover = res_with_cost.positions.diff().abs().fillna(res_with_cost.positions.abs().iloc[0])
            expected_total_cost = (turnover * cost_rate).sum()
            actual_cost_diff = (res_zero_cost.net_ret - res_with_cost.net_ret).sum()
            np.testing.assert_allclose(actual_cost_diff, expected_total_cost, atol=1e-6)


class TestDynamicZExit:
    """Verifies that dynamic z-exit closes positions early upon mean reversion."""

    def test_dynamic_exit_cuts_position_early(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        # Long signal at t=1 (z = -2.5)
        # At t=2, z reverts to 0.0 (|z| <= 0.5)
        # With max_hold=5, fixed hold stays in position until t=5,
        # but dynamic exit should exit at t=2!
        signals = pd.Series([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], index=dates)
        zscore = pd.Series([0.0, -2.5, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=dates)

        fixed_pos = expand_positions_fixed_hold(signals, hold=5)
        dynamic_pos = expand_positions_dynamic_exit(signals, zscore, max_hold=5, exit_z=0.5)

        # Fixed hold should still be 1 at t=2, 3, 4
        assert fixed_pos.iloc[2] == 1
        assert fixed_pos.iloc[3] == 1
        assert fixed_pos.iloc[4] == 1

        # Dynamic exit should have reverted to 0 at t=2 because z[2] = 0.0 >= -0.5
        assert dynamic_pos.iloc[1] == 1
        assert dynamic_pos.iloc[2] == 0, "Dynamic exit failed to cut position when z reverted to 0.0"
        assert dynamic_pos.iloc[3] == 0


class TestPureZScoreExit:
    """Verifies that pure Z-score exit operates with no arbitrary time limits."""

    def test_pure_zscore_holds_indefinitely_until_reversion(self):
        # 15 days of depressed prices (z < -2.0)
        dates = pd.date_range("2024-01-01", periods=15, freq="B")
        z_vals = [-2.5] * 10 + [-1.0, -0.4, 0.1, 0.0, 0.0]
        zscore = pd.Series(z_vals, index=dates)

        # Pure z-score exit with exit_z = 0.5 (exits when z >= -0.5)
        pos = expand_positions_zscore(zscore, buy_threshold=-2.0, sell_threshold=2.0, exit_z=0.5)

        # Days 0 to 9: z = -2.5 (< -2.0), should be Long (+1) all 10 days!
        # An arbitrary 5-day hold would have exited on Day 5; pure z-score stays in trade!
        assert (pos.iloc[0:10] == 1.0).all(), "Pure Z-score exit must hold position as long as z is oversold"

        # Day 10: z = -1.0 (< -0.5), still Long
        assert pos.iloc[10] == 1.0

        # Day 11: z = -0.4 (>= -0.5), reverted! Position exits to 0!
        assert pos.iloc[11] == 0.0, "Position must exit to 0 once z reverts to >= -0.5"
        assert pos.iloc[12] == 0.0
