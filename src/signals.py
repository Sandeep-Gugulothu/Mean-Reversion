"""
signals.py — signal generation, z-score calculations, and position expansion.
"""

import numpy as np
import pandas as pd

try:
    from .config import (
        ZSCORE_WINDOW, BUY_THRESHOLD, SELL_THRESHOLD,
        HOLDING_PERIOD, EXIT_Z_THRESHOLD,
    )
except ImportError:
    from config import (
        ZSCORE_WINDOW, BUY_THRESHOLD, SELL_THRESHOLD,
        HOLDING_PERIOD, EXIT_Z_THRESHOLD,
    )


def compute_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Daily simple returns: r_t = (P_t - P_{t-1}) / P_{t-1}"""
    return prices.pct_change().dropna()


def compute_zscore(
    returns: pd.DataFrame | pd.Series,
    window: int = ZSCORE_WINDOW,
) -> pd.DataFrame | pd.Series:
    """
    Rolling z-score of returns over *window* days.
    Guards against division-by-zero when standard deviation is 0.
    """
    roll_mean = returns.rolling(window=window, min_periods=window).mean()
    roll_std  = returns.rolling(window=window, min_periods=window).std()
    # Replace zero std with NaN to prevent infinity spikes
    safe_std  = roll_std.replace(0.0, np.nan)
    return (returns - roll_mean) / safe_std


def generate_raw_signals(
    zscore: pd.DataFrame | pd.Series,
    buy_threshold: float = BUY_THRESHOLD,
    sell_threshold: float = SELL_THRESHOLD,
) -> pd.DataFrame | pd.Series:
    """
    Generate entry impulse signals:
      +1 -> Buy  (z < buy_threshold, stock oversold)
      -1 -> Sell (z > sell_threshold, stock overbought)
       0 -> Neutral
    """
    signals = pd.DataFrame(0, index=zscore.index, columns=zscore.columns) if isinstance(zscore, pd.DataFrame) else pd.Series(0, index=zscore.index)
    signals[zscore < buy_threshold]  =  1
    signals[zscore > sell_threshold] = -1
    return signals


def expand_positions_fixed_hold(
    signals: pd.Series,
    hold: int = HOLDING_PERIOD,
) -> pd.Series:
    """
    Expand impulse entry signals to a fixed holding period.
    New signals while already in a position are ignored.
    Returns the target position state at date t (before execution lag).
    """
    n = len(signals)
    pos = np.zeros(n)
    sig_arr = signals.values
    counter = 0

    for t in range(n):
        if counter > 0:
            pos[t] = pos[t - 1]
            counter -= 1
        else:
            if sig_arr[t] != 0:
                pos[t] = sig_arr[t]
                counter = hold - 1
            else:
                pos[t] = 0

    return pd.Series(pos, index=signals.index, name=signals.name)


def expand_positions_dynamic_exit(
    signals: pd.Series,
    zscore: pd.Series,
    max_hold: int = HOLDING_PERIOD,
    exit_z: float = EXIT_Z_THRESHOLD,
) -> pd.Series:
    """
    Dynamic Mean-Reversion Exit:
    Holds position until either:
      1. Mean reversion completes (|z| <= exit_z), OR
      2. Max holding period (max_hold days) is reached.
    """
    n = len(signals)
    pos = np.zeros(n)
    sig_arr = signals.values
    z_arr = zscore.values
    counter = 0

    for t in range(n):
        if counter > 0:
            curr_pos = pos[t - 1]
            # Check for mean reversion exit
            z_val = z_arr[t]
            is_reverted = False
            if not np.isnan(z_val):
                if curr_pos > 0 and z_val >= -exit_z:
                    is_reverted = True
                elif curr_pos < 0 and z_val <= exit_z:
                    is_reverted = True

            if is_reverted:
                pos[t] = 0
                counter = 0
            else:
                pos[t] = curr_pos
                counter -= 1
        else:
            if sig_arr[t] != 0:
                pos[t] = sig_arr[t]
                counter = max_hold - 1
            else:
                pos[t] = 0

    return pd.Series(pos, index=signals.index, name=signals.name)


def expand_positions_zscore(
    zscore: pd.Series,
    buy_threshold: float = BUY_THRESHOLD,
    sell_threshold: float = SELL_THRESHOLD,
    exit_z: float = EXIT_Z_THRESHOLD,
) -> pd.Series:
    """
    Pure Z-Score State Machine (No arbitrary holding period):
    - Long entry (+1): when zscore < buy_threshold (stock oversold).
    - Short entry (-1): when zscore > sell_threshold (stock overbought).
    - Long exit (to 0): as soon as zscore >= -exit_z (reverted back towards mean).
    - Short exit (to 0): as soon as zscore <= +exit_z (reverted back towards mean).
    - Positions are held naturally until mean reversion occurs.
    """
    n = len(zscore)
    pos = np.zeros(n)
    z_arr = zscore.values
    curr_pos = 0.0

    for t in range(n):
        z_val = z_arr[t]
        if np.isnan(z_val):
            pos[t] = 0.0
            curr_pos = 0.0
            continue

        if curr_pos == 0.0:
            if z_val < buy_threshold:
                curr_pos = 1.0
            elif z_val > sell_threshold:
                curr_pos = -1.0
        elif curr_pos > 0:  # Currently Long
            if z_val >= -exit_z:  # Reverted to mean!
                curr_pos = 0.0
                if z_val > sell_threshold:
                    curr_pos = -1.0
        elif curr_pos < 0:  # Currently Short
            if z_val <= exit_z:  # Reverted to mean!
                curr_pos = 0.0
                if z_val < buy_threshold:
                    curr_pos = 1.0

        pos[t] = curr_pos

    return pd.Series(pos, index=zscore.index, name=zscore.name)


def apply_execution_lag(positions: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    KILL HIDDEN BIAS (Feedback P1):
    Signals generated at Day t Close (3:30 PM) cannot be traded at Day t.
    We apply explicit signal lag: pos_executed[t] = target_pos[t-1].
    First day position is 0 (no lookahead).
    """
    return positions.shift(1).fillna(0)
