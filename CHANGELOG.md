# Change Log

## 05-09-2026
### Improving the data collection
- Previous collected data is only the close prices
- Storing in csv is not a good idea

- Collecting OHLCV for each stock
- Stored both raw 'Close' and split/dividend-adjusted 'Adj Close' side-by-side in all Parquet files to maintain unadjusted price history while ensuring dividend/split-safe return calculations.
- Storing the data in parquet this is columnar format so less load to RAM and reduced I/O bottleneck.

### Limitation
- Data is static data downloaded as of today.
- The list of the Nifty50 List is as of today with no considerations of the previous lists in past 5 years  (Context- Nifty 50 list changes twice a year in March and Sept). 

### Future
- data\nifty50_constituents.json this is the static file which can be updated if the constituents change list can we found at https://www.nseindia.com/market-data/live-equity-market?symbol=NIFTY%2050. This can be automated.

### Testing & Validation
- Added automated data integrity test suite (`tests/test_data.py`) with real-time streaming progress to validate all 50 constituents, symbol resolution, financial sanity checks (High >= Low, positive Close, non-negative Volume), and Parquet file health.

## 05-09-2026
### Improving the backtest & removing bias
- Previous code had lookahead bias because signals formed at 3:30 PM Close were being traded on the same day.
  - Solved by adding explicit 1-day lag (`pos = signal.shift(1)`), so Day T signals only execute on Day T+1.
- Trade count was wrong because it just counted any day returns changed sign instead of actual buy orders.
  - Solved by counting only real position entries when a stock goes from 0 to +1.
- No transaction costs were included, making high-turnover mean reversion look artificially good.
  - Solved by deducting 10 bps per leg (0.1%) on portfolio turnover.
- Previous version tested 49 isolated theoretical bets and averaged them, which isn't how a real account trades.
  - Solved by combining all active signals into one unified portfolio capped at 100% gross leverage.

### Missing charts & analysis
- Had no check on whether mean reversion alpha decays over time.
  - Solved by adding yearly factor decay analysis from 2021 to 2026.
- Didn't know at what fee level the strategy stops making money.
  - Solved by running a cost sensitivity sweep from 0 to 30 bps per leg.
- Missing required deliverable charts from problem statement.
  - Solved by generating equity curve vs benchmark, underwater drawdown, daily returns distribution, and yearly comparison bar chart.
- No test suite to guarantee logic doesn't break.
  - Solved by adding `tests/test_strategy.py` testing lookahead lag, trade counts, leverage bounds, and cost math (16/16 tests passing).

## 06-09-2026
### Fixing portfolio execution & cash ledger
- Previous backtest was shorting stocks overnight on Z > 2.0 signals, which violates SEBI cash market rules (no naked overnight shorting).
  - Solved by making strategy strictly Buy-Only (Long-Only).
- Previous sizing tried to divide total portfolio value by 50 even when most money was locked in existing stocks, which risked running out of cash.
  - Solved by sizing dynamically from available cash: `cash / (50 - current_holdings)`. Once bought, positions stay untouched; new entries only use leftover cash.
- Exits were not properly defined in the streaming ledger.
  - Solved by exiting on 5th trading day or if yesterday hit Z > 2.0, executed strictly at next day's Adj Close.
- Daily prints were missing so couldn't see what the simulation was doing day-to-day.
  - Solved by adding an interactive day-by-day event inspector in the notebook to view entries, exits, and cash balance for any window.
- Python scripts (`main.py`, `src/backtest.py`, `src/research.py`) were using an older engine and didn't match the notebook numbers.
  - Solved by updating `backtest_event_driven` in `src/backtest.py` and running `python main.py` so both `.py` and `.ipynb` produce the exact same numbers (+1.85% return, -5.86% max DD, 1,234 trades).




