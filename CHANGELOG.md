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
