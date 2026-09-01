# Mean Reversion Strategy — NIFTY 50

Quant Research: does short-term mean reversion exist in NIFTY 50 stocks,
and can it generate profitable trading signals on a **per-stock** basis?

---

## Project Structure

```
mean_reversion_strategy/
│
├── main.py          ← run the full pipeline
├── config.py        ← all parameters in one place
├── data.py          ← download & cache NIFTY 50 price data
├── signals.py       ← z-score helpers (standalone reference)
├── backtest.py      ← per-stock simulation & metrics
├── plots.py         ← all four charts
├── report.py        ← conclusion printer & CSV saver
├── Testing.ipynb    ← exploratory research notebook
├── requirements.txt
│
├── data/            ← auto-created; cached CSVs
├── plots/           ← auto-created; PNG charts
└── results/         ← auto-created; per_stock_metrics.csv
```

---

## Strategy

| Parameter | Value |
|-----------|-------|
| Universe | NIFTY 50 stocks (NSE) |
| Data | Daily close prices, last 5 years |
| Return formula | `r_t = (P_t − P_{t-1}) / P_{t-1}` |
| Z-Score window | 20 trading days |
| Buy signal | z-score < −2.0 (oversold) |
| Sell signal | z-score > +2.0 (overbought) |
| Holding period | 5 trading days |
| Transaction cost | 0.10 % per leg (0.20 % round-trip) |
| Approach | Each stock backtested **independently** |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python main.py
```

First run downloads data from Yahoo Finance (~1–2 min).
Subsequent runs load from the `data/` cache instantly.

---

## Outputs

| File | Description |
|------|-------------|
| `plots/per_stock_equity_curves.png` | Top-10 vs Bottom-10 equity curves |
| `plots/per_stock_cagr.png` | CAGR bar chart — all 49 stocks |
| `plots/per_stock_sharpe.png` | Sharpe ratio bar chart — all 49 stocks |
| `plots/per_stock_drawdown.png` | Drawdown curves — top-5 stocks |
| `results/per_stock_metrics.csv` | Full metrics table (sorted by Sharpe) |

---

## Key Findings

- **9 / 49 stocks** beat both thresholds (Sharpe > 0.5 and CAGR > 0 %)
- Best performers: TATASTEEL, BHARTIARTL, VEDL, KOTAKBANK, GRASIM
- On average the strategy **trails** the NIFTY 50 buy-and-hold benchmark
- Mean reversion works selectively — strongest in high-volatility, cyclical sectors

---

## Note on Survivorship Bias

The backtest uses the current NIFTY 50 constituent list for all historical dates.
Stocks removed from the index (typically underperformers) are excluded, which
may slightly overstate backtest returns.
