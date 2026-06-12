
## 2026-06-12 — Data entitlement verification
- A successful API call returning rows ≠ owned dataset. Nasdaq Data Link serves free SAMPLE data (SF2: 29 Dow tickers; SF3: one 2015 quarter) on unsubscribed tables with HTTP 200. ALWAYS probe breadth (off-sample ticker, e.g. NVDA) and depth (date range) before declaring a dataset owned in the catalog.
- Futures symbol month/year codes recycle each decade AND one symbol can be two live instruments simultaneously (CLZ0 rows in 2012 = Dec-2020 listed 9y out, while 2010 rows = Dec-2010). Disambiguate at instrument_id level via last-trade date, never at symbol level via first-seen. The roll-promotion chain test (new front == prior second) is what caught this.
- Databento bills per byte: quote metadata.get_cost before ANY pull; 'trades'+'ALL_SYMBOLS' would be $1000s where ohlcv-1d is $43.
