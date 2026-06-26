# Sample historical data (SAMPLE — synthetic & fictional)

> ⚠️ **This is a clearly-labelled SAMPLE.** The prices, tickers, firms, and analyst names here are
> **fictional and synthetically generated** (seeded, reproducible). Nothing in this folder reflects
> any real security or any real analyst's track record. It exists so the historical back-test
> builds, runs, and is validated **fully offline**. Replace the files with your own real data
> (same schema) to grade real analysts.

The back-test feeds these files through the **same** look-ahead-safe resolver and scoring funnel
the synthetic engine uses — only the data source changes.

## Files

### `manifest.json`
```json
{ "benchmark_symbol": "SPX", "default_horizon_months": 12, "label": "...", "is_sample": true }
```
- `benchmark_symbol` (required): the symbol used as the "vs the index" benchmark and the trading
  calendar. Must appear in `prices.csv`.
- `default_horizon_months` (optional, default 12): horizon applied to any call lacking one.
- `is_sample` (optional): when true, the app labels the data as sample (not user-supplied).

### `prices.csv` — long format, one row per (date, symbol)
```
date,symbol,close
2017-01-03,SPX,2401.2
2017-01-03,AVTX,30.05
...
```
- `close` is the **split- and dividend-ADJUSTED** close. Corporate actions are assumed already
  handled in the data; the engine computes simple returns `P_resolution/P_call − 1` on these
  values (no in-engine adjustment).
- **Ragged coverage is allowed.** A ticker that delists/halts simply stops having rows after its
  last trading day; an interior missing day is allowed. The benchmark should have a row for every
  market day (it defines the trading calendar).

### `calls.csv` — one row per analyst price-target call
```
call_id,analyst_id,analyst_name,firm,ticker,rating,target_price,call_date,horizon_months,note
```
- `rating` accepts common vocabularies and is normalized: Strong Buy→Buy; Outperform/Accumulate/
  Add→Overweight; Neutral/Market Perform/Equal Weight/In-line→Hold; Underperform/Reduce→Underweight;
  Strong Sell→Sell.
- `target_price` > 0. `call_date` is the **original publication date** (ISO `YYYY-MM-DD`).
- `horizon_months` optional → falls back to `default_horizon_months`.
- `note` is ignored by the engine (free-text for your own reference, e.g. flagging a revision).
- `calls.json` (a JSON array of the same fields) is also accepted in place of `calls.csv`.

## Documented policies (applied uniformly to every analyst)
- **Call-date snapping:** a raw call date is moved forward to the next benchmark trading day
  (weekend/holiday publication → next session). Forward-only, so no future information is used.
- **Deadline:** `resolution_date = call_date + round(horizon_months/12 × 252)` benchmark trading
  days, fixed at record time.
- **Revisions:** record a revised target as a **separate dated row**. Each row is resolved
  independently on its own horizon — this realizes "close the old call at its original horizon,
  open a new call from the revision date".
- **Ingest drops** (cannot form a valid, closeable call — logged, never silently scored):
  `BAD_RATING`, `BAD_TARGET`, `UNKNOWN_TICKER`, `CALL_DATE_OUT_OF_RANGE`, `NO_ENTRY_PRICE`,
  `HORIZON_BEYOND_DATA` (a recent call still open — resolves after the data ends).
- **Resolution skips** (valid call but no look-ahead-safe outcome — logged): `DELISTED_OR_HALTED`
  (no ticker price on the resolution date). Skipped calls never enter any score.

## Tricky cases deliberately included in this sample
- **Perma-bull** (`calloway`, Mornington Capital): only Buys on index-lagging names in a rising
  market → high direction hit-rate but **beat-market ≤ 0**.
- **Skilled picker** (`petrova`, Halberd Research): longs that beat the index + shorts on a
  decliner → **beat-market > 0**.
- **Delisted ticker** (`HALT`): a Buy whose 12-month horizon ends after the ticker stops trading →
  **skipped** with reason `DELISTED_OR_HALTED`.
- **Revised target** (`brandt` on `ORCA`): two dated rows on the same name within one horizon, each
  resolved independently.
- **Still-open calls**: two recent calls whose horizon runs past the data → `HORIZON_BEYOND_DATA`.

## Replacing the sample with your own data
1. Drop your own `prices.csv`, `calls.csv`, and `manifest.json` into a folder (this one, or any).
2. Run the back-test against it:
   ```bash
   python -m analyst_scorecard.backtest_cli --data-dir /path/to/your/folder
   ```
   or point the Streamlit app's "Historical" tab at the folder.

## Regenerating this sample
```bash
python data/sample_historical/_generate_sample.py
```
Deterministic (fixed seed) — same output every time.
