# Input contract: the nine feature channels

The model consumes `(batch, seq_len, 9)` float32 tensors. Each row is one 1-minute bar.
Bars are ordered by time within one security. `seq_len` must be divisible by 5 and should
be at most 10,000 (see "Window length" below).

## Step 1: adjust your raw bars with `adjust_ohlcv.py`

The model was trained on split- and dividend-adjusted prices under one specific
convention. `adjust_ohlcv.py` (pandas and numpy only) reproduces it from unadjusted bars
plus a corporate-action table. It also applies the training pipeline's raw-bar quality
checks: duplicate timestamps dropped (first kept), `high < low` dropped, close outside
`[low, high]` dropped, non-positive or missing prices dropped, negative volume set to 0,
missing volume set to 0.

Input schema for `bars`, one security:

| column | meaning |
|---|---|
| `timestamp` | bar **start** time. Timezone-aware in any zone (UTC is fine), or naive, in which case it is taken as US Eastern. **Naive UTC timestamps are a silent error**: every time-of-day channel shifts by four or five hours. |
| `open`, `high`, `low`, `close` | unadjusted prices: as they printed at the time, with no later restatement for splits or dividends, so bars before a 10-for-1 split still show the old, ten-times-higher prices |
| `volume` | shares traded in the bar |

Column names are lowercase. A yfinance-style frame converts with
`bars = df.rename(columns=str.lower).rename_axis("timestamp").reset_index()`.

```python
import pandas as pd
from adjust_ohlcv import adjust_ohlcv          # from stock_embed_mamba3 import adjust_ohlcv in the GitHub checkout

splits = pd.DataFrame({"date": ["2024-06-10"], "ratio": [10.0]})        # ex-date; 10.0 = 10-for-1, 0.5 = 1-for-2 reverse
dividends = pd.DataFrame({"date": ["2024-08-12"], "amount": [0.25]})    # ex-date, cash per share as declared

adjusted = adjust_ohlcv(bars, splits, dividends)
# columns: timestamp_eastern, open_adj, high_adj, low_adj, close_adj, volume_adj,
#          adj_factor_price, adj_factor_volume
```

The convention, stated so you can check your vendor's numbers against it:

- Factors are cumulative and backward-looking, anchored at 1.0 on the latest bar date.
  Every bar dated strictly before an ex-date (Eastern time) is scaled. The ex-date's own
  bars are not.
- Split: price × denominator/numerator, volume × numerator/denominator. `ratio` is
  numerator/denominator, so 10.0 means 10 new shares for 1 old. You may pass `numerator`
  and `denominator` columns instead of `ratio`; numerator is shares after the split.
- Cash dividend: price × (close_prior − amount) / close_prior, volume unchanged.
  `close_prior` is the close of the last bar stamped at or before 16:00 ET on the last bar
  date before the ex-date, or that day's last bar if none is stamped that early. Pass a
  `close_prior` column in the dividends table to override it. Dividends with
  `amount ≥ close_prior` are skipped and the price is left under-adjusted.
- `amount` must be the cash per share on the same share basis as the bars it adjusts. A
  feed that quotes prices on the historical basis needs dividends on that basis too. If
  the feed reports dividends on today's split-adjusted basis, a dividend that precedes a
  later split inside your bar range must be multiplied by that split's ratio.
- Two actions on the same ex-date are combined: dividend amounts are summed, split ratios
  are multiplied.
- Actions dated on or before the first bar date have no effect. Actions dated on a day
  with no bars (weekend, holiday, feed gap) still apply once, to every bar before that day.

Corporate actions must be keyed by **ex-date**. One free source is yfinance. Its prices,
volumes and dividends are all on today's split-adjusted basis even with
`auto_adjust=False`, so pass the dividends only and no splits:

```python
tk = yfinance.Ticker(sym)
dividends = tk.dividends.rename_axis("date").rename("amount").reset_index()
adjusted = adjust_ohlcv(bars, None, dividends)
```

If your vendor already delivers adjusted bars, call `adjust_ohlcv(bars)` with no actions.
That runs the quality checks and renames the columns with all factors at 1.0. This is only
correct if the vendor's convention matches the one above. A split-adjusted-only feed can
be completed by passing just the dividends, provided no split falls inside your bar range.
Split-only or differently-formulated adjustment shifts the `log_return` channel on every
action day.

## Step 2: the adjusted bar table `make_features` reads

The output of step 1: split- and dividend-adjusted 1-minute OHLCV bars for one security.
All sessions are included, pre-market and post-market too. Timestamps are in Eastern time
(`America/New_York`). Columns used: `open_adj, high_adj, low_adj, close_adj, volume_adj,
timestamp_eastern`.

Only bars that exist in the feed are used. Missing minutes are not filled.

## Window length

Training windows were 30 consecutive trading days per security, capped at 10,000 bars by
keeping the most recent 10,000. A liquid name prints far more than 10,000 bars in 30 days
with extended hours, so most windows were the last 10,000 bars of the span; per-window bar
counts were not recorded. Pass at most 10,000 bars, trimmed from the front; the shipped code applies no cap itself. Shorter inputs run but
were not evaluated.

`log_vol_ratio` needs history. Its rolling mean excludes the current day and spans the
prior 20 bar dates, and in training it was computed over each security's full history, so
every training window started warm. Feed `make_features` at least 20 trading days of bars
before the window you want, then slice the lead-in off the returned array. Without a
lead-in, the first day of your window gets the 0.0 cold-start value and days 2 to 20 use a
short expanding mean.

## Step 3: channels (order is fixed)

Let `prev_close` be the previous bar's `close_adj` (null on the first bar).

| # | name | formula | clip |
|---|---|---|---|
| 0 | `log_return` | `ln(close_adj / prev_close)`; first bar → 0.0 | [−0.2, 0.2] |
| 1 | `hl_range` | `(high_adj − low_adj) / denom`, `denom = prev_close` or `close_adj` on the first bar | [0, 0.2] |
| 2 | `oc_body` | `(close_adj − open_adj) / denom` (same `denom`) | [−0.2, 0.2] |
| 3 | `log_vol_ratio` | `ln(volume_adj / rolling_mean_vol)`. `rolling_mean_vol` is the mean of per-day mean bar volume over the prior 20 bar dates, shifted one day so the current day is excluded (an expanding mean of at least 1 day during cold start). Zero volume → −10.0 sentinel, checked first. Null or zero rolling mean → 0.0 | [−10, 10] |
| 4 | `session_flag` | 1.0 on the first bar of each Eastern-time calendar date, else 0.0 | none |
| 5 | `tod_sin` | `sin(2π · (hour + minute/60) / 24)` in Eastern time | none |
| 6 | `tod_cos` | `cos(2π · (hour + minute/60) / 24)` | none |
| 7 | `dow_sin` | `sin(2π · weekday / 5)`, Monday = 0 … Friday = 4 | none |
| 8 | `dow_cos` | `cos(2π · weekday / 5)` | none |

All channels are cast to float32. No per-stock or cross-sectional normalization is
applied. The clips above are the only scaling.

## Reference implementation (pandas)

Shipped as `features.py` next to this file and paraphrased here for reading; `features.py` is authoritative.

```python
import numpy as np, pandas as pd

def make_features(df: pd.DataFrame) -> np.ndarray:
    """One security sorted by timestamp_eastern, with columns open_adj, high_adj,
    low_adj, close_adj, volume_adj, timestamp_eastern. Returns (len(df), 9) float32."""
    ts = pd.to_datetime(df["timestamp_eastern"])
    ts = ts.dt.tz_convert("America/New_York") if ts.dt.tz is not None else ts.dt.tz_localize("America/New_York")
    date = ts.dt.date
    prev_close = df["close_adj"].shift(1)
    denom = prev_close.fillna(df["close_adj"])

    log_return = np.log(df["close_adj"] / prev_close).fillna(0.0).clip(-0.2, 0.2)
    hl_range = ((df["high_adj"] - df["low_adj"]) / denom).clip(0.0, 0.2)
    oc_body = ((df["close_adj"] - df["open_adj"]) / denom).clip(-0.2, 0.2)

    daily_mean_vol = df.groupby(date)["volume_adj"].mean()
    rolling = daily_mean_vol.rolling(20, min_periods=1).mean().shift(1)
    rolling_per_bar = pd.Series(date).map(rolling)
    bad_mean = rolling_per_bar.isna() | (rolling_per_bar == 0)
    ratio = (df["volume_adj"] / rolling_per_bar).where(~bad_mean & (df["volume_adj"] > 0), 1.0)
    log_vol_ratio = np.where(df["volume_adj"] == 0, -10.0, np.where(bad_mean, 0.0, np.log(ratio)))
    log_vol_ratio = np.clip(log_vol_ratio, -10.0, 10.0)

    session_flag = (pd.Series(date) != pd.Series(date).shift(1)).astype(float).fillna(1.0)
    tod = (ts.dt.hour + ts.dt.minute / 60.0) * (2 * np.pi / 24.0)
    dow = ts.dt.weekday * (2 * np.pi / 5.0)

    feats = np.stack([log_return, hl_range, oc_body, log_vol_ratio, session_flag,
                      np.sin(tod), np.cos(tod), np.sin(dow), np.cos(dow)], axis=1)
    return feats.astype(np.float32)
```

Trim `seq_len` to a multiple of 5, dropping bars from the front, before calling the model.
