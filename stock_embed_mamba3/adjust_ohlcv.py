"""Split/dividend adjustment for a user's raw 1-minute OHLCV bars.

Reproduces the convention the model was trained under (ported from the authors' training
pipeline):

* factors are cumulative and backward-looking, anchored at 1.0 on the latest bar date;
* a split with ex-date D multiplies price by denominator/numerator and volume by
  numerator/denominator on every bar dated strictly before D (Eastern time);
* a cash dividend with ex-date D multiplies price by (close_prior - amount) / close_prior
  on every bar dated strictly before D, where close_prior is the close of the last bar stamped
  at or before 16:00 ET on the last bar date before D (or that day's last bar if none); volume is unchanged;
* dividends with amount >= close_prior are skipped (price left under-adjusted);
* two actions on the same ex-date are combined: amounts summed, split ratios multiplied.

Only pandas and numpy are required.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

EASTERN = "America/New_York"
REGULAR_CLOSE = dt.time(16, 0)


def clean_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Apply the training pipeline's raw-bar quality checks.

    Drops duplicate timestamps (keeps first), bars with high < low, bars whose close lies
    outside [low, high], and bars with a non-positive or missing price; negative or
    missing volume is set to 0.
    """
    df = bars.drop_duplicates(subset="timestamp", keep="first")
    ok = (df["high"] >= df["low"]) & (df["close"] <= df["high"]) & (df["close"] >= df["low"])
    ok &= (df[["open", "high", "low", "close"]] > 0).all(axis=1)
    df = df[ok.fillna(False)].copy()
    df["volume"] = df["volume"].fillna(0).clip(lower=0)
    return df.sort_values("timestamp").reset_index(drop=True)


def _parse_timestamps(ts: pd.Series) -> pd.Series:
    """Parse strings; mixed UTC offsets (e.g. across a DST change) are parsed as UTC.

    pandas 3 raises on mixed offsets, pandas 2 returns object dtype; both fall back.
    """
    try:
        parsed = pd.to_datetime(ts)
    except ValueError:
        return pd.to_datetime(ts, utc=True)
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        return pd.to_datetime(ts, utc=True)
    return parsed


def _eastern(ts: pd.Series) -> pd.Series:
    """Parse if needed, then localize naive as Eastern or convert aware to Eastern."""
    if not pd.api.types.is_datetime64_any_dtype(ts):
        ts = _parse_timestamps(ts)
    return ts.dt.tz_localize(EASTERN) if ts.dt.tz is None else ts.dt.tz_convert(EASTERN)


def _split_events(splits: pd.DataFrame | None) -> dict[dt.date, tuple[float, float]]:
    """{ex_date: (numerator, denominator)}; accepts `ratio` (10.0 = 10-for-1, 0.5 = 1-for-2).

    Same-date splits are multiplied together.
    """
    if splits is None or len(splits) == 0:
        return {}
    s = splits.copy()
    if "numerator" not in s:
        s["numerator"], s["denominator"] = s["ratio"].astype(float), 1.0
    events: dict[dt.date, tuple[float, float]] = {}
    for d, n, m in zip(s["date"], s["numerator"], s["denominator"], strict=True):
        ex_date = pd.Timestamp(d).date()
        n0, m0 = events.get(ex_date, (1.0, 1.0))
        events[ex_date] = (n0 * float(n), m0 * float(m))
    return events


def _close_prior(df: pd.DataFrame, ex_date: dt.date) -> float | None:
    before = df[df["_date"] < ex_date]
    if before.empty:
        return None
    last_day = before[before["_date"] == before["_date"].iloc[-1]]
    regular = last_day[last_day["timestamp_eastern"].dt.time <= REGULAR_CLOSE]
    return float((regular if not regular.empty else last_day)["close"].iloc[-1])


def _add_dividend(events: dict, df: pd.DataFrame, ex_date: dt.date, amount: float, close_prior) -> None:
    """Insert one dividend; a repeat ex-date adds to the amount already recorded."""
    if ex_date in events:
        events[ex_date] = (events[ex_date][0] + amount, events[ex_date][1])
        return
    if pd.isna(close_prior):
        close_prior = _close_prior(df, ex_date)
    if close_prior is not None and close_prior > 0:
        events[ex_date] = (amount, float(close_prior))


def _dividend_events(df: pd.DataFrame, dividends: pd.DataFrame | None) -> dict[dt.date, tuple[float, float]]:
    """{ex_date: (amount, close_prior)}; close_prior is taken from the bars unless given.

    Same-date dividends are summed against one close_prior.
    """
    if dividends is None or len(dividends) == 0:
        return {}
    events: dict[dt.date, tuple[float, float]] = {}
    given = dividends["close_prior"] if "close_prior" in dividends else [None] * len(dividends)
    for d, amount, cp in zip(dividends["date"], dividends["amount"], given, strict=True):
        _add_dividend(events, df, pd.Timestamp(d).date(), float(amount), cp)
    return events


def _apply_event(price: float, volume: float, ev: dt.date, splits: dict, divs: dict) -> tuple[float, float]:
    """Fold one ex-date's split and/or dividend into the running factors."""
    if ev in splits:
        num, den = splits[ev]
        price *= den / num
        volume *= num / den
    if ev in divs:
        amount, close_prior = divs[ev]
        if close_prior > amount:
            price *= (close_prior - amount) / close_prior
    return price, volume


def _factors_by_date(dates: list[dt.date], splits: dict, divs: dict) -> tuple[dict, dict]:
    """Cumulative backward-looking factors per bar date, anchored at 1.0 on the latest date.

    Walking backwards, every event with `current < ex-date <= previous date` scales all
    bars dated `current` and earlier.
    """
    event_dates = sorted(set(splits) | set(divs))
    price, volume = 1.0, 1.0
    price_by, volume_by = {}, {}
    prev = None
    for current in reversed(dates):
        for ev in event_dates:
            if prev is not None and current < ev <= prev:
                price, volume = _apply_event(price, volume, ev, splits, divs)
        price_by[current], volume_by[current] = price, volume
        prev = current
    return price_by, volume_by


def adjust_ohlcv(
    bars: pd.DataFrame,
    splits: pd.DataFrame | None = None,
    dividends: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return cleaned bars with adjusted prices ready for `make_features`.

    Args:
        bars: one security; columns `timestamp` (bar start; tz-aware, or naive taken as
            Eastern), `open`, `high`, `low`, `close`, `volume`, all unadjusted.
        splits: columns `date` (ex-date) and either `ratio` or `numerator`+`denominator`
            (numerator = shares after the split).
        dividends: columns `date` (ex-date), `amount` (cash per share as declared),
            optional `close_prior`.

    Returns:
        DataFrame with `timestamp_eastern`, `open_adj`, `high_adj`, `low_adj`, `close_adj`,
        `volume_adj`, `adj_factor_price`, `adj_factor_volume`, sorted by time.
    """
    df = clean_bars(bars)
    df["timestamp_eastern"] = _eastern(df["timestamp"])
    df["_date"] = df["timestamp_eastern"].dt.date
    price_by, volume_by = _factors_by_date(
        sorted(df["_date"].unique()), _split_events(splits), _dividend_events(df, dividends)
    )
    df["adj_factor_price"] = df["_date"].map(price_by).astype(np.float64)
    df["adj_factor_volume"] = df["_date"].map(volume_by).astype(np.float64)
    for col in ("open", "high", "low", "close"):
        df[f"{col}_adj"] = df[col] * df["adj_factor_price"]
    df["volume_adj"] = df["volume"] * df["adj_factor_volume"]
    keep = [
        "timestamp_eastern",
        "open_adj",
        "high_adj",
        "low_adj",
        "close_adj",
        "volume_adj",
        "adj_factor_price",
        "adj_factor_volume",
    ]
    return df[keep]
