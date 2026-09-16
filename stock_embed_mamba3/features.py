"""Build the nine model input channels from adjusted 1-minute bars.

Reference implementation shipped with the model (pandas and numpy only). The formulas,
clips and conventions are documented in preprocessing_contract.md. This file is the
executable form of that table. It was checked against the training pipeline's
implementation with zero numerical difference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EASTERN = "America/New_York"

CHANNELS = [
    "log_return",
    "hl_range",
    "oc_body",
    "log_vol_ratio",
    "session_flag",
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
]


def make_features(df: pd.DataFrame) -> np.ndarray:
    """Nine channels for one security.

    Args:
        df: one security sorted by `timestamp_eastern`, with columns `open_adj`,
            `high_adj`, `low_adj`, `close_adj`, `volume_adj`, `timestamp_eastern`
            (tz-aware; any zone is converted to Eastern, naive is taken as Eastern).

    Returns:
        (len(df), 9) float32 array in CHANNELS order.
    """
    ts = df["timestamp_eastern"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        ts = pd.to_datetime(ts)
    ts = ts.dt.tz_localize(EASTERN) if ts.dt.tz is None else ts.dt.tz_convert(EASTERN)
    date = ts.dt.date
    prev_close = df["close_adj"].shift(1)
    denom = prev_close.fillna(df["close_adj"])

    log_return = np.log(df["close_adj"] / prev_close).fillna(0.0).clip(-0.2, 0.2)
    hl_range = ((df["high_adj"] - df["low_adj"]) / denom).clip(0.0, 0.2)
    oc_body = ((df["close_adj"] - df["open_adj"]) / denom).clip(-0.2, 0.2)

    daily_mean_vol = df.groupby(date)["volume_adj"].mean()
    rolling = daily_mean_vol.rolling(20, min_periods=1).mean().shift(1)
    rolling_per_bar = date.map(rolling)
    bad_mean = rolling_per_bar.isna() | (rolling_per_bar == 0)
    ratio = (df["volume_adj"] / rolling_per_bar).where(~bad_mean & (df["volume_adj"] > 0), 1.0)
    log_vol_ratio = np.clip(np.where(df["volume_adj"] == 0, -10.0, np.log(ratio)), -10.0, 10.0)

    session_flag = (date != date.shift(1)).astype(float)
    tod = (ts.dt.hour + ts.dt.minute / 60.0) * (2 * np.pi / 24.0)
    dow = ts.dt.weekday * (2 * np.pi / 5.0)

    feats = np.stack(
        [
            log_return,
            hl_range,
            oc_body,
            log_vol_ratio,
            session_flag,
            np.sin(tod),
            np.cos(tod),
            np.sin(dow),
            np.cos(dow),
        ],
        axis=1,
    )
    return feats.astype(np.float32)
