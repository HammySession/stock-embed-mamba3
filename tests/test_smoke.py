"""End-to-end smoke: unadjusted bars -> adjustment -> features -> model.

The model test needs a GPU, mamba_ssm, and the checkpoint. Point
STOCK_MAMBA3_DIR at a local snapshot to skip the Hugging Face download.
Run with `pytest -rs`: a skipped model test means the model was NOT exercised.
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd
import pytest

from stock_embed_mamba3 import CHANNELS, adjust_ohlcv, make_features

HF_REPO = "HamSession/stock-embed-mamba3"
N_DAYS = 3
BARS_PER_DAY = 16 * 60  # 04:00 to 19:59 ET, pre-market and post-market included
SPLIT_DAY = 2  # 0-based; the 2:1 split ex-date is 2024-03-06
DIVIDEND = 0.10  # ex-date 2024-03-05
SPIKE_BAR = 2 * BARS_PER_DAY + 500  # one bar on the last day with volume 1e6 x the mean, to hit the +10 clip

LR = CHANNELS.index("log_return")
LVR = CHANNELS.index("log_vol_ratio")
SF = CHANNELS.index("session_flag")


def synthetic_bars(seed: int = 0) -> pd.DataFrame:
    """One continuous price path over N_DAYS sessions, quoted UNADJUSTED: from the
    2:1 split ex-date (day 3) onward the raw prices halve and raw volume doubles."""
    rng = np.random.default_rng(seed)
    frames = []
    day = dt.datetime(2024, 3, 4, 4, 0)  # a Monday
    level = 100.0
    for d in range(N_DAYS):
        base = day + dt.timedelta(days=d)
        ts = pd.date_range(base, periods=BARS_PER_DAY, freq="min", tz="America/New_York")
        close = level * np.exp(np.cumsum(rng.normal(0, 0.001, BARS_PER_DAY)))
        level = close[-1]
        opn = close * (1 + rng.normal(0, 0.0005, BARS_PER_DAY))
        high = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.0005, BARS_PER_DAY)))
        low = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.0005, BARS_PER_DAY)))
        vol = rng.integers(1, 5000, BARS_PER_DAY).astype(float)
        vol[rng.random(BARS_PER_DAY) < 0.05] = 0.0
        scale = 0.5 if d >= SPLIT_DAY else 1.0  # raw quotes after the 2:1 split
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": ts,
                    "open": opn * scale,
                    "high": high * scale,
                    "low": low * scale,
                    "close": close * scale,
                    "volume": vol / scale,
                }
            )
        )
    bars = pd.concat(frames, ignore_index=True)
    bars.loc[SPIKE_BAR, "volume"] = 1e6 * bars["volume"].mean()
    return bars


@pytest.fixture(scope="module")
def adjusted() -> pd.DataFrame:
    splits = pd.DataFrame({"date": ["2024-03-06"], "ratio": [2.0]})
    dividends = pd.DataFrame({"date": ["2024-03-05"], "amount": [DIVIDEND]})
    return adjust_ohlcv(synthetic_bars(), splits, dividends)


@pytest.fixture(scope="module")
def feats(adjusted: pd.DataFrame) -> np.ndarray:
    return make_features(adjusted)


def test_adjustment_factors_follow_the_convention(adjusted: pd.DataFrame):
    # close_prior for the 03-05 dividend: the last raw bar at or before 16:00 on 03-04
    raw = synthetic_bars()
    raw_ts = pd.to_datetime(raw["timestamp"])
    close_prior = raw.loc[(raw_ts.dt.day == 4) & (raw_ts.dt.time <= dt.time(16, 0)), "close"].iloc[-1]
    div_factor = (close_prior - DIVIDEND) / close_prior

    day = adjusted["timestamp_eastern"].dt.day
    price, volume = adjusted["adj_factor_price"], adjusted["adj_factor_volume"]
    assert np.allclose(price[day == 4], 0.5 * div_factor)  # split and dividend both ahead
    assert np.allclose(price[day == 5], 0.5)  # split ahead only
    assert np.allclose(price[day == 6], 1.0)  # ex-date bars untouched
    assert np.allclose(volume[day <= 5], 2.0)
    assert np.allclose(volume[day == 6], 1.0)


def test_features_follow_the_contract(adjusted: pd.DataFrame, feats: np.ndarray):
    assert feats.shape == (N_DAYS * BARS_PER_DAY, len(CHANNELS))
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()
    # one session boundary per calendar day, and the very first bar counts as one
    assert feats[:, SF].sum() == N_DAYS
    assert feats[0, SF] == 1.0
    # the -10 sentinel marks exactly the zero-volume bars
    zero_vol = adjusted["volume_adj"].to_numpy() == 0
    assert zero_vol.any()
    np.testing.assert_array_equal(feats[:, LVR] == -10.0, zero_vol)
    # the spike bar lands exactly on the +10 clip; every other non-sentinel bar is inside it
    assert feats[SPIKE_BAR, LVR] == 10.0
    others = feats[:, LVR][~zero_vol]
    assert (others <= 10.0).all() and (others[others != 10.0] < 10.0).all()
    # the raw 2:1 split on day 3 is undone by the adjustment: the day-2 -> day-3 boundary
    # return matches the continuous price path, and no jump survives anywhere
    boundary = SPLIT_DAY * BARS_PER_DAY
    c = adjusted["close_adj"].to_numpy()
    assert np.isclose(feats[boundary, LR], np.log(c[boundary] / c[boundary - 1]), atol=1e-6)
    assert np.abs(feats[:, LR]).max() < 0.05


def test_mixed_offset_timestamp_strings_parse():
    # a CSV written across a DST change carries both -04:00 and -05:00 offsets; pandas 3
    # raises on such strings and pandas 2 returns object dtype, and both must still adjust
    bars = synthetic_bars().iloc[:4].copy()
    bars["timestamp"] = [
        "2024-11-01 15:58:00-04:00",
        "2024-11-01 15:59:00-04:00",
        "2024-11-04 09:30:00-05:00",
        "2024-11-04 09:31:00-05:00",
    ]
    out = adjust_ohlcv(bars)
    assert str(out["timestamp_eastern"].dt.tz) == "America/New_York"
    assert list(out["timestamp_eastern"].dt.hour) == [15, 15, 9, 9]


def test_model_embeds_and_inpaints(feats: np.ndarray):
    torch = pytest.importorskip("torch")
    pytest.importorskip("mamba_ssm")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from stock_embed_mamba3.modeling_stock_mamba import MaskedReconstructionModel

    repo = os.environ.get("STOCK_MAMBA3_DIR")
    if not repo:
        from huggingface_hub import snapshot_download

        repo = snapshot_download(HF_REPO, allow_patterns=["config.json", "model.safetensors"])
    model = MaskedReconstructionModel.from_pretrained(repo, device="cuda")

    x = torch.from_numpy(feats[: len(feats) // 5 * 5]).unsqueeze(0).cuda()
    emb = model.embed(x)
    assert emb.shape == (1, 256)
    assert torch.isfinite(emb).all()

    n_patches = x.shape[1] // 5
    patch_mask = torch.zeros(1, n_patches, dtype=torch.bool, device="cuda")
    patch_mask[:, n_patches // 2 : n_patches // 2 + 10] = True
    filled = model.inpaint(x, patch_mask)
    assert filled.shape == x.shape
    bar_mask = patch_mask.repeat_interleave(5, dim=1).unsqueeze(-1).expand_as(x)
    assert torch.equal(filled[~bar_mask], x[~bar_mask]), "visible bars must pass through untouched"
    assert not torch.equal(filled[bar_mask], x[bar_mask]), "masked bars must be reconstructed"
    assert torch.isfinite(filled).all()
    with pytest.raises(TypeError):
        model.inpaint(x, patch_mask.to(torch.uint8))
