"""Embed every ticker fetched by fetch_yfinance.py.

For each ticker: bars -> adjust_ohlcv (dividends only, because Yahoo bars are already
split-adjusted) -> make_features -> drop the lead-in sessions -> model.embed. The lead-in
sessions only warm the volume channel's 20-day rolling mean, as in training, and are not
embedded. Yahoo gives about 20 sessions, so the default splits them 10 and 10; with a
vendor feed use --lead-in 20 and a longer window.

Also computes a hand-made baseline per ticker (mean and standard deviation of each of the
nine channels over the same bars, 18 numbers) so the downstream examples can show what
the embedding adds over trivial summary statistics.

Writes examples/out/embeddings.npz with arrays tickers, sectors, embeddings (N, 256),
baseline (N, 18) and n_bars (N,). Needs a CUDA GPU and mamba_ssm.

    python examples/embed_universe.py [--out examples/out] [--lead-in 10]
"""

from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np
import pandas as pd
import torch

from stock_embed_mamba3 import CHANNELS, adjust_ohlcv, make_features
from stock_embed_mamba3.modeling_stock_mamba import MaskedReconstructionModel

HERE = pathlib.Path(__file__).resolve().parent
HF_REPO = "HamSession/stock-embed-mamba3"
PATCH = 5
MAX_BARS = 10_000  # the trainer kept the most recent 10,000 bars of a window
SESSION_FLAG = CHANNELS.index("session_flag")


def load_model(device: str) -> MaskedReconstructionModel:
    repo = os.environ.get("STOCK_MAMBA3_DIR")
    if not repo:
        from huggingface_hub import snapshot_download

        repo = snapshot_download(HF_REPO, allow_patterns=["config.json", "model.safetensors"])
    return MaskedReconstructionModel.from_pretrained(repo, device=device)


def read_dividends(path: pathlib.Path) -> pd.DataFrame:
    """Yahoo prices are split-adjusted already, so the splits in the file are not applied."""
    actions = pd.read_csv(path)
    return actions[actions["kind"] == "dividend"].rename(columns={"value": "amount"})[["date", "amount"]]


def window_after_lead_in(feats: np.ndarray, lead_in: int) -> np.ndarray | None:
    """Drop the first `lead_in` sessions, then trim from the front to a multiple of PATCH."""
    session = np.cumsum(feats[:, SESSION_FLAG])  # 1 on the first bar of each session
    if len(feats) == 0 or session[-1] < lead_in + 1:
        return None
    window = feats[session > lead_in][-MAX_BARS:]
    return window[len(window) % PATCH :]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", type=pathlib.Path, default=HERE / "universe.csv")
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "out")
    ap.add_argument("--lead-in", type=int, default=10, help="sessions used only to warm the volume channel")
    args = ap.parse_args()

    device = "cuda"
    model = load_model(device)
    universe = pd.read_csv(args.universe)
    rows = []
    for symbol, sector in zip(universe["ticker"], universe["sector"], strict=True):
        bars_path = args.out / "bars" / f"{symbol}.csv"
        if not bars_path.exists():
            print(f"{symbol}: no bars on disk, skipped")
            continue
        bars = pd.read_csv(bars_path)
        dividends = read_dividends(args.out / "actions" / f"{symbol}.csv")
        feats = make_features(adjust_ohlcv(bars, None, dividends))
        window = window_after_lead_in(feats, args.lead_in)
        if window is None:
            print(f"{symbol}: fewer than {args.lead_in + 1} sessions, skipped")
            continue
        x = torch.from_numpy(window).unsqueeze(0).to(device)
        emb = model.embed(x)[0].cpu().numpy()
        baseline = np.concatenate([window.mean(axis=0), window.std(axis=0)])
        rows.append((symbol, sector, emb, baseline, len(window)))
        print(f"{symbol}: {len(window)} bars embedded")

    if not rows:
        raise SystemExit("nothing embedded: run examples/fetch_yfinance.py first")
    tickers, sectors, embeddings, baseline, n_bars = (np.array(col) for col in zip(*rows, strict=True))
    np.savez(
        args.out / "embeddings.npz",
        tickers=tickers,
        sectors=sectors,
        embeddings=embeddings,
        baseline=baseline,
        n_bars=n_bars,
    )
    print(f"saved {len(tickers)} embeddings to {args.out / 'embeddings.npz'}")


if __name__ == "__main__":
    main()
