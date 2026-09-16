"""Download the example universe from Yahoo Finance with yfinance.

Writes, under examples/out/:
    bars/<TICKER>.csv      unadjusted 1-minute bars (regular session unless --prepost)
    actions/<TICKER>.csv   splits and dividends keyed by ex-date (kind, date, value)

Yahoo serves 1-minute bars for the last 30 calendar days only, at most 8 days per
request, so each ticker yields about 20 trading days. Yahoo reports no volume for
pre-market and post-market bars, which the model would read as its zero-volume sentinel,
so the default is the regular session only. A session still in progress is left out.
Re-running skips tickers already on disk.

Yahoo prices and volumes are already split-adjusted even with auto_adjust=False; only
dividends are left unapplied. embed_universe.py therefore applies the dividends only.

    python examples/fetch_yfinance.py [--universe examples/universe.csv] [--out examples/out]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import pathlib

import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).resolve().parent
EASTERN = "America/New_York"
LOOKBACK_DAYS = 29  # Yahoo rejects any 1-minute request starting more than 30 days back
CHUNK_DAYS = 8  # and any single 1-minute request longer than 8 days

logging.getLogger("yfinance").setLevel(logging.ERROR)  # quiets a per-request cookie warning


def download_bars(symbol: str, prepost: bool) -> pd.DataFrame:
    now = pd.Timestamp.now(EASTERN)
    today = now.date()
    start = today - dt.timedelta(days=LOOKBACK_DAYS)
    frames = []
    while start <= today:
        end = min(start + dt.timedelta(days=CHUNK_DAYS), today + dt.timedelta(days=1))
        frames.append(
            yf.download(
                symbol,
                interval="1m",
                start=str(start),
                end=str(end),
                prepost=prepost,
                auto_adjust=False,  # leave dividends unapplied; Yahoo splits are always applied
                progress=False,
                multi_level_index=False,
            )
        )
        start = end
    raw = pd.concat(frames)
    raw = raw[~raw.index.duplicated()].sort_index()
    raw = raw[raw.index.second == 0]  # drop a bar still in progress
    if now.time() < dt.time(16, 0):
        raw = raw[raw.index.date < today]  # drop a session still in progress
    bars = raw.rename(columns=str.lower).rename_axis("timestamp").reset_index()
    return bars[["timestamp", "open", "high", "low", "close", "volume"]]


def download_actions(symbol: str) -> pd.DataFrame:
    tk = yf.Ticker(symbol)
    splits = tk.splits.rename_axis("date").rename("value").reset_index().assign(kind="split")
    dividends = tk.dividends.rename_axis("date").rename("value").reset_index().assign(kind="dividend")
    return pd.concat([splits, dividends], ignore_index=True)[["kind", "date", "value"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", type=pathlib.Path, default=HERE / "universe.csv")
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "out")
    ap.add_argument("--prepost", action="store_true", help="include pre-market and post-market bars")
    args = ap.parse_args()

    bars_dir, actions_dir = args.out / "bars", args.out / "actions"
    bars_dir.mkdir(parents=True, exist_ok=True)
    actions_dir.mkdir(parents=True, exist_ok=True)
    for symbol in pd.read_csv(args.universe)["ticker"]:
        target = bars_dir / f"{symbol}.csv"
        if target.exists():
            print(f"{symbol}: on disk, skipped", flush=True)
            continue
        bars = download_bars(symbol, args.prepost)
        if bars.empty:
            print(f"{symbol}: no bars returned, skipped", flush=True)
            continue
        download_actions(symbol).to_csv(actions_dir / f"{symbol}.csv", index=False)  # before the bars, so a
        bars.to_csv(target, index=False)  # rerun never finds bars without actions
        sessions = bars["timestamp"].dt.date.nunique()
        first, last = bars["timestamp"].iloc[0], bars["timestamp"].iloc[-1]
        print(f"{symbol}: {len(bars)} bars over {sessions} sessions, {first} to {last}", flush=True)


if __name__ == "__main__":
    main()
