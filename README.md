# stock-embed-mamba3

Data preparation, loader and build recipe for
[`HamSession/stock-embed-mamba3`](https://huggingface.co/HamSession/stock-embed-mamba3),
a 3.3M-parameter Mamba-3 encoder pretrained by masked-patch reconstruction on 1-minute US
equity bars, windows of up to 10,000 bars from a 30-trading-day span. Read the model card
first. It is a research artifact released with a documented null result. It has no
demonstrated forecasting or trading value and is not a generative model. Use it as a
stable per-stock fingerprint for nearest-neighbour retrieval, as a documented negative
baseline, or as a warm start for further pretraining.

## What is here

The weights live on Hugging Face. This repo holds the code around them:

| Path | What |
|---|---|
| `stock_embed_mamba3/adjust_ohlcv.py` | split and dividend adjustment of your own unadjusted minute bars under the training convention, plus the raw-bar quality checks (pandas and numpy) |
| `stock_embed_mamba3/features.py` | the nine input channels (pandas and numpy) |
| `stock_embed_mamba3/modeling_stock_mamba.py` | model definition and loader (torch, mamba_ssm, safetensors) |
| `preprocessing_contract.md` | the input contract that `adjust_ohlcv.py` and `features.py` implement |
| `docker/Dockerfile` | CUDA image with Mamba-3 built from source |
| `tests/test_smoke.py` | bars to features to embed and inpaint, end to end |
| `examples/` | free Yahoo data to embeddings, a clustering plot and a sector k-NN, with measured results |

`adjust_ohlcv.py`, `features.py` and `modeling_stock_mamba.py` are byte-identical to the
copies shipped inside the Hugging Face repo, so you can import them from either place.

## Install

Data preparation needs only numpy and pandas:

```bash
pip install -e .
```

The model needs a CUDA GPU and `mamba_ssm` with the `Mamba3` class. The PyPI release of
`mamba_ssm` is source only and needs the CUDA toolkit to build, so the supported path is
the Dockerfile. It builds for compute capability 12.0 (SM_120, RTX 5090) by default, and
`CUDA_ARCH` selects another architecture. Only the SM_120 build has been exercised by the
authors.

```bash
docker build -f docker/Dockerfile -t stock-embed-mamba3 .                          # Blackwell
docker build -f docker/Dockerfile -t stock-embed-mamba3 --build-arg CUDA_ARCH=90 .  # Hopper
docker run --rm --gpus all stock-embed-mamba3 pytest -q -rs
```

An uncached build takes about 30 minutes on an RTX 5090 host, almost all of it the Mamba
source build; a rebuild after editing this repo takes seconds. The GPU test downloads the
13 MB checkpoint from Hugging Face unless `STOCK_MAMBA3_DIR` points at a local copy. Pass `-e HF_TOKEN=...` if the model repo is private to you. A skipped
model test in the `-rs` summary means the model was not exercised, for example when the
container was started without `--gpus all`.

Outside Docker, `pip install -e ".[model,test]"` installs everything except `mamba_ssm`;
note that the default PyPI `torch` wheel is the CUDA build and is several gigabytes.

## Use

```python
import pandas as pd, torch
from huggingface_hub import snapshot_download
from stock_embed_mamba3 import adjust_ohlcv, make_features
from stock_embed_mamba3.modeling_stock_mamba import MaskedReconstructionModel

bars = ...   # one security, UNADJUSTED 1-minute bars, columns: timestamp (bar start;
             # tz-aware, or naive taken as Eastern), open, high, low, close, volume
splits = pd.DataFrame({"date": [...], "ratio": [...]})       # ex-dates; empty frame if none
dividends = pd.DataFrame({"date": [...], "amount": [...]})   # ex-dates; empty frame if none

feats = make_features(adjust_ohlcv(bars, splits, dividends))  # (seq_len, 9) float32
feats = feats[-10_000:]                                        # keep the most recent bars
feats = feats[len(feats) % 5 :]                                # seq_len divisible by 5, trimmed from the front
x = torch.from_numpy(feats).unsqueeze(0).cuda()                # (1, seq_len, 9)

repo = snapshot_download("HamSession/stock-embed-mamba3", allow_patterns=["config.json", "model.safetensors"])
model = MaskedReconstructionModel.from_pretrained(repo, device="cuda")
emb = model.embed(x)                                           # (1, 256)
tokens = model.encoder(x)                                      # (1, seq_len // 5, 256)

n_patches = x.shape[1] // 5
patch_mask = torch.zeros(1, n_patches, dtype=torch.bool, device="cuda")
patch_mask[:, n_patches // 2 : n_patches // 2 + 10] = True     # mask ten 5-bar patches
filled = model.inpaint(x, patch_mask)                          # (1, seq_len, 9); visible bars unchanged
```

To run that inside the image with your own bars on disk:

```bash
docker run --rm --gpus all -v "$PWD":/work -e HF_TOKEN stock-embed-mamba3 python /work/your_script.py
```

Feed `make_features` at least 20 trading days of bars before the window you want and
slice the lead-in off, so the volume channel starts warm as it did in training. The
contract explains why, and covers timestamp conventions, yfinance corporate actions and
vendor-adjusted bars.

## Where to get bars

No data ships with the model or this repo. The helpers take unadjusted 1-minute bars for
one security at a time and produce the model's input; give them at least 20 trading days
of lead-in before the window you embed. Sources checked in September 2026:

| Source | Cost | 1-minute history | Notes |
|---|---|---|---|
| [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance) | free, no key | last 30 calendar days, 8 days per request | pass `auto_adjust=False`; prices are still split-adjusted, so apply `Ticker.dividends` only; no volume on extended-hours bars. Enough for the examples below, not for a backfill |
| [Alpaca](https://docs.alpaca.markets/us/reference/stockbars) | free with an account | since 2016 | raw prices by default; IEX feed on the free plan, so volume is a fraction of the tape |
| [Polygon.io, now Massive](https://massive.com/docs/rest/stocks/aggregates/custom-bars) | free tier, 5 calls per minute | 2 years on the free tier | pass `adjusted=false`; timestamps are UTC milliseconds |
| [EODHD](https://eodhd.com/financial-apis/intraday-historical-data-api) | paid | US since 2004, 120 days per request | extended hours with real volume; the training data came from here |
| [Databento](https://databento.com/docs/venues-and-datasets/equs-mini) | usage based | US equities from 2023 | adjustment factors from the same client |

Whatever the source, keep prices unadjusted and apply splits and dividends with
`adjust_ohlcv`, so the adjustment convention matches training. `preprocessing_contract.md`
documents the convention, the timestamp rules and the yfinance pitfalls. `examples/`
shows the whole path on free Yahoo data.

## Examples

`examples/` goes from free Yahoo data to a clustering plot and a sector classifier on a
66-ticker, 11-sector universe. Four scripts, run in order:

```bash
mkdir -p examples/out
run() { docker run --rm --gpus all -v "$PWD/examples/out":/workspace/examples/out -e HF_TOKEN stock-embed-mamba3 "$@"; }
run python examples/fetch_yfinance.py     # 1-minute bars and corporate actions, about 20 sessions per ticker, about 5 minutes
run python examples/embed_universe.py     # one 256-dim embedding per ticker, plus a summary-statistics baseline
run python examples/cluster_plot.py       # PCA scatter by sector next to k-means clusters
run python examples/sector_knn.py         # leave-one-out k-NN sector accuracy vs baseline and chance
```

Outputs land in `examples/out` on the host. `-e HF_TOKEN` forwards a token exported in
your shell and is needed only while the Hugging Face repo is private.

On Windows, run these from WSL or PowerShell, or prefix each `docker` command in Git Bash
with `MSYS_NO_PATHCONV=1`; without it Git Bash rewrites the `/workspace/...` mount target,
the scripts write into the container instead of `examples/out`, and nothing appears on the
host. In PowerShell pass the token as `-e HF_TOKEN=$env:HF_TOKEN`.

On the authors' run the embedding placed a stock's nearest neighbour in the same sector
23% of the time against 8% chance, and the summary-statistics baseline did better at
every k. `examples/README.md` has the full numbers, the plot and the caveats of the
30-day Yahoo window.

![clusters](examples/clusters.png)

## Disclaimer

Not investment advice. Released for research and educational purposes only. Outputs are
embeddings and reconstructions of derived features, not return forecasts, price targets
or trading signals. In the authors' own evaluation this model carried no cross-sectional
return signal and produced no strategy that was profitable after transaction costs in the
authors' backtests. Out of scope: live or
automated trading, portfolio construction, any routing of outputs to order execution.

## License

Apache-2.0 for the code here and for the weights on Hugging Face.
