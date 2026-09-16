"""Nearest-neighbour sector classification from the universe embeddings.

Leave-one-out: each ticker is assigned the majority sector of its k nearest other
tickers by cosine distance on standardized coordinates (each dimension z-scored across
tickers first); a tied vote goes to the nearest neighbour's sector. Reports accuracy for
the embeddings, for the 18-dimensional summary-statistics baseline, and for chance (the
same procedure on shuffled sector labels, averaged over many shuffles). This is a
description of what the embedding encodes, not a trading signal.

    python examples/sector_knn.py [--out examples/out] [--k 1 3 5]
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def cosine_distances(x: np.ndarray) -> np.ndarray:
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    d = 1.0 - unit @ unit.T
    np.fill_diagonal(d, np.inf)  # leave-one-out: never your own neighbour
    return d


def knn_accuracy(d: np.ndarray, labels: np.ndarray, k: int) -> float:
    neighbours = np.argsort(d, axis=1)[:, :k]
    hits = 0
    for i, row in enumerate(neighbours):
        votes, counts = np.unique(labels[row], return_counts=True)
        top = votes[counts == counts.max()]
        predicted = top[0] if len(top) == 1 else labels[row][0]
        hits += predicted == labels[i]
    return hits / len(labels)


def chance_accuracy(d: np.ndarray, labels: np.ndarray, k: int, n_shuffles: int = 500, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    return float(np.mean([knn_accuracy(d, rng.permutation(labels), k) for _ in range(n_shuffles)]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "out")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    args = ap.parse_args()

    data = np.load(args.out / "embeddings.npz")
    sectors = data["sectors"]
    d_emb, d_base = cosine_distances(data["embeddings"]), cosine_distances(data["baseline"])
    print(f"{len(sectors)} tickers, {len(set(sectors))} sectors, leave-one-out k-NN sector accuracy")
    print(f"{'k':>3}  {'embeddings':>10}  {'baseline':>10}  {'chance':>10}")
    for k in args.k:
        emb, base, chance = (
            knn_accuracy(d_emb, sectors, k),
            knn_accuracy(d_base, sectors, k),
            chance_accuracy(d_emb, sectors, k),
        )
        print(f"{k:>3}  {emb:>10.1%}  {base:>10.1%}  {chance:>10.1%}")


if __name__ == "__main__":
    main()
