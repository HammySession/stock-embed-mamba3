"""Cluster the universe embeddings and draw them in two dimensions.

Left panel: PCA of the standardized embeddings, one point per ticker, colored by GICS
sector. Right panel: the same points colored by k-means cluster, with k equal to the
number of sectors. The title reports the adjusted Rand index (ARI) between clusters and
sectors: 1.0 would mean the clusters are the sectors, 0.0 means chance agreement. The
same number for the 18-dimensional summary-statistics baseline is printed alongside.

    python examples/cluster_plot.py [--out examples/out]
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two leading principal components and their explained-variance fractions."""
    centered = x - x.mean(axis=0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    explained = s**2 / (s**2).sum()
    return centered @ vt[:2].T, explained[:2]


def cluster_ari(x: np.ndarray, sectors: np.ndarray, seed: int = 0) -> tuple[np.ndarray, float]:
    k = len(np.unique(sectors))
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)
    return labels, adjusted_rand_score(sectors, labels)


def scatter(ax, xy: np.ndarray, groups: np.ndarray, tickers: np.ndarray, title: str) -> None:
    names = sorted(set(groups))
    for i, name in enumerate(names):
        m = groups == name
        ax.scatter(xy[m, 0], xy[m, 1], color=plt.cm.tab20(i % 20), label=str(name), s=36)
    for (px, py), t in zip(xy, tickers, strict=True):
        ax.annotate(t, (px, py), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=6, loc="best", ncol=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "out")
    args = ap.parse_args()

    data = np.load(args.out / "embeddings.npz")
    tickers, sectors = data["tickers"], data["sectors"]
    emb = standardize(data["embeddings"])
    xy, explained = pca_2d(emb)
    clusters, ari = cluster_ari(emb, sectors)
    _, ari_baseline = cluster_ari(standardize(data["baseline"]), sectors)
    print(f"k-means vs sector ARI: embeddings {ari:.3f}, summary-statistics baseline {ari_baseline:.3f}")
    print(f"PCA explained variance: PC1 {explained[0]:.1%}, PC2 {explained[1]:.1%}")

    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 7))
    scatter(left, xy, sectors, tickers, "stock-embed-mamba3 embeddings, PCA, colored by sector")
    scatter(right, xy, clusters, tickers, f"k-means clusters (k={len(set(sectors))}), ARI vs sector {ari:.2f}")
    fig.tight_layout()
    fig.savefig(args.out / "clusters.png", dpi=130)
    print(f"wrote {args.out / 'clusters.png'}")


if __name__ == "__main__":
    main()
