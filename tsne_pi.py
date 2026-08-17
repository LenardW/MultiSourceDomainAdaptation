"""Fig. 4 style t-SNE of the DRT routing vectors Pi = {pi_i^l(x)}.

Usage
-----
    python tsne_pi.py pi_dumps/pi_anim1_subspace_seed100.npz
    python tsne_pi.py "pi_dumps/*.npz" --metric cosine --out figs/

Reads the .npz produced by the dump block in train(). Expects `pi` [N, L*K]
and, if present, `domain` [N], `y` [N], `names` [D], `K`. Files with only `pi`
still plot -- just without the domain colouring, which is the interesting panel.
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def load(path):
    z = np.load(path, allow_pickle=True)
    pi = z["pi"].astype(np.float64)
    K = int(z["K"]) if "K" in z else 4
    dom = z["domain"] if "domain" in z else None
    y = z["y"] if "y" in z else None
    names = [str(s) for s in z["names"]] if "names" in z else None
    return pi, dom, y, names, K


def subsample(n, cap, seed=0):
    if n <= cap:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, cap, replace=False)


def entropy(pi, K):
    """Mean normalised entropy per layer block. 1.0 = router is inert."""
    L = pi.shape[1] // K
    out = []
    for l in range(L):
        b = np.clip(pi[:, l * K:(l + 1) * K], 1e-8, 1.0)
        out.append(float(-(b * np.log(b)).sum(1).mean() / np.log(K)))
    return out


def plot_one(path, args):
    pi, dom, y, names, K = load(path)
    sel = subsample(len(pi), args.max_points, args.seed)
    pi, dom, y = pi[sel], (dom[sel] if dom is not None else None), \
        (y[sel] if y is not None else None)

    ents = entropy(pi, K)
    print(f"{os.path.basename(path)}: N={len(pi)} dim={pi.shape[1]} K={K}")
    print("  per-layer normalised pi entropy: "
          + ", ".join(f"{e:.3f}" for e in ents))
    if min(ents) > 0.99:
        print("  WARNING: router is emitting a near-uniform blend -- the "
              "dynamic path carries no sample-dependent information, so the "
              "t-SNE below is noise.")

    emb = TSNE(n_components=2, perplexity=args.perplexity,
               metric=args.metric, init="pca", learning_rate="auto",
               random_state=args.seed).fit_transform(pi)

    n_panels = 1 + (y is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.4),
                             squeeze=False)
    ax = axes[0][0]

    if dom is not None:
        cmap = plt.get_cmap("tab10")
        for d in np.unique(dom):
            m = dom == d
            lbl = names[d] if names is not None else f"domain {d}"
            # domain 0 is the held-out target
            ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.6,
                       color=cmap(d % 10),
                       marker="x" if d == 0 else "o",
                       label=lbl + (" (target)" if d == 0 else ""))
        ax.legend(markerscale=2, fontsize=8, frameon=False)
    else:
        ax.scatter(emb[:, 0], emb[:, 1], s=6, alpha=0.6, color="0.3")
    ax.set_title("routing vector $\\Pi$, coloured by animal")

    if y is not None:
        ax = axes[0][1]
        sc = ax.scatter(emb[:, 0], emb[:, 1], s=6, alpha=0.7, c=y,
                        cmap="viridis")
        fig.colorbar(sc, ax=ax, label="strokework")
        ax.set_title("same embedding, coloured by target")

    tag = os.path.basename(path).replace("pi_", "").replace(".npz", "")
    for a in axes[0]:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"DRT routing t-SNE -- {tag}", y=0.98)
    fig.tight_layout()

    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, f"tsne_{tag}.png")
    fig.savefig(dest, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {dest}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="npz file(s) or a glob")
    p.add_argument("--out", default="figs")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--metric", default="euclidean",
                   choices=["euclidean", "cosine"],
                   help="cosine often separates simplex-valued pi better")
    p.add_argument("--max_points", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    files = []
    for pat in a.paths:
        files.extend(sorted(glob.glob(pat)) or [pat])
    for f in files:
        plot_one(f, a)


if __name__ == "__main__":
    main()