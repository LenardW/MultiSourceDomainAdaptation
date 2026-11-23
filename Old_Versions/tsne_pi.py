"""Fig. 4 / Fig. 5 style t-SNE of the DRT routing vectors Pi = {pi_i^l(x)}.

Usage
-----
    python tsne_pi.py pi_dumps/pi_anim1_seed100.npz
    python tsne_pi.py "pi_dumps/*.npz" --metric cosine --out figs
    python tsne_pi.py pi_dumps/pi_anim3_seed100.npz --no-per_layer

Expects the .npz written by the dump block in train():
    pi      [N, L*K]   routing coefficients, layer blocks concatenated
    domain  [N]        0 = held-out target, 1..D = source animals  (optional)
    names   [D+1]      display names in domain-id order             (optional)
    y       [N]        strokework in original units                 (optional)
    K       scalar     bases per layer                              (optional, default 4)

Only `pi` is required; the domain panel is the interesting one, so a file
without `domain` still plots but says much less.

--per_layer (on by default) embeds each layer's K-dim block separately, which
is the comparison the paper's Fig. 5 makes: routing in lower layers carries
more domain information than in higher layers. --no-per_layer embeds the whole
concatenated Pi instead, which is Fig. 4.
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load(path):
    z = np.load(path)

    if "pi" in z:
        pi = z["pi"].astype(np.float64)
        K = int(z["K"]) if "K" in z else 4
        dom = z["domain"].astype(int) if "domain" in z else None
        y = z["y"].astype(np.float64) if "y" in z else None
        names = [str(s) for s in z["names"]] if "names" in z else None
        return pi, dom, y, names, K

    if "pi_target" in z:
        # Older dump layout: target and source blocks stored separately with no
        # domain array. Sources cannot be told apart, so they share one id.
        pt = z["pi_target"].astype(np.float64)
        ps = z["pi_source"].astype(np.float64) if "pi_source" in z else None
        K = int(z["K"]) if "K" in z else 4
        if ps is None:
            return pt, np.zeros(len(pt), int), None, ["target"], K
        pi = np.concatenate([pt, ps])
        dom = np.concatenate([np.zeros(len(pt), int), np.ones(len(ps), int)])
        return pi, dom, None, ["target", "sources (pooled)"], K

    raise KeyError(f"{path}: no 'pi' or 'pi_target' array in the archive "
                   f"(found {list(z.keys())})")


def subsample(dom, n, cap, seed=0):
    """Cap the point count, stratified by domain when we know it.

    t-SNE is O(n^2); an unstratified cap would silently under-represent the
    smaller animals, which is exactly what the figure is about.
    """
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    if dom is None:
        return np.sort(rng.choice(n, cap, replace=False))

    groups = np.unique(dom)
    per = max(cap // len(groups), 1)
    keep = []
    for g in groups:
        idx = np.flatnonzero(dom == g)
        keep.append(idx if len(idx) <= per
                    else rng.choice(idx, per, replace=False))
    return np.sort(np.concatenate(keep))


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

def entropy_per_layer(pi, K):
    """Mean normalised entropy of each layer's pi block.

    1.0 means the router emits a near-uniform blend for every sample, so the
    dynamic path carries no sample-dependent information and any structure in
    the embedding is noise. 0.0 means hard one-hot routing.
    """
    out = []
    for l in range(pi.shape[1] // K):
        b = np.clip(pi[:, l * K:(l + 1) * K], 1e-12, 1.0)
        out.append(float(-(b * np.log(b)).sum(1).mean() / np.log(K)))
    return out


def spread_per_layer(pi, K):
    """Std of pi across samples, averaged over the K coordinates.

    Complements entropy: a router could be confident yet emit the *same*
    confident vector for everyone (low entropy, zero spread), which is still a
    static layer. Only spread > 0 means routing is sample-dependent.
    """
    out = []
    for l in range(pi.shape[1] // K):
        out.append(float(pi[:, l * K:(l + 1) * K].std(axis=0).mean()))
    return out


def domain_separation(X, dom):
    """Fraction of points whose nearest neighbour shares its domain.

    ~1/n_domains means no clustering; near 1.0 means clean separation. Computed
    on whatever coordinates are passed in -- on the raw pi it is a property of
    the routing itself, on the 2D embedding it also reflects how t-SNE laid
    things out. Report the raw one when the two disagree.

    A number to put next to the figure, so the claim is not purely visual.
    """
    if dom is None or len(np.unique(dom)) < 2:
        return None
    d = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    return float((dom[d.argmin(1)] == dom).mean())


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def embed(X, args):
    perp = min(args.perplexity, max(5.0, (len(X) - 1) / 3.0))
    return TSNE(n_components=2, perplexity=perp, metric=args.metric,
                init="pca", learning_rate="auto",
                random_state=args.seed).fit_transform(X)


def scatter_domains(ax, emb, dom, names):
    cmap = plt.get_cmap("tab10")
    for d in np.unique(dom):
        m = dom == d
        lbl = (names[d] if names is not None and d < len(names)
               else f"domain {d}")
        # domain 0 is the held-out target, drawn as crosses
        ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.6, color=cmap(d % 10),
                   marker="x" if d == 0 else "o", linewidths=0.6,
                   label=lbl + (" (target)" if d == 0 else ""))
    ax.legend(markerscale=2, fontsize=8, frameon=False)


def plot_one(path, args):
    pi, dom, y, names, K = load(path)

    sel = subsample(dom, len(pi), args.max_points, args.seed)
    pi = pi[sel]
    dom = dom[sel] if dom is not None else None
    y = y[sel] if y is not None else None

    L = pi.shape[1] // K
    ents = entropy_per_layer(pi, K)
    spreads = spread_per_layer(pi, K)

    print(f"{os.path.basename(path)}: N={len(pi)} dim={pi.shape[1]} "
          f"K={K} layers={L}")
    print("  normalised pi entropy per layer: "
          + ", ".join(f"{e:.3f}" for e in ents))
    print("  pi spread per layer:             "
          + ", ".join(f"{s:.3f}" for s in spreads))
    if min(ents) > 0.99 or max(spreads) < 1e-3:
        print("  WARNING: the router is effectively static -- near-uniform "
              "and/or identical across samples. The embedding below is noise.")

    # ---- which blocks get their own row ----
    if args.per_layer:
        blocks = [(f"layer {l + 1}", pi[:, l * K:(l + 1) * K])
                  for l in range(L)]
    else:
        blocks = [("$\\Pi$ (all layers)", pi)]

    chance = 1.0 / len(np.unique(dom)) if dom is not None else None

    rows = []
    for title, X in blocks:
        emb = embed(X, args)
        sep_raw = domain_separation(X, dom)
        sep_emb = domain_separation(emb, dom)
        rows.append((title, emb, sep_raw, sep_emb))
        if sep_raw is not None:
            print(f"  {title}: same-domain NN  raw pi {sep_raw:.3f} | "
                  f"embedding {sep_emb:.3f}  (chance {chance:.3f})")

    # ---- figure: one row per block, columns = [by animal, by strokework] ----
    n_cols = 1 + (y is not None)
    fig, axes = plt.subplots(len(rows), n_cols,
                             figsize=(5.4 * n_cols, 5.2 * len(rows)),
                             squeeze=False)

    for r, (title, emb, sep_raw, sep_emb) in enumerate(rows):
        ax = axes[r][0]
        if dom is not None:
            scatter_domains(ax, emb, dom, names)
            ax.set_title(f"{title} — by animal\n"
                         f"same-domain NN: {sep_raw:.2f} raw / "
                         f"{sep_emb:.2f} embedded  (chance {chance:.2f})",
                         fontsize=10)
        else:
            ax.scatter(emb[:, 0], emb[:, 1], s=6, alpha=0.6, color="0.3")
            ax.set_title(title, fontsize=10)

        if y is not None:
            ax = axes[r][1]
            sc = ax.scatter(emb[:, 0], emb[:, 1], s=6, alpha=0.7, c=y,
                            cmap="viridis")
            fig.colorbar(sc, ax=ax, label="strokework")
            ax.set_title(f"{title} — by strokework", fontsize=10)

    for row in axes:
        for a in row:
            a.set_xticks([])
            a.set_yticks([])

    tag = os.path.basename(path).replace("pi_", "").replace(".npz", "")
    fig.suptitle(f"DRT routing t-SNE — {tag}  (metric={args.metric})", y=0.99)
    fig.tight_layout()

    dest = os.path.join(args.out, f"tsne_{tag}"
                        + ("_perlayer" if args.per_layer else "_all")
                        + f"_{args.metric}.png")
    fig.savefig(dest, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {dest}")


# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(
        description="t-SNE of DRT routing coefficients")
    p.add_argument("paths", nargs="*", default=["pi_dumps/*.npz"],
                   help="npz file(s) or a glob (default: pi_dumps/*.npz)")
    p.add_argument("--out", default="figs")
    p.add_argument("--per_layer", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="one row per layer block (Fig. 5) instead of the "
                        "concatenated Pi (Fig. 4); --no-per_layer for the "
                        "latter")
    p.add_argument("--both", action="store_true",
                   help="write both the per-layer and the concatenated figure")
    p.add_argument("--perplexity", type=float, default=30.0,
                   help="lowered automatically if a file has too few points")
    p.add_argument("--metric", default="cosine",
                   choices=["euclidean", "cosine"],
                   help="pi lies on the simplex, so cosine usually separates "
                        "it better than euclidean")
    p.add_argument("--max_points", type=int, default=4000,
                   help="stratified by domain; t-SNE is O(n^2)")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    files = []
    for pat in a.paths:
        files.extend(sorted(glob.glob(pat)) or [pat])
    if not files:
        raise SystemExit(f"no files matched {a.paths}")

    os.makedirs(a.out, exist_ok=True)
    modes = [True, False] if a.both else [a.per_layer]
    for f in files:
        for m in modes:
            a.per_layer = m
            plot_one(f, a)


if __name__ == "__main__":
    main()