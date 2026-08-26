"""
KernelSHAP for the DRT+MCD strokework model. Self-contained: no other custom imports.

Two roles in one file.

1. Training hook -- call once per LOO fold, after training:

       from shap_analysis import run_shap_for_fold
       run_shap_for_fold(G, F1, F2, X_background=..., X_explain=..., ...)

   Writes one .npz per (target, fold) into --shap_dir.

2. Report CLI -- run after all five folds have finished:

       python shap_analysis.py --dir shap_dumps --config mcd1drt0st0

   Prints the importance table and cross-fold Kendall tau, writes CSVs and PNGs.

Why KernelSHAP and not DeepSHAP/GradientSHAP: DynamicLinear computes
(1 + lam(x)) * W0 x plus sum_k pi_k(x) Phi_k x -- both terms are multiplicative
gates. DeepSHAP's rescale rule misattributes across multiplicative interactions;
it returns numbers, but not Shapley values of the composed function. With 14
features, exact enumeration of all 2**14 coalitions is affordable anyway, so
these are exact Shapley values rather than a regression approximation.

Requires: shap, scipy, pandas, matplotlib
    pip install shap scipy pandas matplotlib
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

# ---------------------------------------------------------------------------
# part 1: the training hook
# ---------------------------------------------------------------------------

# targets whose values scale linearly with Y_std, so sv * Y_std -> original units
_LINEAR_TARGETS = {"mean", "head1", "head2", "disc_abs"}


def _make_predict_fn(G, F1, F2, device, target="mean", batch_size=8192):
    """
    Wrap the three modules as f(np.ndarray[N, D]) -> np.ndarray[N].

    target:
        "mean"          -> (F1 + F2) / 2        the ensemble evaluate() scores
        "head1"/"head2" -> a single head
        "disc_abs"      -> |F1 - F2|            linear in Y, rescalable
        "disc_bounded"  -> 1 - exp(-|F1 - F2|)  matches discrepancy_loss exactly

    disc_bounded, not disc_abs, is what Step B/C actually optimises. Shapley
    values are NOT invariant under monotone transforms, so explaining |d| would
    explain a different function than the training objective.
    """
    import torch

    def f(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float32))
        out = np.empty(x.shape[0], dtype=np.float64)
        with torch.no_grad():
            for i in range(0, x.shape[0], batch_size):
                xb = torch.from_numpy(x[i : i + batch_size]).to(device)
                feat = G(xb)
                p1, p2 = F1(feat), F2(feat)

                if target == "mean":
                    yb = 0.5 * (p1 + p2)
                elif target == "head1":
                    yb = p1
                elif target == "head2":
                    yb = p2
                elif target == "disc_abs":
                    yb = (p1 - p2).abs()
                elif target == "disc_bounded":
                    yb = 1.0 - torch.exp(-(p1 - p2).abs())
                else:
                    raise ValueError(f"unknown target {target!r}")

                out[i : i + batch_size] = yb.reshape(-1).cpu().numpy().astype(np.float64)
        return out

    return f


def run_shap_for_fold(
    G,
    F1,
    F2,
    X_background,
    X_explain,
    feature_names,
    out_dir,
    fold,
    seed,
    config,
    device,
    mlflow=None,
    targets=("mean", "disc_bounded"),
    n_background=25,
    n_explain=100,
    y_std=None,
):
    """
    X_background and X_explain must already be in this fold's normalizer space
    (normalizer.transform_X output). Both come out of split_domain(), so that
    holds by construction if you pass trg_train_X / trg_val_X.

    Cost is n_coalitions * n_background rows per explained instance, i.e.
    16382 * 25 ~ 410k forward rows per beat. Lower n_background before
    lowering exactness if it drags.

    Values are stored in normalized-Y units. Y_std differs per fold (the
    normalizer is fit on that fold's source pool), so cross-fold RANK
    comparisons are valid as stored, but MAGNITUDE comparisons need y_std.
    """
    import shap

    os.makedirs(out_dir, exist_ok=True)

    # BatchNorm1d in DynamicBackbone must be in eval mode: in train mode the
    # explainer's synthetic coalition batches would update running stats and
    # leak batch composition into the attributions.
    modes = [G.training, F1.training, F2.training]
    G.eval(); F1.eval(); F2.eval()

    results = {}
    try:
        rng = np.random.default_rng(seed)
        n_ex = min(n_explain, len(X_explain))
        idx = rng.choice(len(X_explain), size=n_ex, replace=False)
        X_ex = np.asarray(X_explain, dtype=np.float64)[idx]

        X_bg = np.asarray(X_background, dtype=np.float64)
        bg = X_bg if len(X_bg) <= n_background else shap.kmeans(X_bg, n_background)
        n_bg = bg.data.shape[0] if hasattr(bg, "data") else len(bg)

        D = X_ex.shape[1]
        nsamples = 2 ** D  # shap clamps to 2**D - 2 -> exact enumeration

        for target in targets:
            f = _make_predict_fn(G, F1, F2, device, target=target)

            explainer = shap.KernelExplainer(f, bg)
            sv = np.asarray(
                explainer.shap_values(X_ex, nsamples=nsamples, l1_reg=0.0, silent=True)
            )
            if sv.ndim == 3:  # (N, D, 1) in shap >= 0.46
                sv = sv[..., 0]
            ev = float(np.asarray(explainer.expected_value).ravel()[0])

            # Efficiency axiom: sum(phi) + E[f] == f(x). If this fails the
            # wrapper is wrong (wrong head, scaler mismatch, BN still training)
            # and nothing below should be interpreted.
            pred = f(X_ex).ravel()
            err = np.abs(pred - (sv.sum(axis=1) + ev))
            rel = float(err.max() / max(np.std(pred), 1e-12))
            ok = rel < 1e-3

            path = os.path.join(out_dir, f"shap_{target}_{fold}_seed{seed}_{config}.npz")
            np.savez(
                path,
                shap_values=sv,
                expected_value=ev,
                X=X_ex,
                index=idx,
                feature_names=np.array(feature_names),
                y_std=np.array(y_std if y_std is not None else np.nan),
                linear_in_y=np.array(target in _LINEAR_TARGETS),
                fold=np.array(str(fold)),
                seed=np.array(str(seed)),
                config=np.array(str(config)),
            )
            results[target] = (sv, ev, rel, path)

            if mlflow is not None:
                mlflow.log_metric(f"shap/{target}/local_acc_rel_err", rel)
                for name, val in zip(feature_names, np.abs(sv).mean(axis=0)):
                    mlflow.log_metric(f"shap/{target}/imp/{name}", float(val))
                mlflow.log_artifact(path)

            print(f"  [SHAP:{target}] n={n_ex} bg={n_bg} rel_err={rel:.2e}"
                  f"{'' if ok else '   <-- LOCAL ACCURACY FAILED'}")
            if not ok:
                print("    efficiency axiom violated: wrong head, scaler mismatch, or "
                      "BatchNorm still in train mode. Do not interpret these values.")
    finally:
        G.train(modes[0]); F1.train(modes[1]); F2.train(modes[2])

    return results


# ---------------------------------------------------------------------------
# part 2: the report CLI
# ---------------------------------------------------------------------------


def _load_folds(directory, target, config=None, seed=None):
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, f"shap_{target}_*.npz"))):
        d = np.load(path, allow_pickle=True)
        cfg, sd = str(d["config"]), str(d["seed"])
        if (config and cfg != config) or (seed and sd != str(seed)):
            continue
        out[str(d["fold"])] = {
            "sv": d["shap_values"],
            "X": d["X"],
            "ev": float(d["expected_value"]),
            "names": [str(n) for n in d["feature_names"]],
            "y_std": float(d["y_std"]),
            "linear": bool(d["linear_in_y"]),
            "config": cfg,
        }
    return out


def report(directory="shap_dumps", target="mean", config=None, seed=None, physical=False):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")  # headless: Bjarne-PC over SSH has no display
    import matplotlib.pyplot as plt
    from scipy.stats import kendalltau

    folds = _load_folds(directory, target, config, seed)
    if not folds:
        raise SystemExit(f"no .npz matched in {directory}/ for target={target} config={config}")

    configs = {v["config"] for v in folds.values()}
    if len(configs) > 1:
        raise SystemExit(f"mixed configs {configs} -- pass --config, else tau compares different models")

    names = folds[next(iter(folds))]["names"]
    out_dir = os.path.join(directory, f"report_{target}_{config or 'all'}")
    os.makedirs(out_dir, exist_ok=True)

    if physical and not all(f["linear"] for f in folds.values()):
        print("  --physical ignored: this target is nonlinear in Y, values stay unitless")
        physical = False
    unit = "strokework units" if physical else "normalized units"

    # importance table
    imps = {a: np.abs(f["sv"] * (f["y_std"] if physical else 1.0)).mean(axis=0)
            for a, f in folds.items()}
    df = pd.DataFrame(imps, index=names)
    df["mean"] = df[list(folds)].mean(axis=1)
    df["std"] = df[list(folds)].std(axis=1, ddof=1) if len(folds) > 1 else 0.0
    df = df.sort_values("mean", ascending=False)
    df.to_csv(os.path.join(out_dir, "importance.csv"))

    print(f"\nmean |SHAP| per feature   ({unit}, target={target}, n_folds={len(folds)})")
    print(df.round(4).to_string())

    # cross-fold rank stability
    if len(folds) >= 2:
        keys = list(folds)
        raw = {a: np.abs(folds[a]["sv"]).mean(axis=0) for a in keys}
        tau = pd.DataFrame(np.eye(len(keys)), index=keys, columns=keys)
        vals = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                t = float(kendalltau(raw[keys[i]], raw[keys[j]])[0])
                tau.loc[keys[i], keys[j]] = tau.loc[keys[j], keys[i]] = t
                vals.append(t)
        tau.to_csv(os.path.join(out_dir, "tau_matrix.csv"))
        print("\npairwise Kendall tau (agreement of importance ranking across folds)")
        print(tau.round(3).to_string())
        m = float(np.mean(vals))
        print(f"\nmean tau {m:.3f}   min tau {min(vals):.3f}")
        if m > 0.7:
            print("  -> folds agree: the importance ranking is animal-invariant")
        elif m > 0.4:
            print("  -> partial agreement: report the spread, not just the LOO mean")
        else:
            print("  -> folds disagree: each learned a different function, "
                  "the LOO mean is hiding heterogeneity")
    else:
        print("\nonly one fold present -- skipping tau (needs >= 2)")

    # plots
    fig, ax = plt.subplots(figsize=(7, 5))
    top = df.head(14)
    ax.barh(top.index[::-1], top["mean"][::-1], xerr=top["std"][::-1],
            color="#4C72B0", ecolor="#888", capsize=3)
    ax.set_xlabel(f"mean |SHAP|  ({unit})")
    ax.set_title(f"feature importance, target={target}, {len(folds)} LOO folds")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "importance.png"), dpi=150)
    plt.close(fig)

    try:
        import shap as _shap
        for animal, f in folds.items():
            plt.figure()
            _shap.summary_plot(f["sv"], f["X"], feature_names=names,
                               show=False, max_display=14)
            plt.title(f"{animal}  ({target})")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"beeswarm_{animal}.png"), dpi=150)
            plt.close()
    except Exception as e:
        print(f"  beeswarm skipped: {e}")

    print(f"\nwrote {out_dir}/")
    for fn in sorted(os.listdir(out_dir)):
        print(f"  {fn}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SHAP report over LOO folds")
    ap.add_argument("--dir", default="shap_dumps")
    ap.add_argument("--target", default="mean",
                    help="mean | disc_bounded | disc_abs | head1 | head2")
    ap.add_argument("--config", default=None,
                    help="e.g. mcd1drt0st0 -- required if several configs are present")
    ap.add_argument("--seed", default=None)
    ap.add_argument("--physical", action="store_true",
                    help="rescale to strokework units using each fold's Y_std")
    a = ap.parse_args()
    report(a.dir, a.target, a.config, a.seed, a.physical)