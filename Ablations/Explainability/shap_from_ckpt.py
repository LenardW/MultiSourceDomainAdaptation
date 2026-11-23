"""
SHAP values for checkpoints of msda_mlp.py, msda_conv.py, msda_conv_ctx.py and
msda_conv_interv.py (saved with --save_dir).

For each checkpoint the script
  1. detects the architecture from the stored fields and imports the matching training
     script (it must lie in the same folder as this file),
  2. rebuilds the fold with the stored arguments (same split, normalisation and windows),
  3. rebuilds the model, loads the weights and, for --target_bn target runs, recomputes the
     BatchNorm statistics on the target's adaptation split as during evaluation,
  4. explains the predicted stroke work (mean of all heads, original unit) for a set of
     beats of the target animal.

Players (one per feature)
  Every feature is one player, whatever the input shape: in the convolutional versions all
  window positions of a feature form one player; in the context versions the routing
  context of a feature joins it (--context joined, default) or forms a player of its own
  (--context separate: 'ctx:<feature>'). This gives 14 (or 28) values per beat for every
  architecture, so the attributions are directly comparable.

Methods (--method)
  kernel    KernelSHAP (Lundberg and Lee 2017) on the players. A coalition keeps the values
            of the explained beat for its players and takes the remaining ones from each
            background sample (interventional Shapley values); the prediction is averaged
            over the background. Model-agnostic, so the same for all models.
            --nsamples: number of coalitions (default 2*M+2048, as shap's 'auto';
            'exact' enumerates all 2^M coalitions). No L1 regularisation.
  deep      DeepSHAP (shap.DeepExplainer), summed over window positions per player. Exact
            rules only for networks without DRT; the DRT layers (multiplicative gating,
            softmax) are passed through with gradients, so for DRT models it is an
            approximation. The additivity error is reported.
  gradient  Expected gradients (shap.GradientExplainer), summed per player.

Background (--background)
  target    random beats of the target's adaptation split (variation within the animal)
  source    random beats of the pooled source animals (difference to the sources)

Explained beats (--explain_split)
  test      the test split of the target (default)
  adapt     the adaptation split
  all       both, i.e. the whole recording; use this for plots over time. For the oracle
            (trained and tested on the same animal) the adaptation split is its training data.
  --n_explain beats are taken evenly spaced in time from the chosen split.

Output: one CSV per checkpoint in --out_dir with one row per explained beat:
  arch, config, target, seed, method, background, beat_row (row in the full recording, the
  same for all architectures), y_true, y_pred, base_value, additivity_error, intervention,
  intervention_phase, condition (baseline = phase 1, else I<code>; from I_<animal>.csv if it
  exists), then one column per player. base_value + sum(players) = y_pred (exactly for
  kernel, approximately for deep and gradient). shap_aggregate.py plots and aggregates them.

Usage:
  python shap_from_ckpt.py ckpt_conv/*mcd_drt*.pt --method kernel --feature_names names.txt
  python shap_from_ckpt.py ckpt_mlp/*source_only*anim3*.pt --method deep --background source
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import re
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Loading
# =============================================================================

def detect_script(ck):
    if "route_sup" in ck:
        return "msda_conv_interv"
    if "ctx_len" in ck:
        return "msda_conv_ctx"
    if "window" in ck:
        return "msda_conv"
    return "msda_mlp"


def arch_label(name, ck):
    """Architecture name for the output; context/intervention models with kernel_size 1 and
    window 1 (the MLP versions) get the suffix _mlp, so they are not mixed up with the
    convolutional versions."""
    if name in ("msda_conv_ctx", "msda_conv_interv") and ck.get("kernel_size") == 1 \
            and ck.get("window") == 1:
        return name + "_mlp"
    return name


_MODULES: dict = {}


def import_script(name):
    if name not in _MODULES:
        path = os.path.join(HERE, f"{name}.py")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found; keep this file next to the training scripts")
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _MODULES[name] = mod
    return _MODULES[name]


def build_model(mod, name, ck, fold, a):
    drt, n_pairs = ck["drt"], ck["n_pairs"]
    if name == "msda_mlp":
        return mod.MSDAModel(fold.src_X[0].shape[1], a.hidden_dim, n_pairs, a.K, drt,
                             a.basis_scale)
    chans = mod.parse_channels(a.channels)
    if name == "msda_conv":
        return mod.MSDAModel(fold.src_X[0].shape[1], chans, a.kernel_size, a.phi_kernel,
                             n_pairs, a.K, drt, a.basis_scale)
    ctx_dim = fold.src_X[0].shape[1] - fold.n_feat
    if name == "msda_conv_ctx":
        return mod.MSDAModel(fold.n_feat, chans, a.kernel_size, a.phi_kernel, n_pairs, a.K,
                             drt, a.basis_scale, ctx_dim=ctx_dim)
    route = a.route_sup if a.route_weight > 0 else "none"
    K = ck.get("K", fold.n_classes if (route == "expert" and drt) else a.K)
    return mod.MSDAModel(fold.n_feat, chans, a.kernel_size, a.phi_kernel, n_pairs, K, drt,
                         a.basis_scale, ctx_dim=ctx_dim, n_classes=fold.n_classes,
                         route_sup=route, aux=a.aux_weight > 0)


def load(path, data_dir, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    name = detect_script(ck)
    mod = import_script(name)
    a = argparse.Namespace(**ck["args"])
    if data_dir:
        a.data_dir = data_dir
    oracle = ck["config"] == "oracle"
    sources = [] if oracle else list(ck["sources"])
    fold = mod.build_fold(a, ck["target"], sources, oracle)
    if not np.array_equal(fold.test_idx, ck["test_idx"]):
        raise RuntimeError(f"{path}: the rebuilt test split differs from the stored one "
                           f"(different data or arguments?)")
    model = build_model(mod, name, ck, fold, a)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    if getattr(a, "target_bn", "source") == "target":
        model = mod.with_target_bn_stats(model, torch.as_tensor(fold.adapt_X, device=device))
    return ck, name, a, fold, model


class Predictor(nn.Module):
    """Predicted stroke work in its original unit: mean of all heads, [N, 1]."""

    def __init__(self, model, y_mean, y_std):
        super().__init__()
        self.model = model
        self.y_mean, self.y_std = float(y_mean), float(y_std)

    def forward(self, x):
        p = self.model.all_heads(self.model.G(x)).mean(0)
        return (p * self.y_std + self.y_mean).unsqueeze(-1)


@torch.no_grad()
def predict(pred, X, batch=8192):
    pred.eval()
    return torch.cat([pred(X[i:i + batch])[:, 0] for i in range(0, len(X), batch)])


def explain_set(fold, split):
    if split == "test":
        return fold.test_X, fold.test_y, fold.test_idx
    if split == "adapt":
        return fold.adapt_X, fold.adapt_y, fold.adapt_idx
    X = np.concatenate([fold.adapt_X, fold.test_X])
    y = np.concatenate([fold.adapt_y, fold.test_y])
    r = np.concatenate([fold.adapt_idx, fold.test_idx])
    order = np.argsort(r, kind="stable")
    return X[order], y[order], r[order]


def recording_rows_and_conditions(ck, name, a, rows, ys, args):
    """Rows in the full recording (msda_mlp.py indexes the beats left after removing
    negative stroke work) and the intervention labels of these rows, if available."""
    aid = ck["target"]
    y_raw = np.loadtxt(os.path.join(a.data_dir, f"Y_{aid}.csv"), delimiter=",",
                       ndmin=1).reshape(-1)
    if name == "msda_mlp" and getattr(a, "drop_negative", True):
        rows = np.flatnonzero(y_raw >= 0)[rows]
    if not np.allclose(y_raw[rows], ys):
        raise RuntimeError(f"{aid}: stroke work of the mapped rows does not match")
    n = len(rows)
    cond = {"intervention": np.full(n, np.nan), "intervention_phase": np.full(n, np.nan),
            "condition": np.full(n, "unknown", dtype=object)}
    path = os.path.join(a.data_dir, args.interv_pattern.format(aid=aid))
    if os.path.exists(path):
        import pandas as pd
        df = pd.read_csv(path)
        if len(df) != len(y_raw):
            raise ValueError(f"{path}: {len(df)} rows, but Y_{aid}.csv has {len(y_raw)}")
        iv = df[args.interv_col].to_numpy(dtype=float)[rows]
        ph = df[args.phase_col].to_numpy(dtype=float)[rows]
        lab = np.array([f"I{int(v)}" if np.isfinite(v) else "unknown" for v in iv], dtype=object)
        lab[ph == 1] = "baseline"
        cond = {"intervention": iv, "intervention_phase": ph, "condition": lab}
    return rows, cond


# =============================================================================
# Players
# =============================================================================

def feature_names(arg, n_feat, a):
    if not arg:
        names = None
    elif os.path.exists(arg):
        names = [l.strip() for l in open(arg) if l.strip()]
    else:
        names = [n.strip() for n in arg.split(",") if n.strip()]
    if names is None:
        return [f"f{i}" for i in range(n_feat)]
    if len(names) == n_feat:
        return names
    drop = {int(i) for i in str(getattr(a, "drop_features", "")).split(",") if i not in ("", "none")}
    kept = [n for i, n in enumerate(names) if i not in drop]
    if len(kept) != n_feat:
        raise ValueError(f"{len(names)} feature names given, but the model has {n_feat} features")
    return kept


def player_groups(n_channels, n_feat, names, context):
    """Player index of every input channel, and the player names."""
    groups = np.arange(n_channels)
    groups[:n_feat] = np.arange(n_feat)
    if n_channels > n_feat:                       # routing context (mean, or mean and SD)
        j = (np.arange(n_feat, n_channels) - n_feat) % n_feat
        groups[n_feat:] = j if context == "joined" else n_feat + j
    players = list(names)
    if n_channels > n_feat and context == "separate":
        players += [f"ctx:{n}" for n in names]
    return groups, players


def per_player(sv, groups, M):
    """Sum SHAP values over channels (and window positions) of each player: [N, M]."""
    sv = sv.reshape(sv.shape[0], sv.shape[1], -1).sum(-1)       # [N, C]
    out = np.zeros((sv.shape[0], M))
    for g in range(M):
        out[:, g] = sv[:, groups == g].sum(1)
    return out


# =============================================================================
# Explainers
# =============================================================================

def kernel_shap(pred, X, bg, groups, M, nsamples, batch_rows):
    import shap
    gidx = torch.as_tensor(groups, device=X.device)
    B = len(bg)
    extra = (1,) * (X.dim() - 2)                   # window axis of the conv inputs

    phis, bases = [], []
    for x in X:
        def f(z):
            z = torch.as_tensor(np.asarray(z), dtype=torch.bool, device=X.device)
            mask = z[:, gidx]                                     # [n, C]
            out = []
            k = max(1, batch_rows // B)
            for i in range(0, len(mask), k):
                m = mask[i:i + k].view(-1, 1, mask.shape[1], *extra)
                comp = torch.where(m, x.unsqueeze(0).unsqueeze(0), bg.unsqueeze(0))
                p = predict(pred, comp.reshape(-1, *x.shape)).view(-1, B).mean(1)
                out.append(p)
            return torch.cat(out).cpu().numpy()

        explainer = shap.KernelExplainer(f, np.zeros((1, M)))
        phi = explainer.shap_values(np.ones((1, M)), nsamples=nsamples, l1_reg=False,
                                    silent=True)
        phi = np.asarray(phi).reshape(-1)[:M]
        phis.append(phi)
        bases.append(float(np.asarray(explainer.expected_value).reshape(-1)[0]))
    return np.stack(phis), np.array(bases)


def deep_or_gradient(pred, X, bg, groups, M, method, ck):
    import shap
    if method == "deep":
        if ck["drt"]:
            print("    note: DeepSHAP passes the DRT layers through with gradients; "
                  "for this model the values are an approximation")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="unrecognized nn.Module")
            e = shap.DeepExplainer(pred, bg)
            sv = e.shap_values(X, check_additivity=False)
        base = float(np.asarray(e.expected_value).reshape(-1)[0])
    else:
        e = shap.GradientExplainer(pred, bg)
        sv = e.shap_values(X, nsamples=200)
        base = float(predict(pred, bg).mean())
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv)
    if sv.ndim == X.dim() + 1:                     # trailing output axis in newer shap
        sv = sv[..., 0]
    return per_player(sv, groups, M), np.full(len(X), base)


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+", help="checkpoint files or glob patterns")
    p.add_argument("--method", choices=["kernel", "deep", "gradient"], default="kernel")
    p.add_argument("--background", choices=["target", "source"], default="target")
    p.add_argument("--n_background", type=int, default=50)
    p.add_argument("--n_explain", type=int, default=200,
                   help="beats of the split to explain, evenly spaced in time (0 = all)")
    p.add_argument("--explain_split", choices=["test", "adapt", "all"], default="test")
    p.add_argument("--interv_pattern", default="I_{aid}.csv")
    p.add_argument("--interv_col", default="intervention")
    p.add_argument("--phase_col", default="intervention_phase")
    p.add_argument("--nsamples", default="auto",
                   help="KernelSHAP coalitions: 'auto' (2M+2048), 'exact' or a number")
    p.add_argument("--context", choices=["joined", "separate"], default="joined",
                   help="routing context as part of each feature's player, or own players")
    p.add_argument("--feature_names", default="",
                   help="comma-separated names or a file with one name per line "
                        "(all 14 features; dropped ones are removed automatically)")
    p.add_argument("--data_dir", default="", help="override the stored data folder")
    p.add_argument("--out_dir", default="shap_results")
    p.add_argument("--seed", type=int, default=0, help="for the background sample")
    p.add_argument("--batch_rows", type=int, default=262144,
                   help="model inputs per forward pass in KernelSHAP")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted({f for pat in args.checkpoints for f in (glob.glob(pat) or [pat])})
    os.makedirs(args.out_dir, exist_ok=True)

    for path in paths:
        ck, name, a, fold, model = load(path, args.data_dir, device)
        seed_match = re.search(r"seed(\d+)", os.path.basename(path))
        seed = int(seed_match.group(1)) if seed_match else -1
        pred = Predictor(model, fold.y_stats.mean, fold.y_stats.std).to(device).eval()

        Xs, ys, rows = explain_set(fold, args.explain_split)
        rows, cond = recording_rows_and_conditions(ck, name, a, rows, ys, args)
        n = len(Xs) if args.n_explain <= 0 else min(args.n_explain, len(Xs))
        idx = np.unique(np.linspace(0, len(Xs) - 1, n).round().astype(int))
        X = torch.as_tensor(Xs[idx], device=device)

        pool = (np.concatenate(fold.src_X) if args.background == "source" else fold.adapt_X)
        rng = np.random.default_rng(args.seed)
        bidx = rng.choice(len(pool), min(args.n_background, len(pool)), replace=False)
        bg = torch.as_tensor(pool[bidx], device=device)

        n_feat = getattr(fold, "n_feat", X.shape[1])
        names = feature_names(args.feature_names, n_feat, a)
        groups, players = player_groups(X.shape[1], n_feat, names, args.context)
        M = len(players)

        print(f"\n{os.path.basename(path)}: {name}, {ck['config']}, target {ck['target']}, "
              f"{len(idx)} beats, {M} players, {args.method}, background {args.background} "
              f"({len(bg)})", flush=True)

        if args.method == "kernel":
            ns = (2 ** M if args.nsamples == "exact" else
                  2 * M + 2048 if args.nsamples == "auto" else int(args.nsamples))
            phi, base = kernel_shap(pred, X, bg, groups, M, ns, args.batch_rows)
        else:
            phi, base = deep_or_gradient(pred, X, bg, groups, M, args.method, ck)

        y_pred = predict(pred, X).cpu().numpy()
        add_err = base + phi.sum(1) - y_pred
        print(f"    additivity error: max {np.abs(add_err).max():.3g}, "
              f"mean {np.abs(add_err).mean():.3g} (stroke work units)")
        order = np.argsort(-np.abs(phi).mean(0))
        print("    mean |SHAP|: " + ", ".join(f"{players[g]}={np.abs(phi[:, g]).mean():.2f}"
                                             for g in order[:8]))

        import pandas as pd
        df = pd.DataFrame(phi, columns=players)
        meta = pd.DataFrame({"arch": arch_label(name, ck), "config": ck["config"], "target": ck["target"],
                             "seed": seed, "method": args.method,
                             "background": args.background, "beat_row": rows[idx],
                             "y_true": ys[idx], "y_pred": y_pred, "base_value": base,
                             "additivity_error": add_err,
                             **{k: v[idx] for k, v in cond.items()}})
        out = os.path.join(args.out_dir,
                           f"shap_{os.path.splitext(os.path.basename(path))[0]}_"
                           f"{args.method}_{args.background}.csv")
        pd.concat([meta, df], axis=1).to_csv(out, index=False)
        print(f"    -> {out}")


if __name__ == "__main__":
    main()