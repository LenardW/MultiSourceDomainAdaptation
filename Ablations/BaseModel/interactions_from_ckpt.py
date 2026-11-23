"""
Pairwise Shapley interactions for checkpoints of msda_mlp.py, msda_conv.py,
msda_conv_ctx.py and msda_conv_interv.py (saved with --save_dir).

Loading, explained beats, background, players and conditions are exactly as in
shap_from_ckpt.py (which must lie in the same folder), so the interactions are comparable
with the SHAP values:
  - one player per feature (window positions and routing context joined),
  - value of a coalition = predicted stroke work with the features of the coalition taken
    from the explained beat and the remaining ones from each background beat, averaged
    over the background (interventional game),
  - beats evenly spaced over the chosen split, with their row in the recording and their
    condition (baseline = phase 1, else I<code>).

Computation
  With 14 players all 2^14 = 16384 coalitions are evaluated per beat, so the indices are
  computed exactly (shapiq.ExactComputer); all requested indices come from the same game
  values at no extra model cost. Cost per beat: 16384 x n_background forward passes.

Indices (--indices, order 2)
  SII    Shapley interaction index; main effects = Shapley values; not efficient
  STII   Shapley-Taylor interaction index (STI); efficient
  FSII   Faith-Shap (faithful Shapley interaction index); efficient
  For STII and FSII: base value + main effects + pair values = prediction (checked).

Output: one CSV per checkpoint and index in --out_dir, one row per explained beat:
  the meta columns of shap_from_ckpt.py, index, one column per feature (main effect) and
  one column per pair '<feature A> × <feature B>'. interactions_aggregate.py plots and
  aggregates them.

Usage:
  python interactions_from_ckpt.py "ckpt_conv/*.pt" --explain_split all --n_explain 100 \\
      --feature_names names.txt --out_dir interaction_results
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shap_from_ckpt as sfc  # noqa: E402

INDEX_NAMES = {"SII": "SII", "STII": "STII", "STI": "STII", "FSII": "FSII",
               "Faith-Shap": "FSII"}
PAIR_SEP = " × "


def make_game(pred, x, bg, groups, batch_rows):
    """Value function of the interventional game for one beat."""
    gidx = torch.as_tensor(groups, device=x.device)
    B = len(bg)
    extra = (1,) * (x.dim() - 1)

    def game(Z):
        Z = torch.as_tensor(np.asarray(Z), dtype=torch.bool, device=x.device)
        mask = Z[:, gidx]                                          # [n, C]
        out, k = [], max(1, batch_rows // B)
        for i in range(0, len(mask), k):
            m = mask[i:i + k].view(-1, 1, mask.shape[1], *extra)
            comp = torch.where(m, x.unsqueeze(0).unsqueeze(0), bg.unsqueeze(0))
            out.append(sfc.predict(pred, comp.reshape(-1, *x.shape)).view(-1, B).mean(1))
        return torch.cat(out).cpu().numpy()

    return game


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--indices", default="SII,STII,FSII")
    p.add_argument("--background", choices=["target", "source"], default="target")
    p.add_argument("--n_background", type=int, default=50)
    p.add_argument("--n_explain", type=int, default=100)
    p.add_argument("--explain_split", choices=["test", "adapt", "all"], default="all")
    p.add_argument("--feature_names", default="")
    p.add_argument("--interv_pattern", default="I_{aid}.csv")
    p.add_argument("--interv_col", default="intervention")
    p.add_argument("--phase_col", default="intervention_phase")
    p.add_argument("--data_dir", default="")
    p.add_argument("--out_dir", default="interaction_results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_rows", type=int, default=262144)
    p.add_argument("--max_players", type=int, default=16,
                   help="exact computation needs 2^players evaluations per beat")
    args = p.parse_args()
    args.context = "joined"

    import shapiq
    indices = [INDEX_NAMES[i.strip()] for i in args.indices.split(",") if i.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted({f for pat in args.checkpoints for f in (glob.glob(pat) or [pat])})
    os.makedirs(args.out_dir, exist_ok=True)

    for path in paths:
        ck, name, a, fold, model = sfc.load(path, args.data_dir, device)
        seed_match = re.search(r"seed(\d+)", os.path.basename(path))
        seed = int(seed_match.group(1)) if seed_match else -1
        pred = sfc.Predictor(model, fold.y_stats.mean, fold.y_stats.std).to(device).eval()

        Xs, ys, rows = sfc.explain_set(fold, args.explain_split)
        rows, cond = sfc.recording_rows_and_conditions(ck, name, a, rows, ys, args)
        n = len(Xs) if args.n_explain <= 0 else min(args.n_explain, len(Xs))
        idx = np.unique(np.linspace(0, len(Xs) - 1, n).round().astype(int))
        X = torch.as_tensor(Xs[idx], device=device)

        pool = np.concatenate(fold.src_X) if args.background == "source" else fold.adapt_X
        rng = np.random.default_rng(args.seed)
        bidx = rng.choice(len(pool), min(args.n_background, len(pool)), replace=False)
        bg = torch.as_tensor(pool[bidx], device=device)

        n_feat = getattr(fold, "n_feat", X.shape[1])
        names = sfc.feature_names(args.feature_names, n_feat, a)
        groups, players = sfc.player_groups(X.shape[1], n_feat, names, "joined")
        M = len(players)
        if M > args.max_players:
            raise SystemExit(f"{M} players: 2^{M} coalitions per beat is too many")
        pairs = list(combinations(range(M), 2))
        pair_cols = [f"{players[i]}{PAIR_SEP}{players[j]}" for i, j in pairs]

        print(f"\n{os.path.basename(path)}: {name}, {ck['config']}, target {ck['target']}, "
              f"{len(idx)} beats, {M} players ({2 ** M} coalitions), "
              f"indices {', '.join(indices)}, background {args.background} ({len(bg)})",
              flush=True)

        results = {ix: [] for ix in indices}
        base, y_pred = np.zeros(len(idx)), np.zeros(len(idx))
        for b, x in enumerate(X):
            game = make_game(pred, x, bg, groups, args.batch_rows)
            ec = shapiq.ExactComputer(game, n_players=M, evaluate_game=True)
            v_empty = float(game(np.zeros((1, M), dtype=bool))[0])
            v_full = float(game(np.ones((1, M), dtype=bool))[0])
            base[b], y_pred[b] = v_empty, v_full
            for ix in indices:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    iv = ec(ix, order=2)
                look = iv.interaction_lookup
                main_eff = [float(iv.values[look[(i,)]]) for i in range(M)]
                pair_val = [float(iv.values[look[pr]]) for pr in pairs]
                results[ix].append(main_eff + pair_val)
            if (b + 1) % 10 == 0 or b + 1 == len(idx):
                print(f"    {b + 1}/{len(idx)} beats", flush=True)

        meta = pd.DataFrame({"arch": sfc.arch_label(name, ck), "config": ck["config"], "target": ck["target"],
                             "seed": seed, "method": "exact", "background": args.background,
                             "beat_row": rows[idx], "y_true": ys[idx], "y_pred": y_pred,
                             "base_value": base,
                             **{k: v[idx] for k, v in cond.items()}})
        for ix in indices:
            vals = pd.DataFrame(np.array(results[ix]), columns=players + pair_cols)
            total = base + vals.sum(axis=1).to_numpy()
            err = total - y_pred if ix in ("STII", "FSII") else np.full(len(idx), np.nan)
            df = pd.concat([meta.assign(index=ix, additivity_error=err), vals], axis=1)
            out = os.path.join(args.out_dir,
                               f"inter_{os.path.splitext(os.path.basename(path))[0]}_"
                               f"{ix}_{args.background}.csv")
            df.to_csv(out, index=False)
            mean_abs = vals[pair_cols].abs().mean().sort_values(ascending=False)
            msg = (f"    {ix}: " + (f"additivity error max {np.abs(err).max():.2g}; "
                                     if ix != "SII" else "")
                   + "strongest pairs: "
                   + "; ".join(f"{k} = {v:.2f}" for k, v in mean_abs.head(3).items()))
            print(msg)
            print(f"    -> {out}")


if __name__ == "__main__":
    main()