"""
Plots and aggregates the pairwise interactions written by interactions_from_ckpt.py, in the
same way as shap_aggregate.py does for the SHAP values (which must lie in the same folder),
so that within-animal models (oracle) and multi-source models can be compared per loading
condition. Each index (SII, STII, FSII) is treated separately.

Several seeds of the same model are combined; sd_seeds is the SD over seeds of the
per-seed mean |value| and ci_seeds the half-width of its 95 % CI (see shap_aggregate.py).
Features keep the fixed order of shap_aggregate.FEATURE_ORDER on the heatmap axes, models
keep their colour from their position in --models, and a pair keeps its colour and line
style in all time plots of an index.

Plot style, --fig_width, --panel_cols, --formats, --font_file and --no_title as in
shap_aggregate.py (every plot as .png and .pdf). In the top-pairs plot each pair is written
on two lines.

Outputs in --out_dir, per index <IX> (plots as .png and .pdf)
  <IX>_by_condition_animal.csv   per model, animal, condition and pair: mean |value|, its SD
                                 and variance over beats, signed mean, SD, variance, n beats,
                                 sd_seeds, ci_seeds, n_seeds
  <IX>_by_condition_all.csv      per model, condition and pair over animals: mean of the
                                 per-animal mean |value|, SD and variance over animals,
                                 sd_seeds and ci_seeds (over seeds of the per-seed mean over
                                 animals)
  <IX>_seed_agreement.csv        per model and animal: Spearman correlation of the pair
                                 ranking between every pair of seeds, relative seed SD
  <IX>_heatmap_<model>_<animal>.png   one panel per condition: mean |value| of every pair
  <IX>_heatmap_<model>_all.png        the same, mean over animals
                                 (one colour scale per index for all models and conditions,
                                 so the panels can be compared directly)
  <IX>_top_pairs_all.png         the --top_pairs strongest pairs per condition, models side
                                 by side, mean over animals with error bars (--error_bars as
                                 in shap_aggregate.py, default 95 % CI over seeds)
  <IX>_time_<model>_<animal>.png (with --time_plots) the strongest pairs over the recording,
                                 mean over the seeds with a 95 % CI band

Usage:
  python interactions_aggregate.py "interaction_results/*.csv" --out_dir interaction_plots \\
      --models conv:oracle,conv:mcd_drt
  python interactions_aggregate.py "interaction_results/*.csv" --error_bars sd
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shap_aggregate as sa  # noqa: E402

PAIR_SEP = " × "
META = sa.META + ["index"]
# sequential ramp in one hue (light = near zero, dark = strong), from the light blue of the
# architecture figures (W_0) to the dark blue of the palette; the diagonal (no pair) is grey
RAMP = ["#F3F8FD", "#CBE4FA", "#8FBEE6", "#3775BA", "#0F4D92", "#0A3363"]
NO_PAIR = "#DCDCDC"


def pair_matrix(sub, players, pairs):
    """Symmetric players x players matrix of mean |value|; diagonal NaN."""
    pos = {p: i for i, p in enumerate(players)}
    M = np.full((len(players), len(players)), np.nan)
    vals = sub.set_index("pair")["mean_abs"]
    for pr in pairs:
        a, b = pr.split(PAIR_SEP)
        v = vals.get(pr, np.nan)
        M[pos[a], pos[b]] = M[pos[b], pos[a]] = v
    return M


def plot_heatmaps(agg, players, pairs, conds, vmax, path, title):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("ramp", RAMP)
    cmap.set_bad(NO_PAIR)
    n_c, n_p = len(conds), len(players)
    ncols = min(sa.PANEL_COLS, n_c)
    nrows = -(-n_c // ncols)
    # panels as wide as the figure allows next to the feature names and the colour bar;
    # the tick labels shrink with the cells (at most 7 pt)
    label_in = 1.35
    panel = (sa.FIG_WIDTH - label_in - 0.75) / ncols
    fs = min(7.0, 0.85 * 72 * panel / n_p)
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False,
                             layout="constrained",
                             figsize=(sa.FIG_WIDTH, nrows * (panel + 0.25) + label_in
                                      + (0.3 if sa.SHOW_TITLE else 0)))
    im = None
    for k, ax in enumerate(axes.flat):
        if k >= n_c:
            ax.axis("off")
            continue
        c = conds[k]
        M = pair_matrix(agg[agg["condition"] == c], players, pairs)
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(c)
        ax.set_xticks(range(n_p))
        ax.set_yticks(range(n_p))
        ax.set_xticklabels(players, rotation=90, fontsize=fs, color=sa.INK)
        ax.set_yticklabels(players, fontsize=fs, color=sa.INK)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.label_outer()
        # a panel without one below it keeps its feature names on the x axis
        if k + ncols >= n_c:
            ax.xaxis.set_tick_params(labelbottom=True)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, aspect=25, pad=0.02)
    cbar.set_label("mean |interaction| (stroke work)")
    cbar.outline.set_visible(False)
    if sa.SHOW_TITLE:
        fig.suptitle(title)
    sa.save(fig, path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--out_dir", default="interaction_plots")
    p.add_argument("--models", default="")
    p.add_argument("--top_pairs", type=int, default=10)
    p.add_argument("--time_plots", action="store_true")
    p.add_argument("--top_k", type=int, default=5, help="pairs shown in the time plots")
    p.add_argument("--error_bars", choices=["ci", "sd", "beats", "seeds", "default"],
                   default="ci", help="top-pairs plot: ci = 95 %% CI over seeds, sd = SD over "
                   "seeds, beats = SD over animals (over beats with one animal)")
    sa.add_style_args(p)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    print("font:", sa.style_from_args(args))

    paths = sorted({f for pat in args.files for f in (glob.glob(pat) or [pat])})
    frames = []
    for f in paths:
        d = pd.read_csv(f)
        d = d.assign(model=d["arch"].map(sa.ARCH_SHORT).fillna(d["arch"]) + ":" + d["config"],
                     file=os.path.splitext(os.path.basename(f))[0])
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    value_cols = [c for c in frames[0].columns if c not in META + ["model", "file"]]
    pairs = [c for c in value_cols if PAIR_SEP in c]
    players = sa.ordered([c for c in value_cols if PAIR_SEP not in c])
    if df["background"].nunique() > 1:
        print(f"WARNING: mixed background: {sorted(df['background'].unique())} - not comparable")
    models = [m for m in args.models.split(",") if m] or sorted(df["model"].unique())
    model_index = {m: i for i, m in enumerate(models)}
    df = df[df["model"].isin(models)]
    os.makedirs(args.out_dir, exist_ok=True)

    for ix, dix in df.groupby("index"):
        if args.time_plots:
            # one fixed style per pair for all time plots of this index: the pairs shown in
            # any plot, ordered by their overall mean |value|
            groups = list(dix.groupby(["model", "target"]))
            shown = set()
            for _, d in groups:
                shown |= set(d[pairs].abs().mean().sort_values(ascending=False)
                             .index[:args.top_k])
            overall = dix[pairs].abs().mean()
            shown = sorted(shown, key=lambda c: -overall[c])
            if len(shown) > len(sa.COLORS) * len(sa.LINESTYLES):
                print(f"{ix}: {len(shown)} pairs in the time plots, styles repeat")
            styles = sa.line_styles(shown)
            for (m, target), d in groups:
                top = list(d[pairs].abs().mean().sort_values(ascending=False)
                           .index[:args.top_k])
                sa.plot_time(d, top, args.top_k, styles,
                             os.path.join(args.out_dir,
                                          f"{ix}_time_{m.replace(':', '-')}_{target}.png"),
                             f"{ix}: {m}, target {target}",
                             ylabel=f"{ix} pair value (stroke work)")

        lab = dix[dix["condition"] != "unknown"]
        if lab.empty:
            print(f"{ix}: no intervention labels (condition = unknown); skipped aggregation")
            continue
        long = lab.melt(id_vars=["model", "target", "condition", "seed"], value_vars=pairs,
                        var_name="pair", value_name="value")
        long["abs"] = long["value"].abs()
        g = long.groupby(["model", "target", "condition", "pair"])
        per_animal = pd.DataFrame({
            "mean_abs": g["abs"].mean(), "sd_abs": g["abs"].std(), "var_abs": g["abs"].var(),
            "mean": g["value"].mean(), "sd": g["value"].std(), "var": g["value"].var(),
            "n_beats": g["value"].size()}).reset_index()
        sd_pa, sd_over = sa.seed_sd(long, "pair")
        per_animal = per_animal.merge(sd_pa, on=["model", "target", "condition", "pair"],
                                      how="left")
        per_animal.to_csv(os.path.join(args.out_dir, f"{ix}_by_condition_animal.csv"),
                          index=False)
        g2 = per_animal.groupby(["model", "condition", "pair"])["mean_abs"]
        over = pd.DataFrame({"mean_abs": g2.mean(), "sd_abs": g2.std(), "var_abs": g2.var(),
                             "n_animals": g2.size()}).reset_index()
        over = over.merge(sd_over, on=["model", "condition", "pair"], how="left")
        over.to_csv(os.path.join(args.out_dir, f"{ix}_by_condition_all.csv"), index=False)

        agree = sa.seed_agreement(dix, pairs)
        if not agree.empty:
            agree.to_csv(os.path.join(args.out_dir, f"{ix}_seed_agreement.csv"), index=False)
            print(f"{ix} seed agreement (mean over animals):")
            print(agree.groupby("model")[["spearman_mean", "spearman_min", "rel_sd_median"]]
                  .mean().round(3).to_string())

        # heatmaps: one colour scale per index and level for all models and conditions
        vmax_animal = float(per_animal["mean_abs"].max())
        for (model, target), d in per_animal.groupby(["model", "target"]):
            plot_heatmaps(d, players, pairs, sa.condition_order(d["condition"]), vmax_animal,
                          os.path.join(args.out_dir,
                                       f"{ix}_heatmap_{model.replace(':', '-')}_{target}.png"),
                          f"{ix}, {model}, {target}: mean |pair value| per condition")
        n_anim = per_animal["target"].nunique()
        if n_anim > 1:
            vmax_all = float(over["mean_abs"].max())
            for model, d in over.groupby("model"):
                plot_heatmaps(d, players, pairs, sa.condition_order(d["condition"]), vmax_all,
                              os.path.join(args.out_dir,
                                           f"{ix}_heatmap_{model.replace(':', '-')}_all.png"),
                              f"{ix}, {model}: mean over {n_anim} animals of the mean "
                              f"|pair value| per condition")

        # strongest pairs, models side by side
        src = over if n_anim > 1 else per_animal
        top = list(src.groupby("pair")["mean_abs"].mean().sort_values(ascending=False)
                   .index[:args.top_pairs])
        err_a, err_o = sa.error_columns(args.error_bars, int(per_animal["n_seeds"].max()))
        err, err_txt = err_o if n_anim > 1 else err_a
        sa.plot_conditions(src[src["pair"].isin(top)].rename(columns={"pair": "player"}),
                           [m for m in models if m in set(src["model"])], top,
                           sa.condition_order(src["condition"]), "mean_abs", err,
                           os.path.join(args.out_dir, f"{ix}_top_pairs_all.png"),
                           f"{ix}: {args.top_pairs} strongest pairs per condition "
                           + (f"(mean over {n_anim} animals, " if n_anim > 1 else "(")
                           + f"error bars: {err_txt})",
                           xlabel="mean |pair value|", model_index=model_index,
                           ylabels=[pr.replace(PAIR_SEP, PAIR_SEP.rstrip() + "\n")
                                    for pr in top])
        print(f"{ix}: {len(set(dix['file']))} files, {len(pairs)} pairs, models: "
              f"{', '.join(sorted(set(dix['model'])))}")
    print(f"written to {args.out_dir}/")


if __name__ == "__main__":
    main()