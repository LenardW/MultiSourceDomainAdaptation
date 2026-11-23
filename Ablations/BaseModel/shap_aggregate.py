"""
Plots and aggregates the SHAP values written by shap_from_ckpt.py, so that the
explanations of different models can be compared, e.g. the within-animal model (oracle:
trained and tested on the same animal) with the multi-source models (source_only, mcd_drt,
...) of the same target animal.

Model label: <architecture>:<config>, e.g. conv:oracle, conv:mcd_drt, mlp:source_only.
If several seeds of the same model are given, they are combined. All seeds explain the
same beats with the same background, so the pooled mean |SHAP| equals the mean over seeds
of the per-seed mean |SHAP|. The seed SD is the SD over seeds of that per-seed mean, and
the 95 % CI is the t-interval of the mean over seeds (t with n_seeds - 1 degrees of freedom).

Every feature keeps the same colour, line style and position in all plots: the order is
FEATURE_ORDER (players not in it follow in their original order), and every model keeps
its colour and hatching from its position in --models (the first model gets the dark blue).

Style (shared with interactions_aggregate.py and matched to the architecture figures):
palette after the figures4papers repository, the thesis font (Latin Modern / CMU Serif if
found, see --font_file), figures drawn at their printed width (--fig_width) with 8-9 pt text,
no top/right axis lines, frameless legends. Every plot is written as .png and .pdf
(--formats); include the .pdf with \\includegraphics[width=\\textwidth]{...}. --no_title
leaves out the titles for the thesis, where the caption takes their place.

Outputs in --out_dir (each as .png and .pdf)
  time_<model>_<animal>.png       per model and animal: stroke work (true, predicted) and
                                  the SHAP values of the --top_k most important players over
                                  the recording, as mean over the seeds with a 95 % CI band,
                                  with the loading conditions marked
  shap_by_condition_animal.csv    per model, animal, condition and player: mean |SHAP| and
                                  its SD and variance over beats, mean SHAP (signed) and its
                                  SD and variance, number of beats; sd_seeds: SD over seeds
                                  of the per-seed mean |SHAP|, n_seeds
  shap_by_condition_all.csv       per model, condition and player, over animals: mean of the
                                  per-animal mean |SHAP|, its SD and variance over animals,
                                  number of animals; sd_seeds: SD over seeds of the per-seed
                                  mean over animals, n_seeds
  shap_seed_agreement.csv         per model and animal: Spearman correlation of the feature
                                  ranking (mean |SHAP| over all beats) between every pair of
                                  seeds (mean and minimum), and the median over players of
                                  SD over seeds / mean (relative seed SD)
  conditions_<animal>.png         one panel per condition (baseline and the interventions),
                                  mean |SHAP| with error bars (see --error_bars), models side
                                  by side
  conditions_all.png              the same, mean over animals

--error_bars  ci     95 % CI over seeds (default; SD over beats/animals with one seed)
              sd     SD over seeds ("seeds" is accepted as well)
              beats  SD over beats (per animal) and over animals (all) ("default" as well)

Conditions: baseline = intervention_phase 1, otherwise I<intervention code> (as in
msda_conv_interv.py). Beats without labels ('unknown') are left out of the aggregation.
The comparison is only meaningful if all files use the same method, background and players;
the script warns otherwise.

Usage:
  python shap_aggregate.py "shap_results/*.csv" --out_dir shap_plots
  python shap_aggregate.py "shap_results/*conv*.csv" --models conv:oracle,conv:mcd_drt
  python shap_aggregate.py "shap_results/*.csv" --error_bars sd
  python shap_aggregate.py "shap_results/*.csv" --no_title --fig_width 6.1 --formats pdf
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

META = ["arch", "config", "target", "seed", "method", "background", "beat_row", "y_true",
        "y_pred", "base_value", "additivity_error", "intervention", "intervention_phase",
        "condition"]
ARCH_SHORT = {"msda_mlp": "mlp", "msda_conv": "conv", "msda_conv_ctx": "ctx",
              "msda_conv_interv": "interv", "msda_conv_ctx_mlp": "mlpctx",
              "msda_conv_interv_mlp": "mlpinterv"}

# ---- house style, shared with the architecture figures -----------------------------------
# Palette after the figures4papers repository (dark blue, red, teal, violet), plus an ochre
# and a light blue for the 5th and 6th series. Checked over all pairs: OKLab distance x100
# >= 11 for normal vision and >= 6.5 under protan, deutan and tritan simulation. Identity is
# never colour alone: legends, line styles and hatching are added.
COLORS = ["#0F4D92", "#B64342", "#42949E", "#9A4D8E", "#D49A1E", "#6CA6DC"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))]
HATCHES = ["", "//", "..", "\\\\", "xx", "--"]
INK, INK_2, GRID = "#272727", "#4D4D4D", "#E6E6E6"
SHADE, SHADE_EDGE = "#EFEFEF", "#CFCECE"      # baseline stretches in the time plots
# figures are drawn at their printed size, so that 8-9 pt in the plot is 8-9 pt in the thesis
FIG_WIDTH = 6.3          # inches; set to \textwidth with --fig_width (pt / 72.27)
PANEL_COLS = 3           # panels per row in the condition plots and heatmaps
FORMATS = ["png", "pdf"]
SHOW_TITLE = True
# serif font of the thesis (Computer Modern / Latin Modern); searched in the usual TeX and
# font folders, otherwise STIX (shipped with matplotlib)
SERIF = ["CMU Serif", "Latin Modern Roman"]
FONT_GLOBS = [
    "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-*.otf",
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf",
    "/usr/local/texlive/*/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf",
    "/Library/TeX/Root/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf",
    "~/texmf/fonts/opentype/public/lm/lmroman10-*.otf",
    "/usr/share/fonts/**/cmun*.otf", "/usr/share/fonts/**/cmun*.ttf",
    "~/.fonts/**/cmun*.[ot]tf", "~/.local/share/fonts/**/cmun*.[ot]tf",
    "~/Library/Fonts/cmun*.[ot]tf", "C:/Windows/Fonts/cmun*.[ot]tf"]

# fixed order of the 14 features (rows of the bar plots, axes of the heatmaps, styles)
FEATURE_ORDER = [
    "Pressure head–flow ratio", "Motor power–LV pressure ratio", "Pulsatile flow integral",
    "Total flow integral", "Pulsatile current integral", "Total current integral",
    "Mean pump speed", "Mean pressure head", "Minimum pressure head", "Maximum pressure head",
    "Maximum flow acceleration", "Maximum flow deceleration",
    "Smoothed max. flow acceleration", "Smoothed max. flow deceleration"]

# two-sided 95 % t quantiles for 1..9 degrees of freedom (fallback without scipy)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262}


def t95(n):
    """Two-sided 95 % t quantile for a mean over n values (NaN for n < 2)."""
    n = np.asarray(n, dtype=float)
    try:
        from scipy.stats import t
        q = t.ppf(0.975, np.maximum(n - 1, 1))
    except ImportError:
        q = np.vectorize(lambda k: _T95.get(int(k) - 1, 1.96))(np.maximum(n, 2))
    return np.where(n >= 2, q, np.nan)


def ordered(players):
    """Players in FEATURE_ORDER, then the remaining ones in their original order."""
    players = list(players)
    known = [p for p in FEATURE_ORDER if p in players]
    return known + [p for p in players if p not in known]


def line_styles(items):
    """Fixed (colour, line style) per item, by its position in items."""
    nc = len(COLORS)
    return {it: (COLORS[i % nc], LINESTYLES[(i // nc) % len(LINESTYLES)])
            for i, it in enumerate(items)}


def condition_order(conds):
    def key(c):
        if c == "baseline":
            return (0, 0)
        m = re.fullmatch(r"I(\d+)", str(c))
        return (1, int(m.group(1))) if m else (2, 0)
    return sorted(set(conds), key=key)


def apply_style(font_file=None, fig_width=None, formats=None, title=True, panel_cols=None):
    """Sets the house style (call once, after matplotlib.use). Returns the font family used."""
    global FIG_WIDTH, FORMATS, SHOW_TITLE, PANEL_COLS
    import matplotlib as mpl
    from matplotlib import font_manager as fm
    FIG_WIDTH = fig_width or FIG_WIDTH
    FORMATS = formats or FORMATS
    SHOW_TITLE = title
    PANEL_COLS = panel_cols or PANEL_COLS
    files = [font_file] if font_file else []
    for pat in FONT_GLOBS:
        files += glob.glob(os.path.expanduser(pat), recursive=True)
    for f in files:
        try:
            fm.fontManager.addfont(f)
        except Exception:
            pass
    names = {f.name for f in fm.fontManager.ttflist}
    if font_file:
        family = fm.FontProperties(fname=font_file).get_name()
    else:
        family = next((n for n in SERIF if n in names), None)
    if family is None:
        print("NOTE: no Computer/Latin Modern font found, using STIX. For the thesis font, "
              "install fonts-lmodern or fonts-cmu, or pass --font_file <lmroman10-regular.otf>")
        family = "STIXGeneral"
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": [family, "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix" if family == "STIXGeneral" else "cm",
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "figure.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.4, "legend.frameon": False, "legend.handlelength": 2.2,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "savefig.facecolor": "white",
        # TrueType fonts are embedded as Type 42 (editable text), OpenType/CFF as Type 3
        "pdf.fonttype": 42 if family in ("STIXGeneral",) or str(font_file).endswith(".ttf")
        else 3,
        "svg.fonttype": "none"})
    return family


def save(fig, path):
    """Writes fig in every format of FORMATS, path with or without extension."""
    stem = os.path.splitext(path)[0]
    for ext in FORMATS:
        fig.savefig(f"{stem}.{ext}")


def style_axes(ax, grid="y"):
    """Light grid along one axis only (None for no grid); spines and ticks come from
    apply_style."""
    ax.grid(False)
    if grid:
        ax.grid(True, axis=grid, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def condition_spans(rows, conds):
    """Contiguous stretches of the same condition along the recording; the borders lie
    halfway between the last explained beat of one condition and the first of the next."""
    spans, start = [], 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or conds[i] != conds[start]:
            a = rows[0] if start == 0 else (rows[start - 1] + rows[start]) / 2
            b = rows[-1] if i == len(rows) else (rows[i - 1] + rows[i]) / 2
            spans.append((a, b, conds[start]))
            start = i
    return spans


def fit_legend(ax, max_cols, handles=None, expand=False, **kw):
    """Legend that is never wider than the plot: at most max_cols columns, only as many as fit
    into the width of ax, with the rows filled evenly. With expand, the columns are spread
    over the full width of the plot."""
    renderer = ax.figure.canvas.get_renderer()
    width = ax.get_window_extent(renderer).width
    if handles is None:
        handles, _ = ax.get_legend_handles_labels()
    ncol = 1
    for n in range(min(max_cols, len(handles)), 0, -1):
        leg = ax.legend(handles=handles, ncol=n, **kw)
        fits = leg.get_window_extent(renderer).width <= width
        leg.remove()
        if fits:
            ncol = n
            break
    # fill the rows evenly: as few rows as possible, and for them as few columns as needed
    # (4 entries that fit into 3 columns are shown as 2 x 2, not 3 + 1)
    rows = -(-len(handles) // ncol)
    ncol = -(-len(handles) // rows)
    if expand:
        x0, y0 = kw.pop("bbox_to_anchor", (0, 0))[:2]
        kw.update(bbox_to_anchor=(x0, y0, 1, 0), mode="expand")
    return ax.legend(handles=handles, ncol=ncol, **kw)


def seed_mean_ci(df, cols):
    """Per beat: mean over seeds of cols and the half-width of the 95 % CI."""
    g = df.groupby("beat_row")
    mean = g[cols].mean()
    half = g[cols].std().mul(t95(g.size().to_numpy()) / np.sqrt(g.size().to_numpy()), axis=0)
    return mean, half


def plot_time(df, cols, top_k, styles, path, title, ylabel="SHAP value (stroke work)",
              include=()):
    """Stroke work and the top_k columns over the recording, mean over the seeds in df with
    a 95 % CI band (no band with a single seed). styles: column -> (colour, line style).
    include: columns shown in addition to the top_k, e.g. a feature that matters only in
    one condition."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    n_seeds = df["seed"].nunique()
    mean, half = seed_mean_ci(df, list(cols) + ["y_true", "y_pred"])
    rows = mean.index.to_numpy()
    conds = df.groupby("beat_row")["condition"].first().reindex(mean.index).to_numpy()
    imp = df[list(cols)].abs().mean().sort_values(ascending=False)
    top = [c for c in ordered(cols) if c in set(imp.index[:top_k]) | set(include)]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(FIG_WIDTH, 0.62 * FIG_WIDTH), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.6]})
    for ax in (ax0, ax1):
        style_axes(ax)
    # baseline stretches shaded, every change of condition marked, interventions labelled
    labels_top = []
    for k, (a, b, c) in enumerate(condition_spans(rows, conds)):
        for ax in (ax0, ax1):
            if c == "baseline":
                ax.axvspan(a, b, color=SHADE, linewidth=0, zorder=0)
            if k > 0:
                ax.axvline(a, color=SHADE_EDGE, linewidth=0.6, zorder=0)
        if c != "baseline" and b - a > 0.03 * (rows[-1] - rows[0]):
            labels_top.append(ax0.text((a + b) / 2, 1.02, c, transform=ax0.get_xaxis_transform(),
                                       ha="center", va="bottom", fontsize="small", color=INK_2))

    ax0.plot(rows, mean["y_true"], color=INK, label="true")
    ax0.plot(rows, mean["y_pred"], color=INK_2, linestyle="--",
             label="predicted" + (" (mean over seeds)" if n_seeds > 1 else ""))
    if n_seeds > 1:
        ax0.fill_between(rows, mean["y_pred"] - half["y_pred"], mean["y_pred"] + half["y_pred"],
                         color=INK_2, alpha=0.2, linewidth=0)
    ax0.set_ylabel("stroke work")
    handles, labels = ax0.get_legend_handles_labels()
    if n_seeds > 1:
        handles.append(Patch(facecolor=INK_2, alpha=0.2, label="95 % CI over seeds"))
    if (conds == "baseline").any():
        handles.append(Patch(facecolor=SHADE, edgecolor=SHADE_EDGE, label="baseline (phase 1)"))

    for c in top:
        color, ls = styles[c]
        ax1.plot(rows, mean[c], color=color, linestyle=ls, label=c)
        if n_seeds > 1:
            ax1.fill_between(rows, mean[c] - half[c], mean[c] + half[c], color=color,
                             alpha=0.15, linewidth=0)
    ax1.axhline(0, color=INK_2, linewidth=0.8)
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel("beat")
    fig.tight_layout()
    # legends only after the layout, so that their columns fit the final width of the plot.
    # Stroke-work legend above the upper plot (over the condition labels, not over the
    # curves), feature legend just below the x label, title above everything
    r = fig.canvas.get_renderer()
    top = max([t.get_window_extent(r).y1 for t in labels_top]
              + [ax0.get_window_extent(r).y1])
    y = ax0.transAxes.inverted().transform((0, top))[1]
    leg0 = fit_legend(ax0, 4, handles=handles, loc="lower left", bbox_to_anchor=(0, y + 0.03),
                      borderaxespad=0)
    y = ax1.transAxes.inverted().transform((0, ax1.xaxis.label.get_window_extent(r).y0))[1]
    fit_legend(ax1, 3, expand=True, loc="upper left", bbox_to_anchor=(0, y - 0.015),
               borderaxespad=0)
    if SHOW_TITLE:
        y = fig.transFigure.inverted().transform((0, leg0.get_window_extent(r).y1))[1]
        fig.suptitle(title + (f" (mean over {n_seeds} seeds, bands: 95 % CI)"
                              if n_seeds > 1 else ""), y=y + 0.01, va="bottom")
    save(fig, path)
    plt.close(fig)


def seed_sd(long, key, value="abs"):
    """SD over seeds of the per-seed mean |value| and the half-width of its 95 % CI: per
    model, animal, condition and key (player or pair), and for the mean over animals (per
    seed: mean over animals, then over seeds)."""
    ps = (long.groupby(["model", "target", "condition", key, "seed"])[value].mean()
          .rename("m").reset_index())
    out = []
    for frame, by in ((ps, ["model", "target", "condition", key]),
                      (ps.groupby(["model", "condition", key, "seed"])["m"].mean()
                       .rename("m").reset_index(), ["model", "condition", key])):
        g = frame.groupby(by)["m"]
        r = pd.DataFrame({"sd_seeds": g.std(), "n_seeds": g.size()}).reset_index()
        r["ci_seeds"] = t95(r["n_seeds"]) * r["sd_seeds"] / np.sqrt(r["n_seeds"])
        out.append(r)
    return out[0], out[1]


def seed_agreement(df, cols):
    """Per model and animal: Spearman correlation of the importance ranking (mean |value|
    over all beats) between every pair of seeds, and the median over columns of the
    relative seed SD (SD over seeds / mean)."""
    rows = []
    for (model, target), d in df.groupby(["model", "target"]):
        imp = d.groupby("seed")[cols].apply(lambda x: x.abs().mean())
        if len(imp) < 2:
            continue
        rho = imp.T.corr(method="spearman").to_numpy()
        r = rho[np.triu_indices(len(imp), 1)]
        rel = (imp.std() / imp.mean()).replace([np.inf, -np.inf], np.nan)
        rows.append({"model": model, "target": target, "n_seeds": len(imp),
                     "spearman_mean": r.mean(), "spearman_min": r.min(),
                     "rel_sd_median": rel.median()})
    return pd.DataFrame(rows)


def plot_conditions(agg, models, players, conds, value, err, path, title,
                    xlabel="mean |SHAP|", model_index=None, ylabels=None):
    """One panel per condition (at most PANEL_COLS per row); horizontal bars per player,
    models side by side. Each model keeps the colour and hatching of its position in
    model_index (default: in models). ylabels: tick labels of the players (may contain line
    breaks), default the player names."""
    import matplotlib.pyplot as plt
    n_c, n_p, n_m = len(conds), len(players), len(models)
    ylabels = ylabels or players
    lines = max(str(l).count("\n") + 1 for l in ylabels)
    ncols = min(PANEL_COLS, n_c)
    nrows = -(-n_c // ncols)
    # height per player: room for its bars and for its label
    slot = max(0.05 * n_m + 0.06, 0.13 * lines + 0.03)
    fig, axes = plt.subplots(nrows, ncols, sharey=True, sharex=True, squeeze=False,
                             figsize=(FIG_WIDTH, nrows * (n_p * slot + 0.45) + 0.15))
    height = 0.8 / n_m
    y = np.arange(n_p)
    model_index = model_index or {m: i for i, m in enumerate(models)}
    for k, ax in enumerate(axes.flat):
        if k >= n_c:
            ax.axis("off")
            continue
        c = conds[k]
        style_axes(ax, grid="x")
        for mi, m in enumerate(models):
            sub = agg[(agg["model"] == m) & (agg["condition"] == c)].set_index("player")
            v = sub[value].reindex(players).to_numpy()
            e = sub[err].reindex(players).to_numpy()
            ax.barh(y + (mi - (n_m - 1) / 2) * height, v, height=height * 0.9,
                    color=COLORS[model_index[m] % len(COLORS)],
                    hatch=HATCHES[model_index[m] % len(HATCHES)],
                    edgecolor="white", linewidth=0.5, xerr=e,
                    error_kw={"elinewidth": 0.7, "ecolor": INK_2, "capsize": 0},
                    label=m if k == 0 else None)
        ax.set_title(c)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_tick_params(labelbottom=True)    # values readable in every row
        if k >= n_c - ncols:            # lowest panel of its column
            ax.set_xlabel(xlabel)
    # only now, when the shared x range has seen the bars of all panels
    axes[0][0].set_xlim(left=0)
    axes[0][0].set_yticks(y)
    axes[0][0].set_yticklabels(ylabels, color=INK)
    axes[0][0].set_ylim(n_p - 0.5, -0.5)
    fig.tight_layout(h_pad=0.8, w_pad=1.4)
    # legend above the panels, title above the legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=min(n_m, 4),
                     bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2)
    if SHOW_TITLE:
        r = fig.canvas.get_renderer()
        top = fig.transFigure.inverted().transform((0, leg.get_window_extent(r).y1))[1]
        fig.suptitle(title, y=top + 0.005, va="bottom")
    save(fig, path)
    plt.close(fig)


def error_columns(choice, n_seeds):
    """(column, label) of the error bars per animal and over animals."""
    choice = {"seeds": "sd", "default": "beats"}.get(choice, choice)
    if choice in ("ci", "sd") and n_seeds < 2:
        print(f"WARNING: --error_bars {choice} needs several seeds; using SD over beats/animals")
        choice = "beats"
    if choice == "ci":
        return ("ci_seeds", "95 % CI over seeds"), ("ci_seeds", "95 % CI over seeds")
    if choice == "sd":
        return ("sd_seeds", "SD over seeds"), ("sd_seeds", "SD over seeds")
    return ("sd_abs", "SD over beats"), ("sd_abs", "SD over animals")


def add_style_args(p):
    """Style options shared by shap_aggregate.py and interactions_aggregate.py."""
    p.add_argument("--fig_width", type=float, default=FIG_WIDTH,
                   help="figure width in inches, best the \\textwidth of the thesis "
                   "(\\the\\textwidth in pt / 72.27); then include with width=\\textwidth")
    p.add_argument("--panel_cols", type=int, default=PANEL_COLS,
                   help="panels per row in the condition plots and heatmaps")
    p.add_argument("--formats", default=",".join(FORMATS),
                   help="comma-separated, e.g. png,pdf (pdf is vector, for the thesis)")
    p.add_argument("--font_file", default=None,
                   help="font file to use instead of the search for Latin Modern / CMU Serif")
    p.add_argument("--no_title", action="store_true",
                   help="no titles in the plots (the caption in the thesis takes their place)")


def style_from_args(args):
    return apply_style(font_file=args.font_file, fig_width=args.fig_width,
                       formats=[f.strip() for f in args.formats.split(",") if f.strip()],
                       title=not args.no_title, panel_cols=args.panel_cols)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="SHAP CSV files or glob patterns")
    p.add_argument("--out_dir", default="shap_plots")
    p.add_argument("--models", default="",
                   help="comma-separated model labels and their order, e.g. conv:oracle,conv:mcd_drt")
    p.add_argument("--top_k", type=int, default=6, help="players shown in the time plots")
    p.add_argument("--time_include", default="",
                   help="comma-separated players always shown in the time plots, in "
                   "addition to the --top_k, e.g. \"Mean pump speed\"")
    p.add_argument("--no_time_plots", action="store_true")
    p.add_argument("--error_bars", choices=["ci", "sd", "beats", "seeds", "default"],
                   default="ci", help="ci: 95 %% CI over seeds, sd: SD over seeds, beats: SD "
                   "over beats (per animal) and over animals (all)")
    add_style_args(p)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    print("font:", style_from_args(args))

    paths = sorted({f for pat in args.files for f in (glob.glob(pat) or [pat])})
    frames = []
    for f in paths:
        d = pd.read_csv(f)
        d = d.assign(model=d["arch"].map(ARCH_SHORT).fillna(d["arch"]) + ":" + d["config"],
                     file=os.path.splitext(os.path.basename(f))[0])
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    players = [c for c in frames[0].columns if c not in META + ["model", "file"]]
    for f, d in zip(paths, frames):
        if [c for c in d.columns if c not in META + ["model", "file"]] != players:
            raise SystemExit(f"{f}: different players than {paths[0]}; use the same "
                             f"--feature_names and --context for all files")
    for col in ("method", "background"):
        if df[col].nunique() > 1:
            print(f"WARNING: mixed {col}: {sorted(df[col].unique())} - not comparable")
    models = [m for m in args.models.split(",") if m] or sorted(df["model"].unique())
    model_index = {m: i for i, m in enumerate(models)}
    df = df[df["model"].isin(models)]
    players = ordered(players)
    styles = line_styles(players)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- plots over time, one per model and animal (mean over seeds) ------
    include = [c.strip() for c in args.time_include.split(",") if c.strip()]
    unknown = [c for c in include if c not in players]
    if unknown:
        print(f"WARNING: --time_include not found among the players: {unknown}")
    if not args.no_time_plots:
        for (m, target), d in df.groupby(["model", "target"]):
            r = d.iloc[0]
            plot_time(d, players, args.top_k, styles,
                      os.path.join(args.out_dir, f"time_{m.replace(':', '-')}_{target}.png"),
                      f"{m}, target {target} ({r['method']}, background {r['background']})",
                      include=include)

    # ---- aggregation per animal and condition -----------------------------
    lab = df[df["condition"] != "unknown"]
    if lab.empty:
        print("No intervention labels found (condition = unknown); only time plots written.")
        return
    long = lab.melt(id_vars=["model", "target", "condition", "seed"], value_vars=players,
                    var_name="player", value_name="shap")
    long["abs"] = long["shap"].abs()
    g = long.groupby(["model", "target", "condition", "player"])
    per_animal = pd.DataFrame({
        "mean_abs": g["abs"].mean(), "sd_abs": g["abs"].std(), "var_abs": g["abs"].var(),
        "mean": g["shap"].mean(), "sd": g["shap"].std(), "var": g["shap"].var(),
        "n_beats": g["shap"].size()}).reset_index()
    sd_pa, sd_over = seed_sd(long, "player")
    per_animal = per_animal.merge(sd_pa, on=["model", "target", "condition", "player"],
                                  how="left")
    per_animal.to_csv(os.path.join(args.out_dir, "shap_by_condition_animal.csv"), index=False)

    g2 = per_animal.groupby(["model", "condition", "player"])["mean_abs"]
    over_animals = pd.DataFrame({"mean_abs": g2.mean(), "sd_abs": g2.std(),
                                 "var_abs": g2.var(), "n_animals": g2.size()}).reset_index()
    over_animals = over_animals.merge(sd_over, on=["model", "condition", "player"], how="left")
    over_animals.to_csv(os.path.join(args.out_dir, "shap_by_condition_all.csv"), index=False)

    agree = seed_agreement(df, players)
    if not agree.empty:
        agree.to_csv(os.path.join(args.out_dir, "shap_seed_agreement.csv"), index=False)
        print("Seed agreement (mean over animals):")
        print(agree.groupby("model")[["spearman_mean", "spearman_min", "rel_sd_median"]]
              .mean().round(3).to_string())

    (err_a, txt_a), (err_o, txt_o) = error_columns(args.error_bars,
                                                   int(per_animal["n_seeds"].max()))

    for target, d in per_animal.groupby("target"):
        conds = condition_order(d["condition"])
        plot_conditions(d, [m for m in models if m in set(d["model"])], players,
                        conds, "mean_abs", err_a,
                        os.path.join(args.out_dir, f"conditions_{target}.png"),
                        f"{target}: mean |SHAP| per condition (error bars: {txt_a})",
                        model_index=model_index)
    if per_animal["target"].nunique() > 1:
        conds = condition_order(over_animals["condition"])
        n_anim = int(per_animal["target"].nunique())
        plot_conditions(over_animals, models, players, conds, "mean_abs", err_o,
                        os.path.join(args.out_dir, "conditions_all.png"),
                        f"Mean over {n_anim} animals of the mean |SHAP| per condition "
                        f"(error bars: {txt_o})", model_index=model_index)

    print(f"{len(paths)} files, models: {', '.join(models)}, players: {len(players)}")
    print(f"written to {args.out_dir}/")


if __name__ == "__main__":
    main()