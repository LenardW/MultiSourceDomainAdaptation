"""
Ablation table across all architectures (msda_mlp.py, msda_conv.py, msda_conv_ctx.py,
msda_conv_interv.py).

Each argument is a directory (all CSV files in it), a CSV file or a glob pattern. Files
without result columns (SHAP outputs etc.) are skipped. Every run is assigned to a model:
mlp, conv, ctx or interv. Runs of the conv, ctx or interv script with kernel_size 1 and
window 1 become mlp_conv, mlp_ctx and mlp_interv. Each run is labelled by the settings in
which it differs from that model's defaults ("default" if none).

Source-only and oracle runs contain no DRT layer and no adaptation. They depend only on the
backbone (MLP or temporal convolution) and on the settings that change it. Copies of them
from another script are therefore the same model as the backbone's own default run and are
listed as "re-run"; their deviation from it is run-to-run noise. A source-only model with
an auxiliary head (aux_weight > 0) is a genuine variant.

Data export: the standard deviation of the stroke work in the test set, SD = RMSE / sqrt(1 - R2),
depends only on the data and the test split, not on the model. It is identical for all runs on
the same export and 10 times larger in the new export. --data new keeps only the runs with the
larger SD per target (--data old the smaller). Without --data, a mix of exports or of test sets
is reported as a warning.

Columns
  RMSE <target>      test RMSE, mean over runs; mean = mean over targets
  vs MLP so ratio    mean over targets of RMSE / RMSE of the default MLP source-only model
  vs MLP so better   targets on which the RMSE is lower than that of this model
  vs own so          the same against the default source-only model of the row's backbone
                     (MLP: msda_mlp.py; temporal convolution: msda_conv.py)
  R2 <target>        test R2, mean over runs, not capped (* = below -1)
  R2 mean            mean over targets after capping each target's mean at -1
  seed_sd            mean over targets of the standard deviation over runs
  n                  runs per target

Usage:
    python ablation_all.py results/mlp results/conv results/ctx results/interv
    python ablation_all.py results/mlp results/conv results/ctx results/interv --data new
    python ablation_all.py results/mlp results/conv "results/interv/interv_*.csv" --exclude anim1
    python ablation_all.py results/* --configs source_only,mcd_drt,oracle --out table.txt --csv table.csv
"""
import argparse
import datetime
import glob
import os
import re

import numpy as np
import pandas as pd

DEFAULTS = {"source_mode": "pooled", "norm": "global", "target_bn": "source", "K": 4,
            "discrepancy": "bounded", "num_k": 4, "eta": 0.1, "gamma": 0.0,
            "warmup_epochs": 100, "ramp_epochs": 20, "epochs": 150,
            "st_keep_quantile": 0.5, "st_weight": 1.0, "drop_features": "none"}
_CONV = {"kernel_size": 3, "phi_kernel": 1, "channels": "256,128", "window": 5}
_CTX = {**_CONV, "ctx_len": 50, "ctx_stats": "mean"}
ARCH_DEFAULTS = {"mlp_drt_linear": {},
                 "conv_drt": _CONV,
                 "conv_drt_ctx": _CTX,
                 # with --route_sup expert the DRT runs use K = number of classes (K_used)
                 "conv_drt_interv": {**_CTX, "route_sup": "expert", "route_weight": 1.0,
                                     "aux_weight": 0.0, "ignore_interv": "none"}}
BASE_MODEL = {"mlp_drt_linear": "mlp", "conv_drt": "conv", "conv_drt_ctx": "ctx",
              "conv_drt_interv": "interv"}
# columns of older result files that did not exist yet (their value was the default)
FILL = {"arch": "mlp_drt_linear", "protocol": "loao", "split": "block", "n_blocks": 50,
        "target_bn": "source", "vpen_cap": 1.0, "ramp_epochs": 20, "st_keep_quantile": 0.5,
        "st_weight": 1.0, "drop_features": "none"}
REQUIRED = ["config", "target", "seed", "test_rmse", "test_r2"]
# settings that change a model without DRT layers and adaptation (source-only, oracle)
BASELINE_KEYS = ["source_mode", "norm", "target_bn", "epochs", "drop_features",
                 "kernel_size", "window", "channels", "aux_weight"]
BASELINE_CONFIGS = ("source_only", "oracle")
CONFIG_ORDER = ["source_only", "drt", "mcd", "mcd_drt", "mcd_drt_st", "oracle"]
MODEL_ORDER = ["mlp", "mlp_conv", "mlp_ctx", "mlp_interv", "conv", "ctx", "interv"]
R2_CAP = -1.0


def missing(v):
    return v is None or (isinstance(v, (float, np.floating)) and np.isnan(v))


def differs(a, b):
    if missing(a):
        return False
    if isinstance(b, str):
        return fmt(a) != b
    try:
        return not np.isclose(float(a), float(b))
    except (TypeError, ValueError):
        return True


def fmt(v):
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def is_one(v):
    try:
        return float(v) == 1.0
    except (TypeError, ValueError):
        return False


def model_of(row):
    base = BASE_MODEL.get(row["arch"], str(row["arch"]))
    if base != "mlp" and is_one(row.get("kernel_size")) and is_one(row.get("window")):
        return "mlp_" + base
    return base


def backbone_of(model):
    return "mlp" if model == "mlp" or model.startswith("mlp_") else "conv"


def model_defaults(arch, model):
    ref = {**DEFAULTS, **ARCH_DEFAULTS.get(arch, {})}
    if model.startswith("mlp_"):
        ref.update(kernel_size=1, window=1)
    return ref


def variant_label(row):
    model = row["model"]
    ref = model_defaults(row["arch"], model)
    if row["config"] in BASELINE_CONFIGS:
        keys = [k for k in BASELINE_KEYS if k in ref]
        aux = not missing(row.get("aux_weight")) and float(row["aux_weight"]) > 0
        if aux and "ignore_interv" in ref:
            keys.append("ignore_interv")
        parts = [f"{k}={fmt(row[k])}" for k in keys if differs(row[k], ref[k])]
        label = ", ".join(parts) or "default"
        if model not in ("mlp", "conv") and not aux:
            label = "re-run" if label == "default" else "re-run, " + label
        return label
    parts = [f"{k}={fmt(row[k])}" for k, v in ref.items() if differs(row[k], v)]
    if not missing(row.get("gamma")) and float(row["gamma"]) > 0:
        parts.append(f"vpen_cap={fmt(row['vpen_cap'])}")
    return ", ".join(parts) or "default"


def collect(paths, pattern, recursive, notes):
    files = []
    for p in paths:
        if os.path.isdir(p):
            pat = os.path.join(p, "**", pattern) if recursive else os.path.join(p, pattern)
            found = sorted(glob.glob(pat, recursive=recursive))
        elif re.search(r"[*?\[]", p):
            found = sorted(glob.glob(p))
        else:
            found = [p] if os.path.isfile(p) else []
        if not found:
            notes.append(f"nothing found for {p}")
        files += found
    return list(dict.fromkeys(os.path.normpath(f) for f in files))


def load(files, notes):
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception as e:  # noqa: BLE001
            notes.append(f"skipped {f} (unreadable: {e})")
            continue
        if any(c not in d for c in REQUIRED):
            notes.append(f"skipped {f} (not a results file)")
            continue
        for c, v in FILL.items():
            if c not in d:
                d[c] = v
        for arch in d["arch"].dropna().unique():
            for c, v in {**DEFAULTS, **ARCH_DEFAULTS.get(arch, {})}.items():
                if c not in d:
                    d[c] = pd.Series([None] * len(d), dtype=object, index=d.index)
                m = (d["arch"] == arch) & d[c].isna()
                if m.any():
                    d[c] = d[c].astype(object)
                    d.loc[m, c] = v
        d["_file"] = f
        frames.append(d)
    if not frames:
        raise SystemExit("no result files found")
    return pd.concat(frames, ignore_index=True)


def check_duplicates(df, notes):
    keys = [c for c in ["arch", "protocol", "split", "n_blocks", "config", "target", "seed",
                        "sources"] if c in df]
    keys += sorted({k for a in ARCH_DEFAULTS.values() for k in a} | set(DEFAULTS))
    cols = df[[k for k in keys if k in df]]
    key = cols.apply(lambda s: s.map(lambda v: "" if missing(v) else fmt(v))).agg("|".join, axis=1)
    dup = key.duplicated(keep=False)
    if dup.any():
        files = sorted(df.loc[dup, "_file"].unique())
        notes.append(f"WARNING: {int(dup.sum())} runs share settings, configuration, target "
                     f"and seed with another run and are averaged (appended re-runs or "
                     f"different data exports?). Files: {', '.join(files)}")


def test_sd(df):
    """Standard deviation of the stroke work in the test set, recovered from RMSE and R2
    (R2 = 1 - RMSE^2 / SD^2). It depends only on the data, not on the model: it is the same
    for all runs on the same export and test split, and 10 times larger in the new export."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return df["test_rmse"] / np.sqrt(1.0 - df["test_r2"])


def select_data(df, which, notes):
    """Keep the runs of the new export (larger stroke-work scale) or of the old one."""
    sd = test_sd(df)
    keep = pd.Series(True, index=df.index)
    single = []
    for t, s in sd.groupby(df["target"]):
        lo, hi = s.min(), s.max()
        if not np.isfinite(hi) or hi / lo < 3:
            single.append(t)
            continue
        keep[s.index] = (s >= hi / 3) if which == "new" else (s <= lo * 3)
    gone = df[~keep]
    if len(gone):
        files = "; ".join(f"{f} ({n})" for f, n in gone["_file"].value_counts().sort_index().items())
        notes.append(f"--data {which}: removed {len(gone)} runs of the "
                     f"{'old' if which == 'new' else 'new'} export: {files}")
    if single:
        notes.append(f"--data {which}: only one data scale found for {', '.join(map(str, single))}; "
                     f"nothing removed there (cannot tell old from new)")
    return df[keep]


def check_test_sets(df, notes):
    sd = test_sd(df)
    for t, s in sd.groupby(df["target"]):
        lo, hi = s.min(), s.max()
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi / lo <= 1.01:
            continue
        files = sorted(df.loc[s.index[(s - lo).abs() > 0.01 * lo], "_file"].unique()) if hi / lo < 3 \
            else sorted(df.loc[s.index[s >= hi / 3], "_file"].unique())
        if hi / lo >= 3:
            notes.append(f"WARNING: {t}: runs of the old and the new data export are mixed "
                         f"(test-set SD {lo:.1f} vs {hi:.1f}); use --data new. New-export files: "
                         f"{', '.join(files)}")
        else:
            notes.append(f"WARNING: {t}: the test sets differ (test-set SD {lo:.1f} to {hi:.1f}; "
                         f"other split or export). Files: {', '.join(files)}")


def reference(df, backbone):
    own = "mlp" if backbone == "mlp" else "conv"
    so = df[df["config"] == "source_only"]
    r = so[(so["model"] == own) & (so["variant"] == "default")]
    if not r.empty:
        return r.groupby("target")["test_rmse"].mean(), f"{own} default ({len(r)} runs)"
    r = so[(so["backbone"] == backbone) & (so["variant"] == "re-run")]
    if not r.empty:
        return (r.groupby("target")["test_rmse"].mean(),
                f"no {own} default run; re-runs of {', '.join(sorted(r['model'].unique()))} "
                f"({len(r)} runs)")
    return None, "missing"


def compare(rmse, ref):
    if ref is None:
        nan = pd.Series(np.nan, index=rmse.index)
        return nan, pd.Series("", index=rmse.index)
    ref = ref.reindex(rmse.columns)
    q = rmse.div(ref, axis=1)
    valid = q.notna()
    better = (rmse.lt(ref, axis=1) & valid).sum(axis=1)
    return q.mean(axis=1), better.astype(str) + "/" + valid.sum(axis=1).astype(str)


def build(df, targets, refs):
    g = df.groupby(["model", "variant", "config", "target"])
    rmse = g["test_rmse"].mean().unstack("target").reindex(columns=targets)
    sd = g["test_rmse"].std().unstack("target").reindex(columns=targets)
    r2 = g["test_r2"].mean().unstack("target").reindex(columns=targets)
    n = g.size().unstack("target").reindex(columns=targets)

    cols = {}
    for t in targets:
        cols[("RMSE", t)] = rmse[t]
    cols[("RMSE", "mean")] = rmse.mean(axis=1)
    cols[("vs MLP so", "ratio")], cols[("vs MLP so", "better")] = compare(rmse, refs["mlp"])
    own_ratio = pd.Series(np.nan, index=rmse.index)
    own_better = pd.Series("", index=rmse.index, dtype=object)
    bb = pd.Series([backbone_of(m) for m in rmse.index.get_level_values("model")],
                   index=rmse.index)
    for b in ("mlp", "conv"):
        m = bb == b
        if m.any():
            r, bt = compare(rmse[m], refs[b])
            own_ratio[m], own_better[m] = r, bt
    cols[("vs own so", "ratio")], cols[("vs own so", "better")] = own_ratio, own_better
    for t in targets:
        cols[("R2", t)] = r2[t]
    cols[("R2", "mean")] = r2.clip(lower=R2_CAP).mean(axis=1)
    cols[("", "seed_sd")] = sd.mean(axis=1)
    lo, hi = n.min(axis=1), n.max(axis=1)
    cols[("", "n")] = [f"{int(a)}" if a == b else f"{int(a)}-{int(b)}" for a, b in zip(lo, hi)]
    table = pd.DataFrame(cols, index=rmse.index)
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    return table


def sort_table(table, how):
    if how == "model":
        idx = table.index.to_frame(index=False)
        mk = idx["model"].map(lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)
        vk = idx["variant"].map(lambda v: 0 if v == "default" else 1 if v.startswith("re-run") else 2)
        ck = idx["config"].map(lambda c: CONFIG_ORDER.index(c) if c in CONFIG_ORDER else 99)
        order = np.lexsort([idx["variant"], ck, vk, mk])
        return table.iloc[order]
    key = ("vs MLP so", "ratio")
    if table[key].isna().all():
        key = ("vs own so", "ratio")
    return table.sort_values(key, na_position="last", kind="stable")


def render(table):
    disp = pd.DataFrame(index=table.index, columns=table.columns, dtype=object)
    for c in table.columns:
        top, sub = c
        col = table[c]
        if top == "RMSE" or sub == "seed_sd":
            f = lambda v: "" if missing(v) else f"{v:.1f}"
        elif top == "R2" and sub != "mean":
            f = lambda v: "" if missing(v) else f"{v:.3f}" + ("*" if v < R2_CAP else "")
        elif top == "R2" or sub == "ratio":
            f = lambda v: "" if missing(v) else f"{v:.3f}"
        else:
            f = str
        disp[c] = col.map(f)
    # model, variant and configuration in every row, left-aligned
    idx = table.index.to_frame(index=False).astype(str)
    labels = {}
    for name in idx.columns:
        w = max(len(name), int(idx[name].str.len().max()))
        labels[(name.ljust(w), "")] = idx[name].str.ljust(w)
    out = pd.concat([pd.DataFrame(labels), disp.reset_index(drop=True)], axis=1)
    return out.to_string(index=False)


def flat(table):
    out = table.copy()
    out.columns = [" ".join(x for x in c if x) for c in table.columns]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="directories, CSV files or glob patterns")
    ap.add_argument("--pattern", default="*.csv", help="file pattern inside directories")
    ap.add_argument("--recursive", action="store_true", help="also search subdirectories")
    ap.add_argument("--configs", default="", help="only these configurations")
    ap.add_argument("--exclude", default="",
                    help="targets left out of an additional table, e.g. anim1")
    ap.add_argument("--sort", choices=["ratio", "model"], default="ratio")
    ap.add_argument("--protocol", default="loao")
    ap.add_argument("--split", default="block")
    ap.add_argument("--n_blocks", type=int, default=50)
    ap.add_argument("--data", choices=["all", "new", "old"], default="all",
                    help="keep only runs of the new export (stroke work x10) or of the old one; "
                         "detected per run from the test-set SD = RMSE / sqrt(1 - R2)")
    ap.add_argument("--out", default="ablation_all.txt", help="text file with the table(s)")
    ap.add_argument("--csv", default="", help="also write the table(s) as CSV")
    args = ap.parse_args()

    notes = []
    files = collect(args.paths, args.pattern, args.recursive, notes)
    df = load(files, notes)
    n0 = len(df)
    nb = pd.to_numeric(df["n_blocks"], errors="coerce")
    df = df[(df["protocol"] == args.protocol) & (df["split"] == args.split)
            & (nb == args.n_blocks)].copy()
    if df.empty:
        raise SystemExit("no runs left after filtering protocol, split and n_blocks")
    df["model"] = df.apply(model_of, axis=1)
    df["backbone"] = df["model"].map(backbone_of)
    df["variant"] = df.apply(variant_label, axis=1)
    if args.configs:
        df = df[df["config"].isin(args.configs.split(","))]
    if args.data != "all":
        df = select_data(df, args.data, notes)
    check_duplicates(df, notes)
    check_test_sets(df, notes)

    def natural(t):
        m = re.search(r"\d+", str(t))
        return (int(m.group()) if m else 0, str(t))

    targets = sorted(df["target"].unique(), key=natural)
    excl = [t for t in args.exclude.split(",") if t]
    sections = [("all targets", targets)]
    if excl:
        sections.append((f"without {', '.join(excl)}", [t for t in targets if t not in excl]))

    head = [f"Ablation table, all architectures ({datetime.datetime.now():%Y-%m-%d %H:%M})",
            f"filter: protocol={args.protocol}, split={args.split}, n_blocks={args.n_blocks} "
            f"({n0 - len(df)} of {n0} rows dropped"
            + (f"; configurations: {args.configs})" if args.configs else ")"),
            "files:"]
    head += [f"  {f} ({c} runs)" for f, c in df["_file"].value_counts().sort_index().items()]

    body, csv_tables = [], []
    for title, tg in sections:
        sub = df[df["target"].isin(tg)]
        refs, desc = {}, []
        for b in ("mlp", "conv"):
            refs[b], d = reference(sub, b)
            desc.append(f"{b} backbone source-only reference: {d}")
        if refs["mlp"] is None:
            desc.append("no MLP source-only run: the 'vs MLP so' columns are empty")
        table = sort_table(build(sub, tg, refs), args.sort)
        body += ["", "=" * 20 + f" {title} " + "=" * 20, *desc, "", render(table)]
        csv_tables.append((title, table))

    legend = ["", "Columns",
              "  RMSE: test RMSE per target, mean over runs; mean = mean over targets",
              "  vs MLP so: ratio = mean over targets of RMSE / RMSE of the default MLP "
              "source-only model; better = targets with a lower RMSE",
              "  vs own so: the same against the default source-only model of the row's "
              "backbone (MLP: msda_mlp.py, temporal convolution: msda_conv.py)",
              f"  R2: mean over runs per target, not capped (* = below {R2_CAP:g}); "
              f"mean = mean over targets after capping each target at {R2_CAP:g}",
              "  seed_sd: mean over targets of the standard deviation over runs; "
              "n: runs per target",
              "  re-run: source-only/oracle run of another script; the same model as the "
              "backbone's own run (differences = run-to-run noise)"]
    text = "\n".join(head + ([""] + notes if notes else []) + body + legend) + "\n"
    print(text)
    with open(args.out, "w") as fh:
        fh.write(text)
    print(f"written: {args.out}")
    if args.csv:
        stem, ext = os.path.splitext(args.csv)
        for i, (title, table) in enumerate(csv_tables):
            path = args.csv if i == 0 else f"{stem}_wo_{'_'.join(excl)}{ext or '.csv'}"
            flat(table).to_csv(path)
            print(f"written: {path}")


if __name__ == "__main__":
    main()