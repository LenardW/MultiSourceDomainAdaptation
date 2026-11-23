#!/usr/bin/env python3
"""
Ablation table relative to the fixed default configuration of msda_mlp.py.

Every run is labelled by the settings in which it differs from the defaults
("default" if none). For each variant and configuration, the table shows the test RMSE
per target (mean over seeds), the mean over targets, the mean ratio to the default
source-only model (every target counts equally), on how many targets it is better than
the default source-only model, and the mean standard deviation over seeds.

Usage:
    python ablation_table.py "nr_*.csv"
    python ablation_table.py "nr_*.csv" --configs mcd_drt --save table_mcd.csv
"""
import argparse
import glob

import numpy as np
import pandas as pd

DEFAULTS = {"source_mode": "pooled", "norm": "global", "target_bn": "source", "K": 4,
            "discrepancy": "bounded", "num_k": 4, "eta": 0.1, "gamma": 0.0,
            "warmup_epochs": 100, "ramp_epochs": 20, "epochs": 150,
            "st_keep_quantile": 0.5, "st_weight": 1.0, "drop_features": "none"}
FILL = {"target_bn": "source", "n_blocks": 50, "vpen_cap": 1.0, "ramp_epochs": 20,
        "st_keep_quantile": 0.5, "st_weight": 1.0, "drop_features": "none"}


def differs(a, b):
    if isinstance(b, str):
        return str(a) != b
    return not np.isclose(float(a), float(b))


def variant_label(row):
    parts = [f"{k}={row[k]}" for k, v in DEFAULTS.items() if differs(row[k], v)]
    if float(row["gamma"]) > 0:
        parts.append(f"vpen_cap={row['vpen_cap']}")
    return ", ".join(parts) if parts else "default"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--configs", default="", help="only these configurations")
    ap.add_argument("--save", default="", help="write the table to this CSV file")
    args = ap.parse_args()

    frames = []
    for pat in args.files:
        for f in sorted(glob.glob(pat)):
            d = pd.read_csv(f)
            for c, v in FILL.items():
                if c not in d:
                    d[c] = v
            frames.append(d)
    if not frames:
        raise SystemExit("no files found")
    df = pd.concat(frames, ignore_index=True)
    df["drop_features"] = df["drop_features"].fillna("none").astype(str)
    df = df[(df["protocol"] == "loao") & (df["split"] == "block") & (df["n_blocks"] == 50)]
    df["variant"] = df.apply(variant_label, axis=1)

    g = df.groupby(["variant", "config", "target"])["test_rmse"]
    mean = g.mean().unstack("target")
    std = g.std().unstack("target")

    if ("default", "source_only") not in mean.index:
        raise SystemExit("the default source-only runs are missing (phase A)")
    ref = mean.loc[("default", "source_only")]

    table = mean.round(1)
    table["mean"] = mean.mean(axis=1).round(1)
    table["ratio"] = (mean / ref).mean(axis=1).round(3)
    table["better"] = (mean < ref).sum(axis=1).astype(str) + "/" + mean.notna().sum(axis=1).astype(str)
    table["seed_std"] = std.mean(axis=1).round(1)

    if args.configs:
        keep = args.configs.split(",")
        table = table[table.index.get_level_values("config").isin(keep)]
    table = table.sort_values("ratio")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(table.to_string())
    if args.save:
        table.to_csv(args.save)


if __name__ == "__main__":
    main()