"""
Loading conditions per animal, counted as msda_conv_interv.py uses them, to decide which
animals and intervention classes to use for architecture 4.

Per animal: labelled beats (stroke work >= 0) per condition. baseline = intervention_phase 1
(whatever the intervention), otherwise the intervention code. All codes in the files are
listed, not only those of --interv_classes.

Per class: in how many animals it occurs with at least --min_beats beats. With leave-one-
animal-out, a class must occur in at least --min_animals animals so that at least
--min_animals - 1 source animals still show it when one of them is the target; classes
below that are suggested for --ignore_interv. The rule depends only on the labels, not on
any result, so it can be applied before training.

Usage:
    python interv_counts.py --data_dir ../../Tierdaten \
        --animals anim3,anim4,anim8,anim10,anim11,anim16,anim20,anim7,anim13,anim19
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="../../Tierdaten")
    p.add_argument("--animals", required=True, help="comma-separated, e.g. anim3,anim4")
    p.add_argument("--interv_pattern", default="I_{aid}.csv")
    p.add_argument("--interv_col", default="intervention")
    p.add_argument("--phase_col", default="intervention_phase")
    p.add_argument("--interv_classes", default="1,2,3,4,10",
                   help="classes of msda_conv_interv.py (others are ignored there)")
    p.add_argument("--min_beats", type=int, default=50,
                   help="beats needed for a class to count as present in an animal")
    p.add_argument("--min_animals", type=int, default=3)
    args = p.parse_args()

    rows = {}
    for aid in [a for a in args.animals.split(",") if a]:
        path = os.path.join(args.data_dir, args.interv_pattern.format(aid=aid))
        if not os.path.exists(path):
            print(f"{aid}: {path} missing")
            continue
        df = pd.read_csv(path)
        y = np.loadtxt(os.path.join(args.data_dir, f"Y_{aid}.csv"), delimiter=",", ndmin=1)
        if len(df) != len(y):
            print(f"{aid}: {len(df)} rows in {path}, but {len(y)} in Y_{aid}.csv")
            continue
        lab = y >= 0
        phase = df[args.phase_col].to_numpy()
        interv = df[args.interv_col].to_numpy()
        cond = np.where(phase == 1, "baseline",
                        pd.Series(interv).map(lambda v: f"I{int(v)}" if pd.notna(v) else "none"))
        counts = pd.Series(cond[lab]).value_counts()
        counts["beats"] = int(lab.sum())
        rows[aid] = counts
    if not rows:
        raise SystemExit("no intervention files found")

    table = pd.DataFrame(rows).T.fillna(0).astype(int)
    used = ["baseline"] + [f"I{int(float(c))}" for c in args.interv_classes.split(",") if c]
    other = sorted(c for c in table.columns if c not in used + ["beats"])
    table = table.reindex(columns=["beats"] + used + other, fill_value=0)
    pd.set_option("display.width", 200)
    print("Labelled beats per animal and condition"
          + (f" (not in --interv_classes: {', '.join(other)})" if other else "") + ":")
    print(table.to_string())

    print(f"\nAnimals with at least {args.min_beats} beats of a class:")
    ignore = []
    for c in used:
        have = table.index[table[c] >= args.min_beats].tolist()
        flag = "" if len(have) >= args.min_animals else "  -> too rare"
        print(f"  {c:9s} {len(have):2d}: {', '.join(have)}{flag}")
        if c != "baseline" and len(have) < args.min_animals:
            ignore.append(c[1:])
    if ignore:
        print(f"\nSuggestion: --ignore_interv {','.join(ignore)}")
    else:
        print("\nAll classes occur in enough animals.")


if __name__ == "__main__":
    main()