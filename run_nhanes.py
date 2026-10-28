"""NHANES 2015-2016, three classes (protocol: protocols/PROTOCOL_NHANES.md).

    python experiments/run_nhanes.py [--out-dir results/nhanes]
"""
import argparse
from pathlib import Path

from _common import NHANES_CSV, RESULTS
from dstfusion.evaluation import (GENERIC, NHANES_FEATURE_SETS, REAL_DATA_VARIANTS, criteria_verdict,
                                  dummy_prior, load_nhanes, nadeau_bengio, run_cv)

COLS = ["acc", "bacc", "f1macro", "brier", "logloss", "ece", "auc", "aurc", "errdet_1mp", "errdet_mTheta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=NHANES_CSV)
    ap.add_argument("--out-dir", type=Path, default=RESULTS / "nhanes")
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    for name, features in NHANES_FEATURE_SETS.items():
        X, y = load_nhanes(a.data, features)
        D = run_cv(X, y, REAL_DATA_VARIANTS, reps=a.reps, extra_models={"Dummy_prior": dummy_prior()})
        D.to_csv(a.out_dir / f"nhanes_{name}_folds.csv", index=False)
        T = D.groupby("model")[COLS].mean()
        T["lam_mean"] = D.groupby("model").lam.mean()
        T.round(3).to_csv(a.out_dir / f"nhanes_{name}_summary.csv")
        print(f"\n=== {name} ({len(features)} variables, n={len(y)}) ===")
        print(T.round(3).to_string())
        best_acc = T.loc[GENERIC, "acc"].idxmax()
        best_brier = T.loc[GENERIC, "brier"].idxmin()
        V = criteria_verdict(T)
        for v in REAL_DATA_VARIANTS:       # 2 variants x 2 metrics = 4 comparisons (Bonferroni)
            da, pa = nadeau_bengio(D, v, best_acc, "acc", 4)
            db, pb = nadeau_bengio(D, v, best_brier, "brier", 4)
            dc, pc = nadeau_bengio(D, v, best_acc, "bacc", 4)
            c = V.loc[v]
            print(f"{v}: c1={c.c1} c2={c.c2} c3={c.c3} | dAcc vs {best_acc}={da:+.3f} p_bonf={pa:.3f}"
                  f" | dBalAcc={dc:+.3f} p_bonf={pc:.3f} | dBrier vs {best_brier}={db:+.3f} p_bonf={pb:.3f}")


if __name__ == "__main__":
    main()
