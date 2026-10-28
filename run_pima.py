"""Pima Indians Diabetes (protocol: protocols/PROTOCOL_PIMA.md).

    python experiments/run_pima.py [--out-dir results/pima]
"""
import argparse
from pathlib import Path

from _common import PIMA_CSV, RESULTS
from dstfusion.evaluation import (GENERIC, REAL_DATA_VARIANTS, criteria_verdict, load_pima,
                                  nadeau_bengio, run_cv)

COLS = ["acc", "brier", "logloss", "ece", "auc", "aurc", "errdet_1mp", "errdet_mTheta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=PIMA_CSV)
    ap.add_argument("--out-dir", type=Path, default=RESULTS / "pima")
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_pima(a.data)
    D = run_cv(X, y, REAL_DATA_VARIANTS, reps=a.reps)
    D.to_csv(a.out_dir / "real_pima_folds.csv", index=False)
    T = D.groupby("model")[COLS].mean()
    T["brier_binary"] = T.brier / 2
    T["lam_mean"] = D.groupby("model").lam.mean()
    T.round(3).to_csv(a.out_dir / "real_pima_summary.csv")
    print(T.round(3).to_string())

    best_acc = T.loc[GENERIC, "acc"].idxmax()
    best_brier = T.loc[GENERIC, "brier"].idxmin()
    print(f"\nBest generic accuracy: {best_acc}; best generic Brier: {best_brier}")
    V = criteria_verdict(T)
    for v in REAL_DATA_VARIANTS:
        da, pa = nadeau_bengio(D, v, best_acc, "acc")       # one comparison, uncorrected (see protocol)
        db, pb = nadeau_bengio(D, v, best_brier, "brier")
        c = V.loc[v]
        print(f"{v}: c1={c.c1} c2={c.c2} c3={c.c3} | dAcc vs {best_acc}={da:+.3f} p={pa:.3f}"
              f" | dBrier vs {best_brier}={db:+.3f} p={pb:.3f}")


if __name__ == "__main__":
    main()
