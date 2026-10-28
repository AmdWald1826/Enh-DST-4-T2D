"""Class-prior hypothesis (protocol: protocols/PROTOCOL_CLASS_PRIOR.md).

    python experiments/run_class_prior.py pima|nhanes [--out-dir results/class_prior]
"""
import argparse
from pathlib import Path

import numpy as np

from _common import NHANES_CSV, PIMA_CSV, RESULTS
from dstfusion.evaluation import (GENERIC, PRIOR_VARIANTS, criteria_verdict, load_nhanes, load_pima,
                                  nadeau_bengio, run_cv)

COLS = ["acc", "bacc", "f1macro", "brier", "ece", "auc", "errdet_1mp", "errdet_mTheta", "sp", "mTheta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["pima", "nhanes"])
    ap.add_argument("--out-dir", type=Path, default=RESULTS / "class_prior")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_pima(PIMA_CSV) if a.dataset == "pima" else load_nhanes(NHANES_CSV)
    D = run_cv(X, y, PRIOR_VARIANTS)
    D.to_csv(a.out_dir / f"prior_{a.dataset}_folds.csv", index=False)
    T = D.groupby("model")[COLS].mean()
    T.round(3).to_csv(a.out_dir / f"prior_{a.dataset}_summary.csv")
    print(f"\n=== {a.dataset} (n={len(y)}, K={len(np.unique(y))}) ===")
    print(T.round(3).to_string())
    for v in ["P1_prior", "P2_prior_cap75"]:    # 2 variants x 3 metrics = 6 comparisons
        diffs = " | ".join(f"d{m}={nadeau_bengio(D, v, 'E3', m, 6)[0]:+.3f} (p_bonf={nadeau_bengio(D, v, 'E3', m, 6)[1]:.3f})"
                           for m in ["acc", "bacc", "brier"])
        print(v, "vs E3:", diffs, "| sp distribution:", D[D.model == v].sp.value_counts().sort_index().to_dict())
    print(criteria_verdict(T).to_string())


if __name__ == "__main__":
    main()
