"""Simulation experiments (protocol: protocols/PROTOCOL_SIMULATION.md).

Single run:
    python experiments/run_simulation.py --seed 0 --reps 10 --out results/simulation/dev_seed0.csv
    python experiments/run_simulation.py --seed 6 --reps 4 --exploratory --out results/simulation/explo_conf_seed6.csv

Whole protocol (development corpus, five confirmation corpora, exploratory variants):
    python experiments/run_simulation.py --suite protocol [--out-dir results/simulation]
"""
import argparse
from pathlib import Path

from _common import RESULTS
from dstfusion.evaluation import EXPLORATORY, VARIANTS, run_cv
from dstfusion.simulate import generate, observe

REPORT_COLS = ["acc", "brier", "logloss", "ece", "auc", "aurc", "errdet_1mp", "errdet_mTheta"]


def run(seed, noise, missing, reps, folds, variants, labels="ada"):
    """One simulated corpus (n=312), observed with noise and missing values, evaluated by CV."""
    X, y, _ = generate(labels=labels, seed=seed)
    Xo = observe(X, noise, missing, seed=seed + 1000)
    D = run_cv(Xo, y, variants, folds=folds, reps=reps)
    D["seed"], D["noise"], D["missing"] = seed, noise, missing
    return D


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.08)
    ap.add_argument("--missing", type=float, default=0.10)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--exploratory", action="store_true", help="run E1-E4 instead of V0-V5")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--suite", choices=["protocol"])
    ap.add_argument("--out-dir", type=Path, default=RESULTS / "simulation")
    a = ap.parse_args()

    if a.suite == "protocol":
        a.out_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(0, 10, VARIANTS, "dev_seed0")]
        jobs += [(s, 4, VARIANTS, f"conf_seed{s}") for s in range(1, 6)]
        jobs += [(0, 10, EXPLORATORY, "explo_dev_seed0")]
        jobs += [(s, 4, EXPLORATORY, f"explo_conf_seed{s}") for s in range(6, 11)]
        for seed, reps, var, name in jobs:
            D = run(seed, a.noise, a.missing, reps, a.folds, var)
            D.to_csv(a.out_dir / f"{name}.csv", index=False)
            print("wrote", a.out_dir / f"{name}.csv")
        return

    if a.out is None:
        ap.error("--out is required unless --suite is used")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    D = run(a.seed, a.noise, a.missing, a.reps, a.folds, EXPLORATORY if a.exploratory else VARIANTS)
    D.to_csv(a.out, index=False)
    print(D.groupby("model")[REPORT_COLS].mean().round(3).to_string())


if __name__ == "__main__":
    main()
