"""Summarise the simulation runs and apply the pre-specified descriptive criteria.

Reads results/simulation/*.csv written by run_simulation.py and writes
summary_dev.csv (development corpus, seed 0) and summary_confirmation.csv
(mean over the confirmation corpora: seeds 1-5 for V0-V5 and the generic models,
seeds 6-10 for the exploratory variants).
"""
import argparse
from pathlib import Path

import pandas as pd

from _common import RESULTS
from dstfusion.evaluation import criteria_verdict

COLS = ["acc", "brier", "logloss", "ece", "auc", "aurc", "errdet_1mp", "errdet_mTheta"]


def read(d, names, explo_names):
    """V0-V5 and the generic models come from `names`; only the E1-E4 rows come from `explo_names`
    (the exploratory files also contain the generic models, evaluated on other corpora)."""
    main = pd.concat([pd.read_csv(d / n) for n in names])
    explo = pd.concat([pd.read_csv(d / n) for n in explo_names])
    return pd.concat([main, explo[explo.model.str.startswith("E")]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=RESULTS / "simulation")
    d = ap.parse_args().dir
    dev = read(d, ["dev_seed0.csv"], ["explo_dev_seed0.csv"])
    conf = read(d, [f"conf_seed{s}.csv" for s in range(1, 6)], [f"explo_conf_seed{s}.csv" for s in range(6, 11)])
    for name, D, title in [("summary_dev.csv", dev, "DEVELOPMENT (seed 0, 5x10 CV)"),
                           ("summary_confirmation.csv", conf, "CONFIRMATION (mean over corpora)")]:
        T = D.groupby("model")[COLS].mean().round(3)
        T.to_csv(d / name)
        print(f"\n== {title} ==")
        print(criteria_verdict(T).to_string())


if __name__ == "__main__":
    main()
