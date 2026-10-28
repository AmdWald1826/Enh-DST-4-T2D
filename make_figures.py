"""Figures of the manuscript: reliability diagrams and error rate by uncertainty quintile.

Out-of-fold predictions are computed with 5x5 repeated stratified CV (random_state=42)
for the simulated corpus (seed 0), Pima and NHANES, and stored in
results/figures/oof_predictions.npz. Figures go to figures/.

    python experiments/make_figures.py
"""
import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import FIGURES, NHANES_CSV, PIMA_CSV, RESULTS
from dstfusion import DSTFusionClassifier
from dstfusion.evaluation import cv_splits, generic_models, load_nhanes, load_pima
from dstfusion.simulate import generate, observe

warnings.filterwarnings("ignore")
REPS = 5
MODELS = {"DST-Fusion (V0)": lambda: DSTFusionClassifier("dempster", "none"),
          "E3 (no double count)": lambda: DSTFusionClassifier("dempster", "learn", ign_shrink=False)}
COLOR = {"DST-Fusion (V0)": "#c0392b", "E3 (no double count)": "#e67e22", "LogReg": "#2c3e50"}
NAME = {"sim": "Simulated", "pima": "Pima", "nhanes": "NHANES 2015-16"}


def load(name):
    if name == "sim":
        X, y, _ = generate(labels="ada", seed=0)
        return observe(X, .08, .10, seed=1000), y
    return load_pima(PIMA_CSV) if name == "pima" else load_nhanes(NHANES_CSV)


def out_of_fold(ds):
    X, y = load(ds)
    res = {m: dict(P=[], mt=[], y=[]) for m in list(MODELS) + ["LogReg"]}
    for tr, te in cv_splits(X, y, 5, REPS, 42):
        for m, make in MODELS.items():
            mod = make().fit(X[tr], y[tr])
            _, mt = mod.masses(X[te])
            res[m]["P"].append(mod.predict_proba(X[te])); res[m]["mt"].append(mt); res[m]["y"].append(y[te])
        lr = generic_models()["LogReg"].fit(X[tr], y[tr])
        res["LogReg"]["P"].append(lr.predict_proba(X[te])); res["LogReg"]["mt"].append(np.zeros(len(te)))
        res["LogReg"]["y"].append(y[te])
    return len(y), {m: {k: np.concatenate(v) for k, v in r.items()} for m, r in res.items()}


def main():
    FIGURES.mkdir(exist_ok=True)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    OUT, N = {}, {}
    for ds in NAME:
        N[ds], OUT[ds] = out_of_fold(ds)
    np.savez_compressed(RESULTS / "figures" / "oof_predictions.npz",
                        **{f"{ds}__{m}__{k}": v for ds, r in OUT.items() for m, d in r.items() for k, v in d.items()})
    title = {ds: f"{NAME[ds]} (n={N[ds]})" for ds in NAME}

    # Figure 1: reliability diagram with equal-mass bins
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for a, ds in zip(ax, OUT):
        lo, hi = 1.0, 0.0
        for m, r in OUT[ds].items():
            conf = r["P"].max(1); ok = (r["P"].argmax(1) == r["y"]).astype(float)
            q = np.quantile(conf, np.linspace(0, 1, 9)); idx = np.clip(np.searchsorted(q, conf, side="right") - 1, 0, 7)
            xs = [conf[idx == i].mean() for i in range(8)]; ys = [ok[idx == i].mean() for i in range(8)]
            a.plot(xs, ys, "o-", color=COLOR[m], label=m, ms=4, lw=1.5)
            lo, hi = min(lo, min(xs + ys)), max(hi, max(xs + ys))
        lo, hi = max(0, lo - .05), min(1.02, hi + .05)
        a.plot([lo, hi], [lo, hi], "k:", lw=1); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
        a.set_title(title[ds], fontsize=10); a.set_xlabel("Mean confidence (equal-mass bins)"); a.grid(alpha=.3)
    ax[0].set_ylabel("Observed accuracy"); ax[0].legend(fontsize=7, loc="lower right")
    plt.tight_layout(); plt.savefig(FIGURES / "fig_reliability.png", dpi=300); plt.close()

    # Figure 2: error rate by quintile of the uncertainty score
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for a, ds in zip(ax, OUT):
        for j, (m, key) in enumerate([("DST-Fusion (V0)", "mt"), ("E3 (no double count)", "mt"), ("LogReg", "1-p")]):
            r = OUT[ds][m]
            u = r["mt"] if key == "mt" else 1 - r["P"].max(1)
            err = (r["P"].argmax(1) != r["y"]).astype(float)
            q = np.quantile(u, [0, .2, .4, .6, .8, 1]); idx = np.clip(np.searchsorted(q, u, side="right") - 1, 0, 4)
            a.bar(np.arange(5) + (j - 1) * 0.27, [err[idx == i].mean() for i in range(5)], 0.27, color=COLOR[m],
                  label=m + (" : m($\\Theta$)" if key == "mt" else " : 1-max p"))
        a.set_title(title[ds], fontsize=10); a.set_xticks(range(5))
        a.set_xticklabels(["Q1\n(low)", "Q2", "Q3", "Q4", "Q5\n(high)"]); a.set_xlabel("Uncertainty quintile"); a.grid(axis="y", alpha=.3)
    ax[0].set_ylabel("Error rate"); ax[0].legend(fontsize=6.5, loc="upper left")
    plt.tight_layout(); plt.savefig(FIGURES / "fig_errdet.png", dpi=300); plt.close()
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
