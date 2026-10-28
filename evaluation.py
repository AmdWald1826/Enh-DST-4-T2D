"""Shared evaluation code: models, metrics, cross-validation loop, tests, criteria.

Everything the experiment scripts have in common lives here so that every data set is
evaluated by exactly the same code.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (enables IterativeImputer)
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .classifier import DSTFusionClassifier

# --------------------------------------------------------------------------- models
# Variants fixed in the simulation protocol (V0-V5).
VARIANTS = {
    "V0_dempster": dict(rule="dempster", lam_mode="none", dep=False),
    "V1_dempster_lam": dict(rule="dempster", lam_mode="learn", dep=False),
    "V2_yager": dict(rule="yager", lam_mode="none", dep=False),
    "V3_yager_lam": dict(rule="yager", lam_mode="learn", dep=False),
    "V4_dempster_dep_lam": dict(rule="dempster", lam_mode="learn", dep=True),
    "V5_yager_dep_lam": dict(rule="yager", lam_mode="learn", dep=True),
}
# Exploratory variants added after the V0-V5 results were seen.
EXPLORATORY = {
    "E1_dempster_pltr_lam": dict(rule="dempster", lam_mode="learn", prob="pltr"),
    "E2_dempster_pltr_dep_lam": dict(rule="dempster", lam_mode="learn", prob="pltr", dep=True),
    "E3_noshrink_betp_lam": dict(rule="dempster", lam_mode="learn", ign_shrink=False),
    "E4_noshrink_pltr_lam": dict(rule="dempster", lam_mode="learn", prob="pltr", ign_shrink=False),
}
# The two DST models carried to the real-data protocols.
REAL_DATA_VARIANTS = {
    "V0_dempster": dict(rule="dempster", lam_mode="none"),
    "E3_noshrink_betp_lam": dict(rule="dempster", lam_mode="learn", ign_shrink=False),
}
# Class-prior protocol: E3 is the reference, P1/P2 add the class frequencies as a source.
PRIOR_VARIANTS = {
    "E3": dict(rule="dempster", lam_mode="learn", ign_shrink=False),
    "P1_prior": dict(rule="dempster", lam_mode="learn", ign_shrink=False, prior="learn"),
    "P2_prior_cap75": dict(rule="dempster", lam_mode="learn", ign_shrink=False, prior="learn",
                           sp_grid=(0.0, .25, .5, .75)),
}
GENERIC = ["LogReg", "RandForest", "HistGB", "GaussNB"]


def generic_models():
    """The four reference classifiers (same settings in every experiment)."""
    imp = lambda: IterativeImputer(max_iter=10, random_state=0)
    return {
        "LogReg": make_pipeline(imp(), StandardScaler(), LogisticRegression(max_iter=2000)),
        "RandForest": make_pipeline(imp(), RandomForestClassifier(300, random_state=0, n_jobs=-1)),
        "HistGB": HistGradientBoostingClassifier(random_state=0),
        "GaussNB": make_pipeline(imp(), StandardScaler(), GaussianNB()),
    }


def dummy_prior():
    """Reference that always outputs the training class frequencies."""
    return DummyClassifier(strategy="prior")


# --------------------------------------------------------------------------- metrics
def ece(P, y, nb=10):
    """Expected calibration error on the top-label confidence (equal-width bins)."""
    conf = P.max(1)
    ok = (P.argmax(1) == y).astype(float)
    e = 0.0
    bins = np.minimum((conf * nb).astype(int), nb - 1)
    for b in range(nb):
        m = bins == b
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return e


def aurc(conf, err):
    """Area under the risk-coverage curve (lower is better)."""
    o = np.argsort(-conf)
    e = err[o]
    return np.mean(np.cumsum(e) / np.arange(1, len(e) + 1))


def metrics(P, y, unc=None):
    """All metrics for one test fold.

    P   (n, K) class probabilities;  y (n,) labels in 0..K-1;
    unc (n,) optional uncertainty score, here m(Theta), used to detect errors.
    The Brier score is summed over classes (for two classes, half of it is the usual
    binary Brier score).
    """
    K = P.shape[1]
    Y = np.eye(K)[y]
    pred = P.argmax(1)
    err = (pred != y).astype(int)
    r = dict(
        acc=1 - err.mean(),
        bacc=balanced_accuracy_score(y, pred),
        f1macro=f1_score(y, pred, average="macro"),
        brier=np.sum((P - Y) ** 2, 1).mean(),
        logloss=-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))),
        ece=ece(P, y),
        auc=roc_auc_score(y, P[:, 1]) if K == 2 else roc_auc_score(y, P, multi_class="ovr"),
        aurc=aurc(P.max(1), err),
    )
    has_both = 0 < err.sum() < len(err)
    r["errdet_1mp"] = roc_auc_score(err, 1 - P.max(1)) if has_both else np.nan
    r["errdet_mTheta"] = (roc_auc_score(err, unc)
                          if (unc is not None and has_both and np.std(unc) > 0) else np.nan)
    return r


# --------------------------------------------------------------------------- CV loop
def cv_splits(X, y, folds=5, reps=10, seed=42):
    """Repeated stratified K-fold; identical splits for every model."""
    return RepeatedStratifiedKFold(n_splits=folds, n_repeats=reps, random_state=seed).split(X, y)


def run_cv(X, y, dst_variants, folds=5, reps=10, extra_models=None, seed=42):
    """Evaluate DST variants and the generic models on the same repeated CV splits.

    Returns one row per (fold, model) with the metrics of `metrics`, plus for DST models
    the learned discount `lam`, the learned prior weight `sp` and the mean m(Theta).
    """
    warnings.filterwarnings("ignore")
    rows = []
    for i, (tr, te) in enumerate(cv_splits(X, y, folds, reps, seed)):
        for name, kw in dst_variants.items():
            m = DSTFusionClassifier(**kw).fit(X[tr], y[tr])
            _, mt = m.masses(X[te])
            r = metrics(m.predict_proba(X[te]), y[te], mt)
            r.update(model=name, fold=i, lam=m.lam_, sp=m.sp_, mTheta=mt.mean())
            rows.append(r)
        models = dict(generic_models())
        models.update(extra_models or {})
        for name, est in models.items():
            est.fit(X[tr], y[tr])
            r = metrics(est.predict_proba(X[te]), y[te])
            r.update(model=name, fold=i)
            rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- statistics
def nadeau_bengio(D, a, b, metric, n_comparisons=1, test_train_ratio=0.25):
    """Corrected resampled t-test (Nadeau & Bengio, 2003) for model a minus model b.

    D must hold one row per (model, fold). With 5-fold CV, n_test / n_train = 1/4.
    Returns (mean difference, p-value); the p-value is multiplied by n_comparisons
    (Bonferroni) and capped at 1.
    """
    xa = D[D.model == a].sort_values("fold")[metric].values
    xb = D[D.model == b].sort_values("fold")[metric].values
    x = xa - xb
    n = len(x)
    t = x.mean() / np.sqrt((1 / n + test_train_ratio) * x.var(ddof=1))
    p = 2 * stats.t.sf(abs(t), n - 1)
    return x.mean(), min(1.0, n_comparisons * p)


def criteria_verdict(T, generic=GENERIC):
    """Pre-specified descriptive criteria, applied to a table of mean metrics `T`.

    c1: Brier <= best generic Brier + 0.03
    c2: accuracy >= best generic accuracy - 0.015
    c3: AUROC of error detection by m(Theta) >= 0.65 and >= best generic (1 - max p) AUROC - 0.02
    """
    b = T.loc[generic]
    out = {}
    for v in [i for i in T.index if i not in generic and i != "Dummy_prior"]:
        c1 = T.loc[v, "brier"] <= b["brier"].min() + .03
        c2 = T.loc[v, "acc"] >= b["acc"].max() - .015
        e = T.loc[v, "errdet_mTheta"]
        c3 = bool(e >= .65 and e >= b["errdet_1mp"].max() - .02)
        out[v] = dict(brier=round(T.loc[v, "brier"], 3), c1=bool(c1), acc=round(T.loc[v, "acc"], 3),
                      c2=bool(c2), errdet=round(e, 3) if pd.notna(e) else np.nan, c3=c3)
    V = pd.DataFrame(out).T
    V["ALL"] = V[["c1", "c2", "c3"]].all(axis=1)
    return V


# --------------------------------------------------------------------------- data
PIMA_COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                "DiabetesPedigree", "Age", "Outcome"]
PIMA_ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
NHANES_CORE = ["Age", "Sex", "BMI", "Waist", "SBP", "DBP", "TotalChol", "HDL", "Triglycerides"]
NHANES_FEATURE_SETS = {"primary9": NHANES_CORE, "secondary11": NHANES_CORE + ["Copper", "Zinc"]}


def load_pima(path):
    """Pima Indians Diabetes data. Physiologically impossible zeros are coded missing."""
    d = pd.read_csv(path)
    for k in PIMA_ZERO_IS_MISSING:
        d.loc[d[k] == 0, k] = np.nan
    return d[PIMA_COLUMNS[:-1]].values.astype(float), d["Outcome"].values


def load_nhanes(path, features=NHANES_CORE):
    """NHANES 2015-2016 cohort built by experiments/build_nhanes_cohort.py."""
    d = pd.read_csv(path)
    return d[list(features)].values.astype(float), d["class"].values.astype(int)
