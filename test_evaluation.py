import numpy as np
import pandas as pd
import pytest

from dstfusion import evaluation as ev
from dstfusion.simulate import generate, observe


def test_metrics_perfect_and_uniform():
    y = np.array([0, 1, 2, 0, 1, 2])
    P = np.eye(3)[y]
    r = ev.metrics(P, y)
    assert r["acc"] == 1 and r["brier"] == 0 and r["ece"] == 0 and r["auc"] == 1
    U = np.full((6, 3), 1 / 3)
    assert ev.metrics(U, y)["brier"] == pytest.approx(2 / 3)       # sum over classes


def test_binary_brier_is_twice_the_usual_value():
    y = np.array([0, 1, 1, 0])
    P = np.array([[.8, .2], [.3, .7], [.4, .6], [.9, .1]])
    usual = np.mean((P[:, 1] - y) ** 2)
    assert ev.metrics(P, y)["brier"] == pytest.approx(2 * usual)


def test_error_detection_uses_uncertainty_score():
    y = np.array([0, 0, 1, 1, 0, 1])
    P = np.array([[.9, .1], [.8, .2], [.7, .3], [.2, .8], [.4, .6], [.1, .9]])   # errors: cases 2 and 4
    unc = np.array([0, 0, 1, 0, 1, 0.])
    assert ev.metrics(P, y, unc)["errdet_mTheta"] == 1.0


def test_nadeau_bengio_detects_a_clear_difference_and_only_that():
    rng = np.random.default_rng(0)
    n = 50
    D = pd.DataFrame({"model": ["a"] * n + ["b"] * n, "fold": list(range(n)) * 2,
                      "acc": np.r_[0.80 + 0.01 * rng.standard_normal(n), 0.70 + 0.01 * rng.standard_normal(n)]})
    d, p = ev.nadeau_bengio(D, "a", "b", "acc")
    assert d == pytest.approx(0.10, abs=0.01) and p < 1e-6
    D["acc"] = rng.standard_normal(2 * n) * 0.01 + 0.75
    assert ev.nadeau_bengio(D, "a", "b", "acc")[1] > 0.05
    assert ev.nadeau_bengio(D, "a", "b", "acc", n_comparisons=1000)[1] == 1.0


def test_criteria_verdict_thresholds():
    T = pd.DataFrame({"acc": [.85, .84, .80, .83, .84], "brier": [.20, .21, .30, .22, .23],
                      "errdet_1mp": [.80, .79, .78, .77, np.nan], "errdet_mTheta": [np.nan] * 4 + [.79]},
                     index=["LogReg", "RandForest", "HistGB", "GaussNB", "X"])
    V = ev.criteria_verdict(T)
    assert bool(V.loc["X", "c1"])                  # 0.23 <= 0.20 + 0.03
    assert bool(V.loc["X", "c2"])                  # 0.84 >= 0.85 - 0.015
    assert bool(V.loc["X", "c3"])                  # 0.79 >= 0.65 and >= 0.80 - 0.02
    T.loc["X", "acc"] = 0.83
    assert not bool(ev.criteria_verdict(T).loc["X", "c2"])
    T.loc["X", "errdet_mTheta"] = 0.70
    assert not bool(ev.criteria_verdict(T).loc["X", "c3"])


def test_simulation_is_deterministic_and_has_expected_shape():
    X1, y1, _ = generate(seed=5)
    X2, y2, _ = generate(seed=5)
    assert X1.shape == (312, 10) and np.array_equal(X1, X2) and np.array_equal(y1, y2)
    assert set(y1) == {0, 1, 2}
    Xo = observe(X1, 0.08, 0.10, seed=1005)
    assert 0.07 < np.isnan(Xo).mean() < 0.13


def test_run_cv_small():
    X, y, _ = generate(seed=0)
    Xo = observe(X, 0.08, 0.10, seed=1000)
    D = ev.run_cv(Xo, y, {"V0_dempster": ev.VARIANTS["V0_dempster"]}, folds=3, reps=1)
    assert set(D.model) == {"V0_dempster", *ev.GENERIC}
    assert len(D) == 3 * 5 and D.acc.between(0, 1).all()
