"""The vectorised classifier against the general engine, plus basic behaviour."""
import numpy as np
import pytest

from dstfusion import DSTFusionClassifier, core
from dstfusion.simulate import generate, observe


@pytest.fixture(scope="module")
def corpus():
    X, y, _ = generate(labels="ada", seed=3)
    return observe(X, noise=0.08, missing=0.10, seed=1003), y


def reference_masses(clf, x):
    """Fuse one case with the general engine from the fitted parameters of `clf`."""
    P = clf.P_
    K = len(P["classes"])
    Q = clf._post(P, x[None, :])[0]            # (d, K) per-variable posterior (NaN where missing)
    acc = None
    for j in range(len(x)):
        if np.isnan(x[j]):
            continue                            # missing value: vacuous source, skipped
        mc = (1 - P["ign"][j]) * Q[j]
        m = {frozenset([c]): mc[c] for c in range(K)}
        m[frozenset(range(K))] = P["ign"][j]
        m = core.discount(m, float(np.clip(P["rel"][j], 0, 1)), K)
        acc = m if acc is None else core.dempster(acc, m)[0]
    mc = np.array([acc.get(frozenset([c]), 0.0) for c in range(K)])
    return mc, acc[frozenset(range(K))]


def test_v0_matches_general_engine(corpus):
    X, y = corpus
    clf = DSTFusionClassifier("dempster", "none").fit(X[:250], y[:250])
    mc, mt = clf.masses(X[250:])
    for i in range(0, len(mc), 6):
        rc, rt = reference_masses(clf, X[250 + i])
        assert np.max(np.abs(mc[i] - rc)) < 1e-9
        assert abs(mt[i] - rt) < 1e-9


def test_masses_are_a_valid_bba(corpus):
    X, y = corpus
    for kw in [dict(), dict(rule="yager"), dict(lam_mode="learn"), dict(ign_shrink=False, lam_mode="learn"),
               dict(ign_shrink=False, lam_mode="learn", prior="learn")]:
        clf = DSTFusionClassifier(**kw).fit(X[:250], y[:250])
        mc, mt = clf.masses(X[250:])
        assert np.all(mc >= -1e-12) and np.all(mt >= -1e-12)
        assert np.allclose(mc.sum(1) + mt, 1.0)
        P = clf.predict_proba(X[250:])
        assert np.allclose(P.sum(1), 1.0)
        assert set(clf.predict(X[250:])) <= set(clf.classes_)


def test_decision_is_argmax_of_class_mass(corpus):
    """On this BBA family argmax Bel = argmax Pl = argmax BetP."""
    X, y = corpus
    clf = DSTFusionClassifier().fit(X[:250], y[:250])
    mc, mt = clf.masses(X[250:])
    assert np.array_equal(clf.predict_proba(X[250:]).argmax(1), mc.argmax(1))


def test_all_missing_case_is_vacuous(corpus):
    X, y = corpus
    clf = DSTFusionClassifier().fit(X[:250], y[:250])
    mc, mt = clf.masses(np.full((1, X.shape[1]), np.nan))
    assert mt[0] == pytest.approx(1.0) and np.allclose(mc, 0.0)


def test_no_test_information_is_used(corpus):
    """Fitting twice on the same training data gives the same masses (no hidden state)."""
    X, y = corpus
    a = DSTFusionClassifier(lam_mode="learn").fit(X[:250], y[:250]).masses(X[250:])
    b = DSTFusionClassifier(lam_mode="learn").fit(X[:250], y[:250]).masses(X[250:])
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_prior_weight_in_grid(corpus):
    X, y = corpus
    clf = DSTFusionClassifier(lam_mode="learn", ign_shrink=False, prior="learn").fit(X[:250], y[:250])
    assert clf.sp_ in clf.sp_grid
