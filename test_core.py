"""Numerical checks of the Dempster-Shafer engine and of the closed form."""

import numpy as np
import pytest

from dstfusion import core


def random_simple_bbas(rng, n_sources, n_classes):
    """Random BBAs with focal sets {singletons, Theta}."""
    M = rng.dirichlet(np.ones(n_classes + 1), size=n_sources)
    return M[:, :n_classes], M[:, n_classes]


def as_dict(mc, mt, n):
    d = {frozenset([c]): v for c, v in enumerate(mc) if v > 0}
    d[frozenset(range(n))] = mt
    return d


@pytest.mark.parametrize("n_sources,n_classes", [(2, 2), (3, 3), (5, 3), (4, 4)])
def test_closed_form_matches_brute_force(n_sources, n_classes):
    rng = np.random.default_rng(n_sources * 10 + n_classes)
    for _ in range(20):
        M, Mt = random_simple_bbas(rng, n_sources, n_classes)
        s = rng.uniform(0.3, 1.0, n_sources)
        # brute force: discount each source, then combine on the power set
        acc = core.discount(as_dict(M[0], Mt[0], n_classes), s[0], n_classes)
        for i in range(1, n_sources):
            acc, _ = core.dempster(acc, core.discount(as_dict(M[i], Mt[i], n_classes), s[i], n_classes))
        mc, mt = core.fuse_simple(M, Mt, s)
        brute_c = np.array([acc.get(frozenset([c]), 0.0) for c in range(n_classes)])
        brute_t = acc[frozenset(range(n_classes))]
        assert np.max(np.abs(mc - brute_c)) < 1e-10
        assert abs(mt - brute_t) < 1e-10


def test_all_decision_rules_agree_on_the_singleton_theta_family():
    rng = np.random.default_rng(0)
    for _ in range(200):
        M, Mt = random_simple_bbas(rng, 4, 3)
        mc, mt = core.fuse_simple(M, Mt)
        sc = core.decision_scores(mc, mt)
        top = {k: int(np.argmax(v)) for k, v in sc.items()}
        assert len(set(top.values())) == 1, top


def test_equation_12_loses_mass():
    """Eq. (12) taken literally has total mass (r1 r2 - lam)/(1 - lam) < 1 whenever r1 r2 < 1."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        mc1, mt1 = random_simple_bbas(rng, 1, 3)
        mc2, mt2 = random_simple_bbas(rng, 1, 3)
        m1, m2 = as_dict(mc1[0], mt1[0], 3), as_dict(mc2[0], mt2[0], 3)
        r1, r2 = rng.uniform(0.3, 0.95, 2)
        out, lam = core.eq12_weighted_rule(m1, m2, r1, r2)
        total = sum(out.values())
        assert total == pytest.approx((r1 * r2 - lam) / (1 - lam))
        assert total < 1.0


def test_total_conflict_raises():
    a = {frozenset([0]): 1.0}
    b = {frozenset([1]): 1.0}
    with pytest.raises(ValueError):
        core.dempster(a, b)


def test_bounded_bba_sums_to_one():
    m = core.bounded_bba(0.2, 0.7, 3, 1)
    assert sum(m.values()) == pytest.approx(1.0)
    assert m[frozenset(range(3))] == pytest.approx(0.5)
