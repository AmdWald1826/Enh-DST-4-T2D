"""Core Dempster-Shafer routines for the DST-Fusion framework.

Two layers:
  1. A general engine on the full power set (small frames) used for unit tests
     and for the worked examples of the manuscript.
  2. A fast closed form for BBAs whose focal elements are the singletons and
     the whole frame Theta (the family produced by the bounded initialisation
     of Section "Bounded Mass Function Initialisation" with Rule 4/5).
     This family is closed under Dempster's rule, which makes the fusion exact
     and linear in the number of sources and classes.
"""
from __future__ import annotations
import itertools
import numpy as np

# ---------------------------------------------------------------- general engine
def subsets(n):
    """All non-empty subsets of {0..n-1} as frozensets, ordered by size."""
    out = []
    for k in range(1, n + 1):
        out += [frozenset(c) for c in itertools.combinations(range(n), k)]
    return out

def discount(m: dict, s: float, n: int) -> dict:
    """Shafer discounting (Algorithm 1): m'(A)=s m(A) for A != Theta,
    m'(Theta)=1-s+s m(Theta)."""
    theta = frozenset(range(n))
    out = {A: s * v for A, v in m.items() if A != theta}
    out[theta] = 1.0 - s + s * m.get(theta, 0.0)
    return out

def dempster(m1: dict, m2: dict):
    """Dempster's rule. Returns (combined mass, conflict K)."""
    out, K = {}, 0.0
    for A, a in m1.items():
        for B, b in m2.items():
            C = A & B
            if C:
                out[C] = out.get(C, 0.0) + a * b
            else:
                K += a * b
    if K >= 1.0 - 1e-12:
        raise ValueError("total conflict")
    return {C: v / (1.0 - K) for C, v in out.items()}, K

def bel(m, A):  return sum(v for B, v in m.items() if B <= A)
def pl(m, A):   return sum(v for B, v in m.items() if B & A)
def betp(m, w): return sum(v / len(B) for B, v in m.items() if w in B)
def score(m, w): A = frozenset([w]); return 0.5 * (bel(m, A) + pl(m, A))

def bounded_bba(min_b, max_b, n, hyp):
    """Rule 5 of the manuscript for one hypothesis A={hyp}:
    m(A)=min_b, m(not A)=1-max_b, m(Theta)=max_b-min_b."""
    assert 0 <= min_b <= max_b <= 1
    theta = frozenset(range(n)); A = frozenset([hyp]); nA = theta - A
    m = {theta: max_b - min_b}
    m[A] = m.get(A, 0) + min_b
    m[nA] = m.get(nA, 0) + 1 - max_b
    return {k: v for k, v in m.items() if v > 0}

def eq12_weighted_rule(m1, m2, r1, r2):
    """Eq. (12) of the manuscript implemented literally: products weighted by
    rho1*rho2 and normalised by 1-lambda, lambda being the weighted conflict.
    Returned unnormalised so that the total mass can be inspected."""
    out, lam = {}, 0.0
    for A, a in m1.items():
        for B, b in m2.items():
            C, w = A & B, a * b * r1 * r2
            if C: out[C] = out.get(C, 0.0) + w
            else: lam += w
    return {C: v / (1.0 - lam) for C, v in out.items()}, lam

# ---------------------------------------------------- singleton + Theta closed form
def fuse_simple(M: np.ndarray, Mtheta: np.ndarray, s: np.ndarray | None = None):
    """Fuse sources with focal sets {singletons, Theta} by Dempster's rule.

    M      (n_sources, n_classes) masses on singletons
    Mtheta (n_sources,)           masses on Theta
    s      (n_sources,)           reliabilities (Shafer discounting), default 1
    Returns (m_class, m_theta, log_unnormalised_conflict_free_mass).
    Computed in log space; exact (no approximation)."""
    M = np.asarray(M, float); Mt = np.asarray(Mtheta, float)
    if s is not None:
        s = np.asarray(s, float)
        M = s[:, None] * M
        Mt = 1.0 - s + s * Mt
    pl_ = M + Mt[:, None]                       # plausibility of each singleton per source
    with np.errstate(divide="ignore"):
        Lp = np.log(pl_).sum(axis=0)            # log prod (m_i(c)+m_i(Theta))
        Lt = np.log(Mt).sum()                   # log prod m_i(Theta)
    ref = max(Lp.max(), Lt)
    P = np.exp(Lp - ref); T = np.exp(Lt - ref)
    mc = np.clip(P - T, 0.0, None)              # m(c) ∝ prod(m_i(c)+m_i(Th)) - prod m_i(Th)
    Z = mc.sum() + T
    return mc / Z, T / Z

def decision_scores(mc, mt):
    """Bel, Pl, BetP and Score for the singleton+Theta family."""
    n = len(mc)
    return dict(bel=mc, pl=mc + mt, betp=mc + mt / n, score=mc + mt / 2.0)
