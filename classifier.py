"""DST-Fusion classifier (vectorised), with the variants studied in the manuscript.

Per feature j (one source): bounded BBA on {singletons, Theta} (Rules 4/5 of the manuscript):
    m_j(c) = (1-ign_j) q_j(c),  m_j(Theta) = ign_j,  ign_j = class overlap (Bhattacharyya).
Reliability then discounts the source (Shafer):  m'(c)=s' m(c),  m'(Theta)=1-s'+s' m(Theta), with
    s'_j = lambda * rel_j * g_j
      rel_j : chance-corrected single-feature training accuracy
      lambda: global discount learned by inner 3-fold CV (Brier) on the training fold   [lam_mode='learn']
      g_j   : 1/(1+sum_k |R_w,jk|), R_w = pooled within-class correlation (training fold)  [dep=True]
Fusion rule:
    'dempster': Dempster's rule (conflict renormalised away)
    'yager'   : the conflict mass K is transferred to Theta, so m(Theta) grows with disagreement between sources
Decision: argmax Score = argmax BetP = argmax m(c) (identical on this BBA family).
Nothing uses test data. Missing values: the source is skipped (vacuous BBA).

Options beyond the manuscript's baseline (variant V0 = rule='dempster', lam_mode='none'):
    ign_shrink=False : the per-variable posterior is discounted by reliability only (variant E3)
    prob='pltr'      : plausibility transform instead of the pignistic transform
    prior='learn'    : class frequencies as an extra Bayesian source (variants P1/P2)
"""
import numpy as np
from itertools import combinations
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold

LAM_GRID = (1.0, .9, .8, .7, .6, .5, .4, .3, .2)

def _bc(m1, s1, m2, s2):
    v = s1 ** 2 + s2 ** 2
    return np.exp(-(0.25 * (m1 - m2) ** 2 / v + 0.5 * np.log(v / (2 * s1 * s2))))

class DSTFusionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, rule="dempster", lam_mode="none", dep=False, sd_floor=1e-3, grid=LAM_GRID, inner_seed=0, prob="betp", ign_shrink=True, prior="none", sp_grid=(0.0,.25,.5,.75,1.0)):
        self.prior = prior; self.sp_grid = sp_grid;
        self.prob = prob; self.ign_shrink = ign_shrink; self.rule = rule; self.lam_mode = lam_mode; self.dep = dep; self.sd_floor = sd_floor
        self.grid = grid; self.inner_seed = inner_seed

    # ---- parameters estimated on a training set
    def _fit_params(self, X, y):
        classes = np.unique(y); K = len(classes); d = X.shape[1]
        mu = np.zeros((K, d)); sd = np.ones((K, d))
        for k, c in enumerate(classes):
            Xc = X[y == c]; mu[k] = np.nanmean(Xc, 0); sd[k] = np.maximum(np.nanstd(Xc, 0, ddof=1), self.sd_floor)
        P = dict(classes=classes, mu=mu, sd=sd)
        ign = np.array([np.mean([_bc(mu[a, j], sd[a, j], mu[b, j], sd[b, j]) for a, b in combinations(range(K), 2)]) for j in range(d)])
        P["ign"] = np.clip(ign, 0, .999)
        Q = self._post(P, X); rel = np.ones(d)
        for j in range(d):
            ok = ~np.isnan(X[:, j]); acc = np.mean(classes[Q[ok, j].argmax(1)] == y[ok])
            rel[j] = np.clip((acc - 1 / K) / (1 - 1 / K), 0, 1)
        P["rel"] = rel
        P["pi"] = (np.bincount(np.searchsorted(classes, y), minlength=K) + 1.0) / (len(y) + K)
        Z = (X - mu[np.searchsorted(classes, y)]) / sd[np.searchsorted(classes, y)]
        Rw = np.ones((d, d))
        for a in range(d):
            for b in range(a + 1, d):
                ok = ~np.isnan(Z[:, a]) & ~np.isnan(Z[:, b])
                r = np.corrcoef(Z[ok, a], Z[ok, b])[0, 1] if ok.sum() > 3 else 0.0
                Rw[a, b] = Rw[b, a] = 0.0 if np.isnan(r) else r
        P["g"] = 1.0 / (1.0 + np.abs(Rw - np.eye(d)).sum(1)); P["Rw"] = Rw
        return P

    @staticmethod
    def _post(P, X):
        mu, sd = P["mu"], P["sd"]
        Xz = np.where(np.isnan(X), 0.0, X)
        dens = np.exp(-0.5 * ((Xz[:, :, None] - mu.T[None]) / sd.T[None]) ** 2) / sd.T[None] + 1e-300
        Q = dens / dens.sum(2, keepdims=True); Q[np.isnan(X)] = np.nan
        return Q

    # ---- fusion for a given lambda
    def _masses(self, P, X, lam, sp=0.0):
        K = len(P["classes"]); mask = ~np.isnan(X); Q = self._post(P, X); Q = np.where(mask[:, :, None], Q, 0.0)
        s = np.clip(lam * P["rel"] * (P["g"] if self.dep else 1.0), 0, 1)
        mc = s[None, :, None] * ((1 - P["ign"]) if self.ign_shrink else np.ones_like(P["ign"]))[None, :, None] * Q
        mt = np.where(mask, (1 - s + s * (P["ign"] if self.ign_shrink else 0.0))[None, :], 1.0)
        Pl = np.where(mask[:, :, None], mc + mt[:, :, None], 1.0)
        Lp = np.log(Pl).sum(1); Lt = np.log(mt).sum(1)
        if sp > 0:   # class prior as an extra Bayesian source, discounted by sp:  m(c)=sp*pi_c, m(Theta)=1-sp
            Lp = Lp + np.log(sp * P["pi"] + (1 - sp))[None, :]; Lt = Lt + np.log(max(1 - sp, 1e-12))
        if self.rule == "dempster":
            ref = np.maximum(Lp.max(1), Lt)[:, None]
            Pp = np.exp(Lp - ref); T = np.exp(Lt[:, None] - ref)
            m = np.clip(Pp - T, 0, None); Zn = m.sum(1, keepdims=True) + T
            return m / Zn, (T / Zn)[:, 0]
        if self.rule == "yager":
            Pp = np.exp(Lp); T = np.exp(Lt)[:, None]
            m = np.clip(Pp - T, 0, None); Kc = np.clip(1 - m.sum(1, keepdims=True) - T, 0, None)
            tot = m.sum(1, keepdims=True) + T + Kc
            return m / tot, ((T + Kc) / tot)[:, 0]
        raise ValueError(self.rule)

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y)
        self.classes_ = np.unique(y); self.P_ = self._fit_params(X, y); self.lam_ = 1.0
        if self.lam_mode == "learn":
            sc = np.zeros(len(self.grid)); n = 0
            for tr, va in StratifiedKFold(3, shuffle=True, random_state=self.inner_seed).split(X, y):
                Pi = self._fit_params(X[tr], y[tr]); Y = np.eye(len(self.classes_))[np.searchsorted(self.classes_, y[va])]
                for i, lam in enumerate(self.grid):
                    mc, mt = self._masses(Pi, X[va], lam); B = self._prob(mc, mt)
                    sc[i] += np.sum((B - Y) ** 2); n += len(va)
            self.lam_ = self.grid[int(np.argmin(sc))]; self.inner_brier_ = sc / n
        self.sp_ = 0.0
        if self.prior == "learn":
            sc = np.zeros(len(self.sp_grid))
            for tr, va in StratifiedKFold(3, shuffle=True, random_state=self.inner_seed).split(X, y):
                Pi = self._fit_params(X[tr], y[tr]); Y = np.eye(len(self.classes_))[np.searchsorted(self.classes_, y[va])]
                for i, sp in enumerate(self.sp_grid):
                    mc, mt = self._masses(Pi, X[va], self.lam_, sp); sc[i] += np.sum((self._prob(mc, mt) - Y) ** 2)
            self.sp_ = self.sp_grid[int(np.argmin(sc))]
        return self

    def masses(self, X):
        return self._masses(self.P_, np.asarray(X, float), self.lam_, self.sp_)

    def _prob(self, mc, mt):
        if self.prob == "betp": return mc + mt[:, None] / mc.shape[1]          # pignistic
        if self.prob == "pltr":                                                 # Cobb-Shenoy plausibility transform
            Pl = mc + mt[:, None]; return Pl / Pl.sum(1, keepdims=True)
        raise ValueError(self.prob)

    def predict_proba(self, X):
        mc, mt = self.masses(X); return self._prob(mc, mt)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
