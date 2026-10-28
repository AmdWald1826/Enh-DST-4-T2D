"""Parametric generator for the simulated 312-case type 2 diabetes corpora.

All distribution parameters below are ASSUMED, plausible ranges chosen by the code author;
they are not taken from a specific publication. The simulated corpora illustrate the
behaviour of the method; they are not evidence about real patients.
"""
import numpy as np, pandas as pd

FEATURES = ["age", "bmi", "fpg", "hba1c", "tg", "hdl", "ldl", "sbp", "family_history", "activity"]
CLASSES = ["Normal", "Prediabetes", "T2D"]
PREV = np.array([0.35, 0.32, 0.33])

#            age  bmi  fpg   hba1c  tg   hdl  ldl  sbp  fam  act
MU = np.array([[44., 25., 90.,  5.3,  110., 55., 105., 118., 0.8, 6.0],
               [52., 28., 108., 6.0,  140., 48., 120., 126., 1.4, 4.5],
               [58., 31., 150., 7.8,  185., 42., 130., 134., 2.1, 3.5]])
SD = np.array([[11., 3.0, 6.0,  0.25, 35., 10., 25., 11., 0.8, 2.0],
               [11., 3.5, 8.0,  0.25, 45., 10., 27., 12., 0.9, 2.0],
               [11., 4.0, 28., 1.10, 70., 10., 30., 13., 1.0, 2.0]])
# correlation imposed among fpg, hba1c, bmi, tg, hdl, ldl
R = np.eye(10)
for (a, b, r) in [("fpg", "hba1c", .75), ("bmi", "fpg", .30), ("bmi", "hba1c", .25), ("bmi", "tg", .35),
                  ("tg", "hdl", -.45), ("tg", "ldl", .30), ("hdl", "bmi", -.30), ("age", "sbp", .35)]:
    i, j = FEATURES.index(a), FEATURES.index(b); R[i, j] = R[j, i] = r
L = np.linalg.cholesky(R)

def ada_label(fpg, hba1c):
    """ADA-style rule used by the manuscript (OGTT not simulated)."""
    lab = np.zeros(len(fpg), int)
    lab[(fpg >= 100) | (hba1c >= 5.7)] = 1
    lab[(fpg >= 126) | (hba1c >= 6.5)] = 2
    return lab

def generate(n=312, labels="ada", seed=0):
    """labels='ada'    : label recomputed from the generated FPG/HbA1c (manuscript protocol)
       labels='latent' : label = latent class used to draw the case"""
    rng = np.random.default_rng(seed)
    counts = np.round(PREV * n).astype(int); counts[0] += n - counts.sum()
    latent = np.repeat(np.arange(3), counts)
    Z = rng.standard_normal((n, 10)) @ L.T
    X = MU[latent] + SD[latent] * Z
    X[:, FEATURES.index("family_history")] = np.clip(np.round(X[:, FEATURES.index("family_history")]), 0, 4)
    X[:, FEATURES.index("activity")] = np.clip(X[:, FEATURES.index("activity")], 0, 10)
    X[:, [FEATURES.index(f) for f in ("fpg", "hba1c", "tg", "hdl", "ldl", "bmi")]] = np.clip(
        X[:, [FEATURES.index(f) for f in ("fpg", "hba1c", "tg", "hdl", "ldl", "bmi")]], 1e-3, None)
    y = ada_label(X[:, FEATURES.index("fpg")], X[:, FEATURES.index("hba1c")]) if labels == "ada" else latent
    perm = rng.permutation(n)
    return X[perm], y[perm], latent[perm]

def observe(X, noise=0.0, missing=0.02, seed=1):
    """What the classifier sees: measurement error (relative sd = noise on lab values)
    and MCAR missingness (NaN)."""
    rng = np.random.default_rng(seed)
    Xo = X.copy()
    if noise > 0:
        for f in ("fpg", "hba1c", "tg", "hdl", "ldl", "bmi", "sbp"):
            j = FEATURES.index(f); Xo[:, j] *= 1 + noise * rng.standard_normal(len(X))
    if missing > 0:
        Xo[rng.random(Xo.shape) < missing] = np.nan
    return Xo

if __name__ == "__main__":
    X, y, lat = generate(labels="ada")
    print("ADA labels  prevalence:", np.bincount(y) / len(y), " agreement with latent class:", (y == lat).mean())
    X, y, lat = generate(labels="latent")
    print("latent prev:", np.bincount(y) / len(y))
