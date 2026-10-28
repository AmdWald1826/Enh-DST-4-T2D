from pathlib import Path

import numpy as np

from dstfusion import evaluation as ev

DATA = Path(__file__).resolve().parents[1] / "data"


def test_pima():
    X, y = ev.load_pima(DATA / "pima.csv")
    assert X.shape == (768, 8) and y.sum() == 268
    assert np.isnan(X).sum() > 0 and not (X[:, 1] == 0).any()      # zero glucose recoded to missing


def test_nhanes():
    X, y = ev.load_nhanes(DATA / "nhanes_2015_2016_cohort.csv")
    assert X.shape == (1739, 9)
    assert np.bincount(y).tolist() == [986, 635, 118]
    X11, _ = ev.load_nhanes(DATA / "nhanes_2015_2016_cohort.csv", ev.NHANES_FEATURE_SETS["secondary11"])
    assert X11.shape == (1739, 11)
