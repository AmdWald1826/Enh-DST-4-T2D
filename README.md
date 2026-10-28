# Enh-DST-4-T2D

Repository: https://github.com/AmdWald1826/Enh-DST-4-T2D

Code, evaluation protocols, data and results for the manuscript

> **Dempster–Shafer Evidence Fusion for Type 2 Diabetes Screening: Ranking Quality, Calibration Failures and the Double Counting of Ignorance**
> Elmehdi Berber and Adda Boualem, Computer Science Department, Ibn Khaldoun University, Tiaret, Algeria.

The manuscript evaluates DST-Fusion, a classifier that turns each clinical variable into a bounded
belief assignment, discounts it by a reliability, and combines the variables with Dempster's rule. On
simulated corpora, the Pima data and NHANES 2015–2016, DST-Fusion ranks cases about as well as
standard classifiers, but its probabilities are poorly calibrated and its ignorance mass m(Θ) is a weak
error detector. The code here reproduces every number, table and figure of the paper.

## Repository layout

```
src/dstfusion/       package: core.py (power-set DST engine and closed form), classifier.py
                     (DSTFusionClassifier), simulate.py (simulated corpora), evaluation.py (metrics, CV, tests)
experiments/         one script per experiment, plus cohort construction, figures and tables
protocols/           the four written evaluation protocols (English)
data/                pima.csv, nhanes_2015_2016_cohort.csv, and a README describing them
results/             per-fold and summary CSV files used in the paper
tests/               pytest suite
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest -q
```

Python ≥ 3.9. The results in `results/` were produced with Python 3.13, numpy 2.5, pandas 3.0,
scipy 1.18, scikit-learn 1.9 and matplotlib 3.11. Other versions can change the last digits of the
random-forest and imputation results.

## Using the classifier

```python
from dstfusion import DSTFusionClassifier

clf = DSTFusionClassifier(rule="dempster", lam_mode="none")   # variant V0, the manuscript's method
clf.fit(X_train, y_train)                 # NaN means "not measured": the source is skipped
class_mass, theta_mass = clf.masses(X_test)   # m(c) per class and the ignorance m(Θ)
proba = clf.predict_proba(X_test)         # pignistic probabilities
```

Variants used in the paper: `V0`–`V5` (rule, learned global discount, redundancy correction),
exploratory `E1`–`E4` (`ign_shrink=False` is E3, the per-variable posterior discounted by reliability
only), and `P1`/`P2` (`prior="learn"`, class frequencies as an extra source). The exact settings
are in `dstfusion/evaluation.py`.

## Reproducing the paper

| Step | Command | Output |
|---|---|---|
| Simulated corpora (V0–V5, E1–E4) | `make simulation` | `results/simulation/` |
| Pima | `make pima` | `results/pima/` |
| NHANES cohort (needs the CDC files) | `python experiments/build_nhanes_cohort.py` | `data/nhanes_2015_2016_cohort.csv` |
| NHANES | `make nhanes` | `results/nhanes/` |
| Class prior | `make class-prior` | `results/class_prior/` |
| Figures / tables | `make figures tables` | `figures/`, `tables/` |
| Everything, compared with the archived results | `make verify` | `results_verify/` |

The full protocol (`make verify`) takes roughly one hour on two CPU cores; the tests take a few seconds. Cross-validation uses
`RepeatedStratifiedKFold(random_state=42)`; simulation seeds are 0 (development), 1–5 (confirmation) and
6–10 (exploratory variants). The NHANES cohort file is included, so `build_nhanes_cohort.py` is only needed to
rebuild it from the raw files (see `data/README.md`).

## Protocols and what "pre-specified" means

Each evaluation step has a written protocol in `protocols/`, written before the corresponding runs.
They are deposited with the code afterwards and were not registered externally, so they document
the intended analysis but carry no independent timestamp. The class-prior protocol was written after
the Pima and NHANES results had been seen; its results are exploratory.

## Limitations

The simulated corpora rest on assumed parameters. Pima concerns one population and a binary task.
NHANES labels come from single laboratory measurements, survey weights are not used, and no second
NHANES cycle has been analysed yet.

## Data sources

- Pima Indians Diabetes: Smith et al. (1988), NIDDK.
- NHANES 2015–2016: National Center for Health Statistics, CDC (public use files).

## Licence and citation

Code: MIT (see `LICENSE`). The licence is the authors' choice and can be changed before the first release.
Citation metadata are in `CITATION.cff`; the Zenodo DOI will be added after the first release.
