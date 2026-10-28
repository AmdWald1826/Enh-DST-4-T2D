# Results

Per-fold (`*_folds.csv`) and summary (`*_summary.csv`) files behind every number in the manuscript.
They were produced by the scripts in `experiments/`; `make verify` re-runs everything into
`results_verify/` and compares it with this folder.

| Folder | Produced by | Content |
|---|---|---|
| `simulation/` | `run_simulation.py --suite protocol`, `summarize_simulation.py` | `dev_seed0` (V0--V5, development corpus), `conf_seed1..5` (V0--V5, confirmation corpora), `explo_dev_seed0` and `explo_conf_seed6..10` (E1--E4), `summary_dev.csv`, `summary_confirmation.csv` |
| `pima/` | `run_pima.py` | V0, E3 and four generic models on Pima |
| `nhanes/` | `run_nhanes.py` | same, for the primary (9 variables) and secondary (11 variables) sets, plus the class-frequency reference |
| `class_prior/` | `run_class_prior.py` | E3, P1, P2 and the generic models on Pima and NHANES |

Columns: `acc` accuracy, `bacc` balanced accuracy, `f1macro`, `brier` Brier score summed over
classes (half of it for two classes is the usual binary Brier score), `logloss`, `ece`, `auc`
(one-vs-rest for three classes), `aurc`, `errdet_1mp` (AUROC of error detection with 1 - max p),
`errdet_mTheta` (AUROC of error detection with m(Θ), DST models only), `lam` learned global
discount, `sp` learned class-prior weight, `mTheta` mean ignorance. The simulation files were
written by an earlier version of the script that did not include `bacc`, `f1macro` and `sp`.
