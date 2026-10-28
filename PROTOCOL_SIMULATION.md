# Protocol fixed before the trials (2026-10-27)

Purpose: correct the over-confidence of DST-Fusion observed with Dempster's rule (Brier 0.36 against 0.20--0.27 for the generic models, `uncertain` scenario) and make the ignorance m(Θ) informative about errors.

## Variants (all valid DST fusions; none uses test labels)
- V0: Dempster's rule, reliability s_j = chance-corrected accuracy (current version).
- V1: V0 + learned global discount lambda (grid {1, .9, .8, .7, .6, .5, .4, .3, .2}).
- V2: Yager's rule (conflict mass transferred to Θ), reliabilities of V0.
- V3: Yager + learned lambda.
- V4: Dempster + within-class redundancy: s'_j = lambda * s_j / (1 + sum_k |R_w,jk|), R_w = pooled within-class correlation (estimated on the training data).
- V5: Yager + within-class redundancy + learned lambda.

lambda is chosen by 3-fold cross-validation inside the training fold, minimising the Brier score. Nothing is tuned on the test fold.

## Evaluation
- Development: corpus with seed 0, scenario `uncertain` (8 % noise, 10 % missing values), 5 folds x 10 repetitions, same folds as before.
- Confirmation: 5 independent corpora (seeds 1 to 5, same parameters), 5 folds x 4 repetitions each. The variant retained on the development corpus is evaluated as is.
- Robustness: sweep over noise {0, .04, .08, .16} x missing {0, .1, .3}, 5 folds x 3 repetitions, on the retained variant, V0 and the 4 generic models.
- Appendix: scenario `protocol`.

## Success criteria (all required, on the development corpus AND on the mean of the 5 confirmation corpora)
1. Calibration: Brier score of the variant <= Brier score of the best generic model + 0.03.
2. Accuracy: at most 1.5 points below the best generic model.
3. Informative ignorance: AUROC of error detection by m(Θ) >= 0.65 and >= the AUROC of 1 - max(probability) of the best generic model minus 0.02.
Retained variant = the one that satisfies the criteria with the best development Brier score; otherwise, a failure is reported as is.

## Known limitations before running
Corpus simulated with assumed parameters; 312 cases; untuned baselines; the thresholds above are arbitrary but fixed before the results.
