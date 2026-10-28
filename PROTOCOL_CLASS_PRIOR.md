# Protocol -- taking class prior frequencies into account (written before execution)

Date: 2026-10-28. Hypothesis H (formulated AFTER seeing the NHANES and Pima results, hence exploratory on these two data sets):
the construction of DST-Fusion ignores the prior class frequencies, which partly explains the lower raw accuracy (balanced accuracy equal to that of LogReg) and the poor calibration.

Variants (all declared in advance, no others):
- E3: reference (no shrinkage by ignorance, BetP, lambda learned).
- P1: E3 + a "prior" source: Bayesian mass m_prior(c) = s * pi_c, m_prior(Θ) = 1 - s (pi_c = smoothed training frequencies, Laplace), combined by Dempster's rule with the existing sources; s chosen in {0, .25, .5, .75, 1} by 3-fold inner CV (Brier). With s = 1 no mass is left on Θ.
- P2: like P1 but s capped at .75, to keep a non-zero ignorance.

Data: Pima (768) and NHANES 2015--2016 primary set with 9 variables (1,739), stratified CV 5x10, random_state=42, a single run. These two data sets were used to formulate H: the result is exploratory. Confirmation requires an independent cohort (another NHANES cycle, e.g. 2013--2014 or 2017--2020, to be supplied). Simulated corpora are excluded: their classes are nearly balanced by construction, so no effect of H is expected there.

Comparisons: P1 and P2 against E3 (accuracy, balanced accuracy, Brier score, ECE, error-detection AUROC by m(Θ)); corrected resampled t (Nadeau--Bengio), Bonferroni over 2 variants x 3 metrics = 6 tests per data set. Descriptive criteria against the generic models unchanged.

A consistency check on a single random split of NHANES (1300/439) was made while coding: s = 1 was chosen and the mean m(Θ) fell to 0. This observation motivates P2; no other trial was made.

Rule: results reported as is, no second run.
