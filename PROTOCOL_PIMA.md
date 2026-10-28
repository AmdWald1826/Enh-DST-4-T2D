# Protocol -- real-data step (written before any execution)

Date: 2026-10-28. Data: Pima Indians Diabetes (NIDDK, 768 women, 8 variables, 268 positives), raw file from GitHub (jbrownlee/Datasets), 23,278 bytes.

Single, frozen preprocessing: physiologically impossible zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are coded missing (NaN). No variable transformation. All models receive the same raw data; the generic models add iterative imputation + standardisation inside the pipeline (fitted in the training fold).

DST models: V0 (the manuscript's method as implemented) and E3 (no shrinkage by ignorance, BetP, lambda learned by inner CV). No other variant, no other setting.

Generic models: LogReg, RandForest, HistGB, GaussNB (same parameters as in the simulation step).

Evaluation: StratifiedKFold 5x10 (random_state=42), a single run. Metrics: accuracy, Brier score (sum over classes = 2x the usual binary Brier score; both are reported), log-loss, ECE, AUC, AURC, AUROC of error detection (m(Θ) and 1 - p_max).

Comparisons: criteria 1 to 3 of the simulation step applied descriptively. Paired Nadeau--Bengio test (corrected resampled t) on accuracy and Brier score between each DST variant and the best generic model.

Rule: the result is reported as is. No second run with variants modified after seeing the numbers. Any new idea becomes a next step, with a new data set or a new protocol.

Known limitations: a single population (Pima women >= 21 years), binary task (diabetes within 5 years), skewed variables treated as Gaussian, small size.
