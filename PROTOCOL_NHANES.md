# Protocol -- NHANES 2015--2016, three classes (written before any model execution)

Date: 2026-10-28. CDC files (XPT) supplied by the user: DEMO_I, BMX_I, BPX_I, GHB_I, OGTT_I, TCHOL_I, HDL_I, TRIGLY_I, DIQ_I, CUSEZN_I. Fasting plasma glucose (GLU_I) is not available.

Cohort: adults >= 18 years, not pregnant, without diagnosed or treated diabetes (DIQ010, DIQ050, DIQ070 != 1), with HbA1c and 2-hour glucose (OGTT) measured -> n = 1,739 (986 / 635 / 118).

Labels (ADA thresholds, single measurement): diabetes if HbA1c >= 6.5 % or 2-h glucose >= 200 mg/dL; prediabetes if HbA1c 5.7--6.4 % or 2-h 140--199; otherwise normal. HbA1c and 2-h glucose are never input variables.

Input variables, primary set (9): age, sex, BMI, waist circumference, systolic and diastolic blood pressure (mean of readings 2 to 4, otherwise reading 1), total cholesterol, HDL, triglycerides. Secondary set (11): primary + serum copper and zinc (67 % missing). LDL excluded (computed from triglycerides).

Survey weights not used: results are not representative of the US population.

Models: V0 (the manuscript's method), E3 (no shrinkage by ignorance, lambda learned), LogReg, RandForest, HistGB, GaussNB (same parameters as before), plus a reference "class frequencies" model (Dummy). No further tuning.

Evaluation: stratified CV 5x10 (random_state=42), one run per variable set. Metrics: accuracy, balanced accuracy, macro F1, Brier score (sum over 3 classes), log-loss, ECE, macro one-vs-rest AUC, AUROC of error detection (m(Θ) and 1 - p_max).

Descriptive criteria, unchanged: (1) Brier <= best generic + 0.03; (2) accuracy >= best generic - 0.015; (3) error-detection AUROC by m(Θ) >= 0.65 and >= best generic - 0.02. Paired comparisons by corrected resampled t (Nadeau--Bengio) against the best generic model; 2 variants x 2 test metrics are corrected by Bonferroni (m = 4 per variable set).

Rule: result reported as is, no second run with modified variants. Known limitations: small diabetes class (118), single-measurement labels, US population 2015--2016.
