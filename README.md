# Data

## `pima.csv`
Pima Indians Diabetes data (NIDDK; Smith et al., 1988): 768 women, 8 variables, outcome = diabetes
onset within five years (268 positives). Public copy of the raw file
(`jbrownlee/Datasets`, `pima-indians-diabetes.data.csv`) with a header row added. Zeros that are
physiologically impossible (Glucose, BloodPressure, SkinThickness, Insulin, BMI) are recoded as
missing by `dstfusion.evaluation.load_pima`, not in the file.

## `nhanes_2015_2016_cohort.csv`
Cohort built by `experiments/build_nhanes_cohort.py` from the public NHANES 2015--2016 files
(CDC/NCHS). 1,739 adults; class counts 986 (normal) / 635 (prediabetes) / 118 (undiagnosed diabetes).

| Column | Meaning |
|---|---|
| `SEQN` | NHANES respondent identifier |
| `Age` | age in years |
| `Sex` | 1 = female, 0 = male |
| `BMI` | body mass index, kg/m² |
| `Waist` | waist circumference, cm |
| `SBP`, `DBP` | systolic / diastolic blood pressure, mmHg (mean of readings 2--4, else reading 1) |
| `TotalChol`, `HDL`, `Triglycerides` | mg/dL |
| `Copper`, `Zinc` | serum, µg/dL (67 % missing; used only in the secondary 11-variable set) |
| `class` | 0 normal, 1 prediabetes, 2 undiagnosed diabetes (ADA thresholds on HbA1c and 2-h OGTT glucose) |

HbA1c and 2-hour glucose define `class` and are not stored as input variables. Survey weights
are not used, so results do not describe the US population.

The raw NHANES XPT files are public and are not redistributed here (see
`experiments/build_nhanes_cohort.py` for the file list and the download page).
Rebuilding the cohort reproduces this file up to floating-point rounding (differences below 1e-13).
