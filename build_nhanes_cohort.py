"""Build the NHANES 2015-2016 cohort from the public CDC XPT files.

Download the following files (cycle 2015-2016, suffix _I) from
https://wwwn.cdc.gov/nchs/nhanes/search/DataPage.aspx?Cycle=2015-2016
into data/raw/nhanes/ (or pass --xpt-dir):

    DEMO_I BMX_I BPX_I GHB_I OGTT_I TCHOL_I HDL_I TRIGLY_I DIQ_I CUSEZN_I

    python experiments/build_nhanes_cohort.py [--xpt-dir data/raw/nhanes]

Cohort: adults >= 18 y, not pregnant, no diagnosed or treated diabetes, HbA1c and 2-hour
OGTT glucose measured. Class labels follow the ADA thresholds (single measurement):
  2 = undiagnosed diabetes : HbA1c >= 6.5 % or 2-h glucose >= 200 mg/dL
  1 = prediabetes          : HbA1c 5.7-6.4 % or 2-h glucose 140-199 mg/dL
  0 = normal
HbA1c and the 2-h glucose define the label and are never used as input variables.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import DATA, NHANES_CSV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xpt-dir", type=Path, default=DATA / "raw" / "nhanes")
    ap.add_argument("--out", type=Path, default=NHANES_CSV)
    a = ap.parse_args()
    r = lambda f: pd.read_sas(a.xpt_dir / f"{f}_I.xpt", format="xport", encoding="utf-8")

    D = r("DEMO")[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDEXPRG"]]
    tables = dict(BMX=["BMXBMI", "BMXWAIST"], GHB=["LBXGH"], OGTT=["LBXGLT"], TCHOL=["LBXTC"],
                  HDL=["LBDHDD"], TRIGLY=["LBXTR"], CUSEZN=["LBXSCU", "LBXSZN"],
                  DIQ=["DIQ010", "DIQ050", "DIQ070"])
    for f, cols in tables.items():
        D = D.merge(r(f)[["SEQN"] + cols], on="SEQN", how="left")

    # Blood pressure: mean of readings 2-4, else reading 1.
    bp = r("BPX")
    sy = bp[["BPXSY2", "BPXSY3", "BPXSY4"]].mean(axis=1)
    di = bp[["BPXDI2", "BPXDI3", "BPXDI4"]].mean(axis=1)
    sy = sy.where(sy.notna(), bp["BPXSY1"])
    di = di.where(di.notna(), bp["BPXDI1"])
    D = D.merge(pd.DataFrame({"SEQN": bp.SEQN, "SBP": sy, "DBP": di}), on="SEQN", how="left")

    steps = [("participants in DEMO_I", len(D))]
    D = D[D.RIDAGEYR >= 18]
    steps.append(("age >= 18", len(D)))
    D = D[D.RIDEXPRG != 1]
    steps.append(("not pregnant", len(D)))
    D = D[(D.DIQ010 != 1) & (D.DIQ050 != 1) & (D.DIQ070 != 1)]
    steps.append(("no diagnosed or treated diabetes", len(D)))
    D = D[D.LBXGH.notna() & D.LBXGLT.notna()]
    steps.append(("HbA1c and 2-h OGTT glucose available", len(D)))

    D["class"] = np.where((D.LBXGH >= 6.5) | (D.LBXGLT >= 200), 2,
                          np.where((D.LBXGH >= 5.7) | (D.LBXGLT >= 140), 1, 0))
    D = D.rename(columns={"RIDAGEYR": "Age", "RIAGENDR": "Sex", "BMXBMI": "BMI", "BMXWAIST": "Waist",
                          "LBXTC": "TotalChol", "LBDHDD": "HDL", "LBXTR": "Triglycerides",
                          "LBXSCU": "Copper", "LBXSZN": "Zinc"})
    D["Sex"] = (D.Sex == 2).astype(float)          # 1 = female, 0 = male
    D["SEQN"] = D.SEQN.astype(int)
    out = D[["SEQN", "Age", "Sex", "BMI", "Waist", "SBP", "DBP", "TotalChol", "HDL", "Triglycerides",
             "Copper", "Zinc", "class"]]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    for s in steps:
        print(s)
    print("classes (0 normal, 1 prediabetes, 2 undiagnosed diabetes):",
          out["class"].value_counts().sort_index().to_dict())
    print("missing values per variable:", out.isna().sum()[lambda s: s > 0].to_dict())


if __name__ == "__main__":
    main()
