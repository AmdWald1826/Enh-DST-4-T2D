"""LaTeX tables of the manuscript, generated from the summary files in results/.

    python experiments/make_tables.py [--out-dir tables]
"""
import argparse
from pathlib import Path

import pandas as pd

from _common import RESULTS, ROOT

rd = lambda f: pd.read_csv(f, index_col=0)
fmt = lambda x, d=3: "--" if pd.isna(x) else f"{x:.{d}f}"
NAME = {"LogReg": "Logistic regression", "RandForest": "Random forest", "HistGB": "Gradient boosting",
        "GaussNB": "Gaussian naive Bayes", "V0_dempster": "DST-Fusion (V0)",
        "E3_noshrink_betp_lam": "E3 (no double count)", "Dummy_prior": "Class frequencies only"}
VARIANT_NAME = {
    "V0_dempster": "V0 Dempster (baseline)", "V1_dempster_lam": "V1 + learned global discount",
    "V2_yager": "V2 Yager's rule", "V3_yager_lam": "V3 Yager + discount",
    "V4_dempster_dep_lam": "V4 redundancy correction + discount",
    "V5_yager_dep_lam": "V5 Yager + redundancy + discount", "E1_dempster_pltr_lam": "E1 plausibility transform",
    "E2_dempster_pltr_dep_lam": "E2 E1 + redundancy", "E3_noshrink_betp_lam": "E3 no double count of ignorance",
    "E4_noshrink_pltr_lam": "E4 E3 + plausibility transform", "P1_prior": "P1 E3 + class prior ($s\\le1$)",
    "P2_prior_cap75": "P2 E3 + class prior ($s\\le.75$)"}
GENERIC = ["LogReg", "RandForest", "HistGB", "GaussNB"]
DST = ["V0_dempster", "E3_noshrink_betp_lam"]


def block(title, T, models):
    out = [f"\\multicolumn{{6}}{{l}}{{\\textit{{{title}}}}}\\\\"]
    for m in models:
        r = T.loc[m]
        err = r.errdet_mTheta if m.startswith(("V0", "E3")) else r.errdet_1mp
        out.append(f"{NAME[m]} & {fmt(r.acc)} & {fmt(r.brier)} & {fmt(r.ece)} & {fmt(r.auc)} & {fmt(err)} \\\\")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "tables")
    a = ap.parse_args()
    a.out_dir.mkdir(exist_ok=True)
    sim = rd(RESULTS / "simulation" / "summary_dev.csv")
    pima = rd(RESULTS / "pima" / "real_pima_summary.csv")
    nh = rd(RESULTS / "nhanes" / "nhanes_primary9_summary.csv")
    pp = rd(RESULTS / "class_prior" / "prior_pima_summary.csv")
    pn = rd(RESULTS / "class_prior" / "prior_nhanes_summary.csv")

    # ---- main results table
    t = ["\\begin{table}[!ht]\\centering\\small",
         "\\caption{Main results (mean over $5\\times10$ stratified cross-validation). Brier score is summed over classes. "
         "The last column is the AUROC for detecting errors, using $m(\\Theta)$ for the DST models and $1-\\max p$ for the others.}"
         "\\label{tab:main}",
         "\\begin{tabular}{lccccc}\\hline", "Model & Accuracy & Brier & ECE & AUC & Err.\\ det.\\\\\\hline"]
    t += block("Simulated corpus (seed 0, $n=312$)", sim, GENERIC + DST)
    t += block("Pima ($n=768$, binary)", pima, GENERIC + DST)
    t += block("NHANES 2015--2016 ($n=1739$, three classes)", nh, ["Dummy_prior"] + GENERIC + DST)
    t += ["\\hline\\end{tabular}", "\\end{table}"]
    (a.out_dir / "tables_main.tex").write_text("\n".join(t) + "\n")

    # ---- variants table
    t = ["\\begin{table}[!ht]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}",
         "\\caption{Variants of the fusion scheme: accuracy and Brier score (mean over cross-validation). Simulated corpus: "
         "development corpus (seed 0). Variants V0--V5 were fixed before the runs; E1--E4 and P1--P2 were added after seeing "
         "results and are exploratory. A dash means that the variant was not run on that data set.}\\label{tab:ablation}",
         "\\begin{tabular}{lcccccc}\\hline",
         " & \\multicolumn{2}{c}{Simulated} & \\multicolumn{2}{c}{Pima} & \\multicolumn{2}{c}{NHANES}\\\\",
         " Variant & Acc. & Brier & Acc. & Brier & Acc. & Brier\\\\\\hline"]
    for v, nm in VARIANT_NAME.items():
        s = sim.loc[v] if v in sim.index else None
        p = pima.loc[v] if v in pima.index else (pp.loc[v] if v in pp.index else (pp.loc["E3"] if v == "E3_noshrink_betp_lam" else None))
        n = nh.loc[v] if v in nh.index else (pn.loc[v] if v in pn.index else None)
        cell = lambda r, c: fmt(r[c]) if r is not None else "--"
        t.append(f"{nm} & {cell(s, 'acc')} & {cell(s, 'brier')} & {cell(p, 'acc')} & {cell(p, 'brier')} & "
                 f"{cell(n, 'acc')} & {cell(n, 'brier')} \\\\")
    t += ["\\hline\\end{tabular}", "\\end{table}"]
    (a.out_dir / "tables_ablation.tex").write_text("\n".join(t) + "\n")
    print("tables written to", a.out_dir)


if __name__ == "__main__":
    main()
