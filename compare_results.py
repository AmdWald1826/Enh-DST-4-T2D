"""Compare two results trees (e.g. results_verify/ against the archived results/).

    python experiments/compare_results.py results_verify results [--tol 1e-9]

For every summary/fold CSV present in both trees, compares the numeric columns they share.
Exits with status 1 if any value differs by more than the tolerance.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new", type=Path)
    ap.add_argument("ref", type=Path)
    ap.add_argument("--tol", type=float, default=1e-9)
    a = ap.parse_args()
    worst, bad, n = 0.0, [], 0
    for f in sorted(a.new.rglob("*.csv")):
        g = a.ref / f.relative_to(a.new)
        if not g.exists():
            print("no reference for", f.relative_to(a.new)); continue
        x, y = pd.read_csv(f), pd.read_csv(g)
        cols = [c for c in x.columns if c in y.columns and pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c])]
        if "model" in x.columns and "model" in y.columns:
            key = [c for c in ["model", "fold"] if c in x.columns and c in y.columns]
            x, y = x.sort_values(key).reset_index(drop=True), y.sort_values(key).reset_index(drop=True)
        else:
            x, y = x.sort_values(x.columns[0]).reset_index(drop=True), y.sort_values(y.columns[0]).reset_index(drop=True)
        if x.shape[0] != y.shape[0]:
            bad.append((f.name, "row count", x.shape[0], y.shape[0])); continue
        d = np.nanmax(np.abs(x[cols].values - y[cols].values)) if cols else 0.0
        both_nan = (x[cols].isna().values != y[cols].isna().values).sum()
        worst = max(worst, d); n += 1
        if d > a.tol or both_nan:
            bad.append((f.name, "max abs diff", float(d), int(both_nan)))
    print(f"compared {n} files, largest numeric difference {worst:.2e}")
    for b in bad:
        print("DIFFERENT:", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
