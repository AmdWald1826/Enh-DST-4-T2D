"""Paths shared by the experiment scripts (all relative to the repository root)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # lets the scripts run without `pip install -e .`

DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PIMA_CSV = DATA / "pima.csv"
NHANES_CSV = DATA / "nhanes_2015_2016_cohort.csv"
