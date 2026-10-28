# Reproduce the results of the manuscript. Each target writes into results/ (or figures/, tables/).
# `make verify` re-runs everything into results_verify/ and compares with the archived results.
PY ?= python3

.PHONY: install test simulation pima nhanes class-prior figures tables all verify

install:
	$(PY) -m pip install -e ".[test]"

test:
	$(PY) -m pytest -q

simulation:
	$(PY) experiments/run_simulation.py --suite protocol
	$(PY) experiments/summarize_simulation.py

pima:
	$(PY) experiments/run_pima.py

nhanes:
	$(PY) experiments/run_nhanes.py

class-prior:
	$(PY) experiments/run_class_prior.py pima
	$(PY) experiments/run_class_prior.py nhanes

figures:
	$(PY) experiments/make_figures.py

tables:
	$(PY) experiments/make_tables.py

all: simulation pima nhanes class-prior figures tables

verify:
	$(PY) experiments/run_simulation.py --suite protocol --out-dir results_verify/simulation
	$(PY) experiments/summarize_simulation.py --dir results_verify/simulation
	$(PY) experiments/run_pima.py --out-dir results_verify/pima
	$(PY) experiments/run_nhanes.py --out-dir results_verify/nhanes
	$(PY) experiments/run_class_prior.py pima --out-dir results_verify/class_prior
	$(PY) experiments/run_class_prior.py nhanes --out-dir results_verify/class_prior
	$(PY) experiments/compare_results.py results_verify results
