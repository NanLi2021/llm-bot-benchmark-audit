PYTHON ?= .venv/bin/python
DATA_ROOT ?= data
.PHONY: test figure check run appendix

test:            ## synthetic and regression tests; no datasets needed
	$(PYTHON) -m unittest discover -s tests -v

figure:          ## Figure 1 from the shipped results
	$(PYTHON) figures/fig_route_column.py

check:           ## shipped results against every printed number, in the paper and the appendix
	$(PYTHON) scripts/check_results.py --actual results/expected
	$(PYTHON) scripts/check_appendix.py

run:             ## rerun everything on authorized inputs: make run DATA_ROOT=/path/to/inputs
	PYTHON=$(PYTHON) scripts/run_all.sh $(DATA_ROOT) build/run

appendix:        ## build paper/appendix.pdf (needs latexmk)
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error appendix.tex && latexmk -c appendix.tex
