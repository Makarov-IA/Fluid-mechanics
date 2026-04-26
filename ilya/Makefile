DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PYTHON := $(DIR).venv/bin/python
VENV := $(DIR).venv

.PHONY: compile run steady venv clean-plots clean-lib clean push

compile:
	bash $(DIR)compile.sh

run:
	cd $(DIR) && $(PYTHON) main.py --mode simulation

steady:
	cd $(DIR) && $(PYTHON) main.py --mode steady

venv:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r $(DIR)requirements.txt

clean-plots:
	rm -rf $(DIR)plots

clean-lib:
	rm -f $(DIR)solver.dylib $(DIR)solver.so $(DIR)solver.dll

clean: clean-plots clean-lib

push:
	git add $(DIR)
	git commit -m "Deafault msg for auto push"
	git push 