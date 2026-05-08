DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PYTHON := $(DIR).venv/bin/python
VENV := $(DIR).venv
KNOWN_TARGETS := compile run steady linearize projected-run venv clean-plots clean-lib clean push
CONFIG_GOALS := $(filter-out $(KNOWN_TARGETS),$(MAKECMDGOALS))
CONFIG_GOAL := $(firstword $(CONFIG_GOALS))
CONFIG ?= $(if $(CONFIG_GOAL),$(CONFIG_GOAL),config)
CONFIG_ARG := --config "$(CONFIG)"

.PHONY: $(KNOWN_TARGETS) $(CONFIG_GOALS)

compile:
	bash $(DIR)compile.sh

run:
	cd $(DIR) && $(PYTHON) main.py --mode simulation $(CONFIG_ARG)

steady:
	cd $(DIR) && $(PYTHON) main.py --mode steady $(CONFIG_ARG)

linearize:
	cd $(DIR) && $(PYTHON) main.py --mode linearize $(CONFIG_ARG)

projected-run:
	cd $(DIR) && $(PYTHON) main.py --mode projected-run $(CONFIG_ARG)

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
	git commit -m "Default msg for auto push"
	git push 

ifneq ($(CONFIG_GOALS),)
$(CONFIG_GOALS):
	@:
endif
