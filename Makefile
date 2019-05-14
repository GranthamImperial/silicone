.DEFAULT_GOAL := help

VENV_DIR ?= ./venv
DATA_DIR ?= ./data
SCRIPTS_DIR ?= ./scripts

SR15_EMISSIONS_SCRAPER = $(SCRIPTS_DIR)/download_sr15_emissions.py
SR15_EMISSIONS_DIR = $(DATA_DIR)/sr15_emissions
SR15_EMISSIONS_FILE = $(SR15_EMISSIONS_DIR)/sr15_emissions.csv

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([0-9a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

sr15-emissions: $(VENV_DIR) $(DATA_DIR) $(SR15_EMISSIONS_SCRAPER)  ## download all SR1.5 emissions data
	mkdir -p $(SR15_EMISSIONS_DIR)
	$(VENV_DIR)/bin/python $(SR15_EMISSIONS_SCRAPER) $(SR15_EMISSIONS_FILE)

$(DATA_DIR):
	mkdir -p $(DATA_DIR)

test: $(VENV_DIR) ## run the full testsuite
	$(VENV_DIR)/bin/pytest --cov -rfsxEX --cov-report term-missing

# first time setup, follow this https://blog.jetbrains.com/pycharm/2017/05/how-to-publish-your-package-on-pypi/
# then this works
.PHONY: publish-on-testpypi
publish-on-testpypi: $(VENV_DIR)  ## publish the current state of the repository to test PyPI
	-rm -rf build dist
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		$(VENV_DIR)/bin/python setup.py sdist bdist_wheel --universal; \
		$(VENV_DIR)/bin/twine upload -r testpypi dist/*; \
	else \
		echo Working directory is dirty >&2; \
	fi;

test-testpypi-install: $(VENV_DIR)  ## test whether installing from test PyPI works
	$(eval TEMPVENV := $(shell mktemp -d))
	python3 -m venv $(TEMPVENV)
	$(TEMPVENV)/bin/pip install pip --upgrade
	# Install dependencies not on testpypi registry
	$(TEMPVENV)/bin/pip install pandas
	# Install pymagicc without dependencies.
	$(TEMPVENV)/bin/pip install \
		-i https://testpypi.python.org/pypi silicone \
		--no-dependencies --pre
		# Remove local directory from path to get actual installed version.
	@echo "This doesn't test dependencies"
	$(TEMPVENV)/bin/python -c "import sys; sys.path.remove(''); import silicone; print(silicone.__version__)"

.PHONY: publish-on-pypi
publish-on-pypi:  $(VENV_DIR) ## publish the current state of the repository to PyPI
	-rm -rf build dist
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		$(VENV_DIR)/bin/python setup.py sdist bdist_wheel --universal; \
		$(VENV_DIR)/bin/twine upload dist/*; \
	else \
		echo Working directory is dirty >&2; \
	fi;

test-pypi-install: $(VENV_DIR)  ## test whether installing from PyPI works
	$(eval TEMPVENV := $(shell mktemp -d))
	python3 -m venv $(TEMPVENV)
	$(TEMPVENV)/bin/pip install pip --upgrade
	$(TEMPVENV)/bin/pip install silicone --pre
	$(TEMPVENV)/bin/python scripts/test_install.py

.PHONY: test-install
test-install: $(VENV_DIR)  ## test whether installing the local setup works
	$(eval TEMPVENV := $(shell mktemp -d))
	python3 -m venv $(TEMPVENV)
	$(TEMPVENV)/bin/pip install pip --upgrade
	$(TEMPVENV)/bin/pip install .
	$(TEMPVENV)/bin/python scripts/test_install.py

virtual-environment:  ## update venv, create a new venv if it doesn't exist
	make $(VENV_DIR)

$(VENV_DIR): setup.py
	[ -d $(VENV_DIR) ] || python3 -m venv $(VENV_DIR)

	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -e .[dev]

	touch $(VENV_DIR)

first-venv: ## create a new virtual environment for the very first repo setup
	python3 -m venv $(VENV_DIR)

	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install versioneer
	# don't touch here as we don't want this venv to persist anyway
