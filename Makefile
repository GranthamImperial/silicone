.DEFAULT_GOAL := help

VENV_DIR ?= ./venv
DATA_DIR ?= ./data
DOCS_DIR=$(PWD)/docs
SCRIPTS_DIR ?= ./scripts
TESTS_DIR=$(PWD)/tests

FILES_TO_FORMAT_PYTHON=setup.py scripts src tests docs/source/conf.py

SR15_EMISSIONS_SCRAPER = $(SCRIPTS_DIR)/download_sr15_emissions.py
SR15_EMISSIONS_DIR = $(DATA_DIR)/sr15_emissions
SR15_EMISSIONS_FILE = $(SR15_EMISSIONS_DIR)/sr15_emissions.csv

NOTEBOOKS_DIR=./notebooks
NOTEBOOKS_SANITIZE_FILE=$(TESTS_DIR)/notebook-tests.cfg

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

.PHONY: format
format:  ## re-format files
	make isort
	make black

.PHONY: flake8
flake8: $(VENV_DIR)  ## check compliance with pep8
	$(VENV_DIR)/bin/flake8 $(FILES_TO_FORMAT_PYTHON)

.PHONY: isort
isort: $(VENV_DIR)  ## format the imports in the source and tests
	$(VENV_DIR)/bin/isort -y --recursive $(FILES_TO_FORMAT_PYTHON)

.PHONY: black
black: $(VENV_DIR)  ## use black to autoformat code
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		$(VENV_DIR)/bin/black --exclude _version.py --target-version py37 $(FILES_TO_FORMAT_PYTHON); \
	else \
		echo Not trying any formatting, working directory is dirty... >&2; \
	fi;

.PHONY: checks
checks: $(VENV_DIR)  ## run checks similar to CI
	@echo "=== black ==="; $(VENV_DIR)/bin/black --check src tests setup.py docs/source/conf.py --exclude silicone/_version.py || echo "--- black failed ---" >&2; \
		echo "\n\n=== isort ==="; $(VENV_DIR)/bin/isort --check-only --quiet --recursive src tests setup.py || echo "--- isort failed ---" >&2; \
		echo "\n\n=== docs ==="; $(VENV_DIR)/bin/sphinx-build -M html docs/source docs/build -qW || echo "--- docs failed ---" >&2; \
		echo "\n\n=== notebook tests ==="; $(VENV_DIR)/bin/pytest notebooks -r a --nbval-lax --sanitize-with tests/notebook-tests.cfg || echo "--- notebook tests failed ---" >&2; \
		echo "\n\n=== tests ==="; $(VENV_DIR)/bin/pytest tests -r a --cov=silicone --cov-report='' \
			&& $(VENV_DIR)/bin/coverage report --fail-under=95 || echo "--- tests failed ---" >&2; \
		echo

.PHONY: test-all
test-all:  ## run the testsuite and test the notebooks
	make test
	make test-notebooks

.PHONY: test
test: $(VENV_DIR) ## run the full testsuite
	$(VENV_DIR)/bin/pytest --cov -rfsxEX --cov-report term-missing

.PHONY: test-notebooks
test-notebooks: $(VENV_DIR)  ## test the notebooks
	$(VENV_DIR)/bin/pytest -r a --nbval-lax $(NOTEBOOKS_DIR) --sanitize $(NOTEBOOKS_SANITIZE_FILE)

.PHONY: format-notebooks
format-notebooks: $(VENV_DIR)  ## format the notebooks
	$(VENV_DIR)/bin/black-nb $(NOTEBOOKS_DIR)

.PHONY: docs
docs: $(VENV_DIR)  ## make docs
	$(VENV_DIR)/bin/sphinx-build -M html docs/source docs/build

$(DOCS_DIR)/build/html/index.html: $(DOCS_DIR)/source/*.py $(DOCS_DIR)/source/_templates/*.html $(DOCS_DIR)/source/*.rst src/silicone/**.py README.rst CHANGELOG.rst $(VENV_DIR)
	cd $(DOCS_DIR); make html

# first time setup, follow this https://blog.jetbrains.com/pycharm/2017/05/how-to-publish-your-package-on-pypi/
# then this works
.PHONY: publish-on-testpypi
publish-on-testpypi: $(VENV_DIR)  ## publish the current state of the repository to test PyPI
	-rm -rf build dist
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		$(VENV_DIR)/bin/python setup.py sdist bdist_wheel --universal; \
		$(VENV_DIR)/bin/twine upload --verbose -r testpypi dist/*; \
	else \
		echo Working directory is dirty >&2; \
	fi;

test-testpypi-install: $(VENV_DIR)  ## test whether installing from test PyPI works
	$(eval TEMPVENV := $(shell mktemp -d))
	python3 -m venv $(TEMPVENV)
	$(TEMPVENV)/bin/pip install pip --upgrade
	# Install pymagicc without dependencies.
	$(TEMPVENV)/bin/pip install \
		-i https://testpypi.python.org/pypi silicone \
		--no-dependencies --pre
		# Remove local directory from path to get actual installed version.
	@echo "This doesn't test all dependencies"
	$(TEMPVENV)/bin/python -c "import sys; sys.path.remove(''); import silicone; print(silicone.__version__)"

.PHONY: publish-on-pypi
publish-on-pypi:  $(VENV_DIR) ## publish the current state of the repository to PyPI
	-rm -rf build dist
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		$(VENV_DIR)/bin/python setup.py sdist bdist_wheel --universal; \
		$(VENV_DIR)/bin/twine upload --verbose dist/*; \
	else \
		echo Working directory is dirty >&2; \
	fi;

test-pypi-install: $(VENV_DIR)  ## test whether installing from PyPI works
	$(eval TEMPVENV := $(shell mktemp -d))
	python3 -m venv $(TEMPVENV)
	$(TEMPVENV)/bin/pip install pip --upgrade
	$(TEMPVENV)/bin/pip install silicone --pre
	$(TEMPVENV)/bin/python scripts/test_install.py

.PHONY: check-pypi-distribution
check-pypi-distribution: $(VENV_DIR)  ## check the PyPI distribution for errors
	-rm -rf build dist
	@status=$$(git status --porcelain); \
	if test "x$${status}" = x; then \
		$(VENV_DIR)/bin/python setup.py sdist bdist_wheel --universal; \
		$(VENV_DIR)/bin/twine check dist/*; \
	else \
		echo Working directory is dirty >&2; \
	fi;


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
