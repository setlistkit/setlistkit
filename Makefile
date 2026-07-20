# Quality gates. `make check` must be green before anything is believed done: lint, test,
# and security, run continuously rather than at the end.
#
# The venv is pyenv/ (python3.14). Bootstrap it with `make dev`.

.PHONY: help dev lint test security check clean

VENV     := pyenv
PYTHON   := $(VENV)/bin/python
# pycodestyle checks pure style everywhere; pylint does deep analysis on shipped source.
STYLE_SRC := src tests
LINT_SRC  := src

help:
	@echo "dev       install setlistkit plus its dev tooling into $(VENV)/"
	@echo "lint      pycodestyle + pylint"
	@echo "test      pytest with coverage"
	@echo "security  bandit + pip-audit"
	@echo "check     lint, test, and security together"
	@echo "clean     drop coverage output and every __pycache__"

dev:
	$(PYTHON) -m pip install -e '.[dev,report]'

lint:
	$(VENV)/bin/pycodestyle --max-line-length=120 $(STYLE_SRC)
	$(VENV)/bin/pylint $(LINT_SRC)

test:
	$(VENV)/bin/pytest -v --cov=setlistkit --cov-report term-missing \
		--cov-report term:skip-covered --cov-report xml:coverage.xml

security:
	$(VENV)/bin/bandit -q -r $(LINT_SRC)
	$(VENV)/bin/pip-audit

check: lint test security

# Not just tidiness. CPython invalidates cached bytecode on (mtime-in-seconds, size), so a
# same-length edit inside the same second can silently reuse the old .pyc. Run this before
# believing any before/after result.
clean:
	rm -rf coverage.xml .coverage .pytest_cache
	find src tests -name __pycache__ -type d -prune -exec rm -rf {} +
