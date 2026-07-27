# Quality gates. `make check` must be green before anything is believed done: lint, test,
# and security, run continuously rather than at the end.
#
# The venv is pyenv/ (python3.14). Bootstrap it with `make dev`.

.PHONY: help dev lint uniquetestnames test security check clean

VENV     := pyenv
PYTHON   := $(VENV)/bin/python
# pycodestyle checks pure style everywhere; pylint does deep analysis on shipped source.
STYLE_SRC := src tests scripts
LINT_SRC  := src scripts

# Tests are linted too, but not by the same standard, and this list is the whole difference.
#
# WHAT THIS IS FOR is not tidiness. A duplicated `def test_...` name shadows the earlier one and
# pytest collects only the survivor, so a test stops running and nothing says so -- the count in
# the green bar just quietly goes down by one. That happened here (`function-redefined` in
# test_model_settings.py, found by running this for the first time), and it is the reason tests
# are now gated rather than checked by eye.
#
# Every entry below is a rule that is RIGHT for shipped source and WRONG for a test, with why:
#
#   missing-*-docstring          a test's name is its documentation; 589 of these would bury
#                                the findings that matter
#   use-implicit-booleaness      `assert shows == []` asserts empty AND a list. `not shows`
#                                also passes for None, 0 and "". The long form is the stronger
#                                assertion, which is the entire point of writing it in a test.
#   unbalanced-tuple-unpacking   `stored, = store.tapes.recordings()` is assert-exactly-one; the
#                                unpack RAISES on any other length, which is the assertion.
#   protected-access             tests test internals. That is their job.
#   import-outside-toplevel      deliberate: an import inside a test is often what it measures.
#   unused-argument              a pytest fixture requested for its side effect has no other way
#                                to be requested.
#   too-few-public-methods       fakes and stubs are small on purpose.
#   duplicate-code               fixtures repeat by design; sharing them is what makes a test
#                                pass for a reason other than the one it names.
#   superfluous-parens           keeps a family of fixture strings written the same way whether
#                                or not one of them happens to fit on a line.
#   use-dict-literal             `dict(slot=..., seconds=...)` reads better than a literal when
#                                every key is an identifier.
# Space-separated and passed as one flag each. A backslash continuation inside a make variable
# joins the lines with a SPACE, so writing this as one comma-separated list across four lines
# produces `a,b, c,d` -- which pylint reads as a single unknown name and silently disables
# nothing. It scored 0.00/10 and looked like the code was broken rather than the list.
TEST_DISABLE := missing-function-docstring missing-class-docstring \
                use-implicit-booleaness-not-comparison unbalanced-tuple-unpacking \
                protected-access import-outside-toplevel unused-argument \
                too-few-public-methods duplicate-code superfluous-parens use-dict-literal

help:
	@echo "dev       install setlistkit plus its dev tooling into $(VENV)/"
	@echo "lint      pycodestyle + pylint"
	@echo "test      pytest with coverage"
	@echo "security  bandit + pip-audit"
	@echo "uniquetestnames  every test function has a name nothing else uses"
	@echo "check     unique names, lint, test, and security together"
	@echo "clean     drop coverage output and every __pycache__"

dev:
	$(PYTHON) -m pip install -e '.[dev,report]'

lint:
	$(VENV)/bin/pycodestyle --max-line-length=120 $(STYLE_SRC)
	$(VENV)/bin/pylint $(LINT_SRC)
	# PYTHONPATH because a test that imports a sibling test's fixtures resolves it the way
	# pytest does -- by the directory it sits in -- and pylint has no such convention.
	PYTHONPATH=tests/unit $(VENV)/bin/pylint $(addprefix --disable=,$(TEST_DISABLE)) tests

# From bitmath, where it guards TestCase class names. Two failures, one rule: a name reused
# inside one file SHADOWS, so a test silently stops running; a name reused across files is only
# ambiguous, but pylint cannot see it -- it reads one module at a time. Needs no venv, so it runs
# first and fails in a second.
uniquetestnames:
	./tests/test_unique_test_names.sh

test:
	$(VENV)/bin/pytest -v --cov=setlistkit --cov-report term-missing \
		--cov-report term:skip-covered --cov-report xml:coverage.xml

security:
	$(VENV)/bin/bandit -q -r $(LINT_SRC)
	$(VENV)/bin/pip-audit

check: uniquetestnames lint test security

# Not just tidiness. CPython invalidates cached bytecode on (mtime-in-seconds, size), so a
# same-length edit inside the same second can silently reuse the old .pyc. Run this before
# believing any before/after result.
clean:
	rm -rf coverage.xml .coverage .pytest_cache
	find src tests -name __pycache__ -type d -prune -exec rm -rf {} +
