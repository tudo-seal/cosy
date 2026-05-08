"""
Pytest configuration file - automatically loaded by pytest.

About PYTHONHASHSEED Requirement:
=================================

Some tests (e.g., test_symbolic_regression_demo) require deterministic behavior across
multiple test runs. This requires PYTHONHASHSEED=0 to be set BEFORE Python starts,
because Python's hash randomization causes sets and dicts to iterate in different orders.

See pytest.ini for details on how to run tests with deterministic behavior.

Recommended usage:
    python run_pytest.py              # Uses wrapper that sets PYTHONHASHSEED=0
    python run_pytest.py -v           # Verbose mode
    python run_pytest.py tests/...    # Run specific tests

Alternative (manual PYTHONHASHSEED):
    PYTHONHASHSEED=0 python -m pytest
"""
