#!/usr/bin/env python3
"""
Pytest wrapper script that ensures PYTHONHASHSEED=0 for deterministic testing.

Usage:
    python run_pytest.py [pytest_args...]
    python run_pytest.py tests/test_evolutionary.py -v
    python run_pytest.py --co  # collect tests only
"""

import os
import subprocess
import sys


def main():
    """Run pytest with PYTHONHASHSEED=0 set."""
    # Ensure PYTHONHASHSEED is set
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"

    # Build pytest command
    cmd = [sys.executable, "-m", "pytest", *sys.argv[1:]]

    # Run pytest
    result = subprocess.run(cmd, check=False, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
