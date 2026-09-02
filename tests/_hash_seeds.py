"""Running a statement in fresh interpreters that differ only in their hash seed.

A process fixes its hash seed at startup, so a test that stays inside the current interpreter
cannot tell a stable order from one that is merely stable for this run. Everything that has to
show reproducibility across processes goes through here.

A plain module rather than a fixture, because the callers are whole test files rather than single
tests, and because the child bodies are literal source text that reads better beside the assertion
it feeds.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# The child bodies import what they need themselves, so that they depend on nothing but ``cosy``
# and the grammars in ``tests/_determinism_grammars``.
_CHILD_PREAMBLE = f"import sys\nsys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})\n"

HASH_SEEDS = ("0", "1", "7", "42", "123")


def printed_across_hash_seeds(body: str) -> set[str]:
    """Run ``body`` in fresh interpreters that differ only in their hash seed.

    Args:
        body (str): Statements to run after the preamble. Whatever they print is what is compared.

    Returns:
        set[str]: The distinct outputs. Reproducibility means there is exactly one.
    """
    printed = set()
    for hash_seed in HASH_SEEDS:
        child = subprocess.run(
            [sys.executable, "-c", _CHILD_PREAMBLE + body],
            capture_output=True,
            check=False,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            text=True,
        )
        if child.returncode != 0:
            pytest.fail(
                f"the child interpreter did not run to completion under PYTHONHASHSEED={hash_seed}; "
                f"this is an environment failure, not a difference in order:\n{child.stderr}"
            )
        printed.add(child.stdout.strip())
    return printed
