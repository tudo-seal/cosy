"""Grammars shared by the reproducibility tests and by the child interpreters they start.

The child processes below ``tests/test_solution_space_determinism.py`` import from here rather than
from the test module itself, so that they depend on nothing but ``cosy``: a child that cannot
import pytest would report an environment failure where a statement about ordering should be. A
grammar that no child needs is defined next to the test that uses it instead.

Only ``mixed_width_space`` is also built in the parent process. The others are built exclusively in
child interpreters, which the coverage run does not measure, so a report shows their bodies as
unexecuted even though every one of them runs five times.
"""

from cosy.core.solution_space import NonTerminalArgument, SolutionSpace

A: NonTerminalArgument[str] = NonTerminalArgument(None, "A")
B: NonTerminalArgument[str] = NonTerminalArgument(None, "B")
C: NonTerminalArgument[str] = NonTerminalArgument(None, "C")


def mixed_width_space() -> SolutionSpace[str, str, None]:
    """Return ``S -> top(C, C)`` with ``C -> un(C) | tri(C, C, C) | bi(C, C) | lf``.

    The rules of ``C`` are added neither in ascending nor in descending order of their number of
    subgoals, so a search that ignores the ordering of new goals, or that reverses it, produces a
    different sequence than one that honours it. Pruning reaches ``S`` only through ``C``, which
    makes this the one grammar here whose discovery order differs from its rule order.

    Returns:
        SolutionSpace[str, str, None]: The grammar.
    """
    space: SolutionSpace[str, str, None] = SolutionSpace()
    space.add_rule("S", "top", (C, C), ())
    space.add_rule("C", "un", (C,), ())
    space.add_rule("C", "tri", (C, C, C), ())
    space.add_rule("C", "bi", (C, C), ())
    space.add_rule("C", "lf", (), ())
    return space
