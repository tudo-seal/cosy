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


def branching_space() -> SolutionSpace[str, str, None]:
    """Return ``S -> top(A, B)`` with ``A -> a1 | a2(A)`` and ``B -> b1 | b2(B) | b3(A)``.

    Two non-terminals produce terms independently of each other, so the enumeration has more than
    one of them queued at a time. That is what makes the order of its working set observable; on a
    grammar with a single recursive non-terminal any implementation looks deterministic.

    Returns:
        SolutionSpace[str, str, None]: The grammar.
    """
    space: SolutionSpace[str, str, None] = SolutionSpace()
    space.add_rule("S", "top", (A, B), ())
    space.add_rule("A", "a1", (), ())
    space.add_rule("A", "a2", (A,), ())
    space.add_rule("B", "b1", (), ())
    space.add_rule("B", "b2", (B,), ())
    space.add_rule("B", "b3", (A,), ())
    return space


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


def three_ground_types_space() -> SolutionSpace[str, str, None]:
    """Return ``S -> top(A, B, C)`` with ``A -> a``, ``B -> b`` and ``C -> c``.

    The only grammar here with more than one ground type. Pruning seeds its queue in a single
    sweep over the rules before the walk begins, and with three entries in that seed the order of
    the sweep is observable in the result; where one ground type is seeded alone it is not.

    Returns:
        SolutionSpace[str, str, None]: The grammar.
    """
    space: SolutionSpace[str, str, None] = SolutionSpace()
    space.add_rule("A", "a", (), ())
    space.add_rule("B", "b", (), ())
    space.add_rule("C", "c", (), ())
    space.add_rule("S", "top", (A, B, C), ())
    return space


def fan_out_space() -> SolutionSpace[str, str, None]:
    """Return ``X -> x`` with ``A -> p(X)``, ``B -> q(X)`` and ``C -> r(X)``.

    One ground type makes three non-terminals productive at once, so the order in which pruning
    walks the consumers of ``X`` decides the order of the pruned grammar.

    Returns:
        SolutionSpace[str, str, None]: The grammar.
    """
    ground: NonTerminalArgument[str] = NonTerminalArgument(None, "X")
    space: SolutionSpace[str, str, None] = SolutionSpace()
    space.add_rule("X", "x", (), ())
    space.add_rule("A", "p", (ground,), ())
    space.add_rule("B", "q", (ground,), ())
    space.add_rule("C", "r", (ground,), ())
    return space
