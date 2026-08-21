"""What a partial term can still be completed to.

A partial-term query prescribes a term everywhere except at one position, where it leaves a
variable. Its success branches describe the subterms that complete it into an inhabitant.

Two failure modes, both pinned here:

* Too few. Returning after the first goal found drops the completions of every other clause.
* Too many. Expanding all open subgoals of a goal at once reaches the same goal once per
  expansion order.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from cosy.core import Constructor, Synthesizer
from cosy.core.solution_space import ConstantArgument, SolutionSpace
from cosy.core.tree import Tree
from cosy.core.types import Arrow, Intersection
from tests.search_fixtures import (
    EXPR,
    PAIR,
    A,
    B,
    C,
    a_only,
    add,
    b_only,
    c_ab,
    constrained_space,
    expression_space,
    lit,
    multi_path_space,
    neg,
    one,
    pair,
    wrap,
    wrap_c,
    zero,
)

T = Constructor("T")


def top_t(inner: str) -> str:
    """Lift a ``C`` to a ``T``.

    Puts the two clauses of ``C`` one step below the root, where the split is decided by a
    traversal step rather than by the initial goals.

    Args:
        inner (str): The rendering of the ``C`` subterm.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"top({inner})"


def nested_multi_path_space():
    """Build ``multi_path_space`` with one clause above it.

    Returns:
        SolutionSpace: The space, started at ``T``.
    """
    specs = {
        c_ab: Intersection(A, B),
        a_only: A,
        b_only: B,
        wrap_c: Intersection(Arrow(A, C), Arrow(B, C)),
        top_t: Arrow(C, T),
    }
    return Synthesizer(specs).construct_solution_space(T)


def positive_digit(substitution: Mapping[str, Any]) -> bool:
    """Read the constant argument ``d``, which needs no non-terminal argument to be decided.

    Args:
        substitution (Mapping[str, Any]): The substitution the rule supplies for its predicates.

    Returns:
        bool: True if the digit is greater than zero.
    """
    return bool(substitution["d"] > 0)


def forbidden_root_space() -> SolutionSpace[str, Any, None]:
    """Build a space whose start symbol has a clause its own predicate forbids.

    ``D ~> digit(d)`` for ``d`` in 0, 1, 2, of which ``d == 0`` is rejected. Stated through
    ``add_rule`` rather than through a specification, so the clause stays in the space.

    Returns:
        SolutionSpace[str, Any, None]: The space, started at ``D``.
    """
    space: SolutionSpace[str, Any, None] = SolutionSpace()
    for value in (0, 1, 2):
        space.add_rule("D", "digit", (ConstantArgument("d", value, None),), (positive_digit,))
    return space


def completions(
    space: SolutionSpace,
    start: object,
    tree: Tree,
    pos: tuple[int, ...],
    candidates: Sequence[Tree],
) -> set[object]:
    """List the candidates that complete ``tree`` at ``pos`` into an inhabitant.

    Decides membership with ``contains_tree``, which resolves nothing and so shares no traversal
    with the query it is used against.

    Args:
        space (SolutionSpace): The space to test against.
        start (object): The queried non-terminal.
        tree (Tree): The prescribed term.
        pos (tuple[int, ...]): The position of the variable.
        candidates (Sequence[Tree]): The subterms to try at ``pos``.

    Returns:
        set[object]: The roots of the candidates that complete the term.
    """
    return {
        candidate.root
        for candidate in candidates
        if space.contains_tree(start, tree.replace_subtree_at(pos, candidate))
    }


def streamed_at(
    space: SolutionSpace,
    start: object,
    tree: Tree,
    pos: tuple[int, ...],
    max_depth: int,
) -> list[object]:
    """Collect the subterms a partial-term query puts at ``pos``.

    Args:
        space (SolutionSpace): The space to query.
        start (object): The queried non-terminal.
        tree (Tree): The prescribed term.
        pos (tuple[int, ...]): The position of the variable.
        max_depth (int): The depth bound handed to the search.

    Returns:
        list[object]: The root of the subterm at ``pos``, once per streamed term.
    """
    return [
        term.subtree_at(pos).root
        for term in space.depth_first_resolution(start, max_depth=max_depth, tree=tree, pos=pos)
    ]


# ---------------------------------------------------------------------------
# The variable at the root: nothing of the prescribed term constrains the query
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "start"),
    [(expression_space, EXPR), (constrained_space, PAIR), (multi_path_space, C)],
)
def test_a_variable_at_the_root_leaves_the_query_unconstrained(build, start):
    """At the root the prescribed term is replaced entirely, so the query streams all of ``start``.

    Building the initial goals through ``_initial_goals_for``, as every other position does,
    matches the clauses against a root that is about to be discarded. What the branch streams
    instead is checked against ``contains_tree``.

    Args:
        build (Callable): Builds the space.
        start: The queried non-terminal.
    """
    space = build()
    inhabitants = list(space.depth_first_resolution(start, max_depth=3))
    assert len(inhabitants) > 1, "a one-element language would prove nothing"

    prescribed = inhabitants[0]
    from_root = list(space.depth_first_resolution(start, max_depth=3, tree=prescribed, pos=()))
    assert from_root == inhabitants
    # The root branch bypasses ``_initial_goals_for``, so check it lets nothing forbidden through.
    assert all(space.contains_tree(start, term) for term in from_root)


def test_a_root_query_is_not_restricted_to_the_prescribed_terminal():
    """The expression space states three clauses with three terminals, so matching keeps one."""
    space = expression_space()
    prescribed = Tree(neg, (Tree(lit, ()),))
    streamed = {term.root for term in space.depth_first_resolution(EXPR, max_depth=2, tree=prescribed, pos=())}
    assert streamed == {lit, neg, add}


def test_a_root_query_skips_a_clause_its_predicate_forbids():
    """At the root the clauses are turned into goals directly, and a forbidden one yields none."""
    space = forbidden_root_space()
    prescribed = Tree("digit", (Tree(0, ()),))

    goals = list(space.goal_from_tree("D", prescribed, ()))
    assert {str(goal.grounded[()][1]) for goal in goals} == {"digit 1", "digit 2"}


# ---------------------------------------------------------------------------
# Too few: the query has to stream every completion
# ---------------------------------------------------------------------------


def test_the_query_streams_every_completion():
    """``wrap_c(_)`` is completed by all three constants, two of them through one clause each."""
    space = multi_path_space()
    partial = Tree(wrap_c, (Tree(c_ab, ()),))
    candidates = [Tree(k, ()) for k in (c_ab, a_only, b_only)]

    expected = completions(space, C, partial, (0,), candidates)
    assert expected == {c_ab, a_only, b_only}, "the oracle must see all three"

    assert set(streamed_at(space, C, partial, (0,), 3)) == expected


def test_a_position_reached_by_two_clauses_yields_two_goals():
    """One goal per matching clause: a goal carries one sort at ``pos``, and the clauses differ."""
    space = multi_path_space()
    partial = Tree(wrap_c, (Tree(c_ab, ()),))

    matching_clauses = [rule for rule in space.get(C) if rule.terminal is wrap_c]
    assert len(matching_clauses) == 2, "the space must offer two clauses here"

    goals = list(space.goal_from_tree(C, partial, (0,)))
    assert {goal.subgoals[(0,)].origin for goal in goals} == {A, B}


def test_a_clause_split_below_the_root_reaches_the_query():
    """Both clauses survive a traversal step.

    At ``(0, 0)`` the split is decided in ``_expand_goal_at``, not in the initial goals.
    """
    space = nested_multi_path_space()
    partial = Tree(top_t, (Tree(wrap_c, (Tree(c_ab, ()),)),))

    goals = list(space.goal_from_tree(T, partial, (0, 0)))
    assert {goal.subgoals[(0, 0)].origin for goal in goals} == {A, B}


def test_an_unconstrained_search_and_the_query_agree_on_the_shape():
    """The completions of ``wrap_c(_)`` are the children of the terms whose root is ``wrap_c``."""
    space = multi_path_space()
    partial = Tree(wrap_c, (Tree(c_ab, ()),))

    unconstrained = {
        term.children[0].root for term in space.depth_first_resolution(C, max_depth=3) if term.root is wrap_c
    }
    assert set(streamed_at(space, C, partial, (0,), 3)) == unconstrained


# ---------------------------------------------------------------------------
# Too many: no goal twice
# ---------------------------------------------------------------------------


def test_a_goal_appears_once_per_derivation_not_once_per_expansion_order():
    """``add(add(lit, lit), neg(lit))`` has one derivation, so each position has one goal.

    Expanding every open subgoal at once reaches that goal six times for the variable at ``(0, 0)``
    and eight times for ``(1, 0)``. The early return hid it.
    """
    space = expression_space()
    tree = Tree(
        add,
        (Tree(add, (Tree(lit, ()), Tree(lit, ()))), Tree(neg, (Tree(lit, ()),))),
    )

    for pos in ((0, 0), (1, 0), (0,), (1,)):
        goals = list(space.goal_from_tree(EXPR, tree, pos))
        assert len(goals) == 1, f"position {pos} produced {len(goals)} goals"


# ---------------------------------------------------------------------------
# A predicate couples the variable to its siblings
# ---------------------------------------------------------------------------


def test_an_external_predicate_restricts_the_completions():
    """``pair`` admits its arguments only when they differ, so the sibling's value is excluded."""
    space = constrained_space()
    tree = Tree(pair, (Tree(zero, ()), Tree(one, ())))
    streamed = streamed_at(space, PAIR, tree, (0,), 3)

    assert streamed, "zero and w(..) complete the term"
    assert one not in streamed
    assert zero in streamed


def test_the_coupled_space_offers_more_than_one_completion_per_position():
    """``wrap`` makes ``W`` recursive, so the predicate has something to rule out."""
    space = constrained_space()
    tree = Tree(pair, (Tree(zero, ()), Tree(one, ())))
    streamed = streamed_at(space, PAIR, tree, (0,), 4)
    assert {zero, wrap} <= set(streamed)


def test_the_expression_space_is_the_control():
    """Without a predicate both positions of ``add`` have the same completions."""
    space = expression_space()
    tree = Tree(add, (Tree(lit, ()), Tree(lit, ())))
    left = set(streamed_at(space, EXPR, tree, (0,), 3))
    right = set(streamed_at(space, EXPR, tree, (1,), 3))
    assert left == right
    assert {lit, neg, add} <= left
