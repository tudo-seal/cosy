"""Tests for the resolution-query vocabulary.

The queries name what is asked of a solution space. These tests pin which kind a query is, that
a half-specified query term is refused, and that the checker decides membership.
"""

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from cosy.core.tree import Tree
from cosy.search import ResolutionQuery, checker, generator_query, residual_query
from tests.search_fixtures import (
    EXPR,
    PAIR,
    WORD,
    add,
    constrained_space,
    expression_space,
    lit,
    neg,
    zero,
)


def unused() -> str:
    """Build a terminal that no specification mentions.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "?"


def test_the_generator_carries_no_query_term():
    """The generator asks for every inhabitant, so it prescribes nothing."""
    query = generator_query(expression_space(), EXPR)

    assert query.tree is None
    assert query.pos is None
    assert query.is_generator
    assert not query.is_partial_term


def test_a_partial_term_query_records_the_term_and_the_position():
    """The partial-term query keeps what it opens and where."""
    space = expression_space()
    tree = Tree(neg, (Tree(lit, ()),))

    query = residual_query(space, EXPR, tree, (0,))

    assert query.tree == tree
    assert query.pos == (0,)
    assert query.is_partial_term
    assert not query.is_generator


def test_the_root_position_is_a_prescribed_term_not_a_missing_one():
    """A hole at the root is a partial-term query, although the empty path is falsy.

    Reading the position for truth rather than for presence turns the root query into the
    generator, which prescribes no term at all.
    """
    space = expression_space()

    query = residual_query(space, EXPR, Tree(neg, (Tree(lit, ()),)), ())

    assert query.is_partial_term
    assert not query.is_generator


@pytest.mark.parametrize(
    ("tree", "pos"),
    [(Tree(lit, ()), None), (None, (0,)), (None, ())],
    ids=["term without position", "position without term", "root position without term"],
)
def test_a_half_specified_query_term_is_rejected(tree, pos):
    """Neither half of a query term means anything without the other.

    Args:
        tree (Tree | None): The prescribed term, or None.
        pos (Path | None): The opened position, or None.
    """
    with pytest.raises(ValueError, match="give both or neither"):
        ResolutionQuery(expression_space(), EXPR, tree, pos)


def test_the_checker_accepts_what_the_engine_derives():
    """Every term the engine streams is a term the checker recognizes."""
    space = expression_space()

    derived = list(space.depth_first_resolution(EXPR, max_count=20, max_depth=3))

    assert len(derived) > 1, "a one-element language would prove nothing"
    for term in derived:
        assert checker(space, EXPR, term), f"the checker rejected the derived term {term}"


@pytest.mark.parametrize(
    "term",
    [
        Tree(neg, ()),
        Tree[Any](add, (Tree(lit, ()),)),
        Tree(lit, (Tree(lit, ()),)),
        Tree(unused, ()),
        Tree[Any](neg, (Tree(unused, ()),)),
    ],
    ids=["arity zero for neg", "arity one for add", "argument for lit", "foreign symbol", "foreign argument"],
)
def test_the_checker_rejects_a_term_the_space_cannot_derive(term):
    """A term of the wrong arity or over a foreign symbol is no inhabitant.

    Args:
        term (Tree): The term offered to the checker.
    """
    assert not checker(expression_space(), EXPR, term)


def test_the_checker_decides_beyond_any_search_depth():
    """Membership does not depend on how deep a search would have gone.

    The checker walks the term it is given, so it answers for terms no bounded enumeration
    reaches, which is what makes it usable as an oracle for the streaming queries.
    """
    space = expression_space()
    deep = Tree(lit, ())
    for _ in range(20):
        deep = Tree(neg, (deep,))

    assert checker(space, EXPR, deep)
    assert deep not in set(space.depth_first_resolution(EXPR, max_count=50, max_depth=4))


def test_the_checker_separates_the_non_terminals_of_one_space():
    """The queried non-terminal decides the answer, not the space.

    The same word inhabits one sort of the coupled space and not the other, so a checker that
    answered for the space as a whole would accept it twice.
    """
    space = constrained_space()
    word = Tree(zero, ())

    assert checker(space, WORD, word)
    assert not checker(space, PAIR, word)
    assert not checker(space, WORD, Tree(unused, ()))


def test_a_query_cannot_be_rewritten_after_it_is_built():
    """The query is immutable, otherwise what it rejects at build time can be set afterwards."""
    query = generator_query(expression_space(), EXPR)

    with pytest.raises(FrozenInstanceError):
        query.tree = Tree(lit, ())  # type: ignore[misc]


def test_a_position_outside_the_prescribed_term_is_rejected():
    """A position no subterm sits at describes no partial term.

    The engine answers such a query with an empty stream, which a caller reads as a term without
    completions rather than as the mistake it is.
    """
    with pytest.raises(ValueError, match="no position of the prescribed term"):
        residual_query(expression_space(), EXPR, Tree(neg, (Tree(lit, ()),)), (5, 7))
