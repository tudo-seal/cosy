"""_summary_."""

import gc
import weakref
from typing import Any

import pytest

from cosy.core.tree import Tree


@pytest.fixture
def term() -> Tree:
    """_summary_.

    Returns:
        Tree: _description_
    """
    return Tree("f", (Tree("x", ()),))


def test_cache_separates_interpretations(term: Tree) -> None:
    """_summary_."""
    doubling: dict[str, Any] = {"f": lambda value: value * 2, "x": lambda: 3}
    squaring: dict[str, Any] = {"f": lambda value: value * value, "x": lambda: 3}
    assert term.interpret(doubling) == 6
    assert term.interpret(squaring) == 9
    assert term.interpret(doubling) == 6
    assert term.interpret(squaring) == 9


def test_repeated_interpretation_hits_cache(term: Tree) -> None:
    """_summary_."""
    boxing: dict[str, Any] = {"f": lambda value: [value], "x": lambda: 3}
    first = term.interpret(boxing)
    second = term.interpret(boxing)
    # same object: result was not recomputed
    assert first is second


def test_cache_neither_grows_or_expires(term: Tree) -> None:
    """_summary_."""

    # exists to be weakly referenced
    class Interpretation(dict):
        pass

    graveyard: list = []
    for _ in range(100):
        interpretation = Interpretation({"f": lambda value: value + 1, "x": lambda: 3})
        graveyard.append(weakref.ref(interpretation))
        assert term.interpret(interpretation) == 4
    # cache remains as only possible anchor
    del interpretation
    gc.collect()

    assert sum(1 for reference in graveyard if reference() is not None) == 1
    assert graveyard[-1]() is not None


def test_indirect_interpretation_is_cached() -> None:
    """_summary_."""
    calls: list = []

    def symbol() -> int:
        calls.append(len(calls))
        return calls[-1]

    term = Tree(symbol, ())
    assert term.interpret(None) == 0
    assert term.interpret(None) == 0
