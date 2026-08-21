"""What ``interpret`` promises a combinator: that it is called, every time.

An interpretation may have side effects and may answer differently on every call.  The evaluation
of a term is therefore not something a term can remember: a memo over results turns the second
evaluation of a term into a replay of the first, which is a different function, not a faster one.

These tests state that promise so that it survives the next attempt to cache.  A node used to keep
``(id(interpretation), interpretation, result)`` and answer from it, which broke every case below.
The memo that replaced it holds the *parameters* of a combinator.  Those follow from the callable
and not from its arguments, so it leaves all of them intact.
"""

import math
import pickle
import random
from typing import Any

import pytest

from cosy.core.tree import Tree


def test_a_combinator_with_a_side_effect_runs_on_every_evaluation() -> None:
    """Two evaluations of one term are two evaluations, not one and a replay.

    This is the case a result memo cannot serve: the effect *is* the point of the call, and a
    cache that skips the call skips the effect.
    """
    log: list[str] = []

    def record() -> str:
        """Append to the log and report what was appended.

        Returns:
            str: The entry just written.
        """
        log.append(f"call {len(log)}")
        return log[-1]

    term = Tree(record, ())

    first = term.interpret(None)
    second = term.interpret(None)

    assert log == ["call 0", "call 1"]
    assert (first, second) == ("call 0", "call 1")


def test_a_non_deterministic_interpretation_answers_afresh_every_time() -> None:
    """An interpretation that draws is asked again, not remembered.

    Fitness in an evolutionary run is an interpretation, and a fitness that averages a noisy
    measurement has to be able to disagree with itself.
    """
    rng = random.Random(20260821)

    def draw() -> float:
        """Draw the next number.

        Returns:
            float: The draw.
        """
        return rng.random()

    term = Tree(draw, ())

    assert term.interpret(None) != term.interpret(None)


def test_a_non_deterministic_value_below_the_root_is_drawn_afresh() -> None:
    """The promise holds inside the term, not only at the node that was asked.

    A memo at the root alone would still let the term as a whole answer twice with one draw.
    """
    rng = random.Random(20260821)

    def draw() -> float:
        """Draw the next number.

        Returns:
            float: The draw.
        """
        return rng.random()

    def keep(value: float) -> float:
        """Pass a value through unchanged.

        Args:
            value (float): The interpreted child.

        Returns:
            float: The same value.
        """
        return value

    term: Tree[Any] = Tree(keep, (Tree(draw, ()),))

    assert term.interpret(None) != term.interpret(None)


def test_a_shared_node_answers_afresh_in_every_term_that_holds_it() -> None:
    """Sharing must not turn one evaluation into an answer for terms built elsewhere.

    Immutability lets one node object sit in many terms, so an entry stored *on a node* is read by
    every term around it.  The same subterm in two individuals of a population would report the
    evaluation of whichever was measured first.
    """
    log: list[str] = []

    def record() -> int:
        """Count the call.

        Returns:
            int: The number of calls so far.
        """
        log.append("x")
        return len(log)

    def left(value: int) -> str:
        """Wrap a value on the left.

        Args:
            value (int): The interpreted child.

        Returns:
            str: The rendering.
        """
        return f"L{value}"

    def right(value: int) -> str:
        """Wrap a value on the right.

        Args:
            value (int): The interpreted child.

        Returns:
            str: The rendering.
        """
        return f"R{value}"

    shared = Tree(record, ())

    assert Tree[Any](left, (shared,)).interpret(None) == "L1"
    assert Tree[Any](right, (shared,)).interpret(None) == "R2"
    assert len(log) == 2


def test_a_combinator_runs_once_per_occurrence() -> None:
    """A term of n nodes over one combinator is n calls.

    The parameter memo asks for a combinator's signature once.  It must not also make the
    combinator itself be applied once.
    """
    depth = 200
    log: list[int] = []

    def leaf() -> int:
        """Start the count.

        Returns:
            int: Zero.
        """
        return 0

    def step(value: int) -> int:
        """Count one application.

        Args:
            value (int): The interpreted child.

        Returns:
            int: One more than the child.
        """
        log.append(value)
        return value + 1

    term: Tree[Any] = Tree(leaf, ())
    for _ in range(depth):
        term = Tree(step, (term,))

    assert term.interpret(None) == depth
    assert len(log) == depth


def test_an_interpretation_changed_in_place_takes_effect_immediately() -> None:
    """The same dictionary with different contents is a different interpretation.

    Keying a result on ``id(interpretation)`` reported the old value here, silently: the address
    is unchanged, so the entry looked valid while the meaning of every symbol in it had moved.
    """
    term = Tree("f", (Tree("x", ()),))
    interpretation: dict[str, Any] = {"f": lambda value: value * 2, "x": lambda: 3}

    assert term.interpret(interpretation) == 6

    interpretation["f"] = lambda value: value * 100

    assert term.interpret(interpretation) == 300


def test_two_interpretations_of_one_term_do_not_shadow_each_other() -> None:
    """A term evaluated under two algebras answers under each of them."""
    term = Tree("f", (Tree("x", ()),))
    doubling: dict[str, Any] = {"f": lambda value: value * 2, "x": lambda: 3}
    squaring: dict[str, Any] = {"f": lambda value: value * value, "x": lambda: 3}

    assert [term.interpret(doubling), term.interpret(squaring)] == [6, 9]
    assert [term.interpret(doubling), term.interpret(squaring)] == [6, 9]


def test_an_evaluated_term_still_pickles_and_carries_no_result() -> None:
    """Evaluating a term must not make it unpicklable, whatever the algebra was built from.

    Terms are pickled to move a population between processes, and an algebra assembled from
    lambdas is the ordinary case, so a node that kept its interpretation could not travel.
    """
    term = Tree("f", (Tree("x", ()),))
    interpretation: dict[str, Any] = {"f": lambda value: value * 2, "x": lambda: 3}

    assert term.interpret(interpretation) == 6
    restored = pickle.loads(pickle.dumps(term))

    assert restored == term
    assert restored.interpret(interpretation) == 6


def test_a_combinator_that_cannot_be_interpreted_is_reported_every_time() -> None:
    """A failure is reported on every evaluation, never remembered and never turned into a value."""
    term = Tree("b", (Tree("leaf", ()),))
    interpretation: dict[str, Any] = {"b": math.log, "leaf": lambda: 1.0}

    for _ in range(2):
        with pytest.raises(TypeError, match="does not expose a signature"):
            term.interpret(interpretation)
