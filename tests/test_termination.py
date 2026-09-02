"""Termination conditions.

Both conditions read the state and keep nothing of their own, so one instance may run any number of
searches. That is the property worth pinning: a condition counting stalled generations in an
attribute would carry one run's count into the next.
"""

import pytest

from cosy.core.tree import Tree
from cosy.evolutionary_algorithms import (
    EAState,
    Generations,
    NoImprovement,
    Termination,
)
from tests.ea_fixtures import a2


def state(generation: int, last_improvement: int) -> EAState:
    """Build a state carrying only what a termination condition reads.

    Args:
        generation (int): The generation number.
        last_improvement (int): The generation in which the best-so-far last changed.

    Returns:
        EAState: The state.
    """
    individual = Tree(a2, ())
    return EAState(
        generation=generation,
        population=[individual],
        fitness={individual: 1.0},
        offspring=[],
        best=individual,
        best_fitness=1.0,
        last_improvement=last_improvement,
    )


def test_generations_admits_exactly_its_count():
    """The initial population is generation 0, so ``Generations(n)`` allows n passes."""
    condition = Generations(3)
    assert not condition.is_satisfied(state(0, 0))
    assert not condition.is_satisfied(state(2, 0))
    assert condition.is_satisfied(state(3, 0))


def test_generations_of_zero_stops_at_once():
    """A budget of nothing is spent immediately."""
    assert Generations(0).is_satisfied(state(0, 0))


def test_generations_refuses_a_negative_count():
    """A count of generations is not negative."""
    with pytest.raises(ValueError, match="negative"):
        Generations(-1)


def test_no_improvement_counts_from_the_last_change():
    """The count is a subtraction on the state, not an attribute of the condition."""
    condition = NoImprovement(2)
    assert not condition.is_satisfied(state(5, 4))
    assert condition.is_satisfied(state(6, 4))


def test_no_improvement_is_reusable_across_runs():
    """One instance, two runs: nothing carries over.

    A condition holding its own counter would report the first run's stall in the second.
    """
    condition = NoImprovement(2)
    assert condition.is_satisfied(state(9, 3))
    assert not condition.is_satisfied(state(0, 0))
    assert not condition.is_satisfied(state(1, 1))


def test_no_improvement_needs_at_least_one_generation():
    """Zero patience would stop before anything could improve."""
    with pytest.raises(ValueError, match="at least 1"):
        NoImprovement(0)


def test_both_conditions_satisfy_the_protocol():
    """The component class is structural."""
    assert isinstance(Generations(1), Termination)
    assert isinstance(NoImprovement(1), Termination)
