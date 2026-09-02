"""Termination conditions.

Every condition reads the state and keeps nothing of its own, so one instance may run any number of
searches. That is the property worth pinning: a condition counting stalled generations in an
attribute would carry one run's count into the next.

:class:`TargetFitness` is the one that needs more than the state, because it needs an order and the
state carries none. It is given the order rather than reading one off, and the tests below pin what
that buys: the same state answers differently under two orders, and an incomparable best does not
count as having arrived.
"""

import math

import pytest

from cosy.core.tree import Tree
from cosy.evolutionary_algorithms import (
    AnyOf,
    EAState,
    Fitness,
    Generations,
    NoImprovement,
    ParetoFitnessComparator,
    ScalarFitnessComparator,
    TargetFitness,
    Termination,
)
from tests.ea_fixtures import a2

MAXIMIZING = ScalarFitnessComparator(greater_is_better=True)
MINIMIZING = ScalarFitnessComparator(greater_is_better=False)


def state(generation: int, last_improvement: int, best_fitness: Fitness = 1.0) -> EAState:
    """Build a state carrying only what a termination condition reads.

    Args:
        generation (int): The generation number.
        last_improvement (int): The generation in which the best-so-far last changed.
        best_fitness (Fitness): The fitness of the best-so-far. (Default value = 1.0)

    Returns:
        EAState: The state.
    """
    individual = Tree(a2, ())
    return EAState(
        generation=generation,
        population=[individual],
        fitness={individual: best_fitness},
        offspring=[],
        best=individual,
        best_fitness=best_fitness,
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


# ---------------------------------------------------------------------------
# TargetFitness: stopping on arrival rather than on a budget
# ---------------------------------------------------------------------------


def test_reaching_the_target_stops_the_run():
    """A run whose objective has a known best value is done when it reaches it."""
    condition = TargetFitness(1.0, MAXIMIZING)
    assert condition.is_satisfied(state(7, 7, best_fitness=1.0))


def test_passing_the_target_stops_the_run_too():
    """The condition is "at least as fit as", not "equal to"."""
    assert TargetFitness(1.0, MAXIMIZING).is_satisfied(state(7, 7, best_fitness=2.0))


def test_falling_short_of_the_target_does_not_stop_the_run():
    """Below the target there is work left."""
    assert not TargetFitness(1.0, MAXIMIZING).is_satisfied(state(7, 7, best_fitness=0.5))


def test_the_target_is_read_in_the_order_it_was_given():
    """The order is the condition's own, and one state answers differently under two of them.

    This is what taking the comparator rather than reading one off the state buys. A loss is
    minimized, so a best of 0.0 has arrived at a target of 0.0 while a best of 2.0 has not, and
    under a maximizing order the same two states answer the other way around.
    """
    minimizing = TargetFitness(0.0, MINIMIZING)
    assert minimizing.is_satisfied(state(1, 1, best_fitness=0.0))
    assert not minimizing.is_satisfied(state(1, 1, best_fitness=2.0))

    maximizing = TargetFitness(0.0, MAXIMIZING)
    assert maximizing.is_satisfied(state(1, 1, best_fitness=2.0))
    assert not maximizing.is_satisfied(state(1, 1, best_fitness=-1.0))


def test_a_best_incomparable_to_the_target_does_not_stop_the_run():
    """The order declined to rank the two, and reading that as arrival would invent a rank.

    Under componentwise dominance a best that trades one objective for another neither reaches the
    target nor falls short of it. Stopping there would report an optimum the order never granted.
    """
    condition = TargetFitness((1.0, 1.0), ParetoFitnessComparator())
    assert not condition.is_satisfied(state(1, 1, best_fitness=(2.0, 0.0)))
    assert condition.is_satisfied(state(1, 1, best_fitness=(1.0, 1.0)))
    assert condition.is_satisfied(state(1, 1, best_fitness=(2.0, 3.0)))


def test_a_failed_measurement_does_not_stop_the_run():
    """A best that could not be measured is incomparable, and that is not arrival either."""
    assert not TargetFitness(1.0, MAXIMIZING).is_satisfied(state(1, 1, best_fitness=math.nan))


def test_the_target_condition_is_reusable_across_runs():
    """It holds the target and the order, and nothing about any run in particular."""
    condition = TargetFitness(1.0, MAXIMIZING)
    assert condition.is_satisfied(state(9, 9, best_fitness=1.0))
    assert not condition.is_satisfied(state(0, 0, best_fitness=0.0))


# ---------------------------------------------------------------------------
# AnyOf: the disjunction
# ---------------------------------------------------------------------------


def test_any_of_stops_when_one_of_its_conditions_does():
    """A budget or the optimum, whichever comes first."""
    condition = AnyOf(Generations(100), TargetFitness(1.0, MAXIMIZING))
    assert condition.is_satisfied(state(3, 3, best_fitness=1.0))
    assert condition.is_satisfied(state(100, 0, best_fitness=0.0))


def test_any_of_runs_on_while_none_of_its_conditions_holds():
    """The disjunction of two conditions that both say go is a condition that says go."""
    condition = AnyOf(Generations(100), TargetFitness(1.0, MAXIMIZING))
    assert not condition.is_satisfied(state(3, 3, best_fitness=0.0))


def test_any_of_keeps_its_components_rather_than_collapsing_them():
    """Both halves stay legible in the record of the run, which is the reason for the component."""
    budget, target = Generations(100), TargetFitness(1.0, MAXIMIZING)
    assert AnyOf(budget, target).conditions == (budget, target)


def test_any_of_nests():
    """A disjunction is a condition, so it combines with the others on equal terms."""
    inner = AnyOf(TargetFitness(1.0, MAXIMIZING))
    assert AnyOf(Generations(100), inner).is_satisfied(state(3, 3, best_fitness=1.0))


def test_any_of_refuses_to_be_empty():
    """The empty disjunction holds on no state, so the run it bounds would never stop."""
    with pytest.raises(ValueError, match="never holds"):
        AnyOf()


def test_every_condition_satisfies_the_protocol():
    """The component class is structural."""
    assert isinstance(Generations(1), Termination)
    assert isinstance(NoImprovement(1), Termination)
    assert isinstance(TargetFitness(1.0, MAXIMIZING), Termination)
    assert isinstance(AnyOf(Generations(1)), Termination)
