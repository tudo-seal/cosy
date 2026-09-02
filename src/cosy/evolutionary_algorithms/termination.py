"""Termination conditions: when the outer loop stops.

Like selection, termination is filled with standard methods rather than with a construction of its
own. The convergence argument for this algorithm runs it *without* a termination condition, since a
condition can only cut the guarantee short and never establish it. Three standard conditions fill
the inventory, and :class:`AnyOf` combines them where a run is bounded by more than one of them at
once.

Every condition reads the state and holds none of its own. That is deliberate: a condition counting
generations without improvement in an attribute of its own would carry the count of one run into
the next, and the same object is meant to be reusable across runs. The driver records when the
best-so-far individual last changed, so the count is a subtraction on the state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

from cosy.evolutionary_algorithms.fitness import Comparison

if TYPE_CHECKING:
    from collections.abc import Hashable

    from cosy.evolutionary_algorithms.evolutionary import EAState
    from cosy.evolutionary_algorithms.fitness import Fitness, FitnessComparator

T = TypeVar("T", bound="Hashable")  # type of terminals

__all__ = ["AnyOf", "Generations", "NoImprovement", "TargetFitness", "Termination"]


@runtime_checkable
class Termination(Protocol[T]):
    """A predicate on the state of a run."""

    def is_satisfied(self, state: EAState[T]) -> bool:
        """Decide whether the run should stop.

        Args:
            state (EAState[T]): The state just produced.

        Returns:
            bool: True if the run should stop after this state.
        """
        ...


class Generations(Termination[T]):
    """Stop after a fixed number of generations.

    The initial population is generation 0, so ``Generations(n)`` admits exactly ``n`` passes of
    the outer loop.

    Attributes:
        count (int): The number of generations to run.
    """

    def __init__(self, count: int) -> None:
        """Build the condition.

        Args:
            count (int): The number of generations, at least 0.

        Raises:
            ValueError: If the count is negative.
        """
        if count < 0:
            msg = f"a number of generations cannot be negative: {count}"
            raise ValueError(msg)
        self.count = count

    def is_satisfied(self, state: EAState[T]) -> bool:
        """Decide whether the generation budget is spent.

        Args:
            state (EAState[T]): The state just produced.

        Returns:
            bool: True once the state's generation has reached the count.
        """
        return state.generation >= self.count


class NoImprovement(Termination[T]):
    """Stop after a fixed number of generations without a fitter best-so-far individual.

    The count is read off the state rather than kept here, so one instance may run any number of
    searches. What counts as an improvement is the driver's decision and the same one the algorithm
    makes: a member strictly fitter than the incumbent in the partial order. Under a partial order
    an incomparable member is not an improvement, so a run producing only incomparable individuals
    terminates. That is the honest outcome, the order having declined to say that anything got
    better.

    Attributes:
        patience (int): Generations without improvement before stopping.
    """

    def __init__(self, patience: int) -> None:
        """Build the condition.

        Args:
            patience (int): The number of generations without improvement, at least 1.

        Raises:
            ValueError: If the patience is smaller than 1. Zero would stop before the first
                generation could improve on anything.
        """
        if patience < 1:
            msg = f"patience counts generations and must be at least 1: {patience}"
            raise ValueError(msg)
        self.patience = patience

    def is_satisfied(self, state: EAState[T]) -> bool:
        """Decide whether the best-so-far individual has stalled for long enough.

        Args:
            state (EAState[T]): The state just produced.

        Returns:
            bool: True once ``patience`` generations have passed without an improvement.
        """
        return state.generation - state.last_improvement >= self.patience


class TargetFitness(Termination[T]):
    """Stop once the best-so-far individual has reached a given fitness.

    The condition an optimum-seeking run wants and neither of the others expresses. A run whose
    objective has a *known* best value has nothing left to do once it reaches that value, and
    spending the remaining generations on it is waste rather than thoroughness. A loss counting
    the rows a candidate gets wrong is the case at hand: it bottoms out at zero, and reaching zero
    certifies the candidate does what was asked of it.

    The comparator is taken here rather than read off the driver, because a termination condition
    is a predicate on the state and the state carries no order. It has to be the order the run
    optimizes under: a condition comparing in one order while the search improves in another would
    stop at the wrong moment, or never.

    Attributes:
        target (Fitness): The fitness value at which the run ends.
        comparator (FitnessComparator): The order to compare in.
    """

    def __init__(self, target: Fitness, comparator: FitnessComparator) -> None:
        """Build the condition.

        Args:
            target (Fitness): The fitness at which to stop.
            comparator (FitnessComparator): The order the run optimizes under.
        """
        self.target = target
        self.comparator = comparator

    def is_satisfied(self, state: EAState[T]) -> bool:
        """Decide whether the best-so-far has reached the target.

        Args:
            state (EAState[T]): The state just produced.

        Returns:
            bool: True once the best-so-far is at least as fit as the target. A best incomparable
                to the target does not stop the run: the order declined to say it had reached
                anything, and reading that as success would invent a rank the order refused.
        """
        return self.comparator.compare(state.best_fitness, self.target) in {
            Comparison.GREATER,
            Comparison.EQUAL,
        }


class AnyOf(Termination[T]):
    """Stop as soon as any one of several conditions is satisfied.

    Conditions are predicates on the state, so their disjunction is one too. A run bounded by a
    generation budget *and* by reaching its optimum, whichever comes first, needs exactly this,
    and writing it as a component keeps both halves legible in the record of the run, which a
    hand-rolled combined condition would not.

    Attributes:
        conditions (tuple[Termination[T], ...]): The conditions, read in the order given.
    """

    def __init__(self, *conditions: Termination[T]) -> None:
        """Build the disjunction.

        Args:
            *conditions (Termination[T]): The conditions to combine, at least one.

        Raises:
            ValueError: If no condition is given. The empty disjunction holds on no state, so the
                run it bounds would never stop, and a run without a bound is a decision to state
                rather than one to arrive at by handing over an empty list.
        """
        if not conditions:
            msg = "a disjunction of no conditions never holds, so the run it bounds would never stop"
            raise ValueError(msg)
        self.conditions = conditions

    def is_satisfied(self, state: EAState[T]) -> bool:
        """Decide whether any of the conditions is satisfied.

        Args:
            state (EAState[T]): The state just produced.

        Returns:
            bool: True if at least one condition says to stop.
        """
        return any(condition.is_satisfied(state) for condition in self.conditions)
