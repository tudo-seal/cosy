"""Termination conditions: when the outer loop stops.

Like selection, termination is filled with standard methods rather than with a construction of its
own. The convergence argument for this algorithm runs it *without* a termination condition, since a
condition can only cut the guarantee short and never establish it. Two standard conditions are
enough for the inventory.

Both read the state and hold no state of their own. That is deliberate: a condition counting
generations without improvement in an attribute of its own would carry the count of one run into
the next, and the same object is meant to be reusable across runs. The driver records when the
best-so-far individual last changed, so the count is a subtraction on the state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Hashable

    from cosy.evolutionary_algorithms.evolutionary import EAState

T = TypeVar("T", bound="Hashable")  # type of terminals

__all__ = ["Generations", "NoImprovement", "Termination"]


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
