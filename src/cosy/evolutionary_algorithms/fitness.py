"""Fitness evaluation and comparison components for evolutionary algorithms.

This module provides a flexible framework for both single-objective and multi-objective optimization.
It defines the Fitness type and various comparators that implement different optimization strategies,
enabling the evolutionary algorithm to handle diverse problem domains.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

# Fitness can be either a single scalar value or a multi-dimensional vector
Fitness = float | Sequence[float]


class FitnessComparator(Protocol):
    """Protocol for comparing fitness values and providing sorting keys.

    Implementations of this protocol define how fitness values are compared and ordered.
    This allows the evolutionary algorithm to support both single-objective and multi-objective
    optimization by simply swapping the comparator.
    """

    def compare(self, first: Fitness, second: Fitness) -> int:
        """Compare two fitness values.

        Args:
            first (Fitness): First fitness value to compare.
            second (Fitness): Second fitness value to compare.

        Returns:
            int: 1 if first is better, -1 if second is better, 0 if tie.
        """
        ...

    def scalarize(self, fitness: Fitness) -> float:
        """Map fitness to a single scalar where larger is better.

        This method is used for selection operators that require a single numeric value
        for weighting or ranking purposes.

        Args:
            fitness (Fitness): The fitness value to convert to a scalar.

        Returns:
            float: A scalar value where larger values represent better fitness.
        """
        ...

    def sort_key(self, fitness: Fitness) -> float:
        """Return a sort key where larger values are better fitness.

        This is primarily used for sorting individuals by fitness.

        Args:
            fitness (Fitness): The fitness value to convert to a sort key.

        Returns:
            float: A numeric value suitable for sorting, where larger is better.
        """
        ...


@dataclass(frozen=True)
class ScalarFitnessComparator:
    """Single-objective fitness comparator for scalar fitness values.

    This comparator handles simple scalar fitness optimization. It can be configured
    for both maximization and minimization problems.

    Attributes:
        greater_is_better (bool): If True, larger fitness values are better (maximization).
            If False, smaller fitness values are better (minimization).
    """

    greater_is_better: bool = True

    def compare(self, first: Fitness, second: Fitness) -> int:
        """Compare two fitness values using scalar comparison.

        Args:
            first (Fitness): _description_
            second (Fitness): _description_

        Returns:
            int: _description_
        """
        first_value = float(first) if isinstance(first, (int, float)) else float(first[0])
        second_value = float(second) if isinstance(second, (int, float)) else float(second[0])
        if first_value == second_value:
            return 0
        if self.greater_is_better:
            return 1 if first_value > second_value else -1
        return 1 if first_value < second_value else -1

    def scalarize(self, fitness: Fitness) -> float:
        """Convert fitness to scalar, negating if minimization is desired.

        Args:
            fitness (Fitness): _description_

        Returns:
            float: _description_
        """
        value = float(fitness) if isinstance(fitness, (int, float)) else float(fitness[0])
        return value if self.greater_is_better else -value

    def sort_key(self, fitness: Fitness) -> float:
        """Return the scalarized fitness as the sort key.

        Args:
            fitness (Fitness): _description_

        Returns:
            float: _description_
        """
        return self.scalarize(fitness)


@dataclass(frozen=True)
class ParetoFitnessComparator:
    """Multi-objective fitness comparator using Pareto dominance.

    This comparator handles multi-dimensional fitness using Pareto domination concepts.
    An individual dominates another if it is at least as good in all objectives
    and strictly better in at least one objective.

    Attributes:
        maximize (Sequence[bool] | None): Sequence of booleans indicating which objectives should be maximized.
            If None, all objectives are maximized by default.
        tie_breaker (Callable[[Sequence[float]], float] | None): Optional function to break ties between non-dominated solutions.
            Takes a normalized fitness vector and returns a scalar value.
    """

    maximize: Sequence[bool] | None = None
    tie_breaker: Callable[[Sequence[float]], float] | None = None

    def _as_vector(self, fitness: Fitness) -> tuple[float, ...]:
        """Convert fitness to a vector representation.

        Args:
            fitness (Fitness): _description_

        Returns:
            tuple[float, ...]: _description_
        """
        if isinstance(fitness, (int, float)):
            return (float(fitness),)
        return tuple(float(v) for v in fitness)

    def _maximize_flags(self, length: int) -> tuple[bool, ...]:
        """Get the maximize flags, using all-True if not specified.

        Args:
            length (int): _description_

        Returns:
            tuple[bool, ...]: _description_

        Raises:
            ValueError: _description_
        """
        if self.maximize is None:
            return tuple(True for _ in range(length))
        if len(self.maximize) != length:
            msg = "maximize length must match fitness dimension"
            raise ValueError(msg)
        return tuple(bool(v) for v in self.maximize)

    def _normalize(self, fitness: Fitness) -> tuple[float, ...]:
        """Normalize fitness vector so that larger values are always better.

        Args:
            fitness (Fitness): _description_

        Returns:
            tuple[float, ...]: _description_
        """
        vector = self._as_vector(fitness)
        flags = self._maximize_flags(len(vector))
        # Negate components that should be minimized
        return tuple(v if maximize else -v for v, maximize in zip(vector, flags, strict=True))

    def compare(self, first: Fitness, second: Fitness) -> int:
        """Compare two fitness values using Pareto dominance.

        Args:
            first (Fitness): _description_
            second (Fitness): _description_

        Returns:
            int: _description_
        """
        first_vec = self._normalize(first)
        second_vec = self._normalize(second)

        # Check if first dominates second
        first_dominates = all(a >= b for a, b in zip(first_vec, second_vec, strict=True)) and any(
            a > b for a, b in zip(first_vec, second_vec, strict=True)
        )

        # Check if second dominates first
        second_dominates = all(b >= a for a, b in zip(first_vec, second_vec, strict=True)) and any(
            b > a for a, b in zip(first_vec, second_vec, strict=True)
        )

        if first_dominates and not second_dominates:
            return 1
        if second_dominates and not first_dominates:
            return -1
        return 0

    def scalarize(self, fitness: Fitness) -> float:
        """Scalarize fitness using the tie_breaker or sum of normalized objectives.

        Args:
            fitness (Fitness): _description_

        Returns:
            float: _description_
        """
        normalized = self._normalize(fitness)
        if self.tie_breaker is not None:
            return float(self.tie_breaker(normalized))
        return sum(normalized)

    def sort_key(self, fitness: Fitness) -> float:
        """Return the scalarized fitness as the sort key.

        Args:
            fitness (Fitness): _description_

        Returns:
            float: _description_
        """
        return self.scalarize(fitness)
