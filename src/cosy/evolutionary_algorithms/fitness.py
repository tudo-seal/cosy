"""Fitness: a partial order to compare in, and a scalarization where numbers are needed.

A fitness function maps the inhabitants into a **partially ordered set**. The codomain is widened
past the reals on purpose, because comparing several objectives componentwise yields a partial
order, and demanding a total one would exclude exactly that. Two individuals may therefore be
*incomparable*, which is a fourth answer beside better, worse and equally good, and
:class:`Comparison` is where that answer lives.

One selection method needs more than comparisons. Fitness-proportional selection draws with
probability proportional to fitness, and an order supplies no numbers. A *scalarization* is a map
``sigma`` from the fitness codomain into the strictly positive reals that is monotone, meaning that
``a <= b`` implies ``sigma(a) <= sigma(b)``. It therefore respects the fitness order and decides
what the order leaves open. Positivity is what makes every population carry positive total weight,
so the proportional draw is defined on every population, and that is how fitness-proportional
parent selection gives every member a positive chance of becoming a parent.

The two levels are therefore separate types here. A :class:`FitnessComparator` carries the partial
order and nothing else, and a :class:`Scalarization` is a component of its own that only the
components needing numbers ask for. They must agree in direction, and nothing in the type system
says so, so :class:`ExpScalarization` and :class:`ScalarFitnessComparator` take the same flag.

**Fitness algebras** need no machinery of their own. A fitness algebra is an algebra over the
function symbols of a repository whose carrier is partially ordered, and the fitness it induces is
the fold, which is ``Tree.interpret``. :func:`induced_fitness` names that construction. Not every
quality measure is compositional: an acquisition function evaluates an inhabitant against all
inhabitants evaluated before, and enters as a fitness function rather than through an algebra.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from cosy.core.tree import Tree

# Fitness can be a single scalar or a vector of objectives. The comparator decides which of the two
# a value is read as, and the type states the two shapes the framework's own components handle.
Fitness = float | Sequence[float]

__all__ = [
    "Comparison",
    "ExpScalarization",
    "Fitness",
    "FitnessComparator",
    "ParetoFitnessComparator",
    "ScalarFitnessComparator",
    "Scalarization",
    "induced_fitness",
]


class Comparison(Enum):
    """The four answers a partial order gives about two fitness values.

    ``GREATER`` means the first value is the fitter one. A comparator configured to minimize
    reports the reversed order, so that "greater" is "fitter" for every consumer.

    ``INCOMPARABLE`` is not a tie. A tie says the two are equally good and either may be preferred.
    Incomparability says the order declines to rank them, and a component that treats the two alike
    is choosing to, which is a decision to document rather than to hide.

    A value incomparable to **itself** is a different matter. A fitness function maps into a
    partially ordered set, and a partial order is reflexive, so such a value is not a member of the
    codomain at all. ``nan`` is the case that arises in practice, from a measurement that failed.
    The comparators here report it as ``INCOMPARABLE`` rather than inventing a rank for it, and
    :class:`~cosy.evolutionary_algorithms.evolutionary.EvolutionarySearch` refuses it
    outright, naming the individual whose measurement failed.
    """

    LESS = -1
    EQUAL = 0
    GREATER = 1
    INCOMPARABLE = 2


@runtime_checkable
class FitnessComparator(Protocol):
    """The partial order on the codomain of the fitness function.

    One method, because a partial order is one relation. The map into the reals that proportional
    drawing needs is :class:`Scalarization`, a component of its own, and it is kept off this object
    on purpose. A comparator carrying one invites a caller to scalarize the values it could not
    order, and a weighted sum in place of a partial order is a different optimization problem.
    """

    def compare(self, first: Fitness, second: Fitness) -> Comparison:
        """Compare two fitness values.

        Args:
            first (Fitness): The first value.
            second (Fitness): The second value.

        Returns:
            Comparison: How ``first`` stands to ``second``.
        """
        ...


@runtime_checkable
class Scalarization(Protocol):
    """A monotone map from the fitness codomain into the strictly positive reals.

    Monotonicity, meaning that ``a <= b`` implies ``sigma(a) <= sigma(b)``, and strict positivity
    are both part of the definition. Neither is enforceable for an arbitrary callable, so both are
    stated here and pinned by tests for the instance this module ships.
    """

    def scalarize(self, fitness: Fitness) -> float:
        """Map a fitness value to a positive real.

        Args:
            fitness (Fitness): The value to map.

        Returns:
            float: A strictly positive number, monotone in the fitness order.
        """
        ...


_COMPARATOR_REMEDY = "Pass a scalar fitness, or use ParetoFitnessComparator, which takes every objective into account."
_SCALARIZATION_REMEDY = "Pass a scalar fitness, or a Scalarization that reads every objective."


def _single_objective(fitness: Fitness, remedy: str) -> float:
    """Read the one objective of a single-objective fitness value.

    A scalar is returned as is. A value carrying exactly one objective yields that objective. Two
    or more are refused rather than truncated to objective 0.

    ``int`` and ``float`` take a fast path, and every other value is placed by whether it *has a
    length* rather than by whether it is a ``Sequence``. ``numpy.ndarray`` carries a length without
    being registered as ``collections.abc.Sequence``, so a ``Sequence`` test would route a
    two-objective array to the scalar path, where the refusal below never runs. Asking for a length
    also places float-like scalars such as ``numpy.float32``, which have none, on the scalar path
    without a registry of scalar types to keep up to date.

    Args:
        fitness (Fitness): The fitness value to read.
        remedy (str): The way out named in the refusal. A comparator and a scalarization have
            different ones.

    Returns:
        float: The single objective, as a float.

    Raises:
        ValueError: If fitness carries two or more objectives.
    """
    if not isinstance(fitness, (int, float)):
        try:
            objectives = len(fitness)
        except TypeError:
            # No length: a float-like scalar rather than a vector.
            return float(fitness)  # type: ignore[arg-type]
        if objectives > 1:
            msg = (
                "a single-objective component was given a fitness with "
                f"{objectives} objectives. Reading objective 0 and discarding the rest would "
                f"optimize a different problem without saying so. {remedy}"
            )
            raise ValueError(msg)
        return float(fitness[0])
    return float(fitness)


@dataclass(frozen=True)
class ScalarFitnessComparator:
    """The total order on the reals, in either direction.

    A fitness may arrive as a scalar or as a one-element vector. Two or more objectives are refused
    with a ValueError rather than silently reduced to objective 0. This class is the framework
    default, so that reduction would make every multi-objective run optimize objective 0 alone
    without saying so. :class:`ParetoFitnessComparator` is the comparator that reads every
    objective.

    A total order never answers ``INCOMPARABLE``, which is what makes this the comparator every
    component behaves most simply under.

    Attributes:
        greater_is_better (bool): If True, larger fitness values are the fitter ones
            (maximization). If False, smaller ones are (minimization).
    """

    greater_is_better: bool = True

    def compare(self, first: Fitness, second: Fitness) -> Comparison:
        """Compare two scalar fitness values.

        Args:
            first (Fitness): The first value.
            second (Fitness): The second value.

        Returns:
            Comparison: ``GREATER`` if ``first`` is the fitter one, ``LESS`` if ``second`` is,
                ``EQUAL`` if neither. Never ``INCOMPARABLE``.
        """
        first_value = _single_objective(first, _COMPARATOR_REMEDY)
        second_value = _single_objective(second, _COMPARATOR_REMEDY)
        if math.isnan(first_value) or math.isnan(second_value):
            # Not a tie and not a ranking: ``nan`` stands in no order relation at all, not even to
            # itself. Reporting ``LESS`` in both directions, which is what the raw float
            # comparisons yield, breaks antisymmetry, and a driver reading it then finds nothing
            # ever fitter than a failed measurement.
            return Comparison.INCOMPARABLE
        if first_value == second_value:
            return Comparison.EQUAL
        fitter = first_value > second_value if self.greater_is_better else first_value < second_value
        return Comparison.GREATER if fitter else Comparison.LESS


@dataclass(frozen=True)
class ParetoFitnessComparator:
    """Componentwise dominance: a genuine partial order on vectors of objectives.

    One value dominates another if it is at least as good in every objective and strictly better
    in one. Two vectors that trade objectives against each other dominate neither way and are
    reported ``INCOMPARABLE``, not ``EQUAL``, which is what a comparator says about two values it
    ranks alike.

    This comparator carries no scalarization. Summing the objectives, which is what a weighted sum
    does, turns the Pareto front into one number and picks a point on it by the weights, and that
    is a different problem from the one a multi-objective run states. Where a component genuinely
    needs numbers, a :class:`Scalarization` is passed to it explicitly, and choosing one is then a
    visible decision.

    **What a run under this comparator has and what it does not.** The search rules that read the
    order alone run under it: :class:`~cosy.evolutionary_algorithms.selection.TournamentSelection`,
    :class:`~cosy.evolutionary_algorithms.selection.RankBasedSelection` and
    :class:`~cosy.evolutionary_algorithms.selection.FitnessBasedReplacement`. The two that draw
    proportionally take a scalarization, and the one shipped here reads a single objective, so a
    multi-objective run needs one of its own for them. The almost sure convergence the package
    documents is stated over a scalarization as well, so it says nothing about a run under this
    comparator. What holds without one is weaker and worth naming: the best-so-far never moves to a
    value the current one dominates, so it is maximal among the values the run has seen.

    Attributes:
        maximize (Sequence[bool] | None): Which objectives are maximized. None maximizes all.
    """

    maximize: Sequence[bool] | None = None

    def _normalize(self, fitness: Fitness) -> tuple[float, ...]:
        """Read a fitness value as a vector in which larger is always better.

        Args:
            fitness (Fitness): The value to read.

        A value is placed by whether it *has a length* rather than by its type, for the reason
        :func:`_single_objective` gives: a float-like scalar such as ``numpy.float32`` has none and
        belongs on the scalar path, and ``numpy.ndarray`` has one without being a ``Sequence``.

        Returns:
            tuple[float, ...]: The objectives, minimized ones negated.

        Raises:
            ValueError: If ``maximize`` was given and its length does not match.
        """
        try:
            objectives = tuple(float(value) for value in fitness)  # type: ignore[union-attr]
        except TypeError:
            objectives = (float(fitness),)  # type: ignore[arg-type]
        vector = objectives
        if self.maximize is None:
            return vector
        if len(self.maximize) != len(vector):
            msg = f"maximize has {len(self.maximize)} flags for {len(vector)} objectives"
            raise ValueError(msg)
        return tuple(value if maximize else -value for value, maximize in zip(vector, self.maximize, strict=True))

    def compare(self, first: Fitness, second: Fitness) -> Comparison:
        """Compare two fitness vectors by componentwise dominance.

        Args:
            first (Fitness): The first vector.
            second (Fitness): The second vector.

        Returns:
            Comparison: ``GREATER`` or ``LESS`` if one dominates, ``EQUAL`` if the two vectors
                agree in every objective, ``INCOMPARABLE`` if they trade objectives or if either
                carries a ``nan``. A ``nan`` compares false in both directions, which is the
                answer a failed measurement gets in every comparator here.

        Raises:
            ValueError: If ``maximize`` was given and its length does not match the number of
                objectives, or if the two values carry a different number of them.
        """
        left = self._normalize(first)
        right = self._normalize(second)
        if len(left) != len(right):
            msg = f"a comparison needs the same objectives on both sides, but got {len(left)} and {len(right)}"
            raise ValueError(msg)
        pairs = list(zip(left, right, strict=True))
        never_worse = all(a >= b for a, b in pairs)
        never_better = all(a <= b for a, b in pairs)
        if never_worse and never_better:
            return Comparison.EQUAL
        if never_worse:
            return Comparison.GREATER
        if never_better:
            return Comparison.LESS
        return Comparison.INCOMPARABLE


@dataclass(frozen=True)
class ExpScalarization:
    """``x -> e^x``, the scalarization of the reals.

    Strictly positive and strictly monotone on all of the reals, so it turns any real-valued
    fitness into proportional-selection weights without a shift, a span or a floor. That is what it
    is for. Raw fitness values as roulette weights make every share depend on where the zero of the
    fitness happens to sit, and lifting negative values into range moves that zero and every share
    with it. The exponential turns differences into ratios, so a constant shift of the whole
    population leaves the shares alone, and ``scale`` below becomes the one place the selection
    pressure is set.

    **A numerical limit, stated rather than patched.** Positivity holds in the reals, while
    ``math.exp`` underflows to ``0.0`` below about ``-745`` and overflows above ``710``. A
    population whose fitness values sit there receives zero weights, and a component drawing
    proportionally then no longer gives every member a positive share. Clamping to the smallest
    positive float would hide that instead of fixing it, so the fix is to bring the values into
    range, and ``scale`` is the parameter for it, because telling a caller to "scale the fitness"
    without offering a way to is not advice. A mean squared error in the thousands is an ordinary
    fitness, and under a minimizing order it lands exactly in the underflowing half.

    Attributes:
        greater_is_better (bool): The direction of the fitness order this scalarization is monotone
            in. It must match the comparator's, because with a minimizing comparator ``e^x`` grows
            where the order falls, which is exactly what monotonicity forbids.
        scale (float): A positive divisor applied before the exponential, so the map is
            ``x -> e^{x/scale}``. Monotonicity is unaffected, dividing by a positive number being
            order-preserving, and the selection pressure falls as the scale grows. It is the
            temperature of the proportional draw.
    """

    greater_is_better: bool = True
    scale: float = 1.0

    def __post_init__(self) -> None:
        """Reject a scale that would break monotonicity.

        Raises:
            ValueError: If the scale is not strictly positive. Zero divides, and a negative scale
                reverses the order, which is what monotonicity forbids.
        """
        if not self.scale > 0.0:
            msg = f"the scale divides and must be strictly positive: {self.scale}"
            raise ValueError(msg)

    def scalarize(self, fitness: Fitness) -> float:
        """Map a real fitness to a positive weight.

        Args:
            fitness (Fitness): The value to map.

        Returns:
            float: ``e^{x/scale}``, or ``e^{-x/scale}`` under a minimizing order. ``inf`` for a
                fitness that is infinitely good, and ``nan`` for a fitness that is not a number.
                Neither is replaced by a substitute, and the consumer states what it does with
                them.

        Raises:
            ValueError: If a finite fitness overflows. A proportional draw reads infinity as an
                infinitely good individual and gives it the whole mass, so a finite fitness must
                not produce one. The scale is the parameter that brings the values into range.
        """
        value = _single_objective(fitness, _SCALARIZATION_REMEDY)
        if not self.greater_is_better:
            value = -value
        if math.isnan(value):
            return value
        try:
            weight = math.exp(value / self.scale)
        except OverflowError:
            weight = math.inf
        if math.isinf(weight) and not math.isinf(value):
            msg = (
                f"the exponential overflows on a fitness of {fitness!r} at a scale of {self.scale}, "
                f"and an infinite weight would take the whole mass of a proportional draw"
            )
            raise ValueError(msg)
        return weight


def induced_fitness(
    interpretation: dict[Hashable, Any] | None = None,
) -> Callable[[Tree[Any]], Any]:
    """Build the fitness function induced by a fitness algebra.

    A fitness algebra is an algebra over the function symbols of a repository whose carrier is
    partially ordered. The algebra is the interpretation, the induced fitness is the fold, and the
    fold is ``Tree.interpret``, so this is a name for a construction rather than machinery of its
    own. The carrier has to be partially ordered, and the comparator the search runs with is the
    order on it. Neither is checked here, because an algebra is a mapping of symbols and its
    carrier is whatever the mapping returns.

    Args:
        interpretation (dict[Hashable, Any] | None): The algebra, a map from function symbols to
            their meaning. None uses each symbol's own callable, which is what a repository built
            from Python functions already carries. (Default value = None)

    Returns:
        Callable[[Tree[Any]], Any]: The induced fitness function.
    """

    def evaluate(individual: Tree[Any]) -> Any:
        """Fold one individual through the algebra.

        Args:
            individual (Tree[Any]): The inhabitant to evaluate.

        Returns:
            Any: Its image under the fold.
        """
        return individual.interpret(interpretation)

    return evaluate
