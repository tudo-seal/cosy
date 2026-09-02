"""Selection: the parent selection and the survivor selection of a run.

This module holds a curated inventory of standard methods rather than a complete one. What it adds
to each of them is the record of which condition for almost sure convergence it discharges, and
that record is the reason the inventory is curated.

**Two contracts, not one.** A parent selection method maps a population to a **pair** of its
members. A survivor selection method maps the parents and the offspring **together** to a
population of mu individuals among them. They are separate protocols with separate method names,
so passing a replacement where a parent selection belongs is a type error rather than a silent
misconfiguration.

**The order is partial**, and every order-based component here consumes it through
:class:`~cosy.evolutionary_algorithms.fitness.Comparison`. Where a total order is needed, as in
ranking and in truncation, the canonical construction over a partial order is used: *dominance
fronts*, the individuals no other dominates, then the same among the rest, and so on. Under a total
order the fronts are exactly the classes of equally fit individuals, so every component below
reduces to its textbook form and equally fit individuals keep sharing a rank.

**Where numbers are needed, a scalarization is passed in.** Only proportional drawing needs them.
A scalarization is strictly positive by definition, so the weights *are* its values and nothing is
lifted, shifted or floored on the way to them. A lift applied to negative values would move every
share along with it, and nothing here needs lifting.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

from cosy.evolutionary_algorithms.fitness import Comparison

if TYPE_CHECKING:
    import random
    from collections.abc import Hashable, Mapping, Sequence

    from cosy.core.tree import Tree
    from cosy.evolutionary_algorithms.fitness import (
        Fitness,
        FitnessComparator,
        Scalarization,
    )

T = TypeVar("T", bound="Hashable")  # type of terminals

__all__ = [
    "FitnessBasedReplacement",
    "FitnessProportionalSelection",
    "GenerousConservativeReplacement",
    "LexicaseSelection",
    "ParentSelection",
    "RankBasedSelection",
    "SurvivorSelection",
    "TournamentSelection",
    "dominance_fronts",
]


def dominance_fronts(
    individuals: Sequence[Tree[T]],
    fitness: Mapping[Tree[T], Fitness],
    comparator: FitnessComparator,
) -> list[list[Tree[T]]]:
    """Sort individuals into dominance fronts, fittest front first.

    The first front holds the individuals no other individual is fitter than, the second holds
    those of the rest, and so on. This is the canonical linearization of a partial order into
    ranks: it uses only the relation, it makes no arbitrary choice among incomparable individuals,
    and under a total order it degenerates to the ordinary ranking with equally fit individuals
    sharing a front.

    Finding one front scans each individual not yet in a front against the others. A scan ends at
    the first individual that is fitter, so a front costs at most the square of the number not yet
    in a front. The construction does that once per front. A population in which no member is
    fitter than any other is a single front, so the cost is quadratic in the number of individuals.
    A total order with no two members equally fit gives every member a front of its own, so the
    cost is at most cubic. A population that arrives fittest first stays quadratic even then, since
    removing a front keeps the order and every scan but the fittest individual's ends at the first
    individual it meets.

    Args:
        individuals (Sequence[Tree[T]]): The individuals to rank. Repetitions are kept, since a
            population is a multiset and a repeated individual occupies a place of its own.
        fitness (Mapping[Tree[T], Fitness]): The fitness of each individual.
        comparator (FitnessComparator): The partial order.

    Returns:
        list[list[Tree[T]]]: The fronts, fittest first. Within a front the input order is kept,
            so a caller that has to cut a front short cuts it deterministically.
    """
    remaining = list(individuals)
    fronts: list[list[Tree[T]]] = []
    while remaining:
        front = [
            candidate
            for candidate in remaining
            if not any(
                comparator.compare(fitness[other], fitness[candidate]) is Comparison.GREATER for other in remaining
            )
        ]
        fronts.append(front)
        # Removing by identity of the front's membership would drop every copy of a repeated
        # individual at once, which is correct: copies are equally fit, so they share a front.
        front_members = set(front)
        remaining = [candidate for candidate in remaining if candidate not in front_members]
    return fronts


def _proportional_weights(values: Sequence[float]) -> list[float]:
    """Turn scalarized fitness values into drawing weights.

    A scalarization takes strictly positive values, so the weights *are* the values and no lift,
    shift or normalization is involved. What remains is the policy for values a measurement
    can produce but the definition of a scalarization excludes. An infinitely good individual is
    the limit of proportional selection and takes the whole mass. A value that is not a number is a
    failed measurement, takes none, and is never drawn. A population in which every value failed is
    drawn uniformly, there being nothing left to prefer.

    The asymmetry between the two policies is deliberate and worth naming, because both end in a
    weight of zero. ``nan`` and ``inf`` are properties of the *fitness*: a measurement failed, or
    an individual is infinitely good, and the scalarization reports them faithfully. A value of
    zero or below is a property of the *scalarization*: it was handed an ordinary number and
    returned something outside its own codomain. The first is data the caller cannot avoid, so it
    gets a documented policy. The second is a broken contract, so it is reported.

    Args:
        values (Sequence[float]): The scalarized fitness values.

    Returns:
        list[float]: Finite, non-negative weights, not all zero.

    Raises:
        ValueError: If a value is zero or negative. A scalarization maps into the *positive* reals,
            so this is a broken component contract rather than a degenerate population, and it is
            reachable numerically. ``math.exp`` underflows to 0.0 below about -745, so a fitness
            that far from zero silently loses the positivity that a generous parent selection rests
            on. The fix is the ``scale`` parameter of
            :class:`~cosy.evolutionary_algorithms.fitness.ExpScalarization`.
    """
    if any(value <= 0.0 for value in values):
        offenders = [value for value in values if value <= 0.0]
        msg = (
            f"a scalarization maps into the strictly positive reals, but produced {offenders!r}. "
            "With an exponential scalarization this is underflow, and the fix is to scale the "
            "fitness rather than to floor the weight"
        )
        raise ValueError(msg)
    if any(value == math.inf for value in values):
        return [1.0 if value == math.inf else 0.0 for value in values]
    weights = [value if math.isfinite(value) else 0.0 for value in values]
    if not any(weight > 0.0 for weight in weights):
        return [1.0] * len(values)
    return weights


@runtime_checkable
class ParentSelection(Protocol[T]):
    """A map from a population to a pair of its members.

    The pair is drawn with replacement, so the two parents may be the same individual, exactly as
    two independent tournaments may have the same winner.
    """

    def select_parents(
        self,
        population: Sequence[Tree[T]],
        fitness: Mapping[Tree[T], Fitness],
        comparator: FitnessComparator,
    ) -> tuple[Tree[T], Tree[T]]:
        """Draw two parents.

        Args:
            population (Sequence[Tree[T]]): The current population as a multiset.
            fitness (Mapping[Tree[T], Fitness]): The fitness of each member.
            comparator (FitnessComparator): The partial order on fitness values.

        Returns:
            tuple[Tree[T], Tree[T]]: The two parents.
        """
        ...


@runtime_checkable
class SurvivorSelection(Protocol[T]):
    """A map from the parents and the offspring together to the next population.

    Receiving both is the contract, not a convenience: generational replacement is the instance
    that ignores the parents, and it is exactly the instance the convergence guarantee excludes.
    A component that only ever sees the offspring could not be anything else.
    """

    def select_survivors(
        self,
        parents: Sequence[Tree[T]],
        offspring: Sequence[Tree[T]],
        fitness: Mapping[Tree[T], Fitness],
        comparator: FitnessComparator,
        size: int,
    ) -> list[Tree[T]]:
        """Choose the next population.

        Args:
            parents (Sequence[Tree[T]]): The population of the finished generation.
            offspring (Sequence[Tree[T]]): The offspring produced from it.
            fitness (Mapping[Tree[T], Fitness]): The fitness of every individual in either.
            comparator (FitnessComparator): The partial order on fitness values.
            size (int): The population size mu.

        Returns:
            list[Tree[T]]: Exactly ``size`` individuals drawn from the parents and the offspring.
        """
        ...


class TournamentSelection(ParentSelection[T]):
    """Draw a tournament uniformly and keep an individual no member of it is fitter than.

    **A tournament of size 2 or more is not generous**, and the reasoning is worth writing out.
    Almost sure convergence asks that *every* member of the population be drawn as a parent with
    positive probability. A tournament is sampled without replacement, so the least fit member of a
    population with pairwise distinct fitness meets a fitter competitor in every tournament it
    enters, and it wins none. Its probability is exactly zero. Only ``tournament_size == 1``, which
    is uniform parent selection, gives every member a positive share, and it does so for the
    trivial reason. Tournament selection is in the inventory because it is the standard method with
    a tunable pressure, not because it carries the guarantee.
    :class:`FitnessProportionalSelection` and :class:`RankBasedSelection` below a pressure of 2 are
    the two that do.

    Under a partial order a tournament can have several undominated members, and one of them is
    taken uniformly. A tie is treated the same way, so equally fit members share the win evenly
    rather than through a chain of coin flips.

    Attributes:
        tournament_size (int): The number of individuals per tournament.
        rng (random.Random): The source of randomness.
    """

    def __init__(self, tournament_size: int, rng: random.Random) -> None:
        """Build the operator.

        Args:
            tournament_size (int): Individuals per tournament, at least 1.
            rng (random.Random): The source of randomness.

        Raises:
            ValueError: If the tournament size is smaller than 1.
        """
        if tournament_size < 1:
            msg = f"a tournament needs at least one participant: {tournament_size}"
            raise ValueError(msg)
        self.tournament_size = tournament_size
        self.rng = rng

    def _winner(
        self,
        population: Sequence[Tree[T]],
        fitness: Mapping[Tree[T], Fitness],
        comparator: FitnessComparator,
    ) -> Tree[T]:
        """Run one tournament.

        Args:
            population (Sequence[Tree[T]]): The population to draw from.
            fitness (Mapping[Tree[T], Fitness]): The fitness of each member.
            comparator (FitnessComparator): The partial order.

        Returns:
            Tree[T]: The winner.
        """
        tournament = self.rng.sample(list(population), min(self.tournament_size, len(population)))
        undominated = [
            candidate
            for candidate in tournament
            if not any(
                comparator.compare(fitness[other], fitness[candidate]) is Comparison.GREATER for other in tournament
            )
        ]
        return self.rng.choice(undominated)

    def select_parents(
        self,
        population: Sequence[Tree[T]],
        fitness: Mapping[Tree[T], Fitness],
        comparator: FitnessComparator,
    ) -> tuple[Tree[T], Tree[T]]:
        """Run two tournaments.

        Args:
            population (Sequence[Tree[T]]): The current population.
            fitness (Mapping[Tree[T], Fitness]): The fitness of each member.
            comparator (FitnessComparator): The partial order.

        Returns:
            tuple[Tree[T], Tree[T]]: The two winners.

        Raises:
            ValueError: If the population is empty.
        """
        if not population:
            msg = "parent selection needs a non-empty population"
            raise ValueError(msg)
        return (
            self._winner(population, fitness, comparator),
            self._winner(population, fitness, comparator),
        )


class FitnessProportionalSelection(ParentSelection[T]):
    """Draw parents with probability proportional to their scalarized fitness.

    Every member receives a positive share *because* a scalarization is strictly positive, so this
    method is generous, and that is the reason the definition of a scalarization demands positivity
    in the first place.

    The scalarization is a parameter of the components that draw proportionally, this one and
    :class:`GenerousConservativeReplacement`, rather than of the search.
    :class:`~cosy.evolutionary_algorithms.evolutionary.EvolutionarySearch` does not take
    one as an input, and this is why: only proportional drawing needs numbers.

    Attributes:
        scalarization (Scalarization): The map into the positive reals.
        rng (random.Random): The source of randomness.
    """

    def __init__(self, scalarization: Scalarization, rng: random.Random) -> None:
        """Build the operator.

        Args:
            scalarization (Scalarization): The map into the positive reals. It must be monotone
                in the same direction as the comparator the search runs with.
            rng (random.Random): The source of randomness.
        """
        self.scalarization = scalarization
        self.rng = rng

    def select_parents(
        self,
        population: Sequence[Tree[T]],
        fitness: Mapping[Tree[T], Fitness],
        comparator: FitnessComparator,
    ) -> tuple[Tree[T], Tree[T]]:
        """Draw two parents proportionally.

        Args:
            population (Sequence[Tree[T]]): The current population.
            fitness (Mapping[Tree[T], Fitness]): The fitness of each member.
            comparator (FitnessComparator): Unused. A proportional draw reads the scalarization,
                not the order. The parameter is part of the contract every parent selection
                shares.

        Returns:
            tuple[Tree[T], Tree[T]]: The two parents.

        Raises:
            ValueError: If the population is empty, or if the scalarization returns a value that is
                zero or below for a member. The second is the case a run meets in practice,
                ``math.exp`` underflowing on a fitness far below zero. The mirror case, ``math.exp``
                overflowing above 710, is reported by
                :class:`~cosy.evolutionary_algorithms.fitness.ExpScalarization` before the value
                arrives here.
        """
        if not population:
            msg = "parent selection needs a non-empty population"
            raise ValueError(msg)
        weights = _proportional_weights([self.scalarization.scalarize(fitness[member]) for member in population])
        first, second = self.rng.choices(list(population), weights=weights, k=2)
        return first, second


class RankBasedSelection(ParentSelection[T]):
    """Draw parents with probability determined by their rank, not by their fitness values.

    The rank comes from the dominance fronts, so the component works under a partial order. The
    members of a front share the average of the ranks they occupy, which keeps the weights summing
    to the population size and decides a tie by the draw rather than by the order the population
    was built in.

    Every member receives a positive share for ``selection_pressure < 2``, so this method is
    generous below that value. At exactly 2 the linear ranking formula gives the last rank weight
    0, so the worst front cannot be drawn at all. That is documented rather than forbidden, because
    the boundary value is the textbook maximum and a caller may want it knowingly.

    Attributes:
        selection_pressure (float): 1.0 is uniform, 2.0 is maximum pressure.
        rng (random.Random): The source of randomness.
    """

    _PRESSURE_LOWER_BOUND = 1.0
    _PRESSURE_UPPER_BOUND = 2.0

    def __init__(self, selection_pressure: float = 1.7, rng: random.Random | None = None) -> None:
        """Build the operator.

        Args:
            selection_pressure (float): Preference for better ranks, in [1.0, 2.0].
                (Default value = 1.7)
            rng (random.Random | None): The source of randomness. (Default value = None)

        Raises:
            ValueError: If the pressure lies outside [1.0, 2.0], or if no generator is given.
        """
        if not self._PRESSURE_LOWER_BOUND <= selection_pressure <= self._PRESSURE_UPPER_BOUND:
            msg = "selection_pressure must be in [1.0, 2.0]"
            raise ValueError(msg)
        if rng is None:
            msg = "rank-based selection draws and needs its own random.Random"
            raise ValueError(msg)
        self.selection_pressure = float(selection_pressure)
        self.rng = rng
        # The ranking of the population this operator was last asked about, and the arguments it
        # was built from. See :meth:`_ranking`.
        self._ranked: list[Tree[T]] = []
        self._weights: list[float] = []
        self._ranked_from: tuple[tuple[Tree[T], ...], Mapping[Tree[T], Fitness], FitnessComparator] | None = None

    def _ranking(
        self,
        population: Sequence[Tree[T]],
        fitness: Mapping[Tree[T], Fitness],
        comparator: FitnessComparator,
    ) -> tuple[list[Tree[T]], list[float]]:
        """Return the population in rank order together with its drawing weights.

        The ranking is a function of the three arguments alone, and a driver asks for one pair per
        variation pass, every pass of a generation over the same population. Building the dominance
        fronts is quadratic in what each front leaves behind, so answering every pass from scratch
        multiplies that cost by the number of passes, which is where a run of a few hundred
        individuals spends nearly all of its time. The last ranking is therefore kept.

        It answers again only if the population holds the same individuals in the same order **and**
        the fitness mapping and the comparator are the very objects of the previous call. Identity
        is the right test for those two: a driver builds a fresh mapping for each generation, and a
        comparator is a component of the run. The population is compared member by member instead,
        so a caller that edits its list in place is ranked again rather than answered from a stale
        ranking.

        Args:
            population (Sequence[Tree[T]]): The current population, of at least two members.
            fitness (Mapping[Tree[T], Fitness]): The fitness of each member.
            comparator (FitnessComparator): The partial order the fronts come from.

        Returns:
            tuple[list[Tree[T]], list[float]]: The members in front order, and the weight of each.
                Both lists belong to this object and are read here, never handed on.
        """
        members = tuple(population)
        if self._ranked_from is not None:
            previous_members, previous_fitness, previous_comparator = self._ranked_from
            if previous_fitness is fitness and previous_comparator is comparator and previous_members == members:
                return self._ranked, self._weights

        count = len(population)
        ranked: list[Tree[T]] = []
        weights: list[float] = []
        pressure = self.selection_pressure
        placed = 0
        for front in dominance_fronts(population, fitness, comparator):
            # The formula is affine in the rank, so sharing the average rank over a front leaves
            # the total weight at the population size.
            shared_rank = (placed + placed + len(front) - 1) / 2
            weight = (2 - pressure) + (2 * (pressure - 1) * (count - shared_rank - 1) / (count - 1))
            ranked.extend(front)
            weights.extend([weight] * len(front))
            placed += len(front)

        self._ranked = ranked
        self._weights = weights
        self._ranked_from = (members, fitness, comparator)
        return ranked, weights

    def select_parents(
        self,
        population: Sequence[Tree[T]],
        fitness: Mapping[Tree[T], Fitness],
        comparator: FitnessComparator,
    ) -> tuple[Tree[T], Tree[T]]:
        """Draw two parents by rank.

        Args:
            population (Sequence[Tree[T]]): The current population.
            fitness (Mapping[Tree[T], Fitness]): The fitness of each member.
            comparator (FitnessComparator): The partial order the fronts come from.

        Returns:
            tuple[Tree[T], Tree[T]]: The two parents.

        Raises:
            ValueError: If the population is empty.
        """
        if not population:
            msg = "parent selection needs a non-empty population"
            raise ValueError(msg)
        if len(population) == 1:
            return population[0], population[0]

        ranked, weights = self._ranking(population, fitness, comparator)
        first, second = self.rng.choices(ranked, weights=weights, k=2)
        return first, second


class LexicaseSelection(ParentSelection[T]):
    """Filter the population case by case in a random order, never aggregating.

    Every other component in this module reads one fitness *value* per individual. This one reads
    the **vector** of per-case outcomes and never sums it: to draw one parent it shuffles the
    cases, then walks them, keeping at each step only the individuals that are elite on the
    current case, until one individual is left or the cases run out.

    **What it is for.** An aggregate hides which cases an individual solves. Where many
    individuals share one aggregate value the order over them is flat and selection has nothing to
    work with, while those same individuals may differ sharply in *which* cases they get right,
    which is the information the sum destroys. Lexicase reads exactly that information, and what
    it selects are **specialists**: individuals excellent on a few cases and unremarkable overall.
    A method reading the aggregate cannot prefer such an individual for what it is good at, the
    aggregate carrying no record of it, so a specialist competes there on a rank that hides its
    specialization and an aggregate-worst specialist is not drawn at all.

    **This method is not generous, and the reason is structural.** Almost sure convergence asks
    that every member of the population be drawn as a parent with positive probability. An
    individual elite on *no* case is filtered out at the first case of every order, so its
    probability is exactly zero, and unlike the tournament there is no parameter to raise that
    changes it. ``uniform_share`` is what changes it: it is the probability that a draw ignores
    the cases and takes a member uniformly, so any positive value restores the condition. The
    default is 0, which is the method as the literature defines it, and a run that wants the
    guarantee sets the share and pays for it in selectivity.

    **Ties and near ties.** Elite means equal to the best value on that case. With ``epsilon`` a
    value within that distance of the best counts as elite too, which is epsilon-lexicase, the
    variant for continuous errors, where exact equality almost never holds and plain lexicase
    collapses to a selection on one case. On integer-valued cases the default 0 is the right one.

    A case on which every individual failed to be measured carries no information about any of
    them, and it is skipped rather than emptying the pool. On a case where only some failed, the
    failures are not elite, which is the answer a failed measurement gets throughout the package.

    Attributes:
        rng (random.Random): The source of randomness, for the case order and for the final tie.
        maximize (bool): Whether a larger per-case value is the better one.
        epsilon (float): How far below the best value on a case still counts as elite.
        uniform_share (float): The probability of drawing uniformly instead, which is what makes
            the method generous.
        case_sample (int | None): Read only this many cases per draw, so fewer cases filter the
            pool and more members reach the final tie. None reads all. Down-sampled lexicase in
            the literature draws its subset *before* evaluating and saves the evaluations of the
            cases it drops. Here the fitness values arrive already computed, so what this
            parameter changes is the filtering and not the cost of it.
    """

    def __init__(
        self,
        rng: random.Random,
        *,
        maximize: bool = False,
        epsilon: float = 0.0,
        uniform_share: float = 0.0,
        case_sample: int | None = None,
    ) -> None:
        """Build the operator.

        Args:
            rng (random.Random): The source of randomness.
            maximize (bool): Whether larger per-case values are better. (Default value = False)
            epsilon (float): The tolerance for counting as elite. (Default value = 0.0)
            uniform_share (float): The probability of a uniform draw. (Default value = 0.0)
            case_sample (int | None): Cases per draw, or None for all. (Default value = None)

        Raises:
            ValueError: If the tolerance is not a finite number that is at least zero, if the
                uniform share is not a probability, or if the case sample is not positive. A
                tolerance wider than the values it compares would empty the pool, which is a
                misconfiguration to report here rather than an index error to raise mid-run.
        """
        if not (epsilon >= 0.0 and math.isfinite(epsilon)):
            msg = f"epsilon is a tolerance and has to be a finite number that is not negative: {epsilon}"
            raise ValueError(msg)
        if not 0.0 <= uniform_share <= 1.0:
            msg = f"uniform_share is a probability: {uniform_share}"
            raise ValueError(msg)
        if case_sample is not None and case_sample < 1:
            msg = f"a draw that reads no case cannot filter anything: {case_sample}"
            raise ValueError(msg)
        self.rng = rng
        self.maximize = maximize
        self.epsilon = float(epsilon)
        self.uniform_share = float(uniform_share)
        self.case_sample = case_sample

    @staticmethod
    def _cases(fitness: Fitness) -> tuple[float, ...]:
        """Read a fitness value as a vector of per-case outcomes.

        A value is placed by whether it *has a length* rather than by its type, for the reason
        :func:`~cosy.evolutionary_algorithms.fitness._single_objective` gives: ``numpy.ndarray``
        carries a length without being a ``Sequence``, and a float-like scalar such as
        ``numpy.float32`` carries none.

        Args:
            fitness (Fitness): The value to read.

        Returns:
            tuple[float, ...]: Its cases. A scalar is a single case.
        """
        try:
            return tuple(float(case) for case in fitness)  # type: ignore[union-attr]
        except TypeError:
            return (float(fitness),)  # type: ignore[arg-type]

    def _elite(self, pool: list[Tree[T]], values: Mapping[Tree[T], float]) -> list[Tree[T]]:
        """Keep the members of the pool that are elite on one case.

        Args:
            pool (list[Tree[T]]): The members still in contention.
            values (Mapping[Tree[T], float]): Their value on the case being read.

        Returns:
            list[Tree[T]]: Those within ``epsilon`` of the best value, in the order they came in.
                A case on which every member failed to be measured leaves the pool untouched.
        """
        measured = [value for value in (values[member] for member in pool) if not math.isnan(value)]
        if not measured:
            return pool
        # A failed measurement compares false in either direction, so it falls out of the bound
        # below on its own and needs no test of its own to keep it from counting as elite.
        if self.maximize:
            bound = max(measured) - self.epsilon
            return [member for member in pool if values[member] >= bound]
        bound = min(measured) + self.epsilon
        return [member for member in pool if values[member] <= bound]

    def _select_one(
        self,
        population: Sequence[Tree[T]],
        fitness: Mapping[Tree[T], Fitness],
    ) -> Tree[T]:
        """Draw one parent.

        Args:
            population (Sequence[Tree[T]]): The population to draw from, of at least one member.
            fitness (Mapping[Tree[T], Fitness]): The fitness of each member.

        Returns:
            Tree[T]: The individual drawn.

        Raises:
            ValueError: If the individuals carry different numbers of cases, which would leave
                "the same case" meaning different things for different individuals.
        """
        if self.uniform_share > 0 and self.rng.random() < self.uniform_share:
            return self.rng.choice(list(population))

        vectors = {member: self._cases(fitness[member]) for member in population}
        widths = {len(vector) for vector in vectors.values()}
        if len(widths) > 1:
            msg = f"lexicase needs one case count for the whole population, got {sorted(widths)}"
            raise ValueError(msg)

        order = list(range(widths.pop()))
        self.rng.shuffle(order)
        if self.case_sample is not None:
            order = order[: self.case_sample]

        pool = list(population)
        for case in order:
            if len(pool) <= 1:
                break
            pool = self._elite(pool, {member: vectors[member][case] for member in pool})
        return self.rng.choice(pool)

    def select_parents(
        self,
        population: Sequence[Tree[T]],
        fitness: Mapping[Tree[T], Fitness],
        comparator: FitnessComparator,
    ) -> tuple[Tree[T], Tree[T]]:
        """Draw two parents, each under a case order of its own.

        The comparator is not consulted. Lexicase orders by cases, and an order over aggregates is
        the thing it exists to avoid. It stays in the signature because the protocol has it.

        Args:
            population (Sequence[Tree[T]]): The current population.
            fitness (Mapping[Tree[T], Fitness]): The fitness of each member.
            comparator (FitnessComparator): Unused, for the reason above.

        Returns:
            tuple[Tree[T], Tree[T]]: The two parents.

        Raises:
            ValueError: If the population is empty.
        """
        del comparator
        if not population:
            msg = "parent selection needs a non-empty population"
            raise ValueError(msg)
        return self._select_one(population, fitness), self._select_one(population, fitness)


class FitnessBasedReplacement(SurvivorSelection[T]):
    """(mu + lambda) truncation: keep the fittest ``size`` of parents and offspring together.

    **Neither generous nor conservative in the sense the guarantee asks.** An individual outside
    the fittest ``size`` survives with probability zero, so it is not generous. It always keeps a
    maximal element, since the first front is never empty, but the convergence condition asks for a
    member of greatest *scalarized* fitness, and under a partial order the two differ: a member of
    the first front can be cut while another of the same front carries the greater scalarized
    value. Under a total order they coincide. It is in the inventory because truncation is the
    standard elitist replacement and converges fast in practice.
    :class:`GenerousConservativeReplacement` is the one to reach for when the guarantee is wanted.

    Under a partial order the fronts are taken in order. A front that does not fit entirely is cut
    in the order the individuals were passed in, which makes the cut deterministic.
    """

    def select_survivors(
        self,
        parents: Sequence[Tree[T]],
        offspring: Sequence[Tree[T]],
        fitness: Mapping[Tree[T], Fitness],
        comparator: FitnessComparator,
        size: int,
    ) -> list[Tree[T]]:
        """Keep the fittest individuals of both populations together.

        Args:
            parents (Sequence[Tree[T]]): The finished generation.
            offspring (Sequence[Tree[T]]): Its offspring.
            fitness (Mapping[Tree[T], Fitness]): The fitness of every individual.
            comparator (FitnessComparator): The partial order.
            size (int): The population size mu.

        Returns:
            list[Tree[T]]: The fittest ``size`` individuals.

        Raises:
            ValueError: If fewer than ``size`` individuals were offered.
        """
        pool = [*parents, *offspring]
        if len(pool) < size:
            msg = f"survivor selection was offered {len(pool)} individuals for {size} places"
            raise ValueError(msg)
        survivors: list[Tree[T]] = []
        for front in dominance_fronts(pool, fitness, comparator):
            survivors.extend(front[: size - len(survivors)])
            if len(survivors) == size:
                break
        return survivors


class GenerousConservativeReplacement(SurvivorSelection[T]):
    """Keep one fittest individual for certain, and fill the rest so nobody is excluded.

    The component that carries the survivor-selection half of almost sure convergence outright.
    Both words are Eiben, Aarts and Van Hee's: a selection function is *conservative* if it always
    keeps one of the strongest individuals of any population, and *generous* if it gives every
    individual a positive chance to survive. Their convergence theorem needs both, and Rudolph's
    results show what the conservative half buys. The canonical genetic algorithm without it does
    not converge, since the probability of sitting in a population without an optimum stays
    positive forever, while the variant maintaining the best solution found does converge.

    The construction: one individual of greatest scalarized fitness takes a place outright, and the
    remaining ``size - 1`` places are drawn proportionally to the scalarization from parents and
    offspring together, with replacement. Positivity of the scalarization makes every individual's
    chance positive, and the reserved place makes the choice conservative. Drawing with replacement
    is what keeps the two properties independent of the population's composition. A population is a
    multiset, so a repeated draw is a legitimate outcome rather than a defect.

    Attributes:
        scalarization (Scalarization): The map into the positive reals that both halves are read
            through.
        rng (random.Random): The source of randomness.
    """

    def __init__(self, scalarization: Scalarization, rng: random.Random) -> None:
        """Build the operator.

        Args:
            scalarization (Scalarization): The map into the positive reals. "Greatest scalarized
                fitness" is read through it, so it must be monotone in the comparator's direction.
            rng (random.Random): The source of randomness.
        """
        self.scalarization = scalarization
        self.rng = rng

    def select_survivors(
        self,
        parents: Sequence[Tree[T]],
        offspring: Sequence[Tree[T]],
        fitness: Mapping[Tree[T], Fitness],
        comparator: FitnessComparator,
        size: int,
    ) -> list[Tree[T]]:
        """Reserve a place for a fittest individual and draw the rest proportionally.

        Args:
            parents (Sequence[Tree[T]]): The finished generation.
            offspring (Sequence[Tree[T]]): Its offspring.
            fitness (Mapping[Tree[T], Fitness]): The fitness of every individual.
            comparator (FitnessComparator): Unused. The reserved place goes to an individual of
                greatest *scalarized* fitness, which the scalarization decides on its own and
                totally. The parameter is part of the contract every survivor selection shares.
            size (int): The population size mu.

        Returns:
            list[Tree[T]]: The survivors, the reserved one first.

        Raises:
            ValueError: If ``size`` is negative, if a positive ``size`` was asked for without a
                single individual being offered, or if the scalarization returns a value that is
                zero or below for a member of the pool. The last is the case a run meets in
                practice, ``math.exp`` underflowing on a fitness far below zero. The mirror case,
                ``math.exp`` overflowing above 710, is reported by
                :class:`~cosy.evolutionary_algorithms.fitness.ExpScalarization` before the value
                arrives here.
        """
        if size < 0:
            msg = f"a population size cannot be negative: {size}"
            raise ValueError(msg)
        if size == 0:
            return []
        pool = [*parents, *offspring]
        if not pool:
            msg = "survivor selection was offered no individuals"
            raise ValueError(msg)

        values = [self.scalarization.scalarize(fitness[member]) for member in pool]
        weights = _proportional_weights(values)
        # The champion is read off the *weights*, not off the raw values. The weights are where
        # the policy for values outside the codomain already lives, a failed measurement weighing
        # nothing and an infinite one taking the mass. Reading ``max`` off the raw values instead
        # makes the choice depend on the order the pool was assembled in: ``max`` returns ``nan``
        # when it comes first and a number when it does not, and in the first case no member equals
        # it, so the draw would be made from an empty sequence.
        best = max(weights)
        # Several individuals may share the greatest weight. Taking one of them uniformly keeps
        # the choice from depending on the pool's order.
        champions = [member for member, weight in zip(pool, weights, strict=True) if weight == best]
        survivors = [self.rng.choice(champions)]
        if size > 1:
            survivors.extend(self.rng.choices(pool, weights=weights, k=size - 1))
        return survivors
