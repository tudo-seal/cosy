"""The selection inventory, measured against the conditions for almost sure convergence.

Every component here is a standard method, and what this file pins is which of the conditions each
one discharges. The findings are not uniform, and two of them are the opposite of what the names
suggest:

* **Tournament selection is not generous for ``tournament_size >= 2``.** A tournament is drawn
  without replacement, so the least fit member of a population with distinct fitness values meets a
  fitter competitor in every tournament it enters and wins none. Convergence asks for positive
  probability for *every* member, and this member gets zero.
* **Truncation is conservative but not generous.** It always keeps a fittest individual, and it
  never keeps anything outside the top mu. A survivor selection needs both halves, so
  ``FitnessBasedReplacement`` does not carry the guarantee either.

:class:`GenerousConservativeReplacement` is the component built to have both halves, and both of
them are measured below. The two contracts, population to a pair versus parents and offspring to a
population, are separate protocols, and that separation is pinned too.
"""

import math
import random
from collections import Counter

import pytest

from cosy.core.tree import Tree
from cosy.evolutionary_algorithms import (
    AggregatedFitnessComparator,
    Comparison,
    ExpScalarization,
    Fitness,
    FitnessBasedReplacement,
    FitnessProportionalSelection,
    GenerousConservativeReplacement,
    LexicaseSelection,
    ParentSelection,
    ParetoFitnessComparator,
    RankBasedSelection,
    ScalarFitnessComparator,
    SurvivorSelection,
    TournamentSelection,
    dominance_fronts,
)
from tests.ea_fixtures import a2, b2, chain, h1

MAXIMIZING = ScalarFitnessComparator(greater_is_better=True)


def individuals(count: int) -> list[Tree]:
    """Build ``count`` distinct individuals.

    Args:
        count (int): How many.

    Returns:
        list[Tree]: Chains of increasing length, so they compare unequal.
    """
    return [chain(index) for index in range(count)]


def scored(values: list[float]) -> tuple[list[Tree], dict[Tree, float]]:
    """Build individuals carrying the given fitness values.

    Args:
        values (list[float]): One fitness per individual.

    Returns:
        tuple[list[Tree], dict[Tree, float]]: The population and its fitness mapping.
    """
    population = individuals(len(values))
    return population, dict(zip(population, values, strict=True))


# ---------------------------------------------------------------------------
# The ranking construction
# ---------------------------------------------------------------------------


def test_dominance_fronts_degenerate_to_the_ordinary_ranking():
    """Under a total order a front is a class of equally fit individuals."""
    population, fitness = scored([3.0, 1.0, 3.0, 2.0])
    fronts = dominance_fronts(population, fitness, MAXIMIZING)
    assert [[fitness[member] for member in front] for front in fronts] == [
        [3.0, 3.0],
        [2.0],
        [1.0],
    ]


def test_dominance_fronts_group_incomparable_individuals():
    """Under a partial order a front holds what nothing else dominates."""
    population = individuals(3)
    fitness = {
        population[0]: [2.0, 1.0],
        population[1]: [1.0, 2.0],
        population[2]: [1.0, 1.0],
    }
    fronts = dominance_fronts(population, fitness, ParetoFitnessComparator())
    assert len(fronts) == 2
    assert set(fronts[0]) == {population[0], population[1]}
    assert fronts[1] == [population[2]]


def test_dominance_fronts_keep_repeated_individuals():
    """A population is a multiset, and a repeated member occupies a place of its own."""
    individual = chain(1)
    fitness = {individual: 1.0}
    fronts = dominance_fronts([individual, individual], fitness, MAXIMIZING)
    assert fronts == [[individual, individual]]


def test_dominance_fronts_keep_the_input_order():
    """Within a front the individuals stay in the order they were passed in."""
    population, fitness = scored([1.0, 1.0, 1.0, 1.0])
    assert dominance_fronts(population, fitness, MAXIMIZING) == [population]


# ---------------------------------------------------------------------------
# Parent selection: the contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "operator",
    [
        TournamentSelection(2, random.Random(0)),
        FitnessProportionalSelection(ExpScalarization(), random.Random(0)),
        RankBasedSelection(1.7, random.Random(0)),
        LexicaseSelection(random.Random(0)),
    ],
)
def test_parent_selection_returns_a_pair_of_members(operator):
    """A parent selection method maps a population to a **pair** of its members.

    Args:
        operator: The parent selection under test.
    """
    population, fitness = scored([1.0, 2.0, 3.0, 4.0])
    parents = operator.select_parents(population, fitness, MAXIMIZING)
    assert len(parents) == 2
    assert all(parent in population for parent in parents)
    assert isinstance(operator, ParentSelection)
    assert not isinstance(operator, SurvivorSelection)


@pytest.mark.parametrize(
    "operator",
    [
        TournamentSelection(2, random.Random(0)),
        FitnessProportionalSelection(ExpScalarization(), random.Random(0)),
        RankBasedSelection(1.7, random.Random(0)),
        LexicaseSelection(random.Random(0)),
    ],
)
def test_parent_selection_refuses_an_empty_population(operator):
    """There is no pair to draw from nothing, and a substitute would be invented.

    Args:
        operator: The parent selection under test.
    """
    with pytest.raises(ValueError, match="non-empty"):
        operator.select_parents([], {}, MAXIMIZING)


@pytest.mark.parametrize(
    "operator",
    [
        TournamentSelection(2, random.Random(0)),
        FitnessProportionalSelection(ExpScalarization(), random.Random(0)),
        RankBasedSelection(1.7, random.Random(0)),
    ],
)
def test_the_two_parents_may_coincide_and_may_differ(operator):
    """In a population of two or more, the two parents are drawn independently and with replacement.

    Args:
        operator: The parent selection under test.
    """
    population, fitness = scored([1.0, 2.0, 3.0, 4.0])
    pairs = [operator.select_parents(population, fitness, MAXIMIZING) for _ in range(50)]
    assert any(len(set(pair)) == 1 for pair in pairs)
    assert any(len(set(pair)) == 2 for pair in pairs)


# ---------------------------------------------------------------------------
# Tournament selection
# ---------------------------------------------------------------------------


def test_a_tournament_covering_the_population_always_returns_the_best():
    """With the whole population in the tournament the winner is determined."""
    population, fitness = scored([1.0, 5.0, 3.0])
    operator = TournamentSelection(3, random.Random(0))
    for _ in range(20):
        assert set(operator.select_parents(population, fitness, MAXIMIZING)) == {population[1]}


def test_tied_individuals_win_equally_often():
    """A tie is decided by a uniform draw among the undominated, not by a chain of coin flips."""
    population, fitness = scored([1.0, 1.0, 1.0])
    operator = TournamentSelection(3, random.Random(1))
    counts = Counter(parent for _ in range(3000) for parent in operator.select_parents(population, fitness, MAXIMIZING))
    assert set(counts) == set(population)
    assert max(counts.values()) - min(counts.values()) < 0.1 * 6000 / 3


def test_the_least_fit_member_never_becomes_a_parent():
    """A tournament of size 2 or more is not generous: the measurement behind the docstring.

    The worst individual meets a fitter competitor in every tournament it is drawn into, because
    the tournament is a sample without replacement.
    """
    population, fitness = scored([1.0, 2.0, 3.0, 4.0])
    operator = TournamentSelection(2, random.Random(2))
    drawn = {parent for _ in range(4000) for parent in operator.select_parents(population, fitness, MAXIMIZING)}
    assert population[0] not in drawn
    assert len(drawn) == 3


def test_a_tournament_of_one_is_uniform_selection_and_is_generous():
    """The one setting under which tournament selection is generous."""
    population, fitness = scored([1.0, 2.0, 3.0, 4.0])
    operator = TournamentSelection(1, random.Random(3))
    counts = Counter(parent for _ in range(4000) for parent in operator.select_parents(population, fitness, MAXIMIZING))
    assert set(counts) == set(population)
    assert max(counts.values()) - min(counts.values()) < 0.1 * 8000 / 4


def test_a_tournament_needs_a_participant():
    """A tournament size is a count of at least one."""
    with pytest.raises(ValueError, match="at least one"):
        TournamentSelection(0, random.Random(0))


# ---------------------------------------------------------------------------
# Fitness-proportional selection
# ---------------------------------------------------------------------------


def test_proportional_selection_gives_every_member_a_positive_share():
    """Every member gets a positive share *because* a scalarization is strictly positive."""
    population, fitness = scored([-3.0, 0.0, 4.0])
    operator = FitnessProportionalSelection(ExpScalarization(), random.Random(4))
    counts = Counter(parent for _ in range(5000) for parent in operator.select_parents(population, fitness, MAXIMIZING))
    assert set(counts) == set(population)


def test_proportional_selection_follows_the_scalarized_ratios():
    """The weights are the scalarization's values, with no lift or shift in between.

    An operator that subtracted the smallest value whenever it was negative would move the worst
    individual to weight zero. A positive weight is what the definition of a scalarization already
    guarantees, so nothing has to be lifted.
    """
    population, fitness = scored([0.0, math.log(3.0)])
    operator = FitnessProportionalSelection(ExpScalarization(), random.Random(5))
    counts = Counter(
        parent for _ in range(20000) for parent in operator.select_parents(population, fitness, MAXIMIZING)
    )
    assert abs(counts[population[1]] / counts[population[0]] - 3.0) < 0.15


def test_an_infinitely_good_individual_takes_the_whole_mass():
    """The documented policy for a non-finite measurement: it takes the whole mass."""
    population, fitness = scored([1.0, math.inf])
    operator = FitnessProportionalSelection(ExpScalarization(), random.Random(6))
    drawn = {parent for _ in range(200) for parent in operator.select_parents(population, fitness, MAXIMIZING)}
    assert drawn == {population[1]}


def test_a_failed_measurement_is_never_drawn():
    """``nan`` takes no share, and is not replaced by a substitute value."""
    population, fitness = scored([1.0, math.nan])
    operator = FitnessProportionalSelection(ExpScalarization(), random.Random(7))
    drawn = {parent for _ in range(200) for parent in operator.select_parents(population, fitness, MAXIMIZING)}
    assert drawn == {population[0]}


def test_a_population_of_nothing_but_failures_is_drawn_uniformly():
    """With nothing left to prefer the draw is uniform rather than undefined."""
    population, fitness = scored([math.nan, math.nan])
    operator = FitnessProportionalSelection(ExpScalarization(), random.Random(8))
    counts = Counter(parent for _ in range(2000) for parent in operator.select_parents(population, fitness, MAXIMIZING))
    assert set(counts) == set(population)


def test_a_scalarization_that_underflows_is_reported_rather_than_floored():
    """A weight of zero costs a member its share, so it is refused where it would take effect.

    ``math.exp`` underflows below about -745. Flooring the weight would hide the loss of
    positivity, and the message says what to do instead.
    """
    population, fitness = scored([-800.0, 1.0])
    operator = FitnessProportionalSelection(ExpScalarization(), random.Random(9))
    with pytest.raises(ValueError, match="strictly positive"):
        operator.select_parents(population, fitness, MAXIMIZING)


def test_a_scalarization_that_overflows_is_reported_rather_than_read_as_infinity():
    """A finite fitness above the exponential's range must not read as infinitely good.

    ``math.exp`` overflows above about 710, and infinity takes the whole mass. The two overflowing
    members would leave 700.0 with a weight of exactly zero, and this component is in the inventory
    for giving every member a positive share.
    """
    population, fitness = scored([900.0, 800.0, 700.0])
    operator = FitnessProportionalSelection(ExpScalarization(), random.Random(22))
    with pytest.raises(ValueError, match="overflows"):
        operator.select_parents(population, fitness, MAXIMIZING)


# ---------------------------------------------------------------------------
# Rank-based selection
# ---------------------------------------------------------------------------


def test_rank_weights_follow_the_linear_ranking_formula():
    """The textbook weights, over the dominance fronts."""
    population, fitness = scored([4.0, 3.0, 2.0, 1.0])
    operator = RankBasedSelection(2.0, random.Random(10))
    counts = Counter(
        parent for _ in range(20000) for parent in operator.select_parents(population, fitness, MAXIMIZING)
    )
    # At pressure 2 the weights are 2, 4/3, 2/3, 0 for the four ranks.
    assert counts[population[3]] == 0
    assert abs(counts[population[0]] / counts[population[1]] - 1.5) < 0.1


def test_rank_selection_is_invariant_to_the_fitness_scale():
    """Ranks, not values: multiplying every fitness changes nothing."""
    population, fitness = scored([1.0, 2.0, 3.0])
    scaled = {member: 1000.0 * value for member, value in fitness.items()}
    plain = RankBasedSelection(1.7, random.Random(11))
    lifted = RankBasedSelection(1.7, random.Random(11))
    first = Counter(parent for _ in range(3000) for parent in plain.select_parents(population, fitness, MAXIMIZING))
    second = Counter(parent for _ in range(3000) for parent in lifted.select_parents(population, scaled, MAXIMIZING))
    assert first == second


def test_rank_selection_decides_a_tie_by_the_draw():
    """A tie is decided by the draw, not by the order the population was built in."""
    population, fitness = scored([2.0, 2.0, 1.0])
    operator = RankBasedSelection(1.7, random.Random(12))
    counts = Counter(parent for _ in range(6000) for parent in operator.select_parents(population, fitness, MAXIMIZING))
    assert abs(counts[population[0]] - counts[population[1]]) < 0.1 * 12000 / 3


def test_rank_selection_is_generous_below_pressure_two():
    """Generous for ``selection_pressure < 2``. At exactly 2 the last rank gets weight zero."""
    population, fitness = scored([1.0, 2.0, 3.0])
    operator = RankBasedSelection(1.9, random.Random(13))
    generous = Counter(
        parent for _ in range(3000) for parent in operator.select_parents(population, fitness, MAXIMIZING)
    )
    assert set(generous) == set(population)


def test_rank_selection_rejects_a_pressure_outside_its_range():
    """The parameter range of the method."""
    with pytest.raises(ValueError, match=r"\[1.0, 2.0\]"):
        RankBasedSelection(2.5, random.Random(0))


def test_rank_selection_needs_a_generator():
    """The default of ``rng`` is ``None``, and the constructor rejects it rather than storing it.

    Without the check the operator would store ``None`` and raise ``AttributeError`` at its first
    draw.
    """
    with pytest.raises(ValueError, match="needs its own"):
        RankBasedSelection(1.7)


def test_rank_selection_of_a_single_individual_repeats_it():
    """A population of one has one pair to offer."""
    population, fitness = scored([1.0])
    parents = RankBasedSelection(1.7, random.Random(14)).select_parents(population, fitness, MAXIMIZING)
    assert parents == (population[0], population[0])


class _RecordingRandom(random.Random):
    """A generator that remembers the weights it was asked to draw with.

    Attributes:
        weights (list[list[float]]): One entry per ``choices`` call, in the order the calls
            happened.
    """

    def __init__(self, seed: int) -> None:
        """Seed the generator.

        Args:
            seed (int): The seed.
        """
        super().__init__(seed)
        self.weights: list[list[float]] = []

    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        """Record the weights and draw with them.

        Args:
            population: The sequence to draw from.
            weights: The weight of each element, recorded before the draw.
            cum_weights: Never used. Kept so the signature stays the one
                ``random.Random.choices`` declares.
            k: How many elements to draw.

        Returns:
            The drawn elements.
        """
        self.weights.append(list(weights))
        return super().choices(population, weights, k=k)


def test_a_front_shares_the_average_of_the_ranks_it_occupies():
    """The shared rank of a front is the average of the ranks it occupies, not either end of them.

    Two of the three members here are equally fit, so they form the first front, occupy ranks 0
    and 1, and share rank 0.5. At pressure 1.7 that gives the weights 1.35, 1.35, and 0.3, which
    sum to the population size. The front's first rank would give the pair 1.7 and its last rank
    1.0, against the same 0.3 either way, and neither of those sums to the population size.
    ``random.Random.choices`` normalizes over the total weight, so only the ratio reaches the draw
    and the weights are read off the generator instead.
    """
    population, fitness = scored([2.0, 2.0, 1.0])
    rng = _RecordingRandom(20)
    RankBasedSelection(1.7, rng).select_parents(population, fitness, MAXIMIZING)
    weights = rng.weights[0]
    assert weights == pytest.approx([1.35, 1.35, 0.3])
    assert sum(weights) == pytest.approx(len(population))


class CountingComparator:
    """A comparator that records how often it was asked, so a rebuild becomes visible."""

    def __init__(self, inner):
        """Wrap a comparator.

        Args:
            inner: The comparator that decides.
        """
        self.inner = inner
        self.calls = 0

    def compare(self, first, second):
        """Delegate and count.

        Args:
            first: The first value.
            second: The second value.

        Returns:
            Comparison: What the wrapped comparator answers.
        """
        self.calls += 1
        return self.inner.compare(first, second)


def test_the_ranking_is_built_once_for_a_population():
    """A driver asks for one pair per offspring, and every pass ranks the same population."""
    population, fitness = scored([3.0, 2.0, 1.0])
    comparator = CountingComparator(MAXIMIZING)
    operator = RankBasedSelection(1.7, random.Random(15))

    operator.select_parents(population, fitness, comparator)
    after_the_first_pair = comparator.calls
    assert after_the_first_pair > 0

    for _ in range(50):
        operator.select_parents(population, fitness, comparator)
    assert comparator.calls == after_the_first_pair


def test_a_population_edited_in_place_is_ranked_again():
    """The ranking is keyed on the members, so an edited list is not answered from it.

    At pressure 2 the last rank carries weight zero, which makes the ranking readable off the draw.
    """
    members, fitness = scored([3.0, 2.0, 1.0, 0.0])
    population = members[:3]
    operator = RankBasedSelection(2.0, random.Random(16))

    before = Counter(parent for _ in range(2000) for parent in operator.select_parents(population, fitness, MAXIMIZING))
    assert before[members[2]] == 0
    assert before[members[0]] > 0

    population[0] = members[3]

    after = Counter(parent for _ in range(2000) for parent in operator.select_parents(population, fitness, MAXIMIZING))
    assert members[0] not in after
    assert after[members[3]] == 0
    assert after[members[2]] > 0


def test_a_new_fitness_mapping_is_ranked_again():
    """The same individuals under a different measurement are a different ranking."""
    population, fitness = scored([3.0, 2.0, 1.0])
    reversed_fitness = dict(zip(population, [1.0, 2.0, 3.0], strict=True))
    operator = RankBasedSelection(2.0, random.Random(17))

    first = Counter(parent for _ in range(2000) for parent in operator.select_parents(population, fitness, MAXIMIZING))
    second = Counter(
        parent for _ in range(2000) for parent in operator.select_parents(population, reversed_fitness, MAXIMIZING)
    )

    assert first[population[2]] == 0
    assert second[population[0]] == 0


def test_a_new_comparator_is_ranked_again():
    """The same individuals under a different order are a different ranking.

    A ranking is built from the fronts, and the fronts come from the comparator, so a caller that
    reverses the order gets the reversed ranking and not the stored one.
    """
    population, fitness = scored([3.0, 2.0, 1.0])
    minimizing = ScalarFitnessComparator(greater_is_better=False)
    operator = RankBasedSelection(2.0, random.Random(19))

    greater = Counter(
        parent for _ in range(2000) for parent in operator.select_parents(population, fitness, MAXIMIZING)
    )
    lesser = Counter(parent for _ in range(2000) for parent in operator.select_parents(population, fitness, minimizing))

    assert greater[population[2]] == 0
    assert lesser[population[0]] == 0


def test_the_stored_ranking_does_not_change_the_draw():
    """A seeded operator draws the same pairs whether the ranking is reused or rebuilt."""
    population, fitness = scored([4.0, 3.0, 2.0, 1.0])
    reusing = RankBasedSelection(1.7, random.Random(18))
    rebuilding = RankBasedSelection(1.7, random.Random(18))

    reused = [reusing.select_parents(population, fitness, MAXIMIZING) for _ in range(200)]
    # A fresh mapping on every call, so the stored ranking never answers.
    rebuilt = [rebuilding.select_parents(population, dict(fitness), MAXIMIZING) for _ in range(200)]

    assert reused == rebuilt


# ---------------------------------------------------------------------------
# Lexicase selection
# ---------------------------------------------------------------------------


def by_case(vectors: list[tuple[float, ...]]) -> tuple[list[Tree], dict[Tree, Fitness]]:
    """Build individuals carrying the given vectors of per-case outcomes.

    Args:
        vectors (list[tuple[float, ...]]): One vector of cases per individual.

    Returns:
        tuple[list[Tree], dict[Tree, Fitness]]: The population and its fitness mapping.
    """
    population = individuals(len(vectors))
    return population, dict(zip(population, vectors, strict=True))


class _RefusingComparator:
    """A comparator that fails the test if anything asks it to compare."""

    def compare(self, first: Fitness, second: Fitness) -> Comparison:
        """Refuse to compare.

        Args:
            first (Fitness): Unused.
            second (Fitness): Unused.

        Returns:
            Comparison: Never returns.

        Raises:
            AssertionError: Always.
        """
        del first, second
        msg = "lexicase orders by cases and must not consult a comparator"
        raise AssertionError(msg)


def test_lexicase_selects_the_specialists_an_aggregate_hides():
    """Three individuals of one aggregate, two of which the cases separate.

    Every vector here totals 4, so an order over aggregates ranks the three equal and has nothing
    to select on. The cases tell them apart: two are best on one case each, and the third, second
    best on both and best on neither, is filtered out at the first case of every order. That is
    what lexicase is for, and it is at the same time the reason the method is not generous. Its
    probability is not small, it is zero, and no parameter but ``uniform_share`` reaches it.
    """
    population, fitness = by_case([(2.0, 2.0), (0.0, 4.0), (4.0, 0.0)])
    aggregate = AggregatedFitnessComparator(greater_is_better=False)
    assert aggregate.compare(fitness[population[0]], fitness[population[1]]) is Comparison.EQUAL
    selection = LexicaseSelection(random.Random(0))
    drawn = {parent for _ in range(200) for parent in selection.select_parents(population, fitness, MAXIMIZING)}
    assert drawn == {population[1], population[2]}


def test_a_positive_uniform_share_makes_every_member_reachable():
    """The parameter that restores the condition the method otherwise fails.

    On the share of draws that ignores the cases every member is equally likely, so the individual
    of the test above gets a positive probability rather than none.
    """
    population, fitness = by_case([(2.0, 2.0), (0.0, 4.0), (4.0, 0.0)])
    selection = LexicaseSelection(random.Random(2), uniform_share=0.5)
    drawn = {parent for _ in range(200) for parent in selection.select_parents(population, fitness, MAXIMIZING)}
    assert drawn == set(population)


def test_the_uniform_draw_is_uniform():
    """At a share of one the cases are never read, and every member is equally likely.

    Reachability alone does not pin this. The never-elite member sits first in the population
    above, so a uniform branch always returning the first member would satisfy that test, the
    case-wise branch supplying the other two.
    """
    population, fitness = by_case([(2.0, 2.0), (0.0, 4.0), (4.0, 0.0)])
    selection = LexicaseSelection(random.Random(3), uniform_share=1.0)
    counts = Counter(parent for _ in range(300) for parent in selection.select_parents(population, fitness, MAXIMIZING))
    assert set(counts) == set(population)
    assert min(counts.values()) > 100


def test_the_direction_decides_which_value_is_elite():
    """Elite is the smallest value on a case under minimization and the largest under maximization."""
    population, fitness = by_case([(0.0, 0.0), (2.0, 2.0), (4.0, 4.0)])
    minimizing = LexicaseSelection(random.Random(0))
    maximizing = LexicaseSelection(random.Random(0), maximize=True)
    assert {p for _ in range(50) for p in minimizing.select_parents(population, fitness, MAXIMIZING)} == {population[0]}
    assert {p for _ in range(50) for p in maximizing.select_parents(population, fitness, MAXIMIZING)} == {population[2]}


def test_a_scalar_fitness_is_a_single_case():
    """A value that was never decomposed is one case, and the method reduces to taking the best."""
    population, fitness = scored([3.0, 1.0, 2.0])
    selection = LexicaseSelection(random.Random(0))
    drawn = {parent for _ in range(50) for parent in selection.select_parents(population, fitness, MAXIMIZING)}
    assert drawn == {population[1]}


def test_epsilon_keeps_the_near_elite():
    """Epsilon-lexicase: within the tolerance of the best on a case is elite too.

    Exact equality almost never holds on continuous errors, and plain lexicase there decides every
    draw on the first case it reads.
    """
    population, fitness = by_case([(1.0,), (1.2,), (1.6,)])
    strict = LexicaseSelection(random.Random(4))
    assert {p for _ in range(50) for p in strict.select_parents(population, fitness, MAXIMIZING)} == {population[0]}
    tolerant = LexicaseSelection(random.Random(4), epsilon=0.5)
    assert {p for _ in range(50) for p in tolerant.select_parents(population, fitness, MAXIMIZING)} == {
        population[0],
        population[1],
    }


def test_epsilon_keeps_the_near_elite_under_maximization_too():
    """The tolerance widens the bound downwards where a larger value is the better one.

    The minimizing case above cannot see the sign of the term on this branch, and neither can the
    maximizing test above it, which runs at the default tolerance of zero.
    """
    population, fitness = by_case([(1.6,), (1.4,), (1.0,)])
    strict = LexicaseSelection(random.Random(4), maximize=True)
    assert {p for _ in range(50) for p in strict.select_parents(population, fitness, MAXIMIZING)} == {population[0]}
    tolerant = LexicaseSelection(random.Random(4), maximize=True, epsilon=0.5)
    assert {p for _ in range(50) for p in tolerant.select_parents(population, fitness, MAXIMIZING)} == {
        population[0],
        population[1],
    }


def test_down_sampling_reads_fewer_cases_and_filters_less():
    """``case_sample`` is the whole difference between a decided draw and an undecided one.

    Case 0 separates nothing and case 1 separates the two. Reading both cases therefore always
    ends at the first individual, in either order. Reading one case ends at both whenever the one
    read is case 0, so the second individual becomes reachable.
    """
    population, fitness = by_case([(0.0, 1.0), (0.0, 2.0)])
    every_case = LexicaseSelection(random.Random(5))
    assert {p for _ in range(50) for p in every_case.select_parents(population, fitness, MAXIMIZING)} == {population[0]}
    sampled = LexicaseSelection(random.Random(5), case_sample=1)
    assert {p for _ in range(50) for p in sampled.select_parents(population, fitness, MAXIMIZING)} == set(population)


def test_a_case_no_one_could_be_measured_on_is_skipped():
    """A case on which every individual failed says nothing about any of them.

    Filtering on it would compare against a value that is not one, and every individual would fail
    the comparison, leaving no one to draw from.
    """
    population, fitness = by_case([(math.nan, 1.0), (math.nan, 0.0)])
    selection = LexicaseSelection(random.Random(6))
    drawn = {parent for _ in range(50) for parent in selection.select_parents(population, fitness, MAXIMIZING)}
    assert drawn == {population[1]}


@pytest.mark.parametrize("maximize", [False, True])
def test_a_failed_measurement_is_not_elite_on_its_case(maximize):
    """Where some individuals were measured, the ones that were not are not the best of them.

    The failed measurement comes first, which is what makes the test see the filtering. Taking the
    extremum over the pool as it stands returns ``nan`` when a ``nan`` is at the front and returns
    the measured value when it is at the back, so a version skipping the filter passes with the
    failure in second place and fails here.

    Args:
        maximize (bool): The direction under test. The bound is read on both branches.
    """
    population, fitness = by_case([(math.nan, 0.0), (1.0, 0.0)])
    selection = LexicaseSelection(random.Random(7), maximize=maximize)
    drawn = {parent for _ in range(50) for parent in selection.select_parents(population, fitness, MAXIMIZING)}
    assert drawn == {population[1]}


def test_the_two_parents_are_drawn_under_case_orders_of_their_own():
    """Each parent gets its own shuffle, so a population with two winners yields both.

    The parametrized contract above leaves lexicase out of the pair that may differ: on a single
    case with one best value the method is deterministic and the two parents always coincide,
    which is a property of the method rather than a breach of the contract.
    """
    population, fitness = by_case([(0.0, 4.0), (4.0, 0.0)])
    selection = LexicaseSelection(random.Random(8))
    pairs = [selection.select_parents(population, fitness, MAXIMIZING) for _ in range(50)]
    assert any(len(set(pair)) == 1 for pair in pairs)
    assert any(len(set(pair)) == 2 for pair in pairs)


def test_lexicase_never_consults_the_comparator():
    """It reads the cases, and an order over aggregates is the thing it exists to avoid."""
    population, fitness = by_case([(0.0, 4.0), (4.0, 0.0)])
    selection = LexicaseSelection(random.Random(9))
    assert len(selection.select_parents(population, fitness, _RefusingComparator())) == 2


def test_a_population_of_mixed_case_counts_is_refused():
    """Two individuals measured on different numbers of cases share no notion of "case k"."""
    population, fitness = by_case([(1.0, 2.0), (1.0,)])
    selection = LexicaseSelection(random.Random(10))
    with pytest.raises(ValueError, match="one case count"):
        selection.select_parents(population, fitness, MAXIMIZING)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"epsilon": -0.1}, "tolerance"),
        ({"epsilon": math.nan}, "finite"),
        ({"epsilon": math.inf}, "finite"),
        ({"uniform_share": 1.5}, "probability"),
        ({"uniform_share": -0.1}, "probability"),
        ({"case_sample": 0}, "reads no case"),
    ],
)
def test_lexicase_refuses_settings_outside_their_range(kwargs, message):
    """Each parameter states its range, and a value outside it is a misconfiguration.

    Args:
        kwargs: The setting under test.
        message (str): Part of the message it should be refused with.
    """
    with pytest.raises(ValueError, match=message):
        LexicaseSelection(random.Random(0), **kwargs)


# ---------------------------------------------------------------------------
# Survivor selection: the contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "operator",
    [
        FitnessBasedReplacement(),
        GenerousConservativeReplacement(ExpScalarization(), random.Random(0)),
    ],
)
def test_survivor_selection_sees_parents_and_offspring(operator):
    """The contract is parents and offspring together, which excludes generational replacement.

    Args:
        operator: The survivor selection under test.
    """
    population, fitness = scored([1.0, 2.0, 3.0, 4.0])
    parents, offspring = population[:2], population[2:]
    survivors = operator.select_survivors(parents, offspring, fitness, MAXIMIZING, 2)
    assert len(survivors) == 2
    assert all(survivor in population for survivor in survivors)
    assert isinstance(operator, SurvivorSelection)
    assert not isinstance(operator, ParentSelection)


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_truncation_keeps_the_fittest_of_both_populations():
    """(mu + lambda): the parents compete with the offspring."""
    population, fitness = scored([5.0, 1.0, 4.0, 2.0])
    survivors = FitnessBasedReplacement().select_survivors(population[:2], population[2:], fitness, MAXIMIZING, 2)
    assert set(survivors) == {population[0], population[2]}


def test_truncation_is_conservative():
    """A fittest individual always survives, which is the conservative half."""
    population, fitness = scored([9.0, 1.0, 2.0, 3.0])
    for size in (1, 2, 3):
        survivors = FitnessBasedReplacement().select_survivors(
            population[:2], population[2:], fitness, MAXIMIZING, size
        )
        assert population[0] in survivors


def test_truncation_is_not_generous():
    """The generous half fails: an individual outside the top mu never survives.

    One call is enough, truncation being deterministic, so repeating it adds no information.
    """
    population, fitness = scored([4.0, 3.0, 2.0, 1.0])
    survivors = FitnessBasedReplacement().select_survivors(population[:2], population[2:], fitness, MAXIMIZING, 2)
    assert population[3] not in survivors
    assert set(survivors) == {population[0], population[1]}


def test_truncation_cuts_an_oversized_front_in_the_input_order():
    """A front that does not fit entirely is cut in the order the individuals were passed in."""
    population, fitness = scored([1.0, 1.0, 1.0, 1.0])
    survivors = FitnessBasedReplacement().select_survivors(population[:2], population[2:], fitness, MAXIMIZING, 2)
    assert survivors == population[:2]


def test_truncation_refuses_to_invent_individuals():
    """Fewer candidates than places is a caller error, not a shortfall to be padded."""
    population, fitness = scored([1.0, 2.0])
    with pytest.raises(ValueError, match="for 5 places"):
        FitnessBasedReplacement().select_survivors(population, [], fitness, MAXIMIZING, 5)


def test_truncation_accepts_a_pool_that_exactly_fills_the_places():
    """As many candidates as places is the smallest pool the component accepts, and all of them
    survive."""
    population, fitness = scored([1.0, 2.0, 3.0, 4.0])
    survivors = FitnessBasedReplacement().select_survivors(population[:2], population[2:], fitness, MAXIMIZING, 4)
    assert set(survivors) == set(population)


# ---------------------------------------------------------------------------
# The generous, conservative replacement
# ---------------------------------------------------------------------------


def test_the_new_replacement_is_conservative():
    """A member of greatest scalarized fitness is kept in every single draw.

    The fitness values sit close together on purpose. With a champion at 7 against 1, 2, 3 the
    exponential gives it 97 % of the weight, so a component *without* a reserved place would still
    keep it in 99.4 % of the runs of this test. The measurement could not tell "guaranteed" from
    "overwhelmingly likely", which is the whole distinction being drawn here. At near-equal values
    the reserved place is the only thing that keeps the champion.
    """
    population, fitness = scored([1.0, 1.01, 1.02, 1.03])
    operator = GenerousConservativeReplacement(ExpScalarization(), random.Random(15))
    for _ in range(300):
        survivors = operator.select_survivors(population[:2], population[2:], fitness, MAXIMIZING, 3)
        assert population[3] in survivors


def test_the_new_replacement_is_generous():
    """Every parent and every offspring survives with positive probability: the generous half.

    Near-equal values again, for the opposite reason: with a dominant champion the weakest member
    has a per-draw chance of 0.2 %, so a run of this test would depend on the seed rather than on
    the property.
    """
    population, fitness = scored([1.0, 1.01, 1.02, 1.03])
    operator = GenerousConservativeReplacement(ExpScalarization(), random.Random(16))
    survivors = {
        survivor
        for _ in range(200)
        for survivor in operator.select_survivors(population[:2], population[2:], fitness, MAXIMIZING, 3)
    }
    assert survivors == set(population)


def test_the_new_replacement_survives_a_failed_measurement():
    """A ``nan`` in the pool must not decide who the champion is, in either pool order.

    ``max`` over the raw scalarized values returns ``nan`` when it comes first and a number when it
    does not, and in the first case no member equals it. Reading the champion off the raw values
    therefore raised ``IndexError`` for one order and worked for the other. The weights are where
    the policy for such values already lives.
    """
    operator = GenerousConservativeReplacement(ExpScalarization(), random.Random(21))
    for values in ([math.nan, 5.0], [5.0, math.nan]):
        population, fitness = scored(values)
        survivors = operator.select_survivors(population[:1], population[1:], fitness, MAXIMIZING, 1)
        assert survivors == [population[values.index(5.0)]]


def test_the_new_replacement_refuses_an_overflowing_pool():
    """An overflowing pool is refused rather than left to decide the reserved place by the draw.

    The reserved place goes to a member of greatest scalarized fitness. ``math.exp`` overflows
    above about 710, so 900.0 and 800.0 would both scalarize to infinity, both carry weight 1.0,
    and the tie would be broken uniformly. Always keeping a fittest individual is the conservative
    half of the convergence condition this component carries.
    """
    population, fitness = scored([900.0, 800.0])
    operator = GenerousConservativeReplacement(ExpScalarization(), random.Random(23))
    with pytest.raises(ValueError, match="overflows"):
        operator.select_survivors(population[:1], population[1:], fitness, MAXIMIZING, 1)


def test_the_new_replacement_fills_exactly_the_population_size():
    """mu places, no more and no fewer."""
    population, fitness = scored([1.0, 2.0, 3.0])
    operator = GenerousConservativeReplacement(ExpScalarization(), random.Random(17))
    for size in (1, 2, 5):
        assert len(operator.select_survivors(population[:1], population[1:], fitness, MAXIMIZING, size)) == size


def test_the_new_replacement_breaks_a_tie_among_champions_uniformly():
    """Several individuals may share the greatest value, and the reserved place is drawn among
    them."""
    population, fitness = scored([5.0, 5.0, 1.0])
    operator = GenerousConservativeReplacement(ExpScalarization(), random.Random(18))
    reserved = Counter(
        operator.select_survivors(population[:1], population[1:], fitness, MAXIMIZING, 1)[0] for _ in range(2000)
    )
    assert set(reserved) == {population[0], population[1]}
    assert abs(reserved[population[0]] - reserved[population[1]]) < 0.1 * 2000


def test_the_new_replacement_refuses_an_empty_pool():
    """There is nothing to select from, and inventing an individual is not an option."""
    operator = GenerousConservativeReplacement(ExpScalarization(), random.Random(19))
    with pytest.raises(ValueError, match="no individuals"):
        operator.select_survivors([], [], {}, MAXIMIZING, 2)


def test_the_new_replacement_refuses_a_negative_population_size():
    """A negative population size is a caller error, and a size of zero is not."""
    population, fitness = scored([1.0, 2.0])
    operator = GenerousConservativeReplacement(ExpScalarization(), random.Random(22))
    with pytest.raises(ValueError, match="cannot be negative"):
        operator.select_survivors(population[:1], population[1:], fitness, MAXIMIZING, -1)


def test_the_new_replacement_answers_zero_places_with_an_empty_population():
    """A size of zero is answered with the empty population, whatever was offered.

    The pool is read only after this, so zero places with an empty pool is the empty population
    and not the error an empty pool raises for a positive size.
    """
    population, fitness = scored([1.0, 2.0])
    operator = GenerousConservativeReplacement(ExpScalarization(), random.Random(23))
    assert operator.select_survivors(population[:1], population[1:], fitness, MAXIMIZING, 0) == []
    assert operator.select_survivors([], [], {}, MAXIMIZING, 0) == []


# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------


def test_no_operator_touches_the_global_random_stream():
    """Every draw goes through the component's own generator."""
    population, fitness = scored([1.0, 2.0, 3.0])
    operators = [
        TournamentSelection(2, random.Random(0)),
        FitnessProportionalSelection(ExpScalarization(), random.Random(0)),
        RankBasedSelection(1.7, random.Random(0)),
    ]
    random.seed(999)
    before = random.random()
    random.seed(999)
    for operator in operators:
        operator.select_parents(population, fitness, MAXIMIZING)
    GenerousConservativeReplacement(ExpScalarization(), random.Random(0)).select_survivors(
        population[:1], population[1:], fitness, MAXIMIZING, 2
    )
    assert random.random() == before


def test_selection_works_under_a_partial_order():
    """Every order-based component consumes :class:`Comparison`, incomparability included."""
    population = [Tree(a2, ()), Tree(b2, ()), Tree(h1, (Tree(a2, ()),))]
    fitness = {
        population[0]: [2.0, 1.0],
        population[1]: [1.0, 2.0],
        population[2]: [1.0, 1.0],
    }
    pareto = ParetoFitnessComparator()
    assert pareto.compare(fitness[population[0]], fitness[population[1]]) is (Comparison.INCOMPARABLE)
    tournament = TournamentSelection(3, random.Random(20))
    winners = {parent for _ in range(200) for parent in tournament.select_parents(population, fitness, pareto)}
    assert winners == {population[0], population[1]}
    survivors = FitnessBasedReplacement().select_survivors(population[:1], population[1:], fitness, pareto, 2)
    assert set(survivors) == {population[0], population[1]}
