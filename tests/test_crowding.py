"""NSGA-II's crowding: the distance within a front and the cut of the last front by it.

Deb, Pratap, Agarwal and Meyarivan, "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II" (2002), §III-B:
per objective a front is sorted, its two boundary solutions get an infinite distance and every other one the gap
between its two neighbours, normalized by the objective's extent; the distances are summed over the objectives. The
crowded comparison prefers the lower nondomination rank and, within one rank, the larger distance. The main loop
(§III-C) cuts the last front that does not fit by that comparison, and draws parents by binary tournaments under it.
The distances below are computed by hand from that procedure, and the distances and the cut are compared with a
transcription of the paper's two listings on random fronts.
"""

import math
import random

import pytest

from cosy.core.tree import Tree
from cosy.evolutionary_algorithms import (
    CrowdedTruncation,
    FitnessBasedReplacement,
    ParetoFitnessComparator,
    crowding_distances,
    dominance_fronts,
)
from tests.ea_fixtures import chain

PARETO = ParetoFitnessComparator()


# ---------------------------------------------------------------------------
# A reference transcribed from the paper's two listings
# ---------------------------------------------------------------------------


def reference_distances(vectors: list[tuple[float, ...]]) -> list[float]:
    """``crowding-distance-assignment(I)`` of Deb et al. (2002), p. 185, line by line.

    ``sort(I, m)`` is left unspecified for ties there; this one is stable, the convention of the
    implementation. The listing divides by ``f_max - f_min``, so it is defined only for fronts of
    three or more with a positive extent in every objective, and is asked only about those.
    """
    count = len(vectors)
    distance = [0.0] * count
    for objective in range(len(vectors[0])):
        order = sorted(range(count), key=lambda i: vectors[i][objective])
        distance[order[0]] = distance[order[count - 1]] = math.inf
        low, high = vectors[order[0]][objective], vectors[order[count - 1]][objective]
        for i in range(1, count - 1):
            distance[order[i]] += (vectors[order[i + 1]][objective] - vectors[order[i - 1]][objective]) / (high - low)
    return distance


def reference_fronts(vectors: list[tuple[float, ...]]) -> list[list[int]]:
    """The nondominated fronts of the positions, by the definition of Pareto dominance, maximizing.

    Within a front the positions keep their order, the convention of the implementation.
    """

    def dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
        return all(x >= y for x, y in zip(a, b, strict=True)) and any(x > y for x, y in zip(a, b, strict=True))

    remaining = list(range(len(vectors)))
    fronts = []
    while remaining:
        front = [i for i in remaining if not any(dominates(vectors[j], vectors[i]) for j in remaining)]
        fronts.append(front)
        remaining = [i for i in remaining if i not in front]
    return fronts


def reference_survivors(vectors: list[tuple[float, ...]], size: int) -> list[int]:
    """The choice of ``P_{t+1}`` in the listing of Deb et al. (2002), p. 186, by position.

    Whole fronts while they fit, then the last front sorted by the crowded comparison in descending
    order, which within one front is the distance, and its first places taken.
    """
    fronts = reference_fronts(vectors)
    chosen: list[int] = []
    index = 0
    while index < len(fronts) and len(chosen) + len(fronts[index]) <= size:
        chosen.extend(fronts[index])
        index += 1
    if len(chosen) < size:
        last = fronts[index]
        distance = reference_distances([vectors[i] for i in last])
        order = sorted(range(len(last)), key=lambda j: -distance[j])
        chosen.extend(last[j] for j in order[: size - len(chosen)])
    return chosen


def scored(vectors: list[tuple[float, ...]]) -> tuple[list[Tree], dict[Tree, tuple[float, ...]]]:
    """Build distinct individuals carrying the given objective vectors.

    Args:
        vectors (list[tuple[float, ...]]): One vector per individual.

    Returns:
        tuple[list[Tree], dict[Tree, tuple[float, ...]]]: The population and its fitness mapping.
    """
    population = [chain(index) for index in range(len(vectors))]
    return population, dict(zip(population, vectors, strict=True))


# ---------------------------------------------------------------------------
# The crowding distance
# ---------------------------------------------------------------------------


def test_evenly_spaced_points_on_a_line_have_equal_inner_distances_and_infinite_ends():
    """Each objective adds the gap 2 / (n - 1) over an extent of 1, so an inner point carries 4 / (n - 1)."""
    count = 11
    population, fitness = scored([(index / (count - 1), 1 - index / (count - 1)) for index in range(count)])
    distances = crowding_distances(population, fitness)
    assert distances[0] == distances[-1] == math.inf
    assert distances[1:-1] == pytest.approx([4 / (count - 1)] * (count - 2))


def test_the_distance_is_the_sum_of_the_normalized_neighbour_gaps():
    """Computed by hand: the first objective spans 6, the second 10, and an inner point reads its neighbours in each."""
    population, fitness = scored([(0.0, 10.0), (1.0, 6.0), (3.0, 5.0), (6.0, 0.0)])
    distances = crowding_distances(population, fitness)
    assert distances[0] == distances[3] == math.inf
    assert distances[1] == pytest.approx((3 - 0) / 6 + (10 - 5) / 10)
    assert distances[2] == pytest.approx((6 - 1) / 6 + (6 - 0) / 10)


def test_an_affine_rescaling_of_an_objective_changes_no_distance():
    """Every objective is normalized by its extent, so its unit and its origin do not matter.

    The control beside it: three objectives mark at most six boundary solutions, so at least nine of the fifteen
    points are inner ones with a finite, positive distance, and the comparison is not one of two lists of infinities
    or zeros.
    """
    rng = random.Random(3)
    population, fitness = scored([(rng.random(), rng.random(), rng.random()) for _ in range(15)])
    rescaled = {member: (7.0 * x + 3.0, 1e6 * y - 2.0, z / 1e3) for member, (x, y, z) in fitness.items()}
    distances = crowding_distances(population, fitness)
    assert crowding_distances(population, rescaled) == pytest.approx(distances)
    assert sum(0 < distance < math.inf for distance in distances) >= 15 - 3 * 2


@pytest.mark.parametrize("count", [1, 2])
def test_a_front_of_one_or_two_is_all_boundary(count):
    """With no inner point every member is a boundary solution.

    Args:
        count (int): The front's size.
    """
    population, fitness = scored([(float(index), float(-index)) for index in range(count)])
    assert crowding_distances(population, fitness) == [math.inf] * count


def test_an_objective_without_extent_adds_nothing():
    """A constant objective has no extent to normalize by and no boundary to mark.

    It neither divides by zero nor adds anything, so the inner point of the other objective stays inner, here where it
    comes first in the input and would be the first of the constant objective's order.
    """
    population, fitness = scored([(1.0, 5.0), (0.0, 5.0), (2.0, 5.0)])
    assert crowding_distances(population, fitness) == [pytest.approx((2 - 0) / 2), math.inf, math.inf]


def test_a_failed_measurement_is_the_most_crowded_and_left_out_of_the_others():
    """A vector with a value that is not a number gets distance 0, and the rest are measured as if it were absent."""
    population, fitness = scored([(0.0, 1.0), (math.nan, 0.5), (0.5, 0.5), (1.0, 0.0)])
    distances = crowding_distances(population, fitness)
    assert distances[1] == 0.0
    assert distances[0] == distances[3] == math.inf
    assert distances[2] == pytest.approx(1.0 + 1.0)


def test_the_distances_follow_the_paper_s_listing_on_random_fronts_with_ties_and_copies():
    """Fronts of three to twelve places in one to four objectives, on a coarse grid, so that values
    tie, and with repeated individuals, against the transcription of the listing.

    The listing is defined where every objective has a positive extent; the other fronts are
    skipped, and counted.
    """
    rng = random.Random(17)
    compared = skipped = 0
    for _trial in range(400):
        count, width = rng.randint(3, 12), rng.randint(1, 4)
        vectors = [tuple(float(rng.randint(0, 4)) for _ in range(width)) for _ in range(count)]
        population, fitness = scored(vectors)
        if rng.random() < 0.3:
            # A copy of the first individual stands in a second place, with its fitness.
            population[1] = population[0]
            vectors[1] = vectors[0]
        if any(len({vector[m] for vector in vectors}) < 2 for m in range(width)):
            skipped += 1
            continue
        assert crowding_distances(population, fitness) == pytest.approx(reference_distances(vectors)), vectors
        compared += 1
    # Counted from the construction alone: 381 fronts compared and 19 skipped for this seed.
    assert compared >= 350, (compared, skipped)


def test_an_infinite_value_is_the_boundary_of_its_objective():
    """An infinite value is the most extreme of its objective, so its solution is a boundary one.

    The driver admits it, since it compares equal to itself, and for a maximized objective it is
    the best value there is. The finite values of that objective are measured among themselves:
    the first objective's inner point reads 3 - 1 over 3 - 1, the second objective's 3 - 1 over
    3 - 0. The finite extremes border an infinitely distant neighbour and are boundary solutions.
    """
    population, fitness = scored([(math.inf, 0.0), (3.0, 1.0), (2.0, 2.0), (1.0, 3.0)])
    assert crowding_distances(population, fitness) == [
        math.inf,
        math.inf,
        pytest.approx(1.0 + 2 / 3),
        math.inf,
    ]
    kept = CrowdedTruncation().select_survivors(population, [], fitness, PARETO, 3)
    assert set(kept) == {population[0], population[1], population[3]}
    lowest, others = scored([(-math.inf, 5.0), (1.0, 3.0), (2.0, 2.0), (3.0, 1.0)])
    assert crowding_distances(lowest, others)[0] == math.inf


def test_a_finite_extreme_beside_infinite_values_is_a_boundary_solution():
    """Latency is infinite for the two designs that do not fit; the one that fits has the smallest latency.

    It is the boundary solution of that objective, the smallest value there is, even though it is
    the only finite one and so spans no extent of its own.
    """
    maximize = ParetoFitnessComparator(maximize=[True, False, False])
    population, fitness = scored([(0.95, math.inf, 60.0), (0.30, math.inf, 5.0), (0.60, 3.0, 30.0)])
    assert dominance_fronts(population, fitness, maximize) == [population]
    assert crowding_distances(population, fitness) == [math.inf, math.inf, math.inf]
    # Offered first, it ties with the two infinite ones rather than being cut as an inner point.
    offered = [population[2], population[0], population[1]]
    kept = CrowdedTruncation().select_survivors(offered, [], fitness, maximize, 2)
    assert population[2] in kept


def test_a_failed_measurement_in_a_later_objective_is_as_failed_as_in_the_first():
    """A value that is not a number fails its vector in whichever objective it stands."""
    population, fitness = scored([(0.0, 1.0), (0.5, math.nan), (0.5, 0.5), (1.0, 0.0)])
    assert crowding_distances(population, fitness) == [math.inf, 0.0, pytest.approx(2.0), math.inf]


def test_two_failed_measurements_leave_one_measured_solution_a_boundary():
    """Of three places two failed: the one measured is all that is left, a front of one."""
    population, fitness = scored([(math.nan, 1.0), (0.0, math.nan), (0.5, 0.5)])
    assert crowding_distances(population, fitness) == [0.0, 0.0, math.inf]


def test_a_front_near_the_largest_float_is_measured_as_the_same_front_scaled_down():
    """An extent past the largest float would overflow, and every distance divided by it with it.

    Measured as if scaled down, the front keeps both boundary solutions when it is cut.
    """
    vectors = [(-1e308, 1e308), (0.0, 0.0), (0.9e308, -0.9e308), (1e308, -1e308)]
    population, huge = scored(vectors)
    small = {member: (x * 1e-308, y * 1e-308) for member, (x, y) in huge.items()}
    assert crowding_distances(population, huge) == pytest.approx(crowding_distances(population, small))
    assert crowding_distances(population, huge)[1:3] == pytest.approx([1.9, 1.0])
    kept = CrowdedTruncation().select_survivors(population, [], huge, PARETO, 2)
    assert set(kept) == {population[0], population[3]}


def test_no_distance_is_ever_not_a_number():
    """Over fronts mixing ordinary, huge, infinite and failed values, every distance is a number.

    A distance that is not a number would make the tournament's comparison depend on the order of
    its draw, so the place it is read in decides nothing in its favour or against it.
    """
    rng = random.Random(13)
    choices = [0.0, 1.0, -2.5, 1e308, -1e308, 1.5e308, math.inf, -math.inf]
    for _trial in range(300):
        vectors = [
            (math.nan, rng.choice(choices)) if rng.random() < 0.1 else (rng.choice(choices), rng.choice(choices))
            for _ in range(rng.randint(1, 9))
        ]
        population, fitness = scored(vectors)
        distances = crowding_distances(population, fitness)
        assert not any(math.isnan(distance) for distance in distances), vectors
        for distance, vector in zip(distances, vectors, strict=True):
            if any(math.isnan(value) for value in vector):
                assert distance == 0.0
            elif any(math.isinf(value) for value in vector):
                assert distance == math.inf


def test_a_front_mixing_numbers_of_objectives_is_refused():
    """A distance sums one gap per objective, so a front whose vectors differ in length has none.

    A custom comparator can put such vectors into one front; the distance is refused rather than
    read off whichever length came first.
    """
    population, fitness = scored([(0.0, 1.0), (0.5,), (1.0, 0.0)])
    with pytest.raises(ValueError, match=r"one number of objectives for the whole front, got \[1, 2\]"):
        crowding_distances(population, fitness)


def test_a_repeated_individual_keeps_a_place_of_its_own():
    """A population is a multiset: the distances are per place, and a copy is a neighbour like any other."""
    population, fitness = scored([(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)])
    repeated = [population[0], population[1], population[1], population[2]]
    distances = crowding_distances(repeated, fitness)
    assert len(distances) == 4
    assert distances[0] == distances[3] == math.inf
    assert sorted(distances[1:3]) == pytest.approx([1.0, 1.0])


# ---------------------------------------------------------------------------
# The crowded truncation
# ---------------------------------------------------------------------------


def test_fronts_that_fit_are_kept_whole_as_the_truncation_keeps_them():
    """Where no front has to be cut, crowding decides nothing: the survivors are the truncation's.

    Over random populations, each with as many places as its first fronts hold, the two lists agree
    place by place; a selection that ignored the fronts would keep a dominated boundary solution in
    some of them.
    """
    rng = random.Random(5)
    left_out = 0
    for _trial in range(100):
        population, fitness = scored([(rng.random(), rng.random()) for _ in range(12)])
        fronts = dominance_fronts(population, fitness, PARETO)
        size = sum(len(front) for front in fronts[: rng.randint(1, len(fronts) - 1)])
        kept = CrowdedTruncation().select_survivors(population[:6], population[6:], fitness, PARETO, size)
        plain = FitnessBasedReplacement().select_survivors(population[:6], population[6:], fitness, PARETO, size)
        assert kept == plain
        left_out += size < len(population)
    assert left_out == 100, "the control: every trial leaves a front out"


def test_the_cut_front_keeps_its_boundary_solutions_and_spans_the_front_in_every_trial():
    """Forty points on x + y = 1 arrive clustered, sorted by x; ten survive.

    The truncation in input order keeps the ten leftmost and spans about a fifth of the front, which the control at
    the end asserts; the crowded cut keeps both boundary solutions, so it spans the whole front in every trial.
    """
    rng = random.Random(0)
    plain_spans = []
    for _trial in range(200):
        xs = sorted(rng.random() for _ in range(40))
        population, fitness = scored([(x, 1 - x) for x in xs])
        kept = CrowdedTruncation().select_survivors(population, [], fitness, PARETO, 10)
        assert len(kept) == 10
        assert population[0] in kept
        assert population[-1] in kept
        plain = FitnessBasedReplacement().select_survivors(population, [], fitness, PARETO, 10)
        plain_spans.append(max(fitness[member][0] for member in plain) - min(fitness[member][0] for member in plain))
    assert sum(plain_spans) / len(plain_spans) < 0.5, "the control: input order does not span the front"


def test_the_cut_keeps_the_larger_distances():
    """Among inner solutions the most crowded go first: here the two pressed against a neighbour."""
    population, fitness = scored([(0.0, 1.0), (0.3, 0.7), (0.31, 0.69), (0.6, 0.4), (1.0, 0.0)])
    kept = CrowdedTruncation().select_survivors(population, [], fitness, PARETO, 3)
    assert set(kept) == {population[0], population[3], population[4]}


def test_the_cut_follows_the_paper_s_listing_on_random_pools():
    """Pools of several fronts with ties and repeated individuals, cut at every size, against the
    transcription of the listing that forms ``P_{t+1}``, place by place and in its order.
    """
    rng = random.Random(19)
    compared = cut = 0
    for _trial in range(150):
        count = rng.randint(2, 14)
        vectors = [(float(rng.randint(0, 5)), float(rng.randint(0, 5))) for _ in range(count)]
        population, fitness = scored(vectors)
        if rng.random() < 0.4:
            population[1] = population[0]
            vectors[1] = vectors[0]
        fronts = reference_fronts(vectors)
        defined = all(
            len(front) < 3 or all(len({vectors[i][m] for i in front}) >= 2 for m in range(2)) for front in fronts
        )
        if not defined:
            continue
        split = rng.randint(0, count)
        parents, offspring = population[:split], population[split:]
        for size in range(count + 1):
            kept = CrowdedTruncation().select_survivors(parents, offspring, fitness, PARETO, size)
            assert kept == [population[i] for i in reference_survivors(vectors, size)], (vectors, size)
            compared += 1
            cut += size not in {sum(len(front) for front in fronts[:k]) for k in range(len(fronts) + 1)}
    # Counted from the construction and the reference alone: 1 250 sizes, 505 of them cutting a
    # front, for this seed.
    assert compared >= 1100
    assert cut >= 450, "the control: many sizes cut a front"


def test_the_crowded_truncation_refuses_to_invent_individuals():
    """Fewer offered than places is an error, as for the truncation."""
    population, fitness = scored([(0.0, 1.0), (1.0, 0.0)])
    with pytest.raises(ValueError, match="offered 2 individuals for 3 places"):
        CrowdedTruncation().select_survivors(population, [], fitness, PARETO, 3)


def test_the_crowded_truncation_refuses_a_negative_size():
    """A negative number of places is no population size, and an empty answer would hide that."""
    population, fitness = scored([(0.0, 1.0), (1.0, 0.0)])
    with pytest.raises(ValueError, match="a population size cannot be negative: -1"):
        CrowdedTruncation().select_survivors(population, [], fitness, PARETO, -1)
