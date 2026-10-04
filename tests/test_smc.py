"""Sequential Monte Carlo over the lazy search tree: particles guided by a construction's own weights, corrected to its target.

A particle walks from the root to a term, choosing each child in proportion to the guide's weight for it, and carries the ratio
of the children's weights summed to its own node's weight; at the term it carries the target's weight over the guide's. Its
weight is then the target over the root's weight and the probability of its path, so the weighted particles follow the
target whatever the guide's errors, and with an exact guide every particle weighs the same. These tests hold that invariant
exactly on every particle, hold the weighted law to the exact target on small spaces where the saddle point's guide alone
is far from it, hold the estimate of the normalizer to its exact value on average, and pin the resampling rule, the
sampler and the refusals.
"""

import bisect
import itertools
import math
import random
from collections import Counter
from typing import ClassVar

import pytest

from cosy.search import generator_query, residual_query
from cosy.search.cost_tables import weighted_cost_table
from cosy.search.samplers import Sampler, SMCSampler
from cosy.search.sampling import log_sum_exp
from cosy.search.smc import WeightedParticles, sequential_monte_carlo
from cosy.search.tilt import saddle_search, tilted_search
from tests.search_fixtures import PRICED, priced_space
from tests.test_tilt import FRACTIONAL, all_terms, digit_sums, priced, space_of

EDGES = (0.0, 2.5, 5.5, 8.5, 13.0)
TARGET = (0.2, 0.3, 0.4, 0.1)


def exact_target_law(construction, terms):
    """The target's law over the terms: each term's target weight, normalized over the terms with one."""
    log_weights = [construction.log_target_of(term) for term in terms]
    total = log_sum_exp(log_weights)
    return {term: math.exp(value - total) for term, value in zip(terms, log_weights, strict=True) if value > -math.inf}


def guide_law(construction):
    """The law of the guide's own first draw: every path walked, each child chosen in proportion to its weight."""
    root, _root_log_weight, expand = construction.expansion()
    law = Counter()
    stack = [(root, 0.0)]
    while stack:
        node, log_probability = stack.pop()
        inhabitant, children = expand(node)
        if inhabitant is not None:
            law[inhabitant] += math.exp(log_probability)
            continue
        if not children:
            continue
        total = log_sum_exp([log_weight for _, log_weight in children])
        stack.extend((child, log_probability + log_weight - total) for child, log_weight in children)
    return law


def weighted_law(result):
    """The weighted particles' law over their terms."""
    total = log_sum_exp(list(result.log_weights))
    law = Counter()
    for term, log_weight in zip(result.terms, result.log_weights, strict=True):
        if term is not None and log_weight > -math.inf:
            law[term] += math.exp(log_weight - total)
    return law


def distance(first, second):
    """Total variation between two laws over terms."""
    return 0.5 * sum(abs(first.get(term, 0.0) - second.get(term, 0.0)) for term in set(first) | set(second))


def saddle_on(parts, holes):
    """The saddle search on a sum of ``holes`` choices of ``parts``, to the module's bins and target, and its terms."""
    query, algebra = digit_sums(parts, holes)
    return saddle_search(query, algebra, EDGES, TARGET), all_terms(query, algebra), algebra


def test_with_an_exact_guide_every_particle_weighs_the_same_and_none_is_resampled():
    """The cost table's weights are the target's mass below each node: every particle's weight is the root's, exactly."""
    query, algebra = digit_sums((0, 1, 3), 6)
    table = weighted_cost_table(query, algebra, lambda cost: 1.0 / (1 + cost), 18)
    result = sequential_monte_carlo(table, random.Random(1), 300)
    assert isinstance(result, WeightedParticles)
    assert all(term is not None for term in result.terms)
    spread = max(result.log_weights) - min(result.log_weights)
    assert spread < 1e-9, spread
    assert result.resamplings == 0
    assert math.isclose(result.ess, 300, rel_tol=1e-9)


@pytest.mark.parametrize("seed", range(3))
def test_a_particles_weight_is_its_target_over_the_root_and_its_paths_probability(seed):
    """Without resampling, every particle: its log weight plus its path's log probability is its target over the root's weight."""
    search, _terms, _algebra = saddle_on((0, 1, 3), 5)
    _root, root_log_weight, _expand = search.expansion()
    result = sequential_monte_carlo(search, random.Random(seed), 200, threshold=0.0)
    assert result.resamplings == 0
    checked = 0
    for term, log_weight, log_proposal in zip(result.terms, result.log_weights, result.log_proposals, strict=True):
        if term is None:
            assert log_weight == -math.inf
            continue
        expected = search.log_target_of(term) - root_log_weight
        assert math.isclose(log_weight + log_proposal, expected, rel_tol=1e-9, abs_tol=1e-9), (
            log_weight,
            log_proposal,
            expected,
        )
        checked += 1
    assert checked >= 150
    # Not trivially: the guide is not exact here, so the weights differ.
    finite = [value for value in result.log_weights if value > -math.inf]
    assert max(finite) - min(finite) > 0.05


def cluster_beside_a_free_term(parts):
    """``S -> f(D x parts) | z``, the parts near 10^4 and ``z`` free: two groups far apart, where one tilt reads both badly.

    Returns:
        tuple: The saddle search to uniform bins of width 3 over the cluster, its terms, and its algebra.
    """
    rules = [("S", "f", ("D",) * parts), ("S", "z", ()), ("D", "d0", ()), ("D", "d1", ()), ("D", "d3", ())]
    query = generator_query(space_of(rules), "S")
    algebra = priced({"f": 0, "z": 0, "d0": 1e4, "d1": 1e4 + 1, "d3": 1e4 + 3})
    edges = [parts * 1e4 + v for v in range(0, 3 * parts + 3, 3)]
    return saddle_search(query, algebra, edges, [1.0] * (len(edges) - 1)), all_terms(query, algebra), algebra


def test_the_weighted_particles_follow_the_target_where_the_guide_alone_does_not():
    """Beside a free term, the saddle point's guide is off its own target by 0.05; the weighted particles are not."""
    search, terms, _algebra = cluster_beside_a_free_term(4)
    target = exact_target_law(search, terms)
    guide = guide_law(search)
    assert distance(guide, target) > 0.045, distance(guide, target)
    pooled = Counter()
    for seed in range(20):
        result = sequential_monte_carlo(search, random.Random(seed), 4000)
        for term, mass in weighted_law(result).items():
            pooled[term] += mass / 20
    assert distance(pooled, target) < 0.03, distance(pooled, target)


def test_the_normalizer_estimate_is_unbiased():
    """Over many runs the mean of the normalizer's estimate is the targets' sum over the root's weight."""
    search, terms, _algebra = saddle_on((0, 1, 3), 5)
    _root, root_log_weight, _expand = search.expansion()
    exact = math.exp(log_sum_exp([search.log_target_of(term) for term in terms]) - root_log_weight)
    estimates = [
        math.exp(sequential_monte_carlo(search, random.Random(seed), 60).log_normalizer) for seed in range(300)
    ]
    mean = sum(estimates) / len(estimates)
    spread = math.sqrt(sum((value - mean) ** 2 for value in estimates) / (len(estimates) - 1))
    assert abs(mean - exact) <= 4 * spread / math.sqrt(len(estimates)), (mean, exact, spread)
    assert spread > 0  # an estimate, not a constant


class TiltToBins:
    """A poor guide on purpose: the tilted search's weights, and a target on cost bins the tilt knows nothing of.

    The target spreads each bin's mass over the bin's terms by their exact number, so that it sums to one, the scale of
    the tilt's weights; every particle then carries a final correction, and on a space whose terms end at different depths
    the particles that end first are corrected while the rest walk on, and resampling acts on uneven weights.
    """

    def __init__(self, query, algebra, theta, edges, target, terms):
        self.search = tilted_search(query, algebra, theta)
        self.algebra = algebra
        self.edges = edges
        counts = Counter(self.bin_of(term) for term in terms)
        self.log_target = [
            math.log(mass) - math.log(counts[index]) if mass > 0 and counts[index] else -math.inf
            for index, mass in enumerate(target)
        ]

    def bin_of(self, term):
        index = bisect.bisect_right(self.edges, self.algebra.fold(term)) - 1
        return index if 0 <= index < len(self.edges) - 1 else None

    def expansion(self):
        return self.search.expansion()

    def log_target_of(self, term):
        index = self.bin_of(term)
        return -math.inf if index is None else self.log_target[index]


PRICED_EDGES = (1.0, 3.0, 4.0, 5.0, 10.0)
PRICED_TARGET = (0.1, 0.3, 0.4, 0.2)


def tilt_to_bins(theta=0.3):
    """The poor guide on the space of 1 750 terms with constants, whose terms end at different depths."""
    query = generator_query(priced_space(), PRICED)
    terms = all_terms(query, FRACTIONAL)
    return TiltToBins(query, FRACTIONAL, theta, PRICED_EDGES, PRICED_TARGET, terms), terms


def bins_law(guide, law):
    """A law over terms read on the guide's bins."""
    out = Counter()
    for term, mass in law.items():
        out[guide.bin_of(term)] += mass
    return out


def test_with_a_poor_guide_every_particle_carries_its_final_correction():
    """The tilt's weight of a term is not the target's: the invariant holds only with the correction at the term."""
    guide, _terms = tilt_to_bins()
    _root, root_log_weight, _expand = guide.expansion()
    result = sequential_monte_carlo(guide, random.Random(7), 300, threshold=0.0)
    corrections = 0
    for term, log_weight, log_proposal in zip(result.terms, result.log_weights, result.log_proposals, strict=True):
        expected = guide.log_target_of(term) - root_log_weight
        assert math.isclose(log_weight + log_proposal, expected, rel_tol=1e-9, abs_tol=1e-9)
        corrections += abs(log_weight) > 1e-6
    assert corrections > 250  # nearly every particle's weight is its correction


@pytest.mark.parametrize("threshold", [0.95, 1.0])
def test_with_a_poor_guide_and_resampling_the_weighted_particles_follow_the_target(threshold):
    """Resampling on uneven weights, below 95 percent or at every step: the weighted particles' bins follow the target."""
    guide, terms = tilt_to_bins()
    target = bins_law(guide, exact_target_law(guide, terms))
    assert max(abs(target[i] - mass) for i, mass in enumerate(PRICED_TARGET)) < 1e-12
    assert 0.5 * sum(abs(bins_law(guide, guide_law(guide))[i] - mass) for i, mass in enumerate(PRICED_TARGET)) > 0.1
    pooled = Counter()
    resamplings = 0
    for seed in range(20):
        result = sequential_monte_carlo(guide, random.Random(seed), 2000, threshold=threshold)
        resamplings += result.resamplings
        for index, mass in bins_law(guide, weighted_law(result)).items():
            pooled[index] += mass / 20
    assert resamplings >= 20
    assert max(abs(pooled[i] - mass) for i, mass in enumerate(PRICED_TARGET)) < 0.02, pooled


def test_with_resampling_the_normalizer_estimate_is_unbiased():
    """Resampling at every step: the mean of the normalizer's estimate is still the targets' sum over the root's weight."""
    guide, terms = tilt_to_bins()
    _root, root_log_weight, _expand = guide.expansion()
    exact = math.exp(log_sum_exp([guide.log_target_of(term) for term in terms]) - root_log_weight)
    runs = [sequential_monte_carlo(guide, random.Random(seed), 60, threshold=1.0) for seed in range(400)]
    assert sum(run.resamplings for run in runs) >= 400
    estimates = [math.exp(run.log_normalizer) for run in runs]
    mean = sum(estimates) / len(estimates)
    spread = math.sqrt(sum((value - mean) ** 2 for value in estimates) / (len(estimates) - 1))
    assert abs(mean - exact) <= 4 * spread / math.sqrt(len(estimates)), (mean, exact, spread)


def test_resampling_happens_when_the_effective_sample_falls_below_the_threshold():
    """A threshold of one resamples at every step the weights are uneven, of zero never; the effective sizes are recorded."""
    search, _terms, _algebra = saddle_on((0, 1, 3), 5)
    never = sequential_monte_carlo(search, random.Random(4), 100, threshold=0.0)
    always = sequential_monte_carlo(search, random.Random(4), 100, threshold=1.0)
    assert never.resamplings == 0
    assert always.resamplings >= 3
    assert len(always.ess_trace) == always.steps
    assert all(0 < value <= 100 + 1e-9 for value in always.ess_trace)


def test_each_constructions_target_weight_is_its_law_computed_on_its_own():
    """``log_target_of`` against an independent computation: the tilt's from Z enumerated, the table's from the counts
    enumerated, the saddle search's from its root counts with the bin found by bisection here."""
    query, algebra = digit_sums((0, 1, 3), 5)
    terms = all_terms(query, algebra)
    costs = {term: algebra.fold(term) for term in terms}
    tilted = tilted_search(query, algebra, 0.3)
    log_z = log_sum_exp([-0.3 * cost for cost in costs.values()])
    for term, cost in costs.items():
        assert math.isclose(tilted.log_target_of(term), -0.3 * cost - log_z, rel_tol=1e-12, abs_tol=1e-12)
    table = weighted_cost_table(query, algebra, lambda cost: 1.0 + cost, 15)
    per_value = Counter(costs.values())
    total = sum(1.0 + value for value in per_value)
    for term, cost in costs.items():
        expected = math.log((1.0 + cost) / total) - math.log(per_value[cost])
        assert math.isclose(table.log_target_of(term), expected, rel_tol=1e-12, abs_tol=1e-12)
    search = saddle_search(query, algebra, EDGES, TARGET)
    reached = [mass for index, mass in enumerate(TARGET) if search.log_root_counts[index] > -math.inf]
    checked = 0
    for term, cost in costs.items():
        index = bisect.bisect_right(EDGES, cost) - 1
        if not 0 <= index < len(TARGET) or search.log_root_counts[index] == -math.inf:
            assert search.log_target_of(term) == -math.inf
            continue
        expected = math.log(TARGET[index] / sum(reached)) - search.log_root_counts[index]
        assert math.isclose(search.log_target_of(term), expected, rel_tol=1e-12, abs_tol=1e-12)
        checked += 1
    assert checked > 100


def test_an_empty_construction_hands_the_particles_a_root_without_weight():
    """A query without a term: its expansion's root weighs nothing, the stream is empty, every particle ends at once."""
    query, algebra = digit_sums((1, 2, 3), 3)
    search = tilted_search(query, algebra, 0.0)
    assert search.expansion()[1] == 0.0
    empty_table = weighted_cost_table(query, algebra, lambda _cost: 1.0, 2)  # every term costs at least 3
    assert empty_table.expansion()[1] == -math.inf
    assert list(empty_table.keyed_stream(random.Random(0))) == []
    result = sequential_monte_carlo(empty_table, random.Random(0), 5)
    assert result.terms == (None,) * 5
    assert result.log_normalizer == -math.inf
    assert result.steps == 0


def test_with_an_exact_guide_a_threshold_of_one_resamples_nothing():
    """Equal weights are equal up to rounding: a threshold of one must not read them as uneven."""
    query, algebra = digit_sums((0, 1, 3), 6)
    table = weighted_cost_table(query, algebra, lambda cost: 1.0 / (1 + cost), 18)
    for particles in (299, 300, 1000):
        result = sequential_monte_carlo(table, random.Random(particles), particles, threshold=1.0)
        assert result.resamplings == 0, particles


def test_draws_follow_the_particles_weights():
    """From one run on a space whose guide is off: draws in proportion to the weights, not one per particle."""
    search, _terms, _algebra = cluster_beside_a_free_term(4)
    result = sequential_monte_carlo(search, random.Random(3), 400, threshold=0.0)
    weighted = weighted_law(result)
    plain = Counter()
    for term in result.terms:
        if term is not None:
            plain[term] += 1 / len(result.terms)
    assert distance(weighted, plain) > 0.03
    drawn = Counter(result.draw(random.Random(4), 200000))
    observed = Counter({term: count / 200000 for term, count in drawn.items()})
    assert distance(observed, weighted) < 0.015, distance(observed, weighted)


class DeadEnds:
    """A hand-built tree with a dead end and terms at three depths; the guide inexact on purpose.

    ``root -> A (0.5) | C (0.3, a dead end) | t4 (0.2)``, ``A -> t1 (0.3) | B (0.4)``, ``B -> t2 (0.2) | t3 (0.1)``; the target
    0.25, 0.15, 0.05, 0.4 on t1 .. t4, so the normalizer is 0.85.
    """

    tree: ClassVar[dict] = {
        "root": [("A", 0.5), ("C", 0.3), ("t4", 0.2)],
        "A": [("t1", 0.3), ("B", 0.4)],
        "B": [("t2", 0.2), ("t3", 0.1)],
        "C": [],
    }
    gamma: ClassVar[dict] = {"t1": 0.25, "t2": 0.15, "t3": 0.05, "t4": 0.4}

    def expansion(self):
        def expand(node):
            if node in self.gamma:
                return node, ()
            return None, [(child, math.log(weight)) for child, weight in self.tree[node]]

        return "root", 0.0, expand

    def log_target_of(self, term):
        return math.log(self.gamma[term])


@pytest.mark.parametrize(("particles", "threshold"), [(1, 1.0), (2, 1.0), (3, 0.5), (4, 0.0)])
def test_a_dead_end_weighs_nothing_and_the_estimates_stay_unbiased(particles, threshold):
    """Over many small runs, the normalizer's estimate and each term's share of it, against the exact values."""
    toy = DeadEnds()
    rng = random.Random(particles * 11 + int(threshold * 10))
    runs = 20000
    estimates = {term: [] for term in DeadEnds.gamma} | {"Z": []}
    dead = 0
    for _ in range(runs):
        result = sequential_monte_carlo(toy, rng, particles, threshold=threshold)
        dead += sum(term is None for term in result.terms)
        normalizer = math.exp(result.log_normalizer) if result.log_normalizer > -math.inf else 0.0
        estimates["Z"].append(normalizer)
        total = sum(math.exp(value) for value in result.log_weights)
        for term in DeadEnds.gamma:
            share = (
                sum(math.exp(w) for t, w in zip(result.terms, result.log_weights, strict=True) if t == term) / total
                if total
                else 0.0
            )
            estimates[term].append(normalizer * share)
    assert dead > 0
    for key, values in estimates.items():
        exact = sum(DeadEnds.gamma.values()) if key == "Z" else DeadEnds.gamma[key]
        mean = sum(values) / runs
        spread = math.sqrt(sum((value - mean) ** 2 for value in values) / (runs - 1))
        assert abs(mean - exact) <= 4 * spread / math.sqrt(runs), (key, mean, exact)


def test_the_sampler_draws_from_the_weighted_particles_to_the_target():
    """The sampler: runs of particles, each run's final set resampled into its draws; the draws' bins follow the target."""
    query, algebra = digit_sums((0, 1, 3), 6)
    sampler = SMCSampler(algebra, EDGES, TARGET, random.Random(6), particles=500, draws_per_run=100)
    assert isinstance(sampler, Sampler)
    drawn = list(itertools.islice(sampler.sample(query), 3000))
    assert sampler.last_run is not None

    def bin_of(term):
        return next(i for i in range(len(TARGET)) if EDGES[i] <= algebra.fold(term) < EDGES[i + 1])

    bins = Counter(bin_of(term) for term in drawn)
    # The particles' target is the saddle search's: the bins carry the target times the truth over its counts.
    expected = Counter()
    for term, mass in exact_target_law(saddle_search(query, algebra, EDGES, TARGET), all_terms(query, algebra)).items():
        expected[bin_of(term)] += mass
    assert len(expected) >= 3
    for index in range(len(TARGET)):
        assert abs(bins[index] / len(drawn) - expected[index]) < 0.04, (index, bins, expected)
    assert sampler.at_least(query, 3**6)
    assert not sampler.at_least(query, 3**6 + 1)


def test_the_sampler_ends_its_stream_where_no_term_lies_in_a_bin_with_a_target():
    """A residual query whose completions all cost more than the target's bins: the stream ends, as a bounded sampler's must."""
    query, algebra = digit_sums((0, 1, 3), 4)
    term = max(all_terms(query, algebra), key=algebra.fold)
    residual = residual_query(query.solution_space, query.start, term, (0,))
    sampler = SMCSampler(algebra, (0.0, 1.0, 2.0), (0.5, 0.5), random.Random(5), particles=20)
    assert list(itertools.islice(sampler.sample(residual), 5)) == []


def test_the_sampler_runs_again_where_every_particle_of_a_run_died():
    """One particle often dies at a dead end the saddle point estimates as reachable; the stream goes on regardless."""
    rules = [
        ("S", "f", ("A", "B")),
        ("S", "h", ("E",)),
        ("A", "a0", ()),
        ("A", "a10", ()),
        ("B", "b0", ()),
        ("B", "b1", ()),
    ]
    rules += [("E", "e4", ()), ("E", "e5", ())]
    query = generator_query(space_of(rules), "S")
    algebra = priced({"f": 0, "h": 0, "a0": 0, "a10": 10, "b0": 0, "b1": 1, "e4": 4, "e5": 5})
    sampler = SMCSampler(algebra, (4.0, 6.0), (1.0,), random.Random(6), particles=1, draws_per_run=1)
    drawn = list(itertools.islice(sampler.sample(query), 200))
    assert len(drawn) == 200
    assert {term.root for term in drawn} == {"h"}


@pytest.mark.parametrize(("arguments", "message"), [({"particles": 0}, "particles"), ({"threshold": 1.5}, "threshold")])
def test_what_the_particles_refuse(arguments, message):
    """Fewer than one particle, a threshold outside [0, 1]."""
    search, _terms, _algebra = saddle_on((0, 1, 3), 4)
    options = {"particles": 10, "threshold": 0.5} | arguments
    with pytest.raises(ValueError, match=message):
        sequential_monte_carlo(search, random.Random(0), options["particles"], threshold=options["threshold"])


@pytest.mark.parametrize(("arguments", "message"), [({"particles": 0}, "particles"), ({"draws_per_run": 0}, "draws")])
def test_what_the_sampler_refuses(arguments, message):
    """Fewer than one particle, fewer than one draw per run."""
    _query, algebra = digit_sums((0, 1, 3), 4)
    options = {"particles": 10, "draws_per_run": 5} | arguments
    with pytest.raises(ValueError, match=message):
        SMCSampler(algebra, EDGES, TARGET, random.Random(0), **options)
