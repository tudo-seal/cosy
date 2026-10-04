"""Counting by rounding: a coarse cost counted exactly bounds a fine one's counts per bin, and rejection reaches its target.

The fine cost of these spaces is a sum of parts that do not divide by the unit, so every term carries a rounding error, and a
coarse value's terms spread over an interval of fine costs. The tests count the fine costs per bin exactly by the tree form
and hold them inside the bounds, at every bin; they walk the coarse table's random search exhaustively for the exact law of a
proposal and hold the accepted law to the target term by term; they hold the bins' masses inside the factor the bounds give;
and they pin what is refused.
"""

import bisect
import math
import random
from collections import Counter

import pytest

from cosy.search import generator_query, residual_query
from cosy.search.costs import AdditiveCostAlgebra, NonNegativeReals
from cosy.search.rounding import BinBounds, RoundedRejection, bin_count_bounds, rounded_rejection
from cosy.search.sampling import log_sum_exp
from tests.test_tilt import all_terms, space_of

PARTS = {"d0": 0, "d1": 7, "d2": 23, "d3": 41}
UNIT = 5


def fine_and_coarse(holes, parts=PARTS, unit=UNIT, head=0):
    """``S -> f(D x holes)``, ``D`` one clause per part; the fine cost the parts', the coarse one each part over the unit, rounded.

    Returns:
        tuple: The query, the fine algebra, the coarse algebra.
    """
    rules = [("S", "f", ("D",) * holes)] + [("D", name, ()) for name in parts]
    fine_costs = {"f": head} | parts
    coarse_costs = {name: round(cost / unit) for name, cost in fine_costs.items()}
    fine = AdditiveCostAlgebra(NonNegativeReals(), fine_costs.__getitem__)
    coarse = AdditiveCostAlgebra(NonNegativeReals(), coarse_costs.__getitem__)
    return generator_query(space_of(rules), "S"), fine, coarse


def exact_per_bin(terms, fine, edges):
    """The number of terms per bin of the fine cost, by enumeration."""
    counts = Counter()
    for term in terms:
        index = bisect.bisect_right(edges, fine.fold(term)) - 1
        if 0 <= index < len(edges) - 1:
            counts[index] += 1
    return [counts[index] for index in range(len(edges) - 1)]


def proposal_law(construction):
    """The law of the coarse table's first draw, every path of its random search walked."""
    root, _root_log_weight, expand = construction.weighted.expansion()
    law = Counter()
    stack = [(root, 0.0)]
    while stack:
        node, log_probability = stack.pop()
        inhabitant, children = expand(node)
        if inhabitant is not None:
            law[inhabitant] += math.exp(log_probability)
            continue
        total = log_sum_exp([weight for _, weight in children])
        stack.extend((child, log_probability + weight - total) for child, weight in children)
    return law


EDGES = (0.0, 40.0, 80.0, 120.0, 160.0, 200.0, 260.0)
TARGET = (0.1, 0.2, 0.25, 0.2, 0.15, 0.1)


@pytest.mark.parametrize("holes", [3, 6])
def test_the_bounds_hold_the_exact_counts_in_every_bin(holes):
    """Counted by enumeration, every bin's number of terms lies inside its bounds; the bounds are not trivial."""
    query, fine, coarse = fine_and_coarse(holes)
    terms = all_terms(query, fine)
    bounds = bin_count_bounds(query, fine, coarse, UNIT, EDGES, 100)
    assert isinstance(bounds, BinBounds)
    exact = exact_per_bin(terms, fine, EDGES)
    for index, count in enumerate(exact):
        assert bounds.lower[index] <= count <= bounds.upper[index], (
            index,
            bounds.lower[index],
            count,
            bounds.upper[index],
        )
    # Every term's fine cost lies in its coarse value's interval, the reason the bounds hold.
    for term in terms:
        low, high = bounds.interval_of(int(coarse.fold(term)))
        assert low - 1e-9 <= fine.fold(term) <= high + 1e-9
    # Not trivial: the bounds hold most terms on both sides, and the error range is what the parts' errors make it.
    assert sum(bounds.lower) >= 0.3 * sum(exact)
    assert max(bounds.upper) < len(terms)
    assert any(low < high for low, high in zip(bounds.lower, bounds.upper, strict=True))
    errors = [cost - UNIT * round(cost / UNIT) for cost in PARTS.values()]
    assert bounds.error_range == (holes * min(errors), holes * max(errors))


def test_where_every_interval_lies_inside_a_bin_the_bounds_are_the_counts():
    """With the parts multiples of the unit there is no error, and the bounds are the exact counts."""
    parts = {"d0": 0, "d1": 5, "d2": 20, "d3": 40}
    query, fine, coarse = fine_and_coarse(5, parts)
    bounds = bin_count_bounds(query, fine, coarse, UNIT, EDGES, 100)
    assert bounds.error_range == (0.0, 0.0)
    exact = exact_per_bin(all_terms(query, fine), fine, EDGES)
    assert list(bounds.lower) == exact == list(bounds.upper)


@pytest.mark.parametrize("given", ["exact", "geometric mean"])
def test_the_accepted_law_is_the_target_spread_by_the_counts_term_by_term(given):
    """Proposal walked exactly, times acceptance: every term weighs target(b) / N-hat(b), normalized, to 1e-12."""
    query, fine, coarse = fine_and_coarse(5)
    terms = all_terms(query, fine)
    exact = exact_per_bin(terms, fine, EDGES)
    log_counts = [math.log(count) if count else -math.inf for count in exact] if given == "exact" else None
    construction = rounded_rejection(query, fine, coarse, UNIT, EDGES, TARGET, 100, log_counts=log_counts)
    assert isinstance(construction, RoundedRejection)
    proposal = proposal_law(construction)
    accepted = Counter({term: mass * math.exp(construction.log_acceptance(term)) for term, mass in proposal.items()})
    total = sum(accepted.values())
    expected = Counter()
    for term in terms:
        index = construction.bin_of(fine.fold(term))
        if index is not None and construction.log_rho[index] > -math.inf:
            expected[term] = math.exp(construction.log_rho[index])
    norm = sum(expected.values())
    assert {term for term, mass in accepted.items() if mass > 0} == set(expected)
    for term in expected:
        assert math.isclose(accepted[term] / total, expected[term] / norm, rel_tol=1e-9)
    if given == "exact":
        by_bin = Counter()
        for term, mass in accepted.items():
            by_bin[construction.bin_of(fine.fold(term))] += mass / total
        for index, mass in enumerate(TARGET):
            assert math.isclose(by_bin[index], mass / sum(TARGET), rel_tol=1e-9)
    # The acceptance is one wherever a coarse value's interval lies inside one bin.
    for term in terms:
        coarse_value = int(coarse.fold(term))
        if construction.bounds.inside(coarse_value) is not None and construction.log_acceptance(term) > -math.inf:
            assert construction.log_acceptance(term) == 0.0


def test_the_bins_carry_the_target_times_the_count_over_its_estimate_within_the_bounds():
    """The accepted law's mass per bin, computed exactly: target(b) N(b) / N-hat(b), normalized; N / N-hat inside the bounds."""
    query, fine, coarse = fine_and_coarse(6)
    terms = all_terms(query, fine)
    exact = exact_per_bin(terms, fine, EDGES)
    construction = rounded_rejection(query, fine, coarse, UNIT, EDGES, TARGET, 100)
    proposal = proposal_law(construction)
    accepted = Counter()
    for term, mass in proposal.items():
        accepted[construction.bin_of(fine.fold(term))] += mass * math.exp(construction.log_acceptance(term))
    total = sum(accepted.values())
    bounds = construction.bounds
    n_hat = [math.exp(math.log(TARGET[i]) - construction.log_rho[i]) for i in range(len(TARGET))]
    expected = [TARGET[i] * exact[i] / n_hat[i] for i in range(len(TARGET))]
    checked = 0
    for index in range(len(TARGET)):
        assert math.isclose(accepted[index] / total, expected[index] / sum(expected), rel_tol=1e-9)
        if exact[index]:
            assert (
                bounds.lower[index] / n_hat[index] - 1e-9
                <= exact[index] / n_hat[index]
                <= bounds.upper[index] / n_hat[index] + 1e-9
            )
            checked += 1
    assert checked >= 4
    assert any(low < high for low, high in zip(bounds.lower, bounds.upper, strict=True))


def test_on_a_residual_query_the_partial_terms_own_error_counts():
    """A head symbol that rounds with an error of its own, and a partial term: the bounds hold and the law is exact there too."""
    query, fine, coarse = fine_and_coarse(4, head=7)
    terms = all_terms(query, fine)
    term = max(terms, key=lambda t: (fine.fold(t), str(t)))
    residual = residual_query(query.solution_space, query.start, term, (1,))
    completions = all_terms(residual, fine)
    assert len(completions) == len(PARTS)
    edges = (60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 140.0)
    bounds = bin_count_bounds(residual, fine, coarse, UNIT, edges, 100)
    exact = exact_per_bin(completions, fine, edges)
    assert all(low <= count <= high for low, count, high in zip(bounds.lower, exact, bounds.upper, strict=True))
    # The head's own error (7 - 5 = 2) and the three fixed parts' (41 - 40 = 1 each) in every completion, the open part's on top.
    errors = [cost - UNIT * round(cost / UNIT) for cost in PARTS.values()]
    assert bounds.error_range == (2 + 3 + min(errors), 2 + 3 + max(errors))
    construction = rounded_rejection(residual, fine, coarse, UNIT, edges, (1,) * 6, 100)
    accepted = Counter({t: m * math.exp(construction.log_acceptance(t)) for t, m in proposal_law(construction).items()})
    total = sum(accepted.values())
    expected = {
        t: math.exp(construction.log_rho[construction.bin_of(fine.fold(t))])
        for t in completions
        if construction.bin_of(fine.fold(t)) is not None
    }
    norm = sum(expected.values())
    for t, mass in expected.items():
        assert math.isclose(accepted[t] / total, mass / norm, rel_tol=1e-9)


def test_the_cap_must_reach_the_coarse_costs_a_bin_can_hold_and_no_further():
    """The cap the bins need is accepted and one below it refused; above the query's dearest term nothing more is needed."""
    query, fine, coarse = fine_and_coarse(3)
    least = 3 * min(cost - UNIT * round(cost / UNIT) for cost in PARTS.values())
    edges = (0.0, 40.0, 80.0)
    needed = max(k for k in range(1000) if UNIT * k + least < edges[-1])
    bin_count_bounds(query, fine, coarse, UNIT, edges, needed)
    with pytest.raises(ValueError, match="does not reach the coarse costs"):
        bin_count_bounds(query, fine, coarse, UNIT, edges, needed - 1)
    dearest = 3 * max(round(cost / UNIT) for cost in PARTS.values())
    far = (0.0, 10_000.0)
    bin_count_bounds(query, fine, coarse, UNIT, far, dearest)
    with pytest.raises(ValueError, match="does not reach the coarse costs"):
        bin_count_bounds(query, fine, coarse, UNIT, far, dearest - 1)


def test_counts_far_beyond_floating_point_are_drawn_from():
    """Ten nested pairs of a part of three: 3^1024 terms, about 10^488, and the sampler is built and draws."""
    rules = [("N0", "d0", ()), ("N0", "d1", ()), ("N0", "d2", ())]
    rules += [(f"N{level + 1}", f"p{level}", (f"N{level}", f"N{level}")) for level in range(10)]
    fine_costs = {"d0": 0, "d1": 1, "d2": 3} | {f"p{level}": 0 for level in range(10)}
    coarse_costs = {name: round(cost / 2) for name, cost in fine_costs.items()}
    fine = AdditiveCostAlgebra(NonNegativeReals(), fine_costs.__getitem__)
    coarse = AdditiveCostAlgebra(NonNegativeReals(), coarse_costs.__getitem__)
    query = generator_query(space_of(rules), "N10")
    edges = (0.0, 1000.0, 2000.0, 3100.0)
    construction = rounded_rejection(query, fine, coarse, 2, edges, (0.0, 0.5, 0.5), 2048)
    assert sum(construction.bounds.upper).bit_length() > 1500
    drawn = list(construction.stream(random.Random(1), 50))
    assert len(drawn) >= 10
    assert all(1000 <= fine.fold(term) < 3100 for term in drawn)


def test_the_stream_draws_the_accepted_law_and_repeats_nothing():
    """Many independent draws, their bins against the exact accepted law; a stream's terms distinct."""
    query, fine, coarse = fine_and_coarse(6)
    construction = rounded_rejection(query, fine, coarse, UNIT, EDGES, TARGET, 100)
    terms = all_terms(query, fine)
    expected = Counter()
    for term in terms:
        index = construction.bin_of(fine.fold(term))
        if index is not None and construction.log_rho[index] > -math.inf:
            expected[index] += math.exp(construction.log_rho[index])
    norm = sum(expected.values())
    seen = Counter()
    draws = 0
    for seed in range(40):
        drawn = list(construction.stream(random.Random(seed), 400))
        assert len(set(drawn)) == len(drawn)
        for term in drawn[:50]:
            seen[construction.bin_of(fine.fold(term))] += 1
            draws += 1
    for index, mass in expected.items():
        assert abs(seen[index] / draws - mass / norm) < 0.03, (index, seen[index] / draws, mass / norm)


def test_a_fractional_fine_cost_is_refused():
    """A part of 7.5: its interval and its fold could round apart, so the construction refuses fractions by name."""
    query, fine, coarse = fine_and_coarse(3, {"d0": 0, "d1": 7.5, "d2": 23})
    with pytest.raises(ValueError, match="whole-number costs"):
        bin_count_bounds(query, fine, coarse, UNIT, EDGES, 100)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"unit": 0}, "unit"),
        ({"unit": -5}, "unit"),
        ({"unit": 2.5}, "unit"),
        ({"cost_cap": 10}, "cost cap"),
        ({"log_counts": [100.0] * 6}, "outside its bounds"),
        ({"log_counts": [0.0] * 6}, "outside its bounds"),
        ({"log_counts": [-math.inf] * 6}, "is zero"),
        ({"target": (0.0,) * 6}, "target needs"),
        ({"target": (0.5, -0.1, 0.2, 0.2, 0.1, 0.1)}, "target needs"),
    ],
)
def test_what_is_refused(arguments, message):
    """A unit not a positive whole number, a cap below what the bins hold, a count outside its bounds or zero where terms may
    lie, a target of nothing or with a negative mass."""
    query, fine, coarse = fine_and_coarse(4)
    options = {"unit": UNIT, "cost_cap": 100, "log_counts": None, "target": TARGET} | arguments
    with pytest.raises(ValueError, match=message):
        rounded_rejection(
            query,
            fine,
            coarse,
            options["unit"],
            EDGES,
            options["target"],
            options["cost_cap"],
            log_counts=options["log_counts"],
        )
