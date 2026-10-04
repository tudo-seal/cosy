"""The Markov chain on terms: Metropolis-Hastings to a target on cost bins, a subtree regrown through its residual query.

The chain's stationary law is ``p(t)`` proportional to ``target(b) / N(b)`` for the bin ``b`` of ``t``'s cost, ``N(b)`` the
number of terms the caller says the bin holds. A move regrows the subtree at one position by the tilted search of the
residual query there, so the regrown term comes with a probability whose normalizer is the residual's tilted mass, the same
for the move back. These tests enumerate small finite spaces completely: the kernel's exact transition probabilities, built
from the chain's own acceptance and the exact law of each regrowth, must leave ``p`` stationary and satisfy detailed balance;
the chain's regrowth must draw the completions of its residual in tilted proportion; a long run must visit the terms in
proportion to ``p``. What the chain refuses is pinned beside it.
"""

import bisect
import itertools
import math
import random
from collections import Counter

import pytest

from cosy.core import Constructor, SpecificationBuilder, Synthesizer
from cosy.core.solution_space import ConstantArgument
from cosy.core.tree import Tree
from cosy.core.types import DataGroup
from cosy.search import generator_query, residual_query
from cosy.search.markov import (
    ChainStep,
    MetropolisChain,
    VisitCounts,
    WangLandau,
    counts_from_visits,
    metropolis_chain,
    wang_landau,
)
from cosy.search.samplers import MarkovChainSampler, Sampler
from cosy.search.sampling import log_sum_exp
from tests.search_fixtures import PRICED, priced_space
from tests.test_tilt import FRACTIONAL, all_terms, digit_sums, priced, space_of

HOLE = Tree("<hole>")


def nested():
    """A small space with depth: ``S -> f(A, B) | g(A)``, ``A -> a1 | a2 | h(B)``, ``B -> b1 | b2 | b3``; 20 terms.

    Returns:
        tuple: The generator query and its algebra.
    """
    rules = [
        ("S", "f", ("A", "B")),
        ("S", "g", ("A",)),
        ("A", "a1", ()),
        ("A", "a2", ()),
        ("A", "h", ("B",)),
        ("B", "b1", ()),
        ("B", "b2", ()),
        ("B", "b3", ()),
    ]
    costs = {"f": 1.0, "g": 0.5, "a1": 0.0, "a2": 2.0, "h": 1.5, "b1": 0.0, "b2": 1.0, "b3": 3.0}
    return generator_query(space_of(rules), "S"), priced(costs)


LIT_TOP = Constructor("LitTop")
LIT_A = Constructor("LitA")


def lit_leaf(d: int) -> str:
    """A leaf of ``LitA`` that fixes a digit, a constant argument."""
    return f"leaf{d}"


def lit_plain() -> str:
    """A leaf of ``LitA`` without a constant."""
    return "plain"


def lit_pair(left: str, right: str) -> str:
    """A ``LitTop`` of two ``LitA``."""
    return f"({left}, {right})"


def lit_wrap(k: int, inner: str) -> str:
    """A ``LitTop`` of a constant before a ``LitA``."""
    return f"{k}:{inner}"


def literal_space():
    """A small space with constants at two depths: ``LitTop -> lit_pair(LitA, LitA) | lit_wrap(k, LitA)``; 24 terms.

    Returns:
        tuple: The generator query and its algebra.
    """
    specs = {
        lit_leaf: SpecificationBuilder().parameter("d", DataGroup("digit", (0, 1, 2))).suffix(LIT_A),
        lit_plain: SpecificationBuilder().suffix(LIT_A),
        lit_pair: SpecificationBuilder().argument("left", LIT_A).argument("right", LIT_A).suffix(LIT_TOP),
        lit_wrap: SpecificationBuilder()
        .parameter("k", DataGroup("k", (0, 1)))
        .argument("inner", LIT_A)
        .suffix(LIT_TOP),
    }
    return generator_query(Synthesizer(specs).construct_solution_space(LIT_TOP), LIT_TOP), FRACTIONAL


def constant_positions(space):
    """The indices of the constant arguments per terminal, read off the program's clauses (independently of the chain)."""
    found = {}
    for nonterminal in space.nonterminals():
        for rule in space.get(nonterminal) or ():
            for index, argument in enumerate(rule.arguments):
                if isinstance(argument, ConstantArgument):
                    found.setdefault(rule.terminal, set()).add(index)
    return found


def eligible_positions(term, space, base=()):
    """The positions at or below ``base`` that are not constant arguments of the clause above them."""
    constants = constant_positions(space)
    out = []
    for position in term.positions():
        if position[: len(base)] != base:
            continue
        if len(position) > len(base) and position[-1] in constants.get(term.subtree_at(position[:-1]).root, ()):
            continue
        out.append(position)
    return out


def switching():
    """``S -> f(A1, B1) | f(A2, B2)``: the clause above the second position switches with the first; six terms, one derivation each.

    The residual at the second position of ``f(a, x)`` holds ``x``, ``z`` and ``y`` (the last by the other clause), while the
    derivation has ``B1`` there, whose terms are ``x`` and ``z``.

    Returns:
        tuple: The generator query and its algebra.
    """
    rules = [
        ("S", "f", ("A1", "B1")),
        ("S", "f", ("A2", "B2")),
        ("A1", "a", ()),
        ("A1", "c", ()),
        ("A2", "a", ()),
        ("A2", "b", ()),
        ("B1", "x", ()),
        ("B1", "z", ()),
        ("B2", "y", ()),
    ]
    costs = {"f": 1.0, "a": 0.0, "b": 2.0, "c": 1.0, "x": 0.0, "y": 1.5, "z": 3.0}
    return generator_query(space_of(rules), "S"), priced(costs)


def bins_of(costs, edges):
    """The bin of each cost; None outside the edges."""
    out = []
    for cost in costs:
        index = bisect.bisect_right(edges, cost) - 1
        out.append(index if 0 <= index < len(edges) - 1 else None)
    return out


def exact_log_counts(terms, algebra, edges):
    """The log of the number of terms per bin, ``-inf`` for an empty bin."""
    counts = Counter(index for index in bins_of([algebra.fold(term) for term in terms], edges) if index is not None)
    return [math.log(counts[index]) if counts[index] else -math.inf for index in range(len(edges) - 1)]


def contexts(terms, space, base=()):
    """Every term grouped by its context at each position a move may choose: the residual's completions, enumerated."""
    groups = {}
    for term in terms:
        for position in eligible_positions(term, space, base):
            groups.setdefault((position, term.replace_subtree_at(position, HOLE)), []).append(term)
    return groups


LANGUAGES: dict[int, tuple[object, dict]] = {}


def language(space, nonterminal, algebra):
    """Every term of a non-terminal, enumerated once per program and non-terminal (the program kept, so its id stays its own)."""
    _space, known = LANGUAGES.setdefault(id(space), (space, {}))
    if nonterminal not in known:
        known[nonterminal] = all_terms(generator_query(space, nonterminal), algebra)
    return known[nonterminal]


def local_completions(chain, term, position, algebra):
    """The terms a local move at a position proposes: the subtree replaced by every term of the derivation's non-terminal."""
    nonterminal = chain.derivation_of(term)[position]
    if nonterminal is None:
        return [term]
    space = chain.query.solution_space
    return [term.replace_subtree_at(position, other) for other in language(space, nonterminal, algebra)]


def exact_kernel(chain, terms, algebra, base=()):
    """The chain's transition probabilities, from its own acceptance and the exact law of every regrowth.

    Args:
        chain (MetropolisChain): The chain.
        terms (list): Every completion of the chain's query.
        algebra: The algebra.
        base (tuple): The query's hole.

    Returns:
        dict: ``kernel[i][j]``, the probability of a step from ``terms[i]`` to ``terms[j]``.
    """
    index = {term: position for position, term in enumerate(terms)}
    space = chain.query.solution_space
    groups = contexts(terms, space, base)
    theta = chain.table.theta
    kernel = {i: Counter() for i in range(len(terms))}
    for i, term in enumerate(terms):
        if chain.log_weight(term) == -math.inf:
            continue
        local = eligible_positions(term, space, base)
        assert chain.positions(term) == len(local)
        assert set(chain.derivation_of(term)) == set(local)
        moves = [(chain.root_share, base, True)] + [((1 - chain.root_share) / len(local), p, False) for p in local]
        for share, position, root_move in moves:
            if share == 0:
                continue
            if root_move:
                completions = groups[position, term.replace_subtree_at(position, HOLE)]
            else:
                completions = local_completions(chain, term, position, algebra)
            log_q = [-theta * algebra.fold(other) for other in completions]
            total = log_sum_exp(log_q)
            for other, value in zip(completions, log_q, strict=True):
                probability = share * math.exp(value - total)
                accept = math.exp(chain.log_acceptance(term, other, root_move=root_move))
                kernel[i][index[other]] += probability * accept
                kernel[i][i] += probability * (1 - accept)
    return kernel


def stationary_check(chain, terms, algebra, base=()):
    """Assert that ``p`` is stationary under the exact kernel and that detailed balance holds; return the kernel."""
    kernel = exact_kernel(chain, terms, algebra, base)
    log_p = [chain.log_weight(term) for term in terms]
    total = log_sum_exp(log_p)
    p = [math.exp(value - total) if value > -math.inf else 0.0 for value in log_p]
    for i in kernel:
        assert math.isclose(sum(kernel[i].values()), 1.0 if p[i] > 0 else 0.0, abs_tol=1e-12)
    flow = Counter()
    for i, row in kernel.items():
        for j, probability in row.items():
            flow[j] += p[i] * probability
            # Detailed balance, pair by pair.
            assert math.isclose(p[i] * probability, p[j] * kernel[j][i], rel_tol=1e-9, abs_tol=1e-15), (
                terms[i],
                terms[j],
            )
    for j, mass in enumerate(p):
        assert math.isclose(flow[j], mass, rel_tol=1e-9, abs_tol=1e-15), terms[j]
    return kernel, p


SPACES = [
    ("nested", nested, (0.0, 1.6, 3.1, 4.6, 7.0), (0.3, 0.2, 0.4, 0.1)),
    ("digit sums", lambda: digit_sums((0, 1, 3), 4), (0.0, 2.5, 5.5, 8.5, 13.0), (0.2, 0.3, 0.4, 0.1)),
    ("constants", literal_space, (1.0, 2.6, 3.6, 4.6, 7.0), (0.25, 0.25, 0.3, 0.2)),
    ("switching", switching, (0.0, 1.6, 2.6, 6.0), (0.3, 0.3, 0.4)),
]


@pytest.mark.parametrize(("name", "build", "edges", "target"), SPACES)
@pytest.mark.parametrize("theta", [0.0, 0.7, -0.4])
@pytest.mark.parametrize("root_share", [0.0, 0.5, 1.0])
def test_the_exact_kernel_leaves_the_target_stationary_and_balances_every_pair(
    name, build, edges, target, theta, root_share
):
    """Every position, every completion, the chain's own acceptance: ``p`` is stationary and every pair is balanced."""
    query, algebra = build()
    terms = all_terms(query, algebra)
    chain = metropolis_chain(
        query, algebra, theta, edges, target, exact_log_counts(terms, algebra, edges), root_share=root_share
    )
    kernel, p = stationary_check(chain, terms, algebra)
    # Not trivially: the kernel moves, and some moves are refused.
    moving = sum(p[i] * (1 - kernel[i][i]) for i in kernel)
    assert moving > 0.1, (name, moving)
    accepts = [
        chain.log_acceptance(a, b, root_move=True) for a in terms[:10] for b in terms[:10] if p[terms.index(a)] > 0
    ]
    assert any(value < 0 for value in accepts)
    assert any(value == 0 for value in accepts)


@pytest.mark.parametrize(("name", "build", "edges", "target"), SPACES)
def test_with_exact_counts_the_bins_carry_the_target_and_a_bin_its_terms_alike(name, build, edges, target):  # noqa: ARG001
    """``p`` itself: with the exact counts the bins carry the target, normalized, and the terms of a bin weigh alike."""
    query, algebra = build()
    terms = all_terms(query, algebra)
    chain = metropolis_chain(query, algebra, 0.3, edges, target, exact_log_counts(terms, algebra, edges))
    log_p = [chain.log_weight(term) for term in terms]
    total = log_sum_exp(log_p)
    by_bin = Counter()
    members = {}
    for value, index in zip(log_p, bins_of([algebra.fold(t) for t in terms], edges), strict=True):
        if value > -math.inf:
            by_bin[index] += math.exp(value - total)
            members.setdefault(index, set()).add(round(value, 12))
    reached = [mass if by_bin[index] > 0 else 0 for index, mass in enumerate(target)]
    for index, mass in enumerate(reached):
        assert math.isclose(by_bin[index], mass / sum(reached), rel_tol=1e-9, abs_tol=1e-15)
    assert all(len(values) == 1 for values in members.values())
    assert len(by_bin) >= 3


def test_any_counts_leave_their_own_law_stationary():
    """Counts that are not the truth: the chain is still balanced, its bins then carry target times truth over counts."""
    query, algebra = nested()
    edges, target = (0.0, 1.6, 3.1, 4.6, 7.0), (0.3, 0.2, 0.4, 0.1)
    terms = all_terms(query, algebra)
    wrong = [
        value + shift
        for value, shift in zip(exact_log_counts(terms, algebra, edges), (0.4, -0.3, 0.0, 1.1), strict=True)
    ]
    chain = metropolis_chain(query, algebra, 0.5, edges, target, wrong)
    _kernel, p = stationary_check(chain, terms, algebra)
    by_bin = Counter()
    for mass, index in zip(p, bins_of([algebra.fold(t) for t in terms], edges), strict=True):
        by_bin[index] += mass
    truth = exact_log_counts(terms, algebra, edges)
    expected = [target[i] * math.exp(truth[i] - wrong[i]) for i in range(4)]
    for index in range(4):
        assert math.isclose(by_bin[index], expected[index] / sum(expected), rel_tol=1e-9)


@pytest.mark.parametrize(("name", "build", "edges", "target"), SPACES)
def test_a_parsed_derivation_puts_at_every_position_a_non_terminal_that_derives_its_subtree_and_fits_its_context(
    name, build, edges, target
):
    """Parsed by a fresh chain: each position's non-terminal derives the subtree there, and every term of it completes the query."""
    query, algebra = build()
    terms = all_terms(query, algebra)
    for term in terms:
        chain = metropolis_chain(query, algebra, 0.0, edges, target, exact_log_counts(terms, algebra, edges))
        derivation = chain.derivation_of(term)
        assert set(derivation) == set(eligible_positions(term, query.solution_space)), name
        for position, nonterminal in derivation.items():
            others = language(query.solution_space, nonterminal, algebra)
            assert term.subtree_at(position) in others
            assert all(term.replace_subtree_at(position, other) in terms for other in others)


def test_a_drawn_term_brings_the_derivation_a_parse_finds():
    """The derivation a draw records is the one a fresh chain parses off the term."""
    query = generator_query(priced_space(), PRICED)
    terms = all_terms(query, FRACTIONAL)
    edges = (1.0, 10.0)
    chain = metropolis_chain(query, FRACTIONAL, 0.3, edges, (1.0,), exact_log_counts(terms, FRACTIONAL, edges))
    rng = random.Random(3)
    for _ in range(40):
        drawn = chain.initial(rng)
        fresh = metropolis_chain(query, FRACTIONAL, 0.3, edges, (1.0,), exact_log_counts(terms, FRACTIONAL, edges))
        assert chain.derivation_of(drawn) == fresh.derivation_of(drawn)


def test_a_local_move_proposes_from_the_derivations_non_terminal_not_the_whole_residual():
    """Where the clause above a position can switch, the residual holds more than the derivation's non-terminal offers."""
    query, algebra = switching()
    terms = all_terms(query, algebra)
    chain = metropolis_chain(query, algebra, 0.0, (0.0, 6.0), (1.0,), exact_log_counts(terms, algebra, (0.0, 6.0)))
    term = next(t for t in terms if t.children[0].root == "a" and t.children[1].root == "x")
    residual = contexts(terms, query.solution_space)[(1,), term.replace_subtree_at((1,), HOLE)]
    assert {other.children[1].root for other in residual} == {"x", "z", "y"}
    assert {other.children[1].root for other in local_completions(chain, term, (1,), algebra)} == {"x", "z"}
    drawn = {chain.regrow(term, (1,), random.Random(seed)).children[1].root for seed in range(60)}
    assert drawn == {"x", "z"}


def test_a_regrowth_draws_the_completions_of_its_residual_in_tilted_proportion():
    """At a position with a hundred or so proposals, many regrowths against the exact law: e^(-theta c) over the tilted mass."""
    query = generator_query(priced_space(), PRICED)
    terms = all_terms(query, FRACTIONAL)
    edges = (0.0, 3.0, 5.0, 10.0)
    chain = metropolis_chain(query, FRACTIONAL, 0.6, edges, (0.3, 0.4, 0.3), exact_log_counts(terms, FRACTIONAL, edges))
    term, position, completions = max(
        (
            (term, position, local_completions(chain, term, position, FRACTIONAL))
            for term in terms[:200]
            for position in chain.derivation_of(term)
        ),
        key=lambda item: (50 <= len(item[2]) <= 200, -abs(len(item[2]) - 125), str(item[0]), item[1]),
    )
    assert 50 <= len(completions) <= 200
    log_q = [-0.6 * FRACTIONAL.fold(other) for other in completions]
    total = log_sum_exp(log_q)
    # The tilted mass a regrowth there can propose is exactly the sum over its enumerated completions.
    assert math.isclose(chain.regrow_log_mass(term, position), total, rel_tol=1e-12)
    rng = random.Random(4)
    draws = 30000
    seen = Counter(chain.regrow(term, position, rng) for _ in range(draws))
    assert set(seen) <= set(completions)
    distance = 0.5 * sum(
        abs(seen[other] / draws - math.exp(value - total)) for other, value in zip(completions, log_q, strict=True)
    )
    assert distance < 0.05, (len(completions), distance)


def test_a_constant_is_not_a_position_and_changes_with_its_clause():
    """The chain does not choose a constant's position; regrowing the clause above it changes the constant."""
    query, algebra = literal_space()
    terms = all_terms(query, algebra)
    chain = metropolis_chain(query, algebra, 0.0, (0.0, 10.0), (1.0,), exact_log_counts(terms, algebra, (0.0, 10.0)))
    term = next(t for t in terms if t.root is lit_wrap and t.children[1].root is lit_leaf)
    assert term.size == 4
    assert chain.positions(term) == 2  # the root and the leaf's clause, not the two constants
    drawn = {chain.regrow(term, (1,), random.Random(seed)) for seed in range(80)}
    assert {t.children[1].children[0].root for t in drawn if t.children[1].root is lit_leaf} == {0, 1, 2}
    assert {t.children[0].root for t in drawn} == {term.children[0].root}
    rng = random.Random(9)
    positions = Counter(chain.step(term, rng).position for _ in range(600))
    assert set(positions) == {(), (1,)}


def test_a_step_chooses_its_position_uniformly_and_proposes_what_the_exact_kernel_says():
    """From one state at root share 0: positions uniform over the derivation's, (position, proposal) at its exact frequency."""
    query, algebra = nested()
    terms = all_terms(query, algebra)
    edges, target = (0.0, 1.6, 3.1, 4.6, 7.0), (0.3, 0.2, 0.4, 0.1)
    chain = metropolis_chain(
        query, algebra, 0.5, edges, target, exact_log_counts(terms, algebra, edges), root_share=0.0
    )
    term = max((t for t in terms if chain.log_weight(t) > -math.inf), key=lambda t: (chain.positions(t), str(t)))
    positions = list(chain.derivation_of(term))
    assert len(positions) >= 3
    expected = Counter()
    for position in positions:
        completions = local_completions(chain, term, position, algebra)
        log_q = [-0.5 * algebra.fold(other) for other in completions]
        total = log_sum_exp(log_q)
        for other, value in zip(completions, log_q, strict=True):
            expected[position, other] += math.exp(value - total) / len(positions)
    rng = random.Random(10)
    draws = 20000
    seen = Counter()
    for _ in range(draws):
        step = chain.step(term, rng)
        seen[step.position, step.proposal] += 1
    by_position = Counter()
    for (position, _other), count in seen.items():
        by_position[position] += count
    assert max(abs(by_position[p] / draws - 1 / len(positions)) for p in positions) < 0.015, by_position
    assert set(seen) <= set(expected)
    assert 0.5 * sum(abs(seen[key] / draws - mass) for key, mass in expected.items()) < 0.03


def test_at_root_share_zero_the_chain_still_reaches_every_root_symbol():
    """Without moves of the whole term, the regrowth at the root's own position reaches both clauses of the start."""
    query, algebra = nested()
    terms = all_terms(query, algebra)
    edges, target = (0.0, 1.6, 3.1, 4.6, 7.0), (0.3, 0.2, 0.4, 0.1)
    chain = metropolis_chain(
        query, algebra, 0.0, edges, target, exact_log_counts(terms, algebra, edges), root_share=0.0
    )
    roots = {step.term.root for step in itertools.islice(chain.run(random.Random(12)), 3000)}
    assert roots == {"f", "g"}


def test_a_long_run_visits_the_terms_in_proportion_to_the_target():
    """Forty thousand steps on a space of 1 750 terms with literals and depth: the visits per bin against the target."""
    query = generator_query(priced_space(), PRICED)
    terms = all_terms(query, FRACTIONAL)
    edges = (1.0, 3.0, 4.0, 5.0, 10.0)
    target = (0.1, 0.3, 0.4, 0.2)
    chain = metropolis_chain(query, FRACTIONAL, 0.4, edges, target, exact_log_counts(terms, FRACTIONAL, edges))
    steps = 40000
    visits = Counter()
    accepted = 0
    for count, step in enumerate(chain.run(random.Random(11))):
        if count >= steps:
            break
        assert isinstance(step, ChainStep)
        accepted += step.accepted
        visits[bisect.bisect_right(edges, FRACTIONAL.fold(step.term)) - 1] += 1
    shares = [visits[index] / steps for index in range(4)]
    assert max(abs(share - mass) for share, mass in zip(shares, target, strict=True)) < 0.03, shares
    assert 0.05 < accepted / steps < 0.95


def test_a_partial_term_query_regrows_only_below_its_hole():
    """On a residual query the chain's every state agrees with the query's term outside its hole, and the kernel balances there."""
    query, algebra = nested()
    terms = all_terms(query, algebra)
    term = next(t for t in terms if t.root == "f")
    partial = residual_query(query.solution_space, query.start, term, (0,))
    completions = all_terms(partial, algebra)
    assert len(completions) >= 4
    edges, target = (0.0, 2.0, 4.0, 7.0), (0.5, 0.3, 0.2)
    chain = metropolis_chain(partial, algebra, 0.2, edges, target, exact_log_counts(completions, algebra, edges))
    stationary_check(chain, completions, algebra, base=(0,))
    outside = term.replace_subtree_at((0,), HOLE)
    for count, step in enumerate(chain.run(random.Random(2))):
        if count >= 300:
            break
        assert step.term.replace_subtree_at((0,), HOLE) == outside
        assert step.position[:1] == (0,)
    stranger = next(t for t in terms if t.root == "f" and t.replace_subtree_at((0,), HOLE) != outside)
    with pytest.raises(ValueError, match="not a completion of the query"):
        next(chain.run(random.Random(2), start=stranger))


def test_a_proposal_outside_the_target_is_refused():
    """A term in a bin without a target, or outside every bin, is never entered."""
    query, algebra = nested()
    terms = all_terms(query, algebra)
    edges, target = (0.0, 1.6, 3.1, 4.6, 7.0), (0.5, 0.0, 0.5, 0.0)
    chain = metropolis_chain(query, algebra, 0.0, edges, target, exact_log_counts(terms, algebra, edges))
    inside = [t for t in terms if chain.log_weight(t) > -math.inf]
    outside = [t for t in terms if chain.log_weight(t) == -math.inf]
    assert inside
    assert outside
    assert all(chain.log_acceptance(inside[0], other, root_move=True) == -math.inf for other in outside)
    for count, step in enumerate(chain.run(random.Random(5))):
        if count >= 500:
            break
        assert chain.log_weight(step.term) > -math.inf


def test_the_chain_starts_on_a_term_inside_its_target():
    """The first state is drawn until it lies in a bin with a target; a start given outside is refused by name."""
    query, algebra = nested()
    terms = all_terms(query, algebra)
    edges, target = (0.0, 1.6, 3.1, 4.6, 7.0), (0.0, 0.0, 0.0, 1.0)
    chain = metropolis_chain(query, algebra, 0.0, edges, target, exact_log_counts(terms, algebra, edges))
    first = next(chain.run(random.Random(3)))
    assert chain.log_weight(first.term) > -math.inf
    outside = next(t for t in terms if chain.log_weight(t) == -math.inf)
    with pytest.raises(ValueError, match="the start lies outside the chain's target"):
        next(chain.run(random.Random(3), start=outside))
    with pytest.raises(ValueError, match="not a completion of the query"):
        next(chain.run(random.Random(3), start=Tree("f", (Tree("b3"), Tree("a2")))))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"root_share": 1.5}, "root share"),
        ({"root_share": -0.1}, "root share"),
        ({"log_counts": [0.0, 0.0]}, "one log count per bin"),
        ({"target": (0.0, 0.0, 0.0, 0.0)}, "target needs"),
        ({"log_counts": [-math.inf] * 4}, "no bin with a target has a term"),
        ({"theta": math.inf}, "theta"),
    ],
)
def test_what_the_chain_refuses(change, message):
    """A root share outside [0, 1], counts not one per bin, a target of nothing, no bin with both, a theta not finite."""
    query, algebra = nested()
    terms = all_terms(query, algebra)
    edges = (0.0, 1.6, 3.1, 4.6, 7.0)
    arguments = {
        "theta": 0.0,
        "target": (0.3, 0.2, 0.4, 0.1),
        "log_counts": exact_log_counts(terms, algebra, edges),
        "root_share": 0.5,
    }
    arguments |= change
    with pytest.raises(ValueError, match=message):
        metropolis_chain(
            query,
            algebra,
            arguments["theta"],
            edges,
            arguments["target"],
            arguments["log_counts"],
            root_share=arguments["root_share"],
        )


def test_the_chain_is_a_dataclass_of_what_it_reads():
    """The construction keeps what a step reads: the tilt's table, the bins, the log weight per bin, the root share."""
    query, algebra = nested()
    terms = all_terms(query, algebra)
    edges = (0.0, 1.6, 3.1, 4.6, 7.0)
    chain = metropolis_chain(
        query, algebra, 0.25, edges, (0.3, 0.2, 0.4, 0.1), exact_log_counts(terms, algebra, edges), root_share=0.3
    )
    assert isinstance(chain, MetropolisChain)
    assert chain.table.theta == 0.25
    assert chain.edges == edges
    assert chain.root_share == 0.3
    assert len(chain.log_rho) == 4


# ---------------------------------------------------------------------------
# Wang-Landau: the counts per bin learned along the chain, with the target as the histogram to flatten against
# ---------------------------------------------------------------------------

WL_EDGES = (1.0, 3.0, 4.0, 5.0, 6.0, 10.0)
WL_TARGET = (0.1, 0.3, 0.3, 0.2, 0.1)


def priced_chain(log_counts=None, theta=0.4):
    """The chain on the space of 1 750 terms with constants, its exact counts per bin, and the counts it is given."""
    query = generator_query(priced_space(), PRICED)
    terms = all_terms(query, FRACTIONAL)
    exact = exact_log_counts(terms, FRACTIONAL, WL_EDGES)
    given = [0.0] * len(exact) if log_counts is None else log_counts
    return metropolis_chain(query, FRACTIONAL, theta, WL_EDGES, WL_TARGET, given), exact


def relative(log_counts):
    """Log counts made relative to their first bin."""
    return [value - log_counts[0] for value in log_counts]


def test_wang_landau_learns_the_counts_per_bin_roughly_from_nothing():
    """From equal counts the walk's stages halve the factor to the final one, and its estimate lands near the counts."""
    chain, exact = priced_chain()
    assert all(value > -math.inf for value in exact)
    assert max(exact) - min(exact) > 1.0  # the bins differ by more than e, so equal counts are far off
    result = wang_landau(chain, random.Random(1), final_log_f=1e-3)
    assert isinstance(result, WangLandau)
    assert result.converged
    learned = relative(result.log_counts)
    truth = relative(exact)
    # Rough: the walk's error stays at the size of its early stages' visit noise (measured 0.12 - 0.45 in log over eight
    # seeds); the correction by visits below is what makes it accurate.
    assert max(abs(a - b) for a, b in zip(learned, truth, strict=True)) < 0.6, (learned, truth)
    factors = [log_f for log_f, _steps in result.stages]
    assert factors[0] == 1.0
    assert all(math.isclose(later, earlier / 2) for earlier, later in itertools.pairwise(factors))
    assert factors[-1] >= 1e-3 > factors[-1] / 2
    assert result.steps == sum(steps for _log_f, steps in result.stages)


def test_the_visits_of_a_run_on_fixed_counts_correct_them():
    """From the walk's rough counts, one run on them: the corrected counts within 0.12 in log, the errors their size."""
    chain, exact = priced_chain()
    rough = wang_landau(chain, random.Random(1), final_log_f=1e-3)
    corrected = counts_from_visits(chain.with_log_counts(rough.log_counts), random.Random(2), 40000, start=rough.term)
    assert isinstance(corrected, VisitCounts)
    assert corrected.steps == 40000
    assert sum(corrected.visits) == 40000
    learned = relative(corrected.log_counts)
    truth = relative(exact)
    errors = [abs(a - b) for a, b in zip(learned, truth, strict=True)]
    assert max(errors) < 0.12, (learned, truth)
    # The reported errors are of the actual errors' size: none of the misses beyond five of them (the first bin's
    # error enters every relative count, so its error is added).
    for index, miss in enumerate(errors):
        bar = corrected.relative_error[index] + corrected.relative_error[0]
        assert miss <= 5 * bar, (index, miss, bar)


def test_the_visits_of_a_run_on_the_exact_counts_stay_near_them():
    """On the exact counts the correction keeps them, up to the visits' noise."""
    chain, exact = priced_chain()
    corrected = counts_from_visits(chain.with_log_counts(exact), random.Random(3), 40000)
    errors = [abs(a - b) for a, b in zip(relative(corrected.log_counts), relative(exact), strict=True)]
    assert max(errors) < 0.1, errors


def test_the_learned_counts_make_the_chain_visit_the_target():
    """The chain run on the walk's counts, corrected by visits, visits the bins in proportion to the target."""
    chain, _exact = priced_chain()
    rough = wang_landau(chain, random.Random(3), final_log_f=1e-3)
    corrected = counts_from_visits(chain.with_log_counts(rough.log_counts), random.Random(4), 30000, start=rough.term)
    learned = chain.with_log_counts(corrected.log_counts)
    visits = Counter()
    steps = 30000
    for count, step in enumerate(learned.run(random.Random(5), start=corrected.term)):
        if count >= steps:
            break
        visits[learned.bin_of(learned.cost_of(step.term))] += 1
    shares = [visits[index] / steps for index in range(len(WL_TARGET))]
    assert max(abs(share - mass) for share, mass in zip(shares, WL_TARGET, strict=True)) < 0.03, shares


def test_a_walk_that_does_not_flatten_in_its_budget_says_so():
    """A step budget too small for the first stage: not converged, every step counted, the stage reported unfinished."""
    chain, _exact = priced_chain()
    result = wang_landau(chain, random.Random(5), max_steps=30)
    assert not result.converged
    assert result.steps == 30
    assert result.stages == ()


def test_a_bin_with_a_target_but_no_term_is_left_out_of_the_flatness_when_the_counts_say_so():
    """A bin the given counts call empty is never required to be visited; the walk still converges."""
    query = generator_query(priced_space(), PRICED)
    terms = all_terms(query, FRACTIONAL)
    edges = (*WL_EDGES, 20.0)  # one more bin, past the dearest term
    target = (*WL_TARGET[:-1], 0.05, 0.05)
    exact = exact_log_counts(terms, FRACTIONAL, edges)
    assert exact[-1] == -math.inf
    chain = metropolis_chain(query, FRACTIONAL, 0.4, edges, target, exact)
    result = wang_landau(chain, random.Random(6), log_counts=exact, log_f=0.05, final_log_f=0.01)
    assert result.converged
    assert result.log_counts[-1] == -math.inf


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"log_f": 0.0}, "factor"),
        ({"final_log_f": -1.0}, "factor"),
        ({"flatness": 1.5}, "flatness"),
        ({"max_steps": -1}, "steps"),
        ({"check_every": 0}, "check"),
        ({"log_counts": [0.0]}, "one log count per bin"),
    ],
)
def test_what_wang_landau_refuses(options, message):
    """A factor not positive, a flatness outside (0, 1], a negative budget, a check interval below one, counts not one per bin."""
    chain, _exact = priced_chain()
    with pytest.raises(ValueError, match=message):
        wang_landau(chain, random.Random(0), **options)


@pytest.mark.parametrize(
    ("steps", "batches", "message"), [(100, 1, "batches"), (5, 10, "steps"), (100, True, "batches")]
)
def test_what_the_correction_by_visits_refuses(steps, batches, message):
    """Fewer than two batches, or fewer steps than batches."""
    chain, _exact = priced_chain()
    with pytest.raises(ValueError, match=message):
        counts_from_visits(chain, random.Random(0), steps, batches=batches)


# ---------------------------------------------------------------------------
# The chain as a sampler
# ---------------------------------------------------------------------------


def test_the_sampler_is_a_sampler_and_streams_the_chain_after_burn_in_every_thin_th_state():
    """Burn-in, then every ``thin``-th state of the chain the same seed runs, with the counts given."""
    query = generator_query(priced_space(), PRICED)
    terms = all_terms(query, FRACTIONAL)
    exact = exact_log_counts(terms, FRACTIONAL, WL_EDGES)
    sampler = MarkovChainSampler(
        FRACTIONAL, 0.4, WL_EDGES, WL_TARGET, random.Random(7), log_counts=exact, burn_in=25, thin=4
    )
    assert isinstance(sampler, Sampler)
    drawn = list(itertools.islice(sampler.sample(query), 30))
    chain = metropolis_chain(query, FRACTIONAL, 0.4, WL_EDGES, WL_TARGET, exact)
    states = [step.term for step in itertools.islice(chain.run(random.Random(7)), 25 + 4 * 30)]
    assert drawn == [states[25 + 4 * index + 3] for index in range(30)]


def test_the_sampler_learns_its_counts_when_none_are_given():
    """Without counts the sampler runs Wang-Landau first, keeps the estimate, and draws from the chain on it."""
    query = generator_query(priced_space(), PRICED)
    sampler = MarkovChainSampler(
        FRACTIONAL, 0.4, WL_EDGES, WL_TARGET, random.Random(8), thin=3, wang_landau={"final_log_f": 1e-3}, visits=30000
    )
    drawn = list(itertools.islice(sampler.sample(query), 4000))
    assert sampler.last_estimate is not None
    assert sampler.last_estimate.converged
    assert sampler.last_correction is not None
    assert sampler.last_correction.steps == 30000
    bins = Counter(bisect.bisect_right(WL_EDGES, FRACTIONAL.fold(term)) - 1 for term in drawn)
    shares = [bins[index] / len(drawn) for index in range(len(WL_TARGET))]
    assert max(abs(share - mass) for share, mass in zip(shares, WL_TARGET, strict=True)) < 0.04, shares


def test_the_sampler_completes_a_partial_term_and_counts_the_terms_exactly():
    """On a residual query the draws complete its term; ``at_least`` reads the exact number of the query's terms."""
    query, algebra = nested()
    terms = all_terms(query, algebra)
    term = next(t for t in terms if t.root == "f")
    partial = residual_query(query.solution_space, query.start, term, (0,))
    completions = all_terms(partial, algebra)
    edges, target = (0.0, 2.0, 4.0, 7.0), (0.5, 0.3, 0.2)
    sampler = MarkovChainSampler(
        algebra, 0.2, edges, target, random.Random(9), log_counts=exact_log_counts(completions, algebra, edges)
    )
    for drawn in itertools.islice(sampler.sample(partial), 50):
        assert drawn.replace_subtree_at((0,), HOLE) == term.replace_subtree_at((0,), HOLE)
    assert sampler.at_least(partial, len(completions))
    assert not sampler.at_least(partial, len(completions) + 1)
    assert sampler.at_least(partial, 0)


@pytest.mark.parametrize(
    ("change", "message"),
    [({"burn_in": -1}, "burn-in"), ({"thin": 0}, "thin"), ({"root_share": 2.0}, "root share")],
)
def test_what_the_sampler_refuses(change, message):
    """A negative burn-in, a thinning below one, a root share outside [0, 1]."""
    arguments = {"burn_in": 0, "thin": 1, "root_share": 0.5} | change
    with pytest.raises(ValueError, match=message):
        MarkovChainSampler(FRACTIONAL, 0.4, WL_EDGES, WL_TARGET, random.Random(0), log_counts=[0.0] * 5, **arguments)
