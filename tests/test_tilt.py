"""The exponential tilt: random search in proportion to ``e^(-theta c(t))``, and what it needs of a program.

A term's tilted weight is ``e^(-theta c(t))`` for an additive cost ``c``. Below a search node the
tilted mass factors over the holes, so one real number per non-terminal carries it: ``Z_A(theta)``,
the sum of the tilted weights of the terms rooted at ``A``. These tests hold ``log Z`` and the tilted
mean cost to the tree form, which counts the terms of every cost value one by one and shares none of
the recursion, on finite spaces, at a positive, a zero and a negative ``theta``, under an algebra
whose costs are fractions. What the tilt refuses is pinned beside it: a language with infinitely
many terms, a cost or a ``theta`` that is not a real number, and a predicate that reads a hole.
"""

import bisect
import heapq
import itertools
import math
import random

import pytest

import cosy.search.samplers as samplers_module
import cosy.search.tilt as tilt_module
from cosy.core import Constructor, SpecificationBuilder, Synthesizer
from cosy.core.solution_space import Goal, NonTerminalArgument, SolutionSpace
from cosy.search import generator_query, residual_query
from cosy.search.cost_tables import cost_table
from cosy.search.costs import AdditiveCostAlgebra, ComponentwiseTuples, NonNegativeReals
from cosy.search.counting import branch_counts
from cosy.search.partial import holes
from cosy.search.rules import deepest_first_subgoal
from cosy.search.samplers import Sampler, TiltSampler
from cosy.search.sampling import log_sum_exp, weighted_tree
from cosy.search.tilt import (
    TiltedSearch,
    _log_gaussian_interval,
    _log_min_over,
    _log_upper_tail,
    _mixture,
    _node_moments,
    _query_spacing,
    saddle_counts,
    saddle_grid,
    saddle_mixture,
    saddle_search,
    theta_for_mean,
    tilt_program,
    tilt_table,
    tilted_mixture,
    tilted_search,
)
from tests.search_fixtures import (
    BOX,
    HOLLOW,
    NOWHERE,
    PRICED,
    TUPLE_SORT,
    USED,
    cut_space,
    hole_tuple_space,
    hollow_space,
    list_space,
    literal_predicate_space,
    priced_space,
)

# Every term of the finite spaces below is smaller than this, so the tree form counts all of them.
SIZE_OF_EVERYTHING = 30


def fractional_symbol_cost(symbol):
    """Charge a combinator a quarter per letter of its name beyond four, and a literal a half more than itself.

    Some combinators cost nothing and the rest fractions, so a table indexed by whole cost values
    could not hold these costs at all.

    Args:
        symbol: A ``Tree`` root: a combinator, or the value of a constant argument.

    Returns:
        float: The cost of the symbol.
    """
    name = getattr(symbol, "__name__", None)
    if name is not None:
        return 0.25 * max(0, len(name) - 4)
    return 0.5 + symbol


FRACTIONAL = AdditiveCostAlgebra(NonNegativeReals(), fractional_symbol_cost)
UNIT = AdditiveCostAlgebra(NonNegativeReals(), lambda _symbol: 1)

FINITE_SPACES = [
    ("priced", priced_space, PRICED),
    ("hole tuples", hole_tuple_space, TUPLE_SORT),
    ("literal predicate", literal_predicate_space, USED),
    ("hollow", hollow_space, HOLLOW),
]
THETAS_OF_THE_TABLE = [0.0, 0.4, 1.3, -0.25]


def counts_by_cost(space, start, algebra):
    """Count the terms rooted at a non-terminal per cost value, by the tree form, one branch at a time.

    Args:
        space: The program.
        start: The non-terminal.
        algebra (AdditiveCostAlgebra): The algebra whose fold is the cost.

    Returns:
        dict: The number of terms per cost value.
    """
    counts = branch_counts(generator_query(space, start), SIZE_OF_EVERYTHING, algebra.fold).counts
    wider = branch_counts(generator_query(space, start), SIZE_OF_EVERYTHING + 10, algebra.fold).counts
    assert counts == wider, (start, "the size bound must hold every term")
    return dict(counts)


def inhabited(space, algebra):
    """Every non-terminal of a program with its terms per cost value, the empty ones left out.

    Args:
        space: The program.
        algebra (AdditiveCostAlgebra): The algebra.

    Returns:
        dict: The non-empty count rows.
    """
    rows = {nonterminal: counts_by_cost(space, nonterminal, algebra) for nonterminal in space.nonterminals()}
    return {nonterminal: row for nonterminal, row in rows.items() if row}


@pytest.mark.parametrize(("name", "build", "start"), FINITE_SPACES)
@pytest.mark.parametrize("theta", THETAS_OF_THE_TABLE)
def test_log_z_is_the_tilted_sum_over_every_term(name, build, start, theta):
    """``Z_A(theta) = sum_a N_A(a) e^(-theta a)`` for every non-terminal with a term, the counts the tree form's.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The space's start symbol, which must be among the non-terminals compared.
        theta (float): The tilt.
    """
    space = build()
    table = tilt_table(space, FRACTIONAL, theta)
    rows = inhabited(space, FRACTIONAL)
    assert start in rows, name
    for nonterminal, row in rows.items():
        expected = math.log(sum(count * math.exp(-theta * cost) for cost, count in row.items()))
        assert math.isclose(table.of(nonterminal), expected, rel_tol=1e-12, abs_tol=1e-12), (name, nonterminal)


def test_the_finite_spaces_say_something():
    """The comparison above needs several non-terminals and several cost values, and each space has them."""
    for name, build, _start in FINITE_SPACES[:2]:
        rows = inhabited(build(), FRACTIONAL)
        assert len(rows) >= 3, name
        assert len({cost for row in rows.values() for cost in row}) >= 4, name
        assert sum(min(row) < max(row) for row in rows.values()) >= 2, (
            name,
            "too few non-terminals whose costs differ",
        )


@pytest.mark.parametrize(("name", "build", "start"), FINITE_SPACES)
def test_at_theta_zero_z_is_the_number_of_terms(name, build, start):
    """No tilt: every term weighs one, so ``Z_A(0)`` counts the terms rooted at ``A``.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The space's start symbol.
    """
    space = build()
    table = tilt_table(space, UNIT, 0)
    for nonterminal, row in inhabited(space, UNIT).items():
        assert math.isclose(table.of(nonterminal), math.log(sum(row.values())), rel_tol=1e-12), (name, nonterminal)
    assert round(math.exp(table.of(start))) == sum(counts_by_cost(space, start, UNIT).values())


@pytest.mark.parametrize(("name", "build", "start"), FINITE_SPACES)
@pytest.mark.parametrize("theta", THETAS_OF_THE_TABLE)
def test_the_tilted_mean_is_the_mean_cost_under_the_tilt(name, build, start, theta):
    """``mean_A(theta) = sum_a a N_A(a) e^(-theta a) / Z_A(theta)``, which is ``-d log Z_A / d theta``.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The space's start symbol.
        theta (float): The tilt.
    """
    space = build()
    table = tilt_table(space, FRACTIONAL, theta)
    step = 1e-5
    above, below = tilt_table(space, FRACTIONAL, theta + step), tilt_table(space, FRACTIONAL, theta - step)
    for nonterminal, row in inhabited(space, FRACTIONAL).items():
        weights = {cost: count * math.exp(-theta * cost) for cost, count in row.items()}
        expected = sum(cost * weight for cost, weight in weights.items()) / sum(weights.values())
        assert math.isclose(table.mean_cost[nonterminal], expected, rel_tol=1e-12, abs_tol=1e-12), (name, nonterminal)
        derivative = -(above.of(nonterminal) - below.of(nonterminal)) / (2 * step)
        assert math.isclose(table.mean_cost[nonterminal], derivative, rel_tol=1e-6, abs_tol=1e-6), (name, nonterminal)
    assert start in table.mean_cost


def test_a_sort_without_a_term_has_no_mass():
    """A hole of a non-terminal the program never mentions weighs nothing: its clause contributes nothing, the rest is unaffected."""
    space = hollow_space()
    table = tilt_table(space, UNIT, 0.5)
    assert table.of(NOWHERE) == -math.inf
    assert table.of(Constructor("NeverMentioned")) == -math.inf
    assert NOWHERE not in table.mean_cost
    assert math.isclose(table.of(HOLLOW), -0.5)
    assert table.mean_cost[HOLLOW] == 1


def test_an_infinite_language_is_refused_and_named():
    """A cycle among non-terminals that have terms pumps without end, and the tilt covers finite languages."""
    with pytest.raises(ValueError, match=r"infinitely many terms.*List"):
        tilt_table(list_space(), UNIT, 2.0)


PUMP = Constructor("Pump")
DEAD = Constructor("Dead")
NEVER = Constructor("Never")


def pump_leaf() -> str:
    """End the pump: the one term of ``Pump``.

    Returns:
        str: Its rendering.
    """
    return "o"


def pump_step(inner: str) -> str:
    """Wrap a ``Dead``, which never has a term.

    Args:
        inner (str): The filler of the hole.

    Returns:
        str: Its rendering.
    """
    return f"s({inner})"


def dead_back(pump: str, never: str) -> str:
    """Close the loop back to ``Pump``, but only beside a ``Never``, which has no clause.

    Args:
        pump (str): The filler of the ``Pump`` hole.
        never (str): The filler of the ``Never`` hole.

    Returns:
        str: Its rendering.
    """
    return f"d({pump}, {never})"


def dead_loop_space():
    """Build ``Pump -> pump_leaf | pump_step(Dead)``, ``Dead -> dead_back(Pump, Never)``, and nothing for ``Never``.

    The loop through ``Dead`` never closes, since ``Never`` has no term, so the language is the one leaf.

    Returns:
        SolutionSpace: The space, started at ``Pump``.
    """
    specs = {
        pump_leaf: SpecificationBuilder().suffix(PUMP),
        pump_step: SpecificationBuilder().argument("inner", DEAD).suffix(PUMP),
        dead_back: SpecificationBuilder().argument("pump", PUMP).argument("never", NEVER).suffix(DEAD),
    }
    return Synthesizer(specs).construct_solution_space(PUMP)


def test_a_loop_through_a_sort_without_terms_is_no_loop():
    """What cannot be completed is pruned first, and what remains of the loop is a single leaf."""
    table = tilt_table(dead_loop_space(), UNIT, 0.0)
    assert math.isclose(table.of(PUMP), 0.0, abs_tol=1e-15)
    assert table.of(DEAD) == -math.inf


@pytest.mark.parametrize(
    "algebra",
    [
        AdditiveCostAlgebra(ComponentwiseTuples(2), lambda _symbol: (1.0, 0.0)),
        AdditiveCostAlgebra(NonNegativeReals(), lambda _symbol: True),
    ],
)
def test_a_cost_that_is_not_a_real_number_is_refused(algebra):
    """The tilt weighs a term by ``e^(-theta c)``, which needs ``c`` to be one real number, not a tuple or a truth value.

    An infinite or undefined cost the domain of the reals refuses itself, before the tilt sees it.

    Args:
        algebra (AdditiveCostAlgebra): An algebra with a cost the tilt cannot use.
    """
    with pytest.raises(ValueError, match="takes finite real costs"):
        tilt_table(priced_space(), algebra, 1.0)


@pytest.mark.parametrize("theta", [math.nan, math.inf, -math.inf, "1", True, None])
def test_theta_must_be_a_finite_real_number(theta):
    """A tilt of infinity would weigh every term but the cheapest nothing, and a string is no tilt at all.

    Args:
        theta: A value that is not a tilt.
    """
    with pytest.raises(ValueError, match="theta must be a finite real number"):
        tilt_table(priced_space(), UNIT, theta)


def test_a_program_whose_predicate_reads_a_hole_is_refused():
    """The holes of a clause must be filled independently, or the product over the holes overcounts."""
    with pytest.raises(ValueError, match="reading a hole in a predicate"):
        tilt_table(cut_space(), UNIT, 1.0)


# ---------------------------------------------------------------------------------------------
# The tilted search: random search in proportion to e^(-theta c(t)), weighed from the tilt table
# ---------------------------------------------------------------------------------------------


def assert_streams_agree(expected, actual):
    """Assert that two keyed streams coincide term for term and key for key.

    Args:
        expected (list): The keyed stream of the oracle.
        actual (list): The keyed stream of the tilted search.
    """
    assert len(expected) == len(actual)
    assert [term for _, term in expected] == [term for _, term in actual]
    for (expected_key, _), (actual_key, _) in zip(expected, actual, strict=True):
        assert math.isclose(expected_key, actual_key, rel_tol=1e-12, abs_tol=1e-12)


def tilted_tree(query, algebra, theta, **options):
    """The tree form under ``pi(a) = N(a) e^(-theta a)``, which weighs every term ``e^(-theta c(t)) / Z``.

    Args:
        query: The query.
        algebra (AdditiveCostAlgebra): The algebra whose fold is the cost.
        theta (float): The tilt.
        **options: Passed on to ``weighted_tree``, the computation rule among them.

    Returns:
        WeightedTree: The tree form, ready to stream from.
    """
    counts = branch_counts(query, SIZE_OF_EVERYTHING, algebra.fold, **options).counts
    return weighted_tree(
        query, SIZE_OF_EVERYTHING, algebra.fold, lambda cost: counts[cost] * math.exp(-theta * cost), **options
    )


@pytest.mark.parametrize(("name", "build", "start"), FINITE_SPACES)
@pytest.mark.parametrize("theta", [0.0, 0.4, -0.25])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_the_tilted_search_streams_what_the_tree_form_streams_under_the_tilted_counts(name, build, start, theta, seed):
    """Every term weighs ``e^(-theta c(t))`` in both, so the streams are the same, term for term and key for key.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The queried non-terminal.
        theta (float): The tilt.
        seed (int): The seed under test.
    """
    query = generator_query(build(), start)
    expected = list(tilted_tree(query, FRACTIONAL, theta).keyed_stream(random.Random(seed)))
    actual = list(tilted_search(query, FRACTIONAL, theta).keyed_stream(random.Random(seed)))
    assert len(actual) >= 1, name
    assert_streams_agree(expected, actual)


def test_a_partial_term_is_completed_as_the_tree_form_completes_it_at_every_position():
    """The prescribed symbols are charged: at the root, the leaves, the literals and everything between."""
    space = priced_space()
    everything = tilted_tree(generator_query(space, PRICED), FRACTIONAL, 0.0)
    parent = max(everything.stream(random.Random(5)), key=lambda term: (term.depth, str(term)))
    positions = sorted(parent.positions())
    assert len(positions) >= 4, "a shallow parent would leave the deep positions untested"
    assert any(not callable(parent.subtree_at(position).root) for position in positions), "no literal leaf"
    for position in positions:
        query = residual_query(space, PRICED, parent, position)
        assert_streams_agree(
            list(tilted_tree(query, FRACTIONAL, 0.7).keyed_stream(random.Random(11))),
            list(tilted_search(query, FRACTIONAL, 0.7).keyed_stream(random.Random(11))),
        )


def shallowest_last_hole(goal):
    """A computation rule other than the engine's: the shallowest open hole, the last of them among equals.

    Args:
        goal (Goal): The search node, not a success node.

    Returns:
        tuple: The position and the argument there.
    """
    position = max(holes(goal), key=lambda path: (-len(path), path))
    return position, goal.subgoals[position]


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_the_tilted_search_follows_a_computation_rule_other_than_the_engines(seed):
    """The rule decides the frontier's children and so the order of the stream, in both constructions alike.

    Args:
        seed (int): The seed under test.
    """
    query = generator_query(priced_space(), PRICED)
    under_the_rule = list(
        tilted_search(query, FRACTIONAL, 0.4, subgoal_selection=shallowest_last_hole).keyed_stream(random.Random(seed))
    )
    expected = list(
        tilted_tree(query, FRACTIONAL, 0.4, subgoal_selection=shallowest_last_hole).keyed_stream(random.Random(seed))
    )
    assert_streams_agree(expected, under_the_rule)
    engines = list(tilted_search(query, FRACTIONAL, 0.4).keyed_stream(random.Random(seed)))
    assert sorted(map(str, (term for _, term in engines))) == sorted(map(str, (term for _, term in under_the_rule)))
    assert [term for _, term in engines] != [term for _, term in under_the_rule]


def an_expanded_position_first(goal):
    """A rule the tilted search cannot follow: an expanded position, while one is still a subgoal.

    Args:
        goal (Goal): The search node, not a success node.

    Returns:
        tuple: The position and the argument there.
    """
    expanded = [position for position in goal.subgoals if position in goal.constructors]
    position = expanded[0] if expanded else deepest_first_subgoal(goal)[0]
    return position, goal.subgoals[position]


def test_a_rule_that_selects_an_expanded_position_is_refused_by_name():
    """A node is weighed by its open holes, so the rule must expand one."""
    search = tilted_search(
        generator_query(priced_space(), PRICED), FRACTIONAL, 0.4, subgoal_selection=an_expanded_position_first
    )
    with pytest.raises(ValueError, match="not an open hole"):
        list(search.stream(random.Random(0)))


WIDE = Constructor("Wide")
WIDE_LEAF = Constructor("WideLeaf")
WIDTH = 30


def _wide_wrap(index):
    """Build the ``index``-th clause ``Wide -> wrap_index(WideLeaf)``.

    Args:
        index (int): Which of the thirty clauses.

    Returns:
        Callable: The combinator, named after its index.
    """

    def wrap(inner: str) -> str:
        return f"w{index}({inner})"

    wrap.__name__ = f"wrap_{index}"
    return wrap


def _wide_leaf(index):
    """Build the ``index``-th leaf of ``WideLeaf``.

    Args:
        index (int): Which of the thirty leaves.

    Returns:
        Callable: The combinator, named after its index.
    """

    def leaf() -> str:
        return f"v{index}"

    leaf.__name__ = f"leaf_{index}"
    return leaf


def wide_space():
    """Build a space of thirty clauses over thirty leaves: every expansion has thirty children.

    Returns:
        SolutionSpace: The space, started at ``Wide``.
    """
    specs = {_wide_wrap(i): SpecificationBuilder().argument("inner", WIDE_LEAF).suffix(WIDE) for i in range(WIDTH)}
    specs |= {_wide_leaf(i): SpecificationBuilder().suffix(WIDE_LEAF) for i in range(WIDTH)}
    return Synthesizer(specs).construct_solution_space(WIDE)


def test_a_child_is_built_when_it_is_popped_and_not_before(monkeypatch):
    """Thirty children per expansion, and the first draw calls ``Goal.update`` once, for the grandchild it pops.

    Args:
        monkeypatch: pytest's monkeypatch.
    """
    search = tilted_search(generator_query(wide_space(), WIDE), UNIT, 0.0)
    calls = []
    update = Goal.update

    def counting(self, rule, position):
        calls.append(rule.terminal)
        return update(self, rule, position)

    monkeypatch.setattr(Goal, "update", counting)
    assert next(search.stream(random.Random(0))) is not None
    assert len(calls) <= 2, len(calls)


def test_a_child_the_engine_refuses_when_it_is_built_is_an_error_not_a_skip(monkeypatch):
    """The weight a child was pushed with assumed the engine applies its clause; if it does not, say so.

    Args:
        monkeypatch: pytest's monkeypatch.
    """
    search = tilted_search(generator_query(wide_space(), WIDE), UNIT, 0.0)
    monkeypatch.setattr(Goal, "update", lambda self, rule, position: None)
    with pytest.raises(ValueError, match="refused a clause the lazy frontier had weighed"):
        next(search.stream(random.Random(0)))


def test_a_table_handed_in_must_be_for_the_same_program_algebra_and_theta():
    """The table's numbers weigh the search's nodes, so they must have been computed for exactly this search."""
    space = priced_space()
    query = generator_query(space, PRICED)
    table = tilt_table(space, FRACTIONAL, 0.4)
    assert tilted_search(query, FRACTIONAL, 0.4, table=table).table is table
    with pytest.raises(ValueError, match="another program"):
        tilted_search(generator_query(priced_space(), PRICED), FRACTIONAL, 0.4, table=table)
    with pytest.raises(ValueError, match="another algebra"):
        tilted_search(query, UNIT, 0.4, table=table)
    with pytest.raises(ValueError, match="another theta"):
        tilted_search(query, FRACTIONAL, 0.5, table=table)


def test_a_query_without_a_term_streams_nothing():
    """A start symbol without a term has no tilted mass, and random search has nothing to draw."""
    search = tilted_search(generator_query(hollow_space(), NOWHERE), UNIT, 0.3)
    assert list(search.stream(random.Random(0))) == []
    assert search.total == 0


@pytest.mark.parametrize(("name", "build", "start"), FINITE_SPACES)
def test_the_sampler_streams_the_tilted_search_and_counts_the_terms_exactly(name, build, start):
    """Behind the sampler protocol: the same stream, and ``at_least`` from the exact number of terms.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The queried non-terminal.
    """
    query = generator_query(build(), start)
    sampler = TiltSampler(FRACTIONAL, 0.4, random.Random(3))
    assert isinstance(sampler, Sampler)
    streamed = list(sampler.sample(query))
    assert streamed == list(tilted_search(query, FRACTIONAL, 0.4).stream(random.Random(3))), name
    number = sum(counts_by_cost(query.solution_space, start, FRACTIONAL).values())
    assert len(streamed) == number, name
    assert sampler.at_least(query, number)
    assert not sampler.at_least(query, number + 1)
    assert sampler.at_least(query, 0)


# ---------------------------------------------------------------------------------------------
# A theta for a target mean: the tilted mean of the query's terms falls as theta grows
# ---------------------------------------------------------------------------------------------


def exact_tilted_mean(query, algebra, theta):
    """The tilted mean cost of a query's terms, from the tree form's counts per cost value.

    Args:
        query: The query.
        algebra (AdditiveCostAlgebra): The algebra whose fold is the cost.
        theta (float): The tilt.

    Returns:
        float: ``sum_a a N(a) e^(-theta a) / sum_a N(a) e^(-theta a)``.
    """
    counts = branch_counts(query, SIZE_OF_EVERYTHING, algebra.fold).counts
    weights = {cost: count * math.exp(-theta * cost) for cost, count in counts.items()}
    return sum(cost * weight for cost, weight in weights.items()) / sum(weights.values())


@pytest.mark.parametrize("theta", [0.0, 0.4, -0.25])
def test_the_query_mean_is_the_tilted_mean_of_its_terms(theta):
    """For the whole language and at every position of a partial term, the prescribed symbols included.

    Args:
        theta (float): The tilt.
    """
    space = priced_space()
    query = generator_query(space, PRICED)
    assert math.isclose(
        tilted_search(query, FRACTIONAL, theta).mean_cost, exact_tilted_mean(query, FRACTIONAL, theta), rel_tol=1e-12
    )
    parent = next(tilted_search(query, FRACTIONAL, 0.0).stream(random.Random(2)))
    for position in sorted(parent.positions()):
        residual = residual_query(space, PRICED, parent, position)
        assert math.isclose(
            tilted_search(residual, FRACTIONAL, theta).mean_cost,
            exact_tilted_mean(residual, FRACTIONAL, theta),
            rel_tol=1e-12,
        ), position


def test_theta_for_a_target_mean_hits_it_and_falls_as_the_target_rises():
    """Targets across the realized costs, each hit to the tolerance, and a dearer target asks for a smaller theta."""
    query = generator_query(priced_space(), PRICED)
    costs = sorted(branch_counts(query, SIZE_OF_EVERYTHING, FRACTIONAL.fold).counts)
    assert len(costs) >= 4
    targets = [costs[0] + (costs[-1] - costs[0]) * share for share in (0.1, 0.35, 0.6, 0.9)]
    thetas = [theta_for_mean(query, FRACTIONAL, target) for target in targets]
    for target, theta in zip(targets, thetas, strict=True):
        # the tolerance is a share of the span of the costs
        assert abs(exact_tilted_mean(query, FRACTIONAL, theta) - target) <= 1e-9 * (costs[-1] - costs[0]), target
    assert thetas == sorted(thetas, reverse=True)
    assert len(set(thetas)) == len(thetas)


def test_the_uniform_mean_asks_for_no_tilt():
    """The mean of all terms alike is the mean at theta zero."""
    query = generator_query(priced_space(), PRICED)
    uniform_mean = exact_tilted_mean(query, FRACTIONAL, 0.0)
    assert math.isclose(theta_for_mean(query, FRACTIONAL, uniform_mean), 0.0, abs_tol=1e-9)


@pytest.mark.parametrize("side", ["below", "above"])
def test_a_target_outside_the_realized_costs_is_refused(side):
    """No tilt moves the mean past the cheapest or the dearest term.

    Args:
        side (str): Which side of the realized costs the target lies on.
    """
    query = generator_query(priced_space(), PRICED)
    costs = sorted(branch_counts(query, SIZE_OF_EVERYTHING, FRACTIONAL.fold).counts)
    target = costs[0] - 0.5 if side == "below" else costs[-1] + 0.5
    with pytest.raises(ValueError, match="outside the costs"):
        theta_for_mean(query, FRACTIONAL, target)


# ---------------------------------------------------------------------------------------------
# A target on cost bins: the counts per bin estimated from a mixture of tilts, and rejection to the target
# ---------------------------------------------------------------------------------------------


def indexed_symbol_cost(symbol):
    """Charge ``wrap_i`` and ``leaf_i`` a tenth of their index, so that the wide space's 900 terms spread over 59 costs.

    Args:
        symbol: A combinator of the wide space.

    Returns:
        float: A tenth of its index.
    """
    return int(symbol.__name__.rsplit("_", 1)[1]) / 10


INDEXED = AdditiveCostAlgebra(NonNegativeReals(), indexed_symbol_cost)
# The wide space's costs run from 0 to 5.8 in tenths, their counts a triangle; four bins over them, a target
# that is not the counts' own shape, and tilts that lean to either side.
EDGES = (0.0, 1.45, 2.95, 4.45, 5.95)
TARGET = (0.4, 0.1, 0.2, 0.3)
THETAS = (-1.0, 0.0, 1.0)


def exact_bin_counts(query, algebra, edges):
    """The number of a query's terms per bin, from the tree form.

    Args:
        query: The query.
        algebra (AdditiveCostAlgebra): The algebra whose fold is the cost.
        edges (Sequence[float]): The bins' boundaries.

    Returns:
        list: The counts per bin.
    """
    counts = [0] * (len(edges) - 1)
    for cost, count in branch_counts(query, SIZE_OF_EVERYTHING, algebra.fold).counts.items():
        for index in range(len(edges) - 1):
            if edges[index] <= cost < edges[index + 1]:
                counts[index] += count
    return counts


def all_terms(query, algebra):
    """Every term of a finite query, from a full stream of the tilted search at no tilt.

    Args:
        query: The query.
        algebra (AdditiveCostAlgebra): The algebra.

    Returns:
        list: The terms.
    """
    return list(tilted_search(query, algebra, 0.0).stream(random.Random(0)))


def test_the_bins_say_something():
    """Every bin holds dozens of terms, so that the estimate, the bound and the uniformity within a bin all say something."""
    counts = exact_bin_counts(generator_query(wide_space(), WIDE), INDEXED, EDGES)
    assert sum(counts) == WIDTH * WIDTH
    assert all(count >= 50 for count in counts), counts


def test_the_estimate_is_unbiased_and_its_error_is_its_spread():
    """Over many pilots the mean estimate per bin is the exact count, and the reported error is the estimates' spread."""
    query = generator_query(wide_space(), WIDE)
    exact = exact_bin_counts(query, INDEXED, EDGES)
    repeats = 200
    estimates = [[] for _ in exact]
    reported = [[] for _ in exact]
    for seed in range(repeats):
        mixture = tilted_mixture(query, INDEXED, THETAS, EDGES, TARGET, 12, random.Random(seed))
        for index in range(len(exact)):
            estimates[index].append(math.exp(mixture.log_estimate[index]))
            reported[index].append(mixture.relative_error[index])
    ratios = []
    for index, count in enumerate(exact):
        mean = sum(estimates[index]) / repeats
        spread = math.sqrt(sum((value - mean) ** 2 for value in estimates[index]) / (repeats - 1))
        assert abs(mean - count) <= 4 * spread / math.sqrt(repeats), (index, mean, count)
        typical = sorted(reported[index])[repeats // 2]
        ratios.append(typical * count / spread)
    # The reported error over the estimates' actual spread: about one per bin, and one on average across the
    # bins, where an error that forgot the draws missing a bin would read about 0.85.
    assert all(0.75 <= ratio <= 1.35 for ratio in ratios), ratios
    assert 0.93 <= sum(ratios) / len(ratios) <= 1.15, ratios


def test_a_cost_on_an_edge_belongs_to_the_bin_the_edge_opens():
    """Bin ``i`` holds ``[edges[i], edges[i + 1])``: an edge opens its bin, and the last edge closes the last one."""
    query = generator_query(wide_space(), WIDE)
    mixture = tilted_mixture(query, INDEXED, THETAS, (0.0, 1.5, 3.0, 5.9), (0.3, 0.3, 0.4), 6, random.Random(8))
    assert [mixture.bin_of(cost) for cost in (0.0, 1.4, 1.5, 2.9, 3.0, 5.8)] == [0, 0, 1, 1, 2, 2]
    assert mixture.bin_of(-0.1) is None
    assert mixture.bin_of(5.9) is None


def exact_mixture(query, algebra, thetas, edges, target):
    """The mixture with the exact counts per bin in place of an estimate, so that the rejection is tested alone.

    Args:
        query: The query.
        algebra (AdditiveCostAlgebra): The algebra.
        thetas (Sequence[float]): The tilts.
        edges (Sequence[float]): The bins' boundaries.
        target (Sequence[float]): The target's mass per bin.

    Returns:
        TiltedMixture: The construction.
    """
    counts = exact_bin_counts(query, algebra, edges)
    searches = [tilted_search(query, algebra, theta) for theta in thetas]
    log_counts = [math.log(count) if count else -math.inf for count in counts]
    return _mixture(searches, edges, target, log_counts, [0.0] * len(counts), 2, "no bin with a target has a term")


def test_every_term_in_a_bin_with_a_target_is_under_the_bound():
    """Rejection is exact only if the ratio of the target to the mixture never exceeds the bound, on any term."""
    query = generator_query(wide_space(), WIDE)
    terms = all_terms(query, INDEXED)
    for mixture in (
        exact_mixture(query, INDEXED, THETAS, EDGES, TARGET),
        tilted_mixture(query, INDEXED, THETAS, EDGES, TARGET, 12, random.Random(1)),
        exact_mixture(query, INDEXED, (1.0,), EDGES, TARGET),
        exact_mixture(query, INDEXED, (-1.0,), EDGES, TARGET),
    ):
        ratios = [mixture.log_acceptance(INDEXED.fold(term)) for term in terms]
        assert max(ratios) <= 1e-12
        assert max(ratios) > math.log(0.5), "a bound far above every ratio would accept far less than it could"


def test_with_exact_counts_the_bins_follow_the_target_and_a_bins_terms_are_alike():
    """The first term of 6 000 streams: the bins in proportion to the target, and the terms of a bin equally often."""
    query = generator_query(wide_space(), WIDE)
    mixture = exact_mixture(query, INDEXED, THETAS, EDGES, TARGET)
    draws = 6000
    by_bin = [0] * len(TARGET)
    by_term = {}
    for seed in range(draws):
        term = next(mixture.stream(random.Random(seed)))
        by_bin[mixture.bin_of(INDEXED.fold(term))] += 1
        by_term[term] = by_term.get(term, 0) + 1
    chi_square = sum(
        (observed - draws * mass) ** 2 / (draws * mass) for observed, mass in zip(by_bin, TARGET, strict=True)
    )
    assert chi_square < 16.27, (by_bin, chi_square)  # the 0.999 quantile of chi-square with three degrees of freedom
    terms = all_terms(query, INDEXED)
    for index in range(len(TARGET)):
        members = [term for term in terms if mixture.bin_of(INDEXED.fold(term)) == index]
        expected = by_bin[index] / len(members)
        within = sum((by_term.get(term, 0) - expected) ** 2 / expected for term in members)
        freedom = len(members) - 1
        # above the 0.999 quantile of chi-square with this many degrees of freedom, by its normal approximation
        assert within < freedom + 3.1 * math.sqrt(2 * freedom), (index, within, freedom)


def test_the_stream_streams_each_term_of_the_target_bins_once_and_then_ends():
    """A small language is exhausted: every term in a bin with a target, once, and nothing outside the bins."""
    query = generator_query(hole_tuple_space(), TUPLE_SORT)
    mixture = tilted_mixture(query, FRACTIONAL, (-0.5, 0.5), (2.2, 2.6, 3.1), (0.5, 0.5), 12, random.Random(4))
    streamed = list(mixture.stream(random.Random(5)))
    inside = [term for term in all_terms(query, FRACTIONAL) if 2.2 <= FRACTIONAL.fold(term) < 3.1]
    assert len(streamed) == len(set(streamed)) == len(inside) >= 3
    assert set(streamed) == set(inside)


def test_a_bin_no_pilot_draw_reaches_drops_its_target_and_says_so():
    """A bin beyond the dearest term has no estimate: its share of the target is reported, and nothing is drawn there."""
    query = generator_query(wide_space(), WIDE)
    mixture = tilted_mixture(query, INDEXED, THETAS, (0.0, 5.95, 9.0), (0.75, 0.25), 12, random.Random(6))
    assert mixture.log_estimate[1] == -math.inf
    assert mixture.relative_error[1] == math.inf
    assert math.isclose(mixture.missing_target, 0.25)
    assert mixture.target == (1.0, 0.0)


@pytest.mark.parametrize(
    ("thetas", "edges", "target", "pilot", "match"),
    [
        ((), EDGES, TARGET, 12, "at least one tilt"),
        ((0.1, 0.1), EDGES, TARGET, 12, "distinct"),
        (THETAS, (3.0,), (), 12, "strictly ascending"),
        (THETAS, (3.0, 3.0, 5.0), (0.5, 0.5), 12, "strictly ascending"),
        (THETAS, EDGES, TARGET[:3], 12, "one nonnegative mass per bin"),
        (THETAS, EDGES, (0.5, -0.1, 0.3, 0.3), 12, "one nonnegative mass per bin"),
        (THETAS, EDGES, (0.0, 0.0, 0.0, 0.0), 12, "one nonnegative mass per bin"),
        (THETAS, EDGES, TARGET, 1, "at least two"),
        (THETAS, EDGES, TARGET, 2.5, "at least two"),
        (THETAS, (100.0, 200.0), (1.0,), 12, "no pilot draw fell in a bin with a target"),
    ],
)
def test_what_the_mixture_refuses(thetas, edges, target, pilot, match):
    """Tilts that do not mix, bins that do not order, a target that is no target, a pilot too small, a target out of reach.

    Args:
        thetas (tuple): The tilts.
        edges (tuple): The bins' boundaries.
        target (tuple): The target per bin.
        pilot: The pilot size.
        match (str): The refusal's wording.
    """
    with pytest.raises(ValueError, match=match):
        tilted_mixture(generator_query(wide_space(), WIDE), INDEXED, thetas, edges, target, pilot, random.Random(0))


# ---------------------------------------------------------------------------------------------
# A prepared program: what does not depend on theta, computed once and shared by every theta
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "build", "start"), FINITE_SPACES)
def test_a_prepared_program_gives_the_same_tables(name, build, start):
    """One preparation, a table per theta: each the table computed from scratch.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The space's start symbol.
    """
    space = build()
    program = tilt_program(space, FRACTIONAL)
    for theta in THETAS_OF_THE_TABLE:
        prepared, fresh = program.table(theta), tilt_table(space, FRACTIONAL, theta)
        assert dict(prepared.log_z) == dict(fresh.log_z), (name, theta)
        assert dict(prepared.mean_cost) == dict(fresh.mean_cost), (name, theta)
        assert dict(prepared.counts) == dict(fresh.counts), (name, theta)
    assert program.counts[start] > 0


def test_a_prepared_program_is_used_and_not_prepared_again(monkeypatch):
    """The search for a theta and the mixture read the program handed in, and never prepare one of their own."""
    query = generator_query(wide_space(), WIDE)
    program = tilt_program(query.solution_space, INDEXED)
    monkeypatch.setattr(tilt_module, "tilt_program", lambda *_args, **_kwargs: pytest.fail("prepared again"))
    theta = theta_for_mean(query, INDEXED, 2.0, program=program)
    # the tolerance is a share of the span of the costs, 0 to 5.8 here
    assert abs(tilted_search(query, INDEXED, theta, table=program.table(theta)).mean_cost - 2.0) <= 1e-9 * 5.8
    mixture = tilted_mixture(query, INDEXED, THETAS, EDGES, TARGET, 4, random.Random(0), program=program)
    assert all(search.table.counts is program.counts for search in mixture.searches)


def test_a_program_prepared_for_another_program_or_algebra_is_refused():
    """The prepared costs and holes are the program's and the algebra's, and read against another they weigh something else."""
    query = generator_query(wide_space(), WIDE)
    other_space = tilt_program(wide_space(), INDEXED)
    other_algebra = tilt_program(query.solution_space, UNIT)
    with pytest.raises(ValueError, match="another program"):
        theta_for_mean(query, INDEXED, 2.0, program=other_space)
    with pytest.raises(ValueError, match="another algebra"):
        tilted_mixture(query, INDEXED, THETAS, EDGES, TARGET, 4, random.Random(0), program=other_algebra)


# ---------------------------------------------------------------------------------------------
# What the first review of the tilt found: a cycle of two, a repeated hole, a query without a term, the sampler's
# cache, a target the mean approaches slowly, the tolerance on small costs, and a theta too large for the costs
# ---------------------------------------------------------------------------------------------


def space_of(rules):
    """Build a program directly from its clauses, with strings for non-terminals and terminals.

    Args:
        rules (list): ``(head, terminal, holes)`` triples.

    Returns:
        SolutionSpace: The program.
    """
    space = SolutionSpace()
    for head, terminal, hole_types in rules:
        space.add_rule(head, terminal, tuple(NonTerminalArgument(None, hole) for hole in hole_types), ())
    return space


def priced(costs):
    """An algebra charging each terminal its entry.

    Args:
        costs (dict): The cost per terminal.

    Returns:
        AdditiveCostAlgebra: The algebra.
    """
    return AdditiveCostAlgebra(NonNegativeReals(), costs.__getitem__)


def test_a_cycle_of_two_non_terminals_is_refused_and_named():
    """``A -> a | f(B)``, ``B -> g(A)``: both have terms and each reads the other, so the language is infinite."""
    space = space_of([("A", "a", ()), ("A", "f", ("B",)), ("B", "g", ("A",))])
    with pytest.raises(ValueError, match="infinitely many terms: A, B lie on a cycle"):
        tilt_table(space, priced({"a": 1, "f": 1, "g": 1}), 0.5)


def test_a_clause_whose_holes_repeat_a_non_terminal_counts_every_pair():
    """``P -> pair(B, B)`` is ``P``'s only clause: it completes once ``B`` has a term, and counts every pair of them."""
    space = space_of([("P", "pair", ("B", "B")), ("B", "b1", ()), ("B", "b2", ())])
    algebra = priced({"pair": 1.0, "b1": 0.5, "b2": 2.0})
    table = tilt_table(space, algebra, 0.3)
    assert table.counts["P"] == 4
    expected = -0.3 * 1.0 + 2 * math.log(math.exp(-0.3 * 0.5) + math.exp(-0.3 * 2.0))
    assert math.isclose(table.of("P"), expected, rel_tol=1e-12)


def test_a_query_without_a_term_has_no_mean_and_a_dead_clause_changes_no_mean():
    """No term, no mean; and a clause whose hole has no term leaves the query's mean to its other clauses."""
    assert math.isnan(tilted_search(generator_query(hollow_space(), NOWHERE), UNIT, 0.3).mean_cost)
    assert tilted_search(generator_query(hollow_space(), HOLLOW), UNIT, 0.3).mean_cost == 1.0


def test_the_sampler_builds_anew_for_another_query_and_after_forget(monkeypatch):
    """The construction is kept for one query at a time, and ``forget`` drops it.

    Args:
        monkeypatch: pytest's monkeypatch.
    """
    built = []
    original = samplers_module.tilted_search
    monkeypatch.setattr(
        samplers_module, "tilted_search", lambda *args, **kwargs: built.append(args[0]) or original(*args, **kwargs)
    )
    sampler = TiltSampler(UNIT, 0.0, random.Random(0))
    first, second = generator_query(hole_tuple_space(), TUPLE_SORT), generator_query(priced_space(), PRICED)
    assert len(list(sampler.sample(first))) == 6
    assert sampler.at_least(first, 6)
    assert len(built) == 1
    assert len(list(sampler.sample(second))) == sum(counts_by_cost(second.solution_space, PRICED, UNIT).values())
    assert built == [first, second]
    sampler.forget()
    assert sampler.at_least(second, 1)
    assert built == [first, second, second]


def test_what_the_tilted_search_and_its_sampler_refuse():
    """A predicate that reads a hole, and a theta that is no number."""
    with pytest.raises(ValueError, match="reading a hole in a predicate"):
        tilted_search(generator_query(cut_space(), BOX), UNIT, 0.5)
    with pytest.raises(ValueError, match="theta must be a finite real number"):
        TiltSampler(UNIT, math.nan, random.Random(0))


@pytest.mark.parametrize(
    ("rules", "costs", "target", "span"),
    [
        # two terms a cost apart, far from zero: the mean moves by little per step
        ([("S", "s", ("B",)), ("B", "x", ()), ("B", "y", ())], {"s": 100_000, "x": 0, "y": 1}, 100_000.25, 1.0),
        # one term of cost 0 beside 100^6 of cost 10: the mean stays near 10 until theta is large
        (
            [("S", "z", ()), ("S", "big", tuple("B" for _ in range(6)))] + [("B", f"b{i}", ()) for i in range(100)],
            {"z": 0, "big": 10} | {f"b{i}": 0 for i in range(100)},
            5.0,
            10.0,
        ),
        # costs of a ten-thousandth: the tolerance is a share of their span, not of one
        ([("S", "p", ()), ("S", "q", ())], {"p": 0.0, "q": 1e-4}, 2.5e-5, 1e-4),
    ],
)
def test_theta_for_mean_meets_targets_the_mean_approaches_slowly(rules, costs, target, span):
    """A target strictly between the cheapest and the dearest cost is met, however slowly the mean moves towards it.

    Args:
        rules (list): The program's clauses.
        costs (dict): The terminals' costs.
        target (float): The mean wanted.
        span (float): The dearest term's cost less the cheapest's, read off the costs by hand: the second
            space has a trillion terms, more than the tree form can count one by one.
    """
    query = generator_query(space_of(rules), "S")
    algebra = priced(costs)
    theta = theta_for_mean(query, algebra, target)
    assert abs(tilted_search(query, algebra, theta).mean_cost - target) <= 1e-9 * span


def test_a_language_of_one_cost_has_that_mean_at_every_theta_and_no_other():
    """Every term costs the same: its cost is the mean at no tilt, and any other target is out of reach."""
    query = generator_query(space_of([("S", "p", ()), ("S", "q", ())]), "S")
    algebra = priced({"p": 3.0, "q": 3.0})
    assert theta_for_mean(query, algebra, 3.0) == 0.0
    with pytest.raises(ValueError, match="outside the costs"):
        theta_for_mean(query, algebra, 3.5)


def test_theta_for_mean_stops_at_its_step_bound():
    """A search that needs more tables than it may compute says so."""
    query = generator_query(priced_space(), PRICED)
    costs = sorted(branch_counts(query, SIZE_OF_EVERYTHING, FRACTIONAL.fold).counts)
    with pytest.raises(ValueError, match="within 2 steps"):
        theta_for_mean(query, FRACTIONAL, costs[0] + 0.01 * (costs[-1] - costs[0]), max_steps=2)


@pytest.mark.parametrize("theta", [1.7e308, -1.7e308])
def test_a_theta_too_large_for_the_costs_is_refused_by_name(theta):
    """``-theta c`` beyond the range of floating point for every term would weigh them all nothing, or nothing sensible.

    Where some term's product stays in range, the others weigh zero, which is their weight rounded, and that is no error.

    Args:
        theta (float): A tilt whose products with every cost overflow.
    """
    space = space_of([("S", "s", ("B",)), ("B", "x", ()), ("B", "y", ())])
    algebra = priced({"s": 5.0, "x": 0.0, "y": 5.0})
    with pytest.raises(ValueError, match="too large in magnitude for the costs"):
        tilt_table(space, algebra, theta)
    rounded = tilt_table(space_of([("S", "p", ()), ("S", "q", ())]), priced({"p": 0.0, "q": 5.0}), abs(theta))
    assert rounded.of("S") == 0.0


@pytest.mark.parametrize(("name", "build", "start"), FINITE_SPACES)
def test_the_cheapest_and_the_dearest_cost_are_the_tree_forms(name, build, start):
    """What the range refusal and the first step of the search for a theta rest on, per non-terminal.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The space's start symbol.
    """
    space = build()
    program = tilt_program(space, FRACTIONAL)
    rows = inhabited(space, FRACTIONAL)
    assert start in rows
    assert set(program.cheapest) == set(rows) == set(program.dearest), name
    for nonterminal, row in rows.items():
        assert program.cheapest[nonterminal] == min(row), (name, nonterminal)
        assert program.dearest[nonterminal] == max(row), (name, nonterminal)


# ---------------------------------------------------------------------------------------------
# What the second review found: the bound read over costs no term has, its minimum on wide bins, a stream that
# never ends, the mean under a large shared cost, an empty query, and the tests that let mutants through
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("low", "high", "where"),
    [
        (-3.0, 3.0, "interior"),
        (1.0, 3.0, "low end"),
        (-3.0, -1.0, "high end"),
        (0.2, 0.2, "one point"),
        (-1e40, 1e40, "vast"),
    ],
)
def test_the_least_log_density_on_an_interval_is_found_wherever_it_lies(low, high, where):
    """``log(a e^c + b e^(-c))`` is least at ``c* = log(b / a) / 2``, where it is ``log(2 sqrt(ab))``.

    Args:
        low (float): The interval's lower end.
        high (float): The interval's upper end.
        where (str): Where the least value lies, for the test id.
    """
    a, b = 0.3, 3.0
    least_at = math.log(b / a) / 2

    def log_density(cost):
        return max(cost, -cost) + math.log(
            a * math.exp(cost - max(cost, -cost)) + b * math.exp(-cost - max(cost, -cost))
        )

    def slope(cost):
        return math.tanh(cost - least_at)

    expected = log_density(min(max(least_at, low), high))
    assert math.isclose(_log_min_over(log_density, slope, low, high), expected, rel_tol=1e-12, abs_tol=1e-12), where


def test_every_term_is_under_the_bound_and_the_bound_is_not_loose_in_random_mixtures():
    """Tilts of either sign, bins cut anywhere, some far past the costs any term has: no ratio above the bound,
    and in every configuration some term within 0.3 nats of it."""
    query = generator_query(wide_space(), WIDE)
    terms = all_terms(query, INDEXED)
    costs = [INDEXED.fold(term) for term in terms]
    rng = random.Random(42)
    checked = 0
    for _ in range(60):
        thetas = tuple(sorted(rng.sample([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], rng.randint(1, 3))))
        edges = sorted(rng.sample([tenth / 10 for tenth in range(-30, 100)], rng.randint(3, 6)))
        target = [rng.random() + 0.05 for _ in range(len(edges) - 1)]
        counts = exact_bin_counts(query, INDEXED, edges)
        if not any(counts):
            continue
        searches = [tilted_search(query, INDEXED, theta) for theta in thetas]
        log_counts = [math.log(count) if count else -math.inf for count in counts]
        mixture = _mixture(
            searches, edges, target, log_counts, [0.0] * len(counts), 2, "no bin with a target has a term"
        )
        ratios = [mixture.log_acceptance(cost) for cost in costs]
        assert max(ratios) <= 1e-12, (thetas, edges)
        assert max(ratios) >= -0.3, (thetas, edges, max(ratios))
        checked += 1
    assert checked >= 40


def test_a_bin_reaching_far_past_the_dearest_term_leaves_the_stream_drawing():
    """The last bin open to a hundred, the dearest term at 5.8: the bound reads the costs terms have, and accepts."""
    query = generator_query(wide_space(), WIDE)
    mixture = tilted_mixture(query, INDEXED, (0.5, 1.0), (0.0, 2.0, 100.0), (0.5, 0.5), 30, random.Random(0))
    near = tilted_mixture(query, INDEXED, (0.5, 1.0), (0.0, 2.0, 5.95), (0.5, 0.5), 30, random.Random(0))
    assert mixture.log_bound == near.log_bound
    assert len(list(mixture.stream(random.Random(1), max_draws=3000))) >= 50


def test_the_error_is_its_spread_with_three_draws_a_tilt():
    """The variance of a tilt's draws divides by one less than their number, which three draws tell from dividing by it."""
    query = generator_query(wide_space(), WIDE)
    exact = exact_bin_counts(query, INDEXED, EDGES)
    repeats = 200
    estimates = [[] for _ in exact]
    reported = [[] for _ in exact]
    for seed in range(repeats):
        mixture = tilted_mixture(query, INDEXED, THETAS, EDGES, TARGET, 3, random.Random(seed))
        for index in range(len(exact)):
            estimates[index].append(
                math.exp(mixture.log_estimate[index]) if mixture.log_estimate[index] > -math.inf else 0.0
            )
            reported[index].append(mixture.relative_error[index])
    ratios = []
    for index, count in enumerate(exact):
        mean = sum(estimates[index]) / repeats
        spread = math.sqrt(sum((value - mean) ** 2 for value in estimates[index]) / (repeats - 1))
        finite = sorted(value for value in reported[index] if math.isfinite(value))
        ratios.append(finite[len(finite) // 2] * count / spread)
    # measured 0.98 on average; dividing by the number of draws would read about 0.80
    assert 0.9 <= sum(ratios) / len(ratios) <= 1.2, ratios


@pytest.mark.parametrize("seed", range(30))
def test_every_stream_of_a_small_language_ends_with_every_term_of_the_target_bins(seed):
    """The tilts lean to the cheap bin, whose terms are drawn often and accepted rarely: every term is drawn long
    before every term of a bin with a target is accepted, and the stream must not end in between.

    Args:
        seed (int): The seed of the pilot and the stream.
    """
    query = generator_query(hole_tuple_space(), TUPLE_SORT)
    mixture = tilted_mixture(query, FRACTIONAL, (1.0, 2.0), (2.0, 2.6, 3.3), (0.5, 0.5), 12, random.Random(seed))
    # a safety net far above what exhausting six terms takes, so that a broken end fails instead of hanging
    streamed = list(mixture.stream(random.Random(seed), max_draws=200_000))
    inside = set()
    for term in all_terms(query, FRACTIONAL):
        index = mixture.bin_of(FRACTIONAL.fold(term))
        if index is not None and mixture.target[index] > 0:
            inside.add(term)
    assert len(inside) >= 4
    assert len(streamed) == len(set(streamed))
    assert set(streamed) == inside


def test_the_missing_share_is_a_share_of_the_whole_target():
    """A target that is not normalized: the bin out of reach carries a quarter of it, whatever its sum."""
    query = generator_query(wide_space(), WIDE)
    mixture = tilted_mixture(query, INDEXED, THETAS, (0.0, 5.95, 9.0), (3.0, 1.0), 12, random.Random(6))
    assert math.isclose(mixture.missing_target, 0.25)


def test_a_bin_without_a_target_is_never_drawn_from():
    """Its terms are reached and estimated, and the stream passes over them without a word."""
    query = generator_query(wide_space(), WIDE)
    mixture = tilted_mixture(query, INDEXED, THETAS, (0.0, 2.95, 5.95), (1.0, 0.0), 12, random.Random(7))
    assert mixture.log_estimate[1] > -math.inf
    drawn = list(itertools.islice(mixture.stream(random.Random(8)), 200))
    assert len(drawn) == 200
    assert all(INDEXED.fold(term) < 2.95 for term in drawn)


def test_a_stream_ends_after_its_draws_and_a_small_language_when_it_is_exhausted():
    """``max_draws`` ends a stream accepted or not; two terms whose hashes agree end it once both are streamed."""
    query = generator_query(wide_space(), WIDE)
    mixture = tilted_mixture(query, INDEXED, THETAS, EDGES, TARGET, 4, random.Random(0))
    assert len(list(mixture.stream(random.Random(1), max_draws=40))) <= 40
    colliding = generator_query(space_of([("S", -1, ()), ("S", -2, ())]), "S")
    assert hash(-1) == hash(-2)
    pair = tilted_mixture(colliding, UNIT, (0.0, 1.0), (0.0, 2.0), (1.0,), 4, random.Random(0))
    calls = []

    class Counting(random.Random):
        def random(self):
            calls.append(1)
            return super().random()

    # a budget far above what two terms take, so that a stream that misses its end fails instead of hanging
    assert len(list(pair.stream(Counting(2), max_draws=100_000))) == 2
    assert len(calls) < 1000, len(calls)


def test_a_query_without_a_term_is_refused_before_any_draw():
    """There is nothing to estimate: said by name, not by a stray ``StopIteration``."""
    with pytest.raises(ValueError, match="the query has no term"):
        tilted_mixture(generator_query(hollow_space(), NOWHERE), UNIT, THETAS, EDGES, TARGET, 3, random.Random(0))


def test_a_large_cost_every_term_shares_costs_the_mean_no_precision():
    """Two terms at 100 000 and 100 001: the mean at theta 0.847 exactly, and the search for theta to its tolerance."""
    query = generator_query(space_of([("S", "a", ()), ("S", "b", ())]), "S")
    algebra = priced({"a": 1e5, "b": 1e5 + 1})

    def exact(theta):
        return (1e5 + (1e5 + 1) * math.exp(-theta)) / (1 + math.exp(-theta))

    assert abs(tilted_search(query, algebra, 0.847).mean_cost - exact(0.847)) <= 1e-9
    for target in (100_000.3, 1e5 + 1e-6):
        assert abs(exact(theta_for_mean(query, algebra, target)) - target) <= 1e-9


def test_the_search_for_theta_computes_a_table_per_step_and_no_search(monkeypatch):
    """Each step is one pass over the prepared program; a search per step would rebuild the query's goals.

    Args:
        monkeypatch: pytest's monkeypatch.
    """
    query = generator_query(wide_space(), WIDE)
    program = tilt_program(query.solution_space, INDEXED)
    monkeypatch.setattr(tilt_module, "tilted_search", lambda *_args, **_kwargs: pytest.fail("a search was built"))
    assert math.isfinite(theta_for_mean(query, INDEXED, 2.0, program=program))


# ---------------------------------------------------------------------------------------------
# The tilted variance: the second derivative of log Z, per non-terminal and for a query
# ---------------------------------------------------------------------------------------------


def exact_tilted_variance(counts, theta):
    """The variance of the cost under the tilt, from the counts per cost value.

    Args:
        counts (dict): The number of terms per cost value.
        theta (float): The tilt.

    Returns:
        float: ``sum_a (a - mean)^2 N(a) e^(-theta a) / Z``.
    """
    weights = {cost: count * math.exp(-theta * cost) for cost, count in counts.items()}
    total = sum(weights.values())
    mean = sum(cost * weight for cost, weight in weights.items()) / total
    return sum((cost - mean) ** 2 * weight for cost, weight in weights.items()) / total


@pytest.mark.parametrize(("name", "build", "start"), FINITE_SPACES)
@pytest.mark.parametrize("theta", THETAS_OF_THE_TABLE)
def test_the_tilted_variance_is_the_variance_of_the_cost_under_the_tilt(name, build, start, theta):
    """Every non-terminal with a term, against the tree form's counts and against the second difference of ``log Z``.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The space's start symbol.
        theta (float): The tilt.
    """
    space = build()
    program = tilt_program(space, FRACTIONAL)
    table = program.table(theta)
    step = 1e-4
    above, below = program.table(theta + step), program.table(theta - step)
    rows = inhabited(space, FRACTIONAL)
    assert start in rows
    if any(min(row) < max(row) for row in rows.values()):
        assert any(table.variance[nonterminal] > 0 for nonterminal in rows), name
    for nonterminal, row in rows.items():
        expected = exact_tilted_variance(row, theta)
        assert math.isclose(table.variance[nonterminal], expected, rel_tol=1e-9, abs_tol=1e-12), (name, nonterminal)
        second = (above.of(nonterminal) - 2 * table.of(nonterminal) + below.of(nonterminal)) / step**2
        assert math.isclose(table.variance[nonterminal], second, rel_tol=1e-4, abs_tol=1e-6), (name, nonterminal)


@pytest.mark.parametrize("theta", [0.0, 0.4, -0.25])
def test_the_query_variance_is_the_variance_of_its_terms(theta):
    """For the whole language and at every position of a partial term, the prescribed symbols included.

    Args:
        theta (float): The tilt.
    """
    space = priced_space()
    query = generator_query(space, PRICED)
    counts = branch_counts(query, SIZE_OF_EVERYTHING, FRACTIONAL.fold).counts
    assert math.isclose(
        tilted_search(query, FRACTIONAL, theta).variance_cost, exact_tilted_variance(counts, theta), rel_tol=1e-9
    )
    parent = next(tilted_search(query, FRACTIONAL, 0.0).stream(random.Random(2)))
    for position in sorted(parent.positions()):
        residual = residual_query(space, PRICED, parent, position)
        counts = branch_counts(residual, SIZE_OF_EVERYTHING, FRACTIONAL.fold).counts
        assert math.isclose(
            tilted_search(residual, FRACTIONAL, theta).variance_cost,
            exact_tilted_variance(counts, theta),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ), position


def test_a_language_of_one_cost_has_no_variance_and_a_query_without_a_term_none_at_all():
    """Every term at one large cost, where subtracting squares would cancel, leaves nothing to vary; no term, no variance."""
    query = generator_query(space_of([("S", "p", ()), ("S", "q", ())]), "S")
    assert tilted_search(query, priced({"p": 1e7 + 0.1, "q": 1e7 + 0.1}), 0.7).variance_cost == 0.0
    assert math.isnan(tilted_search(generator_query(hollow_space(), NOWHERE), UNIT, 0.3).variance_cost)


# ---------------------------------------------------------------------------------------------
# The saddle point: the number of terms per cost bin from the tilt's mass, mean and variance
# ---------------------------------------------------------------------------------------------


def sum_of_choices(holes):
    """A term is ``f`` over ``holes`` independent choices of a digit costing 0, 1 or 3: its cost a sum of them.

    Where many independent parts add up, the cost is near a Gaussian, which is where the saddle point must hold.

    Args:
        holes (int): The number of choices.

    Returns:
        tuple: The query and its algebra.
    """
    space = space_of([("S", "f", tuple("D" for _ in range(holes))), ("D", "d0", ()), ("D", "d1", ()), ("D", "d3", ())])
    return generator_query(space, "S"), priced({"f": 0, "d0": 0, "d1": 1, "d3": 3})


def exact_counts_per_bin(query, algebra, edges, cap):
    """The exact number of terms per bin, from the cost table, which counts what the tree form could not enumerate.

    Args:
        query: The query.
        algebra (AdditiveCostAlgebra): The algebra, whole-number costs.
        edges (Sequence[float]): The bins' boundaries.
        cap (int): A cost cap above every term.

    Returns:
        list: The counts per bin.
    """
    row = cost_table(query.solution_space, algebra, cap).counts[query.start]
    return [sum(count for cost, count in row.items() if edges[i] <= cost < edges[i + 1]) for i in range(len(edges) - 1)]


def central_relative_errors(holes, width):
    """The saddle point's relative error on the bins within two standard deviations of the mean, and how many there are.

    Args:
        holes (int): The number of choices.
        width (int): The bins' width.

    Returns:
        list: One relative error per central bin.
    """
    query, algebra = sum_of_choices(holes)
    mean, deviation = holes * 4 / 3, math.sqrt(holes * 14 / 9)
    edges = [float(value) for value in range(0, 3 * holes + width + 1, width)]
    exact = exact_counts_per_bin(query, algebra, edges, 3 * holes + width)
    estimate = saddle_counts(query, algebra, edges)
    errors = []
    for index, count in enumerate(exact):
        if count and abs((edges[index] + edges[index + 1]) / 2 - mean) <= 2 * deviation:
            errors.append(abs(math.exp(estimate.log_counts[index] - math.log(count)) - 1))
    return errors


def test_the_saddle_point_counts_a_sum_of_many_choices_closely_and_closer_the_more_there_are():
    """The saddle point's error falls as the number of parts grows: measured 3.2, 1.9, 1.0 % for 10, 20, 40 choices."""
    ten, twenty, forty = central_relative_errors(10, 2), central_relative_errors(20, 3), central_relative_errors(40, 4)
    assert min(len(ten), len(twenty), len(forty)) >= 5
    assert max(forty) < 0.015, forty
    assert max(forty) < 0.7 * max(twenty) < 0.7 * 0.7 * max(ten), (ten, twenty, forty)


def test_a_bin_the_query_does_not_reach_counts_nothing():
    """Below the cheapest term and above the dearest there is nothing to approximate."""
    query, algebra = sum_of_choices(8)
    estimate = saddle_counts(query, algebra, (-10.0, -1.0, 0.0, 30.0, 40.0))
    assert estimate.log_counts[0] == -math.inf
    assert estimate.log_counts[1] == -math.inf
    assert estimate.log_counts[2] > 0
    assert estimate.log_counts[3] == -math.inf


@pytest.mark.parametrize(("holes", "edges"), [(20, range(6, 50, 4)), (100, range(200, 301, 2))])
def test_each_bin_is_read_at_a_theta_whose_mean_falls_within_the_cells_of_its_points(holes, edges):
    """The local form holds near the tilted mean, so each bin's tilt puts its mean within its points' cells.

    On the unit lattice a bin ``[a, b)`` holds the points ``a .. b - 1``, and their cells reach half a unit beyond
    them: a bin two units wide may be read at a mean just below its lower edge.

    Args:
        holes (int): The number of choices.
        edges (range): The bins' boundaries, whole numbers.
    """
    query, algebra = sum_of_choices(holes)
    edges = [float(value) for value in edges]
    estimate = saddle_counts(query, algebra, edges)
    assert all(not math.isnan(theta) for theta in estimate.thetas)
    for index, theta in enumerate(estimate.thetas):
        mean = tilted_search(query, algebra, theta).mean_cost
        assert edges[index] - 0.5 <= mean <= edges[index + 1] - 0.5, (index, mean)


def test_a_single_term_is_counted_once_in_its_bin():
    """No spread at all: the whole tilted mass sits at one cost, and the bin that holds it holds one term."""
    query = generator_query(space_of([("S", "p", ())]), "S")
    estimate = saddle_counts(query, priced({"p": 5.0}), (0.0, 4.0, 6.0, 8.0))
    assert estimate.log_counts[0] == -math.inf
    assert math.isclose(math.exp(estimate.log_counts[1]), 1.0, rel_tol=1e-9)
    assert estimate.log_counts[2] == -math.inf


@pytest.mark.parametrize(("edges", "match"), [((1.0,), "strictly ascending"), ((2.0, 1.0), "strictly ascending")])
def test_what_the_saddle_point_refuses(edges, match):
    """Bins that do not order, and a query without a term.

    Args:
        edges (tuple): The bins' boundaries.
        match (str): The refusal's wording.
    """
    query, algebra = sum_of_choices(4)
    with pytest.raises(ValueError, match=match):
        saddle_counts(query, algebra, edges)
    with pytest.raises(ValueError, match="the query has no term"):
        saddle_counts(generator_query(hollow_space(), NOWHERE), UNIT, (0.0, 1.0))


def test_a_bin_holding_only_the_dearest_or_the_cheapest_cost_is_counted_exactly():
    """No tilted mean reaches an extreme cost, so a bin that holds nothing else counts its terms exactly."""
    query, algebra = sum_of_choices(8)
    top = saddle_counts(query, algebra, (24.0, 25.0))
    assert math.exp(top.log_counts[0]) == 1
    bottom = saddle_counts(query, algebra, (-1.0, 0.0, 0.5))
    assert bottom.log_counts[0] == -math.inf
    assert math.isfinite(bottom.log_counts[1])
    duplicated = generator_query(space_of([("S", "p", ()), ("S", "q", ()), ("S", "r", ())]), "S")
    exact = saddle_counts(duplicated, priced({"p": 1.0, "q": 1.0, "r": 4.0}), (4.0, 9.0))
    assert math.isclose(math.exp(exact.log_counts[0]), 1.0)
    # two dearest choices in each of two holes: four terms of the dearest cost, and four of the cheapest the other way round
    pairs = generator_query(space_of([("S", "f", ("D", "D")), ("D", "x", ()), ("D", "y", ()), ("D", "z", ())]), "S")
    assert (
        round(math.exp(saddle_counts(pairs, priced({"f": 0, "x": 0, "y": 3, "z": 3}), (6.0, 7.0)).log_counts[0])) == 4
    )
    assert (
        round(math.exp(saddle_counts(pairs, priced({"f": 0, "x": 3, "y": 0, "z": 0}), (-1.0, 0.0, 0.5)).log_counts[1]))
        == 4
    )


# ---------------------------------------------------------------------------------------------
# The saddle mixture: the mixture of tilts with the saddle point's counts in place of the pilot
# ---------------------------------------------------------------------------------------------

# Forty choices of 0, 1 or 3: mean 53.3, standard deviation 7.9; bins over two deviations either side, on the lattice.
CHOICE_EDGES = (38.0, 42.0, 46.0, 50.0, 54.0, 58.0, 62.0, 66.0, 70.0)
CHOICE_TARGET = (0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05)
CHOICE_THETAS = (-0.08, 0.0, 0.08)


def test_the_saddle_mixture_draws_no_pilot_and_counts_by_the_saddle_point(monkeypatch):
    """No draw before the stream: the counts per bin are the saddle point's, with no error bar of their own.

    Args:
        monkeypatch: pytest's monkeypatch.
    """
    query, algebra = sum_of_choices(40)
    streams = []
    original = TiltedSearch.keyed_stream
    monkeypatch.setattr(TiltedSearch, "keyed_stream", lambda self, rng: streams.append(1) or original(self, rng))
    mixture = saddle_mixture(query, algebra, CHOICE_THETAS, CHOICE_EDGES, CHOICE_TARGET)
    assert streams == []
    assert mixture.pilot_counts == (0, 0, 0)
    assert all(math.isnan(error) for error in mixture.relative_error)
    assert mixture.log_estimate == saddle_counts(query, algebra, CHOICE_EDGES).log_counts
    assert mixture.missing_target == 0.0


def test_the_saddle_mixture_follows_the_target():
    """The first term of 4 000 streams, by bin, against the target: the saddle point's error of about 1 % cannot show."""
    query, algebra = sum_of_choices(40)
    mixture = saddle_mixture(query, algebra, CHOICE_THETAS, CHOICE_EDGES, CHOICE_TARGET)
    draws = 4000
    by_bin = [0] * len(CHOICE_TARGET)
    for seed in range(draws):
        by_bin[mixture.bin_of(algebra.fold(next(mixture.stream(random.Random(seed)))))] += 1
    chi_square = sum(
        (observed - draws * mass) ** 2 / (draws * mass) for observed, mass in zip(by_bin, CHOICE_TARGET, strict=True)
    )
    assert chi_square < 24.32, (by_bin, chi_square)  # the 0.999 quantile of chi-square with seven degrees of freedom


def test_the_saddle_mixture_leaves_out_bins_below_the_least_share_and_says_so():
    """A bin whose share of the target is below the least share is neither estimated nor reached, and reported missing."""
    query, algebra = sum_of_choices(40)
    target = (1e-9, *CHOICE_TARGET[1:])
    mixture = saddle_mixture(query, algebra, CHOICE_THETAS, CHOICE_EDGES, target, least_share=1e-6)
    assert mixture.log_estimate[0] == -math.inf
    assert math.isclose(mixture.missing_target, 1e-9 / sum(target))


def test_what_the_saddle_mixture_refuses():
    """What the mixture refuses, and a query without a term."""
    query, algebra = sum_of_choices(4)
    with pytest.raises(ValueError, match="strictly ascending"):
        saddle_mixture(query, algebra, (0.1,), (2.0, 1.0), (1.0,))
    with pytest.raises(ValueError, match="distinct"):
        saddle_mixture(query, algebra, (0.1, 0.1), (0.0, 5.0), (1.0,))
    with pytest.raises(ValueError, match="the query has no term"):
        saddle_mixture(generator_query(hollow_space(), NOWHERE), UNIT, (0.1,), (0.0, 5.0), (1.0,))


# ---------------------------------------------------------------------------------------------
# The saddle point, reviewed: the lattice of a query's costs, the moments' precision, the step bound, the far tail
# ---------------------------------------------------------------------------------------------


def digit_sums(parts, holes, head=0.0, extra=()):
    """``S -> f(D x holes)``, ``D`` one clause per part: a term's cost is ``head`` plus a part per hole.

    Args:
        parts (Sequence[float]): The parts' costs.
        holes (int): The number of holes.
        head (float): The cost of ``f``. (Default value = 0.0)
        extra (Sequence[tuple]): Further ``(head, terminal, holes, cost)`` clauses. (Default value = ())

    Returns:
        tuple: The query and its algebra.
    """
    rules = [("S", "f", tuple("D" for _ in range(holes)))] + [("D", f"d{index}", ()) for index in range(len(parts))]
    costs = {"f": head} | {f"d{index}": part for index, part in enumerate(parts)}
    for nonterminal, terminal, hole_types, cost in extra:
        rules.append((nonterminal, terminal, hole_types))
        costs[terminal] = cost
    return generator_query(space_of(rules), "S"), priced(costs)


def digit_sum_counts(parts, holes, head=0):
    """The exact number of terms per cost of :func:`digit_sums`, by convolution in whole numbers.

    Args:
        parts (Sequence[int]): The parts' costs, whole numbers.
        holes (int): The number of holes.
        head (int): The cost of ``f``. (Default value = 0)

    Returns:
        dict: The number of terms per cost.
    """
    row = {head: 1}
    for _ in range(holes):
        wider = {}
        for cost, count in row.items():
            for part in parts:
                wider[cost + part] = wider.get(cost + part, 0) + count
        row = wider
    return row


def relative_errors(estimate, row, edges):
    """The estimate's relative error per bin against exact counts: 0 where both are empty, inf where one of them is.

    Args:
        estimate (SaddleCounts): The estimate.
        row (dict): The exact number of terms per cost.
        edges (Sequence[float]): The bins' boundaries.

    Returns:
        list: One relative error per bin.
    """
    errors = []
    for index in range(len(edges) - 1):
        exact = sum(count for cost, count in row.items() if edges[index] <= cost < edges[index + 1])
        log_count = estimate.log_counts[index]
        if exact == 0 or log_count == -math.inf:
            errors.append(0.0 if exact == 0 and log_count == -math.inf else math.inf)
        else:
            errors.append(math.exp(log_count - math.log(exact)) - 1)
    return errors


def test_a_bin_holding_one_lattice_point_counts_the_terms_of_that_cost():
    """Unit bins on the unit lattice, and width-2 bins on a lattice of 2: each bin holds one cost, and its terms."""
    query, algebra = digit_sums([0, 1, 3], 20)
    edges = [float(value) for value in range(20, 41)]
    errors = relative_errors(saddle_counts(query, algebra, edges), digit_sum_counts([0, 1, 3], 20), edges)
    assert max(abs(error) for error in errors) < 0.05, errors
    query, algebra = digit_sums([0, 2, 6], 20)
    edges = [float(value) for value in range(40, 81, 2)]
    errors = relative_errors(saddle_counts(query, algebra, edges), digit_sum_counts([0, 2, 6], 20), edges)
    assert max(abs(error) for error in errors) < 0.05, errors


def central(errors, edges, mean, deviation):
    """The errors of the bins whose middles lie within two deviations of the mean, where the saddle point is sharp.

    Args:
        errors (list): The relative errors per bin.
        edges (Sequence[float]): The bins' boundaries.
        mean (float): The untilted mean cost.
        deviation (float): The untilted standard deviation.

    Returns:
        list: Their absolute values, at least four of them.
    """
    kept = [
        abs(error)
        for index, error in enumerate(errors)
        if abs((edges[index] + edges[index + 1]) / 2 - mean) <= 2 * deviation
    ]
    assert len(kept) >= 4
    return kept


def test_the_lattice_is_the_spacing_of_the_querys_costs_and_not_of_the_clauses():
    """Parts 1 and 3 add up to every other whole number only; a cost of 1 no query reaches changes no spacing.

    On the unit lattice the bins two wide of the first case were off by 35 %, and those of the second, the odd whole
    numbers, by 39 %; on the query's own lattice every bin within two deviations is within 5 %.
    """
    query, algebra = digit_sums([1, 3], 20)
    edges = [float(value) for value in range(30, 52, 2)]
    errors = relative_errors(saddle_counts(query, algebra, edges), digit_sum_counts([1, 3], 20), edges)
    assert max(central(errors, edges, 40, math.sqrt(20))) < 0.05, errors
    query, algebra = digit_sums([0, 2], 20, head=1)
    edges = [float(value) for value in range(9, 34, 4)]
    errors = relative_errors(saddle_counts(query, algebra, edges), digit_sum_counts([0, 2], 20, head=1), edges)
    assert max(central(errors, edges, 21, math.sqrt(20))) < 0.05, errors
    edges = [float(value) for value in range(150, 460, 25)]
    alone = saddle_counts(*digit_sums([0, 100], 6), edges)
    beside = saddle_counts(*digit_sums([0, 100], 6, extra=[("X", "x", (), 1)]), edges)
    assert alone.log_counts == beside.log_counts
    errors = relative_errors(alone, digit_sum_counts([0, 100], 6), edges)
    assert errors.count(0.0) == 9  # the bins between the multiples of 100 hold nothing, and are read so
    assert max(abs(error) for error in errors) < 0.1, errors


def test_costs_in_halves_are_read_on_their_lattice_as_whole_numbers_are_on_theirs():
    """Halving every cost and every edge changes no count."""
    edges = [float(value) for value in range(22, 60, 3)]
    whole = saddle_counts(*digit_sums([0, 1, 3], 30), edges)
    halves = saddle_counts(*digit_sums([0, 0.5, 1.5], 30), [edge / 2 for edge in edges])
    assert all(math.isfinite(log_count) for log_count in whole.log_counts)
    for here, there in zip(whole.log_counts, halves.log_counts, strict=True):
        assert math.isclose(here, there, rel_tol=1e-9), (whole.log_counts, halves.log_counts)


def test_edges_between_lattice_points_take_the_points_between_them():
    """Bins from half-way to half-way hold the whole numbers inside them."""
    query, algebra = digit_sums([0, 1, 3], 20)
    edges = [value + 0.5 for value in range(14, 46, 4)]
    errors = relative_errors(saddle_counts(query, algebra, edges), digit_sum_counts([0, 1, 3], 20), edges)
    assert max(abs(error) for error in errors) < 0.05, errors


def test_a_cost_every_term_shares_moves_the_bins_and_nothing_else():
    """A head costing 100: the counts of the bins moved by 100 are the counts without it."""
    edges = [float(value) for value in range(6, 50, 4)]
    plain = saddle_counts(*digit_sums([0, 1, 3], 20), edges)
    moved = saddle_counts(*digit_sums([0, 1, 3], 20, head=100), [edge + 100 for edge in edges])
    for here, there in zip(plain.log_counts, moved.log_counts, strict=True):
        assert math.isclose(here, there, rel_tol=1e-9), (plain.log_counts, moved.log_counts)


def test_the_extreme_counts_are_exact_below_holes_that_are_not_leaves():
    """``S -> f(A, A)``, ``A -> g(D, D)``: sixteen terms cost nothing and one costs 12, all four digits dear."""
    query = generator_query(
        space_of([("S", "f", ("A", "A")), ("A", "g", ("D", "D")), ("D", "x", ()), ("D", "y", ()), ("D", "z", ())]), "S"
    )
    estimate = saddle_counts(query, priced({"f": 0, "g": 0, "x": 0, "y": 0, "z": 3}), (-1.0, 0.5, 11.5, 13.0))
    assert round(math.exp(estimate.log_counts[0]), 9) == 16
    assert round(math.exp(estimate.log_counts[2]), 9) == 1


def test_only_the_initial_nodes_that_reach_the_cheapest_cost_count_there():
    """``S -> p | q``, ``p`` costing 1 and ``q`` 2: the bin of cost 1 holds one term."""
    query = generator_query(space_of([("S", "p", ()), ("S", "q", ())]), "S")
    estimate = saddle_counts(query, priced({"p": 1, "q": 2}), (1.0, 2.0, 3.0))
    assert round(math.exp(estimate.log_counts[0]), 9) == 1
    assert round(math.exp(estimate.log_counts[1]), 9) == 1


def test_a_language_of_one_cost_is_counted_whole():
    """Two terms of one cost: the bin holding it holds both."""
    query = generator_query(space_of([("S", "p", ()), ("S", "q", ())]), "S")
    assert round(math.exp(saddle_counts(query, priced({"p": 1, "q": 1}), (0.0, 2.0)).log_counts[0]), 9) == 2


def test_a_bin_ending_at_the_cheapest_cost_holds_nothing():
    """Costs off every lattice floating point adds exactly: the bin up to the cheapest cost excludes it."""
    query, algebra = digit_sums([0.5, 0.6, 0.9], 5)
    assert saddle_counts(query, algebra, (0.0, 2.5, 3.0)).log_counts[0] == -math.inf


def test_a_tilt_that_leans_on_one_cost_counts_the_terms_of_that_cost():
    """Costs off every lattice, a bin holding only the cheapest cost and far less: its one term."""
    query, algebra = digit_sums([0.0, 0.1, 0.3], 20)
    estimate = saddle_counts(query, algebra, (-1.0, 1e-300, 1.0))
    assert math.isclose(math.exp(estimate.log_counts[0]), 1.0, rel_tol=1e-6)


def test_the_moments_keep_their_precision_where_the_costs_span_a_billion():
    """A term costing nothing beside two sums of ten parts near 1e8: where the tilt leans on the sums, their moments.

    The free term sets the query's cheapest cost to 0, so the sums' shares are read ``10^9`` above it.
    """
    base = 1e8
    rules = [("S", "f", ("D",) * 10), ("S", "g", ("D",) * 10), ("S", "z", ())]
    rules += [("D", "d0", ()), ("D", "d1", ()), ("D", "d3", ())]
    query = generator_query(space_of(rules), "S")
    algebra = priced({"f": 0, "g": 0, "z": 0, "d0": base, "d1": base + 1, "d3": base + 3})
    offsets = digit_sum_counts([0, 1, 3], 10)
    for theta in (-3.0, -0.4):
        weights = {offset: 2 * count * math.exp(-theta * (offset - 30)) for offset, count in offsets.items()}
        mean = sum(offset * weight for offset, weight in weights.items()) / sum(weights.values())
        variance = sum((offset - mean) ** 2 * weight for offset, weight in weights.items()) / sum(weights.values())
        search = tilted_search(query, algebra, theta)
        assert abs(search.mean_cost - (10 * base + mean)) <= 1e-6, (theta, search.mean_cost - 10 * base, mean)
        assert math.isclose(search.variance_cost, variance, rel_tol=1e-6), (theta, search.variance_cost, variance)


def test_a_spread_beyond_floating_point_leaves_the_table_and_the_search_as_they_were():
    """Costs 0 and 1e160: the table and the tilted search stand, their variance is infinite, and the saddle point refuses."""
    query = generator_query(space_of([("S", "a", ()), ("S", "b", ())]), "S")
    algebra = priced({"a": 0.0, "b": 1e160})
    table = tilt_table(query.solution_space, algebra, 0.0)
    assert math.isclose(table.mean_cost["S"], 5e159)
    assert table.variance["S"] == math.inf
    assert tilted_search(query, algebra, 0.0).variance_cost == math.inf
    with pytest.raises(ValueError, match="variance"):
        saddle_counts(query, algebra, (0.0, 1e150, 2e160))


def count_tables(monkeypatch):
    """Count the tilt tables computed from here on.

    Args:
        monkeypatch: pytest's monkeypatch.

    Returns:
        list: One entry per table computed.
    """
    computed = []
    table = tilt_module.TiltProgram.table
    monkeypatch.setattr(
        tilt_module.TiltProgram, "table", lambda self, theta: computed.append(theta) or table(self, theta)
    )
    return computed


def test_the_step_bound_counts_every_table_the_fallback_included(monkeypatch):
    """With Newton switched off every bin falls back to bracketing and bisection, and the bound still holds.

    Args:
        monkeypatch: pytest's monkeypatch.
    """
    query, algebra = sum_of_choices(40)
    program = tilt_program(query.solution_space, algebra)
    monkeypatch.setattr(tilt_module, "_NEWTON_STEPS", 0)
    computed = count_tables(monkeypatch)
    with pytest.raises(ValueError, match="more than 10 tilt tables"):
        saddle_counts(query, algebra, CHOICE_EDGES, program=program, max_steps=10)
    assert 0 < len(computed) <= 10


def test_newton_finds_each_bins_tilt_from_its_neighbours_without_bisecting(monkeypatch):
    """Eight bins, each tilt warm-started from its neighbour's: eight tables, and no bracketing at all.

    Args:
        monkeypatch: pytest's monkeypatch.
    """
    query, algebra = sum_of_choices(40)
    program = tilt_program(query.solution_space, algebra)
    computed = count_tables(monkeypatch)
    monkeypatch.setattr(tilt_module, "_theta_between", lambda *_args: pytest.fail("a tilt was bracketed and bisected"))
    saddle_counts(query, algebra, CHOICE_EDGES, program=program)
    assert len(computed) <= 12


def simpson_log_mass(alpha, beta):
    """The log of the standard normal mass on a narrow interval by Simpson's rule, which the code does not use.

    Args:
        alpha (float): The lower end.
        beta (float): The upper end.

    Returns:
        float: The log of the mass, its relative error below ``(width * max(1, |x|))**4 / 500``.
    """
    ends = (alpha, (alpha + beta) / 2, beta)
    log_densities = [-x * x / 2 - math.log(2 * math.pi) / 2 for x in ends]
    top = max(log_densities)
    weighted = sum(weight * math.exp(value - top) for weight, value in zip((1, 4, 1), log_densities, strict=True))
    return math.log((beta - alpha) / 6) + top + math.log(weighted)


def fraction_log_tail(x):
    """``log Q(x)`` for ``x >= 5`` by Laplace's continued fraction, which the code does not use.

    Args:
        x (float): The point.

    Returns:
        float: The log of the standard normal's upper tail at ``x``.
    """
    denominator = x
    for depth in range(400, 0, -1):
        denominator = x + depth / denominator
    return -x * x / 2 - math.log(2 * math.pi) / 2 - math.log(denominator)


@pytest.mark.parametrize("x", [5.0, 10.0, 20.0, 36.0, 37.5, 40.0, 60.0, 200.0])
def test_the_upper_tail_is_the_continued_fractions_near_and_far(x):
    """Below the switch the complementary error function, beyond it the asymptotic series: both to 1e-10 in log.

    Args:
        x (float): The point.
    """
    assert abs(_log_upper_tail(x) - fraction_log_tail(x)) <= 1e-10


@pytest.mark.parametrize(("alpha", "beta"), [(40.0, 41.0), (36.0, 38.0), (37.5, 60.0), (-41.0, -40.0), (-38.0, -36.0)])
def test_an_interval_in_the_far_tail_has_the_mass_of_the_tails_between_its_ends(alpha, beta):
    """Both ends beyond the switch, or one on either side of it.

    Args:
        alpha (float): The lower end.
        beta (float): The upper end.
    """
    near, far = (alpha, beta) if alpha > 0 else (-beta, -alpha)
    expected = fraction_log_tail(near) + math.log1p(-math.exp(fraction_log_tail(far) - fraction_log_tail(near)))
    assert abs(_log_gaussian_interval(alpha, beta) - expected) <= 1e-9


@pytest.mark.parametrize(
    ("alpha", "beta"),
    [
        (-1e-17, 1e-17),
        (-1e-16, 1e-16),
        (-1e-300, 1e-300),
        (36.999999999999, 37.000000000001),
        (10.0, 10.000000000001),
        (-10.000000000001, -10.0),
        (3.0, 3.000001),
        (-2.5e-4, 2.5e-4),
    ],
)
def test_a_narrow_interval_has_the_density_times_its_width(alpha, beta):
    """Where two tails or two error functions would cancel, the mass is read off the density.

    Args:
        alpha (float): The lower end.
        beta (float): The upper end.
    """
    assert abs(_log_gaussian_interval(alpha, beta) - simpson_log_mass(alpha, beta)) <= 1e-9


@pytest.mark.parametrize(("alpha", "beta"), [(-1.0, 0.0), (2.0, 3.0), (-3.0, -2.0), (-0.001, 50.0), (-50.0, 50.0)])
def test_a_wide_interval_has_the_mass_the_error_function_gives(alpha, beta):
    """Away from the tails, the error function's difference, which cancels nothing there.

    Args:
        alpha (float): The lower end.
        beta (float): The upper end.
    """
    expected = math.log((math.erf(beta / math.sqrt(2)) - math.erf(alpha / math.sqrt(2))) / 2)
    assert abs(_log_gaussian_interval(alpha, beta) - expected) <= 1e-12


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"max_steps": 0}, "max_steps"),
        ({"max_steps": True}, "max_steps"),
        ({"max_steps": 2.5}, "max_steps"),
        ({"only": [99]}, "bin"),
        ({"only": [-1]}, "bin"),
        ({"only": ["a"]}, "bin"),
    ],
)
def test_what_the_saddle_point_refuses_of_its_options(options, match):
    """A step bound that is not a positive whole number, and bins that are not the edges' bins.

    Args:
        options (dict): The keyword arguments.
        match (str): The refusal's wording.
    """
    query, algebra = sum_of_choices(4)
    with pytest.raises(ValueError, match=match):
        saddle_counts(query, algebra, (0.0, 5.0, 13.0), **options)


@pytest.mark.parametrize("least_share", [math.nan, 1.5, -0.1, True, "0.1"])
def test_the_least_share_is_a_share(least_share):
    """Between 0 and 1, a real number.

    Args:
        least_share: The share passed.
    """
    query, algebra = sum_of_choices(40)
    with pytest.raises(ValueError, match="least share"):
        saddle_mixture(query, algebra, CHOICE_THETAS, CHOICE_EDGES, CHOICE_TARGET, least_share=least_share)


def test_a_saddle_mixture_whose_target_no_term_reaches_says_so_without_a_pilot():
    """The target on a bin past every term: refused, in words about the counts, not about pilot draws."""
    query, algebra = sum_of_choices(4)
    with pytest.raises(ValueError, match="estimated term") as refusal:
        saddle_mixture(query, algebra, (0.1,), (100.0, 200.0), (1.0,))
    assert "pilot" not in str(refusal.value)


def test_the_saddle_point_is_exported_where_the_tilt_is():
    """``cosy.search`` exports what the tilt module lists, the saddle point's names among them."""
    import cosy.search as search_package  # noqa: PLC0415

    for name in ("SaddleCounts", "saddle_counts", "saddle_mixture"):
        assert name in tilt_module.__all__
        assert getattr(search_package, name) is getattr(tilt_module, name)


def test_a_newton_step_that_would_overshoot_is_halved_until_it_helps(monkeypatch):
    """A digit costing nothing beside a thousand costing 1: at theta 0 the variance is tiny, and the full step lands
    where the variance is nearly zero, so that Newton would stall there; halved, the steps land without bisection.

    Args:
        monkeypatch: pytest's monkeypatch.
    """
    rules = [("S", "f", ("D", "D")), ("D", "x0", ())] + [("D", f"x{index}", ()) for index in range(1, 1001)]
    costs = {"f": 0, "x0": 0} | {f"x{index}": 1 for index in range(1, 1001)}
    query, algebra = generator_query(space_of(rules), "S"), priced(costs)
    monkeypatch.setattr(tilt_module, "_theta_between", lambda *_args: pytest.fail("a tilt was bracketed and bisected"))
    estimate = saddle_counts(query, algebra, (0.5, 1.5))
    assert math.isfinite(estimate.log_counts[0])
    assert 0.5 <= tilted_search(query, algebra, estimate.thetas[0]).mean_cost <= 1.5


def test_a_clause_without_weight_adds_nothing_to_the_variance_not_even_an_infinite_one():
    """``A``'s two terms lie 1e160 apart, so its variance leaves floating point; where ``f(A)`` weighs nothing beside
    ``g``, the variance of ``S`` is ``g``'s, which is none."""
    space = space_of([("S", "f", ("A",)), ("S", "g", ()), ("A", "a", ()), ("A", "b", ())])
    table = tilt_table(space, priced({"f": 1e163, "g": 0, "a": 0, "b": 1e160}), 1e-160)
    assert table.variance["A"] == math.inf
    assert table.variance["S"] == 0.0


@pytest.mark.parametrize(
    ("parts", "head", "unit"),
    [((0, 1, 3), 0, 1.0), ((0, 0.5, 1.5), 0, 0.5), ((0, 0.1, 0.3), 0, 0.0), ((0, 1, 3), 2.0**52, 0.0)],
)
def test_the_unit_is_one_that_floating_point_adds_exactly(parts, head, unit):
    """Whole numbers and halves have theirs; tenths are whole multiples only of a power of two so fine that their sums
    leave the exact range; whole numbers from ``2^52`` on are no longer added exactly.

    Args:
        parts (tuple): The digits' costs.
        head (float): The cost every term shares.
        unit (float): The unit expected.
    """
    query, algebra = digit_sums(parts, 2, head=head)
    program = tilt_program(query.solution_space, algebra)
    assert program.unit == unit
    assert program.spacing["S"] == (parts[1] if unit else 0.0)


def test_a_cost_a_partial_term_has_charged_off_the_unit_leaves_no_lattice():
    """A prescribed part costing half a unit puts the terms off the program's lattice, so none is claimed for them."""
    query, algebra = digit_sums([0, 1, 3], 4)
    program = tilt_program(query.solution_space, algebra)
    assert _query_spacing([(0.5, ("D", "D"))], program, 0.5) == 0.0
    assert _query_spacing([(1.0, ("D", "D"))], program, 1.0) == 1.0


# ---------------------------------------------------------------------------------------------
# The saddle point per search node: random search on each node's estimated completions per bin
# ---------------------------------------------------------------------------------------------

# Every non-terminal has one cost, so every node's completions have one cost and the estimate is exact:
# f(a, b) costs 3 (two terms), g(c) costs 5 (three terms), h costs 10 (one term).
ONE_COST_RULES = [
    ("S", "f", ("A", "B")),
    ("S", "g", ("C",)),
    ("S", "h", ()),
    ("A", "a1", ()),
    ("A", "a2", ()),
    ("B", "b", ()),
    ("C", "c1", ()),
    ("C", "c2", ()),
    ("C", "c3", ()),
]
ONE_COST = {"f": 0, "g": 1, "h": 10, "a1": 1, "a2": 1, "b": 2, "c1": 4, "c2": 4, "c3": 4}
ONE_COST_EDGES = (0.0, 4.0, 8.0, 12.0)
ONE_COST_TARGET = (0.5, 0.3, 0.2)


def instrumented_keyed_stream(expansions, terms):
    """``keyed_stream`` as random search runs it, recording what the saddle search's weights are at every step.

    Args:
        expansions (list): Receives, per expanded inner node, the number of its open holes (-1 at the root) and the log
            of its children's weights summed less its own.
        terms (list): Receives, per term, the term and its node's log weight.

    Returns:
        Callable: The instrumented stream.
    """
    from cosy.search.gumbel import condition_on_maximum, gumbel_key  # noqa: PLC0415

    def stream(root, root_log_weight, expand, rng):
        tie_break = 0
        root_key = gumbel_key(root_log_weight, rng)
        frontier = [(-root_key, tie_break, root_key, root, root_log_weight)]
        while frontier:
            _, _, key, node, log_weight = heapq.heappop(frontier)
            inhabitant, children = expand(node)
            if inhabitant is not None:
                terms.append((inhabitant, log_weight))
                yield key, inhabitant
                continue
            if not children:
                continue
            weights = [child_log_weight for _, child_log_weight in children]
            expansions.append(
                (len(node[3]) if node[0] is not None or node[2] is not None else -1, log_sum_exp(weights) - log_weight)
            )
            for (child, child_log_weight), child_key in zip(
                children, condition_on_maximum(key, weights, rng), strict=True
            ):
                tie_break += 1
                heapq.heappush(frontier, (-child_key, tie_break, child_key, child, child_log_weight))

    return stream


def test_where_every_non_terminal_has_one_cost_the_saddle_search_is_exact(monkeypatch):
    """Every node's completions share one cost, so every estimate is a count: the siblings sum to their parent and
    every term weighs its bin's share over the bin's terms.

    Args:
        monkeypatch: pytest's monkeypatch.
    """
    query, algebra = generator_query(space_of(ONE_COST_RULES), "S"), priced(ONE_COST)
    search = saddle_search(query, algebra, ONE_COST_EDGES, ONE_COST_TARGET)
    assert [round(math.exp(log_count), 9) for log_count in search.log_root_counts] == [2, 3, 1]
    expansions, terms = [], []
    monkeypatch.setattr(tilt_module, "keyed_stream", instrumented_keyed_stream(expansions, terms))
    for seed in range(40):
        for _ in search.keyed_stream(random.Random(seed)):
            pass
    assert len(terms) == 40 * 6
    assert all(abs(deviation) <= 1e-12 for _holes, deviation in expansions), expansions
    share = {3.0: 0.5 / 2, 5.0: 0.3 / 3, 10.0: 0.2 / 1}
    for term, log_weight in terms:
        assert math.isclose(log_weight, math.log(share[algebra.fold(term)]), abs_tol=1e-12)


def test_the_initial_nodes_of_a_saddle_search_weigh_one_in_all():
    """The root's estimate is the sum of its initial nodes', so their weights sum to one exactly, whatever the form's error."""
    for query, algebra, edges, target in (
        (*sum_of_choices(40), CHOICE_EDGES, CHOICE_TARGET),
        (generator_query(space_of(ONE_COST_RULES), "S"), priced(ONE_COST), ONE_COST_EDGES, ONE_COST_TARGET),
    ):
        search = saddle_search(query, algebra, edges, target)
        assert abs(log_sum_exp([log_weight for _node, log_weight in search._root_children()])) <= 1e-12  # noqa: SLF001


def test_the_saddle_search_follows_the_target_where_many_parts_add_up():
    """The first term of 2 000 streams, by bin, against the target: forty choices, bins four wide."""
    query, algebra = sum_of_choices(40)
    search = saddle_search(query, algebra, CHOICE_EDGES, CHOICE_TARGET)
    draws = 2000
    by_bin = [0] * len(CHOICE_TARGET)
    for seed in range(draws):
        cost = algebra.fold(next(search.stream(random.Random(seed), 1)))
        by_bin[bisect.bisect_right(CHOICE_EDGES, cost) - 1] += 1
    chi_square = sum(
        (observed - draws * mass) ** 2 / (draws * mass) for observed, mass in zip(by_bin, CHOICE_TARGET, strict=True)
    )
    assert chi_square < 24.32, (by_bin, chi_square)  # the 0.999 quantile of chi-square with seven degrees of freedom


def test_the_siblings_of_a_saddle_search_agree_at_the_root_and_nearly_where_many_holes_remain(monkeypatch):
    """At the root the children sum to the parent exactly; with six holes or more open, within 3 %; with fewer the form
    reads too few parts, and there they may not.

    Args:
        monkeypatch: pytest's monkeypatch.
    """
    query, algebra = sum_of_choices(40)
    search = saddle_search(query, algebra, CHOICE_EDGES, CHOICE_TARGET)
    expansions, terms = [], []
    monkeypatch.setattr(tilt_module, "keyed_stream", instrumented_keyed_stream(expansions, terms))
    for seed in range(100):
        next(search.keyed_stream(random.Random(seed)))
    at_root = [deviation for holes_open, deviation in expansions if holes_open == -1]
    many = [deviation for holes_open, deviation in expansions if holes_open >= 6]
    assert len(at_root) == 100
    assert len(many) >= 100 * 30
    assert max(abs(deviation) for deviation in at_root) <= 1e-12
    assert max(abs(deviation) for deviation in many) <= 0.03


def test_a_saddle_search_draws_each_term_once_until_its_draws_run_out():
    """Six terms, 300 draws: each term once, and all six."""
    query, algebra = generator_query(space_of(ONE_COST_RULES), "S"), priced(ONE_COST)
    drawn = list(saddle_search(query, algebra, ONE_COST_EDGES, ONE_COST_TARGET).stream(random.Random(3), 300))
    assert len(drawn) == len(set(drawn)) == 6


def test_the_grid_holds_the_tables_values_at_each_tilt_in_order():
    """Two tilts given out of order: each non-terminal's three numbers per tilt, ascending."""
    query, algebra = sum_of_choices(6)
    program = tilt_program(query.solution_space, algebra)
    grid = saddle_grid(program, [0.3, -0.2, 0.3])
    assert grid.thetas == (-0.2, 0.3)
    for position, theta in enumerate(grid.thetas):
        table = program.table(theta)
        for nonterminal in program.order:
            assert grid.log_excess[nonterminal][position] == table.log_excess[nonterminal]
            assert grid.mean_excess[nonterminal][position] == table.mean_excess[nonterminal]
            assert grid.variance[nonterminal][position] == table.variance[nonterminal]


def moments_of_every_node(monkeypatch, search, seeds):
    """Every child random search weighs in some streams, with the moments it was weighed by.

    Args:
        monkeypatch: pytest's monkeypatch.
        search (SaddleSearch): The search.
        seeds (range): The streams' seeds.

    Returns:
        list: The children, each a frontier node carrying its cost, its holes and its moments.
    """
    from cosy.search.gumbel import condition_on_maximum, gumbel_key  # noqa: PLC0415

    seen = []

    def stream(root, root_log_weight, expand, rng):
        frontier = [(-gumbel_key(root_log_weight, rng), 0, root)]
        tie_break = 0
        while frontier:
            negated, _, node = heapq.heappop(frontier)
            inhabitant, children = expand(node)
            if inhabitant is not None:
                yield -negated, inhabitant
                continue
            if children:
                seen.extend(child for child, _ in children)
                keys = condition_on_maximum(-negated, [log_weight for _, log_weight in children], rng)
                for (child, _), key in zip(children, keys, strict=True):
                    tie_break += 1
                    heapq.heappush(frontier, (-key, tie_break, child))

    monkeypatch.setattr(tilt_module, "keyed_stream", stream)
    for seed in seeds:
        next(search.keyed_stream(random.Random(seed)), None)
    return seen


def test_every_node_is_weighed_by_the_sums_over_its_own_holes(monkeypatch):
    """A hole of variance 2.5e17 expanded before one of 1.6: the child's moments are those its own holes sum to, not
    its parent's less the expanded hole's, which would cancel the smaller variance away; and with costs floating point
    adds inexactly, a node's least and greatest cost are its cost plus its holes', the same on every path to it.

    Args:
        monkeypatch: pytest's monkeypatch.
    """
    rules = [
        ("S", "f", ("A", "B")),
        ("A", "a0", ()),
        ("A", "a9", ()),
        ("B", "b0", ()),
        ("B", "b1", ()),
        ("B", "b3", ()),
    ]
    query = generator_query(space_of(rules), "S")
    algebra = priced({"f": 0, "a0": 0, "a9": 1e9, "b0": 0, "b1": 1, "b3": 3})
    search = saddle_search(query, algebra, (0.0, 1.0, 3.0, 5.0), (1.0, 1.0, 1.0), thetas=[0.0])
    nodes = [node for node in moments_of_every_node(monkeypatch, search, range(20)) if node[3] == ("B",)]
    assert nodes
    for node in nodes:
        own = _node_moments(search.grid, node[1], node[3])
        assert node[4][:2] == own[:2]
        for ours, theirs in zip(node[4][2:], own[2:], strict=True):
            assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(ours, theirs, strict=True))
        assert node[4][4][0] > 1.5  # B's variance, 14/9, and not the 0 the cancellation leaves
    rules = [
        ("S", "f", ("A", "B")),
        ("A", "g", ("B", "B")),
        ("A", "a", ()),
        ("B", "b0", ()),
        ("B", "b1", ()),
        ("B", "b3", ()),
    ]
    nested = (generator_query(space_of(rules), "S"), priced({"f": 0, "g": 1, "a": 0.1, "b0": 0, "b1": 0.1, "b3": 0.3}))
    for (query, algebra), edges in (
        (digit_sums([0.0, 0.1, 0.3], 4), [round(0.2 * step, 10) for step in range(8)]),
        (nested, [0.0, 0.5, 1.0, 2.5]),
    ):
        search = saddle_search(query, algebra, edges, [1.0] * (len(edges) - 1))
        nodes = moments_of_every_node(monkeypatch, search, range(20))
        assert len(nodes) >= 20
        assert any(len(node[3]) > 1 for node in nodes)
        check_moments(search, nodes)


def check_moments(search, nodes):
    """Each node's moments are the sums over its own holes: its range exactly, its arrays to rounding.

    Args:
        search (SaddleSearch): The search.
        nodes (list): Frontier nodes it weighed.
    """
    for node in nodes:
        own = _node_moments(search.grid, node[1], node[3])
        assert node[4][:2] == own[:2]  # a node's range, whatever the path
        for ours, theirs in zip(node[4][2:], own[2:], strict=True):
            assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(ours, theirs, strict=True))


def test_a_node_whose_completions_all_cost_the_same_is_read_whole_whatever_the_costs():
    """Costs of 0.1, 0.2 and 0.7, which floating point adds inexactly: each complete term costs one amount, and both
    terms are drawn, each once."""
    rules = [("S", "f", ("A", "B")), ("A", "a", ()), ("B", "b1", ()), ("B", "b2", ())]
    query, algebra = generator_query(space_of(rules), "S"), priced({"f": 0, "a": 0.1, "b1": 0.2, "b2": 0.7})
    drawn = list(saddle_search(query, algebra, (0.0, 0.5, 1.0), (0.5, 0.5)).stream(random.Random(0), 200))
    assert sorted(algebra.fold(term) for term in drawn) == [0.1 + 0.2, 0.1 + 0.7]


def test_a_node_whose_costs_add_below_floating_points_resolution_counts_all_its_completions():
    """A head of 1e17 and digits 0 and 1: both terms cost 1e17 in floating point, and the bin holding it holds two."""
    query, algebra = digit_sums([0, 1], 1, head=1e17)
    search = saddle_search(query, algebra, (0.0, 2e17), (1.0,))
    assert round(math.exp(search.log_root_counts[0]), 9) == 2


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"edges": (1.0, 0.0), "target": (1.0,)}, "strictly ascending"),
        ({"edges": (0.0, 4.0, 8.0), "target": (1.0,)}, "one nonnegative mass per bin"),
        ({"edges": (0.0, 4.0), "target": (0.0,)}, "one nonnegative mass per bin"),
        ({"least_share": 2.0}, "between 0 and 1"),
        ({"least_share": math.nan}, "between 0 and 1"),
        ({"thetas": [math.nan]}, "finite real"),
        ({"thetas": []}, "at least one tilt"),
        ({"edges": (100.0, 200.0), "target": (1.0,), "thetas": [0.0]}, "estimated term"),
    ],
)
def test_what_the_saddle_search_refuses(options, match):
    """Bins and a target that are not of their kind, a least share, a grid, and a target no term reaches.

    Args:
        options (dict): The arguments that differ from a valid call.
        match (str): The refusal's wording.
    """
    query, algebra = generator_query(space_of(ONE_COST_RULES), "S"), priced(ONE_COST)
    arguments = {"edges": ONE_COST_EDGES, "target": ONE_COST_TARGET} | options
    with pytest.raises(ValueError, match=match):
        saddle_search(query, algebra, **arguments)


def test_a_saddle_search_whose_rule_selects_an_expanded_position_is_refused_by_name():
    """The saddle search weighs a node by its open holes, as the tilted search does."""
    query = generator_query(priced_space(), PRICED)
    search = saddle_search(
        query, FRACTIONAL, (-1.0, 100.0), (1.0,), thetas=[0.0], subgoal_selection=an_expanded_position_first
    )
    with pytest.raises(ValueError, match="not an open hole"):
        list(search.keyed_stream(random.Random(0)))


def test_the_saddle_search_is_exported_where_the_tilt_is():
    """``cosy.search`` exports the saddle search's names."""
    import cosy.search as search_package  # noqa: PLC0415

    for name in ("SaddleGrid", "SaddleSearch", "saddle_grid", "saddle_search"):
        assert name in tilt_module.__all__
        assert getattr(search_package, name) is getattr(tilt_module, name)


@pytest.mark.parametrize(
    ("parts", "holes", "edges"),
    [
        ((0, 1, 3), 20, [float(value) for value in range(20, 41)]),
        ((0, 1, 3), 20, [value + 0.5 for value in range(14, 46, 4)]),
        ((0, 1, 3), 6, [0.0, 2.0, 5.0, 9.0, 13.0, 16.0, 19.0]),
        ((0, 2, 6), 20, [float(value) for value in range(40, 81, 2)]),
    ],
)
def test_at_the_root_the_saddle_search_counts_what_the_saddle_point_counts(parts, holes, edges):
    """One initial node, the grid the saddle point's own tilts: its estimate per bin is the saddle point's, the bins
    holding the least or the greatest cost beside others and a lattice of 2 included.

    Args:
        parts (tuple): The digits' costs.
        holes (int): The number of digits.
        edges (list): The bins' boundaries.
    """
    query, algebra = digit_sums(parts, holes)
    search = saddle_search(query, algebra, edges, [1.0] * (len(edges) - 1))
    counts = saddle_counts(query, algebra, edges)
    assert sorted({*(theta for theta in counts.thetas if not math.isnan(theta)), 0.0}) == list(search.grid.thetas)
    assert all(math.isfinite(log_count) for log_count in counts.log_counts)
    for ours, theirs in zip(search.log_root_counts, counts.log_counts, strict=True):
        assert math.isclose(ours, theirs, rel_tol=1e-12), (search.log_root_counts, counts.log_counts)


def test_the_saddle_search_sums_the_initial_nodes_that_share_a_bin_and_normalizes_the_target():
    """Two terms of cost 10 from two initial nodes count two; a target three times too heavy is read as its shares."""
    rules = [*ONE_COST_RULES, ("S", "k", ())]
    query, algebra = generator_query(space_of(rules), "S"), priced(ONE_COST | {"k": 10})
    search = saddle_search(query, algebra, ONE_COST_EDGES, [3 * mass for mass in ONE_COST_TARGET])
    assert [round(math.exp(log_count), 9) for log_count in search.log_root_counts] == [2, 3, 2]
    assert math.isclose(sum(search.target), 1.0)
    assert math.isclose(math.exp(search.log_rho[2]), 0.2 / 2)
    assert abs(log_sum_exp([log_weight for _node, log_weight in search._root_children()])) <= 1e-12  # noqa: SLF001


def test_a_bin_the_least_share_leaves_out_is_the_saddle_searchs_missing_share():
    """The third bin below the least share: not estimated, never drawn, and its share reported."""
    query, algebra = generator_query(space_of(ONE_COST_RULES), "S"), priced(ONE_COST)
    search = saddle_search(query, algebra, ONE_COST_EDGES, (0.5, 0.4999, 0.0001), least_share=0.001)
    assert search.log_root_counts[2] == -math.inf
    assert math.isclose(search.missing_target, 0.0001)
    assert math.isclose(sum(search.target), 1.0)
    assert all(algebra.fold(term) < 8 for term in search.stream(random.Random(1), 200))


def test_the_saddle_search_counts_the_extreme_costs_exactly():
    """On a lattice a bin holding only the least or the greatest cost counts their terms, its edge further than a
    spacing away included; off a lattice, a bin from the greatest cost on counts its terms."""
    query = generator_query(
        space_of([("S", "f", ("A", "A")), ("A", "g", ("D", "D")), ("D", "x", ()), ("D", "y", ()), ("D", "z", ())]), "S"
    )
    search = saddle_search(
        query, priced({"f": 0, "g": 0, "x": 0, "y": 0, "z": 3}), (-10.0, 0.5, 11.5, 20.0), (0.4, 0.2, 0.4)
    )
    assert round(math.exp(search.log_root_counts[0]), 9) == 16
    assert round(math.exp(search.log_root_counts[2]), 9) == 1
    query, algebra = digit_sums([0.5, 0.6, 0.9], 3)
    dearest = tilt_program(query.solution_space, algebra).dearest["S"]
    search = saddle_search(query, algebra, (0.0, dearest, dearest + 1.0), (0.5, 0.5))
    assert round(math.exp(search.log_root_counts[1]), 9) == 1


def test_a_tilt_under_which_a_node_has_one_cost_is_read_only_where_no_other_tilt_is():
    """Costs off every lattice; a grid of 0 and a tilt so steep that every completion costs the least: a bin beside
    that cost is read at 0, not at the steep tilt, whose one cost it does not hold; with the steep tilt alone, the bin
    holding the least cost counts its one term and the other bin nothing; between two steep tilts, the nearer."""
    query, algebra = digit_sums([0.0, 0.1, 0.3], 20)
    edges = (-1.0, 1e-300, 1.0)
    search = saddle_search(query, algebra, edges, (0.5, 0.5), thetas=[0.0, 1e5])
    assert math.isfinite(search.log_root_counts[1])
    alone = saddle_search(query, algebra, edges, (0.5, 0.5), thetas=[1e5])
    assert math.isclose(math.exp(alone.log_root_counts[0]), 1.0)
    assert alone.log_root_counts[1] == -math.inf
    # two steep tilts, one collapsing the node onto its least cost and one onto its greatest: the nearer is read
    both = saddle_search(query, algebra, edges, (0.5, 0.5), thetas=[-1e5, 1e5])
    assert math.isclose(math.exp(both.log_root_counts[0]), 1.0)


def test_a_saddle_search_whose_estimate_reaches_a_bin_no_term_lies_in_draws_nothing_and_ends():
    """Digits 0, 1 and 10 and a target on the costs 4 and 5, which no term has: the estimate is positive there, the
    search finds no term, and the stream ends instead of failing."""
    query, algebra = digit_sums([0, 1, 10], 1)
    search = saddle_search(query, algebra, (4.0, 6.0), (1.0,))
    assert math.isfinite(search.log_root_counts[0])
    assert list(search.stream(random.Random(0), 50)) == []


@pytest.mark.parametrize("max_draws", [True, 2.5, -1, "3"])
def test_a_saddle_streams_bound_is_a_whole_number_of_draws(max_draws):
    """Not a truth value, a fraction, a negative number or a string.

    Args:
        max_draws: The bound passed.
    """
    query, algebra = generator_query(space_of(ONE_COST_RULES), "S"), priced(ONE_COST)
    search = saddle_search(query, algebra, ONE_COST_EDGES, ONE_COST_TARGET)
    with pytest.raises(ValueError, match="max_draws"):
        next(search.stream(random.Random(0), max_draws))


def test_a_grid_whose_variance_leaves_floating_point_is_refused():
    """Costs 0 and 1e160: the variance at a tilt of 0 is infinite, and the saddle point cannot read it."""
    program = tilt_program(space_of([("S", "a", ()), ("S", "b", ())]), priced({"a": 0.0, "b": 1e160}))
    with pytest.raises(ValueError, match="leaves floating point"):
        saddle_grid(program, [0.0])


def test_the_default_grid_is_the_saddle_points_tilts_for_the_bins_and_no_tilt():
    """Bins below the untilted mean, none of them read at ``theta = 0``: the grid is their tilts and 0 besides."""
    query, algebra = sum_of_choices(40)
    edges = (30.0, 34.0, 38.0, 42.0)
    counts = saddle_counts(query, algebra, edges)
    assert all(theta > 0 for theta in counts.thetas)
    search = saddle_search(query, algebra, edges, (1.0, 1.0, 1.0))
    assert search.grid.thetas == (0.0, *sorted(counts.thetas))
