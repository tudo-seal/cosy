"""The cost one rule adds under an additive cost algebra: the step the cost table is built from.

The size table counts terms per size because every rule writes a fixed number of symbols, its
terminal and one leaf per constant argument, and that number is what the fill adds when it applies
the rule. Under an additive cost algebra the same holds with the symbols weighted: applying a rule
adds the cost of its terminal plus the cost of each constant argument, and every non-terminal
argument becomes a hole, which carries no symbol and so adds nothing to the cost so far. That
number, :func:`~cosy.search.counting.rule_cost`, is what a table over the cost values adds per rule.

Two things pin it. Under the algebra that charges one per symbol it *is* the size table's step,
rule for rule. And under an algebra that charges the symbols differently, zero included, it is what
every expansion adds to the cost so far, so that summed along a derivation it gives the fold of the
finished term. The second is checked against :meth:`~cosy.search.costs.AdditiveCostAlgebra.cost_so_far`
and :meth:`~cosy.search.costs.AdditiveCostAlgebra.fold`, which read the partial inhabitant and never
the rule, so the two computations share nothing but the algebra.

The rest of the file is the cost table itself (:mod:`cosy.search.cost_tables`), held to three
oracles that share none of its counting. Under the unit algebra it is the size table, row for row,
and it streams what the size table's form streams, key for key. Below the cap its root row is what
the engine's own expansion finds when it is cut at the cap, and on the reference spaces what the
tree form counts under a size bound that provably covers the cap. On a finite space, with the cap
at its dearest term, it streams what the tree form streams, key for key, from every position of a
term and under a distribution that is not uniform. What it refuses is pinned beside that: a loop
that clauses of cost zero close, a cost that is not a whole number, a negative cap, a predicate that
reads a hole, and a table filled under another algebra or to a lower cap.
"""

import inspect
import itertools
import math
import random

import pytest

import cosy.search.cost_tables as cost_tables_module
from cosy.core import Constructor, SpecificationBuilder, Synthesizer
from cosy.core.solution_space import ConstantArgument, Goal, NonTerminalArgument
from cosy.core.types import DataGroup
from cosy.search import generator_query, residual_query
from cosy.search.cost_tables import (
    CostTable,
    _dict_product,
    _initial_cost_nodes,
    _packed_product,
    _product,
    cost_table,
    weighted_cost_table,
)
from cosy.search.costs import AdditiveCostAlgebra, ComponentwiseTuples, NonNegativeReals
from cosy.search.counting import _added_symbols, branch_counts, rule_cost, size_table
from cosy.search.partial import holes, partial_inhabitant, term_size
from cosy.search.rules import deepest_first_subgoal
from cosy.search.samplers import CostTableSampler, Sampler
from cosy.search.sampling import log_sum_exp, weighted_table, weighted_tree
from tests.search_fixtures import (
    AMBIGUOUS_TARGET,
    BOX,
    CHAIN,
    EXPR,
    IDLE,
    LIST,
    PRICED,
    PRICED_Q,
    ROUND,
    TAGGED,
    TUPLE_SORT,
    USED,
    ambiguous_space,
    chain_space,
    cut_space,
    expression_space,
    hole_tuple_space,
    hollow_space,
    idle_space,
    list_space,
    literal_predicate_space,
    loop_space,
    pair_edge_space,
    priced_space,
    round_space,
    split_space,
    ternary_tails_space,
    three_cycle_space,
    two_symbol_clause_space,
)

CONSTANTS = Constructor("Constants")
DIGITS = DataGroup("digit", (0, 1))


def const_leaf() -> str:
    """Build the leaf.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "l"


def const_two(a: int, b: int) -> str:
    """Fix two digits and open no hole.

    Args:
        a (int): The first digit.
        b (int): The second digit.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"two({a},{b})"


def const_after(rest: str, d: int) -> str:
    """Open a hole and fix a digit after it.

    Args:
        rest (str): The interpreted hole.
        d (int): The digit.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"after({rest},{d})"


def const_between(left: str, d: int, right: str) -> str:
    """Fix a digit between two holes.

    Args:
        left (str): The interpreted first hole.
        d (int): The digit.
        right (str): The interpreted second hole.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"between({left},{d},{right})"


def constants_space():
    """Build a space whose clauses fix two constants, a constant after a hole, and one between two.

    Returns:
        SolutionSpace: The space, started at ``Constants``.
    """
    specs = {
        const_leaf: SpecificationBuilder().suffix(CONSTANTS),
        const_two: SpecificationBuilder().parameter("a", DIGITS).parameter("b", DIGITS).suffix(CONSTANTS),
        const_after: SpecificationBuilder().argument("rest", CONSTANTS).parameter("d", DIGITS).suffix(CONSTANTS),
        const_between: SpecificationBuilder()
        .argument("left", CONSTANTS)
        .parameter("d", DIGITS)
        .argument("right", CONSTANTS)
        .suffix(CONSTANTS),
    }
    return Synthesizer(specs).construct_solution_space(CONSTANTS)


# The spaces the per-rule cost is read on, with the start symbol and the size down to which every
# expansion is checked. The last two are the ones whose clauses carry constant arguments.
SPACES = [
    ("list", list_space, LIST, 6),
    ("expression", expression_space, EXPR, 5),
    ("two symbols per clause", two_symbol_clause_space, TAGGED, 7),
    ("constants", constants_space, CONSTANTS, 7),
]

# A cost per combinator, zero among them, so that a rule adding nothing is part of what is checked.
COMBINATOR_COSTS = {
    # the reference spaces and the constants space above
    "const_leaf": 0,
    "const_two": 1,
    "const_after": 1,
    "const_between": 0,
    "nil": 0,
    "cons_0": 1,
    "cons_1": 2,
    "cons_2": 3,
    "stop": 0,
    "tag": 1,
    "lit": 0,
    "neg": 2,
    "add": 1,
    # the cost table's spaces: several clauses of cost zero, one of them on every level
    "s_a": 0,
    "s_b": 1,
    "s_c": 1,
    "r_one": 2,
    "r_wrap": 0,
    "q_pair": 0,
    "q_twin": 0,
    "q_three": 0,
    "p_top": 0,
    "p_join": 1,
    "p_join_again": 1,
    "p_link": 1,
    "round_stop": 0,
    "round_wrap": 0,
    "round_pair": 0,
    "inner_leaf": 0,
    "inner_step": 1,
    "idle_halt": 1,
    "idle_again": 0,
    "a_zero": 0,
    "a_one": 1,
    "b_single": 2,
    "same_holes": 0,
    "mixed_holes": 1,
}


def weighted_symbol_cost(symbol):
    """Charge a combinator its entry in the table above and a literal digit one more than its value.

    Args:
        symbol: A ``Tree`` root: a combinator, or the value of a constant argument.

    Returns:
        int: The cost of the symbol.
    """
    name = getattr(symbol, "__name__", None)
    if name is not None:
        return COMBINATOR_COSTS[name]
    return symbol + 1


def unit_symbol_cost(_symbol):
    """Charge every symbol one, which makes the fold the term size.

    Args:
        _symbol: The symbol. Ignored.

    Returns:
        int: One.
    """
    return 1


WEIGHTED = AdditiveCostAlgebra(NonNegativeReals(), weighted_symbol_cost)
UNIT = AdditiveCostAlgebra(NonNegativeReals(), unit_symbol_cost)
# The same symbol costs with an estimate of five for every hole, which a clause must not be charged.
ESTIMATING = AdditiveCostAlgebra(NonNegativeReals(), weighted_symbol_cost, hole_cost=lambda _hole: 5)


@pytest.mark.parametrize(("name", "build"), [(name, build) for name, build, _start, _bound in SPACES])
def test_under_the_unit_algebra_a_rule_costs_the_symbols_it_writes(name, build):
    """One per symbol: the per-rule cost is the size table's step, rule for rule.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
    """
    space = build()
    rules = [rule for nonterminal in space.nonterminals() for rule in space.get(nonterminal) or ()]
    assert rules, "a space without rules would make the comparison vacuous"
    for rule in rules:
        assert rule_cost(rule, UNIT) == _added_symbols(rule), (name, rule.terminal)


def test_a_constant_argument_is_charged_as_the_symbol_it_writes():
    """The digit a ``tag`` clause fixes is a leaf of the term, and the algebra charges it.

    The two-symbol space is the one whose clauses carry a constant argument, so this is where the
    constant half of the per-rule cost is exercised at all.
    """
    space = two_symbol_clause_space()
    tagged = [
        rule
        for nonterminal in space.nonterminals()
        for rule in space.get(nonterminal) or ()
        if any(isinstance(argument, ConstantArgument) for argument in rule.arguments)
    ]
    assert tagged, "the space was chosen for its constant arguments"
    for rule in tagged:
        (digit,) = (argument.value for argument in rule.arguments if isinstance(argument, ConstantArgument))
        assert rule_cost(rule, WEIGHTED) == COMBINATOR_COSTS["tag"] + digit + 1


@pytest.mark.parametrize(("name", "build", "start", "bound"), SPACES)
def test_a_rule_costs_what_its_expansion_adds_and_a_derivation_sums_to_the_fold(name, build, start, bound):
    """Every expansion adds the rule's cost to ``g``, and the rules of a derivation sum to the fold.

    The expansion is the engine's: the root's children are the goals of the applicable clauses, an
    inner node's are ``Goal.update`` at the deepest open subgoal. Each child is checked against
    ``cost_so_far``, which reads the partial inhabitant, and each finished term against ``fold``.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The queried non-terminal.
        bound (int): The term size below which every expansion is checked.
    """
    space = build()
    pending: list[tuple[Goal, float]] = []
    for rule in space.get(start) or ():
        goal = Goal.from_rhs_rule(rule)
        if goal is None:
            continue
        paid = rule_cost(rule, WEIGHTED)
        assert WEIGHTED.cost_so_far(goal) == paid, (name, rule.terminal)
        pending.append((goal, paid))

    expansions = 0
    finished = 0
    while pending:
        goal, paid = pending.pop()
        if goal.success:
            assert WEIGHTED.fold(goal.grounded[()][1]) == paid, name
            finished += 1
            continue
        if term_size(partial_inhabitant(goal)) > bound:
            continue
        position, argument = deepest_first_subgoal(goal)
        for rule in space.get(argument.origin) or ():
            child = goal.update(rule, position)
            if child is None:
                continue
            step = rule_cost(rule, WEIGHTED)
            assert WEIGHTED.cost_so_far(child) == WEIGHTED.cost_so_far(goal) + step, (name, rule.terminal)
            pending.append((child, paid + step))
            expansions += 1

    assert expansions > 10, "too few expansions to say anything"
    assert finished > 5, "too few finished terms to say anything"


def test_every_constant_is_charged_wherever_it_stands():
    """Two constants on one clause, a constant after a hole, a constant between two: each is charged.

    The reference spaces fix at most one constant per clause, and always before any hole, so a cost
    that charged the first constant only, or the constants before the first hole, or equal digits
    once, would pass on them. Here each of those is wrong on some clause.
    """
    space = constants_space()
    shapes = set()
    for nonterminal in space.nonterminals():
        for rule in space.get(nonterminal) or ():
            digits = [argument.value for argument in rule.arguments if isinstance(argument, ConstantArgument)]
            expected = COMBINATOR_COSTS[rule.terminal.__name__] + sum(digit + 1 for digit in digits)
            assert rule_cost(rule, WEIGHTED) == expected, rule.terminal
            kinds = tuple(isinstance(argument, ConstantArgument) for argument in rule.arguments)
            shapes.add(kinds)
    assert (True, True) in shapes, "two constants on one clause"
    assert (False, True) in shapes, "a constant after a hole"
    assert (False, True, False) in shapes, "a constant between two holes"


@pytest.mark.parametrize(("name", "build"), [(name, build) for name, build, _start, _bound in SPACES])
def test_a_hole_is_charged_nothing_whatever_the_assignment_estimates_for_it(name, build):
    """The variable assignment estimates what a hole's fillers will cost; opening the hole costs nothing.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
    """
    space = build()
    with_holes = 0
    for nonterminal in space.nonterminals():
        for rule in space.get(nonterminal) or ():
            assert rule_cost(rule, ESTIMATING) == rule_cost(rule, WEIGHTED), (name, rule.terminal)
            with_holes += any(isinstance(argument, NonTerminalArgument) for argument in rule.arguments)
    assert with_holes, "a space without holes would make the comparison vacuous"


def test_the_cost_is_summed_in_the_algebras_domain():
    """Over pairs, componentwise: the sum is the domain's, not Python's."""
    pairs = AdditiveCostAlgebra(ComponentwiseTuples(2), lambda symbol: (1.0, float(weighted_symbol_cost(symbol))))
    space = constants_space()
    for nonterminal in space.nonterminals():
        for rule in space.get(nonterminal) or ():
            digits = [argument.value for argument in rule.arguments if isinstance(argument, ConstantArgument)]
            expected = (1.0 + len(digits), float(COMBINATOR_COSTS[rule.terminal.__name__] + sum(d + 1 for d in digits)))
            assert rule_cost(rule, pairs) == expected, rule.terminal


def test_a_symbol_cost_outside_the_domain_is_refused():
    """A negative symbol cost would make ``g`` fall along a branch, so it is refused where it enters."""
    space = list_space()
    negative = AdditiveCostAlgebra(NonNegativeReals(), lambda _symbol: -1)
    rule = next(rule for nonterminal in space.nonterminals() for rule in space.get(nonterminal) or ())
    with pytest.raises(ValueError, match="the symbol cost returned -1"):
        rule_cost(rule, negative)


# ---------------------------------------------------------------------------------------------
# The cost table
# ---------------------------------------------------------------------------------------------

# The reference spaces the size table is checked on, with the cap each one is streamed to in full.
UNIT_SPACES = [
    ("list", list_space, LIST),
    ("expression", expression_space, EXPR),
    ("ambiguous", ambiguous_space, AMBIGUOUS_TARGET),
    ("two symbols per clause", two_symbol_clause_space, TAGGED),
    ("chain", chain_space, CHAIN),
]
FULL_STREAM_CAPS = {"list": 7, "expression": 7, "ambiguous": 4, "two symbols per clause": 9, "chain": 7}

# The recursive spaces the root row is checked on below a cap, under the weighted algebra. The third
# entry bounds the size of a term by its cost: a list of cost ``a`` has at most ``a`` conses, an
# expression at most ``a`` operators and so at most ``a + 1`` literals, a tagged term at most
# ``a / 2`` tags of two symbols each. Under that size bound the tree form counts every term below the
# cap, so its counts are an oracle for the table's. The round space has no such bound worth the tree
# form's cost and is checked against the cut expansion alone.
RECURSIVE_SPACES = [
    ("list", list_space, LIST, lambda cap: cap + 1),
    ("expression", expression_space, EXPR, lambda cap: 2 * cap + 1),
    ("two symbols per clause", two_symbol_clause_space, TAGGED, lambda cap: cap + 1),
    ("round", round_space, ROUND, None),
]

# The finite spaces the streams are compared on, with a size bound above every term they have.
FINITE_SPACES = [
    ("priced", priced_space, PRICED),
    ("hole tuples", hole_tuple_space, TUPLE_SORT),
]
SIZE_OF_EVERYTHING = 30


def uniform(_value):
    """Weight every realized cost value alike.

    Args:
        _value: The cost value. Ignored.

    Returns:
        float: One; the constructions normalize over the realized values.
    """
    return 1.0


def falling(value):
    """Weight a cost value by ``1 / (1 + a)``, a distribution that is not uniform.

    Args:
        value: The cost value.

    Returns:
        float: Its weight.
    """
    return 1.0 / (1.0 + value)


def assert_streams_agree(expected, actual):
    """Assert that two keyed streams coincide term for term and key for key.

    Args:
        expected (list): The keyed stream of the oracle.
        actual (list): The keyed stream of the cost table.
    """
    assert len(expected) == len(actual)
    assert [term for _, term in expected] == [term for _, term in actual]
    for (expected_key, _), (actual_key, _) in zip(expected, actual, strict=True):
        assert math.isclose(expected_key, actual_key, rel_tol=1e-12, abs_tol=1e-12)


def cut_expansion_counts(space, start, algebra, cap):
    """Count the terms below a cost cap by expanding the engine's goals and cutting at the cap.

    The oracle that shares nothing with the table's arithmetic: the retained derivation tree, cut
    where the cost so far passes the cap, which is exact because no cost comes back down. It
    terminates on a space without a loop of clauses of cost zero.

    Args:
        space (SolutionSpace): The program.
        start: The queried non-terminal.
        algebra (AdditiveCostAlgebra): The algebra.
        cap (int): The cost cap.

    Returns:
        dict[int, int]: The number of success branches per fold value up to the cap.
    """
    counts: dict[int, int] = {}
    pending = [goal for rule in space.get(start) or () if (goal := Goal.from_rhs_rule(rule)) is not None]
    while pending:
        goal = pending.pop()
        if algebra.cost_so_far(goal) > cap:
            continue
        if goal.success:
            value = int(algebra.fold(goal.grounded[()][1]))
            counts[value] = counts.get(value, 0) + 1
            continue
        position, argument = deepest_first_subgoal(goal)
        for rule in space.get(argument.origin) or ():
            child = goal.update(rule, position)
            if child is not None:
                pending.append(child)
    return counts


@pytest.mark.parametrize(("name", "build"), [(name, build) for name, build, _start in UNIT_SPACES])
@pytest.mark.parametrize("cap", [1, 2, 3, 5, 6])
def test_under_the_unit_algebra_the_cost_table_is_the_size_table(name, build, cap):
    """One per symbol makes the cost the size, and then the two tables are one, row for row.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        cap (int): The cap, which is then the size bound.
    """
    space = build()
    costs = cost_table(space, UNIT, cap)
    sizes = size_table(space, cap)
    assert set(costs.counts) == set(sizes.counts), name
    for nonterminal in sizes.counts:
        expected = {size: sizes.of(nonterminal, size) for size in range(cap + 1) if sizes.of(nonterminal, size)}
        assert dict(costs.counts[nonterminal]) == expected, (name, nonterminal)
    if cap >= 5:
        assert any(costs.counts.values()), "an empty table would make the comparison vacuous"


@pytest.mark.parametrize(("name", "build", "start"), UNIT_SPACES)
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_under_the_unit_algebra_the_cost_table_streams_what_the_size_table_streams(name, build, start, seed):
    """The same oracle from another table: the same stream, term for term and key for key.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The queried non-terminal.
        seed (int): The seed under test.
    """
    cap = FULL_STREAM_CAPS[name]
    query = generator_query(build(), start)
    sized = list(weighted_table(query, cap, uniform).keyed_stream(random.Random(seed)))
    costed = list(weighted_cost_table(query, UNIT, uniform, cap).keyed_stream(random.Random(seed)))
    assert sized, "an empty stream would make the comparison vacuous"
    assert_streams_agree(sized, costed)


@pytest.mark.parametrize(("name", "build", "start", "size_for"), RECURSIVE_SPACES)
@pytest.mark.parametrize("cap", [0, 1, 3, 4])
def test_below_the_cap_the_root_row_is_what_the_cut_expansion_and_the_tree_form_count(
    name, build, start, size_for, cap
):
    """Clauses of cost zero, a recursion, and the rows agree with two oracles below the cap.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The queried non-terminal.
        size_for (Callable | None): A size bound that covers every term below the cap, or None.
        cap (int): The cost cap.
    """
    space = build()
    table = cost_table(space, WEIGHTED, cap)
    expected = cut_expansion_counts(space, start, WEIGHTED, cap)
    assert dict(table.counts[start]) == expected, name
    if cap >= 3:
        assert len(expected) >= 2, "a row of one value would make the comparison weak"
    if size_for is not None:
        tree = branch_counts(generator_query(space, start), size_for(cap), WEIGHTED.fold)
        assert {int(value): count for value, count in tree.counts.items() if value <= cap} == expected, name


def test_a_loop_of_positive_cost_is_counted_value_by_value():
    """``Idle -> idle_again(Idle) | idle_halt`` with the loop charged: one term per cost value."""
    charged = AdditiveCostAlgebra(NonNegativeReals(), lambda symbol: 1 if getattr(symbol, "__name__", "") else 0)
    table = cost_table(idle_space(), charged, 4)
    assert dict(table.counts[IDLE]) == {1: 1, 2: 1, 3: 1, 4: 1}


@pytest.mark.parametrize(("name", "build", "start"), FINITE_SPACES)
@pytest.mark.parametrize("seed", [0, 1, 7, 23])
def test_on_a_finite_space_the_cost_table_streams_what_the_tree_form_streams(name, build, start, seed):
    """A cap above the dearest term: the same question as the tree form's, and the same stream.

    Under a distribution that is not uniform, so that the unit weights differ between cost values.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The queried non-terminal.
        seed (int): The seed under test.
    """
    query = generator_query(build(), start)
    eager = weighted_tree(query, SIZE_OF_EVERYTHING, WEIGHTED.fold, falling)
    wider = branch_counts(query, SIZE_OF_EVERYTHING + 10, WEIGHTED.fold)
    assert wider.counts == eager.root.counts, (name, "the size bound must hold every term of the finite space")
    dearest = int(max(eager.root.counts))
    assert len(eager.root.counts) >= 3, (name, "too few cost values to say anything")
    lazy = weighted_cost_table(query, WEIGHTED, falling, dearest)
    assert lazy.total == eager.root.total, name
    assert_streams_agree(
        list(eager.keyed_stream(random.Random(seed))),
        list(lazy.keyed_stream(random.Random(seed))),
    )


@pytest.mark.parametrize("cap", [0, 2, 4, 6])
def test_a_cap_below_the_dearest_term_streams_exactly_the_terms_within_it_once_each(cap):
    """The cap cuts the language, and the stream is the language below it, without repeats.

    Args:
        cap (int): The cost cap.
    """
    query = generator_query(priced_space(), PRICED)
    everything = list(weighted_tree(query, SIZE_OF_EVERYTHING, WEIGHTED.fold, uniform).stream(random.Random(0)))
    within = {term for term in everything if WEIGHTED.fold(term) <= cap}
    streamed = list(weighted_cost_table(query, WEIGHTED, uniform, cap).stream(random.Random(3)))
    assert set(streamed) == within
    assert len(streamed) == len(within)
    assert within, "an empty language below the cap would make the comparison vacuous"


def test_a_partial_term_is_completed_as_the_tree_form_completes_it_at_every_position():
    """The prescribed symbols are charged, at the root, the leaves, the literals and everything between.

    The cost of the prescribed part counts against the cap in both constructions, and both start
    from the goals ``goal_from_tree`` derives, so the residual streams agree position by position.
    """
    space = priced_space()
    eager_all = weighted_tree(generator_query(space, PRICED), SIZE_OF_EVERYTHING, WEIGHTED.fold, uniform)
    dearest = int(max(eager_all.root.counts))
    parent = max(eager_all.stream(random.Random(5)), key=lambda term: (term.depth, str(term)))
    positions = sorted(parent.positions())
    assert len(positions) >= 4, "a shallow parent would leave the deep positions untested"
    assert any(not callable(parent.subtree_at(position).root) for position in positions), "no literal leaf"
    for position in positions:
        query = residual_query(space, PRICED, parent, position)
        eager = weighted_tree(query, SIZE_OF_EVERYTHING, WEIGHTED.fold, falling)
        lazy = weighted_cost_table(query, WEIGHTED, falling, dearest)
        assert lazy.total == eager.root.total > 0, position
        assert_streams_agree(
            list(eager.keyed_stream(random.Random(11))),
            list(lazy.keyed_stream(random.Random(11))),
        )


def test_clauses_of_cost_zero_that_close_a_cycle_are_refused_and_named():
    """``idle_again`` costs nothing and opens its own sort: infinitely many terms of one cost."""
    with pytest.raises(ValueError, match="clauses of cost zero close a cycle") as refused:
        cost_table(idle_space(), WEIGHTED, 5)
    assert "idle_again" in str(refused.value)


def test_a_cost_that_is_not_a_whole_number_is_refused():
    """A half is not rounded: a coarser algebra is a different algebra and the caller's to choose."""
    halves = AdditiveCostAlgebra(NonNegativeReals(), lambda _symbol: 0.5)
    with pytest.raises(ValueError, match="takes whole-number costs"):
        cost_table(list_space(), halves, 4)


def test_a_negative_cap_is_refused():
    """A cost value below zero is a caller's mistake, in the table and in the search alike."""
    with pytest.raises(ValueError, match="the cost cap is a cost value"):
        cost_table(list_space(), WEIGHTED, -1)
    with pytest.raises(ValueError, match="the cost cap is a cost value"):
        weighted_cost_table(generator_query(list_space(), LIST), WEIGHTED, uniform, -1)


def test_a_program_whose_predicate_reads_a_hole_is_refused():
    """The holes of a clause must be filled independently, or the product overcounts."""
    with pytest.raises(ValueError, match="reading a hole in a predicate"):
        cost_table(cut_space(), WEIGHTED, 4)
    with pytest.raises(ValueError, match="reading a hole in a predicate"):
        weighted_cost_table(generator_query(cut_space(), BOX), WEIGHTED, uniform, 4)


def test_a_table_handed_in_must_be_filled_under_the_same_algebra_to_at_least_the_cap():
    """A prebuilt table is no way around the algebra or the cap."""
    space = list_space()
    query = generator_query(space, LIST)
    other = AdditiveCostAlgebra(NonNegativeReals(), weighted_symbol_cost)
    with pytest.raises(ValueError, match="filled under another algebra"):
        weighted_cost_table(query, WEIGHTED, uniform, 3, table=cost_table(space, other, 3))
    with pytest.raises(ValueError, match="filled to the cost cap 2"):
        weighted_cost_table(query, WEIGHTED, uniform, 3, table=cost_table(space, WEIGHTED, 2))
    shared = cost_table(space, WEIGHTED, 5)
    assert isinstance(shared, CostTable)
    assert list(weighted_cost_table(query, WEIGHTED, uniform, 5, table=shared).stream(random.Random(2))) == list(
        weighted_cost_table(query, WEIGHTED, uniform, 5).stream(random.Random(2))
    )


def test_the_table_reads_zero_outside_what_it_holds():
    """Outside the cap, for an unknown non-terminal, and for a split beyond the cap."""
    table = cost_table(list_space(), WEIGHTED, 3)
    assert table.of(LIST, 0) == 1
    assert table.of(LIST, 4) == 0
    assert table.of(LIST, -1) == 0
    assert table.of("no such sort", 0) == 0
    assert table.split_counts((), 0) == 1
    assert table.split_counts((LIST, LIST), 0) == 1
    assert table.split_counts((LIST, LIST), 4) == 0


def test_the_sampler_streams_the_weighted_table_and_knows_how_many_terms_it_holds():
    """The ``Sampler`` a pipeline consumes: the same stream, an exact count, a refusal of a negative cap."""
    query = generator_query(priced_space(), PRICED)
    sampler = CostTableSampler(WEIGHTED, falling, 6, random.Random(9))
    assert isinstance(sampler, Sampler)
    total = weighted_cost_table(query, WEIGHTED, falling, 6).total
    assert total > 0
    assert sampler.at_least(query, total)
    assert not sampler.at_least(query, total + 1)
    assert sampler.at_least(query, 0)
    expected = list(weighted_cost_table(query, WEIGHTED, falling, 6).stream(random.Random(9)))
    assert list(sampler.sample(query)) == expected
    sampler.forget()
    with pytest.raises(ValueError, match="the cost cap is a cost value"):
        CostTableSampler(WEIGHTED, falling, -1, random.Random(0))


def test_a_cap_below_the_cheapest_term_gives_an_empty_stream():
    """Emptiness below the cap is a legitimate answer, not an error."""
    charged = AdditiveCostAlgebra(NonNegativeReals(), lambda symbol: 1 if getattr(symbol, "__name__", "") else 0)
    query = generator_query(idle_space(), IDLE)
    lazy = weighted_cost_table(query, charged, uniform, 0)
    assert lazy.total == 0
    assert list(lazy.stream(random.Random(0))) == []
    assert not CostTableSampler(charged, uniform, 0, random.Random(0)).at_least(query, 1)
    # the control: one cost value up, the cheapest term is there, and one more up, the next
    assert weighted_cost_table(query, charged, uniform, 1).total == 1
    assert weighted_cost_table(query, charged, uniform, 2).total == 2
    assert CostTableSampler(charged, uniform, 2, random.Random(0)).at_least(query, 2)


# ---------------------------------------------------------------------------------------------
# What the reviews of the cost table found unexercised
# ---------------------------------------------------------------------------------------------


def by_name(costs, literal=lambda value: value + 1):
    """Build an algebra that charges each combinator its entry and a literal ``literal(value)``.

    Args:
        costs (dict): The combinators' costs, by function name.
        literal (Callable): The cost of a literal value.

    Returns:
        AdditiveCostAlgebra: The algebra.
    """

    def symbol_cost(symbol):
        name = getattr(symbol, "__name__", None)
        return costs[name] if name is not None else literal(symbol)

    return AdditiveCostAlgebra(NonNegativeReals(), symbol_cost)


# The spaces every row of the table is checked on, each with the algebra that reaches the branch it
# was built for: the shared first holes of the priced space, a recursion read at the value it fills
# from a single hole and from a clause of two holes, three different tuples split in one component,
# a component of three members, two ternary tails of one length, and a clause whose hole no clause
# fills.
EVERY_ROW = [
    ("priced", priced_space, WEIGHTED, 6, 3),
    # a cap below several leaves: a clause dearer than the cap contributes nothing, not a value above it
    ("priced, a low cap", priced_space, WEIGHTED, 1, 3),
    (
        "round, the order reversed",
        round_space,
        by_name({"round_stop": 1, "round_wrap": 1, "round_pair": 1, "inner_leaf": 1, "inner_step": 0}),
        5,
        3,
    ),
    (
        "pair edge",
        pair_edge_space,
        by_name({"outer_stop": 0, "outer_wrap": 1, "inner_leaf_pair": 1, "inner_pair": 0}),
        6,
        3,
    ),
    (
        "split",
        split_space,
        by_name(
            {"split_leaf": 1, "split_sum": 1, "split_scale": 1, "split_rescale": 2, "scalar_one": 1, "scalar_two": 2}
        ),
        6,
        3,
    ),
    ("three cycle", three_cycle_space, by_name({"cycle_leaf": 1, "cycle_ab": 1, "cycle_bc": 1, "cycle_ca": 1}), 9, 3),
    (
        "ternary tails",
        ternary_tails_space,
        by_name({"tail_a0": 0, "tail_a1": 1, "tail_b0": 3, "tails_same": 1, "tails_mixed": 1}),
        8,
        3,
    ),
    (
        "a loop that pays for its side",
        loop_space,
        by_name({"loop_leaf": 1, "loop_fold": 0, "loop_side": 1, "loop_there": 1, "loop_back": 1}),
        5,
        3,
    ),
    # one term in all: the point is the clause whose hole nothing fills, counted as nothing
    ("hollow", hollow_space, by_name({"hollow_leaf": 1, "hollow_needs": 0}), 3, 1),
]


@pytest.mark.parametrize(("name", "build", "algebra", "cap", "at_least"), EVERY_ROW)
def test_every_row_is_what_the_cut_expansion_counts_from_its_sort(name, build, algebra, cap, at_least):
    """Not the start row alone: every non-terminal's row, against the engine's own expansion from it.

    Args:
        name (str): The case's name, for the test id.
        build (Callable): Builds the space.
        algebra (AdditiveCostAlgebra): The algebra that reaches the case's branch.
        cap (int): The cost cap.
        at_least (int): How many realized values the case must have for the comparison to say anything.
    """
    space = build()
    table = cost_table(space, algebra, cap)
    realized = 0
    for nonterminal in space.nonterminals():
        expected = cut_expansion_counts(space, nonterminal, algebra, cap)
        assert dict(table.counts[nonterminal]) == expected, (name, nonterminal)
        assert all(value <= cap for value in table.counts[nonterminal]), (name, nonterminal)
        realized += len(expected)
    assert realized >= at_least, (name, "too few realized values to say anything")


@pytest.mark.parametrize(
    ("name", "build"),
    [
        ("priced", priced_space),
        ("round", round_space),
        ("pair edge", pair_edge_space),
        ("split", split_space),
        ("three cycle", three_cycle_space),
        ("ternary tails", ternary_tails_space),
        ("literal predicate", literal_predicate_space),
    ],
)
@pytest.mark.parametrize("cap", [3, 5, 8])
def test_under_the_unit_algebra_every_other_space_is_the_size_table_too(name, build, cap):
    """The grouping, the literal-only predicates and the cyclic fills, held to the size table.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        cap (int): The cap, which is then the size bound.
    """
    space = build()
    costs = cost_table(space, UNIT, cap)
    sizes = size_table(space, cap)
    for nonterminal in sizes.counts:
        expected = {size: sizes.of(nonterminal, size) for size in range(cap + 1) if sizes.of(nonterminal, size)}
        assert dict(costs.counts[nonterminal]) == expected, (name, nonterminal)


def test_the_priced_space_groups_clauses_by_cost_and_first_hole():
    """The weighted algebra gives the clauses that share a first hole one cost, so the fill groups them."""
    space = priced_space()
    shared = {}
    for nonterminal in space.nonterminals():
        for rule in space.get(nonterminal) or ():
            holes_of = tuple(
                argument.origin for argument in rule.arguments if isinstance(argument, NonTerminalArgument)
            )
            if len(holes_of) >= 2:
                key = (nonterminal, rule_cost(rule, WEIGHTED), holes_of[0])
                shared.setdefault(key, []).append(holes_of[1:])
    tails = [tuple(group) for group in shared.values() if len(group) > 1]
    assert tails, "no two clauses share their cost and their first hole"
    assert any(len(set(group)) < len(group) for group in tails), "no tail is repeated within a group"
    assert any(len(set(group)) > 1 for group in tails), "no group has two different tails"


@pytest.mark.parametrize(
    ("algebra", "named"),
    [
        # through a clause's second hole, its first a side that costs nothing
        (by_name({"loop_leaf": 1, "loop_fold": 0, "loop_side": 0, "loop_there": 1, "loop_back": 1}), ["loop_fold"]),
        # the same, with the loop's own sort free too, so that every hole of the clause is a free edge
        (by_name({"loop_leaf": 0, "loop_fold": 0, "loop_side": 0, "loop_there": 1, "loop_back": 1}), ["loop_fold"]),
        # through two clauses of one hole each
        (
            by_name({"loop_leaf": 1, "loop_fold": 1, "loop_side": 0, "loop_there": 0, "loop_back": 0}),
            ["loop_there", "loop_back"],
        ),
    ],
)
def test_a_loop_that_pumps_without_paying_is_refused_and_named(algebra, named):
    """Through a clause of two holes whose side costs nothing, and through two clauses of one hole each.

    Args:
        algebra (AdditiveCostAlgebra): The algebra that closes the loop at no cost.
        named (list[str]): The clauses the message must name.
    """
    with pytest.raises(ValueError, match="close a cycle whose other holes can be filled at cost zero") as refused:
        cost_table(loop_space(), algebra, 4)
    for clause in named:
        assert clause in str(refused.value), clause


def test_a_cap_is_a_whole_number_and_a_whole_float_is_one():
    """A fold of the algebra is a float; four point oh is four, four and a half is refused."""
    query = generator_query(list_space(), LIST)
    as_int = list(weighted_cost_table(query, WEIGHTED, uniform, 4).keyed_stream(random.Random(1)))
    as_float = list(weighted_cost_table(query, WEIGHTED, uniform, 4.0).keyed_stream(random.Random(1)))
    assert as_int
    assert_streams_agree(as_int, as_float)
    assert CostTableSampler(WEIGHTED, uniform, 4.0, random.Random(0)).at_least(query, 1)
    for bad in (4.5, float("inf"), True, "4"):
        with pytest.raises(ValueError, match="must be a nonnegative whole number"):
            weighted_cost_table(query, WEIGHTED, uniform, bad)
        with pytest.raises(ValueError, match="must be a nonnegative whole number"):
            cost_table(list_space(), WEIGHTED, bad)


def test_a_table_filled_for_another_program_is_refused():
    """The rows are indexed by one program's non-terminals; another program's query reads them wrongly."""
    one, other = list_space(), list_space()
    with pytest.raises(ValueError, match="filled for another program"):
        weighted_cost_table(generator_query(other, LIST), WEIGHTED, uniform, 3, table=cost_table(one, WEIGHTED, 3))


def test_a_table_filled_to_a_higher_cap_is_read_up_to_the_searchs():
    """The branch counts stop at the search's cap, not the table's: the same keys as a table of its own."""
    space = list_space()
    query = generator_query(space, LIST)
    wide = cost_table(space, WEIGHTED, 8)
    for seed in (0, 1, 2):
        assert_streams_agree(
            list(weighted_cost_table(query, WEIGHTED, falling, 4).keyed_stream(random.Random(seed))),
            list(weighted_cost_table(query, WEIGHTED, falling, 4, table=wide).keyed_stream(random.Random(seed))),
        )


def test_the_table_reads_the_cap_itself():
    """The cap is included: a term of exactly the cap's cost is counted."""
    table = cost_table(list_space(), WEIGHTED, 3)
    assert table.of(LIST, 3) > 0
    assert table.split_counts((LIST,), 3) == table.of(LIST, 3)
    assert table.split_counts((LIST, LIST), 3) > 0


def test_a_product_skips_a_value_above_the_cap_wherever_it_stands():
    """A row being built is not sorted, so a value above the cap may come first."""
    assert _product({5: 1, 0: 1}, {0: 1, 1: 1}, 2) == {0: 1, 1: 1}


def test_the_root_weight_is_the_sum_of_its_childrens():
    """The root reads the root row, its children the split rows; the two must agree in log space."""
    query = generator_query(priced_space(), PRICED)
    lazy = weighted_cost_table(query, WEIGHTED, falling, 5)
    children = [lazy.log_weight_of(goal, cost) for goal, cost in _initial_cost_nodes(query, WEIGHTED) if cost <= 5]
    assert len(children) >= 3
    assert log_sum_exp(children) == pytest.approx(lazy.log_weight_of(None, 0), abs=1e-12)


def test_the_sampler_counts_again_for_another_query_and_forgets_on_request(monkeypatch):
    """One fill per query in a row, another for another query, another after ``forget``."""
    module = cost_tables_module

    space = priced_space()
    whole, part = generator_query(space, PRICED), generator_query(space, PRICED_Q)
    expected_part = set(weighted_cost_table(part, WEIGHTED, falling, 6).stream(random.Random(0)))
    fills = []
    original = module.cost_table

    def counted(*args, **kwargs):
        fills.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "cost_table", counted)
    sampler = CostTableSampler(WEIGHTED, falling, 6, random.Random(5))
    assert sampler.at_least(whole, 1)
    first = list(sampler.sample(whole))
    assert len(fills) == 1, "asking and then drawing counts once"
    in_part = list(sampler.sample(part))
    assert len(fills) == 2, "another query is counted again"
    assert set(in_part) == expected_part
    assert set(in_part) != set(first)
    sampler.forget()
    list(sampler.sample(part))
    assert len(fills) == 3, "forgotten, the same query is counted again"


def test_the_sampler_draws_from_its_own_source_of_randomness():
    query = generator_query(priced_space(), PRICED)
    drawn = list(CostTableSampler(WEIGHTED, falling, 6, random.Random(5)).sample(query))
    assert drawn == list(weighted_cost_table(query, WEIGHTED, falling, 6).stream(random.Random(5)))
    assert drawn != list(weighted_cost_table(query, WEIGHTED, falling, 6).stream(random.Random(6)))


def test_a_whole_clause_of_fractional_symbols_is_refused():
    """A terminal at 0.1 and a digit at 0.9 make a whole clause, but the fold adds the floats in another order."""
    fractional = by_name({"stop": 1, "tag": 0.1}, literal=lambda _value: 0.9)
    with pytest.raises(ValueError, match="takes whole-number costs"):
        cost_table(two_symbol_clause_space(), fractional, 4)


def test_a_long_tuple_of_holes_is_split_without_the_stack():
    """Three thousand holes are a loop of products, not a recursion three thousand deep."""
    table = cost_table(list_space(), WEIGHTED, 3)
    assert table.split_counts((LIST,) * 3000, 0) == 1
    assert table.split_counts((LIST,) * 3000, 1) == 3000


# ---------------------------------------------------------------------------------------------
# The product of two rows: pair by pair, or packed into two integers and multiplied once
# ---------------------------------------------------------------------------------------------


def product_by_definition(left, right, cap):
    """Multiply two rows by the definition: every pair, no ordering, no early exit.

    Args:
        left (dict): One row.
        right (dict): The other row.
        cap (int): The largest cost value kept.

    Returns:
        dict: The nonzero coefficients up to the cap.
    """
    out = {}
    for first, first_count in left.items():
        for second, second_count in right.items():
            if first + second <= cap:
                out[first + second] = out.get(first + second, 0) + first_count * second_count
    return {value: count for value, count in out.items() if count}


def random_row(rng, entries, span, digits):
    """Draw a row: ``entries`` distinct cost values up to ``span``, counts below ``10 ** digits``.

    Args:
        rng (random.Random): The source of randomness.
        entries (int): How many values the row realizes.
        span (int): The largest value it may realize.
        digits (int): The number of decimal digits its counts may have.

    Returns:
        dict: The row.
    """
    values = rng.sample(range(span + 1), min(entries, span + 1))
    return {value: rng.randrange(1, 10**digits) for value in values}


@pytest.mark.parametrize("seed", range(30))
def test_both_products_are_the_product_on_any_two_rows(seed):
    """Sparse and dense, small counts and counts of two hundred digits, caps inside and beyond the spans."""
    rng = random.Random(seed)
    for _ in range(8):
        span = rng.choice([0, 1, 7, 60, 400])
        cap = rng.choice([0, span // 2, span, 2 * span, 3 * span + 1])
        left = random_row(rng, rng.randint(1, span + 1), span, rng.choice([1, 20, 60, 200]))
        right = random_row(rng, rng.randint(1, span + 1), rng.choice([span, span // 3 + 1]), rng.choice([1, 20, 60]))
        expected = product_by_definition(left, right, cap)
        assert _packed_product(left, right, cap) == expected, (seed, span, cap)
        assert _dict_product(left, right, cap) == expected, (seed, span, cap)
        assert _product(left, right, cap) == expected, (seed, span, cap)


def test_the_packed_product_of_nothing_is_nothing():
    """An empty row, a row entirely above the cap, and a single value zero."""
    assert _packed_product({}, {0: 1}, 5) == {}
    assert _packed_product({7: 3}, {1: 1}, 5) == {}
    assert _packed_product({0: 1}, {0: 1}, 0) == {0: 1}
    assert _packed_product({0: 2, 3: 5}, {0: 7}, 2) == {0: 14}


def test_the_product_packs_dense_wide_rows_and_walks_sparse_ones(monkeypatch):
    """The packed product wins where the rows are dense and wide, and loses badly where they are sparse and far apart."""
    paths = []
    monkeypatch.setattr(cost_tables_module, "_packed_product", lambda *a: paths.append("packed") or _packed_product(*a))
    monkeypatch.setattr(cost_tables_module, "_dict_product", lambda *a: paths.append("walked") or _dict_product(*a))
    dense = {value: value + 1 for value in range(2000)}
    sparse = {value * 10_000: 1 for value in range(30)}
    assert _product(dense, dense, 4000) == product_by_definition(dense, dense, 4000)
    assert _product(sparse, sparse, 1_000_000) == product_by_definition(sparse, sparse, 1_000_000)
    assert paths == ["packed", "walked"]


@pytest.mark.parametrize(
    ("left_bits", "right_bits", "length_bits", "cap"),
    [
        # 4 + 4 + 8 bits: two bytes a slot, and the cap keeps the slot the largest coefficient fills to its top bit
        (4, 4, 8, 254),
        # 4 + 4 + 9 bits: one bit past two bytes, so a slot one bit narrower overflows
        (4, 4, 9, 1020),
        # the same total from unequal counts, so neither count's bits can be spared
        (5, 3, 9, 1020),
        (3, 5, 9, 1020),
    ],
)
def test_the_packed_product_is_exact_where_a_coefficient_fills_its_slot(left_bits, right_bits, length_bits, cap):
    """Every count at the most its bits hold and every row as long as its length's bits allow: the largest
    coefficient, ``(2**length_bits - 1) * (2**left_bits - 1) * (2**right_bits - 1)``, needs every bit of the bound.

    Args:
        left_bits (int): The bit length of the left row's counts.
        right_bits (int): The bit length of the right row's counts.
        length_bits (int): The bit length of the rows' length.
        cap (int): The largest cost value kept.
    """
    length = 2**length_bits - 1
    left = dict.fromkeys(range(length), 2**left_bits - 1)
    right = dict.fromkeys(range(length), 2**right_bits - 1)
    largest = max(product_by_definition(left, right, cap).values())
    assert largest.bit_length() == left_bits + right_bits + length_bits
    assert _packed_product(left, right, cap) == product_by_definition(left, right, cap)


def product_chosen(monkeypatch, left, right, cap):
    """Which product ``_product`` takes for two rows, neither of them computed.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces both products by recorders.
        left (dict): One row.
        right (dict): The other row.
        cap (int): The largest cost value kept.

    Returns:
        str: ``"packed"``, ``"walked"`` or ``"neither"``.
    """
    chosen = []
    monkeypatch.setattr(cost_tables_module, "_packed_product", lambda *_: chosen.append("packed") or {})
    monkeypatch.setattr(cost_tables_module, "_dict_product", lambda *_: chosen.append("walked") or {})
    _product(left, right, cap)
    monkeypatch.undo()
    assert len(chosen) <= 1
    return chosen[0] if chosen else "neither"


def test_the_product_weighs_the_pairs_the_walk_multiplies_and_the_counts_it_multiplies(monkeypatch):
    """The walk stops where a sum passes the cap, and a pair of long counts costs more than a pair of short ones."""
    rng = random.Random(3)
    ending_at_the_cap = {value: rng.getrandbits(110) | 1 for value in range(88_001, 100_001)}
    nearly_past_it = {value: rng.getrandbits(60) | 1 for value in range(4_950, 10_001)}
    sparse = {value * 15_625: 1 for value in range(64)}
    mostly_above = {1: 5, 2: 7} | dict.fromkeys(range(10**7 + 1, 10**7 + 100_001), 3)
    long_counts = {value: rng.getrandbits(1500) | 1 for value in range(500)}
    dense = {value: value + 1 for value in range(3000)}
    a_long_count_above = dense | {3001: 10**3000}
    a_far_value_above = dense | {10**7: 1}
    # every sum past the cap: nothing to walk, whatever the rows' lengths
    assert product_chosen(monkeypatch, ending_at_the_cap, ending_at_the_cap, 100_000) == "walked"
    # the sums reach the cap only from the rows' first hundred values
    assert product_chosen(monkeypatch, nearly_past_it, nearly_past_it, 10_000) == "walked"
    # past the least number of pairs worth weighing, and spread over a million slots
    assert product_chosen(monkeypatch, sparse, sparse, 10**6) == "walked"
    # the values above the cap are neither walked nor packed
    assert product_chosen(monkeypatch, mostly_above, mostly_above, 10**7) == "walked"
    # counts of 1 500 bits: a quarter of a million pairs cost more walked than packed
    assert product_chosen(monkeypatch, long_counts, long_counts, 1000) == "packed"
    # a count above the cap widens no slot, and a value above it lengthens no integer
    assert product_chosen(monkeypatch, a_long_count_above, dict.fromkeys(range(3000), 1), 3000) == "packed"
    assert product_chosen(monkeypatch, a_far_value_above, a_far_value_above, 3000) == "packed"


def test_a_negative_cap_multiplies_to_nothing_on_either_path():
    """Below the least number of pairs worth weighing and past it, as the pair-by-pair product has it."""
    wide = dict.fromkeys(range(100), 1)
    assert _product({0: 1}, {0: 1}, -1) == _dict_product({0: 1}, {0: 1}, -1) == {}
    assert _product(wide, wide, -5) == _dict_product(wide, wide, -5) == {}


def test_the_table_is_the_same_whichever_product_fills_it(monkeypatch):
    """Every row and every split of two or three holes, filled and split by the packed product alone and by the
    pair-by-pair product alone, on every space of EVERY_ROW; the packed product reaches all three places that
    multiply rows."""
    callers = set()

    def packed_only(left, right, cap):
        frame = inspect.currentframe()
        callers.add(frame.f_back.f_code.co_name if frame is not None and frame.f_back is not None else "")
        return _packed_product(left, right, cap)

    compared = 0
    for name, build, algebra, cap, _at_least in EVERY_ROW:
        space = build()
        monkeypatch.setattr(cost_tables_module, "_product", _dict_product)
        walked = cost_table(space, algebra, cap)
        monkeypatch.setattr(cost_tables_module, "_product", packed_only)
        packed = cost_table(space, algebra, cap)
        assert {nt: dict(row) for nt, row in packed.counts.items()} == {
            nt: dict(row) for nt, row in walked.counts.items()
        }, name
        nonterminals = list(space.nonterminals())
        for hole_types in itertools.chain(
            itertools.product(nonterminals, repeat=2), itertools.product(nonterminals, repeat=3)
        ):
            monkeypatch.setattr(cost_tables_module, "_product", _dict_product)
            expected = dict(walked.split_row(hole_types))
            monkeypatch.setattr(cost_tables_module, "_product", packed_only)
            assert dict(packed.split_row(hole_types)) == expected, (name, hole_types)
            compared += 1
        monkeypatch.undo()
    assert compared > 100
    assert callers == {"_acyclic_row", "tuple_row", "split_row"}


# ---------------------------------------------------------------------------------------------
# The lazy frontier: a child is built when it is popped, not when its parent is expanded
# ---------------------------------------------------------------------------------------------

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


def counted_updates(monkeypatch):
    """Count the goals the engine builds from here on.

    Args:
        monkeypatch: pytest's monkeypatch.

    Returns:
        list: One entry per call of ``Goal.update``.
    """
    calls = []
    update = Goal.update

    def counting(self, rule, position):
        calls.append(rule.terminal)
        return update(self, rule, position)

    monkeypatch.setattr(Goal, "update", counting)
    return calls


@pytest.mark.parametrize("form", ["size table", "cost table"])
def test_a_child_is_built_when_it_is_popped_and_not_before(monkeypatch, form):
    """Thirty children per expansion, and the first draw builds one goal with ``Goal.update``, not thirty.

    The first draw pops the root, one of its thirty children, and one of that child's thirty. The root's
    children are built from the start symbol's clauses without ``Goal.update``; a frontier that built
    every child it pushed would then have called it thirty times, once per grandchild, where the lazy
    one builds only the grandchild it pops.

    Args:
        monkeypatch: pytest's monkeypatch.
        form (str): Which table form draws.
    """
    query = generator_query(wide_space(), WIDE)
    lazy = weighted_table(query, 3, uniform) if form == "size table" else weighted_cost_table(query, UNIT, uniform, 3)
    calls = counted_updates(monkeypatch)
    first = next(lazy.stream(random.Random(0)))
    assert first is not None
    assert len(calls) <= 2, (form, len(calls))


@pytest.mark.parametrize("form", ["size table", "cost table"])
def test_a_child_the_engine_refuses_when_it_is_built_is_an_error_not_a_skip(monkeypatch, form):
    """The weight a lazy child was pushed with assumed the engine applies its clause; if it does not, say so.

    Args:
        monkeypatch: pytest's monkeypatch.
        form (str): Which table form draws.
    """
    query = generator_query(wide_space(), WIDE)
    lazy = weighted_table(query, 3, uniform) if form == "size table" else weighted_cost_table(query, UNIT, uniform, 3)
    monkeypatch.setattr(Goal, "update", lambda self, rule, position: None)
    with pytest.raises(ValueError, match="refused a clause the lazy frontier had weighed"):
        next(lazy.stream(random.Random(0)))


def shallowest_last_hole(goal):
    """A computation rule other than the engine's: the shallowest open hole, the last of them among equals.

    Args:
        goal (Goal): The search node, not a success node.

    Returns:
        tuple: The position and the argument there.
    """
    position = max(holes(goal), key=lambda path: (-len(path), path))
    return position, goal.subgoals[position]


def an_expanded_position_first(goal):
    """A rule the table forms cannot follow: an expanded position, while one is still a subgoal.

    ``Goal.subgoals`` keeps an expanded position until its subtree grounds, so a rule reading it
    directly can select one. The engine's rule never does.

    Args:
        goal (Goal): The search node, not a success node.

    Returns:
        tuple: The position and the argument there.
    """
    expanded = [position for position in goal.subgoals if position in goal.constructors]
    position = expanded[0] if expanded else deepest_first_subgoal(goal)[0]
    return position, goal.subgoals[position]


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_both_table_forms_follow_a_computation_rule_other_than_the_engines(seed):
    """The rule decides which hole a node expands, and so the frontier's children and the stream's order.

    Both table forms under the rule against the tree form under the rule, key for key; and, so that the
    comparison says something, the rule changes the order of the stream on this space.

    Args:
        seed (int): The seed under test.
    """
    query = generator_query(priced_space(), PRICED)
    by_cost = weighted_tree(query, SIZE_OF_EVERYTHING, WEIGHTED.fold, falling, subgoal_selection=shallowest_last_hole)
    by_size = weighted_tree(query, SIZE_OF_EVERYTHING, term_size, falling, subgoal_selection=shallowest_last_hole)
    dearest = int(max(by_cost.root.counts))
    cost_form = weighted_cost_table(query, WEIGHTED, falling, dearest, subgoal_selection=shallowest_last_hole)
    size_form = weighted_table(query, SIZE_OF_EVERYTHING, falling, subgoal_selection=shallowest_last_hole)
    under_the_rule = list(cost_form.keyed_stream(random.Random(seed)))
    assert_streams_agree(list(by_cost.keyed_stream(random.Random(seed))), under_the_rule)
    assert_streams_agree(
        list(by_size.keyed_stream(random.Random(seed))), list(size_form.keyed_stream(random.Random(seed)))
    )
    engines = list(weighted_cost_table(query, WEIGHTED, falling, dearest).keyed_stream(random.Random(seed)))
    assert sorted(map(str, (term for _, term in engines))) == sorted(map(str, (term for _, term in under_the_rule)))
    assert [term for _, term in engines] != [term for _, term in under_the_rule]


@pytest.mark.parametrize("form", ["size table", "cost table"])
def test_a_rule_that_selects_an_expanded_position_is_refused_by_name(form):
    """A table weighs a node by its open holes, so it needs a rule that expands one.

    Args:
        form (str): Which table form draws.
    """
    query = generator_query(priced_space(), PRICED)
    lazy = (
        weighted_table(query, SIZE_OF_EVERYTHING, uniform, subgoal_selection=an_expanded_position_first)
        if form == "size table"
        else weighted_cost_table(query, WEIGHTED, uniform, 20, subgoal_selection=an_expanded_position_first)
    )
    with pytest.raises(ValueError, match="not an open hole"):
        list(lazy.stream(random.Random(0)))


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_the_cost_form_never_pushes_a_clause_a_literal_predicate_refuses_below_the_root(seed):
    """``grade(0)`` is refused below ``use``, and under the unit algebra it costs what the two admitted grades cost.

    So a frontier that pushed it would weigh it like them and pop it: the cost form must skip it before
    weighing, as the size form does, and stream what the tree form streams.

    Args:
        seed (int): The seed under test.
    """
    query = generator_query(literal_predicate_space(), USED)
    lazy = weighted_cost_table(query, UNIT, uniform, 5)
    eager = weighted_tree(query, 5, UNIT.fold, uniform)
    streamed = list(lazy.keyed_stream(random.Random(seed)))
    assert_streams_agree(list(eager.keyed_stream(random.Random(seed))), streamed)
    assert len(streamed) == 2
