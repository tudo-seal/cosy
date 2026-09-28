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
"""

import pytest

from cosy.core import Constructor, SpecificationBuilder, Synthesizer
from cosy.core.solution_space import ConstantArgument, Goal, NonTerminalArgument
from cosy.core.types import DataGroup
from cosy.search.costs import AdditiveCostAlgebra, ComponentwiseTuples, NonNegativeReals
from cosy.search.counting import _added_symbols, rule_cost
from cosy.search.partial import partial_inhabitant, term_size
from cosy.search.rules import deepest_first_subgoal
from tests.search_fixtures import (
    EXPR,
    LIST,
    TAGGED,
    expression_space,
    list_space,
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
