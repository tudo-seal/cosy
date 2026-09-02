"""The cost structure and the best-first family.

Three layers are under test here, and they are separable on purpose. The *cost domain* is a
positively ordered commutative monoid whose order may be partial. The *frontier* pops a node of
minimal cost, where minimal means that no node of the frontier lies strictly below it. And the
*cost algebras* read a search node as its partial inhabitant and split its cost into the
cost-so-far ``g`` and the heuristic ``h`` of A* search.

The partiality of the order is the point of most of these tests. Over a total order a frontier
that pops "the smallest" and one that pops "a minimal" agree, and the cost-bounded set of a bound
``c`` coincides with ``{n : f(n) <= c}``. Over a partial order the two part ways, and the
statements that survive are the ones the module makes. ``ComponentwiseTuples`` is the genuinely
partial instance carried through the file for that reason, ``NonNegativeReals`` the total one that
also exercises the heap fast path.

Every reference space is small enough to enumerate, so the target quantities, which is to say
costs, orders and completion costs, are computed by brute force rather than asserted from memory.
"""

import heapq
import math
import random
import sys
from collections import deque

import pytest

import cosy.search as search_package
import cosy.search.costs as costs_module
from cosy.core.solution_space import Goal
from cosy.core.tree import Tree
from cosy.search import (
    branch_counts,
    condition_on_maximum,
    generator_query,
    gumbel_key,
    holes,
    partial_inhabitant,
    residual_query,
    term_size,
    weighted_tree,
)
from cosy.search.costs import (
    AdditiveCostAlgebra,
    ComponentwiseTuples,
    CostDomain,
    CostFunction,
    CostOrder,
    HeapFrontier,
    LinearScanFrontier,
    NonNegativeReals,
    Reals,
    a_star,
    assert_uniform_cost_complete,
    best_first,
    best_first_frontier,
    cost_bounded_nodes,
    greedy,
    uniform_cost,
    zero_assignment,
)
from cosy.search.partial import Hole
from tests.search_fixtures import (
    EXPR,
    LIST,
    TAGGED,
    add,
    cons_0,
    cons_1,
    cons_2,
    expression_space,
    list_space,
    lit,
    neg,
    nil,
    stop,
    tag,
    two_symbol_clause_space,
)

REALS = NonNegativeReals()
PLAIN_REALS = Reals()
PAIRS = ComponentwiseTuples(2)

LIST_SYMBOLS = (nil, cons_0, cons_1, cons_2)

PACKAGE_EXPORTS = (
    "AdditiveCostAlgebra",
    "ComponentwiseTuples",
    "CostDomain",
    "CostFunction",
    "CostOrder",
    "Frontier",
    "HeapFrontier",
    "LinearScanFrontier",
    "NonNegativeReals",
    "Reals",
    "a_star",
    "assert_uniform_cost_complete",
    "best_first",
    "best_first_frontier",
    "greedy",
    "uniform_cost",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def unit_costs(_symbol):
    """Give every function symbol the cost one.

    Args:
        _symbol: The function symbol. Ignored.

    Returns:
        float: One.
    """
    return 1.0


def one_per_hole(_hole):
    """Give every hole the cost one, the sharpest admissible bound under unit symbol costs.

    Every inhabitant carries at least one function symbol, so one is a lower bound on the fold of
    every filler, and the assignment is consistent as well.

    Args:
        _hole: The hole. Ignored.

    Returns:
        float: One.
    """
    return 1.0


def unit_algebra():
    """Build the additive cost algebra with unit symbol costs and the zero assignment.

    Returns:
        AdditiveCostAlgebra: The algebra over the nonnegative reals whose fold is the term size.
    """
    return AdditiveCostAlgebra(REALS, unit_costs)


def goals_of(space, start, size_bound):
    """Collect the search nodes of a query's retained tree, with their parents.

    Args:
        space: The synthesized program.
        start: The queried non-terminal.
        size_bound (int): The bound ``D`` that keeps the retained tree finite.

    Returns:
        tuple[list, list]: The goals, and the (parent, child) goal pairs.
    """
    root = branch_counts(generator_query(space, start), size_bound, term_size)
    collected = []
    edges = []
    pending = [root]
    while pending:
        node = pending.pop()
        if node.goal is not None:
            collected.append(node.goal)
            edges.extend((node.goal, child.goal) for child in node.children if child.goal is not None)
        pending.extend(node.children)
    return collected, edges


def inhabitants_below(node):
    """Collect the inhabitants of the success branches below a retained node.

    Args:
        node: The retained node.

    Returns:
        list: The inhabitants, one per success branch.
    """
    found = []
    pending = [node]
    while pending:
        current = pending.pop()
        if current.inhabitant is not None:
            found.append(current.inhabitant)
        pending.extend(current.children)
    return found


class Node:
    """A frontier item that carries nothing but its cost, so that the frontier is alone under test.

    Attributes:
        label (str): A name, so that a popped item is identifiable.
        value: The item's cost.
    """

    def __init__(self, label, value):
        """Build the item.

        Args:
            label (str): The name.
            value: The cost.
        """
        self.label = label
        self.value = value


def cost_of_node(domain):
    """Build the cost function that reads an item's stored cost.

    Args:
        domain (CostDomain): The cost domain the values live in.

    Returns:
        CostFunction: The cost function.
    """
    return CostFunction(domain, lambda node: node.value)


# ---------------------------------------------------------------------------
# The cost domain: a positively ordered commutative monoid
# ---------------------------------------------------------------------------


def test_reals_are_a_positively_ordered_monoid():
    """The monoid laws on the nonnegative reals, with monotonicity and positivity.

    The sampled values are integral floats, so the monoid laws hold exactly. Testing
    associativity on arbitrary doubles would test IEEE rounding rather than the domain.
    """
    rng = random.Random(20260729)
    values = [float(rng.randrange(0, 100)) for _ in range(24)]
    for a in values:
        assert REALS.le(REALS.zero, a)
        assert REALS.add(a, REALS.zero) == a
    for a in values:
        for b in values:
            assert REALS.add(a, b) == REALS.add(b, a)
            for c in values[:4]:
                assert REALS.add(REALS.add(a, b), c) == REALS.add(a, REALS.add(b, c))
                if REALS.le(a, b):
                    assert REALS.le(REALS.add(a, c), REALS.add(b, c))


def test_componentwise_tuples_are_a_positively_ordered_monoid():
    """The same laws on the componentwise order, which is the genuinely partial instance."""
    rng = random.Random(20260730)
    values = [(float(rng.randrange(0, 20)), float(rng.randrange(0, 20))) for _ in range(24)]
    for a in values:
        assert PAIRS.le(PAIRS.zero, a)
        assert PAIRS.add(a, PAIRS.zero) == a
    for a in values:
        for b in values:
            assert PAIRS.add(a, b) == PAIRS.add(b, a)
            for c in values[:4]:
                assert PAIRS.add(PAIRS.add(a, b), c) == PAIRS.add(a, PAIRS.add(b, c))
                if PAIRS.le(a, b):
                    assert PAIRS.le(PAIRS.add(a, c), PAIRS.add(b, c))


def test_componentwise_order_is_genuinely_partial():
    """Two costs may be incomparable, which is what separates the partial from the total case."""
    assert PAIRS.incomparable((1.0, 0.0), (0.0, 1.0))
    assert not PAIRS.le((1.0, 0.0), (0.0, 1.0))
    assert not PAIRS.le((0.0, 1.0), (1.0, 0.0))
    assert not PAIRS.is_total
    assert REALS.is_total
    assert not REALS.incomparable(1.0, 2.0)


def test_a_single_component_domain_is_total_and_archimedean():
    """One component collapses the partial order into a total one, and the fast path returns.

    ``ComponentwiseTuples`` is carried through this file as the partial domain, so the one arity
    at which it is neither partial nor a counterexample to the archimedean property is worth
    pinning: it is the arity at which the factory hands out a heap again.
    """
    single = ComponentwiseTuples(1)
    assert single.is_total
    assert single.is_archimedean
    assert single.lt((1.0,), (2.0,))
    assert not single.incomparable((1.0,), (2.0,))
    assert any(single.lt((5.0,), single.iterated_sum(m, (0.5,))) for m in range(1, 100))
    assert isinstance(best_first_frontier(cost_of_node(single)), HeapFrontier)


def test_strict_order_is_the_asymmetric_part():
    """``lt`` holds exactly when ``le`` holds one way and not the other."""
    assert REALS.lt(1.0, 2.0)
    assert not REALS.lt(2.0, 1.0)
    assert not REALS.lt(1.0, 1.0)
    assert not PAIRS.lt((1.0, 0.0), (0.0, 1.0))
    assert PAIRS.lt((1.0, 0.0), (1.0, 1.0))


def test_down_set_membership_is_weaker_than_le():
    """In a cost-bounded set "not strictly above" is not the same as "at most".

    An incomparable cost is not strictly above the bound, so its node lies in the cost-bounded set
    although its cost is not at most the bound. The difference is what carries the completeness
    argument for best-first search over a partial order.
    """
    assert PAIRS.in_down_set((0.0, 1.0), (1.0, 0.0))
    assert not PAIRS.le((0.0, 1.0), (1.0, 0.0))
    assert PAIRS.in_down_set((1.0, 0.0), (1.0, 0.0))
    assert not PAIRS.in_down_set((2.0, 0.0), (1.0, 0.0))
    assert REALS.in_down_set(1.0, 2.0)
    assert not REALS.in_down_set(3.0, 2.0)


def test_cost_bounded_nodes_collects_the_down_set():
    """``cost_bounded_nodes`` keeps exactly the nodes whose cost is not strictly above."""
    nodes = [Node("a", (1.0, 0.0)), Node("b", (0.0, 1.0)), Node("c", (2.0, 2.0))]
    bounded = cost_bounded_nodes(nodes, cost_of_node(PAIRS), (1.0, 0.0))
    assert [node.label for node in bounded] == ["a", "b"]


def test_iterated_sum_and_archimedean_property():
    """The iterated sum, and the archimedean property that separates the two domains.

    The reals are archimedean, so every strictly positive cost, summed often enough, passes every
    bound. The componentwise order is not, since ``(m, 0)`` never exceeds ``(0, 1)`` and the two
    stay incomparable. That is exactly the counterexample against reading the property with a
    non-strict inequality.
    """
    assert REALS.iterated_sum(0, 3.0) == 0.0
    assert REALS.iterated_sum(4, 3.0) == 12.0
    assert REALS.is_archimedean
    assert any(REALS.lt(100.0, REALS.iterated_sum(m, 0.5)) for m in range(1, 500))

    assert PAIRS.iterated_sum(3, (2.0, 0.0)) == (6.0, 0.0)
    assert not PAIRS.is_archimedean
    assert not any(PAIRS.lt((0.0, 1.0), PAIRS.iterated_sum(m, (1.0, 0.0))) for m in range(1, 500))


def test_iterated_sum_rejects_a_negative_count():
    """A negative repetition count is a caller error, not a value to be clamped."""
    with pytest.raises(ValueError, match="repetition"):
        REALS.iterated_sum(-1, 1.0)


def test_domains_reject_values_outside_their_carrier():
    """The carrier is the nonnegative part, so a negative cost is not an element of it."""
    assert REALS.is_element(0.0)
    assert not REALS.is_element(-1.0)
    assert not REALS.is_element("free")
    assert PAIRS.is_element((0.0, 3.0))
    assert not PAIRS.is_element((0.0, -3.0))
    assert not PAIRS.is_element((1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="arity"):
        PAIRS.add((1.0, 2.0), (1.0, 2.0, 3.0))


def test_domains_decide_an_int_beyond_the_float_range_instead_of_raising():
    """``is_element`` is a predicate, so it answers False for an int beyond the float range.

    ``cosy.search`` produces such ints, since its size tables count inhabitants exactly and
    those counts grow past the float range. A caller who writes ``if not domain.is_element(x)``
    needs the False branch, and ``AdditiveCostAlgebra`` needs the documented ``ValueError``. A
    conversion of that int to a float would raise ``OverflowError`` in both places instead.

    The upper end of the range is itself a cost, so the bound is inclusive.
    """
    huge = 10**400
    assert not PLAIN_REALS.is_element(huge)
    assert not REALS.is_element(huge)
    assert not PAIRS.is_element((huge, 0.0))
    assert REALS.is_element(10**300), "an int the float range holds is a cost"

    largest = sys.float_info.max
    assert PLAIN_REALS.is_element(largest)
    assert PLAIN_REALS.is_element(-largest)
    assert REALS.is_element(largest)
    assert PAIRS.is_element((largest, 0.0))

    with pytest.raises(ValueError, match="element"):
        PLAIN_REALS.le(huge, 1.0)
    with pytest.raises(ValueError, match="element"):
        REALS.le(huge, 1.0)
    algebra = AdditiveCostAlgebra(REALS, lambda _symbol: huge)
    with pytest.raises(ValueError, match="cost domain"):
        algebra.fold(Tree(nil, ()))


def test_an_overflowing_sum_is_refused_where_it_arises():
    """A monoid is closed under its operation, so ``add`` checks its result as well.

    Two floats near the top of the range add to an infinity, which is outside the carrier.
    Unchecked, that infinity would travel on. ``sum_of`` would return it whenever the overflow
    happens in the last addition. Otherwise the next addition would refuse it as an argument, on a
    value the caller never passed in.
    """
    with pytest.raises(ValueError, match="element"):
        REALS.add(1e308, 1e308)
    with pytest.raises(ValueError, match="element"):
        PAIRS.add((1e308, 0.0), (1e308, 0.0))
    with pytest.raises(ValueError, match="element"):
        REALS.sum_of([1e308, 1e308])
    with pytest.raises(ValueError, match="element"):
        REALS.iterated_sum(2, 1e308)
    assert REALS.add(1e308, 1.0) == 1e308, "a sum inside the carrier is untouched"


def test_componentwise_tuples_reject_a_nonpositive_arity():
    """A cost domain of no components carries no information and is a construction error."""
    with pytest.raises(ValueError, match="arity"):
        ComponentwiseTuples(0)


def test_reals_are_an_order_without_a_monoid():
    """A cost function needs a poset only, and random search's costs need exactly that.

    The randomizing cost function of random search is a *negated* Gumbel key, which is negative as
    often as not. No positively ordered monoid holds it, and none has to, because best-first
    search compares and never sums. The type layer says so: :class:`Reals` is a :class:`CostOrder`
    and not a :class:`CostDomain`, so an additive cost algebra cannot be built over it by
    accident.
    """
    reals = Reals()
    assert isinstance(reals, CostOrder)
    assert not isinstance(reals, CostDomain)
    assert reals.is_element(-3.5)
    assert not reals.is_element(float("nan"))
    assert reals.lt(-3.5, -1.0)
    assert reals.is_total
    assert reals.in_down_set(-3.5, -1.0)
    assert not NonNegativeReals().is_element(-3.5)
    assert isinstance(NonNegativeReals(), CostOrder)


def test_sum_of_folds_in_any_order():
    """``sum_of`` is the commutative fold the additive split relies on."""
    assert REALS.sum_of([]) == 0.0
    assert REALS.sum_of([1.0, 2.0, 3.0]) == REALS.sum_of([3.0, 1.0, 2.0])
    assert PAIRS.sum_of([(1.0, 2.0), (3.0, 4.0)]) == (4.0, 6.0)


# ---------------------------------------------------------------------------
# The frontier: a pop returns a node of minimal cost
# ---------------------------------------------------------------------------


def test_frontier_pops_only_minimal_elements_of_an_antichain():
    """Over a partial order every pop is minimal among what is left.

    The costs form an antichain plus one element strictly above it, so no total order can be read
    off them. A frontier that insisted on a least element would have nothing to return. What is
    asked is only that no remaining node lie strictly below the one popped.
    """
    frontier = LinearScanFrontier(cost_of_node(PAIRS))
    items = [
        Node("x", (2.0, 0.0)),
        Node("y", (0.0, 2.0)),
        Node("z", (1.0, 1.0)),
        Node("high", (3.0, 3.0)),
    ]
    for item in items:
        frontier.push(item)

    remaining = list(items)
    popped = []
    while frontier:
        node = frontier.pop()
        remaining.remove(node)
        assert not any(PAIRS.lt(other.value, node.value) for other in remaining), (
            f"{node.label} was popped although a strictly cheaper node remained"
        )
        popped.append(node.label)
    assert sorted(popped) == ["high", "x", "y", "z"]
    assert popped[-1] == "high"


def test_frontier_accepts_an_antichain_without_ordering_it():
    """Incomparable minima may leave in any order, and the rule returns any one of them."""
    frontier = LinearScanFrontier(cost_of_node(PAIRS))
    frontier.push(Node("x", (2.0, 0.0)))
    frontier.push(Node("y", (0.0, 2.0)))
    first = frontier.pop()
    assert first.label in {"x", "y"}
    assert len(frontier) == 1


def test_heap_and_scan_agree_on_a_total_order():
    """The heap fast path and the general frontier stream the same sequence over a total order.

    Ties included: the heap breaks them by an increasing counter, the scan keeps the first minimal
    element it meets, and both amount to insertion order among equal costs. This is the agreement
    the random-search heap in :mod:`cosy.search.sampling` relies on, since that heap *is* this
    fast path, specialized to the negated Gumbel keys.
    """
    rng = random.Random(4711)
    items = [Node(f"n{i}", float(rng.randrange(0, 6))) for i in range(40)]
    scan = LinearScanFrontier(cost_of_node(REALS))
    heap = HeapFrontier(cost_of_node(REALS))
    for item in items:
        scan.push(item)
        heap.push(item)
    assert [scan.pop().label for _ in items] == [heap.pop().label for _ in items]


def test_heap_and_scan_agree_when_pushes_interleave_with_pops():
    """The agreement survives the interleaving the engine actually produces."""
    rng = random.Random(1009)
    scan = LinearScanFrontier(cost_of_node(REALS))
    heap = HeapFrontier(cost_of_node(REALS))
    scan_order = []
    heap_order = []
    counter = 0
    for _ in range(60):
        for _ in range(rng.randrange(0, 4)):
            item = Node(f"n{counter}", float(rng.randrange(0, 8)))
            counter += 1
            scan.push(item)
            heap.push(item)
        if scan:
            scan_order.append(scan.pop().label)
            heap_order.append(heap.pop().label)
    assert scan_order == heap_order


def test_heap_frontier_reproduces_the_random_search_frontier():
    """The overlap with :mod:`cosy.search.sampling`: random search *is* this fast path.

    Random search is best-first search under the randomizing cost function, and the module that
    implements it holds its frontier in a ``heapq`` of its own rather than building it on this
    class. This test replays that loop with :class:`HeapFrontier` in its place, over the same
    counted tree and the same seeded generator, and requires the two streams to be identical. A
    divergence would mean the two frontiers disagree about which node is minimal, and the
    prefix-sampling guarantee is a statement about the frontier best-first search is defined with,
    not about the one the sampler happens to have.
    """
    query = generator_query(list_space(), LIST)
    weighted = weighted_tree(query, 4, term_size, lambda _value: 1.0)
    produced = list(weighted.stream(random.Random(7)))

    rng = random.Random(7)
    frontier = HeapFrontier(CostFunction(Reals(), lambda item: -item[0]))
    frontier.push((gumbel_key(weighted.log_weight_of(weighted.root), rng), weighted.root))
    replayed = []
    while frontier:
        key, node = frontier.pop()
        if node.inhabitant is not None:
            replayed.append(node.inhabitant)
            continue
        child_keys = condition_on_maximum(key, [weighted.log_weight_of(child) for child in node.children], rng)
        for child, child_key in zip(node.children, child_keys, strict=True):
            frontier.push((child_key, child))
    assert replayed == produced
    # A list of length l is a term of size l + 1, so the bound D = 4 admits the lengths 0 to 3
    # and 3^l lists of each. The stream is nonempty for a reason that is checkable by hand.
    assert len(produced) == 1 + 3 + 9 + 27


def test_heap_frontier_matches_a_raw_heapq_with_a_tie_break_counter():
    """The tie discipline is the same as the sampler's: an increasing counter, insertion order.

    :mod:`cosy.search.sampling` pushes its entries with a counter that only grows. The heap
    frontier does the same through its wrapper, and on values with many ties the two orders agree
    exactly, which is what lets the previous test compare streams rather than sets.
    """
    rng = random.Random(31415)
    values = [float(rng.randrange(0, 4)) for _ in range(40)]
    reference: list[int] = []
    raw: list[tuple[float, int]] = []
    for tie, value in enumerate(values):
        heapq.heappush(raw, (value, tie))
    while raw:
        reference.append(heapq.heappop(raw)[1])

    frontier = HeapFrontier(CostFunction(Reals(), lambda item: item[0]))
    for tie, value in enumerate(values):
        frontier.push((value, tie))
    assert [frontier.pop()[1] for _ in values] == reference


def test_frontier_pops_in_nondecreasing_cost_over_a_total_order():
    """Over a total order minimality is least, so the popped costs never fall."""
    rng = random.Random(99)
    frontier = best_first_frontier(cost_of_node(REALS))
    for index in range(30):
        frontier.push(Node(f"n{index}", float(rng.randrange(0, 50))))
    costs = []
    while frontier:
        costs.append(frontier.pop().value)
    assert costs == sorted(costs)


def test_maximization_is_the_order_dual():
    """A maximizing frontier pops a node of maximal score.

    Kernel scores are to be maximized, and best-first search is stated for minimization. The
    frontier turns around rather than the caller negating, which keeps the construction available
    over partial orders where negation is not defined.
    """
    frontier = LinearScanFrontier(cost_of_node(REALS), maximize=True)
    for label, value in [("low", 1.0), ("high", 9.0), ("mid", 5.0)]:
        frontier.push(Node(label, value))
    assert [frontier.pop().label for _ in range(3)] == ["high", "mid", "low"]

    partial = LinearScanFrontier(cost_of_node(PAIRS), maximize=True)
    items = [Node("a", (2.0, 0.0)), Node("b", (0.0, 2.0)), Node("c", (0.0, 0.0))]
    for item in items:
        partial.push(item)
    assert partial.pop().label in {"a", "b"}
    assert partial.pop().label in {"a", "b"}
    assert partial.pop().label == "c"


def test_maximizing_heap_agrees_with_the_maximizing_scan():
    """The fast path is a fast path in both directions of the order."""
    rng = random.Random(555)
    items = [Node(f"n{i}", float(rng.randrange(0, 5))) for i in range(30)]
    scan = LinearScanFrontier(cost_of_node(REALS), maximize=True)
    heap = HeapFrontier(cost_of_node(REALS), maximize=True)
    for item in items:
        scan.push(item)
        heap.push(item)
    assert [scan.pop().label for _ in items] == [heap.pop().label for _ in items]


def test_best_first_frontier_takes_the_heap_only_on_a_total_order():
    """The factory chooses the specialization the domain admits."""
    assert isinstance(best_first_frontier(cost_of_node(REALS)), HeapFrontier)
    assert isinstance(best_first_frontier(cost_of_node(PAIRS)), LinearScanFrontier)


def test_heap_frontier_refuses_a_partial_order():
    """A binary heap needs a total order, so asking for one over a partial domain is an error."""
    with pytest.raises(ValueError, match="total"):
        HeapFrontier(cost_of_node(PAIRS))


def test_popping_an_empty_frontier_is_an_error():
    """An empty frontier has no minimal element, and no substitute is invented for it."""
    for frontier in (
        LinearScanFrontier(cost_of_node(REALS)),
        HeapFrontier(cost_of_node(REALS)),
    ):
        with pytest.raises(ValueError, match="empty"):
            frontier.pop()


def test_frontier_evaluates_the_cost_once_per_node():
    """The cost is read when a node enters, not on every comparison.

    A cost function on goals materializes the partial inhabitant, so evaluating it per comparison
    would rebuild that inhabitant once per comparison instead of once per node.
    """
    calls = []

    def counted(node):
        calls.append(node.label)
        return node.value

    frontier = LinearScanFrontier(CostFunction(REALS, counted))
    for label, value in [("a", 3.0), ("b", 1.0), ("c", 2.0)]:
        frontier.push(Node(label, value))
    assert calls == ["a", "b", "c"]
    frontier.pop()
    frontier.pop()
    assert calls == ["a", "b", "c"]


def test_variance_strategies_keep_the_engine_queue_in_step():
    """The adapter mirrors the frontier's cardinality into the engine's queue.

    ``SolutionSpace.resolution`` reads nothing of the queue but its emptiness, and the ordered
    storage lives in the frontier. So the adapter keeps the two of equal size and fails loudly if
    a caller hands back a queue that never saw the frontier's pushes.
    """
    frontier = LinearScanFrontier(cost_of_node(REALS))
    push, pop = frontier.variance_strategies()
    queue = push(deque(), [Node("a", 2.0), Node("b", 1.0)])
    assert len(queue) == 2
    queue, node = pop(queue)
    assert node.label == "b"
    assert len(queue) == 1
    with pytest.raises(ValueError, match="queue"):
        pop(deque())


@pytest.mark.parametrize("build_frontier", [LinearScanFrontier, HeapFrontier])
def test_variance_strategies_push_a_batch_in_the_order_it_arrives(build_frontier):
    """The adapter pushes the children of one expansion in the order it receives them.

    Both frontiers break ties by insertion order, so equally cheap nodes leave in the order they
    were pushed. All four nodes here carry the same cost, which makes the popped sequence the
    pushed sequence and every reordering of the batch visible.

    Args:
        build_frontier (type): The frontier class under test.
    """
    frontier = build_frontier(cost_of_node(REALS))
    push, pop = frontier.variance_strategies()
    queue = push(deque(), [Node("a", 1.0), Node("b", 1.0), Node("c", 1.0), Node("d", 1.0)])
    popped = []
    while frontier:
        queue, node = pop(queue)
        popped.append(node.label)
    assert popped == ["a", "b", "c", "d"]


# ---------------------------------------------------------------------------
# The cost algebras: the cost of a term and the cost of a goal
# ---------------------------------------------------------------------------


def test_additive_split_on_a_ground_term():
    """On a term without holes the split is the fold."""
    algebra = AdditiveCostAlgebra(REALS, {lit: 1.0, neg: 2.0, add: 4.0}.__getitem__)
    term = Tree(add, (Tree(lit, ()), Tree(neg, (Tree(lit, ()),))))
    assert algebra.term_assignment(term) == 8.0
    assert algebra.symbol_cost_sum(term) == 8.0
    assert algebra.hole_cost_sum(term) == 0.0
    assert algebra.fold(term) == 8.0


def test_additive_split_on_a_term_with_holes():
    """With holes the split is the symbol occurrences plus the hole occurrences.

    Each occurrence is a summand of its own, which the two-hole term pins.
    """
    algebra = AdditiveCostAlgebra(REALS, {lit: 1.0, neg: 2.0, add: 4.0}.__getitem__, hole_cost=lambda _hole: 5.0)
    term = Tree(add, (Tree(lit, ()), Tree(Hole((1,), EXPR), ())))
    assert algebra.symbol_cost_sum(term) == 5.0
    assert algebra.hole_cost_sum(term) == 5.0
    assert algebra.term_assignment(term) == 10.0
    two_holes = Tree(add, (Tree(Hole((0,), EXPR), ()), Tree(Hole((1,), EXPR), ())))
    assert algebra.symbol_cost_sum(two_holes) == 4.0
    assert algebra.hole_cost_sum(two_holes) == 10.0
    assert algebra.term_assignment(two_holes) == 14.0


def test_fold_refuses_a_term_with_holes():
    """A fold is defined on ground terms, and a hole needs a variable assignment."""
    algebra = unit_algebra()
    with pytest.raises(ValueError, match="hole"):
        algebra.fold(Tree(neg, (Tree(Hole((0,), EXPR), ()),)))


def test_additive_split_agrees_with_the_direct_fold_on_every_node():
    """The split on the real search nodes of a reference space.

    The term assignment is computed by the algebra's recursion, and the split sums the symbol
    occurrences and the hole occurrences separately. The two have to agree, and it is that
    identity the equation ``f(n) = g(n) + h(n)`` rests on. The costs are integral floats and the
    sums stay small, so both sides are exact.
    """
    algebra = AdditiveCostAlgebra(
        REALS,
        {nil: 1.0, cons_0: 2.0, cons_1: 3.0, cons_2: 4.0}.__getitem__,
        hole_cost=lambda _hole: 7.0,
    )
    goals, _ = goals_of(list_space(), LIST, 4)
    assert goals
    for goal in goals:
        term = partial_inhabitant(goal)
        direct = algebra.term_assignment(term)
        split = REALS.add(algebra.symbol_cost_sum(term), algebra.hole_cost_sum(term))
        assert direct == split
        assert algebra.cost_on_goals(goal) == direct
        assert algebra.cost_so_far(goal) == algebra.symbol_cost_sum(term)
        assert algebra.heuristic(goal) == algebra.hole_cost_sum(term)


def test_additive_split_sums_over_every_hole_of_a_node():
    """The split on the search nodes of a space that opens more than one hole at a time.

    ``test_additive_split_agrees_with_the_direct_fold_on_every_node`` runs on ``list_space``,
    whose clauses are nullary or unary, so no node there carries more than one hole and the sum
    over the hole occurrences is a one-element sum. ``add`` of ``expression_space`` is binary, so
    writing it opens two holes at once, and the first assertion pins that such nodes are reached.

    The symbol and hole costs are integral floats and the sums stay small, so every addition here
    is exact. The two sides add the same summands in a different order, and float addition is not
    associative. A cost that is not exactly representable would separate the two by a rounding
    step rather than by a missing summand.
    """
    per_hole = 8.0
    algebra = AdditiveCostAlgebra(
        REALS,
        {lit: 1.0, neg: 2.0, add: 4.0}.__getitem__,
        hole_cost=lambda _hole: per_hole,
    )
    goals, _ = goals_of(expression_space(), EXPR, 5)
    assert max(len(holes(goal)) for goal in goals) > 1
    for goal in goals:
        term = partial_inhabitant(goal)
        assert algebra.hole_cost_sum(term) == len(holes(goal)) * per_hole
        assert algebra.term_assignment(term) == REALS.add(algebra.symbol_cost_sum(term), algebra.hole_cost_sum(term))
        assert a_star(algebra)(goal) == algebra.cost_on_goals(goal)


def test_additive_split_counts_constant_arguments_as_symbols():
    """A clause that writes a literal writes a symbol, and ``g`` is charged for it.

    ``two_symbol_clause_space`` fixes its terminal *and* its constant argument in one application,
    so a partial inhabitant there grows by two symbols per step while it grows by one on most
    other reference spaces. A cost-so-far function that counted applications rather than symbol
    occurrences would agree with the truth everywhere else and be wrong here, and this is the
    shape a repository with literal parameters takes throughout.
    """
    algebra = unit_algebra()
    goals, _ = goals_of(two_symbol_clause_space(), TAGGED, 5)
    assert goals
    for goal in goals:
        term = partial_inhabitant(goal)
        assert algebra.cost_so_far(goal) == float(term_size(term))
    tagged = Tree(tag, (Tree(0, ()), Tree(stop, ())))
    assert algebra.fold(tagged) == 3.0


def test_cost_so_far_is_a_cost_so_far_function():
    """``g(n') = g(n) + delta(n, n')`` on every edge of the retained tree.

    The step cost of filling a hole with ``F`` is ``c_F``, and it is read off the symbol
    occurrences the child added, which needs no subtraction and therefore works over every cost
    domain. The costs are integral floats and the sums stay small, so both sides are exact.
    """
    algebra = AdditiveCostAlgebra(REALS, {lit: 1.0, neg: 2.0, add: 4.0}.__getitem__, hole_cost=one_per_hole)
    _, edges = goals_of(expression_space(), EXPR, 4)
    assert edges
    for parent, child in edges:
        step = algebra.step_cost(parent, child)
        assert REALS.le(REALS.zero, step)
        assert algebra.cost_so_far(child) == REALS.add(algebra.cost_so_far(parent), step)


def test_a_step_that_writes_no_symbol_costs_nothing():
    """The step cost sums the symbols the child added, so an empty sum is the neutral element.

    No expansion of the engine hands over such a pair, because every resolution step fixes at
    least one function-symbol occurrence, and the second measurement below is what says so over a
    whole retained tree. The zero case is still the one that makes the definition work over a
    monoid without subtraction, so it is pinned rather than left to the reader.
    """
    algebra = unit_algebra()
    goals, edges = goals_of(expression_space(), EXPR, 4)
    assert goals
    assert edges
    assert algebra.step_cost(goals[0], goals[0]) == REALS.zero
    assert all(algebra.step_cost(parent, child) > 0.0 for parent, child in edges)


def test_cost_so_far_is_monotone_along_the_branches():
    """Positivity of the step costs makes ``g`` monotone along every branch."""
    algebra = unit_algebra()
    _, edges = goals_of(expression_space(), EXPR, 4)
    assert edges
    for parent, child in edges:
        assert REALS.le(algebra.cost_so_far(parent), algebra.cost_so_far(child))


def test_step_cost_rejects_an_unrelated_goal():
    """A pair that is not an edge of the derivation tree has no step cost, and none is invented."""
    algebra = unit_algebra()
    goals, _ = goals_of(expression_space(), EXPR, 3)
    unrelated = [
        (first, second)
        for first in goals
        for second in goals
        if partial_inhabitant(first).root is add and partial_inhabitant(second).root is neg
    ]
    assert unrelated
    with pytest.raises(ValueError, match="child"):
        algebra.step_cost(*unrelated[0])


def test_heuristic_vanishes_on_success_nodes():
    """No hole remains at a success node, and the empty sum is zero."""
    algebra = AdditiveCostAlgebra(REALS, unit_costs, hole_cost=lambda _hole: 9.0)
    goals, _ = goals_of(list_space(), LIST, 4)
    successes = [goal for goal in goals if goal.success]
    assert successes
    for goal in successes:
        assert algebra.heuristic(goal) == REALS.zero
        assert algebra.cost_on_goals(goal) == algebra.cost_so_far(goal)


def test_symbol_costs_outside_the_domain_are_rejected():
    """A negative symbol cost breaks positivity, so it is refused where it enters."""
    algebra = AdditiveCostAlgebra(REALS, lambda _symbol: -1.0)
    with pytest.raises(ValueError, match="cost domain"):
        algebra.fold(Tree(nil, ()))


def test_hole_costs_outside_the_domain_are_rejected():
    """The variable assignment maps into the cost domain too."""
    algebra = AdditiveCostAlgebra(REALS, unit_costs, hole_cost=lambda _hole: -2.0)
    with pytest.raises(ValueError, match="cost domain"):
        algebra.term_assignment(Tree(Hole((), LIST), ()))


def test_zero_assignment_is_the_trivial_admissible_one():
    """``h(v) = 0`` is admissible for free, and it turns A* into uniform-cost search."""
    assignment = zero_assignment(REALS)
    assert assignment(Hole((), LIST)) == 0.0
    algebra = AdditiveCostAlgebra(REALS, unit_costs, hole_cost=assignment)
    goals, _ = goals_of(list_space(), LIST, 3)
    assert goals
    for goal in goals:
        assert algebra.heuristic(goal) == 0.0


def test_admissible_assignment_bounds_every_completion_cost():
    """Admissibility measured against brute force.

    For every node of the retained tree and every success branch below it, the heuristic must not
    exceed the completion cost. With unit symbol costs the fold of the branch's inhabitant is
    ``g(n)`` plus that completion cost, so the check is ``f(n) <= fold(u)`` for every inhabitant
    ``u`` below ``n``. One hole needs at least one symbol, which is why ``h(v) = 1`` is
    admissible.
    """
    algebra = AdditiveCostAlgebra(REALS, unit_costs, hole_cost=one_per_hole)
    root = branch_counts(generator_query(expression_space(), EXPR), 5, term_size)
    pending = [root]
    checked = 0
    while pending:
        node = pending.pop()
        pending.extend(node.children)
        if node.goal is None:
            continue
        estimate = REALS.add(algebra.cost_so_far(node.goal), algebra.heuristic(node.goal))
        for inhabitant in inhabitants_below(node):
            assert REALS.le(estimate, algebra.fold(inhabitant))
            checked += 1
    assert checked > 50


def test_an_inadmissible_assignment_is_visible_to_the_same_measurement():
    """The admissibility check bites: a hole assignment above the fillers' folds fails it."""
    algebra = AdditiveCostAlgebra(REALS, unit_costs, hole_cost=lambda _hole: 50.0)
    root = branch_counts(generator_query(expression_space(), EXPR), 4, term_size)
    pending = [root]
    violated = False
    while pending:
        node = pending.pop()
        pending.extend(node.children)
        if node.goal is None:
            continue
        estimate = REALS.add(algebra.cost_so_far(node.goal), algebra.heuristic(node.goal))
        violated = violated or any(
            not REALS.le(estimate, algebra.fold(inhabitant)) for inhabitant in inhabitants_below(node)
        )
    assert violated


def test_consistent_heuristic_makes_the_sum_monotone():
    """With a consistent ``h`` the sum ``f(n) = g(n) + h(n)`` never falls along a branch.

    One per hole is consistent under unit symbol costs, since filling a hole pays one and releases
    at most one estimate. The constant assignment of five per hole is not, and the measurement
    below separates the two. The costs are integral floats and the sums stay small, so every
    addition here is exact.
    """
    consistent = AdditiveCostAlgebra(REALS, unit_costs, hole_cost=one_per_hole)
    inconsistent = AdditiveCostAlgebra(REALS, unit_costs, hole_cost=lambda _hole: 5.0)
    _, edges = goals_of(expression_space(), EXPR, 4)
    assert edges
    assert all(REALS.le(a_star(consistent)(parent), a_star(consistent)(child)) for parent, child in edges)
    assert not all(REALS.le(a_star(inconsistent)(parent), a_star(inconsistent)(child)) for parent, child in edges)


def test_uniform_cost_ignores_the_variable_assignment():
    """Uniform-cost search is A* with ``h = 0``, whatever assignment the algebra carries."""
    algebra = AdditiveCostAlgebra(REALS, unit_costs, hole_cost=lambda _hole: 12.0)
    goals, _ = goals_of(list_space(), LIST, 3)
    assert goals
    for goal in goals:
        assert uniform_cost(algebra)(goal) == algebra.cost_so_far(goal)


def test_greedy_is_the_heuristic_alone():
    """Greedy search reads ``f = h``, the estimate without the cost already paid."""
    algebra = AdditiveCostAlgebra(REALS, unit_costs, hole_cost=one_per_hole)
    goals, _ = goals_of(list_space(), LIST, 3)
    assert goals
    for goal in goals:
        assert greedy(algebra)(goal) == algebra.heuristic(goal)


def test_cost_functions_carry_their_domain():
    """A cost function is a map *into a partially ordered set*, and it carries that set along."""
    algebra = unit_algebra()
    assert a_star(algebra).domain is REALS
    assert uniform_cost(algebra).domain is REALS
    assert greedy(algebra).domain is REALS


# ---------------------------------------------------------------------------
# The conditions under which uniform-cost search is complete
# ---------------------------------------------------------------------------


def test_uniform_cost_completeness_conditions_hold_for_unit_costs():
    """An archimedean domain and strictly positive combinator costs are what is asked."""
    assert_uniform_cost_complete(unit_algebra(), LIST_SYMBOLS)


@pytest.mark.parametrize("offender", LIST_SYMBOLS)
def test_uniform_cost_completeness_fails_on_a_zero_cost_combinator(offender):
    """A combinator of cost zero leaves the cost-bounded sets infinite.

    Every position of the family takes its turn as the offender. The condition is over all of
    them, so a check that stops early would pass on the symbols it never reads.

    Args:
        offender (Any): The combinator whose cost is zero.
    """
    symbol_costs = dict.fromkeys(LIST_SYMBOLS, 1.0)
    symbol_costs[offender] = 0.0
    algebra = AdditiveCostAlgebra(REALS, symbol_costs.__getitem__)
    with pytest.raises(ValueError, match="strictly positive") as raised:
        assert_uniform_cost_complete(algebra, LIST_SYMBOLS)
    assert repr(offender) in str(raised.value), "the message names the combinator to repair"


def test_uniform_cost_completeness_fails_on_a_nonarchimedean_domain():
    """Without the archimedean property an infinite antichain fits below a bound."""
    algebra = AdditiveCostAlgebra(PAIRS, lambda _symbol: (1.0, 1.0))
    with pytest.raises(ValueError, match="archimedean"):
        assert_uniform_cost_complete(algebra, LIST_SYMBOLS)


def test_uniform_cost_completeness_refuses_an_empty_symbol_family():
    """The check refuses an empty symbol family, and a consumed iterator arrives as one.

    The first call reads the generator to its end, so the second call has no symbol left to look
    at. Its algebra gives every combinator cost zero, and a silent pass would report completeness
    for exactly the algebra the condition rules out.
    """
    zero_costs = AdditiveCostAlgebra(REALS, dict.fromkeys(LIST_SYMBOLS, 0.0).__getitem__)
    symbols = (symbol for symbol in LIST_SYMBOLS)
    assert_uniform_cost_complete(unit_algebra(), symbols)
    with pytest.raises(ValueError, match="empty symbol family"):
        assert_uniform_cost_complete(zero_costs, symbols)
    with pytest.raises(ValueError, match="empty symbol family"):
        assert_uniform_cost_complete(zero_costs, ())


# ---------------------------------------------------------------------------
# Best-first search over a resolution query
# ---------------------------------------------------------------------------


def test_uniform_cost_search_streams_by_nondecreasing_cost():
    """``g`` never falls along a branch, so the stream is ordered by cost.

    The symbol costs differ per combinator, so the order is not the enumeration order of any
    uninformed rule: a cheap long list precedes an expensive short one.
    """
    algebra = AdditiveCostAlgebra(REALS, {nil: 1.0, cons_0: 1.0, cons_1: 5.0, cons_2: 9.0}.__getitem__)
    query = generator_query(list_space(), LIST)
    streamed = list(best_first(query, uniform_cost(algebra), max_count=25))
    costs = [algebra.fold(tree) for tree in streamed]
    assert len(streamed) == 25
    assert costs == sorted(costs)
    assert costs[0] == 1.0
    assert len(set(streamed)) == 25


def test_uniform_cost_search_finds_the_cheapest_inhabitant_first():
    """The first pop is a cheapest node, which is admissibility in its simplest instance."""
    algebra = AdditiveCostAlgebra(REALS, {lit: 7.0, neg: 1.0, add: 1.0}.__getitem__)
    query = generator_query(expression_space(), EXPR)
    first = next(iter(best_first(query, uniform_cost(algebra), max_count=1)))
    assert algebra.fold(first) == 7.0
    assert first == Tree(lit, ())


def test_a_star_with_an_admissible_heuristic_streams_in_cost_order():
    """A consistent heuristic gives the same cost order as uniform-cost search.

    The heuristic is one per hole, which is consistent under unit symbol costs, so
    ``f(n) = g(n) + h(n)`` never falls along a branch and the stream is ordered by cost. The costs
    are integral floats and the sums stay small, so every addition here is exact. What is asserted
    here is the cost order and the terms of the complete cost levels, not the number of
    expansions.
    """
    plain = AdditiveCostAlgebra(REALS, unit_costs)
    informed = AdditiveCostAlgebra(REALS, unit_costs, hole_cost=one_per_hole)
    query = generator_query(expression_space(), EXPR)
    ucs = list(best_first(query, uniform_cost(plain), max_count=20))
    astar = list(best_first(query, a_star(informed), max_count=20))
    assert [plain.fold(tree) for tree in ucs] == sorted(plain.fold(tree) for tree in ucs)
    assert [plain.fold(tree) for tree in astar] == [plain.fold(tree) for tree in ucs]
    # Both streams are cut mid-cost-level by ``max_count``, and which terms of the last level
    # made it depends on the tie-break. So the sets are compared below that level, where both
    # rules must have streamed every inhabitant there is.
    boundary = plain.fold(ucs[-1])
    complete_levels = {tree for tree in ucs if plain.fold(tree) < boundary}
    assert complete_levels
    assert {tree for tree in astar if plain.fold(tree) < boundary} == complete_levels


def test_best_first_streams_a_linear_extension_over_a_partial_order():
    """Where the order is genuinely partial the stream is a linear extension of it.

    With componentwise costs the stream cannot be sorted, since incomparable costs may appear in
    any order, but no streamed cost is strictly below one streamed before it. The assertion that
    some pair is incomparable keeps the test from passing vacuously on a total order.

    Completeness would need every cost-bounded set to be finite, and here one is not: ``neg``
    costs ``(0, 1)``, no iterated sum of it ever passes ``(1, 0)``, the cost of the cheapest
    inhabitant, so every nesting of ``neg`` stays in that cost-bounded set and is never too
    expensive to pop. A pop may return any minimum, so an unbounded run can go on nesting ``neg``
    and stream nothing further. Bounded, the run halts on whichever minimum each pop returns and
    streams the same terms either way, so the free choice decides only their order, which is what
    the assertions below check.
    """
    algebra = AdditiveCostAlgebra(PAIRS, {lit: (1.0, 0.0), neg: (0.0, 1.0), add: (1.0, 1.0)}.__getitem__)
    query = generator_query(expression_space(), EXPR)
    # lit, neg and add have arity 0, 1 and 2, so the terms of depth at most d number
    # t(d) = 1 + t(d - 1) + t(d - 1) ** 2, which is 1, 3, 13 and 183 for d up to 3. What
    # max_depth bounds is the length of the positions a goal carries, not the term depth. The two
    # agree on the terms a deepest-first computation rule streams, and best_first fixes that
    # rule. The count bound sits above 183 so that a run reaching further stops and fails the
    # count here instead of streaming on.
    streamed = list(best_first(query, uniform_cost(algebra), max_depth=3, max_count=200))
    costs = [algebra.fold(tree) for tree in streamed]
    assert len(streamed) == 1 + 13 + 13 * 13
    for earlier in range(len(costs)):
        for later in range(earlier + 1, len(costs)):
            assert not PAIRS.lt(costs[later], costs[earlier])
    assert any(PAIRS.incomparable(first, second) for first in costs for second in costs)


def test_best_first_is_sound_under_a_greedy_cost_function():
    """Greedy search is a sound best-first search, and its docstring promises no more.

    Greedy search is not complete in general and does not stream in cost order, so the test asks
    only what is granted: every streamed term is an inhabitant.
    """
    space = expression_space()
    algebra = AdditiveCostAlgebra(REALS, unit_costs, hole_cost=one_per_hole)
    query = generator_query(space, EXPR)
    streamed = list(best_first(query, greedy(algebra), max_count=10))
    assert streamed
    assert all(space.contains_tree(EXPR, tree) for tree in streamed)


def test_best_first_agrees_between_its_two_frontiers_on_a_query():
    """The engine sees no difference between the fast path and the general frontier."""
    algebra = AdditiveCostAlgebra(REALS, {nil: 2.0, cons_0: 1.0, cons_1: 3.0, cons_2: 5.0}.__getitem__)
    query = generator_query(list_space(), LIST)
    cost = uniform_cost(algebra)
    with_heap = list(best_first(query, cost, frontier=HeapFrontier(cost), max_count=20))
    with_scan = list(best_first(query, cost, frontier=LinearScanFrontier(cost), max_count=20))
    assert [str(tree) for tree in with_heap] == [str(tree) for tree in with_scan]


def test_best_first_searches_with_the_heap_over_a_total_order():
    """Best-first search builds the frontier its cost domain admits, which is the heap here.

    The two frontiers answer the same specification and stream the same terms. A search pinned to
    the general one would therefore keep every assertion of this file correct and give up the fast
    path the heap exists for. The observation is the number of order comparisons and not a wall
    clock. A pop of the scan compares its running candidate against every node of the frontier, a
    pop of the heap against one path through it. Every comparison of either frontier ends at the
    cost order, so counting it there is exact and deterministic, where a wall-clock bound would
    report how busy the machine running the suite happens to be.
    """

    class CountingReals(NonNegativeReals):
        """The nonnegative reals, recording every comparison asked of them.

        Attributes:
            comparisons (int): The number of comparisons so far.
        """

        def __init__(self):
            """Start the count at zero."""
            super().__init__()
            self.comparisons = 0

        def le(self, left, right):
            """Compare two costs and count the comparison.

            Args:
                left (float): The lower candidate.
                right (float): The upper candidate.

            Returns:
                bool: True if ``left`` is at most ``right``.
            """
            self.comparisons += 1
            return super().le(left, right)

    def comparisons_of(frontier_class):
        """Count the order comparisons of one best-first search over the list space.

        Args:
            frontier_class: The frontier class to search with, or None to leave the choice of
                frontier to ``best_first``.

        Returns:
            int: The number of comparisons the search asked its cost order for.
        """
        domain = CountingReals()
        algebra = AdditiveCostAlgebra(domain, {nil: 2.0, cons_0: 1.0, cons_1: 3.0, cons_2: 5.0}.__getitem__)
        cost = uniform_cost(algebra)
        query = generator_query(list_space(), LIST)
        frontier = None if frontier_class is None else frontier_class(cost)
        for _ in best_first(query, cost, frontier=frontier, max_count=60):
            pass
        return domain.comparisons

    heap = comparisons_of(HeapFrontier)
    scan = comparisons_of(LinearScanFrontier)
    assert heap < scan, "the count has to tell the two frontiers apart, or the assertion below is vacuous"
    assert comparisons_of(None) == heap, "the frontier best_first built for a total order is not the heap"


def test_best_first_on_a_partial_term_query_charges_the_prescribed_symbols():
    """The prescribed symbols of a partial-term query count into ``g`` from the start.

    The query fixes ``cons_1`` at the root and opens its argument, so every completion carries
    that symbol and the cheapest completion is ``cons_1(nil)``.
    """
    algebra = AdditiveCostAlgebra(REALS, {nil: 1.0, cons_0: 1.0, cons_1: 4.0, cons_2: 1.0}.__getitem__)
    prescribed = Tree(cons_1, (Tree(nil, ()),))
    query = residual_query(list_space(), LIST, prescribed, (0,))
    streamed = list(best_first(query, uniform_cost(algebra), max_count=6))
    assert streamed[0] == prescribed
    costs = [algebra.fold(tree) for tree in streamed]
    assert costs == sorted(costs)
    assert costs[0] == 5.0


def test_best_first_maximizes_under_the_order_dual():
    """A maximizing best-first search expands the most expensive node.

    Kernel scores are maximized, and best-first search is stated for minimization, so the frontier
    turns around instead of the caller negating. What is asserted here is the direction and
    soundness alone. Maximizing ``g`` is a dive, since ``g`` grows along the branches and the
    order-dual turns that the wrong way round, so nothing is claimed about the order of the
    stream. The case that does stream in order of score is the monotone one, and it belongs to
    :mod:`cosy.search.kernels`.
    """
    space = expression_space()
    algebra = AdditiveCostAlgebra(REALS, {lit: 1.0, neg: 2.0, add: 3.0}.__getitem__)
    query = generator_query(space, EXPR)
    cheapest = next(iter(best_first(query, uniform_cost(algebra), max_count=1, max_depth=3)))
    dearest = next(iter(best_first(query, uniform_cost(algebra), maximize=True, max_count=1, max_depth=3)))
    assert algebra.fold(dearest) > algebra.fold(cheapest)
    assert space.contains_tree(EXPR, dearest)


def test_best_first_respects_the_goal_filter():
    """The expansion filter of the engine reaches best-first search unchanged."""
    algebra = unit_algebra()
    query = generator_query(list_space(), LIST)

    def small(goal: Goal) -> bool:
        """Keep only goals whose partial inhabitant stays within three symbols.

        Args:
            goal (Goal): The child under test.

        Returns:
            bool: True if the partial inhabitant carries at most three symbols.
        """
        return term_size(partial_inhabitant(goal)) <= 3

    streamed = list(best_first(query, uniform_cost(algebra), goal_filter=small))
    assert streamed
    assert all(term_size(tree) <= 3 for tree in streamed)
    assert len(streamed) == 1 + 3 + 9


# ---------------------------------------------------------------------------
# The edges: contracts that only fire when a caller gets something wrong
# ---------------------------------------------------------------------------


def test_incomparable_is_false_when_one_direction_holds():
    """Incomparability is the failure of *both* directions, not of one.

    On a total order nothing is incomparable, and the asymmetric case is where a one-sided
    definition would pass every other test in this file.
    """
    assert not REALS.incomparable(1.0, 2.0)
    assert not REALS.incomparable(2.0, 1.0)
    assert not REALS.incomparable(1.0, 1.0)

    assert PAIRS.incomparable((1.0, 0.0), (0.0, 1.0))
    assert not PAIRS.incomparable((0.0, 0.0), (1.0, 1.0))
    assert not PAIRS.incomparable((1.0, 1.0), (0.0, 0.0))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_value_outside_the_domain_is_rejected_by_the_comparison(value):
    """``le`` refuses what is not an element, and is the only guard the reals have.

    ``NonNegativeReals`` catches a non-element a second time in ``add``, and ``Reals`` has no
    addition at all. Without this guard a NaN cost is incomparable to every other cost while
    ``is_total`` still reports True. That property is all :class:`HeapFrontier` checks before it
    admits a domain, and its heap can then let a node that is not minimal reach the root.

    Args:
        value (float): The non-element under test.
    """
    with pytest.raises(ValueError, match="element"):
        PLAIN_REALS.le(value, 1.0)
    with pytest.raises(ValueError, match="element"):
        PLAIN_REALS.le(1.0, value)
    with pytest.raises(ValueError, match="element"):
        REALS.le(value, 1.0)
    assert not REALS.is_element(value)


def test_a_negative_cost_is_not_an_element_of_the_positive_domain():
    """Positivity is ``0 <= a`` for every element, so a negative number is none."""
    assert not REALS.is_element(-1.0)
    assert REALS.is_positive(REALS.zero)
    assert REALS.is_positive(3.0)
    assert PLAIN_REALS.is_element(-1.0), "the plain reals are an order, not a positive monoid"


def test_best_first_refuses_a_frontier_that_searches_by_another_cost():
    """Two cost functions and no rule for which one applies is a silent wrong answer.

    The frontier decides the search, so a ``cost`` passed next to it would be read by the caller
    and by nothing else. Across domains it is worse still, since the values would then be compared
    without either domain objecting.
    """
    space = expression_space()
    query = generator_query(space, EXPR)
    cheap = uniform_cost(AdditiveCostAlgebra(REALS, lambda _symbol: 1.0))
    other = uniform_cost(AdditiveCostAlgebra(REALS, lambda _symbol: 2.0))

    with pytest.raises(ValueError, match="own cost function"):
        best_first(query, cheap, frontier=LinearScanFrontier(other))

    # the same frontier with the same cost function is what the pinning tests do, and it passes
    list(zip(best_first(query, cheap, frontier=LinearScanFrontier(cheap)), range(3), strict=False))


def test_best_first_refuses_a_frontier_that_is_not_empty():
    """A search starts from the query's initial goals, not from someone else's leftovers."""
    space = expression_space()
    query = generator_query(space, EXPR)
    cost = uniform_cost(AdditiveCostAlgebra(REALS, lambda _symbol: 1.0))
    frontier = LinearScanFrontier(cost)
    for rule in space.get(EXPR):
        goal = Goal.from_rhs_rule(rule)
        if goal is not None:
            frontier.push(goal)

    with pytest.raises(ValueError, match="starts empty"):
        best_first(query, cost, frontier=frontier)


def test_best_first_refuses_maximize_together_with_a_frontier():
    """The direction is stated once: by the frontier, or by the flag."""
    space = expression_space()
    query = generator_query(space, EXPR)
    cost = uniform_cost(AdditiveCostAlgebra(REALS, lambda _symbol: 1.0))

    with pytest.raises(ValueError, match="own direction"):
        best_first(query, cost, maximize=True, frontier=LinearScanFrontier(cost, maximize=True))


def test_a_maximizing_frontier_is_accepted_without_the_flag():
    """The direction stated once by the frontier is the way to run a maximizing search.

    The previous test pins the refusal of the two together. This one runs the combination that
    remains, so that the refusal cannot be read as a ban on maximizing through a frontier.
    """
    space = expression_space()
    algebra = AdditiveCostAlgebra(REALS, {lit: 1.0, neg: 2.0, add: 3.0}.__getitem__)
    query = generator_query(space, EXPR)
    cost = uniform_cost(algebra)
    frontier = LinearScanFrontier(cost, maximize=True)
    dearest = next(iter(best_first(query, cost, frontier=frontier, max_count=1, max_depth=3)))
    cheapest = next(iter(best_first(query, cost, max_count=1, max_depth=3)))
    assert algebra.fold(dearest) > algebra.fold(cheapest)
    assert space.contains_tree(EXPR, dearest)


# ---------------------------------------------------------------------------
# What the package hands out
# ---------------------------------------------------------------------------


def test_the_package_hands_out_the_module_s_own_objects():
    """A name re-exported from ``cosy.search`` is the module's object and not a copy of it.

    A caller that annotates a parameter as ``cosy.search.Frontier`` and builds one through
    ``cosy.search.costs.best_first_frontier`` has to end up with one class, and a failing
    ``isinstance`` is what an accidental second definition would look like much later.
    ``cost_bounded_nodes`` and ``zero_assignment`` stay behind on purpose, the way the default
    weights of the tree kernels do: they are the module's own building blocks rather than the
    vocabulary the package offers.
    """
    for name in PACKAGE_EXPORTS:
        assert getattr(search_package, name) is getattr(costs_module, name)
        assert name in search_package.__all__
    assert set(costs_module.__all__) - set(PACKAGE_EXPORTS) == {"cost_bounded_nodes", "zero_assignment"}
