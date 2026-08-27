"""Branch counts over the retained derivation tree, checked against exhaustive enumeration.

Random search weights the choice of a child by the inhabitants below it, counted per cost value.
That quantity is the branch count of a node, it obeys a recursion over the tree, and at the root,
under unambiguity within the bound and only there, the branch counts *are* the cost counts ``N_r``,
which is to say the number of inhabitants of the query per cost value.

Exact throughout, with no statistics. The oracle builds every term over the signature up to the
size bound and asks the checker which of them the space contains, so each target quantity is
computed by brute force through a decision procedure independent of the counting recursion, and
compared value by value. What the counts are then used for, which is a draw that follows a chosen
distribution, is a statistical claim and is not made here.

Two claims about the retained tree are checked structurally rather than through counts: it is
finite because each internal step fixes at least one function symbol, so no branch survives more
than ``D`` expansions, and the recursion therefore halts on a recursive space carrying no depth
bound at all.
"""

import pytest

from cosy.core import SpecificationBuilder, Synthesizer
from cosy.core.tree import Tree
from cosy.core.types import Intersection
from cosy.search import generator_query, partial_inhabitant, residual_query, term_size
from cosy.search.counting import (
    assert_unambiguous_within,
    branch_counts,
    branch_multiplicities,
    retained_node_count,
)
from tests._generate_and_check import (
    AMBIGUOUS_SIGNATURE,
    EXPR_SIGNATURE,
    LIST_SIGNATURE,
    PAIR_SIGNATURE,
    TAGGED_SIGNATURE,
    cost_counts,
    inhabitants_within,
)
from tests.search_fixtures import (
    AMBIGUOUS_TARGET,
    EXPR,
    LIST,
    PAIR,
    TAGGED,
    X,
    Y,
    alt,
    ambiguous_space,
    base,
    constrained_space,
    expression_space,
    list_space,
    lit,
    merge,
    neg,
    two_symbol_clause_space,
)


def nullary_merge(left: str, right: str) -> str:
    """Combine an ``X`` and a ``Y``, for the control space of the ambiguity test.

    Args:
        left (str): The interpreted ``X``.
        right (str): The interpreted ``Y``.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"n({left},{right})"


@pytest.mark.parametrize(
    ("build", "start", "signature", "bound", "depth"),
    [
        (list_space, LIST, LIST_SIGNATURE, 4, 3),
        (expression_space, EXPR, EXPR_SIGNATURE, 4, 3),
        (constrained_space, PAIR, PAIR_SIGNATURE, 5, 3),
        (ambiguous_space, AMBIGUOUS_TARGET, AMBIGUOUS_SIGNATURE, 3, 3),
        (two_symbol_clause_space, TAGGED, TAGGED_SIGNATURE, 5, 3),
    ],
)
def test_the_oracle_agrees_with_resolution_where_both_are_feasible(build, start, signature, bound, depth):
    """Generate-and-check and depth-first resolution list the same terms.

    The oracle decides membership with ``contains_tree`` rather than by resolving, so this pins
    the two against each other. Otherwise a shared blind spot in the checker would make every count
    below agree for the wrong reason.

    On the list space the two can hardly disagree: one clause per combinator, no constraint, no
    constant argument. The cases that can are the other four: a binary combinator, an external
    predicate, two clauses sharing a terminal, and a clause that writes a literal.

    Args:
        build (Callable): Builds the space.
        start: The queried non-terminal.
        signature (dict): The function symbols with their arities.
        bound (int): The size bound for the oracle.
        depth (int): The depth bound for the resolution, chosen to cover the size bound.
    """
    space = build()
    by_checker = {str(term) for term in inhabitants_within(space, start, signature, bound)}
    by_resolution = {
        str(term) for term in space.depth_first_resolution(start, max_depth=depth) if term_size(term) <= bound
    }
    assert by_checker == by_resolution
    assert by_checker, "an empty agreement would prove nothing"


def all_nodes(root):
    """Return every node of the retained tree, root first.

    Args:
        root (CountedNode): The root of the retained tree.

    Returns:
        list[CountedNode]: All nodes, in pre-order.
    """
    collected = []
    pending = [root]
    while pending:
        node = pending.pop()
        collected.append(node)
        pending.extend(node.children)
    return collected


def constant_cost(_tree):
    """Map every inhabitant to one value, so all of them share a cost class.

    Args:
        _tree (Tree): The inhabitant. Ignored.

    Returns:
        str: The single cost value.
    """
    return "*"


def leaf_count(tree):
    """Count the leaves of a term, which is a cost function that is not the size.

    Args:
        tree (Tree): The inhabitant.

    Returns:
        int: The number of leaf positions.
    """
    return len(tree.leaf_positions())


# ---------------------------------------------------------------------------
# The counts themselves, against exhaustive enumeration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bound", [1, 2, 3, 4, 5])
def test_root_counts_are_the_cost_counts_on_the_list_space(bound):
    """``B_r(a) = N_r(a)`` on the unambiguous list space, for every bound.

    Args:
        bound (int): The size bound ``D`` under test.
    """
    space = list_space()
    root = branch_counts(generator_query(space, LIST), bound, term_size)
    assert dict(root.counts) == cost_counts(space, LIST, LIST_SIGNATURE, bound, term_size)


@pytest.mark.parametrize("bound", [1, 2, 3, 4, 5, 6])
def test_root_counts_are_the_cost_counts_on_the_expression_space(bound):
    """``B_r(a) = N_r(a)`` where the inhabitants of one size differ in shape.

    Args:
        bound (int): The size bound ``D`` under test.
    """
    space = expression_space()
    root = branch_counts(generator_query(space, EXPR), bound, term_size)
    assert dict(root.counts) == cost_counts(space, EXPR, EXPR_SIGNATURE, bound, term_size)


@pytest.mark.parametrize("bound", [3, 4, 5, 6])
def test_root_counts_are_the_cost_counts_with_an_external_predicate(bound):
    """``B_r(a) = N_r(a)`` where a predicate couples two holes.

    The predicate rejects some completions only once both holes are filled, so a count that
    treated the holes independently would come out too large.

    Args:
        bound (int): The size bound ``D`` under test.
    """
    space = constrained_space()
    root = branch_counts(generator_query(space, PAIR), bound, term_size)
    assert dict(root.counts) == cost_counts(space, PAIR, PAIR_SIGNATURE, bound, term_size)


@pytest.mark.parametrize("bound", [1, 2, 3, 4, 5, 6, 7])
def test_root_counts_are_the_cost_counts_where_a_clause_writes_two_symbols(bound):
    """``B_r(a) = N_r(a)`` where a clause fixes a constant argument as well as its terminal.

    Every other reference space grows a partial inhabitant by one symbol per clause, so a
    bookkeeping that counted clause applications instead of symbols would pass everywhere else.
    Here ``tag`` writes two, and counting it as one lets terms of size ``2D - 1`` pass a bound of
    ``D``. A repository whose combinators take literal parameters is built almost entirely from
    clauses of this shape.

    Args:
        bound (int): The size bound ``D`` under test.
    """
    space = two_symbol_clause_space()
    root = branch_counts(generator_query(space, TAGGED), bound, term_size)
    assert dict(root.counts) == cost_counts(space, TAGGED, TAGGED_SIGNATURE, bound, term_size)


def test_no_counted_term_exceeds_the_bound_where_a_clause_writes_two_symbols():
    """The retained success nodes stay within the bound, symbol for symbol.

    The reading of the test above that does not depend on the oracle: sizes in this space are
    odd, so a bookkeeping that charged ``tag`` one symbol instead of two would report counts at
    even values and at values beyond ``D``.
    """
    for bound in (3, 5, 7):
        root = branch_counts(generator_query(two_symbol_clause_space(), TAGGED), bound, term_size)
        assert set(root.counts) <= set(range(1, bound + 1))
        assert all(value % 2 == 1 for value in root.counts)
        assert dict(root.counts) == {size: 2 ** ((size - 1) // 2) for size in range(1, bound + 1, 2)}


@pytest.mark.parametrize("cost", [constant_cost, leaf_count])
def test_root_counts_hold_for_cost_functions_other_than_size(cost):
    """Random search needs no monotone cost, since the size bound alone secures finiteness.

    A constant cost puts every inhabitant into one class, and the leaf count is monotone in
    neither direction along a branch; both must still be counted exactly.

    Args:
        cost (Callable): The cost function under test.
    """
    space = expression_space()
    root = branch_counts(generator_query(space, EXPR), 5, cost)
    assert dict(root.counts) == cost_counts(space, EXPR, EXPR_SIGNATURE, 5, cost)


def test_list_cost_counts_follow_the_closed_form():
    """A list of length ``l`` has size ``l + 1``, and ``3^l`` lists share that size."""
    root = branch_counts(generator_query(list_space(), LIST), 6, term_size)
    assert dict(root.counts) == {size: 3 ** (size - 1) for size in range(1, 7)}


# ---------------------------------------------------------------------------
# The recursion itself
# ---------------------------------------------------------------------------


def test_every_inner_node_is_the_sum_of_its_children():
    """``B_n(a) = sum over children``, at every node of the retained tree."""
    root = branch_counts(generator_query(expression_space(), EXPR), 5, term_size)
    for node in all_nodes(root):
        if not node.children:
            continue
        summed: dict = {}
        for child in node.children:
            for value, count in child.counts.items():
                summed[value] = summed.get(value, 0) + count
        assert dict(node.counts) == summed


def test_a_success_node_within_the_bound_counts_exactly_its_own_inhabitant():
    """``B_n(a) = 1`` at ``a = c(t)`` and 0 elsewhere, for a success node of inhabitant ``t``."""
    root = branch_counts(generator_query(list_space(), LIST), 4, term_size)
    successes = [node for node in all_nodes(root) if node.inhabitant is not None]
    assert successes, "the list space has success nodes within the bound"
    for node in successes:
        assert dict(node.counts) == {term_size(node.inhabitant): 1}
        assert node.children == ()


def test_a_node_beyond_the_bound_is_not_retained():
    """A success node whose inhabitant exceeds the bound contributes nothing.

    Its branch counts vanish, so random search would never choose it, the expansion drops it, and
    the retained tree must not hold it either.
    """
    root = branch_counts(generator_query(list_space(), LIST), 3, term_size)
    for node in all_nodes(root):
        if node.inhabitant is not None:
            assert term_size(node.inhabitant) <= 3
        assert sum(node.counts.values()) > 0, "a retained node has a nonvanishing count"


def test_an_empty_bound_retains_nothing():
    """No term has size 0, so a bound of 0 leaves the root with vanishing counts."""
    root = branch_counts(generator_query(list_space(), LIST), 0, term_size)
    assert dict(root.counts) == {}
    assert root.children == ()


@pytest.mark.parametrize(
    ("build", "start"),
    [
        (list_space, LIST),
        (expression_space, EXPR),
        (constrained_space, PAIR),
        (two_symbol_clause_space, TAGGED),
    ],
)
def test_the_carried_size_is_the_size_of_the_materialized_partial_inhabitant(build, start):
    """The recursion carries the size along instead of materializing the term at every node.

    Applying a clause writes its terminal at a hole and turns each constant argument into a leaf,
    so the partial inhabitant grows by exactly that many symbols, which is what lets the cutoff
    be an addition rather than a traversal.  If that arithmetic were too generous the retained
    tree would hold nodes beyond the bound, which is what this checks; if it were too strict the
    counts would come out short, which the brute-force comparisons above check.

    Args:
        build (Callable): Builds the space.
        start: The queried non-terminal.
    """
    bound = 5
    root = branch_counts(generator_query(build(), start), bound, term_size)
    for node in all_nodes(root):
        if node.goal is None:
            continue
        assert term_size(partial_inhabitant(node.goal)) <= bound


@pytest.mark.parametrize(
    ("bound", "nodes"),
    [(0, 1), (1, 2), (2, 8), (3, 26), (4, 80)],
    ids=["empty", "the leaf", "two levels", "three", "four"],
)
def test_the_retained_node_count_is_the_number_of_nodes_the_search_holds(bound, nodes):
    """The memory of the search, reported by the function that exists to report it.

    The counts have to be complete before anything can be drawn from them, so the retained tree is
    held whole and its node count is what a caller budgets against. On the list space every node
    but the leaves has three children, so the retained tree at bound ``D`` holds ``3^D - 1`` nodes
    from bound 1 on, and a count that was off by the root or by the leaves would show at once.

    Args:
        bound (int): The size bound ``D``.
        nodes (int): The number of nodes the retained tree holds.
    """
    root = branch_counts(generator_query(list_space(), LIST), bound, term_size)
    assert retained_node_count(root) == nodes

    walked = 0
    pending = [root]
    while pending:
        node = pending.pop()
        walked += 1
        pending.extend(node.children)
    assert walked == nodes


def test_a_negative_size_bound_is_an_error_for_the_tree_form_too():
    """A bound that counts function symbols has no negative value to mean.

    The table form refuses one, and the tree form has to refuse it for the same reason: there is no
    term of negative size, so a caller who passes one has a number that came from somewhere else,
    and answering with an empty count would hide that.
    """
    with pytest.raises(ValueError, match="cannot be negative"):
        branch_counts(generator_query(list_space(), LIST), -1, term_size)


@pytest.mark.parametrize("bound", [2, 3, 4, 5])
def test_a_partial_term_query_counts_the_completions_of_its_prescribed_term(bound):
    """A prescribed term is counted from the symbols it already carries, not from nothing.

    A generator query starts from a bare variable, and its root has size 0. A partial-term query
    starts from the goals the prescribed term derives, whose partial inhabitants already carry the
    prescribed symbols, so the size those goals start at is what the term prescribes. Charging them
    0 would let the counts run one symbol past the bound, and charging the whole term at every goal
    would cut them short.

    The oracle is the same generate-and-check as everywhere else, restricted to the terms whose
    root is the prescribed symbol. It shares nothing with the recursion, and it is what says that
    the completions counted here really are the completions of this term.

    Args:
        bound (int): The size bound ``D`` under test.
    """
    space = expression_space()
    query = residual_query(space, EXPR, Tree(neg, (Tree(lit, ()),)), (0,))
    counted = branch_counts(query, bound, term_size)

    expected: dict[int, int] = {}
    for term in inhabitants_within(space, EXPR, EXPR_SIGNATURE, bound):
        if term.root == neg:
            expected[term_size(term)] = expected.get(term_size(term), 0) + 1

    assert dict(counted.counts) == expected
    assert expected, "the bound must admit completions, or the test asserts nothing"


def test_the_children_of_a_node_stand_in_clause_order():
    """The retained children follow the clause order of the space, not its reverse.

    The recursion walks its stack, so it has to reverse twice to end up in clause order, and
    getting that wrong is invisible in every count: the counts are a sum over the children and do
    not depend on their order. What does depend on it is a sampler driven by them: the
    conditioned keys are handed to the children in this order, so a seeded run reproduces only as
    long as the order does.
    """
    space = expression_space()
    root = branch_counts(generator_query(space, EXPR), 4, term_size)
    terminals = [rule.terminal for rule in space.get(EXPR)]
    assert len(terminals) > 1, "one clause per expansion would make the order unobservable"

    # the root's children come from the initial clause list; the inner ones from an expansion,
    # which is the path that has to reverse its stack twice
    inner = [node for node in all_nodes(root) if node.goal is not None and len(node.children) == len(terminals)]
    assert inner, "the space must produce an expansion with one child per clause"

    for node in [root, *inner]:
        assert [child.goal.constructors[max(child.goal.constructors)] for child in node.children] == terminals


# ---------------------------------------------------------------------------
# The retained tree is finite, on every query
# ---------------------------------------------------------------------------


def test_counting_halts_on_a_recursive_space_without_a_depth_bound():
    """The recursion carries no depth bound, and the size bound is what makes it finite.

    ``wrap`` makes ``W`` recursive, so the derivation tree below a ``pair`` goal is infinite and a
    traversal that did not cut on the size bound would not return. The space also carries a
    predicate, which is what the tree form is allowed to do: it counts branches it actually walks
    and needs no hypothesis about them.
    """
    root = branch_counts(generator_query(constrained_space(), PAIR), 5, term_size)
    assert sum(root.counts.values()) > 0


def test_no_retained_branch_is_longer_than_the_bound():
    """Each internal step fixes one function-symbol occurrence, shared by every completion below.

    So a branch of more than ``D`` expansions has no completion within the bound, and the
    retained tree has depth at most ``D``.
    """
    bound = 5
    root = branch_counts(generator_query(expression_space(), EXPR), bound, term_size)
    depths = []
    pending = [(root, 0)]
    while pending:
        node, depth = pending.pop()
        depths.append(depth)
        pending.extend((child, depth + 1) for child in node.children)
    assert max(depths) <= bound


def test_the_retained_tree_grows_with_the_bound_only():
    """Raising the bound deepens the search, and lowering it must retain strictly less."""
    query = generator_query(expression_space(), EXPR)
    sizes = [len(all_nodes(branch_counts(query, bound, term_size))) for bound in (2, 3, 4, 5)]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


# ---------------------------------------------------------------------------
# Unambiguity within the bound: the validation tool
# ---------------------------------------------------------------------------


def test_the_unambiguity_tool_accepts_a_space_with_one_clause_per_combinator():
    """Each combinator carries one clause, so the derivation of a term is determined by it."""
    assert_unambiguous_within(generator_query(list_space(), LIST), 5)
    assert branch_multiplicities(branch_counts(generator_query(list_space(), LIST), 5, term_size)) == {}


def test_the_unambiguity_tool_names_the_inhabitant_that_is_derived_twice():
    """The tool decides unambiguity within the bound and reports the witness.

    ``merge`` carries two paths of arity one onto ``M``, so the inhabitation emits two clauses
    that differ in the sort of their argument.  ``base`` inhabits both sorts, so ``merge(base)``
    ends two success branches while ``merge(alt)`` ends one, so the space carries both cases, and
    a tool that reported ambiguity for every intersection would fail on the second.
    """
    space = ambiguous_space()
    query = generator_query(space, AMBIGUOUS_TARGET)
    multiplicities = branch_multiplicities(branch_counts(query, 3, term_size))

    assert {str(term): count for term, count in multiplicities.items()} == {str(Tree(merge, (Tree(base, ()),))): 2}
    with pytest.raises(ValueError, match="2 derivations"):
        assert_unambiguous_within(query, 3)


def test_a_nullary_intersection_cannot_make_a_space_ambiguous():
    """An intersection on a nullary combinator collapses to one clause.

    The structural precondition behind the space above, and the reason the first attempt at an
    ambiguous reference space failed: the inhabitation emits one clause per admissible subset of
    paths, and for arity 0 every such subset yields the same clause. Ambiguity therefore needs an
    intersection of *arrows*, which is a structural condition on the space and not one the types
    alone announce.
    """
    specs = {
        base: SpecificationBuilder().suffix(Intersection(X, Y)),
        alt: SpecificationBuilder().suffix(Intersection(X, Y)),
        nullary_merge: SpecificationBuilder().argument("l", X).argument("r", Y).suffix(AMBIGUOUS_TARGET),
    }
    space = Synthesizer(specs).construct_solution_space(AMBIGUOUS_TARGET)
    query = generator_query(space, AMBIGUOUS_TARGET)

    assert branch_multiplicities(branch_counts(query, 3, term_size)) == {}
    assert_unambiguous_within(query, 3)


def test_root_counts_exceed_the_cost_counts_exactly_by_the_ambiguity():
    """Without unambiguity the branch counts exceed the cost counts, and this measures by how much.

    The branch counts count *branches*; the cost counts count *inhabitants*.  Their difference
    is the number of extra derivations, which is precisely what the counting samplers of the
    literature report as sampling in proportion to the parse count.
    """
    space = ambiguous_space()
    root = branch_counts(generator_query(space, AMBIGUOUS_TARGET), 3, term_size)
    multiplicities = branch_multiplicities(root)
    exact = cost_counts(space, AMBIGUOUS_TARGET, AMBIGUOUS_SIGNATURE, 3, term_size)
    surplus: dict = {}
    for inhabitant, count in multiplicities.items():
        surplus[term_size(inhabitant)] = surplus.get(term_size(inhabitant), 0) + count - 1

    assert surplus, "without a surplus this test only repeats the unambiguous case"
    assert dict(root.counts) == {
        value: exact.get(value, 0) + surplus.get(value, 0) for value in set(exact) | set(surplus)
    }
    assert root.total > sum(exact.values()), "the branch count exceeds the term count"
