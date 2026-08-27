"""Subtree swap and subtree graft.

Recombination is closed by **rejection**, so the acceptance test is the whole operator, and four
things about it are pinned here:

* the batch holds **two offspring or none** for the swap, and one or none for the graft. The
  previous operator stopped at the first pair whose two children were not both valid and returned
  the empty batch, so a later pair that would have worked was never reached;
* the pairs are walked in a **uniform permutation of the pair set**, not in the product of two
  separately shuffled position lists, which keeps the first primary position against every
  secondary one and lets the primary parent dominate which exchange is tried;
* membership is the **checker**, and ``contains_tree`` is its decision procedure. The two decide
  the same predicate, which is pinned against genuine enumeration below;
* a maximum size, when given, sits **inside the acceptance test**: a candidate beyond it is
  rejected like one outside the language, and the next pair is tried.

Note on assertion style: an offspring is built by ``replace_subtree_at``, and the comparisons here
go through ``interpret()`` and position sets, which test the structure rather than cached fields.
"""

import itertools
import random
from collections import Counter

import pytest

from cosy.core.tree import Tree
from cosy.evolutionary_algorithms import Recombination, SubtreeGraft, SubtreeSwap
from cosy.search import (
    checker,
    depth_first,
    generator_query,
    partial_inhabitant,
    term_size,
)
from tests.ea_fixtures import (
    ASYMMETRIC_START,
    NULLARY_START,
    RECURSIVE_START,
    CountingSpace,
    a0,
    a2,
    asymmetric_space,
    bi,
    chain,
    f1,
    g1,
    leaf_c,
    nullary_space,
    parent,
    recursive_space,
    rendered,
    root2,
)


@pytest.fixture
def recursive():
    """Return the generator query on the primary recursive space.

    Returns:
        ResolutionQuery: Every ``C`` subtree is interchangeable there, so a pair always works.
    """
    return generator_query(recursive_space(), RECURSIVE_START)


@pytest.fixture
def asymmetric():
    """Return the generator query on the space where exactly one offspring can be valid.

    Returns:
        ResolutionQuery: ``g1`` fits the left argument of ``root2`` but not the right one.
    """
    return generator_query(asymmetric_space(), ASYMMETRIC_START)


def _inner(tree):
    """Return the positions that are neither the root nor a leaf.

    Args:
        tree (Tree): The term.

    Returns:
        list: The inner positions, sorted.
    """
    return sorted(tree.positions() - {()} - tree.leaf_positions())


# ---------------------------------------------------------------------------
# Closure by rejection
# ---------------------------------------------------------------------------


def test_every_offspring_is_an_inhabitant(recursive):
    """Whatever a batch holds lies in the tree language.

    Args:
        recursive: The recursive-space query fixture.
    """
    for seed in range(20):
        swap = SubtreeSwap(random.Random(seed))
        for offspring in swap.recombine(recursive, parent(3, 2), parent(2, 3)):
            assert checker(recursive.solution_space, recursive.start, offspring)


def test_the_parents_are_not_modified(recursive):
    """The operator is functional in its arguments.

    Args:
        recursive: The recursive-space query fixture.
    """
    first, second = parent(3, 2), parent(2, 3)
    before = (rendered(first), rendered(second))
    SubtreeSwap(random.Random(0)).recombine(recursive, first, second)
    assert (rendered(first), rendered(second)) == before


def test_membership_is_tested_once_per_candidate(recursive):
    """No candidate is handed to the checker twice; it is the expensive operation here.

    Args:
        recursive: The recursive-space query fixture.
    """
    counting = CountingSpace(recursive.solution_space)
    watched = generator_query(counting, recursive.start)
    SubtreeSwap(random.Random(1)).recombine(watched, parent(3, 2), parent(2, 3))
    assert counting.duplicate_calls() == []


# ---------------------------------------------------------------------------
# The batch
# ---------------------------------------------------------------------------


def test_a_swap_returns_two_offspring_or_none(recursive):
    """The batch of the swap has size two or zero.

    Args:
        recursive: The recursive-space query fixture.
    """
    for seed in range(20):
        batch = SubtreeSwap(random.Random(seed)).recombine(recursive, parent(2, 2), parent(3, 1))
        assert len(batch) in {0, 2}
    assert len(SubtreeSwap(random.Random(2)).recombine(recursive, parent(2, 2), parent(3, 1))) == 2


def test_a_lone_valid_child_is_not_returned(recursive):
    """The rejected alternative: one valid offspring of a pair is not a batch.

    The discriminator has to be a pair for which **some** exchange yields exactly one acceptable
    child and **none** yields two, since otherwise an operator returning the lone child passes.
    A size bound builds one: swapping a big subterm into the small parent breaks the bound for that
    child while the reverse direction stays under it, and there is no pair at which both fit.

    Args:
        recursive: The recursive-space query fixture.
    """
    small = Tree(parent(1, 1).root, (chain(1), chain(1)))
    big = Tree(parent(1, 1).root, (chain(9), chain(9)))
    bound = 8
    lonely = [
        (left, right)
        for left in _inner(small)
        for right in _inner(big)
        if (
            (term_size(small.replace_subtree_at(left, big.subtree_at(right))) <= bound)
            != (term_size(big.replace_subtree_at(right, small.subtree_at(left))) <= bound)
        )
    ]
    both = [
        (left, right)
        for left in _inner(small)
        for right in _inner(big)
        if term_size(small.replace_subtree_at(left, big.subtree_at(right))) <= bound
        and term_size(big.replace_subtree_at(right, small.subtree_at(left))) <= bound
    ]
    assert lonely, "the fixture must offer lone children"
    assert not both, "the fixture must offer no complete pair"

    for seed in range(40):
        assert SubtreeSwap(random.Random(seed), max_size=bound).recombine(recursive, small, big) == []


def test_a_lone_valid_child_is_not_returned_when_the_language_rejects_the_other(asymmetric):
    """The same, with membership rather than a size bound deciding it.

    ``g1`` inhabits ``P`` but not ``Q``, so exchanging it into the right argument of ``root2``
    produces exactly one valid child.

    Args:
        asymmetric: The asymmetric-space query fixture.
    """
    left = Tree(root2, (Tree(g1, (Tree(a0, ()),)), Tree(f1, (Tree(a0, ()),))))
    right = Tree(root2, (Tree(f1, (Tree(a0, ()),)), Tree(f1, (Tree(a0, ()),))))
    assert checker(asymmetric.solution_space, asymmetric.start, left)
    assert checker(asymmetric.solution_space, asymmetric.start, right)
    for seed in range(30):
        batch = SubtreeSwap(random.Random(seed)).recombine(asymmetric, left, right)
        assert len(batch) in {0, 2}
        for offspring in batch:
            assert checker(asymmetric.solution_space, asymmetric.start, offspring)


def test_a_graft_returns_one_offspring_or_none(recursive):
    """The graft has a single candidate per pair, so its batch has size one or zero.

    Args:
        recursive: The recursive-space query fixture.
    """
    for seed in range(20):
        batch = SubtreeGraft(random.Random(seed)).recombine(recursive, parent(2, 2), parent(3, 1))
        assert len(batch) in {0, 1}


def test_the_graft_builds_from_the_primary_parent(recursive):
    """The offspring is the **primary** parent with a subterm of the secondary one grafted in.

    The two parents have to be told apart by the assertion, and on symmetric parents they cannot
    be: over ``top(chain, chain)`` terms, grafting into the primary and grafting into the
    secondary produce the same *set* of offspring, so an operator with the roles exchanged passes.
    The parents here differ in shape (a binary branch against a chain), and the assertion is that
    every offspring is the primary with exactly one subterm replaced, which pins the direction.

    Args:
        recursive: The recursive-space query fixture.
    """
    primary = Tree(parent(1, 1).root, (Tree(bi, (leaf_c(), leaf_c())), chain(1)))
    secondary = Tree(parent(1, 1).root, (chain(4), chain(4)))

    def is_primary_with_one_replacement(offspring) -> bool:
        """Decide whether an offspring is the primary parent with one subterm exchanged.

        Args:
            offspring (Tree): The candidate.

        Returns:
            bool: True if putting one of the primary's subterms back reconstructs it exactly.
        """
        positions = offspring.positions()
        return any(
            rendered(offspring.replace_subtree_at(left, primary.subtree_at(left))) == rendered(primary)
            for left in _inner(primary)
            if left in positions
        )

    # The control: with the roles exchanged, most candidates fail that characterisation, so the
    # assertion below has something to catch.
    swapped_roles = [
        secondary.replace_subtree_at(right, primary.subtree_at(left))
        for left in _inner(primary)
        for right in _inner(secondary)
    ]
    assert not all(is_primary_with_one_replacement(other) for other in swapped_roles)

    for seed in range(40):
        batch = SubtreeGraft(random.Random(seed)).recombine(recursive, primary, secondary)
        assert len(batch) == 1
        assert is_primary_with_one_replacement(batch[0]), rendered(batch[0])


def test_a_parent_without_inner_positions_yields_the_empty_batch(recursive):
    """Neither root nor leaf leaves nothing to exchange in a term of depth one.

    Args:
        recursive: The recursive-space query fixture.
    """
    flat = parent(0, 0)  # top(lf, lf): the root plus two leaves
    assert SubtreeSwap(random.Random(4)).recombine(recursive, flat, parent(3, 3)) == []
    assert SubtreeGraft(random.Random(4)).recombine(recursive, flat, parent(3, 3)) == []


def test_a_single_node_parent_yields_the_empty_batch():
    """A one-node term has one position, which is both root and leaf.

    Regression: the previous operator removed it twice and raised ``ValueError``.
    """
    query = generator_query(nullary_space(), NULLARY_START)
    single = Tree(a2, ())
    assert SubtreeSwap(random.Random(5)).recombine(query, single, single) == []
    assert SubtreeGraft(random.Random(5)).recombine(query, single, single) == []


def test_offspring_take_material_from_both_parents(recursive):
    """A swap is an exchange: each offspring differs from the parent it was built from.

    Args:
        recursive: The recursive-space query fixture.
    """
    first, second = parent(1, 1), Tree(parent(1, 1).root, (chain(3), chain(3)))
    batch = SubtreeSwap(random.Random(6)).recombine(recursive, first, second)
    assert len(batch) == 2
    assert batch[0] != first
    assert batch[1] != second


def test_an_offspring_may_be_deeper_than_its_parent(recursive):
    """An exchange can deepen a term, and that is not an error.

    It is the reason a run needs a route to keeping its individuals finite, and the ``max_size``
    parameter is that route.

    Args:
        recursive: The recursive-space query fixture.
    """
    shallow, deep = parent(1, 1), Tree(parent(1, 1).root, (chain(6), chain(1)))
    assert any(
        offspring.depth > shallow.depth
        for seed in range(40)
        for offspring in SubtreeSwap(random.Random(seed)).recombine(recursive, shallow, deep)
    )


# ---------------------------------------------------------------------------
# The size bound, inside the acceptance test
# ---------------------------------------------------------------------------


def test_a_candidate_beyond_the_size_bound_is_rejected(recursive):
    """No offspring exceeds ``max_size``.

    Args:
        recursive: The recursive-space query fixture.
    """
    for seed in range(30):
        batch = SubtreeSwap(random.Random(seed), max_size=8).recombine(recursive, parent(3, 3), parent(2, 4))
        assert all(term_size(offspring) <= 8 for offspring in batch)


def test_the_bound_rejects_a_candidate_rather_than_a_position(recursive):
    """The bound is an acceptance test, so a rejected pair does not stop the enumeration.

    A pre-filter over positions would decide what an exchange *could* reach and drop positions
    outright; the test here is that pairs whose candidates are too large are skipped while a later
    pair still succeeds.

    Args:
        recursive: The recursive-space query fixture.
    """
    first = Tree(parent(1, 1).root, (chain(1), chain(5)))
    second = Tree(parent(1, 1).root, (chain(1), chain(5)))
    found = [
        batch
        for seed in range(30)
        if (batch := SubtreeSwap(random.Random(seed), max_size=10).recombine(recursive, first, second))
    ]
    assert found, "every pair was rejected; the bound is acting as a filter, not as a test"
    assert all(term_size(offspring) <= 10 for batch in found for offspring in batch)


def test_a_negative_size_bound_is_refused():
    """A size bound counts symbols."""
    with pytest.raises(ValueError, match="negative"):
        SubtreeSwap(random.Random(0), max_size=-1)


# ---------------------------------------------------------------------------
# The order of the pairs
# ---------------------------------------------------------------------------


def _shared_primary_rate(orders):
    """Measure how often two consecutive pairs share their primary position.

    Args:
        orders (list[list[tuple]]): One enumeration per repetition.

    Returns:
        float: The fraction of consecutive pairs agreeing in the first component.
    """
    shared = sum(1 for order in orders for a, b in itertools.pairwise(order) if a[0] == b[0])
    total = sum(len(order) - 1 for order in orders)
    return shared / total


def test_the_pairs_are_a_uniform_permutation_of_the_pair_set():
    """The swap draws the order over the **pair set**.

    Measured on the internal enumeration, because the operator returns the first acceptable pair
    and never reports the order it walked. The discriminator is how often two consecutive pairs
    share their primary position: under a uniform permutation of the pair set that is
    ``(|right| - 1) / (|left| * |right| - 1)``, and under the product of two shuffles it is almost always.
    """
    first, second = parent(3, 3), parent(3, 3)
    left, right = _inner(first), _inner(second)
    pairs = SubtreeSwap(random.Random(7))._pairs(first, second)  # noqa: SLF001
    assert sorted(pairs) == sorted(itertools.product(left, right))

    orders = [
        SubtreeSwap(random.Random(seed))._pairs(first, second)  # noqa: SLF001
        for seed in range(300)
    ]
    expected = (len(right) - 1) / (len(left) * len(right) - 1)
    assert abs(_shared_primary_rate(orders) - expected) < 0.05


def test_the_product_order_would_fail_that_measurement():
    """Negative control: the previous enumeration, under the same measurement."""
    first, second = parent(3, 3), parent(3, 3)
    left, right = _inner(first), _inner(second)
    orders = []
    for seed in range(300):
        rng = random.Random(seed)
        shuffled_left, shuffled_right = list(left), list(right)
        rng.shuffle(shuffled_left)
        rng.shuffle(shuffled_right)
        orders.append(list(itertools.product(shuffled_left, shuffled_right)))
    expected = (len(right) - 1) / (len(left) * len(right) - 1)
    assert abs(_shared_primary_rate(orders) - expected) > 0.5


def test_no_acceptable_exchange_is_starved(recursive):
    """The consequence of the permutation, measured through the offspring.

    On a space where every pair is type-correct the batch is the exchange at the first pair, so
    the offspring distribution is the distribution over the pairs. Distinct pairs may build the
    same offspring, so what is asserted is that every reachable one keeps a substantial share.

    Args:
        recursive: The recursive-space query fixture.
    """
    first, second = parent(2, 2), parent(3, 1)
    left, right = _inner(first), _inner(second)
    counts: Counter = Counter(
        rendered(SubtreeSwap(random.Random(seed)).recombine(recursive, first, second)[0]) for seed in range(2000)
    )
    outcomes = {rendered(first.replace_subtree_at(p, second.subtree_at(q))) for p in left for q in right}
    assert set(counts) == outcomes
    assert min(counts.values()) > 0.3 * 2000 / len(left) / len(right)


# ---------------------------------------------------------------------------
# The checker
# ---------------------------------------------------------------------------


def _compositions(total, parts):
    """Enumerate the ways to write ``total`` as an ordered sum of positive integers.

    Args:
        total (int): The sum.
        parts (int): The number of summands.

    Yields:
        tuple[int, ...]: One composition.
    """
    if parts == 1:
        if total >= 1:
            yield (total,)
        return
    for head in range(1, total - parts + 2):
        for rest in _compositions(total - head, parts - 1):
            yield (head, *rest)


def _free_terms(symbols, max_size):
    """Enumerate every term of the free algebra over an alphabet, up to a size.

    Args:
        symbols (dict): Symbol to arity.
        max_size (int): The largest number of symbol occurrences.

    Returns:
        list[Tree]: Every term within the bound, members and non-members alike.
    """
    by_size: dict[int, list] = {0: []}
    for size in range(1, max_size + 1):
        built = []
        for symbol, arity in symbols.items():
            if arity == 0:
                if size == 1:
                    built.append(Tree(symbol, ()))
                continue
            for split in _compositions(size - 1, arity):
                built.extend(
                    Tree(symbol, tuple(children)) for children in itertools.product(*(by_size[part] for part in split))
                )
        by_size[size] = built
    return [term for size in range(1, max_size + 1) for term in by_size[size]]


def _within(bound):
    """Build a goal filter bounding the size of the partial inhabitant.

    Args:
        bound (int): The largest number of symbol occurrences.

    Returns:
        Callable: The filter.
    """

    def keep(goal) -> bool:
        """Decide whether a goal still fits the bound.

        Args:
            goal: The search node.

        Returns:
            bool: True if its partial inhabitant is within the bound.
        """
        return term_size(partial_inhabitant(goal)) <= bound

    return keep


def test_the_checker_decides_what_resolution_enumerates(asymmetric):
    """``contains_tree`` is the checker's decision procedure, not a second notion of membership.

    Every term of the free algebra up to size 5 is classified twice: by the checker, and by
    membership in what the generator query enumerates within that size. The verdicts must agree
    on every term, members and non-members alike. Every access runs through a resolution query,
    and this is that claim as a test.

    Args:
        asymmetric: The asymmetric-space query fixture.
    """
    bound = 5
    enumerated = {tree for tree in depth_first(asymmetric, goal_filter=_within(bound)) if term_size(tree) <= bound}
    candidates = _free_terms({a0: 0, f1: 1, g1: 1, root2: 2}, bound)
    assert len(candidates) > len(enumerated) > 0
    for candidate in candidates:
        assert checker(asymmetric.solution_space, asymmetric.start, candidate) == (candidate in enumerated), rendered(
            candidate
        )


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


def test_neither_the_root_nor_a_leaf_is_a_crossover_point():
    """Positions are inner ones, for both parents."""
    first, second = parent(2, 2), parent(2, 2)
    for left, right in SubtreeSwap(random.Random(0))._pairs(first, second):  # noqa: SLF001
        assert left != ()
        assert right != ()
        assert left not in first.leaf_positions()
        assert right not in second.leaf_positions()


def test_a_swap_at_the_root_pair_would_return_the_parents():
    """Why the root is excluded: the exchange there is the identity on the pair.

    Copies of the parents are what the driver produces with the probability left over from its
    two rates, so an operator producing them too would bypass that rate.
    """
    first, second = parent(2, 1), parent(1, 2)
    assert first.replace_subtree_at((), second.subtree_at(())) == second
    pairs = SubtreeSwap(random.Random(0))._pairs(first, second)  # noqa: SLF001
    assert () not in {left for left, _ in pairs}


def test_both_operators_satisfy_the_protocol():
    """The component class is structural, and both operators fill the same slot."""
    assert isinstance(SubtreeSwap(random.Random(0)), Recombination)
    assert isinstance(SubtreeGraft(random.Random(0)), Recombination)


def test_the_operators_do_not_touch_the_global_random_stream(recursive):
    """Reproducibility: the pair order comes from the operator's own generator.

    Args:
        recursive: The recursive-space query fixture.
    """
    random.seed(4321)
    before = random.random()
    random.seed(4321)
    SubtreeSwap(random.Random(0)).recombine(recursive, parent(2, 2), parent(2, 2))
    assert random.random() == before


def test_a_c_subtree_is_not_an_individual():
    """Guard for the fixtures: ``lf`` is a ``C``-term but not an ``S``-term."""
    assert not recursive_space().contains_tree(RECURSIVE_START, leaf_c())


# ---------------------------------------------------------------------------
# The order the pairs are built and drawn in
# ---------------------------------------------------------------------------


class _RecordingRandom(random.Random):
    """A generator that remembers every sequence it was asked to shuffle.

    Attributes:
        pools (list[list[tuple]]): One entry per shuffle, in the order they happened.
    """

    def __init__(self, seed: int) -> None:
        """Seed the generator.

        Args:
            seed (int): The seed.
        """
        super().__init__(seed)
        self.pools: list[list[tuple]] = []

    def shuffle(self, x, *args):
        """Record the sequence and shuffle it.

        Args:
            x: The sequence to shuffle in place.
            *args: Never used. Kept so the signature stays compatible with the one typeshed
                declares for Python 3.10, where ``Random.shuffle`` still carries a second
                parameter that later versions removed.
        """
        self.pools.append(list(x))
        super().shuffle(x)


def test_the_pair_list_is_built_from_sorted_pools(recursive):
    """The permutation depends on the seed alone, not on how the position sets iterate.

    ``positions()`` answers with a set. Shuffling a list whose order came out of set iteration
    makes a seeded run depend on the interpreter, and the permutation measurement above cannot see
    it, because every input order permutes uniformly.

    Args:
        recursive: The recursive-space query fixture.
    """
    first, second = parent(2, 2), parent(2, 3)
    for tree in (first, second):
        raw = tree.positions() - {()} - tree.leaf_positions()
        assert list(raw) != sorted(raw), "the fixture must expose an order differing from sorted"

    rng = _RecordingRandom(0)
    SubtreeSwap(rng).recombine(recursive, first, second)
    assert rng.pools == [list(itertools.product(_inner(first), _inner(second)))]


def test_a_candidate_of_exactly_the_bound_is_accepted_and_later_pairs_are_tried(recursive):
    """The bound is inclusive, and a pair that fails it does not end the search.

    ``parent(1, 1)`` against ``parent(1, 3)`` offers three exchanges, whose offspring sizes are
    ``(5, 7)``, ``(6, 6)`` and ``(7, 5)``. At ``max_size = 6`` exactly one of them passes, and both
    of its children sit on the bound. Two things follow that no other test here holds: a candidate
    of exactly the bound is accepted rather than rejected, and the operator walks past the pairs
    that fail instead of giving up on the first one.

    Args:
        recursive: The recursive-space query fixture.
    """
    first, second = parent(1, 1), parent(1, 3)
    for seed in range(15):
        batch = SubtreeSwap(random.Random(seed), max_size=6).recombine(recursive, first, second)
        assert batch, "the only acceptable pair sits on the bound and must be reached"
        assert max(term_size(offspring) for offspring in batch) == 6


def test_the_graft_rejects_a_candidate_outside_the_language_and_tries_the_next_pair(asymmetric):
    """The graft is closed by rejection too, and here a candidate is actually rejected.

    ``g1`` inhabits ``P`` but not ``Q``, so grafting it into the right argument of ``root2`` leaves
    the language. On the recursive space every pair is type-correct, so the first one is always
    taken and the acceptance test is never seen to say no.

    Args:
        asymmetric: The asymmetric-space query fixture.
    """
    primary = Tree(root2, (Tree(f1, (Tree(a0, ()),)), Tree(f1, (Tree(a0, ()),))))
    secondary = Tree(root2, (Tree(g1, (Tree(a0, ()),)), Tree(f1, (Tree(a0, ()),))))
    rejected = primary.replace_subtree_at((1,), secondary.subtree_at((0,)))
    assert not checker(asymmetric.solution_space, asymmetric.start, rejected), (
        "the fixture must offer a pair the acceptance test refuses"
    )
    for seed in range(30):
        batch = SubtreeGraft(random.Random(seed)).recombine(asymmetric, primary, secondary)
        assert len(batch) == 1
        assert checker(asymmetric.solution_space, asymmetric.start, batch[0])


def test_the_graft_rejects_a_candidate_beyond_the_size_bound(recursive):
    """``max_size`` sits inside the graft's acceptance test as well, not only the swap's.

    Args:
        recursive: The recursive-space query fixture.
    """
    primary, secondary = parent(1, 1), parent(1, 3)
    for seed in range(40):
        batch = SubtreeGraft(random.Random(seed), max_size=6).recombine(recursive, primary, secondary)
        assert all(term_size(offspring) <= 6 for offspring in batch)
