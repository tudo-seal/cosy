"""Tree-kernel costs: the subtree and the subset-tree kernel.

Three levels of evidence, and they check different things.

The **golden tests** fix the two kernels on one pair of terms whose values are small enough to
count by hand: the reference ``f(a, b)`` against ``f(a, hole)`` and against itself.

The **oracles** are naive transcriptions of the two definitions, a double loop over the position
pairs and a recursion for ``C`` with no memoization, run against the implementation on seeded
random terms. They share no code with it: the implementation enumerates subterms into a multiset
and computes ``C`` bottom-up over a memo table, while the oracles read positions off
``Tree.positions`` and recurse. A memo table indexed by the wrong pair, or a production comparison
that forgets the arity, shows up here and nowhere else.

The **monotonicity tests** are the claim the cost layer rests on: filling a hole never lowers the
score against a fixed set of ground reference terms. They do not build partial inhabitants by
hand. They run the engine over a reference space, materialize the partial inhabitant of every goal
it builds, and compare the ones that stand in the refinement order, so the terms are those a
search actually orders.
"""

import random
import sys

import pytest

from cosy.core.tree import Tree
from cosy.search import partial_inhabitant
from cosy.search.kernels import (
    decay_weight,
    k_sst,
    k_st,
    normalized,
    reference_score,
    unit_weight,
)
from cosy.search.partial import Hole
from tests.search_fixtures import (
    EXPR,
    NUM,
    PAIR,
    WIDTH,
    constrained_space,
    equal_width_space,
    expression_space,
    literal_space,
)

# ---------------------------------------------------------------------------
# A small signature, and a few terms over it
# ---------------------------------------------------------------------------

HOLE_SORT = "E"


def hole(position):
    """Build a hole leaf at a position, as ``partial_inhabitant`` would.

    Args:
        position (tuple): The position of the hole in the partial inhabitant.

    Returns:
        Tree: The hole marker as a nullary leaf.
    """
    return Tree(Hole(position, HOLE_SORT), ())


A = Tree("a", ())
B = Tree("b", ())
F_A_B = Tree("f", (A, B))
F_A_HOLE = Tree("f", (A, hole((1,))))


# ---------------------------------------------------------------------------
# Oracles: the definitions transcribed, sharing nothing with the implementation
# ---------------------------------------------------------------------------


def naive_k_st(left, right, weight=unit_weight):
    """Sum the weights over the pairs of equal subterms, by a double loop over positions.

    The definition verbatim: a subterm is a position together with the whole term rooted there,
    and the sum ranges over the pairs of equal ones.

    Args:
        left (Tree): The first term.
        right (Tree): The second term.
        weight (Callable): The weight ``w_s``. (Default value = unit_weight)

    Returns:
        float: The subtree-kernel value.
    """
    total = 0.0
    for position in left.positions():
        subterm = left.subtree_at(position)
        for other in right.positions():
            if subterm == right.subtree_at(other):
                total += weight(subterm)
    return total


def naive_c(left, right, position, other):
    """Count the subset trees rooted at both positions, by the recursion of the definition.

    Args:
        left (Tree): The first term.
        right (Tree): The second term.
        position (tuple): A position of ``left``.
        other (tuple): A position of ``right``.

    Returns:
        float: ``C(p, q)``.
    """
    node = left.subtree_at(position)
    partner = right.subtree_at(other)
    if not node.children or not partner.children:
        return 0.0
    if (node.root, tuple(child.root for child in node.children)) != (
        partner.root,
        tuple(child.root for child in partner.children),
    ):
        return 0.0
    value = 1.0
    for index in range(len(node.children)):
        value *= 1.0 + naive_c(left, right, (*position, index), (*other, index))
    return value


def naive_k_sst(left, right):
    """Sum ``C`` over all position pairs, by the double sum of the definition.

    Args:
        left (Tree): The first term.
        right (Tree): The second term.

    Returns:
        float: The subset-tree-kernel value.
    """
    return sum(naive_c(left, right, position, other) for position in left.positions() for other in right.positions())


RANDOM_SIGNATURE = {"f": 2, "g": 1, "a": 0, "b": 0}


def random_term(rng, budget, position=()):
    """Build a random term over ``{f, g, a, b}`` whose leaves may be holes.

    Holes are drawn at the same rate as the constants, so the random terms cover the partial
    inhabitants as well as the ground ones, and the kernels have to be right on both. The
    branching symbols are drawn twice as often as each leaf: with an even draw two thirds of the
    terms are a bare leaf, and a pair of leaves exercises neither the memo table nor the
    multiplicities.

    Args:
        rng (random.Random): The source of randomness.
        budget (int): The remaining depth; at zero only leaves are drawn.
        position (tuple): The position of the term being built, which a hole records.
            (Default value = ())

    Returns:
        Tree: The term.
    """
    choices = ["a", "b", "hole"] if budget <= 0 else ["f", "f", "g", "g", "a", "b", "hole"]
    symbol = rng.choice(choices)
    if symbol == "hole":
        return hole(position)
    arity = RANDOM_SIGNATURE[symbol]
    return Tree(
        symbol,
        tuple(random_term(rng, budget - 1, (*position, index)) for index in range(arity)),
    )


def chain(length, symbol="cons", leaf="nil"):
    """Build the chain term of a given length: a unary spine over a nullary leaf.

    Args:
        length (int): The number of spine symbols.
        symbol (str): The unary symbol. (Default value = "cons")
        leaf (str): The nullary leaf. (Default value = "nil")

    Returns:
        Tree: The chain.
    """
    term = Tree(leaf, ())
    for _ in range(length):
        term = Tree(symbol, (term,))
    return term


# ---------------------------------------------------------------------------
# The golden values, counted by hand
# ---------------------------------------------------------------------------


def test_the_golden_subtree_kernel_rises_when_the_hole_is_filled():
    """``k_ST`` rises from 1 to 3 when the hole is filled, counted by hand.

    The reference ``f(a, b)`` has the subterms ``a``, ``b`` and ``f(a, b)``. At ``f(a, hole)``
    only ``a`` is shared, because the hole is a fresh symbol and the root's subterm therefore
    matches nothing; filling the hole makes all three match.
    """
    reference = [F_A_B]
    assert reference_score(F_A_HOLE, reference, k_st) == 1
    assert reference_score(F_A_B, reference, k_st) == 3


def test_the_golden_subset_tree_kernel_rises_when_the_hole_is_filled():
    """``k_SST`` rises from 0 to 1 when the hole is filled, counted by hand.

    At ``f(a, hole)`` the production ``f -> a hole`` matches none in the reference, so the only
    internal position contributes nothing; after the fill the two terms share the single subset
    tree ``f(a, b)``.
    """
    reference = [F_A_B]
    assert reference_score(F_A_HOLE, reference, k_sst) == 0
    assert reference_score(F_A_B, reference, k_sst) == 1


# ---------------------------------------------------------------------------
# Against the oracles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_subtree_kernel_agrees_with_the_definition(seed):
    """The implementation reproduces the double loop of the definition on random terms.

    Args:
        seed (int): The seed of this case; each case brings its own generator.
    """
    rng = random.Random(seed)
    for _ in range(20):
        left = random_term(rng, 3)
        right = random_term(rng, 3)
        assert k_st(left, right) == naive_k_st(left, right)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_subset_tree_kernel_agrees_with_the_definition(seed):
    """The memoized bottom-up ``C`` reproduces the plain recursion of the definition.

    This is the sharp test of the memo table: an entry keyed on the wrong pair, or a table filled
    in an order that never hits it, changes the value here while leaving the golden numbers
    intact. Both need a pair of matching productions *nested* inside another, which two
    independently drawn terms rarely have, so each term is also compared against itself, where
    every matching production of the term nests inside its parent.

    Args:
        seed (int): The seed of this case.
    """
    rng = random.Random(seed)
    for _ in range(20):
        left = random_term(rng, 4)
        right = random_term(rng, 4)
        assert k_sst(left, right) == naive_k_sst(left, right)
        assert k_sst(left, left) == naive_k_sst(left, left)
        assert k_sst(right, right) == naive_k_sst(right, right)


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_weighted_subtree_kernel_agrees_with_the_definition(seed):
    """A non-unit weight is applied per shared subterm, not per pair of terms.

    Args:
        seed (int): The seed of this case.
    """
    rng = random.Random(seed)
    weight = decay_weight(0.5)
    for _ in range(20):
        left = random_term(rng, 3)
        right = random_term(rng, 3)
        assert k_st(left, right, weight=weight) == pytest.approx(naive_k_st(left, right, weight))


# ---------------------------------------------------------------------------
# Kernel properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [21, 22, 23])
def test_both_kernels_are_symmetric(seed):
    """``k(t, r) = k(r, t)``, since a kernel is a symmetric function.

    Args:
        seed (int): The seed of this case.
    """
    rng = random.Random(seed)
    for _ in range(20):
        left = random_term(rng, 3)
        right = random_term(rng, 3)
        assert k_st(left, right) == k_st(right, left)
        assert k_sst(left, right) == k_sst(right, left)


@pytest.mark.parametrize("seed", [31, 32, 33])
def test_the_subtree_kernel_is_positive_on_every_term(seed):
    """``k_ST(t, t) > 0``: every term is its own subterm, and every weight is positive.

    Args:
        seed (int): The seed of this case.
    """
    rng = random.Random(seed)
    for _ in range(20):
        term = random_term(rng, 3)
        assert k_st(term, term) > 0


def test_the_subset_tree_kernel_vanishes_exactly_on_leaves():
    """``k_SST(t, t) > 0`` needs an internal position, since a leaf roots no subset tree.

    The documented edge case of the normalization: ``C`` is zero at a leaf by definition, so a
    one-node term has no self-similarity to divide by. Recorded here as behavior rather than
    repaired, so that a later change to the leaf case cannot pass unnoticed.
    """
    assert k_sst(A, A) == 0
    assert k_sst(hole(()), hole(())) == 0
    assert k_sst(F_A_B, F_A_B) > 0


def test_the_subset_tree_kernel_compares_the_whole_production():
    """Equal symbols with different children symbols contribute nothing to ``k_SST``.

    ``f(a, a)`` and ``f(a, b)`` agree in the root symbol and share the subterm ``a``, so the
    subtree kernel scores them positively. Their productions ``f -> a a`` and ``f -> a b``
    differ, so ``C`` is zero at the only pair of internal positions and ``k_SST`` is zero. An
    implementation that compared the symbol alone would report 1 here.
    """
    f_a_a = Tree("f", (A, A))
    assert k_st(f_a_a, F_A_B) == 2
    assert k_sst(f_a_a, F_A_B) == 0
    # And the production does match when the children do: the reference sits one level down.
    wrapped = Tree("g", (F_A_B,))
    assert k_sst(F_A_B, wrapped) == 1


def test_the_subset_tree_kernel_reads_the_children_in_order():
    """A production is the symbol *and the sequence* of its children's symbols.

    The subset-tree kernel is stated over ranked trees of fixed arity, so the child at one index is
    compared with the child at the same index. ``f(a, b)`` and ``f(b, a)`` share no production and
    no subset tree, while they do share the subterms ``a`` and ``b``. A production read as a
    multiset would pass every other test in this file.
    """
    swapped = Tree("f", (Tree("b", ()), Tree("a", ())))
    assert k_sst(F_A_B, swapped) == 0
    assert k_st(F_A_B, swapped) == 2
    assert k_sst(F_A_B, F_A_B) == 1


def test_the_subset_tree_kernel_multiplies_over_every_child_of_a_production():
    """A production with two live children contributes the product of both, counted by hand.

    ``f(g(a), g(b))`` against itself: ``C`` is 1 at each ``g`` (their single child is a leaf) and
    ``(1 + 1) * (1 + 1) = 4`` at the root, so the kernel is 6. Every other hand-computed case in
    this file has at most one child with a live ``C`` below it, and there a product that consults
    only the first or only the last child gives the same answer as the real one.
    """
    left = Tree("f", (Tree("g", (A,)), Tree("g", (B,))))
    twin = Tree("f", (Tree("g", (A,)), Tree("g", (A,))))

    assert k_sst(left, left) == 6
    assert k_sst(twin, twin) == 8
    assert k_sst(twin, left) == 4


def test_the_subset_tree_kernel_accumulates_down_matching_productions():
    """``C`` at a position is a product over its children's ``C``, not a flat one per match.

    Hand-computed on ``g(g(a))`` against itself. The inner position has the production
    ``g -> a``, matches only itself, and contributes ``C = 1 + 0 = 1``. The root has the
    production ``g -> g``, matches only itself, and contributes ``C = 1 + C(inner) = 2``: the two
    subset trees rooted there are ``g(g)`` cut at the inner child and ``g(g(a))`` descended into.
    The kernel is their sum, 3.

    The two cross pairs are zero, since ``g -> g`` and ``g -> a`` are different productions, so an
    implementation whose child lookup never resolves reports 2 instead of 3. That failure is
    invisible in every value computed from a term of depth one, which is what the golden test and
    most random pairs are.
    """
    inner = Tree("g", (A,))
    nested = Tree("g", (inner,))
    assert k_sst(nested, nested) == 3
    # Against the inner term alone only the one matching production survives, with nothing below.
    assert k_sst(nested, inner) == 1
    # Depth three, so the accumulation runs over two levels rather than one.
    deeper = Tree("g", (nested,))
    assert k_sst(deeper, deeper) == 8
    # And across terms of different depth, where the matching positions are *not* the same path:
    # the root of ``nested`` pairs with the middle of ``deeper``. A memo entry filed under the
    # swapped pair is never found again here, while it stays invisible on the diagonal above.
    assert k_sst(deeper, nested) == 4


def test_the_subset_tree_kernel_on_chains_follows_the_recursion():
    """On chains, ``C`` between two spine positions is ``min(k, m)`` plus one when ``k = m``.

    With ``k`` and ``m`` the numbers of spine symbols still below the two positions, the
    productions match while both descend, and the recursion adds one per shared level; the
    branches part as soon as one side reaches ``cons -> nil`` and the other does not. Summed over
    the position pairs this gives the table below, which is an arithmetic identity rather than a
    value read off this implementation, and it grows with the nesting depth. A lookup into the
    memo table that misses would flatten every entry to the number of matching pairs.
    """

    def closed_form(left, right):
        """Sum ``C`` over the spine pairs of two chains.

        Args:
            left (int): The length of the first chain.
            right (int): The length of the second chain.

        Returns:
            int: The subset-tree kernel of the two chains.
        """
        return sum(
            min(below_left, below_right) + (1 if below_left == below_right else 0)
            for below_left in range(left)
            for below_right in range(right)
        )

    for left in range(6):
        for right in range(6):
            assert k_sst(chain(left), chain(right)) == closed_form(left, right)
    assert k_sst(chain(5), chain(5)) == 35


def test_holes_at_different_positions_are_different_symbols():
    """Two holes of one term never match each other, since the position tells them apart.

    ``Hole`` is a frozen dataclass over ``(position, nonterminal)``, so ``f(hole, hole)`` has three
    pairwise distinct subterms and scores 3 against itself. Were the two holes equal, each would
    match the other and the score would be 5. The monotonicity of the kernels reads exactly this
    freshness into the holes, and it is a property of ``Hole``, not an assumption of this module.
    """
    assert Hole((0,), HOLE_SORT) != Hole((1,), HOLE_SORT)
    two_holes = Tree("f", (hole((0,)), hole((1,))))
    assert k_st(two_holes, two_holes) == 3


def test_a_term_without_shared_substructure_scores_zero():
    """A term sharing no substructure with the reference set has score 0 under both kernels.

    Holes are the reason the partial case is covered too: ``f(hole, hole)`` and ``f(a, a)``
    share the root symbol and nothing else.
    """
    disjoint = Tree("q", (Tree("u", ()),))
    assert reference_score(disjoint, [F_A_B], k_st) == 0
    assert reference_score(disjoint, [F_A_B], k_sst) == 0
    two_holes = Tree("f", (hole((0,)), hole((1,))))
    assert k_st(two_holes, Tree("f", (A, A))) == 0
    assert k_sst(two_holes, Tree("f", (A, A))) == 0


def test_the_subtree_kernel_counts_shared_subterms_with_multiplicity():
    """A subterm occurring twice on one side counts twice, since the sum ranges over *pairs*.

    ``f(a, a)`` against ``g(a)`` shares ``a`` at two positions on the left and one on the right,
    which is two pairs. Counting shared subterm *values* instead would report 1.
    """
    assert k_st(Tree("f", (A, A)), Tree("g", (A,))) == 2
    assert k_st(Tree("f", (A, A)), Tree("f", (A, A))) == 5


# ---------------------------------------------------------------------------
# Scores against a reference set
# ---------------------------------------------------------------------------


def test_the_reference_score_sums_over_the_reference_set():
    """``s_R(t) = sum over r in R of k(t, r)``: one summand per member.

    Pinned against hand-computed values rather than against the same sum written twice: ``f(a, b)``
    shares three subterms with itself, one (``a``) with ``g(a)`` and one with ``a``.
    """
    reference = [F_A_B, Tree("g", (A,)), A]
    assert reference_score(F_A_B, reference, k_st) == 3 + 1 + 1
    assert reference_score(F_A_B, reference, k_sst) == 1 + 0 + 0
    assert reference_score(A, reference, k_st) == 1 + 1 + 1


def test_the_reference_score_of_an_empty_reference_set_is_the_empty_sum():
    """An empty reference set scores every term 0, since the sum has no summands.

    Not an error: the empty sum is what the definition says, and what a caller gets is a constant
    cost function, which the docstring says outright.
    """
    assert reference_score(F_A_B, [], k_st) == 0
    assert reference_score(F_A_HOLE, [], k_sst) == 0


def test_a_repeated_reference_term_counts_twice():
    """The score sums over the members it is given; deduplication is the caller's business.

    The reference is a set, and this module does not silently make one out of an iterable that is
    not. It is also walked once, so an iterable that can only be walked once is enough.
    """
    assert reference_score(F_A_B, [F_A_B, F_A_B], k_st) == 2 * k_st(F_A_B, F_A_B)
    assert reference_score(F_A_B, (term for term in [F_A_B, F_A_B]), k_st) == 2 * k_st(F_A_B, F_A_B)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [41, 42, 43])
def test_the_normalized_kernel_lies_in_the_unit_interval(seed):
    """``k(t, r) / sqrt(k(t, t) k(r, r))`` is in [0, 1], by Cauchy-Schwarz on a kernel.

    Args:
        seed (int): The seed of this case.
    """
    rng = random.Random(seed)
    normalized_st = normalized(k_st)
    for _ in range(20):
        left = random_term(rng, 3)
        right = random_term(rng, 3)
        value = normalized_st(left, right)
        assert 0.0 <= value <= 1.0 + 1e-12


def test_the_normalized_kernel_is_one_exactly_for_equal_terms():
    """Normalized self-similarity is 1, and every distinct pair falls strictly below it.

    Equality in Cauchy-Schwarz needs parallel feature vectors, and two distinct terms never have
    them: whichever of the two is not a subterm of the other contributes a coordinate the other
    lacks.
    """
    normalized_st = normalized(k_st)
    normalized_sst = normalized(k_sst)
    terms = [F_A_B, Tree("f", (A, A)), Tree("g", (F_A_B,)), Tree("g", (Tree("g", (A,)),))]
    for term in terms:
        assert normalized_st(term, term) == pytest.approx(1.0)
        assert normalized_sst(term, term) == pytest.approx(1.0)
    for left in terms:
        for right in terms:
            if left != right:
                assert normalized_st(left, right) < 1.0
                assert normalized_sst(left, right) < 1.0


def test_the_normalized_kernel_refuses_a_term_of_zero_self_similarity():
    """A self-similarity of zero is not silently read as "maximally dissimilar".

    ``k_SST`` scores a leaf zero, so ``k(t, t) = 0`` there and the quotient is 0/0. Returning 0
    would make such a term maximally dissimilar to *itself*, which is exactly the value a draw
    over kernel scores would then reward, so the failure has to stay visible.
    """
    normalized_sst = normalized(k_sst)
    with pytest.raises(ValueError, match="self-similarity"):
        normalized_sst(A, F_A_B)
    with pytest.raises(ValueError, match="self-similarity"):
        normalized_sst(F_A_B, A)


def test_normalization_removes_the_size_bias_the_raw_score_carries():
    """The normalization is what keeps a draw over kernel scores from clustering.

    On the *unary* chain, ``k_ST`` between chains of lengths ``a`` and ``b`` is ``min(a, b) + 1``,
    so the raw score cannot tell a term equal to the reference from a much longer one that merely
    contains it: both score 4 against a chain of length 3. The normalization divides by the
    self-similarities, which grow with the term, and separates them: 1.0 against 0.60. (On the
    binary list family the raw kernel does separate the two, since the extra leaves contribute;
    the point is that one may not rely on it.) A draw over unnormalized scores would read the long
    term as no farther away than the identical one.
    """
    reference = chain(3)
    identical = chain(3)
    longer = chain(10)
    assert k_st(identical, reference) == k_st(longer, reference)
    normalized_st = normalized(k_st)
    assert normalized_st(identical, reference) == pytest.approx(1.0)
    assert normalized_st(longer, reference) < 0.7


# ---------------------------------------------------------------------------
# Closed forms on two families
# ---------------------------------------------------------------------------


def test_the_subtree_kernel_on_a_unary_chain_counts_the_shared_tails():
    """On a *unary* chain, ``k_ST(a, b) = min(a, b) + 1``: the shared tails plus the shared leaf.

    A unary chain of length ``l`` has the tails of lengths ``0..l`` as its subterms, each once, so
    two of them share ``min(a, b) + 1``. This is the smallest family on which the subterm
    multiset is obvious by inspection, which is why it is here.
    """
    for left in range(6):
        for right in range(6):
            assert k_st(chain(left), chain(right)) == min(left, right) + 1


def test_the_subtree_kernel_reproduces_the_closed_form_on_binary_lists():
    """``k_ST(l, l') = l*l' + min(l, l') + 1`` on lists over a binary ``cons``.

    A different family from the unary chain above: a list here is ``cons(z, cons(z, ... nil))``
    over the signature ``{z, nil, cons}``, so ``cons``
    is *binary* and every spine position carries a ``z`` leaf of its own. Those leaves are what
    contributes the ``l*l'`` summand, one pair per pair of spine positions, on top of the
    ``min(l, l') + 1`` shared tails.

    Recorded because the closed form reads like double counting when one has the unary chain in
    mind, and the two families are easy to confuse. It is not: at ``l = 10`` the kernel returns
    111.
    """

    def natlist(length):
        """Build the list of ``length`` zeroes.

        Args:
            length (int): The number of elements.

        Returns:
            Tree: The list as a term over ``{z, nil, cons}``.
        """
        term = Tree("nil", ())
        for _ in range(length):
            term = Tree("cons", (Tree("z", ()), term))
        return term

    for left in range(7):
        for right in range(7):
            assert k_st(natlist(left), natlist(right)) == left * right + min(left, right) + 1
    assert k_st(natlist(10), natlist(10)) == 111


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def test_a_nonpositive_weight_is_rejected():
    """A positive weight is part of the subtree kernel, and a violation is the caller's mistake.

    A weight of zero drops a subterm out of the feature space silently, which breaks both the
    positivity of ``k(t, t)`` the normalization divides by and the monotonicity along a branch.
    """

    def zero_weight(_subterm):
        """Return an inadmissible weight.

        Args:
            _subterm (Tree): The subterm. Ignored.

        Returns:
            float: Zero.
        """
        return 0.0

    with pytest.raises(ValueError, match="positive"):
        k_st(F_A_B, F_A_B, weight=zero_weight)

    # both sides, or the kernel would be asymmetric in its failures: a weight admissible on the
    # subterms of one argument and not on the other's would raise in one direction only
    def zero_on_a(subterm):
        """Return an inadmissible weight for ``a`` alone.

        Args:
            subterm (Tree): The subterm to weight.

        Returns:
            float: Zero on ``a``, one elsewhere.
        """
        return 0.0 if subterm.root == "a" else 1.0

    only_a = Tree("g", (Tree("a", ()),))
    only_b = Tree("g", (Tree("b", ()),))
    with pytest.raises(ValueError, match="positive"):
        k_st(only_a, only_b, weight=zero_on_a)
    with pytest.raises(ValueError, match="positive"):
        k_st(only_b, only_a, weight=zero_on_a)
    with pytest.raises(ValueError, match="decay"):
        decay_weight(0.0)
    with pytest.raises(ValueError, match="decay"):
        decay_weight(-1.0)


def test_a_decay_weight_damps_the_large_subterms():
    """``w_s = decay ** size(s)`` weights a shared root below a shared leaf.

    Against itself, ``f(a, b)`` shares ``a`` (size 1), ``b`` (size 1) and ``f(a, b)`` (size 3),
    which at decay 1/2 is 1/2 + 1/2 + 1/8 rather than 3.
    """
    assert k_st(F_A_B, F_A_B, weight=decay_weight(0.5)) == pytest.approx(0.5 + 0.5 + 0.125)
    assert k_st(F_A_B, F_A_B, weight=decay_weight(1.0)) == k_st(F_A_B, F_A_B)


def test_a_hole_carries_the_weight_of_a_nullary_symbol():
    """``term_size`` counts a hole as 0, so a decay weight stays positive on it.

    The weight contract is ``w_s > 0`` for *every* subterm, holes included; a decay weight raised
    to the size of a bare hole is ``decay ** 0 = 1``, which is positive whatever the decay.
    """
    lone_hole = hole(())
    assert k_st(lone_hole, lone_hole, weight=decay_weight(0.5)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Monotonicity: filling a hole never lowers the score
# ---------------------------------------------------------------------------


def refines(coarse, fine):
    """Decide whether one partial inhabitant is the other with some of its holes filled.

    A hole is taken to refine to anything, the non-terminal it carries included, so the relation
    is wider than the one the engine walks. That costs nothing here: the kernels are monotone
    under any filling, so the extra pairs are extra checks rather than false ones.

    Args:
        coarse (Tree): The term whose holes may be filled.
        fine (Tree): The candidate refinement.

    Returns:
        bool: True if ``fine`` arises from ``coarse`` by replacing holes with terms.
    """
    if isinstance(coarse.root, Hole):
        return True
    if coarse.root != fine.root or len(coarse.children) != len(fine.children):
        return False
    return all(refines(left, right) for left, right in zip(coarse.children, fine.children, strict=True))


def search_nodes_of(space, start, bound, count):
    """Materialize the partial inhabitant of every goal the engine builds on a bounded run.

    The expansion filter is consulted on every goal that is created, successful ones included, so
    collecting there yields the search nodes themselves rather than a reconstruction of them.

    Args:
        space (SolutionSpace): The space to search.
        start: The queried non-terminal.
        bound (int): The depth bound of the run.
        count (int): The number of inhabitants to stream before stopping.

    Returns:
        tuple: The distinct partial inhabitants of the goals, and the streamed inhabitants.
    """
    seen = []

    def record(goal):
        """Record the partial inhabitant of one goal and admit the goal.

        Args:
            goal (Goal): The goal being created.

        Returns:
            bool: True, always.
        """
        seen.append(partial_inhabitant(goal))
        return True

    streamed = list(space.depth_first_resolution(start, max_count=count, max_depth=bound, goal_filter=record))
    return list(dict.fromkeys(seen)), streamed


MONOTONE_SPACES = [
    (expression_space, EXPR, 4),
    (equal_width_space, WIDTH, 4),
    (constrained_space, PAIR, 5),
    (literal_space, NUM, 5),
]

# The kernel, and how many pairs of the run below have to score above zero on the coarser side.
# A pair whose coarser term scores zero cannot fall, so those pairs hold for any non-negative
# implementation and prove nothing. The subset-tree kernel matches a production as soon as one is
# complete, so it scores partial terms early and reaches that count everywhere. The subtree kernel
# needs a whole shared subterm, which a space of nested unary clauses only offers once its nullary
# clause has been reached, so no count above zero holds for it across all four spaces.
KERNELS_AND_FALSIFIABLE_PAIRS = [
    pytest.param(k_st, 0, id="k_st"),
    pytest.param(k_sst, 4, id="k_sst"),
]


@pytest.mark.parametrize(("build", "start", "bound"), MONOTONE_SPACES)
@pytest.mark.parametrize(("kernel", "least_falsifiable"), KERNELS_AND_FALSIFIABLE_PAIRS)
def test_the_kernel_score_never_falls_when_a_hole_is_filled(build, start, bound, kernel, least_falsifiable):
    """The monotonicity claim, on the search nodes of four reference spaces.

    Every goal the engine builds is materialized as the term it denotes, and each pair standing in
    the refinement order is required not to lose score. The reference set is drawn from the
    inhabitants of the space itself, so the scores are not uniformly zero: a run in which nothing
    ever matches would pass for the wrong reason.

    Args:
        build (Callable): Builds the space.
        start: The queried non-terminal.
        bound (int): The depth bound of the run.
        kernel (Callable): The kernel to score with.
        least_falsifiable (int): How many pairs have to score above zero on the coarser side.
    """
    nodes, streamed = search_nodes_of(build(), start, bound, 6)
    by_rendering = {str(term): term for term in streamed}
    reference = [by_rendering[rendering] for rendering in sorted(by_rendering)[:3]]
    assert reference, "the space must have inhabitants within the bound"

    scores = [reference_score(term, reference, kernel) for term in nodes]
    compared = 0
    falsifiable = 0
    for index, coarse in enumerate(nodes):
        for other, fine in enumerate(nodes):
            if index == other or not refines(coarse, fine):
                continue
            compared += 1
            falsifiable += scores[index] > 0
            assert scores[index] <= scores[other], f"score fell from {scores[index]} to {scores[other]}"
    assert compared > 0, "the run must produce nodes that refine one another"
    assert falsifiable >= least_falsifiable, f"only {falsifiable} of {compared} pairs could have fallen"


def test_filling_a_hole_can_leave_the_score_where_it_was():
    """Monotone means non-decreasing: a step that matches nothing new keeps the score.

    Worth pinning separately, because an implementation that happened to be *strictly* increasing
    would satisfy every comparison above while contradicting the claim. The witness scores above
    zero on both sides, because a pair of zeros would also be kept by an implementation that
    rises on every match and simply never matched here.
    """
    reference = [F_A_B]
    before = Tree("f", (A, hole((1,))))
    after = Tree("f", (A, Tree("q", ())))

    assert reference_score(before, reference, k_st) == 1
    assert reference_score(after, reference, k_st) == 1


# ---------------------------------------------------------------------------
# Deep terms
# ---------------------------------------------------------------------------


def test_a_deep_term_is_walked_without_recursing_over_it():
    """Both kernels collect their positions with an explicit stack, whatever the depth.

    The naive oracles above recurse and raise on a term like this one. ``k_st`` is handed the
    same object twice, which settles its dict lookups on identity: what a lookup costs when it
    has to compare two equal terms is the test below. ``k_sst`` is quadratic in the internal
    positions, so it is measured on a shorter chain, one that already exceeds what the oracle
    could recurse over.
    """
    deep = chain(sys.getrecursionlimit() * 2)

    assert k_st(deep, deep) == deep.size

    shorter = chain(200)
    assert k_sst(shorter, shorter) > 0


def test_the_subtree_kernel_reaches_the_bottom_of_separately_built_terms():
    """A lookup that cannot settle on identity ends in a comparison, and that no longer bounds depth.

    ``k_st`` collects subterms into a dict, so a lookup whose hash matches ends in ``Tree.__eq__``.
    That comparison used to descend one interpreter frame per level and raised ``RecursionError``
    here, and the kernel was the first caller in the package to reach the bound. Scoring partial
    inhabitants against reference terms built elsewhere is exactly the case where the lookup cannot
    settle on identity, so the answer has to be the one the identical-object case above gives.
    """
    too_deep = sys.getrecursionlimit() * 2
    left, right = chain(too_deep), chain(too_deep)

    assert k_st(left, right) == left.size
