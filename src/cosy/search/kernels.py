"""Tree-kernel costs: scoring a search node by its similarity to a set of reference terms.

A kernel on a set is a symmetric, positive definite function that measures how similar two of its
elements are, and a *tree kernel* is a kernel on terms. A kernel compares two terms while a cost
function scores a single search node, and a fixed reference set bridges the two: a term is scored
by its summed similarity to the reference terms.

**Why this is a cost function on goals.** The recursion of a convolution kernel runs over the term
bottom-up, which is a term assignment into an algebra whose carrier records the scores against the
reference positions: the subtree kernel accumulates the shared subterms, the subset-tree kernel the
values of its table, and a sum over the reference set stays a convolution kernel. Either kernel is
therefore a cost function on goals, and a caller builds it by composition,
``reference_score(partial_inhabitant(goal), reference, kernel)``. The order it is a cost function
over is the order on the reals that the score carries.

**Why a search over these scores maximizes.** Best-first search is stated for minimization, while a
search for a term *similar* to the reference set wants the node of maximal score. That is the
order-dual: the frontier returns a node of maximal score, and the completeness and cost-order
results carry over with the order reversed. This module produces scores and takes no position on
the direction.

**Why the holes are fresh symbols.** A kernel scores a ground term, while the search scores a
partial inhabitant, whose holes are open. Each hole is read as a fresh function symbol occurring in
no reference term, so a substructure reaching into a hole matches nothing and filling a hole can
only add matches. Against a fixed set of ground reference terms both kernels below are therefore
monotone along every branch of the derivation tree, which is what a cost-so-far function has to be.
:class:`~cosy.search.partial.Hole` supplies the freshness: it is a frozen record of position and
non-terminal, so it never equals a combinator, and two holes of one term differ because their
positions do. The freshness is relative to *ground* reference terms, which is what a reference set
holds. Two partial inhabitants sharing a hole position and type do match there, which is also why
``k(t, t)`` is positive on a term of holes.

**Normalization.** :func:`normalized` offers ``k(t, r) / sqrt(k(t, t) k(r, r))``, because counting
kernels score a large term high against itself: the raw score reads small terms as distant and
large ones as close. A draw meant to spread its picks over a reference set needs the normalized
form, since under the raw score a long term that merely contains a reference term scores as close
to it as an identical one does, and the draw clusters instead of spreading.

**What is not monotone.** Two quantities built from these kernels lose the property, both for
the same reason, that ``k(t, t)`` grows as a term is filled. The metric they induce,
``d^2(t, r) = k(t, t) - 2 k(t, r) + k(r, r)``, is one: a search for terms *far* from the
reference set does not fit this construction, and random search, which needs no monotonicity, is
what serves that purpose. The normalized kernel is the other, and it is the one to watch, since
it is offered here and wanted for its own reasons: against the reference ``a``, filling the hole
of ``g(a, .)`` leaves the raw score at 1 and raises the self-similarity from 3 to 4, so the
normalized score falls from 0.577 to 0.5. What monotonicity buys is that a best-first search
streams its inhabitants in order of cost, so that is what a normalized score costs. A search that
randomizes its cost does not depend on it, and neither does a comparison of finished terms.

**Deliberate limits.** A kernel-based A* would need an upper bound on the score a hole can still
gain, which these kernels do not supply. The partial tree kernel of Moschitti, which relaxes the
production constraint, is not implemented.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from math import sqrt
from typing import TYPE_CHECKING, Any

from cosy.core.tree import Tree
from cosy.search.partial import term_size

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cosy.core.tree import Path

TreeKernel = Callable[[Tree[Any], Tree[Any]], float]
"""A kernel on terms: a symmetric function of two terms into the reals."""

Weight = Callable[[Tree[Any]], float]
"""The weight of a subterm in the subtree kernel: a positive value depending only on the subterm."""

Production = tuple[Any, tuple[Any, ...]]
"""The production at a position: its function symbol together with its children's symbols."""

__all__ = [
    "Production",
    "TreeKernel",
    "Weight",
    "decay_weight",
    "k_sst",
    "k_st",
    "normalized",
    "reference_score",
    "unit_weight",
]


def unit_weight(subterm: Tree[Any]) -> float:
    """Return the unit weight of a subterm: the default of the subtree kernel.

    With unit weights the subtree kernel counts the shared subterms.

    Args:
        subterm (Tree[Any]): The subterm being weighted. Ignored, since the weight is constant.

    Returns:
        float: One.
    """
    del subterm
    return 1.0


def decay_weight(decay: float) -> Weight:
    """Build the weight that damps a subterm by its size: ``w_s = decay ** size(s)``.

    The weights are a parameter of the subtree kernel and admit almost arbitrary choices, fixed
    before the comparison; weights decaying with the size damp the large subterms, the way the
    decay parameter of the subset-tree kernel damps its large fragments. ``term_size`` counts holes
    as zero, so a bare hole weighs ``decay ** 0 = 1`` and the contract of a positive weight survives
    on partial inhabitants.

    Args:
        decay (float): The factor per function symbol. Below one it damps, above one it
            amplifies, and at exactly one it reproduces :func:`unit_weight`.

    Returns:
        Weight: The weight function.

    The exponent is the size of the subterm, so the contract of a positive weight is the contract
    of a positive power. In floating point it holds up to the range: a decay below one underflows
    to zero on a subterm of about a thousand symbols, where the kernel then reports the weight as
    inadmissible, and one above one overflows there.

    Raises:
        ValueError: If ``decay`` is not strictly positive. The subtree kernel requires positive
            weights, and a nonpositive decay would drop every subterm above a certain size out of
            the feature space, breaking both the positivity of ``k(t, t)`` and the monotonicity
            along a branch.
    """
    if decay <= 0.0:
        msg = f"a subtree-kernel weight is positive, so the decay must be strictly positive: {decay}"
        raise ValueError(msg)

    def weight(subterm: Tree[Any]) -> float:
        """Return the weight of one subterm.

        Args:
            subterm (Tree[Any]): The subterm being weighted.

        Returns:
            float: ``decay`` raised to the number of function symbols in the subterm.
        """
        return decay ** term_size(subterm)

    return weight


def _subterm_counts(term: Tree[Any]) -> dict[Tree[Any], int]:
    """Return the subterms of a term as a multiset.

    A subterm is a position together with the entire term rooted there, so
    the same term occurring at two positions is two subterms. Collapsing them into a set and
    counting distinct values would undercount every kernel value on a term with repetition, which
    is why the multiplicity is carried rather than the mere presence.

    Args:
        term (Tree[Any]): The term to decompose.

    Returns:
        dict[Tree[Any], int]: One entry per distinct subterm, mapping it to its number of
            occurrences. Iterative, so a term deeper than the interpreter's recursion limit is
            fine.
    """
    counts: dict[Tree[Any], int] = {}
    pending = [term]
    while pending:
        current = pending.pop()
        counts[current] = counts.get(current, 0) + 1
        pending.extend(current.children)
    return counts


def _checked_weights(counts: dict[Tree[Any], int], weight: Weight) -> dict[Tree[Any], float]:
    """Evaluate the weight on every subterm, enforcing that a subtree-kernel weight is positive.

    Every subterm of both arguments is checked, not only the shared ones: which subterms two terms
    happen to share is not something a caller controls, and an error that surfaced only on some
    pairs would make the same weight function look correct on one comparison and fail on the next.

    Args:
        counts (dict[Tree[Any], int]): The subterm multiset whose keys are weighed.
        weight (Weight): The weight function.

    Returns:
        dict[Tree[Any], float]: The weight of each subterm.

    Raises:
        ValueError: If the weight function returns a value that is not strictly positive.
    """
    weights: dict[Tree[Any], float] = {}
    for subterm in counts:
        value = weight(subterm)
        if value <= 0.0:
            msg = (
                f"a subtree-kernel weight is positive, but the weight function returned {value} "
                f"for the subterm {subterm}"
            )
            raise ValueError(msg)
        weights[subterm] = value
    return weights


def k_st(term: Tree[Any], reference: Tree[Any], *, weight: Weight = unit_weight) -> float:
    """Return the subtree kernel of two terms.

    The kernel sums the weight over the pairs of equal subterms, where a subterm is a position
    together with the entire term rooted there. Equal subterms at ``i`` positions of one term and
    ``j`` of the other form ``i * j`` pairs, so the value is the weighted inner product of the two
    subterm multisets, which is what makes it a kernel and what makes ``k(t, t)`` grow with the
    term.

    Holes participate as ordinary nullary symbols. Against a ground reference term a hole matches
    nothing, so a subterm reaching into one contributes nothing and filling the hole can only add
    contributions, which is what makes the subtree kernel monotone along a branch.

    Args:
        term (Tree[Any]): The first term; may be a partial inhabitant with holes.
        reference (Tree[Any]): The second term.
        weight (Weight): The weight ``w_s > 0``, depending only on the subterm.
            (Default value = unit_weight, under which the kernel counts the shared subterms)

    Returns:
        float: The kernel value; zero exactly when the two terms share no subterm.

    Raises:
        ValueError: If the weight function returns a value that is not strictly positive on some
            subterm of either argument.
    """
    left = _subterm_counts(term)
    right = _subterm_counts(reference)
    # The weights of the left subterms are computed for their contract check alone: the sum below
    # reads the weight of a *shared* subterm, which is a key of the right multiset as well, and
    # ``w_s`` depends only on ``s``, so the two agree wherever both are defined.
    _checked_weights(left, weight)
    right_weights = _checked_weights(right, weight)

    total = 0.0
    for subterm, count in left.items():
        shared = right.get(subterm)
        if shared is not None:
            total += right_weights[subterm] * count * shared
    return total


def _production_of(node: Tree[Any]) -> Production:
    """Return the production at a node: its symbol together with its children's symbols.

    Args:
        node (Tree[Any]): The node.

    Returns:
        Production: The symbol and the tuple of child symbols. The arity is part of the tuple, so
            two nodes with equal productions have equally many children and the product of the
            subset-tree kernel runs over both.
    """
    return (node.root, tuple(child.root for child in node.children))


def _internal_positions(
    term: Tree[Any],
) -> list[tuple[Path, Tree[Any], Production]]:
    """Return the non-leaf positions of a term in breadth-first order, with their productions.

    Leaves are dropped here rather than skipped later, because ``C(p, q) = 0`` whenever either
    position is a leaf: the pairs involving one contribute nothing to the sum and need no entry in
    the memo table. Breadth-first order puts a position before all of its descendants, so
    traversing the list backwards computes every child pair before the parent pair that needs it.

    Args:
        term (Tree[Any]): The term to decompose.

    Returns:
        list[tuple[Path, Tree[Any], Production]]: The internal positions with their nodes and
            productions, shallowest first.
    """
    nodes: list[tuple[Path, Tree[Any], Production]] = []
    queue: deque[tuple[Tree[Any], Path]] = deque([(term, ())])
    while queue:
        current, path = queue.popleft()
        if current.children:
            nodes.append((path, current, _production_of(current)))
        for index, child in enumerate(current.children):
            queue.append((child, (*path, index)))
    return nodes


def k_sst(term: Tree[Any], reference: Tree[Any]) -> float:
    """Return the subset-tree kernel of two terms, with decay one.

    A subset tree is a position with, at each child, the choice to cut, keeping the child as a leaf
    of the fragment, or to descend and repeat, so that every retained position keeps *all*
    of its children. Terms are ranked trees of fixed arity, which is why the production, symbol
    plus children's symbols, is the unit of comparison: two positions carrying the same symbol
    over different children share no fragment rooted there. The kernel sums ``C(p, q)`` over the
    position pairs, with ``C(p, q) = 0`` if either position is a leaf or the productions differ,
    and ``C(p, q) = product over j of (1 + C(p.j, q.j))`` otherwise.

    ``C`` is memoized per position pair and computed bottom-up over both terms, so each pair is
    evaluated once; the plain recursion re-derives the deep pairs once per ancestor pair and
    recurses as deep as the terms are.

    Holes participate as ordinary symbols, and a production carrying one matches no production of
    a ground reference term. Filling the hole may complete that production and raise ``C`` from
    zero, and every ancestor's product then has a factor that rose and none that fell, which is
    what makes the subset-tree kernel monotone along a branch as well.

    Args:
        term (Tree[Any]): The first term; may be a partial inhabitant with holes.
        reference (Tree[Any]): The second term.

    Returns:
        float: The kernel value. Zero when either term is a single node, since a leaf roots no
            subset tree. See :func:`normalized` for what that means for the normalization.
    """
    left = _internal_positions(term)
    right = _internal_positions(reference)

    contributions: dict[tuple[Path, Path], float] = {}
    total = 0.0
    for path, node, production in reversed(left):
        for other, _, other_production in reversed(right):
            if production != other_production:
                continue
            value = 1.0
            for index in range(len(node.children)):
                value *= 1.0 + contributions.get(((*path, index), (*other, index)), 0.0)
            contributions[(path, other)] = value
            total += value
    return total


def normalized(kernel: TreeKernel) -> TreeKernel:
    """Wrap a kernel into its normalized form ``k(t, r) / sqrt(k(t, t) k(r, r))``.

    The counting kernels score a large term high against itself, so the raw score carries a size
    bias: a long term that merely contains the reference scores as high as an identical one.
    Dividing by the self-similarities removes the bias and puts every value in ``[0, 1]``, with
    one reached exactly on equal terms. A draw that spreads its picks over a reference set needs
    this form: it is the normalization that makes the draw spread rather than cluster.

    The wrapper closes over the kernel it is given and nothing else, so a kernel that takes further
    parameters has to arrive with them bound: ``normalized(k_st)`` is the unit-weight subtree
    kernel, and a decay-weighted one is ``normalized(partial(k_st, weight=decay_weight(0.5)))``.

    Normalization is applied per pair, before the sum over a reference set, so
    ``reference_score(t, R, normalized(k))`` sums normalized similarities rather than normalizing
    a summed one.

    Args:
        kernel (TreeKernel): The kernel to normalize.

    Returns:
        TreeKernel: The normalized kernel.
    """

    def normalized_kernel(term: Tree[Any], reference: Tree[Any]) -> float:
        """Return the normalized similarity of two terms.

        Args:
            term (Tree[Any]): The first term.
            reference (Tree[Any]): The second term.

        Returns:
            float: The quotient, in ``[0, 1]``.

        Raises:
            ValueError: If either term has self-similarity zero. The quotient is then ``0/0``,
                and no value substitutes for it: returning zero would call the term maximally
                dissimilar to itself, which is exactly the reading a kernel-diverse draw rewards,
                so the term would be drawn *because* it cannot be scored. ``k_sst`` scores a
                single-node term zero, which is the case a caller runs into.
        """
        self_similarity = kernel(term, term)
        reference_self_similarity = kernel(reference, reference)
        if self_similarity <= 0.0 or reference_self_similarity <= 0.0:
            msg = (
                f"the normalized kernel divides by sqrt(k(t, t) * k(r, r)), and a "
                f"self-similarity of zero leaves it undefined: k(t, t) = {self_similarity} for "
                f"{term}, k(r, r) = {reference_self_similarity} for {reference}. The "
                f"subset-tree kernel scores a term without an internal position zero. Score "
                f"such a term with the subtree kernel, or keep it out of the reference set"
            )
            raise ValueError(msg)
        return kernel(term, reference) / sqrt(self_similarity * reference_self_similarity)

    return normalized_kernel


def reference_score(term: Tree[Any], reference: Iterable[Tree[Any]], kernel: TreeKernel) -> float:
    """Return the score of a term against a reference set: ``s_R(t) = sum over r of k(t, r)``.

    This is the step from a kernel, which compares two terms, to a cost function, which scores
    one. Applied to the partial inhabitant of a search node it is a cost function on goals, and a
    search maximizing it looks for a term similar to the reference set, which is the order-dual of
    best-first search.

    The kernel is a required argument rather than a defaulted one: which kernel is in use decides
    what "similar" means. The subtree kernel counts shared subterms, the subset-tree kernel shared
    fragments with complete productions, and the two disagree on terms that share symbols
    without sharing productions.

    Args:
        term (Tree[Any]): The term to score; may be a partial inhabitant with holes.
        reference (Iterable[Tree[Any]]): The reference terms, walked once. Summed as given: the
            reference is a set, and a repeated member is counted once per occurrence rather than
            silently deduplicated. Every term has to be ground. A reference term carrying a hole
            can be matched by a hole of the scored term and stop being matched once that hole is
            filled, which is the one way to lose the monotonicity the module rests on.
        kernel (TreeKernel): The kernel, for instance :func:`k_st`, :func:`k_sst`, or either of
            them wrapped in :func:`normalized`.

    Returns:
        float: The summed similarity. Zero when the term shares no substructure with any
            reference term, and also for an empty reference set, which is the empty sum, and the
            resulting cost function is constant.
    """
    total = 0.0
    for other in reference:
        total += kernel(term, other)
    return total
