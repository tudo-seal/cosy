"""Recombination: exchanging subterms, and testing what comes out.

Recombination is binary, and unlike mutation it can leave the tree language. The subterm a position
admits is described by the residual there, and an arbitrary subterm of another individual need not
lie in it. The operators here are therefore **closed by rejection**, building a candidate and
testing its membership, where resolution mutation is closed by construction. The test is the
*checker*: the resolution query on a ground term denotes a finite derivation tree, so membership is
decidable, and :func:`cosy.search.queries.checker` is the named entry to it.

The result of an operator is a **batch**, a multiset of at most two offspring. A move that finds no
acceptable pair returns the empty batch, and the surrounding procedure draws new parents. A batch
holding one offspring is deliberately absent from the swap, since the two offspring of a swap are
the two halves of one exchange and returning only the valid one would break that.

**Positions are neither root nor leaf**, for both parents. Uniform choice over all positions would
favour the leaves, which a term has most of, and at the pair of roots a swap returns the parents
unchanged, which would smuggle copies past the rates of the driver. The driver already makes
copies, with the probability left over from ``mutation_rate`` and ``recombination_rate``. Mutation
includes the root because reachability needs it there, and the asymmetry is intended.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from cosy.core.solution_space import NT, G, T
from cosy.search.partial import term_size
from cosy.search.queries import checker

if TYPE_CHECKING:
    import random

    from cosy.core.tree import Path, Tree
    from cosy.search.queries import ResolutionQuery

__all__ = ["Recombination", "SubtreeGraft", "SubtreeSwap"]


def _inner_positions(tree: Tree[T]) -> list[Path]:
    """Return the positions of a term that are neither its root nor one of its leaves.

    Args:
        tree (Tree[T]): The term.

    Returns:
        list[Path]: The inner positions, sorted so that the caller's shuffle is the only source
            of order.
    """
    return sorted(tree.positions() - {()} - tree.leaf_positions())


@runtime_checkable
class Recombination(Protocol[NT, T, G]):
    """A binary variation operator: two individuals to a batch of offspring.

    The batch is a multiset of at most two individuals, all of them inhabitants; the empty batch
    means the move found nothing acceptable.
    """

    def recombine(self, query: ResolutionQuery[NT, T, G], first: Tree[T], second: Tree[T]) -> list[Tree[T]]:
        """Recombine two individuals.

        Args:
            query (ResolutionQuery[NT, T, G]): The generator query of the search space; the
                operator tests membership against the space and type it names.
            first (Tree[T]): The first parent.
            second (Tree[T]): The second parent.

        Returns:
            list[Tree[T]]: The batch, possibly empty.
        """
        ...


class _CheckedExchange:
    """The acceptance test and the pair enumeration shared by the two operators.

    Attributes:
        rng (random.Random): The source of randomness for the order of the pairs.
        max_size (int | None): An optional bound on the size of an offspring, tested *inside* the
            acceptance test.
    """

    def __init__(self, rng: random.Random, max_size: int | None = None) -> None:
        """Build the shared part.

        Args:
            rng (random.Random): The source of randomness for the order of the pairs.
            max_size (int | None): The maximum size of an offspring, or None for no bound.
                (Default value = None)

        Raises:
            ValueError: If ``max_size`` is negative.
        """
        if max_size is not None and max_size < 0:
            msg = f"a size bound counts function symbols and cannot be negative: {max_size}"
            raise ValueError(msg)
        self.rng = rng
        self.max_size = max_size

    def _accepts(self, query: ResolutionQuery[NT, T, G], candidate: Tree[T]) -> bool:
        """Decide whether a candidate offspring is accepted.

        Two clauses, and both of them sit *in the acceptance test* rather than in a pre-filter over
        positions. A pre-filter on precomputed subtree heights decides a different question, namely
        what the exchange could reach rather than what this candidate is, and the size bound is
        moreover one route to keeping the individuals a run can hold to finitely many. A candidate beyond the
        bound is rejected exactly like one outside the language, so the next pair is tried.

        Args:
            query (ResolutionQuery[NT, T, G]): The query naming the space and the type.
            candidate (Tree[T]): The candidate offspring.

        Returns:
            bool: True if the candidate is an inhabitant within the bound.
        """
        if self.max_size is not None and term_size(candidate) > self.max_size:
            return False
        return checker(query.solution_space, query.start, candidate)

    def _pairs(self, first: Tree[T], second: Tree[T]) -> list[tuple[Path, Path]]:
        """Enumerate the position pairs in a uniformly drawn order.

        The permutation runs over the **pair set**. Shuffling the two position lists separately and
        taking their product is not the same distribution: it walks the first position of the first
        parent against every position of the second before it ever changes it, so the order of the
        pairs is strongly correlated and the first parent's choice dominates which exchange is
        tried.

        Args:
            first (Tree[T]): The first parent.
            second (Tree[T]): The second parent.

        The pair set is built and permuted in full before the first candidate is tested, so the
        cost is quadratic in the number of inner positions whether or not the first pair is
        accepted. A parent of a thousand nodes therefore costs a million pairs.

        Returns:
            list[tuple[Path, Path]]: The pairs of inner positions, uniformly permuted; empty if
                either parent has no inner position.
        """
        left = _inner_positions(first)
        right = _inner_positions(second)
        pairs = [(p, q) for p in left for q in right]
        self.rng.shuffle(pairs)
        return pairs


class SubtreeSwap(_CheckedExchange, Recombination[NT, T, G]):
    """Exchange the subterms at a pair of inner positions.

    Walk the pairs of inner positions in a uniformly drawn order, exchange the two subterms, and
    return both offspring as soon as a pair passes the acceptance test for **both** of them. If no
    pair does, the batch is empty.

    An exchange may make an offspring deeper than either parent, because the subterm that arrives
    can be deeper than the one that left. That is not an error, and it is the reason a run needs a
    route to keeping its individuals finite. ``max_size`` is one such route.

    What a position admits is its residual, not a symbol. Grammar-guided crossover matches
    nonterminals and gets closure from that. Here the residual is what decides, and it is decided by
    a query rather than by a label.

    Attributes:
        rng (random.Random): The source of randomness for the order of the pairs.
        max_size (int | None): An optional bound on the size of an offspring, tested inside the
            acceptance test.
    """

    def __init__(self, rng: random.Random, max_size: int | None = None) -> None:
        """Build the operator.

        Args:
            rng (random.Random): The source of randomness for the order of the pairs.
            max_size (int | None): The maximum size of an offspring, or None for no bound.
                (Default value = None)

        Raises:
            ValueError: If ``max_size`` is negative.
        """
        super().__init__(rng, max_size)

    def recombine(self, query: ResolutionQuery[NT, T, G], first: Tree[T], second: Tree[T]) -> list[Tree[T]]:
        """Swap subterms until both offspring are accepted.

        Args:
            query (ResolutionQuery[NT, T, G]): The generator query of the search space.
            first (Tree[T]): The first parent.
            second (Tree[T]): The second parent.

        Returns:
            list[Tree[T]]: Both offspring of the first acceptable pair, or the empty batch.
        """
        for left, right in self._pairs(first, second):
            first_child = first.replace_subtree_at(left, second.subtree_at(right))
            second_child = second.replace_subtree_at(right, first.subtree_at(left))
            if self._accepts(query, first_child) and self._accepts(query, second_child):
                return [first_child, second_child]
        return []


class SubtreeGraft(_CheckedExchange, Recombination[NT, T, G]):
    """Graft a subterm of the secondary parent into the primary one.

    The same enumeration as the swap, but the parents are not symmetric. ``first`` is the *primary*
    parent and ``second`` the *secondary* one, the only candidate of a pair is the primary with the
    subterm at its position replaced by the secondary's, and one acceptance test decides it. The
    batch therefore holds one offspring or none.

    A drop-in alternative for the swap in the same component slot.

    Residual-guided grafting, which would draw the replacement from the residual and restrict it to
    subterms of the secondary parent, is deliberately not here: several holes would have to be drawn
    together.

    Attributes:
        rng (random.Random): The source of randomness for the order of the pairs.
        max_size (int | None): An optional bound on the size of an offspring, tested inside the
            acceptance test.
    """

    def __init__(self, rng: random.Random, max_size: int | None = None) -> None:
        """Build the operator.

        Args:
            rng (random.Random): The source of randomness for the order of the pairs.
            max_size (int | None): The maximum size of an offspring, or None for no bound.
                (Default value = None)

        Raises:
            ValueError: If ``max_size`` is negative.
        """
        super().__init__(rng, max_size)

    def recombine(self, query: ResolutionQuery[NT, T, G], first: Tree[T], second: Tree[T]) -> list[Tree[T]]:
        """Graft subterms until one offspring is accepted.

        Args:
            query (ResolutionQuery[NT, T, G]): The generator query of the search space.
            first (Tree[T]): The primary parent, which the offspring is built from.
            second (Tree[T]): The secondary parent, which contributes the subterm.

        Returns:
            list[Tree[T]]: The one offspring of the first acceptable pair, or the empty batch.
        """
        for left, right in self._pairs(first, second):
            candidate = first.replace_subtree_at(left, second.subtree_at(right))
            if self._accepts(query, candidate):
                return [candidate]
        return []
