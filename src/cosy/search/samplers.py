"""Samplers: a resolution query mapped to a stream of completions.

A sampler asks for very little. It maps a resolution query to a stream of inhabitants completing
its query term, and it may make random choices along the way. The whole weight of the definition
sits in one word: *bounded*. A bounded sampler answers every request for a next element, with or
without an inhabitant. Whether a tree language is empty is undecidable, so no component may ever
ask. Within a bound the question changes, and what the caller sees is a stream that ends. Every
clause that reacts to "the stream gave nothing" therefore reacts to a *halting request*, never to
an emptiness test, which in Python terms means a `StopIteration` and never an `is_empty()` oracle.

Four of them, and the difference is what they promise:

* :class:`DepthBoundedRandomSampler` draws independently: one draw runs a depth-first search whose
  clause order is uniformly random, and takes the first inhabitant it yields. It promises
  *positivity*, which is that every completion within the bound can come out, and nothing about
  the distribution. This module claims no more than that.
* :class:`SizeUniformSampler` is the size-uniform stream of random search. Under unambiguity
  within the bound its prefixes are samples without replacement, so it repeats nothing.
* :class:`CostTableSampler` is random search under the cost of an additive cost algebra and a
  distribution on its values that the caller prescribes, counted by the cost table
  (:mod:`cosy.search.cost_tables`), with a cap on the cost in the place of the size bound. It
  promises what the size-uniform sampler promises, for that distribution.
* :class:`TiltSampler` is random search in proportion to ``e^(-theta c(t))`` for the cost of an
  additive cost algebra (:mod:`cosy.search.tilt`): the terms of one cost value alike, the cost values
  in proportion to their counts, tilted, with one real number per non-terminal and no table of
  counts, over a program whose language is finite. Under unambiguity its prefixes are samples without
  replacement as well.

All four run on *any* resolution query of a program they apply to, generator or partial-term alike,
which is what lets a mutation operator take the same sampler parameter as an initialization. The
counting samplers apply where their counts do: the cost table's sampler, like the size table's form,
refuses a program in which a predicate reads a hole, and it refuses one in which clauses of cost
zero pump without paying; the tilt's refuses a predicate that reads a hole too, and a program whose
language is infinite. For the size-uniform sampler that follows from enumerating. For the
depth-bounded one it is a property of the engine, and one that
had to be repaired: its randomness is the clause order and nothing else, so a query whose initial
goals the clause order does not reach is a query it answers with a constant.
``SolutionSpace.goal_from_tree`` takes the order for the walk that derives them, and
``tests/test_samplers.py`` pins positivity on residual queries as it does on generator ones.

The depth bound is a bound on the depth of the **partial inhabitant** (``term_depth``), and it goes
in through ``goal_filter`` rather than through the engine's ``max_depth``. The two measure
different things, since ``max_depth`` bounds the length of a goal's *open* positions and a subtree
leaves that measurement as soon as it grounds, and on the reference spaces they nevertheless agree
on the finished terms, because the computation rule expands the deepest open subgoal first and so
measures every position before it grounds. That agreement is a property of the rule. The bound the
sampler states is not, so the filter states it where it holds regardless.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from cosy.search.cost_tables import WeightedCostTable, _whole_cap, weighted_cost_table
from cosy.search.partial import term_size
from cosy.search.rules import depth_first, uniform_random_clause_order
from cosy.search.sampling import (
    WeightedTable,
    WeightedTree,
    weighted_table,
    weighted_tree,
)
from cosy.search.tilt import TiltedSearch, _real_theta, tilted_search

if TYPE_CHECKING:
    import random
    from collections.abc import Callable, Iterator

    from cosy.core.solution_space import Goal
    from cosy.core.tree import Tree
    from cosy.search.costs import AdditiveCostAlgebra
    from cosy.search.queries import ResolutionQuery

__all__ = ["CostTableSampler", "DepthBoundedRandomSampler", "Sampler", "SizeUniformSampler", "TiltSampler"]


def _uniform(_value: Any) -> float:
    """Give every realized size the same probability, which is the size-uniform distribution.

    Args:
        _value (Any): The size. Ignored.

    Returns:
        float: One. The construction normalizes over the realized values.
    """
    return 1.0


@runtime_checkable
class Sampler(Protocol):
    """A map from a resolution query to a stream of completions of its query term.

    The interface is deliberately narrow: one method to draw, and one question about the bound.
    A sampler requires the stream to be *bounded*, which is to say that every request for a next
    element halts, with or without an inhabitant. In Python that is simply the iterator either
    yielding or raising `StopIteration`, never hanging.
    """

    def sample(self, query: ResolutionQuery[Any, Any, Any]) -> Iterator[Tree[Any]]:
        """Stream completions of the query's term.

        Args:
            query (ResolutionQuery[Any, Any, Any]): The query to complete.

        Returns:
            Iterator[Tree[Any]]: The stream. It ends rather than blocking when nothing is left.
        """
        ...

    def at_least(self, query: ResolutionQuery[Any, Any, Any], count: int) -> bool:
        """Decide whether at least ``count`` completions lie within the bound.

        The question an initialization asks before it starts drawing, so that a population it
        cannot fill fails at once rather than after a stream runs dry. It is decidable *within*
        the bound, which is the whole reason the bound is there.

        Args:
            query (ResolutionQuery[Any, Any, Any]): The query to complete.
            count (int): The number of completions asked for.

        Returns:
            bool: True if the bound admits at least ``count`` distinct completions.
        """
        ...


class DepthBoundedRandomSampler:
    """Independent draws from a depth-first search with a uniformly random clause order.

    One draw is one run of the search rule. The clause order draws a fresh uniform permutation per
    expansion, which is where all of the randomness sits, and the frontier stays a plain stack. The
    draw is the first inhabitant the run yields, and the stream ends at the first draw that yields
    none.

    **Why it halts.** Finitely many terms have depth at most ``depth_bound``, and the filter drops
    every child that exceeds it, so the search tree of one draw is finite and the draw halts. That
    holds on a recursive space, with no depth bound of the engine's own, and on a space whose
    language is empty.

    **What it does not promise.** Not a distribution. The clause order is uniform per expansion,
    which is not the same as a uniform inhabitant, and the search reaches short terms through more
    orders than long ones. Positivity is all there is, and it is exactly what makes this sampler
    the contrast to size-uniform sampling.

    Attributes:
        depth_bound (int): The bound on the depth of a completion.
        rng (random.Random): The source of randomness. Two samplers with equally seeded generators
            draw identically.
    """

    def __init__(self, depth_bound: int, rng: random.Random) -> None:
        """Build the sampler.

        Args:
            depth_bound (int): The bound on the depth of a completion, at least 0.
            rng (random.Random): The source of randomness.

        Raises:
            ValueError: If the bound is negative, since there is no such thing as a term of
                negative depth and a caller passing one has computed it wrongly.
        """
        if depth_bound < 0:
            msg = f"a depth bound is a length and cannot be negative: {depth_bound}"
            raise ValueError(msg)
        self.depth_bound = depth_bound
        self.rng = rng

    def _within_bound(self, goal: Goal[Any, Any, Any]) -> bool:
        """Decide whether a goal's partial inhabitant still fits the depth bound.

        Reads the depth off the goal's positions rather than materializing the partial inhabitant:
        the filter runs once per expanded child, and rebuilding the term each time makes it
        quadratic in the term. A position ``p`` sits at depth ``len(p)``, and if a subtree is
        grounded there it reaches ``len(p) + subtree.depth``, a depth ``Tree`` carries with it.
        Holes and expanded spine positions contribute their own depth alone, which is what
        ``term_depth(partial_inhabitant(goal))`` computes for them too.

        Args:
            goal (Goal[Any, Any, Any]): The search node.

        Returns:
            bool: True if its partial inhabitant is no deeper than the bound.
        """
        for position, (_, subtree) in goal.grounded.items():
            if len(position) + subtree.depth > self.depth_bound:
                return False
        return all(len(position) <= self.depth_bound for position in goal.subgoals)

    def _draw(self, query: ResolutionQuery[Any, Any, Any]) -> Tree[Any] | None:
        """Run one search and return its first inhabitant.

        Args:
            query (ResolutionQuery[Any, Any, Any]): The query to complete.

        Returns:
            Tree[Any] | None: The inhabitant, or None if this run yielded none.
        """
        stream = depth_first(
            query,
            max_count=1,
            clause_order=uniform_random_clause_order(self.rng),
            goal_filter=self._within_bound,
        )
        return next(iter(stream), None)

    def sample(self, query: ResolutionQuery[Any, Any, Any]) -> Iterator[Tree[Any]]:
        """Stream independent draws until one comes up empty.

        Args:
            query (ResolutionQuery[Any, Any, Any]): The query to complete.

        Yields:
            Tree[Any]: One completion per draw. Draws are independent, so the same completion may
                appear more than once, since this sampler draws *with* replacement.
        """
        while True:
            drawn = self._draw(query)
            if drawn is None:
                return
            yield drawn

    def at_least(self, query: ResolutionQuery[Any, Any, Any], count: int) -> bool:
        """Decide whether at least ``count`` completions lie within the depth bound.

        Enumerates the depth-bounded search tree, which is finite for the reason given on the
        class, and stops as soon as the count is reached, so the cost is bounded by the answer
        rather than by the size of the language.

        Args:
            query (ResolutionQuery[Any, Any, Any]): The query to complete.
            count (int): The number of completions asked for.

        Returns:
            bool: True if the bound admits at least ``count`` distinct completions.
        """
        if count <= 0:
            return True
        found = depth_first(query, max_count=count, goal_filter=self._within_bound)
        return sum(1 for _ in found) >= count


class SizeUniformSampler:
    """The size-uniform stream of random search, as a sampler.

    The cost is the term size and the distribution is uniform over the sizes the query realizes
    within the bound, so every realized size carries the same total weight and the completions of
    one size are equally likely.

    Under unambiguity within the bound the stream has the property the depth-bounded sampler lacks:
    it is a sample **without replacement**, so it repeats nothing, and every prefix of it is a
    sample from the intended distribution. Beyond that hypothesis the branch counts count
    derivations rather than terms and the stream repeats an inhabitant once per derivation, which
    is documented in :mod:`cosy.search.sampling` and not repaired here.

    **Which construction, and why it is asked rather than guessed.** ``counting="tree"``
    materializes the retained derivation tree, and ``counting="table"`` computes the same counts
    from ``N_A(s)`` over the program (:func:`cosy.search.counting.size_table`) and never builds
    the tree. Where the table applies the two produce the same stream from the same seed, term for
    term and key for key, but it applies only under a hypothesis on the program (no predicate
    reads a hole), and where that fails it raises. Choosing it for a caller who did not ask would
    turn a program the tree form handles into an error, so the choice stays with the caller.

    **One construction per pair of questions.** An initialization asks ``at_least`` and then
    draws, and counting is the expensive half of both. The sampler therefore keeps the *last*
    construction it built and answers both from it. Two consequences worth knowing: a caller who
    alternates between two queries gets no reuse, and under ``counting="tree"`` the retained tree
    stays alive until another query displaces it, and on a large space that is real memory, which is
    the price of not building it twice. :meth:`forget` gives it back.

    Attributes:
        size_bound (int): The bound ``D`` on ``term_size`` of a completion.
        rng (random.Random): The source of randomness.
        counting (str): ``"tree"`` or ``"table"``.
    """

    def __init__(
        self,
        size_bound: int,
        rng: random.Random,
        *,
        counting: Literal["tree", "table"] = "tree",
    ) -> None:
        """Build the sampler.

        Args:
            size_bound (int): The bound ``D`` on the size of a completion, at least 0.
            rng (random.Random): The source of randomness.
            counting (str): Which construction computes the branch counts, ``"tree"`` or
                ``"table"``. (Default value = "tree")

        Raises:
            ValueError: If the bound is negative, or if ``counting`` is neither of the two names.
        """
        if size_bound < 0:
            msg = f"the size bound counts function symbols and cannot be negative: {size_bound}"
            raise ValueError(msg)
        if counting not in ("tree", "table"):
            msg = f"counting selects the construction of the branch counts and is 'tree' or 'table', not {counting!r}"
            raise ValueError(msg)
        self.size_bound = size_bound
        self.rng = rng
        self.counting = counting
        self._query: ResolutionQuery[Any, Any, Any] | None = None
        self._weighted: WeightedTable[Any, Any, Any] | WeightedTree[Any, Any, Any] | None = None

    def forget(self) -> None:
        """Drop the cached construction.

        The retained tree of a large space is the biggest object this package produces, and a
        caller who is done drawing has no other way to say so.

        Returns:
            None
        """
        self._query = None
        self._weighted = None

    def _construction(
        self, query: ResolutionQuery[Any, Any, Any]
    ) -> WeightedTable[Any, Any, Any] | WeightedTree[Any, Any, Any]:
        """Return the weighted construction for a query, building it at most once in a row.

        Keyed by *identity* rather than by equality: a partial-term query carries a term, and
        comparing terms structurally on every call would cost more than the lookup saves. The
        callers that benefit, an initialization asking and then drawing, pass one and the same
        query object.

        Args:
            query (ResolutionQuery[Any, Any, Any]): The query to complete.

        Returns:
            WeightedTable | WeightedTree: The construction, ready to stream from.
        """
        if self._weighted is None or self._query is not query:
            self._weighted = (
                weighted_table(query, self.size_bound, _uniform)
                if self.counting == "table"
                else weighted_tree(query, self.size_bound, term_size, _uniform)
            )
            self._query = query
        return self._weighted

    def sample(self, query: ResolutionQuery[Any, Any, Any]) -> Iterator[Tree[Any]]:
        """Stream the completions in size-uniform order, without replacement.

        Args:
            query (ResolutionQuery[Any, Any, Any]): The query to complete.

        Yields:
            Tree[Any]: The completions within the bound, each exactly once under unambiguity.
        """
        yield from self._construction(query).stream(self.rng)

    def at_least(self, query: ResolutionQuery[Any, Any, Any], count: int) -> bool:
        """Decide whether at least ``count`` completions lie within the size bound.

        Exact, and it costs nothing beyond the counting the draw needs anyway: the branch counts
        hold the number of success branches within the bound, their sum answers the question
        outright, and the construction they come from is the one :meth:`sample` will use. Under
        ambiguity that sum counts derivations, so it can only overstate, which is the direction
        that keeps the caller from failing on a population it could have filled.

        Args:
            query (ResolutionQuery[Any, Any, Any]): The query to complete.
            count (int): The number of completions asked for.

        Returns:
            bool: True if the bound admits at least ``count`` completions.
        """
        if count <= 0:
            return True
        weighted = self._construction(query)
        total = weighted.total if isinstance(weighted, WeightedTable) else weighted.root.total
        return total >= count


class CostTableSampler:
    """Random search under an additive cost and a prescribed distribution on its values, as a sampler.

    The sampler of the cost table (:mod:`cosy.search.cost_tables`): the cost is the fold of an
    additive cost algebra with whole-number costs, the distribution ``pi`` on its values is the
    caller's, and a term's weight is ``pi(c(t)) / N_r(c(t))`` over the terms of cost at most the
    cap. Every realized cost value up to the cap then carries the probability ``pi`` gives it,
    normalized over those values, and the terms of one value are equally likely. Under unambiguity within the cap the stream is a sample
    without replacement, as the size-uniform one is.

    Like :class:`SizeUniformSampler` it keeps the last construction it built, keyed by the identity
    of the query, so that an initialization asking :meth:`at_least` and then drawing pays for the
    table once. :meth:`forget` gives it back.

    Attributes:
        algebra (AdditiveCostAlgebra): The additive cost algebra whose fold is the cost.
        distribution (Callable[[Any], float]): ``pi`` on the cost values; it must be positive on
            every value the query realizes below the cap.
        cost_cap (int): The largest cost a drawn term may have.
        rng (random.Random): The source of randomness.
    """

    def __init__(
        self,
        algebra: AdditiveCostAlgebra[Any],
        distribution: Callable[[Any], float],
        cost_cap: int,
        rng: random.Random,
    ) -> None:
        """Build the sampler.

        Args:
            algebra (AdditiveCostAlgebra[Any]): The additive cost algebra.
            distribution (Callable[[Any], float]): ``pi`` on the cost values.
            cost_cap (int): The largest cost a drawn term may have, a nonnegative whole number; a
                whole float such as a fold of the algebra is read as the number it is.
            rng (random.Random): The source of randomness.

        Raises:
            ValueError: If the cap is not a nonnegative whole number.
        """
        self.algebra = algebra
        self.distribution = distribution
        self.cost_cap = _whole_cap(cost_cap)
        self.rng = rng
        self._query: ResolutionQuery[Any, Any, Any] | None = None
        self._weighted: WeightedCostTable[Any, Any, Any] | None = None

    def forget(self) -> None:
        """Drop the cached construction.

        Returns:
            None
        """
        self._query = None
        self._weighted = None

    def _construction(self, query: ResolutionQuery[Any, Any, Any]) -> WeightedCostTable[Any, Any, Any]:
        """Return the weighted construction for a query, building it at most once in a row.

        Args:
            query (ResolutionQuery[Any, Any, Any]): The query to complete.

        Returns:
            WeightedCostTable: The construction, ready to stream from.
        """
        if self._weighted is None or self._query is not query:
            self._weighted = weighted_cost_table(query, self.algebra, self.distribution, self.cost_cap)
            self._query = query
        return self._weighted

    def sample(self, query: ResolutionQuery[Any, Any, Any]) -> Iterator[Tree[Any]]:
        """Stream the completions of cost at most the cap, without replacement.

        Args:
            query (ResolutionQuery[Any, Any, Any]): The query to complete.

        Yields:
            Tree[Any]: The completions, each exactly once under unambiguity within the cap.

        Raises:
            ValueError: Where the cost table refuses the query's program -- a predicate that reads a
                hole, clauses of cost zero that pump without paying, a cost that is not a whole
                number -- or ``pi`` is not positive on a realized cost value up to the cap.
        """
        yield from self._construction(query).stream(self.rng)

    def at_least(self, query: ResolutionQuery[Any, Any, Any], count: int) -> bool:
        """Decide whether at least ``count`` completions cost at most the cap.

        Exact, from the counts the draw needs anyway. Under ambiguity the counts are of
        derivations, so they can only overstate.

        Args:
            query (ResolutionQuery[Any, Any, Any]): The query to complete.
            count (int): The number of completions asked for.

        Returns:
            bool: True if at least ``count`` completions cost at most the cap.
        """
        if count <= 0:
            return True
        return self._construction(query).total >= count


class TiltSampler:
    """Random search in proportion to ``e^(-theta c(t))`` for an additive cost, as a sampler.

    The sampler of the exponential tilt (:mod:`cosy.search.tilt`): a term weighs ``e^(-theta c(t))``,
    so the terms of one cost value are equally likely and the cost values are drawn in proportion to
    their counts, tilted. It needs one real number per non-terminal and no table of counts, and it
    covers finite languages. Under unambiguity the stream is a sample without replacement.

    Like :class:`CostTableSampler` it keeps the last construction it built, keyed by the identity of
    the query, and :meth:`forget` gives it back.

    Attributes:
        algebra (AdditiveCostAlgebra): The additive cost algebra whose fold is the cost.
        theta (float): The tilt.
        rng (random.Random): The source of randomness.
    """

    def __init__(self, algebra: AdditiveCostAlgebra[Any], theta: float, rng: random.Random) -> None:
        """Build the sampler.

        Args:
            algebra (AdditiveCostAlgebra[Any]): The additive cost algebra, with finite real costs.
            theta (float): The tilt, any finite real number.
            rng (random.Random): The source of randomness.

        Raises:
            ValueError: If ``theta`` is not a finite real number.
        """
        self.algebra = algebra
        self.theta = _real_theta(theta)
        self.rng = rng
        self._query: ResolutionQuery[Any, Any, Any] | None = None
        self._search: TiltedSearch[Any, Any, Any] | None = None

    def forget(self) -> None:
        """Drop the cached construction.

        Returns:
            None
        """
        self._query = None
        self._search = None

    def _construction(self, query: ResolutionQuery[Any, Any, Any]) -> TiltedSearch[Any, Any, Any]:
        """Return the tilted search for a query, building it at most once in a row.

        Args:
            query (ResolutionQuery[Any, Any, Any]): The query to complete.

        Returns:
            TiltedSearch: The construction, ready to stream from.
        """
        if self._search is None or self._query is not query:
            self._search = tilted_search(query, self.algebra, self.theta)
            self._query = query
        return self._search

    def sample(self, query: ResolutionQuery[Any, Any, Any]) -> Iterator[Tree[Any]]:
        """Stream the completions in proportion to their tilted weights, without replacement.

        Args:
            query (ResolutionQuery[Any, Any, Any]): The query to complete.

        Yields:
            Tree[Any]: The completions, each exactly once under unambiguity.

        Raises:
            ValueError: Where the tilt refuses the query's program -- an infinite language, a predicate
                that reads a hole, a cost that is not a finite real number.
        """
        yield from self._construction(query).stream(self.rng)

    def at_least(self, query: ResolutionQuery[Any, Any, Any], count: int) -> bool:
        """Decide whether the query has at least ``count`` completions.

        Exact, from the number of terms the tilt table keeps in whole numbers. Under ambiguity the
        numbers are of derivations, so they can only overstate.

        Args:
            query (ResolutionQuery[Any, Any, Any]): The query to complete.
            count (int): The number of completions asked for.

        Returns:
            bool: True if the query has at least ``count`` completions.
        """
        if count <= 0:
            return True
        return self._construction(query).total >= count
