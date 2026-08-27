"""Random search: best-first search under a randomizing cost function.

The module is named ``sampling`` rather than ``random_search`` because the package re-exports the
function :func:`random_search` under that name. A module and a function of one name collide on the
package attribute, and ``import cosy.search.random_search`` would have bound the *function*.

Uninformed search fixes the order of a stream by the shape of the derivation tree and informed
search by a cost. Random search draws a random cost instead. It serves the evolutionary and
Bayesian methods twice over, as the baseline they are measured against and as the initialization
they start from. Both roles need more than an arbitrary order, and the construction meets two
requirements: the stream follows a chosen distribution, which is to say that every prefix of it is
a sample from that distribution, and it draws without replacement, so the streamed inhabitants are
distinct.

The construction runs in three steps. The branch counts (:mod:`cosy.search.counting`) count, per
cost value, the inhabitants that realize it. The counts spread the chosen distribution ``pi`` into
a weight per inhabitant, ``w(t) = pi(c(t)) / N_r(c(t))``: a cost value carries the probability
``pi`` gives it, and within the value the inhabitants are uniform. A Gumbel key per inhabitant
finally realizes the draw without replacement, since the greatest key falls on ``t`` with
probability ``w(t)/W``, and listing by decreasing key *is* a sample without replacement. Extended
to search nodes, the negated keys are the randomizing cost function that random search is
best-first search on.

**No key per inhabitant.** The greatest key below a node ``n`` is itself Gumbel, with location
``log sum_a B_n(a) pi(a) / N_r(a)``, a quantity the branch counts already hold. So the search draws
one key per *node*, top-down, conditioning the children of an expansion on the key their parent
carries (:func:`cosy.search.gumbel.condition_on_maximum`). The construction follows Maddison et al.
and Kool et al.

**What is guaranteed, and when.** Under unambiguity within the bound, which is to say that every
inhabitant ends exactly one success branch, both requirements hold: each inhabitant of size at most
``D`` is streamed exactly once, and every prefix is a sample without replacement in proportion to
``w``. Without that hypothesis the branch counts count derivations rather than terms, and the
stream repeats an inhabitant once per derivation, drawing it in proportion to its derivation count,
which is the behavior the counting samplers of the literature show on an ambiguous grammar. This
module does *not* deduplicate the stream to hide that:
:func:`cosy.search.counting.assert_unambiguous_within` decides the hypothesis, and a caller who
needs the guarantee checks it.

**Best-first, specialized.** The keys are reals, so the frontier here is a binary heap over a total
order rather than the partial-order frontier best-first search is defined with. The general
frontier arrives with the cost layer. This one is its total-order fast path, and the two must agree
where they overlap.

**Two ways to the weights, one search.** Read the construction again and notice what the frontier
actually asks of a node: whether it is a success node, its children in clause order, and one number
``log w`` per child. Nothing else. ``condition_on_maximum`` is entirely indifferent to where that
number comes from. So the part of random search that is *not* lazy is not the Gumbel machinery, it
is the oracle

    log w(n) = log sum_a B_n(a) pi(a) / N_r(a),

which :class:`WeightedTree` answers by having materialized the whole retained tree first.
:class:`WeightedTable` answers the same question from ``N_A(s)``
(:func:`cosy.search.counting.size_table`) instead: a node is its partial inhabitant plus a multiset
of open holes, and under the decomposition hypothesis its branch counts are the convolution of the
holes' table rows. The oracle becomes a lookup, the tree is never built, and nodes are expanded
only as the frontier reaches them.

Both drive :func:`keyed_stream`, and on a program that satisfies the hypothesis they produce the
*same stream* from the same seed, term for term and key for key. That is the point: what changes is
the way ``B_n`` is computed, not what ``B_n`` is. The weight per inhabitant, the randomizing cost,
the Gumbel top-k identity and the prefix-sampling guarantee all consume the counts as numbers and
are untouched.

Which one a caller gets is never decided silently. The table form applies only where
:func:`cosy.search.counting.decomposable_or_raise` holds, and where it does not it raises rather
than returning weights that are quietly wrong.

**Logarithms, throughout.** The key asks for ``log w`` and never for ``w``, which is lucky, because
``w`` is not representable at the bounds this construction exists to reach: on the list space at
``D = 500`` a node weight is a ratio of numbers with hundreds of digits, and ``count * unit_weight``
overflows in one direction while underflowing in the other. So the unit weights are kept in log
space and the node weight is a log-sum-exp over them.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

from cosy.core.solution_space import NT, G, Goal, T
from cosy.search.counting import (
    CountedNode,
    branch_counts,
    child_nodes,
    decomposable_or_raise,
    initial_nodes,
    size_table,
)
from cosy.search.gumbel import condition_on_maximum, gumbel_key
from cosy.search.partial import holes, term_size
from cosy.search.rules import deepest_first_subgoal

if TYPE_CHECKING:
    import random
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from cosy.core.solution_space import NonTerminalArgument
    from cosy.core.tree import Path, Tree
    from cosy.search.counting import SizeTable
    from cosy.search.queries import ResolutionQuery

__all__ = [
    "WeightedTable",
    "WeightedTree",
    "keyed_stream",
    "log_sum_exp",
    "random_search",
    "random_search_keyed",
    "size_uniform",
    "weighted_table",
    "weighted_tree",
]

N = Any
"""A node of whatever construction drives :func:`keyed_stream`. It is only ever passed back."""

# An integer of at most this many bits is below 2**1023 and so always converts to a double. From
# 1024 bits on it depends on the value: `float(3**646)` still works, but `float(2**1024 - 1)`
# rounds up past the largest double and raises, and above 1025 bits it never does. The switch is
# drawn where conversion is *guaranteed*, not where it happens to still work.
_ALWAYS_A_DOUBLE = 1023


def log_sum_exp(terms: Sequence[float]) -> float:
    """Return ``log sum_i exp(terms[i])`` without leaving log space.

    Args:
        terms (Sequence[float]): The summands, as logarithms.

    Returns:
        float: The logarithm of the sum. It is ``-inf`` on an empty sequence, and ``-inf`` when
            every summand is ``-inf``, which is a node with no completion within the bound.
    """
    if not terms:
        return -math.inf
    top = max(terms)
    if top == -math.inf:
        return -math.inf
    return top + math.log(math.fsum(math.exp(term - top) for term in terms))


def keyed_stream(
    root: N,
    root_log_weight: float,
    expand: Callable[[N], tuple[Tree[T] | None, Sequence[tuple[N, float]]]],
    rng: random.Random,
) -> Iterator[tuple[float, Tree[T]]]:
    """Run random search over any construction that can weigh a node and name its children.

    This *is* random search: best-first search under the randomizing cost function, which is the
    negated Gumbel key. The frontier pops the node of greatest key. A success node yields its
    inhabitant, an inner node draws keys for its children conditioned on its own, following
    Maddison et al. and Kool et al., and the maximum over a subtree is again Gumbel with location
    ``log w``, which is the only thing ``expand`` has to supply.

    Factored out of :class:`WeightedTree` so that the eager and the table construction run the
    *same* search rather than two copies of it. Two copies would be two chances for the streams
    to drift apart, and their agreement is the evidence that only the computation of ``B_n``
    changed.

    Args:
        root (N): The root node, passed back to ``expand`` unread.
        root_log_weight (float): ``log w`` at the root, the total weight ``W``, so 0 up to
            floating-point error when the distribution is normalized.
        expand (Callable): Maps a node to its inhabitant (None on an inner node) and to its
            retained children with their ``log w``, in clause order. Children of vanishing
            weight must already be dropped, since they carry no key.
        rng (random.Random): The source of randomness.

    Yields:
        tuple[float, Tree[T]]: The key and the inhabitant, in decreasing key order.
    """
    if root_log_weight == -math.inf:
        return
    # The frontier orders by the randomizing cost function, the *negated* key: best-first pops a
    # node of least cost, which is the node of greatest key. The counter breaks ties in the heap
    # without ever comparing nodes, which do not order.
    tie_break = 0
    root_key = gumbel_key(root_log_weight, rng)
    frontier: list[tuple[float, int, float, N]] = [(-root_key, tie_break, root_key, root)]
    while frontier:
        _, _, key, node = heapq.heappop(frontier)
        inhabitant, children = expand(node)
        if inhabitant is not None:
            yield key, inhabitant
            continue
        if not children:
            continue
        child_keys = condition_on_maximum(key, [log_weight for _, log_weight in children], rng)
        for (child, _), child_key in zip(children, child_keys, strict=True):
            tie_break += 1
            heapq.heappush(frontier, (-child_key, tie_break, child_key, child))


@dataclass(frozen=True)
class WeightedTree(Generic[NT, T, G]):
    """A retained derivation tree together with the weight the distribution puts on its inhabitants.

    Counting is the expensive half of random search and does not depend on the randomness, so a
    caller drawing repeatedly from one space builds this once and calls :meth:`stream` per draw.

    Attributes:
        root (CountedNode[NT, T, G]): The root of the retained tree, carrying ``B_r = N_r``.
        unit_weights (Mapping[Any, float]): ``pi(a) / N_r(a)`` per realized cost value, the
            weight of a *single* inhabitant of that cost. ``pi`` is already restricted to the
            realized values and normalized over them.
        log_unit_weights (Mapping[Any, float]): The same numbers in log space, computed from the
            exact integer counts rather than from :attr:`unit_weights`. At large bounds the
            float image of ``pi(a) / N_r(a)`` underflows to zero while its logarithm stays
            perfectly ordinary, and a Gumbel key only ever asks for the logarithm.
    """

    root: CountedNode[NT, T, G]
    unit_weights: Mapping[Any, float]
    log_unit_weights: Mapping[Any, float]

    def weight_of(self, node: CountedNode[NT, T, G]) -> float:
        """Return the total weight of the inhabitants below a node.

        Args:
            node (CountedNode[NT, T, G]): A node of the retained tree.

        Returns:
            float: ``sum_a B_n(a) pi(a) / N_r(a)``. For the root this is the total weight ``W``,
                which is 1 up to floating-point error. Reading it as a plain float is safe only
                at bounds where the weights are representable; :meth:`log_weight_of` is what the
                search itself uses.
        """
        return sum(count * self.unit_weights[value] for value, count in node.counts.items())

    def log_weight_of(self, node: CountedNode[NT, T, G]) -> float:
        """Return the location of the Gumbel distribution a node's key follows.

        Computed as a log-sum-exp over ``log B_n(a) + log pi(a) - log N_r(a)`` rather than as the
        logarithm of :meth:`weight_of`. The difference is not cosmetic: taking the float sum
        first overflows and underflows at exactly the bounds the table form makes reachable, and
        an underflow to zero here would drop part of the space from the sample without a trace.

        Args:
            node (CountedNode[NT, T, G]): A node of the retained tree.

        Returns:
            float: ``log sum_a B_n(a) pi(a) / N_r(a)``.

        Raises:
            ValueError: If the node has no realized cost value in the distribution's support.
                Every retained node has a positive weight in exact arithmetic, so this means the
                counts and the weights have come apart, and continuing would sample from
                something other than ``w``.
        """
        terms = [
            math.log(count) + self.log_unit_weights[value]
            for value, count in node.counts.items()
            if value in self.log_unit_weights
        ]
        total = log_sum_exp(terms)
        if total == -math.inf:
            msg = (
                "the weight of a retained node vanished; every retained node has a positive "
                "weight in exact arithmetic, so the counts and the distribution have come apart"
            )
            raise ValueError(msg)
        return total

    def stream(self, rng: random.Random) -> Iterator[Tree[T]]:
        """Draw one stream from this weighted tree.

        Args:
            rng (random.Random): The source of randomness.

        Yields:
            Tree[T]: The inhabitants, in decreasing key order.
        """
        for _, inhabitant in self.keyed_stream(rng):
            yield inhabitant

    def keyed_stream(self, rng: random.Random) -> Iterator[tuple[float, Tree[T]]]:
        """Draw one stream, keeping the key each inhabitant was streamed under.

        The keys are what carries the guarantee, which is stated about the decreasing order of
        keys, so exposing them lets a caller or a validation test read that order instead of
        inferring it.

        Args:
            rng (random.Random): The source of randomness.

        Yields:
            tuple[float, Tree[T]]: The key and the inhabitant, in decreasing key order.
        """
        if not self.root.counts:
            return

        def expand(
            node: CountedNode[NT, T, G],
        ) -> tuple[Tree[T] | None, Sequence[tuple[CountedNode[NT, T, G], float]]]:
            """Read a node of the retained tree the way :func:`keyed_stream` needs it.

            Args:
                node (CountedNode[NT, T, G]): The node to expand.

            Returns:
                tuple: Its inhabitant (None on an inner node) and its children with their
                    ``log w``, in clause order. Only nodes of nonvanishing count are retained,
                    so every child here already carries a positive weight.
            """
            if node.inhabitant is not None:
                return node.inhabitant, ()
            return None, [(child, self.log_weight_of(child)) for child in node.children]

        yield from keyed_stream(self.root, self.log_weight_of(self.root), expand, rng)


def weighted_tree(
    query: ResolutionQuery[NT, T, G],
    size_bound: int,
    cost: Callable[[Tree[T]], Any],
    distribution: Callable[[Any], float],
    *,
    subgoal_selection: Callable[[Goal[NT, T, G]], tuple[Path, NonTerminalArgument[NT]]] | None = None,
) -> WeightedTree[NT, T, G]:
    """Count a query's branches and spread a distribution over the counts.

    Args:
        query (ResolutionQuery[NT, T, G]): The query to sample from.
        size_bound (int): The bound ``D`` on the term size. Uniform drawing exists only on a
            finite set, and this is what makes each cost value carry finitely many inhabitants.
        cost (Callable[[Tree[T]], Any]): The cost function ``c`` on ground terms. Any computable
            function will do, since random search needs no monotonicity.
        distribution (Callable[[Any], float]): ``pi``, evaluated on each cost value the query
            realizes within the bound. It need not be normalized, since only the ratios matter
            and the values are normalized over the realized ones here, but it must be positive on
            every one of them.
        subgoal_selection (Callable | None): The computation rule ``order_S``, which decides which
            hole is expanded next and hence the shape of the derivation tree. It changes the
            order of the stream and not the terms in it. The table form takes the same parameter,
            and the two must be comparable under a rule other than the default. Otherwise their
            agreement could be an artifact of one shared traversal rather than of the counts. None selects
            the engine's deepest-first rule. (Default value = None)

    Returns:
        WeightedTree[NT, T, G]: The counted tree and the weight per realized cost value.

    Raises:
        ValueError: If ``distribution`` is not positive on some realized cost value. Dropping
            such a value would silently omit realizable inhabitants from every sample.
    """
    root = branch_counts(query, size_bound, cost, subgoal_selection=subgoal_selection)
    unit_weights, log_unit_weights = _spread(distribution, root.counts)
    return WeightedTree(root=root, unit_weights=unit_weights, log_unit_weights=log_unit_weights)


def _spread(
    distribution: Callable[[Any], float], counts: Mapping[Any, int]
) -> tuple[dict[Any, float], dict[Any, float]]:
    """Spread a distribution over the realized cost values and return the unit weights.

    The weight of a single inhabitant in both representations at once: ``pi(a) / N_r(a)`` as a
    float, for callers that read weights, and its logarithm computed from the exact integer count,
    for the search. The two are computed from the same normalization, so they can only disagree by
    rounding, and where the float underflows, the logarithm still holds.

    Args:
        distribution (Callable[[Any], float]): ``pi``, evaluated on each realized cost value. It
            need not be normalized, since only the ratios matter.
        counts (Mapping[Any, int]): ``N_r``, the root's branch counts.

    Returns:
        tuple[dict[Any, float], dict[Any, float]]: The unit weights and their logarithms.

    Raises:
        ValueError: If ``distribution`` is not positive and finite on some realized cost value.
            Dropping such a value would silently omit realizable inhabitants from every sample.
    """
    if not counts:
        # A query with no inhabitant within the bound realizes no cost value, so there is nothing
        # to spread and nothing to normalize over. The caller sees an empty stream, which is the
        # honest answer and not an error, since emptiness within a bound is a legitimate outcome.
        return {}, {}
    probabilities: dict[Any, float] = {}
    for value in counts:
        probability = distribution(value)
        if probability <= 0.0 or not math.isfinite(probability):
            msg = (
                f"the distribution must be positive on every realized cost value, but gives {probability} at {value!r}"
            )
            raise ValueError(msg)
        probabilities[value] = probability
    total = math.fsum(probabilities.values())
    log_total = math.log(total)
    log_unit_weights = {
        value: math.log(probability) - log_total - math.log(counts[value])
        for value, probability in probabilities.items()
    }
    unit_weights = {
        value: (
            probability / total / counts[value]
            # Past the conversion limit the division would raise instead of rounding, at exactly
            # the bounds the table form exists to reach (lists at D = 648). The logarithm is
            # unaffected and is what the search uses. This is the reading copy, so it takes the
            # value the logarithm gives, which is the only one available there. That value can sit
            # one unit in the last place off the correctly rounded one, which is why the search
            # reads the logarithm and not this.
            if counts[value].bit_length() <= _ALWAYS_A_DOUBLE
            else math.exp(log_unit_weights[value])
        )
        for value, probability in probabilities.items()
    }
    return unit_weights, log_unit_weights


@dataclass(frozen=True)
class WeightedTable(Generic[NT, T, G]):
    """Random search weighted from the size table instead of from a materialized tree.

    Same distribution, same stream, different cost: the retained tree is never built, and a node
    is expanded only when the frontier reaches it. What replaces the traversal is the observation
    that a node's branch counts depend on the node only through the *size* of its partial
    inhabitant and the *multiset of its open holes*, so under the decomposition hypothesis

        B_n(s) = #{ways to split s - size(n) over the holes of n} = convolution of their rows,

    which is a table lookup. The weight per inhabitant and the randomizing cost are unchanged.
    Only the way ``B_n`` is computed is.

    The cost function is the term size, and the reason is not laziness. The size-uniform draw is
    the default here, and a general cost function would have to be a fold in a cost algebra with a
    finite carrier before the table could carry it as a second axis. A caller who needs another
    cost gets the tree form, which needs no such hypothesis.

    Attributes:
        query (ResolutionQuery[NT, T, G]): The query being sampled from.
        size_bound (int): The bound ``D``.
        table (SizeTable[NT]): ``N_A(s)`` over the program.
        root_counts (Mapping[int, int]): ``N_r(s)``, summed over the query's initial nodes. For
            a generator query the row of the start symbol, for a partial-term query the counts of
            the completions of the prescribed term.
        unit_weights (Mapping[int, float]): ``pi(s) / N_r(s)`` per realized size.
        log_unit_weights (Mapping[int, float]): The same in log space, and what the search uses.
        subgoal_selection (Callable | None): The computation rule; None selects the engine's
            deepest-first rule, which is what the tree form uses too.
    """

    query: ResolutionQuery[NT, T, G]
    size_bound: int
    table: SizeTable[NT]
    root_counts: Mapping[int, int]
    unit_weights: Mapping[int, float]
    log_unit_weights: Mapping[int, float]
    subgoal_selection: Callable[[Goal[NT, T, G]], tuple[Path, NonTerminalArgument[NT]]] | None = None

    @property
    def total(self) -> int:
        """Return the number of success branches within the bound.

        Returns:
            int: ``sum_s N_r(s)``, the same number as ``branch_counts(...).total``, computed
                without building the tree.
        """
        return sum(self.root_counts.values())

    def _rank(self) -> Mapping[NT, int]:
        """Return a deterministic order on the non-terminals, for canonicalising hole tuples.

        The convolution is commutative, so the order never changes a count. It only decides
        whether two nodes with the same multiset of holes share a cache entry. Sorting by the
        table's own insertion order is deterministic across runs and costs nothing, where sorting
        by ``repr`` would cost a rendering per lookup.

        Returns:
            Mapping[NT, int]: The rank of each non-terminal.
        """
        cached = self.__dict__.get("_rank_cache")
        if cached is None:
            cached = {nonterminal: index for index, nonterminal in enumerate(self.table.counts)}
            object.__setattr__(self, "_rank_cache", cached)
        return cached

    def holes_of(self, goal: Goal[NT, T, G]) -> tuple[NT, ...]:
        """Return the non-terminals of a goal's open holes, in a canonical order.

        ``Goal.subgoals`` is not the set of open holes, since an expanded spine position stays a
        subgoal until its whole subtree grounds, so this reads :func:`cosy.search.partial.holes`
        rather than the raw dict.

        Args:
            goal (Goal[NT, T, G]): The search node.

        Returns:
            tuple[NT, ...]: The hole types, sorted so that equal multisets give equal tuples.
        """
        rank = self._rank()
        return tuple(sorted(holes(goal).values(), key=lambda nt: (rank.get(nt, -1), id(nt))))

    def branch_counts_of(self, goal: Goal[NT, T, G] | None, size: int) -> dict[int, int]:
        """Return ``B_n`` of a node, from the table alone.

        Args:
            goal (Goal[NT, T, G] | None): The search node, or None for the query's root.
            size (int): The size of its partial inhabitant.

        Returns:
            dict[int, int]: The branch counts, one entry per realized size.
        """
        if goal is None:
            return dict(self.root_counts)
        if goal.success:
            return {size: 1} if size <= self.size_bound else {}
        hole_types = self.holes_of(goal)
        # The whole row at once: the loop below asks one hole tuple for every size the bound
        # admits, which is what `split_row` computes in a single walk.
        row = self.table.split_row(hole_types)
        counts: dict[int, int] = {}
        for total in range(size + len(hole_types), self.size_bound + 1):
            value = row[total - size]
            if value:
                counts[total] = value
        return counts

    def log_weight_of(self, goal: Goal[NT, T, G] | None, size: int) -> float:
        """Return the location of the Gumbel distribution a node's key follows.

        Memoised on ``(hole multiset, size)``, which is the whole content of the claim that the
        oracle is a lookup: two nodes agreeing in those two things have the same weight, whatever
        their partial inhabitants look like. On the list space at ``D = 100`` this turns 7 077
        oracle calls into 201 table queries.

        Args:
            goal (Goal[NT, T, G] | None): The search node, or None for the query's root.
            size (int): The size of its partial inhabitant.

        Returns:
            float: ``log sum_a B_n(a) pi(a) / N_r(a)``, and ``-inf`` on a node with no completion
                within the bound. Such a node carries no key and is dropped by the caller.
        """
        if goal is None:
            key: tuple[Any, int] = (None, -1)
        elif goal.success:
            key = (True, size)
        else:
            key = (self.holes_of(goal), size)
        cache = self.__dict__.get("_weight_cache")
        if cache is None:
            cache = {}
            object.__setattr__(self, "_weight_cache", cache)
        cached = cache.get(key)
        if cached is not None:
            return cached
        terms = [
            math.log(count) + self.log_unit_weights[value]
            for value, count in self.branch_counts_of(goal, size).items()
            if value in self.log_unit_weights
        ]
        value = log_sum_exp(terms)
        cache[key] = value
        return value

    def stream(self, rng: random.Random) -> Iterator[Tree[T]]:
        """Draw one stream from this weighted table.

        Args:
            rng (random.Random): The source of randomness.

        Yields:
            Tree[T]: The inhabitants, in decreasing key order.
        """
        for _, inhabitant in self.keyed_stream(rng):
            yield inhabitant

    def keyed_stream(self, rng: random.Random) -> Iterator[tuple[float, Tree[T]]]:
        """Draw one stream, keeping the key each inhabitant was streamed under.

        Args:
            rng (random.Random): The source of randomness.

        Yields:
            tuple[float, Tree[T]]: The key and the inhabitant, in decreasing key order.
        """
        if not self.root_counts:
            return
        select = deepest_first_subgoal if self.subgoal_selection is None else self.subgoal_selection

        def expand(
            node: tuple[Goal[NT, T, G] | None, int],
        ) -> tuple[Tree[T] | None, Sequence[tuple[tuple[Goal[NT, T, G] | None, int], float]]]:
            """Expand one node on demand, and weigh its children from the table.

            The retention rule is the tree form's, position for position: a success node within
            the bound is a leaf, anything past the bound is dropped, and a child whose weight
            vanishes carries no key and never enters the frontier. The tree form decides the last
            of these after expanding the child. Here the table decides it before, which is the
            same decision taken earlier.

            Args:
                node (tuple): The goal (None at the root) and the size of its partial inhabitant.

            Returns:
                tuple: The inhabitant (None on an inner node) and the retained children with
                    their ``log w``, in clause order.
            """
            goal, size = node
            if goal is not None and goal.success:
                return goal.grounded[()][1], ()
            raw = initial_nodes(self.query) if goal is None else child_nodes(self.query, goal, size, select)
            kept: list[tuple[tuple[Goal[NT, T, G] | None, int], float]] = []
            for child, child_size in raw:
                if child_size > self.size_bound:
                    continue
                log_weight = self.log_weight_of(child, child_size)
                if log_weight > -math.inf:
                    kept.append(((child, child_size), log_weight))
            return None, kept

        yield from keyed_stream((None, 0), self.log_weight_of(None, 0), expand, rng)


def weighted_table(
    query: ResolutionQuery[NT, T, G],
    size_bound: int,
    distribution: Callable[[Any], float],
    *,
    subgoal_selection: Callable[[Goal[NT, T, G]], tuple[Path, NonTerminalArgument[NT]]] | None = None,
    table: SizeTable[NT] | None = None,
) -> WeightedTable[NT, T, G]:
    """Build the table form of random search for a query.

    Args:
        query (ResolutionQuery[NT, T, G]): The query to sample from, generator or partial-term.
        size_bound (int): The bound ``D`` on the term size.
        distribution (Callable[[Any], float]): ``pi``, evaluated on each realized size. It need
            not be normalized, but it must be positive on every realized size.
        subgoal_selection (Callable | None): The computation rule. (Default value = None)
        table (SizeTable[NT] | None): A table already filled for this program and bound. Passing
            one is how several queries against one space share the cost of filling it, since the
            table depends on the program and the bound alone and not on the query. (Default value
            = None)

    Returns:
        WeightedTable[NT, T, G]: The construction, ready to stream from.

    Raises:
        ValueError: If a predicate of the program reads a hole, in which case the table would
            overcount and the sample would follow weights other than ``w``, if ``size_bound`` is
            negative, or if ``distribution`` is not positive on some realized size.
    """
    # Both checks run whether or not a table was handed in. Routing them through `size_table`
    # alone would make a prebuilt table an escape hatch out of the very hypothesis this
    # construction depends on, and would turn a negative bound into "this space has no
    # inhabitants", a silent substitute for a caller's mistake.
    if size_bound < 0:
        msg = f"the size bound counts function symbols and cannot be negative: {size_bound}"
        raise ValueError(msg)
    decomposable_or_raise(query.solution_space)
    filled = size_table(query.solution_space, size_bound, check=False) if table is None else table
    if filled.bound < size_bound:
        msg = (
            f"the table was filled to {filled.bound} but the search asks for {size_bound}; "
            f"the missing rows would read as zero and cut the space short"
        )
        raise ValueError(msg)

    root_counts: dict[int, int] = {}
    probe = WeightedTable(
        query=query,
        size_bound=size_bound,
        table=filled,
        root_counts={},
        unit_weights={},
        log_unit_weights={},
        subgoal_selection=subgoal_selection,
    )
    for goal, size in initial_nodes(query):
        for value, count in probe.branch_counts_of(goal, size).items():
            root_counts[value] = root_counts.get(value, 0) + count

    unit_weights, log_unit_weights = _spread(distribution, root_counts)
    return WeightedTable(
        query=query,
        size_bound=size_bound,
        table=filled,
        root_counts=root_counts,
        unit_weights=unit_weights,
        log_unit_weights=log_unit_weights,
        subgoal_selection=subgoal_selection,
    )


def random_search_keyed(
    query: ResolutionQuery[NT, T, G],
    size_bound: int,
    cost: Callable[[Tree[T]], Any],
    distribution: Callable[[Any], float],
    rng: random.Random,
) -> Iterator[tuple[float, Tree[T]]]:
    """Run random search and stream each inhabitant with the key it was drawn under.

    Args:
        query (ResolutionQuery[NT, T, G]): The query to sample from.
        size_bound (int): The bound ``D`` on the term size.
        cost (Callable[[Tree[T]], Any]): The cost function ``c``.
        distribution (Callable[[Any], float]): The distribution ``pi`` on the cost values.
        rng (random.Random): The source of randomness.

    Yields:
        tuple[float, Tree[T]]: The key and the inhabitant, in decreasing key order.
    """
    yield from weighted_tree(query, size_bound, cost, distribution).keyed_stream(rng)


def random_search(
    query: ResolutionQuery[NT, T, G],
    size_bound: int,
    cost: Callable[[Tree[T]], Any],
    distribution: Callable[[Any], float],
    rng: random.Random,
) -> Iterator[Tree[T]]:
    """Run random search on a query.

    Args:
        query (ResolutionQuery[NT, T, G]): The query to sample from.
        size_bound (int): The bound ``D`` on the term size.
        cost (Callable[[Tree[T]], Any]): The cost function ``c``.
        distribution (Callable[[Any], float]): The distribution ``pi`` on the cost values.
        rng (random.Random): The source of randomness.

    Yields:
        Tree[T]: The inhabitants of size at most ``D``, in random order. Under unambiguity
            within the bound each exactly once, with every prefix a sample without replacement
            in proportion to ``w``.
    """
    for _, inhabitant in random_search_keyed(query, size_bound, cost, distribution, rng):
        yield inhabitant


def _uniform(_value: Any) -> float:
    """Give every realized cost value the same probability.

    Args:
        _value (Any): The cost value. Ignored.

    Returns:
        float: One; :func:`weighted_tree` normalizes over the realized values.
    """
    return 1.0


def size_uniform(query: ResolutionQuery[NT, T, G], size_bound: int, rng: random.Random) -> Iterator[Tree[T]]:
    """Run size-uniform sampling: random search under term size and the uniform distribution.

    The size-uniform draw is the default of the package. Its weight is
    ``w(t) = 1 / (|S| * N_r(size(t)))`` over the realized sizes ``S``, so it draws a realized
    size uniformly and then an inhabitant of that size uniformly. It spreads a sample over the
    sizes instead of concentrating it on the many large inhabitants, and it needs nothing of the
    program beyond term size, which is why it is meaningful in every problem domain and the
    right default where no specific cost function is at hand.

    Args:
        query (ResolutionQuery[NT, T, G]): The query to sample from.
        size_bound (int): The bound ``D`` on the term size.
        rng (random.Random): The source of randomness.

    Yields:
        Tree[T]: The inhabitants of size at most ``D``, in random order.
    """
    yield from random_search(query, size_bound, term_size, _uniform, rng)
