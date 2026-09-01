"""Cost orders and the best-first family: informed search over a cost that may be partial.

An uninformed search rule reads the order of its stream off the shape of the derivation tree. An
informed one reads it off a cost function on the search nodes. This module carries the three
layers that reading takes, and they are separable on purpose.

* The **cost domain** a cost function maps into. A partially ordered set, strengthened to a
  positively ordered commutative monoid wherever costs are summed: monotone under addition, with
  ``0`` as its least element.
* The **best-first frontier**, whose pop returns a node of minimal cost, where minimal means that
  no node of the frontier lies strictly below it.
* The **cost algebras**, which read a search node as the partial inhabitant it denotes and split
  its cost into the cost-so-far ``g`` and the heuristic ``h`` of A* search.

**The order stays partial.** This is the recurring source of subtle mistakes. Best-first search
pops a *minimal* node, not a least one. Several incomparable minima may exist at once, and the
rule returns any of them. The cost-bounded sets that carry its completeness collect the nodes
whose cost is **not strictly above** the bound, which over a partial order is strictly weaker
than "at most the bound". A binary heap is therefore not the general implementation but a fast
path, available exactly when the domain is totally ordered. That is the case random search lives
in, where the costs are negated Gumbel keys and the heap in :mod:`cosy.search.sampling` is this
fast path, specialized.

**Name collision, stated once.** Cost-algebraic heuristic search, in the sense of Edelkamp et al.
and of Holte and Zilles, calls its ordered cost structure a *cost algebra*. That notion is the
**cost domain** here, :class:`CostDomain`. The *cost algebra* of this module is an algebra over
the signature of the synthesized program whose carrier is such a domain. The two are kept apart
throughout.

**Two levels of structure, and why they are separate classes.** A cost function is asked for no
more than a map into a partially ordered set, and best-first search consumes no more than that,
which is :class:`CostOrder`. A heuristic search sums estimates with costs already paid, and that
is where the monoid comes in, which is :class:`CostDomain`. The distinction is not academic. The
randomizing cost function of random search is a negated Gumbel key, a real number that is as
often negative as not, so it lives in :class:`Reals`, an order with no positivity to it, and no
additive cost algebra can be built over it.

**Scope.** The constructions here rest on the shape of the synthesized clauses. A clause head
applies one function symbol to pairwise distinct fresh variables, so a hole occurs exactly once
in the partial inhabitant and an expansion fills exactly one of them. They do not carry over to
arbitrary amalgamated logic programs.

**Notation.** Function symbols are written ``F`` and never ``f``, because ``f``, ``g`` and ``h``
name the cost functions. And ``f = g + h`` is never an equation between *functions*, since ``+``
is defined on the cost domain and not on functions. Every equation here is pointwise:
``f(n) = g(n) + h(n)``.

**Costs in floating point.** Both cost domains here add IEEE doubles, the componentwise domain
one component at a time. Double addition is commutative but not associative. Two routes that add
the same costs in a different order therefore agree up to rounding rather than exactly, and the
additive split ``f(n) = g(n) + h(n)`` and the step cost ``g(n') = g(n) + delta(n, n')`` are two
such pairs. The monotonicity of ``g`` survives exactly. Correctly rounded addition is monotone in
each argument, so a family with further nonnegative summands inserted never sums below the family
without them. An expansion only adds symbol occurrences to the partial inhabitant.

**Deliberate limits.** Cost-limited search and its iterative-deepening form, IDA*, are not
implemented. The pair trades frontier memory for repeated expansions, and IDA* assumes a total
cost order besides, which best-first search and A* do not need.
"""

from __future__ import annotations

import heapq
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from cosy.search.partial import Hole, partial_inhabitant
from cosy.search.rules import deepest_first_subgoal

if TYPE_CHECKING:
    from collections import deque
    from collections.abc import Callable, Iterable, Sequence

    from cosy.core.solution_space import NT, G, Goal, T
    from cosy.core.tree import Path, Tree
    from cosy.search.queries import ResolutionQuery

__all__ = [
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
    "cost_bounded_nodes",
    "greedy",
    "uniform_cost",
    "zero_assignment",
]

A = TypeVar("A")
N = TypeVar("N")

# The largest float, the upper end of the float range. Reals, NonNegativeReals, and
# ComponentwiseTuples decide membership by comparing against it, because their carriers are the
# floats and tuples of floats.
#
# They compare rather than call math.isfinite, which raises OverflowError on an int too large
# to convert to a float. is_element is a predicate and has to decide for such an int as well.
# Python compares an int against a float exactly and converts neither, so the comparison decides
# it. A NaN and an infinity fail the comparison, so neither is a cost.
_LARGEST_FLOAT = sys.float_info.max


class CostOrder(ABC, Generic[A]):
    """A partially ordered set of costs: what a cost function maps into.

    A cost function is a map from the search nodes of a derivation tree into a partially ordered
    set ``(A, <=)``, the cost domain, and best-first search consumes nothing else. Its pop returns
    a node of minimal cost, and minimality is a notion of the order alone. Everything that *sums*
    costs, which is to say the heuristics and the additive split below, needs the richer
    :class:`CostDomain`.

    The order is **partial**. Two costs may be incomparable, and every statement this module makes
    is phrased to survive that: minimal rather than least, and "not strictly above" rather than
    "at most" in the cost-bounded sets. :attr:`is_total` reports the special case in which the two
    collapse and a binary heap becomes correct.
    """

    @abstractmethod
    def le(self, left: A, right: A) -> bool:
        """Decide the partial order ``left <= right``.

        Args:
            left (A): The lower candidate.
            right (A): The upper candidate.

        Returns:
            bool: True if ``left`` is at most ``right``. Two costs with ``not le(a, b)`` and
                ``not le(b, a)`` are incomparable, which a partial order permits and which the
                frontier of this module is built to handle.

        Raises:
            ValueError: If an argument is not an element of the carrier.
        """

    @abstractmethod
    def is_element(self, value: object) -> bool:
        """Decide whether a value belongs to the carrier.

        This is where a cost that does not belong is caught, be it a negative symbol cost over a
        positive domain or a tuple of the wrong length, rather than at the first search that
        quietly stops being complete.

        Args:
            value (object): The candidate.

        Returns:
            bool: True if the value is a cost of this domain.
        """

    @property
    @abstractmethod
    def is_total(self) -> bool:
        """Whether the order is total.

        Returns:
            bool: True if any two costs are comparable. Only then does a binary heap order the
                frontier correctly, and only then is the cost-bounded set of a bound ``c`` the
                same as ``{n : f(n) <= c}``.
        """

    def lt(self, left: A, right: A) -> bool:
        """Decide the strict order ``left < right``.

        Args:
            left (A): The lower candidate.
            right (A): The upper candidate.

        Returns:
            bool: True if ``left <= right`` and not ``right <= left``.
        """
        return self.le(left, right) and not self.le(right, left)

    def incomparable(self, left: A, right: A) -> bool:
        """Decide whether two costs are incomparable.

        Args:
            left (A): The first cost.
            right (A): The second cost.

        Returns:
            bool: True if neither is at most the other. Always False on a total order.
        """
        return not self.le(left, right) and not self.le(right, left)

    def in_down_set(self, value: A, bound: A) -> bool:
        """Decide membership in the cost-bounded set ``D_bound``.

        The cost-bounded set of a bound ``c`` is ``D_c = {n : f(n) > c is false}``, the nodes
        whose cost is **not strictly above** the bound, and not those of cost at most the bound.
        Over a total order the two agree. Over a partial one a cost incomparable to the bound lies
        in ``D_c`` and is not at most it, and the difference is what makes the completeness
        argument for best-first search hold there: a node of incomparable cost can be minimal at
        the same time as a success node, so a pop may take it first, and an argument that did not
        count it would miss an infinite antichain of such competitors.

        Args:
            value (A): The cost under test.
            bound (A): The bound ``c``.

        Returns:
            bool: True if ``value`` is not strictly above ``bound``.
        """
        return not self.lt(bound, value)


class Reals(CostOrder[float]):
    """The reals under their usual total order: an order, and nothing more.

    The cost order of random search, which gives every retained search node the *negated* Gumbel
    key of the inhabitants below it. That key is a real number that is negative as often as not,
    so the values are outside every positively ordered monoid, and they need not be inside one,
    because best-first search asks only for a partial order. Keeping this class free of ``+`` and
    ``0`` records that: no additive cost algebra can be built over it, and none is wanted, since
    the randomizing cost function is drawn rather than accumulated.

    That total order is what :class:`HeapFrontier` requires of a cost order.
    """

    def le(self, left: float, right: float) -> bool:
        """Compare two reals.

        Args:
            left (float): The lower candidate.
            right (float): The upper candidate.

        Returns:
            bool: True if ``left <= right``.

        Raises:
            ValueError: If an argument is not a real inside the float range.
        """
        if not self.is_element(left) or not self.is_element(right):
            msg = f"not an element of the cost order of reals: {left!r}, {right!r}"
            raise ValueError(msg)
        return left <= right

    def is_element(self, value: object) -> bool:
        """Decide whether a value is a real inside the float range.

        Args:
            value (object): The candidate.

        Returns:
            bool: True for a value of either sign inside the float range. A NaN is not a cost,
                since it is comparable to nothing and would make the order's laws fail
                silently. An int beyond the float range is not a cost either, since the
                carrier is the floats.
        """
        return isinstance(value, (int, float)) and -_LARGEST_FLOAT <= value <= _LARGEST_FLOAT

    @property
    def is_total(self) -> bool:
        """Whether the order is total.

        Returns:
            bool: True.
        """
        return True


class CostDomain(CostOrder[A], ABC):
    """A positively ordered commutative monoid: a cost order in which costs can be summed.

    A cost domain is a commutative monoid ``(A, +, 0)`` with a partial order ``<=`` that is
    *monotone*, so that ``a <= b`` implies ``a + c <= b + c``, and *positive*, so that ``0 <= a``
    for every ``a``. Then ``0`` is the least element and every cost is positive. The positivity
    terminology follows Fuchs, who does not assume commutativity.

    Commutativity is required here rather than offered as a convenience. The additive split below
    and the cost-so-far function sum the symbol costs of a term in whatever order a traversal
    produces them, and the sums must not depend on it.

    Subclasses report :attr:`is_archimedean` alongside :attr:`CostOrder.is_total` rather than
    leaving it to be guessed. It is one of the two hypotheses under which uniform-cost search is
    complete, and the componentwise domain below fails it while looking entirely ordinary.

    This is *not* the cost algebra of :class:`AdditiveCostAlgebra`. See the module docstring on
    the name collision with cost-algebraic heuristic search, whose "cost algebra" is this class.
    """

    @property
    @abstractmethod
    def zero(self) -> A:
        """Return the neutral element ``0``, the least element of the order.

        Returns:
            A: The neutral element.
        """

    @abstractmethod
    def add(self, left: A, right: A) -> A:
        """Add two costs.

        Args:
            left (A): The first summand.
            right (A): The second summand.

        Returns:
            A: Their sum. The operation is associative, commutative and neutral at :attr:`zero`.

        Raises:
            ValueError: If an argument is not an element of the carrier, or if the sum leaves
                it. A monoid is closed under its operation, so a sum outside the carrier is
                reported where it arises rather than by the next operation to receive it.
        """

    @property
    @abstractmethod
    def is_archimedean(self) -> bool:
        """Whether the domain is archimedean.

        Archimedean means that for every ``a > 0`` and every ``c`` there is an ``m`` with
        ``m (*) a > c``, the iterated sum of :meth:`iterated_sum`. The strict inequality is what
        matters over a partial order. An iterated sum merely *incomparable* to ``c`` keeps its
        node inside the cost-bounded set of ``c`` and does not help. Together with strictly
        positive combinator costs this is the hypothesis :func:`assert_uniform_cost_complete`
        checks.

        Returns:
            bool: True if every strictly positive cost passes every bound when summed often
                enough.
        """

    def is_positive(self, value: A) -> bool:
        """Decide whether a cost is positive, that is ``0 <= value``.

        Args:
            value (A): The cost under test.

        Returns:
            bool: True for every element of the carrier, by positivity of the order.
        """
        return self.le(self.zero, value)

    def is_strictly_positive(self, value: A) -> bool:
        """Decide whether a cost is strictly positive, that is ``0 < value``.

        Args:
            value (A): The cost under test.

        Returns:
            bool: True if the cost is above zero and not zero itself. Strict positivity of the
                combinator costs is the second hypothesis of :func:`assert_uniform_cost_complete`.
        """
        return self.lt(self.zero, value)

    def sum_of(self, values: Iterable[A]) -> A:
        """Sum a family of costs.

        Commutativity and associativity make the result independent of the order the values
        arrive in, which is what lets the additive split sum over the *occurrences* of a term in
        whatever order a traversal produces. Addition over the two domains here is associative only
        up to rounding, and the module docstring says what that leaves standing.

        Args:
            values (Iterable[A]): The summands. The empty family sums to :attr:`zero`.

        Returns:
            A: Their sum.
        """
        total = self.zero
        for value in values:
            total = self.add(total, value)
        return total

    def iterated_sum(self, times: int, value: A) -> A:
        """Return the iterated sum ``times (*) value``.

        Args:
            times (int): How often to add the value. ``0`` gives :attr:`zero`.
            value (A): The summand.

        Returns:
            A: The ``times``-fold sum of ``value`` with itself.

        Raises:
            ValueError: If ``times`` is negative. The iterated sum is defined by recursion on a
                natural number, so a negative count is a caller error and has no value to fall
                back on.
        """
        if times < 0:
            msg = f"the repetition count of an iterated sum is a natural number: {times}"
            raise ValueError(msg)
        total = self.zero
        for _ in range(times):
            total = self.add(total, value)
        return total


class NonNegativeReals(CostDomain[float]):
    """The nonnegative reals under addition: the standard, totally ordered cost domain.

    Positively ordered, commutative, totally ordered and archimedean. So the completeness argument
    for uniform-cost search applies to every additive cost algebra over this domain whose
    combinator costs are strictly positive, and the heap fast path of :class:`HeapFrontier` is
    available. Counting costs live here: one per combinator, the term size, a number of
    parameters, a memory footprint.

    Negative values are outside the carrier, which is the whole point of the positivity axiom and
    not a technicality. A negative step cost would make ``g`` fall along a branch, and with it
    every ordering and completeness statement of this module. Costs that genuinely take both
    signs, the negated Gumbel keys of random search among them, belong to :class:`Reals`, which is
    an order and not a monoid.
    """

    @property
    def zero(self) -> float:
        """Return ``0.0``.

        Returns:
            float: The neutral element.
        """
        return 0.0

    def add(self, left: float, right: float) -> float:
        """Add two nonnegative reals.

        Args:
            left (float): The first summand.
            right (float): The second summand.

        Returns:
            float: Their sum.

        Raises:
            ValueError: If an argument is not a nonnegative real inside the float range, or
                if the sum overflows that range. Closure under addition is a monoid axiom, so the
                sum is checked as well as the arguments. Unchecked, the sum travels on as an
                infinity, and the next operation to receive it reports a value the caller
                never passed in.
        """
        self._require(left)
        self._require(right)
        total = left + right
        self._require(total)
        return total

    def le(self, left: float, right: float) -> bool:
        """Compare two nonnegative reals.

        Args:
            left (float): The lower candidate.
            right (float): The upper candidate.

        Returns:
            bool: True if ``left <= right``.

        Raises:
            ValueError: If an argument is not a nonnegative real inside the float range.
        """
        self._require(left)
        self._require(right)
        return left <= right

    def is_element(self, value: object) -> bool:
        """Decide whether a value is a nonnegative real inside the float range.

        Args:
            value (object): The candidate.

        Returns:
            bool: True for a value at or above zero and inside the float range. An int
                beyond the float range is not a cost, since the carrier is the floats.
        """
        return isinstance(value, (int, float)) and 0.0 <= value <= _LARGEST_FLOAT

    @property
    def is_total(self) -> bool:
        """Whether the order is total.

        Returns:
            bool: True, since any two reals are comparable.
        """
        return True

    @property
    def is_archimedean(self) -> bool:
        """Whether the domain is archimedean.

        Returns:
            bool: True. For ``a > 0`` and any bound ``c``, ``m a > c`` once ``m > c / a``.
        """
        return True

    def _require(self, value: float) -> None:
        """Reject a value outside the carrier.

        Args:
            value (float): The candidate.

        Raises:
            ValueError: If the value is not a nonnegative real inside the float range.
        """
        if not self.is_element(value):
            msg = f"not an element of the cost domain of nonnegative reals: {value!r}"
            raise ValueError(msg)


class ComponentwiseTuples(CostDomain["tuple[float, ...]"]):
    """Tuples of nonnegative reals under the componentwise order: a genuinely partial domain.

    A cost with several independent parts, a number of combinators *and* a number of parameters
    say, has no canonical way of being weighed against another, and the componentwise order says
    so: ``(1, 0)`` and ``(0, 1)`` are incomparable. Best-first search is defined over exactly this
    generality, and the domain is here so that the partiality of the order is exercised rather
    than merely asserted.

    It is **not archimedean** for two components or more. The iterated sum ``m (*) (1, 0)`` is
    ``(m, 0)``, which stays incomparable to ``(0, 1)`` for every ``m``, so no iterated sum ever
    passes that bound. This is the reason the archimedean property is stated with a strict
    inequality, and it means the completeness argument for uniform-cost search gives nothing here:
    the cost-bounded set of ``(0, 1)`` contains every node built from ``(1, 0)`` alone.
    Uniform-cost search over this domain is still sound and still streams a linear extension of
    the cost order. Only completeness needs a separate argument.

    Attributes:
        arity (int): The number of components.
    """

    def __init__(self, arity: int) -> None:
        """Build the domain of tuples of one fixed length.

        Args:
            arity (int): The number of components, at least one.

        Raises:
            ValueError: If the arity is not positive. A cost of no components carries no
                information, and the empty tuple would make every cost equal to every other.
        """
        if arity < 1:
            msg = f"a componentwise cost domain needs at least one component, got arity {arity}"
            raise ValueError(msg)
        self.arity = arity

    @property
    def zero(self) -> tuple[float, ...]:
        """Return the all-zero tuple.

        Returns:
            tuple[float, ...]: The neutral element.
        """
        return (0.0,) * self.arity

    def add(self, left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
        """Add two tuples componentwise.

        Args:
            left (tuple[float, ...]): The first summand.
            right (tuple[float, ...]): The second summand.

        Returns:
            tuple[float, ...]: Their componentwise sum.

        Raises:
            ValueError: If an argument is not an element of the carrier, or if the sum leaves
                it. A component that overflows the float range takes the tuple out of the
                carrier. The sum is checked as well as the arguments, for the reason given at
                :meth:`NonNegativeReals.add`.
        """
        self._require(left)
        self._require(right)
        total = tuple(a + b for a, b in zip(left, right, strict=True))
        self._require(total)
        return total

    def le(self, left: tuple[float, ...], right: tuple[float, ...]) -> bool:
        """Compare two tuples componentwise.

        Args:
            left (tuple[float, ...]): The lower candidate.
            right (tuple[float, ...]): The upper candidate.

        Returns:
            bool: True if every component of ``left`` is at most the corresponding one of
                ``right``. False in both directions for an incomparable pair.

        Raises:
            ValueError: If an argument is not an element of the carrier.
        """
        self._require(left)
        self._require(right)
        return all(a <= b for a, b in zip(left, right, strict=True))

    def is_element(self, value: object) -> bool:
        """Decide whether a value is a tuple of the right length with nonnegative entries inside the float range.

        Args:
            value (object): The candidate.

        Returns:
            bool: True for a tuple of :attr:`arity` entries at or above zero and inside the
                float range. An int beyond the float range is not such an entry, since the
                carrier is tuples of floats.
        """
        return (
            isinstance(value, tuple)
            and len(value) == self.arity
            and all(isinstance(part, (int, float)) and 0.0 <= part <= _LARGEST_FLOAT for part in value)
        )

    @property
    def is_total(self) -> bool:
        """Whether the order is total.

        Returns:
            bool: True only for a single component. From two on, incomparable tuples exist.
        """
        return self.arity == 1

    @property
    def is_archimedean(self) -> bool:
        """Whether the domain is archimedean.

        Returns:
            bool: True only for a single component. With two or more, ``(m, 0)`` never exceeds
                ``(0, 1)``, since the pair stays incomparable, so the property fails.
        """
        return self.arity == 1

    def _require(self, value: tuple[float, ...]) -> None:
        """Reject a value outside the carrier.

        Args:
            value (tuple[float, ...]): The candidate.

        Raises:
            ValueError: If the value is not a tuple of the domain's arity whose entries are
                nonnegative and inside the float range.
        """
        if not self.is_element(value):
            msg = f"not an element of the cost domain of componentwise tuples of arity {self.arity}: {value!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class CostFunction(Generic[N, A]):
    """A cost function: a map from search nodes into a cost domain.

    The domain travels with the map because the definition puts it there. A cost function is a map
    into a *partially ordered set*, and the frontier that consumes it needs the order and not just
    the values. Keeping the two in one object also removes the one mismatch a two-argument
    interface invites, a cost computed in one domain and compared in another.

    Attributes:
        domain (CostOrder[A]): The cost domain, which is to say the partially ordered set the
            values live in. A :class:`CostDomain` where the costs are summed, a bare
            :class:`CostOrder` where they are only compared.
        evaluate (Callable[[N], A]): The map itself. Its argument is a search node, a ``Goal`` for
            the searches of this module and an arbitrary object for a frontier used on its own.
    """

    domain: CostOrder[A]
    evaluate: Callable[[N], A]

    def __call__(self, node: N) -> A:
        """Return the cost of a node.

        Args:
            node (N): The search node.

        Returns:
            A: Its cost.
        """
        return self.evaluate(node)


def cost_bounded_nodes(nodes: Iterable[N], cost: CostFunction[N, A], bound: A) -> tuple[N, ...]:
    """Return the nodes of a cost-bounded set ``D_bound``.

    The membership test is :meth:`CostOrder.in_down_set`, which is to say not strictly above the
    bound, and over a partial order that admits incomparable costs. The use is the completeness
    argument for best-first search: if every cost-bounded set is finite then the search is
    complete, since while a node sits in the frontier every node popped before it lies in that
    node's cost-bounded set. Finiteness is a property of the *whole* derivation tree and therefore
    not decidable by inspection. This function selects from a given collection and does not
    attempt to enumerate one.

    Args:
        nodes (Iterable[N]): The nodes to select from.
        cost (CostFunction[N, A]): The cost function ``f``.
        bound (A): The bound ``c``.

    Returns:
        tuple[N, ...]: The nodes whose cost is not strictly above the bound, in input order.
    """
    return tuple(node for node in nodes if cost.domain.in_down_set(cost(node), bound))


class Frontier(ABC, Generic[N, A]):
    """A best-first frontier: its pop returns a node of minimal cost.

    *Minimal* is the order-theoretic notion and not a least element. No node of the frontier lies
    strictly below the node returned. Several incomparable minima may exist at once, and the rule
    returns any one of them, a nondeterminism the generic search rule already permits.

    The success test happens when a node leaves the frontier, never when it is created. For
    best-first search that late test is exactly what orders the stream by cost: a node yields its
    inhabitant only once its cost is minimal, given a cost function that never falls along a
    branch.

    A frontier can also **maximize**, which is the order-dual of the same construction. Its pop
    then returns a node no other node lies strictly above. Tree-kernel scores are the case in
    hand, since a search for terms *similar* to a reference set maximizes, and turning the
    frontier around rather than negating the values keeps the construction available over domains
    where negation does not exist.

    Attributes:
        cost (CostFunction[N, A]): The cost function ``f`` and its domain.
        maximize (bool): Whether the pop returns a maximal instead of a minimal node.
    """

    def __init__(self, cost: CostFunction[N, A], *, maximize: bool = False) -> None:
        """Build an empty frontier.

        Args:
            cost (CostFunction[N, A]): The cost function ``f``.
            maximize (bool): Pop a maximal node instead of a minimal one, which is the order-dual.
                (Default value = False)
        """
        self.cost = cost
        self.maximize = maximize

    @property
    def domain(self) -> CostOrder[A]:
        """Return the cost order the frontier compares by.

        Returns:
            CostOrder[A]: The domain of :attr:`cost`. A frontier compares and never sums, so the
                order alone is enough, which is what lets random search's negated keys use this
                frontier without a monoid over them.
        """
        return self.cost.domain

    def prefers(self, left: A, right: A) -> bool:
        """Decide whether one cost is strictly better than another, in the frontier's direction.

        Args:
            left (A): The candidate.
            right (A): The incumbent.

        Returns:
            bool: For a minimizing frontier ``left < right``, for a maximizing one
                ``right < left``. False for incomparable costs in both cases, which is what makes
                a pop return *a* minimal element rather than a least one.
        """
        if self.maximize:
            return self.domain.lt(right, left)
        return self.domain.lt(left, right)

    @abstractmethod
    def push(self, node: N) -> None:
        """Add a node to the frontier.

        Args:
            node (N): The search node. Its cost is evaluated once, here. A cost function on goals
                materializes the partial inhabitant, and evaluating it per comparison would rebuild
                that inhabitant once per comparison instead of once per node.
        """

    @abstractmethod
    def pop(self) -> N:
        """Remove and return a node of minimal cost.

        Returns:
            N: The node. No node left in the frontier has a cost strictly below its own, or
                strictly above it when maximizing.

        Raises:
            ValueError: If the frontier is empty. An empty frontier has no minimal element, and
                there is no substitute for one.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of nodes in the frontier.

        Returns:
            int: The size. The emptiness test of the generic search rule is this being zero.
        """

    def __bool__(self) -> bool:
        """Decide whether the frontier holds a node.

        Returns:
            bool: True if it is non-empty.
        """
        return len(self) > 0

    def variance_strategies(
        self,
    ) -> tuple[
        Callable[[deque[Any], Iterable[Any]], deque[Any]],
        Callable[[deque[Any]], tuple[deque[Any], Any]],
    ]:
        """Return the push and pop pair ``SolutionSpace.resolution`` takes as its frontier.

        The engine holds its frontier in a ``deque`` it passes to both callables and never reads
        except for its emptiness. The loop runs while the queue is truthy, and the popped goal
        comes from the callable. A best-first frontier cannot use that deque as its storage, since
        a linear scan for a minimal element and a binary heap both need their own arrangement. The
        adapter therefore keeps the engine's queue as a *cardinality mirror*, one entry per node
        in the frontier with the contents unread, while the ordered storage lives in this object.
        The mirror is checked on every pop, so a caller who hands back a queue that did not see
        the pushes fails loudly instead of silently searching a frontier the engine no longer
        believes in.

        Returns:
            tuple: The ``variance_strategy_push`` and ``variance_strategy_pop`` of the engine.
        """

        def push_all(queue: deque[Any], new_goals: Iterable[Any]) -> deque[Any]:
            """Push a batch of children onto the frontier in the order it arrives.

            Both frontiers of this module break ties by insertion order, so reordering a batch
            here would move equally cheap children past each other in the stream.

            Args:
                queue (deque[Any]): The engine's cardinality mirror.
                new_goals (Iterable[Any]): The children of one expansion.

            Returns:
                deque[Any]: The mirror, grown by one entry per child.
            """
            for goal in new_goals:
                self.push(goal)
                queue.append(goal)
            return queue

        def pop_one(queue: deque[Any]) -> tuple[deque[Any], Any]:
            """Pop a node of minimal cost.

            Args:
                queue (deque[Any]): The engine's cardinality mirror.

            Returns:
                tuple[deque[Any], Any]: The mirror and the popped node.

            Raises:
                ValueError: If the mirror and the frontier disagree on their size.
            """
            if len(queue) != len(self):
                msg = (
                    f"the engine's queue holds {len(queue)} entries and the best-first frontier "
                    f"{len(self)}: the queue passed to pop is not the one the pushes went through"
                )
                raise ValueError(msg)
            queue.pop()
            return queue, self.pop()

        return push_all, pop_one


class LinearScanFrontier(Frontier[N, A]):
    """The general best-first frontier: one linear scan finds a minimal element.

    A partial order admits no heap, since a binary heap maintains an invariant between a node and
    its children that presupposes any two costs are comparable. The general frontier therefore
    keeps its nodes in insertion order and scans. One pass suffices, keeping a running candidate
    and replacing it whenever a node is *strictly* better: the candidates then form a strictly
    decreasing chain, and a node below the final candidate would have been below the candidate
    that held when it was scanned, so it would have replaced it. Ties never replace, which makes
    the scan return the earliest-inserted among equally cheap nodes. That is the same tie-break
    the heap's counter produces, and the reason the two agree on a total order.

    The cost is one pass per pop. For the total orders of practice, random search's Gumbel keys
    above all, :class:`HeapFrontier` is the specialization to use.
    """

    def __init__(self, cost: CostFunction[N, A], *, maximize: bool = False) -> None:
        """Build an empty frontier.

        Args:
            cost (CostFunction[N, A]): The cost function ``f``.
            maximize (bool): Pop a maximal node instead of a minimal one. (Default value = False)
        """
        super().__init__(cost, maximize=maximize)
        self._items: list[tuple[A, N]] = []

    def push(self, node: N) -> None:
        """Add a node, evaluating its cost once.

        Args:
            node (N): The search node.
        """
        self._items.append((self.cost(node), node))

    def pop(self) -> N:
        """Remove and return a node of minimal cost.

        Returns:
            N: The node.

        Raises:
            ValueError: If the frontier is empty.
        """
        if not self._items:
            msg = "cannot pop from an empty best-first frontier: it has no minimal element"
            raise ValueError(msg)
        best = 0
        for index in range(1, len(self._items)):
            if self.prefers(self._items[index][0], self._items[best][0]):
                best = index
        return self._items.pop(best)[1]

    def __len__(self) -> int:
        """Return the number of nodes in the frontier.

        Returns:
            int: The size.
        """
        return len(self._items)


class _Ordered(Generic[A]):
    """A cost and a tie-break counter, wrapped so that a binary heap can compare two entries.

    ``heapq`` compares its entries with ``<``, and a cost domain exposes its order as a method.
    The wrapper carries the domain and the direction, so that the same heap serves a minimizing
    and a maximizing frontier without the values ever being negated. Negation is not available on
    a general cost domain, and the order-dual is.

    The wrapper also carries the tie-break counter, so ``__lt__`` decides every comparison,
    equal costs included. A heap entry pairs a wrapper with its node. The class defines no
    ``__eq__``, so two wrappers are equal only when they are the same object, and every push
    builds its own. A tuple comparison therefore stops at the wrapper.

    Attributes:
        value (A): The wrapped cost.
    """

    __slots__ = ("_domain", "_maximize", "_tie_break", "value")

    def __init__(self, domain: CostOrder[A], value: A, tie_break: int, *, maximize: bool) -> None:
        """Wrap one cost together with the tie-break counter of its push.

        Args:
            domain (CostOrder[A]): The order to compare through.
            value (A): The cost.
            tie_break (int): The counter that breaks ties between equivalent costs.
            maximize (bool): Whether to compare in the dual order.
        """
        self._domain = domain
        self._maximize = maximize
        self._tie_break = tie_break
        self.value = value

    def __lt__(self, other: _Ordered[A]) -> bool:
        """Compare two wrappers by cost in the frontier's direction, equal costs by the counter.

        Args:
            other (_Ordered[A]): The other wrapper.

        Returns:
            bool: True if this wrapper's cost is strictly better, or the two costs are equivalent
                and this wrapper was pushed first.
        """
        left, right = (other.value, self.value) if self._maximize else (self.value, other.value)
        if not self._domain.le(left, right):
            return False
        if not self._domain.le(right, left):
            return True
        return self._tie_break < other._tie_break


class HeapFrontier(Frontier[N, A]):
    """The total-order fast path: a binary heap with a tie-break counter.

    A heap finds a least element in logarithmic time, and over a total order least and minimal
    coincide, so this frontier answers the same specification as :class:`LinearScanFrontier`, and
    the two are pinned to each other by test. Over a partial order it would be wrong and not
    merely slow: the heap invariant compares a node with its children only, and two incomparable
    costs would let a node that is *not* minimal reach the root. The constructor therefore refuses
    a domain that is not total rather than producing an order that looks like an answer.

    The tie-break counter increases with every push, so equal costs leave in insertion order. No
    comparison looks past the cost and the counter, so the nodes themselves are never compared,
    which matters because search nodes carry no order.
    The frontier of random search in :mod:`cosy.search.sampling` is this construction on the
    negated Gumbel keys: the keys are reals, the domain is total, and the heap there is this fast
    path written out in place.
    """

    def __init__(self, cost: CostFunction[N, A], *, maximize: bool = False) -> None:
        """Build an empty heap frontier.

        Args:
            cost (CostFunction[N, A]): The cost function ``f``.
            maximize (bool): Pop a maximal node instead of a minimal one. (Default value = False)

        Raises:
            ValueError: If the cost domain is not totally ordered. A heap cannot represent a
                partial order, and :class:`LinearScanFrontier` is the frontier for that case.
        """
        if not cost.domain.is_total:
            msg = (
                "a binary heap needs a total cost order, and this domain is only partially "
                "ordered. Use LinearScanFrontier, which scans for a minimal element"
            )
            raise ValueError(msg)
        super().__init__(cost, maximize=maximize)
        self._heap: list[tuple[_Ordered[A], N]] = []
        self._pushes = 0

    def push(self, node: N) -> None:
        """Add a node, evaluating its cost once.

        Args:
            node (N): The search node.
        """
        entry = (_Ordered(self.domain, self.cost(node), self._pushes, maximize=self.maximize), node)
        self._pushes += 1
        heapq.heappush(self._heap, entry)

    def pop(self) -> N:
        """Remove and return a node of least cost.

        Returns:
            N: The node.

        Raises:
            ValueError: If the frontier is empty.
        """
        if not self._heap:
            msg = "cannot pop from an empty best-first frontier: it has no minimal element"
            raise ValueError(msg)
        return heapq.heappop(self._heap)[1]

    def __len__(self) -> int:
        """Return the number of nodes in the frontier.

        Returns:
            int: The size.
        """
        return len(self._heap)


def best_first_frontier(cost: CostFunction[N, A], *, maximize: bool = False) -> Frontier[N, A]:
    """Build the best-first frontier a cost function's domain admits.

    The heap where the order is total, the scan otherwise. Both realize the same specification,
    the choice between them is a matter of cost, and it is made here so that no caller has to
    remember which domains admit a heap.

    Args:
        cost (CostFunction[N, A]): The cost function ``f``.
        maximize (bool): Pop a maximal node instead of a minimal one. (Default value = False)

    Returns:
        Frontier[N, A]: The frontier.
    """
    if cost.domain.is_total:
        return HeapFrontier(cost, maximize=maximize)
    return LinearScanFrontier(cost, maximize=maximize)


def best_first(
    query: ResolutionQuery[NT, T, G],
    cost: CostFunction[Goal[NT, T, G], A],
    *,
    maximize: bool = False,
    frontier: Frontier[Goal[NT, T, G], A] | None = None,
    max_count: int | None = None,
    max_depth: int | None = None,
    goal_filter: Callable[[Goal[NT, T, G]], bool] | None = None,
) -> Iterable[Tree[T]]:
    """Run best-first search on a query and stream its inhabitants.

    The engine is ``SolutionSpace.resolution``, the generic search rule, and this function
    supplies the frontier.

    **No clause order.** A search rule is a frontier and a clause order, and this one takes no
    clause order, because the pop selects by cost alone: no arrangement of one expansion's clauses
    can make a strictly more expensive node leave before a cheaper one. What an arrangement still
    decides is the sequence in which equally cheap nodes were pushed, and both frontiers here
    break ties by insertion order, so the engine's default order fixes that much and a caller has
    one order fewer to choose.

    **No computation rule parameter either.** The computation rule picks which hole is filled next,
    and the engine takes it as a parameter. This function fixes it to ``deepest_first_subgoal``,
    the one :func:`~cosy.search.rules.depth_first` and :func:`~cosy.search.rules.breadth_first` run
    under, so that the three rules explore one derivation tree. The choice shapes that tree: which
    nodes an expansion creates, and in which order the nodes of a run reach the frontier. The tree
    form and the table form of :mod:`cosy.search.sampling` do take it, as ``subgoal_selection``, so
    that the two can be compared under a rule other than the one they share by default. A caller
    that needs another computation rule calls ``SolutionSpace.resolution`` itself.

    Sound on every space, as every instance of the generic search rule is. Complete when every
    cost-bounded set is finite. Ordered by cost when the cost function never falls along a branch,
    and then the stream is a *linear extension* of the cost order, with incomparable costs in any
    order. Both hypotheses are properties of the cost function rather than of this function:
    :func:`uniform_cost` supplies them and :func:`greedy` does not.

    Args:
        query (ResolutionQuery[NT, T, G]): The generator or partial-term query to search.
        cost (CostFunction[Goal[NT, T, G], A]): The cost function ``f`` on search nodes, with its
            domain.
        maximize (bool): Search for maximal cost instead of minimal, the order-dual, for scores
            such as the tree kernels of :mod:`cosy.search.kernels`. (Default value = False)
        frontier (Frontier[Goal[NT, T, G], A] | None): A frontier to use instead of the one
            :func:`best_first_frontier` would build, for pinning the two implementations against
            each other, or for a frontier with its own tie-breaking. It must be empty, and it
            carries its own direction. (Default value = None)
        max_count (int | None): Stop after this many inhabitants. (Default value = None)
        max_depth (int | None): Prune goals whose positions exceed this depth. A pragmatic
            addition of the engine with no counterpart in the statements above, and in particular
            not the size bound of random search. (Default value = None)
        goal_filter (Callable[[Goal[NT, T, G]], bool] | None): An expansion filter on goals. A
            child it rejects is dropped where it is created, and the filter runs on the goals a
            query starts with as well. (Default value = None)

    Returns:
        Iterable[Tree[T]]: The stream of inhabitants.

    Raises:
        ValueError: If a frontier is passed together with ``maximize``, which would state the
            direction twice and let the two disagree, if the frontier passed is not empty, or if
            it searches by a different cost function than the one passed here.
    """
    if frontier is None:
        chosen: Frontier[Goal[NT, T, G], A] = best_first_frontier(cost, maximize=maximize)
    else:
        if maximize:
            msg = (
                "a frontier carries its own direction: pass either maximize=True or a frontier built with it, not both"
            )
            raise ValueError(msg)
        if len(frontier) > 0:
            msg = (
                f"the frontier of a search starts empty, but this one holds {len(frontier)} "
                f"nodes left over from an earlier search"
            )
            raise ValueError(msg)
        if frontier.cost is not cost:
            # Otherwise there would be two cost functions and no rule for which one applies: the
            # search would run on the frontier's, silently, while the caller reads its own in the
            # call. Worse when the two live in different domains, since the values would then be
            # compared across domains without either complaining.
            msg = (
                "the frontier searches by its own cost function, which is not the one passed "
                "here. Pass the frontier alone or build it from this cost function"
            )
            raise ValueError(msg)
        chosen = frontier
    push, pop = chosen.variance_strategies()
    return query.solution_space.resolution(
        query.start,
        push,
        pop,
        deepest_first_subgoal,
        max_count,
        max_depth,
        query.tree,
        query.pos,
        goal_filter=goal_filter,
    )


def _symbol_occurrences(term: Tree[Any]) -> dict[Path, Any]:
    """Return the function-symbol occurrences of a term, by position.

    Args:
        term (Tree[Any]): The term. Its leaves may be :class:`~cosy.search.partial.Hole` markers.

    Returns:
        dict[Path, Any]: One entry per position carrying a function symbol. Holes are variables
            and contribute none.
    """
    found: dict[Path, Any] = {}
    pending: list[tuple[Path, Tree[Any]]] = [((), term)]
    while pending:
        position, node = pending.pop()
        if not isinstance(node.root, Hole):
            found[position] = node.root
        pending.extend(((*position, index), child) for index, child in enumerate(node.children))
    return found


def _hole_occurrences(term: Tree[Any]) -> list[Hole[Any]]:
    """Return the hole occurrences of a term.

    Args:
        term (Tree[Any]): The term. Its leaves may be :class:`~cosy.search.partial.Hole` markers.

    Returns:
        list[Hole[Any]]: One entry per occurrence of a variable. At a search node of a generator
            query the holes are pairwise distinct and each occurs exactly once, so the list has no
            repetitions there. The additive split sums over occurrences all the same, and so does
            this function, so that two holes of one term are two summands.
    """
    found: list[Hole[Any]] = []
    pending = [term]
    while pending:
        node = pending.pop()
        if isinstance(node.root, Hole):
            found.append(node.root)
        pending.extend(node.children)
    return found


def zero_assignment(domain: CostDomain[A]) -> Callable[[Hole[Any]], A]:
    """Build the variable assignment that gives every hole the cost zero.

    The trivial admissible assignment. Zero is the least element of the cost domain, so the
    estimate of a hole stays below the fold of every filler and admissibility holds without any
    knowledge of the modeled domain. Its heuristic vanishes at every node, and A* search reduces
    to uniform-cost search, which is the sense in which every additive cost algebra admits
    uniform-cost search.

    Args:
        domain (CostDomain[A]): The cost domain.

    Returns:
        Callable[[Hole[Any]], A]: The assignment.
    """

    def assign(_hole: Hole[Any]) -> A:
        """Return zero for any hole.

        Args:
            _hole (Hole[Any]): The hole. Ignored.

        Returns:
            A: The neutral element of the domain.
        """
        return domain.zero

    return assign


class AdditiveCostAlgebra(Generic[A]):
    """An additive cost algebra and the two cost functions its split yields.

    A cost algebra reads a cost function as an algebra over the signature of the synthesized
    program whose carrier is a cost domain, so that the fold assigns each inhabitant its cost. A
    partial inhabitant carries holes, and holes are variables, so its cost is a *term assignment*
    and needs a variable assignment ``h`` on the holes, an estimate of what each hole's fillers
    will cost.

    An algebra is *additive* when each operation adds a fixed cost to the sum of its arguments,
    ``[[F]](a_1..a_n) = c_F + a_1 + ... + a_n``. Then the term assignment splits into a sum over
    the symbol occurrences and a sum over the variable occurrences, and at a search node of a
    generator query the variables are the pairwise distinct holes, each occurring exactly once.
    Those two sums are the two summands of A* search.

    * :meth:`cost_so_far` is ``g(n)``, the sum of the symbol costs of the partial inhabitant. It
      is a cost-so-far function, since filling a hole with ``F`` costs ``c_F`` and a step that
      writes no symbol costs nothing. Every step cost is positive, so ``g`` never falls along a
      branch.
    * :meth:`heuristic` is ``h(n)``, the sum of the assignments of the node's holes. It is a
      heuristic in the sense that it vanishes at a success node, where no hole is left and the
      empty sum is ``0``. It is admissible when every hole's assignment stays below the folds of
      all its fillers. The exact hole costs are a specification and not an algorithm, since
      inhabitation is undecidable in general, so a practical assignment bounds them from below by
      knowledge of the modeled domain.

    On a partial-term query the prescribed symbols are part of the partial inhabitant from the
    start, so they count into ``g`` before the search takes its first step.

    One point of vocabulary is worth naming. A cost algebra in general also knows *literal holes*,
    which await a literal symbol rather than a term. This engine's clauses fix their constant
    arguments at application, so a literal never appears as an open
    :class:`~cosy.search.partial.Hole`. Its cost enters through :attr:`symbol_cost` at the step
    that writes it, which by the additive split is the same total.

    Attributes:
        domain (CostDomain[A]): The cost domain, a positively ordered monoid.
        symbol_cost (Callable[[Any], A]): The cost ``c_F`` of a function symbol. The argument is a
            ``Tree`` root: a combinator for a clause's terminal, a literal value for a constant
            argument.
        hole_cost (Callable[[Hole[Any]], A]): The variable assignment ``h`` on the holes. The zero
            assignment by default, which turns A* into uniform-cost search.
    """

    def __init__(
        self,
        domain: CostDomain[A],
        symbol_cost: Callable[[Any], A],
        *,
        hole_cost: Callable[[Hole[Any]], A] | None = None,
    ) -> None:
        """Build the algebra.

        Args:
            domain (CostDomain[A]): The cost domain.
            symbol_cost (Callable[[Any], A]): The cost ``c_F`` of a function symbol.
            hole_cost (Callable[[Hole[Any]], A] | None): The variable assignment on the holes.
                None takes :func:`zero_assignment`. (Default value = None)
        """
        self.domain = domain
        self.symbol_cost = symbol_cost
        self.hole_cost = zero_assignment(domain) if hole_cost is None else hole_cost

    def _checked(self, value: A, what: str) -> A:
        """Reject a cost that does not belong to the domain.

        A negative symbol cost would break the positivity of the order, and with it the
        monotonicity of ``g`` along the branches, the admissibility argument and the completeness
        of uniform-cost search, while producing perfectly plausible numbers. It is refused where
        it enters instead.

        Args:
            value (A): The cost returned by an assignment.
            what (str): What produced it, for the message.

        Returns:
            A: The value, unchanged.

        Raises:
            ValueError: If the value is not an element of the cost domain.
        """
        if not self.domain.is_element(value):
            msg = f"{what} returned {value!r}, which is not an element of the cost domain"
            raise ValueError(msg)
        return value

    def cost_of_symbol(self, symbol: Any) -> A:
        """Return the cost ``c_F`` of a function symbol.

        Args:
            symbol (Any): The symbol, a ``Tree`` root.

        Returns:
            A: Its cost.

        Raises:
            ValueError: If the assignment leaves the cost domain.
        """
        return self._checked(self.symbol_cost(symbol), "the symbol cost")

    def cost_of_hole(self, hole: Hole[Any]) -> A:
        """Return the variable assignment ``h(v)`` of a hole.

        Args:
            hole (Hole[Any]): The hole.

        Returns:
            A: Its estimated cost.

        Raises:
            ValueError: If the assignment leaves the cost domain.
        """
        return self._checked(self.hole_cost(hole), "the variable assignment")

    def operation(self, symbol: Any, arguments: Sequence[A]) -> A:
        """Interpret one function symbol: ``[[F]](a_1..a_n) = c_F + a_1 + ... + a_n``.

        Args:
            symbol (Any): The function symbol ``F``.
            arguments (Sequence[A]): The costs of its arguments.

        Returns:
            A: The cost of the application.
        """
        return self.domain.add(self.cost_of_symbol(symbol), self.domain.sum_of(arguments))

    def term_assignment(self, term: Tree[Any]) -> A:
        """Return the term assignment of a term, by the algebra's recursion.

        The recursion evaluated bottom-up: a hole takes its variable assignment, an application
        takes :meth:`operation` of its symbol and its arguments' values. This is the definition.
        :meth:`symbol_cost_sum` and :meth:`hole_cost_sum` are the two halves the additive split
        says it equals, and they are computed independently of it so that the split is a testable
        statement rather than a restatement. The recursion and the two halves add the same costs in
        a different order, so over the float domains here they agree up to rounding.

        Args:
            term (Tree[Any]): The term. Its leaves may be :class:`~cosy.search.partial.Hole`
                markers.

        Returns:
            A: Its cost.

        Raises:
            ValueError: If an assignment leaves the cost domain.
        """
        order: list[Tree[Any]] = []
        pending = [term]
        while pending:
            node = pending.pop()
            order.append(node)
            pending.extend(node.children)
        values: dict[int, A] = {}
        for node in reversed(order):
            if isinstance(node.root, Hole):
                values[id(node)] = self.cost_of_hole(node.root)
            else:
                values[id(node)] = self.operation(node.root, [values[id(child)] for child in node.children])
        return values[id(term)]

    def symbol_cost_sum(self, term: Tree[Any]) -> A:
        """Return the first half of the additive split: the sum over the symbol occurrences.

        Args:
            term (Tree[Any]): The term.

        Returns:
            A: The sum of ``c_F`` over the occurrences of function symbols.

        Raises:
            ValueError: If the symbol cost leaves the cost domain.
        """
        return self.domain.sum_of(self.cost_of_symbol(symbol) for symbol in _symbol_occurrences(term).values())

    def hole_cost_sum(self, term: Tree[Any]) -> A:
        """Return the second half of the additive split: the sum over the hole occurrences.

        Args:
            term (Tree[Any]): The term.

        Returns:
            A: The sum of ``h(v)`` over the occurrences of variables.

        Raises:
            ValueError: If the variable assignment leaves the cost domain.
        """
        return self.domain.sum_of(self.cost_of_hole(hole) for hole in _hole_occurrences(term))

    def fold(self, term: Tree[Any]) -> A:
        """Return the fold of a ground term: its cost, independent of any variable assignment.

        On a term without holes the term assignment does not consult the variable assignment, so
        the fold is the cost of an inhabitant, which is what admissibility is measured against.

        Args:
            term (Tree[Any]): The ground term.

        Returns:
            A: Its cost.

        Raises:
            ValueError: If the term carries a hole. A hole is a variable, and a term with
                variables has a term assignment rather than a fold. Substituting the variable
                assignment here would hide from the caller that the term was not ground.
        """
        open_holes = _hole_occurrences(term)
        if open_holes:
            msg = (
                f"a fold is defined on ground terms, but this one carries the hole "
                f"{open_holes[0]!r}. Use term_assignment for a partial inhabitant"
            )
            raise ValueError(msg)
        return self.symbol_cost_sum(term)

    def cost_so_far(self, goal: Goal[Any, Any, Any]) -> A:
        """Return ``g(n)``: the cost of the symbols the partial inhabitant already carries.

        A cost-so-far function, and therefore one that never falls along a branch, since every
        step cost is positive.

        Args:
            goal (Goal[Any, Any, Any]): The search node.

        Returns:
            A: The sum of the symbol costs of its partial inhabitant.
        """
        return self.symbol_cost_sum(partial_inhabitant(goal))

    def heuristic(self, goal: Goal[Any, Any, Any]) -> A:
        """Return ``h(n)``: the sum of the variable assignments of the node's holes.

        A heuristic in the sense that it vanishes at a success node, where no hole remains and the
        empty sum is ``0``.

        Args:
            goal (Goal[Any, Any, Any]): The search node.

        Returns:
            A: The estimated remaining cost.
        """
        return self.hole_cost_sum(partial_inhabitant(goal))

    def cost_on_goals(self, goal: Goal[Any, Any, Any]) -> A:
        """Return the cost function on goals: the term assignment of the node's partial inhabitant.

        By the additive split this is ``g(n) + h(n)``, the cost function of A* search. It is
        computed here by the algebra's recursion instead, so that the identity is something the
        tests check. :func:`a_star` takes the other route and adds the two halves, so over the
        float domains here the two agree up to rounding.

        Args:
            goal (Goal[Any, Any, Any]): The search node.

        Returns:
            A: Its cost.
        """
        return self.term_assignment(partial_inhabitant(goal))

    def step_cost(self, parent: Goal[Any, Any, Any], child: Goal[Any, Any, Any]) -> A:
        """Return the step cost of one expansion.

        Filling a hole with a function symbol ``F`` of any arity costs ``c_F``. The value is read
        off the symbol occurrences the child added rather than as a difference of the two ``g``
        values, since a cost domain is a monoid and need not have subtraction. A pair that adds no
        symbol therefore costs ``0``, which is the neutral element and not a fallback. Every
        expansion of the engine fixes at least one function-symbol occurrence, so on an edge of a
        derivation tree the step cost is the cost of the symbols that expansion wrote.

        Args:
            parent (Goal[Any, Any, Any]): The node ``n``.
            child (Goal[Any, Any, Any]): A child ``n'`` of it.

        Returns:
            A: The step cost, so that ``g(n') = g(n) + delta(n, n')``. The two sides add the same
                symbol costs in a different order, so over the float domains here they agree up to
                rounding.

        Raises:
            ValueError: If the child does not extend the parent's partial inhabitant. A symbol of
                the parent missing or replaced in the child means the two are not an edge of the
                derivation tree, and no step cost relates them. The check is necessary rather than
                sufficient: it catches a swapped or unrelated pair, not a node from a different
                branch that happens to extend the same prefix.
        """
        before = _symbol_occurrences(partial_inhabitant(parent))
        after = _symbol_occurrences(partial_inhabitant(child))
        for position, symbol in before.items():
            if position not in after or after[position] != symbol:
                msg = (
                    f"the second goal is not a child of the first: the symbol at position "
                    f"{position} of the parent's partial inhabitant is missing or replaced"
                )
                raise ValueError(msg)
        return self.domain.sum_of(
            self.cost_of_symbol(symbol) for position, symbol in after.items() if position not in before
        )


def uniform_cost(algebra: AdditiveCostAlgebra[A]) -> CostFunction[Any, A]:
    """Build the cost function of uniform-cost search.

    ``f(n) = g(n)``, which is A* search with the zero heuristic. Zero is a heuristic and it is
    consistent, since ``0 <= delta(n, n') + 0``. The algebra's own variable assignment plays no
    part here, because uniform-cost search fixes ``h(n) = 0``, so an algebra carrying a nonzero
    assignment yields the same uniform-cost search as one carrying none.

    ``g`` never falls along a branch, so the stream is a linear extension of the cost order.
    Completeness needs finite cost-bounded sets, which an archimedean domain and strictly positive
    combinator costs supply, and :func:`assert_uniform_cost_complete` checks that hypothesis.

    Args:
        algebra (AdditiveCostAlgebra[A]): The additive cost algebra.

    Returns:
        CostFunction[Any, A]: The cost function on goals.
    """
    return CostFunction(algebra.domain, algebra.cost_so_far)


def greedy(algebra: AdditiveCostAlgebra[A]) -> CostFunction[Any, A]:
    """Build the cost function of greedy best-first search.

    ``f = h``, the heuristic alone, so the search expands the node that looks closest to a success
    node and ignores what has been paid to get there.

    **Sound, and in general nothing more.** Completion costs fall toward a success node and their
    estimate falls with them, so a heuristic need not be monotone along the branches. Greedy
    search therefore need not stream in order of cost, and the cost-bounded sets of ``h`` may be
    infinite, so greedy search may follow a misleading estimate down an infinite branch. On a
    finite derivation tree every cost-bounded set is finite and greedy search is complete.

    Args:
        algebra (AdditiveCostAlgebra[A]): The additive cost algebra. Its variable assignment is
            the heuristic.

    Returns:
        CostFunction[Any, A]: The cost function on goals.
    """
    return CostFunction(algebra.domain, algebra.heuristic)


def a_star(algebra: AdditiveCostAlgebra[A]) -> CostFunction[Any, A]:
    """Build the cost function of A* search.

    ``f(n) = g(n) + h(n)``, pointwise, since ``+`` is defined on the cost domain and not on
    functions. Both summands live in the same domain, which the algebra guarantees by
    construction.

    With a *consistent* heuristic ``f`` never falls along a branch even where ``h`` alone does, and
    then the stream is ordered by cost. A consistent heuristic is admissible. Admissibility alone
    secures less, but it still places a cheapest inhabitant first. Completeness carries over from
    uniform-cost search, since ``h(n) >= 0`` puts the cost-bounded sets of ``f`` inside those of
    ``g``. Over the float domains here ``g`` and ``h`` are rounded before they are added, so ``f``
    can come out lower at a child than at its parent even where the heuristic is consistent.

    Args:
        algebra (AdditiveCostAlgebra[A]): The additive cost algebra.

    Returns:
        CostFunction[Any, A]: The cost function on goals.
    """

    def evaluate(goal: Goal[Any, Any, Any]) -> A:
        """Return ``g(n) + h(n)`` for one search node.

        The partial inhabitant is materialized once and both sums are read off it. Calling
        ``cost_so_far`` and ``heuristic`` in turn would build it twice, once per pushed node.

        Args:
            goal (Goal[Any, Any, Any]): The search node.

        Returns:
            A: Its cost.
        """
        term = partial_inhabitant(goal)
        return algebra.domain.add(algebra.symbol_cost_sum(term), algebra.hole_cost_sum(term))

    return CostFunction(algebra.domain, evaluate)


def assert_uniform_cost_complete(algebra: AdditiveCostAlgebra[A], symbols: Iterable[Any]) -> None:
    """Check the hypotheses under which uniform-cost search is complete.

    Two conditions, and between them they make every cost-bounded set of ``g`` finite: the cost
    domain is archimedean, and every combinator has strictly positive cost. The argument is a
    pigeonhole one. A path that builds enough combinators repeats one of the finitely many
    combinator costs often enough for its iterated sum to pass any bound, so every node of a
    cost-bounded set ends a path of bounded length, and the program is finite. It needs nothing of
    the program beyond a cost per combinator, which is what makes it the broadest class of cost
    functions on offer here.

    A* search inherits the conclusion under the same hypotheses, since ``h(n) >= 0`` puts the
    cost-bounded sets of ``f`` inside those of ``g``.

    The check is a validation tool and not something a search runs. The conditions are properties
    of the algebra, and a caller who wants the guarantee states the combinators once. An empty
    family states none of them, and the guarantee would then rest on nothing, so it is refused.

    Args:
        algebra (AdditiveCostAlgebra[A]): The additive cost algebra.
        symbols (Iterable[Any]): The function symbols of the program, the combinators whose costs
            the argument rests on. The family is read into a tuple first, so an iterator that an
            earlier call has consumed arrives as the empty family and is refused.

    Raises:
        ValueError: If the cost domain is not archimedean, or if some symbol has a cost that is
            not strictly positive. Either failure leaves the cost-bounded sets possibly infinite,
            and reporting completeness on those grounds would be a claim the argument does not
            support. Also if the symbol family is empty, since then nothing was checked.
    """
    if not algebra.domain.is_archimedean:
        msg = (
            "uniform-cost search is complete over an archimedean cost domain, and this one is "
            "not: an iterated sum need never pass a bound, so a cost-bounded set may be infinite"
        )
        raise ValueError(msg)
    family = tuple(symbols)
    if not family:
        msg = (
            "uniform-cost search is complete only if every combinator has a strictly positive "
            "cost, and an empty symbol family leaves that unchecked. Pass the combinators of the "
            "program. An iterator that has already been read arrives here empty."
        )
        raise ValueError(msg)
    for symbol in family:
        cost = algebra.cost_of_symbol(symbol)
        if not algebra.domain.is_strictly_positive(cost):
            msg = (
                f"uniform-cost search is complete only if every combinator has a strictly "
                f"positive cost, but {symbol!r} costs {cost!r}"
            )
            raise ValueError(msg)
