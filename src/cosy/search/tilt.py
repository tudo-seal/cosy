"""The exponential tilt: random search in proportion to ``e^(-theta c(t))`` for an additive cost, without counts.

Random search draws a term in proportion to a weight, and the table forms get a node's weight from
the counts of its completions per cost value (:mod:`cosy.search.cost_tables`), one row per
non-terminal. A cost that realizes many values makes those rows long. The *exponential tilt* weighs a
term by ``e^(-theta c(t))`` for a real ``theta`` instead, and that weight needs no row at all.

**Why one number per non-terminal suffices.** The cost of a term below a search node is the cost of
the node's partial inhabitant plus the costs of the fillers of its holes, and the completions below
the node are the tuples of independent completions of its holes (the additive split read along a
success branch, and the decomposition over the holes). So the tilted mass below a node factors,

    sum over the terms t below n of e^(-theta c(t))  =  e^(-theta g(n)) * prod over the holes B of Z_B(theta),

with ``g(n)`` the cost so far and ``Z_A(theta)`` the sum of the tilted weights of the terms rooted at
``A``. The ``Z`` obey the recursion of the counts with the cost in the exponent,

    Z_A = sum over clauses A <- F(B_1..B_k) of  e^(-theta c(F)) * Z_{B_1} * .. * Z_{B_k},

``c(F)`` the cost a clause adds (:func:`~cosy.search.counting.rule_cost`). A term's weight is then the
same for every term of one cost value, so within a cost value the draw is uniform, exactly, and the
marginal on the cost is ``N(a) e^(-theta a) / Z``: the counts, tilted. ``theta = 0`` draws uniformly
from all terms, a positive ``theta`` leans towards cheap terms and a negative one towards dear ones.

**Finite languages.** Random search streams a sample without replacement in proportion to the weights
for a finite set of terms. A non-terminal on a cycle of non-terminals that all have terms has
infinitely many, and whether their tilted weights even sum to something finite depends on ``theta``;
this module covers finite languages and refuses the rest, naming the cycle. The cycle is looked for in
the whole program, as the cost table looks for its loops, so a cycle no query reaches refuses the
program all the same. What cannot be completed is pruned first, so a cycle through a non-terminal
without a term is no cycle. A finite language takes any real ``theta``, negative ones included, and
every finite real cost its algebra's domain admits: fractions, and negative costs under a domain that
has them. Every count and every sample statement here is about derivations, which are the terms
under unambiguity; an ambiguous program counts a term once per derivation, and the tilted search
streams it as often (the mixture below, which skips what it has streamed, once).

**The tilted mean.** The same recursion, differentiated, gives the tilted mean cost of the terms rooted
at each non-terminal, ``-d log Z_A / d theta``: the mean of a clause's cost plus the means of its holes,
weighted by the clause's share of ``Z_A``. Its derivative is minus the tilted variance, so it never
increases with ``theta``, and it falls strictly unless every term costs the same; it runs from the
dearest term's cost at ``-inf`` to the cheapest's at ``+inf``. That is what finding a ``theta`` for a
target mean needs, together with the cheapest and the dearest cost, which the same recursion gives
with the minimum and the maximum in the place of the sum.

**Numerics.** The weights are kept in log space, so no count and no ``Z`` overflows. What floating point
limits is ``theta`` times a cost: the log weights carry an absolute error of about ``|theta| c``
times the machine epsilon, which is a relative error in the weights, and ``theta`` so large that
``-theta c`` leaves the range of floating point is refused by name. Within one cost value the draw is
uniform exactly in exact arithmetic, and up to that error in floating point.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Generic

from cosy.core.solution_space import NT, ConstantArgument, G, Goal, NonTerminalArgument, T
from cosy.search.cost_tables import _components, _named
from cosy.search.counting import _admitted, decomposable_or_raise
from cosy.search.partial import Hole, holes, partial_inhabitant
from cosy.search.rules import deepest_first_subgoal
from cosy.search.sampling import _built, keyed_stream, log_sum_exp

if TYPE_CHECKING:
    import random
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from cosy.core.solution_space import RHSRule, SolutionSpace
    from cosy.core.tree import Path, Tree
    from cosy.search.costs import AdditiveCostAlgebra
    from cosy.search.queries import ResolutionQuery

__all__ = [
    "TiltProgram",
    "TiltTable",
    "TiltedMixture",
    "TiltedSearch",
    "theta_for_mean",
    "tilt_program",
    "tilt_table",
    "tilted_mixture",
    "tilted_search",
]


def _real_cost(value: Any, what: str) -> float:
    """Return a cost as a finite real number, or refuse it.

    Args:
        value (Any): The cost, an element of the algebra's domain.
        what (str): What produced it, for the message.

    Returns:
        float: The cost.

    Raises:
        ValueError: If the cost is not a finite real number: a tuple of costs has no single
            exponential weight, and an infinite or undefined cost none at all.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        msg = f"the exponential tilt takes finite real costs, but {what} is {value!r}"
        raise ValueError(msg)
    return float(value)


def _real_rule_cost(rule: RHSRule[Any, Any, Any], algebra: AdditiveCostAlgebra[Any]) -> float:
    """Return :func:`~cosy.search.counting.rule_cost` as a real number, each symbol checked.

    Args:
        rule (RHSRule): The clause.
        algebra (AdditiveCostAlgebra[Any]): The algebra.

    Returns:
        float: The cost of the terminal plus the cost of each constant argument.
    """
    total = _real_cost(algebra.cost_of_symbol(rule.terminal), f"the cost of the symbol {_named(rule.terminal)}")
    for argument in rule.arguments:
        if isinstance(argument, ConstantArgument):
            total += _real_cost(
                algebra.cost_of_symbol(argument.value), f"the cost of the symbol {_named(argument.value)}"
            )
    return total


def _real_term_cost(term: Tree[Any], algebra: AdditiveCostAlgebra[Any]) -> float:
    """Return the cost of the symbols a (partial) term carries, as a real number; holes cost nothing.

    Args:
        term (Tree[Any]): The term; its leaves may be holes.
        algebra (AdditiveCostAlgebra[Any]): The algebra.

    Returns:
        float: The sum of its symbols' costs.
    """
    total = 0.0
    pending = [term]
    while pending:
        node = pending.pop()
        if not isinstance(node.root, Hole):
            total += _real_cost(algebra.cost_of_symbol(node.root), f"the cost of the symbol {_named(node.root)}")
        pending.extend(node.children)
    return total


def _real_theta(theta: Any) -> float:
    """Return the tilt as a finite real number, or refuse it.

    Args:
        theta (Any): The tilt a caller passed.

    Returns:
        float: The tilt.

    Raises:
        ValueError: If it is not a finite real number.
    """
    if isinstance(theta, bool) or not isinstance(theta, (int, float)) or not math.isfinite(theta):
        msg = f"theta must be a finite real number, not {theta!r}"
        raise ValueError(msg)
    return float(theta)


# A clause as the tilt reads it: its own cost and the non-terminals of its holes.
_Clause = tuple[float, tuple[Any, ...]]


@dataclass(frozen=True)
class TiltTable(Generic[NT]):
    """``log Z_A(theta)`` and the tilted mean cost per non-terminal of a program with a finite language.

    Attributes:
        space (SolutionSpace[NT, Any, Any]): The program the table was computed for.
        algebra (AdditiveCostAlgebra[Any]): The algebra whose costs are tilted.
        theta (float): The tilt.
        log_z (Mapping[NT, float]): Per non-terminal of the program, ``log Z_A(theta)``, the log of the
            sum of ``e^(-theta c(t))`` over the terms rooted at it; ``-inf`` for one without a term.
        mean_cost (Mapping[NT, float]): Per non-terminal with a term, the mean cost of its terms under
            the tilt, ``-d log Z_A / d theta``.
        counts (Mapping[NT, int]): Per non-terminal of the program, the exact number of its derivations, its
            terms under unambiguity, 0 for one without; ``Z_A(0)``, kept in whole numbers because a
            number of terms is asked exactly.
        cheapest (Mapping[NT, float]): Per non-terminal with a term, the least cost of its terms.
        dearest (Mapping[NT, float]): Per non-terminal with a term, the greatest cost of its terms.
        log_excess (Mapping[NT, float]): Per non-terminal with a term, ``log Z_A(theta) + theta m_A``, ``m_A``
            its cheapest cost: the tilt of the costs in excess of the cheapest, which is what the table
            computes, so that a large cost every term shares costs no precision.
        mean_excess (Mapping[NT, float]): Per non-terminal with a term, the tilted mean cost less ``m_A``.
    """

    space: SolutionSpace[NT, Any, Any]
    algebra: AdditiveCostAlgebra[Any]
    theta: float
    log_z: Mapping[NT, float]
    mean_cost: Mapping[NT, float]
    counts: Mapping[NT, int]
    cheapest: Mapping[NT, float]
    dearest: Mapping[NT, float]
    log_excess: Mapping[NT, float]
    mean_excess: Mapping[NT, float]

    def of(self, nonterminal: NT) -> float:
        """Return ``log Z_A(theta)``.

        Args:
            nonterminal (NT): The non-terminal ``A``.

        Returns:
            float: The log of its tilted mass, ``-inf`` for one without a term or unknown to the program.
        """
        return self.log_z.get(nonterminal, -math.inf)


def _productive(clauses: Mapping[NT, list[_Clause]]) -> set[NT]:
    """Return the non-terminals with at least one term: the least fixed point over the clauses.

    A clause completes once each of its holes can be completed, so every clause counts the holes it
    still waits for, and a non-terminal found complete releases the clauses waiting for it.

    Args:
        clauses (Mapping[NT, list[_Clause]]): The admitted clauses per non-terminal.

    Returns:
        set[NT]: The non-terminals with a term.
    """
    waiting: dict[tuple[NT, int], int] = {}
    users: dict[NT, list[tuple[NT, int]]] = {}
    ready: list[NT] = []
    for head, head_clauses in clauses.items():
        for index, (_cost, hole_types) in enumerate(head_clauses):
            waiting[head, index] = len(hole_types)
            for hole in hole_types:
                users.setdefault(hole, []).append((head, index))
            if not hole_types:
                ready.append(head)
    productive: set[NT] = set()
    while ready:
        nonterminal = ready.pop()
        if nonterminal in productive:
            continue
        productive.add(nonterminal)
        for head, index in users.get(nonterminal, ()):
            waiting[head, index] -= 1
            if waiting[head, index] == 0 and head not in productive:
                ready.append(head)
    return productive


@dataclass(frozen=True)
class TiltProgram(Generic[NT]):
    """A program prepared for the tilt: what does not depend on ``theta``, computed once.

    The clauses with their real costs and the non-terminals of their holes, pruned to the non-terminals
    that have a term, in an order in which every hole comes before the clauses that read it, and the
    exact number of terms of every non-terminal. A tilt table for one ``theta`` is then one pass over the
    clauses, which is what a search for ``theta`` or a mixture of several tilts needs.

    Attributes:
        space (SolutionSpace[NT, Any, Any]): The program.
        algebra (AdditiveCostAlgebra[Any]): The algebra whose costs are tilted.
        nonterminals (tuple[NT, ...]): Every non-terminal of the program.
        order (tuple[NT, ...]): The non-terminals with a term, every one after the holes of its clauses.
        clauses (Mapping[NT, tuple[tuple[float, tuple[NT, ...]], ...]]): Per non-terminal with a term, its
            admitted clauses whose holes all have terms: the clause's cost and its holes' non-terminals.
        counts (Mapping[NT, int]): Per non-terminal of the program, the exact number of its derivations, its
            terms under unambiguity, 0 for one without; ``Z_A(0)``, kept in whole numbers because a
            number of terms is asked exactly.
        cheapest (Mapping[NT, float]): Per non-terminal with a term, the least cost of its terms.
        dearest (Mapping[NT, float]): Per non-terminal with a term, the greatest cost of its terms.
    """

    space: SolutionSpace[NT, Any, Any]
    algebra: AdditiveCostAlgebra[Any]
    nonterminals: tuple[NT, ...]
    order: tuple[NT, ...]
    clauses: Mapping[NT, tuple[tuple[float, tuple[NT, ...]], ...]]
    counts: Mapping[NT, int]
    cheapest: Mapping[NT, float]
    dearest: Mapping[NT, float]

    def table(self, theta: float) -> TiltTable[NT]:
        """Compute ``log Z_A(theta)`` and the tilted mean cost for every non-terminal, in one pass.

        Args:
            theta (float): The tilt, any finite real number.

        Returns:
            TiltTable[NT]: The table.

        Raises:
            ValueError: If ``theta`` is not a finite real number, or so large in magnitude that ``-theta c``
                leaves the range of floating point for the costs of this program.
        """
        theta = _real_theta(theta)
        cheapest = self.cheapest
        log_z: dict[NT, float] = dict.fromkeys(self.nonterminals, -math.inf)
        mean_cost: dict[NT, float] = {}
        log_excess: dict[NT, float] = {}
        mean_excess: dict[NT, float] = {}
        for member in self.order:
            member_clauses = self.clauses[member]
            base = cheapest[member]
            # Each clause's cost in excess of the member's cheapest term, its holes at their cheapest: small
            # where every term shares a large cost, so that the weights and the mean keep their precision.
            excesses = [cost + sum(cheapest[hole] for hole in hole_types) - base for cost, hole_types in member_clauses]
            exponents = [
                -theta * excess + sum(log_excess[hole] for hole in hole_types)
                for excess, (_cost, hole_types) in zip(excesses, member_clauses, strict=True)
            ]
            total = log_sum_exp(exponents)
            mean = sum(
                math.exp(exponent - total) * (excess + sum(mean_excess[hole] for hole in hole_types))
                for exponent, excess, (_cost, hole_types) in zip(exponents, excesses, member_clauses, strict=True)
            )
            log_mass = -theta * base + total
            if not (math.isfinite(total) and math.isfinite(mean) and math.isfinite(log_mass)):
                msg = (
                    f"theta {theta} is too large in magnitude for the costs of this program: the tilted weights of "
                    f"{member} leave the range of floating point"
                )
                raise ValueError(msg)
            log_excess[member] = total
            mean_excess[member] = mean
            log_z[member] = log_mass
            mean_cost[member] = base + mean
        return TiltTable(
            space=self.space,
            algebra=self.algebra,
            theta=theta,
            log_z=log_z,
            mean_cost=mean_cost,
            counts=self.counts,
            cheapest=cheapest,
            dearest=self.dearest,
            log_excess=log_excess,
            mean_excess=mean_excess,
        )


def tilt_program(
    space: SolutionSpace[NT, Any, Any], algebra: AdditiveCostAlgebra[Any], *, check: bool = True
) -> TiltProgram[NT]:
    """Prepare a program for the tilt: its clauses' costs and holes, pruned, ordered, and its terms counted.

    Args:
        space (SolutionSpace[NT, Any, Any]): The synthesized program; its language must be finite.
        algebra (AdditiveCostAlgebra[Any]): The additive cost algebra; its symbol costs must be finite
            real numbers.
        check (bool): Whether to decide the decomposition hypothesis first. Leaving it on is the
            supported use. (Default value = True)

    Returns:
        TiltProgram[NT]: The prepared program.

    Raises:
        ValueError: If a symbol's cost is not a finite real number; under ``check``, if a predicate of the
            program reads a hole; or if non-terminals with terms form a cycle, whose language is infinite,
            in which case the message names them. The cycle is looked for in the whole program, so a
            cycle no query reaches refuses it all the same.
    """
    if check:
        decomposable_or_raise(space)

    nonterminals = list(space.nonterminals())
    known = set(nonterminals)
    clauses: dict[NT, list[_Clause]] = {}
    for nonterminal in nonterminals:
        kept: list[_Clause] = []
        for rule in space.get(nonterminal) or ():
            if not _admitted(rule):
                continue
            hole_types = tuple(
                argument.origin for argument in rule.arguments if isinstance(argument, NonTerminalArgument)
            )
            if any(hole not in known for hole in hole_types):
                continue
            kept.append((_real_rule_cost(rule, algebra), hole_types))
        clauses[nonterminal] = kept

    # What cannot be completed weighs nothing, and a cycle through it never closes, so it goes first.
    productive = _productive(clauses)
    live = {
        nonterminal: tuple(clause for clause in clauses[nonterminal] if all(hole in productive for hole in clause[1]))
        for nonterminal in nonterminals
        if nonterminal in productive
    }
    candidates = [nonterminal for nonterminal in nonterminals if nonterminal in productive]
    successors = {
        nonterminal: list(dict.fromkeys(hole for _cost, hole_types in live[nonterminal] for hole in hole_types))
        for nonterminal in candidates
    }
    order: list[NT] = []
    counts: dict[NT, int] = dict.fromkeys(nonterminals, 0)
    cheapest: dict[NT, float] = {}
    dearest: dict[NT, float] = {}
    for component in _components(candidates, successors):
        member = component[0]
        if len(component) > 1 or member in successors[member]:
            names = ", ".join(sorted(str(nonterminal) for nonterminal in component))
            msg = (
                f"the language has infinitely many terms: {names} lie on a cycle and each has a term; the "
                f"exponential tilt covers finite languages"
            )
            raise ValueError(msg)
        order.append(member)
        counts[member] = sum(math.prod(counts[hole] for hole in hole_types) for _cost, hole_types in live[member])
        cheapest[member] = min(cost + sum(cheapest[hole] for hole in hole_types) for cost, hole_types in live[member])
        dearest[member] = max(cost + sum(dearest[hole] for hole in hole_types) for cost, hole_types in live[member])
    return TiltProgram(
        space=space,
        algebra=algebra,
        nonterminals=tuple(nonterminals),
        order=tuple(order),
        clauses=live,
        counts=counts,
        cheapest=cheapest,
        dearest=dearest,
    )


def tilt_table(
    space: SolutionSpace[NT, Any, Any], algebra: AdditiveCostAlgebra[Any], theta: float, *, check: bool = True
) -> TiltTable[NT]:
    """Compute ``log Z_A(theta)`` and the tilted mean cost for every non-terminal of a program.

    :func:`tilt_program` followed by :meth:`TiltProgram.table`; a caller with several ``theta`` prepares
    the program once and asks it for each table.

    Args:
        space (SolutionSpace[NT, Any, Any]): The synthesized program; its language must be finite.
        algebra (AdditiveCostAlgebra[Any]): The additive cost algebra; its symbol costs must be finite
            real numbers.
        theta (float): The tilt, any finite real number.
        check (bool): Whether to decide the decomposition hypothesis first. Leaving it on is the
            supported use. (Default value = True)

    Returns:
        TiltTable[NT]: The table.

    Raises:
        ValueError: If ``theta`` or a symbol's cost is not a finite real number; under ``check``, if a
            predicate of the program reads a hole; or if non-terminals with terms form a cycle, whose
            language is infinite, in which case the message names them.
    """
    theta = _real_theta(theta)
    return tilt_program(space, algebra, check=check).table(theta)


# A node of the tilted search's lazy frontier: its goal (None at the root, and while it is unbuilt), the
# cost of its partial inhabitant, and, while unbuilt, the parent, position and clause that build it.
_TiltNode = tuple[Any, float, Any]


@dataclass(frozen=True)
class TiltedSearch(Generic[NT, T, G]):
    """Random search in proportion to ``e^(-theta c(t))``, each node weighed from a tilt table.

    A node's weight is the tilted mass of the terms below it, ``e^(-theta g(n))`` times the ``Z`` of its
    holes, divided by the query's total; so every term is weighed ``e^(-theta c(t)) / Z``, the terms of
    one cost value alike, and under unambiguity the stream is a sample without replacement in that
    proportion. The frontier is lazy, as the table forms' is: a child is pushed as its parent, position
    and clause and built when popped.

    Attributes:
        query (ResolutionQuery[NT, T, G]): The query being sampled from.
        table (TiltTable[NT]): ``log Z_A(theta)`` over the program.
        root_log_mass (float): The log of the query's total tilted mass, ``-inf`` without a term.
        subgoal_selection (Callable | None): The computation rule; None selects the engine's
            deepest-first rule. It must select an open hole, as the engine's rule does: a node is
            weighed by its open holes, and an expanded position that ``Goal.subgoals`` still lists
            is not one.
    """

    query: ResolutionQuery[NT, T, G]
    table: TiltTable[NT]
    root_log_mass: float
    subgoal_selection: Callable[[Goal[NT, T, G]], tuple[Path, NonTerminalArgument[NT]]] | None = None

    @property
    def total(self) -> int:
        """Return the exact number of the query's derivations, its terms under unambiguity.

        Returns:
            int: The number of derivations below the query's initial nodes.
        """
        known = self.__dict__.get("_total")
        if known is None:
            known = sum(
                math.prod(self.table.counts.get(hole, 0) for hole in holes(goal).values())
                for goal, _cost in _initial_tilt_nodes(self.query, self.table.algebra)
            )
            object.__setattr__(self, "_total", known)
        return known

    def rule_cost_of(self, rule: RHSRule[NT, T, G]) -> float:
        """Return the real cost a clause adds, computed once per clause.

        Args:
            rule (RHSRule[NT, T, G]): The clause.

        Returns:
            float: Its cost.
        """
        costs = self.__dict__.get("_rule_costs")
        if costs is None:
            costs = {}
            object.__setattr__(self, "_rule_costs", costs)
        cost = costs.get(id(rule))
        if cost is None:
            cost = _real_rule_cost(rule, self.table.algebra)
            costs[id(rule)] = cost
        return cost

    @property
    def mean_cost(self) -> float:
        """Return the tilted mean cost of the query's terms, ``-d log Z / d theta`` of the query's mass.

        Each initial node contributes its share of the query's mass times its cost so far plus the tilted
        means of its holes, which is the recursion of the tilt table read at the query's root.

        Returns:
            float: The mean, ``nan`` for a query without a term.
        """
        return _query_mean(_initial_holes(self.query, self.table.algebra), self.table)

    def _root_children(self) -> list[tuple[_TiltNode, float]]:
        """Return the query's initial nodes with their ``log w``, computed once for every stream of this search.

        A goal is never changed once built, so the children of the root can be shared by every stream:
        a fresh stream per independent draw does not rebuild them, though it still draws a key for each.

        Returns:
            list: The retained initial nodes with their ``log w``, in clause order.
        """
        known = self.__dict__.get("_root_children_cache")
        if known is None:
            known = []
            for child, child_cost in _initial_tilt_nodes(self.query, self.table.algebra):
                log_weight = self.log_mass_of(tuple(holes(child).values()), child_cost) - self.root_log_mass
                if log_weight > -math.inf:
                    known.append(((child, child_cost, None), log_weight))
            object.__setattr__(self, "_root_children_cache", known)
        return known

    def log_mass_of(self, hole_types: Sequence[NT], cost: float) -> float:
        """Return the log of the tilted mass below a node: ``-theta g(n) + sum of log Z`` over its holes.

        Args:
            hole_types (Sequence[NT]): The non-terminals of the node's holes.
            cost (float): The cost of its partial inhabitant.

        Returns:
            float: The log of the mass, ``-inf`` where a hole has no term.
        """
        return -self.table.theta * cost + sum(self.table.of(hole) for hole in hole_types)

    def stream(self, rng: random.Random) -> Iterator[Tree[T]]:
        """Draw one stream: the query's terms, each once, in decreasing key order.

        Args:
            rng (random.Random): The source of randomness.

        Yields:
            Tree[T]: The terms.
        """
        for _, inhabitant in self.keyed_stream(rng):
            yield inhabitant

    def keyed_stream(self, rng: random.Random) -> Iterator[tuple[float, Tree[T]]]:
        """Draw one stream, keeping the key each term was streamed under.

        Args:
            rng (random.Random): The source of randomness.

        Yields:
            tuple[float, Tree[T]]: The key and the term, in decreasing key order.
        """
        if self.root_log_mass == -math.inf:
            return
        select = deepest_first_subgoal if self.subgoal_selection is None else self.subgoal_selection
        space = self.query.solution_space
        root_log_mass = self.root_log_mass

        def expand(node: _TiltNode) -> tuple[Tree[T] | None, Sequence[tuple[_TiltNode, float]]]:
            """Expand one node, and weigh its children from the tilt table without building them.

            A child's holes are the parent's open holes without the expanded one, plus the clause's
            non-terminal arguments, and its cost the parent's plus the clause's; its goal is built
            when it is popped.

            Args:
                node (_TiltNode): The goal (None at the root or while unbuilt), the cost of its partial
                    inhabitant, and, while unbuilt, the parent, position and clause that build it.

            Returns:
                tuple: The term (None on an inner node) and the retained children with their ``log w``,
                    in clause order.

            Raises:
                ValueError: If the engine refuses a clause the frontier had weighed, or if the
                    computation rule selects a position that is not an open hole.
            """
            goal, cost, pending = node
            if pending is not None:
                goal = _built(pending)
            if goal is not None and goal.success:
                return goal.grounded[()][1], ()
            kept: list[tuple[_TiltNode, float]] = []
            if goal is None:
                return None, self._root_children()
            position, argument = select(goal)
            open_holes = holes(goal)
            if position not in open_holes:
                msg = (
                    f"the computation rule selected position {position}, which is not an open hole: the tilted "
                    "search weighs a node by its open holes, so the rule must select one, as deepest_first_subgoal does"
                )
                raise ValueError(msg)
            others = [hole_type for hole_position, hole_type in open_holes.items() if hole_position != position]
            for rule in space.get(argument.origin) or ():
                if not _admitted(rule):
                    continue
                child_cost = cost + self.rule_cost_of(rule)
                opened = [arg.origin for arg in rule.arguments if isinstance(arg, NonTerminalArgument)]
                log_weight = self.log_mass_of(others + opened, child_cost) - root_log_mass
                if log_weight > -math.inf:
                    kept.append(((None, child_cost, (goal, position, rule)), log_weight))
            return None, kept

        yield from keyed_stream((None, 0.0, None), 0.0, expand, rng)


def _initial_tilt_nodes(
    query: ResolutionQuery[NT, T, G], algebra: AdditiveCostAlgebra[Any]
) -> list[tuple[Goal[NT, T, G], float]]:
    """Return the children of the query's root with the real cost of their partial inhabitants.

    A partial-term query starts from goals that already carry the prescribed symbols, and those are charged.

    Args:
        query (ResolutionQuery[NT, T, G]): The query.
        algebra (AdditiveCostAlgebra[Any]): The algebra.

    Returns:
        list[tuple[Goal[NT, T, G], float]]: The initial goals with their costs.
    """
    space = query.solution_space
    tree, pos = query.tree, query.pos
    if tree is not None and pos is not None:
        goals = space.goal_from_tree(query.start, tree, pos)
        return [(goal, _real_term_cost(partial_inhabitant(goal), algebra)) for goal in goals]
    initial: list[tuple[Goal[NT, T, G], float]] = []
    for rule in space.get(query.start) or ():
        goal = Goal.from_rhs_rule(rule)
        if goal is not None:
            initial.append((goal, _real_rule_cost(rule, algebra)))
    return initial


def _initial_holes(
    query: ResolutionQuery[NT, T, G], algebra: AdditiveCostAlgebra[Any]
) -> list[tuple[float, tuple[NT, ...]]]:
    """Return the query's initial nodes as the tilt reads them: their cost so far and the non-terminals of their holes.

    Args:
        query (ResolutionQuery[NT, T, G]): The query.
        algebra (AdditiveCostAlgebra[Any]): The algebra.

    Returns:
        list[tuple[float, tuple[NT, ...]]]: One pair per initial node.
    """
    return [(cost, tuple(holes(goal).values())) for goal, cost in _initial_tilt_nodes(query, algebra)]


def _query_mean(initial: Sequence[tuple[float, tuple[NT, ...]]], table: TiltTable[NT]) -> float:
    """Return the tilted mean cost of the terms below a query's initial nodes, in excess of the cheapest of them.

    Args:
        initial (Sequence[tuple[float, tuple[NT, ...]]]): The initial nodes' costs so far and holes.
        table (TiltTable[NT]): The tilt table.

    Returns:
        float: The mean, ``nan`` without a term.
    """
    live = [(cost, hole_types) for cost, hole_types in initial if all(hole in table.cheapest for hole in hole_types)]
    if not live:
        return math.nan
    bases = [cost + sum(table.cheapest[hole] for hole in hole_types) for cost, hole_types in live]
    base = min(bases)
    exponents = [
        -table.theta * (least - base) + sum(table.log_excess[hole] for hole in hole_types)
        for least, (_cost, hole_types) in zip(bases, live, strict=True)
    ]
    total = log_sum_exp(exponents)
    return base + sum(
        math.exp(exponent - total) * (least - base + sum(table.mean_excess[hole] for hole in hole_types))
        for exponent, least, (_cost, hole_types) in zip(exponents, bases, live, strict=True)
    )


def _query_cost_range(
    initial: Sequence[tuple[float, tuple[NT, ...]]], cheapest: Mapping[NT, float], dearest: Mapping[NT, float]
) -> tuple[float, float]:
    """Return the least and the greatest cost of the terms below a query's initial nodes.

    Args:
        initial (Sequence[tuple[float, tuple[NT, ...]]]): The initial nodes' costs so far and holes.
        cheapest (Mapping[NT, float]): The program's cheapest cost per non-terminal with a term.
        dearest (Mapping[NT, float]): The program's dearest cost per non-terminal with a term.

    Returns:
        tuple[float, float]: The cheapest and the dearest cost, ``(inf, -inf)`` for a query without a term.
    """
    least, greatest = math.inf, -math.inf
    for cost, hole_types in initial:
        if all(hole in cheapest for hole in hole_types):
            least = min(least, cost + sum(cheapest[hole] for hole in hole_types))
            greatest = max(greatest, cost + sum(dearest[hole] for hole in hole_types))
    return least, greatest


def tilted_search(
    query: ResolutionQuery[NT, T, G],
    algebra: AdditiveCostAlgebra[Any],
    theta: float,
    *,
    subgoal_selection: Callable[[Goal[NT, T, G]], tuple[Path, NonTerminalArgument[NT]]] | None = None,
    table: TiltTable[NT] | None = None,
) -> TiltedSearch[NT, T, G]:
    """Build random search in proportion to ``e^(-theta c(t))`` for a query of a program whose language is finite.

    Finite as a whole program, not only in the part the query reaches (:func:`tilt_program`).

    Args:
        query (ResolutionQuery[NT, T, G]): The query to sample from, generator or partial-term.
        algebra (AdditiveCostAlgebra[Any]): The additive cost algebra, with finite real costs.
        theta (float): The tilt, any finite real number.
        subgoal_selection (Callable | None): The computation rule, which must select an open hole.
            (Default value = None)
        table (TiltTable[NT] | None): A table already computed for this program, algebra and ``theta``.
            Passing one is how several queries share it. (Default value = None)

    Returns:
        TiltedSearch[NT, T, G]: The construction, ready to stream from.

    Raises:
        ValueError: Where :func:`tilt_table` refuses the program, the algebra or ``theta``; or if a table
            handed in was computed for another program, under another algebra or for another ``theta``.
    """
    theta = _real_theta(theta)
    decomposable_or_raise(query.solution_space)
    computed = tilt_table(query.solution_space, algebra, theta, check=False) if table is None else table
    if computed.space is not query.solution_space:
        msg = "the tilt table was computed for another program than the one the query asks"
        raise ValueError(msg)
    if computed.algebra is not algebra:
        msg = "the tilt table was computed under another algebra than the one the search charges clauses by"
        raise ValueError(msg)
    if computed.theta != theta:
        msg = f"the tilt table was computed for another theta, {computed.theta}, than the search's {theta}"
        raise ValueError(msg)
    probe = TiltedSearch(query=query, table=computed, root_log_mass=-math.inf, subgoal_selection=subgoal_selection)
    root_log_mass = log_sum_exp(
        [probe.log_mass_of(tuple(holes(goal).values()), cost) for goal, cost in _initial_tilt_nodes(query, algebra)]
    )
    return TiltedSearch(query=query, table=computed, root_log_mass=root_log_mass, subgoal_selection=subgoal_selection)


def _prepared(
    query: ResolutionQuery[NT, T, G], algebra: AdditiveCostAlgebra[Any], program: TiltProgram[NT] | None
) -> TiltProgram[NT]:
    """Return the query's program prepared under the algebra: the one handed in, checked, or a fresh one.

    Args:
        query (ResolutionQuery[NT, T, G]): The query.
        algebra (AdditiveCostAlgebra[Any]): The algebra.
        program (TiltProgram[NT] | None): A prepared program, or None.

    Returns:
        TiltProgram[NT]: The prepared program.

    Raises:
        ValueError: If the program handed in was prepared for another program or under another algebra.
    """
    if program is None:
        return tilt_program(query.solution_space, algebra)
    if program.space is not query.solution_space:
        msg = "the prepared program is another program than the one the query asks"
        raise ValueError(msg)
    if program.algebra is not algebra:
        msg = "the prepared program was prepared under another algebra than the one the search charges clauses by"
        raise ValueError(msg)
    return program


def theta_for_mean(
    query: ResolutionQuery[NT, T, G],
    algebra: AdditiveCostAlgebra[Any],
    target: float,
    *,
    tolerance: float = 1e-9,
    max_steps: int = 400,
    program: TiltProgram[NT] | None = None,
) -> float:
    """Find the ``theta`` under which the query's terms have a given tilted mean cost.

    The tilted mean runs from the dearest term's cost at ``theta = -inf`` to the cheapest's at ``+inf``
    and never increases on the way, so a target strictly between the two is met by one ``theta``,
    and one outside them by none. The cheapest and the dearest cost are computed exactly first, and a
    target outside them is refused; then a step of one over the span of the costs, doubled from zero
    in the direction the target lies until the mean crosses it, and bisection.

    Args:
        query (ResolutionQuery[NT, T, G]): The query whose terms' mean is targeted.
        algebra (AdditiveCostAlgebra[Any]): The additive cost algebra, with finite real costs.
        target (float): The mean cost wanted, strictly between the cheapest and the dearest term's cost,
            or equal to it where every term costs the same.
        tolerance (float): The error allowed on the mean, as a share of the span between the cheapest and
            the dearest cost. (Default value = 1e-9)
        max_steps (int): How many tilt tables the search may compute. (Default value = 400)
        program (TiltProgram[NT] | None): The query's program already prepared under this algebra; every
            step then costs one pass over its clauses. (Default value = None)

    Returns:
        float: The ``theta``.

    Raises:
        ValueError: If the target is not a finite real number; if the query has no term; if the target
            lies outside the costs the query's terms realize; if the steps run out first; or if floating
            point cannot meet the tolerance, or ``theta`` would have to leave its range.
    """
    target = _real_cost(target, "the target mean")
    prepared = _prepared(query, algebra, program)
    initial = _initial_holes(query, algebra)
    cheapest, dearest = _query_cost_range(initial, prepared.cheapest, prepared.dearest)
    if cheapest > dearest:
        msg = "the query has no term, so its terms have no mean"
        raise ValueError(msg)
    span = dearest - cheapest
    if target == cheapest == dearest:
        return 0.0
    if not cheapest < target < dearest:
        msg = (
            f"the target mean {target} lies outside the costs the query's terms realize, strictly between "
            f"{cheapest} and {dearest}; no tilt moves the mean to it"
        )
        raise ValueError(msg)
    allowed = tolerance * span
    steps = 0

    def mean_at(theta: float) -> float:
        nonlocal steps
        steps += 1
        if steps > max_steps:
            msg = f"no theta found for the mean {target} within {max_steps} steps"
            raise ValueError(msg)
        return _query_mean(initial, prepared.table(theta))

    at_zero = mean_at(0.0)
    if abs(at_zero - target) <= allowed:
        return 0.0
    direction = 1.0 if at_zero > target else -1.0
    near, far = 0.0, direction / span
    while True:
        mean = mean_at(far)
        if abs(mean - target) <= allowed:
            return far
        if (mean - target) * direction < 0:
            break
        near, far = far, 2 * far
    while True:
        middle = (near + far) / 2
        mean = mean_at(middle)
        if abs(mean - target) <= allowed:
            return middle
        if middle in (near, far):
            msg = f"the mean {target} cannot be met to {tolerance} of the costs' span in floating point"
            raise ValueError(msg)
        if (mean - target) * direction > 0:
            near = middle
        else:
            far = middle


# ---------------------------------------------------------------------------------------------
# A target on cost bins: the mixture of tilts, the counts per bin estimated, and rejection to the target
# ---------------------------------------------------------------------------------------------


# A bin is the interval between two edges, and a standard error needs two draws of each tilt.
_FEWEST_EDGES = 2
_FEWEST_PILOT_DRAWS = 2
# The largest language whose exhaustion a stream of the mixture watches for, keeping every term it draws.
_WATCHED_LANGUAGE = 2**20


def _log_min_over(
    log_density: Callable[[float], float], slope: Callable[[float], float], low: float, high: float
) -> float:
    """Return the least value of a log density on ``[low, high]``, the density a positive combination of exponentials.

    A positive combination of exponentials ``e^(-theta c)`` is convex in ``c``, so its derivative changes
    sign at most once, from falling to rising, and the least value lies at an end or where the slope of
    its logarithm crosses zero. The crossing is bisected until the interval cannot be split in floating
    point, which bounds the steps by the number of floating-point values between the ends.

    Args:
        log_density (Callable[[float], float]): The log of the density, as a function of the cost.
        slope (Callable[[float], float]): The derivative of the log density, whose sign is the density's.
        low (float): The interval's lower end.
        high (float): The interval's upper end.

    Returns:
        float: The least log density on the interval.
    """
    if slope(low) >= 0:
        return log_density(low)
    if slope(high) <= 0:
        return log_density(high)
    while True:
        middle = (low + high) / 2
        if not low < middle < high:
            break
        if slope(middle) < 0:
            low = middle
        else:
            high = middle
    return min(log_density(low), log_density(high))


@dataclass(frozen=True)
class TiltedMixture(Generic[NT, T, G]):
    """Random search to a target on cost bins: a mixture of tilts, and rejection by the estimated counts per bin.

    A tilt draws the terms of one cost value alike, and the cost values in proportion to their counts,
    tilted. A target on the costs needs those counts, which this construction estimates: from pilot
    draws of every tilt in the mixture, whose exact probabilities are known, every draw in a bin counts
    the inverse of its probability under the mixture, averaged, which is unbiased for the number of
    terms in the bin. A draw from the mixture is then accepted in proportion to the bin's target over
    the estimated count and the draw's mixture probability, so within a bin the accepted terms are
    alike, exactly, and the bins carry the target up to the estimate's error. A term already streamed
    is skipped, so the stream is a sample without replacement in proportion to the estimated weights.

    Attributes:
        searches (tuple[TiltedSearch[NT, T, G], ...]): The tilted searches of the mixture, one per ``theta``.
        shares (tuple[float, ...]): The mixture's weight of each, its share of the pilot draws.
        edges (tuple[float, ...]): The bins' boundaries, ascending; bin ``i`` holds the costs in
            ``[edges[i], edges[i + 1])``.
        target (tuple[float, ...]): The target's mass per bin, normalized over the bins with an estimate.
        log_estimate (tuple[float, ...]): The log of the estimated number of terms per bin, ``-inf`` for a
            bin no pilot draw fell in.
        relative_error (tuple[float, ...]): The estimate's standard error over the estimate, per bin;
            ``inf`` without an estimate. A bin one pilot draw reached reports exactly one: a single draw
            carries no estimate of its own spread, and the one says only that the estimate rests on it.
        pilot_counts (tuple[int, ...]): The number of pilot draws of each tilt.
        missing_target (float): The share of the target on bins without an estimate, which the stream
            cannot reach.
        log_bound (float): ``log M``, the bound on the ratio of the target to the mixture over every
            term in a bin with a target.
    """

    searches: tuple[TiltedSearch[NT, T, G], ...]
    shares: tuple[float, ...]
    edges: tuple[float, ...]
    target: tuple[float, ...]
    log_estimate: tuple[float, ...]
    relative_error: tuple[float, ...]
    pilot_counts: tuple[int, ...]
    missing_target: float
    log_bound: float

    def bin_of(self, cost: float) -> int | None:
        """Return the bin a cost falls in.

        Args:
            cost (float): The cost.

        Returns:
            int | None: The bin's index, None outside the edges.
        """
        index = bisect_right(self.edges, cost) - 1
        return index if 0 <= index < len(self.edges) - 1 else None

    def log_proposal(self, cost: float) -> float:
        """Return the log of the probability the mixture gives one term of a cost.

        Args:
            cost (float): The term's cost.

        Returns:
            float: ``log sum_k share_k e^(-theta_k c) / Z_k``.
        """
        return log_sum_exp(
            [
                math.log(share) - search.table.theta * cost - search.root_log_mass
                for share, search in zip(self.shares, self.searches, strict=True)
                if share > 0
            ]
        )

    def log_proposal_slope(self, cost: float) -> float:
        """Return the derivative in the cost of :meth:`log_proposal`.

        Args:
            cost (float): The cost.

        Returns:
            float: ``-sum_k theta_k q_k(c) / Q(c)``, ``q_k`` the share of the tilt ``k`` in ``Q``.
        """
        log_total = self.log_proposal(cost)
        return -math.fsum(
            search.table.theta
            * math.exp(math.log(share) - search.table.theta * cost - search.root_log_mass - log_total)
            for share, search in zip(self.shares, self.searches, strict=True)
            if share > 0
        )

    def log_acceptance(self, cost: float) -> float:
        """Return the log of the probability that a draw of this cost is accepted.

        Args:
            cost (float): The draw's cost.

        Returns:
            float: ``log(target / (N-hat M Q))``, ``-inf`` outside the bins with a target and an estimate.
        """
        index = self.bin_of(cost)
        if index is None or self.target[index] <= 0 or self.log_estimate[index] == -math.inf:
            return -math.inf
        return math.log(self.target[index]) - self.log_estimate[index] - self.log_proposal(cost) - self.log_bound

    def stream(self, rng: random.Random, max_draws: int | None = None) -> Iterator[Tree[T]]:
        """Draw one stream: distinct terms, the bins in proportion to the target, a bin's terms alike.

        Every draw is the first term of a fresh stream of one tilt, chosen by its share, so the draws
        are independent. A language of at most ``2**20`` terms is watched for exhaustion: its stream ends
        once every one of its terms has been drawn and every one in a bin with a target streamed. That
        needs every term's derivation to be the only one, and a term of vanishing probability delays it
        without end; a larger language is not watched. So the stream ends where ``max_draws`` says, and
        otherwise goes on as long as it is asked.

        Args:
            rng (random.Random): The source of randomness.
            max_draws (int | None): The number of draws after which the stream ends, accepted or not;
                None for no such end. (Default value = None)

        Yields:
            Tree[T]: The accepted terms.
        """
        algebra = self.searches[0].table.algebra
        total = self.searches[0].total
        watched = total <= _WATCHED_LANGUAGE
        drawn: set[Tree[T]] = set()
        wanted: set[Tree[T]] = set()
        streamed: set[Tree[T]] = set()
        draws = 0
        while max_draws is None or draws < max_draws:
            if watched and len(drawn) >= total and len(streamed) >= len(wanted):
                return
            draws += 1
            pick, choice = rng.random(), len(self.shares) - 1
            for index, share in enumerate(self.shares):
                pick -= share
                if pick < 0:
                    choice = index
                    break
            _key, term = next(self.searches[choice].keyed_stream(rng))
            cost = _real_cost(algebra.fold(term), "the cost of a drawn term")
            log_acceptance = self.log_acceptance(cost)
            if watched:
                drawn.add(term)
                if log_acceptance > -math.inf:
                    wanted.add(term)
            if term in streamed or rng.random() >= math.exp(min(0.0, log_acceptance)):
                continue
            streamed.add(term)
            yield term


def tilted_mixture(
    query: ResolutionQuery[NT, T, G],
    algebra: AdditiveCostAlgebra[Any],
    thetas: Sequence[float],
    edges: Sequence[float],
    target: Sequence[float],
    pilot: int,
    rng: random.Random,
    *,
    program: TiltProgram[NT] | None = None,
) -> TiltedMixture[NT, T, G]:
    """Estimate the counts per bin from pilot draws of a mixture of tilts, and bound the rejection to a target.

    Args:
        query (ResolutionQuery[NT, T, G]): The query to sample from, generator or partial-term.
        algebra (AdditiveCostAlgebra[Any]): The additive cost algebra, with finite real costs.
        thetas (Sequence[float]): The tilts of the mixture, distinct finite reals; spread so that their
            tilted means cover the target's bins (:func:`theta_for_mean`).
        edges (Sequence[float]): The bins' boundaries, strictly ascending finite reals, at least two.
        target (Sequence[float]): The target's mass per bin, one fewer than the edges, nonnegative, not all zero.
        pilot (int): The number of pilot draws of each tilt, at least two, so that a standard error exists.
        rng (random.Random): The source of randomness for the pilot draws.
        program (TiltProgram[NT] | None): The query's program already prepared under this algebra, which
            every tilt of the mixture then shares. (Default value = None)

    Returns:
        TiltedMixture[NT, T, G]: The construction, ready to stream from.

    Raises:
        ValueError: If the tilts, the edges, the target or the pilot size are not of the kinds above; where
            :func:`tilted_search` refuses the query; or if no pilot draw falls in a bin with a target.
    """
    thetas = [_real_theta(theta) for theta in thetas]
    if not thetas or len(set(thetas)) != len(thetas):
        msg = f"the mixture needs at least one tilt and distinct ones, not {thetas}"
        raise ValueError(msg)
    edges = [_real_cost(edge, "a bin edge") for edge in edges]
    if len(edges) < _FEWEST_EDGES or any(low >= high for low, high in pairwise(edges)):
        msg = f"the bin edges must be at least two strictly ascending numbers, not {edges}"
        raise ValueError(msg)
    target = [_real_cost(mass, "a target mass") for mass in target]
    if len(target) != len(edges) - 1 or any(mass < 0 for mass in target) or sum(target) <= 0:
        msg = "the target needs one nonnegative mass per bin, one fewer than the edges, and not all of them zero"
        raise ValueError(msg)
    if isinstance(pilot, bool) or not isinstance(pilot, int) or pilot < _FEWEST_PILOT_DRAWS:
        msg = f"the pilot size must be a whole number of at least two, not {pilot!r}"
        raise ValueError(msg)

    prepared = _prepared(query, algebra, program)
    searches = tuple(tilted_search(query, algebra, theta, table=prepared.table(theta)) for theta in thetas)
    if searches[0].root_log_mass == -math.inf:
        msg = "the query has no term, so there is nothing to estimate or to draw"
        raise ValueError(msg)
    shares = tuple(1 / len(searches) for _ in searches)
    mixture = TiltedMixture(
        searches=searches,
        shares=shares,
        edges=tuple(edges),
        target=tuple(target),
        log_estimate=(),
        relative_error=(),
        pilot_counts=(pilot,) * len(searches),
        missing_target=0.0,
        log_bound=0.0,
    )
    bins = len(edges) - 1
    # Each pilot draw's contribution to its bin is the inverse of its probability under the mixture, over
    # the number of draws: log(1 / (n Q(c))), grouped by the tilt that drew it for the standard error.
    contributions: list[list[list[float]]] = [[[] for _ in searches] for _ in range(bins)]
    for group, search in enumerate(searches):
        for _ in range(pilot):
            _key, term = next(search.keyed_stream(rng))
            cost = _real_cost(algebra.fold(term), "the cost of a drawn term")
            index = mixture.bin_of(cost)
            if index is not None:
                contributions[index][group].append(-mixture.log_proposal(cost) - math.log(pilot * len(searches)))

    log_estimate: list[float] = []
    relative_error: list[float] = []
    for index in range(bins):
        estimate = log_sum_exp([value for group in contributions[index] for value in group])
        log_estimate.append(estimate)
        if estimate == -math.inf:
            relative_error.append(math.inf)
            continue
        variance = 0.0
        for of_one_tilt in contributions[index]:
            # A tilt's draws that missed the bin contribute zero; its sum over the pilot draws varies as
            # pilot times the variance of one draw's contribution.
            scaled = [math.exp(value - estimate) for value in of_one_tilt] + [0.0] * (pilot - len(of_one_tilt))
            mean = sum(scaled) / pilot
            variance += pilot * sum((value - mean) ** 2 for value in scaled) / (pilot - 1)
        relative_error.append(math.sqrt(variance))

    return _mixture(searches, edges, target, log_estimate, relative_error, pilot)


def _mixture(
    searches: Sequence[TiltedSearch[NT, T, G]],
    edges: Sequence[float],
    target: Sequence[float],
    log_estimate: Sequence[float],
    relative_error: Sequence[float],
    pilot: int,
) -> TiltedMixture[NT, T, G]:
    """Normalize the target over the bins with an estimate and bound the rejection, given the counts per bin.

    Args:
        searches (Sequence[TiltedSearch[NT, T, G]]): The tilted searches, mixed in equal shares.
        edges (Sequence[float]): The bins' boundaries.
        target (Sequence[float]): The target's mass per bin.
        log_estimate (Sequence[float]): The log of the (estimated) number of terms per bin.
        relative_error (Sequence[float]): The estimate's relative standard error per bin.
        pilot (int): The number of pilot draws of each tilt.

    Returns:
        TiltedMixture[NT, T, G]: The construction.

    Raises:
        ValueError: If no bin with a target has an estimate.
    """
    shares = tuple(1 / len(searches) for _ in searches)
    reached = [mass if log_estimate[index] > -math.inf else 0.0 for index, mass in enumerate(target)]
    if sum(reached) <= 0:
        msg = "no pilot draw fell in a bin with a target; spread the tilts over the target's bins or draw more"
        raise ValueError(msg)
    normalized = tuple(mass / sum(reached) for mass in reached)
    probe = TiltedMixture(
        searches=tuple(searches),
        shares=shares,
        edges=tuple(edges),
        target=normalized,
        log_estimate=tuple(log_estimate),
        relative_error=tuple(relative_error),
        pilot_counts=(pilot,) * len(searches),
        missing_target=1 - sum(reached) / sum(target),
        log_bound=0.0,
    )
    # The least mixture probability on a bin is read only over the costs the query's terms realize: an
    # edge far past the dearest term would otherwise bound the ratio by a probability no term has.
    cheapest, dearest = _query_cost_range(
        _initial_holes(searches[0].query, searches[0].table.algebra),
        searches[0].table.cheapest,
        searches[0].table.dearest,
    )
    log_bound = max(
        math.log(normalized[index])
        - log_estimate[index]
        - _log_min_over(
            probe.log_proposal,
            probe.log_proposal_slope,
            max(edges[index], cheapest),
            min(edges[index + 1], dearest),
        )
        for index in range(len(edges) - 1)
        if normalized[index] > 0
    )
    return TiltedMixture(
        searches=probe.searches,
        shares=shares,
        edges=probe.edges,
        target=normalized,
        log_estimate=probe.log_estimate,
        relative_error=probe.relative_error,
        pilot_counts=probe.pilot_counts,
        missing_target=probe.missing_target,
        log_bound=log_bound,
    )
