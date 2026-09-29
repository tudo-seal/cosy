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
from fractions import Fraction
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
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from cosy.core.solution_space import RHSRule, SolutionSpace
    from cosy.core.tree import Path, Tree
    from cosy.search.costs import AdditiveCostAlgebra
    from cosy.search.queries import ResolutionQuery

__all__ = [
    "SaddleCounts",
    "SaddleGrid",
    "SaddleSearch",
    "TiltProgram",
    "TiltTable",
    "TiltedMixture",
    "TiltedSearch",
    "saddle_counts",
    "saddle_grid",
    "saddle_mixture",
    "saddle_search",
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
        variance (Mapping[NT, float]): Per non-terminal with a term, the variance of its terms' cost under the tilt,
            ``d^2 log Z_A / d theta^2``, which the saddle point reads beside the mean; ``inf`` where it leaves floating
            point, which a spread of costs beyond about ``1e154`` does.
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
    variance: Mapping[NT, float]

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
        cheapest_counts (Mapping[NT, int]): Per non-terminal with a term, the exact number of its terms of the
            least cost, where a tilted mean never arrives.
        dearest_counts (Mapping[NT, int]): Per non-terminal with a term, the exact number of its terms of the
            greatest cost.
        unit (float): A power of two whose whole multiples every clause's cost is, where floating point adds the
            costs of every term exactly (each sum of clause costs a multiple of it below ``2^52`` of it); 0 where it
            does not, and then no spacing is known.
        spacing (Mapping[NT, float]): Per non-terminal with a term, the greatest common divisor of the differences
            between its terms' costs, a multiple of ``unit``: its terms cost its cheapest cost plus multiples of it.
            0 where every term costs the same, and everywhere where ``unit`` is 0.
    """

    space: SolutionSpace[NT, Any, Any]
    algebra: AdditiveCostAlgebra[Any]
    nonterminals: tuple[NT, ...]
    order: tuple[NT, ...]
    clauses: Mapping[NT, tuple[tuple[float, tuple[NT, ...]], ...]]
    counts: Mapping[NT, int]
    cheapest: Mapping[NT, float]
    dearest: Mapping[NT, float]
    cheapest_counts: Mapping[NT, int]
    dearest_counts: Mapping[NT, int]
    unit: float
    spacing: Mapping[NT, float]

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
        variance: dict[NT, float] = {}
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
            shares = _normalized(exponents, total)
            # A clause's terms have the clause's excess plus their holes' costs: their mean the sum of the means,
            # their variance the sum of the variances, the holes being filled independently.
            conditional = [
                excess + sum(mean_excess[hole] for hole in hole_types)
                for excess, (_cost, hole_types) in zip(excesses, member_clauses, strict=True)
            ]
            mean = sum(share * value for share, value in zip(shares, conditional, strict=True))
            spread = _spread(shares, conditional, mean, member_clauses, variance)
            log_mass = -theta * base + total
            if not (math.isfinite(total) and math.isfinite(mean) and math.isfinite(log_mass)):
                msg = (
                    f"theta {theta} is too large in magnitude for the costs of this program: the tilted weights of "
                    f"{member} leave the range of floating point"
                )
                raise ValueError(msg)
            log_excess[member] = total
            mean_excess[member] = mean
            variance[member] = spread
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
            variance=variance,
        )


def _normalized(exponents: Sequence[float], total: float) -> list[float]:
    """Return the shares ``e^(exponent - total)``, divided by their sum.

    ``total`` is the log of the sum of the ``e^exponent``, rounded at the scale of ``theta`` times the costs, so
    the shares sum to one only up to that times the machine epsilon; a mean of values as large as the costs would
    carry the difference times them, and a variance its square. Their sum divides it out.

    Args:
        exponents (Sequence[float]): The logs of the weights.
        total (float): The log of their sum.

    Returns:
        list[float]: The shares, summing to one.
    """
    shares = [math.exp(exponent - total) for exponent in exponents]
    whole = math.fsum(shares)
    return [share / whole for share in shares]


def _spread(
    shares: Sequence[float],
    conditional: Sequence[float],
    mean: float,
    clauses: Sequence[tuple[float, tuple[NT, ...]]],
    variance: Mapping[NT, float],
) -> float:
    """Return the variance of a mixture of clauses: their holes' variances and the spread of their means, weighted.

    A clause without weight adds nothing, not even an infinite variance of its holes; a spread beyond floating
    point is ``inf``.

    Args:
        shares (Sequence[float]): The clauses' shares.
        conditional (Sequence[float]): The clauses' mean costs.
        mean (float): The mixture's mean cost.
        clauses (Sequence[tuple[float, tuple[NT, ...]]]): The clauses: their costs and their holes' non-terminals.
        variance (Mapping[NT, float]): The variance per non-terminal of a hole.

    Returns:
        float: The variance, ``inf`` where it leaves floating point.
    """
    return sum(
        share * (sum(variance[hole] for hole in hole_types) + (value - mean) * (value - mean))
        for share, value, (_cost, hole_types) in zip(shares, conditional, clauses, strict=True)
        if share > 0
    )


# Floating point adds whole multiples of a power of two exactly while their sums stay below 2^53 of it; half of that
# leaves room for the cost a partial-term query has already charged.
_EXACT_MULTIPLES = 2**52


def _unit_of(live: Mapping[NT, tuple[_Clause, ...]], order: Sequence[NT]) -> float:
    """Return a power of two whose whole multiples every clause's cost is, where floating point adds them exactly.

    Every float is a whole multiple of a power of two, so the finest of the clauses' is a unit of all of them; the
    sums of a term's clause costs are then exact where no partial sum exceeds ``2^52`` units, which the greatest
    sum of absolute costs below each non-terminal bounds.

    Args:
        live (Mapping[NT, tuple[_Clause, ...]]): The clauses per non-terminal with a term.
        order (Sequence[NT]): The non-terminals with a term, every one after its clauses' holes.

    Returns:
        float: The unit, or 0 where the costs' sums leave the exact range.
    """
    denominator = max((Fraction(cost).denominator for member in order for cost, _holes in live[member]), default=1)
    magnitude: dict[NT, float] = {}
    for member in order:
        magnitude[member] = max(
            abs(cost) + sum(magnitude[hole] for hole in hole_types) for cost, hole_types in live[member]
        )
    top = max(magnitude.values(), default=0.0)
    if not math.isfinite(top) or Fraction(top) * denominator > _EXACT_MULTIPLES:
        return 0.0
    return 1 / denominator


def _spacing_of(
    live: Mapping[NT, tuple[_Clause, ...]], order: Sequence[NT], cheapest: Mapping[NT, float], unit: float
) -> dict[NT, float]:
    """Return per non-terminal the greatest common divisor of the differences between its terms' costs.

    A clause's terms cost the clause's cheapest completion plus each hole's difference from its own cheapest, so
    the differences below a non-terminal are generated by its clauses' cheapest completions less its cheapest cost
    and by its holes' spacings; their greatest common divisor, in units, is exact where the sums are.

    Args:
        live (Mapping[NT, tuple[_Clause, ...]]): The clauses per non-terminal with a term.
        order (Sequence[NT]): The non-terminals with a term, every one after its clauses' holes.
        cheapest (Mapping[NT, float]): The cheapest cost per non-terminal with a term.
        unit (float): The costs' unit, 0 where none is exact.

    Returns:
        dict[NT, float]: The spacing per non-terminal with a term, 0 for one whose terms all cost the same or
            where there is no unit.
    """
    if not unit:
        return dict.fromkeys(order, 0.0)
    units: dict[NT, int] = {}
    for member in order:
        divisor = 0
        for cost, hole_types in live[member]:
            low = cost + sum(cheapest[hole] for hole in hole_types)
            divisor = math.gcd(divisor, round((low - cheapest[member]) / unit), *(units[hole] for hole in hole_types))
        units[member] = divisor
    return {member: units[member] * unit for member in order}


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
    cheapest_counts: dict[NT, int] = {}
    dearest_counts: dict[NT, int] = {}
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
        lows = [cost + sum(cheapest[hole] for hole in hole_types) for cost, hole_types in live[member]]
        highs = [cost + sum(dearest[hole] for hole in hole_types) for cost, hole_types in live[member]]
        cheapest[member], dearest[member] = min(lows), max(highs)
        cheapest_counts[member] = sum(
            math.prod(cheapest_counts[hole] for hole in hole_types)
            for low, (_cost, hole_types) in zip(lows, live[member], strict=True)
            if low == cheapest[member]
        )
        dearest_counts[member] = sum(
            math.prod(dearest_counts[hole] for hole in hole_types)
            for high, (_cost, hole_types) in zip(highs, live[member], strict=True)
            if high == dearest[member]
        )
    unit = _unit_of(live, order)
    return TiltProgram(
        space=space,
        algebra=algebra,
        nonterminals=tuple(nonterminals),
        order=tuple(order),
        clauses=live,
        counts=counts,
        cheapest=cheapest,
        dearest=dearest,
        cheapest_counts=cheapest_counts,
        dearest_counts=dearest_counts,
        unit=unit,
        spacing=_spacing_of(live, order, cheapest, unit),
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
        return _query_moments(_initial_holes(self.query, self.table.algebra), self.table)[0]

    @property
    def variance_cost(self) -> float:
        """Return the variance of the cost of the query's terms under the tilt, ``d^2 log Z / d theta^2`` of the query's mass.

        Returns:
            float: The variance, ``nan`` for a query without a term.
        """
        return _query_moments(_initial_holes(self.query, self.table.algebra), self.table)[1]

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


def _query_moments(initial: Sequence[tuple[float, tuple[NT, ...]]], table: TiltTable[NT]) -> tuple[float, float]:
    """Return the tilted mean and variance of the cost of the terms below a query's initial nodes.

    Args:
        initial (Sequence[tuple[float, tuple[NT, ...]]]): The initial nodes' costs so far and holes.
        table (TiltTable[NT]): The tilt table.

    Returns:
        tuple[float, float]: The mean and the variance, both ``nan`` without a term.
    """
    _log_mass, mean, variance = _query_summary(initial, table)
    return mean, variance


def _query_summary(initial: Sequence[tuple[float, tuple[NT, ...]]], table: TiltTable[NT]) -> tuple[float, float, float]:
    """Return the log of the query's tilted mass, and the tilted mean and variance of its terms' cost.

    Computed in excess of the cheapest of them, as the table is, so that a large shared cost costs no precision.

    Args:
        initial (Sequence[tuple[float, tuple[NT, ...]]]): The initial nodes' costs so far and holes.
        table (TiltTable[NT]): The tilt table.

    Returns:
        tuple[float, float, float]: ``log Z``, the mean and the variance; ``-inf`` and two ``nan`` without a term.
    """
    live = [(cost, hole_types) for cost, hole_types in initial if all(hole in table.cheapest for hole in hole_types)]
    if not live:
        return -math.inf, math.nan, math.nan
    bases = [cost + sum(table.cheapest[hole] for hole in hole_types) for cost, hole_types in live]
    base = min(bases)
    exponents = [
        -table.theta * (least - base) + sum(table.log_excess[hole] for hole in hole_types)
        for least, (_cost, hole_types) in zip(bases, live, strict=True)
    ]
    total = log_sum_exp(exponents)
    shares = _normalized(exponents, total)
    conditional = [
        least - base + sum(table.mean_excess[hole] for hole in hole_types)
        for least, (_cost, hole_types) in zip(bases, live, strict=True)
    ]
    mean = sum(share * value for share, value in zip(shares, conditional, strict=True))
    variance = _spread(shares, conditional, mean, live, table.variance)
    return -table.theta * base + total, base + mean, variance


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
    steps = 0

    def mean_at(theta: float) -> float:
        nonlocal steps
        steps += 1
        if steps > max_steps:
            msg = f"no theta found for the mean {target} within {max_steps} steps"
            raise ValueError(msg)
        return _query_moments(initial, prepared.table(theta))[0]

    return _theta_between(
        mean_at,
        target,
        span,
        tolerance * span,
        f"the mean {target} cannot be met to {tolerance} of the costs' span in floating point",
    )


def _theta_between(mean_at: Callable[[float], float], target: float, span: float, allowed: float, unmet: str) -> float:
    """Return a ``theta`` whose tilted mean lies within ``allowed`` of a target strictly inside the costs' range.

    From zero, a step of one over the span of the costs, doubled in the direction the target lies until the mean
    crosses it, then bisection; each step is one call of ``mean_at``, which counts them.

    Args:
        mean_at (Callable[[float], float]): The tilted mean at a ``theta``.
        target (float): The mean wanted.
        span (float): The span between the cheapest and the dearest cost, positive.
        allowed (float): The error allowed on the mean.
        unmet (str): The message when floating point cannot meet ``allowed``.

    Returns:
        float: The ``theta``.

    Raises:
        ValueError: If floating point cannot meet ``allowed``; and whatever ``mean_at`` raises.
    """
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
            raise ValueError(unmet)
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
            ``nan`` in every bin where the counts carry no error bar of their own (:func:`saddle_mixture`).
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


def _checked_mixture(
    thetas: Sequence[float], edges: Sequence[float], target: Sequence[float]
) -> tuple[list[float], list[float], list[float]]:
    """Return a mixture's tilts, edges and target as real numbers, or refuse them.

    Args:
        thetas (Sequence[float]): The tilts.
        edges (Sequence[float]): The bins' boundaries.
        target (Sequence[float]): The target's mass per bin.

    Returns:
        tuple[list[float], list[float], list[float]]: The three, checked.

    Raises:
        ValueError: If the tilts are none or not distinct, the edges not strictly ascending, or the target not one
            nonnegative mass per bin, not all zero.
    """
    checked_thetas = [_real_theta(theta) for theta in thetas]
    if not checked_thetas or len(set(checked_thetas)) != len(checked_thetas):
        msg = f"the mixture needs at least one tilt and distinct ones, not {checked_thetas}"
        raise ValueError(msg)
    checked_edges = _checked_edges(edges)
    checked_target = [_real_cost(mass, "a target mass") for mass in target]
    if (
        len(checked_target) != len(checked_edges) - 1
        or any(mass < 0 for mass in checked_target)
        or sum(checked_target) <= 0
    ):
        msg = "the target needs one nonnegative mass per bin, one fewer than the edges, and not all of them zero"
        raise ValueError(msg)
    return checked_thetas, checked_edges, checked_target


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
    thetas, edges, target = _checked_mixture(thetas, edges, target)
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

    return _mixture(
        searches,
        edges,
        target,
        log_estimate,
        relative_error,
        pilot,
        "no pilot draw fell in a bin with a target; spread the tilts over the target's bins or draw more",
    )


def _mixture(
    searches: Sequence[TiltedSearch[NT, T, G]],
    edges: Sequence[float],
    target: Sequence[float],
    log_estimate: Sequence[float],
    relative_error: Sequence[float],
    pilot: int,
    unreached: str,
) -> TiltedMixture[NT, T, G]:
    """Normalize the target over the bins with an estimate and bound the rejection, given the counts per bin.

    Args:
        searches (Sequence[TiltedSearch[NT, T, G]]): The tilted searches, mixed in equal shares.
        edges (Sequence[float]): The bins' boundaries.
        target (Sequence[float]): The target's mass per bin.
        log_estimate (Sequence[float]): The log of the (estimated) number of terms per bin.
        relative_error (Sequence[float]): The estimate's relative standard error per bin.
        pilot (int): The number of pilot draws of each tilt.
        unreached (str): Why no bin with a target has an estimate, for the refusal.

    Returns:
        TiltedMixture[NT, T, G]: The construction.

    Raises:
        ValueError: If no bin with a target has an estimate.
    """
    shares = tuple(1 / len(searches) for _ in searches)
    reached = [mass if log_estimate[index] > -math.inf else 0.0 for index, mass in enumerate(target)]
    if sum(reached) <= 0:
        raise ValueError(unreached)
    normalized = tuple(mass / sum(reached) for mass in reached)
    probe = TiltedMixture(
        searches=tuple(searches),
        shares=shares,
        edges=tuple(edges),
        target=normalized,
        log_estimate=tuple(log_estimate),
        relative_error=tuple(relative_error),
        pilot_counts=(pilot,) * len(searches),
        missing_target=math.fsum(
            mass for mass, estimate in zip(target, log_estimate, strict=True) if estimate == -math.inf
        )
        / math.fsum(target),
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


# Beyond this many standard deviations the complementary error function runs out of floating point, and the tail's
# asymptotic series takes over, its error below one part in ten billion there.
_ERFC_RANGE = 37.0
# An interval narrower than this, in units of the larger of 1 and its ends' distance from the mean, has the density
# at its middle times its width as its mass, to the second order: the rest is below one part in 10^15, where two
# tails or two error functions would cancel to their rounding.
_NARROW = 1e-3
# The Newton steps the saddle point takes for one bin before it brackets and bisects instead.
_NEWTON_STEPS = 12


def _log_upper_tail(x: float) -> float:
    """Return ``log Q(x)``, ``Q`` the upper tail of the standard normal distribution.

    Args:
        x (float): The point.

    Returns:
        float: ``log(1 - Phi(x))``, accurate far into the tail.
    """
    if x < _ERFC_RANGE:
        return math.log(0.5 * math.erfc(x / math.sqrt(2)))
    inverse = 1 / (x * x)
    return -x * x / 2 - math.log(x * math.sqrt(2 * math.pi)) + math.log1p(-inverse + 3 * inverse**2 - 15 * inverse**3)


def _log_gaussian_interval(alpha: float, beta: float) -> float:
    """Return ``log(Phi(beta) - Phi(alpha))`` for ``alpha <= beta``.

    A narrow interval from the density at its middle, ``phi(m) w (1 + (m^2 - 1) w^2 / 24)``; one on either side of
    the mean from the two error functions, which add; one on one side from the tail both ends lie in.

    Args:
        alpha (float): The lower end, in standard deviations.
        beta (float): The upper end.

    Returns:
        float: The log of the standard normal mass between them, ``-inf`` for an empty interval.
    """
    if not alpha < beta:
        return -math.inf
    width, middle = beta - alpha, (alpha + beta) / 2
    if width * max(1.0, abs(alpha), abs(beta)) < _NARROW:
        return (
            -middle * middle / 2
            - math.log(2 * math.pi) / 2
            + math.log(width)
            + math.log1p((middle * middle - 1) * width * width / 24)
        )
    if alpha >= 0:
        near, far = _log_upper_tail(alpha), _log_upper_tail(beta)
    elif beta <= 0:
        near, far = _log_upper_tail(-beta), _log_upper_tail(-alpha)
    else:
        return math.log((math.erf(beta / math.sqrt(2)) + math.erf(-alpha / math.sqrt(2))) / 2)
    if far >= near:
        return -math.inf
    return near + math.log1p(-math.exp(far - near))


@dataclass(frozen=True)
class SaddleCounts:
    """The saddle point's estimate of the number of a query's terms per cost bin.

    Where a term's cost is a sum of many independent parts, its distribution under a tilt is near a Gaussian, and so
    the number of terms per unit of cost near the tilted mean is ``Z(theta) e^(theta a) phi((a - mu)/sigma) / sigma``,
    and a point of a lattice of spacing ``d`` holds ``d`` times that: the saddle point. Each bin is read at a tilt whose
    mean falls near it, and the local form is integrated over the part of the bin the query's costs reach, in closed
    form. Its error is the saddle point's: small where many independent parts add up to a law with one mode, large where
    few discrete costs remain, and large however many parts there are where the terms fall into groups whose costs lie
    apart, a cheap term beside a cluster of dear ones, say, which one tilt cannot fit both of. It carries no error bar
    of its own.

    Attributes:
        edges (tuple[float, ...]): The bins' boundaries; bin ``i`` holds the costs in ``[edges[i], edges[i + 1])``.
        log_counts (tuple[float, ...]): The log of the estimated number of terms per bin, ``-inf`` for a bin the
            query's costs do not reach.
        thetas (tuple[float, ...]): The tilt each bin was read at, ``nan`` for one it was not.
    """

    edges: tuple[float, ...]
    log_counts: tuple[float, ...]
    thetas: tuple[float, ...]


def _checked_edges(edges: Sequence[float]) -> list[float]:
    """Return the bins' edges as real numbers, or refuse them.

    Args:
        edges (Sequence[float]): The edges.

    Returns:
        list[float]: The edges.

    Raises:
        ValueError: If they are not at least two strictly ascending finite real numbers.
    """
    checked = [_real_cost(edge, "a bin edge") for edge in edges]
    if len(checked) < _FEWEST_EDGES or any(low >= high for low, high in pairwise(checked)):
        msg = f"the bin edges must be at least two strictly ascending numbers, not {checked}"
        raise ValueError(msg)
    return checked


def _query_extreme_counts(
    initial: Sequence[tuple[float, tuple[NT, ...]]], program: TiltProgram[NT], cheapest: float, dearest: float
) -> tuple[int, int]:
    """Return the exact number of the query's terms of its least and of its greatest cost.

    Args:
        initial (Sequence[tuple[float, tuple[NT, ...]]]): The initial nodes' costs so far and holes.
        program (TiltProgram[NT]): The query's program, prepared.
        cheapest (float): The query's least cost.
        dearest (float): The query's greatest cost.

    Returns:
        tuple[int, int]: The two numbers.
    """
    at_least, at_most = 0, 0
    for cost, hole_types in initial:
        if all(hole in program.cheapest for hole in hole_types):
            if cost + sum(program.cheapest[hole] for hole in hole_types) == cheapest:
                at_least += math.prod(program.cheapest_counts[hole] for hole in hole_types)
            if cost + sum(program.dearest[hole] for hole in hole_types) == dearest:
                at_most += math.prod(program.dearest_counts[hole] for hole in hole_types)
    return at_least, at_most


def _query_spacing(initial: Sequence[tuple[float, tuple[NT, ...]]], program: TiltProgram[NT], cheapest: float) -> float:
    """Return the spacing of the lattice a query's costs lie on, counted from its cheapest cost.

    Args:
        initial (Sequence[tuple[float, tuple[NT, ...]]]): The initial nodes' costs so far and holes.
        program (TiltProgram[NT]): The query's program, prepared.
        cheapest (float): The query's least cost.

    Returns:
        float: The greatest common divisor of the differences between the query's terms' costs; 0 where they all
            cost the same, where the program's costs have no exact unit, or where a cost a partial term has already
            charged is off it.
    """
    unit = program.unit
    if not unit:
        return 0.0
    divisor = 0
    for cost, hole_types in initial:
        if not all(hole in program.cheapest for hole in hole_types):
            continue
        if not (cost / unit).is_integer() or abs(cost) / unit > _EXACT_MULTIPLES:
            return 0.0
        low = cost + sum(program.cheapest[hole] for hole in hole_types)
        divisor = math.gcd(
            divisor, round((low - cheapest) / unit), *(round(program.spacing[hole] / unit) for hole in hole_types)
        )
    return divisor * unit


def _checked_bins(only: Iterable[int] | None, bins: int) -> set[int]:
    """Return the bins asked for by index, or refuse them.

    Args:
        only (Iterable[int] | None): The indices, None for every bin.
        bins (int): The number of bins.

    Returns:
        set[int]: The indices.

    Raises:
        ValueError: If an index is not a whole number naming one of the bins.
    """
    if only is None:
        return set(range(bins))
    wanted = list(only)
    if any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < bins for index in wanted):
        msg = f"only names bins by their index, from 0 to {bins - 1}, not {wanted!r}"
        raise ValueError(msg)
    return set(wanted)


def saddle_counts(
    query: ResolutionQuery[NT, T, G],
    algebra: AdditiveCostAlgebra[Any],
    edges: Sequence[float],
    *,
    program: TiltProgram[NT] | None = None,
    max_steps: int = 4000,
    only: Iterable[int] | None = None,
) -> SaddleCounts:
    """Estimate the number of a query's terms per cost bin by the saddle point.

    Where the query's costs lie on a lattice, its cheapest cost plus whole multiples of a spacing that floating point
    adds exactly, a bin is the cells of the lattice points it holds, each as wide as the spacing, and a bin holding
    only the cheapest or the dearest cost counts its terms exactly, no tilted mean arriving there. Elsewhere a bin is
    the part of it the query's costs reach, and only a bin that touches the dearest cost at its lower edge is counted
    exactly. The other bins are read from the one whose middle lies nearest the untilted mean outwards, each at a tilt
    whose mean falls within its cells, found by damped Newton steps on the tilted mean (whose derivative is minus the
    tilted variance) from its neighbour's tilt, so that a bin costs a few tilt tables; where Newton stalls, the tilt is
    bracketed and bisected.

    Args:
        query (ResolutionQuery[NT, T, G]): The query.
        algebra (AdditiveCostAlgebra[Any]): The additive cost algebra, with finite real costs.
        edges (Sequence[float]): The bins' boundaries, strictly ascending finite reals, at least two.
        program (TiltProgram[NT] | None): The query's program already prepared under this algebra. (Default value = None)
        max_steps (int): How many tilt tables the estimate may compute in all, bracketing and bisection included.
            (Default value = 4000)
        only (Iterable[int] | None): The bins to estimate, by index; None for all. A bin left out reads ``-inf``.
            (Default value = None)

    Returns:
        SaddleCounts: The estimate.

    Raises:
        ValueError: If the edges are not of the kind above, ``max_steps`` is not a positive whole number or ``only``
            names no bin; if the query has no term; if the steps run out; or if the costs' variance under a tilt
            leaves floating point.
    """
    edges = _checked_edges(edges)
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        msg = f"max_steps must be a positive whole number, not {max_steps!r}"
        raise ValueError(msg)
    bins = len(edges) - 1
    wanted = _checked_bins(only, bins)
    prepared = _prepared(query, algebra, program)
    initial = _initial_holes(query, algebra)
    cheapest, dearest = _query_cost_range(initial, prepared.cheapest, prepared.dearest)
    if cheapest > dearest:
        msg = "the query has no term, so there is nothing to count"
        raise ValueError(msg)
    log_counts = [-math.inf] * bins
    thetas = [math.nan] * bins
    reached = [
        index for index in range(bins) if index in wanted and edges[index] <= dearest and cheapest < edges[index + 1]
    ]
    if cheapest == dearest:
        # One cost for every term: the whole count lies in the bin that holds it, exactly.
        log_mass, _mean, _variance = _query_summary(initial, prepared.table(0.0))
        for index in reached:
            log_counts[index], thetas[index] = log_mass, 0.0
        return SaddleCounts(edges=tuple(edges), log_counts=tuple(log_counts), thetas=tuple(thetas))
    steps = 0
    summaries: dict[float, tuple[float, float, float]] = {}

    def summary_at(theta: float) -> tuple[float, float, float]:
        nonlocal steps
        if theta not in summaries:
            steps += 1
            if steps > max_steps:
                msg = f"the saddle point needed more than {max_steps} tilt tables"
                raise ValueError(msg)
            summary = _query_summary(initial, prepared.table(theta))
            if not math.isfinite(summary[2]):
                msg = (
                    f"the variance of the query's costs under the tilt {theta} leaves floating point, and the saddle "
                    "point reads it"
                )
                raise ValueError(msg)
            summaries[theta] = summary
        return summaries[theta]

    def tilt_for(low: float, high: float, start: float) -> tuple[float, tuple[float, float, float]]:
        middle, allowed = (low + high) / 2, (high - low) / 2
        theta, summary = start, summary_at(start)
        for _ in range(_NEWTON_STEPS):
            _log_mass, mean, variance = summary
            if abs(mean - middle) <= allowed:
                return theta, summary
            step = (mean - middle) / variance if variance > 0 else 0.0
            while step:
                trial = summary_at(theta + step)
                if abs(trial[1] - middle) < abs(mean - middle):
                    theta, summary = theta + step, trial
                    break
                step /= 2
                if abs(step) <= abs(theta) * 1e-15:
                    step = 0.0
            if not step:
                break
        theta = _theta_between(
            lambda value: summary_at(value)[1],
            middle,
            dearest - cheapest,
            allowed,
            f"no tilt puts the mean within {allowed} of {middle} in floating point",
        )
        return theta, summary_at(theta)

    spacing = _query_spacing(initial, prepared, cheapest)
    spans: dict[int, tuple[float, float]] = {}
    extremes: dict[int, float] = {}
    for index in reached:
        if spacing:
            first = cheapest + spacing * max(0, math.ceil((edges[index] - cheapest) / spacing))
            last = cheapest + spacing * min(
                (dearest - cheapest) // spacing, math.ceil((edges[index + 1] - cheapest) / spacing) - 1
            )
            if first > last:
                continue
            if first == last and first in (cheapest, dearest):
                extremes[index] = first
            else:
                spans[index] = (first - spacing / 2, last + spacing / 2)
        else:
            low, high = max(edges[index], cheapest), min(edges[index + 1], dearest)
            if low == high:
                extremes[index] = low
            else:
                spans[index] = (low, high)
    if extremes:
        at_least, at_most = _query_extreme_counts(initial, prepared, cheapest, dearest)
        for index, cost in extremes.items():
            log_counts[index] = math.log(at_least if cost == cheapest else at_most)
    reached = sorted(spans)
    if not reached:
        return SaddleCounts(edges=tuple(edges), log_counts=tuple(log_counts), thetas=tuple(thetas))
    _log_mass, untilted_mean, _variance = summary_at(0.0)
    middles = {index: (spans[index][0] + spans[index][1]) / 2 for index in reached}
    first = min(reached, key=lambda index: abs(middles[index] - untilted_mean))
    upward = [index for index in reached if index >= first]
    downward = [index for index in reached if index < first][::-1]
    for sweep in (upward, downward):
        start = thetas[first] if sweep is downward else 0.0
        for index in sweep:
            low, high = spans[index]
            theta, (log_mass, mean, variance) = tilt_for(max(low, cheapest), min(high, dearest), start)
            thetas[index], start = theta, theta
            if variance > 0:
                deviation = math.sqrt(variance)
                shift = mean + theta * variance
                log_counts[index] = (
                    log_mass
                    + theta * mean
                    + theta * theta * variance / 2
                    + _log_gaussian_interval((low - shift) / deviation, (high - shift) / deviation)
                )
            elif low <= mean <= high:
                log_counts[index] = log_mass + theta * mean
    return SaddleCounts(edges=tuple(edges), log_counts=tuple(log_counts), thetas=tuple(thetas))


def saddle_mixture(
    query: ResolutionQuery[NT, T, G],
    algebra: AdditiveCostAlgebra[Any],
    thetas: Sequence[float],
    edges: Sequence[float],
    target: Sequence[float],
    *,
    program: TiltProgram[NT] | None = None,
    least_share: float = 0.0,
) -> TiltedMixture[NT, T, G]:
    """Build the mixture of tilts to a target on cost bins with the saddle point's counts in place of a pilot.

    :func:`tilted_mixture` estimates the number of terms per bin from pilot draws; here :func:`saddle_counts`
    gives it, deterministically and without a draw. The stream is the same: a bin's terms alike, exactly, and the
    bins following the target up to the saddle point's error, which carries no error bar of its own.

    Args:
        query (ResolutionQuery[NT, T, G]): The query to sample from, generator or partial-term.
        algebra (AdditiveCostAlgebra[Any]): The additive cost algebra, with finite real costs.
        thetas (Sequence[float]): The tilts of the mixture, distinct finite reals.
        edges (Sequence[float]): The bins' boundaries, strictly ascending finite reals, at least two.
        target (Sequence[float]): The target's mass per bin, one fewer than the edges, nonnegative, not all zero.
        program (TiltProgram[NT] | None): The query's program already prepared under this algebra. (Default value = None)
        least_share (float): Bins whose share of the target is below this, a real number from 0 to 1, are not
            estimated, their share reported as out of reach; every tilt table a bin needs is a pass over the
            program. (Default value = 0.0)

    Returns:
        TiltedMixture[NT, T, G]: The construction, ready to stream from.

    Raises:
        ValueError: If the tilts, the edges, the target or the least share are not of the kinds above; where
            :func:`tilted_search` or :func:`saddle_counts` refuses the query; if the query has no term; or if no bin
            with a target at or above the least share has an estimated term.
    """
    thetas, edges, target = _checked_mixture(thetas, edges, target)
    if isinstance(least_share, bool) or not isinstance(least_share, (int, float)) or not 0 <= least_share <= 1:
        msg = f"the least share must be a real number between 0 and 1, not {least_share!r}"
        raise ValueError(msg)
    prepared = _prepared(query, algebra, program)
    searches = tuple(tilted_search(query, algebra, theta, table=prepared.table(theta)) for theta in thetas)
    if searches[0].root_log_mass == -math.inf:
        msg = "the query has no term, so there is nothing to estimate or to draw"
        raise ValueError(msg)
    total = sum(target)
    wanted = [index for index, mass in enumerate(target) if mass > 0 and mass / total >= least_share]
    counts = saddle_counts(query, algebra, edges, program=prepared, only=wanted)
    return _mixture(
        searches,
        edges,
        target,
        counts.log_counts,
        [math.nan] * len(target),
        0,
        "no bin with a target at or above the least share has an estimated term of the query",
    )


# ---------------------------------------------------------------------------------------------
# The saddle point per search node: random search weighed by each node's estimated completions per bin
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SaddleGrid(Generic[NT]):
    """The tilted moments of every non-terminal of a program on a grid of tilts.

    A search node's completions fill its holes independently, so their tilted mass, mean and variance add over the
    holes, and the moments of every non-terminal, tabulated once on a grid of tilts, serve every node; a search for a
    tilt per node and bin would cost a table each.

    Attributes:
        program (TiltProgram[NT]): The prepared program.
        thetas (tuple[float, ...]): The grid, ascending.
        log_excess (Mapping[NT, tuple[float, ...]]): Per non-terminal with a term, ``log Z_A + theta m_A`` at each tilt.
        mean_excess (Mapping[NT, tuple[float, ...]]): Per non-terminal with a term, its tilted mean cost less ``m_A``.
        variance (Mapping[NT, tuple[float, ...]]): Per non-terminal with a term, its tilted variance.
    """

    program: TiltProgram[NT]
    thetas: tuple[float, ...]
    log_excess: Mapping[NT, tuple[float, ...]]
    mean_excess: Mapping[NT, tuple[float, ...]]
    variance: Mapping[NT, tuple[float, ...]]


def saddle_grid(program: TiltProgram[NT], thetas: Iterable[float]) -> SaddleGrid[NT]:
    """Tabulate the tilted moments of every non-terminal of a prepared program at each of a grid of tilts.

    Args:
        program (TiltProgram[NT]): The prepared program.
        thetas (Iterable[float]): The tilts, finite reals; duplicates are read once.

    Returns:
        SaddleGrid[NT]: The moments, one table's worth per tilt, the tables themselves not kept.

    Raises:
        ValueError: If there is no tilt or one is not a finite real number, where a table refuses a tilt, or if a
            variance on the grid leaves floating point.
    """
    grid = tuple(sorted({_real_theta(theta) for theta in thetas}))
    if not grid:
        msg = "the grid needs at least one tilt"
        raise ValueError(msg)
    columns: dict[NT, tuple[list[float], list[float], list[float]]] = {member: ([], [], []) for member in program.order}
    for theta in grid:
        table = program.table(theta)
        for member, (log_excess, mean_excess, variance) in columns.items():
            spread = table.variance[member]
            if not math.isfinite(spread):
                msg = (
                    f"the variance of the costs of {member} under the tilt {theta} leaves floating point, and the "
                    "saddle point reads it"
                )
                raise ValueError(msg)
            log_excess.append(table.log_excess[member])
            mean_excess.append(table.mean_excess[member])
            variance.append(spread)
    return SaddleGrid(
        program=program,
        thetas=grid,
        log_excess={member: tuple(column[0]) for member, column in columns.items()},
        mean_excess={member: tuple(column[1]) for member, column in columns.items()},
        variance={member: tuple(column[2]) for member, column in columns.items()},
    )


# A node's moments on the grid: its cheapest and dearest completion's cost, and per tilt the log of its completions'
# tilted mass in excess of the cheapest, their mean cost less the cheapest, and their variance.
_Moments = tuple[float, float, tuple[float, ...], tuple[float, ...], tuple[float, ...]]


def _node_moments(grid: SaddleGrid[NT], cost: float, hole_types: Sequence[NT]) -> _Moments:
    """Return a node's moments on the grid from its holes: the sums over them.

    Args:
        grid (SaddleGrid[NT]): The grid.
        cost (float): The cost of the node's partial inhabitant.
        hole_types (Sequence[NT]): The non-terminals of its open holes, each with a term.

    Returns:
        _Moments: The node's moments.
    """
    program = grid.program
    least = cost + sum(program.cheapest[hole] for hole in hole_types)
    greatest = cost + sum(program.dearest[hole] for hole in hole_types)
    size = len(grid.thetas)
    if not hole_types:
        return least, greatest, (0.0,) * size, (0.0,) * size, (0.0,) * size
    return (
        least,
        greatest,
        tuple(math.fsum(column) for column in zip(*(grid.log_excess[hole] for hole in hole_types), strict=True)),
        tuple(math.fsum(column) for column in zip(*(grid.mean_excess[hole] for hole in hole_types), strict=True)),
        tuple(math.fsum(column) for column in zip(*(grid.variance[hole] for hole in hole_types), strict=True)),
    )


def _with_holes(
    grid: SaddleGrid[NT], base: _Moments, opened: Sequence[NT], cost: float, hole_types: Sequence[NT]
) -> _Moments:
    """Return the moments of a node whose holes are a base's and a clause's.

    Nothing is taken away: the base holds the holes besides the expanded one, summed from scratch once per expansion,
    and every moment is a sum of numbers none of which is negative, so adding a clause's holes costs no precision. The
    least and the greatest cost are the node's cost plus its holes', added as :func:`_node_moments` adds them, so that
    a node's range does not depend on the path to it, and a node whose holes each have one cost gets one cost.

    Args:
        grid (SaddleGrid[NT]): The grid.
        base (_Moments): The moments of the holes besides the expanded one.
        opened (Sequence[NT]): The non-terminals of the clause's holes.
        cost (float): The cost of the node's partial inhabitant.
        hole_types (Sequence[NT]): The non-terminals of all the node's holes.

    Returns:
        _Moments: The node's moments.
    """
    program = grid.program
    least = cost + sum(program.cheapest[hole] for hole in hole_types)
    greatest = cost + sum(program.dearest[hole] for hole in hole_types)
    if not opened:
        return least, greatest, base[2], base[3], base[4]
    return (
        least,
        greatest,
        tuple(math.fsum(column) for column in zip(base[2], *(grid.log_excess[hole] for hole in opened), strict=True)),
        tuple(math.fsum(column) for column in zip(base[3], *(grid.mean_excess[hole] for hole in opened), strict=True)),
        tuple(math.fsum(column) for column in zip(base[4], *(grid.variance[hole] for hole in opened), strict=True)),
    )


def _nearest_tilt(mean_excess: Sequence[float], variance: Sequence[float], target: float) -> int:
    """Return the grid index whose tilted mean lies nearest a target: of the two whose means bracket it, the one nearer
    in standard deviations, or the end the target lies beyond.

    The mean falls as the tilt rises, so the grid, ascending, holds it descending. A tilt under which the variance is 0
    (the node collapsed onto one cost, or its variance underflowed) says nothing about the bin's other costs, so it is
    read only where its neighbour has no variance either, and then the nearer mean wins.

    Args:
        mean_excess (Sequence[float]): The node's tilted mean less its cheapest cost, per grid tilt.
        variance (Sequence[float]): The node's tilted variance per grid tilt.
        target (float): The middle of the bin's reach, less the node's cheapest cost.

    Returns:
        int: The index.
    """
    last = len(mean_excess) - 1
    if mean_excess[0] <= target:
        return 0
    if mean_excess[last] >= target:
        return last
    low, high = 0, last
    while high - low > 1:
        middle = (low + high) // 2
        if mean_excess[middle] >= target:
            low = middle
        else:
            high = middle

    def distance(index: int) -> float:
        return abs(mean_excess[index] - target) / math.sqrt(variance[index]) if variance[index] > 0 else math.inf

    if distance(low) == distance(high) == math.inf:
        return low if abs(mean_excess[low] - target) <= abs(mean_excess[high] - target) else high
    return low if distance(low) <= distance(high) else high


def _node_log_counts(
    grid: SaddleGrid[NT],
    moments: _Moments,
    hole_types: Sequence[NT],
    edges: Sequence[float],
    bins: Sequence[int],
) -> list[tuple[int, float]]:
    """Estimate the number of a node's completions in each of some bins by the saddle point on the grid.

    Exact where the local form has nothing to say: nothing in a bin the node's costs do not reach, every completion in
    the bin holding its one cost, on a lattice the completions of its least or greatest cost in a bin holding only
    that, and off a lattice those of its greatest cost in a bin that begins there. Otherwise the bin is read at the grid
    tilt whose mean for this node lies nearest the middle of the bin's reach within the node's costs, and the local
    form integrated as :func:`saddle_counts` integrates it: over the whole cells of the bin's lattice points, or off a
    lattice over the bin's reach.

    Args:
        grid (SaddleGrid[NT]): The grid.
        moments (_Moments): The node's moments on it.
        hole_types (Sequence[NT]): The non-terminals of the node's open holes.
        edges (Sequence[float]): The bins' boundaries.
        bins (Sequence[int]): The bins asked for, by index.

    Returns:
        list[tuple[int, float]]: The bins the node's costs reach, each with the log of its estimated count.
    """
    program = grid.program
    least, greatest, log_excess, mean_excess, variance = moments
    spacing = (
        math.gcd(*(round(program.spacing[hole] / program.unit) for hole in hole_types)) * program.unit
        if program.unit and hole_types
        else 0.0
    )
    counts: list[tuple[int, float]] = []
    for index in bins:
        if edges[index] > greatest or edges[index + 1] <= least:
            continue
        if least == greatest:
            counts.append((index, math.log(math.prod(program.counts[hole] for hole in hole_types))))
            continue
        if spacing:
            first = least + spacing * max(0, math.ceil((edges[index] - least) / spacing))
            last = least + spacing * min(
                (greatest - least) // spacing, math.ceil((edges[index + 1] - least) / spacing) - 1
            )
            if first > last:
                continue
            low, high = first - spacing / 2, last + spacing / 2
            only = first if first == last and first in (least, greatest) else None
        else:
            low, high = max(edges[index], least), min(edges[index + 1], greatest)
            only = low if low == high else None
        if only is not None:
            extreme = program.cheapest_counts if only == least else program.dearest_counts
            counts.append((index, math.log(math.prod(extreme[hole] for hole in hole_types))))
            continue
        # The local form in excess of the node's cheapest cost, where the products of the tilt and the costs are small.
        tilt = _nearest_tilt(mean_excess, variance, (max(low, least) + min(high, greatest)) / 2 - least)
        low, high = low - least, high - least
        theta, mass, mean, spread = grid.thetas[tilt], log_excess[tilt], mean_excess[tilt], variance[tilt]
        if spread > 0:
            deviation = math.sqrt(spread)
            shift = mean + theta * spread
            log_count = (
                mass
                + theta * mean
                + theta * theta * spread / 2
                + _log_gaussian_interval((low - shift) / deviation, (high - shift) / deviation)
            )
        else:
            log_count = mass + theta * mean if low <= mean <= high else -math.inf
        if log_count > -math.inf:
            counts.append((index, log_count))
    return counts


# A node of the saddle search's lazy frontier: the tilted search's three, and the non-terminals of its open holes with
# its moments on the grid, which its children's are computed from.
_SaddleNode = tuple[Any, float, Any, tuple[Any, ...], Any]


@dataclass(frozen=True)
class SaddleSearch(Generic[NT, T, G]):
    """Random search to a target on cost bins, each node weighed by the saddle point's estimate of its completions.

    The target spreads each bin's share evenly over the bin's terms, ``P(t) = target(b) / N(b)``, so a node's share is
    ``w(n) = sum_b rho(b) N_n(b)``, ``rho(b) = target(b) / N(b)``, ``N_n(b)`` the number of its completions in the bin.
    With exact counts random search on ``w`` draws exactly that, without rejection. Here each ``N_n(b)`` is the saddle
    point's, read on a grid of tilts from the node's moments, which add over its holes, and ``N(b)`` the sum of the
    initial nodes' estimates, so that the initial nodes' weights sum to one. Siblings' estimates need not sum to their
    parent's: random search chooses among siblings in proportion to theirs, and the draws then follow the target only
    as far as the estimates are right. Every draw is the first term of a fresh stream.

    Attributes:
        query (ResolutionQuery[NT, T, G]): The query being sampled from.
        grid (SaddleGrid[NT]): The tilted moments of every non-terminal on the grid.
        edges (tuple[float, ...]): The bins' boundaries; bin ``i`` holds the costs in ``[edges[i], edges[i + 1])``.
        target (tuple[float, ...]): The target's mass per bin, normalized over the bins the estimate reaches; a bin
            the estimate reaches but no term lies in keeps its share, which no draw then carries.
        log_root_counts (tuple[float, ...]): The log of the estimated number of the query's terms per bin, ``-inf``
            for a bin left out or not reached.
        log_rho (tuple[float, ...]): Per bin, ``log(target / N)``: the log weight of one of its terms; ``-inf``
            where the bin has no target or no term.
        missing_target (float): The share of the target on bins without an estimate, which no draw reaches; the
            share of a bin with an estimate but no term is not in it.
        subgoal_selection (Callable | None): The computation rule, which must select an open hole; None selects
            the engine's deepest-first rule.
    """

    query: ResolutionQuery[NT, T, G]
    grid: SaddleGrid[NT]
    edges: tuple[float, ...]
    target: tuple[float, ...]
    log_root_counts: tuple[float, ...]
    log_rho: tuple[float, ...]
    missing_target: float
    subgoal_selection: Callable[[Goal[NT, T, G]], tuple[Path, NonTerminalArgument[NT]]] | None = None

    def log_weight(self, moments: _Moments, hole_types: Sequence[NT]) -> float:
        """Return the log of a node's share of the target: ``log sum_b rho(b) N_n(b)`` over the estimated ``N_n``.

        Args:
            moments (_Moments): The node's moments on the grid.
            hole_types (Sequence[NT]): The non-terminals of its open holes.

        Returns:
            float: The log weight, ``-inf`` for a node with no completion in a bin with a target.
        """
        bins = [index for index, log_rho in enumerate(self.log_rho) if log_rho > -math.inf]
        counts = _node_log_counts(self.grid, moments, hole_types, self.edges, bins)
        if not counts:
            return -math.inf
        return log_sum_exp([self.log_rho[index] + log_count for index, log_count in counts])

    def _root_children(self) -> list[tuple[_SaddleNode, float]]:
        """Return the query's initial nodes with their log weights, computed once for every stream of this search.

        Returns:
            list: The initial nodes that have a completion in a bin with a target, with their log weights.
        """
        known = self.__dict__.get("_root_children_cache")
        if known is None:
            known = []
            program = self.grid.program
            for child, child_cost in _initial_tilt_nodes(self.query, program.algebra):
                hole_types = tuple(holes(child).values())
                if not all(hole in program.cheapest for hole in hole_types):
                    continue
                moments = _node_moments(self.grid, child_cost, hole_types)
                log_weight = self.log_weight(moments, hole_types)
                if log_weight > -math.inf:
                    known.append(((child, child_cost, None, hole_types, moments), log_weight))
            object.__setattr__(self, "_root_children_cache", known)
        return known

    def keyed_stream(self, rng: random.Random) -> Iterator[tuple[float, Tree[T]]]:
        """Run random search on the estimated weights, keeping each term's key.

        Its first term is a draw in proportion to the weights along its path; a node whose estimate is positive while
        no term lies below it in a bin with a target is passed over, and the search goes on with the next key. The
        later terms are an enumeration of the rest, in the order of keys conditioned on estimates that need not agree
        with each other.

        Args:
            rng (random.Random): The source of randomness.

        Yields:
            tuple[float, Tree[T]]: The key and the term, in decreasing key order.

        Raises:
            ValueError: If the computation rule selects a position that is not an open hole.
        """
        select = deepest_first_subgoal if self.subgoal_selection is None else self.subgoal_selection
        space = self.query.solution_space
        program = self.grid.program
        rule_costs: dict[int, float] = {}

        def expand(node: _SaddleNode) -> tuple[Tree[T] | None, Sequence[tuple[_SaddleNode, float]]]:
            """Expand one node, and weigh its children by their moments without building them.

            Args:
                node (_SaddleNode): The goal (None at the root or while unbuilt), the cost of its partial
                    inhabitant, while unbuilt the parent, position and clause that build it, and its holes' non-terminals
                    with its moments.

            Returns:
                tuple: The term (None on an inner node) and the retained children with their log weights.

            Raises:
                ValueError: If the computation rule selects a position that is not an open hole.
            """
            goal, cost, pending, _hole_types, _moments = node
            if pending is not None:
                goal = _built(pending)
            if goal is not None and goal.success:
                return goal.grounded[()][1], ()
            if goal is None:
                return None, self._root_children()
            position, argument = select(goal)
            open_holes = holes(goal)
            if position not in open_holes:
                msg = (
                    f"the computation rule selected position {position}, which is not an open hole: the saddle "
                    "search weighs a node by its open holes, so the rule must select one, as deepest_first_subgoal does"
                )
                raise ValueError(msg)
            others = [hole_type for hole_position, hole_type in open_holes.items() if hole_position != position]
            base = _node_moments(self.grid, cost, others)
            kept: list[tuple[_SaddleNode, float]] = []
            for rule in space.get(argument.origin) or ():
                if not _admitted(rule):
                    continue
                opened = [arg.origin for arg in rule.arguments if isinstance(arg, NonTerminalArgument)]
                if not all(hole in program.cheapest for hole in opened):
                    continue
                clause_cost = rule_costs.get(id(rule))
                if clause_cost is None:
                    clause_cost = _real_rule_cost(rule, program.algebra)
                    rule_costs[id(rule)] = clause_cost
                child_holes = (*others, *opened)
                child_moments = _with_holes(self.grid, base, opened, cost + clause_cost, child_holes)
                log_weight = self.log_weight(child_moments, child_holes)
                if log_weight > -math.inf:
                    kept.append(
                        ((None, cost + clause_cost, (goal, position, rule), child_holes, child_moments), log_weight)
                    )
            return None, kept

        yield from keyed_stream((None, 0.0, None, (), None), 0.0, expand, rng)

    def stream(self, rng: random.Random, max_draws: int) -> Iterator[Tree[T]]:
        """Draw terms: the first term of a fresh stream each, a term drawn before skipped, until the draws run out.

        The stream ends early where a fresh stream finds no term: then no term lies in a bin with a target and an
        estimate, and none ever will.

        Args:
            rng (random.Random): The source of randomness.
            max_draws (int): How many draws to make, repeats included, a whole number not below 0; a language whose
                target bins are exhausted draws only repeats, so the bound is not optional.

        Yields:
            Tree[T]: The terms, each once.

        Raises:
            ValueError: If ``max_draws`` is not a whole number not below 0.
        """
        if isinstance(max_draws, bool) or not isinstance(max_draws, int) or max_draws < 0:
            msg = f"max_draws must be a whole number not below 0, not {max_draws!r}"
            raise ValueError(msg)
        streamed: set[Tree[T]] = set()
        for _ in range(max_draws):
            drawn = next(self.keyed_stream(rng), None)
            if drawn is None:
                return
            term = drawn[1]
            if term in streamed:
                continue
            streamed.add(term)
            yield term


def saddle_search(
    query: ResolutionQuery[NT, T, G],
    algebra: AdditiveCostAlgebra[Any],
    edges: Sequence[float],
    target: Sequence[float],
    *,
    program: TiltProgram[NT] | None = None,
    thetas: Iterable[float] | None = None,
    least_share: float = 0.0,
    subgoal_selection: Callable[[Goal[NT, T, G]], tuple[Path, NonTerminalArgument[NT]]] | None = None,
) -> SaddleSearch[NT, T, G]:
    """Build random search to a target on cost bins, each node weighed by the saddle point's estimate.

    Args:
        query (ResolutionQuery[NT, T, G]): The query to sample from, generator or partial-term.
        algebra (AdditiveCostAlgebra[Any]): The additive cost algebra, with finite real costs.
        edges (Sequence[float]): The bins' boundaries, strictly ascending finite reals, at least two.
        target (Sequence[float]): The target's mass per bin, one fewer than the edges, nonnegative, not all zero.
        program (TiltProgram[NT] | None): The query's program already prepared under this algebra. (Default value = None)
        thetas (Iterable[float] | None): The grid of tilts; None reads the tilts :func:`saddle_counts` finds for the
            query's bins with a target, and 0. (Default value = None)
        least_share (float): Bins whose share of the target is below this, a real number from 0 to 1, are left out,
            their share reported as out of reach. (Default value = 0.0)
        subgoal_selection (Callable | None): The computation rule, which must select an open hole. (Default value = None)

    Returns:
        SaddleSearch[NT, T, G]: The construction, ready to stream from.

    Raises:
        ValueError: If the edges, the target, the least share or a tilt are not of the kinds above; where
            :func:`saddle_counts` or :func:`saddle_grid` refuses; or if no bin with a target at or above the least
            share has an estimated term of the query.
    """
    edges = _checked_edges(edges)
    checked = [_real_cost(mass, "a target mass") for mass in target]
    if len(checked) != len(edges) - 1 or any(mass < 0 for mass in checked) or sum(checked) <= 0:
        msg = "the target needs one nonnegative mass per bin, one fewer than the edges, and not all of them zero"
        raise ValueError(msg)
    if isinstance(least_share, bool) or not isinstance(least_share, (int, float)) or not 0 <= least_share <= 1:
        msg = f"the least share must be a real number between 0 and 1, not {least_share!r}"
        raise ValueError(msg)
    prepared = _prepared(query, algebra, program)
    total = sum(checked)
    wanted = [index for index, mass in enumerate(checked) if mass > 0 and mass / total >= least_share]
    if thetas is None:
        root = saddle_counts(query, algebra, edges, program=prepared, only=wanted)
        thetas = [theta for theta in root.thetas if not math.isnan(theta)] + [0.0]
    grid = saddle_grid(prepared, thetas)
    per_bin: list[list[float]] = [[] for _ in checked]
    for child, child_cost in _initial_tilt_nodes(query, algebra):
        hole_types = tuple(holes(child).values())
        if all(hole in prepared.cheapest for hole in hole_types):
            moments = _node_moments(grid, child_cost, hole_types)
            for index, log_count in _node_log_counts(grid, moments, hole_types, edges, wanted):
                per_bin[index].append(log_count)
    log_root_counts = tuple(log_sum_exp(values) if values else -math.inf for values in per_bin)
    reached = [mass if log_root_counts[index] > -math.inf else 0.0 for index, mass in enumerate(checked)]
    if sum(reached) <= 0:
        msg = "no bin with a target at or above the least share has an estimated term of the query"
        raise ValueError(msg)
    normalized = tuple(mass / sum(reached) for mass in reached)
    return SaddleSearch(
        query=query,
        grid=grid,
        edges=tuple(edges),
        target=normalized,
        log_root_counts=log_root_counts,
        log_rho=tuple(
            math.log(mass) - log_count if mass > 0 else -math.inf
            for mass, log_count in zip(normalized, log_root_counts, strict=True)
        ),
        missing_target=math.fsum(
            mass for mass, log_count in zip(checked, log_root_counts, strict=True) if log_count == -math.inf
        )
        / math.fsum(checked),
        subgoal_selection=subgoal_selection,
    )
