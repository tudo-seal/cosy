"""Counting by rounding: an additive cost counted exactly in units, its counts per bin bounded, its target reached by rejection.

Approximate counting with a guarantee, transferred from sequences of items to tree grammars. Stefankovic, Vempala and Vigoda
count the solutions of a knapsack within ``1 +- epsilon`` by a recursion over the items on the inverse counting function at
geometric levels of the count; the recursion is a sum of two shifted functions per item, and a tree grammar's clause, whose
cost is the sum of its holes' costs, needs a convolution instead, which that recursion does not contain. The approach they
cite before theirs, Dyer's, transfers: scale the costs down, round them, count the rounded problem exactly, and reject to the
original. An additive cost rounded symbol by symbol is again additive, so the cost table (:mod:`cosy.search.cost_tables`)
counts it exactly, without a size bound; and the rounding error of a term is itself an additive cost.

**The coarse and the fine algebra.** The caller gives the FINE algebra, whose cost the target is about, a COARSE one, and the
UNIT that relates them, all three in whole numbers: a term's fine cost is ``unit`` times its coarse cost plus its rounding error
``e(t)``, the sum of its symbols' errors ``c(F) - unit * c_coarse(F)``. Whole numbers keep every one of these sums exact, so
a term's cost and the interval it is placed in are computed alike; fractional fine costs are refused. The least and the
greatest error over the query's terms, ``E_min`` and ``E_max``, are the min and max recursions of that additive error over the
program. So every term of coarse cost ``k`` has its fine cost in ``[unit k + E_min, unit k + E_max]``.

**The bounds.** For a bin ``b`` of fine costs, the terms of coarse cost ``k`` all lie in ``b`` where that interval lies inside
it, and none of them does where the interval misses it. So the coarse table's exact counts bound the fine counts per bin:

    N_lo(b) = sum over k with [unit k + E_min, unit k + E_max] inside b of N_coarse(k)  <=  N(b)
            <=  sum over k with that interval meeting b of N_coarse(k) = N_hi(b).

Both are counted exactly; neither is estimated. They are tight where the coarse values' intervals lie inside the bins, which a
unit small against the bins' widths and the number of rounded symbols per term make the rule; a bin no interval fits inside
has a lower bound of zero.

**The sampler.** The target spreads a bin's mass over its terms by counts ``N-hat(b)`` the caller chooses inside the bounds,
``p(t)`` proportional to ``target(b) / N-hat(b)``. It is reached by rejection from the coarse table's random search: a term of
coarse cost ``k`` is proposed in proportion to the largest ``target(b) / N-hat(b)`` over the bins its interval meets, its
ceiling, and accepted with its own bin's value over that ceiling, which is one wherever the interval lies inside a bin. Every
draw is the first term of a fresh stream, so the accepted terms are independent and distributed exactly as ``p``: a bin's
terms alike, the bins in proportion to ``target(b) N(b) / N-hat(b)``. Skipping the terms drawn before makes the stream a
sample without replacement in proportion to ``p``. With ``N-hat`` inside the bounds, each bin's factor ``N(b) / N-hat(b)``
lies between ``N_lo(b) / N-hat(b)`` and ``N_hi(b) / N-hat(b)``: the bounds give it, and it is unbounded above where a bin's
lower bound is zero.

The weights are kept in log space throughout, so counts of any size are drawn from. The program's language must be finite, as
the tilt's and the cost table's are, and every statement is about derivations, which are the terms under unambiguity.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

from cosy.core.solution_space import NT, G, T
from cosy.search.cost_tables import WeightedCostTable, _whole_cap, cost_table, weighted_cost_table
from cosy.search.sampling import log_sum_exp
from cosy.search.tilt import TargetOutOfReach, _checked_edges, _initial_holes, _real_cost, tilt_program

if TYPE_CHECKING:
    import random
    from collections.abc import Iterator, Sequence

    from cosy.core.tree import Tree
    from cosy.search.cost_tables import CostTable
    from cosy.search.costs import AdditiveCostAlgebra
    from cosy.search.queries import ResolutionQuery

__all__ = ["BinBounds", "RoundedRejection", "bin_count_bounds", "rounded_rejection"]


@dataclass(frozen=True)
class BinBounds:
    """Two-sided bounds on the number of a query's terms per bin of a fine cost, from the exact counts of a coarse one.

    Attributes:
        edges (tuple[float, ...]): The bins' boundaries of the fine cost; bin ``i`` holds ``[edges[i], edges[i + 1])``.
        lower (tuple[int, ...]): Per bin, the number of terms whose coarse cost puts them inside the bin for certain.
        upper (tuple[int, ...]): Per bin, the number of terms whose coarse cost lets them fall into the bin.
        error_range (tuple[int, int]): ``E_min`` and ``E_max``, the least and the greatest rounding error of the query's
            terms, fine cost less ``unit`` times coarse cost.
        unit (int): The fine cost one unit of the coarse cost stands for.
    """

    edges: tuple[float, ...]
    lower: tuple[int, ...]
    upper: tuple[int, ...]
    error_range: tuple[int, int]
    unit: int

    def interval_of(self, coarse: int) -> tuple[int, int]:
        """Return the fine costs a term of a coarse cost can have, whole numbers computed exactly.

        Args:
            coarse (int): The coarse cost.

        Returns:
            tuple[int, int]: ``[unit k + E_min, unit k + E_max]``.
        """
        return self.unit * coarse + self.error_range[0], self.unit * coarse + self.error_range[1]

    def bins_met(self, coarse: int) -> list[int]:
        """Return the bins the fine costs of a coarse cost's terms can fall into.

        Args:
            coarse (int): The coarse cost.

        Returns:
            list[int]: The indices of the bins its interval meets, in order.
        """
        low, high = self.interval_of(coarse)
        first = max(0, bisect_right(self.edges, low) - 1)
        met = []
        for index in range(first, len(self.edges) - 1):
            if self.edges[index] > high:
                break
            if self.edges[index + 1] > low:
                met.append(index)
        return met

    def inside(self, coarse: int) -> int | None:
        """Return the bin that holds every fine cost a coarse cost's terms can have, if one does.

        Args:
            coarse (int): The coarse cost.

        Returns:
            int | None: The bin's index, None where the interval meets two bins, or none.
        """
        low, high = self.interval_of(coarse)
        index = bisect_right(self.edges, low) - 1
        if 0 <= index < len(self.edges) - 1 and high < self.edges[index + 1]:
            return index
        return None


def _whole(value: Any, what: str) -> int:
    """Return a cost as a whole number, or refuse it.

    Args:
        value (Any): The cost.
        what (str): What it is, for the message.

    Returns:
        int: The cost.

    Raises:
        ValueError: If the cost is not a finite whole number.
    """
    real = _real_cost(value, what)
    if not real.is_integer():
        msg = f"counting by rounding takes whole-number costs, so that a term's cost and its interval are computed alike, but {what} is {value!r}"
        raise ValueError(msg)
    return int(real)


def _ranges(
    query: ResolutionQuery[NT, T, G], fine: AdditiveCostAlgebra[Any], coarse: AdditiveCostAlgebra[Any], unit: int
) -> tuple[int, int, int]:
    """Return the least and the greatest rounding error over the query's terms, and the greatest coarse cost of one.

    The error is additive, a clause's being its fine cost less ``unit`` times its coarse one, but it takes both signs, which
    no cost domain holds; so its least and greatest values are found by the min and max recursion over the two programs the
    tilt prepares, whose clauses are the same clauses in the same order, the one with fine costs, the other with coarse.

    Args:
        query (ResolutionQuery[NT, T, G]): The query.
        fine (AdditiveCostAlgebra[Any]): The fine algebra, with whole-number costs.
        coarse (AdditiveCostAlgebra[Any]): The coarse algebra, with whole-number costs.
        unit (int): The fine cost of one coarse unit.

    Returns:
        tuple[int, int, int]: ``E_min``, ``E_max`` and the query's greatest coarse cost.

    Raises:
        ValueError: If a cost is not a whole number; if the query has no term (as :class:`TargetOutOfReach`); where the
            tilt refuses the program (a language with infinitely many terms, a predicate that reads a hole).
    """
    fine_program = tilt_program(query.solution_space, fine)
    coarse_program = tilt_program(query.solution_space, coarse, check=False)
    least: dict[Any, int] = {}
    greatest: dict[Any, int] = {}
    dearest: dict[Any, int] = {}
    for nonterminal in fine_program.order:
        lows, highs, dears = [], [], []
        for (fine_cost, holes_fine), (coarse_cost, holes_coarse) in zip(
            fine_program.clauses[nonterminal], coarse_program.clauses[nonterminal], strict=True
        ):
            if holes_fine != holes_coarse:  # pragma: no cover - the two programs prune and order alike
                msg = "the fine and the coarse program disagree about a clause's holes"
                raise ValueError(msg)
            fine_whole = _whole(fine_cost, "a clause's fine cost")
            coarse_whole = _whole(coarse_cost, "a clause's coarse cost")
            error = fine_whole - unit * coarse_whole
            lows.append(error + sum(least[hole] for hole in holes_fine))
            highs.append(error + sum(greatest[hole] for hole in holes_fine))
            dears.append(coarse_whole + sum(dearest[hole] for hole in holes_fine))
        least[nonterminal], greatest[nonterminal], dearest[nonterminal] = min(lows), max(highs), max(dears)
    low: int | None = None
    high: int | None = None
    dear: int | None = None
    for (fine_cost, holes_fine), (coarse_cost, _holes) in zip(
        _initial_holes(query, fine), _initial_holes(query, coarse), strict=True
    ):
        if all(hole in least for hole in holes_fine):
            error = _whole(fine_cost, "the query term's fine cost") - unit * _whole(
                coarse_cost, "the query term's coarse cost"
            )
            node_low = error + sum(least[hole] for hole in holes_fine)
            node_high = error + sum(greatest[hole] for hole in holes_fine)
            node_dear = _whole(coarse_cost, "the query term's coarse cost") + sum(dearest[hole] for hole in holes_fine)
            low = node_low if low is None else min(low, node_low)
            high = node_high if high is None else max(high, node_high)
            dear = node_dear if dear is None else max(dear, node_dear)
    if low is None or high is None or dear is None:
        msg = "the query has no term, so there is nothing to bound"
        raise TargetOutOfReach(msg)
    return low, high, dear


def _checked_unit(unit: Any) -> int:
    """Return the unit as a positive whole number, or refuse it.

    Args:
        unit (Any): The unit a caller passed.

    Returns:
        int: The unit.

    Raises:
        ValueError: If the unit is not a positive whole number.
    """
    if isinstance(unit, bool) or not isinstance(unit, (int, float)) or not math.isfinite(unit) or unit <= 0:
        whole = None
    else:
        whole = int(unit) if float(unit).is_integer() else None
    if whole is None:
        msg = f"the unit is the fine cost of one coarse unit, a positive whole number, not {unit!r}"
        raise ValueError(msg)
    return whole


def _needed(edges: Sequence[float], unit: int, least_error: int, dearest: int) -> int:
    """Return the greatest coarse cost a term in some bin can have: what the cost table must count up to.

    The last edge is exclusive, so a coarse value whose interval starts at it holds no term of a bin; and no term costs more
    than the query's dearest. Computed in whole numbers against the edge, exactly.

    Args:
        edges (Sequence[float]): The bins' boundaries.
        unit (int): The unit.
        least_error (int): ``E_min``.
        dearest (int): The query's greatest coarse cost.

    Returns:
        int: The coarse cost the cap must reach.
    """
    last = edges[-1]
    k = math.floor((last - least_error) / unit)
    while k >= 0 and unit * k + least_error >= last:
        k -= 1
    while unit * (k + 1) + least_error < last:
        k += 1
    return max(0, min(k, dearest))


def bin_count_bounds(
    query: ResolutionQuery[NT, T, G],
    fine: AdditiveCostAlgebra[Any],
    coarse: AdditiveCostAlgebra[Any],
    unit: int,
    edges: Sequence[float],
    cost_cap: int,
    *,
    table: CostTable[NT] | None = None,
) -> BinBounds:
    """Bound the number of a query's terms per bin of a fine cost by the exact counts of a coarse one.

    Args:
        query (ResolutionQuery[NT, T, G]): The query, of a program whose language is finite.
        fine (AdditiveCostAlgebra[Any]): The algebra whose cost the bins are of, with whole-number costs.
        coarse (AdditiveCostAlgebra[Any]): The coarse algebra, with whole-number costs, counted by the cost table.
        unit (int): The fine cost one unit of the coarse cost stands for, a positive whole number.
        edges (Sequence[float]): The bins' boundaries of the fine cost, strictly ascending finite reals, at least two.
        cost_cap (int): The largest coarse cost the table counts; it must reach every coarse cost a term in a bin can have
            (the greatest whose interval starts below the last edge, or the query's dearest, whichever is less).
        table (CostTable[NT] | None): A cost table of the query's program under the coarse algebra, filled to the cap.
            (Default value = None)

    Returns:
        BinBounds: The bounds per bin.

    Raises:
        ValueError: If the unit is not a positive whole number, the edges not strictly ascending, a cost not a whole number,
            or the cap below the coarse costs the bins can hold; if the query has no term (as :class:`TargetOutOfReach`);
            where the cost table or the tilt refuses the program.
    """
    bounds, _filled, _root = _counted(query, fine, coarse, unit, edges, cost_cap, table)
    return bounds


def _counted(
    query: ResolutionQuery[NT, T, G],
    fine: AdditiveCostAlgebra[Any],
    coarse: AdditiveCostAlgebra[Any],
    unit: Any,
    edges: Sequence[float],
    cost_cap: Any,
    table: CostTable[NT] | None,
) -> tuple[BinBounds, CostTable[NT], dict[int, int]]:
    """Check the arguments, then fill the coarse table once and bound the fine counts per bin from it.

    Args:
        query (ResolutionQuery[NT, T, G]): The query.
        fine (AdditiveCostAlgebra[Any]): The fine algebra.
        coarse (AdditiveCostAlgebra[Any]): The coarse algebra.
        unit (Any): The unit.
        edges (Sequence[float]): The bins' boundaries.
        cost_cap (Any): The cap.
        table (CostTable[NT] | None): A filled table, or None.

    Returns:
        tuple: The bounds, the table, and the query's counts per coarse cost.

    Raises:
        ValueError: As :func:`bin_count_bounds`.
    """
    whole_unit = _checked_unit(unit)
    checked = _checked_edges(edges)
    cap = _whole_cap(cost_cap)
    least, greatest, dearest = _ranges(query, fine, coarse, whole_unit)
    needed = _needed(checked, whole_unit, least, dearest)
    if cap < needed:
        msg = (
            f"the cost cap {cap} does not reach the coarse costs a term in the bins can have (up to {needed}); the terms "
            "above the cap would be missing from the bounds"
        )
        raise ValueError(msg)
    filled = cost_table(query.solution_space, coarse, cap) if table is None else table
    root = _root_counts(query, coarse, cap, filled)
    return _bounds(root, checked, whole_unit, (least, greatest)), filled, root


def _bounds(root: dict[int, int], edges: list[float], unit: int, error_range: tuple[int, int]) -> BinBounds:
    """Return the bounds per bin from the coarse counts.

    Args:
        root (dict[int, int]): The query's number of terms per coarse cost.
        edges (list[float]): The bins' boundaries.
        unit (int): The unit.
        error_range (tuple[int, int]): ``E_min`` and ``E_max``.

    Returns:
        BinBounds: The bounds.
    """
    probe = BinBounds(edges=tuple(edges), lower=(), upper=(), error_range=error_range, unit=unit)
    lower = [0] * (len(edges) - 1)
    upper = [0] * (len(edges) - 1)
    for value, count in root.items():
        for index in probe.bins_met(value):
            upper[index] += count
        holder = probe.inside(value)
        if holder is not None:
            lower[holder] += count
    return BinBounds(edges=probe.edges, lower=tuple(lower), upper=tuple(upper), error_range=error_range, unit=unit)


def _root_counts(
    query: ResolutionQuery[NT, T, G], coarse: AdditiveCostAlgebra[Any], cap: int, table: CostTable[NT]
) -> dict[int, int]:
    """Return the number of the query's terms per coarse cost up to the cap, from the table.

    Args:
        query (ResolutionQuery[NT, T, G]): The query.
        coarse (AdditiveCostAlgebra[Any]): The coarse algebra.
        cap (int): The cap.
        table (CostTable[NT]): The cost table.

    Returns:
        dict[int, int]: The counts per coarse cost.
    """
    return dict(weighted_cost_table(query, coarse, _one, cap, table=table).root_counts)


def _one(_value: Any) -> float:
    """Weigh every coarse value alike; used where only the counts are read.

    Args:
        _value (Any): The coarse value. Ignored.

    Returns:
        float: One.
    """
    return 1.0


@dataclass(frozen=True)
class RoundedRejection(Generic[NT, T, G]):
    """Rejection from the coarse table's random search to a target on bins of the fine cost.

    Attributes:
        weighted (WeightedCostTable[NT, T, G]): The coarse table's random search, a term of coarse cost ``k`` weighed in
            proportion to its ceiling; the values no bin with a target is in reach of carry no weight.
        fine (AdditiveCostAlgebra[Any]): The fine algebra.
        bounds (BinBounds): The bounds on the fine counts per bin.
        log_rho (tuple[float, ...]): Per bin, ``log(target / N-hat)``; ``-inf`` for a bin without a target.
        log_ceiling (dict[int, float]): Per coarse value with a target in reach, the largest ``log_rho`` over the bins its
            interval meets.
    """

    weighted: WeightedCostTable[NT, T, G]
    fine: AdditiveCostAlgebra[Any]
    bounds: BinBounds
    log_rho: tuple[float, ...]
    log_ceiling: dict[int, float]

    def bin_of(self, cost: float) -> int | None:
        """Return the bin a fine cost falls in.

        Args:
            cost (float): The fine cost.

        Returns:
            int | None: The bin's index, None outside the edges.
        """
        index = bisect_right(self.bounds.edges, cost) - 1
        return index if 0 <= index < len(self.bounds.edges) - 1 else None

    def log_acceptance(self, term: Tree[T]) -> float:
        """Return the log of the probability of accepting a drawn term.

        Args:
            term (Tree[T]): A term drawn from the coarse table.

        Returns:
            float: ``log rho(b(t))`` less its coarse value's ceiling, at most 0; ``-inf`` outside the target.

        Raises:
            ValueError: If the term's bin lies outside the bins its coarse value's interval meets, which Lemma 1 rules out:
                the bounds would not hold either, and the draw is not one the construction can weigh.
        """
        index = self.bin_of(_whole(self.fine.fold(term), "a term's fine cost"))
        if index is None or self.log_rho[index] == -math.inf:
            return -math.inf
        coarse = _whole(self.weighted.algebra.fold(term), "a term's coarse cost")
        ceiling = self.log_ceiling.get(coarse)
        if ceiling is None or index not in self.bounds.bins_met(coarse):
            msg = (
                f"a term of fine cost {self.fine.fold(term)!r} and coarse cost {coarse} lies outside the interval its coarse "
                "cost was given; the error range does not hold for it"
            )
            raise ValueError(msg)
        return self.log_rho[index] - ceiling

    def stream(self, rng: random.Random, max_draws: int) -> Iterator[Tree[T]]:
        """Draw terms: the first term of a fresh stream each, accepted in proportion, a term drawn before skipped.

        Args:
            rng (random.Random): The source of randomness.
            max_draws (int): How many draws to make at most, accepted or not, a whole number not below 0.

        Yields:
            Tree[T]: The accepted terms, each once.

        Raises:
            ValueError: If ``max_draws`` is not a whole number not below 0.
        """
        if isinstance(max_draws, bool) or not isinstance(max_draws, int) or max_draws < 0:
            msg = f"max_draws must be a whole number not below 0, not {max_draws!r}"
            raise ValueError(msg)
        streamed: set[Tree[T]] = set()
        for _ in range(max_draws):
            drawn = next(self.weighted.keyed_stream(rng), None)
            if drawn is None:
                return
            term = drawn[1]
            log_acceptance = self.log_acceptance(term)
            if log_acceptance == -math.inf or term in streamed:
                continue
            if log_acceptance < 0 and rng.random() >= math.exp(log_acceptance):
                continue
            streamed.add(term)
            yield term


def rounded_rejection(
    query: ResolutionQuery[NT, T, G],
    fine: AdditiveCostAlgebra[Any],
    coarse: AdditiveCostAlgebra[Any],
    unit: int,
    edges: Sequence[float],
    target: Sequence[float],
    cost_cap: int,
    *,
    log_counts: Sequence[float] | None = None,
    table: CostTable[NT] | None = None,
) -> RoundedRejection[NT, T, G]:
    """Build rejection from the coarse table to a target on bins of the fine cost, the counts per bin inside their bounds.

    Args:
        query (ResolutionQuery[NT, T, G]): The query, of a program whose language is finite.
        fine (AdditiveCostAlgebra[Any]): The algebra whose cost the bins are of, with whole-number costs.
        coarse (AdditiveCostAlgebra[Any]): The coarse algebra, with whole-number costs.
        unit (int): The fine cost of one coarse unit, a positive whole number.
        edges (Sequence[float]): The bins' boundaries of the fine cost.
        target (Sequence[float]): The target's mass per bin, one fewer than the edges, nonnegative, not all zero.
        cost_cap (int): The largest coarse cost the table counts (:func:`bin_count_bounds`).
        log_counts (Sequence[float] | None): The log of the counts per bin ``N-hat`` the target is spread by, each inside its
            bounds and finite where a bin with a target may hold a term; None takes the geometric mean of the bounds, the
            upper bound where the lower is zero. (Default value = None)
        table (CostTable[NT] | None): A cost table of the query's program under the coarse algebra, filled to the cap.
            (Default value = None)

    Returns:
        RoundedRejection[NT, T, G]: The construction, ready to stream from.

    Raises:
        ValueError: Where :func:`bin_count_bounds` refuses; if the target is not of the kind above; if a given count lies
            outside its bounds, or is zero where a bin with a target may hold a term; if no bin with a target may hold a term
            (as :class:`TargetOutOfReach`).
    """
    whole_unit = _checked_unit(unit)
    checked = _checked_edges(edges)
    masses = [_real_cost(mass, "a target mass") for mass in target]
    if len(masses) != len(checked) - 1 or any(mass < 0 for mass in masses) or sum(masses) <= 0:
        msg = "the target needs one nonnegative mass per bin, one fewer than the edges, and not all of them zero"
        raise ValueError(msg)
    if log_counts is not None and len(log_counts) != len(masses):
        msg = "the counts need one log count per bin"
        raise ValueError(msg)
    bounds, filled, root = _counted(query, fine, coarse, whole_unit, checked, cost_cap, table)
    chosen = _chosen_counts(bounds, masses, log_counts)
    log_rho = tuple(
        math.log(mass) - count if mass > 0 and count > -math.inf else -math.inf
        for mass, count in zip(masses, chosen, strict=True)
    )
    log_ceiling: dict[int, float] = {}
    for value in root:
        reachable = [log_rho[index] for index in bounds.bins_met(value) if log_rho[index] > -math.inf]
        if reachable:
            log_ceiling[value] = max(reachable)
    if not log_ceiling:
        msg = "no bin with a target may hold a term of the query"
        raise TargetOutOfReach(msg)
    # A term of coarse value k weighs its ceiling over the total, the values out of reach nothing: in log space, so that counts
    # of any size are weighed, and the out-of-reach values left out of the map the search skips.
    log_total = log_sum_exp([ceiling + math.log(root[value]) for value, ceiling in log_ceiling.items()])
    log_unit_weights = {value: ceiling - log_total for value, ceiling in log_ceiling.items()}
    weighted = WeightedCostTable(
        query=query,
        algebra=coarse,
        cost_cap=_whole_cap(cost_cap),
        table=filled,
        root_counts=root,
        unit_weights={value: math.exp(weight) for value, weight in log_unit_weights.items()},
        log_unit_weights=log_unit_weights,
    )
    return RoundedRejection(weighted=weighted, fine=fine, bounds=bounds, log_rho=log_rho, log_ceiling=log_ceiling)


def _chosen_counts(bounds: BinBounds, masses: list[float], log_counts: Sequence[float] | None) -> list[float]:
    """Return the log counts per bin the target is spread by: the given ones, checked, or the geometric mean of the bounds.

    Args:
        bounds (BinBounds): The bounds.
        masses (list[float]): The target per bin.
        log_counts (Sequence[float] | None): The given log counts, or None.

    Returns:
        list[float]: The log counts.

    Raises:
        ValueError: If a given count lies outside its bounds, or is zero where a bin with a target may hold a term.
    """
    if log_counts is None:
        return [
            0.5 * (math.log(low) + math.log(high)) if low > 0 else (math.log(high) if high > 0 else -math.inf)
            for low, high in zip(bounds.lower, bounds.upper, strict=True)
        ]
    chosen = [float(value) for value in log_counts]
    for index, value in enumerate(chosen):
        low, high = bounds.lower[index], bounds.upper[index]
        if masses[index] <= 0 or high == 0:
            continue
        if value == -math.inf or math.isnan(value):
            msg = f"the count given for bin {index} is zero, but its bounds [{low}, {high}] let it hold terms"
            raise ValueError(msg)
        if not ((math.log(low) if low > 0 else -math.inf) - 1e-12 <= value <= math.log(high) + 1e-12):
            msg = f"the count given for bin {index} lies outside its bounds [{low}, {high}]"
            raise ValueError(msg)
    return chosen
