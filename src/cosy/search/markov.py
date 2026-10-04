"""The Markov chain on terms: Metropolis-Hastings to a target on cost bins, a subtree regrown from its non-terminal.

Random search draws a term in proportion to a weight that it reads off counts: per cost value in the table forms, through a
tilt and an estimate per bin elsewhere. A Markov chain needs no count of the terms below a node at all. It walks from term
to term, and its stationary law is any weight it can evaluate on ONE term, up to a constant. The weight here is the target
of the programme spread over the terms of a bin,

    p(t)  proportional to  target(b) / N(b),    b the bin of c(t),

``N(b)`` the number of the query's terms the caller says the bin holds: exact, estimated by a pilot or the saddle point, or
learned along the chain. With the exact counts the bins carry the target and a bin's terms are alike; with others the bins
carry ``target(b) N(b) / N-hat(b)``, normalized, and a bin's terms are still alike.

**The state** is a term together with its derivation: the non-terminal at each of its positions at or below the query's
hole that a clause fills with a non-terminal, the positions ``m(t)`` counts. A term the chain draws brings its derivation
with it; a start the caller gives is parsed once.

**The move.** With probability ``root_share`` the whole term is regrown: the query is drawn once in proportion to
``e^(-theta c(t))``, its initial goals weighed by the tilt (:mod:`cosy.search.tilt`) and their holes filled from their
non-terminals. Otherwise one of the ``m(t)`` positions is chosen uniformly and the subtree there is regrown from the
non-terminal ``A`` the derivation has there, by the same top-down draw: each clause of a non-terminal in proportion to
``e^(-theta c(clause))`` times its holes' ``Z(theta)``, each hole filled the same way. The regrown term ``t'`` then comes
with probability ``e^(-theta c(t'))`` over ``e^(-theta c(context)) Z_A(theta)``. Under unambiguity the derivation of ``t'``
keeps the context's and has ``A`` at the position, so the move back regrows the same position from the same non-terminal,
the normalizer cancels in the Metropolis-Hastings ratio, and the acceptance needs the two terms alone:

    alpha = min(1, p(t') / p(t) * m(t) / m(t') * e^(theta (c(t') - c(t)))),

without the factor ``m(t) / m(t')`` for the move of the whole term. Detailed balance holds for each position, the position
being part of the proposal, and the mixture of the two kinds of move keeps it. A position a clause fills with a constant is
not one of the ``m(t)``: the clause above fixes the constant, which changes when the position above it is regrown.

Regrowing from the derivation's non-terminal rather than through the residual query at the position is a choice of cost:
the residual admits the completions of every derivation of the context, but finding its initial goals walks the whole
context, about 200 000 partial goals on a term of 813 nodes of a large program (nine seconds), where the draw from a
non-terminal costs the regrown subtree alone. The two kernels are both reversible for ``p``; this one proposes fewer terms
per position, and the move of the whole term still reaches every term.

**What the chain promises, and what it does not.** In the limit, the law ``p``: the visits to the terms converge to it,
for any start inside the target, the chain being irreducible on a finite language because the move of the whole term
reaches every term. Nothing about a prefix: the states are correlated, and a term recurs. How fast the chain mixes is
measured, not promised; every step reports its proposal and whether it was accepted. Every statement assumes an
unambiguous program, one derivation per term, which is what the chain keeps; on an ambiguous one no law is claimed.

**Where it applies** is where the tilt applies: an additive cost algebra with finite real costs, a program whose language is
finite and whose predicates read no hole (:func:`cosy.search.tilt.tilt_program`), generator and partial-term queries alike.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field, replace
from itertools import accumulate
from typing import TYPE_CHECKING, Any, Generic

from cosy.core.solution_space import NT, ConstantArgument, G, NonTerminalArgument, T
from cosy.core.tree import Tree
from cosy.search.counting import _admitted, decomposable_or_raise
from cosy.search.partial import holes
from cosy.search.sampling import log_sum_exp
from cosy.search.tilt import (
    TargetOutOfReach,
    _checked_edges,
    _initial_tilt_nodes,
    _prepared,
    _real_cost,
    _real_rule_cost,
    _real_theta,
)

if TYPE_CHECKING:
    import random
    from collections.abc import Iterator, Sequence

    from cosy.core.solution_space import RHSRule
    from cosy.core.tree import Path
    from cosy.search.costs import AdditiveCostAlgebra
    from cosy.search.queries import ResolutionQuery
    from cosy.search.tilt import TiltProgram, TiltTable

__all__ = [
    "ChainStep",
    "MetropolisChain",
    "VisitCounts",
    "WangLandau",
    "counts_from_visits",
    "metropolis_chain",
    "wang_landau",
]

# A spread needs two values: the fewest batches a run's error is read from, and the fewest a bin must be visited in.
_FEWEST_BATCHES = 2
# How many terms' derivations the chain keeps: the state's and the proposal's, with room for a caller's few.
_REMEMBERED = 8
# Stands for a hole when two terms are compared outside one position.
_HOLE: Tree[Any] = Tree("<hole>")

# The non-terminal at each position of a term at or below the query's hole that a clause fills with a non-terminal; None at
# the hole itself when a clause fills it with a constant.
Derivation = dict[tuple[int, ...], Any]


@dataclass(frozen=True)
class ChainStep(Generic[T]):
    """One step of the chain: the state after it, and what was proposed.

    Attributes:
        term (Tree[T]): The state after the step: the proposal if it was accepted, the state before it otherwise.
        proposal (Tree[T]): The proposed term.
        position (Path): The position whose subtree was regrown; the query's hole for a move of the whole term.
        root_move (bool): Whether the whole term was regrown.
        log_acceptance (float): The log of the acceptance probability, at most 0; ``-inf`` for a proposal outside the
            target.
        accepted (bool): Whether the proposal became the state.
    """

    term: Tree[T]
    proposal: Tree[T]
    position: Path
    root_move: bool
    log_acceptance: float
    accepted: bool


class _Memory:
    """What a chain computes once and shares with the copies ``with_log_counts`` makes: the clauses' laws, the query's
    initial goals, the index a parse reads, and the derivations of the last few terms."""

    def __init__(self) -> None:
        self.clauses: dict[Any, tuple[list[RHSRule[Any, Any, Any]], list[float]]] = {}
        self.goals: list[tuple[float, Any, Any]] | None = None
        self.index: dict[Any, dict[tuple[int, ...], dict[tuple[Any, ...], list[tuple[Any, Any]]]]] | None = None
        self.derivations: dict[Tree[Any], Derivation] = {}

    def remember(self, term: Tree[Any], derivation: Derivation) -> None:
        """Keep a term's derivation, forgetting the oldest beyond the last few.

        Args:
            term (Tree[Any]): The term.
            derivation (Derivation): Its derivation.
        """
        self.derivations.pop(term, None)
        self.derivations[term] = derivation
        while len(self.derivations) > _REMEMBERED:
            self.derivations.pop(next(iter(self.derivations)))


@dataclass(frozen=True)
class MetropolisChain(Generic[NT, T, G]):
    """A Metropolis-Hastings chain on a query's terms, to a target on cost bins, regrowing subtrees from their non-terminals.

    Attributes:
        query (ResolutionQuery[NT, T, G]): The query whose completions the chain walks on.
        table (TiltTable[NT]): The tilt table of the query's program at the chain's tilt, which every regrowth draws by.
        edges (tuple[float, ...]): The bins' boundaries; bin ``i`` holds the costs in ``[edges[i], edges[i + 1])``.
        log_target (tuple[float, ...]): Per bin, the log of the target's mass; ``-inf`` for a bin without a target.
        log_counts (tuple[float, ...]): Per bin, the log of the number of terms the chain takes the bin to hold; ``-inf``
            for a bin without a term.
        log_rho (tuple[float, ...]): Per bin, ``log(target / N)``: the log weight of one of its terms, up to a constant;
            ``-inf`` for a bin without a target or without a term.
        root_share (float): The probability of regrowing the whole term in a step, from 0 to 1.
    """

    query: ResolutionQuery[NT, T, G]
    table: TiltTable[NT]
    edges: tuple[float, ...]
    log_target: tuple[float, ...]
    log_counts: tuple[float, ...]
    log_rho: tuple[float, ...]
    root_share: float
    memory: _Memory = field(default_factory=_Memory, compare=False, repr=False)

    def with_log_counts(self, log_counts: Sequence[float]) -> MetropolisChain[NT, T, G]:
        """Return the same chain with other counts per bin, and so with another law.

        Args:
            log_counts (Sequence[float]): The log of the number of terms per bin, one per bin, a real number or ``-inf``.

        Returns:
            MetropolisChain[NT, T, G]: The chain on those counts, sharing what this one computed once.

        Raises:
            ValueError: If the counts are not one per bin of the kinds above, or no bin has both a target and a term.
        """
        counts = _checked_log_counts(log_counts, len(self.log_target))
        return replace(self, log_counts=counts, log_rho=_log_rho(self.log_target, counts))

    @property
    def base(self) -> Path:
        """Return the query's hole: the position at and below which the chain regrows.

        Returns:
            Path: ``()`` for a generator query, the query's position for a partial-term one.
        """
        return () if self.query.pos is None else self.query.pos

    def bin_of(self, cost: float) -> int | None:
        """Return the bin a cost falls in.

        Args:
            cost (float): The cost.

        Returns:
            int | None: The bin's index, None outside the edges.
        """
        index = bisect_right(self.edges, cost) - 1
        return index if 0 <= index < len(self.edges) - 1 else None

    def cost_of(self, term: Tree[T]) -> float:
        """Return a term's cost as a real number.

        Args:
            term (Tree[T]): The term.

        Returns:
            float: Its cost under the table's algebra.
        """
        return _real_cost(self.table.algebra.fold(term), "the cost of a term")

    def log_weight(self, term: Tree[T]) -> float:
        """Return the log of a term's weight under the chain's law, up to a constant.

        Args:
            term (Tree[T]): The term.

        Returns:
            float: ``log(target(b) / N(b))`` for the bin of its cost; ``-inf`` outside the target.
        """
        index = self.bin_of(self.cost_of(term))
        return -math.inf if index is None else self.log_rho[index]

    def _clauses_of(self, nonterminal: NT) -> tuple[list[RHSRule[NT, T, G]], list[float]]:
        """Return a non-terminal's clauses with a term, and the cumulative probabilities a draw picks one by.

        A clause weighs ``e^(-theta c(clause))`` times its holes' ``Z(theta)``; their sum is ``Z`` of the non-terminal.

        Args:
            nonterminal (NT): The non-terminal, one with a term.

        Returns:
            tuple: The clauses and their cumulative probabilities, the last one 1.
        """
        known = self.memory.clauses.get(nonterminal)
        if known is None:
            theta = self.table.theta
            kept: list[RHSRule[NT, T, G]] = []
            log_weights: list[float] = []
            for rule in self.query.solution_space.get(nonterminal) or ():
                if not _admitted(rule):
                    continue
                log_weight = -theta * _real_rule_cost(rule, self.table.algebra) + sum(
                    self.table.of(argument.origin)
                    for argument in rule.arguments
                    if isinstance(argument, NonTerminalArgument)
                )
                if log_weight > -math.inf:
                    kept.append(rule)
                    log_weights.append(log_weight)
            total = log_sum_exp(log_weights)
            known = (kept, list(accumulate(math.exp(value - total) for value in log_weights)))
            self.memory.clauses[nonterminal] = known
        return known

    def _draw(self, nonterminal: NT, rng: random.Random) -> tuple[Tree[T], Derivation]:
        """Draw a term of a non-terminal in proportion to ``e^(-theta c(t))``, top-down, with its derivation.

        Each clause is chosen in proportion to its tilted weight and each hole filled independently, which is the tilted
        law because the tilted mass of a clause's terms factors over its holes. Iterative, because terms grow deeper than
        a recursive descent survives.

        Args:
            nonterminal (NT): The non-terminal, one with a term.
            rng (random.Random): The source of randomness.

        Returns:
            tuple[Tree[T], Derivation]: The term and its derivation, positions relative to its root.
        """
        derivation: Derivation = {(): nonterminal}
        # A frame: the chosen clause, the children built so far, and the frame's position.
        stack: list[tuple[RHSRule[NT, T, G], list[Tree[T]], tuple[int, ...]]] = [
            (self._choose(nonterminal, rng), [], ())
        ]
        while True:
            rule, children, position = stack[-1]
            index = len(children)
            if index == len(rule.arguments):
                built = Tree(rule.terminal, tuple(children))
                stack.pop()
                if not stack:
                    return built, derivation
                stack[-1][1].append(built)
                continue
            argument = rule.arguments[index]
            if isinstance(argument, ConstantArgument):
                children.append(Tree(argument.value, ()))
                continue
            child = (*position, index)
            derivation[child] = argument.origin
            stack.append((self._choose(argument.origin, rng), [], child))

    def _choose(self, nonterminal: NT, rng: random.Random) -> RHSRule[NT, T, G]:
        """Choose one clause of a non-terminal in proportion to its tilted weight.

        Args:
            nonterminal (NT): The non-terminal.
            rng (random.Random): The source of randomness.

        Returns:
            RHSRule[NT, T, G]: The clause.

        Raises:
            ValueError: If the non-terminal has no term.
        """
        clauses, cumulative = self._clauses_of(nonterminal)
        if not clauses:
            msg = f"the non-terminal {nonterminal} has no term to draw"
            raise ValueError(msg)
        return clauses[min(bisect_right(cumulative, rng.random() * cumulative[-1]), len(clauses) - 1)]

    def _goals(self) -> list[tuple[float, Any, Any]]:
        """Return a partial-term query's initial goals as a move of the whole term draws them: log mass, goal, hole's non-terminal.

        Computed once per chain, since finding them walks the query's term.

        Returns:
            list: Per initial goal with a term, its log tilted mass, the goal, and the non-terminal of its hole at the
                query's position (None where a clause fills the position with a constant and the goal is a term).
        """
        known = self.memory.goals
        if known is None:
            known = []
            for goal, cost in _initial_tilt_nodes(self.query, self.table.algebra):
                open_holes = holes(goal)
                log_mass = -self.table.theta * cost + sum(self.table.of(hole) for hole in open_holes.values())
                if log_mass > -math.inf:
                    known.append((log_mass, goal, next(iter(open_holes.values()), None)))
            self.memory.goals = known
        return known

    def _draw_query(self, rng: random.Random) -> tuple[Tree[T], Derivation]:
        """Draw a completion of the query in proportion to ``e^(-theta c(t))``, with its derivation below the hole.

        A generator's draw is the start's; a partial-term query's chooses one of its initial goals by its tilted mass and
        fills the goal's hole from the hole's non-terminal.

        Args:
            rng (random.Random): The source of randomness.

        Returns:
            tuple[Tree[T], Derivation]: The term and its derivation, positions absolute.

        Raises:
            ValueError: If the query has no term.
        """
        if self.query.tree is None:
            if self.table.of(self.query.start) == -math.inf:
                msg = "the query has no term, so the chain has nowhere to start"
                raise ValueError(msg)
            return self._draw(self.query.start, rng)
        goals = self._goals()
        if not goals:
            msg = "the query has no term, so the chain has nowhere to start"
            raise ValueError(msg)
        total = log_sum_exp([log_mass for log_mass, _goal, _hole in goals])
        cumulative = list(accumulate(math.exp(log_mass - total) for log_mass, _goal, _hole in goals))
        _log_mass, goal, hole = goals[min(bisect_right(cumulative, rng.random() * cumulative[-1]), len(goals) - 1)]
        base = self.base
        if hole is None:
            return goal.grounded[()][1], {base: None}
        subtree, below = self._draw(hole, rng)
        return self.query.tree.replace_subtree_at(base, subtree), {(*base, *path): nt for path, nt in below.items()}

    def derivation_of(self, term: Tree[T]) -> Derivation:
        """Return a term's derivation below the query's hole: remembered when the chain drew the term, parsed otherwise.

        Args:
            term (Tree[T]): A completion of the query.

        Returns:
            Derivation: The non-terminal at each position the chain may regrow.

        Raises:
            ValueError: If the term is not a completion of the query.
        """
        known = self.memory.derivations.get(term)
        if known is None:
            known = self._parse(term)
            self.memory.remember(term, known)
        return known

    def positions(self, term: Tree[T]) -> int:
        """Return the number of positions a local move chooses among: those at and below the query's hole, constants left out.

        Args:
            term (Tree[T]): A completion of the query.

        Returns:
            int: ``m(t)``, at least 1: the query's hole itself.
        """
        return len(self.derivation_of(term))

    def regrow_log_mass(self, term: Tree[T], position: Path) -> float:
        """Return the log of the tilted mass of the terms a regrowth at a position can propose.

        Args:
            term (Tree[T]): A completion of the query.
            position (Path): One of its positions the chain may regrow.

        Returns:
            float: ``-theta c(context) + log Z_A(theta)``, ``A`` the derivation's non-terminal at the position.
        """
        nonterminal = self.derivation_of(term)[position]
        context = self.cost_of(term) - self.cost_of(term.subtree_at(position))
        return -self.table.theta * context + self.table.of(nonterminal)

    def regrow(self, term: Tree[T], position: Path, rng: random.Random) -> Tree[T]:
        """Regrow the subtree at a position from the non-terminal the term's derivation has there.

        Args:
            term (Tree[T]): A completion of the query.
            position (Path): One of its positions the chain may regrow.
            rng (random.Random): The source of randomness.

        Returns:
            Tree[T]: The regrown term, its derivation remembered.

        Raises:
            KeyError: If the position is not one the chain may regrow.
        """
        derivation = self.derivation_of(term)
        nonterminal = derivation[position]
        if nonterminal is None:  # a constant at the query's hole: only the move of the whole term changes it
            return term
        subtree, below = self._draw(nonterminal, rng)
        regrown = term.replace_subtree_at(position, subtree)
        depth = len(position)
        kept = {path: nt for path, nt in derivation.items() if path[:depth] != position}
        kept.update({(*position, *path): nt for path, nt in below.items()})
        self.memory.remember(regrown, kept)
        return regrown

    def log_acceptance(self, term: Tree[T], proposal: Tree[T], *, root_move: bool) -> float:
        """Return the log of the probability of accepting a proposal, ``min(0, log of the ratio above)``.

        Args:
            term (Tree[T]): The state, a term inside the target.
            proposal (Tree[T]): The proposal, regrown at one position of the state.
            root_move (bool): Whether the whole term was regrown, in which case the numbers of positions do not enter.

        Returns:
            float: The log acceptance, at most 0; ``-inf`` for a proposal outside the target.

        Raises:
            ValueError: If the state lies outside the target, where the chain never stands.
        """
        current = self.log_weight(term)
        if current == -math.inf:
            msg = "the term lies outside the chain's target, where the chain never stands"
            raise ValueError(msg)
        proposed = self.log_weight(proposal)
        if proposed == -math.inf:
            return -math.inf
        log_ratio = proposed - current + self.table.theta * (self.cost_of(proposal) - self.cost_of(term))
        if not root_move:
            log_ratio += math.log(self.positions(term)) - math.log(self.positions(proposal))
        return min(0.0, log_ratio)

    def step(self, term: Tree[T], rng: random.Random) -> ChainStep[T]:
        """Take one step from a state.

        Args:
            term (Tree[T]): The state, a completion of the query inside the target.
            rng (random.Random): The source of randomness.

        Returns:
            ChainStep[T]: The step.
        """
        root_move = rng.random() < self.root_share
        if root_move:
            position = self.base
            proposal, derivation = self._draw_query(rng)
            self.memory.remember(proposal, derivation)
        else:
            derivation = self.derivation_of(term)
            position = list(derivation)[rng.randrange(len(derivation))]
            proposal = self.regrow(term, position, rng)
        log_acceptance = self.log_acceptance(term, proposal, root_move=root_move)
        accepted = log_acceptance == 0.0 or (log_acceptance > -math.inf and rng.random() < math.exp(log_acceptance))
        return ChainStep(
            term=proposal if accepted else term,
            proposal=proposal,
            position=position,
            root_move=root_move,
            log_acceptance=log_acceptance,
            accepted=accepted,
        )

    def initial(self, rng: random.Random, max_draws: int = 10_000) -> Tree[T]:
        """Draw a first state: completions of the query in tilted proportion, until one lies inside the target.

        Args:
            rng (random.Random): The source of randomness.
            max_draws (int): How many terms to draw at most. (Default value = 10_000)

        Returns:
            Tree[T]: A term inside the target, its derivation remembered.

        Raises:
            ValueError: If the query has no term, or none of ``max_draws`` drawn terms lies inside the target.
        """
        for _ in range(max_draws):
            term, derivation = self._draw_query(rng)
            if self.log_weight(term) > -math.inf:
                self.memory.remember(term, derivation)
                return term
        msg = f"none of {max_draws} drawn terms lies inside the chain's target; give the chain a start"
        raise TargetOutOfReach(msg)

    def run(self, rng: random.Random, start: Tree[T] | None = None) -> Iterator[ChainStep[T]]:
        """Run the chain, one step per element, without end.

        Args:
            rng (random.Random): The source of randomness.
            start (Tree[T] | None): The first state, a completion of the query inside the target; None draws one
                (:meth:`initial`). (Default value = None)

        Yields:
            ChainStep[T]: The steps, each from the state the one before left.

        Raises:
            ValueError: If the start lies outside the chain's target or is not a completion of the query; or, without a
                start, if the query has no term or none of the drawn terms lies inside the target.
        """
        term = self.initial(rng) if start is None else start
        if self.log_weight(term) == -math.inf:
            msg = "the start lies outside the chain's target, where the chain never stands"
            raise ValueError(msg)
        self.derivation_of(term)
        while True:
            step = self.step(term, rng)
            term = step.term
            yield step

    def _parse(self, term: Tree[T]) -> Derivation:
        """Return the derivation of a completion of the query below its hole, from the program's clauses.

        Bottom-up, every node is given the non-terminals with a clause that derives it from its children's; top-down, the
        one the query admits at its hole is followed down. The clauses are found by their terminal and constants.

        Args:
            term (Tree[T]): The term.

        Returns:
            Derivation: Its derivation, positions absolute.

        Raises:
            ValueError: If the term is not a completion of the query, or has two derivations.
        """
        base = self.base
        tree = self.query.tree
        if tree is not None:
            try:
                differs = term.replace_subtree_at(base, _HOLE) != tree.replace_subtree_at(base, _HOLE)
            except IndexError:
                differs = True
            if differs:
                msg = "the term is not a completion of the query: it differs from the query's term outside its hole"
                raise ValueError(msg)
        if tree is None:
            holes_admitted = {self.query.start}
        else:
            holes_admitted = {hole for _log_mass, _goal, hole in self._goals()}
            if None in holes_admitted and term == tree:
                return {base: None}
        subtree = term.subtree_at(base)
        derives = self._derives(subtree)
        root = [nonterminal for nonterminal in derives[()] if nonterminal in holes_admitted]
        if len(root) != 1:
            reason = "it has no derivation" if not root else "it has more than one derivation"
            msg = f"the term is not a completion of the query the chain can walk on: {reason}"
            raise ValueError(msg)
        derivation: Derivation = {}
        pending: list[tuple[tuple[int, ...], Any]] = [((), root[0])]
        while pending:
            position, nonterminal = pending.pop()
            derivation[(*base, *position)] = nonterminal
            rule = derives[position][nonterminal]
            for index, argument in enumerate(rule.arguments):
                if isinstance(argument, NonTerminalArgument):
                    pending.append(((*position, index), argument.origin))
        return derivation

    def _derives(self, subtree: Tree[T]) -> dict[tuple[int, ...], dict[Any, RHSRule[NT, T, G]]]:
        """Return, per position of a term, the non-terminals that derive the subtree there, each with its clause.

        Args:
            subtree (Tree[T]): The term.

        Returns:
            dict: Per position, the deriving non-terminals and the clause each uses.

        Raises:
            ValueError: If a non-terminal derives a subtree by two clauses, which is a second derivation.
        """
        index = self._index()
        derives: dict[tuple[int, ...], dict[Any, RHSRule[NT, T, G]]] = {}
        order: list[tuple[tuple[int, ...], Tree[T]]] = []
        stack: list[tuple[tuple[int, ...], Tree[T]]] = [((), subtree)]
        while stack:
            position, node = stack.pop()
            order.append((position, node))
            stack.extend(((*position, child_index), child) for child_index, child in enumerate(node.children))
        for position, node in reversed(order):
            found: dict[Any, RHSRule[NT, T, G]] = {}
            for fixed, buckets in index.get(node.root, {}).items():
                if any(i >= len(node.children) for i in fixed):
                    continue
                for nonterminal, rule in buckets.get(tuple(node.children[i].root for i in fixed), ()):
                    if len(rule.arguments) != len(node.children) or not all(
                        (not node.children[i].children and node.children[i].root == argument.value)
                        if isinstance(argument, ConstantArgument)
                        else argument.origin in derives.get((*position, i), {})
                        for i, argument in enumerate(rule.arguments)
                    ):
                        continue
                    if nonterminal in found and found[nonterminal] is not rule:
                        msg = (
                            "the term is not a completion of the query the chain can walk on: it has more than one "
                            "derivation"
                        )
                        raise ValueError(msg)
                    found[nonterminal] = rule
            derives[position] = found
        return derives

    def _index(self) -> dict[Any, dict[tuple[int, ...], dict[tuple[Any, ...], list[tuple[Any, RHSRule[NT, T, G]]]]]]:
        """Return the program's clauses with a term, by terminal, by the indices of its constants, and by their values.

        Built once per chain, in one pass over the program.

        Returns:
            dict: Per terminal, per tuple of constant indices, per tuple of their values, the non-terminals and clauses.
        """
        known = self.memory.index
        if known is None:
            known = {}
            space = self.query.solution_space
            for nonterminal in space.nonterminals():
                for rule in space.get(nonterminal) or ():
                    if not _admitted(rule) or any(
                        self.table.of(argument.origin) == -math.inf
                        for argument in rule.arguments
                        if isinstance(argument, NonTerminalArgument)
                    ):
                        continue
                    fixed = tuple(
                        i for i, argument in enumerate(rule.arguments) if isinstance(argument, ConstantArgument)
                    )
                    values = tuple(rule.arguments[i].value for i in fixed)  # type: ignore[union-attr]
                    known.setdefault(rule.terminal, {}).setdefault(fixed, {}).setdefault(values, []).append(
                        (nonterminal, rule)
                    )
            self.memory.index = known
        return known


def _checked_log_counts(log_counts: Sequence[float], bins: int) -> tuple[float, ...]:
    """Return the log counts per bin as real numbers or ``-inf``, or refuse them.

    Args:
        log_counts (Sequence[float]): The log of the number of terms per bin.
        bins (int): The number of bins.

    Returns:
        tuple[float, ...]: The log counts.

    Raises:
        ValueError: If they are not one per bin, each a real number or ``-inf``.
    """
    if len(log_counts) != bins or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value) or value == math.inf
        for value in log_counts
    ):
        msg = "the counts need one log count per bin, a real number or -inf for a bin without a term"
        raise ValueError(msg)
    return tuple(float(value) for value in log_counts)


def _log_rho(log_target: Sequence[float], log_counts: Sequence[float]) -> tuple[float, ...]:
    """Return ``log(target / N)`` per bin, ``-inf`` where the bin has no target or no term, or refuse an empty law.

    Args:
        log_target (Sequence[float]): The log of the target's mass per bin.
        log_counts (Sequence[float]): The log of the number of terms per bin.

    Returns:
        tuple[float, ...]: The log weight of one term per bin.

    Raises:
        ValueError: If no bin has both a target and a term.
    """
    log_rho = tuple(
        target - count if target > -math.inf and count > -math.inf else -math.inf
        for target, count in zip(log_target, log_counts, strict=True)
    )
    if all(value == -math.inf for value in log_rho):
        msg = "no bin with a target has a term, so the chain has no law to walk to"
        raise TargetOutOfReach(msg)
    return log_rho


def metropolis_chain(
    query: ResolutionQuery[NT, T, G],
    algebra: AdditiveCostAlgebra[Any],
    theta: float,
    edges: Sequence[float],
    target: Sequence[float],
    log_counts: Sequence[float],
    *,
    program: TiltProgram[NT] | None = None,
    root_share: float = 0.5,
) -> MetropolisChain[NT, T, G]:
    """Build the chain to a target on cost bins, spread over the terms of a bin by the counts the caller gives.

    Args:
        query (ResolutionQuery[NT, T, G]): The query to walk on, generator or partial-term.
        algebra (AdditiveCostAlgebra[Any]): The additive cost algebra, with finite real costs.
        theta (float): The tilt every regrowth draws by, any finite real number.
        edges (Sequence[float]): The bins' boundaries, strictly ascending finite reals, at least two.
        target (Sequence[float]): The target's mass per bin, one fewer than the edges, nonnegative, not all zero.
        log_counts (Sequence[float]): The log of the number of the query's terms per bin, exact or estimated, one per
            bin; ``-inf`` for a bin without a term.
        program (TiltProgram[NT] | None): The query's program already prepared under this algebra. (Default value = None)
        root_share (float): The probability of regrowing the whole term in a step, a real number from 0 to 1.
            (Default value = 0.5)

    Returns:
        MetropolisChain[NT, T, G]: The chain, ready to run.

    Raises:
        ValueError: If ``theta``, the edges, the target, the counts or the root share are not of the kinds above; if a
            predicate of the program reads a hole, or the tilt refuses the program otherwise; or if no bin has both a
            target and a term.
    """
    theta = _real_theta(theta)
    checked_edges = _checked_edges(edges)
    masses = [_real_cost(mass, "a target mass") for mass in target]
    if len(masses) != len(checked_edges) - 1 or any(mass < 0 for mass in masses) or sum(masses) <= 0:
        msg = "the target needs one nonnegative mass per bin, one fewer than the edges, and not all of them zero"
        raise ValueError(msg)
    counts = _checked_log_counts(log_counts, len(masses))
    if isinstance(root_share, bool) or not isinstance(root_share, (int, float)) or not 0 <= root_share <= 1:
        msg = f"the root share is a probability, a real number from 0 to 1, not {root_share!r}"
        raise ValueError(msg)
    log_target = tuple(math.log(mass) if mass > 0 else -math.inf for mass in masses)
    log_rho = _log_rho(log_target, counts)
    decomposable_or_raise(query.solution_space)
    prepared = _prepared(query, algebra, program)
    return MetropolisChain(
        query=query,
        table=prepared.table(theta),
        edges=tuple(checked_edges),
        log_target=log_target,
        log_counts=counts,
        log_rho=log_rho,
        root_share=float(root_share),
    )


@dataclass(frozen=True)
class WangLandau(Generic[T]):
    """What a Wang-Landau walk learned: the counts per bin, and how it got there.

    Attributes:
        log_counts (tuple[float, ...]): Per bin, the learned log of the number of terms, up to one constant shared by
            every bin; ``-inf`` for a bin without a target, or one the given counts call empty.
        stages (tuple[tuple[float, int], ...]): Per finished stage, its factor ``log f`` and its number of steps.
        steps (int): The steps taken in all, an unfinished last stage's included.
        converged (bool): Whether the factor fell below the final one, every stage flat.
        term (Tree[T]): The state the walk ended in, inside the target, a start for the chain on the learned counts.
    """

    log_counts: tuple[float, ...]
    stages: tuple[tuple[float, int], ...]
    steps: int
    converged: bool
    term: Tree[T]


def wang_landau(
    chain: MetropolisChain[NT, T, G],
    rng: random.Random,
    *,
    start: Tree[T] | None = None,
    log_counts: Sequence[float] | None = None,
    log_f: float = 1.0,
    final_log_f: float = 1e-4,
    flatness: float = 0.8,
    check_every: int | None = None,
    max_steps: int = 1_000_000,
) -> WangLandau[T]:
    """Learn the number of terms per bin along the chain, with the target as the histogram to flatten against.

    Wang and Landau walk in proportion to the reciprocal of an estimated density of states and multiply the estimate
    of the state's energy by a factor ``f`` at every visit, so that the visits even out; once the histogram of
    visits is flat, every value at least 80 percent of their mean, the factor is reduced to its square root and the
    histogram reset, until the factor falls below a final one; the estimate is then relative. Here the energy is
    the bin of a term's cost and the walk is the chain: it walks in proportion to ``target(b) / N-hat(b)``, adds
    ``log f / (k s(b))`` to ``log N-hat(b)`` at every step for the bin the step leaves the walk in, ``s(b)`` the bin's
    share of the target and ``k`` the number of bins, and calls the visits flat when each bin's visits over its
    share are at least ``flatness`` times their mean. The estimates of two bins keep their ratio once the visits'
    rates over the shares agree, which is where the visits follow the target and ``N-hat`` is the truth; with equal
    increments, as in the flat case, they would settle where the visits are equal instead, at ``N`` times the target. Every bin with a target must have been
    visited before a stage may end, except the bins the given counts call empty. While the factor is positive the walk
    is not a Markov chain with a fixed law; it is the learned counts that go on, into a chain that is.

    Args:
        chain (MetropolisChain[NT, T, G]): The chain whose bins, target and moves the walk uses; its counts are replaced.
        rng (random.Random): The source of randomness.
        start (Tree[T] | None): The first state, inside the target; None draws one (:meth:`MetropolisChain.initial`).
            (Default value = None)
        log_counts (Sequence[float] | None): The counts to start from, one per bin; None starts from equal counts on
            every bin with a target. (Default value = None)
        log_f (float): The first factor ``log f``, positive; Wang and Landau start at 1. (Default value = 1.0)
        final_log_f (float): The factor below which the walk ends, positive. (Default value = 1e-4)
        flatness (float): How close to their mean every bin's visits over its share must be, in ``(0, 1]``; Wang and
            Landau take 0.8. (Default value = 0.8)
        check_every (int | None): How many steps pass between two checks of flatness, at least one; None checks
            every ten steps per bin with a target. (Default value = None)
        max_steps (int): The budget of steps, at least 0; a walk that exhausts it reports that it did not converge.
            (Default value = 1_000_000)

    Returns:
        WangLandau[T]: The learned counts, the stages, the steps, and the state the walk ended in.

    Raises:
        ValueError: If a factor is not positive, the flatness is outside ``(0, 1]``, the budget negative, the check
            interval below one, or the counts not one per bin.
    """
    for name, value in (("first factor", log_f), ("final factor", final_log_f)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < math.inf:
            msg = f"the {name} log f must be a positive real number, not {value!r}"
            raise ValueError(msg)
    if isinstance(flatness, bool) or not isinstance(flatness, (int, float)) or not 0 < flatness <= 1:
        msg = f"the flatness must be a real number in (0, 1], not {flatness!r}"
        raise ValueError(msg)
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        msg = f"the budget of steps must be a whole number not below 0, not {max_steps!r}"
        raise ValueError(msg)
    bins = len(chain.log_target)
    wanted = [index for index, target in enumerate(chain.log_target) if target > -math.inf]
    if check_every is None:
        check_every = 10 * len(wanted)
    if isinstance(check_every, bool) or not isinstance(check_every, int) or check_every < 1:
        msg = f"the steps between two checks of flatness must be a whole number of at least one, not {check_every!r}"
        raise ValueError(msg)
    if log_counts is None:
        estimate = [0.0 if index in wanted else -math.inf for index in range(bins)]
    else:
        estimate = list(_checked_log_counts(log_counts, bins))
    required = [index for index in wanted if estimate[index] > -math.inf]
    total = math.fsum(math.exp(chain.log_target[index]) for index in required)
    shares = {index: math.exp(chain.log_target[index]) / total for index in required}
    increments = {index: 1 / (len(required) * share) for index, share in shares.items()}
    walking = chain.with_log_counts(estimate)
    term = walking.initial(rng) if start is None else start
    if walking.log_weight(term) == -math.inf:
        msg = "the start lies outside the chain's target, where the chain never stands"
        raise ValueError(msg)
    visits = [0] * bins
    stages: list[tuple[float, int]] = []
    stage_steps = 0
    steps = 0
    converged = False
    while steps < max_steps:
        term = walking.step(term, rng).term
        steps += 1
        stage_steps += 1
        index = walking.bin_of(walking.cost_of(term))
        if index is None:  # the chain never stands outside its target
            msg = "the walk left the chain's target"
            raise ValueError(msg)
        estimate[index] += log_f * increments[index]
        visits[index] += 1
        walking = replace(walking, log_counts=tuple(estimate), log_rho=_log_rho(walking.log_target, estimate))
        if stage_steps % check_every:
            continue
        relative = [visits[index] / shares[index] for index in required]
        if min(relative) <= 0 or min(relative) < flatness * (sum(relative) / len(relative)):
            continue
        stages.append((log_f, stage_steps))
        stage_steps = 0
        visits = [0] * bins
        log_f /= 2
        if log_f < final_log_f:
            converged = True
            break
    return WangLandau(log_counts=tuple(estimate), stages=tuple(stages), steps=steps, converged=converged, term=term)


@dataclass(frozen=True)
class VisitCounts(Generic[T]):
    """The counts per bin a run of the chain on fixed counts reads off its visits.

    Attributes:
        log_counts (tuple[float, ...]): Per bin, the corrected log of the number of terms, up to one constant shared by
            every bin; ``-inf`` for a bin the run never visited.
        relative_error (tuple[float, ...]): Per bin, the standard error of the correction over its value, from the
            spread of the visits across consecutive batches of the run; ``inf`` for a bin never visited, ``nan`` where
            a bin was visited in one batch only.
        visits (tuple[int, ...]): Per bin, the run's visits.
        steps (int): The steps of the run.
        term (Tree[T]): The state the run ended in.
    """

    log_counts: tuple[float, ...]
    relative_error: tuple[float, ...]
    visits: tuple[int, ...]
    steps: int
    term: Tree[T]


def counts_from_visits(
    chain: MetropolisChain[NT, T, G],
    rng: random.Random,
    steps: int,
    *,
    start: Tree[T] | None = None,
    batches: int = 10,
) -> VisitCounts[T]:
    """Correct the chain's counts per bin by its visits: a run on fixed counts, read as the multicanonical recursion does.

    On fixed counts ``N-hat`` the chain's visits to a bin follow ``target(b) N(b) / N-hat(b)`` in the limit, so the
    truth is ``N(b)`` proportional to ``N-hat(b) H(b) / target(b)`` for the visits ``H(b)`` of a long run: one step of
    the recursion multicanonical sampling iterates. It is what turns a Wang-Landau walk's rough counts into accurate
    ones, its error that of the visits, and it reads any counts a caller has, a pilot's or the saddle point's, against
    the chain's own visits. The run's correlation enters the error through batches: the visits are counted per batch of
    consecutive steps, and the spread of the batches' corrections is the reported error.

    Args:
        chain (MetropolisChain[NT, T, G]): The chain, on the counts to correct.
        rng (random.Random): The source of randomness.
        steps (int): The steps of the run, at least ``batches``.
        start (Tree[T] | None): The first state, inside the target; None draws one. (Default value = None)
        batches (int): The number of batches the run is cut into for the error, at least two. (Default value = 10)

    Returns:
        VisitCounts[T]: The corrected counts, their errors, the visits, and the state the run ended in.

    Raises:
        ValueError: If the number of batches is below two or the steps fewer than the batches, or the start lies
            outside the target.
    """
    if isinstance(batches, bool) or not isinstance(batches, int) or batches < _FEWEST_BATCHES:
        msg = f"the batches must be a whole number of at least two, not {batches!r}"
        raise ValueError(msg)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < batches:
        msg = f"the steps must be a whole number of at least the batches, {batches}, not {steps!r}"
        raise ValueError(msg)
    bins = len(chain.log_target)
    per_batch = [[0] * bins for _ in range(batches)]
    run = chain.run(rng, start)
    term = start
    for step_index in range(steps):
        term = next(run).term
        index = chain.bin_of(chain.cost_of(term))
        if index is not None:
            per_batch[min(batches - 1, step_index * batches // steps)][index] += 1
    visits = [sum(batch[index] for batch in per_batch) for index in range(bins)]
    log_counts: list[float] = []
    relative_error: list[float] = []
    for index in range(bins):
        if visits[index] == 0 or chain.log_target[index] == -math.inf or chain.log_counts[index] == -math.inf:
            log_counts.append(-math.inf)
            relative_error.append(math.inf)
            continue
        log_counts.append(chain.log_counts[index] + math.log(visits[index]) - chain.log_target[index])
        shares = [batch[index] / sum(batch) for batch in per_batch if sum(batch)]
        mean = math.fsum(shares) / len(shares)
        if sum(1 for batch in per_batch if batch[index]) < _FEWEST_BATCHES or mean <= 0:
            relative_error.append(math.nan)
            continue
        spread = math.sqrt(math.fsum((share - mean) ** 2 for share in shares) / (len(shares) - 1))
        relative_error.append(spread / math.sqrt(len(shares)) / mean)
    if term is None:  # pragma: no cover - steps >= batches >= 2 makes at least one step
        msg = "the run took no step"
        raise ValueError(msg)
    return VisitCounts(
        log_counts=tuple(log_counts),
        relative_error=tuple(relative_error),
        visits=tuple(visits),
        steps=steps,
        term=term,
    )
