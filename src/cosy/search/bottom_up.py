"""Bottom-up search: the least Herbrand model, computed forward from the facts.

The immediate consequence operator of a synthesized program collects the heads of the ground
clause instances whose bodies already hold. Iterated from the empty set it produces an ascending
chain of finite sets whose least fixpoint is the least Herbrand model, and a term is an
inhabitant of a non-terminal exactly when the model holds its atom. Bottom-up search runs that
iteration and streams, after each step, the terms of the newly derived atoms of the queried
non-terminal, so the stream is exactly the tree language that non-terminal denotes.

This is the one search rule that does *not* traverse the derivation tree of a query. It runs
data-driven, following the clauses forward with no goal to direct it, and that is where it halts
while the goal-driven rules do not. A clause is applied only to ground instances whose body already
holds, so an external predicate is decided the moment the clause is applied rather than at the end
of a branch, and a non-terminal whose language is empty simply contributes nothing instead of
receiving a descent that never returns.

What it gives up in exchange is control, and the condition it halts under is a property of the
whole program rather than of the query. A round derives for every non-terminal, so a start whose
own language is finite does not end the run while some other non-terminal keeps growing. It cannot
be steered toward a region of the space or toward a preferred solution, and it constructs its
iterates in full where a goal-driven descent would hold a single path.

Deliberately naive. :meth:`cosy.core.solution_space.SolutionSpace.enumerate_trees` is a
long-optimized enumeration of the same language, bucketed by term size and driven by an inverse
grammar so that a new term is combined only with the rules that can consume it. This module is the
textbook operator instead: one full pass over every clause per round, against the whole model. The
two are not interchangeable as measurements, and that difference is the point when they are
compared, which is what :class:`BottomUpCounters` is for.

The predicates of a clause are decided on the chosen terms themselves. ``enumerate_trees`` also
takes an ``interpretation`` and evaluates the arguments through it before it decides a clause,
and bottom-up search has no counterpart to that, so the two agree on the programs whose predicates
read terms rather than interpreted values.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Any

from cosy.core.solution_space import NT, G, NonTerminalArgument, T
from cosy.core.tree import Tree

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from cosy.core.solution_space import RHSRule, SolutionSpace

__all__ = ["BottomUpCounters", "bottom_up", "least_herbrand_model"]

# The model of one non-terminal. A dict rather than a set because the stream is read off it: a set
# iterates in an order the hash seed decides, which would make the terms a bounded run returns
# differ between processes, and the enumeration this module is compared against promises they do
# not. The values carry nothing, only the keys and their insertion order. The engine keeps a
# private ``_OrderedSet`` for the same reason and builds it on a dict, which a package outside
# ``cosy.core`` has no business importing and does not need: nothing here asks for set operations.
_Terms = dict["Tree[T]", None]


class BottomUpCounters:
    """The work a bottom-up run has done, in units the algorithm itself defines.

    A bottom-up run has no search nodes, so it expands none, and the node expansions the
    goal-driven rules report have no counterpart here. What it does instead is apply clauses to
    ground instances, and that is the unit counted: ``applications`` is the number of candidate
    ground instances the operator formed, ``derivations`` the number of them whose predicates
    held. Neither is comparable with an expansion count. Both are comparable across the size
    ladder of one program, which is what a growth statement needs, and both are comparable with
    the enumeration of ``enumerate_trees``, which advances the same two counters.

    Attributes:
        rounds (int): Iterations of the immediate consequence operator begun. A budget stops a
            run inside a round, and that round is counted like any other.
        applications (int): Candidate ground instances formed, predicates included.
        derivations (int): Candidate ground instances whose predicates all held.
        atoms (int): Atoms in the model the last round produced, which is a partial model when a
            budget stopped that round.
    """

    __slots__ = ("applications", "atoms", "derivations", "rounds")

    def __init__(self) -> None:
        """Start every counter at zero."""
        self.rounds = 0
        self.applications = 0
        self.derivations = 0
        self.atoms = 0


class _Budget:
    """The caller's stopping condition, consulted where the operator does its work.

    Bottom-up search has no stopping condition of its own: it runs to the fixpoint or it does not
    halt. A measurement needs one anyway, and it has to bite inside a round rather than between
    rounds. On a program whose model is infinite, a single round already outgrows the memory of
    the machine once the model is large, so a check taken between rounds would not be reached.
    The predicate is therefore consulted per candidate ground instance, and a run it stops is
    reported as stopped rather than as finished.
    """

    __slots__ = ("_stop", "stopped")

    def __init__(self, stop: Callable[[BottomUpCounters], bool] | None) -> None:
        """Wrap a stopping predicate.

        Args:
            stop (Callable[[BottomUpCounters], bool] | None): Returns True when the run must
                stop. None never stops.
        """
        self._stop = stop
        self.stopped = False

    def exhausted(self, counters: BottomUpCounters) -> bool:
        """Ask whether the run must stop, and remember the answer.

        Args:
            counters (BottomUpCounters): The work done so far.

        Returns:
            bool: True when the run must stop.
        """
        if self._stop is not None and self._stop(counters):
            self.stopped = True
        return self.stopped


def _ground_instances(
    rule: RHSRule[NT, T, G],
    model: dict[NT, _Terms[T]],
    counters: BottomUpCounters,
    budget: _Budget,
) -> Iterator[Tree[T]]:
    """Enumerate the terms one clause derives from a model.

    A ground instance chooses a term of the model for every non-terminal argument and keeps the
    literal arguments as they stand. The predicates are decided on the substitution the named
    arguments carry, and an instance whose predicates hold contributes the term the clause builds.
    A clause with a body atom the model does not derive yet contributes nothing at all this round.

    Args:
        rule (RHSRule[NT, T, G]): The clause.
        model (dict[NT, _Terms[T]]): The current model, one term set per non-terminal.
        counters (BottomUpCounters): The counters to advance.
        budget (_Budget): The stopping condition, consulted per candidate.

    Yields:
        Tree[T]: The term of each ground instance whose predicates hold.
    """
    choices: list[tuple[Tree[T], ...]] = []
    for argument in rule.arguments:
        if isinstance(argument, NonTerminalArgument):
            known = model.get(argument.origin)
            if not known:
                # A body atom with no derivation yet: the clause contributes nothing this round.
                return
            choices.append(tuple(known))
        else:
            # A literal argument stands for itself, so its position offers the one choice.
            choices.append((Tree(argument.value, ()),))

    literal_substitution = rule.literal_substitution

    for combination in product(*choices):
        counters.applications += 1
        if budget.exhausted(counters):
            return

        if rule.predicates:
            substitution: dict[str, Any] = dict(literal_substitution)
            for argument, chosen in zip(rule.arguments, combination, strict=True):
                if isinstance(argument, NonTerminalArgument) and argument.name is not None:
                    substitution[argument.name] = chosen
            if not all(predicate(substitution) for predicate in rule.predicates):
                continue

        counters.derivations += 1
        yield Tree(rule.terminal, combination)


def _immediate_consequence(
    space: SolutionSpace[NT, T, G],
    model: dict[NT, _Terms[T]],
    counters: BottomUpCounters,
    budget: _Budget,
) -> dict[NT, _Terms[T]]:
    """Apply the operator once: derive everything one clause application reaches from the model.

    The result contains the model itself, since the operator is applied to a model that already
    holds its facts, and the fixpoint test in :func:`_iterates` compares the two.

    Args:
        space (SolutionSpace[NT, T, G]): The synthesized program.
        model (dict[NT, _Terms[T]]): The current model.
        counters (BottomUpCounters): The counters to advance.
        budget (_Budget): The stopping condition.

    Returns:
        dict[NT, _Terms[T]]: The next iterate.
    """
    consequence: dict[NT, _Terms[T]] = {nonterminal: dict(terms) for nonterminal, terms in model.items()}
    for nonterminal, rules in space.as_tuples():
        derived = consequence.setdefault(nonterminal, {})
        for rule in rules:
            for term in _ground_instances(rule, model, counters, budget):
                derived[term] = None
            if budget.stopped:
                return consequence
    return consequence


def _iterates(
    space: SolutionSpace[NT, T, G],
    counters: BottomUpCounters,
    budget: _Budget,
) -> Iterator[tuple[dict[NT, _Terms[T]], dict[NT, _Terms[T]]]]:
    """Iterate the operator from the empty set and yield each step as it is taken.

    Both entry points of this module ascend the same chain and differ only in what they read off
    it, so the fixpoint test lives here and is written once. The chain ends when a step adds
    nothing, which is the fixpoint, or when the budget stops the run inside a step.

    Args:
        space (SolutionSpace[NT, T, G]): The synthesized program.
        counters (BottomUpCounters): The counters to advance.
        budget (_Budget): The stopping condition.

    Yields:
        tuple[dict[NT, _Terms[T]], dict[NT, _Terms[T]]]: The model a step started from and the
            iterate it produced.
    """
    model: dict[NT, _Terms[T]] = {}
    while True:
        consequence = _immediate_consequence(space, model, counters, budget)
        counters.rounds += 1
        counters.atoms = sum(len(terms) for terms in consequence.values())

        yield model, consequence

        if budget.stopped:
            return
        # The consequence contains the model it was computed from, so equal sizes mean equal sets
        # and the chain has reached its fixpoint.
        if all(len(consequence.get(n, {})) == len(model.get(n, {})) for n in consequence):
            return
        model = consequence


def bottom_up(
    space: SolutionSpace[NT, T, G],
    start: NT,
    *,
    max_count: int | None = None,
    stop: Callable[[BottomUpCounters], bool] | None = None,
    counters: BottomUpCounters | None = None,
) -> Iterator[Tree[T]]:
    """Run bottom-up search on a synthesized program and stream its inhabitants.

    Iterate the immediate consequence operator from the empty set, and after each step yield the
    term of every atom of ``start`` the step added. The stream is sound and complete: every term
    it yields inhabits ``start``, and every inhabitant of ``start`` is yielded, on the programs
    whose predicates read terms rather than values under an interpretation.

    It halts exactly when the least Herbrand model is finite, which is a condition on the whole
    program and not on ``start``. ``max_count`` therefore bounds the stream but not the run: once
    the language of ``start`` is exhausted, no further term arrives to reach the bound, and the
    rounds go on for as long as any other non-terminal still grows. ``stop`` is the bound that
    always bites.

    The stream is ordered by the round that derives a term, which is by the depth of the
    derivation rather than by the size of the term. Within one round the order follows the clauses
    of the program and the terms the model already held. It is therefore the same in every process,
    which is what a bound needs, but it carries no meaning of its own.

    Args:
        space (SolutionSpace[NT, T, G]): The synthesized program.
        start (NT): The queried non-terminal.
        max_count (int | None): Stop after this many inhabitants. (Default value = None)
        stop (Callable[[BottomUpCounters], bool] | None): A budget, consulted per candidate
            ground instance. The algorithm has no stopping condition of its own. This one is the
            caller's, and a run it stops has not finished. A budget over ``rounds`` or ``atoms``
            bites one round late, since those two are written between rounds rather than during
            one. (Default value = None)
        counters (BottomUpCounters | None): The counters to advance, for a caller that wants to
            read them after the stream ends or is abandoned. None allocates fresh ones.
            (Default value = None)

    Yields:
        Tree[T]: The inhabitants of ``start``, in the order the rounds derive them.
    """
    work = BottomUpCounters() if counters is None else counters
    streamed = 0

    for model, consequence in _iterates(space, work, _Budget(stop)):
        previous = model.get(start, {})
        for term in (t for t in consequence.get(start, {}) if t not in previous):
            # Before the yield rather than after it, so that a bound of zero streams nothing, as
            # it does in the enumeration. The second test ends the run without paying for the
            # round that would follow.
            if max_count is not None and streamed >= max_count:
                return
            yield term
            streamed += 1
        if max_count is not None and streamed >= max_count:
            return


def least_herbrand_model(
    space: SolutionSpace[NT, T, G],
    *,
    stop: Callable[[BottomUpCounters], bool] | None = None,
    counters: BottomUpCounters | None = None,
) -> dict[NT, set[Tree[T]]]:
    """Compute the least Herbrand model of a synthesized program.

    The fixpoint :func:`bottom_up` ascends to, returned whole rather than streamed through one
    queried non-terminal. Its size per non-terminal is what a bottom-up run must hold in memory,
    which is the quantity to read from a program when the memory of a run is the question.

    A model is a set of atoms and comes back as one. The order the stream needs is the order a
    bound cuts, and nothing here is cut.

    Args:
        space (SolutionSpace[NT, T, G]): The synthesized program.
        stop (Callable[[BottomUpCounters], bool] | None): A budget, as in :func:`bottom_up`. A
            model it stops is partial. (Default value = None)
        counters (BottomUpCounters | None): The counters to advance. (Default value = None)

    Returns:
        dict[NT, set[Tree[T]]]: The model, one term set per non-terminal of the program. A
            non-terminal the program never derives a term for carries an empty set rather than
            being absent.
    """
    work = BottomUpCounters() if counters is None else counters

    model: dict[NT, _Terms[T]] = {}
    for _previous, consequence in _iterates(space, work, _Budget(stop)):
        model = consequence
    return {nonterminal: set(terms) for nonterminal, terms in model.items()}
