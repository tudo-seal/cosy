"""Predicate determinization: pushing a recognizable constraint into the non-terminals.

The table form of counting (:func:`cosy.search.counting.size_table`) applies to a program in which
no predicate reads a hole, and :func:`cosy.search.counting.decomposable_or_raise` refuses the rest
rather than returning numbers that are quietly too large. Where a repository states its conditions
as predicates over several holes, that refusal is not occasional but total, and a treatment
covering only the decoupled case would leave such a repository exactly where it started.

This module covers the coupled case, under the condition **(REC)** of
:mod:`cosy.core.recognizable`: a predicate that factors through a finite abstraction ``alpha`` and
a relation ``R`` on its values can be *compiled away*. The construction is the product ``NT x Q``:

* a non-terminal of the determinized program is a pair of an original non-terminal and a state,
  :class:`ProductNonTerminal`, read as "an ``A`` whose term abstracts to ``q``";
* a clause becomes one instance per assignment of states to its holes that ``R`` admits, with the
  head state computed by ``alpha`` from the argument states;
* the predicate is gone. Nothing decides it at search time any more, because no term violating it
  is derivable.

Goldstein and Pierce (2022) name the reason AVL trees defeat a generator: it must guess the
correct height to cache at each node. That guess is what disappears here, the height being part of
the non-terminal and therefore derived rather than guessed.

**Why this preserves the distribution.** ``alpha`` is a function, so every term of the original
program abstracts to exactly one state, and its derivation maps to exactly one derivation of the
product program. The map is a bijection on success branches, so the branch counts, the
unambiguity of the program and the weights a draw reads off it are all invariant. Determinism is
what carries this. A *non*-deterministic abstraction would multiply branches and change the
weights silently, which is why the abstraction is an algebra and not a relation.

**Fixed point and rule generation are one pass, and each combination is visited once.** The
reachable states are not known in advance, being the least fixed point of "a clause instance whose
holes are reachable makes its head reachable", so the instances have to be enumerated while the
state sets still grow. Enumerating them per round would produce the same instance again in the
next round, and a duplicated rule is a second derivation of the same term: it makes the program
ambiguous and inflates the branch counts. The enumeration below is therefore semi-naive. In the
round in which a hole's state set grows from ``old`` to ``new``, a clause sees exactly the
combinations that were not available before, so every combination is instantiated exactly once
and :func:`cosy.search.counting.branch_multiplicities` on the result is empty. That is checked for
every space in the test suite rather than assumed here.

**Cost.** A clause with ``k`` holes has up to ``|Q|^k`` instances, so the construction is
polynomial in the size of the program and exponential in the arity of the coupling. That is the
price of turning a filter into a specification, and it is still incomparably cheaper than the tree
form, whose cost is the number of inhabitants. What bounds it in practice is ``|Q|``, and that is
a property of the repository's abstraction rather than of this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, Generic, cast

from cosy.core.recognizable import RecognizableConstraint
from cosy.core.solution_space import (
    NT,
    G,
    NonTerminalArgument,
    SolutionSpace,
    T,
)
from cosy.search.counting import CoupledClause

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from cosy.core.recognizable import StateRelation, TreeAbstraction
    from cosy.core.solution_space import Argument, RHSRule

__all__ = [
    "Determinization",
    "MergedNonTerminal",
    "ProductNonTerminal",
    "determinize",
    "recognizable_or_raise",
    "unabstracted_clauses",
]

DEFAULT_STATE_LIMIT = 100_000
"""How many product states the fixed point may reach before it is called non-terminating.

(REC) asks for a *finite* carrier, and an abstraction without one makes the fixed point run
forever. The term size without a cap is the usual slip. A limit turns the hang into a message
naming the abstraction, and it sits deliberately far above any state count a usable construction
reaches: the AVL space of the test suite has 94 product states over ten keys.
"""


@dataclass(frozen=True)
class ProductNonTerminal(Generic[NT]):
    """A non-terminal of the determinized program: ``(A, q)``.

    Attributes:
        nonterminal (NT): The non-terminal of the original program.
        state (tuple[Any, ...]): The value of the abstraction, one component per abstraction the
            program states. A program with no recognizable constraint has the empty tuple
            throughout, which makes the determinization the identity on it up to this wrapper.
    """

    nonterminal: NT
    state: tuple[Any, ...]

    def __str__(self) -> str:
        """Render the pair the way an error message or a printed grammar needs it.

        Returns:
            str: ``A@q``.
        """
        return f"{self.nonterminal}@{self.state}"


@dataclass(frozen=True)
class MergedNonTerminal(Generic[NT]):
    """The start symbol of the determinized program: ``A`` with the abstraction forgotten.

    A query is posed against a non-terminal, and the caller asks for the terms of ``A``, not for
    the terms of ``A`` that abstract to some particular ``q``. So the determinized program carries
    one extra symbol whose clauses are the clauses of every ``(A, q)``. Since each instance
    produces exactly one head state, a term still has exactly one derivation from here, and the
    branch count at the root is the sum over the states.

    A chain rule ``Merged(A) <- (A, q)`` would be the textbook alternative and is not available:
    every clause of a solution space writes a terminal, so a chain rule would add a symbol to the
    term and shift every size by one.

    Attributes:
        nonterminal (NT): The non-terminal whose states are merged.
    """

    nonterminal: NT

    def __str__(self) -> str:
        """Render the symbol the way an error message or a printed grammar needs it.

        Returns:
            str: ``A@*``.
        """
        return f"{self.nonterminal}@*"


@dataclass(frozen=True)
class Determinization(Generic[NT, T, G]):
    """The determinized program, together with what the construction learned about the repository.

    Attributes:
        space (SolutionSpace[Any, T, G]): The product program. It is predicate-free wherever the
            original's predicates were recognizable, so
            :func:`cosy.search.counting.decomposable_or_raise` passes on it and the table form
            applies.
        start (MergedNonTerminal[NT]): The symbol to query: the start non-terminal with the
            abstraction forgotten.
        states (Mapping[NT, tuple[tuple[Any, ...], ...]]): The reachable states per original
            non-terminal, in the order the fixed point found them. Their number is what decides
            whether the construction is affordable, so it is reported rather than hidden.
        abstractions (tuple[TreeAbstraction, ...]): The distinct abstractions the program states, in
            the order they occur in it. A state is a tuple over exactly these, which is the
            product construction over several constraints.
    """

    space: SolutionSpace[Any, T, G]
    start: MergedNonTerminal[NT]
    states: Mapping[NT, tuple[tuple[Any, ...], ...]]
    abstractions: tuple[TreeAbstraction, ...]

    @property
    def state_count(self) -> int:
        """Return the number of product non-terminals the construction reached.

        Returns:
            int: The sum of the reachable states over all original non-terminals.
        """
        return sum(len(reached) for reached in self.states.values())


def _named_holes(rule: RHSRule[NT, T, G]) -> tuple[int, ...]:
    """Return the argument positions of a clause a predicate can read.

    Args:
        rule (RHSRule[NT, T, G]): The clause.

    Returns:
        tuple[int, ...]: The positions of the named non-terminal arguments.
    """
    return tuple(
        index
        for index, argument in enumerate(rule.arguments)
        if isinstance(argument, NonTerminalArgument) and argument.name is not None
    )


def unabstracted_clauses(
    space: SolutionSpace[NT, T, G],
) -> list[CoupledClause[NT, T]]:
    """Return the clauses that read a hole through a predicate without a finite abstraction.

    These are exactly the clauses :func:`cosy.search.counting.coupled_clauses` reports and this
    module cannot compile away: a plain predicate records nothing about what it reads, so there is
    no way to decide it on states. Restating it with
    :meth:`cosy.core.SpecificationBuilder.recognizable_constraint` is what makes it determinizable.

    Args:
        space (SolutionSpace[NT, T, G]): The synthesized program.

    Returns:
        list[CoupledClause[NT, T]]: One entry per offending clause, empty exactly when
            :func:`determinize` can run.
    """
    offenders: list[CoupledClause[NT, T]] = []
    for nonterminal in space.nonterminals():
        for rule in space.get(nonterminal) or ():
            positions = _named_holes(rule)
            if not rule.predicates or not positions:
                continue
            if all(isinstance(predicate, RecognizableConstraint) for predicate in rule.predicates):
                continue
            offenders.append(
                CoupledClause(
                    nonterminal=nonterminal,
                    terminal=rule.terminal,
                    positions=positions,
                    nonterminals=tuple(rule.arguments[index].origin for index in positions),
                )
            )
    return offenders


def recognizable_or_raise(space: SolutionSpace[NT, T, G]) -> None:
    """Check that every hole-reading predicate of a program states its abstraction.

    Args:
        space (SolutionSpace[NT, T, G]): The program to decide for.

    Raises:
        ValueError: If some clause reads a hole through a plain predicate. The message names
            every such clause, because which ones they are is what a caller needs in order to
            repair the repository.
    """
    offenders = unabstracted_clauses(space)
    if not offenders:
        return
    listed = "; ".join(clause.describe() for clause in offenders)
    msg = (
        f"determinization needs every hole-reading predicate to state the finite abstraction it "
        f"factors through, which {len(offenders)} clause(s) of this program do not: {listed}. "
        f"State them with SpecificationBuilder.recognizable_constraint(alpha, R), or keep the "
        f"predicate and count with branch_counts, which needs no abstraction and pays the size of "
        f"the search tree."
    )
    raise ValueError(msg)


@dataclass
class _Clause(Generic[NT, T, G]):
    """One clause of the original program, prepared for instantiation.

    Attributes:
        head (NT): The non-terminal the clause belongs to.
        index (int): Its position among that non-terminal's clauses, so that the instances can be
            emitted in program order however the fixed point happened to find them.
        rule (RHSRule[NT, T, G]): The clause itself.
        hole_origins (tuple[NT, ...]): The non-terminals of its holes, named and anonymous alike,
            in clause order.
        slots (tuple[int | None, ...]): Per argument, the index of its hole, or None for a
            constant.
        names (tuple[str | None, ...]): Per argument, the variable name a relation sees it under.
        relations (tuple[tuple[StateRelation, int], ...]): The relations to satisfy, each with
            the index of the abstraction it decides on.
        constant_states (tuple[tuple[Any, ...], ...]): Per argument, the state of its constant
            value under each abstraction, and the entry of a hole is empty and never read. Filled in a
            second pass, because the first one is what discovers the abstractions.
    """

    head: NT
    index: int
    rule: RHSRule[NT, T, G]
    hole_origins: tuple[NT, ...]
    slots: tuple[int | None, ...]
    names: tuple[str | None, ...]
    relations: tuple[tuple[StateRelation, int], ...]
    constant_states: tuple[tuple[Any, ...], ...] = ()


def _abstraction_index(abstractions: list[TreeAbstraction], abstraction: TreeAbstraction) -> int:
    """Return the index of an abstraction, appending it if it is new.

    Linear rather than hashed: a program states one abstraction, or a handful, and requiring them
    to be hashable would rule out perfectly good abstractions for no gain. Equal abstractions
    share a component, so a repository that reuses one does not pay the product for it.

    Args:
        abstractions (list[TreeAbstraction]): The abstractions found so far, in order of occurrence.
        abstraction (TreeAbstraction): The one to look up.

    Returns:
        int: Its index.
    """
    for index, known in enumerate(abstractions):
        if known is abstraction or known == abstraction:
            return index
    abstractions.append(abstraction)
    return len(abstractions) - 1


def _prepare(
    space: SolutionSpace[NT, T, G],
) -> tuple[list[_Clause[NT, T, G]], list[TreeAbstraction]]:
    """Read the program into instantiable clauses and collect its abstractions.

    A clause without a named hole is decided here, once and for all, exactly as ``Goal.update``
    and the fill of the size table decide it: its predicates can only read its literals, so either
    they hold for it or the clause is never applicable and is dropped. Dropping it is not an
    optimization: keeping it would make its head state reachable through a rule no search can ever
    apply.

    Args:
        space (SolutionSpace[NT, T, G]): The synthesized program.

    Returns:
        tuple[list[_Clause[NT, T, G]], list[TreeAbstraction]]: The applicable clauses and the
            distinct abstractions, both in program order.
    """
    abstractions: list[TreeAbstraction] = []
    clauses: list[_Clause[NT, T, G]] = []
    for nonterminal in space.nonterminals():
        for index, rule in enumerate(space.get(nonterminal) or ()):
            relations: tuple[tuple[StateRelation, int], ...] = ()
            if rule.predicates and not _named_holes(rule):
                if not all(predicate(rule.literal_substitution) for predicate in rule.predicates):
                    continue
            elif rule.predicates:
                # Every predicate here is recognizable: the clause reads a hole, and
                # ``recognizable_or_raise`` has already named it otherwise. The cast is what says
                # so, and a second isinstance check would be a branch no input reaches.
                recognizable = cast("tuple[RecognizableConstraint, ...]", rule.predicates)
                relations = tuple(
                    (predicate.relation, _abstraction_index(abstractions, predicate.abstraction))
                    for predicate in recognizable
                )
            hole_origins: list[NT] = []
            slots: list[int | None] = []
            for argument in rule.arguments:
                if isinstance(argument, NonTerminalArgument):
                    slots.append(len(hole_origins))
                    hole_origins.append(argument.origin)
                else:
                    slots.append(None)
            clauses.append(
                _Clause(
                    head=nonterminal,
                    index=index,
                    rule=rule,
                    hole_origins=tuple(hole_origins),
                    slots=tuple(slots),
                    names=tuple(argument.name for argument in rule.arguments),
                    relations=relations,
                )
            )
    return clauses, abstractions


def _abstract_constants(clauses: Sequence[_Clause[NT, T, G]], abstractions: Sequence[TreeAbstraction]) -> None:
    """Abstract the literal values of every clause, once.

    A literal is a nullary symbol of the synthesis alphabet, so its state is ``alpha(value, ())``,
    which is the same fold :func:`cosy.core.recognizable.state_of` performs on it. That is what
    keeps the states this construction computes and the states the derived predicate computes the
    same numbers.

    A second pass, because a clause may be the one that introduces an abstraction: the states of a
    literal are only known once every abstraction of the program is.

    Args:
        clauses (Sequence[_Clause[NT, T, G]]): The prepared clauses.
        abstractions (Sequence[TreeAbstraction]): The program's abstractions.
    """
    for clause in clauses:
        clause.constant_states = tuple(
            ()
            if isinstance(argument, NonTerminalArgument)
            else tuple(abstraction(argument.value, ()) for abstraction in abstractions)
            for argument in clause.rule.arguments
        )


def _fresh_combinations(old: tuple[int, ...], sizes: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    """Enumerate the hole-state combinations that became available since the last round.

    A combination is fresh when at least one of its holes takes a state that was not there before.
    Splitting on the *smallest* such hole partitions the fresh combinations, so each one is
    enumerated exactly once. That is the whole reason the construction may emit rules while the
    fixed point is still running, which is semi-naive evaluation.

    Args:
        old (tuple[int, ...]): The number of states per hole at the previous visit.
        sizes (tuple[int, ...]): The number of states per hole now.

    Yields:
        tuple[int, ...]: One index per hole.
    """
    for split in range(len(sizes)):
        if sizes[split] == old[split]:
            continue
        ranges = [
            *(range(old[before]) for before in range(split)),
            range(old[split], sizes[split]),
            *(range(sizes[after]) for after in range(split + 1, len(sizes))),
        ]
        yield from product(*ranges)


def determinize(
    space: SolutionSpace[NT, T, G],
    start: NT,
    *,
    state_limit: int | None = DEFAULT_STATE_LIMIT,
) -> Determinization[NT, T, G]:
    """Compile the recognizable constraints of a program into its non-terminals.

    The result derives exactly the terms the original derives, along exactly one branch each, and
    carries no predicate over a hole, so it can be counted from the program
    (:func:`cosy.search.counting.size_table`) instead of from a materialized search tree.

    Args:
        space (SolutionSpace[NT, T, G]): The synthesized program. It should be pruned, since the
            construction only ever reaches productive states in any case.
        start (NT): The non-terminal the caller queries. Its states are merged into
            :class:`MergedNonTerminal`, which is the start symbol of the result.
        state_limit (int | None): How many product states to allow before reporting the
            abstraction as infinite. None disables the check, for a caller who has established
            finiteness by other means. (Default value = :data:`DEFAULT_STATE_LIMIT`)

    Returns:
        Determinization[NT, T, G]: The product program, its start symbol, the reachable states and
            the abstractions they range over.

    Raises:
        ValueError: If some clause reads a hole through a predicate without an abstraction, or if
            the reachable states exceed ``state_limit``. Both messages name what to repair.
    """
    if start not in space:
        msg = f"the start symbol {start} has no clause in this program, so there is nothing to determinize for it"
        raise ValueError(msg)
    recognizable_or_raise(space)
    clauses, abstractions = _prepare(space)
    _abstract_constants(clauses, abstractions)
    width = len(abstractions)
    merged = MergedNonTerminal(start)

    states: dict[NT, list[tuple[Any, ...]]] = {nonterminal: [] for nonterminal in space.nonterminals()}
    reached: dict[NT, set[tuple[Any, ...]]] = {nonterminal: set() for nonterminal in states}
    produced: dict[Any, list[tuple[int, T, tuple[Argument, ...]]]] = {}
    total_states = 0

    def instantiate(clause: _Clause[NT, T, G], hole_states: tuple[tuple[Any, ...], ...]) -> None:
        """Emit the instance of one clause for one assignment of states to its holes.

        Args:
            clause (_Clause[NT, T, G]): The clause to instantiate.
            hole_states (tuple[tuple[Any, ...], ...]): One state per hole, in clause order.

        Raises:
            ValueError: If the abstraction has produced more states than ``state_limit`` allows.
        """
        nonlocal total_states
        per_abstraction = [
            tuple(
                clause.constant_states[position][component] if slot is None else hole_states[slot][component]
                for position, slot in enumerate(clause.slots)
            )
            for component in range(width)
        ]
        for relation, component in clause.relations:
            substitution = {
                name: per_abstraction[component][position]
                for position, name in enumerate(clause.names)
                if name is not None
            }
            if not relation(substitution):
                return
        head_state = tuple(
            abstraction(clause.rule.terminal, per_abstraction[component])
            for component, abstraction in enumerate(abstractions)
        )
        if head_state not in reached[clause.head]:
            reached[clause.head].add(head_state)
            states[clause.head].append(head_state)
            total_states += 1
            if state_limit is not None and total_states > state_limit:
                msg = (
                    f"the abstraction reached more than {state_limit} product states, so it is "
                    f"not the finite abstraction condition (REC) asks for. An unbounded "
                    f"quantity such as a term size without a cap is the usual cause. Raise "
                    f"state_limit if the carrier really is this large."
                )
                raise ValueError(msg)
        arguments = tuple(
            argument
            if slot is None
            else NonTerminalArgument(
                argument.name,
                ProductNonTerminal(argument.origin, hole_states[slot]),
            )
            for argument, slot in zip(clause.rule.arguments, clause.slots, strict=True)
        )
        instance = (clause.index, clause.rule.terminal, arguments)
        produced.setdefault(ProductNonTerminal(clause.head, head_state), []).append(instance)
        if clause.head == start:
            produced.setdefault(merged, []).append(instance)

    inner = [clause for clause in clauses if clause.hole_origins]
    for clause in clauses:
        if not clause.hole_origins:
            instantiate(clause, ())

    visited: list[tuple[int, ...]] = [(0,) * len(clause.hole_origins) for clause in inner]
    while True:
        progressed = False
        for position, clause in enumerate(inner):
            sizes = tuple(len(states.get(origin, ())) for origin in clause.hole_origins)
            old = visited[position]
            if sizes == old:
                continue
            for combination in _fresh_combinations(old, sizes):
                instantiate(
                    clause,
                    tuple(
                        states[origin][index] for origin, index in zip(clause.hole_origins, combination, strict=True)
                    ),
                )
            visited[position] = sizes
            progressed = True
        if not progressed:
            break

    determinized: SolutionSpace[Any, T, G] = SolutionSpace()
    for head, instances in produced.items():
        # By clause index, so that the program order of the original survives the order in which
        # the fixed point happened to reach the states. The sort is stable, so the instances of
        # one clause keep the order the enumeration produced them in, and the result of the
        # construction does not depend on the shape of the fixed-point iteration.
        for _index, terminal, arguments in sorted(instances, key=lambda instance: instance[0]):
            determinized.add_rule(head, terminal, arguments, ())

    return Determinization(
        space=determinized.prune(),
        start=merged,
        states={nonterminal: tuple(reached) for nonterminal, reached in states.items()},
        abstractions=tuple(abstractions),
    )
