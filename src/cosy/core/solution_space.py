"""Solution space given by a logic program."""

from __future__ import annotations

import random
from collections import defaultdict, deque
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, MutableSet, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from itertools import product
from queue import PriorityQueue
from types import FunctionType
from typing import Any, Generic, TypeVar

from cosy.core.tree import Tree

NT = TypeVar("NT", bound=Hashable)  # type of non-terminals
T = TypeVar("T", bound=Hashable)  # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants
_E = TypeVar("_E", bound=Hashable)  # element type of _OrderedSet


@dataclass(frozen=True)
class ConstantArgument(Generic[T, G]):
    """_summary_.

    Attributes:
        name (str): _description_
        value (T): _description_
        origin (G): _description_
    """

    name: str
    value: T
    origin: G


@dataclass(frozen=True)
class NonTerminalArgument(Generic[NT]):
    """_summary_.

    Attributes:
        name (str | None): _description_
        origin (NT): _description_
    """

    name: str | None
    origin: NT


Argument = ConstantArgument[T, G] | NonTerminalArgument[NT]


@dataclass(frozen=True)
class RHSRule(Generic[NT, T, G]):
    """_summary_.

    Attributes:
        arguments (tuple[Argument, ...]): _description_
        predicates (tuple[Callable[[dict[str, Any]], bool], ...]): _description_
        terminal (T): _description_
        non_terminals (frozenset[NT]): _description_
        literal_substitution (dict[str, T]): _description_
    """

    arguments: tuple[Argument, ...]
    predicates: tuple[Callable[[dict[str, Any]], bool], ...]
    terminal: T

    @property
    def non_terminals(self) -> frozenset[NT]:
        """Set of non-terminals occurring in the body of the rule.

        Returns:
            frozenset[NT]: _description_
        """
        return frozenset(arg.origin for arg in self.arguments if isinstance(arg, NonTerminalArgument))

    @property
    def literal_substitution(self) -> dict[str, T]:
        """_summary_.

        Returns:
            dict[str, T]: _description_
        """
        return {n.name: n.value for n in self.arguments if isinstance(n, ConstantArgument)}


Path = tuple[int, ...]


class Goal(Generic[NT, T, G]):
    """A goal models a Tree/Combinatory Term with variables.

    To enable non-recursive algorithms, a goal models a Tree with several mappings from positions in a tree
    (indexed by their paths) to the information stored at this position.
    A position is either a variable/non-terminal or a grounded subtree.
    Additionally, a position has a combinator, which is applied to all children positions when all of them are grounded.
    And a position may have constraints on its children.
    If all positions are grounded and all constraints are satisfied, the goal is successful and
    the tree at the root position is a solution.
    """

    constructors: dict[Path, T]
    subgoals: dict[Path, NonTerminalArgument[NT]]
    grounded: dict[Path, tuple[str, Tree[T]]]
    constraints: dict[tuple[Path, ...], tuple[tuple[Callable[[dict[str, Any]], bool], ...], dict[str, T]]]
    success: bool

    def __init__(
        self,
        root: dict[Path, T],
        subgoals: dict[Path, NonTerminalArgument[NT]],
        grounded: dict[Path, tuple[str, Tree[T]]],
        constraints: dict[tuple[Path, ...], tuple[tuple[Callable[[dict[str, Any]], bool], ...], dict[str, T]]],
        success,
    ):
        """_summary_.

        Args:
            root (dict[Path, T]): _description_
            subgoals (dict[Path, NonTerminalArgument[NT]]): _description_
            grounded (dict[Path, tuple[str, Tree[T]]]): _description_
            constraints (dict[tuple[Path, ...], tuple[tuple[Callable[[dict[str, Any]], bool], ...], dict[str, T]]]): _description_
            success (_type_): _description_
        """
        self.constructors = root
        self.subgoals = subgoals
        self.grounded = grounded
        self.constraints = constraints
        self.success = success

    @classmethod
    def from_rhs_rule(cls, rhs: RHSRule[NT, T, G]) -> Goal[NT, T, G] | None:
        """Create a goal from an RHSRule.

        The terminal becomes the combinator applied at the root.
        The arguments become the children of the root and are either grounded (ConstantArgument)
        or ungrounded (NonTerminalArgument).
        The constraints are the predicates from the RHSRule and to ensure a correct substitution of variable names,
        the local variable names from the RHSRule are stored additionally to the predicates that are applied at
        the given positions.
        If the rule has no named non-terminal argument, there is no such position to store its predicates at.
        They are then decided right here, on the literal substitution of the rule, and None is returned if one of
        them is violated.

        Args:
            rhs (RHSRule[NT, T, G]): _description_

        Returns:
            Goal[NT, T, G] | None: _description_

        Raises:
            TypeError: _description_
        """
        subgoals: dict[Path, NonTerminalArgument[NT]] = {}
        grounded: dict[Path, tuple[str, Tree[T]]] = {}
        named: tuple[Path, ...] = ()
        for i, arg in enumerate(rhs.arguments):
            if isinstance(arg, NonTerminalArgument):
                subgoals[(i,)] = arg
                if arg.name is not None:
                    named += ((i,),)
            elif isinstance(arg, ConstantArgument):
                grounded[(i,)] = arg.name, Tree(arg.value, ())
            else:
                msg = f"Argument {arg} is neither a NonTerminalArgument nor a ConstantArgument"
                raise TypeError(msg)
        root: dict[Path, T] = {(): rhs.terminal}
        constraints = ({named: (rhs.predicates, rhs.literal_substitution)} if named else {}) if rhs.predicates else {}
        # A rule without a named non-terminal argument cannot deposit its predicates: `constraints` is keyed by
        # the tuple of named positions, and the cascade in `update` indexes those keys with `ps[0][:-1]`, which
        # an empty tuple cannot answer. No deposit is needed either. As the docstring of `resolution` states,
        # the predicates of a rule are applied to the substitution given by its constant arguments and its
        # named non-terminal arguments; an unnamed argument is invisible to them. So `literal_substitution` is
        # already the full substitution, and the rule decides here -- the ground rule as well as the rule whose
        # non-terminal arguments are all unnamed, which is the shape an arrow type in a suffix produces.
        # `enumerate_trees` and `contains_tree` decide these predicates the same way; without this branch the
        # resolution is the only one of the three that does not, at the root for the rule with only unnamed
        # non-terminal arguments and below it (see `update`) for the ground rule as well. The check replaced
        # here decided the ground rule at the root on `dict(grounded.values()) | rhs.literal_substitution`,
        # which is the same dict: `grounded` holds only constant arguments at this point, so the literal
        # substitution wins every key it contributes.
        if rhs.predicates and not named and not all(c(rhs.literal_substitution) for c in rhs.predicates):
            return None
        if not subgoals:
            grounded[()] = "", Tree(rhs.terminal, tuple(grounded[p][1] for p in sorted(grounded.keys())))
            return Goal(root, subgoals, grounded, constraints, success=True)
        return Goal(root, subgoals, grounded, constraints, success=False)

    def update(self, rhs: RHSRule[NT, T, G], position: Path) -> Goal[NT, T, G] | None:
        """Update the goal by applying the given rule at the given position.

        If the rule cannot be applied (because a constraint/predicate is violated) at the given position, return None.

        Args:
            rhs (RHSRule[NT, T, G]): _description_
            position (Path): _description_

        Returns:
            Goal[NT, T, G] | None: _description_

        Raises:
            TypeError: _description_
            ValueError: _description_
            AssertionError: _description_
        """
        new_subgoals: dict[Path, NonTerminalArgument[NT]] = self.subgoals.copy()
        new_grounded: dict[Path, tuple[str, Tree[T]]] = self.grounded.copy()
        named: tuple[Path, ...] = ()

        is_ground = True

        children: tuple[Tree[T], ...] = ()

        # apply the rule at the given position
        for i, arg in enumerate(rhs.arguments):
            new_position = (*position, i)
            if isinstance(arg, NonTerminalArgument):
                is_ground = False
                new_subgoals[new_position] = arg
                if arg.name is not None:
                    named += (new_position,)
            elif isinstance(arg, ConstantArgument):
                new_grounded[new_position] = arg.name, Tree(arg.value, ())
                children += (Tree(arg.value, ()),)
            else:
                msg = f"Argument {arg} is neither a NonTerminalArgument nor a ConstantArgument"
                raise TypeError(msg)

        new_constructors = self.constructors.copy()
        new_constructors[position] = rhs.terminal
        new_constraints = self.constraints.copy()
        if rhs.predicates and named:
            new_constraints[named] = rhs.predicates, rhs.literal_substitution
        # The same decision for a rule applied below the root: without a named non-terminal argument there is no
        # position to deposit the predicates at, and `literal_substitution` is the whole substitution they are
        # applied to (see `from_rhs_rule`). Without this branch such a rule applies unchecked everywhere but at
        # the root, while `enumerate_trees` and `contains_tree` reject the resulting term.
        elif rhs.predicates and not named and not all(c(rhs.literal_substitution) for c in rhs.predicates):
            return None

        common_prefix = position[:-1]

        if is_ground:
            """
                   If applying the rule leads to a ground tree at the position, we check bottom up,
                   if the constraints are satisfied and if all subgoals on the same level are grounded.
                   If this is the case, we can ground the parent goal as well,
                   which can lead to a cascade of cumulating subtrees into bigger ones up to the root.
                   If the root grounded, we have found a solution.
            """
            nt: NonTerminalArgument[NT] = new_subgoals.pop(position)
            tree = Tree(rhs.terminal, children)
            new_grounded[position] = (nt.name, tree) if nt.name is not None else ("", tree)
            # if a parent position becomes grounded, there is no need to map the children position to subtrees anymore
            for p in [x for x in new_grounded if x[:-1] == position]:
                new_grounded.pop(p)
            # move the path bottom up and check if all children are grounded and the parent can be grounded as well
            while common_prefix:
                subgoal_level_pos = [p for p in new_subgoals if p[:-1] == common_prefix]
                grounded_level_pos = [p for p in new_grounded if p[:-1] == common_prefix]
                if subgoal_level_pos:
                    break
                # if all arguments are grounded, we can ground the parent as well, if the constraints are satisfied
                # check all constraints
                preds = [ps for ps in new_constraints if ps[0][:-1] == common_prefix]
                for ps in preds:
                    constraints, literal_substitution = new_constraints[ps]
                    args: tuple[tuple[str, Tree[T]], ...] = tuple(new_grounded[p] for p in ps)
                    substitution = dict(args) | literal_substitution
                    if not all(c(substitution) for c in constraints):
                        return None
                # sort the positions by their last element,
                # which corresponds to the position in the arguments of the parent position
                sorted_positions = sorted(grounded_level_pos, key=lambda p: p[-1])
                children = tuple(new_grounded[p][1] for p in sorted_positions)
                # construct the tree for the parent position
                tree = Tree(new_constructors[position[:-1]], children)
                if position[:-1] in new_subgoals:
                    nt = new_subgoals.pop(position[:-1])
                    new_grounded[position[:-1]] = (nt.name, tree) if nt.name is not None else ("", tree)
                else:
                    msg = "the parent to a nonterminal must be a nonterminal as well"
                    raise ValueError(msg)
                # tidy up
                for p in grounded_level_pos:
                    new_grounded.pop(p)
                if position in new_subgoals:
                    new_subgoals.pop(position)
                position = position[:-1]
                common_prefix = position[:-1]

            if len(new_subgoals) == 0:
                # if there are no subgoals left, the root must be grounded
                if common_prefix != ():
                    msg = "common_prefix should be empty when all subgoals are grounded"
                    raise AssertionError(msg)
                # check all constraints and return None if a not all constraints are satisfied
                preds = [ps for ps in new_constraints if ps[0][:-1] == ()]
                for ps in preds:
                    constraints, literal_substitution = new_constraints[ps]
                    args = tuple(new_grounded[p] for p in ps)
                    substitution = dict(args) | literal_substitution
                    if not all(c(substitution) for c in constraints):
                        return None
                # sort the positions by their last element,
                # which corresponds to the position in the arguments of the parent position
                sorted_positions = sorted(new_grounded.keys(), key=lambda p: p[-1])
                children = tuple(new_grounded[p][1] for p in sorted_positions)
                # construct the tree for the root position, the derivation
                tree = Tree(new_constructors[()], children)
                new_grounded[()] = "", tree
            return Goal(new_constructors, new_subgoals, new_grounded, new_constraints, success=len(new_subgoals) == 0)
        """
        If applying the rule does not lead to a ground tree at the position,
        we return the updated goal.
        """
        return Goal(new_constructors, new_subgoals, new_grounded, new_constraints, success=False)


class _OrderedSet(MutableSet[_E]):
    """A set that iterates in the order its elements were first added.

    ``set`` iterates in hash order, which varies with ``PYTHONHASHSEED`` and, for elements whose
    hash falls back to ``id()``, with the memory addresses of one particular run. A tree whose
    terminal is a plain function object -- as a CoSy combinator usually is -- hashes by identity,
    so a plain set makes an enumeration built on it unrepeatable. This class keeps the set
    semantics and fixes the order; ``dict`` supplies the ordering, and its keys are a set.

    Only ``__contains__``, ``__iter__``, ``__len__``, ``add`` and ``discard`` are defined here; the
    rest of the set interface, ``pop`` included, comes from ``MutableSet``. ``pop`` therefore
    returns ``next(iter(self))``, which is the element that was added first.
    """

    __slots__ = ("_items",)

    def __init__(self, items: Iterable[_E] = ()) -> None:
        """Build an ordered set, keeping the order of first occurrence.

        Args:
            items (Iterable[_E]): The initial elements. (Default value = ())
        """
        self._items: dict[_E, None] = dict.fromkeys(items)

    def __contains__(self, item: object) -> bool:
        """Check membership.

        Args:
            item (object): The candidate element.

        Returns:
            bool: True if the element is in the set.
        """
        return item in self._items

    def __iter__(self) -> Iterator[_E]:
        """Iterate over the elements in insertion order.

        Returns:
            Iterator[_E]: The elements, oldest first.
        """
        return iter(self._items)

    def __len__(self) -> int:
        """Return the number of elements.

        Returns:
            int: The number of elements.
        """
        return len(self._items)

    def __repr__(self) -> str:
        """Return a representation that shows the order.

        Returns:
            str: The representation.
        """
        return f"{type(self).__name__}({list(self._items)!r})"

    def add(self, value: _E) -> None:
        """Add an element, keeping the position of an element that is already present.

        Args:
            value (_E): The element to add.
        """
        self._items[value] = None

    def discard(self, value: _E) -> None:
        """Remove an element if it is present.

        Args:
            value (_E): The element to remove.
        """
        self._items.pop(value, None)


@dataclass
class _RuleProgress(Generic[T]):
    """_summary_.

    Attributes:
        done_parameters (list[tuple[Tree[T] | None, ...]]): _description_
        seen_parameters (set[tuple[Tree[T] | None, ...]]): _description_
        done_arguments (list[tuple[Tree[T] | None, ...]]): _description_
        seen_arguments (set[tuple[Tree[T] | None, ...]]): _description_
        pending_parameters (list[tuple[Tree[T] | None, ...]]): _description_
        pending_arguments (list[tuple[Tree[T] | None, ...]]): _description_
    """

    done_parameters: list[tuple[Tree[T] | None, ...]] = field(default_factory=list)
    seen_parameters: set[tuple[Tree[T] | None, ...]] = field(default_factory=set)
    done_arguments: list[tuple[Tree[T] | None, ...]] = field(default_factory=list)
    seen_arguments: set[tuple[Tree[T] | None, ...]] = field(default_factory=set)
    pending_parameters: list[tuple[Tree[T] | None, ...]] = field(default_factory=list)
    pending_arguments: list[tuple[Tree[T] | None, ...]] = field(default_factory=list)


class SolutionSpace(Generic[NT, T, G]):
    """_summary_."""

    _rules: defaultdict[NT, deque[RHSRule[NT, T, G]]]

    def __init__(self, rules: dict[NT, deque[RHSRule[NT, T, G]]] | None = None) -> None:
        """_summary_.

        Args:
            rules (dict[NT, deque[RHSRule[NT, T, G]]] | None): _description_ (Default value = None)
        """
        if rules is None:
            rules = defaultdict(deque)
        self._rules = defaultdict(deque, rules)

    def get(self, nonterminal: NT) -> deque[RHSRule[NT, T, G]] | None:
        """_summary_.

        Args:
            nonterminal (NT): _description_

        Returns:
            deque[RHSRule[NT, T, G]] | None: _description_
        """
        return self._rules.get(nonterminal)

    def __contains__(self, nonterminal: object) -> bool:
        """Report whether a non-terminal has an entry in this solution space.

        This is the cheap membership test. ``nonterminals()`` returns a snapshot, so testing
        against it is linear in the size of the grammar and rebuilds the snapshot every time.

        Args:
            nonterminal (object): The non-terminal to look for.

        Returns:
            bool: True if the solution space has an entry for it.
        """
        return nonterminal in self._rules

    def __getitem__(self, nonterminal: NT) -> deque[RHSRule[NT, T, G]]:
        """Return the rules of a non-terminal, without creating it.

        Reading is not a mutation. Going through ``defaultdict.__getitem__`` used to insert an
        empty entry for every unknown non-terminal that was merely looked at, so that enumerating
        or inspecting a solution space grew its set of non-terminals. A missing non-terminal is
        now reported instead of invented; ``get`` is the accessor for "maybe present".

        Args:
            nonterminal (NT): The non-terminal to look up.

        Returns:
            deque[RHSRule[NT, T, G]]: The stored deque of rules. It is the live one, so appending
                to it adds a rule, exactly as ``add_rule`` does.

        Raises:
            KeyError: If the solution space has no entry for this non-terminal.
        """
        rules = self._rules.get(nonterminal)
        if rules is None:
            raise KeyError(nonterminal)
        return rules

    def _rules_of(self, nonterminal: NT) -> deque[RHSRule[NT, T, G]]:
        """Return the rules of a non-terminal for reading only.

        Internal read path for the places that treat "no rules" and "no such non-terminal" alike
        and would otherwise need a guard around every lookup. The empty deque it returns for an
        unknown non-terminal is not stored, which is why this is private: appending to it is lost.

        Args:
            nonterminal (NT): The non-terminal to look up.

        Returns:
            deque[RHSRule[NT, T, G]]: The stored rules, or a fresh empty deque.
        """
        rules = self._rules.get(nonterminal)
        return rules if rules is not None else deque()

    def nonterminals(self) -> tuple[NT, ...]:
        """Return the non-terminals of this solution space, in the order they were added.

        Returns:
            tuple[NT, ...]: A snapshot. Returning the live ``keys()`` view meant the result changed
                underneath a caller that was still iterating it. Because this is a copy, use
                ``nonterminal in space`` rather than ``in space.nonterminals()`` to test membership.
        """
        return tuple(self._rules.keys())

    def as_tuples(self) -> Iterable[tuple[NT, deque[RHSRule[NT, T, G]]]]:
        """_summary_.

        Returns:
            Iterable[tuple[NT, deque[RHSRule[NT, T, G]]]]: _description_
        """
        return self._rules.items()

    def add_rule(
        self,
        nonterminal: NT,
        terminal: T,
        arguments: tuple[Argument, ...],
        predicates: tuple[Callable[[dict[str, Any]], bool], ...],
    ) -> None:
        """_summary_.

        Args:
            nonterminal (NT): _description_
            terminal (T): _description_
            arguments (tuple[Argument, ...]): _description_
            predicates (tuple[Callable[[dict[str, Any]], bool], ...]): _description_
        """
        self._rules[nonterminal].append(RHSRule(arguments, predicates, terminal))

    def show(self) -> str:
        """_summary_.

        Returns:
            str: _description_
        """
        return "\n".join(
            f"{nt!s} ~> {' | '.join([str(subrule) for subrule in rule])}" for nt, rule in self._rules.items()
        )

    def prune(self) -> SolutionSpace[NT, T, G]:
        """Keep only productive rules.

        Returns:
            SolutionSpace[NT, T, G]: _description_
        """

        # Insertion-ordered sets: the keys of a dict are a set, and unlike a plain set they keep
        # the order in which they were added. That order is observable here -- the order in which
        # ground types are discovered becomes the key order of the pruned space and therefore the
        # order of its nonterminals(). Plain sets made it depend on PYTHONHASHSEED, down to the
        # order of the inverse grammar, which decides which non-terminal is discovered next.
        ground_types: dict[NT, None] = {}
        queue: dict[NT, None] = {}
        inverse_grammar: dict[NT, dict[tuple[NT, frozenset[NT]], None]] = defaultdict(dict)

        for n, exprs in self._rules.items():
            for expr in exprs:
                non_terminals = expr.non_terminals
                for m in non_terminals:
                    inverse_grammar[m][(n, non_terminals)] = None
                if not non_terminals:
                    queue[n] = None

        while queue:
            n = next(iter(queue))  # oldest entry first, so the traversal follows discovery order
            del queue[n]
            if n not in ground_types:
                ground_types[n] = None
                for m, non_terminals in inverse_grammar[n]:
                    if m not in ground_types and all(t in ground_types for t in non_terminals):
                        queue[m] = None

        return SolutionSpace[NT, T, G](
            defaultdict(
                deque,
                {
                    target: deque(
                        possibility
                        for possibility in self._rules_of(target)
                        if all(t in ground_types for t in possibility.non_terminals)
                    )
                    for target in ground_types
                },
            )
        )

    def _enumerate_tree_vectors(
        self,
        non_terminals: Sequence[NT | None],
        existing_terms: Mapping[NT, AbstractSet[Tree[T]]],
        nt_term: tuple[NT, Tree[T]] | None = None,
    ) -> Iterable[tuple[Tree[T] | None, ...]]:
        """Enumerate possible term vectors for a given list of non-terminals and existing terms. Use nt_term at least once (if given).

        Args:
            non_terminals (Sequence[NT | None]): _description_
            existing_terms (Mapping[NT, AbstractSet[Tree[T]]]): _description_
            nt_term (tuple[NT, Tree[T]] | None): _description_ (Default value = None)

        Yields:
            tuple[Tree[T] | None, ...]: _description_
        """
        if nt_term is None:
            yield from product(*([n] if n is None else existing_terms[n] for n in non_terminals))
        else:
            nt, term = nt_term
            for i, n in enumerate(non_terminals):
                if n == nt:
                    arg_lists: Iterable[Iterable[Tree[T] | None]] = (
                        [None] if m is None else [term] if i == j else existing_terms[m]
                        for j, m in enumerate(non_terminals)
                    )
                    yield from product(*arg_lists)

    def _generate_new_trees(
        self,
        rule: RHSRule[NT, T, G],
        existing_terms: Mapping[NT, AbstractSet[Tree[T]]],
        interpretation: dict[T, Any] | None = None,
        max_count: int | None = None,
        nt_old_term: tuple[NT, Tree[T]] | None = None,
        progress: _RuleProgress[T] | None = None,
    ) -> _OrderedSet[Tree[T]]:
        # Genererate new terms for rule `rule` from existing terms up to `max_count`
        # the term `old_term` should be a subterm of all resulting terms, at a position, that corresponds to `nt`

        """_summary_.

        Args:
            rule (RHSRule[NT, T, G]): _description_
            existing_terms (Mapping[NT, AbstractSet[Tree[T]]]): _description_
            interpretation (dict[T, Any] | None): _description_ (Default value = None)
            max_count (int | None): _description_ (Default value = None)
            nt_old_term (tuple[NT, Tree[T]] | None): _description_ (Default value = None)
            progress (_RuleProgress[T] | None): _description_ (Default value = None)

        Returns:
            _OrderedSet[Tree[T]]: The new terms, in the order they were generated. The caller
                enumerates them in that order, so a plain set would leave it to the hash seed
                which terms an interrupted enumeration returns.
        """
        output_set: _OrderedSet[Tree[T]] = _OrderedSet()
        if max_count == 0:
            return output_set

        named_non_terminals = [
            a.origin if isinstance(a, NonTerminalArgument) and a.name is not None else None for a in rule.arguments
        ]
        unnamed_non_terminals = [
            a.origin if isinstance(a, NonTerminalArgument) and a.name is None else None for a in rule.arguments
        ]
        literal_arguments = [Tree(a.value, ()) if isinstance(a, ConstantArgument) else None for a in rule.arguments]

        def interleave(
            parameters: Sequence[Tree[T] | None],
            literal_arguments: Sequence[Tree[T] | None],
            arguments: Sequence[Tree[T] | None],
        ) -> Iterable[Tree[T]]:
            """Interleave parameters, literal arguments and arguments.

            Args:
                parameters (Sequence[Tree[T] | None]): _description_
                literal_arguments (Sequence[Tree[T] | None]): _description_
                arguments (Sequence[Tree[T] | None]): _description_

            Yields:
                Tree[T]: _description_

            Raises:
                ValueError: _description_
            """
            for parameter, literal_argument, argument in zip(parameters, literal_arguments, arguments, strict=True):
                if parameter is not None:
                    yield parameter
                elif literal_argument is not None:
                    yield literal_argument
                elif argument is not None:
                    yield argument
                else:
                    msg = "All arguments of interleave are None"
                    raise ValueError(msg)

        def construct_tree(
            rule: RHSRule[NT, T, G],
            parameters: Sequence[Tree[T] | None],
            literal_arguments: Sequence[Tree[T] | None],
            arguments: Sequence[Tree[T] | None],
        ) -> Tree[T]:
            """Construct a new tree from the rule and the given specific arguments.

            Args:
                rule (RHSRule[NT, T, G]): _description_
                parameters (Sequence[Tree[T] | None]): _description_
                literal_arguments (Sequence[Tree[T] | None]): _description_
                arguments (Sequence[Tree[T] | None]): _description_

            Returns:
                Tree[T]: _description_
            """
            return Tree(
                rule.terminal,
                tuple(interleave(parameters, literal_arguments, arguments)),
            )

        def specific_substitution(parameters: Sequence[Tree[T] | None]):
            """_summary_.

            Args:
                parameters (Sequence[Tree[T] | None]): _description_

            Returns:
                _type_: _description_
            """
            return {
                a.name: p if interpretation is None else p.interpret(interpretation)
                for p, a in zip(parameters, rule.arguments, strict=True)
                if isinstance(a, NonTerminalArgument) and a.name is not None and p is not None
            } | rule.literal_substitution

        def valid_parameters(
            nt_term: tuple[NT, Tree[T]] | None,
        ) -> Iterable[tuple[Tree[T] | None, ...]]:
            """Enumerate all valid parameters for the rule.

            Args:
                nt_term (tuple[NT, Tree[T]] | None): _description_

            Yields:
                tuple[Tree[T] | None, ...]: _description_
            """
            for parameters in self._enumerate_tree_vectors(named_non_terminals, existing_terms, nt_term):
                if rule.predicates:
                    # compute the specific substitution only if there are predicates
                    substitution = specific_substitution(parameters)
                    if all(predicate(substitution) for predicate in rule.predicates):
                        yield parameters
                else:
                    yield parameters

        # Legacy behaviour intact if no progress object passed
        if progress is None:
            for parameters in valid_parameters(nt_old_term):
                for arguments in self._enumerate_tree_vectors(unnamed_non_terminals, existing_terms):
                    output_set.add(construct_tree(rule, parameters, literal_arguments, arguments))
                    if max_count is not None and len(output_set) >= max_count:
                        return output_set

            if nt_old_term is not None:
                all_parameters: deque[tuple[Tree[T] | None, ...]] | None = None
                for arguments in self._enumerate_tree_vectors(unnamed_non_terminals, existing_terms):
                    all_parameters = all_parameters if all_parameters is not None else deque(valid_parameters(None))
                    for parameters in all_parameters:
                        output_set.add(construct_tree(rule, parameters, literal_arguments, arguments))
                        if max_count is not None and len(output_set) >= max_count:
                            return output_set
            return output_set

        # NOTE: THIS WILL BREAK IF PREDICATES ARE NON-DETERMINISTIC

        # seed with prior interrupted work
        incoming_parameters = progress.pending_parameters
        progress.pending_parameters = []

        # only new parameters, old ones fetched from progress
        for parameters in valid_parameters(nt_old_term):
            if parameters not in progress.seen_parameters:
                progress.seen_parameters.add(parameters)
                incoming_parameters.append(parameters)
        incoming_arguments = progress.pending_arguments
        progress.pending_arguments = []

        # only new arguments, old ones fetched from progress
        for arguments in self._enumerate_tree_vectors(unnamed_non_terminals, existing_terms, nt_old_term):
            if arguments not in progress.seen_arguments:
                progress.seen_arguments.add(arguments)
                incoming_arguments.append(arguments)

        # Don't recompute combinations of previously seen arguments and parameters, they get deduped anyway later
        # Instead only compute new combinations: new params with new args, new params with old args, and new args with old params.
        known = len(progress.done_parameters)
        if incoming_parameters:
            # instead of recomputing this expensively, build it from progress
            every_argument = progress.done_arguments + incoming_arguments
            for index, parameters in enumerate(incoming_parameters):
                for arguments in every_argument:
                    output_set.add(construct_tree(rule, parameters, literal_arguments, arguments))
                    if max_count is not None and len(output_set) >= max_count:
                        # If max count interrupts, unprocessed but seen things are pending work next time around
                        progress.pending_parameters = incoming_parameters[index:]
                        progress.pending_arguments = incoming_arguments
                        return output_set
                progress.done_parameters.append(parameters)
        if incoming_arguments:
            # new_parameters are after known length snapshot, so old are before that
            old_parameters = progress.done_parameters[:known]
            for index, arguments in enumerate(incoming_arguments):
                for parameters in old_parameters:
                    output_set.add(construct_tree(rule, parameters, literal_arguments, arguments))
                    if max_count is not None and len(output_set) >= max_count:
                        # parameters are done already, only arguments can be pending
                        progress.pending_arguments = incoming_arguments[index:]
                        return output_set
                progress.done_arguments.append(arguments)
        return output_set

    def enumerate_trees(
        self,
        start: NT,
        max_count: int | None = None,
        max_bucket_size: int | None = None,
        interpretation: dict[T, Any] | None = None,
    ) -> Iterable[Tree[T]]:
        """Enumerate terms as an iterator efficiently - all terms are enumerated.

        The term order is not specified, but it is reproducible: the same solution space yields the
        same sequence in every process. Under ``max_count`` that also fixes which terms are
        returned, not merely the order they arrive in.

        Args:
            start (NT): _description_
            max_count (int | None): _description_ (Default value = None)
            max_bucket_size (int | None): _description_ (Default value = None)
            interpretation (dict[T, Any] | None): _description_ (Default value = None)

        Yields:
            Tree[T]: _description_
        """
        if start not in self:  # O(1), and unlike `in self.nonterminals()` it builds no snapshot
            return

        # nonterminals() now returns a snapshot, so take it once instead of once per use.
        all_nonterminals = self.nonterminals()
        queues: dict[NT, PriorityQueue[Tree[T]]] = {n: PriorityQueue() for n in all_nonterminals}
        existing_terms: dict[NT, _OrderedSet[Tree[T]]] = {n: _OrderedSet() for n in all_nonterminals}
        inverse_grammar: dict[NT, deque[tuple[NT, RHSRule[NT, T, G]]]] = {n: deque() for n in all_nonterminals}
        all_results: set[Tree[T]] = set()
        progressed: dict[tuple[NT, int], _RuleProgress[T]] = {}

        for n, exprs in self._rules.items():
            for expr in exprs:
                if all(m in self for m in expr.non_terminals):
                    for m in expr.non_terminals:
                        inverse_grammar[m].append((n, expr))
                    for new_term in self._generate_new_trees(
                        expr,
                        existing_terms,
                        interpretation,
                        progress=progressed.setdefault((n, id(expr)), _RuleProgress()),
                    ):
                        queues[n].put(new_term)
                        if n == start and new_term not in all_results:
                            if max_count is not None and len(all_results) >= max_count:
                                return
                            yield new_term
                            all_results.add(new_term)

        current_bucket_size = 1

        while (max_bucket_size is None or current_bucket_size <= max_bucket_size) and any(
            not queue.empty() for queue in queues.values()
        ):
            # The working set decides which non-terminal is expanded next, and with it which
            # terms an enumeration under max_count returns; a plain set left that to the hash seed.
            non_terminals = _OrderedSet(n for n in all_nonterminals if not queues[n].empty())

            while non_terminals:
                n = non_terminals.pop()
                results = existing_terms[n]
                while len(results) < current_bucket_size and not queues[n].empty():
                    term = queues[n].get()
                    if term in results:
                        continue
                    results.add(term)
                    for m, expr in inverse_grammar[n]:
                        if len(existing_terms[m]) < current_bucket_size:
                            non_terminals.add(m)
                        if m == start:
                            for new_term in self._generate_new_trees(
                                expr,
                                existing_terms,
                                interpretation,
                                max_count,
                                (n, term),
                                progressed.setdefault((m, id(expr)), _RuleProgress()),
                            ):
                                if new_term not in all_results:
                                    if max_count is not None and len(all_results) >= max_count:
                                        return
                                    yield new_term
                                    all_results.add(new_term)
                                    queues[start].put(new_term)
                        else:
                            for new_term in self._generate_new_trees(
                                expr,
                                existing_terms,
                                interpretation,
                                max_bucket_size,
                                (n, term),
                                progressed.setdefault((m, id(expr)), _RuleProgress()),
                            ):
                                queues[m].put(new_term)
            current_bucket_size += 1
        return

    # --- helpers for goal_from_tree / contains_tree ---------------------------------
    def _rule_matches_subtree(self, rhs: RHSRule[NT, T, G], subtree: Tree[T]) -> bool:
        """Return True if rhs can match the given subtree head (arity, terminal and constant leaf args).

        Args:
            rhs (RHSRule[NT, T, G]): _description_
            subtree (Tree[T]): _description_

        Returns:
            bool: _description_
        """
        if len(rhs.arguments) != len(subtree.children):
            return False
        if rhs.terminal != subtree.root:
            return False
        # all constant arguments must match a leaf child with the same root
        for argument, child in zip(rhs.arguments, subtree.children, strict=True):
            if isinstance(argument, ConstantArgument) and not (
                len(child.children) == 0 and argument.value == child.root
            ):
                return False
        return True

    def _initial_goals_for(self, start: NT, tree: Tree[T]) -> list[Goal[NT, T, G]]:
        """Build initial goals from rules for `start` that match the root `tree`.

        Filters out None results from Goal.from_rhs_rule.

        Args:
            start (NT): _description_
            tree (Tree[T]): _description_

        Returns:
            list[Goal[NT, T, G]]: _description_
        """
        goals: list[Goal[NT, T, G]] = []
        for rhs in self._rules_of(start):
            if self._rule_matches_subtree(rhs, tree):
                g = Goal.from_rhs_rule(rhs)
                if g is not None:
                    goals.append(g)
        return goals

    def _expand_goal_at(self, goal: Goal[NT, T, G], child_pos: Path, tree: Tree[T]) -> list[Goal[NT, T, G]]:
        """Expand a single subgoal at `child_pos` of `goal` and return the list of resulting goals.

        This calls `goal.update(...)` for every rule that matches the corresponding subtree and
        filters out None results.

        Args:
            goal (Goal[NT, T, G]): _description_
            child_pos (Path): _description_
            tree (Tree[T]): _description_

        Returns:
            list[Goal[NT, T, G]]: _description_
        """
        try:
            subtree = tree.subtree_at(child_pos)
        except IndexError:
            return []
        nt = goal.subgoals[child_pos].origin
        results: list[Goal[NT, T, G]] = []
        for rhs in self._rules_of(nt):
            if self._rule_matches_subtree(rhs, subtree):
                new = goal.update(rhs, child_pos)
                if new is not None:
                    results.append(new)
        return results

    def _is_goal_for_position(self, goal: Goal[NT, T, G], pos: Path, is_pos_leaf: bool) -> bool:
        """Return True if goal successfully represents pos as the single variation point.

        This is true if:
        - pos is a nonterminal combinator (exactly one open subgoal at pos), or
        - pos is a leaf literal (goal is successful with no open subgoals)

        Args:
            goal (Goal[NT, T, G]): _description_
            pos (Path): _description_
            is_pos_leaf (bool): _description_

        Returns:
            bool: _description_
        """
        valid_children = [
            p for p in goal.subgoals if not any(p != other and p == other[: len(p)] for other in goal.subgoals)
        ]
        # Case: pos is a nonterminal (combinator) — exactly one open subgoal at pos
        has_open_at_pos = len(valid_children) == 1 and pos in valid_children

        # Case: pos is a leaf literal — no open subgoals and goal is successful
        is_complete_leaf = len(goal.subgoals) == 0 and is_pos_leaf and goal.success

        return has_open_at_pos or is_complete_leaf

    def goal_from_tree(self, start: NT, tree: Tree[T], pos: Path) -> Iterable[Goal[NT, T, G]]:
        """Build the goals of the partial-term query for ``tree`` with a variable at ``pos``.

        Yields each goal once. For ``pos == ()`` these are the initial goals of every clause of
        ``start``; otherwise a goal fixes ``tree`` around ``pos`` and leaves one open subgoal
        there -- or none, where a clause covers a leaf ``pos`` with a constant argument. Those
        goals describe the subterms that complete the partial term into an inhabitant.

        A non-terminal is in general reached by more than one clause: for an intersection of
        arrows the inhabitation emits one clause per admissible subset of paths, and those clauses
        ask different sorts of their arguments. Stopping at the first goal found drops the
        completions of every other clause.

        The traversal expands one subgoal per step, the deepest open one other than ``pos``.
        Expanding all open subgoals of a goal at once and pushing every result reaches the same
        goal once per expansion order.

        Args:
            start (NT): The queried non-terminal.
            tree (Tree[T]): The prescribed term. Nothing at or below ``pos`` constrains the query,
                except where the enclosing clause prescribes a constant argument at ``pos``.
            pos (Path): The position of the variable.

        Yields:
            Goal[NT, T, G]: One goal per success branch of the query.
        """

        if start not in self:  # O(1), and unlike `in self.nonterminals()` it builds no snapshot
            return

        # validate pos by attempting to access the subtree once (avoids materializing all positions)
        try:
            _ = tree.subtree_at(pos) if pos != () else tree
        except IndexError:
            return

        if pos == ():
            # The variable is the whole term, so nothing of ``tree`` constrains the query.
            # ``_initial_goals_for`` matches the clauses against the root of ``tree``, which the
            # caller is about to replace, and would keep only those sharing its terminal.
            for rhs in self._rules_of(start):
                initial = Goal.from_rhs_rule(rhs)
                if initial is not None:
                    yield initial
            return

        # compute leaf positions once, reuse for all checks
        leaf_positions = tree.leaf_positions()
        is_pos_leaf = pos in leaf_positions

        pending_goals: deque[Goal[NT, T, G]] = deque(self._initial_goals_for(start, tree))

        while pending_goals:
            goal = pending_goals.pop()
            if self._is_goal_for_position(goal, pos, is_pos_leaf):
                yield goal
                continue
            # a child subgoal has a position as key that is no prefix of another key position
            open_children = [
                p
                for p in goal.subgoals
                if p != pos and not any(p != other and p == other[: len(p)] for other in goal.subgoals)
            ]
            # ``open_children`` is not empty: it is empty only when every non-prefix subgoal is
            # ``pos``, which is what ``_is_goal_for_position`` accepts above.
            # Deepest open subgoal, leftmost among those: the selection the uninformed
            # resolution strategies use.
            deepest = max(len(child) for child in open_children)
            child_pos = min(child for child in open_children if len(child) == deepest)
            pending_goals.extend(self._expand_goal_at(goal, child_pos, tree))

    def resolution(
        self,
        start: NT,
        variance_strategy_push: Callable[[deque[Goal], Iterable[Goal]], deque[Goal]],
        variance_strategy_pop: Callable[[deque[Goal]], tuple[deque[Goal], Goal]],
        subgoal_selection_strategy: Callable[[Goal], tuple[Path, NonTerminalArgument[NT]]],
        max_count: int | None = None,
        max_depth: int | None = None,
        tree: Tree[T] | None = None,
        pos: Path | None = None,
    ) -> Iterable[Tree[T]]:
        """Enumerate terms implemented via SLD-Resolution.

        The NT start is the request/ first goal.

        If the solution space is not pruned, resolution may lead to unexpected behavior.

        It is important to note, that a solution space differs from a logic program as follows:
        While the CLSP synthesizes a logic program of the following form:

        NT(T(X_0, ..., X_n)) :- NT_0(X_0), ..., NT_n(X_n), P_1(X_0, ..., X_n), ..., P_m(X_0, ..., X_n).
        NT(T()).
        Start(X) :- NT(X).
        Where NT, NT_0, ..., NT_n are non-terminals, T is a terminal, X_0, ..., X_n are variables and P_1, ..., P_m are
        (black box) predicates.

        The solution space contains rules of the following form:

        NT ~> (arguments, predicates, terminal)

        Where arguments is a tuple of Arguments, which can be either constant arguments (ConstantArgument)
        or non-terminal arguments (NonTerminalArgument).
        These arguments can be named, where the name corresponds to the variable in the logic program, or unnamed,
        where the name is None. An unnamed argument still corresponds to NT_i(X_i) in the logic program,
        but it cannot be used in the predicates, and we need to choose a free variable name for it
        when we apply the rule.
        The predicates are the same as in the logic program, but they are applied to the specific substitution of the
        rule, which is given by the constant arguments and the non-terminals arguments that have a name.
        The terminal is the terminal of the rule, which corresponds to T in the logic program.

        The SLD-Resolution for solution spaces works as follows:
        Initialize: We start with the initial goal, which is the unary conjunction with the NT start.

        Selection: We select a non-terminal (subgoal) in the current goal.
        (We apply the subgoal_selection_strategy here.)

        Unification: SolutionSpace doesn't require unification, as the heads of the rules are just non-terminals,
        but we still need to select a rule which matches the current goal.
        All rules with the current goal as head are applicable, and variance in our SolutionSpace comes from
        multiple applicable rules for the same non-terminal.
        In SLD-Resolution this variance leads to a SLD-Derivation-Tree, where the children of a node are
        the new goals derived by applying all the applicable rules to the selected subgoal in the parent node.
        The strategy to traverse the SLD-Derivation-Tree and therefore select the next rule to apply
        is given by the variance_strategy_push and variance_strategy_pop.

        Derivation: We replace the selected non-terminal in the current goal with the RHS of the selected rule.

        Termination: If there are no more non-terminals (or NonTerminalArguments) in the current goal,
        we have derived a grounded tree, which is a solution. If there are still non-terminals,
        we continue with the selection step.

        Args:
            start (NT): _description_
            variance_strategy_push (Callable[[deque[Goal], Iterable[Goal]], deque[Goal]]): _description_
            variance_strategy_pop (Callable[[deque[Goal]], tuple[deque[Goal], Goal]]): _description_
            subgoal_selection_strategy (Callable[[Goal], tuple[Path, NonTerminalArgument[NT]]]): _description_
            max_count (int | None): _description_ (Default value = None)
            max_depth (int | None): _description_ (Default value = None)
            tree (Tree[T] | None): _description_ (Default value = None)
            pos (Path | None): _description_ (Default value = None)

        Yields:
            Tree[T]: _description_
        """

        if start not in self:  # O(1), and unlike `in self.nonterminals()` it builds no snapshot
            return

        all_results: set[Tree[T]] = set()

        # Initialize
        # goals = [Goal.from_rhs_rule(rhs) for rhs in self._rules[start]]
        goals: Iterable[Goal[NT, T, G] | None] = []
        if tree is not None and pos is not None:
            goals = self.goal_from_tree(start, tree, pos)
        else:
            goals = [Goal.from_rhs_rule(rhs) for rhs in self._rules_of(start)]
        # yield all solutions for already successful initial goals
        non_successful_goals: list[Goal[NT, T, G]] = []
        for goal in goals:
            if goal is not None:
                if goal.success:
                    new_term = goal.grounded[()][1]
                    if new_term not in all_results:
                        # depth 0
                        yield new_term
                        all_results.add(new_term)
                        if max_count is not None and len(all_results) >= max_count:
                            return
                else:
                    # Append, do not prepend: sorted() is stable, so prepending would break ties
                    # among the initial goals in reverse rule order while every later level, whose
                    # goals arrive in rule order, breaks them in rule order.
                    non_successful_goals.append(goal)

        if max_depth is not None and max_depth == 0:
            return

        variance: deque[Goal] = variance_strategy_push(deque(), non_successful_goals)

        # Selection, Unification, Derivation and Termination
        while variance:
            variance, current_goal = variance_strategy_pop(variance)
            # Selection:
            p, nt = subgoal_selection_strategy(current_goal)
            # Unification
            applicable_rules = self._rules_of(nt.origin)
            # A list, not a set: Goal defines neither __eq__ nor __hash__, so a set never
            # deduplicated anything here -- it only scattered the goals over their object
            # addresses and handed them back in an allocation-dependent order, which is what made
            # every derivation irreproducible. A list therefore removes no capability.
            new_goals: list[Goal] = []
            for r in applicable_rules:
                # Derivation
                new_goal = current_goal.update(r, p)
                if new_goal is not None:
                    if max_depth is not None:
                        paths = list(new_goal.grounded.keys()) + list(new_goal.subgoals.keys())
                        depth = max(len(p) for p in paths)
                        if depth > max_depth:
                            continue
                    # Termination
                    if new_goal.success:
                        new_term = new_goal.grounded[()][1]
                        if new_term not in all_results:
                            yield new_term
                            all_results.add(new_term)
                            if max_count is not None and len(all_results) >= max_count:
                                return
                    else:
                        new_goals.append(new_goal)
            variance = variance_strategy_push(variance, new_goals)
        return

    def depth_first_resolution(
        self,
        start: NT,
        max_count: int | None = None,
        max_depth: int | None = None,
        tree: Tree[T] | None = None,
        pos: Path | None = None,
    ) -> Iterable[Tree[T]]:
        """A simple implementation of SLD-Resolution with depth-first search in the SLD-Derivation-Tree.

        The goal selection is not leftmost: among the open subgoals it takes the deepest ones and,
        among those, the leftmost. Whether true leftmost selection would be the right semantics is
        a question about the inhabitation algorithm itself and is not decided here.

        Args:
            start (NT): _description_
            max_count (int | None): _description_ (Default value = None)
            max_depth (int | None): _description_ (Default value = None)
            tree (Tree[T] | None): _description_ (Default value = None)
            pos (Path | None): _description_ (Default value = None)

        Returns:
            Iterable[Tree[T]]: _description_
        """

        def variance_strategy_push(queue: deque[Goal], new_goals: Iterable[Goal]) -> deque[Goal]:
            """_summary_.

            Args:
                queue (deque[Goal]): _description_
                new_goals (Iterable[Goal]): _description_

            Returns:
                deque[Goal]: _description_
            """
            ordered = sorted(new_goals, key=lambda g: len(g.subgoals))  # fewest subgoals first
            # extendleft inserts in reverse, so pushing `ordered` directly would make popleft
            # return the goal with the MOST subgoals first -- the opposite of the intent.
            queue.extendleft(reversed(ordered))  # depth-first search <~> LIFO
            return queue

        def variance_strategy_pop(queue: deque[Goal]) -> tuple[deque[Goal], Goal]:
            """_summary_.

            Args:
                queue (deque[Goal]): _description_

            Returns:
                tuple[deque[Goal], Goal]: _description_
            """
            return queue, queue.popleft()  # depth-first search <~> LIFO

        def goal_selection_strategy(goal: Goal) -> tuple[Path, NonTerminalArgument[NT]]:
            """_summary_.

            Args:
                goal (Goal): _description_

            Returns:
                tuple[Path, NonTerminalArgument[NT]]: _description_
            """
            max_len = max(len(p) for p in goal.subgoals)
            filtered = filter(lambda x: len(x[0]) == max_len, goal.subgoals.items())
            return min(filtered, key=lambda item: item[0][-1])  # leftmost selection,
            # assuming new subgoals (deeper positions) are added "to the left" of the old ones

        return self.resolution(
            start,
            variance_strategy_push,
            variance_strategy_pop,
            goal_selection_strategy,
            max_count,
            max_depth,
            tree,
            pos,
        )

    def sample_tree(
        self,
        start: NT,
        max_depth: int | None = None,
        tree: Tree[T] | None = None,
        pos: Path | None = None,
        rng: random.Random | None = None,
    ) -> Tree[T] | None:
        """This method samples a tree top-down with possibly limited depth.

        Be aware, that this method is not guaranteed to terminate if the solution space contains recursive rules and
        max_depth is None, as it may get stuck in an infinite branch of the SLD-Derivation-Tree.
        Additionally the user has to ensure that the solution space is not empty, as an empty solution space can lead
        to nontermination as well.

        TODO: Because resolution directly returns all successful goals after the first derivation step without pushing
              goals to the stack, this method currently doens't work with requests, were depth 0 terms are allowed!

        Args:
            start (NT): _description_
            max_depth (int | None): _description_ (Default value = None)
            tree (Tree[T] | None): _description_ (Default value = None)
            pos (Path | None): _description_ (Default value = None)
            rng (random.Random | None): _description_ (Default value = None)

        Returns:
            Tree[T] | None: _description_
        """
        # allow deterministic sampling by providing an RNG instance
        rndm: random.Random = rng if rng is not None else random.Random()

        def variance_strategy_push(queue: deque[Goal], new_goals: Iterable[Goal]) -> deque[Goal]:
            """_summary_.

            Args:
                queue (deque[Goal]): _description_
                new_goals (Iterable[Goal]): _description_

            Returns:
                deque[Goal]: _description_
            """
            goals = list(new_goals)
            rndm.shuffle(goals)
            queue.extendleft(goals)  # depth-first search <~> LIFO
            return queue

        def variance_strategy_pop(queue: deque[Goal]) -> tuple[deque[Goal], Goal]:
            """_summary_.

            Args:
                queue (deque[Goal]): _description_

            Returns:
                tuple[deque[Goal], Goal]: _description_
            """
            return queue, queue.popleft()  # depth-first search <~> LIFO

        def goal_selection_strategy(goal: Goal) -> tuple[Path, NonTerminalArgument[NT]]:
            """_summary_.

            Args:
                goal (Goal): _description_

            Returns:
                tuple[Path, NonTerminalArgument[NT]]: _description_
            """
            max_len = max(len(p) for p in goal.subgoals)
            filtered = list(filter(lambda x: len(x[0]) == max_len, goal.subgoals.items()))
            return rndm.choice(filtered)
            # assuming new subgoals (deeper positions) are added "to the left" of the old ones

        trees: Iterable[Tree[T]] = self.resolution(
            start,
            variance_strategy_push,
            variance_strategy_pop,
            goal_selection_strategy,
            max_depth=max_depth,
            tree=tree,
            pos=pos,
        )

        try:
            iterator = iter(trees)
            tree = next(iterator)
        except StopIteration:
            return None

        return tree

    def breadth_first_resolution(
        self,
        start: NT,
        max_count: int | None = None,
        max_depth: int | None = None,
        tree: Tree[T] | None = None,
        pos: Path | None = None,
    ) -> Iterable[Tree[T]]:
        """A simple implementation of SLD-Resolution with breadth-first search in the SLD-Derivation-Tree.

        The goal selection is not leftmost: among the open subgoals it takes the deepest ones and,
        among those, the leftmost. Whether true leftmost selection would be the right semantics is
        a question about the inhabitation algorithm itself and is not decided here.

        Args:
            start (NT): _description_
            max_count (int | None): _description_ (Default value = None)
            max_depth (int | None): _description_ (Default value = None)
            tree (Tree[T] | None): _description_ (Default value = None)
            pos (Path | None): _description_ (Default value = None)

        Returns:
            Iterable[Tree[T]]: _description_
        """

        def variance_strategy_push(queue: deque[Goal], new_goals: Iterable[Goal]) -> deque[Goal]:
            """_summary_.

            Args:
                queue (deque[Goal]): _description_
                new_goals (Iterable[Goal]): _description_

            Returns:
                deque[Goal]: _description_
            """
            ordered = sorted(new_goals, key=lambda g: len(g.subgoals))  # fewest subgoals first
            # extend appends in order and popleft reads from the front, so unlike the depth-first
            # push above this one needs no reversal.
            queue.extend(ordered)  # breadth-first search <~> FIFO
            return queue

        def variance_strategy_pop(queue: deque[Goal]) -> tuple[deque[Goal], Goal]:
            """_summary_.

            Args:
                queue (deque[Goal]): _description_

            Returns:
                tuple[deque[Goal], Goal]: _description_
            """
            return queue, queue.popleft()  # breadth-first search <~> FIFO

        def goal_selection_strategy(goal: Goal) -> tuple[Path, NonTerminalArgument[NT]]:
            """_summary_.

            Args:
                goal (Goal): _description_

            Returns:
                tuple[Path, NonTerminalArgument[NT]]: _description_
            """
            max_len = max(len(p) for p in goal.subgoals)
            filtered = filter(lambda x: len(x[0]) == max_len, goal.subgoals.items())
            return min(filtered, key=lambda item: item[0][-1])  # leftmost selection,
            # assuming new subgoals (deeper positions) are added "to the left" of the old ones

        return self.resolution(
            start,
            variance_strategy_push,
            variance_strategy_pop,
            goal_selection_strategy,
            max_count,
            max_depth,
            tree,
            pos,
        )

    def contains_tree(self, start: NT, tree: Tree[T], interpretation: dict[T, Any] | None = None) -> bool:
        """Check if the solution space contains a given `tree` derivable from `start`.

        Args:
            start (NT): _description_
            tree (Tree[T]): _description_
            interpretation (dict[T, Any] | None): _description_ (Default value = None)

        Returns:
            bool: _description_

        Raises:
            ValueError: _description_
        """
        if start not in self:  # O(1), and unlike `in self.nonterminals()` it builds no snapshot
            return False

        stack: deque[tuple | Callable] = deque([(start, tree)])
        results: deque[bool] = deque()

        def get_inputs(count: int) -> list[bool]:
            """_summary_.

            Args:
                count (int): _description_

            Returns:
                list[bool]: _description_
            """
            return [results.pop() for _ in range(count)]

        while stack:
            task = stack.pop()
            if isinstance(task, tuple):
                nt, tree = task
                # use shared helper to check whether a rule matches the current tree head
                relevant_rhss = [rhs for rhs in self._rules_of(nt) if self._rule_matches_subtree(rhs, tree)]

                # disjunction of the results for individual rules
                def or_inputs(count: int = len(relevant_rhss)) -> None:
                    """_summary_.

                    Args:
                        count (int): _description_ (Default value = len(relevant_rhss))
                    """
                    results.append(any(get_inputs(count)))

                stack.append(or_inputs)

                for rhs in relevant_rhss:
                    substitution = {
                        argument.name: child.root
                        if isinstance(argument, ConstantArgument)
                        else (child if interpretation is None else child.interpret(interpretation))
                        for argument, child in zip(rhs.arguments, tree.children, strict=True)
                        if argument.name is not None
                    }

                    # conjunction of the results for individual arguments in the rule
                    def and_inputs(
                        count: int = sum(1 for argument in rhs.arguments if isinstance(argument, NonTerminalArgument)),
                        substitution: dict[str, Any] = substitution,
                        predicates=rhs.predicates,
                    ) -> None:
                        """_summary_.

                        Args:
                            count (int): _description_ (Default value = sum((1 for argument in rhs.arguments if isinstance(argument, NonTerminalArgument))))
                            substitution (dict[str, Any]): _description_ (Default value = substitution)
                            predicates (_type_): _description_ (Default value = rhs.predicates)
                        """
                        results.append(
                            all(get_inputs(count)) and all(predicate(substitution) for predicate in predicates)
                        )

                    stack.append(and_inputs)
                    for argument, child in zip(rhs.arguments, tree.children, strict=True):
                        if isinstance(argument, NonTerminalArgument):
                            stack.append((argument.origin, child))
            elif isinstance(task, FunctionType):
                # task is a function to execute
                task()

        if len(results) != 1:
            msg: str = "Number of results in contains_tree is not 1"
            raise ValueError(msg)

        return results.pop()
