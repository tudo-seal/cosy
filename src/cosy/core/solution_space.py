"""Solution space given by a logic program."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from queue import PriorityQueue
from types import FunctionType
from typing import Any, Generic, TypeVar

from cosy.core.tree import Tree

NT = TypeVar("NT", bound=Hashable)  # type of non-terminals
T = TypeVar("T", bound=Hashable)  # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants


@dataclass(frozen=True)
class ConstantArgument(Generic[T, G]):
    name: str
    value: T
    origin: G


@dataclass(frozen=True)
class NonTerminalArgument(Generic[NT]):
    name: str | None
    origin: NT


Argument = ConstantArgument[T, G] | NonTerminalArgument[NT]


@dataclass(frozen=True)
class RHSRule(Generic[NT, T, G]):
    arguments: tuple[Argument, ...]
    predicates: tuple[Callable[[dict[str, Any]], bool], ...]
    terminal: T

    @property
    def non_terminals(self) -> frozenset[NT]:
        """Set of non-terminals occurring in the body of the rule."""
        return frozenset(arg.origin for arg in self.arguments if isinstance(arg, NonTerminalArgument))

    @property
    def literal_substitution(self):
        return {n.name: n.value for n in self.arguments if isinstance(n, ConstantArgument)}


Path = tuple[int, ...]


class Goal(Generic[NT, T, G]):
    """
    A goal models a Tree/Combinatory Term with variables.
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

    def __init__(self, root: dict[Path, T], subgoals: dict[Path, NonTerminalArgument[NT]], grounded: dict[Path, tuple[str, Tree[T]]],
                 constraints: dict[tuple[Path, ...], tuple[tuple[Callable[[dict[str, Any]], bool], ...], dict[str, T]]], success=False):
        self.constructors = root
        self.subgoals = subgoals
        self.grounded = grounded
        self.constraints = constraints
        self.success = success

    @classmethod
    def from_rhs_rule(self, rhs: RHSRule[NT, T, G]) -> Goal[NT, T, G] | None:
        """
        Create a goal from an RHSRule.
        The terminal becomes the combinator applied at the root.
        The arguments become the children of the root and are either grounded (ConstantArgument)
        or ungrounded (NonTerminalArgument).
        The constraints are the predicates from the RHSRule and to ensure a correct substitution of variable names,
        the local variable names from the RHSRule are stored additionally to the predicates that are applied at
        the given positions.
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
                raise ValueError(msg)
        root: dict[Path, T] = {(): rhs.terminal}
        if rhs.predicates:
            constraints = {named: (rhs.predicates, rhs.literal_substitution)} if named else {}
        else:
            constraints = {}
        if not subgoals:
            substitution = dict(grounded.values()) | rhs.literal_substitution
            if not all([c(substitution) for c in rhs.predicates]):
                return None
            grounded[()] = "", Tree(rhs.terminal, tuple(grounded[p][1] for p in sorted(grounded.keys())))
            return Goal(root, subgoals, grounded, constraints, success=True)
        return Goal(root, subgoals, grounded, constraints)


    def update(self, rhs: RHSRule[NT, T, G], position: Path) -> Goal[NT, T, G] | None:
        """
        Update the goal by applying the given rule at the given position.
        If the rule cannot be applied (because a constraint/predicate is violated) at the given position, return None.
        """
        new_subgoals: dict[Path, NonTerminalArgument[NT]] = self.subgoals.copy()
        new_grounded: dict[Path, tuple[str, Tree[T]]] = self.grounded.copy()
        named: tuple[Path, ...] = ()

        isGround = True

        children: tuple[Tree[T], ...] = ()

        # apply the rule at the given position
        for i, arg in enumerate(rhs.arguments):
            new_position = position + (i,)
            if isinstance(arg, NonTerminalArgument):
                isGround = False
                new_subgoals[new_position] = arg
                if arg.name is not None:
                    named += (new_position,)
            elif isinstance(arg, ConstantArgument):
                new_grounded[new_position] = arg.name, Tree(arg.value, ())
                children += (Tree(arg.value, ()),)
            else:
                msg = f"Argument {arg} is neither a NonTerminalArgument nor a ConstantArgument"
                raise ValueError(msg)

        new_constructors = self.constructors.copy()
        new_constructors[position] = rhs.terminal
        new_constraints = self.constraints.copy()
        if rhs.predicates:
            if named:
                new_constraints[named] = rhs.predicates, rhs.literal_substitution

        common_prefix = position[:-1]

        if isGround:
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
            for p in [x for x in new_grounded.keys() if x[:-1] == position]:
                new_grounded.pop(p)
            # move the path bottom up and check if all children are grounded and the parent can be grounded as well
            while common_prefix:
                subgoal_level_pos = [p for p in new_subgoals.keys() if p[:-1] == common_prefix]
                grounded_level_pos = [p for p in new_grounded.keys() if p[:-1] == common_prefix]
                if subgoal_level_pos:
                    break
                # if all arguments are grounded, we can ground the parent as well, if the constraints are satisfied
                else:
                    # check all constraints
                    preds = [ps for ps in new_constraints.keys() if ps[0][:-1] == common_prefix]
                    for ps in preds:
                        constraints, literal_substitution = new_constraints[ps]
                        args: tuple[tuple[str, Tree[T]], ...] = tuple(new_grounded[p] for p in ps)
                        substitution = dict(args) | literal_substitution
                        if not all([c(substitution) for c in constraints]):
                            return None
                    # sort the positions by their last element,
                    # which corresponds to the position in the arguments of the parent position
                    sorted_positions = sorted(grounded_level_pos, key=lambda p: p[-1])
                    children = tuple(new_grounded[p][1] for p in sorted_positions)
                    # construct the tree for the parent position
                    tree = Tree(new_constructors[position[:-1]], children)
                    if position[:-1] in new_subgoals.keys():
                        nt = new_subgoals.pop(position[:-1])
                        new_grounded[position[:-1]] = (nt.name, tree) if nt.name is not None else ("", tree)
                    else:
                        raise ValueError("the parent to a nonterminal must be a nonterminal as well")
                    # tidy up
                    for p in grounded_level_pos:
                        new_grounded.pop(p)
                    if position in new_subgoals.keys():
                        new_subgoals.pop(position)
                    position = position[:-1]
                    common_prefix = position[:-1]

            if len(new_subgoals) == 0:
                # if there are no subgoals left, the root must be grounded
                if not common_prefix == ():
                    raise AssertionError("common_prefix should be empty when all subgoals are grounded")
                # check all constraints and return None if a not all constraints are satisfied
                preds = [ps for ps in new_constraints.keys() if ps[0][:-1] == ()]
                for ps in preds:
                    constraints, literal_substitution = new_constraints[ps]
                    args = tuple(new_grounded[p] for p in ps)
                    substitution = dict(args) | literal_substitution
                    if not all([c(substitution) for c in constraints]):
                        return None
                # sort the positions by their last element,
                # which corresponds to the position in the arguments of the parent position
                sorted_positions = sorted(new_grounded.keys(), key=lambda p: p[-1])
                children = tuple(new_grounded[p][1] for p in sorted_positions)
                # construct the tree for the root position, the derivation
                tree = Tree(new_constructors[()], children)
                new_grounded[()] = "", tree
            return Goal(new_constructors, new_subgoals, new_grounded, new_constraints, success=len(new_subgoals) == 0)
        else:
            """
            If applying the rule does not lead to a ground tree at the position, 
            we return the updated goal.
            """
            return Goal(new_constructors, new_subgoals, new_grounded, new_constraints, success=False)




class SolutionSpace(Generic[NT, T, G]):
    _rules: defaultdict[NT, deque[RHSRule[NT, T, G]]]

    def __init__(self, rules: dict[NT, deque[RHSRule[NT, T, G]]] | None = None) -> None:
        if rules is None:
            rules = defaultdict(deque)
        self._rules = defaultdict(deque, rules)

    def get(self, nonterminal: NT) -> deque[RHSRule[NT, T, G]] | None:
        return self._rules.get(nonterminal)

    def __getitem__(self, nonterminal: NT) -> deque[RHSRule[NT, T, G]]:
        return self._rules[nonterminal]

    def nonterminals(self) -> Iterable[NT]:
        return self._rules.keys()

    def as_tuples(self) -> Iterable[tuple[NT, deque[RHSRule[NT, T, G]]]]:
        return self._rules.items()

    def add_rule(
        self,
        nonterminal: NT,
        terminal: T,
        arguments: tuple[Argument, ...],
        predicates: tuple[Callable[[dict[str, Any]], bool], ...],
    ) -> None:
        self._rules[nonterminal].append(RHSRule(arguments, predicates, terminal))

    def show(self) -> str:
        return "\n".join(
            f"{nt!s} ~> {' | '.join([str(subrule) for subrule in rule])}" for nt, rule in self._rules.items()
        )

    def prune(self) -> SolutionSpace[NT, T, G]:
        """Keep only productive rules."""

        ground_types: set[NT] = set()
        queue: set[NT] = set()
        inverse_grammar: dict[NT, set[tuple[NT, frozenset[NT]]]] = defaultdict(set)

        for n, exprs in self._rules.items():
            for expr in exprs:
                non_terminals = expr.non_terminals
                for m in non_terminals:
                    inverse_grammar[m].add((n, non_terminals))
                if not non_terminals:
                    queue.add(n)

        while queue:
            n = queue.pop()
            if n not in ground_types:
                ground_types.add(n)
                for m, non_terminals in inverse_grammar[n]:
                    if m not in ground_types and all(t in ground_types for t in non_terminals):
                        queue.add(m)

        return SolutionSpace[NT, T, G](
            defaultdict(
                deque,
                {
                    target: deque(
                        possibility
                        for possibility in self._rules[target]
                        if all(t in ground_types for t in possibility.non_terminals)
                    )
                    for target in ground_types
                },
            )
        )

    def _enumerate_tree_vectors(
        self,
        non_terminals: Sequence[NT | None],
        existing_terms: Mapping[NT, set[Tree[T]]],
        nt_term: tuple[NT, Tree[T]] | None = None,
    ) -> Iterable[tuple[Tree[T] | None, ...]]:
        """Enumerate possible term vectors for a given list of non-terminals and existing terms. Use nt_term at least once (if given)."""
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
        existing_terms: Mapping[NT, set[Tree[T]]],
        interpretation: dict[T, Any] | None = None,
        max_count: int | None = None,
        nt_old_term: tuple[NT, Tree[T]] | None = None,
    ) -> set[Tree[T]]:
        # Genererate new terms for rule `rule` from existing terms up to `max_count`
        # the term `old_term` should be a subterm of all resulting terms, at a position, that corresponds to `nt`

        output_set: set[Tree[T]] = set()
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
            """Interleave parameters, literal arguments and arguments."""
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
            """Construct a new tree from the rule and the given specific arguments."""
            return Tree(
                rule.terminal,
                tuple(interleave(parameters, literal_arguments, arguments)),
            )

        def specific_substitution(parameters: Sequence[Tree[T] | None]):
            return {
                a.name: p if interpretation is None else p.interpret(interpretation)
                for p, a in zip(parameters, rule.arguments, strict=True)
                if isinstance(a, NonTerminalArgument) and a.name is not None and p is not None
            } | rule.literal_substitution

        def valid_parameters(
            nt_term: tuple[NT, Tree[T]] | None,
        ) -> Iterable[tuple[Tree[T] | None, ...]]:
            """Enumerate all valid parameters for the rule."""
            for parameters in self._enumerate_tree_vectors(named_non_terminals, existing_terms, nt_term):
                if rule.predicates:
                    # compute the specific substitution only if there are predicates
                    substitution = specific_substitution(parameters)
                    if all(predicate(substitution) for predicate in rule.predicates):
                        yield parameters
                else:
                    yield parameters

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

    def enumerate_trees(
        self,
        start: NT,
        max_count: int | None = None,
        max_bucket_size: int | None = None,
        interpretation: dict[T, Any] | None = None,
    ) -> Iterable[Tree[T]]:
        """
        Enumerate terms as an iterator efficiently - all terms are enumerated, no guaranteed term order.
        """
        if start not in self.nonterminals():
            return

        queues: dict[NT, PriorityQueue[Tree[T]]] = {n: PriorityQueue() for n in self.nonterminals()}
        existing_terms: dict[NT, set[Tree[T]]] = {n: set() for n in self.nonterminals()}
        inverse_grammar: dict[NT, deque[tuple[NT, RHSRule[NT, T, G]]]] = {n: deque() for n in self.nonterminals()}
        all_results: set[Tree[T]] = set()

        for n, exprs in self._rules.items():
            for expr in exprs:
                if all(m in self.nonterminals() for m in expr.non_terminals):
                    for m in expr.non_terminals:
                        inverse_grammar[m].append((n, expr))
                    for new_term in self._generate_new_trees(expr, existing_terms, interpretation):
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
            non_terminals = {n for n in self.nonterminals() if not queues[n].empty()}

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
                                expr, existing_terms, interpretation, max_count, (n, term)
                            ):
                                if new_term not in all_results:
                                    if max_count is not None and len(all_results) >= max_count:
                                        return
                                    yield new_term
                                    all_results.add(new_term)
                                    queues[start].put(new_term)
                        else:
                            for new_term in self._generate_new_trees(
                                expr, existing_terms, interpretation, max_bucket_size, (n, term)
                            ):
                                queues[m].put(new_term)
            current_bucket_size += 1
        return

    def resolution(
            self,
            start: NT,
            variance_strategy_push: Callable[[deque[Goal], Iterable[Goal]], deque[Goal]],
            variance_strategy_pop: Callable[[deque[Goal]], tuple[deque[Goal], Goal]],
            subgoal_selection_strategy: Callable[[Goal], tuple[Path, NonTerminalArgument[NT]]],
            max_count: int | None = None,
    ) -> Iterable[Tree[T]]:
        """
        Enumerate terms implemented via SLD-Resolution.
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
        """

        if start not in self.nonterminals():
            return

        all_results: set[Tree[T]] = set()

        # Initialize
        goals = [Goal.from_rhs_rule(rhs) for rhs in self._rules[start]]
        # yield all solutions for already successful initial goals
        non_successful_goals = []
        for goal in goals:
            if goal is not None:
                if goal.success:
                    new_term = goal.grounded[()][1]
                    if new_term not in all_results:
                        yield new_term
                        all_results.add(new_term)
                        if max_count is not None and len(all_results) >= max_count:
                            return
                else:
                    non_successful_goals.append(goal)
        non_successful_goals.reverse()
        variance: deque[Goal] = variance_strategy_push(deque(), non_successful_goals)

        # Selection, Unification, Derivation and Termination
        while variance:
            variance, current_goal = variance_strategy_pop(variance)
            # Selection:
            p, nt = subgoal_selection_strategy(current_goal)
            # Unification
            applicable_rules = self._rules[nt.origin]
            new_goals: set[Goal] = set()
            for r in applicable_rules:
                # Derivation
                new_goal = current_goal.update(r, p)
                if new_goal is not None:
                    # Termination
                    if new_goal.success:
                        new_term = new_goal.grounded[()][1]
                        if new_term not in all_results:
                            yield new_term
                            all_results.add(new_term)
                            if max_count is not None and len(all_results) >= max_count:
                                return
                    else:
                        new_goals.add(new_goal)
            variance = variance_strategy_push(variance, new_goals)
        return

    def depth_first_resolution(self,
                               start: NT,
                               max_count: int | None = None, ) -> Iterable[Tree[T]]:
        """A simple implementation of SLD-Resolution with leftmost goal selection and depth-first search in the SLD-Derivation-Tree."""
        def variance_strategy_push(queue: deque[Goal], new_goals: Iterable[Goal]) -> deque[Goal]:
            sorted(new_goals, key=lambda g: len(g.subgoals))  # sort by number of subgoals
            queue.extendleft(new_goals)  # depth-first search <~> LIFO
            return queue

        def variance_strategy_pop(queue: deque[Goal]) -> tuple[deque[Goal], Goal]:
            return queue, queue.popleft()  # depth-first search <~> LIFO

        def goal_selection_strategy(goal: Goal) -> tuple[Path, NonTerminalArgument[NT]]:
            max_len = max(len(p) for p in goal.subgoals.keys())
            filtered = filter(lambda x: len(x[0]) == max_len, goal.subgoals.items())
            return min(filtered, key=lambda item: item[0][-1])  # leftmost selection,
            # assuming new subgoals (deeper positions) are added "to the left" of the old ones

        return self.resolution(start, variance_strategy_push, variance_strategy_pop, goal_selection_strategy, max_count)

    def breadth_first_resolution(self,
            start: NT,
            max_count: int | None = None,) -> Iterable[Tree[T]]:
        """A simple implementation of SLD-Resolution with leftmost goal selection and breadth-first search in the SLD-Derivation-Tree."""
        def variance_strategy_push(queue: deque[Goal], new_goals: Iterable[Goal]) -> deque[Goal]:
            sorted(new_goals, key=lambda g: len(g.subgoals))  # sort by number of subgoals
            queue.extend(new_goals)  # breadth-first search <~> FIFO
            return queue

        def variance_strategy_pop(queue: deque[Goal]) -> tuple[deque[Goal], Goal]:
            return queue, queue.popleft()  # breadth-first search <~> FIFO

        def goal_selection_strategy(goal: Goal) -> tuple[Path, NonTerminalArgument[NT]]:
            max_len = max(len(p) for p in goal.subgoals.keys())
            filtered = filter(lambda x: len(x[0]) == max_len, goal.subgoals.items())
            return min(filtered, key=lambda item: item[0][-1])  # leftmost selection,
            # assuming new subgoals (deeper positions) are added "to the left" of the old ones

        return self.resolution(start, variance_strategy_push, variance_strategy_pop, goal_selection_strategy, max_count)


    def contains_tree(self, start: NT, tree: Tree[T], interpretation: dict[T, Any] | None = None) -> bool:
        """Check if the solution space contains a given `tree` derivable from `start`."""
        if start not in self.nonterminals():
            return False

        stack: deque[tuple | Callable] = deque([(start, tree)])
        results: deque[bool] = deque()

        def get_inputs(count: int) -> list[bool]:
            return [results.pop() for _ in range(count)]

        while stack:
            task = stack.pop()
            if isinstance(task, tuple):
                nt, tree = task
                relevant_rhss = [
                    rhs
                    for rhs in self._rules[nt]
                    if len(rhs.arguments) == len(tree.children)
                    and rhs.terminal == tree.root
                    and all(
                        argument.value == child.root and len(child.children) == 0
                        for argument, child in zip(rhs.arguments, tree.children, strict=True)
                        if isinstance(argument, ConstantArgument)
                    )
                ]

                # disjunction of the results for individual rules
                def or_inputs(count: int = len(relevant_rhss)) -> None:
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
