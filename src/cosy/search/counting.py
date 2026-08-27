"""Branch counts over the retained derivation tree: what random search weights its choices by.

Random search is best-first search under a randomizing cost function, and two things are asked of
it: every prefix of its stream is a sample from a chosen distribution, and it draws without
replacement. A search that picks clauses uniformly delivers neither. The uniform choice ignores how
many inhabitants lie below each child, so it concentrates the sample on the terms the grammar
reaches quickly and never reaches the long ones, whatever distribution was intended. The choice of
a child has to be weighted by the inhabitants below it, counted per cost value.

That count is the *branch count* of a node: the number of success branches through it whose
inhabitant has size at most ``D`` and cost ``a``. It obeys a recursion over the tree,

    success node with inhabitant t, size <= D:   B_n(a) = 1 at a = c(t), else 0
    other node without children:                 B_n(a) = 0
    node with children:                          B_n(a) = sum over children

and at the root the branch counts are the *cost counts* ``N_r``, the number of inhabitants of the
query per cost value, *provided* the query is unambiguous within the bound, which is to say that
every inhabitant ends exactly one success branch. Without that hypothesis the two differ by the
number of extra derivations, and a sampler driven by them draws in proportion to the derivation
count rather than the term count, exactly as the counting samplers of the literature do on an
ambiguous grammar. :func:`assert_unambiguous_within` decides the hypothesis. It is a validation
tool and not a check the counting path runs.

**Why the recursion terminates.** Each internal resolution step fixes one function-symbol
occurrence that every completion below it shares, so the partial inhabitant of a node grows by at
least one symbol per expansion. A node whose partial inhabitant already exceeds ``D`` therefore has
no completion within the bound, its branch counts vanish, and the recursion cuts there. No retained
branch is longer than ``D`` expansions, and each node has finitely many children, one per
applicable clause, so the retained nodes form a *finite* tree on every query, recursive or not.
This is why random search halts where depth-first search does not, and it is also why the whole
tree is materialized here: the frontier memory of random search *is* the branch counts.

**Cost.** The recursion is deliberately naive, with no memoization across nodes. Memoizing by goal
would be wrong in general. An external predicate or a repeated variable couples subgoals, so the
subtree below a node depends on the accumulated substitution and the deferred atoms, and not on the
selected subgoal alone. A memoizing variant needs its own correctness argument for the class of
goals it applies to.

**The second way to the same numbers.** Materializing the tree costs the *number of inhabitants*,
which is what makes random search unusable where it is most wanted: the counts have to be complete
before the first element leaves the stream, so a space with a million inhabitants pays for all of
them in order to draw forty. :func:`size_table` computes the same counts from the program instead,
one row ``N_A(s)`` per non-terminal and size, filled by increasing ``s``, at a cost in the size of
the program and the bound alone. This is the table form of the classical counting samplers, and a
caller can drive the same construction from it.

The two agree only under a hypothesis, and :func:`decomposable_or_raise` decides it. A clause whose
predicate reads a hole makes the residual at that hole a *proper subset* of the hole's language,
and a table indexed by the non-terminal cannot see the difference. On a space whose single
predicate cuts one hole, the exact counts are ``{2: 2, 3: 2}`` where the table says
``{2: 2, 3: 2, 4: 2, ..., 8: 2}``. Coupled holes are the same failure with two arguments. So the
table is offered *with* its condition and never without it: a violation raises and names the
offending clause and its positions, rather than returning numbers that are quietly too large.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic

from cosy.core.solution_space import (
    NT,
    ConstantArgument,
    G,
    Goal,
    NonTerminalArgument,
    T,
)
from cosy.search.partial import partial_inhabitant, term_size
from cosy.search.rules import deepest_first_subgoal

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from cosy.core.solution_space import RHSRule, SolutionSpace
    from cosy.core.tree import Path, Tree
    from cosy.search.queries import ResolutionQuery

__all__ = [
    "CountedNode",
    "CoupledClause",
    "SizeTable",
    "assert_unambiguous_within",
    "branch_counts",
    "branch_multiplicities",
    "child_nodes",
    "coupled_clauses",
    "decomposable_or_raise",
    "initial_nodes",
    "retained_node_count",
    "size_table",
]


@dataclass(frozen=True)
class CountedNode(Generic[NT, T, G]):
    """A node of the retained derivation tree, together with its branch counts.

    Only nodes with a nonvanishing branch count are retained: a child all of whose completions
    exceed the size bound weighs nothing, so random search would never choose it, and it is
    discarded at expansion. The root is the exception. It is returned even when the query has no
    inhabitant within the bound, so that a caller can tell an empty space from a failed call.

    Attributes:
        goal (Goal[NT, T, G] | None): The engine's search node; None at the root of a generator
            query, whose query term is a bare variable and which has no goal of its own.
        inhabitant (Tree[T] | None): The solution of a success node within the bound; None on
            every other node.
        children (tuple[CountedNode[NT, T, G], ...]): The retained children, in clause order.
        counts (Mapping[Any, int]): ``B_n``, one entry per realized cost value.  Empty exactly
            when the node is the root of a query without inhabitants within the bound.
    """

    goal: Goal[NT, T, G] | None
    inhabitant: Tree[T] | None
    children: tuple[CountedNode[NT, T, G], ...]
    counts: Mapping[Any, int]

    @property
    def total(self) -> int:
        """Return the number of retained success branches below this node.

        Returns:
            int: ``sum_a B_n(a)``.
        """
        return sum(self.counts.values())


def _added_symbols(rule: RHSRule[NT, T, G]) -> int:
    """Return the number of function symbols one application of a rule fixes.

    Applying a rule at a hole writes its terminal there and turns each constant argument into a
    leaf. Each non-terminal argument becomes a fresh hole, which counts 0 under
    :func:`~cosy.search.partial.term_size`, so the partial inhabitant grows by exactly this many
    symbols. That is what lets the recursion carry the size along instead of measuring the term
    at every node.

    Args:
        rule (RHSRule[NT, T, G]): The clause being applied.

    Returns:
        int: One for the terminal, plus one per constant argument.
    """
    return 1 + sum(1 for argument in rule.arguments if isinstance(argument, ConstantArgument))


def initial_nodes(
    query: ResolutionQuery[NT, T, G],
) -> list[tuple[Goal[NT, T, G], int]]:
    """Return the children of the query's root node, with the size of their partial inhabitants.

    The root of a generator query is ``<- Q_tau(v)``: its query term is a variable, its partial
    inhabitant has size 0, and its children are the goals of the applicable clauses.  A
    partial-term query starts from the goals ``goal_from_tree`` derives for the prescribed term,
    whose partial inhabitants already carry the prescribed symbols.

    Args:
        query (ResolutionQuery[NT, T, G]): The query to expand.

    Returns:
        list[tuple[Goal[NT, T, G], int]]: The initial goals paired with their sizes.
    """
    space = query.solution_space
    tree, pos = query.tree, query.pos
    if tree is not None and pos is not None:
        goals = space.goal_from_tree(query.start, tree, pos)
        return [(goal, term_size(partial_inhabitant(goal))) for goal in goals]
    initial: list[tuple[Goal[NT, T, G], int]] = []
    for rule in space.get(query.start) or ():
        goal = Goal.from_rhs_rule(rule)
        if goal is not None:
            initial.append((goal, _added_symbols(rule)))
    return initial


def child_nodes(
    query: ResolutionQuery[NT, T, G],
    goal: Goal[NT, T, G],
    size: int,
    select: Callable[[Goal[NT, T, G]], tuple[Path, NonTerminalArgument[NT]]],
) -> list[tuple[Goal[NT, T, G], int]]:
    """Expand one node: one child per applicable clause that stays consistent.

    This is the engine's expansion with its early abort: ``Goal.update`` returns None when
    applying a clause violates a ground constraint, and such a child is dropped at creation rather
    than carried and rejected later.

    Args:
        query (ResolutionQuery[NT, T, G]): The query being searched.
        goal (Goal[NT, T, G]): The node to expand.
        size (int): The size of the node's partial inhabitant.
        select (Callable): The computation rule.

    Returns:
        list[tuple[Goal[NT, T, G], int]]: The children, in clause order, with their sizes.
    """
    position, argument = select(goal)
    children: list[tuple[Goal[NT, T, G], int]] = []
    for rule in query.solution_space.get(argument.origin) or ():
        child = goal.update(rule, position)
        if child is not None:
            children.append((child, size + _added_symbols(rule)))
    return children


@dataclass
class _Frame(Generic[NT, T, G]):
    """One node under construction while the recursion runs iteratively.

    Attributes:
        goal (Goal[NT, T, G] | None): The node's goal; None for the root.
        pending (list[tuple[Goal[NT, T, G], int]]): Children not yet processed, reversed so that
            ``pop`` yields them in clause order.
        built (list[CountedNode[NT, T, G]]): The retained children, in clause order.
    """

    goal: Goal[NT, T, G] | None
    pending: list[tuple[Goal[NT, T, G], int]]
    built: list[CountedNode[NT, T, G]]


def branch_counts(
    query: ResolutionQuery[NT, T, G],
    size_bound: int,
    cost: Callable[[Tree[T]], Any],
    *,
    subgoal_selection: Callable[[Goal[NT, T, G]], tuple[Path, NonTerminalArgument[NT]]] | None = None,
) -> CountedNode[NT, T, G]:
    """Build the retained derivation tree of a query and count its success branches per cost value.

    Args:
        query (ResolutionQuery[NT, T, G]): The generator or partial-term query to count over.
        size_bound (int): The bound ``D`` on the number of function symbols a term may carry
            (:func:`~cosy.search.partial.term_size`). It is what makes the counts finite, and it is
            *not* the engine's ``max_depth``.
        cost (Callable[[Tree[T]], Any]): The cost function ``c``, evaluated on the inhabitant of
            a success node. Any computable function will do. Random search needs no monotonicity,
            since the bound alone secures finiteness. The values must be hashable and are compared
            by equality only.
        subgoal_selection (Callable | None): The computation rule.  None selects the engine's
            deepest-first rule. (Default value = None)

    Returns:
        CountedNode[NT, T, G]: The root of the retained tree.  Its ``counts`` are ``B_r``, which
            equal the cost counts ``N_r`` exactly when the query is unambiguous within the bound.

    Raises:
        ValueError: If ``size_bound`` is negative.
    """
    if size_bound < 0:
        msg = f"the size bound counts function symbols and cannot be negative: {size_bound}"
        raise ValueError(msg)
    select = deepest_first_subgoal if subgoal_selection is None else subgoal_selection

    initial = initial_nodes(query)
    initial.reverse()
    stack: list[_Frame[NT, T, G]] = [_Frame(goal=None, pending=initial, built=[])]

    while True:
        frame = stack[-1]
        if frame.pending:
            goal, size = frame.pending.pop()
            if goal.success:
                inhabitant = goal.grounded[()][1]
                if size <= size_bound:
                    frame.built.append(
                        CountedNode(
                            goal=goal,
                            inhabitant=inhabitant,
                            children=(),
                            counts={cost(inhabitant): 1},
                        )
                    )
                continue
            if size > size_bound:
                continue
            children = child_nodes(query, goal, size, select)
            children.reverse()
            stack.append(_Frame(goal=goal, pending=children, built=[]))
            continue

        counts: dict[Any, int] = {}
        for child in frame.built:
            for value, count in child.counts.items():
                counts[value] = counts.get(value, 0) + count
        node = CountedNode(
            goal=frame.goal,
            inhabitant=None,
            children=tuple(frame.built),
            counts=counts,
        )
        stack.pop()
        if not stack:
            return node
        if counts:
            stack[-1].built.append(node)


def retained_node_count(root: CountedNode[NT, T, G]) -> int:
    """Count the nodes of a retained tree, which is the memory the search holds.

    Args:
        root (CountedNode[NT, T, G]): The root of the retained tree.

    Returns:
        int: The number of nodes, root included.
    """
    total = 0
    pending = [root]
    while pending:
        node = pending.pop()
        total += 1
        pending.extend(node.children)
    return total


def branch_multiplicities(root: CountedNode[NT, T, G]) -> dict[Tree[T], int]:
    """Return the inhabitants that end more than one success branch, with their branch count.

    Deciding unambiguity within the bound is exactly this: the retained tree holds every success
    branch within the bound, so counting branches per inhabitant answers the question. An empty
    result means every inhabitant has exactly one derivation, and the branch counts at the root are
    therefore the cost counts.

    Args:
        root (CountedNode[NT, T, G]): The root of the retained tree.

    Returns:
        dict[Tree[T], int]: The ambiguous inhabitants and their number of derivations.
    """
    seen: dict[Tree[T], int] = {}
    pending = [root]
    while pending:
        node = pending.pop()
        if node.inhabitant is not None:
            seen[node.inhabitant] = seen.get(node.inhabitant, 0) + 1
        pending.extend(node.children)
    return {tree: count for tree, count in seen.items() if count > 1}


@dataclass(frozen=True)
class CoupledClause(Generic[NT, T]):
    """A clause whose predicate reads at least one hole, which is what the table form cannot count.

    The engine hands a predicate the whole substitution of its clause and records nowhere which
    variables it reads, so the only thing that can be decided from the program is which names are
    *available* to it: the named non-terminal arguments plus the literals.  A clause is therefore
    reported here as soon as it carries a predicate and a named non-terminal argument, whether or
    not the predicate really looks at it. That is deliberately conservative. The alternative is to
    guess, and a wrong guess changes the counts without a trace.

    A predicate over the literals alone is *not* reported: it reads nothing that depends on how a
    hole is filled, decides on the clause itself, and leaves the counts of the table intact.

    Attributes:
        nonterminal (NT): The head the clause belongs to.
        terminal (T): The clause's function symbol, which is what identifies it to a reader.
        positions (tuple[int, ...]): The argument positions the predicate can read, in clause
            order. One entry means the predicate cuts a single hole's language, two or more that
            it couples holes, so that the residual at the clause is a relation between them rather
            than the product of their languages.
        nonterminals (tuple[NT, ...]): The non-terminals of those positions, in the same order.
    """

    nonterminal: NT
    terminal: T
    positions: tuple[int, ...]
    nonterminals: tuple[NT, ...]

    def describe(self) -> str:
        """Render the clause the way an error message needs it.

        Returns:
            str: The head, the terminal and the readable positions with their non-terminals.
        """
        listed = ", ".join(
            f"argument {position} : {nonterminal}"
            for position, nonterminal in zip(self.positions, self.nonterminals, strict=True)
        )
        kind = "couples" if len(self.positions) > 1 else "cuts"
        return f"{self.nonterminal} <- {self.terminal} ({listed}): the predicate {kind} them"


def coupled_clauses(
    space: SolutionSpace[NT, T, G],
) -> list[CoupledClause[NT, T]]:
    """Return the clauses whose predicates reach into a hole.

    Args:
        space (SolutionSpace[NT, T, G]): The synthesized program.

    Returns:
        list[CoupledClause[NT, T]]: One entry per offending clause, empty exactly when the
            program satisfies the decomposition hypothesis of :func:`size_table`.
    """
    offenders: list[CoupledClause[NT, T]] = []
    for nonterminal in space.nonterminals():
        for rule in space.get(nonterminal) or ():
            if not rule.predicates:
                continue
            readable = tuple(
                (index, argument.origin)
                for index, argument in enumerate(rule.arguments)
                if isinstance(argument, NonTerminalArgument) and argument.name is not None
            )
            if readable:
                offenders.append(
                    CoupledClause(
                        nonterminal=nonterminal,
                        terminal=rule.terminal,
                        positions=tuple(index for index, _ in readable),
                        nonterminals=tuple(origin for _, origin in readable),
                    )
                )
    return offenders


def decomposable_or_raise(space: SolutionSpace[NT, T, G]) -> None:
    """Check that no predicate of the program reads a hole, and say which one does if one does.

    This is the hypothesis under which the residual at a node is the *product* of the residuals
    at its holes, and hence the hypothesis under which :func:`size_table` computes the branch
    counts. Where it fails, the table overcounts, by the pairs a coupling predicate rejects or by
    the terms a cutting predicate rejects, and a sampler driven by it would draw in proportion to
    the wrong weights, silently.

    There is no fallback here on purpose.  A caller who hits this either uses the tree form
    (:func:`branch_counts`), which needs no hypothesis and pays the size of the search tree, or
    turns the predicate into a finite abstraction so that it becomes part of the non-terminal.

    Args:
        space (SolutionSpace[NT, T, G]): The synthesized program to decide for.

    Raises:
        ValueError: If some clause carries a predicate over a named non-terminal argument.  The
            message names every such clause, because which ones they are is what a caller needs
            in order to repair the repository.
    """
    offenders = coupled_clauses(space)
    if not offenders:
        return
    listed = "; ".join(clause.describe() for clause in offenders)
    msg = (
        f"the table form counts a node's completions as the product of its holes' completions, "
        f"which {len(offenders)} clause(s) of this program break by reading a hole in a "
        f"predicate: {listed}. Use branch_counts for the exact count, or replace the predicate "
        f"by a finite abstraction carried in the non-terminal."
    )
    raise ValueError(msg)


def _split_over(head_occupied: Sequence[tuple[int, int]], tail_row: Sequence[int], total: int, arity: int) -> int:
    """Split a size between the first hole and a tail whose row is already known.

    The one arithmetic step of the recursion, and the only place the fill and the finished table
    do any: give the first hole a size it occupies, read the tail's row at what is left, multiply,
    sum. Everything else in this module decides *which* rows exist to read.

    The walk runs over the sizes the first hole *occupies*, not over every size below the total.
    Which sizes a non-terminal occupies is a property of the program, and on the programs this
    exists for it is a thin subset of the bound: a determinized space whose abstraction grows along
    the term realizes a handful of sizes below a bound in the hundreds, and a walk over every size
    below the total would read a zero at nearly every step. The counts come out of the same iteration as the sizes, so a
    dense program, where the walk wins nothing, does not pay a second lookup for the privilege.

    Args:
        head_occupied (Sequence[tuple[int, int]]): The ``(size, N_A(size))`` pairs of the first
            hole's non-zero rows, by increasing size.
        tail_row (Sequence[int]): The row of the remaining holes, readable at every index below
            ``total``.
        total (int): The size to distribute.
        arity (int): How many holes there are in all, so that the tail keeps its minimum of one
            symbol per hole.

    Returns:
        int: ``sum over first of N_head(first) * tail_row[total - first]``.
    """
    limit = total - arity + 1
    value = 0
    for first, head_count in head_occupied:
        if first > limit:
            break
        value += head_count * tail_row[total - first]
    return value


# The rows one fill task writes to, with how many clauses of the program contribute to each: the
# row itself, its occupancy list, and the multiplicity.
_Targets = tuple[tuple[list[int], list[tuple[int, int]], int], ...]

# What :func:`_split_over` handles in one step: it gives the first hole a size and reads the tail's
# row at what is left, so a tail of this many holes or more needs a row of its own before the clause
# can be convolved at all.
_DIRECTLY_SPLIT_HOLES = 2


def _recursive_suffixes(
    decomposed: Mapping[NT, Sequence[tuple[int, tuple[NT, ...]]]],
) -> list[tuple[NT, ...]]:
    """Return the hole-tuple suffixes a longer tuple splits into, shortest first.

    A clause of arity three or more cannot be split in one step: ``N_(A,B,C)(s)`` is a sum over
    the sizes of ``A`` of ``N_(B,C)`` at what is left, so ``(B, C)`` has to be known at a whole
    range of sizes.  Such a suffix gets a row of its own, filled by increasing size exactly as a
    non-terminal's is. It obeys the same recursion, and it depends only on rows below its own size
    for the same reason: every hole takes at least one symbol.

    That is what keeps the memory in the size of the *program*. Giving a row to every hole tuple
    would cost the number of tuples times ``D``, and on a determinized space the tuples are nearly
    all distinct, so most of those rows would be written once and never read again. Only the
    suffixes a longer tuple recurses into are ever asked for twice, and a program has far fewer of
    those than it has distinct clause tuples.

    Shortest first, so that filling them in order is well founded: a longer suffix reads the
    shorter ones and never the other way round.  Suffixes of equal length are independent, so
    their order is free. It is fixed to the order they were found in anyway, since a result that
    depends on how a set happens to iterate is a result that changes between runs.

    Args:
        decomposed (Mapping[NT, Sequence[tuple[int, tuple[NT, ...]]]]): Per non-terminal, its
            applicable clauses as ``(base cost, hole non-terminals)``.

    Returns:
        list[tuple[NT, ...]]: The suffixes of length at least two, shortest first.
    """
    needed: dict[tuple[NT, ...], None] = {}
    pending = [
        holes for clauses in decomposed.values() for _base, holes in clauses if len(holes) > _DIRECTLY_SPLIT_HOLES
    ]
    while pending:
        suffix = pending.pop()[1:]
        if len(suffix) >= _DIRECTLY_SPLIT_HOLES and suffix not in needed:
            needed[suffix] = None
            pending.append(suffix)
    return sorted(needed, key=len)


def _largest_term(
    decomposed: Mapping[NT, Sequence[tuple[int, tuple[NT, ...]]]],
    suffixes: Sequence[tuple[NT, ...]],
    bound: int,
) -> int:
    """Return the size of the largest term the program has, never above ``bound``.

    A program whose language is finite has rows that are zero from some size on, and filling the
    table above that size counts nothing. A program whose clauses cannot recurse past a fixed
    budget is such a program, and so is every determinized space whose abstraction grows along the
    term, which is the case the table form exists for. The largest term is the least fixed point of
    "a clause reaches its base plus the largest term of each of its holes", which is exact on an
    acyclic program and saturates at ``bound`` on a recursive one.

    Saturating is what keeps a recursive program from iterating up to the bound one round at a
    time: a value that has reached ``bound + 1`` cannot grow again.  The round limit does the same
    for the shape of the dependency graph. An acyclic one converges in as many rounds as it has
    topological levels, and there are no more of those than non-terminals, so still growing after
    that many rounds means the program is recursive and the bound is the only limit there is.

    A hole tuple reaches as far as its holes reach together, and that can be further than any
    single non-terminal: the clause that splits into a suffix carries the reach up to its head, so
    where that clause is unproductive the head never records it. The suffix rows are read back
    through :meth:`SizeTable.split_row`, so they are filled to their own reach and not to the reach
    of the non-terminals alone.

    Args:
        decomposed (Mapping[NT, Sequence[tuple[int, tuple[NT, ...]]]]): Per non-terminal, its
            applicable clauses as ``(base cost, hole non-terminals)``.
        suffixes (Sequence[tuple[NT, ...]]): The hole-tuple suffixes that get a row of their own.
        bound (int): The size bound ``D``.

    Returns:
        int: The largest size any non-terminal or suffix row can reach, capped at ``bound``. Zero
            when the program derives nothing at all.
    """
    largest = dict.fromkeys(decomposed, 0)
    for _round in range(len(decomposed) + 1):
        changed = False
        for nonterminal, clauses in decomposed.items():
            best = largest[nonterminal]
            for base, holes in clauses:
                # A hole whose non-terminal the program never mentions has no terms at all, which
                # is the same answer as a non-terminal that has not been reached yet. The synthesis
                # does not prune, so such a hole is an ordinary shape and not a broken program.
                if any(largest.get(hole, 0) == 0 for hole in holes):
                    continue
                best = max(best, min(base + sum(largest[hole] for hole in holes), bound + 1))
            if best > largest[nonterminal]:
                largest[nonterminal] = best
                changed = True
        if not changed:
            reached = [*largest.values()]
            reached += [sum(largest.get(hole, 0) for hole in suffix) for suffix in suffixes]
            return min(bound, max(reached, default=0))
    # Still growing after that many rounds: the program is recursive, the values would climb to
    # the cap one round at a time, and the bound is the answer.  Returning the value reached so
    # far instead would cut the table short, which is a wrong count and not a slow one.
    return bound


@dataclass(frozen=True)
class SizeTable(Generic[NT]):
    """``N_A(s)``: the branches rooted at a non-terminal whose term has exactly size ``s``.

    The table is what replaces the traversal of the retained tree. The recursion for the branch
    counts, read on the *program* rather than on the search tree, is

        N_A(s) = sum over clauses A <- f(B_1..B_k) of  #{(s_1..s_k) : sum s_i = s - base(f),
                                                        s_i >= 1}  prod_i N_{B_i}(s_i),

    where ``base(f)`` is the number of symbols one application writes, the terminal plus one per
    constant argument. Every clause fixes at least its terminal, so ``base`` is at least one and
    row ``s`` depends only on rows ``< s``: the table fills by increasing ``s``, with no fixed-point
    iteration and no termination argument beyond that. A clause paying a symbol before it recurses
    is a cheaper and stronger argument than the finiteness of the retained tree.

    Cost is ``O(|clauses| * arity * D * occupancy)``, so it is in the size of the *program* and the
    bound, and independent of how many inhabitants the program has. That independence is the whole
    point. ``occupancy`` is how many sizes a non-terminal actually has terms of, which is ``D`` in
    the worst case and far below it on the programs this exists for: the fill walks the occupied
    sizes rather than every size, and stops at the largest term the program has rather than at
    ``D``.

    Attributes:
        bound (int): The size bound ``D`` the table was filled to.
        counts (Mapping[NT, tuple[int, ...]]): ``counts[A][s] = N_A(s)`` for ``0 <= s <= bound``;
            entry 0 is always 0, since no term has size 0.
        suffix_counts (Mapping[tuple[NT, ...], tuple[int, ...]]): The same, for the hole-tuple
            suffixes a clause of arity three or more splits into (:func:`_recursive_suffixes`).
            Filled to the size the suffix itself reaches, above which every entry is zero anyway.
            Empty on a program whose clauses have at most two holes, which is the common case.
    """

    bound: int
    counts: Mapping[NT, tuple[int, ...]]
    suffix_counts: Mapping[tuple[NT, ...], tuple[int, ...]] = field(default_factory=dict)

    def of(self, nonterminal: NT, size: int) -> int:
        """Return ``N_A(s)``.

        Args:
            nonterminal (NT): The non-terminal ``A``.
            size (int): The size ``s``.

        Returns:
            int: The number of branches, and 0 outside the table. A non-terminal the program never
                mentions has no terms, and neither does a size beyond the bound.
        """
        row = self.counts.get(nonterminal)
        if row is None or not 0 <= size <= self.bound:
            return 0
        return row[size]

    def split_counts(self, nonterminals: Sequence[NT], total: int) -> int:
        """Return the number of ways to fill a tuple of holes with a given total size.

        The convolution the recursion asks for: each hole takes at least one symbol, the sizes sum
        to ``total``, and the ways multiply because the holes are independent, which is exactly what
        :func:`decomposable_or_raise` secures.

        Args:
            nonterminals (Sequence[NT]): The non-terminals of the holes, in clause order.
            total (int): The size to distribute over them.

        Returns:
            int: ``sum over compositions of prod_i N_{A_i}(s_i)``.  One on an empty tuple with
                ``total == 0``, which is the empty product, and zero on an empty tuple otherwise.
        """
        if not 0 <= total <= self.bound:
            return 0
        return self.split_row(nonterminals)[total]

    def split_row(self, nonterminals: Sequence[NT]) -> tuple[int, ...]:
        """Return :meth:`split_counts` for a tuple of holes at *every* size up to the bound.

        The row rather than the value, because the row is what the caller wants.  A sampler asks
        a node for its branch counts, and a node's branch counts are one entry per size the bound
        admits, so the value form answered ``D`` questions about one hole tuple, hashing that tuple
        ``D`` times to do it. Computed as a row it is one walk, and the result is one array instead
        of ``D`` dictionary entries.

        Rows are kept, so a hole tuple is convolved once per table however often it is asked for.
        Only the tuples a caller actually names are built. The fill builds none of these, since it
        needs the suffixes (:func:`_recursive_suffixes`) and nothing else, which is what keeps a
        determinized program's distinct clause tuples out of memory.

        Args:
            nonterminals (Sequence[NT]): The non-terminals of the holes, in clause order.

        Returns:
            tuple[int, ...]: Entry ``s`` is the number of ways to fill the holes with total size
                ``s``, for ``0 <= s <= bound``.
        """
        holes = tuple(nonterminals)
        cached = self._rows.get(holes)
        if cached is not None:
            return cached
        if not holes:
            # The empty product: one way to fill no holes with nothing, none with anything.
            row = (1, *(0,) * self.bound)
        elif len(holes) == 1:
            row = self.counts.get(holes[0]) or (0,) * (self.bound + 1)
        else:
            known = self.suffix_counts.get(holes)
            if known is not None:
                row = known
            else:
                tail = self.split_row(holes[1:])
                occupied = self._occupied_row(holes[0])
                arity = len(holes)
                row = tuple(_split_over(occupied, tail, size, arity) for size in range(self.bound + 1))
        self._rows[holes] = row
        return row

    def _occupied_row(self, nonterminal: NT) -> tuple[tuple[int, int], ...]:
        """Return the ``(size, N_A(size))`` pairs of a non-terminal's non-zero rows.

        Args:
            nonterminal (NT): The non-terminal ``A``.

        Returns:
            tuple[tuple[int, int], ...]: The occupied sizes with their counts, by increasing size.
                Empty for a non-terminal the program never mentions.
        """
        return self._occupied.get(nonterminal, ())

    @property
    def _occupied(self) -> dict[NT, tuple[tuple[int, int], ...]]:
        """Return the occupied sizes per non-terminal, derived on first use.

        Derived data over a frozen table, installed the same way the convolution cache is.
        :func:`size_table` hands its own copy over instead, since the fill builds it row by row
        anyway and recomputing it would walk every row of every non-terminal a second time.

        Returns:
            dict[NT, tuple[tuple[int, int], ...]]: Per non-terminal, its occupied sizes and counts.
        """
        cache = self.__dict__.get("_occupied_sizes")
        if cache is None:
            cache = {
                nonterminal: tuple((size, count) for size, count in enumerate(row) if count)
                for nonterminal, row in self.counts.items()
            }
            object.__setattr__(self, "_occupied_sizes", cache)
        return cache

    @property
    def _rows(self) -> dict[tuple[NT, ...], tuple[int, ...]]:
        """Return the cache of convolved rows, created on first use.

        The dataclass is frozen so that a table can be shared between draws without any caller
        being able to disturb the counts; the cache is derived data and is installed through
        ``object.__setattr__`` for that reason.

        Returns:
            dict[tuple[NT, ...], tuple[int, ...]]: The rows built so far, by hole tuple.
        """
        cache = self.__dict__.get("_split_rows")
        if cache is None:
            cache = {}
            object.__setattr__(self, "_split_rows", cache)
        return cache


def size_table(space: SolutionSpace[NT, T, G], bound: int, *, check: bool = True) -> SizeTable[NT]:
    """Fill ``N_A(s)`` for every non-terminal of a program and every size up to a bound.

    Args:
        space (SolutionSpace[NT, T, G]): The synthesized program.
        bound (int): The size bound ``D``.
        check (bool): Whether to decide the decomposition hypothesis first.  Leaving it on is the
            supported use; turning it off is for a caller that has already decided it for this
            program and is filling several tables from it. (Default value = True)

    Returns:
        SizeTable[NT]: The filled table.

    Raises:
        ValueError: If ``bound`` is negative, or, under ``check``, if a predicate of the program
            reads a hole, in which case the table would overcount.
    """
    if bound < 0:
        msg = f"the size bound counts function symbols and cannot be negative: {bound}"
        raise ValueError(msg)
    if check:
        decomposable_or_raise(space)

    nonterminals = list(space.nonterminals())
    decomposed: dict[NT, list[tuple[int, tuple[NT, ...]]]] = {}
    for nonterminal in nonterminals:
        clauses: list[tuple[int, tuple[NT, ...]]] = []
        for rule in space.get(nonterminal) or ():
            # A predicate over a clause without named non-terminal arguments reads the literals
            # and nothing else, so it decides the clause once and for all, exactly as the engine
            # decides it in `Goal.from_rhs_rule` and `Goal.update`. Counting such a clause anyway
            # would put terms in the table that no search can produce.
            #
            # The `name is not None` guard is what keeps this from calling a *hole-reading*
            # predicate with a substitution that has no entry for the hole: under `check` those
            # clauses have already raised, but with the check off they would reach this line and
            # fail inside user code with a KeyError, an error about the wrong thing entirely.
            if (
                rule.predicates
                and not any(
                    isinstance(argument, NonTerminalArgument) and argument.name is not None
                    for argument in rule.arguments
                )
                and not all(predicate(rule.literal_substitution) for predicate in rule.predicates)
            ):
                continue
            holes = tuple(argument.origin for argument in rule.arguments if isinstance(argument, NonTerminalArgument))
            clauses.append((_added_symbols(rule), holes))
        decomposed[nonterminal] = clauses

    rows: dict[NT, list[int]] = {nt: [0] * (bound + 1) for nt in nonterminals}
    occupied: dict[NT, list[tuple[int, int]]] = {nt: [] for nt in nonterminals}
    suffixes = _recursive_suffixes(decomposed)
    suffix_rows: dict[tuple[NT, ...], list[int]] = {s: [0] * (bound + 1) for s in suffixes}

    def tail_row(holes: tuple[NT, ...]) -> list[int] | None:
        """Return the row a split's tail reads, or None where the program has none.

        Args:
            holes (tuple[NT, ...]): The tail of a hole tuple, of length at least one.

        Returns:
            list[int] | None: The row, still filling.
        """
        return rows.get(holes[0]) if len(holes) == 1 else suffix_rows.get(holes)

    # One task per distinct (base, holes) over the whole program, with the rows it contributes to
    # and how often. Two clauses that agree in both compute the same number in the same round, and
    # a determinized program has many times more clauses than distinct tasks. This is where that is
    # decided, once, instead of being rediscovered by hashing a tuple of non-terminals on every
    # call.
    grouped: dict[tuple[int, tuple[NT, ...]], dict[NT, int]] = {}
    for nonterminal in nonterminals:
        for base, holes in decomposed[nonterminal]:
            # A hole whose non-terminal the program never mentions has no terms, so the clause
            # contributes nothing at any size.
            if any(hole not in rows for hole in holes):
                continue
            heads = grouped.setdefault((base, holes), {})
            heads[nonterminal] = heads.get(nonterminal, 0) + 1

    nullary: list[tuple[int, _Targets]] = []
    unary: list[tuple[int, list[int], _Targets]] = []
    splitting: list[tuple[int, int, list[tuple[int, int]], list[int], _Targets]] = []
    for (base, holes), heads in grouped.items():
        targets: _Targets = tuple((rows[head], occupied[head], count) for head, count in heads.items())
        if not holes:
            nullary.append((base, targets))
        elif len(holes) == 1:
            unary.append((base, rows[holes[0]], targets))
        else:
            tail = tail_row(holes[1:])
            if tail is not None:
                splitting.append((base, len(holes), occupied[holes[0]], tail, targets))

    # A suffix is only ever a tail, never the head of a split, so it needs no occupancy list of
    # its own: `_split_over` walks the occupied sizes of the first hole, and that first hole is a
    # non-terminal.
    suffix_tasks: list[tuple[list[int], list[tuple[int, int]], list[int], int]] = []
    for suffix in suffixes:
        head_occupancy = occupied.get(suffix[0])
        tail = tail_row(suffix[1:])
        if head_occupancy is not None and tail is not None:
            suffix_tasks.append((suffix_rows[suffix], head_occupancy, tail, len(suffix)))

    # Every clause writes at least its terminal, so `remaining` is strictly below `size` and every
    # read below reaches a row this loop has already finished.  That is what makes one pass by
    # increasing size enough, and it is why nothing has to be remembered between rounds.
    #
    # Above the largest term the program has, every row is zero and filling it counts nothing.
    # The rows stay allocated to the bound either way, so the table still answers for every size
    # the caller asked about. It just stops doing arithmetic where the answer is already known.
    reachable = _largest_term(decomposed, suffixes, bound)
    for size in range(1, reachable + 1):
        # Suffix rows first, shortest first: a longer suffix reads the shorter ones, and a clause
        # reads all of them. Each depends only on sizes below its own, so no order is unsafe, and
        # this one keeps every read from touching a row that is half written in this round.
        for row, head_occupied, tail, arity in suffix_tasks:
            value = _split_over(head_occupied, tail, size, arity)
            if value:
                row[size] = value
        touched: list[tuple[list[int], list[tuple[int, int]]]] = []
        for base, targets in nullary:
            if size == base:
                for row, occupancy, multiplicity in targets:
                    if not row[size]:
                        touched.append((row, occupancy))
                    row[size] += multiplicity
        for base, hole_row, targets in unary:
            remaining = size - base
            if remaining < 1:
                continue
            value = hole_row[remaining]
            if value:
                for row, occupancy, multiplicity in targets:
                    if not row[size]:
                        touched.append((row, occupancy))
                    row[size] += value * multiplicity
        for base, arity, head_occupied, tail, targets in splitting:
            remaining = size - base
            if remaining < arity:
                continue
            value = _split_over(head_occupied, tail, remaining, arity)
            if value:
                for row, occupancy, multiplicity in targets:
                    if not row[size]:
                        touched.append((row, occupancy))
                    row[size] += value * multiplicity
        for row, occupancy in touched:
            occupancy.append((size, row[size]))

    # Read-only views, not copies: a table is shared between readers, and a rebinding through one
    # of them would leave `_rows` holding convolutions of the numbers that were there before.
    table = SizeTable(
        bound=bound,
        counts=MappingProxyType({nt: tuple(row) for nt, row in rows.items()}),
        suffix_counts=MappingProxyType({suffix: tuple(row) for suffix, row in suffix_rows.items()}),
    )
    object.__setattr__(table, "_occupied_sizes", {nt: tuple(sizes) for nt, sizes in occupied.items()})
    return table


def assert_unambiguous_within(query: ResolutionQuery[NT, T, G], size_bound: int) -> None:
    """Check that every inhabitant within the bound ends exactly one success branch.

    Unambiguity within the bound is the hypothesis under which the branch counts at the root are
    the cost counts, and hence the hypothesis under which a prefix of the stream is a sample from
    the intended distribution. Without it the branch counts count derivations rather than terms, and
    a sampler draws in proportion to the derivation count. Deciding it costs a full traversal of the
    retained tree, so this is a validation tool and the counting path does not run it.

    Args:
        query (ResolutionQuery[NT, T, G]): The query to decide for.
        size_bound (int): The bound ``D``.

    Raises:
        ValueError: If some inhabitant within the bound has more than one derivation.  The
            message names the offending terms, since which ones they are is what a caller needs
            to repair the space.
    """
    multiplicities = branch_multiplicities(branch_counts(query, size_bound, term_size))
    if multiplicities:
        listed = ", ".join(
            f"{tree} ({count} derivations)"
            for tree, count in sorted(multiplicities.items(), key=lambda item: str(item[0]))
        )
        msg = f"the query is not unambiguous within the bound {size_bound}: {listed}"
        raise ValueError(msg)
