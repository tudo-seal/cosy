"""Named search rules over resolution queries.

A search rule is a frontier and a clause order: the frontier decides which node leaves next, and
the clause order arranges the applicable clauses of one expansion into a sequence, possibly at
random. ``SolutionSpace.resolution`` is the engine behind every rule, and it takes both halves as
parameters, along with the computation rule that picks which hole is filled next. This module
names the two rules that reach for no information beyond the space itself, depth-first and
breadth-first search, as functions over a query rather than over a solution space, and supplies
the clause orders they can be given. A rule that scores its nodes needs a cost order instead and
is not one of these.

Neither named rule holds state between calls. They exist so that a component can take *a search
rule* as its argument, in the same way that a query lets it take *a query* instead of reaching
into the space itself, and so that both the default clause order and the computation rule are
values a caller can name, replace, or pass on. The computation rule is re-exported here for that
reason and is not a parameter of the two rules: a caller that has to expand nodes the way the
engine expands them needs the function the engine uses, and two copies of it agree only by
coincidence, while a stream of terms does not show that they have stopped agreeing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from cosy.core.solution_space import deepest_first_subgoal, fewest_arguments_first

if TYPE_CHECKING:
    import random
    from collections.abc import Callable, Iterable, Sequence

    from cosy.core.solution_space import NT, ClauseOrder, G, Goal, T
    from cosy.core.tree import Tree
    from cosy.search.queries import ResolutionQuery

__all__ = [
    "breadth_first",
    "deepest_first_subgoal",
    "depth_first",
    "fewest_arguments_first",
    "uniform_random_clause_order",
]

R = TypeVar("R")


def uniform_random_clause_order(rng: random.Random) -> Callable[[Sequence[R]], Sequence[R]]:
    """Build the clause order that arranges every clause sequence uniformly at random.

    The order draws from the given generator and from nothing else, so one seeded generator per
    search makes that search reproducible. Two searches must not share a generator: the streams
    are lazy, so consuming them in turn interleaves their draws. A randomizing search rule puts
    all of its randomness here and keeps its frontier a plain stack.

    Args:
        rng (random.Random): The source of randomness.

    Returns:
        Callable[[Sequence[R]], Sequence[R]]: A clause order drawing a fresh uniform permutation
            per expansion.
    """

    def order(applicable: Sequence[R]) -> Sequence[R]:
        """Arrange the applicable clauses of one expansion uniformly at random.

        Args:
            applicable (Sequence[R]): The applicable clauses of one expansion.

        Returns:
            Sequence[R]: The same clauses in a freshly drawn uniform order.
        """
        drawn = list(applicable)
        return rng.sample(drawn, len(drawn))

    return order


def depth_first(
    query: ResolutionQuery[NT, T, G],
    *,
    max_count: int | None = None,
    max_depth: int | None = None,
    clause_order: ClauseOrder[NT, T, G] | None = None,
    goal_filter: Callable[[Goal[NT, T, G]], bool] | None = None,
) -> Iterable[Tree[T]]:
    """Run depth-first search on a query and stream its inhabitants.

    The frontier is a last-in-first-out queue, so the rule follows one branch of the derivation
    tree to its end before it takes up another. It is sound on every space and complete on those
    whose derivation tree is finite. On a recursive space the branches are infinite, so what the
    default clause order buys is that a clause opening no holes is reached at all, and with it a
    first inhabitant; the search still needs a bound to end.

    Args:
        query (ResolutionQuery[NT, T, G]): The generator or partial-term query to search.
        max_count (int | None): Stop after this many inhabitants. (Default value = None)
        max_depth (int | None): Prune goals whose positions exceed this depth.
            (Default value = None)
        clause_order (ClauseOrder[NT, T, G] | None): The clause order of the rule. None takes
            ``fewest_arguments_first``. (Default value = None)
        goal_filter (Callable[[Goal[NT, T, G]], bool] | None): An expansion filter on goals. A
            child it rejects is dropped where it is created. (Default value = None)

    Returns:
        Iterable[Tree[T]]: The stream of inhabitants.
    """
    return query.solution_space.depth_first_resolution(
        query.start,
        max_count=max_count,
        max_depth=max_depth,
        tree=query.tree,
        pos=query.pos,
        clause_order=clause_order,
        goal_filter=goal_filter,
    )


def breadth_first(
    query: ResolutionQuery[NT, T, G],
    *,
    max_count: int | None = None,
    max_depth: int | None = None,
    clause_order: ClauseOrder[NT, T, G] | None = None,
    goal_filter: Callable[[Goal[NT, T, G]], bool] | None = None,
) -> Iterable[Tree[T]]:
    """Run breadth-first search on a query and stream its inhabitants.

    The frontier is a first-in-first-out queue, so the rule visits the derivation tree level by
    level and reaches an inhabitant of a shallower derivation before one of a deeper derivation.
    That is a statement about expansions, not about term size: one expansion grounds every
    constant argument of its clause at once, so a clause carrying constants adds several symbols
    where another adds one. It is sound and complete on every space, at the price of a frontier
    that holds a whole level.

    Args:
        query (ResolutionQuery[NT, T, G]): The generator or partial-term query to search.
        max_count (int | None): Stop after this many inhabitants. (Default value = None)
        max_depth (int | None): Prune goals whose positions exceed this depth.
            (Default value = None)
        clause_order (ClauseOrder[NT, T, G] | None): The clause order of the rule. None takes
            ``fewest_arguments_first``. (Default value = None)
        goal_filter (Callable[[Goal[NT, T, G]], bool] | None): An expansion filter on goals. A
            child it rejects is dropped where it is created. (Default value = None)

    Returns:
        Iterable[Tree[T]]: The stream of inhabitants.
    """
    return query.solution_space.breadth_first_resolution(
        query.start,
        max_count=max_count,
        max_depth=max_depth,
        tree=query.tree,
        pos=query.pos,
        clause_order=clause_order,
        goal_filter=goal_filter,
    )
