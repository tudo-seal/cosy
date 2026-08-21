"""Partial inhabitants: reading a search node as the term it denotes.

A search node of a generator or partial-term query denotes the instantiated query term, the
*partial inhabitant*, and its open positions are the *holes*. This module materializes that
reading from the engine's ``Goal`` representation and measures the result.

``Goal`` stores more than the open holes, and the difference matters. An expanded position stays
in ``Goal.subgoals``, with its symbol recorded in ``Goal.constructors``, until its whole
subtree grounds and the cascade in ``Goal.update`` folds it into ``Goal.grounded``. The holes of
a node are therefore the subgoal positions that were never expanded. Treating all of ``subgoals``
as open work would re-derive positions that already carry a symbol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

from cosy.core.solution_space import NT, G, T
from cosy.core.tree import Tree

if TYPE_CHECKING:
    from cosy.core.solution_space import Goal
    from cosy.core.tree import Path

__all__ = ["Hole", "holes", "partial_inhabitant", "term_depth", "term_size"]


@dataclass(frozen=True)
class Hole(Generic[NT]):
    """A hole of a partial inhabitant: a nullary symbol standing for an open position.

    A hole stands for a symbol that occurs in no term of the language, so nothing of the language
    matches it. The position and the non-terminal together keep two holes of one term distinct.

    Attributes:
        position (Path): The position of the hole in the partial inhabitant.
        nonterminal (NT): The non-terminal the hole must be completed from (the hole's type).
    """

    position: Path
    nonterminal: NT


def holes(goal: Goal[NT, T, G]) -> dict[Path, NT]:
    """Return the holes of a goal: its unexpanded subgoal positions and their non-terminals.

    A position that grounds leaves ``subgoals`` in the same step, so the open positions are the
    subgoal positions without a symbol. A grounded position is not among them to begin with.

    Args:
        goal (Goal[NT, T, G]): The search node.

    Returns:
        dict[Path, NT]: One entry per hole, mapping its position to the non-terminal it must be
            completed from. Empty exactly on nodes whose partial inhabitant is ground.
    """
    return {
        position: argument.origin for position, argument in goal.subgoals.items() if position not in goal.constructors
    }


def term_size(term: Tree[Any]) -> int:
    """Return the number of function-symbol occurrences in a term.

    A variable has size 0 and ``F(t_1, .., t_n)`` has size ``1 + sum size(t_i)``. On a ground
    term this is the node count, which is what ``Tree.size`` reports. On a partial inhabitant the
    two differ, because a hole is a variable and carries no symbol. A search that bounds the size
    of the terms it keeps has to measure it this way, since charging a partial inhabitant for its
    holes would cut branches whose completions still fit the bound.

    Args:
        term (Tree[Any]): The term to measure. Its leaves may be Hole markers.

    Returns:
        int: The number of function-symbol occurrences.
    """
    total = 0
    pending = [term]
    while pending:
        current = pending.pop()
        if not isinstance(current.root, Hole):
            total += 1
        pending.extend(current.children)
    return total


def term_depth(term: Tree[Any]) -> int:
    """Return the depth of a term: the length of its longest root-to-leaf path.

    A hole is a leaf, so it ends the path it sits on and a partial inhabitant reaches no deeper
    along that path. This is a different quantity from ``term_size``, which counts symbols, and
    from the engine's ``max_depth``, which bounds the length of the positions a goal still
    carries and stops seeing a subtree once it grounds.

    Args:
        term (Tree[Any]): The term to measure. Its leaves may be Hole markers.

    Returns:
        int: The depth, 0 for a single node.
    """
    depth = 0
    pending = [(term, 0)]
    while pending:
        current, level = pending.pop()
        depth = max(depth, level)
        pending.extend((child, level + 1) for child in current.children)
    return depth


def partial_inhabitant(goal: Goal[NT, T, G]) -> Tree[Any]:
    """Materialize the partial inhabitant a goal denotes.

    Grounded positions contribute their subtree, expanded positions contribute their symbol
    applied to the materialized children, and each hole contributes a nullary Hole leaf. On a
    success goal the result is the solution itself, hole-free and equal to ``goal.grounded[()][1]``.

    Args:
        goal (Goal[NT, T, G]): The search node to read.

    Returns:
        Tree[Any]: The partial inhabitant. Its leaves are terminals, constants, or Hole markers.
    """

    def child_positions(position: Path) -> list[Path]:
        """List the positions of the children of an expanded position.

        A child is known to the goal through one of its three maps, and which one depends on the
        kind of argument: a constant argument is grounded from the start, a non-terminal argument
        is a subgoal until it is expanded and grounds. The arity is where all three run out.

        Args:
            position (Path): The expanded position.

        Returns:
            list[Path]: The child positions, in argument order.
        """
        children: list[Path] = []
        index = 0
        while True:
            child = (*position, index)
            if child in goal.subgoals or child in goal.grounded or child in goal.constructors:
                children.append(child)
                index += 1
            else:
                return children

    built: dict[Path, Tree[Any]] = {}
    # (position, children_materialized): the second visit of an expanded position assembles it
    pending: list[tuple[Path, bool]] = [((), False)]
    while pending:
        position, materialized = pending.pop()
        if materialized:
            built[position] = Tree(
                goal.constructors[position], tuple(built[child] for child in child_positions(position))
            )
            continue
        grounded = goal.grounded.get(position)
        if grounded is not None:
            built[position] = grounded[1]
        elif position in goal.constructors:
            pending.append((position, True))
            pending.extend((child, False) for child in child_positions(position))
        else:
            built[position] = Tree(Hole(position, goal.subgoals[position].origin), ())
    return built[()]
