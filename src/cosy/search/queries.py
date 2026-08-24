"""Resolution queries: the access path to a synthesized solution space.

A query names what is asked of a solution space, so that a component can take *a query* as its
argument instead of reaching into the space itself. Three kinds exist, told apart by their
query term:

* the **generator**: the query term is a variable, and the query streams every inhabitant.
* the **checker**: the query term is ground, and the query decides membership.
* the **partial-term query**: the query term carries a hole, and the query streams the
  completions of the term.

``SolutionSpace`` realizes the generator and the partial-term query through ``resolution`` (the
``tree``/``pos`` pair selects the kind) and the checker through ``contains_tree``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic

from cosy.core.solution_space import NT, G, SolutionSpace, T

if TYPE_CHECKING:
    from cosy.core.tree import Path, Tree

__all__ = ["ResolutionQuery", "checker", "generator_query", "residual_query"]


@dataclass(frozen=True)
class ResolutionQuery(Generic[NT, T, G]):
    """A resolution query against a synthesized solution space.

    ``tree`` and ``pos`` select the kind. Without them the query term is a fresh variable and
    the query is the generator. With them the query term is ``tree`` with the subterm at ``pos``
    replaced by a fresh variable, and the query streams the completions of that partial term.

    Attributes:
        solution_space (SolutionSpace[NT, T, G]): The synthesized solution space.
        start (NT): The queried non-terminal, whose inhabitants are asked for.
        tree (Tree[T] | None): The term the query term is derived from, or None for the
            generator. (Default value = None)
        pos (Path | None): The position of ``tree`` that becomes the hole, or None for the
            generator. (Default value = None)
    """

    solution_space: SolutionSpace[NT, T, G]
    start: NT
    tree: Tree[T] | None = None
    pos: Path | None = None

    def __post_init__(self) -> None:
        """Reject a query term the space can never be asked about.

        A position outside ``tree`` describes no partial term. The engine answers such a query
        with an empty stream, which reads like a term without completions rather than like the
        mistake it is, so the query refuses to be built.

        Raises:
            ValueError: If exactly one of ``tree`` and ``pos`` is given, or if ``pos`` is no
                position of ``tree``.
        """
        if (self.tree is None) != (self.pos is None):
            msg = "tree and pos select the partial-term query together: give both or neither"
            raise ValueError(msg)
        if self.tree is not None and self.pos is not None:
            try:
                self.tree.subtree_at(self.pos)
            except IndexError as error:
                msg = f"the query opens {self.pos}, which is no position of the prescribed term"
                raise ValueError(msg) from error

    @property
    def is_generator(self) -> bool:
        """Whether the query term is a fresh variable.

        Returns:
            bool: True for the generator, which streams every inhabitant of ``start``.
        """
        return self.tree is None

    @property
    def is_partial_term(self) -> bool:
        """Whether the query term carries a hole.

        Returns:
            bool: True for the partial-term query, which streams the completions of the term.
        """
        return self.tree is not None


def generator_query(solution_space: SolutionSpace[NT, T, G], start: NT) -> ResolutionQuery[NT, T, G]:
    """Build the generator query for a non-terminal.

    Args:
        solution_space (SolutionSpace[NT, T, G]): The synthesized solution space.
        start (NT): The queried non-terminal.

    Returns:
        ResolutionQuery[NT, T, G]: The query whose stream ranges over every inhabitant.
    """
    return ResolutionQuery(solution_space, start)


def residual_query(
    solution_space: SolutionSpace[NT, T, G], start: NT, tree: Tree[T], pos: Path
) -> ResolutionQuery[NT, T, G]:
    """Build the partial-term query at one position of a term.

    The query term is ``tree`` with the subterm at ``pos`` replaced by a fresh variable, and the
    success branches of the query are the substitutions that complete it into an inhabitant,
    the residual of the language at the term.

    Args:
        solution_space (SolutionSpace[NT, T, G]): The synthesized solution space.
        start (NT): The queried non-terminal.
        tree (Tree[T]): The term whose subterm is opened.
        pos (Path): The position that becomes the hole. ``()`` opens the whole term.

    Returns:
        ResolutionQuery[NT, T, G]: The partial-term query.
    """
    return ResolutionQuery(solution_space, start, tree, pos)


def checker(solution_space: SolutionSpace[NT, T, G], start: NT, tree: Tree[T]) -> bool:
    """Decide whether a ground term is an inhabitant.

    Implemented by ``SolutionSpace.contains_tree``, an AND/OR derivability procedure that decides
    membership without resolving anything.

    Args:
        solution_space (SolutionSpace[NT, T, G]): The synthesized solution space.
        start (NT): The queried non-terminal.
        tree (Tree[T]): The ground term whose membership is asked.

    Returns:
        bool: True exactly when ``tree`` is derivable from ``start``.
    """
    return solution_space.contains_tree(start, tree)
