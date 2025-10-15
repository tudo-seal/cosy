from collections.abc import Hashable, Iterable, Mapping
from typing import Any, Generic, TypeVar

from cosy.dsl import DSL
from cosy.solution_space import SolutionSpace
from cosy.subtypes import Subtypes, Taxonomy
from cosy.synthesizer import Specification, Synthesizer
from cosy.types import Arrow, Constructor, Intersection, Literal, Omega, Type, Var

__all__ = [
    "DSL",
    "Literal",
    "Var",
    "Subtypes",
    "Type",
    "Omega",
    "Constructor",
    "Arrow",
    "Intersection",
    "Synthesizer",
    "SolutionSpace",
]

T = TypeVar("T", bound=Hashable)


class CoSy(Generic[T]):
    component_specifications: Mapping[T, Specification]
    taxonomy: Taxonomy | None = None
    _synthesizer: Synthesizer

    def __init__(
        self,
        component_specifications: Mapping[T, Specification],
        taxonomy: Taxonomy | None = None,
    ) -> None:
        self.component_specifications = component_specifications
        self.taxonomy = taxonomy if taxonomy is not None else {}
        self._synthesizer = Synthesizer(component_specifications, self.taxonomy)

    def solve(self, query: Type, max_count: int = 100) -> Iterable[Any]:
        """
        Solves the given query by constructing a solution space and enumerating and interpreting the resulting trees.

        :param query: The query to solve.
        :param max_count: The maximum number of trees to enumerate.
        :return: An iterable of interpreted trees.
        """
        if not isinstance(query, Type):
            msg = "Query must be of type Type"
            raise TypeError(msg)
        solution_space = self._synthesizer.construct_solution_space(query).prune()

        trees = solution_space.enumerate_trees(query, max_count=max_count)
        for tree in trees:
            yield tree.interpret()
