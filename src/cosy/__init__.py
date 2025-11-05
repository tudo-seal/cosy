from collections.abc import Callable, Hashable, Iterable, Sequence
from itertools import groupby
from typing import Any, Generic, TypeVar

from cosy.solution_space import SolutionSpace
from cosy.specification_builder import SpecificationBuilder
from cosy.subtypes import Subtypes, Taxonomy
from cosy.synthesizer import Specification, Synthesizer
from cosy.types import Arrow, Constructor, Intersection, Literal, Omega, Type, Var

__all__ = [
    "Arrow",
    "Constructor",
    "CoSy",
    "Intersection",
    "Literal",
    "Omega",
    "SolutionSpace",
    "SpecificationBuilder",
    "Subtypes",
    "Synthesizer",
    "Type",
    "Var",
]

T = TypeVar("T", bound=Hashable)


class CoSy(Generic[T]):
    named_components_with_specifications: Sequence[tuple[T, Callable, Specification]]
    taxonomy: Taxonomy | None = None
    _synthesizer: Synthesizer

    def __init__(
        self,
        named_components_with_specifications: Sequence[tuple[T, Callable, Specification]],
        taxonomy: Taxonomy | None = None,
    ) -> None:
        if len(list(groupby(named_components_with_specifications, key=lambda x: x[0]))) != len(
            named_components_with_specifications
        ):
            msg = "Duplicate names: component's names should be unique"
            raise ValueError(msg)

        self.named_components_with_specifications = named_components_with_specifications
        self.taxonomy = taxonomy if taxonomy is not None else {}

        self.component_specifications = {
            name: specification for name, _, specification in self.named_components_with_specifications
        }
        self.component_interpretations = {
            name: interpretation for name, interpretation, _ in self.named_components_with_specifications
        }

        self._synthesizer = Synthesizer(self.component_specifications, self.taxonomy)

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
        _solution_space = self._synthesizer.construct_solution_space(query).prune()

        trees = _solution_space.enumerate_trees(
            query, max_count=max_count, interpretation=self.component_interpretations
        )
        for tree in trees:
            yield tree.interpret(interpretation=self.component_interpretations)
