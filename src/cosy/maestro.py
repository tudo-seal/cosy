from collections.abc import Callable, Hashable, Iterable, Sequence
from itertools import groupby
from typing import Any, Generic, TypeVar

from cosy.solutions import MaestroSolutions
from cosy.subtypes import Taxonomy
from cosy.synthesizer import Specification, Synthesizer
from cosy.types import Type

T = TypeVar("T", bound=Hashable)


class Maestro(Generic[T]):
    named_components_with_specifications: Sequence[tuple[T, Callable, Specification]]
    taxonomy: Taxonomy | None = None
    _synthesizer: Synthesizer

    def __init__(
        self,
        named_components_with_specifications: Sequence[tuple[T, Callable, Specification]],
        taxonomy: Taxonomy | None = None,
    ) -> None:
        duplicate_component_names = [
            key
            for key, group in groupby(named_components_with_specifications, key=lambda x: x[0])
            if len(list(group)) > 1
        ]
        if len(duplicate_component_names) != 0:
            msg = f"Component's names should be unique, but the following names are duplicated: {duplicate_component_names}"
            raise ValueError(msg)

        non_callable_interpretations_by_component_name = [
            name for name, interpretation, _ in named_components_with_specifications if not callable(interpretation)
        ]
        if len(non_callable_interpretations_by_component_name) != 0:
            msg = f"Component's interpretations should be callable, but interpretations of components with the following names are not: {non_callable_interpretations_by_component_name}"
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

    def query(self, target: Type, max_count: int = 100) -> MaestroSolutions[T]:
        """
        Query the Maestro for solutions that fulfill given target; by constructing a solution space and enumerating and interpreting the resulting trees.

        :param target: The target for which solutions should be queried.
        :param max_count: The maximum number of trees to enumerate.
        :return: An iterable of interpreted trees, the results.
        """
        if not isinstance(target, Type):
            msg = "Target must be of type Type"
            raise TypeError(msg)
        solution_space = self._synthesizer.construct_solution_space(target).prune()

        trees = solution_space.enumerate_trees(
            target, max_count=max_count, interpretation=self.component_interpretations
        )
        return MaestroSolutions(trees, component_interpretations=self.component_interpretations)


