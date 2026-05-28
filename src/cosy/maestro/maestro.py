"""Contains the implementation of the Maestro wrapper."""

from collections.abc import Callable, Hashable, Sequence
from itertools import groupby
from typing import Generic, TypeVar

from cosy.core.subtypes import Taxonomy
from cosy.core.synthesizer import Specification, Synthesizer
from cosy.core.types import Type
from cosy.extensions.solutions import _MaestroSolutions

T = TypeVar("T", bound=Hashable)


class Maestro(Generic[T]):
    """The Maestro provides an easy to use but opinionated wrapper around the Synthesizer.

    Attributes:
        named_components_with_specifications (Sequence[tuple[T, Callable, Specification]]): Stores the sequence of tuples passed in __init__.
        taxonomy (Taxonomy | None): A taxonomy containing subtyping information. (Default value = None)
        component_specifications (dict[T, Specification]): Subset of named_components_with_specifications.
        component_interpretations (dict[T, Callable]): Subset of named_components_with_specifications.
    """

    named_components_with_specifications: Sequence[tuple[T, Callable, Specification]]
    taxonomy: Taxonomy | None = None
    _synthesizer: Synthesizer

    def __init__(
        self,
        named_components_with_specifications: Sequence[tuple[T, Callable, Specification]],
        taxonomy: Taxonomy | None = None,
    ) -> None:
        """Initializes the Maestro.

        Args:
            named_components_with_specifications (Sequence[tuple[T, Callable, Specification]]):  A sequence of tuples that link a name to a corresponding type and callable.
            taxonomy (Taxonomy | None): A taxonomy containing subtyping information. (Default value = None)

        Raises:
            ValueError: Raised if the names assigned to components are not unique.
            ValueError: Raised if there are components whose interpretations are not callable.
        """
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

    def query(self, target: Type, max_count: int | None = 100) -> _MaestroSolutions[T]:
        """Query the Maestro for solutions that fulfill given target; by constructing a solution space and enumerating and interpreting the resulting trees.

        Args:
            target (Type): The target for which solutions should be queried.
            max_count (int): The maximum number of trees to enumerate. (Default value = 100)

        Returns:
            MaestroSolutions[T]: An iterable of interpreted trees, the results.

        Raises:
            TypeError: Raised if the request to the synthesizer is not a Type.
        """
        if not isinstance(target, Type):
            msg = "Target must be of type Type"
            raise TypeError(msg)
        solution_space = self._synthesizer.construct_solution_space(target).prune()

        trees = solution_space.enumerate_trees(
            target, max_count=max_count, interpretation=self.component_interpretations
        )
        return _MaestroSolutions(
            trees,
            component_interpretations=self.component_interpretations,
            named_components_with_specifications=self.named_components_with_specifications,
            taxonomy=None
            if self.taxonomy is None
            else self._synthesizer.subtypes.taxonomy,  # This way we get the closure of the taxonomy
            max_count=max_count,
        )
