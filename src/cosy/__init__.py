from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import FrozenInstanceError, dataclass
from functools import update_wrapper
from typing import Any, Generic, TypeVar

from cosy.solution_space import SolutionSpace
from cosy.specification_builder import SpecificationBuilder
from cosy.subtypes import Subtypes, Taxonomy
from cosy.synthesizer import Specification, Synthesizer
from cosy.types import Arrow, Constructor, Intersection, Literal, Omega, Type, Var

__all__ = [
    "Arrow",
    "Component",
    "Constructor",
    "CoSy",
    "Intersection",
    "Literal",
    "Omega",
    "SolutionSpace",
    "Subtypes",
    "Synthesizer",
    "Type",
    "Var",
    "SpecificationBuilder",
]

T = TypeVar("T", bound=Hashable)


@dataclass
class Component:
    name: str
    interpretation: Callable

    def __post_init__(self):
        """
        Wrapping a Callable within a class makes the Python inspect module retrieve a wrong signature, statically
        returning 2 arguments (*args,*kwargs). To fix this, update_wrapper() copies the signature of the
        wrapped Callable to the Component class (which is itself a Callable).
        """

        update_wrapper(self, self.interpretation)
        self._frozen = True

    def __call__(self, *args, **kwargs):
        return self.interpretation(*args, **kwargs)

    def __setattr__(self, attr, value):
        """
        This dataclass can not be frozen, due to the modifications detailed in __post_init__. To emulate the behavior
        of a frozen dataclass after __post_init__ is executed, this method prevents modification of attributes.

        :param attr: The attribute to set.
        :param value: The value to be set.
        """

        if getattr(self, "_frozen", None):
            msg = f"cannot assign to field '{attr}'"
            raise FrozenInstanceError(msg)
        return super().__setattr__(attr, value)

    def __hash__(self):
        hash(self.name)

    def __str__(self):
        return str(self.name)


class CoSy(Generic[T]):
    component_specifications: Mapping[Component, Specification]
    taxonomy: Taxonomy | None = None
    _synthesizer: Synthesizer

    def __init__(
        self,
        component_specifications: Mapping[Component, Specification],
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
        _solution_space = self._synthesizer.construct_solution_space(query).prune()

        trees = _solution_space.enumerate_trees(query, max_count=max_count)
        for tree in trees:
            yield tree.interpret()
