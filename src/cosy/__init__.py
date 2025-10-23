from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import FrozenInstanceError, dataclass
from functools import update_wrapper
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


@dataclass(unsafe_hash=True)
class Component(Callable):
    name: str
    interpretation: Callable

    def __post_init__(self):
        update_wrapper(self, self.interpretation)
        self._frozen = True

    def __call__(self, *args, **kwargs):
        return self.interpretation(*args, **kwargs)

    def __setattr__(self, attr, value):
        if getattr(self, "_frozen", None):
            msg = f"cannot assign to field '{attr}'"
            raise FrozenInstanceError(msg)
        return super().__setattr__(attr, value)


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
