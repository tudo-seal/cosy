from collections.abc import Callable, Generator, Hashable, Iterable, Sequence
from typing import Any, Generic, TypeVar

from cosy.core.tree import Tree
from cosy.core.types import Abstraction, Implication, Type
from cosy.extensions import visualize

T = TypeVar("T", bound=Hashable)


class Solutions(Generic[T]):
    def __init__(self, tree_generator: Iterable[Tree[T]]):
        self._generated_trees: list[Tree[T]] = []
        self.tree_generator = tree_generator
        # def caching_tree_generator():
        #     for tree in self._generated_trees:
        #         yield tree
        #     for tree in tree_generator:
        #         self._generated_trees.append(tree)
        #         yield tree
        #
        # self.trees = caching_tree_generator()

    def trees(self) -> Generator[Tree[T], None, None]:
        for tree in self._generated_trees:
            yield tree
        for tree in self.tree_generator:
            self._generated_trees.append(tree)
            yield tree


class MaestroSolutions(Solutions[T]):
    def __init__(
        self,
        trees: Iterable[Tree[T]],
        component_interpretations: dict[T, Callable[..., Any]],
        named_components_with_specifications: Sequence[tuple[T, Callable, Abstraction | Implication | Type]],
    ):
        super().__init__(trees)
        self.component_interpretations = component_interpretations
        self.named_components_with_specifications = named_components_with_specifications

    def __iter__(self) -> Generator[Any, None, None]:
        for result in self.trees():
            yield result.interpret(interpretation=self.component_interpretations)

    def visualize(self, amount: int = 10):
        visualize.visualize(
            amount=amount,
            trees=self.trees(),
            named_components_with_specifications=self.named_components_with_specifications,
        )
