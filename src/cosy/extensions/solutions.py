from collections.abc import Callable, Generator, Hashable, Iterable
from typing import Any, Generic, TypeVar

from cosy.core.tree import Tree

T = TypeVar("T", bound=Hashable)


class Solutions(Generic[T]):
    def __init__(self, tree_generator: Iterable[Tree[T]]):
        self.generated_trees: list[Tree[T]] = []

        def caching_tree_generator():
            for tree in tree_generator:
                self.generated_trees.append(tree)
                yield tree

        self.trees = caching_tree_generator()


class MaestroSolutions(Solutions[T]):
    def __init__(self, trees: Iterable[Tree[T]], component_interpretations: dict[T, Callable[..., Any]]):
        super().__init__(trees)
        self.component_interpretations = component_interpretations

    def __iter__(self) -> Generator[Any, None, None]:
        for result in self.trees:
            yield result.interpret(interpretation=self.component_interpretations)
