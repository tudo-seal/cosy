from dataclasses import dataclass, field
from typing import Generic, Iterable, Callable, Any, Generator, List, TypeVar
from cosy.tree import Tree

T: TypeVar = TypeVar("T")

@dataclass
class Solutions(Generic[T]):
    tree_generator: Iterable[Tree[T]]
    generated_trees: List[Tree[T]] = field(init=False, default_factory=list)

@dataclass
class SynthesizerSolutions(Solutions[T]):
    def __iter__(self):
        for tree in self.tree_generator:
            self.generated_trees.append(tree)
            yield tree

@dataclass
class MaestroSolutions(Solutions[T]):
    component_interpretations: dict[T, Callable[[...], Any]]

    def __iter__(self) -> Generator[Any, None, None]:
        for tree in self.tree_generator:
            self.generated_trees.append(tree)
            yield tree.interpret(interpretation=self.component_interpretations)
