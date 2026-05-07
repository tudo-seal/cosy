import random
from typing import Generic, TypeVar
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence

from src.cosy.core.tree import Tree
from src.cosy.core.solution_space import SolutionSpace

NT = TypeVar("NT", bound=Hashable)  # type of non-terminals
T = TypeVar("T", bound=Hashable)  # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants


class Mutation(ABC, Generic[NT, T, G]):

    def __init__(self, solution_space: SolutionSpace[NT, T, G], start: NT, max_depth: int | None = None) -> None:
        self.solution_space = solution_space
        self.start = start
        self.max_depth = max_depth

    @abstractmethod
    def mutate(self, tree: Tree[T]) -> Iterable[Tree[T]]:
        pass


class ResolutionMutation(Mutation[NT, T, G], Generic[NT, T, G]):

    def mutate(self, tree: Tree[T]) -> list[Tree[T]]:
        positions = list(tree.positions())
        positions.remove(())
        for leaf in tree.leaf_positions():
            positions.remove(leaf)
        if not positions:
            return []

        mutation_point = random.choice(positions)
        positions.remove(mutation_point)
        mutant = self.solution_space.sample_tree(self.start, tree=tree, pos=mutation_point, max_depth=self.max_depth)
        while mutant is None and positions:
            mutation_point = random.choice(positions)
            positions.remove(mutation_point)
            mutant = self.solution_space.sample_tree(self.start, tree=tree, pos=mutation_point, max_depth=self.max_depth)
        if mutant is not None:
            return [mutant]
        return []

