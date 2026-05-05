import random
from typing import Generic, TypeVar
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence

from src.cosy.core.tree import Tree
from src.cosy.core.solution_space import SolutionSpace

NT = TypeVar("NT", bound=Hashable)  # type of non-terminals
T = TypeVar("T", bound=Hashable)  # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants


class Initialization(ABC, Generic[NT, T, G]):

    def __init__(self, solution_space: SolutionSpace[NT, T, G], start: NT):
        self.solution_space = solution_space
        self.start = start

    @abstractmethod
    def initialize_population(self, population_size: int) -> Iterable[Tree[T]]:
        pass


class RandomLimitedDepthFirstInitialization(Initialization[NT, T, G], Generic[NT, T, G]):

    def __init__(self, solution_space: SolutionSpace[NT, T, G], start: NT, max_depth: int):
        super().__init__(solution_space, start)
        self.max_depth = max_depth

    def initialize_population(self, population_size: int) -> Iterable[Tree[T]]:
        for _ in range(population_size):
            tree = self.solution_space.sample_tree(self.start, self.max_depth)
            if tree is not None:
                yield tree
        return

