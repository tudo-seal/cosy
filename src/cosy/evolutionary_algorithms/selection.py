import random
from typing import Generic, TypeVar
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence

from src.cosy.core.tree import Tree
from src.cosy.core.solution_space import SolutionSpace

NT = TypeVar("NT", bound=Hashable)  # type of non-terminals
T = TypeVar("T", bound=Hashable)  # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants


class Selection(ABC, Generic[NT, T, G]):

    @abstractmethod
    def select(self, population: Iterable[Tree[T]], n: int) -> Iterable[Tree[T]]:
        pass


