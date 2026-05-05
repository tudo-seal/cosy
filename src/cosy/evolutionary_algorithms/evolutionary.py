import random
from typing import Generic, TypeVar, Any
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence

from src.cosy.core.tree import Tree
from src.cosy.core.solution_space import SolutionSpace

from src.cosy.evolutionary_algorithms.mutation import Mutation
from src.cosy.evolutionary_algorithms.recombination import Recombination
from src.cosy.evolutionary_algorithms.selection import Selection
from src.cosy.evolutionary_algorithms.initialisation import Initialization

NT = TypeVar("NT", bound=Hashable)  # type of non-terminals
T = TypeVar("T", bound=Hashable)  # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants


class Evolutionary(ABC, Generic[NT, T, G]):

    def __init__(self, solution_space: SolutionSpace[NT, T, G], start: NT,
                 termination_condition: Callable[[Any], bool],
                 fitness_function: Callable[[Tree[T]], float],
                 initialization: Initialization[NT, T, G],
                 mutation: Mutation[NT, T, G],
                 recombination: Recombination[NT, T, G],
                 parent_selection: Selection[NT, T, G],
                 survivor_selection: Selection[NT, T, G],):
        self.solution_space = solution_space
        self.start = start
        self.termination_condition = termination_condition
        self.objective_function = fitness_function
        self.initialization = initialization
        self.mutation = mutation
        self.recombination = recombination
        self.parent_selection = parent_selection
        self.survivor_selection = survivor_selection

    @abstractmethod
    def evolutionary_search(self, population_size: int, mutation_rate: float, recombination_rate: float) -> Tree[T]:
        pass





