import random
from typing import Generic, TypeVar
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence

from src.cosy.core.tree import Tree
from src.cosy.core.solution_space import SolutionSpace
from itertools import product

NT = TypeVar("NT", bound=Hashable)  # type of non-terminals
T = TypeVar("T", bound=Hashable)  # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants


class Recombination(ABC, Generic[NT, T, G]):

    def __init__(self, solution_space: SolutionSpace[NT, T, G], start: NT):
        self.solution_space = solution_space
        self.start = start

    @abstractmethod
    def recombine(self, primary: Tree[T], secondary: Tree[T]) -> list[Tree[T]]:
        pass


class Crossover(Recombination[NT, T, G], Generic[NT, T, G]):

    def recombine(self, primary: Tree[T], secondary: Tree[T]) -> list[Tree[T]]:
        primary_positions = list(primary.positions())
        primary_positions.remove(())
        for leaf in primary.leaf_positions():
            primary_positions.remove(leaf)
        if not primary_positions:
            return []

        secondary_positions = list(secondary.positions())
        secondary_positions.remove(())
        for leaf in secondary.leaf_positions():
            secondary_positions.remove(leaf)
        if not secondary_positions:
            return []

        possible_recombination_points = list(product(primary_positions, secondary_positions))

        primary_crossover_point, secondary_crossover_point = random.choice(possible_recombination_points)
        possible_recombination_points.remove((primary_crossover_point, secondary_crossover_point))

        primary_subtree = primary.subtree_at(primary_crossover_point)
        secondary_subtree = secondary.subtree_at(secondary_crossover_point)

        primary_child = primary.replace_subtree_at(primary_crossover_point, secondary_subtree)
        secondary_child = secondary.replace_subtree_at(secondary_crossover_point, primary_subtree)

        if (self.solution_space.contains_tree(self.start, primary_child)
                and self.solution_space.contains_tree(self.start, secondary_child)):
            return [primary_child, secondary_child]

        while (not self.solution_space.contains_tree(self.start, primary_child)
               and not self.solution_space.contains_tree(self.start, secondary_child)
               and primary_positions and secondary_positions):
            primary_crossover_point, secondary_crossover_point = random.choice(possible_recombination_points)
            possible_recombination_points.remove((primary_crossover_point, secondary_crossover_point))

            primary_subtree = primary.subtree_at(primary_crossover_point)
            secondary_subtree = secondary.subtree_at(secondary_crossover_point)

            primary_child = primary.replace_subtree_at(primary_crossover_point, secondary_subtree)
            secondary_child = secondary.replace_subtree_at(secondary_crossover_point, primary_subtree)

            if (self.solution_space.contains_tree(self.start, primary_child)
                    and self.solution_space.contains_tree(self.start, secondary_child)):
                return [primary_child, secondary_child]

        return []


