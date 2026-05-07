"""
Recombination operators for evolutionary algorithms.

Recombination (crossover) operators combine genetic material from two parent solutions
to create offspring. This is essential for exploiting promising areas of the search space
by combining good building blocks from different solutions.
"""

import random
from typing import Generic, TypeVar
from abc import ABC, abstractmethod
from collections.abc import Hashable

from cosy.core.tree import Tree
from cosy.core.solution_space import SolutionSpace
from itertools import product

NT = TypeVar("NT", bound=Hashable)  # type of non-terminals
T = TypeVar("T", bound=Hashable)  # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants

Path = tuple[int, ...]

class Recombination(ABC, Generic[NT, T, G]):
    """Abstract base class for recombination (crossover) operators.
    
    Recombination operators combine two parent individuals to create offspring.
    Subclasses implement specific crossover strategies by implementing the recombine method.
    
    Type Parameters:
        NT: Type of non-terminals in the grammar/search space
        T: Type of terminals in the grammar/search space
        G: Type of constants/ground symbols
    """

    def __init__(self, solution_space: SolutionSpace[NT, T, G], start: NT, max_depth: int | None = None):
        """Initialize the recombination operator.
        
        Args:
            solution_space: The search space that defines valid individuals.
            start: The start non-terminal for generating new individuals.
            max_depth: The maximum depth of the trees in the search space.
        """
        self.solution_space = solution_space
        self.start = start
        self.max_depth = max_depth

    @abstractmethod
    def recombine(self, primary: Tree[T], secondary: Tree[T]) -> list[Tree[T]]:
        """Recombine two parent trees to create offspring.
        
        Args:
            primary: The first parent tree.
            secondary: The second parent tree.
        
        Returns:
            A list of valid offspring, or an empty list if recombination failed.
        """
        pass


class Crossover(Recombination[NT, T, G], Generic[NT, T, G]):
    """Subtree crossover operator.
    
    This operator performs standard genetic programming crossover by swapping
    random subtrees between two parents. The crossover points are selected from
    non-leaf positions to ensure meaningful recombination.
    
    Type Parameters:
        NT: Type of non-terminals in the grammar/search space
        T: Type of terminals in the grammar/search space
        G: Type of constants/ground symbols
    """

    def maximum_leaf_length(self, leaf_positions: set[Path], position: Path) -> int:
        """Calculate the maximum leaf length for a given position in the tree.

        This is used to ensure that when swapping subtrees, the resulting offspring
        do not exceed the maximum depth constraint. The maximum leaf length is
        determined by the distance from the position to the nearest leaf in the tree.

        Args:
            leaf_positions: A list of paths to leaf nodes in the tree.
            position: The path to the current position being evaluated.

        Returns:
            The maximum allowed depth for a subtree at the given position.
        """
        return max(len(leaf) - len(position) for leaf in leaf_positions if leaf[:len(position)] == position)

    def recombine(self, primary: Tree[T], secondary: Tree[T]) -> list[Tree[T]]:
        """Exchange subtrees at randomly selected crossover points.
        
        Algorithm:
        1. Collect valid crossover points (non-leaf positions) in both parents
        2. Randomly select a crossover point pair
        3. Swap the corresponding subtrees
        4. Check if offspring are valid (contained in the solution space)
        5. If invalid, retry with other point pairs
        6. Return valid offspring or empty list if none found
        
        Args:
            primary: The primary parent tree.
            secondary: The secondary parent tree.
        
        Returns:
            A list containing up to two offspring if valid, or an empty list if no valid
            offspring could be produced.
        """
        # Collect valid crossover points in the primary parent
        # (exclude root and leaves to ensure meaningful swaps)
        primary_positions = list(primary.positions())
        primary_positions.remove(())  # Remove root
        for leaf in primary.leaf_positions():
            primary_positions.remove(leaf)  # Remove leaves
        if not primary_positions:
            return []

        # Collect valid crossover points in the secondary parent
        secondary_positions = list(secondary.positions())
        secondary_positions.remove(())  # Remove root
        for leaf in secondary.leaf_positions():
            secondary_positions.remove(leaf)  # Remove leaves
        if not secondary_positions:
            return []

        random.shuffle(primary_positions)
        random.shuffle(secondary_positions)

        # Generate all possible crossover point pairs
        possible_recombination_points = product(primary_positions, secondary_positions)
        iterator = iter(possible_recombination_points)

        # Try a random crossover point pair
        #primary_crossover_point, secondary_crossover_point = possible_recombination_points.
        #possible_recombination_points.remove((primary_crossover_point, secondary_crossover_point))
        try:
            primary_crossover_point, secondary_crossover_point = next(iterator)
            primary_max_depth = self.maximum_leaf_length(primary.leaf_positions(), primary_crossover_point)
            secondary_max_depth = self.maximum_leaf_length(secondary.leaf_positions(), secondary_crossover_point)
            if self.max_depth is not None:
                while (len(primary_crossover_point) + secondary_max_depth > self.max_depth or
                       len(secondary_crossover_point) + primary_max_depth > self.max_depth):
                    primary_crossover_point, secondary_crossover_point = next(iterator)
        except StopIteration:
            return []


        # Extract subtrees at crossover points
        primary_subtree = primary.subtree_at(primary_crossover_point)
        secondary_subtree = secondary.subtree_at(secondary_crossover_point)

        # Create offspring by swapping subtrees
        primary_child = primary.replace_subtree_at(primary_crossover_point, secondary_subtree)
        secondary_child = secondary.replace_subtree_at(secondary_crossover_point, primary_subtree)

        # Check if offspring are valid
        if (self.solution_space.contains_tree(self.start, primary_child)
                and self.solution_space.contains_tree(self.start, secondary_child)):
            return [primary_child, secondary_child]

        # If offspring are invalid, retry with other crossover points
        while (not self.solution_space.contains_tree(self.start, primary_child)
               and not self.solution_space.contains_tree(self.start, secondary_child)
               and primary_positions and secondary_positions):
            try:
                primary_crossover_point, secondary_crossover_point = next(iterator)
                primary_max_depth = self.maximum_leaf_length(primary.leaf_positions(), primary_crossover_point)
                secondary_max_depth = self.maximum_leaf_length(secondary.leaf_positions(), secondary_crossover_point)
                if self.max_depth is not None:
                    while len(primary_crossover_point) + secondary_max_depth > self.max_depth or len(
                            secondary_crossover_point) + primary_max_depth > self.max_depth:
                        primary_crossover_point, secondary_crossover_point = next(iterator)
            except StopIteration:
                break

            primary_subtree = primary.subtree_at(primary_crossover_point)
            secondary_subtree = secondary.subtree_at(secondary_crossover_point)

            primary_child = primary.replace_subtree_at(primary_crossover_point, secondary_subtree)
            secondary_child = secondary.replace_subtree_at(secondary_crossover_point, primary_subtree)

            if (self.solution_space.contains_tree(self.start, primary_child)
                    and self.solution_space.contains_tree(self.start, secondary_child)):
                return [primary_child, secondary_child]

        return []


