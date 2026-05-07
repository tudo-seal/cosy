"""
Mutation operators for evolutionary algorithms.

Mutation operators modify individual solutions to explore the search space.
They are crucial for maintaining genetic diversity and escaping local optima.
"""

import random
from typing import Generic, TypeVar
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable

from cosy.core.tree import Tree
from cosy.core.solution_space import SolutionSpace

NT = TypeVar("NT", bound=Hashable)  # type of non-terminals
T = TypeVar("T", bound=Hashable)  # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants


class Mutation(ABC, Generic[NT, T, G]):
    """Abstract base class for mutation operators.
    
    Mutation operators transform individuals by modifying their structure.
    Subclasses implement specific mutation strategies by implementing the mutate method.
    
    Type Parameters:
        NT: Type of non-terminals in the grammar/search space
        T: Type of terminals in the grammar/search space
        G: Type of constants/ground symbols
    """

    def __init__(self, solution_space: SolutionSpace[NT, T, G], start: NT, max_depth: int | None = None) -> None:
        """Initialize the mutation operator.
        
        Args:
            solution_space: The search space that defines valid individuals.
            start: The start non-terminal for generating new subtrees.
            max_depth: Optional maximum depth constraint for generated subtrees.
                      If None, no depth constraint is applied.
        """
        self.solution_space = solution_space
        self.start = start
        self.max_depth = max_depth

    @abstractmethod
    def mutate(self, tree: Tree[T]) -> Iterable[Tree[T]]:
        """Apply mutation to an individual tree.
        
        Args:
            tree: The individual to mutate.
        
        Yields:
            One or more mutated variants of the input tree,
            or nothing if mutation is not possible.
        """
        pass


class ResolutionMutation(Mutation[NT, T, G], Generic[NT, T, G]):
    """Mutation by replacing subtrees at non-leaf positions.
    
    This operator selects a non-leaf position in the tree and replaces the subtree
    at that position with a newly generated subtree, thus modifying the tree structure.
    The new subtree is generated respecting the search space constraints and optional
    depth limits.
    
    Type Parameters:
        NT: Type of non-terminals in the grammar/search space
        T: Type of terminals in the grammar/search space
        G: Type of constants/ground symbols
    """

    def mutate(self, tree: Tree[T]) -> list[Tree[T]]:
        """Replace a random non-leaf subtree with a newly sampled one.
        
        Algorithm:
        1. Collect all non-leaf positions in the tree
        2. Randomly select one as the mutation point
        3. Sample a new subtree at that position respecting constraints
        4. If sampling fails, retry with other positions up to the pool limit
        
        Args:
            tree: The tree to mutate.
        
        Returns:
            A list containing the mutated tree, or an empty list if mutation failed.
        """
        # Get all non-leaf positions (excluding root and leaves)
        positions = list(tree.positions())
        positions.remove(())  # Remove root
        for leaf in tree.leaf_positions():
            positions.remove(leaf)  # Remove leaves
        if not positions:
            return []

        # Try to replace a random non-leaf position with a new subtree
        mutation_point = random.choice(positions)
        positions.remove(mutation_point)
        mutant = self.solution_space.sample_tree(self.start, tree=tree, pos=mutation_point, max_depth=self.max_depth)
        
        # If the first attempt fails, retry with other positions
        while mutant is None and positions:
            mutation_point = random.choice(positions)
            positions.remove(mutation_point)
            mutant = self.solution_space.sample_tree(self.start, tree=tree, pos=mutation_point, max_depth=self.max_depth)
        
        if mutant is not None:
            return [mutant]
        return []

