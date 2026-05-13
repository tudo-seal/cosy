"""
Mutation operators for evolutionary algorithms.

Mutation operators modify individual solutions to explore the search space.
They are crucial for maintaining genetic diversity and escaping local optima.
"""

import random
from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import Generic, TypeVar

from cosy.core.solution_space import SolutionSpace
from cosy.core.tree import Tree

NT = TypeVar("NT", bound=Hashable)  # type of non-terminals
T = TypeVar("T", bound=Hashable)  # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants

Path = tuple[int, ...]


class Mutation(ABC, Generic[NT, T, G]):
    """Abstract base class for mutation operators.

    Mutation operators transform individuals by modifying their structure.
    Subclasses implement specific mutation strategies by implementing the mutate method.
    """

    def __init__(
        self,
        solution_space: SolutionSpace[NT, T, G],
        start: NT,
        max_depth: int | None = None,
        rng: random.Random | None = None,
    ) -> None:
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
        self.rng = rng if rng is not None else random.Random()

    @abstractmethod
    def mutate(self, tree: Tree[T], trim: int = 1, min_trim_length: int = 1) -> list[Tree[T]]:
        """Apply mutation to an individual tree.

        Args:
            tree: The individual to mutate.
            trim: Enforce mutation points nearer to the root by removing a suffix of length n from the leaf-paths.
                  For example, trim=1 means only consider positions that are not leaves.
            min_trim_length: Optional minimum depth of a path that is allowed to be trimmed.

        Yields:
            One or more mutated variants of the input tree,
            or nothing if mutation is not possible.
        """


class ResolutionMutation(Mutation[NT, T, G], Generic[NT, T, G]):
    """Mutation by replacing subtrees at non-leaf positions.

    This operator selects a non-leaf position in the tree and replaces the subtree
    at that position with a newly generated subtree, thus modifying the tree structure.
    The new subtree is generated respecting the search space constraints and optional
    depth limits.
    """

    def mutate(self, tree: Tree[T], trim: int = 1, min_trim_length: int = 1) -> list[Tree[T]]:
        """Replace a random non-leaf subtree with a newly sampled one.

        Algorithm:
        1. Collect all non-leaf positions in the tree
        2. Randomly select one as the mutation point
        3. Sample a new subtree at that position respecting constraints
        4. If sampling fails, retry with other positions up to the pool limit

        Args:
            tree: The tree to mutate.
            trim: Enforce mutation points nearer to the root by removing a suffix of length n from the leaf-paths.
                  For example, trim=1 means only consider positions that are not leaves.
            min_trim_length: Optional minimum depth of a path that is allowed to be trimmed.

        Returns:
            A list containing the mutated tree, or an empty list if mutation failed.
        """
        # Get all non-leaf positions (excluding root and leaves)
        positions = list(tree.positions())
        positions.remove(())  # Remove root
        leafs: set[Path] = tree.leaf_positions()
        for i in range(trim):
            for leaf in leafs:
                positions.remove(leaf)  # Remove leaves
            if not positions:
                return []
            if i < trim - 1:
                leafs = set()
                # leaf positions are all positions that are no prefix of another position
                # a prefix of a position is defined as follows: p is a prefix of q if p == q or p is a prefix of q[:-1]
                for pos in positions:
                    if (min_trim_length <= len(pos)) and not any(
                        pos != other and pos == other[: len(pos)] for other in positions
                    ):
                        leafs.add(pos)

        # Try to replace a random non-leaf position with a new subtree
        mutation_point = self.rng.choice(positions)
        positions.remove(mutation_point)
        mutant = self.solution_space.sample_tree(
            self.start, tree=tree, pos=mutation_point, max_depth=self.max_depth, rng=self.rng
        )
        print(f"Mutation point: {mutation_point}")
        # If the first attempt fails, retry with other positions
        while mutant is None and positions:
            mutation_point = self.rng.choice(positions)
            print(f"Mutation point failed, updated to: {mutation_point}")
            positions.remove(mutation_point)
            mutant = self.solution_space.sample_tree(
                self.start, tree=tree, pos=mutation_point, max_depth=self.max_depth, rng=self.rng
            )

        if mutant is not None:
            return [mutant]
        return []
