"""
Population initialization strategies for evolutionary algorithms.

This module provides components for creating initial populations of candidate solutions.
Different initialization strategies can significantly impact the quality and diversity
of the initial population and thus the overall search performance.
"""

import random
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable
from typing import Generic, TypeVar

from cosy.core.solution_space import SolutionSpace
from cosy.core.tree import Tree

NT = TypeVar("NT", bound=Hashable)  # type of non-terminals
T = TypeVar("T", bound=Hashable)  # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants


class Initialization(ABC, Generic[NT, T, G]):
    """Abstract base class for population initialization strategies.

    Subclasses must implement the initialize_population method to generate
    valid individuals according to their specific strategy.
    """

    def __init__(self, solution_space: SolutionSpace[NT, T, G], start: NT, rng: random.Random | None = None):
        """Initialize the strategy with a search space and start symbol.

        Args:
            solution_space: The search space that defines valid individuals.
            start: The start non-terminal for generating individuals.
        """
        self.solution_space = solution_space
        self.start = start
        self.rng = rng if rng is not None else random.Random()

    @abstractmethod
    def initialize_population(self, population_size: int) -> Iterable[Tree[T]]:
        """Generate an initial population of candidate solutions.

        Args:
            population_size: The desired size of the population.

        Yields:
            Valid individuals (Tree objects) from the search space.
        """


class RandomLimitedDepthFirstInitialization(Initialization[NT, T, G], Generic[NT, T, G]):
    """Initialize population with random trees limited to a maximum depth.

    This strategy generates individuals using the solution space's sampling method,
    constraining the maximum depth to ensure reasonable computational complexity
    and control tree size.
    """

    def __init__(
        self, solution_space: SolutionSpace[NT, T, G], start: NT, max_depth: int, rng: random.Random | None = None
    ):
        """Initialize the random limited-depth initialization strategy.

        Args:
            solution_space: The search space that defines valid individuals.
            start: The start non-terminal for generating individuals.
            max_depth: Maximum tree depth for generated individuals.
        """
        super().__init__(solution_space, start, rng)
        self.max_depth = max_depth

    def initialize_population(self, population_size: int) -> Iterable[Tree[T]]:
        """Generate population_size random individuals with limited depth.

        Args:
            population_size: The desired population size.

        Yields:
            Valid Tree individuals sampled from the solution space,
            or skips None results from failed sampling attempts.
        """
        for _ in range(population_size):
            tree = self.solution_space.sample_tree(self.start, self.max_depth, rng=self.rng)
            if tree is not None:
                yield tree
        return
