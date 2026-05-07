"""
Selection operators for evolutionary algorithms.

Selection operators are responsible for choosing individuals from a population based on their fitness.
They are used for both parent selection (choosing individuals for reproduction) and survivor selection
(choosing individuals to survive to the next generation).

This module provides several selection strategies that balance exploitation (favoring good solutions)
with exploration (maintaining diversity).
"""

import random
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Sequence, Mapping
from typing import Generic, TypeVar

from cosy.core.tree import Tree
from cosy.evolutionary_algorithms.fitness import Fitness, FitnessComparator

NT = TypeVar("NT", bound=Hashable)  # type of non-terminals
T = TypeVar("T", bound=Hashable)  # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants


class Selection(ABC, Generic[NT, T, G]):
    """Abstract base class for selection operators.
    
    Selection operators are responsible for choosing individuals from a population.
    They can be used for parent selection (producing a mating pool) or
    survivor selection (choosing individuals for the next generation).
    
    Type Parameters:
        NT: Type of non-terminals in the grammar/search space
        T: Type of terminals in the grammar/search space
        G: Type of constants/ground symbols
    """

    @abstractmethod
    def select(self, population_fitness: Mapping[Tree[T], Fitness], population_size: int,
               comparator: FitnessComparator,
               previous_generation_fitness: Mapping[Tree[T], Fitness] | None = None,
               ages: Mapping[Tree[T], int] | None = None,
               previous_generation_ages: Mapping[Tree[T], int] | None = None) -> Iterable[Tree[T]]:
        """Select population_size individuals from the given population.
        
        Args:
            population_fitness: Mapping from individuals to their fitness values.
            population_size: The number of individuals to select.
            comparator: The fitness comparator for comparing individuals.
            previous_generation_fitness: Optional fitness values from the previous generation (for survivor selection).
            ages: Optional ages of individuals (for age-aware selection).
            previous_generation_ages: Optional ages from the previous generation.
        
        Yields:
            Selected individuals up to population_size.
        """
        pass


class TournamentSelection(Selection[NT, T, G]):
    """Tournament selection: select best from random tournaments.
    
    In each round, this operator selects tournament_size random individuals from the population
    and returns the best one. This process is repeated to generate population_size selections.
    
    Tournament selection is effective for parent selection as it balances selection pressure
    and diversity without requiring global fitness rankings.
    
    Attributes:
        tournament_size: The number of individuals in each tournament.
    """

    def __init__(self, tournament_size: int, rng: random.Random | None = None):
        """Initialize tournament selection.
        
        Args:
            tournament_size: Number of individuals per tournament (typically 2-5).
            rng: Optional random number generator for reproducibility.
        """
        self.tournament_size = tournament_size
        self.rng = rng if rng is not None else random.Random()

    def select(self, population_fitness: Mapping[Tree[T], Fitness], population_size: int,
               comparator: FitnessComparator,
               previous_generation_fitness: Mapping[Tree[T], Fitness] | None = None,
               ages: Mapping[Tree[T], int] | None = None,
               previous_generation_ages: Mapping[Tree[T], int] | None = None) -> Iterable[Tree[T]]:
        """Conduct population_size tournaments and yield the winner of each.
        
        Args:
            population_fitness: Mapping from individuals to fitness values.
            population_size: Number of individuals to select.
            comparator: The fitness comparator.
            previous_generation_fitness: Unused (for interface compatibility).
            ages: Unused (for interface compatibility).
            previous_generation_ages: Unused (for interface compatibility).
        
        Yields:
            Winners of each tournament.
        """
        population = list(population_fitness.keys())
        if not population_size or not population:
            return
        
        for _ in range(population_size):
            # Create a tournament with random individuals
            tournament = self.rng.sample(population, min(self.tournament_size, len(population)))
            best = tournament[0]
            best_score = comparator.sort_key(population_fitness[best])
            
            # Find the best individual in the tournament
            for candidate in tournament[1:]:
                candidate_score = comparator.sort_key(population_fitness[candidate])
                if candidate_score > best_score:
                    best = candidate
                    best_score = candidate_score
                elif candidate_score == best_score and self.rng.random() < 0.5:
                    best = candidate
                    best_score = candidate_score
            
            yield best


class FitnessProportionalSelection(Selection[NT, T, G]):
    """Fitness proportional selection: probability proportional to fitness.
    
    Individuals with higher fitness have higher probability of selection.
    Handles negative fitness values by shifting them, and avoids division by zero
    when all individuals have zero fitness.
    
    This is a classic selection method that provides strong selection pressure
    but can suffer from loss of diversity in later generations.
    """

    def __init__(self, rng: random.Random | None = None):
        """Initialize fitness proportional selection.
        
        Args:
            rng: Optional random number generator for reproducibility.
        """
        self.rng = rng if rng is not None else random.Random()

    @staticmethod
    def _weights(fitness_values: Sequence[Fitness], comparator: FitnessComparator) -> list[float]:
        """Compute selection weights from fitness values.
        
        Handles negative fitness by shifting all values above zero.
        If all weights are zero, uses uniform weights.
        
        Args:
            fitness_values: The fitness values.
            comparator: The fitness comparator.
        
        Returns:
            A list of non-negative weights usable for weighted selection.
        """
        values = [comparator.scalarize(fitness) for fitness in fitness_values]
        minimum = min(values)
        weights = [value - minimum if minimum < 0 else value for value in values]

        if not any(weight > 0 for weight in weights):
            return [1.0] * len(fitness_values)
        return [float(weight) for weight in weights]

    def select(self, population_fitness: Mapping[Tree[T], Fitness], population_size: int,
               comparator: FitnessComparator,
               previous_generation_fitness: Mapping[Tree[T], Fitness] | None = None,
               ages: Mapping[Tree[T], int] | None = None,
               previous_generation_ages: Mapping[Tree[T], int] | None = None) -> Iterable[Tree[T]]:
        """Select individuals proportional to their fitness.
        
        Args:
            population_fitness: Mapping from individuals to fitness values.
            population_size: Number of individuals to select.
            comparator: The fitness comparator.
            previous_generation_fitness: Unused (for interface compatibility).
            ages: Unused (for interface compatibility).
            previous_generation_ages: Unused (for interface compatibility).
        
        Yields:
            population_size individuals selected proportional to fitness.
        """
        population = list(population_fitness.keys())
        if not population_size or not population:
            return

        fitness_values = [population_fitness[tree] for tree in population]
        weights = self._weights(fitness_values, comparator)

        for _ in range(population_size):
            yield self.rng.choices(population, weights=weights, k=1)[0]


class RankBasedSelection(Selection[NT, T, G]):
    """Rank-based selection: probability based on rank not absolute fitness.
    
    Individuals are ranked by fitness and selected with probabilities determined by their rank.
    This provides more stable selection pressure than fitness-proportional selection since
    it doesn't depend on absolute fitness differences.
    
    Attributes:
        selection_pressure: Controls the pressure toward higher-ranked individuals (1.0-2.0).
                          1.0 = uniform selection, 2.0 = maximum pressure toward best.
    """

    def __init__(self, selection_pressure: float = 1.7, rng: random.Random | None = None):
        """Initialize rank-based selection.
        
        Args:
            selection_pressure: Strength of preference for better-ranked individuals (1.0-2.0).
            rng: Optional random number generator for reproducibility.
            
        Raises:
            ValueError: If selection_pressure is not in [1.0, 2.0].
        """
        if not 1.0 <= selection_pressure <= 2.0:
            raise ValueError("selection_pressure must be in [1.0, 2.0]")
        self.selection_pressure = float(selection_pressure)
        self.rng = rng if rng is not None else random.Random()

    def select(self, population_fitness: Mapping[Tree[T], Fitness], population_size: int,
               comparator: FitnessComparator,
               previous_generation_fitness: Mapping[Tree[T], Fitness] | None = None,
               ages: Mapping[Tree[T], int] | None = None,
               previous_generation_ages: Mapping[Tree[T], int] | None = None) -> Iterable[Tree[T]]:
        """Select individuals based on their fitness rank.
        
        Args:
            population_fitness: Mapping from individuals to fitness values.
            population_size: Number of individuals to select.
            comparator: The fitness comparator.
            previous_generation_fitness: Unused (for interface compatibility).
            ages: Unused (for interface compatibility).
            previous_generation_ages: Unused (for interface compatibility).
        
        Yields:
            population_size individuals selected according to their rank.
        """
        population = list(population_fitness.keys())
        if not population_size or not population:
            return

        # Sort population by fitness (best first)
        ranked = sorted(
            population,
            key=lambda tree: comparator.sort_key(population_fitness[tree]),
            reverse=True,
        )
        count = len(ranked)
        
        # Special case: single individual
        if count == 1:
            for _ in range(population_size):
                yield ranked[0]
            return

        # Compute rank-based weights using linear ranking formula
        pressure = self.selection_pressure
        weights = [
            (2 - pressure) + (2 * (pressure - 1) * (count - rank - 1) / (count - 1))
            for rank in range(count)
        ]
        
        for _ in range(population_size):
            yield self.rng.choices(ranked, weights=weights, k=1)[0]


class FitnessBasedReplacement(Selection[NT, T, G]):
    """Fitness-based survivor selection: select best individuals from all candidates.
    
    This strategy combines offspring and previous generation members (if provided),
    and selects the population_size best individuals by fitness. This implements
    (μ+λ) survivor selection and can be used for elitist replacement strategies.
    """

    def __init__(self):
        """Initialize fitness-based replacement."""
        pass

    def select(self, population_fitness: Mapping[Tree[T], Fitness], population_size: int,
               comparator: FitnessComparator,
               previous_generation_fitness: Mapping[Tree[T], Fitness] | None = None,
               ages: Mapping[Tree[T], int] | None = None,
               previous_generation_ages: Mapping[Tree[T], int] | None = None) -> Iterable[Tree[T]]:
        """Select the population_size best individuals from the current and previous generation.
        
        Args:
            population_fitness: Mapping from offspring to fitness values.
            population_size: Number of individuals to select.
            comparator: The fitness comparator.
            previous_generation_fitness: Fitness values from the previous generation (combined with current).
            ages: Unused (for interface compatibility).
            previous_generation_ages: Unused (for interface compatibility).
        
        Yields:
            The population_size best individuals by fitness.
        """
        if not population_size:
            return

        # Combine current and previous generation fitness
        combined: dict[Tree[T], Fitness] = dict(population_fitness)
        if previous_generation_fitness is not None:
            for tree, fitness in previous_generation_fitness.items():
                combined.setdefault(tree, fitness)

        # Select the best individuals
        ranked = sorted(combined.keys(),
                        key=lambda tree: comparator.sort_key(combined[tree]),
                        reverse=True)
        for tree in ranked[:population_size]:
            yield tree


class AgeBasedReplacement(Selection[NT, T, G]):
    """Age-based survivor selection: prefer younger individuals, break ties by fitness.
    
    This strategy selects survivors primarily based on age (younger is better),
    using fitness as a tie-breaker. This is useful for maintaining diversity
    by preventing individuals from dominating for too many generations.
    """

    def __init__(self):
        """Initialize age-based replacement."""
        pass

    def select(self, population_fitness: Mapping[Tree[T], Fitness], population_size: int,
               comparator: FitnessComparator,
               previous_generation_fitness: Mapping[Tree[T], Fitness] | None = None,
               ages: Mapping[Tree[T], int] | None = None,
               previous_generation_ages: Mapping[Tree[T], int] | None = None) -> Iterable[Tree[T]]:
        """Select population_size individuals primarily by age, secondarily by fitness.
        
        Args:
            population_fitness: Mapping from offspring to fitness values.
            population_size: Number of individuals to select.
            comparator: The fitness comparator.
            previous_generation_fitness: Optional fitness values from previous generation.
            ages: Ages of current generation individuals.
            previous_generation_ages: Ages from the previous generation.
        
        Yields:
            population_size individuals selected by age (younger first) and fitness.
        """
        if not population_size:
            return

        # If no age information, fall back to fitness-based selection
        if ages is None:
            selected: list[Tree[T]] = list(population_fitness.keys())
            if len(selected) < population_size and previous_generation_fitness is not None:
                for tree in previous_generation_fitness.keys():
                    if tree not in population_fitness:
                        selected.append(tree)
                    if len(selected) >= population_size:
                        break
            for tree in selected[:population_size]:
                yield tree
            return

        # Combine current and previous generation fitness and ages
        combined_fitness: dict[Tree[T], Fitness] = dict(population_fitness)
        combined_ages: dict[Tree[T], int] = dict(ages)
        
        if previous_generation_fitness is not None:
            for tree, fitness in previous_generation_fitness.items():
                combined_fitness.setdefault(tree, fitness)
                if previous_generation_ages is not None:
                    combined_ages.setdefault(tree, previous_generation_ages.get(tree, combined_ages.get(tree, 0) + 1))
                else:
                    combined_ages.setdefault(tree, combined_ages.get(tree, 0) + 1)

        # Sort by age (ascending) and then by fitness (descending)
        ranked = sorted(
            combined_fitness.keys(),
            key=lambda tree: (combined_ages.get(tree, 0), -comparator.sort_key(combined_fitness[tree])),
        )
        for tree in ranked[:population_size]:
            yield tree

