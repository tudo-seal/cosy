"""
Component-oriented framework for evolutionary algorithms.

This module provides the core infrastructure for building customizable evolutionary algorithms
by composing independent components (initialization, mutation, recombination, selection).
The framework uses SolutionSpace to define a constraint-based search space and Tree to represent
individuals in the population.
"""

import random
from dataclasses import dataclass
from typing import Generic, TypeVar
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable

from cosy.core.tree import Tree
from cosy.core.solution_space import SolutionSpace

from cosy.evolutionary_algorithms.mutation import Mutation
from cosy.evolutionary_algorithms.recombination import Recombination
from cosy.evolutionary_algorithms.selection import Selection
from cosy.evolutionary_algorithms.initialisation import Initialization
from cosy.evolutionary_algorithms.fitness import Fitness, FitnessComparator, ScalarFitnessComparator

NT = TypeVar("NT", bound=Hashable)  # type of non-terminals
T = TypeVar("T", bound=Hashable)  # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants


@dataclass
class EAState(Generic[T]):
    """Snapshot of one generation in an evolutionary run.
    
    Attributes:
        generation: The generation number (starts at 0).
        population: The current population of individuals (Tree objects).
        fitness: Mapping from individuals to their fitness values.
        offspring: The individuals created in this generation through variation.
        ages: Mapping from individuals to their age (generations alive).
    """
    generation: int
    population: list[Tree[T]]
    fitness: dict[Tree[T], Fitness]
    offspring: list[Tree[T]]
    ages: dict[Tree[T], int]


class Evolutionary(ABC, Generic[NT, T, G]):
    """Abstract base class for component-oriented evolutionary algorithms.
    
    This class defines the interface for evolutionary algorithms composed from reusable components.
    Subclasses implement specific evolutionary strategies by defining the evolutionary_stream method.
    
    Type Parameters:
        NT: Type of non-terminals in the grammar/search space
        T: Type of terminals in the grammar/search space
        G: Type of constants/ground symbols
    """

    def __init__(self, solution_space: SolutionSpace[NT, T, G], start: NT,
                 termination_condition: Callable[[EAState[T]], bool],
                 initialization: Initialization[NT, T, G],
                 mutation: Mutation[NT, T, G],
                 recombination: Recombination[NT, T, G],
                 parent_selection: Selection[NT, T, G],
                 survivor_selection: Selection[NT, T, G],
                 fitness_comparator: FitnessComparator = ScalarFitnessComparator(),
                 rng: random.Random | None = None,
                 ):
        """Initialize the evolutionary algorithm with the given components.
        
        Args:
            solution_space: Defines the search space and constraint satisfaction.
            start: The start non-terminal for generating new individuals.
            termination_condition: A function that returns True when the EA should stop.
            initialization: Component for creating initial populations.
            mutation: Component for applying mutations to individuals.
            recombination: Component for recombining individuals.
            parent_selection: Component for selecting parents for variation.
            survivor_selection: Component for selecting survivors for the next generation.
            fitness_comparator: Component for comparing fitness values (mono- or multi-objective).
            rng: Optional random number generator for reproducibility.
        """
        self.solution_space = solution_space
        self.start = start
        self.termination_condition = termination_condition
        self.fitness_comparator = fitness_comparator
        self.rng = rng if rng is not None else random.Random()
        self.initialization = initialization
        self.mutation = mutation
        self.recombination = recombination
        self.parent_selection = parent_selection
        self.survivor_selection = survivor_selection

    @abstractmethod
    def evolutionary_stream(self, fitness_function: Callable[[Tree[T]], Fitness], population_size: int,
                            mutation_rate: float, recombination_rate: float) -> Iterable[EAState[T]]:
        """Yield successive EA states until the termination condition is met.
        
        Args:
            fitness_function: Function mapping individuals to fitness values.
            population_size: Target population size for each generation.
            mutation_rate: Probability of applying mutation during variation [0, 1].
            recombination_rate: Probability of applying recombination during variation [0, 1].
                               mutation_rate + recombination_rate should be <= 1.
        
        Yields:
            EAState: Snapshots of each generation until termination_condition returns True.
        """
        pass

    def evolutionary_last_generation(self, fitness_function: Callable[[Tree[T]], Fitness], population_size: int,
                                     mutation_rate: float, recombination_rate: float, verbose: bool = False) -> list[Tree[T]]:
        """Return the final generation, sorted by fitness (best first).
        
        Args:
            fitness_function: Function to evaluate individuals.
            population_size: Population size for the evolutionary run.
            mutation_rate: Mutation probability during variation.
            recombination_rate: Recombination probability during variation.
            verbose: Print generation numbers during the run if True (default: False).
        
        Returns:
            A list of individuals from the final generation, sorted by fitness (best first).
        """
        last_state: EAState[T] | None = None
        for state in self.evolutionary_stream(fitness_function, population_size, mutation_rate, recombination_rate):
            last_state = state
            if verbose:
                print(f"Generation {state.generation}")
        if last_state is None:
            return []
        return sorted(
            last_state.population,
            key=lambda tree: self.fitness_comparator.sort_key(last_state.fitness[tree]),
            reverse=True,
        )

    def evolutionary_best(self, fitness_function: Callable[[Tree[T]], Fitness], population_size: int,
                          mutation_rate: float, recombination_rate: float, verbose: bool = False) -> Tree[T] | None:
        """Return the best individual from the final generation, if any.
        
        Args:
            fitness_function: Function to evaluate individuals.
            population_size: Population size for the evolutionary run.
            mutation_rate: Mutation probability during variation.
            recombination_rate: Recombination probability during variation.
            verbose: Print generation numbers during the run if True (default: False).
        
        Returns:
            The best individual from the final generation, or None if no individuals were generated.
        """
        last_generation = self.evolutionary_last_generation(
            fitness_function, population_size, mutation_rate, recombination_rate, verbose
        )
        return last_generation[0] if last_generation else None

    def evolutionary_search(self, fitness_function: Callable[[Tree[T]], Fitness], population_size: int,
                            mutation_rate: float, recombination_rate: float, verbose: bool = False) -> Iterable[Tree[T]]:
        """Backward-compatible alias for returning the final generation.
        
        Args:
            fitness_function: Function to evaluate individuals.
            population_size: Population size for the evolutionary run.
            mutation_rate: Mutation probability during variation.
            recombination_rate: Recombination probability during variation.
            verbose: Print generation numbers during the run if True (default: False).
        
        Returns:
            An iterable of individuals from the final generation, sorted by fitness (best first).
        """
        return self.evolutionary_last_generation(
            fitness_function, population_size, mutation_rate, recombination_rate, verbose
        )


class SimpleGeneticProgramming(Evolutionary[NT, T, G], Generic[NT, T, G]):
    """A straightforward genetic programming implementation using component-based operators.
    
    This implemention applies one of three operations per individual proportional to their rates:
    - Mutation: Transform an individual using the mutation operator
    - Recombination: Create offspring by combining two parent individuals
    - Survival: Select an unchanged individual
    
    Features:
    - Elitism: Preserves the best individuals across generations
    - Fitness caching: Avoids recomputing fitness for identical individuals
    - Robustness: Retries variation operators multiple times if they produce invalid individuals
    - Age tracking: Maintains individual age for diversity-aware selection
    
    Type Parameters:
        NT: Type of non-terminals in the grammar/search space
        T: Type of terminals in the grammar/search space
        G: Type of constants/ground symbols
    """

    def __init__(self, solution_space: SolutionSpace[NT, T, G], start: NT,
                 termination_condition: Callable[[EAState[T]], bool],
                 initialization: Initialization[NT, T, G],
                 mutation: Mutation[NT, T, G],
                 recombination: Recombination[NT, T, G],
                 parent_selection: Selection[NT, T, G],
                 survivor_selection: Selection[NT, T, G],
                 fitness_comparator: FitnessComparator = ScalarFitnessComparator(),
                 rng: random.Random | None = None,
                 elite_count: int = 1,
                 max_attempts_factor: int = 5,
                 min_attempts: int = 10,):
        """Initialize a Simple Genetic Programming search strategy.
        
        Args:
            solution_space: Defines the search space and constraint satisfaction.
            start: The start non-terminal for generating new individuals.
            termination_condition: A function that returns True when the EA should stop.
            initialization: Component for creating initial populations.
            mutation: Component for applying mutations to individuals.
            recombination: Component for recombining individuals.
            parent_selection: Component for selecting parents for variation.
            survivor_selection: Component for selecting survivors for the next generation.
            fitness_comparator: Component for comparing fitness values (mono- or multi-objective).
            rng: Optional random number generator for reproducibility.
            elite_count: Number of best individuals to preserve unchanged each generation (default: 1).
            max_attempts_factor: Maximum attempts = population_size * this factor (default: 5).
            min_attempts: Minimum number of variation attempts per generation (default: 10).
        """
        super().__init__(
            solution_space=solution_space,
            start=start,
            termination_condition=termination_condition,
            initialization=initialization,
            mutation=mutation,
            recombination=recombination,
            parent_selection=parent_selection,
            survivor_selection=survivor_selection,
            fitness_comparator=fitness_comparator,
            rng=rng,
        )
        self.elite_count = max(0, elite_count)
        self.max_attempts_factor = max(1, max_attempts_factor)
        self.min_attempts = max(0, min_attempts)

    def evolutionary_stream(self, fitness_function: Callable[[Tree[T]], Fitness], population_size: int,
                            mutation_rate: float, recombination_rate: float) -> Iterable[EAState[T]]:
        """Yield EA states while optimizing the provided fitness function.
        
        The algorithm proceeds as follows:
        1. Initialize population using the initialization component
        2. Evaluate fitness for all individuals
        3. In each generation:
           a. Select elite individuals to preserve unchanged
           b. Create offspring through variation (mutation, recombination, or survival)
           c. Evaluate offspring fitness
           d. Select survivors for the next generation
           e. Yield the new state
        
        Args:
            fitness_function: Function mapping individuals to fitness values.
            population_size: Target population size for each generation.
            mutation_rate: Probability of applying mutation [0, 1].
            recombination_rate: Probability of applying recombination [0, 1].
            
        Yields:
            EAState: Snapshots of each generation until termination_condition returns True.
            
        Raises:
            ValueError: If mutation_rate + recombination_rate > 1.
        """
        if mutation_rate + recombination_rate > 1:
            raise ValueError("mutation_rate + recombination_rate > 1 not supported")

        # Initialize the population
        population: list[Tree[T]] = list(self.initialization.initialize_population(population_size))

        # Cache fitness values across generations to avoid recomputation for unchanged individuals.
        fitness_cache: dict[Tree[T], Fitness] = {}

        def get_fitness(tree: Tree[T]) -> Fitness:
            """Retrieve fitness from cache or compute and cache it."""
            if tree not in fitness_cache:
                fitness_cache[tree] = fitness_function(tree)
            return fitness_cache[tree]

        population_fitness: dict[Tree[T], Fitness] = {tree: get_fitness(tree) for tree in population}
        population_ages: dict[Tree[T], int] = {tree: 0 for tree in population}

        generation: int = 0
        state = EAState(
            generation=generation,
            population=population,
            fitness=population_fitness,
            offspring=[],
            ages=population_ages,
        )
        while True:
            yield state
            if self.termination_condition(state):
                return

            # Build a mating pool; when exhausted, re-sample from the selection operator.
            mating_pool = self.parent_selection.select(
                population_fitness,
                population_size,
                self.fitness_comparator,
                ages=population_ages,
            )
            iterator = iter(mating_pool)

            def next_parent() -> Tree[T] | None:
                """Get next parent from current pool or resample if pool is exhausted."""
                nonlocal iterator, mating_pool
                try:
                    return next(iterator)
                except StopIteration:
                    mating_pool = self.parent_selection.select(
                        population_fitness,
                        population_size,
                        self.fitness_comparator,
                        ages=population_ages,
                    )
                    iterator = iter(mating_pool)
                    try:
                        return next(iterator)
                    except StopIteration:
                        return None

            # Extract elite individuals
            elite_count = min(self.elite_count, population_size)
            elites = sorted(
                population,
                key=lambda tree: self.fitness_comparator.sort_key(population_fitness[tree]),
                reverse=True,
            )[:elite_count]

            # Create offspring through variation operators
            target_offspring = max(population_size - elite_count, 0)
            offspring: list[Tree[T]] = []
            attempts = 0
            max_attempts = max(population_size * self.max_attempts_factor, self.min_attempts)

            while len(offspring) < target_offspring and attempts < max_attempts:
                # Select a variation operator based on rates
                variation_operator = self.rng.choices(
                    ["mutate", "crossover", "survive"],
                    weights=[mutation_rate, recombination_rate, 1 - mutation_rate - recombination_rate],
                    k=1,
                )[0]
                
                # Apply the selected operator
                if variation_operator == "mutate":
                    parent = next_parent()
                    if parent is None:
                        break
                    candidates = list(self.mutation.mutate(parent))
                elif variation_operator == "crossover":
                    parent1 = next_parent()
                    parent2 = next_parent()
                    if parent1 is None or parent2 is None:
                        break
                    candidates = list(self.recombination.recombine(parent1, parent2))
                else:
                    survivor = next_parent()
                    if survivor is None:
                        break
                    candidates = [survivor]

                # Add valid candidates to offspring
                if not candidates:
                    attempts += 1
                    continue

                remaining = target_offspring - len(offspring)
                offspring.extend(candidates[:remaining])

            # Fill remaining slots with survivors if operators produced too few candidates.
            while len(offspring) < target_offspring:
                survivor = next_parent()
                if survivor is None:
                    break
                offspring.append(survivor)

            # Prepare candidates for survivor selection (elites + new offspring)
            previous_ages = population_ages
            candidate_fitness: dict[Tree[T], Fitness] = {tree: get_fitness(tree) for tree in offspring}
            candidate_ages: dict[Tree[T], int] = {}
            
            # Age tracking: new individuals start at age 0, retained individuals age by 1
            for tree in offspring:
                candidate_ages[tree] = previous_ages.get(tree, -1) + 1
            for elite in elites:
                if elite in population_fitness:
                    candidate_fitness.setdefault(elite, population_fitness[elite])
                    candidate_ages.setdefault(elite, previous_ages.get(elite, -1) + 1)

            # Select survivors for next generation
            selected = list(
                self.survivor_selection.select(
                    candidate_fitness,
                    population_size - elite_count,
                    self.fitness_comparator,
                    population_fitness,
                    ages=candidate_ages,
                    previous_generation_ages=previous_ages,
                )
            )
            
            # Combine elites with selected survivors
            population = elites + selected
            population_fitness = {
                tree: candidate_fitness[tree] if tree in candidate_fitness else population_fitness[tree] for tree in population
            }
            population_ages = {tree: candidate_ages.get(tree, previous_ages.get(tree, 0) + 1) for tree in population}
            generation += 1
            
            # Create the next state
            state = EAState(
                generation=generation,
                population=population,
                fitness=population_fitness,
                offspring=offspring,
                ages=population_ages,
            )
