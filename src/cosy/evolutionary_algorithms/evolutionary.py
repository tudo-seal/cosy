"""Component-oriented framework for evolutionary algorithms.

This module provides the core infrastructure for building customizable evolutionary algorithms
by composing independent components (initialization, mutation, recombination, selection).
The framework uses SolutionSpace to define a constraint-based search space and Tree to represent
individuals in the population.
"""

import inspect
import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, cast, get_origin, get_type_hints

from cosy.core.solution_space import SolutionSpace
from cosy.core.tree import Tree
from cosy.evolutionary_algorithms.fitness import Fitness, FitnessComparator, ScalarFitnessComparator
from cosy.evolutionary_algorithms.initialisation import Initializer
from cosy.evolutionary_algorithms.mutation import Mutation
from cosy.evolutionary_algorithms.recombination import Recombination
from cosy.evolutionary_algorithms.rng.factory import RNGFactory
from cosy.evolutionary_algorithms.selection import Selection
from cosy.search.queries import generator_query

NT = TypeVar("NT", bound=Hashable)  # type of non-terminals
T = TypeVar("T", bound=Hashable)  # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants

FitnessFunctionMode = Literal["auto", "single", "batch"]
FitnessFunctionSingle = Callable[[Tree[T]], Fitness]
FitnessFunctionBatch = Callable[[list[Tree[T]]], Mapping[Tree[T], Fitness]]
FitnessFunction = FitnessFunctionSingle | FitnessFunctionBatch


@dataclass
class EAState(Generic[T]):
    """Snapshot of one generation in an evolutionary run.

    Attributes:
        generation (int): The generation number (starts at 0).
        population (list[Tree[T]]): The current population of individuals (Tree objects).
        fitness (dict[Tree[T], Fitness]): Mapping from individuals to their fitness values.
        offspring (list[Tree[T]]): The individuals created in this generation through variation.
        ages (dict[Tree[T], int]): Mapping from individuals to their age (generations alive).
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
    """

    def __init__(
        self,
        solution_space: SolutionSpace[NT, T, G],
        start: NT,
        termination_condition: Callable[[EAState[T]], bool],
        initialization: Initializer[NT, T, G],
        mutation: Mutation[NT, T, G],
        recombination: Recombination[NT, T, G],
        parent_selection: Selection[NT, T, G],
        survivor_selection: Selection[NT, T, G],
        fitness_comparator: FitnessComparator = ScalarFitnessComparator(),
        rng: random.Random | None = None,
    ):
        """Initialize the evolutionary algorithm with the given components.

        Args:
            solution_space (SolutionSpace[NT, T, G]): Defines the search space and constraint satisfaction.
            start (NT): The start non-terminal for generating new individuals.
            termination_condition (Callable[[EAState[T]], bool]): A function that returns True when the EA should stop.
            initialization (Initializer[NT, T, G]): Component for creating initial populations.
            mutation (Mutation[NT, T, G]): Component for applying mutations to individuals.
            recombination (Recombination[NT, T, G]): Component for recombining individuals.
            parent_selection (Selection[NT, T, G]): Component for selecting parents for variation.
            survivor_selection (Selection[NT, T, G]): Component for selecting survivors for the next generation.
            fitness_comparator (FitnessComparator): Component for comparing fitness values (mono- or multi-objective). (Default value = ScalarFitnessComparator())
            rng (random.Random | None): Optional random number generator for reproducibility. (Default value = None)
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

    @staticmethod
    def _deduplicate_population(population: Iterable[Tree[T]]) -> list[Tree[T]]:
        """_summary_.

        Args:
            population (Iterable[Tree[T]]): _description_

        Returns:
            list[Tree[T]]: _description_
        """
        return list(dict.fromkeys(population))

    @staticmethod
    def _looks_like_batch_fitness_function(fitness_function: Callable[..., Any]) -> bool:
        """_summary_.

        Args:
            fitness_function (Callable[..., Any]): _description_

        Returns:
            bool: _description_
        """
        try:
            signature = inspect.signature(fitness_function)
            type_hints = get_type_hints(fitness_function)
        except (TypeError, ValueError):
            return False

        parameters = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]
        if len(parameters) != 1:
            return False

        parameter_name = parameters[0].name
        annotation = type_hints.get(parameter_name, parameters[0].annotation)
        if annotation is inspect.Signature.empty:
            return False

        origin = get_origin(annotation)
        return origin in {list, Iterable, tuple, set} or annotation in {list, Iterable, tuple, set}

    def _evaluate_population_fitness(
        self,
        fitness_function: Callable[..., Any],
        population: Iterable[Tree[T]],
        fitness_cache: dict[Tree[T], Fitness],
        *,
        fitness_function_mode: FitnessFunctionMode = "auto",
    ) -> dict[Tree[T], Fitness]:
        """_summary_.

        Args:
            fitness_function (Callable[..., Any]): _description_
            population (Iterable[Tree[T]]): _description_
            fitness_cache (dict[Tree[T], Fitness]): _description_
            fitness_function_mode (FitnessFunctionMode): _description_ (Default value = 'auto')

        Returns:
            dict[Tree[T], Fitness]: _description_

        Raises:
            ValueError: _description_
            TypeError: _description_
            ValueError: _description_
        """
        unique_population = self._deduplicate_population(population)
        missing = [tree for tree in unique_population if tree not in fitness_cache]

        if missing:
            if fitness_function_mode == "single":
                for tree in missing:
                    fitness_cache[tree] = cast("FitnessFunctionSingle", fitness_function)(tree)
            elif fitness_function_mode == "batch":
                evaluated = cast("FitnessFunctionBatch", fitness_function)(missing)
                missing_fitness = dict(evaluated)
                missing_keys = [tree for tree in missing if tree not in missing_fitness]
                if missing_keys:
                    msg = "Batch fitness function must return a fitness value for every requested tree"
                    raise ValueError(msg)
                fitness_cache.update({tree: missing_fitness[tree] for tree in missing})
            elif self._looks_like_batch_fitness_function(fitness_function):
                evaluated = cast("Callable[[list[Tree[T]]], Any]", fitness_function)(missing)
                if not isinstance(evaluated, Mapping):
                    msg = "Batch fitness function must return a mapping from tree to fitness"
                    raise TypeError(msg)
                missing_fitness = dict(evaluated)
                missing_keys = [tree for tree in missing if tree not in missing_fitness]
                if missing_keys:
                    msg = "Batch fitness function must return a fitness value for every requested tree"
                    raise ValueError(msg)
                fitness_cache.update({tree: missing_fitness[tree] for tree in missing})
            else:
                for tree in missing:
                    fitness_cache[tree] = cast("FitnessFunctionSingle", fitness_function)(tree)

        return {tree: fitness_cache[tree] for tree in unique_population}

    @abstractmethod
    def evolutionary_stream(
        self,
        fitness_function: Callable[..., Any],
        population_size: int,
        mutation_rate: float,
        recombination_rate: float,
        fitness_function_mode: FitnessFunctionMode = "auto",
    ) -> Iterable[EAState[T]]:
        """Yield successive EA states until the termination condition is met.

        Args:
            fitness_function (Callable[..., Any]): Function mapping individuals to fitness values. Also supports a batch
                variant that accepts a list of trees and returns a mapping from tree to fitness.
            population_size (int): Target population size for each generation.
            mutation_rate (float): Probability of applying mutation during variation [0, 1].
            recombination_rate (float): Probability of applying recombination during variation [0, 1].
                mutation_rate + recombination_rate should be <= 1.
            fitness_function_mode (FitnessFunctionMode): How to interpret fitness_function. "single" expects a tree at
                a time, "batch" expects a list of trees, and "auto" tries batch first and falls
                back to single-tree evaluation. (Default value = 'auto')

        Yields:
            EAState[T]: Snapshots of each generation until termination_condition returns True.
        """

    def evolutionary_last_generation(
        self,
        fitness_function: Callable[..., Any],
        population_size: int,
        mutation_rate: float,
        recombination_rate: float,
        verbose: bool = False,
        fitness_function_mode: FitnessFunctionMode = "auto",
    ) -> list[Tree[T]]:
        """Return the final generation, sorted by fitness (best first).

        Args:
            fitness_function (Callable[..., Any]): Function to evaluate individuals. Also supports a batch variant.
            population_size (int): Population size for the evolutionary run.
            mutation_rate (float): Mutation probability during variation.
            recombination_rate (float): Recombination probability during variation.
            verbose (bool): Print generation numbers during the run if True (default: False).
            fitness_function_mode (FitnessFunctionMode): How to interpret fitness_function (see evolutionary_stream). (Default value = 'auto')

        Returns:
            list[Tree[T]]: A list of individuals from the final generation, sorted by fitness (best first).
        """
        last_state: EAState[T] | None = None
        for state in self.evolutionary_stream(
            fitness_function,
            population_size,
            mutation_rate,
            recombination_rate,
            fitness_function_mode,
        ):
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

    def evolutionary_best(
        self,
        fitness_function: Callable[..., Any],
        population_size: int,
        mutation_rate: float,
        recombination_rate: float,
        verbose: bool = False,
        fitness_function_mode: FitnessFunctionMode = "auto",
    ) -> Tree[T] | None:
        """Return the best individual from the final generation, if any.

        Args:
            fitness_function (Callable[..., Any]): Function to evaluate individuals. Also supports a batch variant.
            population_size (int): Population size for the evolutionary run.
            mutation_rate (float): Mutation probability during variation.
            recombination_rate (float): Recombination probability during variation.
            verbose (bool): Print generation numbers during the run if True (default: False).
            fitness_function_mode (FitnessFunctionMode): How to interpret fitness_function (see evolutionary_stream). (Default value = 'auto')

        Returns:
            Tree[T] | None: The best individual from the final generation, or None if no individuals were generated.
        """
        last_generation = self.evolutionary_last_generation(
            fitness_function,
            population_size,
            mutation_rate,
            recombination_rate,
            verbose,
            fitness_function_mode,
        )
        return last_generation[0] if last_generation else None

    def evolutionary_search(
        self,
        fitness_function: Callable[..., Any],
        population_size: int,
        mutation_rate: float,
        recombination_rate: float,
        verbose: bool = False,
        fitness_function_mode: FitnessFunctionMode = "auto",
    ) -> Iterable[Tree[T]]:
        """Backward-compatible alias for returning the final generation.

        Args:
            fitness_function (Callable[..., Any]): Function to evaluate individuals. Also supports a batch variant.
            population_size (int): Population size for the evolutionary run.
            mutation_rate (float): Mutation probability during variation.
            recombination_rate (float): Recombination probability during variation.
            verbose (bool): Print generation numbers during the run if True (default: False).
            fitness_function_mode (FitnessFunctionMode): How to interpret fitness_function (see evolutionary_stream). (Default value = 'auto')

        Returns:
            Iterable[Tree[T]]: An iterable of individuals from the final generation, sorted by fitness (best first).
        """
        return self.evolutionary_last_generation(
            fitness_function,
            population_size,
            mutation_rate,
            recombination_rate,
            verbose,
            fitness_function_mode,
        )


class SimpleGeneticProgramming(Evolutionary[NT, T, G], Generic[NT, T, G]):
    """A straightforward genetic programming implementation using component-based operators.

    This implementation applies one of three operations per individual proportional to their rates:
    - Mutation: Transform an individual using the mutation operator
    - Recombination: Create offspring by combining two parent individuals
    - Survival: Select an unchanged individual

    Features:
    - Elitism: Preserves the best individuals across generations
    - Fitness caching: Avoids recomputing fitness for identical individuals
    - Robustness: Retries variation operators multiple times if they produce invalid individuals
    - Age tracking: Maintains individual age for diversity-aware selection
    """

    def __init__(
        self,
        solution_space: SolutionSpace[NT, T, G],
        start: NT,
        termination_condition: Callable[[EAState[T]], bool],
        initialization: Initializer[NT, T, G],
        mutation: Mutation[NT, T, G],
        recombination: Recombination[NT, T, G],
        parent_selection: Selection[NT, T, G],
        survivor_selection: Selection[NT, T, G],
        fitness_comparator: FitnessComparator = ScalarFitnessComparator(),
        rng: random.Random | None = None,
        elite_count: int = 1,
        max_attempts_factor: int = 5,
        min_attempts: int = 10,
        rng_factory: RNGFactory | None = None,
        distribute_rngs: bool = True,
    ):
        """Initialize a Simple Genetic Programming search strategy.

        Args:
            solution_space (SolutionSpace[NT, T, G]): Defines the search space and constraint satisfaction.
            start (NT): The start non-terminal for generating new individuals.
            termination_condition (Callable[[EAState[T]], bool]): A function that returns True when the EA should stop.
            initialization (Initializer[NT, T, G]): Component for creating initial populations.
            mutation (Mutation[NT, T, G]): Component for applying mutations to individuals.
            recombination (Recombination[NT, T, G]): Component for recombining individuals.
            parent_selection (Selection[NT, T, G]): Component for selecting parents for variation.
            survivor_selection (Selection[NT, T, G]): Component for selecting survivors for the next generation.
            fitness_comparator (FitnessComparator): Component for comparing fitness values (mono- or multi-objective). (Default value = ScalarFitnessComparator())
            rng (random.Random | None): Optional random number generator for reproducibility (default: new unseeded Random()).
            elite_count (int): Number of best individuals to preserve unchanged each generation (default: 1).
            max_attempts_factor (int): Maximum attempts = population_size * this factor (default: 5).
            min_attempts (int): Minimum number of variation attempts per generation (default: 10).
            rng_factory (RNGFactory | None): Optional RNGFactory for producing independent child RNGs for components.
                If None, one is created from self.rng automatically. Each child RNG is
                deterministically seeded, ensuring reproducibility and independence. (Default value = None)
            distribute_rngs (bool): If True (default), attempt to assign child RNGs from rng_factory to
                components that expose an 'rng' attribute. Gracefully skips components
                without 'rng' attribute. Set to False to disable distribution.
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
        # RNG factory and optional distribution to components
        if rng_factory is None:
            self.rng_factory = RNGFactory.from_random(self.rng)
        else:
            self.rng_factory = rng_factory

        if distribute_rngs:
            # Create and assign child RNGs for components that expose a 'rng' attribute.
            # Child RNGs are independent (different sequences) yet reproducible (same master seed
            # → same child RNGs) via deterministic seeding.
            #
            # This reaches a component's own generator only. A component that draws through a
            # sampler carries the sampler's generator separately, and that one is not reseeded
            # here, so seeding such a component at construction time is what makes a run
            # reproducible.
            init_rng = self.rng_factory.child()
            mutation_rng = self.rng_factory.child()
            recomb_rng = self.rng_factory.child()
            parent_sel_rng = self.rng_factory.child()
            survivor_sel_rng = self.rng_factory.child()

            if hasattr(self.initialization, "rng"):
                self.initialization.rng = init_rng
            if hasattr(self.mutation, "rng"):
                self.mutation.rng = mutation_rng
            if hasattr(self.recombination, "rng"):
                self.recombination.rng = recomb_rng
            if hasattr(self.parent_selection, "rng"):
                self.parent_selection.rng = parent_sel_rng
            if hasattr(self.survivor_selection, "rng"):
                self.survivor_selection.rng = survivor_sel_rng

    def evolutionary_stream(
        self,
        fitness_function: Callable[..., Any],
        population_size: int,
        mutation_rate: float,
        recombination_rate: float,
        fitness_function_mode: FitnessFunctionMode = "auto",
    ) -> Iterable[EAState[T]]:
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
            fitness_function (Callable[..., Any]): Function mapping individuals to fitness values. Also supports a batch
                variant that accepts a list of trees and returns a mapping from tree to fitness.
            population_size (int): Target population size for each generation.
            mutation_rate (float): Probability of applying mutation [0, 1].
            recombination_rate (float): Probability of applying recombination [0, 1].
            fitness_function_mode (FitnessFunctionMode): How to interpret fitness_function. "single" expects a tree at
                a time, "batch" expects a list of trees, and "auto" tries batch first and falls
                back to single-tree evaluation. (Default value = 'auto')

        Yields:
            EAState[T]: Snapshots of each generation until termination_condition returns True.

        Raises:
            ValueError: If mutation_rate + recombination_rate > 1.
        """
        if mutation_rate + recombination_rate > 1:
            msg = "mutation_rate + recombination_rate > 1 not supported"
            raise ValueError(msg)

        # Initialize the population
        query = generator_query(self.solution_space, self.start)
        population: list[Tree[T]] = list(self.initialization.initialize(query, population_size))

        # Cache fitness values across generations to avoid recomputation for unchanged individuals.
        fitness_cache: dict[Tree[T], Fitness] = {}

        population_fitness = self._evaluate_population_fitness(
            fitness_function,
            population,
            fitness_cache,
            fitness_function_mode=fitness_function_mode,
        )
        population_ages: dict[Tree[T], int] = dict.fromkeys(population, 0)

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
                """Get next parent from current pool or resample if pool is exhausted.

                Returns:
                    Tree[T] | None: _description_
                """
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
                    mutant = self.mutation.mutate(query, parent)
                    candidates = [] if mutant is None else [mutant]
                elif variation_operator == "crossover":
                    parent1 = next_parent()
                    parent2 = next_parent()
                    if parent1 is None or parent2 is None:
                        break
                    candidates = self.recombination.recombine(query, parent1, parent2)
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
            candidate_fitness = self._evaluate_population_fitness(
                fitness_function,
                offspring,
                fitness_cache,
                fitness_function_mode=fitness_function_mode,
            )
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
                tree: candidate_fitness[tree] if tree in candidate_fitness else population_fitness[tree]
                for tree in population
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
