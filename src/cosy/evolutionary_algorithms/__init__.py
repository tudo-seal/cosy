"""
Component-oriented evolutionary algorithms framework.

This package provides a flexible, modular framework for building evolutionary algorithms by
composing independent, reusable components. Instead of implementing specific algorithms from scratch,
users can combine components to create custom algorithms tailored to their specific problems.

Core Components:
    - Initialization: Create initial populations
    - Mutation: Apply local modifications to individuals
    - Recombination: Combine genetic material from multiple parents
    - Selection: Choose individuals for reproduction or survival
    - Fitness: Evaluate and compare solution quality (single and multi-objective)

Main Algorithm:
    - SimpleGeneticProgramming: A straightforward GP implementation using the above components

Key Concepts:
    - SolutionSpace: Defines the search space and constraint satisfaction
    - Tree: Represents individual solutions (program trees)
    - EAState: A snapshot of one generation in the evolutionary run
    - Fitness: Can be a scalar value or multi-dimensional vector
    - FitnessComparator: Determines how fitness values are compared
"""

# Core classes
from src.cosy.evolutionary_algorithms.evolutionary import (
    Evolutionary,
    SimpleGeneticProgramming,
    EAState,
)

# Initialization
from src.cosy.evolutionary_algorithms.initialisation import (
    Initialization,
    RandomLimitedDepthFirstInitialization,
)

# Mutation
from src.cosy.evolutionary_algorithms.mutation import (
    Mutation,
    ResolutionMutation,
)

# Recombination
from src.cosy.evolutionary_algorithms.recombination import (
    Recombination,
    Crossover,
)

# Selection
from src.cosy.evolutionary_algorithms.selection import (
    Selection,
    TournamentSelection,
    FitnessProportionalSelection,
    RankBasedSelection,
    FitnessBasedReplacement,
    AgeBasedReplacement,
)

# Fitness
from src.cosy.evolutionary_algorithms.fitness import (
    Fitness,
    FitnessComparator,
    ScalarFitnessComparator,
    ParetoFitnessComparator,
)

__all__ = [
    # Core algorithm classes
    "Evolutionary",
    "SimpleGeneticProgramming",
    "EAState",
    # Initialization
    "Initialization",
    "RandomLimitedDepthFirstInitialization",
    # Mutation
    "Mutation",
    "ResolutionMutation",
    # Recombination
    "Recombination",
    "Crossover",
    # Selection
    "Selection",
    "TournamentSelection",
    "FitnessProportionalSelection",
    "RankBasedSelection",
    "FitnessBasedReplacement",
    "AgeBasedReplacement",
    # Fitness
    "Fitness",
    "FitnessComparator",
    "ScalarFitnessComparator",
    "ParetoFitnessComparator",
]
