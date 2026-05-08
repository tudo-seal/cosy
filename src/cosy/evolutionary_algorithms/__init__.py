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
from cosy.evolutionary_algorithms.evolutionary import (
    EAState,
    Evolutionary,
    SimpleGeneticProgramming,
)

# Fitness
from cosy.evolutionary_algorithms.fitness import (
    Fitness,
    FitnessComparator,
    ParetoFitnessComparator,
    ScalarFitnessComparator,
)

# Initialization
from cosy.evolutionary_algorithms.initialisation import (
    Initialization,
    RandomLimitedDepthFirstInitialization,
)

# Mutation
from cosy.evolutionary_algorithms.mutation import (
    Mutation,
    ResolutionMutation,
)

# Recombination
from cosy.evolutionary_algorithms.recombination import (
    Crossover,
    Recombination,
)

# Selection
from cosy.evolutionary_algorithms.selection import (
    AgeBasedReplacement,
    FitnessBasedReplacement,
    FitnessProportionalSelection,
    RankBasedSelection,
    Selection,
    TournamentSelection,
)

__all__ = [
    "AgeBasedReplacement",
    "Crossover",
    "EAState",
    # Core algorithm classes
    "Evolutionary",
    # Fitness
    "Fitness",
    "FitnessBasedReplacement",
    "FitnessComparator",
    "FitnessProportionalSelection",
    # Initialization
    "Initialization",
    # Mutation
    "Mutation",
    "ParetoFitnessComparator",
    "RandomLimitedDepthFirstInitialization",
    "RankBasedSelection",
    # Recombination
    "Recombination",
    "ResolutionMutation",
    "ScalarFitnessComparator",
    # Selection
    "Selection",
    "SimpleGeneticProgramming",
    "TournamentSelection",
]
