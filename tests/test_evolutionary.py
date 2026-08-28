"""_summary_."""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

from cosy.core.tree import Tree
from cosy.evolutionary_algorithms.evolutionary import EAState, SimpleGeneticProgramming
from cosy.evolutionary_algorithms.fitness import ScalarFitnessComparator
from examples.example_symbolic_regression import (
    SymbolicRegression,
    run_symbolic_regression,
)


class StaticInitialization:
    """_summary_.

    Attributes:
        population (_type_): _description_
    """

    def __init__(self, population: list[Tree[str]]) -> None:
        """_summary_.

        Args:
            population (list[Tree[str]]): _description_
        """
        self.population = list(population)

    def initialize(self, query: Any, population_size: int):
        """_summary_.

        Args:
            query (Any): _description_
            population_size (int): _description_

        Returns:
            _type_: _description_
        """
        return self.population[:population_size]


class EmptyInitialization:
    """_summary_."""

    def initialize(self, query: Any, population_size: int):
        """_summary_.

        Args:
            query (Any): _description_
            population_size (int): _description_

        Returns:
            _type_: _description_
        """
        return []


class RecordingMutation:
    """_summary_.

    Attributes:
        calls (list[Tree[str]]): _description_
    """

    def __init__(self) -> None:
        """_summary_."""
        self.calls: list[Tree[str]] = []

    def mutate(self, query: Any, tree: Tree[str]):
        """_summary_.

        Args:
            query (Any): _description_
            tree (Tree[str]): _description_

        Returns:
            _type_: _description_
        """
        self.calls.append(tree)


class EmptyRecombination:
    """_summary_."""

    def recombine(self, query: Any, primary: Tree[str], secondary: Tree[str]):
        """_summary_.

        Args:
            query (Any): _description_
            primary (Tree[str]): _description_
            secondary (Tree[str]): _description_

        Returns:
            _type_: _description_
        """
        return []


class RecordingSelection:
    """_summary_.

    Attributes:
        response (list[Tree[str]] | None): _description_
        calls (list[dict[str, object]]): _description_
    """

    def __init__(self, response: list[Tree[str]] | None = None) -> None:
        """_summary_.

        Args:
            response (list[Tree[str]] | None): _description_ (Default value = None)
        """
        self.response = response
        self.calls: list[dict[str, object]] = []

    def select(
        self,
        population_fitness,
        population_size,
        comparator,
        previous_generation_fitness=None,
        ages=None,
        previous_generation_ages=None,
    ):
        """_summary_.

        Args:
            population_fitness (_type_): _description_
            population_size (_type_): _description_
            comparator (_type_): _description_
            previous_generation_fitness (_type_): _description_ (Default value = None)
            ages (_type_): _description_ (Default value = None)
            previous_generation_ages (_type_): _description_ (Default value = None)

        Returns:
            _type_: _description_
        """
        self.calls.append(
            {
                "population_fitness": dict(population_fitness),
                "population_size": population_size,
                "comparator": comparator,
                "previous_generation_fitness": None
                if previous_generation_fitness is None
                else dict(previous_generation_fitness),
                "ages": None if ages is None else dict(ages),
                "previous_generation_ages": None
                if previous_generation_ages is None
                else dict(previous_generation_ages),
            }
        )
        if self.response is None:
            return list(population_fitness.keys())[:population_size]
        return list(self.response)[:population_size]


def test_ea_state_is_a_plain_snapshot() -> None:
    """_summary_."""
    tree = Tree("leaf")
    state = EAState(generation=1, population=[tree], fitness={tree: 1.5}, offspring=[], ages={tree: 0})

    assert state == EAState(generation=1, population=[tree], fitness={tree: 1.5}, offspring=[], ages={tree: 0})


def test_simple_gp_rejects_invalid_operator_rates() -> None:
    """_summary_."""
    solution_space: Any = object()
    initialization: Any = StaticInitialization([])
    mutation: Any = RecordingMutation()
    recombination: Any = EmptyRecombination()
    parent_selection: Any = RecordingSelection()
    survivor_selection: Any = RecordingSelection()

    gp = SimpleGeneticProgramming(
        solution_space,
        "start",
        lambda _state: True,
        initialization,
        mutation,
        recombination,
        parent_selection,
        survivor_selection,
        ScalarFitnessComparator(),
        rng=random.Random(0),
    )

    with pytest.raises(ValueError, match=r"mutation_rate \+ recombination_rate > 1"):
        list(gp.evolutionary_stream(lambda tree: 0.0, 1, 0.8, 0.3))


def test_simple_gp_last_generation_and_best_use_fitness_order() -> None:
    """_summary_."""
    low = Tree("low")
    high = Tree("high")
    fitness_calls: list[Tree[str]] = []
    solution_space: Any = object()
    initialization: Any = StaticInitialization([low, high])
    mutation: Any = RecordingMutation()
    recombination: Any = EmptyRecombination()
    parent_selection: Any = RecordingSelection()
    survivor_selection: Any = RecordingSelection()

    def fitness(tree: Tree[str]) -> float:
        """_summary_.

        Args:
            tree (Tree[str]): _description_

        Returns:
            float: _description_
        """
        fitness_calls.append(tree)
        return 2.0 if tree.root == "low" else 1.0

    gp: Any = SimpleGeneticProgramming(
        solution_space,
        "start",
        lambda state: state.generation >= 0,
        initialization,
        mutation,
        recombination,
        parent_selection,
        survivor_selection,
        ScalarFitnessComparator(False),
        rng=random.Random(0),
    )

    last_generation = gp.evolutionary_last_generation(fitness, 2, 0.0, 0.0)
    assert last_generation == [high, low]
    assert fitness_calls == [low, high]

    fitness_calls.clear()
    assert gp.evolutionary_best(fitness, 2, 0.0, 0.0) == high
    assert fitness_calls == [low, high]


def test_simple_gp_supports_batch_fitness_functions() -> None:
    """_summary_."""
    low = Tree("low")
    high = Tree("high")
    solution_space: Any = object()
    initialization: Any = StaticInitialization([low, high])
    mutation: Any = RecordingMutation()
    recombination: Any = EmptyRecombination()
    parent_selection: Any = RecordingSelection()
    survivor_selection: Any = RecordingSelection()
    batch_calls: list[list[Tree[str]]] = []

    def fitness(trees: list[Tree[str]]) -> dict[Tree[str], float]:
        """_summary_.

        Args:
            trees (list[Tree[str]]): _description_

        Returns:
            dict[Tree[str], float]: _description_
        """
        batch_calls.append(list(trees))
        return {tree: 2.0 if tree.root == "low" else 1.0 for tree in trees}

    gp: Any = SimpleGeneticProgramming(
        solution_space,
        "start",
        lambda state: state.generation >= 0,
        initialization,
        mutation,
        recombination,
        parent_selection,
        survivor_selection,
        ScalarFitnessComparator(False),
        rng=random.Random(0),
    )

    last_generation = gp.evolutionary_last_generation(fitness, 2, 0.0, 0.0)

    assert last_generation == [high, low]
    assert batch_calls == [[low, high]]


def test_simple_gp_returns_none_for_an_empty_initial_population() -> None:
    """_summary_."""
    solution_space: Any = object()
    initialization: Any = EmptyInitialization()
    mutation: Any = RecordingMutation()
    recombination: Any = EmptyRecombination()
    parent_selection: Any = RecordingSelection()
    survivor_selection: Any = RecordingSelection()

    gp = SimpleGeneticProgramming(
        solution_space,
        "start",
        lambda state: state.generation >= 0,
        initialization,
        mutation,
        recombination,
        parent_selection,
        survivor_selection,
        ScalarFitnessComparator(),
        rng=random.Random(0),
    )

    assert gp.evolutionary_last_generation(lambda tree: 0.0, 3, 0.0, 0.0) == []
    assert gp.evolutionary_best(lambda tree: 0.0, 3, 0.0, 0.0) is None


def test_simple_gp_falls_back_to_survivors_and_tracks_ages() -> None:
    """_summary_."""
    best = Tree("best")
    other = Tree("other")
    fitness_calls: list[Tree[str]] = []
    solution_space: Any = object()

    def fitness(tree: Tree[str]) -> float:
        """_summary_.

        Args:
            tree (Tree[str]): _description_

        Returns:
            float: _description_
        """
        fitness_calls.append(tree)
        return 2.0 if tree.root == "best" else 1.0

    initialization: Any = StaticInitialization([best, other])
    mutation: Any = RecordingMutation()
    recombination: Any = EmptyRecombination()
    parent_selection: Any = RecordingSelection(response=[other, best])
    survivor_selection: Any = RecordingSelection()

    gp = SimpleGeneticProgramming(
        solution_space,
        "start",
        lambda state: state.generation >= 1,
        initialization,
        mutation,
        recombination,
        parent_selection,
        survivor_selection,
        ScalarFitnessComparator(),
        rng=random.Random(0),
        elite_count=1,
        max_attempts_factor=1,
        min_attempts=1,
    )

    states = list(gp.evolutionary_stream(fitness, 2, 1.0, 0.0))
    assert [state.generation for state in states] == [0, 1]

    initial_state, next_state = states
    assert initial_state.population == [best, other]
    assert mutation.calls
    assert mutation.calls[0] == other
    assert len(next_state.offspring) == 1
    assert next_state.population == [best, other]
    assert next_state.ages == {best: 1, other: 1}
    assert fitness_calls == [best, other]

    assert parent_selection.calls[0]["population_size"] == 2
    assert parent_selection.calls[0]["ages"] == {best: 0, other: 0}
    assert survivor_selection.calls[0]["population_size"] == 1
    assert survivor_selection.calls[0]["previous_generation_fitness"] == initial_state.fitness
    assert survivor_selection.calls[0]["ages"] == {other: 1, best: 1}
    assert survivor_selection.calls[0]["previous_generation_ages"] == {best: 0, other: 0}


def test_symbolic_regression() -> None:
    # TODO: This is the most simple integration test.
    #       Everything is prepared to enforce determinism, but a suitable integration test must still be written.
    """_summary_."""
    best_tree, train_mse, test_mse = run_symbolic_regression(
        # seed=0,
        population_size=30,
        max_generations=10,
        max_depth=5,
    )

    repo = SymbolicRegression(max_depth=4, variables=["x"], constants=[2.5382, 1.2345, 0.5678])

    assert isinstance(best_tree, Tree)
    assert isinstance(best_tree.interpret(repo.pretty_term_algebra()), str)
    assert math.isfinite(train_mse)
    assert math.isfinite(test_mse)
    assert train_mse >= 0
    assert test_mse >= 0


class _SingleOffspringMutation:
    """A mutation that always produces the same offspring.

    Attributes:
        offspring (Tree[str]): What every call returns.
    """

    def __init__(self, offspring: Tree[str]) -> None:
        """Store what the operator returns.

        Args:
            offspring (Tree[str]): What every call returns.
        """
        self.offspring = offspring

    def mutate(self, query: Any, tree: Tree[str]):
        """Return the fixed offspring.

        Args:
            query (Any): Unused.
            tree (Tree[str]): Unused.

        Returns:
            Tree[str]: The fixed offspring.
        """
        return self.offspring


def test_a_mutation_offspring_reaches_the_next_generation() -> None:
    """What the mutation returns is what the driver carries forward.

    The operator answers with one individual or with None, while the driver collects a list of
    candidates, so the two meet through an adapter. A driver that dropped the offspring, or that
    wrapped a None into a candidate, would still run and still terminate, and every other test here
    would stay green because none of them lets a mutation succeed.
    """
    parent_tree, mutant = Tree("parent"), Tree("mutant")
    solution_space: Any = object()
    parent_selection: Any = RecordingSelection([parent_tree])
    survivor_selection: Any = RecordingSelection([parent_tree])
    gp = SimpleGeneticProgramming(
        solution_space,
        "start",
        lambda state: state.generation >= 1,
        StaticInitialization([parent_tree, parent_tree]),
        _SingleOffspringMutation(mutant),
        EmptyRecombination(),
        parent_selection,
        survivor_selection,
        ScalarFitnessComparator(),
        rng=random.Random(0),
        elite_count=0,
        distribute_rngs=False,
    )

    states = list(gp.evolutionary_stream(lambda tree: 1.0 if tree == mutant else 0.0, 2, 1.0, 0.0))
    assert any(mutant in state.offspring for state in states)
