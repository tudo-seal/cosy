"""_summary_."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any

import pytest

from cosy.core.solution_space import SolutionSpace
from cosy.core.tree import Path, Tree
from cosy.evolutionary_algorithms.evolutionary import EAState, SimpleGeneticProgramming
from cosy.evolutionary_algorithms.fitness import ScalarFitnessComparator
from cosy.evolutionary_algorithms.mutation import ResolutionMutation
from cosy.evolutionary_algorithms.recombination import Crossover
from examples.example_symbolic_regression import (
    SymbolicRegression,
    run_symbolic_regression,
)

if TYPE_CHECKING:
    from collections.abc import MutableSequence


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

    def initialize_population(self, population_size: int):
        """_summary_.

        Args:
            population_size (int): _description_

        Returns:
            _type_: _description_
        """
        return self.population[:population_size]


class EmptyInitialization:
    """_summary_."""

    def initialize_population(self, population_size: int):
        """_summary_.

        Args:
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

    def mutate(self, tree: Tree[str]):
        """_summary_.

        Args:
            tree (Tree[str]): _description_

        Returns:
            _type_: _description_
        """
        self.calls.append(tree)
        return []


class EmptyRecombination:
    """_summary_."""

    def recombine(self, primary: Tree[str], secondary: Tree[str]):
        """_summary_.

        Args:
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


class RecordingRandom(random.Random):
    """A generator that remembers every sequence it was asked to draw from.

    The operators below hand the pool of candidate positions to ``choice`` or ``shuffle``.  What
    that pool contains, and in which order, is the whole of their randomness, so recording it
    inspects the decision itself rather than the tree that comes out the other end.

    Attributes:
        pools (list[list[Path]]): One entry per draw, in the order the draws happened.
    """

    def __init__(self, seed: int) -> None:
        """Seed the generator.

        Args:
            seed (int): The seed.
        """
        super().__init__(seed)
        self.pools: list[list[Any]] = []

    def shuffle(self, x: MutableSequence[Any], *args: Any) -> None:
        """Record the sequence and shuffle it.

        Args:
            x (MutableSequence[Any]): The sequence to shuffle in place.
            *args (Any): Never used. Kept so the signature stays compatible with the one typeshed
                declares for Python 3.10, where ``Random.shuffle`` still carries the second
                parameter that later versions removed.
        """
        self.pools.append(list(x))
        super().shuffle(x)

    def choice(self, seq: Any) -> Any:
        """Record the sequence and draw from it.

        Args:
            seq (Any): The sequence to draw from.

        Returns:
            Any: The drawn element.
        """
        self.pools.append(list(seq))
        return super().choice(seq)


class PermissiveSolutionSpace(SolutionSpace[str, str, str]):
    """A solution space that accepts everything and samples a constant.

    The operators are under test here, not the space: accepting every candidate makes them run
    their position bookkeeping to the end instead of bailing out on the first rejection, and a
    constant sample makes the outcome depend on the chosen position alone.
    """

    def contains_tree(self, start: str, tree: Tree[str], interpretation: dict[str, Any] | None = None) -> bool:
        """Accept every tree.

        Args:
            start (str): The start non-terminal.
            tree (Tree[str]): The candidate.
            interpretation (dict[str, Any] | None): Unused. (Default value = None)

        Returns:
            bool: Always ``True``.
        """
        return True

    def sample_tree(
        self,
        start: str,
        max_depth: int | None = None,
        tree: Tree[str] | None = None,
        pos: Path | None = None,
        rng: random.Random | None = None,
    ) -> Tree[str] | None:
        """Return a fixed subtree.

        Args:
            start (str): The start non-terminal.
            max_depth (int | None): Unused. (Default value = None)
            tree (Tree[str] | None): Unused. (Default value = None)
            pos (Path | None): Unused. (Default value = None)
            rng (random.Random | None): Unused. (Default value = None)

        Returns:
            Tree[str] | None: A single node.
        """
        return Tree("SAMPLED")


def sample_individual() -> Tree[str]:
    """Return ``f(g(h(x), y), z)``.

    Returns:
        Tree[str]: A tree with one branch deeper than the other, so the pool of inner positions
            has more than one element and its order is therefore observable.
    """
    return Tree("f", (Tree("g", (Tree("h", (Tree("x"),)), Tree("y"))), Tree("z")))


def test_mutation_draws_from_a_pool_in_a_fixed_order() -> None:
    """The mutation point is decided by the seed alone, not by set iteration order.

    ``positions()`` answers with a set, and the order a set iterates in is an implementation
    detail -- it shifts between interpreter versions, and it shifted when the position sets
    changed shape.  Sorting the pool before drawing from it is what keeps a seeded run
    reproducible across the whole test matrix.
    """
    rng = RecordingRandom(0)
    mutation: ResolutionMutation[str, str, str] = ResolutionMutation(PermissiveSolutionSpace(), "S", rng=rng)

    mutation.mutate(sample_individual())

    assert rng.pools == [[(0,), (0, 0)]]


def test_mutation_trims_one_level_of_leaves_per_trim_step() -> None:
    """Each pass drops the leaves, then the positions that pass turned into leaves -- and only those.

    Every pass after the first has to work out which of the remaining positions are childless
    now; a pass that reused the leaves of the original term, or that came up with nothing at all,
    would leave the pool one level too deep and offer a mutation point the caller excluded.  Of
    ``f(g(h(x), y), z)`` the first pass leaves ``(0,)`` and ``(0, 0)``, the second only ``(0,)``.

    The deeper term takes three passes, and that is what pins the other half: the positions a pass
    trims are collected in a set that starts empty each time.  A set carried across the passes
    would still hold what the pass before it trimmed and ask for those positions to be removed a
    second time.  Two passes cannot show it, because the last pass collects nothing.
    """
    rng = RecordingRandom(0)
    mutation: ResolutionMutation[str, str, str] = ResolutionMutation(PermissiveSolutionSpace(), "S", rng=rng)

    mutation.mutate(sample_individual(), trim=2)

    assert rng.pools == [[(0,)]]

    deeper = Tree("f", (Tree("g", (Tree("h", (Tree("k", (Tree("x"),)), Tree("y"))), Tree("z"))), Tree("w")))
    rng = RecordingRandom(0)
    mutation = ResolutionMutation(PermissiveSolutionSpace(), "S", rng=rng)

    mutation.mutate(deeper, trim=3)

    assert rng.pools == [[(0,)]]


def test_crossover_draws_from_pools_in_a_fixed_order() -> None:
    """Both parents' crossover points are shuffled from a sorted pool -- each parent from its own.

    Same reason as for mutation: the shuffle is reproducible from the seed only if the sequence
    going into it does not depend on how a set happens to be laid out.  Each parent has more than
    one inner position, because a pool of one element is in order whatever produced it, and the
    two are shaped differently, because pools that read alike say nothing about which parent
    either of them was collected from.
    """
    rng = RecordingRandom(0)
    crossover: Crossover[str, str, str] = Crossover(PermissiveSolutionSpace(), "S", rng=rng)

    second = Tree("F", (Tree("G", (Tree("H", (Tree("X"),)), Tree("Y"))), Tree("K", (Tree("Z"),))))
    crossover.recombine(sample_individual(), second)

    assert rng.pools == [[(0,), (0, 0)], [(0,), (0, 0), (1,)]]
