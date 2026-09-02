"""The driver: closure, the shape of one generation, and what the run returns.

Nine decisions of the algorithm are pinned here, and every one of them is a line of it:

* **recombination, then mutation**, per offspring, rather than one operator per pass chosen from a
  weighted triple, and with no "survive" branch, since copies of the parents are what the crossover
  draw produces when it fails, with probability ``1 - p_c``,
* **a pass whose batch is incomplete is discarded** and new parents are drawn. Nothing is filled up
  with unchanged parents, and the attempt cap raises instead of degrading quietly,
* **the result is the fittest individual encountered over the whole run**, not the best of the
  final generation,
* **initialization failure is an error**, not an empty result,
* **survivor selection receives the parents and the offspring together**, which the driver
  enforces,
* **no elitism in the driver**, keeping a fittest individual being a condition on the survivor
  selection component rather than a slot the driver reserves,
* **no age tracking**,
* **arguments at the call, parameters in the constructor**,
* ``p_c + p_m > 1`` is allowed, the two draws being independent.
"""

from __future__ import annotations

import math
import random
from itertools import islice
from typing import Any

import pytest

from cosy.core.tree import Tree
from cosy.evolutionary_algorithms import (
    EAState,
    EvolutionarySearch,
    ExpScalarization,
    FitnessBasedReplacement,
    FitnessProportionalSelection,
    Generations,
    GenerousConservativeReplacement,
    InitializationError,
    NoImprovement,
    ParetoFitnessComparator,
    ResolutionMutation,
    SampledInitialization,
    ScalarFitnessComparator,
    SubtreeGraft,
    SubtreeSwap,
    TournamentSelection,
    induced_fitness,
)
from cosy.search import (
    DepthBoundedRandomSampler,
    SizeUniformSampler,
    checker,
    generator_query,
    term_size,
)
from examples.example_symbolic_regression import (
    SymbolicRegression,
    run_symbolic_regression,
)
from tests.ea_fixtures import (
    NULLARY_START,
    RECURSIVE_START,
    a2,
    b2,
    bi,
    h1,
    lf,
    nullary_space,
    parent,
    recursive_space,
    top,
    un,
)


@pytest.fixture
def recursive():
    """Return the generator query on the primary recursive space.

    Returns:
        ResolutionQuery: The argument the driver takes at the call.
    """
    return generator_query(recursive_space(), RECURSIVE_START)


@pytest.fixture
def tiny():
    """Return the generator query on ``A -> a | b | h(A)``.

    Returns:
        ResolutionQuery: Small enough to state the optimum outright.
    """
    return generator_query(nullary_space(), NULLARY_START)


def by_size(individual) -> float:
    """Score an individual by its size.

    Args:
        individual (Tree): The individual.

    Returns:
        float: Its number of symbols.
    """
    return float(term_size(individual))


def search(seed: int, **overrides) -> EvolutionarySearch:
    """Build a driver over the recursive space with workable defaults.

    Args:
        seed (int): Base seed. Every component gets a distinct derived one.
        **overrides: Constructor arguments to replace.

    Returns:
        EvolutionarySearch: The driver.
    """
    parameters: dict[str, Any] = {
        "initializer": SampledInitialization(SizeUniformSampler(9, random.Random(seed))),
        "mutation": ResolutionMutation(DepthBoundedRandomSampler(4, random.Random(seed + 1)), random.Random(seed + 2)),
        "recombination": SubtreeSwap(random.Random(seed + 3), max_size=20),
        "parent_selection": TournamentSelection(2, random.Random(seed + 4)),
        "survivor_selection": FitnessBasedReplacement(),
        "termination": Generations(4),
        "population_size": 6,
        "crossover_rate": 0.8,
        "mutation_rate": 0.3,
        "rng": random.Random(seed + 5),
    }
    parameters.update(overrides)
    return EvolutionarySearch(**parameters)


def converging(seed: int, **overrides) -> EvolutionarySearch:
    """Build a driver meeting every convergence condition on the two-symbol space.

    Args:
        seed (int): Base seed.
        **overrides: Constructor arguments to replace.

    Returns:
        EvolutionarySearch: The driver.
    """
    parameters: dict[str, Any] = {
        "initializer": SampledInitialization(SizeUniformSampler(4, random.Random(seed))),
        "mutation": ResolutionMutation(SizeUniformSampler(4, random.Random(seed + 1)), random.Random(seed + 2)),
        "recombination": SubtreeSwap(random.Random(seed + 3), max_size=4),
        "parent_selection": FitnessProportionalSelection(ExpScalarization(), random.Random(seed + 4)),
        "survivor_selection": GenerousConservativeReplacement(ExpScalarization(), random.Random(seed + 5)),
        "termination": Generations(40),
        "population_size": 6,
        "crossover_rate": 0.7,
        "mutation_rate": 0.4,
        "rng": random.Random(seed + 6),
        "comparator": ScalarFitnessComparator(greater_is_better=True),
    }
    parameters.update(overrides)
    return EvolutionarySearch(**parameters)


# ---------------------------------------------------------------------------
# Stubs that record what the driver does
# ---------------------------------------------------------------------------


class _ScriptedRecombination:
    """Return a fixed batch and record the parents it was given.

    Attributes:
        calls (list): One ``(first, second)`` pair per call.
        batch (list): What every call returns.
    """

    def __init__(self, batch) -> None:
        """Store the batch.

        Args:
            batch (list): What to return.
        """
        self.calls: list = []
        self.batch = batch

    def recombine(self, query, first, second):
        """Record the call and return the batch.

        Args:
            query: Ignored.
            first (Tree): The first parent.
            second (Tree): The second parent.

        Returns:
            list: The stored batch.
        """
        self.calls.append((first, second))
        return list(self.batch)


class _ScriptedMutation:
    """Answer from a callable and record what it was asked about.

    Attributes:
        calls (list): The individuals it was asked to mutate.
        answer (Callable): Maps an individual to an offspring or None.
    """

    def __init__(self, answer) -> None:
        """Store the answer function.

        Args:
            answer (Callable): Maps an individual to an offspring or None.
        """
        self.calls: list = []
        self.answer = answer

    def mutate(self, query, individual):
        """Record the call and answer it.

        Args:
            query: Ignored.
            individual (Tree): The parent.

        Returns:
            Tree | None: Whatever the answer function returns.
        """
        self.calls.append(individual)
        return self.answer(individual)


class _RecordingSurvivors:
    """Record what survivor selection was handed, and keep the first ``size`` of the pool.

    Deliberately neither conservative nor generous: it keeps the offspring first, so the best
    individual can drop out of the population, which is what makes the best-so-far observable.

    Attributes:
        calls (list): One ``(parents, offspring)`` pair per call.
    """

    def __init__(self) -> None:
        """Start with no calls."""
        self.calls: list = []

    def select_survivors(self, parents, offspring, fitness, comparator, size):
        """Record the two populations and return a slice of them.

        Args:
            parents (list): The finished generation.
            offspring (list): Its offspring.
            fitness: Ignored.
            comparator: Ignored.
            size (int): The population size.

        Returns:
            list: The first ``size`` individuals of the offspring, then of the parents.
        """
        self.calls.append((list(parents), list(offspring)))
        return [*offspring, *parents][:size]


class _FixedInitializer:
    """Return a fixed population.

    Attributes:
        population (list): The population returned.
    """

    def __init__(self, population) -> None:
        """Store the population.

        Args:
            population (list): What to return.
        """
        self.population = population

    def initialize(self, query, size):
        """Return the stored population.

        Args:
            query: Ignored.
            size: Ignored.

        Returns:
            list: The stored population.
        """
        return list(self.population)


# ---------------------------------------------------------------------------
# Closure
# ---------------------------------------------------------------------------


def test_every_individual_the_run_holds_is_an_inhabitant(recursive):
    """Closure is about every population the run passes through, not only the result.

    Args:
        recursive: The recursive-space query fixture.
    """
    for state in search(0).evolutionary_stream(recursive, by_size):
        for individual in [*state.population, *state.offspring, state.best]:
            assert checker(recursive.solution_space, recursive.start, individual)


def test_the_result_is_the_fittest_individual_encountered(recursive):
    """The best-so-far individual is carried across the run, not read off the final population.

    Args:
        recursive: The recursive-space query fixture.
    """
    states = list(search(1).evolutionary_stream(recursive, by_size))
    encountered = {individual for state in states for individual in [*state.population, *state.offspring]}
    best = search(1).evolutionary_best(recursive, by_size)
    assert by_size(best) == max(by_size(individual) for individual in encountered)


def test_the_best_so_far_never_gets_worse(recursive):
    """The incumbent is replaced only by something strictly fitter.

    Args:
        recursive: The recursive-space query fixture.
    """
    scores = [state.best_fitness for state in search(2).evolutionary_stream(recursive, by_size)]
    assert scores == sorted(scores)


def test_the_incumbent_survives_a_population_that_drops_it(recursive):
    """Why the best-so-far exists: without elitism the final population can be worse than the run.

    Returning the best of the final generation would lose an individual that a later generation
    dropped. ``_RecordingSurvivors`` keeps the offspring first and is neither conservative nor
    generous, which is exactly the case the convergence guarantee excludes.

    Args:
        recursive: The recursive-space query fixture.
    """

    # Smaller is fitter, and variation grows terms, so the incumbent is regularly bred out.
    # Under "larger is fitter" the population improves monotonically and nothing is dropped.
    def by_smallness(individual) -> float:
        """Score an individual by how small it is.

        Args:
            individual (Tree): The individual.

        Returns:
            float: The negated size.
        """
        return -float(term_size(individual))

    driver = search(3, survivor_selection=_RecordingSurvivors(), termination=Generations(8))
    states = list(driver.evolutionary_stream(recursive, by_smallness))
    dropped = [
        state
        for state in states
        if max(by_smallness(individual) for individual in state.population) < state.best_fitness
    ]
    assert dropped, "the survivor selection never dropped the incumbent, so nothing was tested"
    assert states[-1].best_fitness >= max(by_smallness(individual) for individual in states[-1].population)


# ---------------------------------------------------------------------------
# The inner loop
# ---------------------------------------------------------------------------


def test_recombination_is_followed_by_mutation_on_the_same_offspring(recursive):
    """Sequential, per offspring, rather than one operator per pass.

    Args:
        recursive: The recursive-space query fixture.
    """
    children = [parent(1, 1), parent(2, 1)]
    recombination = _ScriptedRecombination(children)
    mutation = _ScriptedMutation(lambda individual: individual)
    driver = search(
        4,
        recombination=recombination,
        mutation=mutation,
        crossover_rate=1.0,
        mutation_rate=1.0,
        termination=Generations(1),
        population_size=2,
    )
    list(driver.evolutionary_stream(recursive, by_size))
    assert recombination.calls
    assert mutation.calls[:2] == children


def test_a_failed_crossover_draw_copies_the_parents(recursive):
    """There is no third branch: ``1 - p_c`` is where copies come from.

    Args:
        recursive: The recursive-space query fixture.
    """
    recombination = _ScriptedRecombination([])
    mutation = _ScriptedMutation(lambda individual: individual)
    driver = search(
        5,
        recombination=recombination,
        mutation=mutation,
        crossover_rate=0.0,
        mutation_rate=1.0,
        termination=Generations(1),
        population_size=2,
    )
    states = list(driver.evolutionary_stream(recursive, by_size))
    assert recombination.calls == []
    assert len(states[-1].offspring) == 2


def test_what_a_mutation_returns_is_what_the_offspring_carries(tiny):
    """The adapter between an operator answering with one individual and a list of candidates.

    A mutation answers with an individual or with None, while the driver collects a list, so the
    two meet through an adapter. A driver that appended the parent instead of the mutant would
    still run and still terminate, and no other test in this file would notice: the ones that let a
    mutation succeed at all hand it an identity mutation, and the one that does not only exercises
    the None branch.

    Args:
        tiny: The two-symbol space query fixture.
    """
    original, mutant = Tree(a2, ()), Tree(b2, ())
    driver = search(
        27,
        initializer=_FixedInitializer([original, original]),
        mutation=_ScriptedMutation(lambda _: mutant),
        crossover_rate=0.0,
        mutation_rate=1.0,
        population_size=2,
        termination=Generations(1),
    )
    states = list(driver.evolutionary_stream(tiny, by_size))
    assert states[-1].offspring == [mutant, mutant]


def test_a_pass_losing_an_offspring_to_mutation_is_discarded(recursive):
    """No filling up with unchanged parents. The whole pass goes.

    The mutation drops the second member of every batch, so no pass can complete. The driver must
    run into its attempt cap rather than quietly keeping the first member.

    Args:
        recursive: The recursive-space query fixture.
    """
    seen: list = []

    def drop_every_second(individual):
        """Return the individual for the first of a pair and nothing for the second.

        Args:
            individual (Tree): The parent.

        Returns:
            Tree | None: The individual, or None on every second call.
        """
        seen.append(individual)
        return None if len(seen) % 2 == 0 else individual

    driver = search(
        6,
        mutation=_ScriptedMutation(drop_every_second),
        crossover_rate=0.0,
        mutation_rate=1.0,
        termination=Generations(1),
        population_size=2,
    )
    with pytest.raises(RuntimeError, match="not filled up with unchanged parents"):
        list(driver.evolutionary_stream(recursive, by_size))


def test_an_empty_batch_discards_the_pass(recursive):
    """A recombination that finds no acceptable pair contributes nothing.

    Args:
        recursive: The recursive-space query fixture.
    """
    driver = search(
        7,
        recombination=_ScriptedRecombination([]),
        crossover_rate=1.0,
        mutation_rate=0.0,
        termination=Generations(1),
        population_size=2,
    )
    with pytest.raises(RuntimeError, match="Variation is failing systematically"):
        list(driver.evolutionary_stream(recursive, by_size))


def test_the_attempt_cap_counts_passes_and_reports_them(recursive):
    """The engineering guard reports rather than shrinking the population.

    Args:
        recursive: The recursive-space query fixture.
    """
    driver = search(
        8,
        recombination=_ScriptedRecombination([]),
        crossover_rate=1.0,
        mutation_rate=0.0,
        attempt_factor=3,
        population_size=4,
    )
    with pytest.raises(RuntimeError, match="in 12 passes"):
        list(driver.evolutionary_stream(recursive, by_size))


def test_a_graft_fills_the_recombination_slot(recursive):
    """The drop-in alternative: a batch of one is complete for a graft.

    A swap yields two offspring and a graft yields one, and both fill the same slot, so the rule
    the driver reads is the completeness of the batch rather than its size.

    Args:
        recursive: The recursive-space query fixture.
    """
    driver = search(
        9,
        recombination=SubtreeGraft(random.Random(90), max_size=20),
        crossover_rate=1.0,
        termination=Generations(3),
    )
    states = list(driver.evolutionary_stream(recursive, by_size))
    assert all(len(state.population) == 6 for state in states)
    assert states[-1].offspring


def test_offspring_are_the_pure_yield_of_variation(recursive):
    """``EAState.offspring`` holds what variation produced, and nothing else.

    Args:
        recursive: The recursive-space query fixture.
    """
    states = list(search(10).evolutionary_stream(recursive, by_size))
    assert states[0].offspring == []
    assert all(len(state.offspring) >= 6 for state in states[1:])


# ---------------------------------------------------------------------------
# Survivor selection and elitism
# ---------------------------------------------------------------------------


def test_survivor_selection_receives_the_parents_and_the_offspring(recursive):
    """The driver enforces the contract, rather than each component deciding for itself.

    Args:
        recursive: The recursive-space query fixture.
    """
    survivors = _RecordingSurvivors()
    driver = search(11, survivor_selection=survivors, termination=Generations(2))
    states = list(driver.evolutionary_stream(recursive, by_size))
    assert len(survivors.calls) == 2
    for index, (parents, offspring) in enumerate(survivors.calls):
        assert parents == states[index].population
        assert offspring == states[index + 1].offspring


def test_the_population_is_exactly_what_survivor_selection_returned(recursive):
    """No elitism in the driver: nothing is added to or removed from the component's answer.

    Args:
        recursive: The recursive-space query fixture.
    """
    survivors = _RecordingSurvivors()
    driver = search(12, survivor_selection=survivors, termination=Generations(2))
    states = list(driver.evolutionary_stream(recursive, by_size))
    for index in range(1, 3):
        parents, offspring = survivors.calls[index - 1]
        assert states[index].population == [*offspring, *parents][:6]


def test_a_survivor_selection_returning_the_wrong_count_is_refused(recursive):
    """The driver enforces both halves of the contract "mu individuals among them".

    Args:
        recursive: The recursive-space query fixture.
    """

    class _TooFew:
        """Return two fewer survivors than asked for."""

        def select_survivors(self, parents, offspring, fitness, comparator, size):
            """Return a short population.

            Args:
                parents (list): The finished generation.
                offspring (list): Its offspring.
                fitness: Ignored.
                comparator: Ignored.
                size (int): The population size.

            Returns:
                list: Two individuals fewer than requested.
            """
            return [*parents, *offspring][: size - 2]

    driver = search(24, survivor_selection=_TooFew(), termination=Generations(2))
    with pytest.raises(ValueError, match="for a population of 6"):
        list(driver.evolutionary_stream(recursive, by_size))


def test_a_survivor_selection_returning_a_stranger_is_refused(recursive):
    """A survivor selection chooses among what it was given.

    Args:
        recursive: The recursive-space query fixture.
    """

    class _Inventive:
        """Return an individual that was in neither population."""

        def select_survivors(self, parents, offspring, fitness, comparator, size):
            """Return a population with one stranger in it.

            Args:
                parents (list): The finished generation.
                offspring (list): Its offspring.
                fitness: Ignored.
                comparator: Ignored.
                size (int): The population size.

            Returns:
                list: ``size`` individuals, one of them from nowhere.
            """
            return [parent(7, 7), *parents, *offspring][:size]

    driver = search(25, survivor_selection=_Inventive(), termination=Generations(2))
    with pytest.raises(ValueError, match="neither the parents nor the offspring"):
        list(driver.evolutionary_stream(recursive, by_size))


def test_the_state_carries_no_ages(recursive):
    """Age tracking left with the age-based replacement that was the only reader of it.

    Args:
        recursive: The recursive-space query fixture.
    """
    state = next(iter(search(13).evolutionary_stream(recursive, by_size)))
    assert isinstance(state, EAState)
    assert not hasattr(state, "ages")


# ---------------------------------------------------------------------------
# Failures and parameters
# ---------------------------------------------------------------------------


def test_an_initialization_failure_is_an_error(tiny):
    """The search fails when the initialization does, rather than returning nothing.

    Args:
        tiny: The two-symbol space query fixture.
    """
    driver = search(
        14,
        initializer=SampledInitialization(SizeUniformSampler(1, random.Random(0))),
        population_size=5,
    )
    with pytest.raises(InitializationError):
        list(driver.evolutionary_stream(tiny, by_size))


@pytest.mark.parametrize("delivered", [0, 1, 7])
def test_an_initializer_returning_the_wrong_count_is_refused(tiny, delivered):
    """The driver enforces the initializer's contract, as it enforces the survivor selection's.

    Seven individuals is the case a guard written with ``<`` would let through.

    Args:
        tiny: The two-symbol space query fixture.
        delivered (int): The number of individuals the initializer answers with.
    """
    driver = search(
        28,
        initializer=_FixedInitializer([Tree(a2, ())] * delivered),
        population_size=5,
        termination=Generations(0),
    )
    with pytest.raises(ValueError, match=f"returned {delivered} individuals for a population of 5"):
        list(driver.evolutionary_stream(tiny, by_size))


def test_rates_summing_above_one_are_allowed(recursive):
    """The two draws are independent, so ``p_m + p_c <= 1`` is not required of a configuration.

    Args:
        recursive: The recursive-space query fixture.
    """
    driver = search(15, crossover_rate=0.9, mutation_rate=0.9, termination=Generations(2))
    assert list(driver.evolutionary_stream(recursive, by_size))


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("population_size", 0),
        ("crossover_rate", 1.5),
        ("mutation_rate", -0.1),
        ("attempt_factor", 0),
    ],
)
def test_a_parameter_outside_its_range_is_refused(parameter, value):
    """The numbers are fixed before the run, so they are checked before it.

    Args:
        parameter (str): The constructor argument.
        value: The value outside its range.
    """
    with pytest.raises(ValueError, match=r".+"):
        search(16, **{parameter: value})


def test_the_same_seed_yields_the_same_run(recursive):
    """Reproducibility: every draw goes through a component's own generator.

    Args:
        recursive: The recursive-space query fixture.
    """
    first = [state.population for state in search(17).evolutionary_stream(recursive, by_size)]
    second = [state.population for state in search(17).evolutionary_stream(recursive, by_size)]
    assert first == second


def test_the_driver_does_not_touch_the_global_random_stream(recursive):
    """The rate draws come from the driver's own generator.

    Args:
        recursive: The recursive-space query fixture.
    """
    random.seed(2024)
    before = random.random()
    random.seed(2024)
    search(18).evolutionary_best(recursive, by_size)
    assert random.random() == before


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------


def test_a_batch_fitness_function_sees_a_whole_generation(recursive):
    """A fitness function that scores a whole generation in one call receives it whole.

    Args:
        recursive: The recursive-space query fixture.
    """
    batches: list[int] = []

    def measure(individuals: list) -> dict:
        """Score a whole generation at once.

        Args:
            individuals (list): The individuals to score.

        Returns:
            dict: Their fitness.
        """
        batches.append(len(individuals))
        return {individual: by_size(individual) for individual in individuals}

    list(search(19).evolutionary_stream(recursive, measure))
    assert batches
    assert max(batches) > 1


def test_a_batch_function_omitting_an_individual_is_refused(recursive):
    """A missing answer is an error, not a substituted value.

    Args:
        recursive: The recursive-space query fixture.
    """

    def incomplete(individuals: list) -> dict:
        """Score all but the last individual.

        Args:
            individuals (list): The individuals to score.

        Returns:
            dict: The fitness of all but one.
        """
        return {individual: by_size(individual) for individual in individuals[:-1]}

    with pytest.raises(ValueError, match="every individual"):
        list(search(20).evolutionary_stream(recursive, incomplete))


def test_the_fitness_cache_pays_for_an_individual_once(recursive):
    """A survivor is not re-evaluated, which is what makes an expensive measure affordable.

    Args:
        recursive: The recursive-space query fixture.
    """
    seen: list = []

    def measure(individual) -> float:
        """Score one individual and record that it was asked for.

        Args:
            individual (Tree): The individual.

        Returns:
            float: Its size.
        """
        seen.append(individual)
        return by_size(individual)

    list(search(21).evolutionary_stream(recursive, measure))
    assert len(seen) == len(set(seen))


def test_an_individual_the_population_holds_twice_is_paid_for_once(tiny):
    """Two copies of an individual in one population cost one measurement.

    The fitness cache is empty when the initial population is evaluated, so it cannot spare the
    second measurement. Deduplicating the individuals handed to that evaluation does.

    Args:
        tiny: The two-symbol space query fixture.
    """
    twin, other = Tree(h1, (Tree(a2, ()),)), Tree(b2, ())
    seen: list = []

    def measure(individual) -> float:
        """Score one individual and record that it was asked for.

        Args:
            individual (Tree): The individual.

        Returns:
            float: Its size.
        """
        seen.append(individual)
        return by_size(individual)

    driver = search(
        22,
        initializer=_FixedInitializer([twin, twin, other]),
        population_size=3,
        termination=Generations(0),
    )
    states = list(driver.evolutionary_stream(tiny, measure))
    assert states[0].population.count(twin) == 2
    assert seen.count(twin) == 1
    assert seen.count(other) == 1


def test_the_state_carries_the_fitness_of_its_population(tiny):
    """The keys of ``EAState.fitness`` are the population, not everything the run has measured.

    The run's cache keeps every individual it has measured, so the mapping and the cache come
    apart as soon as a measured individual is not in the population.

    Args:
        tiny: The two-symbol space query fixture.
    """
    original, mutant = Tree(a2, ()), Tree(b2, ())
    driver = search(
        29,
        initializer=_FixedInitializer([original, original]),
        mutation=_ScriptedMutation(lambda _: mutant),
        survivor_selection=_RecordingSurvivors(),
        crossover_rate=0.0,
        mutation_rate=1.0,
        population_size=2,
        termination=Generations(1),
    )
    states = list(driver.evolutionary_stream(tiny, by_size))
    assert original in states[0].fitness
    assert original not in states[-1].population
    assert states[-1].fitness == {mutant: by_size(mutant)}


def test_a_batch_function_that_does_not_return_a_mapping_is_refused(recursive):
    """A batch function answers with a mapping from individual to fitness, not with a sequence.

    Fitness values in the order of the individuals would be an equally plausible contract. The
    driver states one of the two and refuses the other rather than reading whichever it is handed.

    Args:
        recursive: The recursive-space query fixture.
    """

    def positional(individuals: list) -> list:
        """Score a whole generation and answer positionally.

        Args:
            individuals (list): The individuals to score.

        Returns:
            list: Their fitness, in the order they arrived in.
        """
        return [by_size(individual) for individual in individuals]

    with pytest.raises(TypeError, match="must return a mapping"):
        list(search(22).evolutionary_stream(recursive, positional))


def test_the_batch_mode_hands_an_unannotated_function_a_whole_generation(recursive):
    """``fitness_function_mode="batch"`` is read where there is no annotation to read.

    The function below carries no annotation, so "auto" would find nothing that says batch and
    would call it once per individual. It answers to either call, so what it is handed is the only
    thing under test.

    Args:
        recursive: The recursive-space query fixture.
    """
    handed: list = []

    def measure(individuals):
        """Score one individual or a whole generation, whichever arrives.

        Args:
            individuals (Tree | list): What the driver passes.

        Returns:
            float | dict: One fitness value, or a mapping for the whole generation.
        """
        handed.append(individuals)
        if isinstance(individuals, Tree):
            return by_size(individuals)
        return {individual: by_size(individual) for individual in individuals}

    list(search(23).evolutionary_stream(recursive, measure, fitness_function_mode="batch"))
    assert handed
    assert all(isinstance(argument, list) for argument in handed)


def test_the_single_mode_hands_a_batch_annotated_function_one_individual(recursive):
    """``fitness_function_mode="single"`` is read before the annotation is.

    The parameter below is annotated as a list, which is what "auto" reads as a batch function.
    The stated mode decides against it.

    Args:
        recursive: The recursive-space query fixture.
    """
    handed: list = []

    def measure(individuals: list):
        """Score one individual or a whole generation, whichever arrives.

        Args:
            individuals (list): What the driver passes.

        Returns:
            float | dict: One fitness value, or a mapping for the whole generation.
        """
        handed.append(individuals)
        if isinstance(individuals, Tree):
            return by_size(individuals)
        return {individual: by_size(individual) for individual in individuals}

    list(search(24).evolutionary_stream(recursive, measure, fitness_function_mode="single"))
    assert handed
    assert all(isinstance(argument, Tree) for argument in handed)


def test_a_fitness_function_whose_annotation_does_not_resolve_still_runs(recursive):
    """A fitness function whose annotation does not resolve is scored one individual at a time.

    ``induced_fitness`` returns a function annotated ``Tree[Any]`` from a module that has
    ``from __future__ import annotations`` and imports ``Tree`` under ``TYPE_CHECKING``, so
    ``typing.get_type_hints`` has a string to resolve and no binding to resolve it against.
    Reading the annotation is what fails there, and "auto" runs without it.

    Args:
        recursive: The recursive-space query fixture.
    """
    counting = {lf: 1.0, un: lambda c: c + 1.0, bi: lambda c, d: c + d + 1.0, top: lambda c, d: c + d + 1.0}
    scored = induced_fitness(counting)

    states = list(search(25).evolutionary_stream(recursive, scored))

    assert states
    final = states[-1]
    assert final.fitness == {individual: scored(individual) for individual in final.population}


# ---------------------------------------------------------------------------
# Almost sure convergence
# ---------------------------------------------------------------------------


def test_a_run_meeting_every_condition_reaches_the_optimum(tiny):
    """The smoke test for almost sure convergence: not a proof, a run of the stated instance.

    All five conditions hold. The individuals the run can hold are the terms of size at most 4,
    the sampler bounding initialization and mutation and ``max_size`` bounding recombination. The
    size-uniform sampler is exhaustive, proportional parent selection gives every member a positive
    share, the replacement is generous and conservative, and ``p_c < 1`` and ``p_m > 0``. Under
    "larger is fitter" the optimum is a term of size 4.

    Args:
        tiny: The two-symbol space query fixture.
    """
    best = converging(30).evolutionary_best(tiny, by_size)
    assert term_size(best) == 4
    assert checker(tiny.solution_space, tiny.start, best)


def test_the_conservative_condition_keeps_the_optimum_once_it_is_found(tiny):
    """The conservative half keeps a fittest member in every later population.

    Args:
        tiny: The two-symbol space query fixture.
    """
    driver = converging(40, termination=Generations(25))
    reached = False
    for state in driver.evolutionary_stream(tiny, by_size):
        best_in_population = max(by_size(individual) for individual in state.population)
        if reached:
            assert best_in_population == state.best_fitness
        reached = reached or best_in_population == state.best_fitness
    # Not vacuous: the run has to reach its incumbent at all for the loop above to assert
    # anything, and it does so within the generation budget.
    assert reached


def test_no_improvement_stops_a_stalled_run(tiny):
    """The second termination condition, on a run that converges early.

    ``NoImprovement`` is the only stopping rule of this run, so a bookkeeping error that recorded an
    improvement in every generation would leave the stream without an end. Reading a prefix reports
    that as a failing assertion rather than as a job that hangs. Its length is one more than a
    correct run can produce. Fitness is the term size and the individuals here are the terms of size
    at most 4, so the best-so-far takes at most four values and improves at most three times. The
    stall counter starts at generation 0 and restarts at every improvement, and no restart carries
    the run more than three generations, so no improvement falls later than generation 9 and no
    state later than generation 12. That is thirteen states, generation 0 included.

    Args:
        tiny: The two-symbol space query fixture.
    """
    bound = 13
    driver = converging(50, termination=NoImprovement(3))
    states = list(islice(driver.evolutionary_stream(tiny, by_size), bound + 1))
    assert len(states) <= bound
    assert states[-1].generation - states[-1].last_improvement == 3


# ---------------------------------------------------------------------------
# The partial order, in the driver
# ---------------------------------------------------------------------------


def test_the_incumbent_is_replaced_by_any_member_that_beats_it(tiny):
    """The algorithm asks whether **a** member of the population is fitter than the incumbent.

    Under a partial order that is not the same as asking whether *the* fittest member is fitter. A
    scan for one maximal element can stop at an individual incomparable to the incumbent while
    another member strictly dominates it, and then the incumbent is never replaced although the
    population improved on it.

    Args:
        tiny: The two-symbol space query fixture.
    """
    incumbent, dominating, sideways = Tree(a2, ()), Tree(b2, ()), Tree(h1, (Tree(a2, ()),))
    scores = {incumbent: [1.0, 1.0], dominating: [2.0, 2.0], sideways: [3.0, 0.0]}

    class _KeepBoth:
        """Return the two new individuals, so the incumbent leaves the population."""

        def select_survivors(self, parents, offspring, fitness, comparator, size):
            """Return the offspring.

            Args:
                parents (list): The finished generation.
                offspring (list): Its offspring.
                fitness: Ignored.
                comparator: Ignored.
                size (int): The population size.

            Returns:
                list: ``size`` offspring.
            """
            return list(offspring)[:size]

    class _FixedBatch:
        """Produce the same two offspring in every pass, ``sideways`` first."""

        def recombine(self, query, first, second):
            """Return the fixed batch.

            Args:
                query: Ignored.
                first (Tree): Ignored.
                second (Tree): Ignored.

            Returns:
                list: The two offspring.
            """
            return [sideways, dominating]

    driver = EvolutionarySearch(
        initializer=_FixedInitializer([incumbent, incumbent]),
        mutation=_ScriptedMutation(lambda individual: individual),
        recombination=_FixedBatch(),
        parent_selection=TournamentSelection(1, random.Random(70)),
        survivor_selection=_KeepBoth(),
        termination=Generations(1),
        population_size=2,
        crossover_rate=1.0,
        mutation_rate=0.0,
        rng=random.Random(71),
        comparator=ParetoFitnessComparator(),
    )
    states = list(driver.evolutionary_stream(tiny, lambda individual: scores[individual]))
    # ``sideways`` is maximal and incomparable to the incumbent, while ``dominating`` beats it.
    assert states[-1].best == dominating
    assert states[-1].last_improvement == 1


def test_the_incumbent_is_replaced_although_the_pool_leads_with_an_incomparable(tiny):
    """The same question, asked where a scan for one maximal element would stop too early.

    The best-so-far reads the parents and the offspring together, and the parents come first, so a
    scan that starts at the head of the pool starts at the previous generation. Generation 1 puts
    ``sideways`` there, which is incomparable to the incumbent and therefore leaves the best-so-far
    where it is. Generation 2 adds ``dominating``, which is incomparable to ``sideways`` but beats
    the incumbent. A scan that maximizes first and compares afterwards stops at ``sideways`` and
    reports nothing.

    Args:
        tiny: The two-symbol space query fixture.
    """
    incumbent, dominating, sideways = Tree(a2, ()), Tree(b2, ()), Tree(h1, (Tree(a2, ()),))
    scores = {incumbent: [1.0, 1.0], dominating: [2.0, 2.0], sideways: [3.0, 0.0]}

    class _KeepBoth:
        """Return the two new individuals, so the previous generation leaves the population."""

        def select_survivors(self, parents, offspring, fitness, comparator, size):
            """Return the offspring.

            Args:
                parents (list): The finished generation.
                offspring (list): Its offspring.
                fitness: Ignored.
                comparator: Ignored.
                size (int): The population size.

            Returns:
                list: ``size`` offspring.
            """
            return list(offspring)[:size]

    class _SidewaysThenDominating:
        """Produce ``sideways`` in the first pass and ``dominating`` in every later one."""

        def __init__(self) -> None:
            """Start before the first pass."""
            self.passes = 0

        def recombine(self, query, first, second):
            """Return the batch this pass calls for.

            Args:
                query: Ignored.
                first (Tree): Ignored.
                second (Tree): Ignored.

            Returns:
                list: Two copies of one individual.
            """
            self.passes += 1
            return [sideways, sideways] if self.passes == 1 else [dominating, dominating]

    driver = EvolutionarySearch(
        initializer=_FixedInitializer([incumbent, incumbent]),
        mutation=_ScriptedMutation(lambda individual: individual),
        recombination=_SidewaysThenDominating(),
        parent_selection=TournamentSelection(1, random.Random(72)),
        survivor_selection=_KeepBoth(),
        termination=Generations(2),
        population_size=2,
        crossover_rate=1.0,
        mutation_rate=0.0,
        rng=random.Random(73),
        comparator=ParetoFitnessComparator(),
    )
    states = list(driver.evolutionary_stream(tiny, lambda individual: scores[individual]))

    assert states[1].best == incumbent, "sideways is incomparable to the incumbent"
    assert states[-1].best == dominating
    assert states[-1].last_improvement == 2


def test_an_offspring_the_survivor_selection_drops_is_still_the_best_so_far(tiny):
    """The best-so-far reads the pool, so an offspring that beats it counts even where it is cut.

    A survivor selection keeps ``size`` individuals and is a parameter of the search, so under a
    partial order it may drop an offspring that beats the incumbent. A dropped individual never
    returns to challenge again, so reading the survivors alone would lose it for good.

    Args:
        tiny: The two-symbol space query fixture.
    """
    incumbent, dominating = Tree(a2, ()), Tree(b2, ())
    scores = {incumbent: [1.0, 1.0], dominating: [2.0, 2.0]}

    class _KeepTheParents:
        """Return the parents, so every offspring is dropped after it was evaluated."""

        def select_survivors(self, parents, offspring, fitness, comparator, size):
            """Return the parents.

            Args:
                parents (list): The finished generation.
                offspring (list): Its offspring, dropped here.
                fitness: Ignored.
                comparator: Ignored.
                size (int): The population size.

            Returns:
                list: ``size`` parents.
            """
            return list(parents)[:size]

    class _FixedBatch:
        """Produce the same two offspring in every pass."""

        def recombine(self, query, first, second):
            """Return the fixed batch.

            Args:
                query: Ignored.
                first (Tree): Ignored.
                second (Tree): Ignored.

            Returns:
                list: The two offspring.
            """
            return [dominating, dominating]

    driver = EvolutionarySearch(
        initializer=_FixedInitializer([incumbent, incumbent]),
        mutation=_ScriptedMutation(lambda individual: individual),
        recombination=_FixedBatch(),
        parent_selection=TournamentSelection(1, random.Random(74)),
        survivor_selection=_KeepTheParents(),
        termination=Generations(1),
        population_size=2,
        crossover_rate=1.0,
        mutation_rate=0.0,
        rng=random.Random(75),
        comparator=ParetoFitnessComparator(),
    )
    states = list(driver.evolutionary_stream(tiny, lambda individual: scores[individual]))

    assert states[-1].population == [incumbent, incumbent], "the survivor selection kept the parents"
    assert states[-1].best == dominating
    assert states[-1].last_improvement == 1


def test_the_incumbent_moves_to_a_maximal_challenger(tiny):
    """Among the members that beat the incumbent, a maximal one becomes the new best-so-far.

    Two members of the new population beat the incumbent, and one of them dominates the other.
    Taking the first would put the dominated one in the best-so-far place.

    Args:
        tiny: The two-symbol space query fixture.
    """
    incumbent, weaker, stronger = Tree(a2, ()), Tree(b2, ()), Tree(h1, (Tree(a2, ()),))
    scores = {incumbent: [1.0, 1.0], weaker: [2.0, 2.0], stronger: [2.0, 3.0]}
    driver = search(
        27,
        initializer=_FixedInitializer([incumbent, incumbent]),
        recombination=_ScriptedRecombination([weaker, stronger]),
        survivor_selection=_RecordingSurvivors(),
        termination=Generations(1),
        population_size=2,
        crossover_rate=1.0,
        mutation_rate=0.0,
        comparator=ParetoFitnessComparator(),
    )
    states = list(driver.evolutionary_stream(tiny, lambda individual: scores[individual]))
    assert states[-1].best == stronger


def test_the_initial_best_so_far_is_maximal_in_its_population(tiny):
    """The initial best-so-far is an individual no member of the initial population is fitter than.

    The initial population holds a maximal member, a second one incomparable to it, and a third one
    that the first dominates and the second is incomparable to. A scan that moves to every candidate
    it does not lose to takes both incomparable steps and ends on the third member.

    Args:
        tiny: The two-symbol space query fixture.
    """
    maximal, sideways, dominated = Tree(a2, ()), Tree(b2, ()), Tree(h1, (Tree(a2, ()),))
    scores = {maximal: [2.0, 2.0], sideways: [3.0, 0.0], dominated: [1.0, 1.0]}
    driver = search(
        28,
        initializer=_FixedInitializer([maximal, sideways, dominated]),
        termination=Generations(0),
        population_size=3,
        comparator=ParetoFitnessComparator(),
    )
    states = list(driver.evolutionary_stream(tiny, lambda individual: scores[individual]))
    assert states[0].best == maximal


def test_a_fitness_the_order_cannot_place_is_refused(recursive):
    """A value incomparable to itself is outside the codomain of a fitness function.

    Nothing is ever fitter than such a value, so once one holds the best-so-far place nothing
    dislodges it again and it is returned as the fittest encountered, and a failed measurement
    would decide the search. The driver names the individual instead.

    Args:
        recursive: The recursive-space query fixture.
    """
    driver = search(26, termination=Generations(1))
    with pytest.raises(ValueError, match="cannot compare to itself"):
        list(driver.evolutionary_stream(recursive, lambda individual: math.nan))


# ---------------------------------------------------------------------------
# The population as a multiset
# ---------------------------------------------------------------------------


def test_an_individual_may_repeat_in_a_population(tiny):
    """A population is a multiset, and nothing deduplicates it.

    Args:
        tiny: The two-symbol space query fixture.
    """
    twin = Tree(h1, (Tree(a2, ()),))
    driver = search(
        23,
        initializer=_FixedInitializer([twin, twin, Tree(b2, ())]),
        mutation=ResolutionMutation(SizeUniformSampler(3, random.Random(60)), random.Random(61)),
        recombination=SubtreeSwap(random.Random(62)),
        population_size=3,
        termination=Generations(0),
    )
    states = list(driver.evolutionary_stream(tiny, by_size))
    assert states[0].population.count(twin) == 2


# ---------------------------------------------------------------------------
# The example, end to end
# ---------------------------------------------------------------------------


def test_symbolic_regression() -> None:
    """Run the symbolic-regression example end to end, and twice from the same seed.

    The second run is what makes this more than a smoke test. Every component of the example takes
    its own generator from a factory seeded with ``seed``, and the driver draws its two rates from
    a generator of its own, so two runs of one process with the same seed must agree in every
    digit. Comparing the errors rather than the trees keeps the assertion readable while still
    failing if any draw moved: a different draw anywhere changes the population and with it the
    error of the individual that is returned.

    The comparison is within one process on purpose. The solution space is built through Python
    sets, whose iteration order depends on the hash seed, so two *processes* agree only with
    ``PYTHONHASHSEED`` fixed. That is a property of the space construction rather than of the
    search, and pinning it here would test the wrong thing.
    """
    best_tree, train_mse, test_mse = run_symbolic_regression(
        seed=0,
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

    repeated_tree, repeated_train, repeated_test = run_symbolic_regression(
        seed=0,
        population_size=30,
        max_generations=10,
        max_depth=5,
    )
    assert repeated_tree == best_tree
    assert (repeated_train, repeated_test) == (train_mse, test_mse)


# ---------------------------------------------------------------------------
# mu = 1: the population that has no second parent
# ---------------------------------------------------------------------------


def test_a_population_of_one_never_recombines(recursive):
    """``p_c`` cannot mean anything when there is one individual, so the operator is not asked.

    ``select_parents`` returns the same term twice at ``mu = 1``, and a binary operator handed one
    term twice cannot mix two individuals. ``SubtreeSwap`` draws from the pairs of that term's own
    inner positions: two distinct positions put one of its subterms into another of its own, which
    is a mutation under another name, a single inner position leaves only the pair of that position
    with itself and hands the term back unchanged, and a term with no inner position leaves no pair
    at all and the batch is empty. None of the three is what ``p_c`` was set for.

    ``crossover_rate=1.0`` here, so that every pass would reach the operator if it were asked.

    Args:
        recursive: The recursive-space query fixture.
    """
    recombination = _ScriptedRecombination([])
    driver = search(
        11,
        population_size=1,
        crossover_rate=1.0,
        mutation_rate=1.0,
        recombination=recombination,
        termination=Generations(5),
    )

    list(driver.evolutionary_stream(recursive, by_size))

    assert recombination.calls == [], (
        "recombination was attempted with a single parent, which is the call that cannot succeed"
    )


def test_a_population_of_one_runs_to_its_termination(recursive):
    """The whole point: the configuration completes instead of raising in some later generation.

    Args:
        recursive: The recursive-space query fixture.
    """
    driver = search(
        12,
        population_size=1,
        crossover_rate=0.85,
        mutation_rate=1.0,
        termination=Generations(25),
    )

    states = list(driver.evolutionary_stream(recursive, by_size))

    assert len(states) == 26, "a (1+1) run must reach its generation bound"
    assert all(len(state.population) == 1 for state in states)


def test_a_population_of_one_yields_one_offspring_per_pass(recursive):
    """Not two copies of the same parent, a (1+lambda) pool having to stay free of duplicates.

    Args:
        recursive: The recursive-space query fixture.
    """
    mutation = _ScriptedMutation(lambda individual: individual)
    driver = search(
        13,
        population_size=1,
        crossover_rate=0.85,
        mutation_rate=1.0,
        mutation=mutation,
        termination=Generations(3),
    )

    list(driver.evolutionary_stream(recursive, by_size))

    # One mutation call per generation and not two, the batch being a single parent.
    assert len(mutation.calls) == 3


def test_two_individuals_still_recombine(recursive):
    """The guard is about ``mu = 1`` alone. At two parents nothing changes.

    Args:
        recursive: The recursive-space query fixture.
    """
    recombination = _ScriptedRecombination([Tree("lf"), Tree("lf")])
    driver = search(
        14,
        population_size=2,
        crossover_rate=1.0,
        mutation_rate=0.0,
        recombination=recombination,
        termination=Generations(3),
    )

    list(driver.evolutionary_stream(recursive, by_size))

    assert recombination.calls, "at mu = 2 the recombination operator must still be used"
