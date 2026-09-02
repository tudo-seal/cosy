"""Evolutionary search on a synthesized search space.

An evolutionary search procedure takes a synthesized solution space, a type and a fitness function,
returns an inhabitant or fails, and may make random choices. It is **closed** if every individual
it holds during a run lies in the tree language, not merely the one it returns. Closure holds for
every parameter choice, and the argument for it is short because the operators carry it. The
initializer delivers completions of the generator query, mutation is closed by construction,
recombination by rejection, and a survivor selection returns members of what it was given.

**Arguments against parameters.** The arguments of the search are the problem, namely the search
space and the fitness function, and they enter at the call. The parameters are the component
choices and the numbers mu, p_c and p_m, and they are fixed before the run, which here means the
constructor. One instance therefore runs any number of problems under one configuration.

**Recombination, then mutation, per offspring.** This is a deliberate departure from the tradition
of picking one operator per pass. The algorithm applies the crossover draw first and then the
mutation draw to each member of the resulting batch. There is no third "survive" branch: copies of
the parents are what the crossover draw produces when it *fails*, with probability ``1 - p_c``.

**A pass whose batch is incomplete is discarded, and new parents are drawn.** Nothing is filled in
with unchanged parents. Under the conditions for almost sure convergence the inner loop terminates
almost surely. The attempt cap here is an engineering guard, and it raises rather than degrading
quietly.

**No elitism in the driver.** Keeping a fittest individual is a condition on the survivor selection
*component*, discharged by
:class:`~cosy.evolutionary_algorithms.selection.GenerousConservativeReplacement`, and the
bookkeeping of the best-so-far individual is independent of it. That individual is what the
algorithm returns, and what it returns is a fittest individual **encountered**, not the best of the
final generation.
"""

from __future__ import annotations

import inspect
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeVar,
    cast,
    get_origin,
    get_type_hints,
)

from cosy.core.solution_space import NT, G, T
from cosy.evolutionary_algorithms.fitness import (
    Comparison,
    Fitness,
    FitnessComparator,
    ScalarFitnessComparator,
)

if TYPE_CHECKING:
    import random
    from collections.abc import Callable, Iterator

    from cosy.core.tree import Tree
    from cosy.evolutionary_algorithms.initialisation import Initializer
    from cosy.evolutionary_algorithms.mutation import Mutation
    from cosy.evolutionary_algorithms.recombination import Recombination
    from cosy.evolutionary_algorithms.selection import (
        ParentSelection,
        SurvivorSelection,
    )
    from cosy.evolutionary_algorithms.termination import Termination
    from cosy.search.queries import ResolutionQuery

FitnessFunctionMode = Literal["auto", "single", "batch"]

_T = TypeVar("_T")

__all__ = ["EAState", "EvolutionarySearch", "FitnessFunctionMode"]


@dataclass
class EAState(Generic[T]):
    """A snapshot of one generation.

    Attributes:
        generation (int): The generation number. The initial population is generation 0.
        population (list[Tree[T]]): The current population, a multiset in which an individual may
            repeat.
        fitness (dict[Tree[T], Fitness]): The fitness of every member of the population.
        offspring (list[Tree[T]]): The pure yield of variation in this generation. It is empty in
            generation 0 and holds at least mu individuals in every later one, the inner loop
            either filling it or raising at the attempt cap.
        best (Tree[T]): The fittest individual encountered so far, over the whole run.
        best_fitness (Fitness): Its fitness.
        last_improvement (int): The generation in which ``best`` was last replaced. A termination
            condition counting stalled generations subtracts, so it needs no state of its own.
    """

    generation: int
    population: list[Tree[T]]
    fitness: dict[Tree[T], Fitness]
    offspring: list[Tree[T]]
    best: Tree[T]
    best_fitness: Fitness
    last_improvement: int


class EvolutionarySearch(Generic[NT, T, G]):
    """The components and the numbers of a run, assembled into one algorithm.

    Attributes:
        initializer (Initializer[NT, T, G]): Builds the initial population.
        mutation (Mutation[NT, T, G]): The unary operator.
        recombination (Recombination[NT, T, G]): The binary operator.
        parent_selection (ParentSelection[T]): Draws a pair of parents per pass.
        survivor_selection (SurvivorSelection[T]): Maps parents and offspring to the next
            population.
        termination (Termination[T]): Decides when the outer loop stops.
        population_size (int): mu.
        crossover_rate (float): p_c.
        mutation_rate (float): p_m.
        comparator (FitnessComparator): The partial order on fitness values.
        rng (random.Random): The source of the two rate draws.
        attempt_factor (int): Passes per generation are capped at this multiple of mu.
    """

    # Recombination is a binary operator, so a population of one has nothing for it to work on.
    _RECOMBINATION_ARITY = 2

    def __init__(
        self,
        initializer: Initializer[NT, T, G],
        mutation: Mutation[NT, T, G],
        recombination: Recombination[NT, T, G],
        parent_selection: ParentSelection[T],
        survivor_selection: SurvivorSelection[T],
        termination: Termination[T],
        population_size: int,
        crossover_rate: float,
        mutation_rate: float,
        rng: random.Random,
        comparator: FitnessComparator | None = None,
        attempt_factor: int = 10,
    ) -> None:
        """Fix the components and the numbers of a run.

        Args:
            initializer (Initializer[NT, T, G]): Builds the initial population.
            mutation (Mutation[NT, T, G]): The unary operator.
            recombination (Recombination[NT, T, G]): The binary operator.
            parent_selection (ParentSelection[T]): Draws a pair of parents per pass.
            survivor_selection (SurvivorSelection[T]): Chooses the next population.
            termination (Termination[T]): Decides when to stop.
            population_size (int): mu, at least 1.
            crossover_rate (float): p_c, in [0, 1]. Almost sure convergence needs ``p_c < 1``, so
                that a pass can copy its parents. 1.0 is permitted and documented, not forbidden.
            mutation_rate (float): p_m, in [0, 1]. Almost sure convergence needs ``p_m > 0``.
            rng (random.Random): The source of the crossover and mutation draws. Each component
                carries its own generator, and nothing is distributed to them here, because
                assigning one by looking for an ``rng`` attribute is exactly the kind of invariant
                a caller cannot read off the API.
            comparator (FitnessComparator | None): The partial order on fitness values.
                (Default value = None, meaning a maximizing scalar comparator)
            attempt_factor (int): The cap on passes per generation, as a multiple of mu.
                (Default value = 10)

        Raises:
            ValueError: If a number lies outside its range.
        """
        if population_size < 1:
            msg = f"a population holds at least one individual: {population_size}"
            raise ValueError(msg)
        if not 0.0 <= crossover_rate <= 1.0:
            msg = f"the crossover rate is a probability: {crossover_rate}"
            raise ValueError(msg)
        if not 0.0 <= mutation_rate <= 1.0:
            msg = f"the mutation rate is a probability: {mutation_rate}"
            raise ValueError(msg)
        if attempt_factor < 1:
            msg = f"a generation needs at least mu passes to fill its offspring: {attempt_factor}"
            raise ValueError(msg)
        self.initializer = initializer
        self.mutation = mutation
        self.recombination = recombination
        self.parent_selection = parent_selection
        self.survivor_selection = survivor_selection
        self.termination = termination
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.rng = rng
        self.comparator = comparator if comparator is not None else ScalarFitnessComparator()
        self.attempt_factor = attempt_factor

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_batch_fitness_function(fitness_function: Callable[..., Any]) -> bool:
        """Decide whether a fitness function takes a list of individuals.

        Args:
            fitness_function (Callable[..., Any]): The function to inspect.

        Returns:
            bool: True if its single positional parameter is annotated as a collection.
        """
        # ``from __future__ import annotations`` leaves an annotation as a string, and
        # ``get_type_hints`` resolves that string in the globals of the module the function was
        # defined in. ``induced_fitness`` returns a function annotated ``Tree[Any]`` from a module
        # that imports ``Tree`` under ``TYPE_CHECKING``, so resolving it raises NameError.
        # Returning False is right for that function, which takes one individual.
        try:
            signature = inspect.signature(fitness_function)
            type_hints = get_type_hints(fitness_function)
        except (NameError, TypeError, ValueError):
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

        # ``get_origin(Sequence[Tree])`` is ``collections.abc.Sequence`` and not ``list``, so the
        # abstract containers stand in this set beside the concrete ones. Without them a function
        # annotated ``Sequence[Tree[T]]`` would not be recognized as a batch function.
        collections = {list, tuple, set, frozenset, Iterable, Sequence, Collection}
        origin = get_origin(annotation)
        return origin in collections or annotation in collections

    def _evaluate(
        self,
        fitness_function: Callable[..., Any],
        individuals: Sequence[Tree[T]],
        cache: dict[Tree[T], Fitness],
        mode: FitnessFunctionMode,
    ) -> None:
        """Fill the cache with the fitness of every individual not already in it.

        The cache lives for one run, and it is what makes a batch fitness function worth having:
        an acquisition function evaluates a whole generation in one call, and an individual that
        survived is not paid for twice.

        Args:
            fitness_function (Callable[..., Any]): The quality measure.
            individuals (Sequence[Tree[T]]): The individuals to evaluate.
            cache (dict[Tree[T], Fitness]): The run's cache, updated in place.
            mode (FitnessFunctionMode): How to call the function.

        Raises:
            TypeError: If a batch function does not return a mapping.
            ValueError: If a batch function omits a requested individual.
        """
        missing = list(dict.fromkeys(tree for tree in individuals if tree not in cache))
        if not missing:
            return

        use_batch = mode == "batch" or (mode == "auto" and self._looks_like_batch_fitness_function(fitness_function))
        if not use_batch:
            for tree in missing:
                cache[tree] = fitness_function(tree)
                self._check_orderable(tree, cache[tree])
            return

        evaluated = fitness_function(missing)
        if not isinstance(evaluated, Mapping):
            msg = "a batch fitness function must return a mapping from individual to fitness"
            raise TypeError(msg)
        omitted = [tree for tree in missing if tree not in evaluated]
        if omitted:
            msg = (
                f"a batch fitness function must answer for every individual it was given. "
                f"{len(omitted)} of {len(missing)} are missing from its result"
            )
            raise ValueError(msg)
        cache.update({tree: evaluated[tree] for tree in missing})
        for tree in missing:
            self._check_orderable(tree, cache[tree])

    def _check_orderable(self, individual: Tree[T], fitness: Fitness) -> None:
        """Refuse a fitness value that the comparator cannot place in its order.

        A fitness function maps into a *partially ordered set*, and a partial order is reflexive,
        so every value is comparable to itself. ``nan`` is not. It comes from a measurement that
        failed, and it is outside the codomain rather than at the bottom of it.

        Letting it through is worse than it looks. Nothing is ever fitter than a value that
        compares to nothing, so once such a value holds the best-so-far place nothing dislodges it
        again and it is returned as the fittest individual encountered, and a truncating survivor
        selection ranks it in the top front and drops the genuine optimum for it. A failed
        measurement would decide the search, which is the opposite of staying visible.

        Args:
            individual (Tree[T]): The individual whose fitness was measured.
            fitness (Fitness): The measured value.

        Raises:
            ValueError: If the value is not comparable to itself.
        """
        if self.comparator.compare(fitness, fitness) is not Comparison.EQUAL:
            msg = (
                f"the fitness of {individual} is {fitness!r}, which {type(self.comparator).__name__}"
                " cannot compare to itself. A fitness function maps into a partially ordered set, "
                "and a value outside that order (a failed measurement, typically nan) is not "
                "replaced by a substitute here. Decide what a failure means: drop the individual "
                "before it is evaluated, or give the comparator an order that places it."
            )
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # The order
    # ------------------------------------------------------------------

    def _fittest(self, individuals: Sequence[Tree[T]], fitness: Mapping[Tree[T], Fitness]) -> Tree[T]:
        """Return an individual no other is fitter than.

        Under a partial order there may be several, and the first one in the given order is taken:
        the choice is arbitrary either way, and a deterministic one keeps a seeded run
        reproducible.

        Args:
            individuals (Sequence[Tree[T]]): The individuals to search, non-empty.
            fitness (Mapping[Tree[T], Fitness]): Their fitness.

        Returns:
            Tree[T]: A fittest individual.
        """
        best = individuals[0]
        for candidate in individuals[1:]:
            if self.comparator.compare(fitness[candidate], fitness[best]) is Comparison.GREATER:
                best = candidate
        return best

    def _challenge(
        self,
        incumbent: Fitness,
        pool: Sequence[Tree[T]],
        fitness: Mapping[Tree[T], Fitness],
    ) -> Tree[T] | None:
        """Return a member of the pool fitter than the incumbent, or None.

        The pool is the parents and the offspring together, which is what the run has evaluated in
        this generation. Reading it rather than the survivors is what makes the best-so-far a
        fittest individual the run has seen: a survivor selection keeps mu individuals, so under a
        partial order it may drop an offspring that beats the incumbent, and a dropped individual
        never gets a second chance to be reported.

        The question is whether **a** member is fitter than the incumbent, and under a partial
        order that is not the same as asking whether *the* fittest member is. A scan for one
        maximal element can land on an individual incomparable to the incumbent while another
        member strictly dominates it, and the incumbent would then never be replaced although the
        pool improved on it.

        Among the members that do beat the incumbent, a maximal one is taken, so the incumbent
        improves monotonically. Simply taking a fittest member of the pool would not guarantee
        that under a partial order, since a fittest member may be incomparable to the incumbent.

        Args:
            incumbent (Fitness): The fitness of the current best-so-far.
            pool (Sequence[Tree[T]]): The parents and the offspring together.
            fitness (Mapping[Tree[T], Fitness]): Their fitness.

        Returns:
            Tree[T] | None: A fitter member, or None if none beats the incumbent.
        """
        beating = [
            member for member in pool if self.comparator.compare(fitness[member], incumbent) is Comparison.GREATER
        ]
        return self._fittest(beating, fitness) if beating else None

    # ------------------------------------------------------------------
    # The algorithm
    # ------------------------------------------------------------------

    def _variation_pass(
        self,
        query: ResolutionQuery[NT, T, G],
        population: Sequence[Tree[T]],
        fitness: Mapping[Tree[T], Fitness],
    ) -> list[Tree[T]]:
        """Run one pass of the inner loop.

        Args:
            query (ResolutionQuery[NT, T, G]): The generator query of the search space.
            population (Sequence[Tree[T]]): The population to draw parents from.
            fitness (Mapping[Tree[T], Fitness]): Its fitness.

        Returns:
            list[Tree[T]]: The batch the pass produced, or an empty list if the pass is to be
                discarded. A pass is complete when every member of the batch the recombination
                step produced is still there after the mutation step: two for a swap or a pair of
                copies, one for a graft. The rule is the completeness of the batch rather than its
                size, because the swap and the graft fill the same slot and produce batches of
                different sizes.
        """
        first_parent, second_parent = self.parent_selection.select_parents(population, fitness, self.comparator)
        # ``mu = 1`` has no second parent, since ``select_parents`` returns the same individual
        # twice. Handing that pair to a binary operator asks it to mix an individual with itself,
        # and every way that can go is wrong. ``SubtreeSwap`` then draws from the pairs of the
        # individual's own inner positions. A pair of two distinct ones puts one of its subterms
        # into another of its own positions, which is a mutation wearing a crossover's name. An
        # individual with a single inner position leaves only the pair of that position with
        # itself, which exchanges a subterm with itself and leaves both children equal to it. An
        # individual with no inner position leaves no pair at all, so the batch is empty, the pass
        # is discarded, and at ``p_c = 1`` nothing else fills the generation and the run dies at
        # the attempt cap below.
        #
        # So recombination is skipped rather than attempted. This is not a silent substitution.
        # ``p_c`` cannot mean anything at ``mu = 1`` under any implementation, the operator being a
        # binary one where there is only one individual, so nothing is being decided here that the
        # configuration left open. **A run that wants both operators needs ``mu >= 2``**, and that
        # is a fact about the operator, not a setting of this class.
        can_recombine = len(population) >= self._RECOMBINATION_ARITY
        if can_recombine and self.rng.random() < self.crossover_rate:
            batch = self.recombination.recombine(query, first_parent, second_parent)
        elif can_recombine:
            batch = [first_parent, second_parent]
        else:
            # One parent in, one offspring out, and the mutation below is what varies it. Handing
            # back two copies would let a (1+lambda) run fill its pool with duplicates.
            batch = [first_parent]
        if not batch:
            return []

        varied: list[Tree[T]] = []
        for offspring in batch:
            if self.rng.random() < self.mutation_rate:
                mutant = self.mutation.mutate(query, offspring)
                if mutant is not None:
                    varied.append(mutant)
            else:
                varied.append(offspring)
        return varied if len(varied) == len(batch) else []

    def evolutionary_stream(
        self,
        query: ResolutionQuery[NT, T, G],
        fitness_function: Callable[..., Any],
        fitness_function_mode: FitnessFunctionMode = "auto",
    ) -> Iterator[EAState[T]]:
        """Run the search, yielding one state per generation.

        Args:
            query (ResolutionQuery[NT, T, G]): The generator query naming the search space and the
                type the run works over.
            fitness_function (Callable[..., Any]): The quality measure ``q``. Either one
                individual at a time, or a whole list returning a mapping.
            fitness_function_mode (FitnessFunctionMode): How to call it. "auto" reads the
                annotation of its parameter, and where it cannot, calls the function with one
                individual at a time. (Default value = 'auto')

        Yields:
            EAState[T]: The initial population, then one state per generation, ending with the
                state on which the termination condition first held.

        Raises:
            RuntimeError: If a generation cannot fill its offspring within the attempt cap.
        """
        population = self.initializer.initialize(query, self.population_size)
        # The initializer's contract is "exactly ``size`` individuals". A population shorter than
        # mu runs generation 0 at a size nobody configured, and at a population of one the passes
        # that build generation 1 skip recombination. An empty population leaves ``_fittest`` no
        # individual to return. This is enforced here rather than trusted, because the initializer
        # is a parameter and may be the caller's own.
        if len(population) != self.population_size:
            msg = (
                f"{type(self.initializer).__name__} returned {len(population)} individuals for a "
                f"population of {self.population_size}"
            )
            raise ValueError(msg)
        cache: dict[Tree[T], Fitness] = {}
        self._evaluate(fitness_function, population, cache, fitness_function_mode)

        best = self._fittest(population, cache)
        state = EAState(
            generation=0,
            population=population,
            fitness={tree: cache[tree] for tree in population},
            offspring=[],
            best=best,
            best_fitness=cache[best],
            last_improvement=0,
        )

        while True:
            yield state
            if self.termination.is_satisfied(state):
                return

            offspring: list[Tree[T]] = []
            attempts = 0
            cap = self.attempt_factor * self.population_size
            while len(offspring) < self.population_size:
                if attempts >= cap:
                    msg = (
                        f"generation {state.generation + 1} produced {len(offspring)} of "
                        f"{self.population_size} offspring in {attempts} passes. Variation is "
                        "failing systematically. The parents may have grown past the sampler's "
                        "bound, or the recombination operator may find no acceptable pair. The "
                        "population is not filled up with unchanged parents."
                    )
                    raise RuntimeError(msg)
                attempts += 1
                offspring.extend(self._variation_pass(query, state.population, state.fitness))

            self._evaluate(fitness_function, offspring, cache, fitness_function_mode)
            pool = [*state.population, *offspring]
            pool_fitness = {tree: cache[tree] for tree in pool}
            population = self.survivor_selection.select_survivors(
                state.population,
                offspring,
                pool_fitness,
                self.comparator,
                self.population_size,
            )
            # The contract is "mu individuals among them". A component returning a different
            # number would run the search at a population size nobody configured, and one returning
            # a stranger would break closure. Both halves are enforced here rather than trusted,
            # because the component is a parameter and may be the caller's own.
            if len(population) != self.population_size:
                msg = (
                    f"{type(self.survivor_selection).__name__} returned {len(population)} "
                    f"individuals for a population of {self.population_size}"
                )
                raise ValueError(msg)
            strangers = [tree for tree in population if tree not in pool_fitness]
            if strangers:
                msg = (
                    f"{type(self.survivor_selection).__name__} returned {len(strangers)} "
                    "individuals that are in neither the parents nor the offspring. A survivor "
                    "selection chooses among what it was given"
                )
                raise ValueError(msg)

            generation = state.generation + 1
            population_fitness = {tree: cache[tree] for tree in population}
            challenger = self._challenge(state.best_fitness, pool, pool_fitness)
            state = EAState(
                generation=generation,
                population=population,
                fitness=population_fitness,
                offspring=offspring,
                best=challenger if challenger is not None else state.best,
                best_fitness=(pool_fitness[challenger] if challenger is not None else state.best_fitness),
                last_improvement=(generation if challenger is not None else state.last_improvement),
            )

    def evolutionary_best(
        self,
        query: ResolutionQuery[NT, T, G],
        fitness_function: Callable[..., Any],
        fitness_function_mode: FitnessFunctionMode = "auto",
    ) -> Tree[T]:
        """Run the search and return the fittest individual encountered.

        Not the best of the final generation. The best-so-far individual is carried across the
        whole run, so an individual a later generation dropped is still the answer if nothing beat
        it.

        Args:
            query (ResolutionQuery[NT, T, G]): The generator query of the search space.
            fitness_function (Callable[..., Any]): The quality measure.
            fitness_function_mode (FitnessFunctionMode): How to call it. (Default value = 'auto')

        Returns:
            Tree[T]: The fittest individual encountered. There is always one, initialization either
                filling the population or raising.
        """
        final: EAState[T] = cast(
            "EAState[T]",
            _last(self.evolutionary_stream(query, fitness_function, fitness_function_mode)),
        )
        return final.best


def _last(values: Iterator[_T]) -> _T:
    """Return the last element of a non-empty iterator.

    Args:
        values (Iterator[_T]): The iterator. The caller guarantees at least one element.

    Returns:
        _T: Its last element.
    """
    last: _T | None = None
    for value in values:
        last = value
    return cast("_T", last)
