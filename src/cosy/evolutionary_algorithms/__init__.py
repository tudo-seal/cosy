"""Evolutionary search over synthesized search spaces.

The components that touch the search space reach it only through resolution queries, which is what
lets each of them work on any synthesized search space. The initializer poses the generator query,
mutation poses a residual query per call, and recombination decides acceptance with a query. The
representation is the inhabitants themselves. Selection, termination and evaluation see individuals
and fitness values and never the space at all, and they are filled with standard methods, kept here
for what each of them contributes to the convergence argument below.

| Component class | What fills it |
| --- | --- |
| Representation | the inhabitants themselves, with no encoding detour |
| Initialization | :class:`SampledInitialization`, :class:`MixtureInitializer` |
| Mutation | :class:`ResolutionMutation` |
| Recombination | :class:`SubtreeSwap`, :class:`SubtreeGraft` |
| Parent selection | :class:`TournamentSelection`, :class:`RankBasedSelection`, :class:`FitnessProportionalSelection` |
| Survivor selection | :class:`GenerousConservativeReplacement`, :class:`FitnessBasedReplacement` |
| Termination | :class:`Generations`, :class:`NoImprovement` |
| Evaluation | :class:`FitnessComparator`, plus a :class:`Scalarization` where numbers are needed |

:class:`EvolutionarySearch` assembles them into one algorithm. The samplers the
initializer and the mutation draw from live in :mod:`cosy.search.samplers`, because a sampler is a
search-rule construction rather than an evolutionary one.

**Almost sure convergence.** A run reaches a fittest individual with probability one if five
conditions hold together, and every component that bears on one of them says so in its own
docstring:

1. **Bounded.** The individuals the run can hold form a finite set.
2. **Reachable.** The sampler the initializer and the mutation draw from can produce every one of
   them.
3. **A generous parent selection.** Every member of the population becomes a parent with positive
   probability.
4. **A generous and conservative replacement.** Every individual survives with positive
   probability, and an individual of greatest scalarized fitness always survives.
5. **Neither rate is degenerate.** The crossover rate stays below 1, so that a pass can copy its
   parents, and the mutation rate stays above 0, so that mutation happens at all.

Two of the five are easy to miss:

* **Tournament selection is not generous** for a tournament size of 2 or more. The least fit member
  of a population wins no tournament it can enter, so its probability is zero.
  :class:`FitnessProportionalSelection` and :class:`RankBasedSelection` below a selection pressure
  of 2 are the two that give every member a positive share.
* **The two bounds must measure the same thing.** Reachability asks that every individual the run
  can hold lie within the bound *of the sampler*. The ``max_size`` of the recombination acceptance
  test establishes that against a :class:`~cosy.search.samplers.SizeUniformSampler`, which bounds
  the same quantity. Combined with a :class:`~cosy.search.samplers.DepthBoundedRandomSampler` it
  does not, because a swap offspring can stay within a size bound and exceed a depth bound. The
  other route is a search space that is finite by its modeling, and then neither bound is needed
  for the condition.
"""

from cosy.evolutionary_algorithms.evolutionary import (
    EAState,
    EvolutionarySearch,
    FitnessFunctionMode,
)
from cosy.evolutionary_algorithms.fitness import (
    Comparison,
    ExpScalarization,
    Fitness,
    FitnessComparator,
    ParetoFitnessComparator,
    ScalarFitnessComparator,
    Scalarization,
    induced_fitness,
)
from cosy.evolutionary_algorithms.initialisation import (
    InitializationError,
    Initializer,
    MixtureInitializer,
    SampledInitialization,
)
from cosy.evolutionary_algorithms.mutation import Mutation, ResolutionMutation
from cosy.evolutionary_algorithms.recombination import (
    Recombination,
    SubtreeGraft,
    SubtreeSwap,
)
from cosy.evolutionary_algorithms.selection import (
    FitnessBasedReplacement,
    FitnessProportionalSelection,
    GenerousConservativeReplacement,
    ParentSelection,
    RankBasedSelection,
    SurvivorSelection,
    TournamentSelection,
    dominance_fronts,
)
from cosy.evolutionary_algorithms.termination import (
    Generations,
    NoImprovement,
    Termination,
)

__all__ = [
    "Comparison",
    "EAState",
    "EvolutionarySearch",
    "ExpScalarization",
    "Fitness",
    "FitnessBasedReplacement",
    "FitnessComparator",
    "FitnessFunctionMode",
    "FitnessProportionalSelection",
    "Generations",
    "GenerousConservativeReplacement",
    "InitializationError",
    "Initializer",
    "MixtureInitializer",
    "Mutation",
    "NoImprovement",
    "ParentSelection",
    "ParetoFitnessComparator",
    "RankBasedSelection",
    "Recombination",
    "ResolutionMutation",
    "SampledInitialization",
    "ScalarFitnessComparator",
    "Scalarization",
    "SubtreeGraft",
    "SubtreeSwap",
    "SurvivorSelection",
    "Termination",
    "TournamentSelection",
    "dominance_fronts",
    "induced_fitness",
]
