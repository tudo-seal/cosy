"""Search over synthesized solution spaces.

A resolution query denotes what is asked of a solution space, and a search rule traverses the
derivation tree of that query lazily, streaming the inhabitants its success branches determine.
This package carries the query vocabulary (generator, checker, partial-term query), the reading
of a search node as the partial inhabitant it denotes, the tree kernels that score such a node by
its similarity to a set of reference terms, the two uninformed search rules with the clause orders
they are built from, and the branch counts that say how many inhabitants a node still reaches,
which is what a search has to weight its choices by in order to draw from a chosen distribution.
It carries the cost layer an informed rule reads its order off, which is to say the orders a cost
function may map into, the best-first frontier over them, and the additive cost algebras that split
the cost of a search node into what its partial inhabitant has already cost and what its holes are
estimated to add. It also carries random search itself, which is best-first search under a
randomizing cost function, and the samplers the evolutionary and Bayesian methods draw their
populations from, one of them under a prescribed distribution on an additive cost, and a Markov
chain on terms that walks to such a distribution where random search would draw from it, and
particles that walk a search tree on estimated weights and are weighted to the distribution, and the
counts of a cost rounded to units, which bound the counts of the cost itself and reach its distribution by
rejection. Beside all of
these stands the one rule that traverses no derivation tree at all:
bottom-up search, which iterates the immediate consequence operator of a program to the least
Herbrand model and reads the inhabitants off it.

One thing here searches nothing. Determinization rewrites a program before any search runs. A
predicate that factors through a finite abstraction is compiled into the non-terminals by a product
construction, and what comes out carries no predicate over a hole, so the branch counts apply to a
program whose original form they refuse.
"""

from cosy.search.bottom_up import BottomUpCounters, bottom_up, least_herbrand_model
from cosy.search.cost_tables import CostTable, WeightedCostTable, cost_table, weighted_cost_table
from cosy.search.costs import (
    AdditiveCostAlgebra,
    ComponentwiseTuples,
    CostDomain,
    CostFunction,
    CostOrder,
    Frontier,
    HeapFrontier,
    LinearScanFrontier,
    NonNegativeReals,
    Reals,
    a_star,
    assert_uniform_cost_complete,
    best_first,
    best_first_frontier,
    greedy,
    uniform_cost,
)
from cosy.search.counting import (
    CountedNode,
    CoupledClause,
    SizeTable,
    assert_unambiguous_within,
    branch_counts,
    branch_multiplicities,
    coupled_clauses,
    decomposable_or_raise,
    retained_node_count,
    rule_cost,
    size_table,
)
from cosy.search.determinize import (
    Determinization,
    MergedNonTerminal,
    ProductNonTerminal,
    determinize,
    recognizable_or_raise,
    unabstracted_clauses,
)
from cosy.search.gumbel import condition_on_maximum, gumbel_key, gumbel_noise
from cosy.search.kernels import k_sst, k_st, normalized, reference_score
from cosy.search.markov import (
    ChainStep,
    MetropolisChain,
    VisitCounts,
    WangLandau,
    counts_from_visits,
    metropolis_chain,
    wang_landau,
)
from cosy.search.partial import Hole, holes, partial_inhabitant, term_depth, term_size
from cosy.search.queries import ResolutionQuery, checker, generator_query, residual_query
from cosy.search.rounding import BinBounds, RoundedRejection, bin_count_bounds, rounded_rejection
from cosy.search.rules import (
    breadth_first,
    deepest_first_subgoal,
    depth_first,
    fewest_arguments_first,
    uniform_random_clause_order,
)
from cosy.search.samplers import (
    CostTableSampler,
    DepthBoundedRandomSampler,
    MarkovChainSampler,
    Sampler,
    SizeUniformSampler,
    SMCSampler,
    TiltSampler,
)
from cosy.search.sampling import (
    WeightedTable,
    WeightedTree,
    random_search,
    random_search_keyed,
    size_uniform,
    weighted_table,
    weighted_tree,
)
from cosy.search.smc import Guided, WeightedParticles, sequential_monte_carlo
from cosy.search.tilt import (
    SaddleCounts,
    SaddleGrid,
    SaddleSearch,
    TargetOutOfReach,
    TiltedMixture,
    TiltedSearch,
    TiltProgram,
    TiltTable,
    saddle_counts,
    saddle_grid,
    saddle_mixture,
    saddle_search,
    theta_for_mean,
    tilt_program,
    tilt_table,
    tilted_mixture,
    tilted_search,
)

__all__ = [
    "AdditiveCostAlgebra",
    "BinBounds",
    "BottomUpCounters",
    "ChainStep",
    "ComponentwiseTuples",
    "CostDomain",
    "CostFunction",
    "CostOrder",
    "CostTable",
    "CostTableSampler",
    "CountedNode",
    "CoupledClause",
    "DepthBoundedRandomSampler",
    "Determinization",
    "Frontier",
    "Guided",
    "HeapFrontier",
    "Hole",
    "LinearScanFrontier",
    "MarkovChainSampler",
    "MergedNonTerminal",
    "MetropolisChain",
    "NonNegativeReals",
    "ProductNonTerminal",
    "Reals",
    "ResolutionQuery",
    "RoundedRejection",
    "SMCSampler",
    "SaddleCounts",
    "SaddleGrid",
    "SaddleSearch",
    "Sampler",
    "SizeTable",
    "SizeUniformSampler",
    "TargetOutOfReach",
    "TiltProgram",
    "TiltSampler",
    "TiltTable",
    "TiltedMixture",
    "TiltedSearch",
    "VisitCounts",
    "WangLandau",
    "WeightedCostTable",
    "WeightedParticles",
    "WeightedTable",
    "WeightedTree",
    "a_star",
    "assert_unambiguous_within",
    "assert_uniform_cost_complete",
    "best_first",
    "best_first_frontier",
    "bin_count_bounds",
    "bottom_up",
    "branch_counts",
    "branch_multiplicities",
    "breadth_first",
    "checker",
    "condition_on_maximum",
    "cost_table",
    "counts_from_visits",
    "coupled_clauses",
    "decomposable_or_raise",
    "deepest_first_subgoal",
    "depth_first",
    "determinize",
    "fewest_arguments_first",
    "generator_query",
    "greedy",
    "gumbel_key",
    "gumbel_noise",
    "holes",
    "k_sst",
    "k_st",
    "least_herbrand_model",
    "metropolis_chain",
    "normalized",
    "partial_inhabitant",
    "random_search",
    "random_search_keyed",
    "recognizable_or_raise",
    "reference_score",
    "residual_query",
    "retained_node_count",
    "rounded_rejection",
    "rule_cost",
    "saddle_counts",
    "saddle_grid",
    "saddle_mixture",
    "saddle_search",
    "sequential_monte_carlo",
    "size_table",
    "size_uniform",
    "term_depth",
    "term_size",
    "theta_for_mean",
    "tilt_program",
    "tilt_table",
    "tilted_mixture",
    "tilted_search",
    "unabstracted_clauses",
    "uniform_cost",
    "uniform_random_clause_order",
    "wang_landau",
    "weighted_cost_table",
    "weighted_table",
    "weighted_tree",
]
