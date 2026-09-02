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
randomizing cost function, and the two samplers the evolutionary and Bayesian methods draw their
populations from. Beside all of these stands the one rule that traverses no derivation tree at all:
bottom-up search, which iterates the immediate consequence operator of a program to the least
Herbrand model and reads the inhabitants off it.
"""

from cosy.search.bottom_up import BottomUpCounters, bottom_up, least_herbrand_model
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
    size_table,
)
from cosy.search.gumbel import condition_on_maximum, gumbel_key, gumbel_noise
from cosy.search.kernels import k_sst, k_st, normalized, reference_score
from cosy.search.partial import Hole, holes, partial_inhabitant, term_depth, term_size
from cosy.search.queries import ResolutionQuery, checker, generator_query, residual_query
from cosy.search.rules import (
    breadth_first,
    deepest_first_subgoal,
    depth_first,
    fewest_arguments_first,
    uniform_random_clause_order,
)
from cosy.search.samplers import DepthBoundedRandomSampler, Sampler, SizeUniformSampler
from cosy.search.sampling import (
    WeightedTable,
    WeightedTree,
    random_search,
    random_search_keyed,
    size_uniform,
    weighted_table,
    weighted_tree,
)

__all__ = [
    "AdditiveCostAlgebra",
    "BottomUpCounters",
    "ComponentwiseTuples",
    "CostDomain",
    "CostFunction",
    "CostOrder",
    "CountedNode",
    "CoupledClause",
    "DepthBoundedRandomSampler",
    "Frontier",
    "HeapFrontier",
    "Hole",
    "LinearScanFrontier",
    "NonNegativeReals",
    "Reals",
    "ResolutionQuery",
    "Sampler",
    "SizeTable",
    "SizeUniformSampler",
    "WeightedTable",
    "WeightedTree",
    "a_star",
    "assert_unambiguous_within",
    "assert_uniform_cost_complete",
    "best_first",
    "best_first_frontier",
    "bottom_up",
    "branch_counts",
    "branch_multiplicities",
    "breadth_first",
    "checker",
    "condition_on_maximum",
    "coupled_clauses",
    "decomposable_or_raise",
    "deepest_first_subgoal",
    "depth_first",
    "fewest_arguments_first",
    "generator_query",
    "greedy",
    "gumbel_key",
    "gumbel_noise",
    "holes",
    "k_sst",
    "k_st",
    "least_herbrand_model",
    "normalized",
    "partial_inhabitant",
    "random_search",
    "random_search_keyed",
    "reference_score",
    "residual_query",
    "retained_node_count",
    "size_table",
    "size_uniform",
    "term_depth",
    "term_size",
    "uniform_cost",
    "uniform_random_clause_order",
    "weighted_table",
    "weighted_tree",
]
