"""Search over synthesized solution spaces.

A resolution query denotes what is asked of a solution space, and a search rule traverses the
derivation tree of that query lazily, streaming the inhabitants its success branches determine.
This package carries the query vocabulary (generator, checker, partial-term query), the reading
of a search node as the partial inhabitant it denotes, the tree kernels that score such a node by
its similarity to a set of reference terms, the two uninformed search rules with the clause orders
they are built from, and the branch counts that say how many inhabitants a node still reaches,
which is what a search has to weight its choices by in order to draw from a chosen distribution.
"""

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

__all__ = [
    "CountedNode",
    "CoupledClause",
    "Hole",
    "ResolutionQuery",
    "SizeTable",
    "assert_unambiguous_within",
    "branch_counts",
    "branch_multiplicities",
    "breadth_first",
    "checker",
    "coupled_clauses",
    "decomposable_or_raise",
    "deepest_first_subgoal",
    "depth_first",
    "fewest_arguments_first",
    "generator_query",
    "holes",
    "k_sst",
    "k_st",
    "normalized",
    "partial_inhabitant",
    "reference_score",
    "residual_query",
    "retained_node_count",
    "size_table",
    "term_depth",
    "term_size",
    "uniform_random_clause_order",
]
