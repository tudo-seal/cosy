"""Search over synthesized solution spaces.

A resolution query denotes what is asked of a solution space, and
``SolutionSpace.resolution`` traverses the derivation tree of that query lazily, streaming the
inhabitants its success branches determine.
This package carries the query vocabulary (generator, checker, partial-term query) and the
reading of a search node as the partial inhabitant it denotes.
"""

from cosy.search.partial import Hole, holes, partial_inhabitant, term_depth, term_size
from cosy.search.queries import ResolutionQuery, checker, generator_query, residual_query

__all__ = [
    "Hole",
    "ResolutionQuery",
    "checker",
    "generator_query",
    "holes",
    "partial_inhabitant",
    "residual_query",
    "term_depth",
    "term_size",
]
