"""Mutation: one position, one residual query.

Resolution mutation discards the subterm at a drawn position, which leaves a hole behind, and the
*residual* of the language at the remaining term describes exactly the replacements that keep the
individual inside the language. The partial-term query streams the completions of the opened term,
and a sampler on that query draws the offspring outright. The operator therefore carries a sampler
as its parameter and needs nothing else from the search space.

Three properties follow, and each of them is a departure from what a tree-swapping mutation does:

* **Closure by construction.** The query delivers completions and nothing else, so the offspring is
  an inhabitant without a membership test. Recombination is closed by rejection instead. The two
  are not the same mechanism and the difference is worth naming.
* **The position ranges over the term's non-leaves, root included.** At the root the whole
  individual is replaced by a fresh variable, the residual query *is* the generator query, and an
  exhaustive sampler then reaches every inhabitant within its bound from every other. An operator
  restricted to inner positions keeps the root symbol of its parent, so a population whose members
  agree on their root symbol would never leave that part of the language, at any mutation rate.
  :class:`ResolutionMutation` carries the reasons for excluding the leaves.
* **One position, one request, no retry.** If the request delivers no inhabitant, which happens
  when no completion of the opened term lies within the bound, the operator returns no offspring.
  That is an ordinary case rather than an error, and the surrounding procedure draws new parents.
  Trying position after position until one succeeds would replace the uniform position distribution
  with a distribution nobody stated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from cosy.core.solution_space import NT, G, T
from cosy.search.queries import residual_query

if TYPE_CHECKING:
    import random

    from cosy.core.tree import Path, Tree
    from cosy.search.queries import ResolutionQuery
    from cosy.search.samplers import Sampler

__all__ = ["Mutation", "ResolutionMutation"]


@runtime_checkable
class Mutation(Protocol[NT, T, G]):
    """A unary variation operator: one individual to at most one offspring.

    Mutation is unary and recombination is binary, and that is the whole of the distinction between
    the two component slots.
    """

    def mutate(self, query: ResolutionQuery[NT, T, G], individual: Tree[T]) -> Tree[T] | None:
        """Produce an offspring of one individual.

        Args:
            query (ResolutionQuery[NT, T, G]): The generator query of the search space; the
                operator derives its own queries from the space and type it names.
            individual (Tree[T]): The parent.

        Returns:
            Tree[T] | None: The offspring, or None if the operator produced none.
        """
        ...


class ResolutionMutation(Mutation[NT, T, G]):
    """Replace the subterm at a uniformly drawn position by a draw from the residual.

    The offspring is the first element of the sampler's stream on the residual query at the drawn
    position: one request, and whatever it delivers.

    The distribution over the offspring is therefore the product of two. The position is uniform,
    and everything below it is the sampler's. Both are parameters, and only the first is fixed here.

    **Which positions.** Uniform over the positions that are not leaves, and over the root, which
    stays a candidate even when it is one. Two reasons, and the second is the one that decides it:

    * Uniform choice over all positions favors the leaves, since a branching term has most of its
      positions there. The same bias is reported for grammar-guided genetic programming.
    * A leaf is where the operator has the least to do and, at a literal, nothing at all. A constant
      argument becomes a childless node, and the clause matcher pins a constant argument to its
      value even at the opened position, so the residual query there answers with the term already
      present. On a repository built from literal parameters that is a large share of the positions,
      every one of them a draw spent on the identity.

    The root is the exception and it has to be, because the residual query at the root *is* the
    generator query. Excluding it would break the reachability that carries every individual within
    the bound to every other, and excluding leaves outright would leave a single-node individual,
    root and leaf at once, with no position to draw.

    Attributes:
        sampler (Sampler): The sampler drawn from on the residual query. It bounds the whole
            individual rather than the replacement: on a residual query the query term is the
            entire term, so a bound of the sampler is a bound on the offspring.
        rng (random.Random): The source of randomness for the position.
    """

    def __init__(self, sampler: Sampler, rng: random.Random) -> None:
        """Build the operator.

        Args:
            sampler (Sampler): The sampler. A sampler is bounded by definition, so every request
                halts, which is what makes "no offspring" an observation rather than a hang.
            rng (random.Random): The source of randomness for the position.
        """
        self.sampler = sampler
        self.rng = rng

    @staticmethod
    def mutation_points(individual: Tree[T]) -> frozenset[Path]:
        """Return the positions the operator draws among: the non-leaves, and the root.

        A set rather than a filtered list, and a method rather than an expression, because it is
        the operator's position distribution written down, the one parameter this class fixes.

        Args:
            individual (Tree[T]): The parent.

        Returns:
            frozenset[Path]: The eligible positions. Never empty, since the root is always among
                them, so every individual offers a draw.
        """
        return (individual.positions() - individual.leaf_positions()) | {()}

    def mutate(self, query: ResolutionQuery[NT, T, G], individual: Tree[T]) -> Tree[T] | None:
        """Draw a position uniformly among the mutation points and complete the residual there.

        A single-node individual has exactly one position, which is its root and its only leaf.
        It is an ordinary input: the root stays eligible, the residual query there is the
        generator query, and the draw regenerates the individual.

        Args:
            query (ResolutionQuery[NT, T, G]): The generator query of the search space.
            individual (Tree[T]): The parent.

        Returns:
            Tree[T] | None: The offspring, or None if the request delivered no inhabitant, which
                is to say that no completion of the opened term lies within the sampler's bound.
        """
        # Sorted before the draw so that the choice depends on the rng alone and not on the
        # iteration order of the set. The root is one of the candidates, which is the point.
        position = self.rng.choice(sorted(self.mutation_points(individual)))
        opened = residual_query(query.solution_space, query.start, individual, position)
        return next(iter(self.sampler.sample(opened)), None)
