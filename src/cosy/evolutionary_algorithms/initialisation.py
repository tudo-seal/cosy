"""Initialization: a population size mapped to a population of individuals.

A *population* is a finite multiset of individuals, and an *initializer* is a map from a population
size to a population. :class:`SampledInitialization` poses the generator query and collects a
prefix of a sampler's stream. :class:`MixtureInitializer` combines two initializers, which is what
ramped half-and-half becomes once its two methods are read as initializers of their own.

**Both failure clauses are errors.** Initialization fails if fewer than the requested number of
inhabitants lie within the bound of the sampler, and it fails if the stream runs dry before the
population is full. Neither shrinks the population. A caller that asked for mu individuals and
receives fewer is running a different algorithm than the one it configured, so the population size
is not a quantity to substitute a smaller value for.

The first clause is the one that carries weight, and it carries it for the depth-bounded sampler.
That sampler draws independently, so it would fill a population of any size with repeats of
finitely many terms and never end its stream. Only the count within the bound tells the caller that
the space is too small for the population it wants. The size-uniform stream ends on its own when
the inhabitants run out, so there the second clause would catch it too. The check stays because it
fails before the run rather than during it.
"""

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from cosy.core.solution_space import NT, G, T

if TYPE_CHECKING:
    import random

    from cosy.core.tree import Tree
    from cosy.search.queries import ResolutionQuery
    from cosy.search.samplers import Sampler

__all__ = [
    "InitializationError",
    "Initializer",
    "MixtureInitializer",
    "SampledInitialization",
]


class InitializationError(RuntimeError):
    """Raised when an initializer cannot deliver the population it was asked for.

    A failure rather than a short population. A space that cannot fill the population needs none of
    the machinery here, since its inhabitants within the bound can be enumerated outright, so the
    condition marks a modeling boundary and is worth reporting as one.
    """


@runtime_checkable
class Initializer(Protocol[NT, T, G]):
    """A map from a population size to a population.

    The initializer reaches the search space the way every operator here does, through a resolution
    query, and returns a multiset, which is to say a list in which an individual may repeat.
    """

    def initialize(self, query: ResolutionQuery[NT, T, G], size: int) -> list[Tree[T]]:
        """Build a population of ``size`` individuals.

        Args:
            query (ResolutionQuery[NT, T, G]): The generator query of the search space.
            size (int): The population size mu.

        Returns:
            list[Tree[T]]: Exactly ``size`` individuals, repetitions included.

        Raises:
            InitializationError: If the population cannot be filled.
        """
        ...


class SampledInitialization(Initializer[NT, T, G]):
    """Collect a population from one stream of a sampler.

    The initializer poses the generator query once and requests one element after the other until
    the population is full. **One stream, not mu draws**, and the distinction is not cosmetic. Each
    call to :meth:`~cosy.search.samplers.Sampler.sample` re-poses the query, and on a search space
    of realistic size the query is what the draw costs, so mu separate calls pay for it mu times.
    The single stream is also what makes a size-uniform population a sample without replacement:
    mu independent streams would repeat.

    A counting sampler keeps the construction it built for the query alive behind the initializer,
    which is the point on a repeated draw but is also real memory on a large space. A caller that
    is done initializing can release it through the sampler.

    Attributes:
        sampler (Sampler): The sampler whose stream the population is a prefix of.
    """

    def __init__(self, sampler: Sampler) -> None:
        """Build the initializer.

        Args:
            sampler (Sampler): The sampler. It carries the bound, and the bound is what makes
                both failure clauses decidable.
        """
        self.sampler = sampler

    def initialize(self, query: ResolutionQuery[NT, T, G], size: int) -> list[Tree[T]]:
        """Collect ``size`` individuals from the sampler's stream on the query.

        Args:
            query (ResolutionQuery[NT, T, G]): The generator query of the search space.
            size (int): The population size mu.

        Returns:
            list[Tree[T]]: Exactly ``size`` individuals, in the order the stream delivered them.

        Raises:
            ValueError: If ``size`` is negative.
            InitializationError: If fewer than ``size`` inhabitants lie within the bound of the
                sampler, or if the stream ended before the population was full.
        """
        if size < 0:
            msg = f"a population size cannot be negative: {size}"
            raise ValueError(msg)
        if not self.sampler.at_least(query, size):
            msg = (
                f"fewer than {size} inhabitants lie within the bound of {self.sampler!r}; "
                "a population larger than the bounded search space is an error, not a "
                "degenerate draw. Widen the bound, or enumerate the space outright"
            )
            raise InitializationError(msg)
        population = list(islice(self.sampler.sample(query), size))
        if len(population) < size:
            msg = (
                f"the stream of {self.sampler!r} ended after {len(population)} of {size} "
                "requested individuals; the population is not filled with substitutes"
            )
            raise InitializationError(msg)
        return population


class MixtureInitializer(Initializer[NT, T, G]):
    """Split the population binomially between two initializers.

    For a population size mu the mixture draws ``k`` from the binomial distribution with mu trials
    and success probability ``p``, and returns the individuals of ``first(k)`` and
    ``second(mu - k)`` together. Ramped half-and-half is this component at ``p = 1/2`` over the two
    classical methods. The pairing it is built for is sampled initialization with the size-uniform
    sampler against sampled initialization with the depth-bounded one.

    Each component checks the bound of its own sampler on the exact count it was handed, and that
    is why the mixture combines *initializers* rather than samplers. A mixture of samplers would
    carry a probability where the bound check needs a bound, and its two leaves may carry bounds of
    different kinds, a size bound and a depth bound, exactly the pairing above. A failure of either
    component is a failure of the mixture and is not caught here.

    Attributes:
        probability (float): The success probability ``p`` of the binomial split.
        first (Initializer[NT, T, G]): The initializer that receives the successes.
        second (Initializer[NT, T, G]): The initializer that receives the rest.
        rng (random.Random): The source of randomness for the split.
    """

    def __init__(
        self,
        probability: float,
        first: Initializer[NT, T, G],
        second: Initializer[NT, T, G],
        rng: random.Random,
    ) -> None:
        """Build the mixture.

        Args:
            probability (float): The success probability ``p``, in [0, 1].
            first (Initializer[NT, T, G]): The initializer for the successes.
            second (Initializer[NT, T, G]): The initializer for the remaining places.
            rng (random.Random): The source of randomness for the split.

        Raises:
            ValueError: If ``probability`` lies outside [0, 1].
        """
        if not 0.0 <= probability <= 1.0:
            msg = f"a probability lies in [0, 1]: {probability}"
            raise ValueError(msg)
        self.probability = probability
        self.first = first
        self.second = second
        self.rng = rng

    def initialize(self, query: ResolutionQuery[NT, T, G], size: int) -> list[Tree[T]]:
        """Draw the split and let both components fill their share.

        The binomial draw is mu Bernoulli trials rather than a closed-form inverse. cosy carries no
        runtime dependencies, and mu is a population size, so the linear cost is not worth an
        approximation that would change the distribution.

        Args:
            query (ResolutionQuery[NT, T, G]): The generator query of the search space.
            size (int): The population size mu.

        Returns:
            list[Tree[T]]: The individuals of both components together, the successes first.

        Raises:
            ValueError: If ``size`` is negative.
            InitializationError: If either component cannot fill its share. The failure is not
                caught here.
        """
        if size < 0:
            msg = f"a population size cannot be negative: {size}"
            raise ValueError(msg)
        successes = sum(1 for _ in range(size) if self.rng.random() < self.probability)
        return self.first.initialize(query, successes) + self.second.initialize(query, size - successes)
