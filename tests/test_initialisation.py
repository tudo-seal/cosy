"""Sampled initialization and the mixture.

Two things are pinned here that the previous initializer did not do. It collected what it could and
yielded a short population, and it called the sampler once per individual. Both are wrong. A
population that cannot be filled is a *failure*, in either of two clauses, and the individuals come
from **one stream**. The singular is what makes a size-uniform prefix a sample without replacement,
and on a realistic space it is also the difference between one query and mu of them.
"""

import random

import pytest

from cosy.core.tree import Tree
from cosy.evolutionary_algorithms import (
    InitializationError,
    Initializer,
    MixtureInitializer,
    SampledInitialization,
)
from cosy.search import (
    DepthBoundedRandomSampler,
    SizeUniformSampler,
    checker,
    generator_query,
    term_size,
)
from tests.ea_fixtures import (
    NULLARY_START,
    RECURSIVE_START,
    a2,
    b2,
    nullary_space,
    recursive_space,
)


@pytest.fixture
def tiny():
    """Return the query on the smallest space, which holds two terms of size 1.

    Returns:
        ResolutionQuery: The generator query on ``A -> a | b | h(A)``.
    """
    return generator_query(nullary_space(), NULLARY_START)


@pytest.fixture
def recursive():
    """Return the query on the primary recursive space.

    Returns:
        ResolutionQuery: The generator query on ``S -> top(C, C)``.
    """
    return generator_query(recursive_space(), RECURSIVE_START)


class _ListSampler:
    """A sampler that streams a fixed list and always claims the bound is wide enough.

    The second failure clause is unreachable through the real samplers, because their bound check
    is exact and a stream that ends early would contradict it. Reaching the clause needs a sampler
    whose two answers disagree, which is what this one is for.

    Attributes:
        trees (list[Tree]): What the stream delivers.
    """

    def __init__(self, trees) -> None:
        """Store the stream contents.

        Args:
            trees (list[Tree]): What the stream delivers, in order.
        """
        self.trees = trees
        self.streams = 0

    def sample(self, query):
        """Stream the fixed list.

        Args:
            query: Ignored.

        Yields:
            Tree: Each stored tree once.
        """
        self.streams += 1
        yield from self.trees

    def at_least(self, query, count) -> bool:
        """Claim the bound admits any number.

        Args:
            query: Ignored.
            count: Ignored.

        Returns:
            bool: Always True.
        """
        return True


class _MarkedInitializer:
    """An initializer that returns a fixed individual, as often as it is asked for.

    Attributes:
        individual (Tree): The individual returned.
        sizes (list[int]): Every count this initializer was asked for.
    """

    def __init__(self, individual) -> None:
        """Store the individual.

        Args:
            individual (Tree): The individual to return.
        """
        self.individual = individual
        self.sizes: list[int] = []

    def initialize(self, query, size):
        """Return ``size`` copies.

        Args:
            query: Ignored.
            size (int): The number of copies.

        Returns:
            list[Tree]: The copies.
        """
        self.sizes.append(size)
        return [self.individual] * size


# ---------------------------------------------------------------------------
# What the initializer delivers
# ---------------------------------------------------------------------------


def test_the_population_has_exactly_the_requested_size(recursive):
    """A population is filled or the attempt fails; it is never short.

    Args:
        recursive: The recursive-space query fixture.
    """
    initializer = SampledInitialization(SizeUniformSampler(9, random.Random(0)))
    assert len(initializer.initialize(recursive, 7)) == 7


def test_every_member_is_an_inhabitant(recursive):
    """Every individual the initializer produces is an inhabitant of the space.

    Args:
        recursive: The recursive-space query fixture.
    """
    initializer = SampledInitialization(SizeUniformSampler(9, random.Random(1)))
    for individual in initializer.initialize(recursive, 10):
        assert checker(recursive.solution_space, recursive.start, individual)


def test_members_respect_the_bound_of_the_sampler(recursive):
    """What initialization produces stays within the bound of its sampler.

    Args:
        recursive: The recursive-space query fixture.
    """
    initializer = SampledInitialization(SizeUniformSampler(7, random.Random(2)))
    assert all(term_size(individual) <= 7 for individual in initializer.initialize(recursive, 8))


def test_the_population_comes_from_one_stream(recursive):
    """One stream, not one draw per individual.

    A prefix of one size-uniform stream samples without replacement, whereas independent streams
    repeat, and the single stream is also the difference between one query and mu of them on a
    realistic space.

    Args:
        recursive: The recursive-space query fixture.
    """
    trees = [Tree(a2, ()), Tree(b2, ()), Tree(a2, ())]
    sampler = _ListSampler(trees)
    population = SampledInitialization(sampler).initialize(recursive, 3)
    assert sampler.streams == 1
    assert population == trees


def test_a_size_uniform_population_repeats_nothing(recursive):
    """Under unambiguity the prefix is a sample without replacement.

    Args:
        recursive: The recursive-space query fixture.
    """
    initializer = SampledInitialization(SizeUniformSampler(11, random.Random(3)))
    population = initializer.initialize(recursive, 12)
    assert len(set(population)) == 12


def test_a_depth_bounded_population_may_repeat(recursive):
    """The other sampler draws independently, and the initializer does not deduplicate.

    A population is a multiset. Repetition is a property of the sampler rather than a defect for
    the initializer to repair, which is why the two samplers are documented apart.

    Args:
        recursive: The recursive-space query fixture.
    """
    # The bound admits far more terms than the 40 drawn here, and the population repeats anyway,
    # because independent draws through a randomized clause order concentrate on short terms.
    initializer = SampledInitialization(DepthBoundedRandomSampler(3, random.Random(4)))
    population = initializer.initialize(recursive, 40)
    assert len(population) == 40
    assert len(set(population)) < 40


# ---------------------------------------------------------------------------
# The two failure clauses
# ---------------------------------------------------------------------------


def test_a_population_larger_than_the_bounded_space_fails(tiny):
    """First clause: fewer than mu inhabitants within the bound is an error.

    The bound admits exactly ``a`` and ``b``, so a population of three cannot be filled. The
    depth-bounded sampler would otherwise return three individuals happily, two of them repeats,
    and the caller would never learn that its search space is smaller than its population.

    Args:
        tiny: The two-term space query fixture.
    """
    initializer = SampledInitialization(DepthBoundedRandomSampler(0, random.Random(5)))
    assert len(initializer.initialize(tiny, 2)) == 2
    with pytest.raises(InitializationError, match="within the bound"):
        initializer.initialize(tiny, 3)


def test_the_same_clause_holds_for_the_size_uniform_sampler(tiny):
    """The count within the bound is exact for the size-uniform sampler too.

    Args:
        tiny: The two-term space query fixture.
    """
    initializer = SampledInitialization(SizeUniformSampler(1, random.Random(6)))
    with pytest.raises(InitializationError, match="within the bound"):
        initializer.initialize(tiny, 3)


def test_a_stream_ending_early_fails(recursive):
    """Second clause: a request delivering no inhabitant is an error, not a short population.

    Args:
        recursive: The recursive-space query fixture.
    """
    sampler = _ListSampler([Tree(a2, ()), Tree(b2, ())])
    with pytest.raises(InitializationError, match="ended after 2 of 5"):
        SampledInitialization(sampler).initialize(recursive, 5)


def test_a_stream_one_short_of_the_population_fails(recursive):
    """The second clause triggers on a stream that is short by one, not only by many.

    The comparison is against the requested size itself. A bound that were off by one would let
    the last individual go missing and return a population the caller never asked for, which is
    exactly the silent shortening the clause exists to prevent.

    Args:
        recursive: The recursive-space query fixture.
    """
    sampler = _ListSampler([Tree(a2, ()), Tree(b2, ()), Tree(a2, ()), Tree(b2, ())])
    with pytest.raises(InitializationError, match="ended after 4 of 5"):
        SampledInitialization(sampler).initialize(recursive, 5)


def test_a_negative_population_size_is_refused(recursive):
    """A population size is a count.

    Args:
        recursive: The recursive-space query fixture.
    """
    initializer = SampledInitialization(SizeUniformSampler(9, random.Random(7)))
    with pytest.raises(ValueError, match="negative"):
        initializer.initialize(recursive, -1)


def test_an_empty_population_is_not_an_error(recursive):
    """Zero individuals is a degenerate request, not a failing one.

    The mixture asks for it whenever the binomial split puts every place on one side.

    Args:
        recursive: The recursive-space query fixture.
    """
    initializer = SampledInitialization(SizeUniformSampler(9, random.Random(8)))
    assert initializer.initialize(recursive, 0) == []


# ---------------------------------------------------------------------------
# The mixture
# ---------------------------------------------------------------------------


def test_the_mixture_splits_the_population_between_its_components(recursive):
    """Both components contribute, and together they fill the population exactly.

    Args:
        recursive: The recursive-space query fixture.
    """
    first = _MarkedInitializer(Tree(a2, ()))
    second = _MarkedInitializer(Tree(b2, ()))
    population = MixtureInitializer(0.5, first, second, random.Random(9)).initialize(recursive, 20)
    assert len(population) == 20
    assert first.sizes[0] + second.sizes[0] == 20


def test_the_split_is_binomial(recursive):
    """The mixture draws k binomially with mu trials and success probability p.

    Mean and variance are checked against ``mu p`` and ``mu p (1 - p)`` over many draws with a fixed
    seed. A split that merely rounded ``mu p`` would pass the mean and fail the variance, which is
    why both are here.

    Args:
        recursive: The recursive-space query fixture.
    """
    size, probability, repeats = 20, 0.3, 4000
    first = _MarkedInitializer(Tree(a2, ()))
    second = _MarkedInitializer(Tree(b2, ()))
    mixture = MixtureInitializer(probability, first, second, random.Random(10))
    for _ in range(repeats):
        mixture.initialize(recursive, size)

    drawn = first.sizes
    mean = sum(drawn) / repeats
    variance = sum((value - mean) ** 2 for value in drawn) / repeats
    assert abs(mean - size * probability) < 0.1
    assert abs(variance - size * probability * (1 - probability)) < 0.25
    # Both tails are reached, which a split that merely rounded ``mu p`` would not do.
    assert min(drawn) <= 1
    assert max(drawn) >= 12


def test_a_mixture_of_the_two_samplers_is_ramped_half_and_half(recursive):
    """The pairing this component is built for: size-uniform and depth-bounded at p = 1/2.

    Args:
        recursive: The recursive-space query fixture.
    """
    mixture = MixtureInitializer(
        0.5,
        SampledInitialization(SizeUniformSampler(9, random.Random(11))),
        SampledInitialization(DepthBoundedRandomSampler(3, random.Random(12))),
        random.Random(13),
    )
    population = mixture.initialize(recursive, 16)
    assert len(population) == 16
    assert all(checker(recursive.solution_space, recursive.start, individual) for individual in population)


def test_a_failing_component_fails_the_mixture(tiny):
    """Each component checks the bound of its own sampler, and its failure is the mixture's.

    Args:
        tiny: The two-term space query fixture.
    """
    mixture = MixtureInitializer(
        1.0,
        SampledInitialization(SizeUniformSampler(1, random.Random(14))),
        _MarkedInitializer(Tree(a2, ())),
        random.Random(15),
    )
    with pytest.raises(InitializationError):
        mixture.initialize(tiny, 5)


def test_a_probability_outside_the_unit_interval_is_refused():
    """p is a probability."""
    marked = _MarkedInitializer(Tree(a2, ()))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        MixtureInitializer(1.5, marked, marked, random.Random(16))


def test_the_mixture_refuses_a_negative_population_size():
    """A population size is a count for the mixture too, and it is checked before the split.

    The guard sits ahead of the binomial draw, so a negative size cannot reach the components and
    be turned into a pair of counts that happen to sum correctly.
    """
    marked = _MarkedInitializer(Tree(a2, ()))
    with pytest.raises(ValueError, match="cannot be negative"):
        MixtureInitializer(0.5, marked, marked, random.Random(19)).initialize(None, -1)


def test_a_probability_below_zero_is_refused():
    """Both ends of the unit interval are checked, not only the upper one."""
    marked = _MarkedInitializer(Tree(a2, ()))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        MixtureInitializer(-0.5, marked, marked, random.Random(20))


def test_the_successes_come_first_in_the_mixture(recursive):
    """The individuals of the first component precede those of the second.

    The two components are distinguishable here, so the order is observable. It is the order the
    class documents, and a caller that pairs a mixture with a component-dependent post-step reads
    it.

    Args:
        recursive: The recursive-space query fixture.
    """
    first = _MarkedInitializer(Tree(a2, ()))
    second = _MarkedInitializer(Tree(b2, ()))
    population = MixtureInitializer(0.5, first, second, random.Random(21)).initialize(recursive, 20)
    boundary = first.sizes[0]
    assert population[:boundary] == [Tree(a2, ())] * boundary
    assert population[boundary:] == [Tree(b2, ())] * (20 - boundary)


def test_both_initializers_satisfy_the_protocol():
    """The component class is structural, so a caller may pass its own."""
    sampled = SampledInitialization(SizeUniformSampler(5, random.Random(17)))
    assert isinstance(sampled, Initializer)
    assert isinstance(MixtureInitializer(0.5, sampled, sampled, random.Random(18)), Initializer)
    assert isinstance(_MarkedInitializer(Tree(a2, ())), Initializer)
