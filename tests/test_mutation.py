"""Resolution mutation.

Four departures from the previous operator are pinned here:

* the position is drawn **uniformly over the non-leaf positions, the root included**, because
  reachability rests on the root: there the residual query is the generator query, and an
  exhaustive sampler reaches every inhabitant within its bound;
* **one position, one request**. A failed request means no offspring, and the surrounding
  procedure draws new parents. Trying position after position would replace the stated position
  distribution with an unstated one;
* the parameter is **a sampler**, not a space plus a start symbol plus a depth;
* there is **no membership test**, since the query itself delivers only completions.

The distribution test also covers the full residual: it compares the offspring distribution against
``position-uniform x size-uniform over the full residual``, and a residual truncated to its first
goal would fail it.
"""

import random
from collections import Counter

import pytest

from cosy.core.tree import Tree
from cosy.evolutionary_algorithms import Mutation, ResolutionMutation
from cosy.search import (
    DepthBoundedRandomSampler,
    SizeUniformSampler,
    checker,
    generator_query,
    residual_query,
    term_depth,
    term_size,
)
from tests.ea_fixtures import (
    NULLARY_START,
    RECURSIVE_START,
    CountingSpace,
    a2,
    b2,
    chain,
    h1,
    leaf_c,
    nullary_space,
    parent,
    recursive_space,
    rendered,
)


@pytest.fixture
def recursive():
    """Return the generator query on the primary recursive space.

    Returns:
        ResolutionQuery: The query the operator derives its residual queries from.
    """
    return generator_query(recursive_space(), RECURSIVE_START)


@pytest.fixture
def tiny():
    """Return the generator query on ``A -> a | b | h(A)``.

    Returns:
        ResolutionQuery: Small enough for an exact offspring distribution.
    """
    return generator_query(nullary_space(), NULLARY_START)


class _SpySampler:
    """Delegate to a real sampler and record the position of every query it is given.

    Attributes:
        inner: The sampler delegated to.
        positions (list): The ``pos`` of each query, one entry per request.
    """

    def __init__(self, inner) -> None:
        """Wrap a sampler.

        Args:
            inner: The sampler to delegate to.
        """
        self.inner = inner
        self.positions: list = []

    def sample(self, query):
        """Record the query's position and delegate.

        Args:
            query: The residual query.

        Returns:
            Iterator[Tree]: The wrapped sampler's stream.
        """
        self.positions.append(query.pos)
        return self.inner.sample(query)

    def at_least(self, query, count) -> bool:
        """Delegate the bound question.

        Args:
            query: The query.
            count (int): The number asked for.

        Returns:
            bool: The wrapped sampler's answer.
        """
        return self.inner.at_least(query, count)


class _EmptySampler:
    """A sampler whose stream is always empty, counting the requests it received.

    Attributes:
        requests (int): The number of streams asked for.
    """

    def __init__(self) -> None:
        """Start with no requests."""
        self.requests = 0

    def sample(self, query):
        """Return an empty stream.

        Args:
            query: Ignored.

        Yields:
            Tree: Nothing.
        """
        self.requests += 1
        return
        yield  # pragma: no cover, makes this a generator

    def at_least(self, query, count) -> bool:
        """Claim nothing lies within the bound.

        Args:
            query: Ignored.
            count: Ignored.

        Returns:
            bool: Always False.
        """
        return False


def _mutation(seed: int, bound: int = 6) -> ResolutionMutation:
    """Build a seeded operator over a depth-bounded sampler.

    Args:
        seed (int): Seed for both the sampler and the position draw.
        bound (int): The depth bound. (Default value = 6)

    Returns:
        ResolutionMutation: The operator.
    """
    return ResolutionMutation(DepthBoundedRandomSampler(bound, random.Random(seed)), random.Random(seed + 100))


# ---------------------------------------------------------------------------
# Closure by construction
# ---------------------------------------------------------------------------


def test_an_offspring_is_an_inhabitant(recursive):
    """Whatever comes back lies in the tree language.

    Args:
        recursive: The recursive-space query fixture.
    """
    for seed in range(20):
        offspring = _mutation(seed).mutate(recursive, parent(2, 2))
        if offspring is not None:
            assert checker(recursive.solution_space, recursive.start, offspring), rendered(offspring)


def test_the_operator_never_tests_membership(recursive):
    """Closure by construction means no checker call, which is recombination's mechanism instead.

    Args:
        recursive: The recursive-space query fixture.
    """
    counting = CountingSpace(recursive.solution_space)
    watched = generator_query(counting, recursive.start)
    for seed in range(5):
        _mutation(seed).mutate(watched, parent(2, 2))
    assert counting.calls == []


def test_the_parent_is_not_modified(recursive):
    """The operator is functional in its argument.

    Args:
        recursive: The recursive-space query fixture.
    """
    original = parent(2, 3)
    before = rendered(original)
    _mutation(3).mutate(recursive, original)
    assert rendered(original) == before


def test_an_offspring_respects_the_bound_of_the_sampler(recursive):
    """The sampler bounds the whole individual, since a residual query's term is all of it.

    Args:
        recursive: The recursive-space query fixture.
    """
    for seed in range(20):
        offspring = _mutation(seed, bound=4).mutate(recursive, parent(2, 2))
        if offspring is not None:
            assert term_depth(offspring) <= 4


# ---------------------------------------------------------------------------
# One position, one request
# ---------------------------------------------------------------------------


def test_a_failed_request_yields_no_offspring_and_no_retry(recursive):
    """Exactly one request; a stream that gives nothing ends the operator.

    The previous operator shuffled the eligible positions and tried them in turn until one
    succeeded, which silently reweighted the position distribution towards positions whose
    residual is easy to complete.

    Args:
        recursive: The recursive-space query fixture.
    """
    sampler = _EmptySampler()
    operator = ResolutionMutation(sampler, random.Random(0))
    assert operator.mutate(recursive, parent(2, 2)) is None
    assert sampler.requests == 1


def test_the_position_is_uniform_over_the_mutation_points(recursive):
    """The position is drawn uniformly among the non-leaves and the root.

    The previous operator excluded the root outright *and* trimmed the leaves. The leaf exclusion
    is kept here, with the reasons in :class:`~cosy.evolutionary_algorithms.mutation.ResolutionMutation`.
    The root exclusion is not, because reachability is what the root carries.

    Args:
        recursive: The recursive-space query fixture.
    """
    individual = parent(1, 1)  # positions: (), (0), (0,0), (1), (1,0)
    points = sorted(ResolutionMutation.mutation_points(individual))
    assert points == [(), (0,), (1,)], "the two leaves are not mutation points"
    assert individual.positions() - set(points) == {(0, 0), (1, 0)}

    spy = _SpySampler(DepthBoundedRandomSampler(6, random.Random(0)))
    operator = ResolutionMutation(spy, random.Random(1))
    draws = 5000
    for _ in range(draws):
        operator.mutate(recursive, individual)

    counts = Counter(spy.positions)
    assert set(counts) == set(points)
    expected = draws / len(points)
    for position in points:
        assert abs(counts[position] - expected) < 0.15 * expected, f"{position}: {counts}"


def test_a_single_node_individual_is_an_ordinary_input(tiny):
    """One position, the root, and mutating it regenerates the individual.

    Args:
        tiny: The two-symbol space query fixture.
    """
    operator = ResolutionMutation(SizeUniformSampler(3, random.Random(0)), random.Random(1))
    offspring = operator.mutate(tiny, Tree(a2, ()))
    assert offspring is not None
    assert checker(tiny.solution_space, tiny.start, offspring)


def test_a_c_subtree_is_not_a_valid_individual_of_the_recursive_space(recursive):
    """Guard for the fixtures below: ``lf`` alone does not inhabit ``S``.

    Args:
        recursive: The recursive-space query fixture.
    """
    assert not checker(recursive.solution_space, recursive.start, leaf_c())


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------


# Both bounds admit exactly ``a``, ``b``, ``h(a)``, ``h(b)`` on the tiny space, since size 2 counts
# symbol occurrences and depth 1 counts edges, so the two samplers are asked for the same language
# and the reachability claims can be stated once for both.
#
# Running these over *both* is the point. Reachability is a statement about the residual, and the
# residual is what the sampler is handed. A sampler whose randomness sits entirely in the clause
# order tests a path through the engine that the size-uniform one, which enumerates, never touches.
BOUNDED_SAMPLERS = [
    pytest.param(lambda seed: SizeUniformSampler(2, random.Random(seed)), id="size-uniform"),
    pytest.param(lambda seed: DepthBoundedRandomSampler(1, random.Random(seed)), id="depth-bounded"),
]


@pytest.mark.parametrize("build_sampler", BOUNDED_SAMPLERS)
def test_the_root_position_regenerates_the_whole_individual(tiny, build_sampler):
    """At the root the residual query *is* the generator query.

    Args:
        tiny: The two-symbol space query fixture.
        build_sampler (Callable): Builds the sampler from a seed.
    """
    spy = _SpySampler(build_sampler(0))
    operator = ResolutionMutation(spy, random.Random(2))
    individual = Tree(h1, (Tree(a2, ()),))
    seen = set()
    for _ in range(200):
        offspring = operator.mutate(tiny, individual)
        if spy.positions[-1] == ():
            seen.add(rendered(offspring))
    # A completion at the root ranges over the whole language within the bound, so terms that
    # share no symbol with the parent's root position must appear.
    assert {"a", "b"} <= seen


@pytest.mark.parametrize("build_sampler", BOUNDED_SAMPLERS)
def test_any_individual_within_the_bound_is_reachable_from_any_other(tiny, build_sampler):
    """Reachability concretely: r1 reaches r2, both small, r1 != r2.

    With the root excluded, ``h(a)`` could never become ``b``: every offspring would keep the
    root symbol ``h``. That is the point about inner-position operators, and
    this is it as a test.

    Args:
        tiny: The two-symbol space query fixture.
        build_sampler (Callable): Builds the sampler from a seed.
    """
    operator = ResolutionMutation(build_sampler(3), random.Random(4))
    source = Tree(h1, (Tree(a2, ()),))
    target = Tree(b2, ())
    assert any(operator.mutate(tiny, source) == target for _ in range(200))


# ---------------------------------------------------------------------------
# The offspring distribution
# ---------------------------------------------------------------------------


def test_the_offspring_distribution_is_position_uniform_times_the_residual(tiny):
    """The full product distribution, on a space small enough to state it exactly.

    Parent ``h(a)`` has two positions and exactly one mutation point: ``(0,)`` holds the leaf
    ``a``, so the root is the only draw. There the residual is the whole language within size 2,
    that is ``a``, ``b`` of size 1 and ``h(a)``, ``h(b)`` of size 2, and size-uniform gives each
    realized size 1/2 and splits it evenly, so 1/4 each.

    Under the previous rule the leaf was drawn half the time, its residual was ``h(_)`` with a
    completion of size at most 1, and the four terms came out 1/8, 1/8, 3/8, 3/8. The uniform
    answer here is what excluding the leaf *does*: it is a different distribution, not a tidier
    one, and this is where the difference is stated. A residual computed from only the first goal
    of the query would still shift the mass at the root, which is what this measurement pins.

    Args:
        tiny: The two-symbol space query fixture.
    """
    operator = ResolutionMutation(SizeUniformSampler(2, random.Random(5)), random.Random(6))
    individual = Tree(h1, (Tree(a2, ()),))
    draws = 8000
    counts = Counter(rendered(operator.mutate(tiny, individual)) for _ in range(draws))
    expected = {"a": 1 / 4, "b": 1 / 4, "h(a)": 1 / 4, "h(b)": 1 / 4}
    assert set(counts) == set(expected)
    total_variation = sum(abs(counts[term] / draws - probability) for term, probability in expected.items()) / 2
    assert total_variation < 0.02, counts


def test_the_distribution_test_would_reject_a_root_excluding_operator(tiny):
    """Negative control: without the root, the same measurement must fail.

    A statistical claim with no control is empty, so this is the sample the assertion above would
    have to reject, produced by the operator this one is contrasted with.

    Args:
        tiny: The two-symbol space query fixture.
    """
    sampler = SizeUniformSampler(2, random.Random(7))
    rng = random.Random(8)
    individual = Tree(h1, (Tree(a2, ()),))
    inner = sorted(individual.positions() - {()})
    draws = 8000
    counts: Counter = Counter()
    for _ in range(draws):
        opened = residual_query(tiny.solution_space, tiny.start, individual, rng.choice(inner))
        counts[rendered(next(iter(sampler.sample(opened))))] += 1
    expected = {"a": 1 / 8, "b": 1 / 8, "h(a)": 3 / 8, "h(b)": 3 / 8}
    total_variation = sum(abs(counts[term] / draws - probability) for term, probability in expected.items()) / 2
    assert total_variation > 0.2, counts


def test_the_operator_satisfies_the_protocol():
    """The component class is structural."""
    assert isinstance(_mutation(0), Mutation)


def test_offspring_sizes_stay_within_a_size_bounded_sampler(recursive):
    """The size-uniform sampler bounds the offspring by size rather than by depth.

    Args:
        recursive: The recursive-space query fixture.
    """
    operator = ResolutionMutation(SizeUniformSampler(9, random.Random(9)), random.Random(10))
    individual = parent(1, 1)
    for _ in range(50):
        offspring = operator.mutate(recursive, individual)
        if offspring is not None:
            assert term_size(offspring) <= 9


def test_a_parent_beyond_the_bound_can_yield_nothing(recursive):
    """No completion within the bound is an ordinary outcome, not an error.

    Args:
        recursive: The recursive-space query fixture.
    """
    operator = ResolutionMutation(SizeUniformSampler(4, random.Random(11)), random.Random(12))
    overgrown = parent(6, 6)
    assert term_size(overgrown) > 4
    # Only the root opens the whole term; at any other position the rest of the parent stays and
    # already exceeds the bound, so most draws come back empty.
    results = [operator.mutate(recursive, overgrown) for _ in range(20)]
    assert any(result is None for result in results)
    assert all(result is None or term_size(result) <= 4 for result in results)


def test_the_operator_does_not_touch_the_global_random_stream(recursive):
    """Reproducibility: every draw goes through the operator's own generators.

    Args:
        recursive: The recursive-space query fixture.
    """
    random.seed(1234)
    before = random.random()
    random.seed(1234)
    _mutation(0).mutate(recursive, parent(2, 2))
    assert random.random() == before


def test_a_parent_deeper_than_the_interpreter_recurses_is_accepted(recursive):
    """A deep parent is an ordinary input: the position walk must not recurse over the term.

    Args:
        recursive: The recursive-space query fixture.
    """
    deep = Tree(parent(1, 1).root, (chain(1200), chain(1)))
    operator = ResolutionMutation(DepthBoundedRandomSampler(4, random.Random(13)), random.Random(14))
    offspring = operator.mutate(recursive, deep)
    assert offspring is None or checker(recursive.solution_space, recursive.start, offspring)


# ---------------------------------------------------------------------------
# The position distribution: which positions the operator draws among
# ---------------------------------------------------------------------------


def test_a_leaf_is_never_opened(tiny):
    """Leaves are not mutation points, and a literal is always a leaf.

    The rule is "uniform non-leaf", and recombination carries the same one, phrased there as
    neither the root nor a leaf. Mutation keeps it too, as
    ``trim=1`` and excluded the root with them. Here the leaf exclusion stays and the root comes
    back, because reachability rests on it.

    What that cost is visible on a literal, because a constant argument becomes a childless node
    and is therefore a leaf. Opening one asks the residual query to complete a term whose literal
    the clause matcher still pins, so the query answers with the term already there: the operator
    is the identity at those positions. A repository built from literal parameters spends that
    share of its position draws on nothing.

    Args:
        tiny: The two-symbol space query fixture.
    """
    spy = _SpySampler(SizeUniformSampler(4, random.Random(0)))
    operator = ResolutionMutation(spy, random.Random(1))
    individual = Tree(h1, (Tree(h1, (Tree(a2, ()),)),))
    leaves = individual.leaf_positions()
    assert leaves, "the fixture must have a leaf, or the test says nothing"

    for _ in range(200):
        operator.mutate(tiny, individual)

    assert set(spy.positions) <= individual.positions() - leaves


def test_the_root_stays_a_mutation_point_even_when_it_is_a_leaf(tiny):
    """A single-node individual is a leaf and a root at once, and it must stay mutable.

    Excluding leaves outright would leave such an individual with no position to draw, and the
    operator would return no offspring for it, for every sampler and every seed. The root is
    what reachability rests on, so it is the one leaf that stays eligible; there the residual query
    is the generator query and the draw regenerates the individual.

    Args:
        tiny: The two-symbol space query fixture.
    """
    spy = _SpySampler(SizeUniformSampler(2, random.Random(2)))
    operator = ResolutionMutation(spy, random.Random(3))
    single = Tree(a2, ())
    assert single.positions() == single.leaf_positions() == {()}

    offspring = [operator.mutate(tiny, single) for _ in range(200)]
    assert set(spy.positions) == {()}
    assert all(child is not None for child in offspring)
    assert {rendered(child) for child in offspring} > {"a"}, (
        "the root residual is the whole language, so the draw must reach more than the parent"
    )


# ---------------------------------------------------------------------------
# The order the position pool is drawn from
# ---------------------------------------------------------------------------


class _RecordingRandom(random.Random):
    """A generator that remembers every sequence it was asked to draw from.

    Attributes:
        pools (list[list[tuple[int, ...]]]): One entry per draw, in the order the draws happened.
    """

    def __init__(self, seed: int) -> None:
        """Seed the generator.

        Args:
            seed (int): The seed.
        """
        super().__init__(seed)
        self.pools: list[list[tuple[int, ...]]] = []

    def choice(self, seq):
        """Record the sequence and draw from it.

        Args:
            seq: The sequence to draw from.

        Returns:
            The drawn element.
        """
        self.pools.append(list(seq))
        return super().choice(seq)


def test_the_position_pool_is_sorted_before_the_draw(recursive):
    """The drawn position depends on the seed alone, not on how the set iterates.

    ``mutation_points`` answers with a frozenset, and the order a set iterates in is an
    implementation detail. It shifts between interpreter versions and with any change to how the
    set is built, so a run seeded for reproducibility would stop being reproducible across the
    matrix. The uniformity measurement above cannot see this, because every order is uniform.

    Args:
        recursive: The recursive-space query fixture.
    """
    individual = parent(2, 3)
    points = ResolutionMutation.mutation_points(individual)
    assert list(points) != sorted(points), "the fixture must expose an order differing from sorted"

    rng = _RecordingRandom(0)
    ResolutionMutation(SizeUniformSampler(12, random.Random(0)), rng).mutate(recursive, individual)
    assert rng.pools == [sorted(points)]
