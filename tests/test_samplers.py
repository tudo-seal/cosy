"""The samplers: a resolution query mapped to a stream of completions.

A sampler asks for a stream of completions and one guarantee: *bounded*, every request for a next
element halts. That word carries the whole definition, because emptiness of a tree language is
undecidable, so nothing may test for it, and a component reacting to "the stream gave nothing"
must be reacting to a halting request instead. These tests pin the halting, the bound, and the one
distributional promise the depth-bounded sampler makes, which is positivity and nothing beyond it.

The two bounds are different quantities and the tests keep them apart: ``term_depth`` for the
depth-bounded sampler, ``term_size`` for the size-uniform one, and neither is the engine's
``max_depth``, which bounds the length of a goal's *open* positions and stops seeing a subtree once
it grounds, so it admits completions of any depth. One test measures exactly that gap.
"""

import random
from itertools import islice

import pytest

import cosy.search.counting as counting_module
import cosy.search.sampling as sampling_module
from cosy.core import Constructor, SpecificationBuilder, Synthesizer
from cosy.core.tree import Tree
from cosy.search import (
    DepthBoundedRandomSampler,
    Sampler,
    SizeUniformSampler,
    checker,
    depth_first,
    generator_query,
    partial_inhabitant,
    random_search,
    random_search_keyed,
    residual_query,
    size_uniform,
    term_depth,
    term_size,
    weighted_table,
)
from tests.search_fixtures import (
    EXPR,
    LIST,
    MIXED,
    PAIR,
    TUPLE_SORT,
    WIDTH,
    add,
    cons_0,
    constrained_space,
    equal_width_space,
    expression_space,
    hole_tuple_space,
    list_space,
    lit,
    mixed_arity_space,
    nil,
)

EMPTY = Constructor("Empty")
NEEDS = Constructor("Needs")


def unreachable(inner: str) -> str:
    """Build the combinator whose argument no clause inhabits.

    Args:
        inner (str): The interpreted child.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"u({inner})"


def empty_space():
    """Build a space whose start symbol has no inhabitant at all.

    ``unreachable`` needs a ``Needs``, and nothing produces one. A sampler over this space must
    end its stream rather than search forever, which is the case boundedness is about.

    Returns:
        SolutionSpace: The space, started at ``Empty``.
    """
    specs = {unreachable: SpecificationBuilder().argument("inner", NEEDS).suffix(EMPTY)}
    return Synthesizer(specs).construct_solution_space(EMPTY)


# ---------------------------------------------------------------------------
# Boundedness: every request halts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth_bound", [0, 1, 2, 3])
def test_the_depth_sampler_halts_on_a_recursive_space(depth_bound):
    """Every draw returns, on a space with no depth bound of the engine's own.

    Finitely many terms have depth at most ``D``, and the filter drops every child beyond it, so
    one draw searches a finite tree.

    Args:
        depth_bound (int): The bound under test.
    """
    sampler = DepthBoundedRandomSampler(depth_bound, random.Random(11))
    query = generator_query(list_space(), LIST)
    stream = sampler.sample(query)
    drawn = [next(stream, None) for _ in range(20)]
    assert all(tree is None or term_depth(tree) <= depth_bound for tree in drawn)


@pytest.mark.parametrize("depth_bound", [0, 1, 2])
def test_the_bound_holds_where_a_grounded_subterm_carries_the_depth(depth_bound):
    """The filter reads the depth of a grounded subterm and not only the position of a hole.

    A goal whose subgoals are all solved has no open position left, so the check over ``subgoals``
    is a check over nothing and passes. What decides such a goal is the depth its grounded subterms
    reach, which is the reason the filter reads it and the reason ``Tree`` carries it. Half of the
    goals one draw filters are of that kind, so this is the common case rather than a corner of it.

    On ``mixed_arity_space`` the distinction bites: an argument is itself composite, so a subterm of
    depth one grounds at a position of length one and the completion comes out a level deeper than
    the bound admits. The recursive spaces above cannot catch this, since there the depth of a term
    follows the length of its positions and the check over the subgoals already decides it.

    Args:
        depth_bound (int): The bound under test.
    """
    sampler = DepthBoundedRandomSampler(depth_bound, random.Random(11))
    query = generator_query(mixed_arity_space(), MIXED)
    drawn = [tree for tree, _ in zip(sampler.sample(query), range(30), strict=False)]

    assert drawn, "the bound admits completions here, so an empty stream would make the test vacuous"
    assert all(term_depth(tree) <= depth_bound for tree in drawn)


def test_a_bound_that_admits_nothing_ends_the_stream_rather_than_searching_on():
    """A nonempty language whose shallowest term is out of reach still halts, and says so.

    This is the case boundedness is about, and it is not the empty language: every clause of
    ``hole_tuple_space`` that reaches the start symbol opens two holes, so the shallowest
    completion has depth 2 and a bound of 0 admits none of them. The search must exhaust its
    finite tree and end rather than run on looking for one.
    """
    sampler = DepthBoundedRandomSampler(0, random.Random(11))
    query = generator_query(hole_tuple_space(), TUPLE_SORT)

    assert list(sampler.sample(query)) == []
    assert not sampler.at_least(query, 1)

    # The same space one level up does admit its completions, so the bound is what decided it and
    # not the space.
    reaching = DepthBoundedRandomSampler(2, random.Random(11))
    drawn = [tree for tree, _ in zip(reaching.sample(query), range(20), strict=False)]
    assert drawn
    assert all(term_depth(tree) <= 2 for tree in drawn)


def test_the_depth_sampler_ends_its_stream_on_an_empty_language():
    """A space with no inhabitant produces an empty stream, not a hanging one.

    The distinction boundedness insists on: the caller learns that nothing came, by
    the stream ending, and never by asking whether the language is empty.
    """
    sampler = DepthBoundedRandomSampler(5, random.Random(3))
    assert list(sampler.sample(generator_query(empty_space(), EMPTY))) == []


def test_the_size_sampler_ends_its_stream_on_an_empty_language():
    """The same for the size-uniform sampler."""
    sampler = SizeUniformSampler(5, random.Random(3))
    assert list(sampler.sample(generator_query(empty_space(), EMPTY))) == []


@pytest.mark.parametrize("build_sampler", [DepthBoundedRandomSampler, SizeUniformSampler])
def test_a_negative_bound_is_an_error(build_sampler):
    """A bound is a length or a count; a negative one is a miscomputation, not a default.

    Args:
        build_sampler (type): The sampler class under test.
    """
    with pytest.raises(ValueError, match="cannot be negative"):
        build_sampler(-1, random.Random(0))


def test_both_samplers_satisfy_the_protocol():
    """Both are ``Sampler``s, which is the narrow interface a caller takes as a parameter."""
    assert isinstance(DepthBoundedRandomSampler(2, random.Random(0)), Sampler)
    assert isinstance(SizeUniformSampler(2, random.Random(0)), Sampler)


# ---------------------------------------------------------------------------
# The bound is on the partial inhabitant, not on the goal's open positions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth_bound", [0, 1, 2, 3, 4])
def test_every_draw_respects_the_depth_bound(depth_bound):
    """No draw is deeper than the bound, on a space where deeper terms exist.

    Args:
        depth_bound (int): The bound under test.
    """
    sampler = DepthBoundedRandomSampler(depth_bound, random.Random(5))
    query = generator_query(expression_space(), EXPR)
    drawn = [tree for tree, _ in zip(sampler.sample(query), range(60), strict=False)]
    assert drawn, "the bound admits at least the literal"
    assert all(term_depth(tree) <= depth_bound for tree in drawn)


@pytest.mark.parametrize(("build", "start"), [(list_space, LIST), (expression_space, EXPR)])
@pytest.mark.parametrize("bound", [1, 2, 3])
def test_max_depth_and_the_term_depth_agree_on_the_streamed_terms(build, start, bound):
    """``max_depth`` and ``term_depth`` agree on what is *streamed*, and this records why.

    They measure different things. ``max_depth`` bounds the longest **open** position of a goal,
    and a subtree leaves that measurement as soon as it grounds: over the retained trees of three
    reference spaces, 117 of 241 goals report a depth below their partial inhabitant's. On the
    finished terms the two nevertheless coincide, and the reason is the computation rule, which
    engine expands the deepest open subgoal first, so every position is measured as an open one
    before it grounds.

    That is a property of the rule, not a promise of ``max_depth``, and the sampler's bound is
    stated on the partial inhabitant. It therefore goes through
    ``goal_filter``, where it holds whatever the rule does, and this test records the agreement
    rather than relying on it. Should it ever break, the sampler is unaffected and this test
    says so.

    Args:
        build (Callable): Builds the space.
        start: The queried non-terminal.
        bound (int): The bound handed to both.
    """
    query = generator_query(build(), start)
    streamed = [tree for tree, _ in zip(depth_first(query, max_depth=bound), range(400), strict=False)]
    assert streamed
    assert all(term_depth(tree) <= bound for tree in streamed)

    sampler = DepthBoundedRandomSampler(bound, random.Random(9))
    drawn = [tree for tree, _ in zip(sampler.sample(query), range(100), strict=False)]
    assert {str(tree) for tree in drawn} <= {str(tree) for tree in streamed}


def test_the_depth_bound_is_measured_on_the_partial_inhabitant():
    """The filter reads the bound off the materialized term, not off the goal's positions.

    Measured directly, since the two agree on the finished terms: a goal deep in a chain reports
    an open-position depth of 1 while its partial inhabitant is four levels deep, and it is the
    latter the sampler bounds.
    """
    space = list_space()
    goals_seen: list[int] = []

    def record(goal):
        """Record the two depths of one goal and admit it.

        Args:
            goal: The search node.

        Returns:
            bool: Always True.
        """
        engine_depth = max(len(position) for position in list(goal.grounded.keys()) + list(goal.subgoals.keys()))
        goals_seen.append(term_depth(partial_inhabitant(goal)) - engine_depth)
        return True

    list(zip(depth_first(generator_query(space, LIST), goal_filter=record), range(60), strict=False))
    assert any(difference > 0 for difference in goals_seen), (
        "no goal reported a depth below its partial inhabitant's, the two measure the same "
        "thing after all, and the filter's justification needs revisiting"
    )


# ---------------------------------------------------------------------------
# Positivity, and for the depth sampler nothing more
# ---------------------------------------------------------------------------


def test_every_completion_within_the_bound_is_drawn_by_some_seed():
    """Each completion comes out with positive probability.

    The claim is about the *first* element of the stream, so the test takes one draw per seed
    rather than a prefix of one stream.
    """
    bound = 2
    query = generator_query(list_space(), LIST)
    reachable = {str(tree) for tree in depth_first(query, max_depth=6) if term_depth(tree) <= bound}
    drawn = {
        str(next(iter(DepthBoundedRandomSampler(bound, random.Random(seed)).sample(query)))) for seed in range(400)
    }
    assert drawn == reachable


def test_the_size_sampler_reaches_every_completion_too():
    """The same for the size-uniform sampler, over its own bound.

    Its stream is exhaustive by construction rather than by chance, since it enumerates every
    inhabitant within the bound, so one stream suffices where the depth sampler needs seeds.
    """
    bound = 4
    query = generator_query(list_space(), LIST)
    streamed = {str(tree) for tree in SizeUniformSampler(bound, random.Random(2)).sample(query)}
    expected = {str(tree) for tree in depth_first(query, max_depth=6) if term_size(tree) <= bound}
    assert streamed == expected


def _residual_witness():
    """Build a term with a root and one inner position.

    Returns:
        tuple: The space, its generator query, and a term of depth 1, so that ``()`` and ``(0,)``
            are both positions of it, which is what the residual tests vary over.
    """
    space = equal_width_space()
    query = generator_query(space, WIDTH)
    witness = next(tree for tree in depth_first(query, max_depth=6) if term_depth(tree) == 1)
    return space, query, witness


@pytest.mark.parametrize("position", [(), (0,)])
def test_every_completion_of_a_residual_is_drawn_by_some_seed(position):
    """Positivity on a *residual* query, which is where a mutation operator asks.

    The generator query is not the only query a sampler is handed: a resolution mutation
    opens a position of an individual and samples the residual there, and
    the reachability an evolutionary algorithm's convergence rests on is exactly the claim that
    every completion can come out.

    The depth-bounded sampler puts all of its randomness in the clause order, so this holds only if
    the clause order reaches the initial goals of a partial-term query as well. It did not:
    those goals came out in program order on every seed, the search rule explored them in that
    one order, and the operator returned the same term for every generator it was given. The
    root is the sharp case, because there the residual is the whole language.

    Args:
        position (tuple): The position that becomes the hole.
    """
    bound = 2
    space, _, witness = _residual_witness()
    residual = residual_query(space, WIDTH, witness, position)
    reachable = {str(tree) for tree in depth_first(residual, max_depth=6) if term_depth(tree) <= bound}
    drawn = {
        str(next(iter(DepthBoundedRandomSampler(bound, random.Random(seed)).sample(residual)))) for seed in range(400)
    }
    assert drawn == reachable


def test_at_the_root_the_residual_draws_exactly_as_the_generator_does():
    """The identity a resolution mutation rests on, read off the draws themselves.

    In words: the residual query at the root is the generator query, and every root mutation is
    that draw. Nothing of the individual constrains the
    query there, so the two are the same query and a sampler must not be able to tell them apart:
    seed for seed, not merely in the set they reach. This is the sharp form of the test above and
    the one that would catch a clause order reaching only *some* of the residual's expansions.
    """
    bound = 2
    space, query, witness = _residual_witness()
    residual = residual_query(space, WIDTH, witness, ())
    from_residual = [
        str(next(iter(DepthBoundedRandomSampler(bound, random.Random(seed)).sample(residual)))) for seed in range(200)
    ]
    from_generator = [
        str(next(iter(DepthBoundedRandomSampler(bound, random.Random(seed)).sample(query)))) for seed in range(200)
    ]
    assert from_residual == from_generator


@pytest.mark.parametrize("position", [(), (0,)])
def test_the_clause_order_does_not_change_which_completions_a_residual_has(position):
    """A clause order permutes the goals of a residual query; it never adds or drops one.

    The counterpart to the test above, and the reason a randomized order is safe here: the
    residual is a *set*, the order is a bijection on each expansion, so the set of completions is
    invariant. Only the order in which they are streamed moves, which
    is what the counting path relies on when it asks ``goal_from_tree`` without an order.

    Args:
        position (tuple): The position that becomes the hole.
    """
    space, _, witness = _residual_witness()
    residual = residual_query(space, WIDTH, witness, position)
    in_program_order = {str(tree) for tree in depth_first(residual, max_depth=5)}
    for seed in range(20):
        rng = random.Random(seed)

        def shuffled(applicable, rng=rng):
            drawn = list(applicable)
            rng.shuffle(drawn)
            return tuple(drawn)

        streamed = {str(tree) for tree in depth_first(residual, max_depth=5, clause_order=shuffled)}
        assert streamed == in_program_order


def test_the_size_sampler_repeats_nothing():
    """Under unambiguity the prefixes are samples without replacement."""
    streamed = [
        str(tree) for tree in SizeUniformSampler(4, random.Random(13)).sample(generator_query(expression_space(), EXPR))
    ]
    assert len(streamed) == len(set(streamed))


def test_the_depth_sampler_draws_with_replacement():
    """Independent draws repeat, and the sampler promises no more than that.

    Recorded rather than lamented: this is the difference to the size-uniform sampler, and it is
    why an initialization fills a population by asking repeatedly rather than by taking a prefix.
    """
    query = generator_query(list_space(), LIST)
    drawn = [
        tree for tree, _ in zip(DepthBoundedRandomSampler(1, random.Random(4)).sample(query), range(30), strict=False)
    ]
    assert len(drawn) > len({str(tree) for tree in drawn})


# ---------------------------------------------------------------------------
# Determinism, residual queries, and the bound question
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build_sampler", [DepthBoundedRandomSampler, SizeUniformSampler])
def test_equally_seeded_samplers_draw_identically(build_sampler):
    """All randomness comes from the generator handed in.

    Args:
        build_sampler (type): The sampler class under test.
    """
    query = generator_query(expression_space(), EXPR)
    first = [str(tree) for tree, _ in zip(build_sampler(3, random.Random(21)).sample(query), range(15), strict=False)]
    second = [str(tree) for tree, _ in zip(build_sampler(3, random.Random(21)).sample(query), range(15), strict=False)]
    assert first == second


@pytest.mark.parametrize("build_sampler", [DepthBoundedRandomSampler, SizeUniformSampler])
def test_the_samplers_run_on_a_residual_query(build_sampler):
    """Both complete a partial term, which is what lets mutation reuse the initializer's sampler.

    Args:
        build_sampler (type): The sampler class under test.
    """
    space = expression_space()
    prescribed = Tree(add, (Tree(lit, ()), Tree(lit, ())))
    query = residual_query(space, EXPR, prescribed, (0,))

    drawn = [tree for tree, _ in zip(build_sampler(3, random.Random(6)).sample(query), range(10), strict=False)]
    assert drawn, "the residual is not empty"
    for tree in drawn:
        assert tree.root is add, "the prescribed shape survives"
        assert checker(space, EXPR, tree)


@pytest.mark.parametrize("build_sampler", [DepthBoundedRandomSampler, SizeUniformSampler])
def test_the_bound_question_is_answered_exactly(build_sampler):
    """``at_least`` agrees with counting the completions by hand.

    The question an initialization asks before drawing: a population it cannot fill has to fail
    at once, and the answer is decidable within the bound.

    Args:
        build_sampler (type): The sampler class under test.
    """
    bound = 3
    query = generator_query(list_space(), LIST)
    sampler = build_sampler(bound, random.Random(1))
    measure = term_depth if build_sampler is DepthBoundedRandomSampler else term_size
    available = sum(1 for tree in depth_first(query, max_depth=6) if measure(tree) <= bound)
    assert available > 1

    assert sampler.at_least(query, 0)
    assert sampler.at_least(query, 1)
    assert sampler.at_least(query, available)
    assert not sampler.at_least(query, available + 1)


def test_the_bound_question_is_false_on_an_empty_language():
    """Nothing is available, so no positive count is."""
    query = generator_query(empty_space(), EMPTY)
    assert not DepthBoundedRandomSampler(4, random.Random(0)).at_least(query, 1)
    assert not SizeUniformSampler(4, random.Random(0)).at_least(query, 1)
    assert DepthBoundedRandomSampler(4, random.Random(0)).at_least(query, 0)


def test_a_depth_bound_of_zero_admits_only_the_nullary_clauses():
    """The smallest bound is not a special case: depth 0 is a single node."""
    query = generator_query(list_space(), LIST)
    drawn = [
        tree for tree, _ in zip(DepthBoundedRandomSampler(0, random.Random(8)).sample(query), range(20), strict=False)
    ]
    assert drawn
    assert {str(tree.root) for tree in drawn} == {str(nil)}
    assert all(tree.children == () for tree in drawn)
    assert str(cons_0) not in {str(tree.root) for tree in drawn}


def test_the_sampler_streams_what_the_bare_size_uniform_search_streams():
    """The sampler is the search under a uniform distribution over the realized sizes.

    Two things are pinned at once. That the sampler adds nothing to the search but the interface,
    and that the distribution it hands the search is the *uniform* one. A sampler weighting the
    sizes by anything else would still stream every inhabitant within the bound, in an order no
    structural assertion here would notice, and only the term-for-term comparison against the bare
    search separates the two.
    """
    query = generator_query(list_space(), LIST)
    bound = 6

    by_sampler = list(SizeUniformSampler(bound, random.Random(29)).sample(query))
    by_search = list(size_uniform(query, bound, random.Random(29)))

    assert [str(term) for term in by_sampler] == [str(term) for term in by_search]
    assert by_sampler


def test_size_uniform_is_random_search_under_term_size_and_a_flat_distribution():
    """``size_uniform`` is the shorthand, and the two must agree from the same seed.

    The cost function is what makes it size-uniform. Under a constant cost every inhabitant falls
    into one class and the draw becomes uniform over *terms* rather than over sizes, which is the
    thing the name promises not to do.
    """
    query = generator_query(list_space(), LIST)
    bound = 6

    shorthand = list(size_uniform(query, bound, random.Random(31)))
    spelled_out = list(random_search(query, bound, term_size, lambda _value: 1.0, random.Random(31)))

    assert [str(term) for term in shorthand] == [str(term) for term in spelled_out]
    assert shorthand


def test_every_realized_size_carries_the_same_total_weight():
    """Size-uniform means the sizes are equally likely, not the terms.

    Exact rather than statistical: on the list space the number of inhabitants of size ``s`` is
    ``3 ** (s - 1)``, so the weight of a single one of them has to be the reciprocal of that,
    divided by the number of realized sizes. If the two agree for every size, the distribution over
    sizes is flat, whatever the counts do.
    """
    bound = 6
    weighted = weighted_table(generator_query(list_space(), LIST), bound, lambda _value: 1.0)
    realized = [size for size, count in weighted.root_counts.items() if count]

    assert realized
    for size in realized:
        assert weighted.root_counts[size] == 3 ** (size - 1)
        assert weighted.root_counts[size] * weighted.unit_weights[size] == pytest.approx(1 / len(realized))


def test_the_keyed_search_hands_out_the_key_each_term_was_drawn_under():
    """``random_search_keyed`` is the same stream with its keys, and they decrease.

    The keys are what carries the guarantee, so a caller that wants to read the order the guarantee
    is about needs them rather than having to infer them from the sequence.
    """
    query = generator_query(list_space(), LIST)
    bound = 5

    keyed = list(random_search_keyed(query, bound, term_size, lambda _value: 1.0, random.Random(37)))
    plain = list(random_search(query, bound, term_size, lambda _value: 1.0, random.Random(37)))

    assert keyed
    assert [str(term) for _key, term in keyed] == [str(term) for term in plain]
    assert [key for key, _term in keyed] == sorted((key for key, _term in keyed), reverse=True)


# ---------------------------------------------------------------------------
# One construction per pair of questions, and which construction it is
# ---------------------------------------------------------------------------


def counted_branch_counts(monkeypatch):
    """Replace ``branch_counts`` by a counting wrapper, in both modules that reach it.

    ``sampling`` imported the name at module load, so patching only ``counting`` would leave the
    path the sampler actually takes unpatched, and the test would then pass on a sampler that
    builds the tree twice.

    Args:
        monkeypatch: pytest's patcher.

    Returns:
        list[int]: A one-element list holding the number of calls.
    """
    calls = [0]
    original = counting_module.branch_counts

    def wrapper(*arguments, **keywords):
        """Count one call and delegate.

        Args:
            *arguments: Passed through.
            **keywords: Passed through.

        Returns:
            CountedNode: What the real function returns.
        """
        calls[0] += 1
        return original(*arguments, **keywords)

    monkeypatch.setattr(counting_module, "branch_counts", wrapper)
    monkeypatch.setattr(sampling_module, "branch_counts", wrapper)
    return calls


def test_asking_and_then_drawing_counts_the_space_once(monkeypatch):
    """An initialization asks ``at_least`` and then draws, and counting must not happen twice.

    Counting is the expensive half of size-uniform sampling, on the list space at D = 10 it is
    59 048 retained nodes, and the initialization asks the bound question immediately before it
    draws. Building the retained tree for the question and then discarding it to build the same
    tree again for the draw doubles the cost of every initialization, and of the Bayesian loop's
    fallback path, which does this in a loop.
    """
    calls = counted_branch_counts(monkeypatch)
    query = generator_query(list_space(), LIST)
    sampler = SizeUniformSampler(7, random.Random(1))

    assert sampler.at_least(query, 40)
    drawn = list(islice(sampler.sample(query), 40))

    assert len(drawn) == 40
    assert calls[0] == 1


def test_a_different_query_is_counted_again(monkeypatch):
    """The reuse is a cache of one, and it must not answer for the wrong query.

    Resolution mutation poses a fresh residual query per call, so a cache that never invalidated
    would hand every mutation the completions of the first term it ever saw.
    """
    calls = counted_branch_counts(monkeypatch)
    space = list_space()
    sampler = SizeUniformSampler(5, random.Random(1))
    first = generator_query(space, LIST)
    second = generator_query(space, LIST)

    assert sampler.at_least(first, 1)
    assert sampler.at_least(second, 1)

    assert calls[0] == 2


def test_forgetting_releases_the_retained_tree(monkeypatch):
    """A caller who is done drawing has to be able to say so.

    The retained tree of a large space is the biggest object this package produces, and the cache
    would otherwise hold it until another query displaced it.
    """
    calls = counted_branch_counts(monkeypatch)
    query = generator_query(list_space(), LIST)
    sampler = SizeUniformSampler(5, random.Random(1))

    assert sampler.at_least(query, 1)
    sampler.forget()
    assert sampler.at_least(query, 1)

    assert calls[0] == 2


@pytest.mark.parametrize(("bound", "size"), [(5, 10), (7, 40)])
def test_both_constructions_of_the_counts_give_the_same_sample(bound, size):
    """``counting`` chooses how the branch counts are computed, not what is sampled.

    The two constructions answer the same question by different means, so with equally seeded
    generators they must produce the same stream, not merely the same distribution. Anything
    less would mean the table is not computing ``B_n`` but something that resembles it.
    """
    query = generator_query(list_space(), LIST)
    by_tree = list(islice(SizeUniformSampler(bound, random.Random(9), counting="tree").sample(query), size))
    by_table = list(islice(SizeUniformSampler(bound, random.Random(9), counting="table").sample(query), size))

    assert len(by_tree) == size
    assert by_tree == by_table


def test_both_constructions_agree_on_the_bound_question():
    """And on the number they answer it from.

    On the expression space, where the counts follow no closed form.
    """
    query = generator_query(expression_space(), EXPR)
    by_tree = SizeUniformSampler(6, random.Random(1), counting="tree")
    by_table = SizeUniformSampler(6, random.Random(1), counting="table")

    for count in (1, 37, 38, 39, 1000):
        assert by_tree.at_least(query, count) == by_table.at_least(query, count)


def test_the_table_construction_refuses_a_program_it_cannot_count():
    """No silent fallback: where the hypothesis fails, the sampler says so.

    ``constrained_space`` couples two holes, so the table would count the pairs the predicate
    rejects and the sampler would draw in proportion to weights that are wrong. Quietly falling
    back to the tree form would be defensible on the numbers and indefensible on the contract,
    since a caller who asked for the table asked because the tree form is out of reach.
    """
    query = generator_query(constrained_space(), PAIR)
    sampler = SizeUniformSampler(6, random.Random(1), counting="table")

    with pytest.raises(ValueError, match="reading a hole in a predicate"):
        sampler.at_least(query, 1)
    with pytest.raises(ValueError, match="reading a hole in a predicate"):
        next(iter(sampler.sample(query)))

    assert SizeUniformSampler(6, random.Random(1)).at_least(query, 1)


def test_a_construction_that_is_neither_is_an_error():
    """The parameter names a construction, and a typo must not silently pick one."""
    with pytest.raises(ValueError, match="'tree' or 'table'"):
        SizeUniformSampler(5, random.Random(1), counting="tabel")
