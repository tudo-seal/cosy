"""The clause order and the expansion filter of a search rule.

A search rule is a pair of a frontier and a clause order. The engine carried the frontier, in
the variance strategies, and the computation rule, in the subgoal selection, but the order of
the clauses of one expansion was fixed. These tests pin the parameter that supplies it, and the
consequence of testing a goal for success when it leaves the frontier rather than when it is
created.
"""

import random
from collections import Counter

import pytest

from cosy.core import Constructor, Synthesizer
from cosy.core.solution_space import (
    ConstantArgument,
    Goal,
    NonTerminalArgument,
    SolutionSpace,
    deepest_first_subgoal,
    fewest_arguments_first,
)
from cosy.core.tree import Tree
from cosy.core.types import Arrow, Intersection
from cosy.search import breadth_first, depth_first, generator_query, residual_query, uniform_random_clause_order
from cosy.search import deepest_first_subgoal as exported_deepest_first_subgoal
from cosy.search import fewest_arguments_first as exported_fewest_arguments_first
from tests.search_fixtures import (
    EXPR,
    START,
    WIDTH,
    A,
    B,
    C,
    D,
    a_only,
    add,
    b_only,
    c0,
    c_ab,
    disjoint_multi_path_space,
    equal_width_space,
    expression_space,
    lit,
    multi_path_space,
    nullary_start_space,
    wide_first_space,
    wrap_c,
)


def reversed_order(rules):
    """Arrange a clause sequence in reverse.

    Args:
        rules (Sequence[RHSRule]): The applicable clauses of one expansion.

    Returns:
        list: The same clauses, reversed.
    """
    return list(reversed(list(rules)))


def by_argument_sort(*, ascending):
    """Build a clause order that is directed: it sorts the clauses by their arguments' sorts.

    Every order the engine ships is uniform or arity based, and neither can show a direction. A
    uniform order is invariant under reversal, and an arity-based one cannot reorder inside the
    walk of a partial-term query at all, because every clause that survives the subtree match
    shares the prescribed arity. This one can.

    Args:
        ascending (bool): Whether to sort by the sort name ascending.

    Returns:
        Callable: The clause order.
    """

    def order(applicable):
        """Sort one expansion by the sorts of its arguments.

        Args:
            applicable (Sequence[RHSRule]): The applicable clauses of one expansion.

        Returns:
            tuple: The clauses, sorted by their arguments' sorts.
        """
        return tuple(
            sorted(
                applicable,
                key=lambda rule: [str(argument.origin) for argument in rule.arguments],
                reverse=not ascending,
            )
        )

    return order


def leaf_of(term):
    """Return the terminal of the leftmost leaf of a term.

    Args:
        term (Tree): The streamed term.

    Returns:
        The terminal that leaf carries.
    """
    while term.children:
        term = term.children[0]
    return term.root


# ---------------------------------------------------------------------------
# The clause order as a parameter
# ---------------------------------------------------------------------------


def test_a_clause_order_permutes_the_stream_without_changing_it():
    """Reversing the clause order permutes the stream and leaves its content alone.

    ``u1`` and ``u2`` open equally many holes, and the default clause order is a stable sort, so
    it keeps the order it was handed for them. The streams have to differ as sequences and agree
    as sets.
    """
    space = equal_width_space()
    baseline = [str(term) for term in space.depth_first_resolution(WIDTH, max_depth=3)]
    reordered = [str(term) for term in space.depth_first_resolution(WIDTH, max_depth=3, clause_order=reversed_order)]

    assert sorted(baseline) == sorted(reordered)
    assert baseline != reordered


def test_the_default_clause_order_puts_the_narrowest_clause_first():
    """``fewest_arguments_first`` is the default, and a caller's order replaces it.

    A stack frontier walks one branch of a recursive space to its end, and the branches of such a
    space have no end. The default puts the clause that opens fewest holes in front, so a success
    node is reached at a bounded depth. This space declares its clauses widest first, so the
    program order and the default disagree, and passing the program order through unchanged shows
    that the default is gone rather than applied on top.
    """
    space = wide_first_space()
    default = [term.root for term in space.depth_first_resolution(EXPR, max_depth=2)]
    program_order = [term.root for term in space.depth_first_resolution(EXPR, max_depth=2, clause_order=lambda r: r)]

    assert default[0] is lit
    assert program_order[0] is not lit
    assert sorted(map(str, default)) == sorted(map(str, program_order))


def test_the_clause_order_reaches_the_initial_goals():
    """The order reaches the first expansion of a run, not only the ones below it.

    Both clauses of ``S`` succeed at once, so the initial goals are the whole stream, and the
    first element flips with the order.
    """
    space = nullary_start_space()
    first_default = next(iter(space.depth_first_resolution(START)))
    first_reversed = next(iter(space.depth_first_resolution(START, clause_order=reversed_order)))

    assert str(first_default) != str(first_reversed)


def test_the_clause_order_is_consulted_at_every_expansion():
    """The engine asks the order once per expansion, and hands it the applicable clauses."""
    space = expression_space()
    sizes = []

    def record(rules):
        """Record the size of one applicable-clause sequence and pass it through.

        Args:
            rules (Sequence[RHSRule]): The applicable clauses of one expansion.

        Returns:
            list: The clauses, unchanged.
        """
        as_list = list(rules)
        sizes.append(len(as_list))
        return as_list

    list(space.depth_first_resolution(EXPR, max_count=5, max_depth=3, clause_order=record))

    assert sizes, "the clause order was never consulted"
    assert set(sizes) == {3}, f"every expansion carries the three clauses of E, got {set(sizes)}"


def test_fewest_arguments_first_sorts_by_the_holes_a_clause_opens():
    """The default order puts the clause that opens fewest holes first."""
    ordered = fewest_arguments_first(tuple(wide_first_space()[EXPR]))
    counts = [sum(1 for argument in rule.arguments if isinstance(argument, NonTerminalArgument)) for rule in ordered]

    assert counts == [0, 1, 2]
    assert ordered[0].terminal is lit


def test_fewest_arguments_first_keeps_ties_in_program_order():
    """Clauses that open equally many holes keep the order they were handed in.

    A stable sort is what lets a clause order decide anything between them, since both uninformed
    instances run the default over what the caller's order produced.
    """
    clauses = tuple(equal_width_space()[WIDTH])
    unary = [rule for rule in clauses if len(rule.arguments) == 1]
    assert len(unary) == 2, "the fixture has to offer a tie for this to say anything"

    ordered = [rule for rule in fewest_arguments_first(clauses) if len(rule.arguments) == 1]

    assert [rule.terminal for rule in ordered] == [rule.terminal for rule in unary]


# ---------------------------------------------------------------------------
# The success test belongs where a goal leaves the frontier
# ---------------------------------------------------------------------------


def first_terminals_over_seeds(space, count=50):
    """Collect the terminal each seed of a random clause order streams first.

    Args:
        space (SolutionSpace): The space to enumerate.
        count (int): The number of seeds to draw. (Default value = 50)

    Returns:
        set: The distinct roots of the first streamed term, one draw per seed.
    """
    return {
        next(
            iter(
                space.depth_first_resolution(
                    WIDTH, max_depth=3, clause_order=uniform_random_clause_order(random.Random(seed))
                )
            )
        ).root
        for seed in range(count)
    }


def test_the_clause_order_decides_which_inhabitant_is_streamed_first():
    """The clause order has to reach the first element of the stream.

    A goal handed out the moment it turns out successful never enters the frontier, so the
    nullary clauses of the start symbol come first whatever order was asked for. A search rule
    that draws its randomness from the clause order then draws that one term on every seed.
    """
    firsts = first_terminals_over_seeds(equal_width_space())

    assert len(firsts) > 1, f"50 seeds produced one and the same first inhabitant: {firsts}"


def test_a_random_clause_order_reaches_every_clause():
    """Each of the three clauses is the first one explored under some seed.

    The stronger reading of the test above, and it needs the frontier sort to be gone. That sort
    ran after the clause order had arranged an expansion, so a random order could permute clauses
    of equal width and nothing else.
    """
    firsts = first_terminals_over_seeds(equal_width_space())

    assert len(firsts) == 3, f"the three clauses of the space produced {len(firsts)} first inhabitants"


def test_reversing_the_clause_order_moves_the_nullary_clause():
    """The deterministic reading of the two tests above.

    With ``c0`` first in the rule list, the reversed order puts it behind ``u1`` and ``u2``, and
    a depth-first search reaches an inhabitant through those before it reaches ``c0`` itself.
    """
    space = equal_width_space()
    reordered = [term.root for term in space.depth_first_resolution(WIDTH, max_depth=3, clause_order=reversed_order)]

    assert reordered[0] is not c0


def test_the_stream_holds_the_same_terms_whatever_the_order():
    """Reordering clauses permutes the stream and does not change what is in it.

    The guard for the change above. Moving the success test to the frontier must not add or drop
    an inhabitant, on either frontier.
    """
    space = equal_width_space()
    baseline = {str(term) for term in space.depth_first_resolution(WIDTH, max_depth=3)}
    reordered = {str(term) for term in space.depth_first_resolution(WIDTH, max_depth=3, clause_order=reversed_order)}
    breadth = {str(term) for term in space.breadth_first_resolution(WIDTH, max_depth=3)}

    assert baseline == reordered == breadth
    assert len(baseline) > 1


def test_breadth_first_still_streams_the_shallow_inhabitants_first():
    """The frontier decides the order, and a queue reaches the shallow goals first.

    That includes the successful ones, which is what stops being true when a goal is handed out
    where it is created.
    """
    space = equal_width_space()
    sizes = [term.size for term in space.breadth_first_resolution(WIDTH, max_depth=4)]

    assert sizes == sorted(sizes), f"breadth-first streamed the sizes {sizes}"


# ---------------------------------------------------------------------------
# The order of a partial-term query
# ---------------------------------------------------------------------------


def test_the_clause_order_reaches_a_partial_term_query():
    """A query that prescribes a term is ordered like any other expansion."""
    space = expression_space()
    witness = next(iter(space.depth_first_resolution(EXPR, max_count=1, max_depth=2)))
    baseline = [str(term) for term in space.depth_first_resolution(EXPR, tree=witness, pos=(), max_depth=2)]
    reordered = [
        str(term)
        for term in space.depth_first_resolution(EXPR, tree=witness, pos=(), max_depth=2, clause_order=reversed_order)
    ]

    assert sorted(baseline) == sorted(reordered)
    assert baseline != reordered


def test_a_directed_clause_order_means_the_same_thing_at_every_position():
    """The order reaches the walk of a partial-term query the same way round as everything else.

    ``goal_from_tree`` pops its frontier from the right, so pushing the goals of an expansion in
    clause order would explore them back to front, while the shortcut for a hole at the root
    yields straight to its caller and runs with the order. Nothing that ships notices, because
    every order that ships is symmetric under reversal, so stating it needs a directed order: the
    root residual and a residual two levels down have to pick the same clause first, and
    reversing the order has to move both.
    """
    root_space = disjoint_multi_path_space(start=C)
    deep_space = disjoint_multi_path_space()
    root_witness = next(iter(root_space.depth_first_resolution(C, max_count=1, max_depth=4)))
    deep_witness = next(iter(deep_space.depth_first_resolution(D, max_count=1, max_depth=4)))

    picks = {}
    for ascending in (True, False):
        order = by_argument_sort(ascending=ascending)
        at_root = next(
            iter(root_space.depth_first_resolution(C, tree=root_witness, pos=(), max_depth=4, clause_order=order))
        )
        deep_down = next(
            iter(deep_space.depth_first_resolution(D, tree=deep_witness, pos=(0, 0), max_depth=4, clause_order=order))
        )
        picks[ascending] = (leaf_of(at_root), leaf_of(deep_down))

    assert picks[True][0] != picks[False][0], "the order has to move the root residual, or the comparison is empty"
    assert picks[True][0] == picks[True][1], f"ascending picked {picks[True]}"
    assert picks[False][0] == picks[False][1], f"descending picked {picks[False]}"


# ---------------------------------------------------------------------------
# The expansion filter
# ---------------------------------------------------------------------------


def test_a_goal_filter_drops_the_children_it_rejects():
    """A goal the filter rejects is dropped where it is created, so nothing below it is reached."""
    space = expression_space()
    unfiltered = {str(term) for term in space.depth_first_resolution(EXPR, max_count=20, max_depth=3)}
    filtered = {
        str(term)
        for term in space.depth_first_resolution(
            EXPR, max_count=20, max_depth=3, goal_filter=lambda goal: len(goal.subgoals) < 2
        )
    }

    assert filtered < unfiltered
    assert filtered, "a filter that admits the nullary clause must leave something"


def test_the_goal_filter_runs_on_successful_children():
    """A successful goal passes the filter like any other, so what it rejects is unreachable.

    A filter consulted only on goals that still carry work would let a rejected node through as
    soon as it happens to be a solution.
    """
    space = expression_space()
    stream = list(space.depth_first_resolution(EXPR, max_count=20, max_depth=3, goal_filter=lambda goal: False))

    assert stream == []


def test_a_sample_reaches_more_than_the_nullary_clause_of_the_start_symbol():
    """A seeded sample can draw any inhabitant within its bound, not one fixed term.

    ``sample_tree`` draws from the frontier, and a successful goal never entered it, so on a
    space whose start symbol has a nullary clause every draw returned that clause's term. The
    docstring of the method carried the defect as a note, which this change removes.
    """
    space = equal_width_space()

    drawn = {space.sample_tree(WIDTH, max_depth=3, rng=random.Random(seed)) for seed in range(30)}

    assert None not in drawn, "the fixture has to be inhabited within the bound"
    assert len({tree.root for tree in drawn}) == 3


def test_a_clause_order_that_drops_a_clause_is_refused():
    """An order is a permutation of what it receives.

    An order that returns fewer rules leaves the inhabitants of the dropped clause out of the
    stream, and a caller reads the result as a space without them.
    """
    space = expression_space()

    with pytest.raises(ValueError, match="a clause order is a permutation"):
        list(space.depth_first_resolution(EXPR, max_depth=2, clause_order=lambda rules: tuple(rules)[1:]))


TOP = Constructor("Top")


def program_order(rules):
    """Keep the clauses in the order they were declared.

    Args:
        rules (Sequence[RHSRule]): The applicable clauses of one expansion.

    Returns:
        Sequence[RHSRule]: The same clauses, unchanged.
    """
    return rules


def terminals_of(term):
    """Yield the terminal of every node of a term.

    Args:
        term (Tree): The term to walk.

    Yields:
        The terminal of one node.
    """
    yield term.root
    for child in term.children:
        yield from terminals_of(child)


def bin_c(left, right):
    """Pair two terms of sort ``C``.

    Args:
        left (str): The interpreted left operand.
        right (str): The interpreted right operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"<{left}|{right}>"


def two_branch_space():
    """Build the space in which a prescribed term has two independently ambiguous positions.

    ``wrap_c`` reaches ``C`` through two clauses and ``c_ab`` completes either, so each argument
    of ``bin_c`` is a binary choice of its own.

    Returns:
        SolutionSpace: The space, started at ``Top``.
    """
    specs = {
        c_ab: Intersection(A, B),
        a_only: A,
        b_only: B,
        wrap_c: Intersection(Arrow(A, C), Arrow(B, C)),
        bin_c: Arrow(C, Arrow(C, TOP)),
    }
    return Synthesizer(specs).construct_solution_space(TOP)


def pair_of(left, right):
    """Build the term of the clause whose two arguments are constants.

    Args:
        left (str): The first constant.
        right (str): The second constant.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"[{left},{right}]"


def wrap_s(inner):
    """Build the term of the recursive clause.

    Args:
        inner (str): The interpreted operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"w({inner})"


def constants_and_holes_space():
    """Build the space whose hole-free clause carries more arguments than its recursive one.

    Returns:
        SolutionSpace: The space, over the non-terminal ``S``.
    """
    space = SolutionSpace()
    space.add_rule("S", wrap_s, (NonTerminalArgument("inner", "S"),), ())
    space.add_rule("S", pair_of, (ConstantArgument("left", "0", None), ConstantArgument("right", "1", None)), ())
    return space


def test_a_depth_bound_of_zero_keeps_reading_the_frontier():
    """A goal that ``max_depth`` forbids to expand is skipped, and the run goes on.

    Ending the run there loses every goal behind it in the frontier, and in program order this
    space puts its only inhabitant of depth zero last.
    """
    space = wide_first_space()

    streamed = [term.root for term in space.depth_first_resolution(EXPR, max_depth=0, clause_order=program_order)]

    assert streamed == [lit]


def test_a_depth_bound_of_zero_keeps_reading_a_queue_frontier():
    """The same on the other frontier."""
    space = wide_first_space()

    streamed = [term.root for term in space.breadth_first_resolution(EXPR, max_depth=0, clause_order=program_order)]

    assert streamed == [lit]


def test_the_goal_filter_reaches_the_children_of_an_expansion():
    """A child the filter rejects is dropped, so nothing below it is reached.

    Every clause applying ``add`` leaves a goal with two open subgoals, which this filter rejects,
    so no streamed inhabitant can contain ``add``. A filter consulted only on the initial goals
    passes this space anyway, because it drops ``add`` at the root either way.
    """
    space = expression_space()

    streamed = list(
        space.depth_first_resolution(EXPR, max_count=20, max_depth=3, goal_filter=lambda goal: len(goal.subgoals) < 2)
    )

    assert streamed, "the filter admits the nullary and the unary clause"
    offenders = [str(term) for term in streamed if add in set(terminals_of(term))]
    assert offenders == [], f"the filter rejects every goal that applies add, yet {offenders} were streamed"


def test_the_goal_filter_never_sees_a_child_the_depth_bound_rejects():
    """``max_depth`` is the bound the engine carries itself, so it runs first.

    The filter is caller code. Consulting it about a child that is dropped anyway costs a call
    and shows the caller a node the search never reaches.
    """
    space = expression_space()
    seen = []

    def record(goal):
        """Record the depth of one goal and admit it.

        Args:
            goal (Goal): The goal offered to the filter.

        Returns:
            bool: True, always.
        """
        paths = list(goal.grounded.keys()) + list(goal.subgoals.keys())
        seen.append(max(len(path) for path in paths))
        return True

    list(space.depth_first_resolution(EXPR, max_count=20, max_depth=2, goal_filter=record))

    assert seen, "the filter was never consulted"
    assert max(seen) <= 2, f"the filter saw a goal of depth {max(seen)}"


def test_a_partial_term_query_survives_a_consumer_that_adds_a_clause():
    """``goal_from_tree`` yields inside its loop over the clauses, so it loops over a copy.

    ``_rules_of`` hands out the live deque of a non-terminal, and a consumer of the goal stream
    runs between two yields.
    """
    space = expression_space()
    goals = []

    for goal in space.goal_from_tree(EXPR, Tree(lit), ()):
        goals.append(goal)
        space.add_rule(EXPR, lit, (), ())

    assert len(goals) == 3


def test_the_clause_order_receives_a_copy_of_the_stored_clauses():
    """An order that passes its input through must not leave the engine on the live deque.

    ``resolution`` runs the ``goal_filter`` inside its loop over the ordered clauses, so a filter
    that touches the space would raise where the same code worked before.
    """
    space = expression_space()

    def grow(_goal):
        """Add a clause to the space and admit every goal.

        Args:
            _goal (Goal): The goal offered to the filter, which this one does not read.

        Returns:
            bool: True, always.
        """
        space.add_rule(EXPR, lit, (), ())
        return True

    streamed = list(
        space.depth_first_resolution(EXPR, max_count=3, max_depth=2, clause_order=lambda rules: rules, goal_filter=grow)
    )

    assert streamed


def test_the_initial_goals_of_a_walk_are_in_clause_order():
    """The first expansion of the walk is ordered, and it keeps the direction of the frontier.

    Starting at ``C``, whose two clauses share their terminal and differ in their argument sort,
    makes both visible: a directed order has to move the whole stream, and the reversal has to
    keep it that way round.
    """
    space = disjoint_multi_path_space(start=C)
    witness = next(iter(space.depth_first_resolution(C, max_count=1, max_depth=4)))

    streamed = {}
    for ascending in (True, False):
        terms = space.depth_first_resolution(
            C, tree=witness, pos=(0,), max_depth=4, clause_order=by_argument_sort(ascending=ascending)
        )
        streamed[ascending] = [leaf_of(term) for term in terms]

    assert streamed[True] == [a_only, b_only], streamed
    assert streamed[False] == [b_only, a_only], streamed


def test_the_walk_takes_the_leftmost_of_the_deepest_open_subgoals():
    """The walk fills the positions of a query left to right, so its goals group by the left one.

    Both arguments of ``bin_c`` are reached by the two clauses of ``C``, and ``c_ab`` completes
    either, so the query holds two independent binary choices. Expanding the right one first
    interleaves the four goals instead of grouping them.
    """
    space = two_branch_space()
    tree = Tree(bin_c, (Tree(wrap_c, (Tree(c_ab),)), Tree(wrap_c, (Tree(c_ab),))))

    origins = [str(goal.subgoals[(0, 0)].origin) for goal in space.goal_from_tree(TOP, tree, (0, 0))]

    assert len(origins) == 4, origins
    assert origins[0] == origins[1], origins
    assert origins[2] == origins[3], origins
    assert origins[0] != origins[2], origins


def test_the_default_clause_order_counts_holes_and_not_arguments():
    """A constant argument is no hole, so it must not push a clause back in the order."""
    space = constants_and_holes_space()

    first = next(iter(space.depth_first_resolution("S", max_depth=3)))

    assert first.root is pair_of, f"the default order streamed {first.root} first"


def test_deepest_first_subgoal_refuses_a_success_node():
    """A success node has nothing to expand, and returning something would hide the caller's bug."""
    space = expression_space()
    goals = [Goal.from_rhs_rule(rhs) for rhs in space[EXPR]]
    success = next(goal for goal in goals if goal is not None and goal.success)

    with pytest.raises(ValueError, match="success node"):
        deepest_first_subgoal(success)


def test_the_stream_hands_out_each_inhabitant_once():
    """Distinct success branches can determine the same term, and it is streamed once.

    ``wrap_c(c_ab)`` is derivable through both clauses of ``C``.
    """
    space = multi_path_space()

    streamed = [str(term) for term in space.depth_first_resolution(C, max_depth=3)]

    assert len(streamed) == len(set(streamed)), streamed


# ---------------------------------------------------------------------------
# The named rules over a query
# ---------------------------------------------------------------------------


NAMED_RULES = [pytest.param(depth_first, id="depth_first"), pytest.param(breadth_first, id="breadth_first")]

ENGINE_ENTRIES = [
    pytest.param(depth_first, SolutionSpace.depth_first_resolution, id="depth_first"),
    pytest.param(breadth_first, SolutionSpace.breadth_first_resolution, id="breadth_first"),
]


@pytest.mark.parametrize(("rule", "engine"), ENGINE_ENTRIES)
def test_a_named_rule_streams_what_its_engine_entry_streams(rule, engine):
    """The named rule is the engine entry of its frontier, read off a query.

    Both rules stream the same terms and only their order tells them apart, so a rule that
    reaches for the other frontier is visible in the order and nowhere else.
    """
    space = expression_space()

    assert [str(term) for term in rule(generator_query(space, EXPR), max_count=50, max_depth=2)] == [
        str(term) for term in engine(space, EXPR, max_count=50, max_depth=2)
    ]


@pytest.mark.parametrize("rule", NAMED_RULES)
def test_a_named_rule_passes_its_bounds_to_the_engine(rule):
    """Both bounds reach the engine, each one observed while the other one holds the space finite."""
    query = generator_query(equal_width_space(), WIDTH)

    assert len(list(rule(query, max_count=5, max_depth=3))) == 5
    shallow = list(rule(query, max_count=20, max_depth=1))
    deeper = list(rule(query, max_count=20, max_depth=3))

    assert 0 < len(shallow) < len(deeper)


@pytest.mark.parametrize("rule", NAMED_RULES)
def test_a_named_rule_answers_a_partial_term_query(rule):
    """The query term of a partial-term query reaches the engine, its prescribed part included.

    A rule that drops it asks the generator instead, which streams the whole language rather
    than the completions of one term, and every completion below then carries the wrong head.
    """
    space = expression_space()
    witness = next(term for term in rule(generator_query(space, EXPR), max_count=20, max_depth=2) if term.children)

    completions = list(rule(residual_query(space, EXPR, witness, (0,)), max_count=50, max_depth=2))

    assert completions, "the opened position has completions within the bound"
    assert all(term.root is witness.root for term in completions)


@pytest.mark.parametrize(("rule", "engine"), ENGINE_ENTRIES)
def test_a_named_rule_passes_the_clause_order_and_the_expansion_filter(rule, engine):
    """The two callbacks a caller supplies reach the engine, and the clause order reaches it whole.

    The stream under the caller's order is compared against the engine under the same order, not
    merely against the unordered one. An order the rule joined with the default instead of
    replacing it also permutes the stream, so a comparison against the unordered stream cannot
    tell the two apart, and joining is what the engine stopped doing.
    """
    space = equal_width_space()
    query = generator_query(space, WIDTH)
    baseline = [str(term) for term in rule(query, max_count=50, max_depth=3)]
    reordered = [str(term) for term in rule(query, max_count=50, max_depth=3, clause_order=reversed_order)]

    assert sorted(baseline) == sorted(reordered)
    assert baseline != reordered
    assert reordered == [
        str(term) for term in engine(space, WIDTH, max_count=50, max_depth=3, clause_order=reversed_order)
    ]
    assert list(rule(query, max_count=50, max_depth=3, goal_filter=lambda goal: False)) == []


def test_uniform_random_clause_order_is_a_function_of_its_generator():
    """Same seed, same stream: the property every search built on the order inherits."""
    query = generator_query(equal_width_space(), WIDTH)

    def stream(seed):
        """Enumerate under a clause order drawn from one seed.

        Args:
            seed (int): The seed of the generator the order draws from.

        Returns:
            list: The streamed terms, rendered.
        """
        order = uniform_random_clause_order(random.Random(seed))
        return [str(term) for term in depth_first(query, max_count=50, max_depth=4, clause_order=order)]

    assert stream(7) == stream(7)
    assert stream(7) != stream(1)
    assert sorted(stream(7)) == sorted(stream(1))


def test_uniform_random_clause_order_draws_a_fresh_permutation_per_expansion():
    """Every draw is a permutation, and the draws differ from one expansion to the next.

    The engine only counts what an order returns, so an order that keeps the clauses but repeats
    one and drops another passes its check; that is why the set is compared here as well. And an
    order that drew once and replayed its result would still be a permutation, so the draws have
    to be told apart from each other rather than from program order alone.
    """
    clauses = tuple(expression_space().get(EXPR))
    order = uniform_random_clause_order(random.Random(3))

    draws = [tuple(order(clauses)) for _ in range(10)]

    assert all(set(drawn) == set(clauses) and len(drawn) == len(clauses) for drawn in draws)
    assert len(set(draws)) > 1, "an order that draws once and replays it is not drawing per expansion"


def test_uniform_random_clause_order_reaches_every_permutation_about_equally_often():
    """Uniform is in the name, so it is what the draws have to be.

    A biased order can be seeded, reproducible and a permutation, and still put one clause in
    front far more often than the others. Nothing else in this file would notice: the tests over
    the streamed terms only ask that each clause be reached under some seed. Three clauses have
    six permutations, and 600 draws leave every count far from the bound below.
    """
    clauses = tuple(expression_space().get(EXPR))
    order = uniform_random_clause_order(random.Random(11))

    counts = Counter(tuple(id(clause) for clause in order(clauses)) for _ in range(600))

    assert len(counts) == 6, "the six permutations of three clauses are all drawn"
    assert min(counts.values()) > 50, f"a permutation is drawn far below its share: {sorted(counts.values())}"


@pytest.mark.parametrize("rule", NAMED_RULES)
def test_a_named_rule_hands_back_a_stream(rule):
    """The rule returns the engine's stream rather than a materialized list.

    A rule that collects the stream before returning it answers an unbounded query by not
    returning at all, and the bounds are the caller's to choose.
    """
    stream = rule(generator_query(expression_space(), EXPR), max_count=50, max_depth=2)

    assert iter(stream) is stream


def test_the_named_rules_hand_out_the_engine_s_own_functions():
    """The computation rule and the default clause order are the engine's, not a copy of them.

    A caller that has to expand nodes the way the engine expands them needs the function the
    engine uses. Two copies of it agree by coincidence, and nothing in a stream of terms shows
    that they have stopped agreeing.
    """
    assert exported_deepest_first_subgoal is deepest_first_subgoal
    assert exported_fewest_arguments_first is fewest_arguments_first
