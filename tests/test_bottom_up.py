"""Bottom-up search and the clause applications it is counted in.

The oracle throughout is the enumeration the framework already carries: bottom-up search computes
the least Herbrand model of a program, and the terms of one non-terminal in that model are exactly
the terms ``enumerate_trees`` streams from it. Two implementations of one language check each
other, so one test writes its expected terms out by hand instead, in case both are wrong together.

The counters are the second subject. A goal-driven search rule reports node expansions, and this
rule has no nodes, so it reports clause applications instead. The enumeration advances the same
two counters, which is what makes the work of the two comparable rather than only their seconds.
"""

import pytest

from cosy.core import Constructor, Synthesizer
from cosy.core.solution_space import SolutionSpace
from cosy.core.types import Arrow
from cosy.search import bottom_up, least_herbrand_model, term_depth
from cosy.search.bottom_up import BottomUpCounters
from tests._hash_seeds import printed_across_hash_seeds
from tests.search_fixtures import (
    AMBIGUOUS_TARGET,
    BOX,
    CHAIN,
    EXPR,
    LIST,
    NUM,
    PAIR,
    START,
    TAGGED,
    TUPLE_SORT,
    USED,
    C,
    D,
    ambiguous_space,
    chain_space,
    constrained_space,
    cut_space,
    disjoint_multi_path_space,
    expression_space,
    hole_tuple_space,
    list_space,
    literal_predicate_space,
    literal_space,
    multi_path_space,
    nullary_start_space,
    two_predicate_space,
    two_symbol_clause_space,
)

# The spaces whose least Herbrand model is finite, so a run of either kind ends on its own and the
# two can be compared in full. They cover what the operator has to get right: clauses whose
# non-terminal arguments carry no name, clauses that carry predicates, and clauses that write a
# literal into the term.
FINITE_SPACES = [
    ("ambiguous", ambiguous_space, AMBIGUOUS_TARGET),
    ("multi path", multi_path_space, C),
    ("disjoint multi path", disjoint_multi_path_space, D),
    ("literal predicate", literal_predicate_space, USED),
    ("two predicates", two_predicate_space, USED),
    ("hole tuple", hole_tuple_space, TUPLE_SORT),
    ("nullary start", nullary_start_space, START),
]

# The spaces whose model is infinite, so every run over them needs a bound.
INFINITE_SPACES = [
    ("constrained", constrained_space, PAIR),
    ("list", list_space, LIST),
    ("expression", expression_space, EXPR),
    ("chain", chain_space, CHAIN),
    ("two symbols per clause", two_symbol_clause_space, TAGGED),
    ("literal", literal_space, NUM),
]


# ---------------------------------------------------------------------------
# The language: sound and complete
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "build", "start"), FINITE_SPACES)
def test_the_stream_is_the_language_the_enumeration_streams(name, build, start) -> None:
    """Every streamed term inhabits the start, and every inhabitant is streamed.

    Args:
        name (str): The name of the space, for the test id.
        build (Callable[[], SolutionSpace]): Builds the space.
        start (NT): The queried non-terminal.
    """
    space = build()

    assert set(bottom_up(space, start)) == set(space.enumerate_trees(start)), name


def test_the_ambiguous_space_streams_the_two_terms_it_holds() -> None:
    """One space has its answer written out, so the two implementations cannot agree on a wrong one.

    ``merge`` reaches ``M`` from either of two argument sorts and ``base`` inhabits both, so the
    space derives ``m(b)`` twice and ``m(a)`` once. A set of terms holds each of them once.
    """
    streamed = [term.interpret() for term in bottom_up(ambiguous_space(), AMBIGUOUS_TARGET)]

    assert sorted(streamed) == ["m(a)", "m(b)"]


@pytest.mark.parametrize(("name", "build", "start"), FINITE_SPACES + INFINITE_SPACES)
def test_every_streamed_term_is_one_the_space_contains(name, build, start) -> None:
    """A third opinion, so that the two enumerations cannot be wrong together.

    ``contains_tree`` decides membership by walking the derivation tree of the term rather than by
    building the language, so it shares no machinery with either enumeration. It answers soundness
    alone, which is why it stands beside the comparison above rather than replacing it. Under a
    bound it also reaches the spaces whose model never ends, which is where the clauses of more
    than one argument and the clauses that write a literal live.

    Args:
        name (str): The name of the space, for the test id.
        build (Callable[[], SolutionSpace]): Builds the space.
        start (NT): The queried non-terminal.
    """
    space = build()

    streamed = list(bottom_up(space, start, max_count=20))

    assert streamed, name
    for term in streamed:
        assert space.contains_tree(start, term), f"{name}: {term}"


@pytest.mark.parametrize(("name", "build", "start"), INFINITE_SPACES)
def test_the_stream_never_returns_to_a_shallower_derivation(name, build, start) -> None:
    """The rounds order the stream, so a term never follows one that took more rounds to derive.

    On these spaces one clause application adds one level to the term, so the round a term is
    derived in is its depth and the order is observable. Within one round the order is not
    specified and carries no meaning.

    Args:
        name (str): The name of the space, for the test id.
        build (Callable[[], SolutionSpace]): Builds the space.
        start (NT): The queried non-terminal.
    """
    depths = [term_depth(term) for term in bottom_up(build(), start, max_count=40)]

    assert depths == sorted(depths), f"{name}: {depths}"


def test_a_predicate_over_two_argument_choices_rejects_without_ending_the_clause() -> None:
    """A clause whose predicate couples two argument positions has to try every combination.

    Every other predicate in the reference spaces decides on the literals of its clause and falls
    the same way for all of its argument choices, so leaving the clause at the first rejection
    instead of moving to the next combination would pass unnoticed. Here it does not: the
    predicate holds for the pairs of distinct words and fails for the rest, and a clause abandoned
    at its first failure streams nothing at all.
    """
    counters = BottomUpCounters()

    streamed = list(bottom_up(constrained_space(), PAIR, counters=counters, stop=lambda c: c.rounds >= 5))

    assert len(streamed) == 56
    assert counters.applications == 151
    assert counters.derivations == 130
    for term in streamed:
        left, right = term.children
        assert left != right


# ---------------------------------------------------------------------------
# The two bounds a caller can put on a run
# ---------------------------------------------------------------------------


def test_max_count_stops_the_stream_at_a_prefix_of_it() -> None:
    """A bounded run streams the same terms in the same order as the unbounded one, and stops."""
    space = list_space()

    bounded = list(bottom_up(space, LIST, max_count=7))
    full = list(bottom_up(list_space(), LIST, max_count=20))

    assert len(bounded) == 7
    assert bounded == full[:7]


def test_a_bound_of_zero_streams_nothing() -> None:
    """A caller who asks for no inhabitants gets none, as the enumeration gives none.

    Testing the bound after the yield rather than before it let one term through, which is the
    term a caller who asked for none would have had to discard.
    """
    space = list_space()

    assert list(bottom_up(space, LIST, max_count=0)) == []
    assert list(space.enumerate_trees(LIST, max_count=0)) == []


def test_max_count_still_binds_when_a_budget_is_set_too() -> None:
    """Two bounds on one run, and the one that is reached first ends it.

    A budget that would allow far more work does not release the count, and the counters show
    that the run stopped at the term rather than at the budget. Reading them is also what pins
    the second test of the bound: without it the run pays for a round it no longer needs.
    """
    counters = BottomUpCounters()

    streamed = list(bottom_up(list_space(), LIST, max_count=3, counters=counters, stop=lambda c: c.applications >= 100))

    assert len(streamed) == 3
    assert counters.applications == 5
    assert counters.rounds == 2


def test_a_run_bounded_at_one_term_pays_for_one_round() -> None:
    """The bound ends the run where it is reached, not after the round that would follow."""
    counters = BottomUpCounters()

    list(bottom_up(list_space(), LIST, max_count=1, counters=counters))

    assert counters.rounds == 1
    assert counters.applications == 1
    assert counters.derivations == 1
    assert counters.atoms == 1


def test_max_count_beyond_the_language_yields_all_of_it() -> None:
    """A bound the model never reaches leaves the stream whole and does not hang."""
    space = multi_path_space()

    assert set(bottom_up(space, C, max_count=1000)) == set(space.enumerate_trees(C))


def test_max_count_does_not_end_a_run_whose_start_is_already_exhausted() -> None:
    """The halting condition is a property of the program, not of the queried non-terminal.

    ``cut_space`` bounds the size of what its predicate lets into ``Box``, so ``Box`` holds four
    terms and no more, while the sort those terms are built from grows without end. A round
    derives for every non-terminal, so the run goes on deriving that sort long after the fourth
    term was streamed, and a bound of ten is never reached. ``stop`` is the bound that bites.
    """
    counters = BottomUpCounters()

    streamed = list(bottom_up(cut_space(), BOX, max_count=10, counters=counters, stop=lambda c: c.rounds >= 12))

    assert len(streamed) == 4
    assert counters.rounds == 13
    assert counters.atoms == 28


def test_a_budget_stops_a_program_whose_model_is_infinite() -> None:
    """A run that would not halt ends at the budget, and says that it was stopped.

    The counters are the evidence: a run stopped at its budget has done the work the budget
    allowed and no more. The numbers are written out, because a run that stopped too early would
    also satisfy "it stopped".
    """
    counters = BottomUpCounters()

    streamed = list(bottom_up(expression_space(), EXPR, counters=counters, stop=lambda c: c.applications >= 500))

    assert counters.applications == 500
    assert counters.derivations == 499
    assert counters.rounds == 5
    assert len(streamed) == 455
    assert counters.atoms == 455


def test_a_budget_is_consulted_inside_a_round_and_not_between_rounds() -> None:
    """The stopping condition has to bite where the operator works, not where a round ends.

    A round of the operator on a large model outgrows the memory of the machine before it ends,
    so a check taken between rounds would not be reached. Three applications is less than the
    second round of this space costs, and the run stops at exactly three.
    """
    counters = BottomUpCounters()

    list(bottom_up(expression_space(), EXPR, counters=counters, stop=lambda c: c.applications >= 3))

    assert counters.applications == 3
    assert counters.rounds == 2


# ---------------------------------------------------------------------------
# The counters
# ---------------------------------------------------------------------------


def test_the_counters_a_caller_passes_are_the_ones_the_run_advances() -> None:
    """A caller that brings its own counters reads the work of the run off them."""
    counters = BottomUpCounters()

    streamed = list(bottom_up(disjoint_multi_path_space(), D, counters=counters))

    assert len(streamed) == 2
    assert counters.rounds == 4
    assert counters.atoms == 6
    assert counters.applications == counters.derivations == 18


def test_counters_survive_a_stream_the_caller_abandons() -> None:
    """The work of an abandoned run is readable, which is what a measurement under a budget needs."""
    counters = BottomUpCounters()

    stream = bottom_up(list_space(), LIST, counters=counters)
    next(stream)

    assert counters.rounds == 1
    assert counters.applications == 1
    assert counters.atoms == 1


def test_predicates_separate_the_applications_from_the_derivations() -> None:
    """An instance whose predicates fail is an application that derives nothing."""
    counters = BottomUpCounters()

    streamed = list(bottom_up(two_predicate_space(), USED, counters=counters))

    assert len(streamed) == 1
    assert counters.applications == 11
    assert counters.derivations == 5


@pytest.mark.parametrize(("name", "build", "start"), [FINITE_SPACES[0], FINITE_SPACES[1], FINITE_SPACES[5]])
def test_a_program_without_predicates_derives_every_application(name, build, start) -> None:
    """With no predicate to reject an instance, every application derives a term.

    Args:
        name (str): The name of the space, for the test id.
        build (Callable[[], SolutionSpace]): Builds the space.
        start (NT): The queried non-terminal.
    """
    counters = BottomUpCounters()

    list(bottom_up(build(), start, counters=counters))

    assert counters.applications == counters.derivations > 0, name


def test_the_rounds_and_the_atoms_report_the_iteration() -> None:
    """The last round is counted, and the atom count is that of the model it produced."""
    counters = BottomUpCounters()

    list(bottom_up(nullary_start_space(), START, counters=counters))

    assert counters.rounds == 2
    assert counters.atoms == 2


# ---------------------------------------------------------------------------
# The model as a whole
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "build"), [(name, build) for name, build, _start in FINITE_SPACES])
def test_the_model_holds_the_inhabitants_of_every_non_terminal(name, build) -> None:
    """The fixpoint is a model of the whole program, not of the queried non-terminal alone.

    Args:
        name (str): The name of the space, for the test id.
        build (Callable[[], SolutionSpace]): Builds the space.
    """
    space = build()

    model = least_herbrand_model(space)

    for nonterminal in space.nonterminals():
        assert model.get(nonterminal, set()) == set(space.enumerate_trees(nonterminal)), f"{name}/{nonterminal}"


def test_a_model_a_budget_stopped_is_partial() -> None:
    """A budget cuts the ascent short, and what comes back is an iterate rather than the fixpoint."""
    counters = BottomUpCounters()

    stopped = least_herbrand_model(expression_space(), counters=counters, stop=lambda c: c.applications >= 50)

    assert counters.applications == 50
    assert counters.rounds == 4
    assert sum(len(terms) for terms in stopped.values()) == 35


def test_the_model_carries_every_non_terminal_of_the_program() -> None:
    """A non-terminal the program derives nothing for is present and empty, not absent."""
    model = least_herbrand_model(hole_tuple_space())

    assert {str(nonterminal): len(terms) for nonterminal, terms in model.items()} == {"Ht": 6, "Ha": 2, "Hb": 1}


def test_a_non_terminal_the_program_derives_nothing_for_stays_in_the_model() -> None:
    """An empty set is an answer. Dropping the entry would make an empty language look absent.

    ``wrap`` needs a term of its own sort to build one, and the program offers no way to start,
    so the sort is a non-terminal of the program with no inhabitant at all.
    """
    sort = Constructor("W")

    def wrap(inner: str) -> str:
        """Wrap a term of the sort in another one.

        Args:
            inner (str): The interpreted argument.

        Returns:
            str: Its rendering under interpret.
        """
        return f"w({inner})"

    space = Synthesizer({wrap: Arrow(sort, sort)}).construct_solution_space(sort)

    assert least_herbrand_model(space) == {sort: set()}
    assert list(bottom_up(space, sort)) == []


def test_a_program_without_rules_has_an_empty_model() -> None:
    """The empty program reaches its fixpoint at once, on the first round."""
    counters = BottomUpCounters()
    empty: SolutionSpace[str, str, None] = SolutionSpace()

    assert least_herbrand_model(empty, counters=counters) == {}
    assert counters.rounds == 1


def test_a_non_terminal_the_program_does_not_have_streams_nothing() -> None:
    """A query for a non-terminal outside the program is empty, not an error."""
    assert list(bottom_up(multi_path_space(), Constructor("Absent"))) == []


# ---------------------------------------------------------------------------
# Reproducibility across processes
# ---------------------------------------------------------------------------


def test_the_stream_is_reproducible_across_processes() -> None:
    """Which terms a bounded run returns, and in which order, must not vary per process.

    The model of a non-terminal was a set of trees, and a tree over a string terminal hashes by a
    value the interpreter chooses at startup. Under a bound that decided not only the order of the
    stream but which terms it held, and the enumeration this search is compared against promises
    the opposite. Written out rather than compared for self-consistency, since an order that is
    stable but wrong would satisfy "all runs agree" and nothing else.
    """
    printed = printed_across_hash_seeds(
        "from cosy.search import bottom_up\n"
        "from tests._determinism_grammars import branching_space, mixed_width_space\n"
        'print([str(t) for t in bottom_up(mixed_width_space(), "S", max_count=5)])\n'
        'print([str(t) for t in bottom_up(branching_space(), "S", max_count=6)])\n'
    )

    assert printed == {
        "['top lf lf', 'top lf (un lf)', 'top lf (tri lf lf lf)', 'top lf (bi lf lf)', 'top (un lf) lf']\n"
        "['top a1 b1', 'top a1 (b2 b1)', 'top a1 (b3 a1)', 'top (a2 a1) b1', 'top (a2 a1) (b2 b1)', "
        "'top (a2 a1) (b3 a1)']"
    }
