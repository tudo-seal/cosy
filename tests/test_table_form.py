"""The table form of the branch counts: the same numbers, computed from the program.

:func:`~cosy.search.counting.branch_counts` materializes the retained derivation tree and counts
its success branches. That is exact and needs no hypothesis, and it costs the *number of
inhabitants*: the counts have to be complete before the first element leaves the stream, so a space
with a million inhabitants pays for all of them in order to draw forty. A bound of a hundred on the
list space puts the retained tree on the order of ``3^100`` nodes, which the tree form cannot reach
at all.

:func:`~cosy.search.counting.size_table` computes the same numbers from the *program*: one row
``N_A(s)`` per non-terminal and size, filled by increasing ``s``, at a cost in the size of the
program and the bound alone. This file pins the two things that have to hold for that to be a
replacement rather than a second implementation.

* **The numbers agree.** At the root the table row *is* the branch count, on every reference space
  that satisfies the hypothesis, for every bound.
* **The hypothesis is decided, never assumed.** The table is indexed by the non-terminal, so it
  cannot see a predicate that reads a hole: such a predicate makes the residual at the hole a
  proper subset of the hole's language. :func:`~cosy.search.counting.decomposable_or_raise` decides
  this and raises with the offending clause named, rather than returning numbers that are quietly
  too large. Two tests carry the point that "no coupling" is *not* the hypothesis, since a
  predicate over a *single* hole breaks the table just as thoroughly as one over two.

With the sampling layer in place the file also pins what the two constructions *stream*, and not
only what they count. Where the hypothesis holds they produce the same stream from the same seed,
term for term and key for key, which is the evidence that only the computation of ``B_n`` changed
and not what ``B_n`` is. The reference example is drawn at a bound the tree form misses by a factor
of ``3^100``.

Nothing here needs statistics: the counts, the weights, the closed forms and the agreement of the
two constructions are all exact. The distributional statements, that a prefix of the stream is a
sample without replacement in proportion to ``w``, are not made here.
"""

import math
import random

import pytest

from cosy.core import Constructor, SpecificationBuilder, Synthesizer
from cosy.core.tree import Tree
from cosy.search import checker, depth_first, generator_query, residual_query, term_size
from cosy.search.counting import (
    CountedNode,
    CoupledClause,
    SizeTable,
    branch_counts,
    coupled_clauses,
    decomposable_or_raise,
    initial_nodes,
    size_table,
)
from cosy.search.sampling import WeightedTree, keyed_stream, log_sum_exp, weighted_table, weighted_tree
from tests._generate_and_check import (
    AMBIGUOUS_SIGNATURE,
    CHAIN_SIGNATURE,
    EXPR_SIGNATURE,
    LIST_SIGNATURE,
    TAGGED_SIGNATURE,
    inhabitants_within,
)
from tests.search_fixtures import (
    AMBIGUOUS_TARGET,
    BOX,
    CHAIN,
    CUT_SORT,
    EXPR,
    GRADED,
    HELD,
    LIST,
    MARKED,
    MIXED,
    PAIR,
    SORT_A,
    SORT_B,
    TAGGED,
    TERNARY,
    TUPLE_SORT,
    USED,
    WORD,
    add,
    ambiguous_space,
    anonymous_hole_space,
    below_two,
    chain_space,
    cons_0,
    cons_1,
    constrained_space,
    cut_space,
    expression_space,
    grade,
    halt,
    hole_tuple_space,
    list_space,
    lit,
    literal_predicate_space,
    marked,
    mixed_arity_space,
    neg,
    nil,
    offset_cut_space,
    positive,
    stop,
    tag,
    ternary_space,
    two_offender_space,
    two_predicate_space,
    two_symbol_clause_space,
    use,
    zero,
)

# The reference spaces the table form applies to: no predicate reads a hole in any of them. The
# ambiguous space is in the list on purpose, since the table counts *branches*, exactly as the tree
# form does, and the two have to agree there as well. The chain space is the two-symbol space with
# its clauses in the other order, and it is here rather than in one test of its own because the
# order matters to every claim in the file and not only to the fill: it is the one space whose
# start symbol realizes nothing at all at some bounds this parametrization runs.
DECOMPOSABLE_SPACES = [
    ("list", list_space, LIST),
    ("expression", expression_space, EXPR),
    ("ambiguous", ambiguous_space, AMBIGUOUS_TARGET),
    ("two symbols per clause", two_symbol_clause_space, TAGGED),
    ("chain", chain_space, CHAIN),
]

# The signature of each space, for the one test whose oracle enumerates terms rather than reading
# the space. It is kept beside the list rather than in it, because a fourth element would be an
# unused argument in every other parametrization over it.
SIGNATURES = {
    "list": LIST_SIGNATURE,
    "expression": EXPR_SIGNATURE,
    "ambiguous": AMBIGUOUS_SIGNATURE,
    "two symbols per clause": TAGGED_SIGNATURE,
    "chain": CHAIN_SIGNATURE,
}

# The bound each space is streamed to in full. They differ because the spaces grow at different
# rates and the oracle enumerates every term of the signature up to it.
FULL_STREAM_BOUNDS = {
    "list": 7,
    "expression": 7,
    "ambiguous": 4,
    "two symbols per clause": 9,
    "chain": 7,
}

# The bound the list example is stated at. Out of reach for the tree form by a factor of 3^100.
GOLDEN_BOUND = 100


def realized_row(table, nonterminal):
    """Return the nonzero entries of one row of a size table.

    Args:
        table (SizeTable): The filled table.
        nonterminal: The non-terminal whose row is read.

    Returns:
        dict[int, int]: ``N_A(s)`` for every size the non-terminal realizes within the bound.
    """
    return {size: table.of(nonterminal, size) for size in range(table.bound + 1) if table.of(nonterminal, size)}


@pytest.mark.parametrize(("name", "build", "start"), DECOMPOSABLE_SPACES)
@pytest.mark.parametrize("bound", [0, 1, 2, 3, 5, 6])
def test_the_table_row_of_the_start_symbol_is_the_root_branch_count(name, build, start, bound):
    """``N_start(s) = B_r(s)``: the two ways to the counts arrive at the same numbers.

    The whole claim of the table form in one line.  The tree form is validated against exhaustive
    generate-and-check in ``test_branch_counts``, so pinning the table against the tree pins it
    against brute force. One comparison rather than two, and the intermediate is the quantity both
    constructions are defined by.

    The bound runs from 0 (nothing realized) upwards, because the fill is a single pass by
    increasing size and an off-by-one in it would show at the ends first.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The queried non-terminal.
        bound (int): The size bound ``D`` under test.
    """
    space = build()
    table = size_table(space, bound)
    counted = branch_counts(generator_query(space, start), bound, term_size)
    assert realized_row(table, start) == dict(counted.counts), name


def test_the_list_table_follows_its_closed_form():
    """``N_List(s) = 3^(s-1)``: the one reference space whose answers are known in advance.

    A list of length ``l`` is a term of size ``l + 1`` and there are ``3^l`` of them, so this is
    the only check in the file that depends on no implementation at all.  The bound is past what
    the tree form is comfortable with, which is the point of having the table.
    """
    bound = 12
    table = size_table(list_space(), bound)
    assert realized_row(table, LIST) == {size: 3 ** (size - 1) for size in range(1, bound + 1)}


def test_the_literal_table_realizes_only_the_odd_sizes():
    """``N_Tagged(s) = 2^((s-1)//2)`` at odd ``s``, and nothing at even ``s``.

    ``tag`` writes its terminal *and* its literal argument, so one clause application grows the
    term by two symbols and every inhabitant has odd size.  A table that charged a clause for its
    terminal alone would fill the even rows too, and would admit terms of size ``2D - 1`` under a
    bound of ``D``.
    """
    bound = 11
    table = size_table(two_symbol_clause_space(), bound)
    row = realized_row(table, TAGGED)
    assert row == {size: 2 ** ((size - 1) // 2) for size in range(1, bound + 1, 2)}
    assert all(size % 2 == 1 for size in row)


def test_the_table_counts_branches_and_not_terms_on_an_ambiguous_space():
    """The identity of branch counts and cost counts needs unambiguity, and the table inherits it.

    ``merge(base)`` is derived twice, once through each of the two clauses ``merge`` emits, so
    the branch count exceeds the number of distinct inhabitants by one.  The table reproduces the
    excess exactly: it is the same recursion on the same clauses, so it counts derivations for the
    same reason the tree form does.  A table that silently deduplicated would agree with the term
    count here and disagree with the tree form, which is what a draw actually follows.
    """
    bound = 4
    space = ambiguous_space()
    table = size_table(space, bound)
    counted = branch_counts(generator_query(space, AMBIGUOUS_TARGET), bound, term_size)
    terms = inhabitants_within(space, AMBIGUOUS_TARGET, AMBIGUOUS_SIGNATURE, bound)

    assert realized_row(table, AMBIGUOUS_TARGET) == dict(counted.counts) == {2: 3}
    assert len({str(term) for term in terms}) == 2
    assert counted.total > len(terms), "an unambiguous space would prove nothing here"


def test_a_clause_without_holes_admits_exactly_one_split_and_only_of_size_zero():
    """The empty product: no holes to fill means one way to fill them, and only at size 0.

    The base case of the convolution.  Returning 0 at ``total == 0`` would make every nullary
    clause vanish from the table and empty it altogether; returning 1 at ``total > 0`` would hand
    a nullary clause the size budget of a recursive one.

    A negative total is on the list because the docstring answers for it, with one *only* at zero,
    and because a base case that answered one there would quietly cover for the guard in the fill
    that skips a clause too heavy for the current size.  Two mistakes that cancel are the kind
    this file exists to keep apart.
    """
    table = size_table(list_space(), 5)
    assert table.split_counts((), 0) == 1
    assert table.split_counts((), 1) == 0
    assert table.split_counts((), 5) == 0
    assert table.split_counts((), -1) == 0


def test_a_total_below_the_number_of_holes_admits_no_split():
    """Every hole takes at least one symbol, so ``k`` holes need at least ``k``.

    The compositions the recursion sums over are into *positive* parts. Allowing a part
    of zero would count a hole as fillable by the empty term, which no term language has.

    The last two lines pass a *list*, which the signature admits and every other call site here
    happens not to use.  The convolution caches on the hole tuple, so a sequence that is not
    already hashable has to be made so before it becomes a key, and a caller who builds the
    holes by comprehension would otherwise meet a ``TypeError`` from inside the cache.
    """
    table = size_table(list_space(), 5)
    assert table.split_counts((LIST, LIST, LIST), 2) == 0
    assert table.split_counts((LIST, LIST, LIST), 3) == 1
    assert table.split_counts((LIST,), 0) == 0
    assert table.split_counts([LIST, LIST, LIST], 3) == 1
    assert table.split_counts([LIST, LIST], 4) == 27, "3^0*3^2 + 3^1*3^1 + 3^2*3^0"


def test_a_negative_size_bound_is_an_error():
    """A negative bound is a caller's mistake, not a table with nothing in it.

    The bound counts function symbols, so there is no reading under which a negative one means
    anything.  Returning an empty table would let a mistyped bound run through the whole sampler
    and come out the other end as "this space has no inhabitants".

    The second call is against a program that *also* fails the decomposition hypothesis, so both
    refusals are available and only one of them answers the caller's question.  Both raise
    ``ValueError``, so which one arrives is decided by the order of the two checks and is visible
    only in the message: a caller who mistyped the bound must not be sent to rewrite a clause.
    """
    with pytest.raises(ValueError, match="cannot be negative"):
        size_table(list_space(), -1)
    with pytest.raises(ValueError, match="cannot be negative"):
        size_table(constrained_space(), -1)


def test_the_table_reads_as_zero_outside_the_rows_it_holds():
    """A non-terminal the program never mentions has no terms, and neither has a size past ``D``.

    The convolution reads the table at sizes it has not reached yet and at whatever non-terminals
    a clause names, so the total function is what keeps the fill from needing a case distinction
    per lookup.  Reading zero is the honest answer in all three directions: no clause, no room, no
    such size.
    """
    table = size_table(list_space(), 5)
    assert table.of(Constructor("Nowhere"), 1) == 0
    assert table.of(LIST, 6) == 0
    assert table.of(LIST, -1) == 0
    assert table.of(LIST, 0) == 0, "no term has size 0"
    assert table.of(LIST, 5) == 81, "an always-zero table would pass the three above"


def test_a_shared_table_cannot_be_written_through_by_one_of_its_readers():
    """A table handed to several callers is theirs to read, and to nobody's to change.

    Sharing one filled table between queries is what makes the table form worth its hypothesis in
    the experiment layer, and it turns the table into a mutable object seen by callers who know
    nothing of each other.  The counts are therefore rows of a type that refuses assignment: a
    caller who wrote into one would change every other caller's answer, at a place none of them
    would think to look, and the sampler would go on drawing in proportion to the new number
    without anything to mark that it had changed.
    """
    table = size_table(list_space(), 5)
    assert isinstance(table.counts[LIST], tuple)
    with pytest.raises(TypeError):
        table.counts[LIST][3] = 999  # type: ignore[index]
    # A row that refuses assignment is only half of it. Rebinding the row under its non-terminal
    # would change the same answer, and it would leave the convolution cache holding rows built
    # from the numbers that were there before, so `of` and `split_counts` would disagree.
    with pytest.raises(TypeError):
        table.counts[LIST] = (0, 0, 0, 999, 0, 0)  # type: ignore[index]
    with pytest.raises(AttributeError):
        table.counts.clear()  # type: ignore[attr-defined]
    assert table.of(LIST, 3) == 9, "3^2, and untouched"
    assert table.split_counts((LIST,), 3) == 9


def triple(dead: str, left: str, right: str) -> str:
    """Combine three arguments, for the spaces that need a clause of arity three.

    Args:
        dead (str): The interpreted first argument.
        left (str): The interpreted second argument.
        right (str): The interpreted third argument.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"t({dead},{left},{right})"


def add_again(left: str, right: str) -> str:
    """Join two expressions, for the space that states one shape of clause twice.

    Args:
        left (str): The interpreted left expression.
        right (str): The interpreted right expression.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"A({left},{right})"


@pytest.mark.parametrize("bound", [1, 4, 7, 10])
def test_a_clause_of_three_holes_is_counted_through_a_row_for_its_tail(bound):
    """The one shape the convolution cannot do in a single step, against a closed form.

    A clause of two holes is convolved by giving the first hole a size and reading the second
    hole's row at what is left. A clause of three cannot be, since the pair of remaining holes has
    to be known over a whole range of sizes, so that pair gets a row of its own and is filled by
    increasing size exactly as a non-terminal is. Every other reference space here has clauses of
    at most two holes, so nothing else in this file reaches that machinery at all.

    The oracle is closed: a ternary tree with ``n`` inner nodes has ``3n + 1`` symbols and there
    are ``binomial(3n, n) / (2n + 1)`` of them, which is 1, 1, 3, 12 at the bounds under test.

    Args:
        bound (int): The size bound ``D`` under test.
    """
    space = ternary_space()
    table = size_table(space, bound)
    counted = branch_counts(generator_query(space, TERNARY), bound, term_size)

    expected = {3 * n + 1: math.comb(3 * n, n) // (2 * n + 1) for n in range(bound) if 3 * n + 1 <= bound}
    assert realized_row(table, TERNARY) == expected
    assert dict(counted.counts) == expected
    assert list(table.suffix_counts) == [(TERNARY, TERNARY)], (
        "the tail of the ternary clause is what gets a row of its own"
    )


def test_two_clauses_of_one_shape_on_one_head_are_counted_twice():
    """Clauses that agree in their terminal cost and their holes compute one number, not one count.

    The fill groups the program by ``(base, holes)``, so two clauses that agree in both are one
    task. What separates them is how often each head takes that task: the group carries a
    multiplicity, and dropping it would count such a pair once. The unary case is covered by the
    list space, whose three clauses share a shape, and this is the case for a clause with holes to
    split.

    The oracle is closed: with two binary clauses every inner node of a binary tree carries a
    choice of two, so the count at size ``2k + 1`` is the ``k``-th Catalan number times ``2^k``.
    """
    bound = 7
    space = Synthesizer(
        {
            lit: SpecificationBuilder().suffix(EXPR),
            add: SpecificationBuilder().argument("left", EXPR).argument("right", EXPR).suffix(EXPR),
            add_again: SpecificationBuilder().argument("left", EXPR).argument("right", EXPR).suffix(EXPR),
        }
    ).construct_solution_space(EXPR)

    table = size_table(space, bound)
    counted = branch_counts(generator_query(space, EXPR), bound, term_size)
    expected = {2 * k + 1: (math.comb(2 * k, k) // (k + 1)) * 2**k for k in range((bound + 1) // 2)}

    assert realized_row(table, EXPR) == expected
    assert dict(counted.counts) == expected


def test_a_size_two_clauses_write_into_is_occupied_once():
    """A size is occupied once, however many clauses wrote into it.

    The fill walks the sizes a non-terminal occupies rather than every size below the total, which
    is what keeps the convolution cheap on a program whose sizes have gaps. The list it walks has
    to hold each size once: a clause that reads it multiplies by the count standing there, so a
    size listed twice is counted twice, and every row above it comes out too large.

    Two clauses write into size 2 here, a nullary one that fixes a literal beside its terminal and
    a unary one above a term of size 1, and ``h_join`` is the clause that reads the result.
    """
    bound = 7
    space = mixed_arity_space()
    table = size_table(space, bound)
    counted = branch_counts(generator_query(space, MIXED), bound, term_size)

    assert realized_row(table, MIXED) == dict(counted.counts)
    assert table.of(MIXED, 2) == 3, "two literals and one step above the leaf"
    assert table.of(MIXED, 7) == 201


def test_split_counts_answers_zero_outside_the_bound_for_a_tuple_that_has_holes():
    """The bound check answers for a real hole tuple, not only for the empty one.

    :meth:`SizeTable.split_counts` reads its row at the total it is given, and a total outside the
    table has no row to read: below zero there is nothing to distribute, and above the bound the
    table was never filled. The empty tuple hides this, because its row is one at zero and zero
    everywhere else, so a missing check reads a zero there and looks right.
    """
    table = size_table(list_space(), 5)
    assert table.split_counts((LIST,), -1) == 0
    assert table.split_counts((LIST,), 6) == 0
    assert table.split_counts((LIST,), 3) == 9, "3^2, inside the bound"


def test_a_hole_whose_sort_the_program_never_derives_is_counted_as_empty():
    """The synthesis does not prune, so a clause may name a sort that has no clauses at all.

    ``construct_solution_space`` keeps a clause whose argument type turned out uninhabited: that
    sort never becomes a head, and the clause can derive nothing. The tree form already reads such
    a hole as empty, because it expands through what the program holds for the sort and finds
    nothing there. The table has to arrive at the same answer rather than at an exception, since a
    caller who did not prune has an ordinary program and not a broken one.
    """
    ghost = Constructor("Ghost")
    space = Synthesizer(
        {
            lit: SpecificationBuilder().suffix(EXPR),
            neg: SpecificationBuilder().argument("inner", ghost).suffix(EXPR),
        }
    ).construct_solution_space(EXPR)

    table = size_table(space, 4)
    counted = branch_counts(generator_query(space, EXPR), 4, term_size)
    assert realized_row(table, EXPR) == dict(counted.counts) == {1: 1}


def test_a_stored_tail_is_filled_to_its_own_reach_and_not_to_the_reach_of_a_sort():
    """A hole tuple reaches as far as its holes together, which no single sort need reach.

    A clause of three holes or more is convolved through a row for its tail, and that row is read
    back through :meth:`SizeTable.split_row`. The fill stops at the largest term the program has,
    and that is a maximum over *sorts*: the clause that splits into a tail carries the tail's reach
    up to its head, so where that clause derives nothing the head never records it and the tail's
    row would stop short of what it holds. The table would then answer two different numbers for
    one convolution, depending on whether the tuple happens to be a stored tail, and a caller
    filling holes from the left asks for exactly the stored ones.

    Here ``E`` and ``W`` reach one symbol each and nothing reaches two, while the tail ``(E, W)``
    reaches two. The clause that splits into it is dead, because its first hole is.
    """
    ghost = Constructor("Ghost")
    space = Synthesizer(
        {
            lit: SpecificationBuilder().suffix(EXPR),
            zero: SpecificationBuilder().suffix(WORD),
            neg: SpecificationBuilder().argument("inner", ghost).suffix(ghost),
            triple: SpecificationBuilder()
            .argument("dead", ghost)
            .argument("left", EXPR)
            .argument("right", WORD)
            .suffix(PAIR),
        }
    ).construct_solution_space(PAIR)

    table = size_table(space, 7)
    assert realized_row(table, EXPR) == {1: 1}
    assert realized_row(table, WORD) == {1: 1}
    # One symbol in each hole, so there is exactly one way to fill the pair with two symbols. The
    # reversed tuple is the same product and is not a stored tail, which is what makes the two
    # answers comparable at all.
    assert list(table.suffix_counts) == [(EXPR, WORD)], "the tail is stored, so it is read back"
    assert table.split_counts((EXPR, WORD), 2) == 1
    assert table.split_counts((WORD, EXPR), 2) == 1
    # The clause stays dead, so the start symbol derives nothing at any bound.
    counted = branch_counts(generator_query(space, PAIR), 7, term_size)
    assert realized_row(table, PAIR) == dict(counted.counts) == {}


def test_two_programs_sharing_a_nonterminal_name_get_their_own_counts():
    """Two tables filled in one process must not read each other's convolutions.

    ``Constructor`` compares by name, so two unrelated programs that both call their sort ``E``
    have equal hole tuples and not merely similar ones. The convolution cache is keyed by those
    tuples and the size, and nothing in the key says which program the rows came from: a cache
    outliving one fill would answer the second program's ``(E, E)`` at total 3 with the first
    program's number, and the whole point of the table is that its numbers come from *this*
    program.

    Sharing a name across programs is not exotic here.  The experiment layer fills one table per
    variant of a repository, in one process, with the sorts named the same in every variant.
    """
    bound = 7
    with_neg = expression_space()
    without_neg = Synthesizer(
        {
            lit: SpecificationBuilder().suffix(EXPR),
            add: SpecificationBuilder().argument("left", EXPR).argument("right", EXPR).suffix(EXPR),
        }
    ).construct_solution_space(EXPR)

    first = realized_row(size_table(with_neg, bound), EXPR)
    second = realized_row(size_table(without_neg, bound), EXPR)

    assert first == dict(branch_counts(generator_query(with_neg, EXPR), bound, term_size).counts)
    assert second == dict(branch_counts(generator_query(without_neg, EXPR), bound, term_size).counts)
    # Without `neg` every inhabitant is a binary tree, so the sizes are odd and the counts are the
    # Catalan numbers, an oracle that shares nothing with either construction.
    assert second == {1: 1, 3: 1, 5: 2, 7: 5}
    assert first != second


def test_a_finished_table_answers_a_repeated_convolution_without_reading_a_row_again():
    """The convolution cache is the reason the table form is cheap, so its effect is asserted.

    The cost claim of the table form is ``O(|clauses| * arity * D * occupancy)``, so it is in the
    size of the *program*, not in the number of inhabitants.  The convolution is what could break that: over a
    clause with three holes it decomposes into overlapping suffix problems, and without a cache it
    recomputes them.  A cache that stored under a different key than it looked up would keep every
    number in this file correct and quietly cost an order of magnitude, which is precisely the
    claim the table exists to make.

    The observation is the *number of convolutions*, not a wall clock: a repeated question about
    the same holes must not walk a non-terminal's occupied sizes again.  That is exact,
    deterministic and reads only the public interface, through a table subclass that records which
    non-terminals a convolution walked, where a time budget would flake on a loaded CI machine and a
    look into the cache dictionary would pin an attribute name rather than a behavior.
    """
    walks: list = []

    class RecordingTable(SizeTable):
        """A size table that records every convolution it performs."""

        def _occupied_row(self, nonterminal):
            """Walk a non-terminal's occupied sizes and record the walk.

            Args:
                nonterminal: The non-terminal being split off.

            Returns:
                tuple: Its occupied sizes with their counts.
            """
            walks.append(nonterminal)
            return super()._occupied_row(nonterminal)

    filled = size_table(list_space(), 9)
    table = RecordingTable(bound=filled.bound, counts=filled.counts)

    first = table.split_counts((LIST, LIST, LIST), 9)
    assert first == 20412, "3^6 * the compositions of 9 into three positive parts"
    assert walks, "an uncached table has to convolve, or the assertion below is vacuous"

    walks.clear()
    assert table.split_counts((LIST, LIST, LIST), 9) == first
    assert walks == []
    # The same holes at a different size, too: the row holds every size at once, so asking for
    # another one is a lookup and not a second convolution.
    assert table.split_counts((LIST, LIST, LIST), 7) == 1215, "3^4 * the compositions of 7"
    assert walks == []


# ---------------------------------------------------------------------------
# The hypothesis: decomposable_or_raise, and the error that has to say which clause
# ---------------------------------------------------------------------------


def test_a_predicate_over_two_holes_is_reported_with_both_of_its_positions():
    """The residual at a coupled node is no product of the residuals at its holes.

    ``pair`` rejects the completions in which the two words agree, so the table, which multiplies
    ``N_W(s_1)`` by ``N_W(s_2)``, counts pairs the language does not contain. The message has to
    name the clause *and* the positions: a repository is repaired by rewriting a clause, and which
    one it is cannot be read off "the table does not apply".
    """
    space = constrained_space()
    offenders = coupled_clauses(space)
    assert len(offenders) == 1
    assert offenders[0].positions == (0, 1)
    assert offenders[0].nonterminals == (WORD, WORD)

    with pytest.raises(ValueError, match="reading a hole in a predicate") as raised:
        decomposable_or_raise(space)
    message = str(raised.value)
    assert "argument 0" in message
    assert "argument 1" in message
    assert "couples" in message
    assert "pair" in message, "the clause is identified by its terminal"


def test_a_predicate_over_a_single_hole_is_reported_with_that_one_position():
    """ "No coupling" is not the hypothesis the table needs: one hole is enough to break it.

    ``cut_space`` has one clause, one hole and one predicate, and nothing in it is coupled to
    anything.  The predicate still cuts the hole's language down to a proper subset, and a table
    indexed by the non-terminal cannot represent the difference: the exact counts are
    ``{2: 2, 3: 2}`` where the table says ``{2: 2, 3: 2, 4: 2, ...}`` for as far as the bound
    reaches.  A condition phrased as "no predicate couples two holes" would accept this space and
    return numbers that are wrong from size 4 on.
    """
    space = cut_space()
    offenders = coupled_clauses(space)
    assert len(offenders) == 1
    assert offenders[0].positions == (0,)

    with pytest.raises(ValueError, match="reading a hole in a predicate") as raised:
        decomposable_or_raise(space)
    message = str(raised.value)
    assert "argument 0" in message
    assert "argument 1" not in message
    assert "cuts" in message
    assert "box" in message

    bound = 8
    exact = branch_counts(generator_query(space, BOX), bound, term_size)
    assert dict(exact.counts) == {2: 2, 3: 2}, "the predicate really does cut"


def test_a_predicate_over_literals_alone_leaves_the_table_form_applicable():
    """A predicate that reads no hole decides on its clause, and the table can drop the clause.

    ``grade``'s predicate reads the literal ``d`` and nothing that depends on how a hole is
    filled, so it neither couples nor cuts: it removes one clause once and for all, which is a
    thing the table can do while it is being filled.  Reporting it would make the condition
    useless on a repository built largely from clauses with literal parameters.
    """
    space = literal_predicate_space()
    assert coupled_clauses(space) == []
    decomposable_or_raise(space)
    assert realized_row(size_table(space, 5), USED) == {3: 2}


def test_a_predicate_over_an_anonymous_hole_leaves_the_table_form_applicable():
    """A hole with no name is invisible to the predicate, so the table still applies.

    An arrow type in a suffix produces a hole the inhabitation leaves anonymous, and a predicate on
    such a clause reads the literals and nothing else, since an anonymous argument never enters the
    substitution it is handed. The clause therefore decides once and for all, and the condition
    must not report it: it reports a clause that carries a predicate *and* a named non-terminal
    argument, and this one has no named argument at all.

    That leaves the fill responsible for the clause. It has to drop what the predicate rejects, as
    the engine does, or the table counts a term the space does not contain. The space grades three
    values and forbids one of them, so a fill that ignored the predicate would say three where the
    tree form says two.
    """
    space = anonymous_hole_space()
    assert coupled_clauses(space) == []
    decomposable_or_raise(space)

    table = size_table(space, 5)
    for start, expected in ((MARKED, {3: 2}), (HELD, {4: 2})):
        query = generator_query(space, start)
        counted = branch_counts(query, 5, term_size)
        assert dict(counted.counts) == expected
        assert realized_row(table, start) == dict(counted.counts)
        # The counts are only half of it. A fill that ignored the predicate would also stream the
        # term the space does not contain, so the stream is asserted against the space itself.
        streamed = list(weighted_table(query, 5, uniform).stream(random.Random(5)))
        assert len(streamed) == 2
        for term in streamed:
            assert checker(space, start, term), "the stream must stay inside the space"


@pytest.mark.parametrize(("name", "build"), [(name, build) for name, build, _start in DECOMPOSABLE_SPACES])
def test_a_program_without_predicates_couples_nothing(name, build):
    """No predicate, nothing to report: the condition must not be a blanket refusal.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
    """
    space = build()
    assert coupled_clauses(space) == [], name
    decomposable_or_raise(space)


def test_turning_the_check_off_lets_the_table_count_what_the_predicate_cuts():
    """What the hypothesis is *for*: without it the numbers are too large, and say nothing.

    ``check=False`` fills the table anyway, and the table counts one ``box`` per value of ``V`` at
    every size, since the predicate is invisible to it. From size 4 on the two constructions part
    company, and nothing in the returned table records that.  A sampler driven by it would draw
    terms outside the language, in proportion to weights that are wrong.
    """
    bound = 8
    space = cut_space()
    table = size_table(space, bound, check=False)
    exact = branch_counts(generator_query(space, BOX), bound, term_size)

    assert realized_row(table, BOX) == dict.fromkeys(range(2, bound + 1), 2)
    assert dict(exact.counts) == {2: 2, 3: 2}
    assert realized_row(table, BOX) != dict(exact.counts)
    assert sum(realized_row(table, BOX).values()) > exact.total


@pytest.mark.parametrize(
    ("build", "start", "bound", "exact"),
    [
        (cut_space, BOX, 8, {2: 2, 3: 2}),
        (constrained_space, PAIR, 6, {3: 2, 4: 8, 5: 10, 6: 16}),
    ],
    ids=["one hole", "two holes"],
)
def test_a_hole_reading_predicate_is_never_called_while_the_table_is_filled(build, start, bound, exact):
    """The fill evaluates a predicate only where it can supply every argument it might read.

    A predicate that reads a hole has no entry for that hole in its clause's *literal*
    substitution, so calling it there fails inside user code, with a ``KeyError`` where the
    predicate subscripts and a wrong answer where it uses ``.get``.  Neither failure says anything about the
    hypothesis, and which of the two happens is decided by how a user wrote their predicate.  So
    the fill evaluates a predicate only on a clause with no *named* non-terminal argument, which
    is exactly the class :func:`coupled_clauses` does not report.

    Both predicates here read their holes by subscript, which is what makes this test the
    evidence: were the fill to call them, these calls would raise instead of returning a table.
    What a caller who turns the check off gets is the honest failure mode, which is numbers that
    are too large, and not an exception about a missing dictionary key.
    """
    table = size_table(build(), bound, check=False)
    counted = realized_row(table, start)

    assert counted != exact
    assert sum(counted.values()) > sum(exact.values())


def test_a_program_with_two_offending_clauses_has_both_of_them_named():
    """ "Every such clause", as the docstring promises, and not the first one found.

    A repository is repaired one clause at a time, so a report that stopped at the first offender
    would send its reader through as many failed runs as the program has offending clauses, each
    one ending in the same message with one name removed.  ``two_offender_space`` breaks the
    hypothesis twice and in both ways at once: ``pair`` couples its two holes, ``box`` cuts its
    one, and both are reachable from the same start.  The two reference spaces that carry a
    violation have exactly one clause each, so neither can tell a complete report from a partial
    one.
    """
    space = two_offender_space()
    offenders = coupled_clauses(space)
    assert len(offenders) == 2
    assert {clause.positions for clause in offenders} == {(0, 1), (0,)}

    with pytest.raises(ValueError, match="reading a hole in a predicate") as raised:
        decomposable_or_raise(space)
    message = str(raised.value)
    assert "2 clause(s)" in message
    assert "pair" in message
    assert "box" in message
    assert "couples" in message
    assert "cuts" in message


def test_a_literal_parameter_before_the_hole_does_not_shift_the_reported_position():
    """The reported position indexes the clause's arguments, not its holes.

    ``dbox`` declares a literal parameter and only then the hole its predicate reads, so the hole
    is the clause's *first* hole and its *second* argument.  A report that counted holes would say
    ``argument 0`` and send the reader of this program to the literal, which is a well-formed
    message naming the wrong thing and worse than none. This is a shape a real repository is built
    from throughout, and every other space here would accept either numbering.
    """
    space = offset_cut_space()
    offenders = coupled_clauses(space)
    assert offenders, "the predicate reads a hole, so the program must be reported"
    assert all(clause.positions == (1,) for clause in offenders)
    assert all(clause.nonterminals == (CUT_SORT,) for clause in offenders)

    with pytest.raises(ValueError, match="reading a hole in a predicate") as raised:
        decomposable_or_raise(space)
    message = str(raised.value)
    assert "argument 1" in message
    assert "argument 0" not in message
    assert "dbox" in message


@pytest.mark.parametrize(
    ("positions", "nonterminals", "expected"),
    [
        (
            (0, 1),
            (WORD, WORD),
            "Pair <- pair (argument 0 : W, argument 1 : W): the predicate couples them",
        ),
        ((0,), (WORD,), "Pair <- pair (argument 0 : W): the predicate cuts them"),
    ],
    ids=["two positions", "one position"],
)
def test_an_offending_clause_renders_its_head_terminal_and_positions_in_that_order(positions, nonterminals, expected):
    """The rendering is asserted whole, because every part of it is load-bearing.

    Each of the four pieces answers a different question a reader has: which non-terminal to look
    under, which clause of it, which argument of that clause, and what is wrong with it.  Tested
    by substring, which is how the messages are checked where they are raised, the head and
    the terminal are interchangeable and the non-terminals may vanish entirely, since
    ``argument 0 : 0`` still contains ``argument 0``.  Here the string is the claim.

    Args:
        positions (tuple): The argument positions the predicate can read.
        nonterminals (tuple): The non-terminals at those positions.
        expected (str): The rendering, in full.
    """
    clause = CoupledClause(
        nonterminal=PAIR,
        terminal="pair",
        positions=positions,
        nonterminals=nonterminals,
    )
    assert clause.describe() == expected


@pytest.mark.parametrize("bound", [0, 1, 2, 5])
def test_the_table_refuses_a_program_it_cannot_count_at_every_bound(bound):
    """The hypothesis is decided before the fill, so no bound is small enough to skip it.

    At bound 0 the fill loop does not run at all and every row is zero whatever the program says,
    so a check tied to the fill, or to a bound past which someone thought the numbers start to
    matter, would let this call through and hand back a table that looks like any other empty
    one.  The refusal is a statement about the *program*, and a program either satisfies the
    hypothesis or it does not.

    The call is to ``size_table`` directly rather than through the sampler: the sampler has a
    check of its own, and testing only through it would leave the table's own default unexercised.

    Args:
        bound (int): The size bound under test.
    """
    with pytest.raises(ValueError, match="reading a hole in a predicate"):
        size_table(constrained_space(), bound)


@pytest.mark.parametrize(
    ("name", "build", "start", "tight", "expected"),
    [
        ("finite", two_predicate_space, USED, 5, {3: 1}),
        (
            "recursive",
            list_space,
            LIST,
            7,
            {1: 1, 2: 3, 3: 9, 4: 27, 5: 81, 6: 243, 7: 729},
        ),
        (
            "two symbols per clause",
            two_symbol_clause_space,
            TAGGED,
            9,
            {1: 1, 3: 2, 5: 4, 7: 8, 9: 16},
        ),
    ],
)
def test_a_bound_far_above_the_program_counts_what_a_tight_bound_counts(name, build, start, tight, expected):
    """Stopping the fill early must save arithmetic, never a term.

    The fill runs up to the largest term the program has rather than up to ``D``, which is what
    makes a generous bound affordable on a finite program, whose rows go empty long before the
    bound does. The saving is only legitimate if the bound above which nothing is counted really is
    above everything the program derives, and the way to get that wrong is a recursive program,
    whose largest term *is* the bound: cut there and the table silently reports a fraction of the
    language. On the list space at ``D = 7`` a fill that stopped before the fixed point converged
    returned a handful of branches where there are 1093.

    So both cases are pinned here, and they are pinned against each other: within the tight
    bound, the generous table has to say exactly what the tight one says.

    Args:
        name (str): The case's name, for the test id.
        build (Callable): Builds the space.
        start: The queried non-terminal.
        tight (int): A bound the program's terms actually reach.
        expected (dict[int, int]): The row within that bound.
    """
    space = build()
    generous = realized_row(size_table(space, 20 * tight), start)

    assert realized_row(size_table(space, tight), start) == expected, name
    assert {size: count for size, count in generous.items() if size <= tight} == expected, name
    # A recursive program keeps producing above the tight bound; a finite one must not, or the
    # early stop has hidden nothing and the test proves nothing.
    assert (max(generous) > tight) == (name != "finite"), name


def uniform(_value):
    """Weight every realized cost value alike, which is the distribution of size-uniform sampling.

    Args:
        _value: The cost value. Ignored.

    Returns:
        float: One; the constructions normalize over the realized values themselves.
    """
    return 1.0


def assert_streams_agree(eager, lazy):
    """Assert that two keyed streams coincide term for term and key for key.

    The keys are compared with :func:`math.isclose` rather than for equality: both constructions
    reach them through a log-sum-exp, and the summands arrive in the order the counts happen to
    be stored in, so the last bits may differ. What must not differ is anything a reader could
    notice, since the order of the stream is decided by the keys, so a discrepancy large enough to
    matter reorders the stream and is caught by the term comparison anyway.

    Args:
        eager (list): The keyed stream of the tree form.
        lazy (list): The keyed stream of the table form.
    """
    assert len(eager) == len(lazy)
    assert [term for _, term in eager] == [term for _, term in lazy]
    for (eager_key, _), (lazy_key, _) in zip(eager, lazy, strict=True):
        assert math.isclose(eager_key, lazy_key, rel_tol=1e-12, abs_tol=1e-12)


def expression_parent():
    """Build the parent term the expression residual queries are posed on.

    Built per call rather than shared: ``Tree`` memoises its positions on the instance, and the
    suite runs randomized and parallel.

    Returns:
        Tree: ``add(lit, neg(lit))``, a term with an inner position of every shape.
    """
    return Tree(add, (Tree(lit, ()), Tree(neg, (Tree(lit, ()),))))


def list_parent():
    """Build the parent term the list residual queries are posed on.

    Returns:
        Tree: ``cons_0(cons_1(nil))``.
    """
    return Tree(cons_0, (Tree(cons_1, (Tree(nil, ()),)),))


def literal_parent():
    """Build the parent term the literal residual queries are posed on.

    Returns:
        Tree: ``tag(0, stop)``, a term whose root clause writes two symbols.
    """
    return Tree(tag, (Tree(0, ()), Tree(stop, ())))


@pytest.mark.parametrize("filled_to", [4, 3], ids=["one row short", "two rows short"])
def test_a_table_filled_too_small_names_both_bounds(filled_to):
    """A table shorter than the search would read as zero past its end and cut the space short.

    The missing rows are not distinguishable from "no inhabitants of that size", so the sampler
    would draw from a strictly smaller language and report nothing unusual. The message has to
    carry both numbers, because the caller's mistake is the *relation* between them.

    Both shortfalls are here because the numbers stay *correct* under an off-by-one in this
    guard: the search reads the row of its own bound and no further, so a table one row short is
    read exactly to its end and streams 121 terms rather than raising. Nothing but this check
    stands between a caller who reused a table under a larger bound and a silently smaller
    language, so the check has to fire on the smallest violation there is.

    Args:
        filled_to (int): The bound the handed-in table was filled to, against a search for 5.
    """
    space = list_space()
    query = generator_query(space, LIST)
    with pytest.raises(ValueError, match=rf"filled to {filled_to}.*asks for 5"):
        weighted_table(query, 5, uniform, table=size_table(space, filled_to))


def test_one_table_shared_between_queries_gives_what_freshly_built_tables_give():
    """The table depends on the program and the bound, not on the query, so it is shareable.

    That is what makes the table form worth its hypothesis where many queries share a program: one
    fill, then
    a residual query per mutation. The convolution cache is filled while the table is used, so a
    shared table is a mutating object seen by several callers; if the cache were keyed carelessly,
    the second caller would be the one who noticed.
    """
    space = list_space()
    shared = size_table(space, 6)
    parent = list_parent()

    for bound, query in (
        (5, generator_query(space, LIST)),
        (4, generator_query(space, LIST)),
        (6, residual_query(space, LIST, parent, (0,))),
    ):
        from_shared = weighted_table(query, bound, uniform, table=shared)
        fresh = weighted_table(query, bound, uniform)
        assert dict(from_shared.root_counts) == dict(fresh.root_counts)
        assert [str(term) for term in from_shared.stream(random.Random(11))] == [
            str(term) for term in fresh.stream(random.Random(11))
        ]


def test_two_hole_tuples_of_one_length_do_not_share_a_split():
    """The convolution is keyed by the holes, not by how many of them there are.

    ``same_holes`` opens ``(Ha, Ha)`` and ``mixed_holes`` opens ``(Ha, Hb)``: both are two holes
    and both are convolved at total 2 while the same row is being filled, but ``Ha`` has two
    inhabitants of size 1 where ``Hb`` has one, so the two tuples admit 4 splits and 2. A cache
    that recognized
    the second tuple as "two holes, total 2" would answer it with the first one's number, and the
    start row would read 8 where it must read 6.

    The consequence is not only a wrong total. The root then draws the two clauses with equal
    probability instead of 2:1, which is a change in the sampled distribution that no count in
    the stream reveals, so the root probabilities are asserted here as well.
    """
    bound = 4
    space = hole_tuple_space()
    table = size_table(space, bound)
    query = generator_query(space, TUPLE_SORT)
    counted = branch_counts(query, bound, term_size)

    assert realized_row(table, SORT_A) == {1: 2}
    assert realized_row(table, SORT_B) == {1: 1}
    assert table.split_counts((SORT_A, SORT_A), 2) == 4
    assert table.split_counts((SORT_A, SORT_B), 2) == 2
    assert realized_row(table, TUPLE_SORT) == dict(counted.counts) == {3: 6}

    weighted = weighted_table(query, bound, uniform)
    root = weighted.log_weight_of(None, 0)
    probabilities = sorted(math.exp(weighted.log_weight_of(goal, size) - root) for goal, size in initial_nodes(query))
    assert probabilities == pytest.approx([1 / 3, 2 / 3])


def test_the_fill_skips_a_clause_too_heavy_for_the_current_size():
    """A clause that does not fit into ``s`` is passed over, and the ones behind it still count.

    The fill walks the clauses of a non-terminal once per size, and a clause whose own symbols
    already exceed that size contributes nothing to it. Passing over such a clause and abandoning
    the whole clause list at it are the same thing in every space where the lightest clause comes
    first, which is why this space states the heavy one first: ``marked`` needs two symbols, so at
    size 1 it is met before ``halt``, and a fill that stopped there would return an *empty* table
    for a space with infinitely many inhabitants.

    Nothing downstream would object. An empty table is what a bound below the smallest inhabitant
    legitimately produces, so the sampler would report "no inhabitants within the bound" for a
    space whose smallest inhabitant is a single symbol.
    """
    bound = 7
    space = chain_space()
    query = generator_query(space, CHAIN)
    counted = branch_counts(query, bound, term_size)

    assert realized_row(size_table(space, bound), CHAIN) == dict(counted.counts)
    assert dict(counted.counts) == {1: 1, 3: 2, 5: 4, 7: 8}

    streamed = weighted_table(query, 3, uniform).stream(random.Random(4))
    assert sorted(str(term) for term in streamed) == sorted(
        [
            str(Tree(halt, ())),
            str(Tree(marked, (Tree(0, ()), Tree(halt, ())))),
            str(Tree(marked, (Tree(1, ()), Tree(halt, ())))),
        ]
    )


@pytest.mark.parametrize("weight", [0.0, -1.0, math.inf, math.nan])
def test_a_distribution_that_is_not_positive_on_a_realized_size_is_an_error(weight):
    """The condition three docstrings state is checked rather than assumed.

    A size the query realizes carries inhabitants, so a distribution giving it no weight leaves
    them unreachable while the counts still say they are there. Zero is the case that matters
    most: it reaches ``log(0)`` and would surface as a domain error from the arithmetic rather
    than as the caller's mistake. A non-finite weight is worse, because it makes every log-weight
    a NaN and the search then orders its frontier by values that compare false against everything.

    Args:
        weight (float): The weight the distribution returns everywhere.
    """
    query = generator_query(list_space(), LIST)

    with pytest.raises(ValueError, match="positive"):
        weighted_table(query, 5, lambda _value: weight)

    with pytest.raises(ValueError, match="positive"):
        weighted_tree(query, 5, term_size, lambda _value: weight)


def test_the_tree_form_reads_the_weight_of_a_node_as_the_sum_over_its_cost_values():
    """``weight_of`` is the unnormalized weight the distribution puts below a node.

    At the root it is the whole weight, which is 1 because the distribution is normalized over the
    realized values. Below the root it is the same sum restricted to what that node reaches, so it
    is what the log-weight the search uses stands for, in the representation a caller reads.
    """
    eager = weighted_tree(generator_query(list_space(), LIST), 5, term_size, uniform)

    assert eager.weight_of(eager.root) == pytest.approx(1.0)

    for child in eager.root.children:
        expected = sum(count * eager.unit_weights[value] for value, count in child.counts.items())
        assert eager.weight_of(child) == pytest.approx(expected)


def test_a_prebuilt_table_is_no_way_around_the_bound_check_or_the_hypothesis():
    """Handing in a table is a way to share the cost of filling it, not to skip what guards it.

    ``weighted_table`` accepts a filled table so that several queries against one program pay for
    one fill. Both of its refusals used to sit on the path that *builds* the table, so a caller
    who supplied one bypassed them: a negative bound turned into an empty range and came back as
    "this space has no inhabitants", and a table built with the check off carried a program that
    breaks the decomposition hypothesis straight into the sampler. Neither is a fallback anybody
    asked for, and both replace a caller's mistake with a plausible-looking answer.
    """
    space = list_space()
    with pytest.raises(ValueError, match="cannot be negative"):
        weighted_table(generator_query(space, LIST), -1, uniform, table=size_table(space, 5))

    cut = cut_space()
    with pytest.raises(ValueError, match="reading a hole in a predicate"):
        weighted_table(
            generator_query(cut, BOX),
            8,
            uniform,
            table=size_table(cut, 8, check=False),
        )


def test_the_table_form_refuses_a_query_against_a_program_it_cannot_count():
    """The refusal reaches the sampler, not only the table: no silent fallback anywhere.

    A caller who asks ``weighted_table`` for a space with a coupled predicate must not receive a
    stream, neither a wrong one nor a tree-form one behind their back. Which construction is
    used changes the cost by orders of magnitude and the hypothesis by everything, so it is never
    decided silently.
    """
    space = constrained_space()
    with pytest.raises(ValueError, match="reading a hole in a predicate"):
        weighted_table(generator_query(space, PAIR), 5, uniform)


@pytest.mark.parametrize(("name", "build", "start"), DECOMPOSABLE_SPACES)
@pytest.mark.parametrize("seed", [1, 2, 3, 17, 99])
def test_the_table_form_streams_what_the_tree_form_streams(name, build, start, seed):
    """Same seed, same stream, term for term and key for key, over the whole stream.

    The central claim of the table form. ``keyed_stream`` is shared by both constructions, so what
    is compared here is exactly the oracle ``log w(n) = log sum_a B_n(a) pi(a) / N_r(a)``: the tree
    form answers it by having materialized the retained tree, the table form by a convolution of
    table rows. Comparing the keys and not only the terms is what makes this a statement about the
    weights rather than about the set of inhabitants, since two constructions could stream the same
    terms in the same order under different keys only by coincidence, and the keys are what
    the sampling guarantee is about.

    The comparison runs to the end of the stream on purpose. A prefix would leave the deep, light
    part of the space, where the counts are largest and a convolution is most likely to drift,
    untested.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The queried non-terminal.
        seed (int): The seed under test.
    """
    bound = FULL_STREAM_BOUNDS[name]
    space = build()
    query = generator_query(space, start)
    eager = list(weighted_tree(query, bound, term_size, uniform).keyed_stream(random.Random(seed)))
    lazy = list(weighted_table(query, bound, uniform).keyed_stream(random.Random(seed)))
    assert eager, "an empty stream would make the comparison vacuous"
    assert_streams_agree(eager, lazy)


@pytest.mark.parametrize(("name", "build", "start"), DECOMPOSABLE_SPACES)
def test_the_table_knows_how_many_branches_the_stream_will_hold(name, build, start):
    """``WeightedTable.total == branch_counts(...).total``, without building the tree.

    The number the experiment layer budgets against: how many draws a bound admits at all. The
    table answers it in milliseconds where the tree form answers it by enumerating the space.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The queried non-terminal.
    """
    bound = FULL_STREAM_BOUNDS[name]
    space = build()
    query = generator_query(space, start)
    assert weighted_table(query, bound, uniform).total == branch_counts(query, bound, term_size).total, name


def test_the_reference_example_is_reachable_at_the_bound_it_is_stated_at():
    """The reference example at ``D = 100``, which the tree form misses by a factor of ``3^100``.

    Every claim the example makes, at the bound it makes them at: ``N_r(s) = 3^(s-1)`` for every
    size within the bound, every length carrying total weight ``1/D``, and the root taking ``nil``
    with probability ``1/D`` and each ``cons_x`` with ``(D-1)/(3D)``, which is ``0.01`` and ``0.33``,
    against the naive sampler's ``0.25`` apiece.

    That this test terminates at all is the result. ``test_random_search`` states the same closed
    form at bounds of 4 to 9 because the retained tree at ``D = 100`` has on the order of ``3^100``
    nodes. Here the table is filled in milliseconds, from the program alone.
    """
    space = list_space()
    query = generator_query(space, LIST)
    weighted = weighted_table(query, GOLDEN_BOUND, uniform)

    assert dict(weighted.root_counts) == {size: 3 ** (size - 1) for size in range(1, GOLDEN_BOUND + 1)}
    for size, count in weighted.root_counts.items():
        assert count * weighted.unit_weights[size] == pytest.approx(1 / GOLDEN_BOUND)

    root = weighted.log_weight_of(None, 0)
    probabilities = sorted(math.exp(weighted.log_weight_of(goal, size) - root) for goal, size in initial_nodes(query))
    assert probabilities[0] == pytest.approx(1 / GOLDEN_BOUND)
    for probability in probabilities[1:]:
        assert probability == pytest.approx((GOLDEN_BOUND - 1) / (3 * GOLDEN_BOUND))
    assert sum(probabilities) == pytest.approx(1.0)


def test_forty_draws_at_the_golden_bound_stay_within_it_and_differ():
    """The bound is not only countable at ``D = 100``, it is drawable from.

    Forty is the initial population of the experiment layer, so this is the smallest thing that
    counts as "the sampler works there". The lengths must spread: size-uniform sampling puts
    ``1/100`` on each length, so forty draws that all came out short would mean the weights never
    reached the deep part of the space.
    """
    query = generator_query(list_space(), LIST)
    weighted = weighted_table(query, GOLDEN_BOUND, uniform)
    drawn = [term for _, term in zip(range(40), weighted.stream(random.Random(7)), strict=False)]

    assert len(drawn) == 40
    assert len({str(term) for term in drawn}) == 40, "the draw is without replacement"
    assert all(term_size(term) <= GOLDEN_BOUND for term in drawn)
    assert max(term_size(term) for term in drawn) > GOLDEN_BOUND // 2


def test_a_bound_below_the_smallest_inhabitant_gives_an_empty_stream():
    """No inhabitant within the bound means no stream, and no error.

    Emptiness within a bound is a legitimate outcome, not a failure: the caller asked what fits
    into ``D`` symbols and the answer is nothing.
    """
    query = generator_query(list_space(), LIST)
    weighted = weighted_table(query, 0, uniform)
    assert dict(weighted.root_counts) == {}
    assert weighted.total == 0
    assert list(weighted.stream(random.Random(1))) == []


def test_a_query_on_an_unknown_start_symbol_gives_an_empty_stream():
    """A start symbol the program has no clause for inhabits nothing.

    The table holds a row per non-terminal of the program, so an unknown start has no row at all
    a case the lookup has to answer rather than raise on, since a query is allowed to ask.
    """
    query = generator_query(list_space(), Constructor("Nowhere"))
    weighted = weighted_table(query, 5, uniform)
    assert dict(weighted.root_counts) == {}
    assert list(weighted.stream(random.Random(1))) == []


@pytest.mark.parametrize(
    ("name", "build", "start", "parent", "position"),
    [
        ("expression-root", expression_space, EXPR, expression_parent, ()),
        ("expression-left", expression_space, EXPR, expression_parent, (0,)),
        ("expression-right", expression_space, EXPR, expression_parent, (1,)),
        ("expression-deep", expression_space, EXPR, expression_parent, (1, 0)),
        ("list-root", list_space, LIST, list_parent, ()),
        ("list-tail", list_space, LIST, list_parent, (0,)),
        ("list-deep", list_space, LIST, list_parent, (0, 0)),
        ("two-symbol-root", two_symbol_clause_space, TAGGED, literal_parent, ()),
        ("two-symbol-rest", two_symbol_clause_space, TAGGED, literal_parent, (1,)),
    ],
)
def test_the_two_forms_agree_on_the_completions_of_a_partial_term(name, build, start, parent, position):
    """The residual through the table: the same completions, in the same order.

    A mutation operator poses a residual query, so the table form is only usable there if it
    reproduces the tree form's stream at *every* position of a term, the root
    included, where the query term is a bare variable and the residual is the whole language, and
    the deep positions, where the prescribed symbols already fill part of the size budget.

    The table is indexed by non-terminals alone and knows nothing of the prescribed term; what
    carries the agreement is that the initial nodes come from ``goal_from_tree`` in both
    constructions and that the size of the prescribed part is charged to the bound in both.

    Args:
        name (str): The case's name, for the test id.
        build (Callable): Builds the space.
        start: The queried non-terminal.
        parent (Callable): Builds the prescribed term.
        position (tuple): The position that becomes the hole.
    """
    bound = 5
    space = build()
    query = residual_query(space, start, parent(), position)
    eager = weighted_tree(query, bound, term_size, uniform)
    lazy = weighted_table(query, bound, uniform)

    assert lazy.total == eager.root.total > 0, name
    for stream_seed in (1, 2, 7):
        assert_streams_agree(
            list(eager.keyed_stream(random.Random(stream_seed))),
            list(lazy.keyed_stream(random.Random(stream_seed))),
        )


def test_no_drawn_term_carries_the_grade_the_predicate_rejects():
    """A clause the predicate rejects is dropped once, while the table is filled.

    ``grade(0)`` is not in the language, so it must not be in the stream. The table drops the
    clause at fill time, since the predicate reads literals only and its verdict is therefore the
    same at every position, and this is what pins that the drop happens rather than being assumed.
    """
    space = literal_predicate_space()
    weighted = weighted_table(generator_query(space, USED), 5, uniform)
    drawn = list(weighted.stream(random.Random(5)))

    assert len(drawn) == 2, "grade(1) and grade(2), and nothing else"
    for term in drawn:
        literals = {term.subtree_at(position).root for position in term.leaf_positions()}
        assert 0 not in literals


@pytest.mark.parametrize(("name", "build", "start"), DECOMPOSABLE_SPACES)
def test_the_stream_holds_every_inhabitant_within_the_bound_exactly_once(name, build, start):
    """Through the table: the stream *is* the language within ``D``.

    The oracle is generate-and-check over every term of the signature, so this compares the table
    form against a decision procedure that shares nothing with it, neither the counting
    recursion nor the search. On the ambiguous space each inhabitant appears once per derivation,
    which is the documented behavior without unambiguity and is checked as such.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The queried non-terminal.
    """
    bound = 4
    space = build()
    streamed = list(weighted_table(generator_query(space, start), bound, uniform).stream(random.Random(23)))
    expected = inhabitants_within(space, start, SIGNATURES[name], bound)

    assert set(streamed) == set(expected), name
    if name == "ambiguous":
        assert len(streamed) == 3, "merge(base) ends two branches"
    else:
        assert len(streamed) == len(expected), name
        assert len(set(streamed)) == len(streamed), name


def test_log_sum_exp_returns_minus_infinity_where_there_is_nothing_to_sum():
    """An empty sum is 0, whose logarithm is ``-inf``; so is a sum of vanishing weights.

    ``-inf`` is the value the search reads as "this node has no completion within the bound", so
    it has to arrive by arithmetic rather than by a special case: the table form drops a child
    exactly on it.
    """
    assert log_sum_exp([]) == -math.inf
    assert log_sum_exp([-math.inf]) == -math.inf
    assert log_sum_exp([-math.inf, -math.inf]) == -math.inf


@pytest.mark.parametrize(
    "terms",
    [
        [0.0, 0.0],
        [-1.5],
        [-1.0, -2.0, -3.0],
        [-math.inf, -1.0, -2.0],
        [2.0, -4.0, 0.5, 1.25],
    ],
)
def test_log_sum_exp_agrees_with_the_sum_it_stands_for(terms):
    """On benign values the shifted computation must give what the plain one gives.

    ``log sum_i exp(terms[i])``, written out. The ``-inf`` summand is in the list because it is
    the value a vanishing weight arrives as, and it must contribute nothing rather than poison the
    sum.

    Args:
        terms (list[float]): The summands, as logarithms.
    """
    expected = math.log(sum(math.exp(term) for term in terms))
    assert log_sum_exp(terms) == pytest.approx(expected)


def test_log_sum_exp_holds_where_the_exponentials_no_longer_fit_a_double():
    """The reason the function exists: ``exp(710)`` is not a double, ``710`` is an ordinary float.

    Node weights at the bounds the table form reaches are of this magnitude, and on the list space a
    log weight of ``-1104`` was measured at ``D = 1000``, so summing the exponentials first is
    not an option in either direction. Shifting by the maximum first leaves the summands in
    ``(0, 1]`` and the result exact to rounding.
    """
    with pytest.raises(OverflowError):
        math.exp(710.0)
    assert log_sum_exp([710.0, 709.0]) == pytest.approx(710.0 + math.log1p(math.exp(-1.0)))
    assert log_sum_exp([-710.0, -711.0]) == pytest.approx(-710.0 + math.log1p(math.exp(-1.0)))


@pytest.mark.parametrize(("name", "build", "start"), DECOMPOSABLE_SPACES)
def test_the_root_carries_the_whole_weight_in_both_constructions(name, build, start):
    """``log W = 0``: the distribution is normalized over the realized values, in log space.

    The root's key is drawn from a Gumbel at this location, and every conditioned key below it is
    derived from that one. A root weight that was not 1 would not break the *relative* weights,
    but it would mean the two constructions normalize differently, and then their streams could
    only agree by luck.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The queried non-terminal.
    """
    bound = 5
    space = build()
    query = generator_query(space, start)
    eager = weighted_tree(query, bound, term_size, uniform)
    lazy = weighted_table(query, bound, uniform)
    assert eager.log_weight_of(eager.root) == pytest.approx(0.0, abs=1e-12), name
    assert lazy.log_weight_of(None, 0) == pytest.approx(0.0, abs=1e-12), name


def test_the_table_form_survives_a_bound_where_the_weights_no_longer_fit_a_double():
    """At ``D = 1000`` a unit weight is not representable, and its logarithm is unremarkable.

    ``N_r(1000) = 3^999`` has more than 1024 bits, so ``pi(a) / N_r(a)`` cannot even be *computed*
    as a float: the division raises ``OverflowError`` on the integer rather than rounding. This
    is why the weights are kept in log space and the node weight is a log-sum-exp, and why the
    float copy takes its value from the logarithm where the direct computation would fail. It
    rounds to 0.0, which is the correctly rounded double, and nothing in the search reads it.

    Only the construction is exercised, not a draw: drawing at this bound is slow for an unrelated
    reason (the frontier, not the arithmetic).
    """
    bound = 1000
    weighted = weighted_table(generator_query(list_space(), LIST), bound, uniform)

    assert weighted.root_counts[bound] == 3 ** (bound - 1)
    assert all(math.isfinite(value) for value in weighted.log_unit_weights.values())
    assert weighted.log_unit_weights[bound] == pytest.approx(-math.log(bound) - (bound - 1) * math.log(3))
    assert weighted.unit_weights[bound] == 0.0
    assert weighted.unit_weights[1] == pytest.approx(1 / bound)
    assert weighted.log_weight_of(None, 0) == pytest.approx(0.0, abs=1e-9)


def test_a_clause_with_two_predicates_is_admissible_only_where_all_of_them_hold():
    """A clause is dropped as soon as *one* of its predicates rejects, not only when all do.

    ``grade`` carries two predicates, and each of the two rejected grades passes exactly one of
    them: 0 is below two but not positive, 2 is positive but not below two. So a filter reading
    "some predicate holds" keeps all three grades, and one reading "all of them hold" keeps one.
    Every other space in this suite has at most one predicate per clause, where the two conditions
    coincide, which is why this was worth a space of its own.

    Three places make the same decision and are asserted together: the fill of the size table
    drops the clause once, ``Goal.from_rhs_rule`` decides it at the root, and ``Goal.update``
    decides it below the root. Getting it wrong in the fill alone would leave the stream correct
    and the weights normalized against 3 instead of 1, which is a change in the sampled
    distribution with nothing in the stream to show for it.
    """
    space = two_predicate_space()
    assert positive({"d": 2})
    assert not below_two({"d": 2})
    assert below_two({"d": 0})
    assert not positive({"d": 0})

    query = generator_query(space, USED)
    counted = branch_counts(query, 5, term_size)
    assert realized_row(size_table(space, 5), USED) == dict(counted.counts) == {3: 1}

    weighted = weighted_table(query, 5, uniform)
    streamed = list(weighted.stream(random.Random(5)))
    assert weighted.total == len(streamed) == 1
    assert [str(term) for term in streamed] == [str(Tree(use, (Tree(grade, (Tree(1, ()),)),)))]

    at_root = depth_first(generator_query(space, GRADED), max_depth=3)
    assert [str(term) for term in at_root] == [str(Tree(grade, (Tree(1, ()),)))]
    for digit in (0, 2):
        assert not checker(space, GRADED, Tree(grade, (Tree(digit, ()),)))


# ---------------------------------------------------------------------------
# The shared search, and the two states it must refuse to search
# ---------------------------------------------------------------------------


def test_random_search_draws_nothing_from_a_root_of_vanishing_weight():
    """A root that weighs nothing has no key, and the search must not ask it for one.

    ``gumbel_key`` of ``-inf`` is ``-inf`` again, a key that orders below every other and would put
    a node with no inhabitant at the head of the frontier for as long as the frontier lasts. The
    stream ends instead, and ``expand`` is never reached: a caller who built the construction over
    a bound the space does not meet gets an empty stream rather than a search over nothing.
    """

    def expand(node):
        """Fail, since the search must not reach this.

        Args:
            node: The node the search would expand.

        Raises:
            AssertionError: Always.
        """
        msg = f"expand must not be called for a root of vanishing weight, was called on {node}"
        raise AssertionError(msg)

    assert list(keyed_stream("root", -math.inf, expand, random.Random(1))) == []


def test_random_search_steps_over_a_node_that_is_neither_a_success_nor_expandable():
    """A node with no inhabitant and no retained children is dropped, and the stream goes on.

    Such a node is what remains when every child of an inner node weighed nothing and was
    discarded at expansion. It carries a key like any other and is popped like any other, so the
    search has to recognize it and take the next node rather than yield or stall. The two children
    here are given equal weight, and the surviving one must come out whichever order the keys put
    them in.
    """
    solution = Tree("found", ())

    def expand(node):
        """Map the three nodes of a hand-built search tree.

        Args:
            node (str): The node to expand.

        Returns:
            tuple: Its inhabitant, or None, and its retained children with their log-weights.
        """
        if node == "root":
            return None, [("dead", math.log(0.5)), ("alive", math.log(0.5))]
        if node == "dead":
            return None, []
        return solution, []

    for seed in range(20):
        drawn = list(keyed_stream("root", 0.0, expand, random.Random(seed)))
        assert [str(tree) for _, tree in drawn] == [str(solution)]


def test_a_node_the_distribution_leaves_weightless_is_an_error_and_not_a_silent_zero():
    """A retained node weighs something in exact arithmetic, so a vanishing weight is a defect.

    Retention is decided by the branch counts and the weights by the distribution, and the two are
    computed apart. If they come apart the node still carries counts while its weight is zero, and
    a zero weight is not a small weight: the node would take a key of ``-inf``, never be drawn, and
    remove its whole subtree from the sample with nothing in the stream to show for it. Both ways
    the support can fail to cover a node are asserted, since they arrive at the guard by different
    routes: an explicit ``-inf`` for the realized value, and a support that omits it, where the
    sum runs over no terms at all.

    Raises:
        ValueError: From both calls, which is what the test asserts.
    """
    node = CountedNode(goal=None, inhabitant=None, children=(), counts={2: 5})

    with pytest.raises(ValueError, match="come apart"):
        WeightedTree(root=node, unit_weights={2: 0.0}, log_unit_weights={2: -math.inf}).log_weight_of(node)

    with pytest.raises(ValueError, match="come apart"):
        WeightedTree(root=node, unit_weights={}, log_unit_weights={}).log_weight_of(node)
