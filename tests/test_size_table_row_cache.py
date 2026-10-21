"""The rows a size table convolves for a stream: kept within a bound on their entries.

A stream asks the size table for the row of every tuple of holes it weighs, one count per size up
to the bound. Kept for the table's lifetime, those rows grew by about a gigabyte a draw on a
synthesized program at a bound of three thousand, where a row holds 3 001 counts. So they are kept
as the cost table keeps its exact rows: within a bound on their entries, the least recently used
evicted first, and convolved again when they are asked for again.

The tests pin what the bound must not change, a draw, a key, a row, and what it must: the entries
it keeps. A row of one hole and a suffix row the fill stored are the table's own and cost the
bound nothing.
"""

import copy
import dataclasses
import itertools
import random
import tracemalloc

import pytest

from cosy.search import counting as counting_module
from cosy.search import generator_query
from cosy.search.counting import SizeTable, size_table
from cosy.search.sampling import weighted_table
from tests.search_fixtures import (
    EXPR,
    PRICED,
    ROUND,
    SPLIT,
    TERNARY,
    expression_space,
    priced_space,
    round_space,
    split_space,
    ternary_space,
)

# A bound no stream here comes near: the table keeps every row it convolves.
MANY_ENTRIES = 10**12

# The spaces streamed below, with a size bound and a number of draws: each weighs tuples of two or
# more holes, and the ternary and priced spaces also carry suffix rows the fill stored.
ROW_SPACES = [
    ("split", split_space, SPLIT, 30, 300),
    ("ternary", ternary_space, TERNARY, 20, 300),
    ("priced", priced_space, PRICED, 20, 300),
    ("expression", expression_space, EXPR, 20, 300),
    ("round", round_space, ROUND, 30, 300),
]


def uniform(_value):
    """Weigh every realized size alike, the distribution of size-uniform sampling.

    Args:
        _value: The size. Ignored.

    Returns:
        float: One.
    """
    return 1.0


def held_entries(table):
    """Return the number of entries the rows a table keeps hold.

    Args:
        table (SizeTable): The table.

    Returns:
        int: The sum of the lengths of the rows it keeps.
    """
    return sum(len(row) for row in (table.__dict__.get("_split_rows") or {}).values())


def independent_split_row(table, hole_types):
    """Return the row of a tuple of holes by multiplying their rows as polynomials in the size.

    No convolution helper of the table and no cache: each hole's row has a zero at size 0, so the
    product counts exactly the fills that give every hole at least one symbol.

    Args:
        table (SizeTable): The table whose rows are multiplied.
        hole_types (tuple): The holes.

    Returns:
        tuple[int, ...]: Entry ``s`` is the number of ways to fill the holes with total size ``s``.
    """
    product = [1] + [0] * table.bound
    for hole in hole_types:
        row = table.counts.get(hole) or (0,) * (table.bound + 1)
        product = [sum(product[i] * row[size - i] for i in range(size + 1)) for size in range(table.bound + 1)]
    return tuple(product)


def test_a_stream_keeps_the_rows_it_convolved_within_the_bound(monkeypatch):
    """The rows a stream convolves are kept within a bound on their entries, never for the table's lifetime.

    Args:
        monkeypatch: pytest's monkeypatch.
    """
    space = split_space()
    query = generator_query(space, SPLIT)
    bound = 5 * 31
    unbounded = size_table(space, 30)
    list(itertools.islice(weighted_table(query, 30, uniform, table=unbounded).stream(random.Random(3)), 300))
    monkeypatch.setattr(counting_module, "ROW_CACHE_ENTRIES", bound, raising=False)
    bounded = size_table(space, 30)
    streamed = 0
    for _term in itertools.islice(weighted_table(query, 30, uniform, table=bounded).stream(random.Random(3)), 300):
        streamed += 1
        assert held_entries(bounded) <= bound, f"after {streamed} terms the rows hold {held_entries(bounded)} entries"
    assert streamed == 300
    if held_entries(unbounded) <= 4 * bound:
        pytest.fail(
            f"the stream needs only {held_entries(unbounded)} entries kept, so a bound of {bound} is never tested"
        )


@pytest.mark.parametrize(("name", "build", "start", "size_bound", "draws"), ROW_SPACES)
@pytest.mark.parametrize("seed", [0, 3, 11])
@pytest.mark.parametrize("rows_kept", [0, 2], ids=["keeping no row", "keeping two rows"])
def test_the_bound_on_the_kept_rows_changes_no_draw_and_no_key(
    monkeypatch, name, build, start, size_bound, draws, seed, rows_kept
):
    """A table that keeps two rows or none streams what one that keeps every row streams, term for term and key for key.

    Args:
        monkeypatch: pytest's monkeypatch.
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The queried non-terminal.
        size_bound (int): The size bound.
        draws (int): How many terms to stream.
        seed (int): The seed under test.
        rows_kept (int): How many rows of this bound the small bound holds.
    """
    query = generator_query(build(), start)
    monkeypatch.setattr(counting_module, "ROW_CACHE_ENTRIES", MANY_ENTRIES, raising=False)
    keeping = list(
        itertools.islice(weighted_table(query, size_bound, uniform).keyed_stream(random.Random(seed)), draws)
    )
    monkeypatch.setattr(counting_module, "ROW_CACHE_ENTRIES", max(1, rows_kept * (size_bound + 1)), raising=False)
    forgetting = list(
        itertools.islice(weighted_table(query, size_bound, uniform).keyed_stream(random.Random(seed)), draws)
    )
    assert keeping, name
    assert [term for _key, term in keeping] == [term for _key, term in forgetting], name
    assert [key for key, _term in keeping] == [key for key, _term in forgetting], name


@pytest.mark.parametrize(("name", "build", "start", "size_bound", "draws"), ROW_SPACES)
def test_a_bounded_table_returns_the_rows_an_unbounded_one_returns(monkeypatch, name, build, start, size_bound, draws):
    """Whatever the bound evicts is convolved again when asked for, to the same counts.

    Args:
        monkeypatch: pytest's monkeypatch.
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The queried non-terminal.
        size_bound (int): The size bound.
        draws (int): How many terms to stream.
    """
    space = build()
    weighted = weighted_table(generator_query(space, start), size_bound, uniform)
    list(itertools.islice(weighted.stream(random.Random(5)), draws))
    tuples = sorted(
        {holes for holes, _size in weighted.__dict__.get("_weight_cache") or {} if holes and len(holes) >= 2}, key=repr
    )
    assert len(tuples) >= 3, f"{name}: a stream that weighs few tuples of holes evicts little"
    bound = 2 * (size_bound + 1)
    monkeypatch.setattr(counting_module, "ROW_CACHE_ENTRIES", bound, raising=False)
    bounded = size_table(space, size_bound)
    for _round in range(2):
        for holes in tuples:
            assert bounded.split_row(holes) == independent_split_row(bounded, holes), (name, holes)
            assert held_entries(bounded) <= bound, (name, holes)


def recording_table(walks, **fields):
    """Return a size table that records each convolution it performs, by the hole it splits off.

    Args:
        walks (list): Receives one entry per convolution.
        **fields: The table's fields.

    Returns:
        SizeTable: The recording table.
    """

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

    return RecordingTable(**fields)


def test_tuples_that_share_a_suffix_convolve_it_once_under_a_bound_that_holds_it():
    """Four hundred tuples end in one suffix of eight holes; a bound of five rows keeps it, so it is convolved once."""
    shared = tuple(f"a{index}" for index in range(8))
    one_symbol = (0, 1, *(0,) * 19)
    counts = dict.fromkeys(shared, one_symbol) | {f"b{index}": one_symbol for index in range(400)}
    expected = tuple(1 if size == 9 else 0 for size in range(21))

    def convolved(bound):
        walks: list = []
        table = recording_table(walks, bound=20, counts=counts, row_cache_entries=bound)
        for index in range(400):
            assert table.split_row((f"b{index}", *shared)) == expected
        return len(walks)

    assert convolved(MANY_ENTRIES) == 7 + 400, "the shared suffix is seven convolutions, and each tuple one more"
    assert convolved(5 * 21) == 7 + 400, "a bound that holds the shared suffix convolves it once"


@pytest.mark.parametrize("bound", [0, 20])
def test_a_bound_below_one_row_keeps_nothing_and_every_row_is_still_right(bound):
    """Every row of a size table holds bound + 1 entries, so a bound below that keeps nothing and evicts nothing.

    Args:
        bound (int): The table's bound on its kept entries, below one row of 21.
    """
    rng = random.Random(4)
    names = [f"s{index}" for index in range(5)]
    counts = {name: (0, *(rng.randint(0, 3) for _ in range(20))) for name in names}
    table = SizeTable(bound=20, counts=counts, row_cache_entries=bound)
    for _ in range(50):
        holes = tuple(rng.choice(names) for _ in range(rng.randint(2, 4)))
        assert table.split_row(holes) == independent_split_row(table, holes), holes
        assert held_entries(table) == 0


def test_a_single_holes_row_and_a_stored_suffix_row_are_the_tables_own_and_cost_the_bound_nothing():
    """The row of one hole and a suffix row the fill stored are handed back as they are, and never kept again."""
    table = size_table(ternary_space(), 20, row_cache_entries=MANY_ENTRIES)
    assert table.suffix_counts, "a space whose fill stores no suffix row tests half of this"
    for suffix, stored in table.suffix_counts.items():
        assert table.split_row(suffix) is stored
    for nonterminal, row in table.counts.items():
        assert table.split_row((nonterminal,)) is row
    assert held_entries(table) == 0


@pytest.mark.parametrize("bound", [0, 62])
def test_a_table_keeps_its_own_bound_whatever_the_modules(monkeypatch, bound):
    """``row_cache_entries`` is the table's bound, and zero keeps nothing; the module's default does not override it.

    Args:
        monkeypatch: pytest's monkeypatch.
        bound (int): The table's own bound.
    """
    monkeypatch.setattr(counting_module, "ROW_CACHE_ENTRIES", MANY_ENTRIES)
    space = split_space()
    table = dataclasses.replace(size_table(space, 30), row_cache_entries=bound)
    sampler = weighted_table(generator_query(space, SPLIT), 30, uniform, table=table)
    streamed = 0
    for _term in itertools.islice(sampler.stream(random.Random(3)), 300):
        streamed += 1
        assert held_entries(table) <= bound, f"after {streamed} terms the rows hold {held_entries(table)} entries"
    assert streamed == 300


def test_the_entries_the_table_counts_are_the_entries_it_keeps():
    """After every question the running count is the sum of the kept rows' lengths, and within the bound."""
    rng = random.Random(9)
    names = [f"s{index}" for index in range(6)]
    counts = {name: (0, *(rng.randint(0, 3) for _ in range(12))) for name in names}
    bound = 4 * 13
    table = SizeTable(bound=12, counts=counts, row_cache_entries=bound)
    for _ in range(200):
        holes = tuple(rng.choice(names) for _ in range(rng.randint(2, 5)))
        assert table.split_row(holes) == independent_split_row(table, holes), holes
        kept = table.__dict__["_split_rows"]
        assert kept.entries == sum(len(row) for row in kept.values()) <= bound


def test_a_row_asked_for_again_is_kept_over_one_that_was_not():
    """The least recently used row goes first: a row asked for again outlives one asked for once, later."""
    one_or_two = (0, 1, 1, 0, 0, 0, 0)
    table = SizeTable(bound=6, counts=dict.fromkeys("ABCDEF", one_or_two), row_cache_entries=2 * 7)
    table.split_row(("A", "B"))
    table.split_row(("C", "D"))
    table.split_row(("A", "B"))
    table.split_row(("E", "F"))
    kept = table.__dict__["_split_rows"]
    assert ("A", "B") in kept, "the row asked for again was evicted"
    assert ("C", "D") not in kept, "the least recently used row was kept"
    assert ("E", "F") in kept


def test_a_stored_suffix_row_is_the_base_of_a_tuple_that_ends_in_it():
    """A tuple that ends in a suffix row the fill stored is convolved once, onto that row, not convolved again."""
    filled = size_table(ternary_space(), 20)
    assert filled.suffix_counts, "a space whose fill stores no suffix row tests nothing here"
    suffix = next(iter(filled.suffix_counts))
    head = next(iter(filled.counts))
    walks: list = []
    table = recording_table(walks, bound=filled.bound, counts=filled.counts, suffix_counts=filled.suffix_counts)
    assert table.split_row((head, *suffix)) == independent_split_row(table, (head, *suffix))
    assert walks == [head], f"the stored suffix was convolved again: {walks}"


# What a refuting review of the bound found, each finding now a test here: a module bound lowered on a table in use
# is honoured at the table's next kept row; a shallow copy shares the kept rows and their count; a longer kept suffix
# is the base over a shorter stored one; the memory the counting module holds, not only the attribute the tests read,
# stays flat under the bound; the bound is a whole number of entries, not negative; and two tables that count alike
# compare equal whatever their bounds.


def oracle_row(counts, bound, holes):
    """Return a tuple of holes' row from plain counts, as :func:`independent_split_row` does from a table.

    Args:
        counts (dict): The holes' rows.
        bound (int): The size bound.
        holes (tuple): The holes.

    Returns:
        tuple[int, ...]: Entry ``s`` is the number of ways to fill the holes with total size ``s``.
    """
    return independent_split_row(SizeTable(bound=bound, counts=counts), holes)


def test_a_longer_kept_suffix_is_the_base_over_a_shorter_stored_one():
    """``(H, A, B, C)`` with ``(B, C)`` stored and ``(A, B, C)`` kept is one convolution onto the kept row, not two."""
    bound = 12
    rng = random.Random(2)
    counts = {name: (0, *(rng.randint(0, 3) for _ in range(bound))) for name in "HABC"}
    stored = {("B", "C"): oracle_row(counts, bound, ("B", "C"))}
    walks: list = []
    table = recording_table(walks, bound=bound, counts=counts, suffix_counts=stored, row_cache_entries=MANY_ENTRIES)
    assert table.split_row(("A", "B", "C")) == oracle_row(counts, bound, ("A", "B", "C"))
    walks.clear()
    assert table.split_row(("H", "A", "B", "C")) == oracle_row(counts, bound, ("H", "A", "B", "C"))
    assert walks == ["H"], f"convolved {walks}, so the kept (A, B, C) was not the base"


@pytest.mark.parametrize("bound", [0, MANY_ENTRIES])
def test_tuples_two_or_more_holes_longer_than_a_stored_suffix_have_the_oracles_rows(bound):
    """A tuple two or three holes longer than a stored suffix is right at every size, with and without its middle kept.

    Args:
        bound (int): The table's bound on its kept entries.
    """
    size_bound = 14
    rng = random.Random(5)
    counts = {name: (0, *(rng.randint(0, 3) for _ in range(size_bound))) for name in "GHBC"}
    stored = {("B", "C"): oracle_row(counts, size_bound, ("B", "C"))}
    table = SizeTable(bound=size_bound, counts=counts, suffix_counts=stored, row_cache_entries=bound)
    for holes in (("G", "H", "B", "C"), ("H", "G", "H", "B", "C"), ("G", "G", "G", "B", "C")):
        assert table.split_row(holes) == oracle_row(counts, size_bound, holes), holes


def test_a_lowered_module_bound_is_honoured_at_the_next_kept_row(monkeypatch):
    """The module's bound is read when a row is kept, and a lowered one evicts what was kept above it.

    Args:
        monkeypatch: pytest's monkeypatch.
    """
    monkeypatch.setattr(counting_module, "ROW_CACHE_ENTRIES", MANY_ENTRIES)
    table = SizeTable(bound=6, counts=dict.fromkeys("ABCD", (0, 1, 1, 0, 0, 0, 0)))
    for holes in (("A", "B"), ("B", "C"), ("C", "D")):
        table.split_row(holes)
    assert held_entries(table) == 21, "the control: three rows kept under the first bound"
    monkeypatch.setattr(counting_module, "ROW_CACHE_ENTRIES", 7)
    table.split_row(("D", "A"))
    assert held_entries(table) == 7, f"{held_entries(table)} entries kept under a bound of 7"
    assert ("D", "A") in table.__dict__["_split_rows"], "the row kept last was not the one kept"
    monkeypatch.setattr(counting_module, "ROW_CACHE_ENTRIES", 0)
    table.split_row(("B", "D"))
    assert held_entries(table) == 0, f"{held_entries(table)} entries kept under a bound of 0"


def test_a_shallow_copy_shares_the_kept_rows_and_their_count():
    """A copy of a table shares its kept rows, and both count them as one, so the bound holds over the two."""
    table = SizeTable(bound=6, counts=dict.fromkeys("ABCDEF", (0, 1, 1, 0, 0, 0, 0)), row_cache_entries=14)
    table.split_row(("A", "B"))
    twin = copy.copy(table)
    twin.split_row(("C", "D"))
    table.split_row(("E", "F"))
    assert table.__dict__["_split_rows"] is twin.__dict__["_split_rows"], "the control: a shallow copy shares the cache"
    assert held_entries(table) <= 14, f"{held_entries(table)} entries kept between the two under a bound of 14"


def test_keeping_a_row_under_a_kept_tuple_counts_it_once():
    """A row kept again under its own tuple replaces it, and the entries count it once."""
    table = SizeTable(bound=6, counts=dict.fromkeys("AB", (0, 1, 1, 0, 0, 0, 0)), row_cache_entries=MANY_ENTRIES)
    row = table.split_row(("A", "B"))
    table._keep(table._rows, ("A", "B"), row)  # noqa: SLF001
    assert len(table.__dict__["_split_rows"]) == 1
    assert held_entries(table) == 7
    table._keep(table._rows, ("A", "A"), row)  # noqa: SLF001
    table.split_row(("B", "B"))
    kept = table.__dict__["_split_rows"]
    assert kept.entries == held_entries(table) == 21


def held_by_counting(stream, early, late):
    """Return the bytes the counting module allocated and still holds after the early and after the late draws.

    Args:
        stream: A stream of terms.
        early (int): The draws before the first reading.
        late (int): The draws between the two readings.

    Returns:
        tuple[int, int]: The bytes after ``early`` draws and after ``early + late``.
    """
    only_counting = [tracemalloc.Filter(True, counting_module.__file__)]

    def held() -> int:
        snapshot = tracemalloc.take_snapshot().filter_traces(only_counting)
        return sum(stat.size for stat in snapshot.statistics("filename"))

    tracemalloc.start()
    try:
        list(itertools.islice(stream, early))
        first = held()
        list(itertools.islice(stream, late))
        second = held()
    finally:
        tracemalloc.stop()
    return first, second


def test_a_bounded_stream_holds_no_growing_memory_in_the_counting_module():
    """Memory, not an attribute: what the counting module allocated and holds grows without a bound, not under one."""
    space = split_space()
    query = generator_query(space, SPLIT)
    grown = {}
    for name, bound in (("unbounded", MANY_ENTRIES), ("bounded", 2 * 31)):
        table = size_table(space, 30, row_cache_entries=bound)
        stream = weighted_table(query, 30, uniform, table=table).stream(random.Random(3))
        first, second = held_by_counting(stream, 20, 280)
        grown[name] = second - first
    assert grown["unbounded"] >= 8_000, f"unbounded, the cache grows by {grown['unbounded']} bytes: it tests nothing"
    assert grown["bounded"] <= grown["unbounded"] // 10, f"under the bound the counting module grew by {grown}"


@pytest.mark.parametrize("bound", [-1, 2.5, True, "62"])
def test_a_bound_that_is_not_a_whole_number_of_entries_is_refused(bound):
    """A negative bound, a fraction, a truth value and a string are refused when the table is built, not when it keeps.

    Args:
        bound: A bound that is not a non-negative whole number.
    """
    with pytest.raises(ValueError, match="row_cache_entries"):
        SizeTable(bound=6, counts={}, row_cache_entries=bound)


def test_two_tables_that_count_alike_compare_equal_whatever_their_bounds():
    """The bound decides what a table keeps, not what it counts, so it is no part of a table's equality."""
    counts = {"A": (0, 1, 0)}
    assert SizeTable(bound=2, counts=counts) == SizeTable(bound=2, counts=counts, row_cache_entries=10)
