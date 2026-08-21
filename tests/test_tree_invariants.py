"""Invariants of :class:`cosy.core.tree.Tree`, including its equality and hash contract.

``Tree`` is the type every cache, every ``set`` and every deduplication in this package is keyed
on, and its docstring asks callers to treat it as immutable.  ``replace_subtree_at`` used to mutate
``children`` in place after copying, so ``size`` and ``_hash`` kept the values they had before the
replacement.  The consequences reached far beyond this class: an offspring assembled by crossover
compared unequal to the structurally identical tree built directly, and no ``set`` of trees
recognized it.

The tests below are in three groups.  The first fixes the properties that must hold whatever the
implementation does.  The second is about the cached fields specifically: a replacement has to
recompute them.  The third asks the same question about pickling, which is the other way a node can
be asked what belongs to it: the derived fields must stay out of the stream.  ``_hash`` because a
hash computed under another process's seed breaks the very contract the first group fixes, the
interpretation cache because its key stops naming anything once it has travelled, and the position
sets because carrying them is waste.
"""

import os
import pickle
import random
import subprocess
import sys

import pytest

from cosy.core.tree import Path, Tree


def random_tree(rng: random.Random, depth: int = 3, max_arity: int = 3) -> Tree[str]:
    """Build a random tree of at most the given depth.

    Args:
        rng (random.Random): Seeded source of randomness; the test suite runs in random order and
            in parallel, so a test must never draw from the global ``random`` module.
        depth (int): Remaining depth budget. (Default value = 3)
        max_arity (int): Largest number of children a node may get. (Default value = 3)

    Returns:
        Tree[str]: A tree whose labels are drawn from a small alphabet, so structurally equal
            subtrees occur often enough to exercise equality and hashing.
    """
    label = rng.choice(["f", "g", "h", "x", "y"])
    if depth <= 0:
        return Tree(label)
    arity = rng.randint(0, max_arity)
    return Tree(label, tuple(random_tree(rng, depth - 1, max_arity) for _ in range(arity)))


def rebuild(tree: Tree[str]) -> Tree[str]:
    """Rebuild a tree bottom-up through ``__init__``, so every cached field is recomputed.

    Args:
        tree (Tree[str]): The tree to rebuild.

    Returns:
        Tree[str]: A structurally identical tree whose ``size`` and ``_hash`` are correct by
            construction.  This is the reference any tree returned by ``replace_subtree_at`` has to
            be indistinguishable from.
    """
    return Tree(tree.root, tuple(rebuild(child) for child in tree.children))


def node_at(tree: Tree[str], pos: Path) -> Tree[str]:
    """Descend to a position without going through ``subtree_at``.

    Args:
        tree (Tree[str]): The tree to descend into.
        pos (Path): The position to reach.

    Returns:
        Tree[str]: The node stored at ``pos``.  Written independently of ``subtree_at`` so a test
            of that method cannot be satisfied by its own implementation.
    """
    current = tree
    for index in pos:
        current = current.children[index]
    return current


def replaced_at(tree: Tree[str], pos: Path, subtree: Tree[str]) -> Tree[str]:
    """Put a subtree at a position without going through ``replace_subtree_at``.

    Args:
        tree (Tree[str]): The tree to rebuild.
        pos (Path): The position whose node is replaced.
        subtree (Tree[str]): What to put there.

    Returns:
        Tree[str]: The tree with ``subtree`` at ``pos``, built by rebuilding every node on the way
            down.  Written independently of ``replace_subtree_at``, because a test that compared
            its result against another of its results -- or read back only the position it had just
            written -- would be satisfied by any implementation that is merely consistent with
            itself.  A replacement that dropped, duplicated or reordered the siblings past the
            replacement point is exactly that.
    """
    if not pos:
        return subtree
    index = pos[0]
    children = list(tree.children)
    children[index] = replaced_at(children[index], pos[1:], subtree)
    return Tree(tree.root, tuple(children))


@pytest.fixture
def rng() -> random.Random:
    """Return the seeded RNG shared by the property-style tests.

    Returns:
        random.Random: A generator seeded with a fixed value, so a failure is reproducible.
    """
    return random.Random(20260728)


@pytest.fixture
def sample() -> Tree[str]:
    """Return a small asymmetric tree with a repeated label.

    Returns:
        Tree[str]: ``f(g(x), y)`` -- asymmetric, so a defect that swaps children is visible, and
            deep enough for replacements below the root.
    """
    return Tree("f", (Tree("g", (Tree("x"),)), Tree("y")))


# ---------------------------------------------------------------------------
# Properties that hold whatever the implementation does
# ---------------------------------------------------------------------------


def test_equality_is_an_equivalence_relation(sample: Tree[str]) -> None:
    """Equality is reflexive and symmetric, and unequal structures stay unequal.

    Args:
        sample (Tree[str]): The shared sample tree.
    """
    assert sample in {sample}
    assert sample == rebuild(sample)
    assert rebuild(sample) == sample
    assert sample != Tree("f", (Tree("y"), Tree("g", (Tree("x"),))))
    assert sample != "not a tree"


def test_positions_are_prefix_closed(rng: random.Random) -> None:
    """Every prefix of a position is itself a position, and the root is always present.

    Args:
        rng (random.Random): Seeded RNG fixture.
    """
    for _ in range(30):
        tree = random_tree(rng)
        positions = tree.positions()
        assert () in positions
        assert len(positions) == tree.size
        for pos in positions:
            for cut in range(len(pos)):
                assert pos[:cut] in positions


def test_leaf_positions_are_exactly_the_childless_nodes(rng: random.Random) -> None:
    """``leaf_positions`` holds every position whose node has no children, and nothing else.

    Asked first, before anything else has been asked of the term: the two sets are filled by one
    traversal, so every other question warms both of them and would leave this one answered from a
    cache rather than computed.

    Args:
        rng (random.Random): Seeded RNG fixture.
    """
    for _ in range(30):
        tree = random_tree(rng)
        leaves = tree.leaf_positions()
        expected = {pos for pos in tree.positions() if not node_at(tree, pos).children}
        assert leaves == expected
        assert leaves <= tree.positions()


def test_subtree_at_rejects_invalid_paths(sample: Tree[str]) -> None:
    """An unreachable path raises ``IndexError`` rather than returning something wrong.

    The negative cases are the ones a bare tuple index would not catch: ``children[-1]`` is the
    last child, so without an explicit check ``subtree_at`` would answer a position that is not
    in ``positions()`` instead of rejecting it.  A caller relies on exactly that rejection --
    resolving a term at a position uses ``subtree_at`` as its validity test and treats
    ``IndexError`` as "no such position".

    Args:
        sample (Tree[str]): The shared sample tree.
    """
    for pos in ((5,), (1, 0), (-1,), (0, -1), (0, -1, 0)):
        with pytest.raises(IndexError):
            sample.subtree_at(pos)


def test_replace_then_read_back(rng: random.Random) -> None:
    """Reading a position back after replacing it yields the replacement.

    Args:
        rng (random.Random): Seeded RNG fixture.
    """
    replacement = Tree("REPL", (Tree("a"), Tree("b")))
    for _ in range(30):
        tree = random_tree(rng)
        for pos in tree.positions():
            assert tree.replace_subtree_at(pos, replacement).subtree_at(pos) == replacement


def test_replace_leaves_both_operands_untouched(sample: Tree[str]) -> None:
    """Neither operand is modified by a replacement -- its caches included.

    The class asks to be used immutably, so a caller holding either tree must not see it change.
    The position sets are part of that: they are the caches of the node itself, and an operation
    that discarded or refilled them would make every other term holding that node pay for a walk
    it had already done.

    Args:
        sample (Tree[str]): The shared sample tree.
    """
    replacement = Tree("z", (Tree("q"),))
    before_self = rebuild(sample)
    before_replacement = rebuild(replacement)
    positions_before = sample.positions()
    leaves_before = sample.leaf_positions()

    sample.replace_subtree_at((0, 0), replacement)

    assert sample == before_self
    assert sample.size == before_self.size
    assert replacement == before_replacement
    assert replacement.size == before_replacement.size
    assert sample.positions() is positions_before
    assert sample.leaf_positions() is leaves_before


def test_replace_rejects_invalid_paths(sample: Tree[str]) -> None:
    """An unreachable path raises ``IndexError``, wherever along it the tree runs out of children.

    One kind of mistake, one exception type.  The previous implementation also raised
    ``IndexError`` here, through a separate pre-validating traversal; the ``ValueError`` its
    second traversal could raise was unreachable, because the pre-validation had already checked
    every index it looked at.  That branch disappeared along with the traversal it belonged to,
    so what a caller sees is unchanged -- this test holds it that way.  The negative cases are
    the ones a bare tuple index would not reject.

    Args:
        sample (Tree[str]): The shared sample tree.
    """
    replacement = Tree("z")
    for pos in ((5,), (1, 0), (0, 5, 0), (-1,), (0, -1, 0)):
        with pytest.raises(IndexError):
            sample.replace_subtree_at(pos, replacement)


# ---------------------------------------------------------------------------
# The cached fields: size and hash
# ---------------------------------------------------------------------------


def test_a_replacement_is_indistinguishable_from_the_tree_built_directly(sample: Tree[str]) -> None:
    """A replaced tree equals, sizes and hashes like the same structure built directly.

    This is the defect the commit is about: the replacement mutated a copy, so ``size`` and
    ``_hash`` kept the values they had before it.  Both consequences are here, because they fail
    differently.  ``a == b`` with ``hash(a) != hash(b)`` breaks Python's own contract -- no
    ``set`` or ``dict`` finds the tree again, and a term that was already evaluated looks new.
    A stale ``size`` is worse still: ``__eq__`` compares it before the structure, so the two trees
    do not even compare equal.  Only the second case, which changes the size, tells them apart.

    Args:
        sample (Tree[str]): The shared sample tree.
    """
    bigger = Tree("B", (Tree("C"), Tree("D")))
    for replacement, expected in (
        (Tree("z"), Tree("f", (Tree("g", (Tree("z"),)), Tree("y")))),
        (bigger, Tree("f", (Tree("g", (bigger,)), Tree("y")))),
    ):
        replaced = sample.replace_subtree_at((0, 0), replacement)

        assert replaced.size == expected.size
        assert replaced == expected
        assert hash(replaced) == hash(expected)
        assert len({replaced, expected}) == 1


def test_repeated_replacements_stay_consistent(rng: random.Random) -> None:
    """Replacing every position in turn must build the right tree and keep its caches correct.

    This is the shape crossover produces: an offspring is a sequence of replacements, and each one
    has to leave a tree that the next operation can still recognize.  Two claims, because the
    caches and the structure fail separately.  The structure is checked against ``replaced_at``
    rather than against the result itself; the caches are checked against the same structure built
    bottom-up through ``__init__``, which is where they are correct by construction.

    Args:
        rng (random.Random): Seeded RNG fixture.
    """
    replacement = Tree("REPL", (Tree("a"),))
    for _ in range(20):
        tree = random_tree(rng)
        for pos in sorted(tree.positions()):
            replaced = tree.replace_subtree_at(pos, replacement)
            assert replaced == replaced_at(tree, pos, replacement)
            assert hash(replaced) == hash(rebuild(replaced))


# ---------------------------------------------------------------------------
# What a node carries into a pickle stream
# ---------------------------------------------------------------------------


class _TaggedTree(Tree[str]):
    """A subclass of ``Tree``, used to check that loading does not change what an object is.

    Defined at module level because pickle stores a class by reference and has to be able to
    import it back.  There is no subclass of ``Tree`` in the package itself; this one exists so
    that the choice between ``self.__class__`` and a named class in the reduction is pinned by a
    test rather than left to whoever edits the line next.
    """


def _sum_of(left: int, right: int) -> int:
    """Add two numbers.

    Defined at module level, and named distinctly, so that a pickle stream carrying the
    interpretation can be recognized by looking for this name in it.

    Args:
        left (int): First summand.
        right (int): Second summand.

    Returns:
        int: Their sum.
    """
    return left + right


def _one() -> int:
    """Return one.

    Returns:
        int: The number one.
    """
    return 1


def _two() -> int:
    """Return two.

    Returns:
        int: The number two.
    """
    return 2


def balanced(levels: int) -> Tree[str]:
    """Build a perfect binary tree of the given depth.

    Args:
        levels (int): Number of levels below the root.

    Returns:
        Tree[str]: A term of ``2 ** (levels + 1) - 1`` nodes, large enough that a position set
            travelling in a pickle stream would be plainly visible in its size.
    """
    if levels == 0:
        return Tree("x")
    return Tree("f", (balanced(levels - 1), balanced(levels - 1)))


def _written_by_a_process_seeded_with(seed: str) -> tuple[int, bytes]:
    """Build the sample term in a fresh interpreter and bring back the stream it wrote.

    Args:
        seed (str): The ``PYTHONHASHSEED`` that interpreter runs under.

    Returns:
        tuple[int, bytes]: What that interpreter computed for ``hash("alpha")``, so that the caller
            can tell whether it hashes strings differently from this one at all, and the term as it
            wrote it.
    """
    source = (
        "import pickle, sys\n"
        "from cosy.core.tree import Tree\n"
        "term = Tree('f', (Tree('alpha'), Tree('beta')))\n"
        "term.positions()\n"
        "sys.stdout.buffer.write(pickle.dumps((hash('alpha'), pickle.dumps(term))))\n"
    )
    environment = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=os.pathsep.join(sys.path))
    written = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        check=True,
        env=environment,
    ).stdout
    return pickle.loads(written)


def test_a_term_that_arrives_by_stream_hashes_like_one_built_here() -> None:
    """Equal terms hash equally, even when one of them was written by another process.

    ``_hash`` is ``hash((root, children))``, and hashing a string is randomized per process, so a
    ``_hash`` that travels in the stream is the writing process's answer.  A term loaded from such
    a stream compared equal to the same term built here and hashed differently, which is a broken
    hash contract and not an abstract one: a ``set`` held both copies and a ``dict`` deduplicated
    neither, so a pool of terms merged from two runs counted every shared term twice.

    Two child processes rather than one, because this process's own seed is not known here; at
    most one of the two can share it.
    """
    written = [_written_by_a_process_seeded_with(seed) for seed in ("1", "2")]
    foreign = [blob for hash_of_alpha, blob in written if hash_of_alpha != hash("alpha")]
    assert foreign, "no child process hashed strings differently from this one"

    here = Tree("f", (Tree("alpha"), Tree("beta")))
    for blob in foreign:
        back = pickle.loads(blob)
        assert back == here
        assert hash(back) == hash(here)
        assert len({back, here}) == 1


def test_pickle_keeps_shared_subtrees_shared() -> None:
    """A node that occurs several times in a term is one object again after a round trip.

    Sharing is what this class buys by being immutable -- ``subtree_at`` hands out the node and
    ``replace_subtree_at`` shares everything off the path it rebuilds -- so a term of a thousand
    nodes can be a handful of distinct objects.  A round trip that unfolded the sharing would
    silently turn that back into a thousand.  Pickle memoizes the objects it has already written
    whichever way they are reduced; this test holds that property for whatever reduction the
    class grows next.
    """
    shared = Tree("s", (Tree("a"), Tree("b")))
    tree = Tree("f", (shared, Tree("g", (shared,)), shared))

    back = pickle.loads(pickle.dumps(tree))

    assert back == tree
    first = back.children[0]
    assert back.children[1].children[0] is first
    assert back.children[2] is first


def test_the_interpretation_is_left_behind_and_never_blocks_pickling() -> None:
    """The interpretation does not travel with the term.

    A node used to keep the result of its last evaluation, together with the interpretation that
    produced it, and that was a bug in both directions.  Outward: the interpretation held a lambda,
    and a single interpreted node therefore failed to pickle at all.  Terms are pickled to move a
    population between processes, and an algebra assembled from lambdas is the ordinary case.  Inward: the entry was keyed on ``id(interpretation)``, and a round trip
    rebuilds that dictionary elsewhere, so the key stopped naming what the entry held.  A node
    holds no result at all now, and this test keeps it that way.
    """
    tree = Tree("add", (Tree("one"), Tree("two")))
    assert tree.interpret({"add": _sum_of, "one": _one, "two": _two}) == 3

    blob = pickle.dumps(tree)
    assert b"_sum_of" not in blob

    with_lambdas = Tree("add", (Tree("one"), Tree("two")))
    assert with_lambdas.interpret({"add": lambda left, right: left + right, "one": lambda: 1, "two": lambda: 2}) == 3
    assert pickle.loads(pickle.dumps(with_lambdas)) == with_lambdas


def test_nothing_derived_travels_and_everything_derived_comes_back(rng: random.Random) -> None:
    """A term survives a round trip whole, with every derived field recomputed rather than read.

    Two claims, because neither is worth much alone.  That nothing derived is written: the derived
    fields are a second encoding of a structure the stream already carries, they are filled by any
    ordinary use of the term, and so they would travel almost always -- whether a term has been
    used before must not change what it costs to pickle.  That everything derived comes back
    anyway: a field left out of the stream has to be recomputed on the way in, or the term that
    arrives is not the term that was sent.

    The two streams are compared against each other rather than against an absolute size, which is
    not stable across Python versions.  The stream is searched for the names of the derived fields
    as well; they are what the default protocol would have written.

    Args:
        rng (random.Random): Seeded RNG fixture.
    """
    cold = balanced(6)
    warm = balanced(6)
    warm.positions()
    warm.leaf_positions()

    assert pickle.dumps(warm) == pickle.dumps(cold)

    for _ in range(30):
        tree = random_tree(rng)
        tree.positions()
        tree.leaf_positions()

        blob = pickle.dumps(tree)
        assert b"_hash" not in blob
        assert b"size" not in blob
        assert b"_positions" not in blob

        back = pickle.loads(blob)

        assert back._positions is None  # noqa: SLF001
        assert back._leaf_positions is None  # noqa: SLF001
        assert back == tree
        assert hash(back) == hash(tree)
        assert back.size == tree.size
        assert back.positions() == tree.positions()
        assert back.leaf_positions() == tree.leaf_positions()


def test_a_subclass_comes_back_as_itself() -> None:
    """Loading rebuilds the class the term was written as, not the base class.

    Nothing in the package subclasses ``Tree``, so this pins a decision rather than a use: the
    reduction reconstructs through ``self.__class__``, because a stream may not silently change
    what an object is.
    """
    tagged = _TaggedTree("f", (Tree("x"),))

    back = pickle.loads(pickle.dumps(tagged))

    assert type(back) is _TaggedTree
    assert back == tagged
    assert hash(back) == hash(tagged)
    assert back.size == tagged.size


def test_a_stream_that_still_carries_an_instance_dictionary_is_rebuilt(monkeypatch: pytest.MonkeyPatch) -> None:
    """A term written before the reduction existed is repaired on the way in, not adopted.

    The reduction decides what is written; what is read is decided by what stands in the stream,
    and terms written by an earlier version are on disk already -- they are exactly the ones worth
    keeping, since a pool of them is the record of a long run.  Such a stream carries ``_hash``,
    and a ``_hash`` from another process is the defect this commit is about, so the reading side
    recomputes rather than adopts.

    The old format is produced by writing the term through ``object.__reduce__`` for the duration
    of the write, which is the reduction the class had before this one: a reconstructor plus the
    instance dictionary.  The root's ``_hash`` is set to a value it cannot have computed itself,
    standing in for the different value another process's hash seed would have produced, and its
    ``size`` likewise: that is the field the earlier, mutating ``replace_subtree_at`` left stale,
    so a stream from back then carries a wrong one and repairing it is half of why this method
    exists.

    Args:
        monkeypatch (pytest.MonkeyPatch): Used to write one stream in the old format.
    """
    tree = Tree("f", (Tree("a"), Tree("b")))
    tree.positions()
    tree.leaf_positions()
    assert tree.interpret({"f": _sum_of, "a": _one, "b": _two}) == 3
    tree._hash = 0  # noqa: SLF001
    tree.size = 99

    monkeypatch.setattr(Tree, "__reduce__", object.__reduce__, raising=False)
    legacy = pickle.dumps(tree)
    monkeypatch.undo()

    assert b"_hash" in legacy
    assert b"_positions" in legacy

    back = pickle.loads(legacy)

    fresh = Tree("f", (Tree("a"), Tree("b")))
    assert back == fresh
    assert hash(back) == hash(fresh)
    assert back.size == fresh.size
    assert back._positions is None  # noqa: SLF001
    assert back._leaf_positions is None  # noqa: SLF001
