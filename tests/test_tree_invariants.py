"""Invariants of :class:`cosy.core.tree.Tree`, including its equality and hash contract.

``Tree`` is the type every cache, every ``set`` and every deduplication in this package is keyed
on, and its docstring asks callers to treat it as immutable.  ``replace_subtree_at`` used to mutate
``children`` in place after copying, so ``size`` and ``_hash`` kept the values they had before the
replacement.  The consequences reached far beyond this class: an offspring assembled by crossover
compared unequal to the structurally identical tree built directly, and no ``set`` of trees
recognized it.

The tests below are in two groups.  The first fixes the properties that must hold whatever the
implementation does.  The second is about the cached fields specifically: a replacement has to
recompute them.
"""

import random

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
