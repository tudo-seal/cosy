"""What the tree operations cost, written as invariants rather than as timings.

``Tree`` is immutable: every operation that changes a term builds a new one and shares whatever it
did not change (``replace_subtree_at`` is written that way, and the equality and hash contract
depends on it).  Three operations were written before that was true and still defended against a
mutation that cannot happen:

* ``subtree_at`` copied the subtree it descended into, once per level, so reading a position cost
  a full copy of everything below it.  Its callers then held a clone, which quietly defeated the
  sharing the rest of the class is built on.
* ``__copy__`` recursed into every child, so a "copy" of a term was a copy of the whole term.
* ``leaf_positions`` filtered the position set against itself, which is quadratic in the number of
  nodes -- and the leaves of a term are asked for once per query against it.

The tests below count operations instead of measuring time.  A counted operation says the same
thing on a loaded machine as on an idle one, whereas a wall-clock bound would only say how busy
the machine running the suite happens to be.  Where the claim is about growth rather than about a
single call -- the quadratic filter above -- the count comes from ``sys.setprofile``, which reports
every call the interpreter makes and so measures work done rather than time taken.
"""

import sys
from collections.abc import Callable
from copy import copy
from types import FrameType
from typing import Any

import pytest

from cosy.core.tree import Path, Tree


def chain(depth: int) -> Tree[str]:
    """Build a unary chain of the given depth.

    Args:
        depth (int): The number of unary nodes above the leaf.

    Returns:
        Tree[str]: The chain.
    """
    node: Tree[str] = Tree("leaf", ())
    for _ in range(depth):
        node = Tree("f", (node,))
    return node


def shared_layers(depth: int) -> Tree[str]:
    """Build a complete binary tree of the given depth out of one node object per level.

    Both children of every level are the same object, so the result is a term of
    ``2**(depth+1) - 1`` *positions* held in only ``depth + 1`` node objects.  That is the sharing
    this module is about, and it is what makes the large cases below cheap to build: nothing here
    depends on the two subtrees being distinct, and every test reasons in positions.

    Args:
        depth (int): The number of levels above the leaves.

    Returns:
        Tree[str]: The tree, with ``2**(depth+1) - 1`` positions.
    """
    node: Tree[str] = Tree("leaf", ())
    for _ in range(depth):
        node = Tree("g", (node, node))
    return node


@pytest.fixture
def walk_counter(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count ``Tree._walk`` calls for the duration of one test.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.

    Returns:
        list[int]: A single-element list holding the number of calls so far.
    """
    calls = [0]
    original = Tree._walk  # noqa: SLF001

    def counting(self: Tree[Any]) -> tuple[frozenset[Path], frozenset[Path]]:
        """Count one call and delegate.

        Args:
            self (Tree[Any]): The tree being walked.

        Returns:
            tuple[frozenset[Path], frozenset[Path]]: Whatever the original returns.
        """
        calls[0] += 1
        return original(self)

    monkeypatch.setattr(Tree, "_walk", counting)
    return calls


def count_operations(work: Callable[[], object], budget: int) -> int:
    """Count the interpreter events ``work`` causes, giving up once ``budget`` is passed.

    ``sys.setprofile`` fires on every call and return, into Python and into C alike, so the total
    is a measure of work performed and not of time spent: it is the same number on a loaded
    machine as on an idle one, and on every interpreter in the test matrix.  Counting stops at the
    budget, so an implementation that is off by a factor of the term size fails in the time the
    budget allows rather than running to completion.

    Args:
        work (Callable[[], object]): The operation to measure, called once with no arguments.
        budget (int): The event count at which measuring stops.

    Returns:
        int: The number of events, never more than ``budget + 1``.
    """
    events = [0]

    def profile(_frame: FrameType, _event: str, _arg: Any) -> None:
        """Count one event and disarm once the budget is spent.

        Args:
            _frame (FrameType): The frame the event happened in.
            _event (str): The kind of event.
            _arg (Any): The event argument.
        """
        events[0] += 1
        if events[0] > budget:
            sys.setprofile(None)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        work()
    finally:
        sys.setprofile(previous)
    return events[0]


# ---------------------------------------------------------------------------
# subtree_at -- reading a position must not build anything
# ---------------------------------------------------------------------------


def test_subtree_at_returns_the_node_itself() -> None:
    """The subtree at a position is the node that sits there, not a clone of it.

    Sharing is the contract of the class: ``replace_subtree_at`` shares every node off the path
    it rebuilds, and a reader that hands out clones makes that sharing unobservable -- and turns
    every read into a copy of the subtree.
    """
    tree = shared_layers(3)
    assert tree.subtree_at((0,)) is tree.children[0]
    assert tree.subtree_at((1, 0)) is tree.children[1].children[0]
    assert tree.subtree_at(()) is tree


def test_a_deep_chain_can_be_read_to_the_bottom() -> None:
    """Reading the bottom of a chain works at any depth a sampler can produce.

    Terms grow to hundreds of nodes, and the recursions that build them materialize partial terms
    along the way, so a term deeper than the interpreter's recursion limit is not a pathological
    case here.
    """
    deep = chain(5000)
    bottom = tuple(0 for _ in range(5000))
    assert deep.subtree_at(bottom).root == "leaf"


# ---------------------------------------------------------------------------
# replace_subtree_at -- rebuild the path, share the rest
# ---------------------------------------------------------------------------


def test_replace_subtree_at_shares_everything_off_the_path() -> None:
    """Only the nodes between the root and the replaced position are new.

    This is the contract the docstring states, and the reason a replacement is cheap: an offspring
    assembled by crossover costs the depth of the crossover point, not the size of the parent.  It
    is also what keeps the per-node interpretation cache useful across a generation -- a rebuilt
    node starts out with an empty one, so a copying implementation would make every individual
    look new to ``interpret`` even where nothing about it changed.
    """
    tree = Tree("f", (Tree("g", (Tree("x"), Tree("w"))), Tree("y")))
    replacement = Tree("z")

    replaced = tree.replace_subtree_at((0, 0), replacement)

    assert replaced.children[1] is tree.children[1]
    assert replaced.children[0].children[1] is tree.children[0].children[1]
    assert replaced.children[0].children[0] is replacement
    assert replaced is not tree
    assert replaced.children[0] is not tree.children[0]


# ---------------------------------------------------------------------------
# __copy__ -- shallow, because the nodes are immutable
# ---------------------------------------------------------------------------


def test_copy_shares_the_children() -> None:
    """A shallow copy of an immutable node shares its children.

    ``copy`` on an immutable structure is a new root over the same children; the recursive version
    is ``deepcopy`` under another name, and nothing in this code base needs it.
    """
    tree = shared_layers(3)
    duplicate = copy(tree)
    assert duplicate == tree
    assert duplicate is not tree
    for original_child, copied_child in zip(tree.children, duplicate.children, strict=True):
        assert copied_child is original_child


# ---------------------------------------------------------------------------
# leaf_positions -- one pass over the nodes, not a self-join over the positions
# ---------------------------------------------------------------------------


def test_one_walk_fills_both_position_sets(walk_counter: list[int]) -> None:
    """Asking for either set fills the other one too, so a term is walked once and not twice.

    The two sets fall out of the same traversal -- a node without children is a leaf -- so
    computing them separately would walk every node a second time for an answer already known.
    Asking again must not walk at all: resolving a term reads the leaves once per query against
    it, and crossover reads them four times per attempt.

    Args:
        walk_counter (list[int]): The ``_walk`` call counter fixture.
    """
    from_the_leaves = shared_layers(4)
    from_the_leaves.leaf_positions()
    assert from_the_leaves._positions is not None  # noqa: SLF001
    assert walk_counter[0] == 1

    from_the_positions = shared_layers(4)
    from_the_positions.positions()
    assert from_the_positions._leaf_positions is not None  # noqa: SLF001
    assert walk_counter[0] == 2

    assert from_the_leaves.positions() == from_the_positions.positions()
    assert from_the_leaves.leaf_positions() == from_the_positions.leaf_positions()
    assert walk_counter[0] == 2


def test_a_deep_chain_can_be_walked_to_the_bottom() -> None:
    """Collecting the positions of a term works at any depth, not only at the depth of a fixture.

    Every other term in these tests is shallow enough that a recursive traversal would survive it,
    and the terms this package produces are not: a chain of five thousand nodes is what a sampler
    or a long run of mutations leaves behind, and asking such a term for its positions must answer
    rather than exhaust the interpreter's stack.  ``subtree_at`` and ``interpret`` are iterative
    for the same reason.
    """
    depth = 5000
    deep = chain(depth)

    assert len(deep.positions()) == depth + 1
    assert deep.leaf_positions() == {tuple(0 for _ in range(depth))}


def test_leaf_positions_costs_one_pass_over_the_term() -> None:
    """Reading the leaves is linear in the term, not quadratic in it.

    This is the regression the rewrite was for, and the one the tests above cannot see: a filter
    that lives inside the traversal and compares the positions it has just collected against each
    other never asks ``positions()`` for anything, so counting those calls says nothing about it.
    What it cannot hide is the work: the quadratic form costs on the order of one comparison per
    pair of positions, the walk costs a constant number of steps per position.

    The budget is fifty events per position against about seven that the walk actually spends, so
    it is not a bound on how the traversal is written -- only on how it grows.  On this term the
    quadratic form passes fifty per position after the first few hundred of them.
    """
    tree = shared_layers(11)
    positions = 2**12 - 1
    budget = 50 * positions

    events = count_operations(tree.leaf_positions, budget)

    assert events <= budget, f"leaf_positions spent more than {budget} events on {positions} positions"


# ---------------------------------------------------------------------------
# positions()/leaf_positions() -- the caches are not the caller's to keep
# ---------------------------------------------------------------------------


def test_a_shared_node_reports_one_set_to_every_term_that_holds_it() -> None:
    """One node, one pair of sets, however many terms are built around it.

    This is why the sets are handed out frozen rather than merely by convention: ``subtree_at``
    hands out the node itself and ``replace_subtree_at`` grafts that same node into the term it
    builds, so a node belongs to several terms at once and the sets it caches are the sets all of
    them read.  A caller able to change one would silently change what every holder of that node
    reports, which is what the last two lines ask about.
    """
    tree = Tree("f", (shared_layers(3), Tree("y")))
    subtree = tree.subtree_at((0,))
    assert subtree is tree.children[0]

    grafted = Tree("h", (Tree("a"), subtree))
    assert grafted.children[1].positions() is subtree.positions()
    assert grafted.children[1].leaf_positions() is subtree.leaf_positions()

    with pytest.raises(AttributeError):
        subtree.positions().add((42,))  # type: ignore[attr-defined]
    assert (42,) not in grafted.children[1].positions()
