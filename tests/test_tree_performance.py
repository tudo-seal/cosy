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

``interpret`` had a fourth cost: it asked ``inspect.signature`` for the parameters of every
combinator *occurrence*, and that call dominated the evaluation of a term.

A fifth cost is paid in interpreter recursion rather than in time.  Comparing two terms compared
their children as tuples, and rendering a term descended into every child, so both spent one level
of recursion per level of the term.  That bounded the depth a term could have at a fixed fraction
of the interpreter's recursion limit, and terms grow past it.  Every ``dict`` and every ``set``
keyed on terms carried the same bound, the fitness cache of the evolutionary algorithms among
them.  The tests at the end do not count that cost the way the ones below count calls.  They work
on terms twice the recursion limit deep, which is the statement about the bound that does not
depend on which interpreter runs it.

The tests below count operations instead of measuring time.  A counted operation says the same
thing on a loaded machine as on an idle one, whereas a wall-clock bound would only say how busy
the machine running the suite happens to be.  Where the claim is about growth rather than about a
single call -- the quadratic filter above -- the count comes from ``sys.setprofile``, which reports
every call the interpreter makes and so measures work done rather than time taken.
"""

import math
import sys
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from functools import lru_cache
from inspect import Signature, signature
from types import FrameType
from typing import Any

import pytest

from cosy.core.tree import Path, Tree, _parameters_cached, _parameters_of


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


def cycling_chain(labels: tuple[str, ...], depth: int) -> Tree[str]:
    """Build a unary chain whose labels cycle through ``labels``.

    Args:
        labels (tuple[str, ...]): The labels to cycle through, innermost first.
        depth (int): The number of unary nodes above the leaf.

    Returns:
        Tree[str]: The chain.  Neighboring nodes carry different labels, so evaluating it asks for
            the combinators in rotation rather than in blocks.
    """
    node: Tree[str] = Tree("leaf", ())
    for index in range(depth):
        node = Tree(labels[index % len(labels)], (node,))
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
    assembled by crossover costs the depth of the crossover point, not the size of the parent.
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


# ---------------------------------------------------------------------------
# interpret: one signature per combinator, not one per occurrence
# ---------------------------------------------------------------------------


def counting_signatures(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record every callable whose signature is asked for, against an empty memo.

    The memo is module state that the whole process shares, so a test that read the shipped one
    would depend on what ran before it, and one that filled it would leave its closures behind.

    The bound is read off the shipped memo rather than repeated here, so the test still asks about
    what is shipped while the memo the rest of the process shares is neither emptied nor left
    holding this test's closures.

    Args:
        monkeypatch (pytest.MonkeyPatch): Used to stand an empty memo in for the shared one.

    Returns:
        list[Any]: The list the recorded callables are appended to.
    """
    calls: list[Any] = []

    def recording(obj: Any) -> Signature:
        """Record the object and delegate.

        Args:
            obj (Any): The callable whose signature is asked for.

        Returns:
            Signature: The signature of ``obj``.
        """
        calls.append(obj)
        return signature(obj)

    monkeypatch.setattr("cosy.core.tree.signature", recording)
    monkeypatch.setattr("cosy.core.tree._parameters_cached", empty_memo())
    return calls


def empty_memo():
    """Return an empty memo of the shipped size, wrapping the shipped function.

    Returns:
        Any: The memo, ready to stand in for ``_parameters_cached``.
    """
    return lru_cache(maxsize=_parameters_cached.cache_parameters()["maxsize"])(_parameters_cached.__wrapped__)


def test_interpret_asks_once_per_combinator_even_when_they_alternate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A signature is asked for once per combinator, not once per node that carries it.

    A chain over a single combinator is served by a memo of any size at all, so it says nothing.
    Here the labels rotate, which is the shape a real term has: every step asks for a different
    combinator and comes back to the first one five steps later.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
    """
    calls = counting_signatures(monkeypatch)

    labels = ("a", "b", "c", "d", "e")
    interpretation: dict[str, Any] = {
        label: (lambda value, step=step: value + step) for step, label in enumerate(labels, start=1)
    }
    interpretation["leaf"] = lambda: 0

    assert cycling_chain(labels, 300).interpret(interpretation) == 900

    assert len(calls) == len(interpretation), (
        f"asked for {len(calls)} signatures over {len(interpretation)} combinators"
    )


def test_a_whole_algebra_of_repository_size_stays_in_the_memo(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bound holds a whole algebra, so a round trip through one keeps hitting.

    An LRU one entry short of its working set evicts exactly the entry the next lookup asks for,
    so it does not degrade gradually but all at once.  That is what the bound is generous for.
    Sixty-five combinators is well past the algebras in use here.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
    """
    calls = counting_signatures(monkeypatch)

    labels = tuple(f"c{index}" for index in range(64))
    algebra: dict[str, Any] = {label: (lambda value: value + 1) for label in labels}
    algebra["leaf"] = lambda: 0

    assert cycling_chain(labels, 3 * len(labels)).interpret(algebra) == 3 * len(labels)

    assert len(calls) == len(algebra)


def test_a_fresh_algebra_per_evaluation_does_not_grow_the_memo(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller that rebuilds its algebra for every evaluation must not grow the memo without end.

    The pattern is the ordinary one: an algebra assembled inside the call that evaluates the term,
    so every set of callables is used once and is unreachable afterwards.  Unbounded, the memo
    would keep all of them, together with whatever they close over.  Bounded, it holds what the
    last evaluations needed and forgets the rest.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
    """
    memo = empty_memo()
    monkeypatch.setattr("cosy.core.tree._parameters_cached", memo)
    bound = memo.cache_parameters()["maxsize"]

    tree: Tree[str] = Tree("leaf", ())
    for _ in range(20):
        tree = Tree("step", (tree,))
    for _ in range(2 * bound):
        assert tree.interpret({"leaf": lambda: 0, "step": lambda value: value + 1}) == 20

    assert memo.cache_info().currsize <= bound


def test_a_combinator_without_parameters_is_remembered_like_any_other(monkeypatch: pytest.MonkeyPatch) -> None:
    """A nullary combinator's answer is the empty tuple, and an empty answer is still an answer.

    Leaves are most of the nodes of a term, and their combinators take no arguments.  A memo that
    tested its stored answer for truth rather than for presence would miss on every one of them
    while looking fully effective on the inner nodes.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
    """
    calls = counting_signatures(monkeypatch)

    interpretation: dict[str, Any] = {"pair": lambda left, right: left + right, "leaf": lambda: 1}
    tree: Tree[str] = Tree("leaf", ())
    for _ in range(50):
        tree = Tree("pair", (tree, Tree("leaf", ())))

    assert tree.interpret(interpretation) == 51

    assert len(calls) == 2, f"asked for {len(calls)} signatures over 2 combinators"


@dataclass
class UncacheableAdd:
    """A combinator that cannot key the memo.

    A dataclass keeps the default ``__eq__``, which sets ``__hash__`` to ``None``.  A combinator
    written as a value object rather than as a function ends up unhashable that way, without its
    author ever thinking about caching.
    """

    def __call__(self, left: int, right: int) -> int:
        """Add two numbers.

        Args:
            left (int): The first summand.
            right (int): The second summand.

        Returns:
            int: Their sum.
        """
        return left + right


def test_interpret_accepts_a_combinator_that_cannot_be_cached() -> None:
    """An unhashable combinator is inspected directly instead of being rejected.

    This is the one failure mode the memo introduces: before it existed every callable worked, and
    a combinator that cannot be a dictionary key must not start raising ``TypeError`` from inside
    the cache.  The memo is skipped for it, not consulted and not blamed.
    """
    combinator = UncacheableAdd()
    assert type(combinator).__hash__ is None

    tree = Tree("add", (Tree("one"), Tree("two")))
    assert tree.interpret({"add": combinator, "one": lambda: 1, "two": lambda: 2}) == 3


def test_interpret_still_reports_a_combinator_without_a_signature() -> None:
    """A built-in without an introspectable signature is still a hard error.

    The memo must not turn the ``TypeError`` of ``interpret`` into a silent success or into an
    error raised by the memo itself: a failure is reported, never replaced by a value.
    """
    tree = Tree("b", (Tree("leaf", ()),))
    with pytest.raises(TypeError, match="does not expose a signature"):
        tree.interpret({"b": math.log, "leaf": lambda: 1.0})


def test_the_memo_hands_out_something_a_caller_cannot_corrupt() -> None:
    """The stored answer is a tuple, because the memo hands out the object it stored.

    A caller that appended to a list it received would change what every later evaluation of that
    combinator reads.
    """

    def combinator(left: int, right: int) -> int:
        """Add two numbers.

        Args:
            left (int): The first summand.
            right (int): The second summand.

        Returns:
            int: Their sum.
        """
        return left + right

    parameters = _parameters_of(combinator)

    assert isinstance(parameters, tuple)
    assert _parameters_of(combinator) is parameters


# ---------------------------------------------------------------------------
# Depth -- what an operation spends per level of a term
# ---------------------------------------------------------------------------


def test_two_equal_terms_built_apart_compare_equal_at_any_depth() -> None:
    """Equality does not care how deep the two terms are.

    Comparing the children compared two tuples, and that compared their elements, so a comparison
    descended one level of interpreter recursion per level of the term.  Both terms are built here
    rather than compared against themselves, because a tuple comparison settles on identity before
    it looks at anything, which is what hid the bound for as long as it was there.
    """
    depth = sys.getrecursionlimit() * 2

    assert chain(depth) == chain(depth)


def test_a_deep_term_that_differs_at_its_bottom_compares_unequal() -> None:
    """A comparison that reaches the bottom of a deep term still reports what it finds there.

    The two terms agree at every position but the last, so anything that stops early -- or that
    answers from the size alone, which is equal here -- calls them equal.
    """
    depth = sys.getrecursionlimit() * 2
    bottom = tuple(0 for _ in range(depth))
    other_leaf = chain(depth).replace_subtree_at(bottom, Tree("other"))

    assert chain(depth).size == other_leaf.size
    assert chain(depth) != other_leaf


def test_a_deep_term_is_found_in_a_dict_keyed_by_terms() -> None:
    """A term looked up in a dict is compared against the key, and the key was built elsewhere.

    This is the shape the bound was reached in.  The fitness cache of the evolutionary algorithms
    is a dict keyed on terms, and the terms it is asked about are the ones just assembled, so the
    lookup cannot settle on identity and ends in a comparison.
    """
    depth = sys.getrecursionlimit() * 2
    scored = {chain(depth): "reference"}

    assert scored[chain(depth)] == "reference"


def test_a_deep_term_renders_as_the_chain_it_is() -> None:
    """Rendering does not care how deep the term is, and writes the brackets it always wrote.

    The expected string is spelled out from the shape rather than taken from another rendering, so
    an implementation that is merely consistent with itself does not satisfy this.
    """
    depth = sys.getrecursionlimit() * 2
    expected = "f " + "(f " * (depth - 1) + "leaf" + ")" * (depth - 1)

    assert str(chain(depth)) == expected
