"""Reproducibility of what a ``SolutionSpace`` hands back.

Every test here fixes an order that used to be an implementation accident: the iteration order of a
``set`` whose elements are hashed by object address, or by a string hash that the interpreter
chooses anew at startup. The two consequences need different setups. Within one process, terminals
that are function objects -- what a CoSy combinator normally is -- hash by identity, so building
the same grammar over and over exposes an address-ordered set. Across processes,
``PYTHONHASHSEED`` moves the hashes of everything else, which only a fresh interpreter can vary.

Where a sequence is asserted it is written out rather than checked for a property of itself: an
order that is stable but wrong would satisfy "all runs agree" without satisfying anything else.
"""

import os
import random
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from cosy.core.solution_space import NonTerminalArgument, SolutionSpace
from cosy.core.tree import Tree
from tests._determinism_grammars import A, B, C, mixed_width_space


@pytest.fixture
def space() -> SolutionSpace[str, str, None]:
    """Return a fresh instance of the shared grammar.

    Returns:
        SolutionSpace[str, str, None]: The grammar.
    """
    return mixed_width_space()


# ---------------------------------------------------------------------------
# Running a child interpreter: only a fresh process can vary PYTHONHASHSEED
# ---------------------------------------------------------------------------


# The child bodies import the grammar they need themselves, so that they depend on nothing but
# ``cosy`` -- see ``tests/_determinism_grammars``.
_CHILD_PREAMBLE = f"import sys\nsys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})\n"

_HASH_SEEDS = ("0", "1", "7", "42", "123")


def _printed_across_hash_seeds(body: str) -> set[str]:
    """Run ``body`` in fresh interpreters that differ only in their hash seed.

    A process fixes its hash seed at startup, so a test that stays inside the current interpreter
    cannot tell a stable order from one that is merely stable for this run.

    Args:
        body (str): Statements to run after the preamble. Whatever they print is what is compared.

    Returns:
        set[str]: The distinct outputs. Reproducibility means there is exactly one.
    """
    printed = set()
    for hash_seed in _HASH_SEEDS:
        child = subprocess.run(
            [sys.executable, "-c", _CHILD_PREAMBLE + body],
            capture_output=True,
            check=False,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            text=True,
        )
        if child.returncode != 0:
            pytest.fail(
                f"the child interpreter did not run to completion under PYTHONHASHSEED={hash_seed}; "
                f"this is an environment failure, not a difference in order:\n{child.stderr}"
            )
        printed.add(child.stdout.strip())
    return printed


# ---------------------------------------------------------------------------
# The order of the goals a derivation step produces
# ---------------------------------------------------------------------------


def _equally_wide_space() -> SolutionSpace[str, str, None]:
    """Return ``C -> a(C) | b(C) | c(C) | lf``, three rules of the same width.

    Returns:
        SolutionSpace[str, str, None]: The grammar. Sorting by the number of subgoals decides
            nothing between ``a``, ``b`` and ``c``, so what is left is the order the rules were
            added in -- at the root of a derivation as well as below it.
    """
    space: SolutionSpace[str, str, None] = SolutionSpace()
    for terminal in ("a", "b", "c"):
        space.add_rule("C", terminal, (C,), ())
    space.add_rule("C", "lf", (), ())
    return space


def test_depth_first_search_takes_the_fewest_subgoals_first(space: SolutionSpace[str, str, None]) -> None:
    """Bounded to depth one, depth-first search returns the four terms in order of rule width.

    This is the whole ordering claim on the shortest possible input: the sorted goals must actually
    be assigned (dropping the assignment yields ``un, tri, bi``, the order the rules were added in)
    and they must be reversed before the push, because ``extendleft`` inserts in reverse (pushing
    them unreversed expands the widest rule first, ``tri, bi, un``).
    """
    terms = [str(tree) for tree in space.depth_first_resolution("C", max_depth=1)]
    assert terms == ["lf", "un lf", "bi lf lf", "tri lf lf lf"]


def test_breadth_first_search_takes_the_fewest_subgoals_first(space: SolutionSpace[str, str, None]) -> None:
    """Breadth-first search reaches the terms by size, and within one size by rule width.

    A term of ``n`` nodes needs ``n`` derivation steps whatever the queue looks like, so the size is
    not what the ordering decides -- the sequence within one size is. Seven terms are what it takes
    for every such change to show somewhere: reversing the sorted goals the way the depth-first push
    has to already moves the third term, but leaving them unassigned moves nothing before the
    seventh.
    """
    terms = [str(tree) for tree in space.breadth_first_resolution("C", max_count=7, max_depth=3)]
    assert terms == [
        "lf",
        "un lf",
        "un (un lf)",
        "bi lf lf",
        "un (un (un lf))",
        "un (bi lf lf)",
        "bi lf (un lf)",
    ]


def test_equally_wide_rules_are_taken_in_the_order_they_were_added() -> None:
    """Sorting by width decides nothing between equally wide rules, so the rule order has to.

    That has to hold at the root of a derivation as well as below it, and in both searches. The
    initial goals used to be collected by prepending, and ``sorted`` is stable, so the root broke
    ties in reverse rule order while every deeper level broke them in rule order: ``c`` first at
    the top, ``a`` first below it. Neither search above can see this: the rules of the grammar
    they use all have a different width, so the sort alone decides their order.
    """
    depth_first = [str(tree) for tree in _equally_wide_space().depth_first_resolution("C", max_count=8, max_depth=2)]
    breadth_first = [str(tree) for tree in _equally_wide_space().breadth_first_resolution("C", max_count=4)]

    assert depth_first == ["lf", "a lf", "a (a lf)", "a (b lf)", "a (c lf)", "b lf", "b (a lf)", "b (b lf)"]
    assert breadth_first == ["lf", "a lf", "b lf", "c lf"]


# ---------------------------------------------------------------------------
# Terminals that hash by identity: the shape a real repository has
# ---------------------------------------------------------------------------


# Two rounds agree with each other far more often than not, so a couple of them would let an
# address-ordered set through. Against the unfixed implementation the worst case here needed seven
# before the order moved; thirty leaves room to spare and still costs milliseconds.
_ROUNDS = 30


def _combinator(name: str) -> Callable[..., str]:
    """Return a fresh function object under the given name.

    A CoSy combinator is a plain function, and a function hashes by identity, so every call
    produces a terminal that hashes differently from the one the call before it produced.

    Args:
        name (str): The name to report the combinator under.

    Returns:
        Callable[..., str]: The combinator. Applying it to the renderings of its arguments renders
            the term rooted at it, which is what ``_render`` does with it; the ``__name__`` it
            carries is what a repr of the bare object shows.
    """

    def combinator(*arguments: str) -> str:
        return f"{name}({', '.join(arguments)})" if arguments else name

    combinator.__name__ = name
    return combinator


def _function_terminal_space() -> SolutionSpace[str, Callable[..., str], None]:
    """Return ``S -> top(A, B)`` with ``A -> a1 | a2(A)`` and ``B -> b1 | b2(B) | b3(A)``, over functions.

    Returns:
        SolutionSpace[str, Callable[..., str], None]: The grammar. Its terminals are plain function
            objects -- what a CoSy combinator normally is -- so they differ between calls within a
            single process and no child interpreter is needed to move their hashes.
    """
    top, a1, a2, b1, b2, b3 = (_combinator(name) for name in ("top", "a1", "a2", "b1", "b2", "b3"))
    space: SolutionSpace[str, Callable[..., str], None] = SolutionSpace()
    space.add_rule("S", top, (A, B), ())
    space.add_rule("A", a1, (), ())
    space.add_rule("A", a2, (A,), ())
    space.add_rule("B", b1, (), ())
    space.add_rule("B", b2, (B,), ())
    space.add_rule("B", b3, (A,), ())
    return space


def _render(tree: Tree[Callable[..., str]]) -> str:
    """Render a term by applying its combinators to the renderings of their arguments.

    Args:
        tree (Tree[Callable[..., str]]): The term to render.

    Returns:
        str: The term. ``str(tree)`` would render a function terminal as ``<function a1 at 0x...>``,
            which changes from run to run even when the enumeration does not.
    """
    return tree.root(*(_render(child) for child in tree.children))


# ---------------------------------------------------------------------------
# Sampling: the same seed has to give the same term
# ---------------------------------------------------------------------------


def test_a_seeded_sample_repeats() -> None:
    """A seeded sample has to be repeatable, otherwise no experiment on this framework is.

    The goals of a derivation step were collected in a set, and a ``Goal`` is hashed by its address,
    so the set handed them back in allocation order. ``sample_tree`` shuffles that order with the
    seeded generator, which made the seed permute an already random list. Combinators that are
    functions show this within one process; a set of goals with a value-based hash would instead
    order them by string hashes, which only fresh interpreters vary.
    """
    sampled = set()
    for _ in range(_ROUNDS):
        _ = [object() for _ in range(50)]  # move the addresses the next grammar will be built at
        tree = _function_terminal_space().sample_tree("S", max_depth=4, rng=random.Random(0))
        assert tree is not None
        sampled.add(_render(tree))

    assert sampled == {"top(a1, b2(b3(a2(a1))))"}, f"one seed produced {len(sampled)} different terms: {sampled}"

    printed = _printed_across_hash_seeds(
        "import random\n"
        "from tests._determinism_grammars import mixed_width_space\n"
        "space = mixed_width_space()\n"
        'print([str(space.sample_tree("S", max_depth=4, rng=random.Random(seed))) for seed in range(3)])\n'
    )
    assert printed == {"['top lf lf', 'top (un lf) lf', 'top (un (bi (tri lf lf lf) (un lf))) lf']"}


# ---------------------------------------------------------------------------
# Reading the rules
# ---------------------------------------------------------------------------


def _dangling_reference_space() -> SolutionSpace[str, str, None]:
    """Return ``S -> f(Missing) | g`` where ``Missing`` has no rules of its own.

    Returns:
        SolutionSpace[str, str, None]: The grammar. Every read path of the solution space runs
            into ``Missing`` and has to leave the grammar alone.
    """
    space: SolutionSpace[str, str, None] = SolutionSpace()
    space.add_rule("S", "f", (NonTerminalArgument(None, "Missing"),), ())
    space.add_rule("S", "g", (), ())
    return space


def test_reading_the_rules_never_creates_a_nonterminal(space: SolutionSpace[str, str, None]) -> None:
    """Reading is not a mutation. Reads went through ``defaultdict.__getitem__``, which inserts."""
    before = set(space.nonterminals())

    assert len(space["C"]) == 4
    with pytest.raises(KeyError):
        _ = space["NoSuchNonTerminal"]
    assert space.get("NoSuchNonTerminal") is None
    assert "NoSuchNonTerminal" not in space

    assert set(space.nonterminals()) == before


def test_searching_never_creates_a_nonterminal() -> None:
    """The searches read the rules of every non-terminal a rule refers to, including unknown ones.

    Those reads went through ``defaultdict.__getitem__`` as well, so running a search over a
    grammar that refers to a non-terminal without rules used to add that non-terminal to it. The
    tree handed to ``contains_tree`` has to match the dangling rule for one step, otherwise the
    check never descends into the non-terminal it is about.

    Starting from ``Missing`` itself is the one case whose answer is observable at all. The other
    entry points have nothing to report either way, but ``contains_tree`` has to answer False
    rather than treat the unknown non-terminal as satisfied.
    """
    space = _dangling_reference_space()

    assert [str(tree) for tree in space.depth_first_resolution("S", max_depth=3)] == ["g"]
    assert [str(tree) for tree in space.breadth_first_resolution("S", max_depth=3)] == ["g"]
    assert [str(tree) for tree in space.enumerate_trees("S")] == ["g"]
    assert space.contains_tree("S", Tree("g")) is True
    assert space.contains_tree("S", Tree("f", (Tree("x"),))) is False
    assert space.contains_tree("Missing", Tree("x")) is False

    assert space.nonterminals() == ("S",)


def test_the_rules_of_a_known_nonterminal_are_the_stored_ones(space: SolutionSpace[str, str, None]) -> None:
    """``__getitem__`` hands out the live deque, so a caller that appends to it adds a rule.

    Refusing the unknown non-terminal instead of inventing an empty deque for it is what makes that
    unambiguous: there is no key for which the same append is silently discarded.
    """
    space["C"].append(space["C"][0])

    assert len(space["C"]) == 5


def test_nonterminals_is_a_snapshot(space: SolutionSpace[str, str, None]) -> None:
    """What ``nonterminals`` reports must not change under a caller who is still iterating it.

    It used to be the live ``keys()`` view of the rule mapping. Two claims ride along that a
    snapshot alone would not pin: it is in the order the non-terminals were added rather than in
    any sorted order, and ``in space`` still sees one that was added after it was taken.
    """
    reported = space.nonterminals()

    space.add_rule("D", "d", (), ())

    assert list(reported) == ["S", "C"]
    assert "D" in space
