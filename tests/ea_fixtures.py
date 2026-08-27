"""Shared search spaces, tree builders and stubs for the evolutionary_algorithms tests.

Three grammars, because no single one covers what has to be covered:

* ``recursive_space`` is the primary one. It is recursive, so depth bounds are testable at all,
  and every ``C``-subtree is interchangeable with every other, so a crossover point pair always
  yields two valid offspring, which is what makes it right for the "always succeeds" invariants.
* ``asymmetric_space`` exists solely to produce the case where *exactly one* offspring is valid.
  That needs an intersection type, and a single-sorted recursive grammar cannot express it.
* ``nullary_space`` exists solely to show that a single-node term is reachable from a real space,
  which is the input that exercises the root as a mutation point.

A plain module rather than ``conftest.py``: each test file wraps what it needs in its own
function-scoped fixture, so nothing is shared across tests. That matters because ``Tree`` memoises
``positions()``/``leaf_positions()`` on the instance, and CI runs with ``--randomize --parallel``.
"""

from cosy.core import Constructor, SpecificationBuilder, Synthesizer
from cosy.core.tree import Tree
from cosy.core.types import Intersection

# ---------------------------------------------------------------------------
# G1, recursive:  S -> top(C, C);  C -> lf | un(C) | bi(C, C)
# ---------------------------------------------------------------------------


def lf() -> str:
    """Leaf combinator.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "."


def un(c: str) -> str:
    """Unary combinator.

    Args:
        c (str): The interpreted child.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"u({c})"


def bi(c: str, d: str) -> str:
    """Binary combinator.

    Args:
        c (str): The interpreted left child.
        d (str): The interpreted right child.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"b({c},{d})"


def top(c: str, d: str) -> str:
    """Root combinator, the only one producing the start symbol.

    Args:
        c (str): The interpreted left child.
        d (str): The interpreted right child.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"T({c},{d})"


RECURSIVE_SPECS = {
    lf: SpecificationBuilder().suffix(Constructor("C")),
    un: SpecificationBuilder().argument("c", Constructor("C")).suffix(Constructor("C")),
    bi: SpecificationBuilder().argument("c", Constructor("C")).argument("d", Constructor("C")).suffix(Constructor("C")),
    top: SpecificationBuilder()
    .argument("c", Constructor("C"))
    .argument("d", Constructor("C"))
    .suffix(Constructor("S")),
}
RECURSIVE_START = Constructor("S")


def recursive_space():
    """Build the primary search space.

    Returns:
        SolutionSpace: Recursive, single-sorted below the root, so every crossover point pair is
            type-compatible.
    """
    return Synthesizer(RECURSIVE_SPECS).construct_solution_space(RECURSIVE_START)


# ---------------------------------------------------------------------------
# G2, asymmetric:  f1 : A -> P & Q,  g1 : A -> P,  root2 : (P, Q) -> S
# ---------------------------------------------------------------------------


def a0() -> str:
    """Nullary combinator of sort ``A``.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "a"


def f1(z: str) -> str:
    """Unary combinator producing both ``P`` and ``Q``.

    Args:
        z (str): The interpreted child.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"f({z})"


def g1(z: str) -> str:
    """Unary combinator producing only ``P``.

    Args:
        z (str): The interpreted child.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"g({z})"


def root2(left: str, right: str) -> str:
    """Root combinator taking a ``P`` and a ``Q``.

    Args:
        left (str): The interpreted ``P`` child.
        right (str): The interpreted ``Q`` child.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"root({left},{right})"


ASYMMETRIC_SPECS = {
    a0: SpecificationBuilder().suffix(Constructor("A")),
    f1: SpecificationBuilder().argument("z", Constructor("A")).suffix(Intersection(Constructor("P"), Constructor("Q"))),
    g1: SpecificationBuilder().argument("z", Constructor("A")).suffix(Constructor("P")),
    root2: SpecificationBuilder()
    .argument("l", Constructor("P"))
    .argument("r", Constructor("Q"))
    .suffix(Constructor("S")),
}
ASYMMETRIC_START = Constructor("S")


def asymmetric_space():
    """Build the space in which exactly one of two offspring can be invalid.

    Returns:
        SolutionSpace: ``g1`` fits the left argument of ``root2`` but not the right one, so
            swapping it into the right position produces a term outside the space.
    """
    return Synthesizer(ASYMMETRIC_SPECS).construct_solution_space(ASYMMETRIC_START)


# ---------------------------------------------------------------------------
# G3, nullary:  A -> a2 | b2 | h1(A)
# ---------------------------------------------------------------------------


def a2() -> str:
    """First nullary combinator.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "a"


def b2() -> str:
    """Second nullary combinator.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "b"


def h1(x: str) -> str:
    """Unary combinator.

    Args:
        x (str): The interpreted child.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"h({x})"


NULLARY_SPECS = {
    a2: SpecificationBuilder().suffix(Constructor("A")),
    b2: SpecificationBuilder().suffix(Constructor("A")),
    h1: SpecificationBuilder().argument("x", Constructor("A")).suffix(Constructor("A")),
}
NULLARY_START = Constructor("A")


def nullary_space():
    """Build the smallest space whose start symbol admits a single-node term.

    Returns:
        SolutionSpace: Sampling from it can return a leaf, so a single-node individual is
            reachable.
    """
    return Synthesizer(NULLARY_SPECS).construct_solution_space(NULLARY_START).prune()


# ---------------------------------------------------------------------------
# Deterministic tree builders for G1
# ---------------------------------------------------------------------------


def leaf_c() -> Tree:
    """Build a single-node ``C``-term.

    Returns:
        Tree: A single-node term. Its only position is its root, so it is an ordinary input for
            ``ResolutionMutation`` and yields the empty batch from ``SubtreeSwap``.
    """
    return Tree(lf, ())


def chain(n: int) -> Tree:
    """Build ``un`` applied ``n`` times to a leaf.

    Args:
        n (int): Number of applications.

    Returns:
        Tree: A ``C``-subtree of depth ``n``.
    """
    tree = leaf_c()
    for _ in range(n):
        tree = Tree(un, (tree,))
    return tree


def parent(left: int, right: int) -> Tree:
    """Build the ``S``-term ``top(un^left(lf), un^right(lf))``.

    Args:
        left (int): Chain length below the left argument.
        right (int): Chain length below the right argument.

    Returns:
        Tree: A term of depth ``max(left, right) + 1``.
    """
    return Tree(top, (chain(left), chain(right)))


def rendered(tree: Tree) -> str:
    """Render a tree through its own interpretation.

    Args:
        tree (Tree): The tree to render.

    Returns:
        str: A structural fingerprint. It was introduced because ``replace_subtree_at`` used to
            leave ``size`` and ``_hash`` stale, so comparing an offspring with ``==`` failed for a
            reason unrelated to the operator under test. The replacement is rebuilt through
            ``__init__`` now, so that is no longer true and ``==`` is sound again. Rendering is
            kept because it makes a failing assertion readable.
    """
    return tree.interpret()


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class CountingSpace:
    """Wrap a real ``SolutionSpace`` and record membership tests by object identity.

    Identity, not equality: the point is to see the *same object* tested twice, which equality
    cannot distinguish from two structurally identical trees.
    """

    def __init__(self, inner) -> None:
        """Wrap a space.

        Args:
            inner (SolutionSpace): The space to delegate to.
        """
        self._inner = inner
        self.calls: list = []

    def contains_tree(self, start, tree, interpretation=None):
        """Delegate the membership test and record it.

        Args:
            start: The start symbol.
            tree (Tree): The tree being tested.
            interpretation: Passed through. (Default value = None)

        Returns:
            bool: The wrapped space's verdict.
        """
        result = self._inner.contains_tree(start, tree, interpretation)
        self.calls.append((id(tree), result))
        return result

    def depth_first_resolution(self, *args, **kwargs):
        """Delegate the search unchanged.

        Args:
            *args: Passed through.
            **kwargs: Passed through.

        Returns:
            Iterable[Tree]: The wrapped space's stream.
        """
        return self._inner.depth_first_resolution(*args, **kwargs)

    def duplicate_calls(self) -> list[int]:
        """Return the object ids that were tested more than once.

        Returns:
            list[int]: One entry per object tested at least twice.
        """
        seen: dict[int, int] = {}
        for identity, _ in self.calls:
            seen[identity] = seen.get(identity, 0) + 1
        return [identity for identity, count in seen.items() if count > 1]


__all__ = [
    "ASYMMETRIC_START",
    "NULLARY_START",
    "RECURSIVE_START",
    "CountingSpace",
    "a0",
    "a2",
    "asymmetric_space",
    "b2",
    "bi",
    "chain",
    "f1",
    "g1",
    "h1",
    "leaf_c",
    "lf",
    "nullary_space",
    "parent",
    "recursive_space",
    "rendered",
    "root2",
    "top",
    "un",
]
