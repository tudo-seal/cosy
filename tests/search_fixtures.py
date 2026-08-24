"""Reference solution spaces shared by the search tests.

Small enough to enumerate exhaustively, so expected answers are computed rather than asserted.

* ``expression_space``: a unary and a binary combinator over one sort, so terms of one size
  differ in shape.
* ``constrained_space``: an external predicate couples two argument positions, so completing
  them independently produces terms the space does not contain.

A plain module rather than a ``conftest.py``: ``Tree`` memoizes its positions on the instance and
the suite runs randomized and in parallel, so a shared instance would let tests observe each
other's cache state.
"""

from collections.abc import Mapping
from typing import Any

from cosy.core import Constructor, SpecificationBuilder, Synthesizer
from cosy.core.types import Arrow, DataGroup, Intersection

# ---------------------------------------------------------------------------
# Expressions: E -> lit | neg(E) | add(E, E)
# ---------------------------------------------------------------------------

EXPR = Constructor("E")


def lit() -> str:
    """Build the literal.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "x"


def neg(inner: str) -> str:
    """Negate an expression.

    Args:
        inner (str): The interpreted operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"-{inner}"


def add(left: str, right: str) -> str:
    """Add two expressions.

    Args:
        left (str): The interpreted left operand.
        right (str): The interpreted right operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"({left}+{right})"


def expression_space():
    """Build the expression space.

    A unary and a binary combinator, so terms of one size differ in shape.

    Returns:
        SolutionSpace: The space, started at ``E``.
    """
    specs = {
        lit: SpecificationBuilder().suffix(EXPR),
        neg: SpecificationBuilder().argument("inner", EXPR).suffix(EXPR),
        add: SpecificationBuilder().argument("left", EXPR).argument("right", EXPR).suffix(EXPR),
    }
    return Synthesizer(specs).construct_solution_space(EXPR)


# ---------------------------------------------------------------------------
# Coupled arguments: W -> zero | one | wrap(W),  Pair -> pair(W, W), left != right
# ---------------------------------------------------------------------------

WORD = Constructor("W")
PAIR = Constructor("Pair")


def zero() -> str:
    """Build the first word.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "0"


def one() -> str:
    """Build the second word.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "1"


def wrap(inner: str) -> str:
    """Wrap a word. Makes ``W`` recursive, so a depth bound bites.

    Args:
        inner (str): The interpreted operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"w({inner})"


def pair(left: str, right: str) -> str:
    """Pair two words.

    Args:
        left (str): The interpreted left word.
        right (str): The interpreted right word.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"<{left},{right}>"


def different(substitution: Mapping[str, Any]) -> bool:
    """Decide whether the two paired words differ.

    Couples the two argument positions: what completes one depends on what stands in the other.

    Args:
        substitution (Mapping[str, Any]): The chosen arguments of the clause, by variable name.

    Returns:
        bool: True when the two words are different terms.
    """
    return bool(substitution["left"] != substitution["right"])


def constrained_space():
    """Build the space whose predicate couples two argument positions.

    Returns:
        SolutionSpace: The space, started at ``Pair``.
    """
    specs = {
        zero: SpecificationBuilder().suffix(WORD),
        one: SpecificationBuilder().suffix(WORD),
        wrap: SpecificationBuilder().argument("inner", WORD).suffix(WORD),
        pair: SpecificationBuilder().argument("left", WORD).argument("right", WORD).constraint(different).suffix(PAIR),
    }
    return Synthesizer(specs).construct_solution_space(PAIR)


# ---------------------------------------------------------------------------
# N -- a constant argument:  N -> val(v) | plus(v, N),  v in {0, 1}
# ---------------------------------------------------------------------------

NUM = Constructor("N")


def val(v: int) -> str:
    """Build the literal of the numeric space.

    Args:
        v (int): The chosen constant.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return str(v)


def plus(v: int, inner: str) -> str:
    """Add a constant to a numeric term.

    Args:
        v (int): The chosen constant.
        inner (str): The interpreted operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"({v}+{inner})"


def literal_space():
    """Build the space whose clauses carry a constant argument next to a non-terminal one.

    The two kinds of argument are recorded in different places of a goal. A constant argument is
    grounded from the start and never becomes a subgoal, so a goal of this space carries a
    grounded position that has no entry in ``constructors``. The spaces above have no constant
    argument, and on them a goal's ``grounded`` map holds nothing a walk over ``constructors``
    could not rebuild.

    Returns:
        SolutionSpace: The space, started at ``N``.
    """
    digits = DataGroup("int", [0, 1])
    specs = {
        val: SpecificationBuilder().parameter("v", digits).suffix(NUM),
        plus: SpecificationBuilder().parameter("v", digits).argument("inner", NUM).suffix(NUM),
    }
    return Synthesizer(specs).construct_solution_space(NUM)


def wide_first_space():
    """Build the expression space with its clauses declared widest first.

    ``expression_space`` declares ``lit`` before ``neg`` before ``add``, so its program order
    already puts the narrowest clause in front and a clause order that does the same cannot be
    told from it. Here the declaration is reversed, which separates the two.

    Returns:
        SolutionSpace: The space, started at ``EXPR``.
    """
    specs = {
        add: SpecificationBuilder().argument("left", EXPR).argument("right", EXPR).suffix(EXPR),
        neg: SpecificationBuilder().argument("inner", EXPR).suffix(EXPR),
        lit: SpecificationBuilder().suffix(EXPR),
    }
    return Synthesizer(specs).construct_solution_space(EXPR)


# ---------------------------------------------------------------------------
# Width, equal width: Width -> c0 | u1(Width) | u2(Width)
# ---------------------------------------------------------------------------

WIDTH = Constructor("Width")


def c0() -> str:
    """Build the nullary clause of the equal-width space.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "."


def u1(inner: str) -> str:
    """Wrap a term with the first of the two unary clauses.

    Args:
        inner (str): The interpreted operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"u1({inner})"


def u2(inner: str) -> str:
    """Wrap a term with the second of the two unary clauses.

    Args:
        inner (str): The interpreted operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"u2({inner})"


def equal_width_space():
    """Build the space whose two unary clauses open equally many holes.

    The default clause order sorts by the number of holes a clause opens and is stable, so it
    keeps the order it was handed between clauses of equal width. ``u1`` and ``u2`` are that pair,
    which is what makes an ordering effect observable here at all.

    Returns:
        SolutionSpace: The space, started at ``Width``.
    """
    specs = {
        c0: SpecificationBuilder().suffix(WIDTH),
        u1: SpecificationBuilder().argument("inner", WIDTH).suffix(WIDTH),
        u2: SpecificationBuilder().argument("inner", WIDTH).suffix(WIDTH),
    }
    return Synthesizer(specs).construct_solution_space(WIDTH)


# ---------------------------------------------------------------------------
# S, two immediate successes: S -> s1 | s2
# ---------------------------------------------------------------------------

START = Constructor("S")


def s1() -> str:
    """Build the first of the two immediately succeeding clauses.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "s1"


def s2() -> str:
    """Build the second of the two immediately succeeding clauses.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "s2"


def nullary_start_space():
    """Build the space whose start symbol has two immediately succeeding clauses.

    Both initial goals are success nodes, so this space observes what a search rule does with
    them before any expansion has happened.

    Returns:
        SolutionSpace: The space, started at ``S``.
    """
    specs = {s1: SpecificationBuilder().suffix(START), s2: SpecificationBuilder().suffix(START)}
    return Synthesizer(specs).construct_solution_space(START)


A = Constructor("A")
B = Constructor("B")
C = Constructor("C")


# ---------------------------------------------------------------------------
# A space whose target is reached by two clauses sharing their terminal
# ---------------------------------------------------------------------------


def c_ab() -> str:
    """Build the constant that inhabits both argument sorts.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "c"


def a_only() -> str:
    """Build the constant that inhabits ``A`` alone.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "a"


def b_only() -> str:
    """Build the constant that inhabits ``B`` alone.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "b"


def wrap_c(x: str) -> str:
    """Build the unary combinator with two paths onto ``C``.

    Args:
        x (str): The interpreted child.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"f({x})"


def multi_path_space():
    """Build the space in which one position is reached by two clauses.

    ``wrap_c : (A -> C) & (B -> C)`` has two paths of arity one onto ``C``, so the inhabitation
    emits two clauses for ``C`` that share their terminal and differ in their argument sort. The
    completions of a variable under ``wrap_c`` are the union: ``c_ab`` and ``a_only`` through ``A``,
    ``c_ab`` and ``b_only`` through ``B``.

    Returns:
        SolutionSpace: The space, started at ``C``.
    """
    specs = {
        c_ab: Intersection(A, B),
        a_only: A,
        b_only: B,
        wrap_c: Intersection(Arrow(A, C), Arrow(B, C)),
    }
    return Synthesizer(specs).construct_solution_space(C)


D = Constructor("D")


def top_d(inner: str) -> str:
    """Put one more level above ``C``.

    Args:
        inner (str): The interpreted operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"top({inner})"


def disjoint_multi_path_space(start=None):
    """Build the multi-path space in which the two clauses share no completion.

    ``multi_path_space`` cannot show which clause a search explored first, because ``c_ab``
    inhabits both argument sorts and is reachable either way. Dropping it makes the two clauses
    disjoint: the ``A`` clause reaches only ``a_only``, the ``B`` clause only ``b_only``, so the
    first streamed term names the clause that was taken.

    ``top_d`` sits above ``C`` so that a hole at ``(0, 0)`` reaches the ambiguous expansion
    through the walk of ``goal_from_tree`` rather than through its initial goals.

    Args:
        start (NT | None): The non-terminal to start from, ``C`` or ``D``. (Default value = None,
            which starts at ``D``)

    Returns:
        SolutionSpace: The space.
    """
    specs = {
        a_only: A,
        b_only: B,
        wrap_c: Intersection(Arrow(A, C), Arrow(B, C)),
        top_d: Arrow(C, D),
    }
    return Synthesizer(specs).construct_solution_space(D if start is None else start)
