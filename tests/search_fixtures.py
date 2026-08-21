"""Reference solution spaces shared by the search tests.

Small enough to enumerate exhaustively, so expected answers are computed rather than asserted.

* :func:`expression_space`: a unary and a binary combinator over one sort, so terms of one size
  differ in shape.
* :func:`constrained_space`: an external predicate couples two argument positions, so completing
  them independently produces terms the space does not contain.

A plain module rather than a ``conftest.py``: ``Tree`` memoises its positions on the instance and
the suite runs randomized and in parallel, so a shared instance would let tests observe each
other's cache state.
"""

from collections.abc import Mapping
from typing import Any

from cosy.core import Constructor, SpecificationBuilder, Synthesizer
from cosy.core.types import DataGroup

# ---------------------------------------------------------------------------
# E -- expressions:  E -> lit | neg(E) | add(E, E)
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
# Pair -- coupled arguments:  W -> zero | one | wrap(W) ;  Pair -> pair(W, W), left != right
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
