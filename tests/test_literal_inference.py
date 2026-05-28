"""_summary_."""
# test for corrrect literal inference in the presence of multiple parameters

from collections.abc import Callable

import pytest

from cosy.core.specification_builder import SpecificationBuilder
from cosy.core.synthesizer import Synthesizer
from cosy.core.tree import Tree
from cosy.core.types import Arrow, Constructor, Group, Intersection, Literal, Var


def leaf_nat(x: int) -> str:
    """_summary_.

    Args:
        x (int): _description_

    Returns:
        str: _description_
    """
    return str(x)


def leaf_bools(y: tuple[bool]) -> str:
    """_summary_.

    Args:
        y (tuple[bool]): _description_

    Returns:
        str: _description_
    """
    return str(y)


def node(_x: int, _y: tuple[bool], argument: str) -> str:
    """_summary_.

    Args:
        argument (str): _description_

    Returns:
        str: _description_
    """
    return f"(Node {argument})"


@pytest.fixture
def component_specifications():
    """_summary_.

    Returns:
        _type_: _description_
    """

    class Nat(Group):
        """_summary_."""

        name = "nat"

        def __contains__(self, x):
            """_summary_.

            Args:
                x (_type_): _description_

            Returns:
                _type_: _description_
            """
            return x is None or (isinstance(x, int) and x >= 0)

        def __iter__(self):
            """_summary_.

            Yields:
                _type_: _description_
            """
            yield None  # default value, do not enumerate all natural numbers

    class Bools(Group):
        """_summary_."""

        name = "bools"

        def __contains__(self, x):
            """_summary_.

            Args:
                x (_type_): _description_

            Returns:
                _type_: _description_
            """
            return x is None or (isinstance(x, tuple) and all(isinstance(b, bool) for b in x))

        def __iter__(self):
            """_summary_.

            Yields:
                _type_: _description_
            """
            yield None  # default value, do not enumerate all boolean tuples

    return {
        leaf_nat: SpecificationBuilder().parameter("x", Nat()).suffix(Constructor("a", Var("x"))),
        leaf_bools: SpecificationBuilder().parameter("y", Bools()).suffix(Constructor("b", Var("y"))),
        node: SpecificationBuilder()
        .parameter("x", Nat())
        .parameter("y", Bools())
        .suffix(
            Intersection(
                Arrow(Constructor("a", Var("x")), Constructor("p", Var("x"))),
                Arrow(Constructor("b", Var("y")), Constructor("q", Var("y"))),
            )
        ),
    }


@pytest.fixture
def query_nat():
    """_summary_.

    Returns:
        _type_: _description_
    """
    return Constructor("p", Literal(100))


@pytest.fixture
def query_bools():
    """_summary_.

    Returns:
        _type_: _description_
    """
    return Constructor("q", Literal((True, True, False, True)))


T = int | Callable | None | tuple


def test_literal_inference_nat(query_nat, component_specifications) -> None:
    """_summary_.

    Args:
        query_nat (_type_): _description_
        component_specifications (_type_): _description_
    """
    solution_space = Synthesizer(component_specifications).construct_solution_space(query_nat)

    result = Tree[T](
        node,
        [
            Tree[T](100),
            Tree[T](None),
            Tree[T](leaf_nat, [Tree[T](100)]),
        ],
    )

    assert solution_space.contains_tree(query_nat, result)


def test_literal_inference_bools(query_bools, component_specifications) -> None:
    """_summary_.

    Args:
        query_bools (_type_): _description_
        component_specifications (_type_): _description_
    """
    solution_space = Synthesizer(component_specifications).construct_solution_space(query_bools)

    result = Tree[T](
        node,
        [
            Tree[T](None),
            Tree[T]((True, True, False, True)),
            Tree[T](leaf_bools, [Tree[T]((True, True, False, True))]),
        ],
    )

    assert solution_space.contains_tree(query_bools, result)
