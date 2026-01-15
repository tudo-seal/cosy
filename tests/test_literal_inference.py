# test for corrrect literal inference in the presence of multiple parameters

from collections.abc import Callable

import pytest

from cosy.core.specification_builder import SpecificationBuilder
from cosy.core.synthesizer import Synthesizer
from cosy.core.tree import Tree
from cosy.core.types import Arrow, Constructor, Group, Intersection, Literal, Var


def leaf_nat(x: int) -> str:
    return str(x)


def leaf_bools(y: tuple[bool]) -> str:
    return str(y)


def node(_x: int, _y: tuple[bool], argument: str) -> str:
    return f"(Node {argument})"


@pytest.fixture
def component_specifications():
    class Nat(Group):
        name = "nat"

        def __contains__(self, x):
            return x is None or (isinstance(x, int) and x >= 0)

        def __iter__(self):
            yield None  # default value, do not enumerate all natural numbers

    class Bools(Group):
        name = "bools"

        def __contains__(self, x):
            return x is None or (isinstance(x, tuple) and all(isinstance(b, bool) for b in x))

        def __iter__(self):
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
    return Constructor("p", Literal(100))


@pytest.fixture
def query_bools():
    return Constructor("q", Literal((True, True, False, True)))


T = int | Callable | None | tuple


def test_literal_inference_nat(query_nat, component_specifications) -> None:
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
