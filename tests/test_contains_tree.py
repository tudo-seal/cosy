# regression test for contains_tree
from collections.abc import Callable

import pytest
from cosy.dsl import DSL
from cosy.synthesizer import Synthesizer
from cosy.tree import Tree
from cosy.types import DataGroup, Literal, Var


def leaf() -> str:
    return "."


def branch(depth: int, _new_depth: int, left: str, right: str) -> str:
    return f"(B {depth} {left} {right})"


@pytest.fixture
def component_specifications():
    return {
        # recursive unproductive specification
        leaf: DSL().suffix(Literal(0)),
        branch: DSL()
        .parameter("depth", DataGroup("int", [0, 1, 2, 3]))
        .parameter("new_depth", DataGroup("int", [0, 1, 2, 3]), lambda vs: [vs["depth"] - 1])
        .argument("left", Var("new_depth"))
        .argument("right", Var("new_depth"))
        .constraint(lambda vs: vs["left"] == vs["right"])
        .suffix(Var("depth")),
    }


@pytest.fixture
def query():
    return Literal(2)


T = int | Callable


def test_contains_tree(query, component_specifications) -> None:
    solution_space = Synthesizer(component_specifications).construct_solution_space(query)

    tree_correct = Tree[T](
        branch,
        [
            Tree(2),
            Tree(1),
            Tree(branch, [Tree(1), Tree(0), Tree(leaf), Tree(leaf)]),
            Tree(branch, [Tree(1), Tree(0), Tree(leaf), Tree(leaf)]),
        ],
    )

    # a literals 0 are wrongly set to 1
    tree_wrong_1 = Tree[T](
        branch,
        [
            Tree(2),
            Tree(1),
            Tree(branch, [Tree(1), Tree(1), Tree(leaf), Tree(leaf)]),
            Tree(branch, [Tree(1), Tree(1), Tree(leaf), Tree(leaf)]),
        ],
    )

    # a subtree is missing
    tree_wrong_2 = Tree[T](
        branch,
        [
            Tree(2),
            Tree(1),
            Tree(branch, [Tree(1), Tree(0), Tree(leaf), Tree(leaf)]),
            Tree(leaf),
        ],
    )

    assert solution_space.contains_tree(query, tree_correct)
    assert not solution_space.contains_tree(query, tree_wrong_1)
    assert not solution_space.contains_tree(query, tree_wrong_2)
