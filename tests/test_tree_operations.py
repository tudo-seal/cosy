# test for subtree_at and replace_subtree_at
from collections.abc import Callable

import pytest

from cosy.core.specification_builder import SpecificationBuilder
from cosy.core.synthesizer import Synthesizer
from cosy.core.tree import Tree
from cosy.core.types import DataGroup, Literal, Var


def leaf() -> str:
    return "."


def branch(depth: int, _new_depth: int, left: str, right: str) -> str:
    return f"(B {depth} {left} {right})"


@pytest.fixture
def component_specifications():
    return {
        # recursive unproductive specification
        leaf: SpecificationBuilder().suffix(Literal(0)),
        branch: SpecificationBuilder()
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

    tree = Tree[T](
        branch,
        [
            Tree(2),
            Tree(1),
            Tree(branch, [Tree(1), Tree(0), Tree(leaf), Tree(leaf)]),
            Tree(branch, [Tree(1), Tree(0), Tree(leaf), Tree(leaf)]),
        ],
    )

    assert solution_space.contains_tree(query, tree)

    tree_correct_positions = tree.positions()

    for pos in tree_correct_positions:
        subtree = tree.subtree_at(pos)
        assert tree.replace_subtree_at(pos, subtree) == tree


