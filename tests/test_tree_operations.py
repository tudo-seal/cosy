"""_summary_."""

# test for subtree_at and replace_subtree_at
from collections.abc import Callable

import pytest

from cosy.core.specification_builder import SpecificationBuilder
from cosy.core.synthesizer import Synthesizer
from cosy.core.tree import Tree
from cosy.core.types import DataGroup, Literal, Var


def leaf() -> str:
    """_summary_.

    Returns:
        str: _description_
    """
    return "."


def branch(depth: int, _new_depth: int, left: str, right: str) -> str:
    """_summary_.

    Args:
        depth (int): _description_
        left (str): _description_
        right (str): _description_

    Returns:
        str: _description_
    """
    return f"(B {depth} {left} {right})"


@pytest.fixture
def component_specifications():
    """_summary_.

    Returns:
        _type_: _description_
    """
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
    """_summary_.

    Returns:
        _type_: _description_
    """
    return Literal(2)


T = int | Callable


def test_contains_tree(query, component_specifications) -> None:
    """_summary_.

    Args:
        query (_type_): _description_
        component_specifications (_type_): _description_
    """
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
