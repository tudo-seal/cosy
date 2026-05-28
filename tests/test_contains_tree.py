"""_summary_."""

# regression test for contains_tree
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

    tree_correct_positions = tree_correct.positions()

    for pos in tree_correct_positions:
        trees = solution_space.depth_first_resolution(query, tree=tree_correct, pos=pos)
        trees = list(trees)
        assert trees
        assert all(solution_space.contains_tree(query, t) for t in trees)

    tree_wrong_1_leaf_positions = tree_wrong_1.leaf_positions()
    for pos in tree_wrong_1_leaf_positions:
        trees = solution_space.breadth_first_resolution(query, tree=tree_wrong_1, pos=pos)
        trees = list(trees)
        assert not trees
        assert not any(solution_space.contains_tree(query, t) for t in trees)

    trees = solution_space.breadth_first_resolution(query, tree=tree_wrong_2, pos=(3,))
    trees = list(trees)
    assert tree_wrong_2 not in trees
    assert all(solution_space.contains_tree(query, t) for t in trees)
