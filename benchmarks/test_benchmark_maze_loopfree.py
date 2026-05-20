"""_summary_."""

from collections.abc import Callable, Iterable, Mapping
from itertools import product

import pytest

from cosy.core.specification_builder import SpecificationBuilder
from cosy.core.synthesizer import Specification, Synthesizer
from cosy.core.tree import Tree
from cosy.core.types import Constructor, DataGroup, Literal, Type, Var


def is_free(pos: tuple[int, int]) -> bool:
    """_summary_.

    Args:
        pos (tuple[int, int]): _description_

    Returns:
        bool: _description_
    """
    col, row = pos
    seed = 0
    if row == col:
        return True
    return pow(11, (row + col + seed) * (row + col + seed) + col + 7, 1000003) % 5 > 0


@pytest.fixture
def component_specifications() -> Mapping[
    Callable[[tuple[int, int], tuple[int, int], str], str] | str,
    Specification,
]:
    """_summary_.

    Returns:
        Mapping[Callable[[tuple[int, int], tuple[int, int], str], str] | str, Specification]: _description_
    """

    def up(b: tuple[int, int], _a: tuple[int, int], p: str) -> str:
        """_summary_.

        Args:
            b (tuple[int, int]): _description_
            p (str): _description_

        Returns:
            str: _description_
        """
        return f"{p} => UP({b})"

    def down(b: tuple[int, int], _a: tuple[int, int], p: str) -> str:
        """_summary_.

        Args:
            b (tuple[int, int]): _description_
            p (str): _description_

        Returns:
            str: _description_
        """
        return f"{p} => DOWN({b})"

    def left(b: tuple[int, int], _a: tuple[int, int], p: str) -> str:
        """_summary_.

        Args:
            b (tuple[int, int]): _description_
            p (str): _description_

        Returns:
            str: _description_
        """
        return f"{p} => LEFT({b})"

    def right(b: tuple[int, int], _a: tuple[int, int], p: str) -> str:
        """_summary_.

        Args:
            b (tuple[int, int]): _description_
            p (str): _description_

        Returns:
            str: _description_
        """
        return f"{p} => RIGHT({b})"

    def pos(ab: str) -> Type:
        """_summary_.

        Args:
            ab (str): _description_

        Returns:
            Type: _description_
        """
        return Constructor("pos", Var(ab))

    def getpath(path: Tree) -> Iterable[tuple[int, int]]:
        """_summary_.

        Args:
            path (Tree): _description_

        Yields:
            tuple[int, int]: _description_

        Raises:
            TypeError: _description_
        """
        while path.root != "START":
            position = path.children[0].root
            path = path.children[2]
            if isinstance(position, tuple) and isinstance(path, Tree):
                yield position
            else:
                msg = "Expected position to be a tuple and path to be a tree."
                raise TypeError(msg)
        yield (0, 0)
        return

    int2 = DataGroup("int2", frozenset(filter(is_free, product(range(SIZE), range(SIZE)))))

    return {
        up: SpecificationBuilder()
        .parameter("b", int2)
        .parameter("a", int2, lambda vs: [(vs["b"][0], vs["b"][1] + 1)])
        .argument("pos", pos("a"))
        .constraint(lambda vs: vs["b"] not in getpath(vs["pos"]))
        .suffix(pos("b")),
        down: SpecificationBuilder()
        .parameter("b", int2)
        .parameter("a", int2, lambda vs: [(vs["b"][0], vs["b"][1] - 1)])
        .argument("pos", pos("a"))
        .constraint(lambda vs: vs["b"] not in getpath(vs["pos"]))
        .suffix(pos("b")),
        left: SpecificationBuilder()
        .parameter("b", int2)
        .parameter("a", int2, lambda vs: [(vs["b"][0] + 1, vs["b"][1])])
        .argument("pos", pos("a"))
        .constraint(lambda vs: vs["b"] not in getpath(vs["pos"]))
        .suffix(pos("b")),
        right: SpecificationBuilder()
        .parameter("b", int2)
        .parameter("a", int2, lambda vs: [(vs["b"][0] - 1, vs["b"][1])])
        .argument("pos", pos("a"))
        .constraint(lambda vs: vs["b"] not in getpath(vs["pos"]))
        .suffix(pos("b")),
        "START": "pos" @ (Literal((0, 0))),
    }


SIZE = 50


def test_benchmark_maze_loopfree(component_specifications, benchmark):
    """_summary_.

    Args:
        component_specifications (_type_): _description_
        benchmark (_type_): _description_
    """
    fin = "pos" @ (Literal((SIZE - 1, SIZE - 1)))

    synthesizer = Synthesizer(component_specifications)
    benchmark(synthesizer.construct_solution_space, fin)
