from collections.abc import Callable, Iterable, Mapping
from itertools import product

import pytest
from cosy.dsl import DSL
from cosy.synthesizer import Specification, Synthesizer
from cosy.tree import Tree
from cosy.types import Constructor, Group, Literal, Type, Var


def is_free(pos: tuple[int, int]) -> bool:
    col, row = pos
    seed = 0
    if row == col:
        return True
    return pow(11, (row + col + seed) * (row + col + seed) + col + 7, 1000003) % 5 > 0


@pytest.fixture
def component_specifications() -> (
    Mapping[
        Callable[[tuple[int, int], tuple[int, int], str], str] | str,
        Specification,
    ]
):
    def up(b: tuple[int, int], _a: tuple[int, int], p: str) -> str:
        return f"{p} => UP({b})"

    def down(b: tuple[int, int], _a: tuple[int, int], p: str) -> str:
        return f"{p} => DOWN({b})"

    def left(b: tuple[int, int], _a: tuple[int, int], p: str) -> str:
        return f"{p} => LEFT({b})"

    def right(b: tuple[int, int], _a: tuple[int, int], p: str) -> str:
        return f"{p} => RIGHT({b})"

    def pos(ab: str) -> Type:
        return Constructor("pos", Var(ab))

    def getpath(path: Tree) -> Iterable[tuple[int, int]]:
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

    class Int2(Group):
        name = "int2"
        _positions = frozenset(filter(is_free, product(range(SIZE), range(SIZE))))

        def __contains__(self, x):
            return x in self._positions

        def __iter__(self):
            yield from self._positions

    int2 = Int2()

    return {
        up: DSL()
        .parameter("b", int2)
        .parameter("a", int2, lambda vs: [(vs["b"][0], vs["b"][1] + 1)])
        .argument("pos", pos("a"))
        .constraint(lambda vs: vs["b"] not in getpath(vs["pos"]))
        .suffix(pos("b")),
        down: DSL()
        .parameter("b", int2)
        .parameter("a", int2, lambda vs: [(vs["b"][0], vs["b"][1] - 1)])
        .argument("pos", pos("a"))
        .constraint(lambda vs: vs["b"] not in getpath(vs["pos"]))
        .suffix(pos("b")),
        left: DSL()
        .parameter("b", int2)
        .parameter("a", int2, lambda vs: [(vs["b"][0] + 1, vs["b"][1])])
        .argument("pos", pos("a"))
        .constraint(lambda vs: vs["b"] not in getpath(vs["pos"]))
        .suffix(pos("b")),
        right: DSL()
        .parameter("b", int2)
        .parameter("a", int2, lambda vs: [(vs["b"][0] - 1, vs["b"][1])])
        .argument("pos", pos("a"))
        .constraint(lambda vs: vs["b"] not in getpath(vs["pos"]))
        .suffix(pos("b")),
        "START": "pos" @ (Literal((0, 0))),
    }


SIZE = 50


def test_benchmark_maze(component_specifications, benchmark):
    fin = "pos" @ (Literal((SIZE - 1, SIZE - 1)))

    synthesizer = Synthesizer(component_specifications)
    benchmark(synthesizer.construct_solution_space, fin)
