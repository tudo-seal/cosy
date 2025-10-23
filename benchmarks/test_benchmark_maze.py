from collections.abc import Callable, Mapping
from itertools import product

import pytest
from cosy.specification_builder import SpecificationBuilder
from cosy.synthesizer import Specification, Synthesizer
from cosy.types import Constructor, DataGroup, Literal, Type, Var


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

    int2 = DataGroup("int2", frozenset(filter(is_free, product(range(SIZE), range(SIZE)))))

    return {
        up: SpecificationBuilder()
        .parameter("b", int2)
        .parameter("a", int2, lambda vs: [(vs["b"][0], vs["b"][1] + 1)])
        .argument("pos", pos("a"))
        .suffix(pos("b")),
        down: SpecificationBuilder()
        .parameter("b", int2)
        .parameter("a", int2, lambda vs: [(vs["b"][0], vs["b"][1] - 1)])
        .argument("pos", pos("a"))
        .suffix(pos("b")),
        left: SpecificationBuilder()
        .parameter("b", int2)
        .parameter("a", int2, lambda vs: [(vs["b"][0] + 1, vs["b"][1])])
        .argument("pos", pos("a"))
        .suffix(pos("b")),
        right: SpecificationBuilder()
        .parameter("b", int2)
        .parameter("a", int2, lambda vs: [(vs["b"][0] - 1, vs["b"][1])])
        .argument("pos", pos("a"))
        .suffix(pos("b")),
        "START": "pos" @ (Literal((0, 0))),
    }


SIZE = 50


def test_benchmark_maze(component_specifications, benchmark):
    fin = "pos" @ (Literal((SIZE - 1, SIZE - 1)))

    synthesizer = Synthesizer(component_specifications)
    benchmark(synthesizer.construct_solution_space, fin)
