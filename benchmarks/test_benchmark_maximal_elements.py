# benchmark for the maximal_elements function

from random import Random

import pytest
from cosy.combinatorics import maximal_elements


@pytest.fixture
def elements():
    bound = 20
    dimension = 10
    count = 500
    rand = Random(0)

    def random_element() -> tuple[int, ...]:
        return tuple(rand.randint(0, bound) for _ in range(dimension))

    return [random_element() for _ in range(count)]


def test_benchmark_maximal_elements(elements, benchmark):
    """Benchmark maximal_elements function."""

    def compare(x, y):
        return all(a <= b for a, b in zip(x, y, strict=False))

    benchmark(maximal_elements, elements, compare)
