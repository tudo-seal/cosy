"""_summary_."""
# benchmark for the maximal_elements function

from random import Random

import pytest

from cosy.core.combinatorics import maximal_elements


@pytest.fixture
def elements():
    """_summary_.

    Returns:
        _type_: _description_
    """
    bound = 20
    dimension = 10
    count = 500
    rand = Random(0)

    def random_element() -> tuple[int, ...]:
        """_summary_.

        Returns:
            tuple[int, ...]: _description_
        """
        return tuple(rand.randint(0, bound) for _ in range(dimension))

    return [random_element() for _ in range(count)]


def test_benchmark_maximal_elements(elements, benchmark):
    """Benchmark maximal_elements function.

    Args:
        elements (_type_): _description_
        benchmark (_type_): _description_
    """

    def compare(x, y):
        """_summary_.

        Args:
            x (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        return all(a <= b for a, b in zip(x, y, strict=False))

    benchmark(maximal_elements, elements, compare)
