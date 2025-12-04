# regression tests combinatorics.py
from itertools import combinations
from random import Random

import pytest
from cosy.combinatorics import maximal_elements, minimal_covers


@pytest.fixture
def elements():
    bound = 10
    dimension = 4
    count = 200
    rand = Random(0)

    def random_element() -> tuple[int, ...]:
        return tuple(rand.randint(0, bound) for _ in range(dimension))

    return [random_element() for _ in range(count)]


@pytest.fixture
def sets():
    bound = 10
    size = 4
    count = 13
    rand = Random(0)

    def random_set() -> frozenset[int]:
        return frozenset(rand.randint(0, bound) for _ in range(size))

    return {random_set() for _ in range(count)}


@pytest.fixture
def to_cover():
    bound = 10
    return set(range(bound))


def test_maximal_elements(elements) -> None:
    """Test maximal_elements function."""

    def compare(x, y):
        return all(a <= b for a, b in zip(x, y, strict=False))

    maximal = maximal_elements(elements, compare)
    assert all(
        any(compare(x, y) for y in maximal) for x in elements
    ), "Some element is not dominated by maximal elements"

    for i, x in enumerate(maximal):
        for j, y in enumerate(maximal):
            if i != j:
                assert not compare(x, y), f"Maximal elements {x} and {y} are not incomparable"


def test_minimal_covers(sets, to_cover) -> None:
    """Test minimal_covers function."""
    covers = minimal_covers(
        list(sets),
        to_cover,
        lambda s, e: e in s,
    )
    assert all(
        any(e in s for s in cover) for cover in covers for e in to_cover
    ), "Some element is not covered by minimal covers"
    assert all(s in sets for cover in covers for s in cover), "Some set in a cover is not included in sets"

    for i, cover1 in enumerate(covers):
        for j, cover2 in enumerate(covers):
            if i != j:
                assert not all(
                    s in cover2 for s in cover1
                ), f"Minimal covers {cover1} and {cover2} are not incomparable"

    # check if all covers are found
    for i in range(len(sets) + 1):
        for cover in combinations(sets, i):
            if all(any(e in s for s in cover) for e in to_cover):
                assert any(
                    all(s in cover for s in cover2) for cover2 in covers
                ), f"Cover {cover} is not found in minimal covers"
