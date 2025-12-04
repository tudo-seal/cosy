# regression test for recursive unproductive specification
import pytest
from cosy.synthesizer import Synthesizer
from cosy.types import Arrow, Constructor


@pytest.fixture
def component_specifications():
    def ab(s: str) -> str:
        return f"AB {s}"

    def ba(s: str) -> str:
        return f"BA {s}"

    return {
        # recursive unproductive specification
        ab: Arrow(Constructor("a"), Constructor("b")),
        ba: Arrow(Constructor("b"), Constructor("a")),
    }
    return


@pytest.fixture
def query():
    return Constructor("a")


def test_param(query, component_specifications) -> None:
    solution_space = Synthesizer(component_specifications).construct_solution_space(query)
    for tree in solution_space.enumerate_trees(query):
        msg = f"This should not be reached {tree}"
        raise NotImplementedError(msg)
