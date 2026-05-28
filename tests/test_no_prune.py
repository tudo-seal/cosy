"""_summary_."""

# regression test for recursive unproductive specification
import pytest

from cosy.core.synthesizer import Synthesizer
from cosy.core.types import Arrow, Constructor


@pytest.fixture
def component_specifications():
    """_summary_.

    Returns:
        _type_: _description_
    """

    def ab(s: str) -> str:
        """_summary_.

        Args:
            s (str): _description_

        Returns:
            str: _description_
        """
        return f"AB {s}"

    def ba(s: str) -> str:
        """_summary_.

        Args:
            s (str): _description_

        Returns:
            str: _description_
        """
        return f"BA {s}"

    return {
        # recursive unproductive specification
        ab: Arrow(Constructor("a"), Constructor("b")),
        ba: Arrow(Constructor("b"), Constructor("a")),
    }
    return


@pytest.fixture
def query():
    """_summary_.

    Returns:
        _type_: _description_
    """
    return Constructor("a")


def test_param(query, component_specifications) -> None:
    """_summary_.

    Args:
        query (_type_): _description_
        component_specifications (_type_): _description_

    Raises:
        NotImplementedError: _description_
    """
    solution_space = Synthesizer(component_specifications).construct_solution_space(query)
    for tree in solution_space.enumerate_trees(query):
        msg = f"This should not be reached {tree}"
        raise NotImplementedError(msg)
