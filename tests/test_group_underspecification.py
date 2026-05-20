"""_summary_."""

# test for missing iter implementation in a Group subclass
import re

import pytest

from cosy.core.specification_builder import SpecificationBuilder
from cosy.core.synthesizer import Synthesizer
from cosy.core.types import Constructor, Group


def test_infinite_enumeration() -> None:
    """_summary_."""

    def c(x) -> str:
        """_summary_.

        Args:
            x (_type_): _description_

        Returns:
            str: _description_
        """
        return f"(C {x})"

    # group with missing iter method
    class Contains(Group):
        """_summary_."""

        name = "contains"

        def __iter__(self):
            """_summary_."""

        def __contains__(self, _value: object) -> bool:
            """_summary_.

            Returns:
                bool: _description_
            """
            return True

    component_specifications = {
        c: SpecificationBuilder().parameter("x", Contains()).suffix(Constructor("c")),
    }

    synthesizer = Synthesizer(component_specifications)
    target = Constructor("c")

    with pytest.raises(ValueError, match=re.escape("Group contains is not iterable.")):
        # iter is necessary to determine the value of the literal variable x
        synthesizer.construct_solution_space(target)
