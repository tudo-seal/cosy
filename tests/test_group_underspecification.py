# test for missing iter implementation in a Group subclass
import re

import pytest
from cosy.specification_builder import SpecificationBuilder
from cosy.synthesizer import Synthesizer
from cosy.types import Constructor, Group


def test_infinite_enumeration() -> None:
    def c(x) -> str:
        return f"(C {x})"

    # group with missing iter method
    class Contains(Group):
        name = "contains"

        def __iter__(self):
            pass

        def __contains__(self, _value: object) -> bool:
            return True

    component_specifications = {
        c: SpecificationBuilder().parameter("x", Contains()).suffix(Constructor("c")),
    }

    synthesizer = Synthesizer(component_specifications)
    target = Constructor("c")

    with pytest.raises(ValueError, match=re.escape("Group contains is not iterable.")):
        # iter is necessary to determine the value of the literal variable x
        synthesizer.construct_solution_space(target)
