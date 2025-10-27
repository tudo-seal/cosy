##Constraints with inlined Interpretations##
"""
Demonstrates constraints in CoSy, highlighting how the Component class can be used to inline the interpretation.
"""

import re

from cosy import Component, CoSy
from cosy.specification_builder import SpecificationBuilder
from cosy.types import Constructor, Group, Literal, Type, Var


def main():
    # regular expressions
    class RegularExpression(Group):
        name = "regex"

        def __contains__(self, value: object) -> bool:
            return isinstance(value, str)

        def __iter__(self):
            pass

    component_specifications = {
        Component(name="empty", interpretation=lambda: ""):  #
        SpecificationBuilder().suffix(Constructor("str")),
        #
        Component(name="zero", interpretation=lambda s: s + "0"):  #
        SpecificationBuilder().argument("s", Constructor("str")).suffix(Constructor("str")),
        #
        Component(name="one", interpretation=lambda s: s + "1"):  #
        SpecificationBuilder().argument("s", Constructor("str")).suffix(Constructor("str")),
        #
        Component(name="fin", interpretation=lambda _, s: s):  #
        SpecificationBuilder()
        .parameter("r", RegularExpression())
        .argument("s", Constructor("str"))
        .constraint(
            lambda vs: bool(re.fullmatch(vs["r"], vs["s"].interpret()))  # ensure s matches regular expression r
        )
        .suffix(Constructor("matches", Var("r"))),
    }

    # CoSy instance with the component specifications and parameter space
    cosy = CoSy(component_specifications)

    # query for heavy strings
    query: Type = Constructor("matches", Literal("01+0"))

    # solve the query and print the solutions
    for solution in cosy.solve(query):
        print(solution)


if __name__ == "__main__":
    main()
