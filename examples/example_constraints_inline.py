##Constraints with Inlined Interpretations##
"""
Demonstrates constraints in CoSy, highlighting how the Component class can be used to inline the interpretation.
"""

import re

from cosy.core.specification_builder import SpecificationBuilder
from cosy.core.types import Constructor, Group, Literal, Type, Var
from cosy.maestro import Maestro


def main():
    # regular expressions
    class RegularExpression(Group):
        name = "regex"

        def __contains__(self, value: object) -> bool:
            return isinstance(value, str)

        def __iter__(self):
            pass

    named_components_with_specifications = [
        (
            "empty",
            lambda: "",
            SpecificationBuilder().suffix(Constructor("str")),
        ),
        (
            "zero",
            lambda s: s + "0",
            SpecificationBuilder().argument("s", Constructor("str")).suffix(Constructor("str")),
        ),
        (
            "one",
            lambda s: s + "1",
            SpecificationBuilder().argument("s", Constructor("str")).suffix(Constructor("str")),
        ),
        (
            "fin",
            lambda _, s: s,
            SpecificationBuilder()
            .parameter("r", RegularExpression())
            .argument("s", Constructor("str"))
            .constraint(lambda vs: bool(re.fullmatch(vs["r"], vs["s"])))
            .suffix(Constructor("matches", Var("r"))),
        ),
    ]

    # Tell the Maestro about the component specifications
    maestro = Maestro(named_components_with_specifications)

    # Query for heavy strings
    target: Type = Constructor("matches", Literal("01+0"))

    # Query the Maestro with the target, then visualize and print results
    results = maestro.query(target)
    results.visualize(amount=10)
    print("Now printing all infinite results in order:")
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
