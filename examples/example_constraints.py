##Constraints##
"""
Demonstrates constraints in CoSy.
"""

import re

from cosy import CoSy
from cosy.dsl import DSL
from cosy.types import Constructor, Group, Literal, Type, Var


def empty() -> str:
    """
    Return an empty string.

    :return: An empty string.
    """
    return ""


def zero(s: str) -> str:
    """
    Append the string "0" to the input string.

    :param s: The input string to which "0" will be appended.
    :return: The input string with "0" appended.
    """
    return s + "0"


def one(s: str) -> str:
    """
    Append the string "1" to the input string.

    :param s: The input string to which "1" will be appended.
    :return: The input string with "1" appended.
    """
    return s + "1"


def fin(_r: str, s: str) -> str:
    """
    Return the input string. The input regular expression is not used in this function as it does not contribute to
    the resulting string interpretation of synthesized results. However, it can not be omitted, as the type of the
    combinator specifies its presence.

    :param _r: The input regular expression.
    :param s: The input string.
    :return: The unmodified input string.
    """
    return s


def main():
    # regular expressions
    class RegularExpression(Group):
        name = "regex"

        def __contains__(self, value: object) -> bool:
            return isinstance(value, str)

        def __iter__(self):
            pass

    component_specifications = {
        empty: DSL().suffix(Constructor("str")),
        zero: DSL().argument("s", Constructor("str")).suffix(Constructor("str")),
        one: DSL().argument("s", Constructor("str")).suffix(Constructor("str")),
        fin: DSL()
        .parameter("r", RegularExpression())
        .argument("s", Constructor("str"))
        # parameter constraint to ensure that s matches the regular expression r
        .constraint(lambda vs: re.fullmatch(vs["r"], vs["s"].interpret()))
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
