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


def fin(_b: bool, s: str) -> str:
    return s


def is_heavy(s: str) -> bool:
    """
    Check if the number of '1's in the string is greater than the number of '0's.

    :param s: A string containing '0's and '1's.
    :return: True if the number of '1's is greater than the number of '0's, False otherwise.
    """
    return s.count("0") < s.count("1")


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
