##Constraints##
"""
Demonstrates constraints in CoSy.
"""

import re

from cosy import Maestro
from cosy.core.specification_builder import SpecificationBuilder
from cosy.core.types import Constructor, Group, Literal, Type, Var


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

    named_components_with_specifications = [
        (
            "empty",
            empty,
            SpecificationBuilder().suffix(Constructor("str")),
        ),
        (
            "zero",
            zero,
            SpecificationBuilder().argument("s", Constructor("str")).suffix(Constructor("str")),
        ),
        (
            "one",
            one,
            SpecificationBuilder().argument("s", Constructor("str")).suffix(Constructor("str")),
        ),
        (
            "fin",
            fin,
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

    # Query the Maestro with the target and print results
    for result in maestro.query(target):
        print(result)


if __name__ == "__main__":
    main()
