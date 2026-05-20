##Constraints##
"""Demonstrates constraints in CoSy."""

import re

from cosy.core.specification_builder import SpecificationBuilder
from cosy.core.types import Constructor, Group, Literal, Type, Var
from cosy.maestro import Maestro


def empty() -> str:
    """Return an empty string.

    Returns:
        str: An empty string.
    """
    return ""


def zero(s: str) -> str:
    """Append the string "0" to the input string.

    Args:
        s (str): The input string to which "0" will be appended.

    Returns:
        str: The input string with "0" appended.
    """
    return s + "0"


def one(s: str) -> str:
    """Append the string "1" to the input string.

    Args:
        s (str): The input string to which "1" will be appended.

    Returns:
        str: The input string with "1" appended.
    """
    return s + "1"


def fin(_r: str, s: str) -> str:
    """Return the input string. The input regular expression is not used in this function as it does not contribute to.

    the resulting string interpretation of synthesized results. However, it can not be omitted, as the type of the
    combinator specifies its presence.

    Args:
        _r (_type_): The input regular expression.
        s (str): The input string.

    Returns:
        str: The unmodified input string.
    """
    return s


def main():
    # regular expressions
    class RegularExpression(Group):
        """_summary_."""

        name = "regex"

        def __contains__(self, value: object) -> bool:
            """_summary_.

            Args:
                value (object): _description_

            Returns:
                bool: _description_
            """
            return isinstance(value, str)

        def __iter__(self):
            """_summary_."""

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

    # Query the Maestro with the target, then visualize and print results
    results = maestro.query(target)
    results.visualize(amount=3)
    print("Now printing all infinite results in order:")
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
