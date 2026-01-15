##Fibonacci##
"""
Overall description of this example goes here.
"""

from cosy.core.specification_builder import SpecificationBuilder
from cosy.core.types import Constructor, DataGroup, Literal, Type, Var
from cosy.maestro import Maestro


def fib_zero() -> int:
    """
    The Fibonacci number at index 0.

    :return: The Fibonacci number at index 0.
    """
    return 0


def fib_one() -> int:
    """
    The Fibonacci number at index .

    :return: The Fibonacci number at index 1.
    """
    return 1


def fib_next(_z: int, _y: int, _x: int, f1: int, f2: int) -> int:
    """
    Calculate the Fibonacci number at a given index z using the Fibonacci numbers
    at indices x = z - 2 and y = z - 1.

    :param _z: The index for which the Fibonacci number is calculated.
    :param _y: The index z - 1.
    :param _x: The index z - 2.
    :param f1: The Fibonacci number at index (z - 1).
    :param f2: The Fibonacci number at index (z - 2).
    :return: The Fibonacci number at index z.
    """
    return f1 + f2


def main():
    # range of relevant indices for Fibonacci numbers
    bound = 20

    named_components_with_specifications = [
        (
            "fib_zero",
            fib_zero,
            SpecificationBuilder().suffix(Constructor("fib") & Constructor("at", Literal(0))),
        ),
        (
            "fib_one",
            fib_one,
            SpecificationBuilder().suffix(Constructor("fib") & Constructor("at", Literal(1))),
        ),
        (
            "fib_next",
            fib_next,
            SpecificationBuilder()
            .parameter("z", DataGroup("int", range(bound)))
            .parameter("y", DataGroup("int", range(bound)), lambda vs: [vs["z"] - 1])
            .parameter("x", DataGroup("int", range(bound)), lambda vs: [vs["z"] - 2])
            .argument("f1", Constructor("fib") & Constructor("at", Var("y")))
            .argument("f2", Constructor("fib") & Constructor("at", Var("x")))
            .suffix(Constructor("fib") & Constructor("at", Var("z"))),
        ),
    ]

    # Tell the Maestro about the component specifications
    maestro = Maestro(named_components_with_specifications)

    # Target describing Fibonacci numbers at relevant indices
    target: Type = Constructor("fib")

    # Query the Maestro with the target and print the compositions
    for result in maestro.query(target):
        print(result)

    for i in range(20):
        # Target for Fibonacci numbers at index i
        target = Constructor("fib") & Constructor("at", Literal(i))

        # Query the Maestro with the target and print the only composition
        print(i, next(iter(maestro.query(target))))


if __name__ == "__main__":
    main()
