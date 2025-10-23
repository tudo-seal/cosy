##Fibonacci##
"""
Overall description of this example goes here.
"""

from cosy import Component, CoSy
from cosy.specification_builder import SpecificationBuilder
from cosy.types import Constructor, DataGroup, Literal, Type, Var


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

    component_specifications = {
        Component(name="fibonacci_zero", interpretation=fib_zero):  #
        SpecificationBuilder().suffix(Constructor("fib") & Constructor("at", Literal(0))),
        #
        Component(name="fibonacci_one", interpretation=fib_one):  #
        SpecificationBuilder().suffix(Constructor("fib") & Constructor("at", Literal(1))),
        #
        Component(name="next_fibonacci_number", interpretation=fib_next):  #
        SpecificationBuilder()
        .parameter("z", DataGroup("int", range(bound)))
        .parameter("y", DataGroup("int", range(bound)), lambda vs: [vs["z"] - 1])
        .parameter("x", DataGroup("int", range(bound)), lambda vs: [vs["z"] - 2])
        .argument("f1", Constructor("fib") & Constructor("at", Var("y")))
        .argument("f2", Constructor("fib") & Constructor("at", Var("x")))
        .suffix(Constructor("fib") & Constructor("at", Var("z"))),
    }

    # CoSy instance with the component specifications and parameter space
    cosy = CoSy(component_specifications)

    # query for Fibonacci numbers at relevant indices
    query: Type = Constructor("fib")

    # solve the query and print the solutions
    for solution in cosy.solve(query):
        print(solution)

    for i in range(20):
        # query for Fibonacci numbers at index i
        query = Constructor("fib") & Constructor("at", Literal(i))

        # solve the query and print the only solution
        print(i, next(iter(cosy.solve(query))))


if __name__ == "__main__":
    main()
