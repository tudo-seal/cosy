##Fibonacci Linear##
"""
Overall description of this example goes here.
"""

from cosy.core.specification_builder import SpecificationBuilder
from cosy.core.types import Constructor, Group, Literal, Var
from cosy.maestro import Maestro


def fst(_x: int, f: tuple[int, int]) -> int:
    """
    Get the first element of a pair.

    :param x: A pair containing two integers.
    :return: The first element of the pair.
    """
    return f[0]


def fib_zero_one() -> tuple[int, int]:
    """
    The pair of Fibonacci numbers at indices 0 and 1.

    :return: The pair of Fibonacci numbers at indices 0 and 1.
    """
    return (0, 1)


def fib_next(_y: int, _x: int, f: tuple[int, int]) -> tuple[int, int]:
    """
    Calculate the pair Fibonacci numbers at a given indices y and y + 1
    using the pair of Fibonacci numbers at indices x = y - 1 and y.

    :param _y: The indices for which the Fibonacci number is calculated.
    :param _x: The indices decremented by 1.
    :param f: The pair of Fibonacci numbers at indices y - 1 and y.
    :return: The pair of Fibonacci numbers at indices y and y + 1.
    """
    return (f[1], f[0] + f[1])


def main():
    # range of relevant indices for Fibonacci numbers
    class Int(Group):
        name = "int"
        _bound = 6000

        def __contains__(self, item: object) -> bool:
            return isinstance(item, int) and 0 <= item < self._bound

        def __iter__(self):
            yield from frozenset(range(self._bound))

    component_specifications = [
        (
            "fst",
            fst,
            SpecificationBuilder()
            .parameter("x", Int())
            .argument("f", Constructor("fibs") & Constructor("at", Var("x")))
            .suffix(Constructor("fib") & Constructor("at", Var("x"))),
        ),
        (
            "fib_zero_one",
            fib_zero_one,
            SpecificationBuilder().suffix(Constructor("fibs") & Constructor("at", Literal(0))),
        ),
        (
            "fib_next",
            fib_next,
            SpecificationBuilder()
            .parameter("y", Int())
            .parameter("x", Int(), lambda vs: [vs["y"] - 1])
            .argument("f", Constructor("fibs") & Constructor("at", Var("x")))
            .suffix(Constructor("fibs") & Constructor("at", Var("y"))),
        ),
    ]

    # Tell the Maestro about the component specifications
    maestro = Maestro(component_specifications)

    # Target for the Fibonacci number at index 5000
    target = Constructor("fib") & Constructor("at", Literal(5000))

    # Query the Maestro with the target and print the only composition
    results = maestro.query(target)
    print("5000th Fibonacci number:", next(iter(results)))
    # Visualize that result
    results.visualize()


if __name__ == "__main__":
    main()
