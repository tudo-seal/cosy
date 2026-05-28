"""_summary_."""
# test for candidate generation for assigning values to literal variables

from cosy.core.specification_builder import SpecificationBuilder
from cosy.core.synthesizer import Synthesizer
from cosy.core.types import Constructor, Group, Literal, Omega, Type, Var


def test_candidates() -> None:
    # literal varibles can be assigned computed values
    """_summary_."""

    def c(x: bool, y: bool, z: bool) -> str:
        """_summary_.

        Args:
            x (bool): _description_
            y (bool): _description_
            z (bool): _description_

        Returns:
            str: _description_
        """
        return f"C {x} {y} {z}"

    class Bool(Group):
        """_summary_."""

        name = "bool"

        def __contains__(self, x):
            """_summary_.

            Args:
                x (_type_): _description_

            Returns:
                _type_: _description_
            """
            return super().__contains__(x)

        def __iter__(self):
            """_summary_.

            Yields:
                _type_: _description_
            """
            yield from [True, False]

    component_specifications = {
        c: SpecificationBuilder()
        .parameter("x", Bool())
        .parameter_constraint(lambda vs: vs["x"])  # x is True
        .parameter("y", Bool(), lambda _vs: [False])  # y is False
        .parameter("z", Bool(), lambda vs: [vs["x"]])  # z is equal to x
        .suffix(Constructor("a", Var("x")) & Constructor("b", Var("y")) & Constructor("c", Var("z")))
    }

    def xyz(x: bool | None, y: bool | None, z: bool | None) -> Type:
        """_summary_.

        Args:
            x (bool | None): _description_
            y (bool | None): _description_
            z (bool | None): _description_

        Returns:
            Type: _description_
        """
        return (
            Constructor("a", Omega() if x is None else Literal(x))
            & Constructor("b", Omega() if y is None else Literal(y))
            & Constructor("c", Omega() if z is None else Literal(z))
        )

    synthesizer = Synthesizer(component_specifications)

    for x in [True, False, None]:
        for y in [True, False, None]:
            for z in [True, False, None]:
                target = xyz(x, y, z)
                solution_space = synthesizer.construct_solution_space(target)
                result = {tree.interpret() for tree in solution_space.enumerate_trees(target)}
                if (x is not None and not x) or y or (z is not None and not z):
                    assert len(result) == 0
                else:
                    assert result == {"C True False True"}
                result = {tree.interpret() for tree in solution_space.depth_first_resolution(target)}
                if (x is not None and not x) or y or (z is not None and not z):
                    assert len(result) == 0
                else:
                    assert result == {"C True False True"}
                result = {tree.interpret() for tree in solution_space.breadth_first_resolution(target)}
                if (x is not None and not x) or y or (z is not None and not z):
                    assert len(result) == 0
                else:
                    assert result == {"C True False True"}


def test_multi_values1() -> None:
    # a literal varible can be assigned multiple computed values
    """_summary_."""

    def c(a: int, b: int) -> str:
        """_summary_.

        Args:
            a (int): _description_
            b (int): _description_

        Returns:
            str: _description_
        """
        return f"C {a} {b}"

    class Int(Group):
        """_summary_."""

        name = "int"

        def __contains__(self, x):
            """_summary_.

            Args:
                x (_type_): _description_

            Returns:
                _type_: _description_
            """
            return super().__contains__(x)

        def __iter__(self):
            """_summary_.

            Yields:
                _type_: _description_
            """
            yield from [0, 1, 2, 3]

    component_specifications = {
        c: SpecificationBuilder()
        .parameter("a", Int())  # a in [0, 1, 2, 3]
        .parameter("b", Int(), lambda vs: [vs["a"] - 1, vs["a"] + 1])  # b in [a-1, a+1]
        .suffix(Constructor("c", Var("a")))
    }

    synthesizer = Synthesizer(component_specifications)
    target = Constructor("c", Literal(0))
    solution_space = synthesizer.construct_solution_space(target)
    assert [tree.interpret() for tree in solution_space.enumerate_trees(target)] == ["C 0 1"]
    assert [tree.interpret() for tree in solution_space.depth_first_resolution(target)] == ["C 0 1"]
    assert [tree.interpret() for tree in solution_space.breadth_first_resolution(target)] == ["C 0 1"]


def test_multi_values2() -> None:
    # a literal varible can be assigned multiple computed values
    """_summary_."""

    def c(a: int, b: int) -> str:
        """_summary_.

        Args:
            a (int): _description_
            b (int): _description_

        Returns:
            str: _description_
        """
        return f"C {a} {b}"

    class Int(Group):
        """_summary_."""

        name = "int"

        def __contains__(self, x):
            """_summary_.

            Args:
                x (_type_): _description_

            Returns:
                _type_: _description_
            """
            return super().__contains__(x)

        def __iter__(self):
            """_summary_.

            Yields:
                _type_: _description_
            """
            yield from [0, 1, 2, 3]

    component_specifications = {
        c: SpecificationBuilder()
        .parameter("a", Int())
        .parameter("b", Int(), lambda vs: [vs["a"] - 1, vs["a"] + 1])
        .suffix(Constructor("c", Var("a")))
    }

    synthesizer = Synthesizer(component_specifications)
    target = Constructor("c", Literal(1))
    solution_space = synthesizer.construct_solution_space(target)
    assert {tree.interpret() for tree in solution_space.enumerate_trees(target)} == {
        "C 1 2",
        "C 1 0",
    }
    assert {tree.interpret() for tree in solution_space.depth_first_resolution(target)} == {
        "C 1 2",
        "C 1 0",
    }
    assert {tree.interpret() for tree in solution_space.breadth_first_resolution(target)} == {
        "C 1 2",
        "C 1 0",
    }


def test_infinite_values() -> None:
    # the number of values for a literal variable can be infinite
    """_summary_."""

    class Nat(Group):
        """_summary_."""

        name = "nat"

        # represents the set of (arbitrary large) natural numbers
        def __contains__(self, value: object) -> bool:
            """_summary_.

            Args:
                value (object): _description_

            Returns:
                bool: _description_
            """
            return isinstance(value, int) and value >= 0

        def __iter__(self):
            """_summary_."""

    def c(x: int, _y: int, b: str) -> str:
        """_summary_.

        Args:
            x (int): _description_
            b (str): _description_

        Returns:
            str: _description_
        """
        return f"C {x} ({b})"

    target = "c" @ Literal(3)

    component_specifications = {
        c: SpecificationBuilder()
        .parameter("a", Nat())  # a in [0, 1, 2, ...]
        .parameter("b", Nat(), lambda vs: [vs["a"] - 1])  # b in [a-1]
        .suffix(("c" @ Var("b")) ** ("c" @ Var("a"))),  # c(b) -> c(a)
        "ZERO": "c" @ Literal(0),  # c(0)
    }

    synthesizer = Synthesizer(component_specifications)
    solution_space = synthesizer.construct_solution_space(target)

    assert [tree.interpret() for tree in solution_space.enumerate_trees(target)] == ["C 3 (C 2 (C 1 (ZERO)))"]
    assert [tree.interpret() for tree in solution_space.depth_first_resolution(target)] == ["C 3 (C 2 (C 1 (ZERO)))"]
    assert [tree.interpret() for tree in solution_space.breadth_first_resolution(target)] == ["C 3 (C 2 (C 1 (ZERO)))"]

    for tree in solution_space.enumerate_trees(target):
        assert solution_space.contains_tree(target, tree)
    for tree in solution_space.depth_first_resolution(target):
        assert solution_space.contains_tree(target, tree)
    for tree in solution_space.breadth_first_resolution(target):
        assert solution_space.contains_tree(target, tree)
