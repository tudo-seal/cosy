"""_summary_."""
# regression test for literal substitution inference on overlapping substitutions

from cosy.core import Constructor, Literal, SpecificationBuilder, Synthesizer, Var
from cosy.core.synthesizer import Specification
from cosy.core.tree import Tree
from cosy.core.types import Group

T = int | str | None | bool


def test_param() -> None:
    """_summary_."""

    class Nat(Group):
        """_summary_."""

        name = "Nat"

        def __init__(self):
            """_summary_."""
            super().__init__()

        def __iter__(self):
            """_summary_.

            Yields:
                _type_: _description_
            """
            yield from []

        def __contains__(self, item):
            """_summary_.

            Args:
                item (_type_): _description_

            Returns:
                _type_: _description_
            """
            return isinstance(item, int) and item >= 0

    class Bool(Group):
        """_summary_.

        Attributes:
            values (_type_): _description_
        """

        name = "Bool"

        def __init__(self):
            """_summary_."""
            super().__init__()
            self.values = [True, False, None]

        def __iter__(self):
            """_summary_.

            Yields:
                _type_: _description_
            """
            yield from self.values

        def __contains__(self, item):
            """_summary_.

            Args:
                item (_type_): _description_

            Returns:
                _type_: _description_
            """
            return item in self.values

    # assert that only the value 42 is tried for n
    def assert_42(vs):
        """_summary_.

        Args:
            vs (_type_): _description_

        Returns:
            _type_: _description_

        Raises:
            ValueError: _description_
        """
        if vs["n"] != 42:
            raise ValueError
        return True

    # assert that m is not None
    def assert_m(vs):
        """_summary_.

        Args:
            vs (_type_): _description_

        Returns:
            _type_: _description_
        """
        return vs["m"] is not None

    repo: dict[T, Specification] = {
        "C": SpecificationBuilder()
        .parameter("n", Nat())
        .parameter("m", Bool())
        .parameter_constraint(assert_42)
        .constraint(assert_m)
        .suffix(
            (((("a" @ Var("n")) & ("b" @ Var("m"))) ** Constructor("d")) ** Constructor("d"))
            & (((("a" @ Var("n")) & ("c" @ Var("m"))) ** Constructor("d")) ** Constructor("d"))
        ),
    }

    assert isinstance(repo["C"], Specification)

    synthesizer: Synthesizer[T] = Synthesizer(repo, {})

    # n is constrained to 42, for m both Bools are possible
    target = (("a" @ Literal(42)) ** Constructor("d")) ** Constructor("d")

    solution_space = synthesizer.construct_solution_space(target)
    terms = {str(t) for t in solution_space.enumerate_trees(target, 10)}

    assert terms == {"C 42 True", "C 42 False"}

    for t in solution_space.enumerate_trees(target, 10):
        assert solution_space.contains_tree(target, t)

    assert {str(t) for t in solution_space.depth_first_resolution(target, 10)} == {"C 42 True", "C 42 False"}

    for t in solution_space.depth_first_resolution(target, 10):
        assert solution_space.contains_tree(target, t)

    assert {str(t) for t in solution_space.breadth_first_resolution(target, 10)} == {"C 42 True", "C 42 False"}

    for t in solution_space.breadth_first_resolution(target, 10):
        assert solution_space.contains_tree(target, t)

    # the tree "C 42 None" is not in the solution space because of assert_m
    tree_none: Tree[T] = Tree[T](
        "C",
        [
            Tree[T](42),
            Tree[T](None),
        ],
    )

    assert not solution_space.contains_tree(target, tree_none)
