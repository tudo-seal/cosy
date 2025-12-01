# regression test for literal substitution inference on overlapping substitutions

from cosy import Constructor, Literal, SpecificationBuilder, Synthesizer, Var
from cosy.types import Group


def test_param() -> None:
    class Nat(Group):
        name = "Nat"

        def __init__(self):
            super().__init__()

        def __iter__(self):
            yield from []

        def __contains__(self, item):
            return isinstance(item, int) and item >= 0

    class Bool(Group):
        name = "Bool"

        def __init__(self):
            super().__init__()
            self.values = [True, False]

        def __iter__(self):
            yield from self.values

        def __contains__(self, item):
            return item in self.values

    # assert that only the value 42 is tried for n
    def assert_42(vs):
        if vs["n"] != 42:
            raise ValueError
        return True

    repo = {
        "C": SpecificationBuilder()
        .parameter("n", Nat())
        .parameter("m", Bool())
        .parameter_constraint(assert_42)
        .suffix(
            (((("a" @ Var("n")) & ("b" @ Var("m"))) ** Constructor("d")) ** Constructor("d"))
            & (((("a" @ Var("n")) & ("c" @ Var("m"))) ** Constructor("d")) ** Constructor("d"))
        ),
    }

    synthesizer = Synthesizer(repo, {})

    # n is constrained to 42, for m both Bools are possible
    target = (("a" @ Literal(42)) ** Constructor("d")) ** Constructor("d")

    solution_space = synthesizer.construct_solution_space(target)
    terms = {str(t) for t in solution_space.enumerate_trees(target, 10)}

    assert terms == {"C 42 True", "C 42 False"}
