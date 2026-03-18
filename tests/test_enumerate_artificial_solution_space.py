# Tests enumeration with multiple non-terminal arguments and cross-argument constraints.
# Verifies that constants are satisfied and no duplicates or extra trees are produced.

from cosy.core.solution_space import ConstantArgument, NonTerminalArgument, SolutionSpace


def test_andrej():
    solution_space = SolutionSpace()
    arguments = (
        ConstantArgument("x", 0, None),
        NonTerminalArgument("v", "T1"),
        ConstantArgument("y", 1, None),
        NonTerminalArgument("w", "T1"),
        NonTerminalArgument(None, "T1"),
        ConstantArgument("z", 2, None),
        NonTerminalArgument(None, "T1"),
    )
    predicates = (
        lambda vs: vs["v"] == vs["w"],
        lambda vs: vs["x"] == 0 and vs["y"] == 1 and vs["z"] == 2,
    )
    solution_space.add_rule("T0", "t", arguments, predicates)
    solution_space.add_rule("T1", "l", (), ())
    solution_space.add_rule("T1", "r", (), ())

    terms = solution_space.enumerate_trees("T0", max_count=100)
    terms_1 = solution_space.depth_first_resolution("T0", max_count=100)
    terms_2 = solution_space.breadth_first_resolution("T0", max_count=100)

    trees = set()
    trees_1 = set()
    trees_2 = set()

    exptected_results = {
        "t 0 l 1 l l 2 l",
        "t 0 r 1 r r 2 r",
        "t 0 r 1 r r 2 l",
        "t 0 r 1 r l 2 r",
        "t 0 r 1 r l 2 l",
        "t 0 l 1 l r 2 r",
        "t 0 l 1 l r 2 l",
        "t 0 l 1 l l 2 r",
    }

    for tree in terms:
        trees.add(str(tree))

    for tree in terms_1:
        trees_1.add(str(tree))

    for tree in terms_2:
        trees_2.add(str(tree))

    assert trees == exptected_results
    assert trees_1 == exptected_results
    assert trees_2 == exptected_results
