# regression test for position-wise intersection of arguments
from cosy.core.synthesizer import Synthesizer
from cosy.core.types import Constructor, Type


def test_intersect_arguments() -> None:
    component_specifications = {
        # the specification is an intersection of multi-arrows (arguments will be point-wise intersected)
        lambda x, y: f"f({x}, {y})": Type.intersect(
            [
                Type.curry([Constructor("a1"), Constructor("b1")], Constructor("b1")),
                Type.curry([Constructor("a2"), Constructor("b2")], Constructor("b2")),
                Type.curry([Constructor("a3"), Constructor("b3")], Constructor("b3")),
            ]
        ),
        "a": Type.intersect([Constructor("a1"), Constructor("a2"), Constructor("a3")]),
        "b": Type.intersect([Constructor("b1"), Constructor("b2"), Constructor("b3")]),
    }

    query = Type.intersect([Constructor("b1"), Constructor("b2"), Constructor("b3")])
    solution_space = Synthesizer(component_specifications).construct_solution_space(query)
    expected_results = {"b", "f(a, b)", "f(a, f(a, b))"}
    results = {tree.interpret() for tree in solution_space.enumerate_trees(query, max_count=3)}
    assert results == expected_results
    results_2 = {tree.interpret() for tree in solution_space.depth_first_resolution(query, max_count=3)}
    assert results_2 == expected_results
    results_3 = {tree.interpret() for tree in solution_space.breadth_first_resolution(query, max_count=3)}
    assert results_3 == expected_results

    for tree in solution_space.enumerate_trees(query, max_count=3):
        assert solution_space.contains_tree(query, tree)
    for tree in solution_space.depth_first_resolution(query, max_count=3):
        assert solution_space.contains_tree(query, tree)
    for tree in solution_space.breadth_first_resolution(query, max_count=3):
        assert solution_space.contains_tree(query, tree)
