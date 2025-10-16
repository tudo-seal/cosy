# test of the DSL for non-inferrable infinite parameter spaces

from collections.abc import Iterator

import pytest
from cosy.dsl import DSL
from cosy.solution_space import SolutionSpace
from cosy.synthesizer import Synthesizer
from cosy.tree import Tree
from cosy.types import Constructor, Group, Var


def test_infinite_enumeration() -> None:
    # literal varibles can be assigned computed values
    def c(x: int, y: int, t1: str, t2: str, t3: str) -> str:
        return f"(C {x} {y} {t1} {t2} {t3})"

    def d(x: int) -> str:
        return f"(D {x})"

    def e(x: int) -> str:
        return f"(E {x})"

    def f() -> str:
        return "F"

    # infinite enumeration of natural numbers
    class Nat(Group):
        name = "nat"

        def __iter__(self) -> Iterator[int]:
            i: int = 0
            while True:
                yield i
                i += 1

        def __contains__(self, value: object) -> bool:
            return isinstance(value, int) and value >= 0

    component_specifications = {
        c: DSL()
        .parameter("x", Nat())
        .parameter("y", Nat())
        .argument("t1", Var("x"))
        .argument("t2", Var("y"))
        .suffix(Constructor("a") ** Constructor("a")),
        d: DSL().parameter("x", Nat()).suffix(Var("x")),
        e: DSL().parameter("y", Nat()).suffix(Var("y")),
        f: DSL().suffix(Constructor("a")),
    }

    synthesizer = Synthesizer(component_specifications)
    target = Constructor("a")

    # intermediate solution space
    solution_space: SolutionSpace = SolutionSpace()
    bound = 20

    tree1 = Tree(f)
    tree2 = Tree(c, [Tree(2), Tree(1), Tree(d, [Tree(2)]), Tree(e, [Tree(1)]), Tree(f)])
    tree3 = Tree(
        c,
        [
            Tree(0),
            Tree(1),
            Tree(d, [Tree(0)]),
            Tree(e, [Tree(1)]),
            Tree(c, [Tree(1), Tree(0), Tree(e, [Tree(1)]), Tree(e, [Tree(0)]), Tree(f)]),
        ],
    )

    trees_of_interest = frozenset([tree1, tree2, tree3])

    # iterate over (infinitely many) rules which in the solutions space
    for i, (nt, rule) in enumerate(synthesizer.construct_solution_space_rules(target)):
        # add the rule to the intermediate solution space
        solution_space.add_rule(nt, rule.terminal, rule.arguments, rule.predicates)
        if all(solution_space.contains_tree(target, tree) for tree in trees_of_interest):
            return
        if i > bound:
            pytest.fail(f"Lazy enumeration did not find solutions of interest after {bound} iterations.")
