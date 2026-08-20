"""Regression tests for predicates of rules without a named non-terminal argument.

The predicates of a rule are deposited under the tuple of its *named* non-terminal positions, so a
rule without such a position deposits nothing and the resolution never gets to evaluate them. Two
shapes go unchecked that way: a rule whose non-terminal arguments are all unnamed -- the shape an
arrow type in a suffix produces -- at every position, and a ground rule at every position below the
root, where the resolution used to decide it directly. `enumerate_trees` and `contains_tree` decide
such predicates on the literal substitution of the rule, so the three procedures used to disagree.
"""

import random
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pytest

from cosy.core import Constructor, SpecificationBuilder, Synthesizer, Var
from cosy.core.solution_space import ConstantArgument, NonTerminalArgument, SolutionSpace
from cosy.core.tree import Tree
from cosy.core.types import Group

Term = Tree[Any]
Repository = SolutionSpace[str, Any, None]

LARGEST_ALLOWED_DIGIT = 1


def positive_digit(substitution: dict[str, Any]) -> bool:
    """Read the constant argument `d`, which needs no non-terminal argument to be decided.

    Args:
        substitution (dict[str, Any]): The substitution the rule supplies for its predicates.

    Returns:
        bool: True if the digit is greater than zero.
    """
    return substitution["d"] > 0


def small_digit(substitution: dict[str, Any]) -> bool:
    """Read the same constant argument `d`, and reject the digits `positive_digit` accepts.

    Args:
        substitution (dict[str, Any]): The substitution the rule supplies for its predicates.

    Returns:
        bool: True if the digit does not exceed `LARGEST_ALLOWED_DIGIT`.
    """
    return substitution["d"] <= LARGEST_ALLOWED_DIGIT


def leftmost_leaf(substitution: dict[str, Any]) -> bool:
    """Read the named non-terminal argument `t`, which is a derived subterm.

    Args:
        substitution (dict[str, Any]): The substitution the rule supplies for its predicates.

    Returns:
        bool: True if the subterm bound to `t` is the leaf `a`.
    """
    return substitution["t"].root == "a"


def use(digit_value: int, leaf: str) -> Term:
    """Build `use digit_value leaf`.

    Args:
        digit_value (int): The value of the constant argument `d`.
        leaf (str): The terminal derived for the unnamed non-terminal argument.

    Returns:
        Term: The application `use digit_value leaf`.
    """
    return Tree("use", (Tree(digit_value, ()), Tree(leaf, ())))


def digit(value: int) -> Term:
    """Build `digit value`.

    Args:
        value (int): The value of the constant argument `d`.

    Returns:
        Term: The application `digit value`.
    """
    return Tree("digit", (Tree(value, ()),))


def wrap(terminal: str, subterm: Term) -> Term:
    """Build a unary application of `terminal`.

    Args:
        terminal (str): The terminal at the root of the result.
        subterm (Term): The single argument of `terminal`.

    Returns:
        Term: The application `terminal subterm`.
    """
    return Tree(terminal, (subterm,))


def unnamed_hole_space() -> Repository:
    """`S ~> use(d, _)`: a predicate on `d`, and the single non-terminal argument is unnamed.

    Returns:
        Repository: A solution space whose start symbol carries the rule with the unnamed hole.
    """
    space: Repository = SolutionSpace()
    for value in (0, 1, 2):
        arguments = (ConstantArgument("d", value, None), NonTerminalArgument(None, "V"))
        space.add_rule("S", "use", arguments, (positive_digit,))
    space.add_rule("V", "a", (), ())
    space.add_rule("V", "b", (), ())
    return space


def nested_unnamed_hole_space() -> Repository:
    """The same rule one level down: `S ~> box(x: U)` and `U ~> use(d, _)`.

    Below the root the rule is applied by `update` rather than by `from_rhs_rule`, and it is not
    ground -- its unnamed argument stays an open subgoal.

    Returns:
        Repository: A solution space whose rule with the unnamed hole sits below the start symbol.
    """
    space: Repository = SolutionSpace()
    for value in (0, 1, 2):
        arguments = (ConstantArgument("d", value, None), NonTerminalArgument(None, "V"))
        space.add_rule("U", "use", arguments, (positive_digit,))
    space.add_rule("S", "box", (NonTerminalArgument("x", "U"),), ())
    space.add_rule("V", "a", (), ())
    space.add_rule("V", "b", (), ())
    return space


def ground_rule_space() -> Repository:
    """`D ~> digit(d)`: a predicate on `d`, and no non-terminal argument at all.

    `S ~> box(x: D)` puts the very same rule one level below the root.

    Returns:
        Repository: A solution space reachable at the root as `D` and below the root as `S`.
    """
    space: Repository = SolutionSpace()
    for value in (0, 1, 2):
        space.add_rule("D", "digit", (ConstantArgument("d", value, None),), (positive_digit,))
    space.add_rule("S", "box", (NonTerminalArgument("x", "D"),), ())
    return space


def two_predicate_space() -> Repository:
    """`T ~> use(d, _)` carrying two predicates, of which only the second rejects `d == 2`.

    A rule with a single predicate cannot tell a conjunction over the predicates from a disjunction,
    nor from a check of the first predicate alone.

    Returns:
        Repository: A solution space reachable at the root as `T` and below the root as `S`.
    """
    space: Repository = SolutionSpace()
    for value in (0, 1, 2):
        arguments = (ConstantArgument("d", value, None), NonTerminalArgument(None, "V"))
        space.add_rule("T", "use", arguments, (positive_digit, small_digit))
    space.add_rule("S", "box", (NonTerminalArgument("x", "T"),), ())
    space.add_rule("V", "a", (), ())
    space.add_rule("V", "b", (), ())
    return space


def named_argument_space() -> Repository:
    """`P ~> pick(t)`: a predicate reading a named non-terminal argument, at the root and below it.

    Returns:
        Repository: A solution space reachable at the root as `P` and below the root as `Q`.
    """
    space: Repository = SolutionSpace()
    space.add_rule("Q", "hold", (NonTerminalArgument("p", "P"),), ())
    space.add_rule("P", "pick", (NonTerminalArgument("t", "V"),), (leftmost_leaf,))
    space.add_rule("V", "a", (), ())
    space.add_rule("V", "b", (), ())
    return space


def mixed_space() -> Repository:
    """All three shapes in one repository: an unnamed hole, a ground rule below the root, a term predicate.

    Returns:
        Repository: A solution space in which every rule shape is reachable from the start symbol `S`.
    """
    space: Repository = SolutionSpace()
    for value in (0, 1):
        arguments = (ConstantArgument("d", value, None), NonTerminalArgument(None, "V"))
        space.add_rule("S", "use", arguments, (positive_digit,))
        space.add_rule("D", "digit", (ConstantArgument("d", value, None),), (positive_digit,))
    space.add_rule("S", "box", (NonTerminalArgument("x", "D"),), ())
    space.add_rule("S", "pick", (NonTerminalArgument("t", "V"),), (leftmost_leaf,))
    space.add_rule("V", "a", (), ())
    space.add_rule("V", "b", (), ())
    return space


UNCHECKED_PREDICATE_CASES = [
    pytest.param(
        unnamed_hole_space,
        "S",
        {use(1, "a"), use(1, "b"), use(2, "a"), use(2, "b")},
        {use(0, "a"), use(0, "b")},
        id="unnamed-holes-at-root",
    ),
    pytest.param(
        nested_unnamed_hole_space,
        "S",
        {wrap("box", use(value, leaf)) for value in (1, 2) for leaf in ("a", "b")},
        {wrap("box", use(0, leaf)) for leaf in ("a", "b")},
        id="unnamed-holes-below-root",
    ),
    pytest.param(
        ground_rule_space,
        "D",
        {digit(1), digit(2)},
        {digit(0)},
        id="ground-rule-at-root",
    ),
    pytest.param(
        ground_rule_space,
        "S",
        {wrap("box", digit(1)), wrap("box", digit(2))},
        {wrap("box", digit(0))},
        id="ground-rule-below-root",
    ),
    pytest.param(
        two_predicate_space,
        "T",
        {use(1, "a"), use(1, "b")},
        {use(value, leaf) for value in (0, 2) for leaf in ("a", "b")},
        id="two-predicates-at-root",
    ),
    pytest.param(
        two_predicate_space,
        "S",
        {wrap("box", use(1, "a")), wrap("box", use(1, "b"))},
        {wrap("box", use(value, leaf)) for value in (0, 2) for leaf in ("a", "b")},
        id="two-predicates-below-root",
    ),
]

RESOLUTIONS = ["depth_first_resolution", "breadth_first_resolution"]


@pytest.mark.parametrize(("build", "start", "expected", "forbidden"), UNCHECKED_PREDICATE_CASES)
@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_predicate_without_named_argument_is_enforced(
    resolution: str,
    build: Callable[[], Repository],
    start: str,
    expected: set[Term],
    forbidden: set[Term],
) -> None:
    """The resolution derives exactly the terms the predicates allow, and `contains_tree` agrees.

    Args:
        resolution (str): Name of the resolution procedure under test.
        build (Callable[[], Repository]): Builds a fresh solution space for the case.
        start (str): The non-terminal the derivation starts from.
        expected (set[Term]): The terms the predicates of the repository allow.
        forbidden (set[Term]): Terms of the same shape that the predicates reject.
    """
    space = build()
    assert set(getattr(space, resolution)(start, max_count=20)) == expected
    assert {term for term in expected | forbidden if space.contains_tree(start, term)} == expected


@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_predicate_is_enforced_when_resolving_into_a_skeleton(resolution: str) -> None:
    """Resolving an open position of a given term checks the same predicates.

    Args:
        resolution (str): Name of the resolution procedure under test.
    """
    space = ground_rule_space()
    skeleton = wrap("box", digit(0))
    derived = set(getattr(space, resolution)("S", tree=skeleton, pos=(0,)))
    assert derived == {wrap("box", digit(1)), wrap("box", digit(2))}


def test_sample_tree_never_draws_a_forbidden_term() -> None:
    """Every draw satisfies the predicate, and the seeds do reach more than a single term."""
    drawn = {unnamed_hole_space().sample_tree("S", rng=random.Random(seed)) for seed in range(20)}
    assert drawn <= {use(1, "a"), use(1, "b"), use(2, "a"), use(2, "b")}
    assert len(drawn) > 1


def test_sample_tree_into_a_skeleton_never_draws_a_forbidden_term() -> None:
    """Sampling into an open position of a given term satisfies the predicate as well."""
    skeleton = wrap("box", digit(0))
    drawn = {
        ground_rule_space().sample_tree("S", tree=skeleton, pos=(0,), rng=random.Random(seed)) for seed in range(20)
    }
    assert drawn <= {wrap("box", digit(1)), wrap("box", digit(2))}


def test_resolution_enumeration_and_membership_agree() -> None:
    """All three procedures answer the same for one repository carrying all three rule shapes."""
    space = mixed_space()
    expected = {use(1, "a"), use(1, "b"), wrap("box", digit(1)), wrap("pick", Tree("a", ()))}
    forbidden = {use(0, "a"), use(0, "b"), wrap("box", digit(0)), wrap("pick", Tree("b", ()))}

    assert set(space.depth_first_resolution("S", max_count=20)) == expected
    assert set(space.breadth_first_resolution("S", max_count=20)) == expected
    assert set(space.enumerate_trees("S", max_count=20)) == expected
    assert {term for term in expected | forbidden if space.contains_tree("S", term)} == expected


@pytest.mark.parametrize(
    ("start", "expected"),
    [("P", {wrap("pick", Tree("a", ()))}), ("Q", {wrap("hold", wrap("pick", Tree("a", ())))})],
)
@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_predicate_reading_a_named_argument_is_unaffected(resolution: str, start: str, expected: set[Term]) -> None:
    """A predicate over a derived subterm keeps being decided on the subterm, not on the literals.

    Args:
        resolution (str): Name of the resolution procedure under test.
        start (str): The non-terminal the derivation starts from.
        expected (set[Term]): The terms the predicate over the subterm allows.
    """
    space = named_argument_space()
    assert set(getattr(space, resolution)(start, max_count=20)) == expected


class Digits(Group):
    """The literal parameters 0, 1 and 2."""

    name = "Digits"

    def __iter__(self) -> Iterator[int]:
        """Enumerate the group.

        Yields:
            int: The digits 0, 1 and 2.
        """
        yield from (0, 1, 2)

    def __contains__(self, x: Any) -> bool:
        """Membership in the group.

        Args:
            x (Any): The candidate value.

        Returns:
            bool: True if the value is one of the digits 0, 1 and 2.
        """
        return isinstance(x, int) and x in (0, 1, 2)


def test_predicate_on_an_arrow_in_a_suffix_is_enforced() -> None:
    """A specification whose suffix is an arrow type produces the rule with only unnamed holes.

    The synthesizer names only the arguments a component declares with `argument`; the arguments it
    derives from the arrow in the suffix stay unnamed. `F` therefore ends up with a predicate, a
    constant argument and an unnamed non-terminal argument -- and no named one.
    """

    def positive_parameter(substitution: Mapping[str, Any]) -> bool:
        """Read the literal parameter `n` of the component `F`.

        Args:
            substitution (Mapping[str, Any]): The substitution the rule supplies for its predicates.

        Returns:
            bool: True if the parameter is greater than zero.
        """
        return substitution["n"] > 0

    repository = {
        "F": SpecificationBuilder()
        .parameter("n", Digits())
        .constraint(positive_parameter)
        .suffix(Constructor("x") ** Constructor("d", Var("n"))),
        "G": SpecificationBuilder().suffix(Constructor("x")),
        "H": SpecificationBuilder()
        .parameter("k", Digits())
        .argument("y", Constructor("d", Var("k")))
        .suffix(Constructor("top")),
    }
    synthesizer: Synthesizer[Any] = Synthesizer(repository)
    target = Constructor("top")
    space = synthesizer.construct_solution_space(target)

    rules_with_predicates = [rule for _, rules in space.as_tuples() for rule in rules if rule.predicates]
    assert rules_with_predicates
    for rule in rules_with_predicates:
        arguments = rule.arguments
        assert any(isinstance(a, NonTerminalArgument) for a in arguments)
        assert not any(isinstance(a, NonTerminalArgument) and a.name is not None for a in arguments)

    expected = {"H 1 (F 1 G)", "H 2 (F 2 G)"}
    assert {str(term) for term in space.depth_first_resolution(target, 20)} == expected
    assert {str(term) for term in space.breadth_first_resolution(target, 20)} == expected
    assert {str(term) for term in space.enumerate_trees(target, 20)} == expected
    for term in space.enumerate_trees(target, 20):
        assert space.contains_tree(target, term)
    forbidden = Tree("H", (Tree(0, ()), Tree("F", (Tree(0, ()), Tree("G", ())))))
    assert not space.contains_tree(target, forbidden)
