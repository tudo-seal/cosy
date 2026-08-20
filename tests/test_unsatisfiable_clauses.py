"""Rules that a predicate already forbids at construction time must not enter the solution space."""

from collections.abc import Mapping
from typing import Any

from cosy.core.solution_space import ConstantArgument, NonTerminalArgument, SolutionSpace
from cosy.core.specification_builder import SpecificationBuilder
from cosy.core.synthesizer import Synthesizer
from cosy.core.tree import Tree
from cosy.core.types import Arrow, Constructor, DataGroup, Intersection, Type

DIGITS = DataGroup("digits", range(10))
JUST_ZERO = DataGroup("just_zero", [0])
JUST_NINE = DataGroup("just_nine", [9])

DIGIT: Type = Constructor("Digit")
START: Type = Constructor("Start")
EXTRA: Type = Constructor("Extra")
LEFT: Type = Constructor("Left")
RIGHT: Type = Constructor("Right")


def digit(d: int) -> str:
    """Turn a literal value into a term."""
    return f"digit {d}"


def wrap(t: str) -> str:
    """Wrap a digit term into another digit term."""
    return f"wrap ({t})"


def use(t: str) -> str:
    """Lift a digit term to the start symbol."""
    return f"use ({t})"


def tag(d: int, t: str) -> str:
    """Lift a digit term to the start symbol and label it with a literal value."""
    return f"tag {d} ({t})"


def name_digit(k: int, t: str) -> str:
    """Label a digit term with a literal value."""
    return f"named {k} as {t}"


def extra() -> str:
    """Provide a term for a type that is only reachable through a rejected rule."""
    return "extra"


def either(d: int, t: str) -> str:
    """Consume one of two alternative argument types."""
    return f"either {d} ({t})"


def above_seven(values: Mapping[str, Any]) -> bool:
    """Accept only literal values greater than seven."""
    return values["d"] > 7


def below_nine(values: Mapping[str, Any]) -> bool:
    """Accept only literal values smaller than nine."""
    return values["d"] < 9


def names_its_digit(values: Mapping[str, Any]) -> bool:
    """Accept a term argument only if it interprets to the digit named by the literal parameter."""
    return values["x"].interpret() == f"digit {values['k']}"


def never(_values: Mapping[str, Any]) -> bool:
    """Reject every substitution, including the empty one."""
    return False


def negative_key(values: Mapping[str, Any]) -> bool:
    """Reject every literal value, while reading nothing but literal parameters."""
    return values["k"] < 0


def digit_space() -> SolutionSpace:
    """Synthesize `Digit` from ten literal values, eight of which the predicate rejects."""
    specifications = {digit: SpecificationBuilder().parameter("d", DIGITS).constraint(above_seven).suffix(DIGIT)}
    return Synthesizer(specifications).construct_solution_space(DIGIT)


def test_rejected_literals_yield_no_rules() -> None:
    """Only the two admissible literal values reach the solution space."""
    space = digit_space()
    assert list(space.nonterminals()) == [DIGIT]
    assert {rule.arguments for rule in space[DIGIT]} == {
        (ConstantArgument("d", 8, DIGITS),),
        (ConstantArgument("d", 9, DIGITS),),
    }
    assert {rule.terminal for rule in space[DIGIT]} == {digit}


def test_a_rule_survives_only_if_all_of_its_predicates_hold() -> None:
    """The predicates of a rule are a conjunction, so a single rejecting one is enough to drop it."""
    specifications = {
        digit: SpecificationBuilder()
        .parameter("d", DIGITS)
        .constraint(above_seven)
        .constraint(below_nine)
        .suffix(DIGIT)
    }
    space = Synthesizer(specifications).construct_solution_space(DIGIT)
    assert [rule.literal_substitution for rule in space[DIGIT]] == [{"d": 8}]


def test_a_predicate_reading_no_literal_at_all_is_decided_as_well() -> None:
    """Without literal parameters the substitution is the empty one, which is still complete."""
    specifications = {extra: SpecificationBuilder().constraint(never).suffix(EXTRA)}
    space = Synthesizer(specifications).construct_solution_space(EXTRA)
    assert list(space.nonterminals()) == []


def test_result_sets_are_unchanged_for_rules_without_arguments() -> None:
    """A rule whose arguments are all constant was already rejected on every query; only the rule is gone."""
    expected = {"digit 8", "digit 9"}
    assert {tree.interpret() for tree in digit_space().enumerate_trees(DIGIT)} == expected
    assert {tree.interpret() for tree in digit_space().depth_first_resolution(DIGIT)} == expected
    assert {tree.interpret() for tree in digit_space().breadth_first_resolution(DIGIT)} == expected
    assert digit_space().contains_tree(DIGIT, Tree(digit, (Tree(8),)))
    assert not digit_space().contains_tree(DIGIT, Tree(digit, (Tree(3),)))


def test_resolution_of_a_rule_with_an_anonymous_argument_agrees_with_enumeration() -> None:
    """A predicate on a rule whose only non-terminal argument is anonymous now decides the rule everywhere."""
    specifications = {
        digit: SpecificationBuilder().parameter("d", JUST_NINE).suffix(DIGIT),
        tag: SpecificationBuilder().parameter("d", DIGITS).constraint(above_seven).suffix(Arrow(DIGIT, START)),
    }
    space = Synthesizer(specifications).construct_solution_space(START)
    assert {rule.arguments for rule in space[START]} == {
        (ConstantArgument("d", 8, DIGITS), NonTerminalArgument(None, DIGIT)),
        (ConstantArgument("d", 9, DIGITS), NonTerminalArgument(None, DIGIT)),
    }
    expected = {"tag 8 (digit 9)", "tag 9 (digit 9)"}
    assert {tree.interpret() for tree in space.enumerate_trees(START)} == expected
    assert {tree.interpret() for tree in space.depth_first_resolution(START, max_count=10, max_depth=10)} == expected
    assert {tree.interpret() for tree in space.breadth_first_resolution(START, max_count=10, max_depth=10)} == expected


def test_unproductive_nonterminal_is_pruned() -> None:
    """A non-terminal whose only rule is rejected is gone, and everything depending on it is unproductive."""
    specifications = {
        digit: SpecificationBuilder().parameter("d", JUST_ZERO).constraint(above_seven).suffix(DIGIT),
        use: Arrow(DIGIT, START),
    }
    space = Synthesizer(specifications).construct_solution_space(START)
    assert list(space.nonterminals()) == [START]
    assert list(space.prune().nonterminals()) == []


def test_recursive_unproductive_nonterminal_terminates() -> None:
    """`Digit -> zero` rejected plus `Digit -> wrap(Digit)` leaves nothing to descend into."""
    specifications = {
        digit: SpecificationBuilder().parameter("d", JUST_ZERO).constraint(above_seven).suffix(DIGIT),
        wrap: Arrow(DIGIT, DIGIT),
    }
    space = Synthesizer(specifications).construct_solution_space(DIGIT)
    assert [rule.terminal for rule in space[DIGIT]] == [wrap]
    pruned = space.prune()
    assert list(pruned.nonterminals()) == []
    assert list(pruned.depth_first_resolution(DIGIT, max_count=1)) == []
    # Even unpruned, no derivation of bounded depth succeeds; the bound only keeps a regression finite.
    assert list(space.depth_first_resolution(DIGIT, max_count=1, max_depth=64)) == []


def test_a_rejected_rule_does_not_make_its_argument_types_targets() -> None:
    """The rule is dropped before its anonymous arguments are turned into further synthesis targets."""
    specifications = {
        tag: SpecificationBuilder().parameter("d", JUST_ZERO).constraint(above_seven).suffix(Arrow(EXTRA, DIGIT)),
        extra: EXTRA,
    }
    space = Synthesizer(specifications).construct_solution_space(DIGIT)
    assert list(space.nonterminals()) == []


def test_a_rejected_instantiation_is_rejected_for_every_minimal_cover() -> None:
    """An intersection suffix offers several ways to cover the target; the predicate rules out all of them."""
    specifications = {
        either: SpecificationBuilder()
        .parameter("d", JUST_ZERO)
        .constraint(above_seven)
        .suffix(Intersection(Arrow(LEFT, START), Arrow(RIGHT, START))),
        digit: SpecificationBuilder().parameter("d", JUST_NINE).suffix(Intersection(LEFT, RIGHT)),
    }
    space = Synthesizer(specifications).construct_solution_space(START)
    assert list(space.nonterminals()) == []


def test_a_rejected_rule_leaves_the_other_rules_for_its_target_alone() -> None:
    """Rejecting one combinator for a target must not affect the combinators considered after it."""
    specifications = {
        digit: SpecificationBuilder().parameter("d", JUST_ZERO).constraint(above_seven).suffix(START),
        name_digit: SpecificationBuilder().parameter("k", JUST_NINE).argument("x", DIGIT).suffix(START),
        wrap: SpecificationBuilder().parameter("d", JUST_NINE).suffix(DIGIT),
    }
    space = Synthesizer(specifications).construct_solution_space(START)
    assert [rule.terminal for rule in space[START]] == [name_digit]
    assert {tree.interpret() for tree in space.enumerate_trees(START)} == {"named 9 as wrap (9)"}


def test_predicate_over_a_term_argument_keeps_its_rules() -> None:
    """A predicate reading a named non-terminal argument is undecided at construction time, so its rule stays."""
    specifications = {
        digit: SpecificationBuilder().parameter("d", DIGITS).suffix(DIGIT),
        name_digit: SpecificationBuilder()
        .parameter("k", JUST_NINE)
        .argument("x", DIGIT)
        .constraint(names_its_digit)
        .suffix(START),
    }
    space = Synthesizer(specifications).construct_solution_space(START)
    assert [rule.arguments for rule in space[START]] == [
        (ConstantArgument("k", 9, JUST_NINE), NonTerminalArgument("x", DIGIT)),
    ]
    assert {rule.arguments for rule in space[DIGIT]} == {(ConstantArgument("d", value, DIGITS),) for value in range(10)}
    assert {tree.interpret() for tree in space.enumerate_trees(START)} == {"named 9 as digit 9"}


def test_a_rule_with_a_term_argument_stays_even_if_its_predicate_reads_only_literals() -> None:
    """The filter asks whether the answer is already fixed, not whether the predicate happens to read literals."""
    specifications = {
        digit: SpecificationBuilder().parameter("d", DIGITS).suffix(DIGIT),
        name_digit: SpecificationBuilder()
        .parameter("k", JUST_NINE)
        .argument("x", DIGIT)
        .constraint(negative_key)
        .suffix(START),
    }
    space = Synthesizer(specifications).construct_solution_space(START)
    assert [rule.terminal for rule in space[START]] == [name_digit]
    assert list(space.enumerate_trees(START)) == []
