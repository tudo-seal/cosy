"""Tests for reading a search node as the term it denotes.

A goal records an expanded position in ``constructors`` while the position itself stays in
``subgoals``. These tests pin that the holes are the unexpanded positions only, that the
materialized term shows the prescribed structure around them, and that both measures read a hole
as a variable rather than as a symbol.
"""

import sys
from typing import Any

from cosy.core.solution_space import Goal, NonTerminalArgument
from cosy.core.tree import Tree
from cosy.search import Hole, checker, holes, partial_inhabitant, term_depth, term_size
from tests.search_fixtures import (
    EXPR,
    NUM,
    WORD,
    add,
    expression_space,
    lit,
    literal_space,
    neg,
    plus,
)


def a_goal_for(space, tree, pos):
    """Return one goal of the partial-term query at a position.

    Any goal will do for the properties tested here, so the count is left open. The engine may
    offer one goal per clause that reaches the position.

    Args:
        space (SolutionSpace): The space to query.
        tree (Tree): The prescribed term.
        pos (Path): The position that becomes the hole.

    Returns:
        Goal: The first goal the query offers.
    """
    goals = list(space.goal_from_tree(EXPR, tree, pos))
    assert goals, f"the fixture produced no goal at {pos}, so it can prove nothing"
    return goals[0]


def test_the_open_position_is_the_hole():
    """A goal derived for one position has exactly that position open."""
    space = expression_space()
    tree = Tree(add, (Tree(lit, ()), Tree(lit, ())))

    goal = a_goal_for(space, tree, (0,))

    assert set(holes(goal)) == {(0,)}


def test_an_expanded_position_is_no_hole():
    """A position that already carries a symbol is not open work.

    ``Goal`` keeps an expanded position in ``subgoals`` until its whole subtree grounds, so
    reading every subgoal as a hole would report a position that has a symbol already, and a
    computation rule fed from that reading would derive it a second time.
    """
    space = expression_space()
    tree = Tree(add, (Tree(neg, (Tree(lit, ()),)), Tree(lit, ())))

    goal = a_goal_for(space, tree, (0, 0))

    assert (0,) in goal.subgoals, "the fixture must carry an expanded subgoal to prove anything"
    assert (0,) in goal.constructors
    assert set(holes(goal)) == {(0, 0)}


def test_a_position_that_grounds_leaves_the_subgoals():
    """A position whose subtree is complete is no longer open work.

    ``holes`` reads ``subgoals`` and filters by ``constructors`` alone, which is only the open
    positions because a position that grounds is removed from ``subgoals`` in the same step.
    """
    space = expression_space()
    tree = Tree(add, (Tree(lit, ()), Tree(lit, ())))

    goal = a_goal_for(space, tree, (0,))

    assert goal.grounded, "the fixture must have a grounded position to prove anything"
    assert not set(goal.grounded) & set(goal.subgoals)


def test_every_open_position_is_a_hole_of_its_own():
    """A goal with two open arguments reports both, at their own positions."""
    space = expression_space()
    binary_rule = next(rule for rule in space[EXPR] if len(rule.arguments) == 2)

    goal = Goal.from_rhs_rule(binary_rule)

    assert goal is not None
    assert set(holes(goal)) == {(0,), (1,)}
    term = partial_inhabitant(goal)
    assert [child.root for child in term.children] == [Hole((0,), EXPR), Hole((1,), EXPR)]
    assert term_size(term) == 1


def test_the_hole_names_the_non_terminal_its_completions_come_from():
    """The hole carries the sort a completion has to inhabit."""
    space = expression_space()
    tree = Tree(add, (Tree(lit, ()), Tree(lit, ())))

    goal = a_goal_for(space, tree, (0,))
    ((position, nonterminal),) = holes(goal).items()

    assert position == (0,)
    assert checker(space, nonterminal, Tree(lit, ())), "lit must inhabit the hole's non-terminal"


def test_a_success_goal_is_ground():
    """A goal that has found its solution has no holes left."""
    space = expression_space()
    rules = space.get(EXPR)
    assert rules is not None
    constant_rule = next(rule for rule in rules if not rule.arguments)

    goal = Goal.from_rhs_rule(constant_rule)

    assert goal is not None
    assert goal.success
    assert holes(goal) == {}
    assert partial_inhabitant(goal) == goal.grounded[()][1]


def test_the_materialized_term_keeps_the_prescribed_structure():
    """Around the hole the partial inhabitant is the term the query prescribed."""
    space = expression_space()
    tree = Tree(add, (Tree(neg, (Tree(lit, ()),)), Tree(lit, ())))

    term = partial_inhabitant(a_goal_for(space, tree, (0, 0)))

    assert term.root is add
    assert term.children[1].root is lit
    assert term.children[0].root is neg
    hole = term.children[0].children[0].root
    assert isinstance(hole, Hole)
    assert hole.position == (0, 0)


def test_the_hole_of_the_term_is_the_hole_of_the_goal():
    """Every hole of the goal shows up as a Hole leaf at its own position, and no other does."""
    space = expression_space()
    tree = Tree(add, (Tree(neg, (Tree(lit, ()),)), Tree(lit, ())))
    goal = a_goal_for(space, tree, (0, 0))

    term = partial_inhabitant(goal)
    marked = {
        position: term.subtree_at(position).root
        for position in term.positions()
        if isinstance(term.subtree_at(position).root, Hole)
    }

    assert set(marked) == set(holes(goal))
    for position, hole in marked.items():
        assert hole.position == position
        assert hole.nonterminal == holes(goal)[position]


def test_two_holes_of_one_term_are_distinct():
    """Position and non-terminal both belong to a hole's identity.

    Two holes of one term differ in their position, and two holes of one position differ in the
    sort a completion has to inhabit.
    """
    assert Hole((0,), EXPR) == Hole((0,), EXPR)
    assert Hole((0,), EXPR) != Hole((1,), EXPR)
    assert Hole((0,), EXPR) != Hole((0,), WORD)
    assert len({Hole((0,), EXPR), Hole((1,), EXPR), Hole((0,), WORD)}) == 3


def test_term_size_counts_symbols_and_a_hole_is_none():
    """A hole is a variable, so it adds no symbol, unlike a node, which ``Tree.size`` counts."""
    ground = Tree(add, (Tree(lit, ()), Tree(lit, ())))
    with_hole = Tree(add, (Tree(Hole((0,), EXPR), ()), Tree(lit, ())))

    assert term_size(ground) == ground.size == 3
    assert with_hole.size == 3
    assert term_size(with_hole) == 2


def test_term_size_of_a_lopsided_term_counts_every_branch():
    """The measure sums over all children, not along one path."""
    term = Tree(add, (Tree(neg, (Tree(lit, ()),)), Tree(add, (Tree(lit, ()), Tree(lit, ())))))

    assert term_size(term) == 6
    assert term_size(term) == term.size, "on a ground term the measure is the node count"


def test_term_depth_measures_the_longest_path():
    """The depth is the longest root-to-leaf path, not the shortest and not the node count."""
    lopsided = Tree(add, (Tree(neg, (Tree(neg, (Tree(lit, ()),)),)), Tree(lit, ())))

    assert term_depth(Tree(lit, ())) == 0
    assert term_depth(Tree(neg, (Tree(lit, ()),))) == 1
    assert term_depth(lopsided) == 3


def test_term_depth_finds_the_deepest_branch_wherever_it_sits():
    """The depth does not depend on which branch the longest path runs through.

    The walk pops its stack, so the branch it visits last is a property of the traversal. Taking
    the level of the last node rather than the largest agrees with the depth on one of the two
    terms here and not on the other.
    """
    spine = Tree(neg, (Tree(neg, (Tree(lit, ()),)),))

    assert term_depth(Tree[Any](add, (spine, Tree(lit, ())))) == 3
    assert term_depth(Tree[Any](add, (Tree(lit, ()), spine))) == 3


def test_a_hole_ends_the_path_it_sits_on():
    """A partial inhabitant reaches as deep as its holes and no deeper."""
    term = Tree(neg, (Tree(Hole((0,), EXPR), ()),))

    assert term_depth(term) == 1
    assert term_size(term) == 1


def deep_spine(depth):
    """Build a goal whose expanded spine is deeper than the interpreter's recursion limit.

    Args:
        depth (int): The number of expanded positions above the hole.

    Returns:
        Goal: The goal, with its only hole at the bottom of the spine.
    """
    constructors = {(0,) * level: neg for level in range(depth)}
    subgoals = {(0,) * level: NonTerminalArgument(None, EXPR) for level in range(1, depth + 1)}
    return Goal(constructors, subgoals, {}, {}, False)


def test_a_goal_deeper_than_the_recursion_limit_is_materialized():
    """Materializing a partial inhabitant walks the goal iteratively.

    The spine of a goal grows with the derivation, and a term deeper than the interpreter's
    recursion limit is reachable long before any bound of the search stops it.
    """
    depth = sys.getrecursionlimit() * 2
    goal = deep_spine(depth)

    term = partial_inhabitant(goal)

    # walked rather than compared: Tree equality descends through its children tuples
    node = term
    for _ in range(depth):
        assert node.root is neg
        node = node.children[0]
    assert isinstance(node.root, Hole)
    assert node.root.position == (0,) * depth


def test_a_term_deeper_than_the_recursion_limit_is_measured():
    """Both measures walk the term iteratively, for the same reason."""
    depth = sys.getrecursionlimit() * 2
    term = Tree(lit, ())
    for _ in range(depth):
        term = Tree(neg, (term,))

    assert term_size(term) == depth + 1
    assert term_depth(term) == depth


def test_a_constant_argument_is_materialized_from_the_grounded_map():
    """A constant argument is grounded from the start and never carries a constructor.

    A walk that rebuilt every position from ``constructors`` would not find it, and one that
    rebuilt it from its symbol would put an empty argument list where the value stands.
    """
    space = literal_space()
    recursive = next(rule for rule in space[NUM] if any(isinstance(a, NonTerminalArgument) for a in rule.arguments))

    goal = Goal.from_rhs_rule(recursive)

    assert goal is not None
    assert set(goal.grounded) == {(0,)}, "the constant argument has to be grounded from the start"
    assert (0,) not in goal.constructors
    term = partial_inhabitant(goal)
    assert term.root is plus
    assert term.children[0] == goal.grounded[(0,)][1]
    assert isinstance(term.children[1].root, Hole)


def test_a_grounded_subtree_is_taken_whole():
    """A position that grounds contributes its subtree, not its symbol applied to nothing."""
    space = literal_space()
    rules = list(space[NUM])
    recursive = next(rule for rule in rules if any(isinstance(a, NonTerminalArgument) for a in rule.arguments))
    ground = next(rule for rule in rules if not any(isinstance(a, NonTerminalArgument) for a in rule.arguments))
    goal = Goal.from_rhs_rule(recursive)
    assert goal is not None

    completed = goal.update(ground, next(iter(goal.subgoals)))

    assert completed is not None
    assert completed.success
    assert partial_inhabitant(completed) == completed.grounded[()][1]
    assert holes(completed) == {}
