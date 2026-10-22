"""Membership decides each pair of a non-terminal and a subtree once.

A term derivable in several ways asks the same question of the same subtree through every rule that matches above
it. Without remembering the answers the work multiplies with the depth -- two derivations a level make ``2 ** depth``
questions of the leaf -- and with them it is one decision a pair. The answers are the derivability procedure's
throughout: the tests hold them against the definition, written out recursively and without memory, on members and
non-members of spaces with ambiguity, literals and predicates.
"""

from __future__ import annotations

import itertools
import random
from typing import Any

from cosy.core import Constructor, SpecificationBuilder, Synthesizer
from cosy.core.solution_space import ConstantArgument, NonTerminalArgument, SolutionSpace
from cosy.core.tree import Tree
from cosy.core.types import Arrow, Intersection
from cosy.search import DepthBoundedRandomSampler, generator_query
from tests.search_fixtures import (
    AMBIGUOUS_TARGET,
    EXPR,
    LIST,
    PAIR,
    PRICED,
    USED,
    ambiguous_space,
    constrained_space,
    expression_space,
    list_space,
    priced_space,
    two_predicate_space,
)

A = Constructor("A")
B = Constructor("B")


def leaf() -> str:
    """Build the term that inhabits both sorts.

    Returns:
        str: Its rendering.
    """
    return "l"


def up(inner: str) -> str:
    """Lift an ``A`` or a ``B`` into an ``A`` or a ``B``: two derivations of each sort at every level.

    Args:
        inner (str): The interpreted argument.

    Returns:
        str: Its rendering.
    """
    return f"u({inner})"


def doubly_ambiguous_space() -> SolutionSpace:
    """A chain whose every node is derivable from ``A`` and from ``B``, each through two clauses."""
    specs = {
        leaf: SpecificationBuilder().suffix(Intersection(A, B)),
        up: SpecificationBuilder().suffix(
            Intersection(Intersection(Arrow(A, A), Arrow(B, A)), Intersection(Arrow(A, B), Arrow(B, B)))
        ),
    }
    return Synthesizer(specs).construct_solution_space(A)


def chain(depth: int) -> Tree[Any]:
    """``up`` applied ``depth`` times to ``leaf``."""
    tree: Tree[Any] = Tree(leaf)
    for _ in range(depth):
        tree = Tree(up, (tree,))
    return tree


def heads_match(rhs, tree: Tree[Any]) -> bool:
    """A rule matches a tree's head: its arity, its terminal, and every constant argument a leaf of that value."""
    return (
        len(rhs.arguments) == len(tree.children)
        and rhs.terminal == tree.root
        and all(
            not child.children and argument.value == child.root
            for argument, child in zip(rhs.arguments, tree.children, strict=True)
            if isinstance(argument, ConstantArgument)
        )
    )


def definition(space: SolutionSpace, nonterminal, tree: Tree[Any]) -> bool:
    """Derivability as the procedure defines it -- some matching rule whose arguments derive and whose predicates
    hold -- written out recursively from the public rules, remembering nothing."""
    for rhs in space.get(nonterminal) or ():
        if not heads_match(rhs, tree):
            continue
        arguments = list(zip(rhs.arguments, tree.children, strict=True))
        if not all(
            definition(space, argument.origin, child)
            for argument, child in arguments
            if isinstance(argument, NonTerminalArgument)
        ):
            continue
        substitution = {
            argument.name: child.root if isinstance(argument, ConstantArgument) else child
            for argument, child in arguments
            if argument.name is not None
        }
        if all(predicate(substitution) for predicate in rhs.predicates):
            return True
    return False


def test_a_doubly_ambiguous_chain_is_decided_with_work_linear_in_its_depth(monkeypatch):
    space = doubly_ambiguous_space()
    rules = sum(len(space.get(nonterminal) or ()) for nonterminal in (A, B))
    assert rules >= 4, "both sorts carry the two clauses of up"
    original = SolutionSpace._rule_matches_subtree  # noqa: SLF001 -- the work is counted where the procedure does it
    calls = []

    def counted(self, rhs, subtree):
        calls.append(subtree)
        return original(self, rhs, subtree)

    monkeypatch.setattr(SolutionSpace, "_rule_matches_subtree", counted)
    depth = 16
    assert space.contains_tree(A, chain(depth))
    assert len(calls) <= rules * (depth + 1), len(calls)
    calls.clear()
    assert not space.contains_tree(A, Tree(up, (Tree(up, (Tree(up),)),)))
    assert len(calls) <= rules * 3, len(calls)


X = Constructor("X")
Y = Constructor("Y")
M = Constructor("M")


def only_x() -> str:
    """Build the term of sort ``X`` alone.

    Returns:
        str: Its rendering.
    """
    return "x"


def only_y() -> str:
    """Build the term of sort ``Y`` alone.

    Returns:
        str: Its rendering.
    """
    return "y"


def pick(inner: str) -> str:
    """Turn an ``X`` or a ``Y`` into an ``M``, through one clause each.

    Args:
        inner (str): The interpreted argument.

    Returns:
        str: Its rendering.
    """
    return f"p({inner})"


def two_sorted_space() -> SolutionSpace:
    """``pick`` asks its argument of ``X`` through one clause and of ``Y`` through the other."""
    specs = {
        only_x: SpecificationBuilder().suffix(X),
        only_y: SpecificationBuilder().suffix(Y),
        pick: SpecificationBuilder().suffix(Intersection(Arrow(X, M), Arrow(Y, M))),
    }
    return Synthesizer(specs).construct_solution_space(M)


def test_a_subtree_derivable_from_one_sort_only_is_answered_per_sort():
    """One subtree, asked of two sorts with two answers: whichever is asked first, the other keeps its own."""
    space = two_sorted_space()
    for argument in (only_x, only_y):
        tree = Tree(pick, (Tree(argument),))
        assert space.contains_tree(M, tree), argument.__name__
        assert definition(space, M, tree)
    assert not space.contains_tree(M, Tree(pick, (Tree(pick, (Tree(only_x),)),)))


def _members(space: SolutionSpace, start, rng: random.Random, count: int) -> list[Tree]:
    stream = DepthBoundedRandomSampler(12, rng).sample(generator_query(space, start))
    return list(itertools.islice(stream, count))


def _grafts(members: list[Tree], rng: random.Random, count: int) -> list[Tree]:
    grafted = []
    for _ in range(count):
        first, second = rng.choice(members), rng.choice(members)
        position = rng.choice(sorted(first.positions()))
        grafted.append(first.replace_subtree_at(position, second.subtree_at(rng.choice(sorted(second.positions())))))
    return grafted


def test_the_answers_are_the_definition_s_on_members_and_non_members():
    rng = random.Random(0)
    spaces = [
        (doubly_ambiguous_space(), A),
        (ambiguous_space(), AMBIGUOUS_TARGET),
        (list_space(), LIST),
        (expression_space(), EXPR),
        (priced_space(), PRICED),
        (constrained_space(), PAIR),
        (two_predicate_space(), USED),
    ]
    seen = {True: 0, False: 0}
    for space, start in spaces:
        members = _members(space, start, rng, 25)
        assert members, start
        trees = members + _grafts(members, rng, 60)
        if start == A:
            trees += [chain(depth) for depth in range(8)] + [Tree(up, (Tree(up),)), Tree(up, (chain(3), chain(2)))]
        for tree in trees:
            answer = space.contains_tree(start, tree)
            assert answer == definition(space, start, tree), (start, tree)
            seen[answer] += 1
    assert seen[True] >= 150, seen
    assert seen[False] >= 100, seen
