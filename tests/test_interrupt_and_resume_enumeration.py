"""_summary_."""

import pytest

from cosy.core.solution_space import NonTerminalArgument, RHSRule, SolutionSpace, _RuleProgress
from cosy.core.tree import Tree

_TERMS = (Tree("t1"), Tree("t2"), Tree("t3"))


@pytest.fixture
def mixed_slot_rule() -> RHSRule:
    """_summary_.

    Returns:
        RHSRule: _description_
    """
    return RHSRule(
        arguments=(NonTerminalArgument("x", "a"), NonTerminalArgument(None, "a")),
        predicates=(),
        terminal="f",
    )


def _recomputed(rule: RHSRule) -> set:
    """_summary_.

    Returns:
        set: _description_
    """
    space: SolutionSpace = SolutionSpace()
    existing: dict = {"a": set()}
    trees: set = set()
    for term in _TERMS:
        existing["a"].add(term)
        trees |= space._generate_new_trees(rule, existing, None, None, ("a", term))  # noqa: SLF001
    return trees


def _resumed(rule: RHSRule) -> set:
    """_summary_.

    Returns:
        set: _description_
    """
    space: SolutionSpace = SolutionSpace()
    existing: dict = {"a": set()}
    trees: set = set()
    progress: _RuleProgress = _RuleProgress()
    trees |= space._generate_new_trees(rule, existing, None, None, None, progress)  # noqa: SLF001
    for term in _TERMS:
        existing["a"].add(term)
        trees |= space._generate_new_trees(rule, existing, None, None, ("a", term), progress)  # noqa: SLF001
    return trees


def test_resume_matches_recompute(mixed_slot_rule: RHSRule) -> None:
    """_summary_."""
    recomputed = _recomputed(mixed_slot_rule)
    assert _resumed(mixed_slot_rule) == recomputed
    assert len(recomputed) == 9


def test_interrupted_call_resumes_without_losing_trees(mixed_slot_rule: RHSRule) -> None:
    """_summary_."""
    space: SolutionSpace = SolutionSpace()
    resumed: set = set()
    existing: dict = {"a": set()}
    progress: _RuleProgress = _RuleProgress()
    resumed |= space._generate_new_trees(mixed_slot_rule, existing, None, None, None, progress)  # noqa: SLF001
    for index, term in enumerate(_TERMS):
        existing["a"].add(term)
        # interrupt
        max_count = 1 if index == 2 else None
        resumed |= space._generate_new_trees(mixed_slot_rule, existing, None, max_count, ("a", term), progress)  # noqa: SLF001
    # check if truly interrupted
    assert progress.pending_parameters
    assert progress.pending_arguments
    # check resuming is not lossy and drains
    resumed |= space._generate_new_trees(mixed_slot_rule, existing, None, None, None, progress)  # noqa: SLF001
    assert not progress.pending_parameters
    assert not progress.pending_arguments
    assert resumed == _recomputed(mixed_slot_rule)
