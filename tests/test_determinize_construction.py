"""How the determinization builds its product program, held against the construction it replaced.

The construction keeps one product non-terminal per sort and state and one argument per name, sort
and state, builds each rule once, and does not prune, because it instantiates a clause only over
reached states. None of that may change the program it builds. The construction as it stood before
is kept below as the oracle, its body unchanged, and the two are compared on every reference space
of the determinization tests: the states in their order, every rule of every non-terminal in its
order, the size table, and the first draws of the table-counting size-uniform sampler. The order of
the non-terminals themselves is the one thing allowed to differ, the pruned program having listed
them in the order its prune discovered them.
"""

import random
from itertools import islice
from typing import Any

import pytest

from cosy.core.solution_space import NonTerminalArgument, SolutionSpace
from cosy.search import SizeUniformSampler, determinize, generator_query, size_table
from cosy.search.determinize import (
    Determinization,
    MergedNonTerminal,
    ProductNonTerminal,
    _abstract_constants,
    _fresh_combinations,
    _prepare,
    recognizable_or_raise,
)
from tests.search_fixtures import AVL, avl_space
from tests.test_determinize import RECOGNIZABLE_SPACES, UNCONSTRAINED_SPACES

SPACES = [
    *UNCONSTRAINED_SPACES,
    *RECOGNIZABLE_SPACES,
    ("avl over four keys", lambda: avl_space((0, 1, 2, 3)), AVL, 17),
]


def reference_determinize(space, start, *, state_limit=100_000):
    """The construction before it shared its objects, built each rule once and stopped pruning.

    Its body as it stood, so that the comparison is with the construction that measured every
    count before, not with a restatement of it.

    Returns:
        Determinization: The pruned product program.
    """
    if start not in space:
        msg = f"the start symbol {start} has no clause in this program, so there is nothing to determinize for it"
        raise ValueError(msg)
    recognizable_or_raise(space)
    clauses, abstractions = _prepare(space)
    _abstract_constants(clauses, abstractions)
    width = len(abstractions)
    merged = MergedNonTerminal(start)

    states: dict = {nonterminal: [] for nonterminal in space.nonterminals()}
    reached: dict = {nonterminal: set() for nonterminal in states}
    produced: dict = {}
    total_states = 0

    def instantiate(clause, hole_states):
        nonlocal total_states
        per_abstraction = [
            tuple(
                clause.constant_states[position][component] if slot is None else hole_states[slot][component]
                for position, slot in enumerate(clause.slots)
            )
            for component in range(width)
        ]
        for relation, component in clause.relations:
            substitution = {
                name: per_abstraction[component][position]
                for position, name in enumerate(clause.names)
                if name is not None
            }
            if not relation(substitution):
                return
        head_state = tuple(
            abstraction(clause.rule.terminal, per_abstraction[component])
            for component, abstraction in enumerate(abstractions)
        )
        if head_state not in reached[clause.head]:
            reached[clause.head].add(head_state)
            states[clause.head].append(head_state)
            total_states += 1
            if state_limit is not None and total_states > state_limit:
                msg = f"the abstraction reached more than {state_limit} product states"
                raise ValueError(msg)
        arguments = tuple(
            argument
            if slot is None
            else NonTerminalArgument(argument.name, ProductNonTerminal(argument.origin, hole_states[slot]))
            for argument, slot in zip(clause.rule.arguments, clause.slots, strict=True)
        )
        instance = (clause.index, clause.rule.terminal, arguments)
        produced.setdefault(ProductNonTerminal(clause.head, head_state), []).append(instance)
        if clause.head == start:
            produced.setdefault(merged, []).append(instance)

    inner = [clause for clause in clauses if clause.hole_origins]
    for clause in clauses:
        if not clause.hole_origins:
            instantiate(clause, ())

    visited = [(0,) * len(clause.hole_origins) for clause in inner]
    while True:
        progressed = False
        for position, clause in enumerate(inner):
            sizes = tuple(len(states.get(origin, ())) for origin in clause.hole_origins)
            old = visited[position]
            if sizes == old:
                continue
            for combination in _fresh_combinations(old, sizes):
                instantiate(
                    clause,
                    tuple(
                        states[origin][index] for origin, index in zip(clause.hole_origins, combination, strict=True)
                    ),
                )
            visited[position] = sizes
            progressed = True
        if not progressed:
            break

    determinized: SolutionSpace = SolutionSpace()
    for head, instances in produced.items():
        for _index, terminal, arguments in sorted(instances, key=lambda instance: instance[0]):
            determinized.add_rule(head, terminal, arguments, ())

    return Determinization(
        space=determinized.prune(),
        start=merged,
        states={nonterminal: tuple(found) for nonterminal, found in states.items()},
        abstractions=tuple(abstractions),
    )


def rules_of(space: SolutionSpace, nonterminal: Any) -> list:
    return [(rule.terminal, rule.arguments, rule.predicates) for rule in space.get(nonterminal) or ()]


@pytest.mark.parametrize(("name", "build", "start", "bound"), SPACES)
def test_the_construction_builds_the_program_it_replaced(name, build, start, bound):
    """States, rules, counts and draws as the construction before built them; only the sort order moves."""
    space = build()
    built, oracle = determinize(space, start), reference_determinize(space, start)

    assert built.start == oracle.start
    assert built.abstractions == oracle.abstractions
    assert built.states == oracle.states, f"{name}: the states or their order differ"
    assert set(built.space.nonterminals()) == set(oracle.space.nonterminals())
    for nonterminal in oracle.space.nonterminals():
        assert rules_of(built.space, nonterminal) == rules_of(oracle.space, nonterminal), f"{name}: {nonterminal}"
    built_table, oracle_table = size_table(built.space, bound), size_table(oracle.space, bound)
    assert dict(built_table.counts) == dict(oracle_table.counts)
    assert sum(oracle_table.counts[oracle.start]) > 0, f"{name}: a space with no term tests nothing"
    built_draws = SizeUniformSampler(bound, random.Random(7), counting="table").sample(
        generator_query(built.space, built.start)
    )
    oracle_draws = SizeUniformSampler(bound, random.Random(7), counting="table").sample(
        generator_query(oracle.space, oracle.start)
    )
    assert list(islice(built_draws, 30)) == list(islice(oracle_draws, 30))


def test_a_state_is_one_object_in_every_rule_that_names_it():
    """One product non-terminal per sort and state, one argument per name, sort and state."""
    determinization = determinize(avl_space((0, 1, 2)), AVL)
    space = determinization.space
    keys = {nonterminal: nonterminal for nonterminal in space.nonterminals()}
    arguments: dict = {}
    holes = 0
    for nonterminal in space.nonterminals():
        for rule in space.get(nonterminal) or ():
            for argument in rule.arguments:
                if isinstance(argument, NonTerminalArgument):
                    holes += 1
                    # the hole names the very object the program keys that sort's rules under
                    assert argument.origin is keys[argument.origin]
                    arguments.setdefault((argument.name, argument.origin), set()).add(id(argument))
    assert holes > len(arguments), "every hole a different argument tests no sharing"
    assert all(len(identities) == 1 for identities in arguments.values())


def test_the_construction_never_prunes(monkeypatch):
    """Every rule is instantiated over reached states, so a prune would remove nothing; it is not run."""

    def refused(_space):
        msg = "determinize pruned"
        raise AssertionError(msg)

    space = avl_space((0, 1, 2))  # the fixture prunes the synthesized program itself
    monkeypatch.setattr(SolutionSpace, "prune", refused)
    determinization = determinize(space, AVL)
    assert any(isinstance(nonterminal, MergedNonTerminal) for nonterminal in determinization.space.nonterminals())


def test_the_merged_start_holds_the_rules_of_its_states_themselves():
    """Each rule is built once: the start symbol's rules are the rule objects of the start's states."""
    determinization = determinize(avl_space((0, 1, 2)), AVL)
    space = determinization.space
    of_states = {
        id(rule)
        for nonterminal in space.nonterminals()
        if isinstance(nonterminal, ProductNonTerminal) and nonterminal.nonterminal == AVL
        for rule in space.get(nonterminal) or ()
    }
    merged = list(space.get(determinization.start) or ())
    assert merged, "a start without rules tests nothing"
    assert all(id(rule) in of_states for rule in merged)
