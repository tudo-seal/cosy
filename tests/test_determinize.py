"""Predicate determinization: compiling a coupling constraint into the non-terminals.

The table form of counting applies to a program in which no predicate reads a hole. Where a
repository states its conditions over several holes at once, no clause of it qualifies, so either
the coupled case is covered or such a repository gets nothing.

:func:`cosy.search.determinize.determinize` covers it under condition **(REC)**: a predicate that
factors through a finite abstraction is pushed into the non-terminals by the product construction
``NT x Q``, and the program that comes out is predicate-free. This file pins what that has to
mean:

* **The language is the same.** The determinized program derives exactly the terms the original
  derives. That is checked against the *hand-written* coupled space and not only against the
  abstraction's own space, so an abstraction quietly disagreeing with the predicate it replaces is
  caught rather than confirmed.
* **Every term keeps exactly one derivation.** ``alpha`` is a function, so the map on success
  branches is a bijection and the branch counts, the unambiguity and the weights a draw reads off
  the program are all invariant. This is the one place where an error would change the sampled
  distribution without any other symptom, and it has a specific failure mode: enumerating the
  clause instances once per fixed-point round rather than once in total duplicates rules, and a
  duplicated rule is a second derivation of the same term.
* **The states are the abstraction.** A term derived from ``(A, q)`` abstracts to ``q``. The
  product construction and the fold of :func:`cosy.core.recognizable.state_of` compute the same
  thing, which is what makes the head state of a clause instance the right one.
* **What cannot be determinized says so.** A predicate without an abstraction is named, with its
  clause and the positions it reads, and an abstraction with an infinite carrier is reported
  rather than left to run forever.

The AVL space here is the benchmark of Goldstein and Pierce (2022), trees over the keys 0 to 9
under the validity condition of a balanced AVL tree. The keys 0 to 2 are hand-checked: one leaf,
three one-node trees, ``C(3, 2) * 2 = 6`` two-node trees, and one balanced three-node tree.
"""

import random

import pytest

from cosy.core import Constructor, SpecificationBuilder, Synthesizer, state_of
from cosy.core.tree import Tree
from cosy.core.types import DataGroup
from cosy.search import (
    branch_counts,
    decomposable_or_raise,
    determinize,
    generator_query,
    size_table,
    term_size,
    weighted_table,
)
from cosy.search.counting import branch_multiplicities
from cosy.search.determinize import (
    MergedNonTerminal,
    ProductNonTerminal,
    recognizable_or_raise,
    unabstracted_clauses,
)
from tests.search_fixtures import (
    AVL,
    BOX,
    CHAIN,
    CUT_SORT,
    EXPR,
    HELD,
    LIST,
    PAIR,
    TAGGED,
    USED,
    anonymous_hole_space,
    at_most_two,
    avl_coupled_space,
    avl_leaf,
    avl_node,
    avl_space,
    box,
    capped_size,
    chain_space,
    constrained_space,
    cut_space,
    expression_space,
    list_space,
    literal_predicate_space,
    recognizable_cut_space,
    recognizable_pair_space,
    two_offender_space,
    two_symbol_clause_space,
    v_one,
    v_wrap,
    v_zero,
)

# The reference spaces with no recognizable constraint in them. Determinizing one is the identity
# up to the wrapper, there being a single state and no abstraction, and that is worth pinning: the
# construction must not become a special case that only runs where a constraint is present.
UNCONSTRAINED_SPACES = [
    ("list", list_space, LIST, 6),
    ("expression", expression_space, EXPR, 7),
    ("two symbol clauses", two_symbol_clause_space, TAGGED, 9),
    ("chain", chain_space, CHAIN, 9),
    ("literal predicate", literal_predicate_space, USED, 4),
    ("anonymous hole", anonymous_hole_space, HELD, 4),
]

# The spaces whose constraint states its abstraction, with the bound each is checked at.
RECOGNIZABLE_SPACES = [
    ("cut", recognizable_cut_space, BOX, 8),
    ("pair", recognizable_pair_space, PAIR, 8),
    ("avl", lambda: avl_space((0, 1, 2)), AVL, 13),
]

# The AVL benchmark, at the two key counts the coupled space can still be counted at.
AVL_COUNTS = {
    2: {1: 1, 5: 2, 9: 2},
    3: {1: 1, 5: 3, 9: 6, 13: 1},
}


def uniform(_value):
    """Weight every realized size alike, which is the distribution of size-uniform sampling.

    Args:
        _value: The size. Ignored.

    Returns:
        float: One. The construction normalizes over the realized sizes itself.
    """
    return 1.0


def exact_counts(space, start, bound):
    """Return the branch counts of a generator query, by size.

    Args:
        space (SolutionSpace): The program.
        start: The queried non-terminal.
        bound (int): The size bound ``D``.

    Returns:
        dict[int, int]: ``N_r(s)`` for every realized size.
    """
    root = branch_counts(generator_query(space, start), bound, term_size)
    return dict(sorted(root.counts.items()))


def inhabitants_of(space, start, bound):
    """Return the inhabitants of a query within a size bound.

    Reads them off the retained tree rather than through ``enumerate_trees``: the enumeration is
    bounded by a term *count*, so on a sort with fewer terms than asked for it keeps enumerating
    the recursive sorts of the same program forever. A size bound is what makes the set finite
    here, and it is the bound the whole construction is stated under anyway.

    Args:
        space (SolutionSpace): The program.
        start: The queried non-terminal.
        bound (int): The size bound ``D``.

    Returns:
        list[Tree]: The inhabitants, one per success branch.
    """
    found = []
    pending = [branch_counts(generator_query(space, start), bound, term_size)]
    while pending:
        node = pending.pop()
        if node.inhabitant is not None:
            found.append(node.inhabitant)
        pending.extend(node.children)
    return found


def valid_avl(term, leaf_symbol, node_symbol):
    """Decide whether a term is a valid AVL tree, independently of the machinery that built it.

    Written as a recursive checker over the finished term rather than through the abstraction or
    the predicate, so that a sample can be validated by something neither of them produced.

    Args:
        term (Tree): The term to check.
        leaf_symbol: The combinator of the empty tree.
        node_symbol: The combinator of an inner node.

    Returns:
        bool: True when the tree is ordered, balanced and caches its heights correctly.
    """

    def walk(subtree):
        """Return validity, height and extreme keys of a subtree.

        Args:
            subtree (Tree): The subtree.

        Returns:
            tuple: ``(valid, height, smallest key, greatest key)``.
        """
        if subtree.root is leaf_symbol:
            return True, 0, None, None
        if subtree.root is not node_symbol or len(subtree.children) != 4:
            return False, 0, None, None
        key, cached = subtree.children[0].root, subtree.children[1].root
        left_ok, left_height, left_low, left_high = walk(subtree.children[2])
        right_ok, right_height, right_low, right_high = walk(subtree.children[3])
        return (
            left_ok
            and right_ok
            and abs(left_height - right_height) <= 1
            and cached == 1 + max(left_height, right_height)
            and (left_high is None or left_high < key)
            and (right_low is None or right_low > key),
            1 + max(left_height, right_height),
            left_low if left_low is not None else key,
            right_high if right_high is not None else key,
        )

    return walk(term)[0]


@pytest.mark.parametrize(("name", "build", "start", "bound"), UNCONSTRAINED_SPACES)
def test_a_program_without_abstractions_is_determinized_to_itself(name, build, start, bound):
    """A program stating no recognizable constraint keeps its counts, with one state per sort.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The start non-terminal.
        bound (int): The size bound.
    """
    space = build()
    determinization = determinize(space, start)

    assert determinization.abstractions == (), f"{name} states no abstraction"
    assert all(states == ((),) for states in determinization.states.values()), (
        f"{name} has one product state per sort, the empty tuple"
    )
    assert exact_counts(determinization.space, determinization.start, bound) == exact_counts(space, start, bound)


@pytest.mark.parametrize(("name", "build", "start", "bound"), RECOGNIZABLE_SPACES)
def test_determinization_preserves_the_counts(name, build, start, bound):
    """The determinized program derives the same terms, per size, as the program it came from.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The start non-terminal.
        bound (int): The size bound.
    """
    space = build()
    determinization = determinize(space, start)
    assert exact_counts(determinization.space, determinization.start, bound) == exact_counts(space, start, bound), (
        f"{name} changed its counts under determinization"
    )


@pytest.mark.parametrize(("name", "build", "start", "bound"), RECOGNIZABLE_SPACES)
def test_determinization_keeps_one_derivation_per_term(name, build, start, bound):
    """No inhabitant of the determinized program ends more than one success branch.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The start non-terminal.
        bound (int): The size bound.
    """
    determinization = determinize(build(), start)
    root = branch_counts(generator_query(determinization.space, determinization.start), bound, term_size)
    assert branch_multiplicities(root) == {}, f"{name} derives some term twice"


@pytest.mark.parametrize(("name", "build", "start", "bound"), RECOGNIZABLE_SPACES)
def test_the_determinized_program_is_countable_from_the_program(name, build, start, bound):
    """The point of the construction: the table form applies to the result, and agrees with it.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The start non-terminal.
        bound (int): The size bound.
    """
    determinization = determinize(build(), start)
    decomposable_or_raise(determinization.space)
    table = size_table(determinization.space, bound)
    row = {
        size: table.of(determinization.start, size)
        for size in range(bound + 1)
        if table.of(determinization.start, size)
    }
    assert row == exact_counts(determinization.space, determinization.start, bound), (
        f"the table disagrees with the tree form on the determinized {name} space"
    )


@pytest.mark.parametrize(("name", "build", "start", "bound"), RECOGNIZABLE_SPACES)
def test_every_term_abstracts_to_the_state_it_is_derived_from(name, build, start, bound):
    """A term derived from ``(A, q)`` folds to ``q`` under the abstraction.

    The identity the head state of a clause instance rests on. The construction computes the
    state from the argument states and :func:`state_of` computes it from the finished term, and
    the two agree because they are the same function.

    Args:
        name (str): The space's name, for the test id.
        build (Callable): Builds the space.
        start: The start non-terminal.
        bound (int): The size bound the terms are collected within.
    """
    determinization = determinize(build(), start)
    abstractions = determinization.abstractions
    checked = 0
    for nonterminal in determinization.space.nonterminals():
        if not isinstance(nonterminal, ProductNonTerminal):
            continue
        for term in inhabitants_of(determinization.space, nonterminal, bound):
            folded = tuple(state_of(term, abstraction) for abstraction in abstractions)
            assert folded == nonterminal.state, (
                f"{name}: {term} is derived from {nonterminal} but abstracts to {folded}"
            )
            checked += 1
    assert checked > 0, f"{name} produced no term to check"


@pytest.mark.parametrize("key_count", [2, 3])
def test_the_avl_benchmark_is_counted_exactly(key_count):
    """The coupled space, the abstracted space and the determinized program count alike.

    The coupled space is the control. Its predicate reads both grounded subtrees and knows
    nothing of the abstraction, so agreement is evidence that the abstraction decides the same
    condition, and not merely that the construction is consistent with itself.

    Args:
        key_count (int): How many keys the space admits.
    """
    keys = tuple(range(key_count))
    bound = 4 * key_count + 1
    coupled = exact_counts(avl_coupled_space(keys), AVL, bound)
    abstracted = exact_counts(avl_space(keys), AVL, bound)
    determinization = determinize(avl_space(keys), AVL)
    determinized = exact_counts(determinization.space, determinization.start, bound)

    assert coupled == AVL_COUNTS[key_count]
    assert abstracted == coupled, "the abstraction decides a different condition than the predicate"
    assert determinized == coupled, "the product construction changed the language"


def test_the_determinized_avl_space_draws_valid_trees():
    """Size-uniform sampling on the determinized program returns valid, distinct AVL trees.

    The end-to-end claim: what comes out of the table form on the product program are terms an
    independent checker accepts. The height a generator would have to guess is in the
    non-terminal, so no draw can get it wrong.
    """
    keys = tuple(range(5))
    determinization = determinize(avl_space(keys), AVL)
    query = generator_query(determinization.space, determinization.start)
    weighted = weighted_table(query, 4 * len(keys) + 1, uniform)
    stream = weighted.stream(random.Random(4))
    drawn = [term for _index, term in zip(range(40), stream, strict=False)]

    assert len(drawn) == 40
    assert len(set(drawn)) == 40, "random search draws without replacement"
    assert all(valid_avl(term, avl_leaf, avl_node) for term in drawn)


def test_two_abstractions_are_taken_as_a_product():
    """A program stating two different abstractions carries both components in its states.

    Nothing else here has more than one abstraction, so nothing else can tell a construction that
    takes the product from one that silently uses the first abstraction it finds.
    """
    short_sort, odd_sort, top_sort = Constructor("Short"), Constructor("Odd"), Constructor("Top")

    def size_cap(_symbol, states):
        """Abstract by the term size, capped at three.

        Args:
            _symbol: The function symbol.
            states: The sizes of the arguments.

        Returns:
            int: The capped size.
        """
        return min(1 + sum(states), 3)

    def first_letter(symbol, states):
        """Abstract by the letter at the core of a value.

        Args:
            symbol: The function symbol.
            states: The abstractions of the arguments.

        Returns:
            int: 0 or 1.
        """
        return states[0] if states else (0 if symbol is v_zero else 1)

    def short_value(inner: str) -> str:
        """Keep a value of at most two symbols.

        Args:
            inner (str): The interpreted value.

        Returns:
            str: Its rendering.
        """
        return f"s({inner})"

    def odd_value(inner: str) -> str:
        """Keep a value whose core is the second letter.

        Args:
            inner (str): The interpreted value.

        Returns:
            str: Its rendering.
        """
        return f"o({inner})"

    def both(left: str, right: str) -> str:
        """Pair the two kinds of value.

        Args:
            left (str): The interpreted short value.
            right (str): The interpreted odd value.

        Returns:
            str: Its rendering.
        """
        return f"<{left},{right}>"

    specs = {
        v_zero: SpecificationBuilder().suffix(CUT_SORT),
        v_one: SpecificationBuilder().suffix(CUT_SORT),
        v_wrap: SpecificationBuilder().argument("inner", CUT_SORT).suffix(CUT_SORT),
        short_value: SpecificationBuilder()
        .argument("inner", CUT_SORT)
        .recognizable_constraint(size_cap, lambda states: states["inner"] <= 2)
        .suffix(short_sort),
        odd_value: SpecificationBuilder()
        .argument("inner", CUT_SORT)
        .recognizable_constraint(first_letter, lambda states: states["inner"] == 1)
        .suffix(odd_sort),
        both: SpecificationBuilder().argument("left", short_sort).argument("right", odd_sort).suffix(top_sort),
    }
    space = Synthesizer(specs).construct_solution_space(top_sort)
    determinization = determinize(space, top_sort)

    assert len(determinization.abstractions) == 2
    assert all(len(state) == 2 for states in determinization.states.values() for state in states)
    assert exact_counts(determinization.space, determinization.start, 9) == exact_counts(space, top_sort, 9)


def test_a_predicate_without_an_abstraction_is_named():
    """A hole-reading predicate that states no abstraction is refused, with its clause named.

    The refusal is what makes the condition visible in the repository rather than in the numbers:
    the alternative is a construction that ignores the predicate and produces a program deriving
    terms the repository forbids.
    """
    assert unabstracted_clauses(recognizable_cut_space()) == []

    offenders = unabstracted_clauses(two_offender_space())
    assert len(offenders) == 2, "both offending clauses are reported, not only the first"

    with pytest.raises(ValueError, match="recognizable_constraint") as raised:
        determinize(cut_space(), BOX)
    assert "box" in str(raised.value)
    assert "argument 0" in str(raised.value)

    with pytest.raises(ValueError, match="recognizable_constraint"):
        recognizable_or_raise(constrained_space())


def test_the_cut_stated_as_an_abstraction_is_the_cut_stated_as_a_predicate():
    """The abstracted cut space and the hand-written one derive the same terms.

    The AVL pair carries this comparison for a coupling condition. This is the same comparison for
    a condition on one hole, and it is the check that ``capped_size`` and ``at_most_two`` together
    decide what ``short`` decides. Without it the cut space's counts are only ever compared to a
    construction built from the same abstraction, which confirms an abstraction rather than
    testing it.
    """
    assert exact_counts(recognizable_cut_space(), BOX, 8) == exact_counts(cut_space(), BOX, 8)


def test_the_pair_space_holds_the_number_of_pairs_its_construction_predicts():
    """Two words per size and two admitted core combinations per split give ``2 (n - 2)``.

    ``constrained_space`` is not a control for this space, stating term inequality where this one
    states core inequality, so the derived count stands in for one. A word of size ``s`` is
    ``wrap`` applied ``s - 1`` times to a letter, a pair of size ``n`` splits ``n - 1`` symbols
    over two words of at least one symbol each, and of the four core combinations at each split
    the relation admits the two that differ.
    """
    counts = exact_counts(recognizable_pair_space(), PAIR, 9)
    assert counts == {size: 2 * (size - 2) for size in range(3, 10)}


def test_a_clause_mixing_an_abstraction_with_a_plain_predicate_is_refused():
    """One clause, two predicates, one of them without an abstraction. The clause is refused.

    This is the only input on which "every predicate of this clause states an abstraction" differs
    from "some predicate does". Reading it the loose way would send the clause into the
    construction, where the plain predicate is neither compiled away nor evaluated, and the
    program that came out would derive terms the repository forbids.
    """
    specs = {
        v_zero: SpecificationBuilder().suffix(CUT_SORT),
        v_one: SpecificationBuilder().suffix(CUT_SORT),
        v_wrap: SpecificationBuilder().argument("inner", CUT_SORT).suffix(CUT_SORT),
        box: SpecificationBuilder()
        .argument("inner", CUT_SORT)
        .recognizable_constraint(capped_size, at_most_two)
        .constraint(lambda substitution: substitution["inner"].size > 0)
        .suffix(BOX),
    }
    space = Synthesizer(specs).construct_solution_space(BOX)

    offenders = unabstracted_clauses(space)
    assert len(offenders) == 1, "a clause is an offender as soon as one of its predicates has no abstraction"

    with pytest.raises(ValueError, match="recognizable_constraint"):
        determinize(space, BOX)


def test_two_abstracted_constraints_on_one_clause_are_both_enforced():
    """A clause may state more than one recognizable constraint, and every one of them holds.

    Nothing else here puts two on a single clause, so nothing else can tell a construction that
    enforces every relation from one that stops after the first. The two conditions here cut in
    different directions: one bounds the size of the boxed value and the other fixes its core, so
    keeping only the first admits values the second forbids.
    """

    def core_letter(symbol, states):
        """Abstract a value by the letter at its core.

        Args:
            symbol: The function symbol.
            states: The abstractions of the arguments.

        Returns:
            int: 0 or 1.
        """
        return states[0] if states else (0 if symbol is v_zero else 1)

    def core_is_one(substitution) -> bool:
        """Accept a value whose core is the second letter.

        Args:
            substitution: The clause's substitution under the core abstraction.

        Returns:
            bool: True when the core is 1.
        """
        return substitution["inner"] == 1

    specs = {
        v_zero: SpecificationBuilder().suffix(CUT_SORT),
        v_one: SpecificationBuilder().suffix(CUT_SORT),
        v_wrap: SpecificationBuilder().argument("inner", CUT_SORT).suffix(CUT_SORT),
        box: SpecificationBuilder()
        .argument("inner", CUT_SORT)
        .recognizable_constraint(capped_size, at_most_two)
        .recognizable_constraint(core_letter, core_is_one)
        .suffix(BOX),
    }
    space = Synthesizer(specs).construct_solution_space(BOX)
    determinization = determinize(space, BOX)

    # v1 and vw(v1) satisfy both, v0 and vw(v0) fail the core, vw(vw(v1)) fails the size.
    assert exact_counts(space, BOX, 6) == {2: 1, 3: 1}
    assert exact_counts(determinization.space, determinization.start, 6) == {2: 1, 3: 1}
    assert len(determinization.abstractions) == 2, "both abstractions are carried, as a product"


def test_an_abstraction_that_is_not_the_identity_on_literals():
    """A literal's state is what the abstraction says it is, and not the literal itself.

    Every other space here abstracts a literal to its own value, so none of them can tell the
    construction's treatment of a literal from a construction that passes the raw value through.
    Here the abstraction buckets the digit by parity and the relation reads the bucket, so passing
    the value through admits digit 1 and drops digit 3.

    The clause carries a hole beside the literal on purpose. A clause of literals alone is decided
    once at build time and never reaches the states the construction computes for its constants.
    """
    top_sort, inner_sort = Constructor("Top"), Constructor("Inner")

    def parity(symbol, states):
        """Abstract a literal by its parity and a term by its argument's state.

        Args:
            symbol: The function symbol, or a literal value.
            states: The abstractions of the arguments.

        Returns:
            int: The parity of a literal, and the state of its first argument on a term.
        """
        if isinstance(symbol, int):
            return symbol % 2
        return states[0] if states else 0

    def odd_digit(substitution) -> bool:
        """Accept an odd digit beside a hole, both read through the parity abstraction.

        Args:
            substitution: The clause's substitution under the parity abstraction.

        Returns:
            bool: True when the digit's parity is 1.
        """
        return substitution["d"] == 1

    def plain() -> str:
        """Build the one term of the inner sort.

        Returns:
            str: Its rendering under ``interpret``.
        """
        return "p"

    def hold(d: int, inner: str) -> str:
        """Build a held digit over an inner term.

        Args:
            d (int): The digit.
            inner (str): The interpreted inner term.

        Returns:
            str: Its rendering under ``interpret``.
        """
        return f"[{d}{inner}]"

    specs = {
        plain: SpecificationBuilder().suffix(inner_sort),
        hold: SpecificationBuilder()
        .parameter("d", DataGroup("digit", (0, 1, 2, 3)))
        .argument("inner", inner_sort)
        .recognizable_constraint(parity, odd_digit)
        .suffix(top_sort),
    }
    space = Synthesizer(specs).construct_solution_space(top_sort)
    determinization = determinize(space, top_sort)

    # Digits 1 and 3 are odd. A held digit is three symbols, the terminal, the literal and the
    # inner term. Reading the raw value instead of its parity would keep only digit 1.
    assert exact_counts(space, top_sort, 4) == {3: 2}
    assert exact_counts(determinization.space, determinization.start, 4) == {3: 2}


def test_the_instances_of_a_head_follow_the_program_and_not_the_fixed_point():
    """Where the two orders disagree, the emitted instances follow the program.

    The AVL space cannot show this, its clauses being reached in the order it declares them. Here
    the recursive clause is declared first and reached second, so a construction emitting
    instances in the order the fixed point produced them puts the base clause in front.

    Both clauses reach the same product sort, which is what puts their instances under one head
    where an order between them exists at all.
    """
    sort = Constructor("Both")

    def constant(_symbol, states):
        """Abstract everything to one state, so both clauses land under one head.

        Args:
            _symbol: The function symbol.
            states: The abstractions of the arguments.

        Returns:
            int: Always 0.
        """
        del states
        return 0

    def always(substitution) -> bool:
        """Accept every state.

        Args:
            substitution: The clause's substitution. Unused.

        Returns:
            bool: True.
        """
        del substitution
        return True

    def step(inner: str) -> str:
        """Extend a term, declared before the clause that starts one.

        Args:
            inner (str): The interpreted inner term.

        Returns:
            str: Its rendering under ``interpret``.
        """
        return f"s({inner})"

    def start() -> str:
        """Start a term, declared after the clause that extends one.

        Returns:
            str: Its rendering under ``interpret``.
        """
        return "z"

    specs = {
        step: SpecificationBuilder().argument("inner", sort).recognizable_constraint(constant, always).suffix(sort),
        start: SpecificationBuilder().suffix(sort),
    }
    determinization = determinize(Synthesizer(specs).construct_solution_space(sort), sort)

    terminals = [step, start]
    heads = [
        [terminals.index(rule.terminal) for rule in rules]
        for _head, rules in determinization.space.as_tuples()
        if len(rules) > 1
    ]
    assert heads, "no head carries both clauses, so there is no order to read"
    for indices in heads:
        assert indices == sorted(indices), "the fixed point reached the base clause first, the program declares it last"


def test_a_repeated_subterm_is_abstracted_once():
    """``state_of`` folds each distinct subterm once, however often the term repeats it.

    The saving is the reason the fold is keyed on the subterms rather than on their positions, and
    a term that shares a subtree between its arguments is where the two differ. Without the saving
    the number of calls is the size of the term, which doubles with every level of sharing.
    """
    calls = []

    def counting(symbol, states):
        """Abstract by size, recording every call.

        Args:
            symbol (Any): The function symbol.
            states (tuple): The sizes of the arguments.

        Returns:
            int: The size of the term.
        """
        del symbol
        calls.append(1)
        return 1 + sum(states)

    shared = Tree("f", (Tree("a", ()), Tree("a", ())))
    term = Tree("g", (shared, shared))

    assert state_of(term, counting) == 7, "the fold reads the term, whose size counts every position"
    assert len(calls) == 3, "three distinct subterms: the leaf, the shared node, and the root"


def test_an_abstraction_without_a_finite_carrier_is_reported():
    """An abstraction whose carrier is infinite is reported instead of run forever.

    The uncapped term size is the standard slip, and on a recursive sort it produces a new state
    per size for as long as the fixed point is allowed to run.
    """
    specs = {
        v_zero: SpecificationBuilder().suffix(CUT_SORT),
        v_one: SpecificationBuilder().suffix(CUT_SORT),
        v_wrap: SpecificationBuilder()
        .argument("inner", CUT_SORT)
        .recognizable_constraint(lambda _symbol, states: 1 + sum(states), lambda states: states["inner"] >= 0)
        .suffix(CUT_SORT),
    }
    space = Synthesizer(specs).construct_solution_space(CUT_SORT)
    with pytest.raises(ValueError, match="product states"):
        determinize(space, CUT_SORT, state_limit=200)


def test_an_unknown_start_symbol_is_reported():
    """Determinizing for a non-terminal the program does not have says so."""
    with pytest.raises(ValueError, match="no clause"):
        determinize(list_space(), Constructor("Absent"))


def test_the_construction_does_not_depend_on_the_fixed_point_order():
    """Determinizing twice produces the same program, rule for rule and in the same order.

    The instances are emitted while the state sets still grow, so the order in which the fixed
    point reaches them is an implementation detail, and sorting by clause index makes the result
    depend on the program instead. CI runs randomized and parallel, and a grammar whose rule
    order drifts makes every stream irreproducible.
    """
    space = avl_space((0, 1, 2))
    first, second = determinize(space, AVL), determinize(space, AVL)

    # One space, determinized twice. Two *separately built* spaces would compare unequal for a
    # reason that has nothing to do with this construction: a repository builds a fresh
    # ``DataGroup`` per call, and the constant arguments carry it.
    assert first.states == second.states
    assert [nonterminal for nonterminal, _rules in first.space.as_tuples()] == [
        nonterminal for nonterminal, _rules in second.space.as_tuples()
    ]
    for (_head, left), (_other, right) in zip(first.space.as_tuples(), second.space.as_tuples(), strict=True):
        assert [(rule.terminal, rule.arguments) for rule in left] == [(rule.terminal, rule.arguments) for rule in right]


def test_the_instances_of_a_head_stay_in_the_order_of_the_program():
    """Within one product sort the instances follow the clause order of the original program.

    The comparison above is between two runs of the same function, so it sees an order that varies
    between runs and not one that varies from the program. This reads the property off a single
    run: an instance carries the index of the clause it came from, and those indices rise along
    the rules of a head. A construction emitting them in the order the fixed point happened to
    reach them would interleave two clauses of one sort.
    """
    determinization = determinize(avl_space((0, 1, 2)), AVL)
    terminals = [avl_leaf, avl_node]  # the order the AVL specification declares its clauses in
    seen_any = False
    for _head, rules in determinization.space.as_tuples():
        indices = [terminals.index(rule.terminal) for rule in rules]
        assert indices == sorted(indices)
        seen_any = seen_any or len(indices) > 1
    assert seen_any, "no head carries two instances, so the order was never observed"


def test_the_fixed_point_reaches_only_productive_states():
    """Pruning the determinized program removes nothing.

    A state enters the fixed point only through a clause instance whose holes are already
    reachable, so every product sort has a term by construction. If pruning ever removed
    something, the construction would be reaching states nothing realizes, which is how a
    product construction usually blows up.
    """
    determinization = determinize(avl_space((0, 1, 2)), AVL)
    # Read against what the fixed point recorded, not against a second prune. ``determinize``
    # prunes before returning, so pruning its result again removes nothing whatever the fixed
    # point did, and the comparison would hold for a construction reaching unrealizable states.
    recorded = {(nonterminal, state) for nonterminal, states in determinization.states.items() for state in states}
    survived = {
        (nonterminal.nonterminal, nonterminal.state)
        for nonterminal in determinization.space.nonterminals()
        if isinstance(nonterminal, ProductNonTerminal)
    }
    assert recorded == survived


def test_the_start_symbol_merges_the_states_of_the_queried_sort():
    """The queried sort loses its abstraction, and nothing else does.

    A caller asks for the terms of ``A``, not for those of ``A`` that abstract to some ``q``, so
    the start symbol carries the clauses of every reachable ``(A, q)``, and each of them once,
    since an instance produces exactly one head state.
    """
    determinization = determinize(avl_space((0, 1, 2)), AVL)
    assert determinization.start == MergedNonTerminal(AVL)

    merged = len(determinization.space[determinization.start])
    per_state = sum(len(determinization.space[ProductNonTerminal(AVL, state)]) for state in determinization.states[AVL])
    assert merged == per_state
    assert (
        sum(1 for nonterminal in determinization.space.nonterminals() if isinstance(nonterminal, MergedNonTerminal))
        == 1
    )


def test_the_coupled_avl_space_is_the_case_the_table_form_refuses():
    """The contrast the whole construction exists for, stated as a test.

    The same language, twice: as a predicate over two holes, where counting means materializing
    the search tree, and as an abstraction, where it is a table over the program. The refusal is
    not a limitation of the implementation. The table would overcount, because the residual at a
    coupled node is a relation and not a product.
    """
    with pytest.raises(ValueError, match="reading a hole"):
        size_table(avl_coupled_space((0, 1, 2)), 13)

    determinization = determinize(avl_space((0, 1, 2)), AVL)
    size_table(determinization.space, 13)


def test_a_bound_far_above_the_language_says_what_the_tight_bound_says():
    """A table filled far past the largest tree agrees with the tight one and adds nothing to it.

    The AVL space over ten keys is empty above a size of 41, a valid tree having at most ten nodes,
    and only eleven sizes are occupied at all. A caller who does not know that bound in advance
    passes a generous one instead, and what this pins is that the generosity costs correctness
    nothing: the wide table says exactly what the tight one says within the tight bound, and it
    carries no count at all above the largest tree.

    The determinization's own cost does not depend on the bound. What the bound reaches is the
    table fill that runs on the program afterwards.
    """
    determinization = determinize(avl_space(tuple(range(10))), AVL)
    tight = size_table(determinization.space, 41)
    generous = size_table(determinization.space, 401)

    row = {size: tight.of(determinization.start, size) for size in range(42) if tight.of(determinization.start, size)}
    assert row == {
        1: 1,
        5: 10,
        9: 90,
        13: 120,
        17: 840,
        21: 1512,
        25: 840,
        29: 2040,
        33: 1440,
        37: 440,
        41: 60,
    }
    assert sum(row.values()) == 7393, "the AVL benchmark has 7393 trees over ten keys"
    assert determinization.state_count == 94, "the figure the module docstring names for ten keys"
    assert all(
        generous.of(determinization.start, size) == tight.of(determinization.start, size) for size in range(42)
    ), "the generous table disagrees with the tight one within the tight bound"
    assert not any(
        generous.of(nonterminal, size)
        for nonterminal in determinization.space.nonterminals()
        for size in range(42, 402)
    ), "the space is empty above 41, so no row above it may carry a count"
