"""Reference solution spaces shared by the search tests.

Small enough to enumerate exhaustively, so expected answers are computed rather than asserted.

* ``expression_space``: a unary and a binary combinator over one sort, so terms of one size
  differ in shape.
* ``constrained_space``: an external predicate couples two argument positions, so completing
  them independently produces terms the space does not contain.
* ``list_space``, ``ambiguous_space``, ``two_symbol_clause_space``, ``chain_space``: the reference
  spaces the branch counts are checked against, exhaustively and against a closed form. The
  ambiguous one derives one term twice, and the last two write two symbols per clause, which a
  size bookkeeping that counted applications would get wrong.
* ``cut_space``, ``literal_predicate_space``, ``two_offender_space``, ``anonymous_hole_space``,
  ``offset_cut_space``: the spaces that decide which predicates a table indexed by the
  non-terminal can count, one shape of predicate each. ``two_predicate_space`` carries two
  predicates on one clause, where "all of them hold" and "some of them holds" differ.
* ``recognizable_cut_space``, ``recognizable_pair_space``, ``avl_space``: predicates stated as
  an abstraction and a relation on its values, so that a determinization can compile them into the
  non-terminals. Two of them have a control that states the *same* condition as a plain predicate,
  ``cut_space`` and ``avl_coupled_space``, and comparing the languages is what makes those pairs
  say something. ``constrained_space`` is not that for the third: it states a *different*
  condition, term inequality rather than core inequality, and the pair is a contrast between a
  condition inside the recognizable class and one outside it.

A plain module rather than a ``conftest.py``: ``Tree`` memoizes its positions on the instance and
the suite runs randomized and in parallel, so a shared instance would let tests observe each
other's cache state.
"""

from collections.abc import Mapping
from typing import Any

from cosy.core import Constructor, SpecificationBuilder, Synthesizer
from cosy.core.types import Arrow, DataGroup, Intersection

# ---------------------------------------------------------------------------
# Expressions: E -> lit | neg(E) | add(E, E)
# ---------------------------------------------------------------------------

EXPR = Constructor("E")


def lit() -> str:
    """Build the literal.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "x"


def neg(inner: str) -> str:
    """Negate an expression.

    Args:
        inner (str): The interpreted operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"-{inner}"


def add(left: str, right: str) -> str:
    """Add two expressions.

    Args:
        left (str): The interpreted left operand.
        right (str): The interpreted right operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"({left}+{right})"


def expression_space():
    """Build the expression space.

    A unary and a binary combinator, so terms of one size differ in shape.

    Returns:
        SolutionSpace: The space, started at ``E``.
    """
    specs = {
        lit: SpecificationBuilder().suffix(EXPR),
        neg: SpecificationBuilder().argument("inner", EXPR).suffix(EXPR),
        add: SpecificationBuilder().argument("left", EXPR).argument("right", EXPR).suffix(EXPR),
    }
    return Synthesizer(specs).construct_solution_space(EXPR)


# ---------------------------------------------------------------------------
# Coupled arguments: W -> zero | one | wrap(W),  Pair -> pair(W, W), left != right
# ---------------------------------------------------------------------------

WORD = Constructor("W")
PAIR = Constructor("Pair")


def zero() -> str:
    """Build the first word.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "0"


def one() -> str:
    """Build the second word.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "1"


def wrap(inner: str) -> str:
    """Wrap a word. Makes ``W`` recursive, so a depth bound bites.

    Args:
        inner (str): The interpreted operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"w({inner})"


def pair(left: str, right: str) -> str:
    """Pair two words.

    Args:
        left (str): The interpreted left word.
        right (str): The interpreted right word.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"<{left},{right}>"


def different(substitution: Mapping[str, Any]) -> bool:
    """Decide whether the two paired words differ.

    Couples the two argument positions: what completes one depends on what stands in the other.

    Args:
        substitution (Mapping[str, Any]): The chosen arguments of the clause, by variable name.

    Returns:
        bool: True when the two words are different terms.
    """
    return bool(substitution["left"] != substitution["right"])


def constrained_space():
    """Build the space whose predicate couples two argument positions.

    Returns:
        SolutionSpace: The space, started at ``Pair``.
    """
    specs = {
        zero: SpecificationBuilder().suffix(WORD),
        one: SpecificationBuilder().suffix(WORD),
        wrap: SpecificationBuilder().argument("inner", WORD).suffix(WORD),
        pair: SpecificationBuilder().argument("left", WORD).argument("right", WORD).constraint(different).suffix(PAIR),
    }
    return Synthesizer(specs).construct_solution_space(PAIR)


# ---------------------------------------------------------------------------
# N -- a constant argument:  N -> val(v) | plus(v, N),  v in {0, 1}
# ---------------------------------------------------------------------------

NUM = Constructor("N")


def val(v: int) -> str:
    """Build the literal of the numeric space.

    Args:
        v (int): The chosen constant.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return str(v)


def plus(v: int, inner: str) -> str:
    """Add a constant to a numeric term.

    Args:
        v (int): The chosen constant.
        inner (str): The interpreted operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"({v}+{inner})"


def literal_space():
    """Build the space whose clauses carry a constant argument next to a non-terminal one.

    The two kinds of argument are recorded in different places of a goal. A constant argument is
    grounded from the start and never becomes a subgoal, so a goal of this space carries a
    grounded position that has no entry in ``constructors``. The spaces above have no constant
    argument, and on them a goal's ``grounded`` map holds nothing a walk over ``constructors``
    could not rebuild.

    Returns:
        SolutionSpace: The space, started at ``N``.
    """
    digits = DataGroup("int", [0, 1])
    specs = {
        val: SpecificationBuilder().parameter("v", digits).suffix(NUM),
        plus: SpecificationBuilder().parameter("v", digits).argument("inner", NUM).suffix(NUM),
    }
    return Synthesizer(specs).construct_solution_space(NUM)


def wide_first_space():
    """Build the expression space with its clauses declared widest first.

    ``expression_space`` declares ``lit`` before ``neg`` before ``add``, so its program order
    already puts the narrowest clause in front and a clause order that does the same cannot be
    told from it. Here the declaration is reversed, which separates the two.

    Returns:
        SolutionSpace: The space, started at ``EXPR``.
    """
    specs = {
        add: SpecificationBuilder().argument("left", EXPR).argument("right", EXPR).suffix(EXPR),
        neg: SpecificationBuilder().argument("inner", EXPR).suffix(EXPR),
        lit: SpecificationBuilder().suffix(EXPR),
    }
    return Synthesizer(specs).construct_solution_space(EXPR)


# ---------------------------------------------------------------------------
# Width, equal width: Width -> c0 | u1(Width) | u2(Width)
# ---------------------------------------------------------------------------

WIDTH = Constructor("Width")


def c0() -> str:
    """Build the nullary clause of the equal-width space.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "."


def u1(inner: str) -> str:
    """Wrap a term with the first of the two unary clauses.

    Args:
        inner (str): The interpreted operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"u1({inner})"


def u2(inner: str) -> str:
    """Wrap a term with the second of the two unary clauses.

    Args:
        inner (str): The interpreted operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"u2({inner})"


def equal_width_space():
    """Build the space whose two unary clauses open equally many holes.

    The default clause order sorts by the number of holes a clause opens and is stable, so it
    keeps the order it was handed between clauses of equal width. ``u1`` and ``u2`` are that pair,
    which is what makes an ordering effect observable here at all.

    Returns:
        SolutionSpace: The space, started at ``Width``.
    """
    specs = {
        c0: SpecificationBuilder().suffix(WIDTH),
        u1: SpecificationBuilder().argument("inner", WIDTH).suffix(WIDTH),
        u2: SpecificationBuilder().argument("inner", WIDTH).suffix(WIDTH),
    }
    return Synthesizer(specs).construct_solution_space(WIDTH)


# ---------------------------------------------------------------------------
# S, two immediate successes: S -> s1 | s2
# ---------------------------------------------------------------------------

START = Constructor("S")


def s1() -> str:
    """Build the first of the two immediately succeeding clauses.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "s1"


def s2() -> str:
    """Build the second of the two immediately succeeding clauses.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "s2"


def nullary_start_space():
    """Build the space whose start symbol has two immediately succeeding clauses.

    Both initial goals are success nodes, so this space observes what a search rule does with
    them before any expansion has happened.

    Returns:
        SolutionSpace: The space, started at ``S``.
    """
    specs = {s1: SpecificationBuilder().suffix(START), s2: SpecificationBuilder().suffix(START)}
    return Synthesizer(specs).construct_solution_space(START)


A = Constructor("A")
B = Constructor("B")
C = Constructor("C")


# ---------------------------------------------------------------------------
# A space whose target is reached by two clauses sharing their terminal
# ---------------------------------------------------------------------------


def c_ab() -> str:
    """Build the constant that inhabits both argument sorts.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "c"


def a_only() -> str:
    """Build the constant that inhabits ``A`` alone.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "a"


def b_only() -> str:
    """Build the constant that inhabits ``B`` alone.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "b"


def wrap_c(x: str) -> str:
    """Build the unary combinator with two paths onto ``C``.

    Args:
        x (str): The interpreted child.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"f({x})"


def multi_path_space():
    """Build the space in which one position is reached by two clauses.

    ``wrap_c : (A -> C) & (B -> C)`` has two paths of arity one onto ``C``, so the inhabitation
    emits two clauses for ``C`` that share their terminal and differ in their argument sort. The
    completions of a variable under ``wrap_c`` are the union: ``c_ab`` and ``a_only`` through ``A``,
    ``c_ab`` and ``b_only`` through ``B``.

    Returns:
        SolutionSpace: The space, started at ``C``.
    """
    specs = {
        c_ab: Intersection(A, B),
        a_only: A,
        b_only: B,
        wrap_c: Intersection(Arrow(A, C), Arrow(B, C)),
    }
    return Synthesizer(specs).construct_solution_space(C)


D = Constructor("D")


def top_d(inner: str) -> str:
    """Put one more level above ``C``.

    Args:
        inner (str): The interpreted operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"top({inner})"


def disjoint_multi_path_space(start=None):
    """Build the multi-path space in which the two clauses share no completion.

    ``multi_path_space`` cannot show which clause a search explored first, because ``c_ab``
    inhabits both argument sorts and is reachable either way. Dropping it makes the two clauses
    disjoint: the ``A`` clause reaches only ``a_only``, the ``B`` clause only ``b_only``, so the
    first streamed term names the clause that was taken.

    ``top_d`` sits above ``C`` so that a hole at ``(0, 0)`` reaches the ambiguous expansion
    through the walk of ``goal_from_tree`` rather than through its initial goals.

    Args:
        start (NT | None): The non-terminal to start from, ``C`` or ``D``. (Default value = None,
            which starts at ``D``)

    Returns:
        SolutionSpace: The space.
    """
    specs = {
        a_only: A,
        b_only: B,
        wrap_c: Intersection(Arrow(A, C), Arrow(B, C)),
        top_d: Arrow(C, D),
    }
    return Synthesizer(specs).construct_solution_space(D if start is None else start)


# ---------------------------------------------------------------------------
# Lists over {0, 1, 2}: List -> nil | cons_0(List) | cons_1(List) | cons_2(List)
# ---------------------------------------------------------------------------

LIST = Constructor("List")


def nil() -> str:
    """Build the empty list.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "[]"


def cons_0(rest: str) -> str:
    """Prepend the digit 0.

    Args:
        rest (str): The interpreted tail.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"0:{rest}"


def cons_1(rest: str) -> str:
    """Prepend the digit 1.

    Args:
        rest (str): The interpreted tail.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"1:{rest}"


def cons_2(rest: str) -> str:
    """Prepend the digit 2.

    Args:
        rest (str): The interpreted tail.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"2:{rest}"


def list_space():
    """Build the list space: three unary clauses over one nullary one.

    A list of length ``l`` is a term of size ``l + 1``, and ``3^l`` lists share that size, so the
    realized sizes within a bound ``D`` are ``1..D`` and the size-uniform weight of a list is
    ``1 / (D * 3^l)``.  Under it every length carries total weight ``1/D``, which is the claim
    a closed form makes.

    Returns:
        SolutionSpace: The space, started at ``List``.
    """
    specs = {
        nil: SpecificationBuilder().suffix(LIST),
        cons_0: SpecificationBuilder().argument("rest", LIST).suffix(LIST),
        cons_1: SpecificationBuilder().argument("rest", LIST).suffix(LIST),
        cons_2: SpecificationBuilder().argument("rest", LIST).suffix(LIST),
    }
    return Synthesizer(specs).construct_solution_space(LIST)


# ---------------------------------------------------------------------------
# Ambiguous: base : X & Y, alt : X, merge : (X -> M) & (Y -> M), target M
# ---------------------------------------------------------------------------

X = Constructor("X")
Y = Constructor("Y")
AMBIGUOUS_TARGET = Constructor("M")


def base() -> str:
    """Build the term that inhabits both argument sorts.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "b"


def alt() -> str:
    """Build the term that inhabits the first argument sort alone.

    The contrast to ``base``: ``merge(alt)`` has one derivation, ``merge(base)`` has two, so the
    space carries both cases and a tool that reported ambiguity everywhere would fail here.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "a"


def merge(inner: str) -> str:
    """Turn an ``X`` or a ``Y`` into an ``M``.

    Args:
        inner (str): The interpreted argument.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"m({inner})"


def ambiguous_space():
    """Build a space in which one inhabitant ends more than one success branch.

    ``merge`` has two paths of arity one onto ``M``, so the inhabitation emits two clauses for
    ``M`` that share their terminal and differ in the sort they ask of their argument.  ``base``
    inhabits both sorts, so ``merge(base)`` is derived twice, once through each clause, while
    ``merge(alt)`` is derived once.

    The earlier version of this space put the intersection on the nullary combinators instead
    (``base, alt : X & Y`` with ``merge : (X, Y) -> M``) and was measured to be *unambiguous*:
    for arity 0 all admissible subsets of paths yield the same clause, so no nullary combinator
    can produce ambiguity, whatever its type. Unambiguity within a bound therefore rests on a
    structural precondition and not on the types alone.

    Returns:
        SolutionSpace: The space, started at ``M``.
    """
    specs = {
        base: SpecificationBuilder().suffix(Intersection(X, Y)),
        alt: SpecificationBuilder().suffix(X),
        merge: SpecificationBuilder().suffix(Intersection(Arrow(X, AMBIGUOUS_TARGET), Arrow(Y, AMBIGUOUS_TARGET))),
    }
    return Synthesizer(specs).construct_solution_space(AMBIGUOUS_TARGET)


# ---------------------------------------------------------------------------
# Two symbols per clause: Digit -> 0 | 1, Tagged -> stop | tag(d: Digit, Tagged)
# ---------------------------------------------------------------------------

TAGGED = Constructor("Tagged")
DIGITS = DataGroup("digit", (0, 1))


def stop() -> str:
    """End a tagged chain.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "."


def tag(d: int, rest: str) -> str:
    """Prepend a literal digit to a tagged chain.

    Args:
        d (int): The literal argument.
        rest (str): The interpreted rest of the chain.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"{d}{rest}"


def two_symbol_clause_space():
    """Build the space whose clauses write two symbols each.

    A clause fixes its terminal *and* every constant argument, so ``tag`` grows a partial
    inhabitant by two symbols per application while every other reference space here grows it by
    one.  A size bookkeeping that counted applications instead of symbols would agree with the
    truth everywhere else and be wrong only here, and it is the shape a repository with literal
    parameters takes throughout.

    Sizes are odd throughout (``stop`` is 1, each ``tag`` adds 2), which makes the space the
    natural test of a draw that spreads over realized sizes when those have gaps in them.

    Returns:
        SolutionSpace: The space, started at ``Tagged``.
    """
    specs = {
        stop: SpecificationBuilder().suffix(TAGGED),
        tag: SpecificationBuilder().parameter("d", DIGITS).argument("rest", TAGGED).suffix(TAGGED),
    }
    return Synthesizer(specs).construct_solution_space(TAGGED)


# ---------------------------------------------------------------------------
# A predicate on ONE hole: V -> v0 | v1 | vw(V), Box -> box(inner: V) with size(inner) <= 2
# ---------------------------------------------------------------------------

CUT_SORT = Constructor("V")
BOX = Constructor("Box")


def v_zero() -> str:
    """Build the first value.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "0"


def v_one() -> str:
    """Build the second value.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "1"


def v_wrap(inner: str) -> str:
    """Wrap a value, so that the sort is recursive and the bound bites.

    Args:
        inner (str): The interpreted operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"v({inner})"


def box(inner: str) -> str:
    """Box a value.

    Args:
        inner (str): The interpreted operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"[{inner}]"


def short(substitution: Mapping[str, Any]) -> bool:
    """Accept only small values: a predicate over a single hole.

    Args:
        substitution (Mapping[str, Any]): The grounded arguments of the clause, by variable name.

    Returns:
        bool: True when the boxed value has at most two symbols.
    """
    return substitution["inner"].size <= 2


def cut_space():
    """Build the space whose predicate reads exactly one hole.

    The contrast to :func:`constrained_space`: nothing here is *coupled*, since there is one
    clause, one hole and one predicate, and the table form is wrong all the same. The residual at
    the hole is a proper subset of the hole's language, and ``N_V(s)`` cannot see the difference: the exact
    counts are ``{2: 2, 3: 2}`` where a table indexed by the non-terminal says ``{2: 2, 3: 2,
    4: 2, ...}`` for as far as the bound reaches.  So "no coupling" is not the hypothesis the
    table needs; "no predicate reaching into a hole" is.

    Returns:
        SolutionSpace: The space, started at ``Box``.
    """
    specs = {
        v_zero: SpecificationBuilder().suffix(CUT_SORT),
        v_one: SpecificationBuilder().suffix(CUT_SORT),
        v_wrap: SpecificationBuilder().argument("inner", CUT_SORT).suffix(CUT_SORT),
        box: SpecificationBuilder().argument("inner", CUT_SORT).constraint(short).suffix(BOX),
    }
    return Synthesizer(specs).construct_solution_space(BOX)


# ---------------------------------------------------------------------------
# A predicate on literals alone: Graded -> grade(d: Digit) with d > 0, Used -> use(Graded)
# ---------------------------------------------------------------------------

GRADED = Constructor("Graded")
USED = Constructor("Used")
GRADES = DataGroup("grade", (0, 1, 2))


def grade(d: int) -> str:
    """Build a graded value from a literal digit.

    Args:
        d (int): The literal.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return str(d)


def use(v: str) -> str:
    """Wrap a graded value, so that ``grade`` is applied at an inner position.

    The wrapper is the whole point: at the root a clause without holes runs through
    ``Goal.from_rhs_rule``, which checks its predicate; at an inner position it runs through
    ``Goal.update``, which is where the check used to be missing.

    Args:
        v (str): The interpreted graded value.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"u({v})"


def positive(substitution: Mapping[str, Any]) -> bool:
    """Reject the grade zero: a predicate over the literals of its clause alone.

    Args:
        substitution (Mapping[str, Any]): The grounded arguments of the clause, by variable name.

    Returns:
        bool: True for a nonzero grade.
    """
    return substitution["d"] > 0


def literal_predicate_space():
    """Build the space whose predicate reads no hole at all.

    Such a predicate decides on its clause and nothing else, so it neither couples nor cuts a
    hole: the table form applies, provided it drops the clauses the predicate rejects exactly as
    the engine does. This space is what pins that the two agree: ``grade(0)`` must be absent
    from both, at the root and below it.

    Returns:
        SolutionSpace: The space, started at ``Used``.
    """
    specs = {
        grade: SpecificationBuilder().parameter("d", GRADES).constraint(positive).suffix(GRADED),
        use: SpecificationBuilder().argument("v", GRADED).suffix(USED),
    }
    return Synthesizer(specs).construct_solution_space(USED)


# ---------------------------------------------------------------------------
# The heavy clause first: Chain -> marked(m: Digit, Chain) | halt
# ---------------------------------------------------------------------------

CHAIN = Constructor("Chain")


def halt() -> str:
    """End a marked chain: the clause that fits into the smallest size.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "|"


def marked(m: int, rest: str) -> str:
    """Prepend a literal mark to a chain: the clause that needs two symbols of its own.

    Args:
        m (int): The literal argument.
        rest (str): The interpreted rest of the chain.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"{m}~{rest}"


def chain_space():
    """Build the space that states its heaviest clause first.

    Structurally this is :func:`two_symbol_clause_space` with the clause order reversed, and the order is
    the whole point.  The fill of the size table walks the clauses of a non-terminal at each size
    and has to *skip* the ones that do not fit into it yet.  Where the light clause stands first
    (as it does in every other space here), skipping and abandoning the clause list are the
    same thing, because nothing follows the first clause that does not fit.  Here ``marked`` needs
    two symbols and stands first, so at size 1 the fill meets a clause that does not fit before it
    meets ``halt``: a fill that abandoned the list there would return an empty table for a space
    with infinitely many inhabitants, and "empty within the bound" is a legitimate answer that
    nothing else would flag.

    Returns:
        SolutionSpace: The space, started at ``Chain``.
    """
    specs = {
        marked: SpecificationBuilder().parameter("m", DIGITS).argument("rest", CHAIN).suffix(CHAIN),
        halt: SpecificationBuilder().suffix(CHAIN),
    }
    return Synthesizer(specs).construct_solution_space(CHAIN)


# ---------------------------------------------------------------------------
# Two offending clauses: W -> zero | one | wrap(W), Box -> box(W), Pair -> pair(Box, Box)
# ---------------------------------------------------------------------------


def two_offender_space():
    """Build the program that breaks the decomposition hypothesis in two places at once.

    ``pair`` couples its two holes and ``box`` cuts its one, and both are reachable from the same
    start symbol, so the program has two offending clauses of *different* kind.  The report of
    :func:`cosy.search.counting.coupled_clauses` promises to name every such clause; with one
    offender per program, which is all :func:`constrained_space` and :func:`cut_space` have,
    a report that stopped after the first would be indistinguishable from one that does not.

    Returns:
        SolutionSpace: The space, started at ``Pair``.
    """
    specs = {
        zero: SpecificationBuilder().suffix(WORD),
        one: SpecificationBuilder().suffix(WORD),
        wrap: SpecificationBuilder().argument("inner", WORD).suffix(WORD),
        box: SpecificationBuilder().argument("inner", WORD).constraint(short).suffix(BOX),
        pair: SpecificationBuilder().argument("left", BOX).argument("right", BOX).constraint(different).suffix(PAIR),
    }
    return Synthesizer(specs).construct_solution_space(PAIR)


# ---------------------------------------------------------------------------
# A predicate over an anonymous hole: Seed -> seed, mark : Grade -> (Seed -> Marked)
# ---------------------------------------------------------------------------

SEED = Constructor("Seed")
MARKED = Constructor("Marked")
HELD = Constructor("Held")


def seed() -> str:
    """Build the term the marked clause consumes through its anonymous hole.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "s"


def mark(d: int, inner: str) -> str:
    """Mark a seed with a literal grade.

    Args:
        d (int): The literal argument, read by the clause's predicate.
        inner (str): The interpreted argument of the anonymous hole.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"m{d}({inner})"


def hold(inner: str) -> str:
    """Wrap a marked seed, so that ``mark`` is applied at an inner position too.

    Args:
        inner (str): The interpreted marked seed.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"h({inner})"


def anonymous_hole_space():
    """Build the space whose predicate sits on a clause with an anonymous hole.

    ``mark`` declares one literal parameter and takes its second argument from the *arrow* in its
    suffix, which the inhabitation turns into a hole with no name.  A predicate on such a clause
    can read the literals and nothing else, since an anonymous argument never enters the
    substitution it is handed, so the clause decides once and for all, exactly like the hole-free
    clause of :func:`literal_predicate_space`, and the table form applies to it unchanged.

    This is the only space here that combines a predicate with a hole and still satisfies the
    hypothesis, which makes it the one that can tell a report keyed on "has a hole" from one keyed
    on "has a *named* hole".  It is also the space on which the engine's missing root check was
    found: ``Goal.from_rhs_rule`` evaluated a predicate only for a clause with no subgoals at all,
    so ``mark(0, seed)`` left the search at the root while the table dropped it, and the two forms
    disagreed, and the engine was wrong.

    Returns:
        SolutionSpace: The space, started at ``Held``.
    """
    specs = {
        seed: SpecificationBuilder().suffix(SEED),
        mark: SpecificationBuilder().parameter("d", GRADES).constraint(positive).suffix(Arrow(SEED, MARKED)),
        hold: SpecificationBuilder().argument("held", MARKED).suffix(HELD),
    }
    return Synthesizer(specs).construct_solution_space(HELD)


# ---------------------------------------------------------------------------
# A literal parameter before the hole: Dbox -> dbox(d: Digit, inner: V) with size <= 2
# ---------------------------------------------------------------------------

DEEP_BOX = Constructor("Dbox")


def dbox(d: int, inner: str) -> str:
    """Box a value behind a literal digit.

    Args:
        d (int): The literal argument, standing *before* the hole.
        inner (str): The interpreted boxed value.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"[{d}:{inner}]"


def offset_cut_space():
    """Build the space whose predicate reads a hole that is not the clause's first argument.

    ``dbox`` declares a literal parameter and only then the hole its predicate reads, so the hole
    sits at argument position 1 while it is the clause's *first* hole.  In every other space here
    the two indices coincide, so a report that numbered the holes instead of the arguments would
    be right everywhere else and send the reader of this program to the literal.  A repository is
    repaired by rewriting one argument of one clause, and it is a shape a real repository is built
    from throughout.

    Returns:
        SolutionSpace: The space, started at ``Dbox``.
    """
    specs = {
        v_zero: SpecificationBuilder().suffix(CUT_SORT),
        v_one: SpecificationBuilder().suffix(CUT_SORT),
        v_wrap: SpecificationBuilder().argument("inner", CUT_SORT).suffix(CUT_SORT),
        dbox: SpecificationBuilder()
        .parameter("d", DIGITS)
        .argument("inner", CUT_SORT)
        .constraint(short)
        .suffix(DEEP_BOX),
    }
    return Synthesizer(specs).construct_solution_space(DEEP_BOX)


# ---------------------------------------------------------------------------
# Two predicates on one clause: Graded -> grade(d: Grade) with d > 0 and d < 2
# ---------------------------------------------------------------------------


def below_two(substitution: Mapping[str, Any]) -> bool:
    """Reject the grade two: the second predicate of a clause that carries two.

    Paired with :func:`positive` it leaves exactly one admissible grade, and each of the two
    rejected grades fails exactly one of the two: 0 fails ``positive`` and passes this one,
    2 passes ``positive`` and fails this one. That is what a filter reading "some predicate
    holds" cannot survive.

    Args:
        substitution (Mapping[str, Any]): The grounded arguments of the clause, by variable name.

    Returns:
        bool: True for a grade below two.
    """
    return substitution["d"] < 2


def two_predicate_space():
    """Build the space whose clause carries two predicates at once.

    ``grade`` is admissible only where *both* predicates hold, so of the three grades exactly one
    survives.  Every other space here has at most one predicate per clause, and with one predicate
    "all of them hold" and "some of them holds" are the same condition, so this space is the only
    one on which the two come apart, in the engine and in the fill of the size table alike.

    Returns:
        SolutionSpace: The space, started at ``Used``.
    """
    specs = {
        grade: SpecificationBuilder().parameter("d", GRADES).constraint(positive).constraint(below_two).suffix(GRADED),
        use: SpecificationBuilder().argument("v", GRADED).suffix(USED),
    }
    return Synthesizer(specs).construct_solution_space(USED)


# ---------------------------------------------------------------------------
# A clause of three holes: Ter -> leaf | tri(Ter, Ter, Ter)
# ---------------------------------------------------------------------------

TERNARY = Constructor("Ter")


def leaf() -> str:
    """End a ternary tree.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "l"


def tri(left: str, middle: str, right: str) -> str:
    """Join three ternary trees.

    Args:
        left (str): The interpreted first subtree.
        middle (str): The interpreted second subtree.
        right (str): The interpreted third subtree.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"t({left},{middle},{right})"


def ternary_space():
    """Build the space whose clause has three holes.

    A clause of two holes is convolved in one step: give the first hole a size, read the second
    hole's row at what is left. A clause of three cannot be, because the pair of remaining holes
    has to be known at a whole range of sizes, so that pair gets a row of its own. Every other
    reference space here has clauses of at most two holes and never reaches that machinery.

    The counts are known in advance. A ternary tree with ``n`` inner nodes has ``3n + 1`` symbols,
    and there are ``binomial(3n, n) / (2n + 1)`` such trees, so the sizes are one apart by three
    and the counts are 1, 1, 3, 12, 55.

    Returns:
        SolutionSpace: The space, started at ``Ter``.
    """
    specs = {
        leaf: SpecificationBuilder().suffix(TERNARY),
        tri: SpecificationBuilder()
        .argument("left", TERNARY)
        .argument("middle", TERNARY)
        .argument("right", TERNARY)
        .suffix(TERNARY),
    }
    return Synthesizer(specs).construct_solution_space(TERNARY)


# ---------------------------------------------------------------------------
# Every arity on one head: H -> h_leaf | h_lit(d: Digit) | h_step(H) | h_join(H, H)
# ---------------------------------------------------------------------------

MIXED = Constructor("H")


def h_leaf() -> str:
    """End a mixed-arity term without writing a literal.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "l"


def h_lit(d: int) -> str:
    """End a mixed-arity term by writing a literal, which costs a second symbol.

    Args:
        d (int): The literal argument.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"{d}"


def h_step(inner: str) -> str:
    """Extend a mixed-arity term by one symbol.

    Args:
        inner (str): The interpreted rest.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"s({inner})"


def h_join(left: str, right: str) -> str:
    """Join two mixed-arity terms, which is the clause that reads the sizes the head occupies.

    Args:
        left (str): The interpreted left term.
        right (str): The interpreted right term.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"j({left},{right})"


def mixed_arity_space():
    """Build the space whose head is written by a clause of every arity, two of them at one size.

    The fill walks the sizes a non-terminal occupies, and it records a size as occupied the first
    time a clause writes into it. Here two clauses write into size 2: ``h_lit`` fixes its terminal
    and its literal, and ``h_step`` fixes its terminal above a term of size 1. A fill that recorded
    the size once per clause instead of once per row would hand ``h_join`` the same size twice, and
    every count above it would come out too large.

    Returns:
        SolutionSpace: The space, started at ``H``.
    """
    specs = {
        h_leaf: SpecificationBuilder().suffix(MIXED),
        h_lit: SpecificationBuilder().parameter("d", DIGITS).suffix(MIXED),
        h_step: SpecificationBuilder().argument("inner", MIXED).suffix(MIXED),
        h_join: SpecificationBuilder().argument("left", MIXED).argument("right", MIXED).suffix(MIXED),
    }
    return Synthesizer(specs).construct_solution_space(MIXED)


# ---------------------------------------------------------------------------
# Two hole tuples of one length: Ha -> a_zero | a_one ; Hb -> b_single ;
# Ht -> same_holes(Ha, Ha) | mixed_holes(Ha, Hb)
# ---------------------------------------------------------------------------

SORT_A = Constructor("Ha")
SORT_B = Constructor("Hb")
TUPLE_SORT = Constructor("Ht")


def a_zero() -> str:
    """Build the first inhabitant of the two-element sort.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "a0"


def a_one() -> str:
    """Build the second inhabitant of the two-element sort.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "a1"


def b_single() -> str:
    """Build the single inhabitant of the one-element sort.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "b"


def same_holes(left: str, right: str) -> str:
    """Build the term whose two holes carry the same sort.

    Args:
        left (str): The interpreted left operand.
        right (str): The interpreted right operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"s({left},{right})"


def mixed_holes(left: str, right: str) -> str:
    """Build the term whose two holes carry different sorts.

    Args:
        left (str): The interpreted left operand.
        right (str): The interpreted right operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"g({left},{right})"


def hole_tuple_space():
    """Build the space with two different hole tuples of one length.

    ``same_holes`` opens ``(Ha, Ha)`` and ``mixed_holes`` opens ``(Ha, Hb)``: two tuples of two
    holes each, and the sorts are of *different* size (``N_Ha(1) = 2``, ``N_Hb(1) = 1``), so the
    two tuples admit different numbers of splits at the same total, 4 against 2 at total 2. A
    convolution cache keyed by the number of holes rather than by the holes themselves would
    answer the second tuple with the first tuple's number, and the start row would read 8 where it
    must read 6.

    Every other space here has at most one hole tuple per length, so nothing else can tell the two
    keys apart.

    Returns:
        SolutionSpace: The space, started at ``Ht``.
    """
    specs = {
        a_zero: SpecificationBuilder().suffix(SORT_A),
        a_one: SpecificationBuilder().suffix(SORT_A),
        b_single: SpecificationBuilder().suffix(SORT_B),
        same_holes: SpecificationBuilder().argument("left", SORT_A).argument("right", SORT_A).suffix(TUPLE_SORT),
        mixed_holes: SpecificationBuilder().argument("left", SORT_A).argument("right", SORT_B).suffix(TUPLE_SORT),
    }
    return Synthesizer(specs).construct_solution_space(TUPLE_SORT)


# ---------------------------------------------------------------------------
# The cut of V, stated through an abstraction: alpha = capped size, R = "at most two"
# ---------------------------------------------------------------------------


def capped_size(_symbol, states) -> int:
    """Abstract a term by its size, capped at three.

    An algebra with a three-element carrier: sizes 1 and 2 are told apart and everything from 3 on
    is one class. That is enough to decide "at most two symbols" and no more, which is what makes
    it finite. The uncapped size is the standard slip and has no finite carrier at all.

    Args:
        _symbol: The function symbol. The size does not depend on which one it is.
        states: The sizes of the arguments.

    Returns:
        int: ``min(1 + sum of the argument sizes, 3)``.
    """
    return min(1 + sum(states), 3)


def at_most_two(substitution) -> bool:
    """Accept a boxed value of at most two symbols, which is the relation on states.

    Args:
        substitution: The clause's substitution, with the hole carrying its abstraction.

    Returns:
        bool: True when the capped size of the boxed value is at most two.
    """
    return substitution["inner"] <= 2


def recognizable_cut_space():
    """Build :func:`cut_space` with its predicate stated as an abstraction and a relation.

    Same language, same counts, same clause structure, and countable from the program, because a
    determinization can split ``V`` into the sorts "one symbol", "two symbols" and "three or more"
    and drop the predicate. The pair with :func:`cut_space` is the point: a repository decides
    whether the table form applies to it, by how it states its constraint.

    Returns:
        SolutionSpace: The space, started at ``Box``.
    """
    specs = {
        v_zero: SpecificationBuilder().suffix(CUT_SORT),
        v_one: SpecificationBuilder().suffix(CUT_SORT),
        v_wrap: SpecificationBuilder().argument("inner", CUT_SORT).suffix(CUT_SORT),
        box: SpecificationBuilder()
        .argument("inner", CUT_SORT)
        .recognizable_constraint(capped_size, at_most_two)
        .suffix(BOX),
    }
    return Synthesizer(specs).construct_solution_space(BOX)


# ---------------------------------------------------------------------------
# Coupled and still recognizable: Pair -> pair(W, W) with different innermost letters
# ---------------------------------------------------------------------------


def innermost(symbol, states) -> int:
    """Abstract a word by the letter at its core.

    Args:
        symbol: The function symbol.
        states: The abstractions of the arguments, empty on a letter.

    Returns:
        int: 0 or 1, the letter ``wrap`` was applied to.
    """
    return states[0] if states else (0 if symbol is zero else 1)


def different_cores(substitution) -> bool:
    """Accept two words whose innermost letters differ, which is the relation on states.

    Args:
        substitution: The clause's substitution, with both holes carrying their abstraction.

    Returns:
        bool: True when the two letters differ.
    """
    return substitution["left"] != substitution["right"]


def recognizable_pair_space():
    """Build a space whose predicate couples two holes and still factors through two states.

    The contrast is to :func:`constrained_space`, which is this space with ``left != right`` on
    the *terms*. Over a sort with infinitely many terms, term equality has no finite abstraction,
    so that space is outside (REC) and this one is inside it. Both couple, both break the
    decomposition the counting recursion assumes, and only one of them can be determinized, which
    is what makes (REC) the condition rather than "no coupling".

    A word of size ``s`` is ``wrap`` applied ``s - 1`` times to a letter, so there are two words
    of every size, one per core. A pair of size ``n`` spends one symbol on ``pair`` and splits the
    remaining ``n - 1`` over two words of at least one symbol each, which is ``n - 2`` splits, and
    of the four core combinations at each split the relation admits the two that differ. So the
    space holds ``2 (n - 2)`` pairs of size ``n`` for ``n >= 2``.

    Returns:
        SolutionSpace: The space, started at ``Pair``.
    """
    specs = {
        zero: SpecificationBuilder().suffix(WORD),
        one: SpecificationBuilder().suffix(WORD),
        wrap: SpecificationBuilder().argument("inner", WORD).suffix(WORD),
        pair: SpecificationBuilder()
        .argument("left", WORD)
        .argument("right", WORD)
        .recognizable_constraint(innermost, different_cores)
        .suffix(PAIR),
    }
    return Synthesizer(specs).construct_solution_space(PAIR)


# ---------------------------------------------------------------------------
# AVL trees: AVL -> avl_node(key, cached height, AVL, AVL) | avl_leaf
# ---------------------------------------------------------------------------

AVL = Constructor("AVL")
AVL_KEYS = tuple(range(10))
"""The key set of the AVL trees Goldstein and Pierce (2022) generate in their Table 1."""

LEAF_STATE = (0, None, None)
"""``alpha`` of the empty tree: height zero, no smallest key, no greatest key."""


def avl_leaf() -> str:
    """Build the empty tree.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "E"


def avl_node(x: int, h: int, left: str, right: str) -> str:
    """Build an inner node carrying its key and its cached height.

    Args:
        x (int): The key.
        h (int): The cached height.
        left (str): The interpreted left subtree.
        right (str): The interpreted right subtree.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"({left} {x}:{h} {right})"


def avl_summary(symbol, states):
    """Abstract a tree by its height and its extreme keys, which is ``alpha`` for the AVL condition.

    All three parts of the validity condition read only this much of a subtree: balance reads the
    heights, the cache reads the heights, and the ordering reads the extreme keys. The extreme keys
    are bounded by the key set. The height is not bounded by this function, which adds one per
    level without a cap, but by the relation: it admits a node only where the cached height comes
    from a finite group and the ordering is strict, so at most as many nodes nest as there are
    keys. What the determinization reaches is therefore finite even though ``alpha`` alone would
    not make it so.

    Args:
        symbol: The function symbol, or a literal value.
        states: The abstractions of the arguments.

    Returns:
        tuple: ``(height, smallest key, greatest key)`` on a tree, the value itself on a literal.
    """
    if symbol is avl_leaf:
        return LEAF_STATE
    if symbol is avl_node:
        key, _cached, left, right = states
        return (
            1 + max(left[0], right[0]),
            left[1] if left[1] is not None else key,
            right[2] if right[2] is not None else key,
        )
    return symbol  # a literal is its own abstraction


def avl_relation(substitution) -> bool:
    """Decide the validity condition on the states of one node.

    Args:
        substitution: The clause's substitution, with both subtrees carrying their abstraction.

    Returns:
        bool: True when the node is balanced, caches its height correctly and orders its keys.
    """
    key, cached = substitution["x"], substitution["h"]
    left, right = substitution["l"], substitution["r"]
    if abs(left[0] - right[0]) > 1:  # balance
        return False
    if cached != 1 + max(left[0], right[0]):  # the cached height is the true one
        return False
    if left[2] is not None and left[2] >= key:  # ordering, to the left
        return False
    return not (right[1] is not None and right[1] <= key)  # ordering, to the right


def tree_summary(term):
    """Compute height and extreme keys of a grounded AVL term, without the abstraction.

    Written independently of :func:`avl_summary` on purpose. It is what the coupled space's
    predicate reads, and if the two agreed by construction the contrast between the two spaces
    would prove nothing.

    Args:
        term (Tree): A grounded term over ``avl_leaf`` and ``avl_node``.

    Returns:
        tuple: ``(height, smallest key, greatest key)``.
    """
    if not term.children:
        return LEAF_STATE
    key = term.children[0].root
    left = tree_summary(term.children[2])
    right = tree_summary(term.children[3])
    return (
        1 + max(left[0], right[0]),
        left[1] if left[1] is not None else key,
        right[2] if right[2] is not None else key,
    )


def avl_valid(substitution) -> bool:
    """Decide the validity condition on the grounded subtrees of one node.

    Reads both subtrees in full, plus the two literals: a genuine relation over the holes, stated
    the way a repository states one when it has no abstraction to state instead.

    Args:
        substitution: The clause's substitution, two grounded subtrees and two literals.

    Returns:
        bool: True when the node is a valid AVL node.
    """
    key, cached = substitution["x"], substitution["h"]
    left, right = tree_summary(substitution["l"]), tree_summary(substitution["r"])
    if abs(left[0] - right[0]) > 1:
        return False
    if cached != 1 + max(left[0], right[0]):
        return False
    if left[2] is not None and left[2] >= key:
        return False
    return not (right[1] is not None and right[1] <= key)


def _avl_specification(keys, constrain):
    """Build the AVL specification, with the constraint stated either way.

    Args:
        keys (tuple): The admissible key values.
        constrain (Callable): Applied to the ``avl_node`` builder to attach the constraint.

    Returns:
        dict: The specification.
    """
    return {
        avl_leaf: SpecificationBuilder().suffix(AVL),
        avl_node: constrain(
            SpecificationBuilder()
            .parameter("x", DataGroup("key", tuple(keys)))
            .parameter("h", DataGroup("height", tuple(range(1, len(keys) + 1))))
            .argument("l", AVL)
            .argument("r", AVL)
        ).suffix(AVL),
    }


def avl_coupled_space(keys=AVL_KEYS):
    """Build the AVL space with the validity condition as a plain predicate over the holes.

    This is what a repository writes without an abstraction, and what the framework cannot count
    from the program: the predicate reads both subtrees, so the residual at ``avl_node`` is a
    relation and not a product. Counting it means building the retained tree, which is why the
    tests below run it on a few keys and its recognizable twin on all of them.

    A node costs three symbols, the terminal and two literals, and a leaf costs one, so a tree of
    ``k`` nodes has size ``4k + 1``. Ordering is strict, so at most ``len(keys)`` nodes fit and no
    tree exceeds a size of ``4 len(keys) + 1``. That size is reached rather than approached, the
    tree using every key being a tree of the language.

    Args:
        keys (tuple): The admissible key values. (Default value = ``AVL_KEYS``)

    Returns:
        SolutionSpace: The space, started at ``AVL``.
    """
    specs = _avl_specification(keys, lambda builder: builder.constraint(avl_valid))
    return Synthesizer(specs).construct_solution_space(AVL).prune()


def avl_space(keys=AVL_KEYS):
    """Build the AVL space with the validity condition stated as an abstraction and a relation.

    The same language as :func:`avl_coupled_space`, in the form the determinization consumes. The
    height a generator would otherwise have to guess becomes part of the non-terminal, and the
    program that comes out carries no predicate at all.

    Args:
        keys (tuple): The admissible key values. (Default value = ``AVL_KEYS``)

    Returns:
        SolutionSpace: The space, started at ``AVL``.
    """
    specs = _avl_specification(keys, lambda builder: builder.recognizable_constraint(avl_summary, avl_relation))
    return Synthesizer(specs).construct_solution_space(AVL).prune()
