"""A membership oracle for the counting tests, independent of the counting recursion.

The oracle must not share machinery with what it checks. Enumerating the space with depth-first
search would be circular in the wrong direction as well as intractable: a depth bound admits
exponentially more terms than a size bound, so a binary combinator makes the enumeration explode
long before the size filter sees it. Instead this builds *every* term over the signature up to the
size bound, which is a purely syntactic construction, and asks the checker which of them the space
contains. The checker is a separate decision procedure (``contains_tree``), so a count compared
against it is genuinely independent of the recursion that produced it.

A signature is the combinators of a reference space with their arities. The literal values of a
space are leaves of its terms, so they carry arity 0 here and the checker rejects whatever
combination of them the space does not contain.

A plain module rather than a ``conftest.py``, for the reason given in :mod:`tests.search_fixtures`.
"""

from itertools import product

from cosy.core.tree import Tree
from cosy.search import checker
from tests.search_fixtures import (
    add,
    alt,
    base,
    cons_0,
    cons_1,
    cons_2,
    halt,
    lit,
    marked,
    merge,
    neg,
    nil,
    one,
    pair,
    stop,
    tag,
    wrap,
    zero,
)

LIST_SIGNATURE = {nil: 0, cons_0: 1, cons_1: 1, cons_2: 1}
EXPR_SIGNATURE = {lit: 0, neg: 1, add: 2}
PAIR_SIGNATURE = {zero: 0, one: 0, wrap: 1, pair: 2}
AMBIGUOUS_SIGNATURE = {base: 0, alt: 0, merge: 1}
TAGGED_SIGNATURE = {stop: 0, tag: 2, 0: 0, 1: 0}
CHAIN_SIGNATURE = {halt: 0, marked: 2, 0: 0, 1: 0}


def terms_up_to(signature, bound):
    """Build every term over a signature with at most ``bound`` function-symbol occurrences.

    Args:
        signature (dict): The function symbols with their arities.
        bound (int): The size bound ``D``.

    Returns:
        list[Tree]: All terms of size 1 to ``bound``, whether or not any space contains them.
    """
    by_size: dict[int, list[Tree]] = {size: [] for size in range(bound + 1)}
    for size in range(1, bound + 1):
        for symbol, arity in signature.items():
            if arity == 0:
                if size == 1:
                    by_size[size].append(Tree(symbol, ()))
                continue
            for split in _compositions(size - 1, arity):
                for children in product(*(by_size[part] for part in split)):
                    by_size[size].append(Tree(symbol, children))
    return [term for size in range(1, bound + 1) for term in by_size[size]]


def _compositions(total, parts):
    """Split a total into a fixed number of positive parts, in every way.

    Args:
        total (int): The number to split.
        parts (int): The number of parts.

    Yields:
        tuple[int, ...]: One composition per yield.
    """
    if parts == 1:
        if total >= 1:
            yield (total,)
        return
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


def inhabitants_within(space, start, signature, bound):
    """List every inhabitant of size at most ``bound``, by generate-and-check.

    Args:
        space (SolutionSpace): The space to test against.
        start: The queried non-terminal.
        signature (dict): The function symbols with their arities.
        bound (int): The size bound ``D``.

    Returns:
        list[Tree]: The inhabitants of size at most ``bound``.
    """
    return [term for term in terms_up_to(signature, bound) if checker(space, start, term)]


def cost_counts(space, start, signature, bound, cost):
    """Count the inhabitants within the bound per cost value, by brute force.

    Args:
        space (SolutionSpace): The space to test against.
        start: The queried non-terminal.
        signature (dict): The function symbols with their arities.
        bound (int): The size bound ``D``.
        cost (Callable): The cost function on ground terms.

    Returns:
        dict: ``N_r``, mapping each realized cost value to its number of inhabitants.
    """
    counts: dict = {}
    for tree in inhabitants_within(space, start, signature, bound):
        value = cost(tree)
        counts[value] = counts.get(value, 0) + 1
    return counts
