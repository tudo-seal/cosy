"""The cost table: branch counts per cost value of an additive cost algebra, computed from the program.

Random search draws from a prescribed distribution ``pi`` on a cost function: the weight of an
inhabitant is ``w(t) = pi(c(t)) / N_r(c(t))``, and a node's key is drawn from the Gumbel
distribution located at ``log sum_a B_n(a) pi(a) / N_r(a)`` (:mod:`cosy.search.sampling`). The
table form of that construction (:class:`~cosy.search.sampling.WeightedTable`) computes the branch
counts ``B_n`` from the program instead of from the retained derivation tree, but only for the term
size: its rows are indexed by size. This module is the same construction for the cost of an
additive cost algebra (:class:`~cosy.search.costs.AdditiveCostAlgebra`), whose rows are indexed by
cost value instead.

**The recursion.** Applying a clause adds a fixed cost to the partial inhabitant,
:func:`~cosy.search.counting.rule_cost`, the cost of its terminal and of its constant arguments,
and every hole it opens is filled independently, which is what
:func:`~cosy.search.counting.decomposable_or_raise` secures. So the number of terms rooted at a
non-terminal with exactly cost ``a`` obeys

    N_A(a) = sum over clauses A <- F(B_1..B_k) of  sum over a_1 + .. + a_k = a - c(F)  prod_i N_{B_i}(a_i),

the size table's recursion with the clause's cost where the size table has the number of symbols it
writes. Under the algebra that charges one per symbol the two tables coincide, row for row.

**Whole numbers, and a cap.** The rows are indexed by cost value, so the costs must be whole
numbers, and a clause that charges ``0.5`` is refused rather than rounded: rounding changes the
algebra, and a coarsened algebra, parameters counted in thousands, say, is a *different* algebra
that a caller chooses and states, not an approximation this module should make silently. The cap
``C`` is a whole number too. The table counts the terms of cost at most ``C``; a node's branch
counts stop there, and the distribution is spread over the values realized up to it, so a term above
the cap is neither weighted nor streamed. Every cost is nonnegative, so a partial inhabitant that
already costs more than ``C`` has no completion within it and is never expanded.

**When the counts are finite.** The size table needs a size bound because every clause writes a
symbol, so the size grows along every branch. A cost need not grow: a clause may cost nothing. A
loop of clauses of cost zero, each of whose other holes can be filled at cost zero too, pumps without
paying, so one cost value may then be realized by infinitely many terms, and no table can hold that.
The table refuses such a program and names the loop, and it refuses a loop of that shape in a sort
that has no term at all as well, which pruning the program first avoids. A clause of cost zero whose other holes all cost
something is no such edge: going round a loop through it pays for them every time, which is the
shape of a free sequencing combinator over priced parts. Which sorts can be filled at cost zero is
decided first, as a least fixed point over the clauses of cost zero. The refusal is decided on the
whole program, so a loop no query reaches refuses the table all the same. Without such a loop every
cost value is realized by finitely many terms: a term of cost ``a`` applies at most ``a`` clauses of
positive cost and passes at most ``a`` times through a side that costs something, every other step
follows an edge of the acyclic graph of free steps, and every clause has finitely many holes.

**Two fills.** A program is processed one strongly connected component of its dependency graph at a
time, children first, so every row a component reads outside itself is complete.

* A component without a cycle is one non-terminal, and its row is a sum of products of finished
  rows. The clauses of one head that share their cost and their first hole are multiplied once:
  their tails are summed first, since the product distributes over the sum. A determinized program
  can have many binary clauses per head that share their first hole and differ in the second, and
  there the grouping turns a product per clause into a product per head and first hole.
* A component with a cycle is filled by increasing cost value, as the size table is filled by
  increasing size, since a clause of positive cost reads its holes below the value it fills. A clause
  of cost zero reads a hole *at* that value exactly when its other holes can be filled at cost zero,
  and so the members of the component are filled in an order in which every such hole comes first,
  which is the order the refusal above guarantees to exist.

**What it costs.** A row holds one entry per cost value the non-terminal realizes up to the cap, so
the table is as large as the realized values are many. The size table is small because sizes are few;
a cost like a parameter count realizes so many values that only a coarsened algebra keeps the rows
short. That is the caller's modelling decision, and why the table takes the algebra as given. A
component with a cycle is filled one cost value after the other up to the cap, whether or not the
values are realized, so there the time grows with the cap itself.

**The weighted form.** :class:`WeightedCostTable` answers the oracle :func:`~cosy.search.sampling.keyed_stream`
asks for, from the table. A node's branch counts depend on the node only through the cost of its
partial inhabitant and the multiset of its holes, ``B_n(a) = (conv of the holes' rows)(a - g(n))``,
and the weight per inhabitant, the Gumbel keys and the search are those of the size table's form,
unchanged: every realized cost value up to the cap carries the probability ``pi`` gives it,
normalized over those values, and the terms of one value are equally likely. Under unambiguity within
the cap every inhabitant of cost at most ``C`` is streamed exactly once, in decreasing key order.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic

from cosy.core.solution_space import NT, ConstantArgument, G, Goal, NonTerminalArgument, T
from cosy.search.counting import _admitted, decomposable_or_raise
from cosy.search.partial import Hole, holes, partial_inhabitant
from cosy.search.rules import deepest_first_subgoal
from cosy.search.sampling import _built, _Node, _spread, keyed_stream, log_sum_exp

if TYPE_CHECKING:
    import random
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from cosy.core.solution_space import RHSRule, SolutionSpace
    from cosy.core.tree import Path, Tree
    from cosy.search.costs import AdditiveCostAlgebra
    from cosy.search.queries import ResolutionQuery

__all__ = ["CostTable", "WeightedCostTable", "cost_table", "weighted_cost_table"]


def _whole_cost(value: Any, what: str) -> int:
    """Return a cost as a whole number, or refuse it.

    Args:
        value (Any): The cost, an element of the algebra's domain.
        what (str): What produced it, for the message.

    Returns:
        int: The cost.

    Raises:
        ValueError: If the cost is not a nonnegative whole number. Rounding it would change the
            algebra, and which coarser algebra to use is the caller's decision.
    """
    if isinstance(value, bool):
        whole = None
    elif isinstance(value, int):
        whole = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        whole = int(value)
    else:
        whole = None
    if whole is None or whole < 0:
        msg = (
            f"the cost table is indexed by cost value and takes whole-number costs, but {what} is "
            f"{value!r}; an algebra with other costs has to be coarsened to whole numbers first, "
            f"which is a different algebra and the caller's choice"
        )
        raise ValueError(msg)
    return whole


def _whole_symbol_cost(algebra: AdditiveCostAlgebra[Any], symbol: Any) -> int:
    """Return the cost the algebra charges one symbol, as a whole number, or refuse it.

    Checked per symbol and not per clause: a terminal at 0.1 and a constant at 0.9 add up to a whole
    number, but the fold of a term adds the same floats in another order and need not.

    Args:
        algebra (AdditiveCostAlgebra[Any]): The algebra.
        symbol (Any): A ``Tree`` root: a combinator, or the value of a literal.

    Returns:
        int: Its cost.
    """
    return _whole_cost(algebra.cost_of_symbol(symbol), f"the cost of the symbol {_named(symbol)}")


def _whole_rule_cost(rule: RHSRule[Any, Any, Any], algebra: AdditiveCostAlgebra[Any]) -> int:
    """Return :func:`~cosy.search.counting.rule_cost` in whole numbers, summed exactly.

    Args:
        rule (RHSRule): The clause.
        algebra (AdditiveCostAlgebra[Any]): The algebra.

    Returns:
        int: The cost of the terminal plus the cost of each constant argument.
    """
    return _whole_symbol_cost(algebra, rule.terminal) + sum(
        _whole_symbol_cost(algebra, argument.value)
        for argument in rule.arguments
        if isinstance(argument, ConstantArgument)
    )


def _whole_term_cost(term: Tree[Any], algebra: AdditiveCostAlgebra[Any]) -> int:
    """Return the cost of the symbols a (partial) term carries, in whole numbers; holes cost nothing.

    Args:
        term (Tree[Any]): The term; its leaves may be holes.
        algebra (AdditiveCostAlgebra[Any]): The algebra.

    Returns:
        int: The sum of its symbols' costs.
    """
    total = 0
    pending = [term]
    while pending:
        node = pending.pop()
        if not isinstance(node.root, Hole):
            total += _whole_symbol_cost(algebra, node.root)
        pending.extend(node.children)
    return total


def _named(symbol: Any) -> str:
    """Render a symbol the way a message needs it: a combinator by its name, a literal by its value.

    Args:
        symbol (Any): The symbol.

    Returns:
        str: Its rendering.
    """
    return getattr(symbol, "__name__", None) or repr(symbol)


def _whole_cap(cost_cap: Any) -> int:
    """Return the cost cap as a whole number, or refuse it.

    Args:
        cost_cap (Any): The cap a caller passed, often a fold of the algebra and so a float.

    Returns:
        int: The cap.

    Raises:
        ValueError: If the cap is not a nonnegative whole number. The table is filled one cost value
            at a time, and a cap between two of them would be read as the lower one without a word.
    """
    if isinstance(cost_cap, bool) or not isinstance(cost_cap, (int, float)):
        whole = None
    elif isinstance(cost_cap, float):
        whole = int(cost_cap) if math.isfinite(cost_cap) and cost_cap.is_integer() else None
    else:
        whole = cost_cap
    if whole is None or whole < 0:
        msg = f"the cost cap is a cost value and must be a nonnegative whole number, not {cost_cap!r}"
        raise ValueError(msg)
    return whole


def _add_shifted(target: dict[int, int], row: Mapping[int, int], shift: int, cap: int, times: int = 1) -> None:
    """Add a row, shifted by a cost and repeated, into a row being built, below the cap.

    Args:
        target (dict[int, int]): The row being built.
        row (Mapping[int, int]): The row to add.
        shift (int): The cost to add to each of its values.
        cap (int): The largest cost value kept.
        times (int): How often to add it. (Default value = 1)
    """
    for cost, count in row.items():
        total = cost + shift
        if total <= cap:
            target[total] = target.get(total, 0) + count * times


def _dict_product(left: Mapping[int, int], right: Mapping[int, int], cap: int) -> dict[int, int]:
    """Multiply two rows as polynomials in the cost, keeping the values up to the cap, pair by pair.

    Every cost is nonnegative, so a product cut at the cap and then multiplied again gives what the
    uncut product would give below the cap: nothing above the cap can come back down.

    Args:
        left (Mapping[int, int]): One row, nonnegative whole-number costs to positive counts, as
            every row of a table is.
        right (Mapping[int, int]): The other row, of the same kind.
        cap (int): The largest cost value kept.

    Returns:
        dict[int, int]: ``(left * right)(a) = sum over a_1 + a_2 = a of left(a_1) right(a_2)`` for
            ``a <= cap``, with its zero entries left out.
    """
    if not left or not right:
        return {}
    if len(left) > len(right):
        left, right = right, left
    ordered = sorted(right.items())
    out: dict[int, int] = {}
    for first, first_count in left.items():
        room = cap - first
        if room < 0:
            continue
        for second, second_count in ordered:
            if second > room:
                break
            total = first + second
            out[total] = out.get(total, 0) + first_count * second_count
    return out


def _packed_product(left: Mapping[int, int], right: Mapping[int, int], cap: int) -> dict[int, int]:
    """Multiply two rows as polynomials in the cost by one multiplication of two integers.

    Kronecker substitution: each row becomes one integer whose base-``2**(8 w)`` digits are its
    counts, slot ``a`` holding the count at cost ``a``, and the product of the two integers holds the
    product of the rows in the same slots, provided a slot is wide enough that no coefficient of the
    product spills into the next. It is: a coefficient is a sum of at most ``min(len)`` products of
    two counts, so ``w`` bytes with ``8 w`` at least the sum of the two largest counts' bit lengths and
    of that length's bit length hold it. The multiplication runs in the interpreter's C code, where
    the pair-by-pair product runs one Python step per pair, so on dense wide rows it is many times
    faster; on sparse rows spread far apart it multiplies mostly zeros and loses. :func:`_product`
    decides between the two.

    Args:
        left (Mapping[int, int]): One row, nonnegative whole-number costs to positive counts. A
            negative cost has no slot, and a zero count is dropped where the pair-by-pair product
            keeps it; no row of a table holds either.
        right (Mapping[int, int]): The other row, of the same kind.
        cap (int): The largest cost value kept.

    Returns:
        dict[int, int]: The same coefficients as :func:`_dict_product`, exactly.
    """
    left_values = [value for value in left if value <= cap]
    right_values = [value for value in right if value <= cap]
    if not left_values or not right_values:
        return {}
    left_span, right_span = max(left_values), max(right_values)
    largest_left = max(left[value] for value in left_values)
    largest_right = max(right[value] for value in right_values)
    bits = (
        largest_left.bit_length() + largest_right.bit_length() + min(len(left_values), len(right_values)).bit_length()
    )
    width = (bits + 7) // 8

    def packed(row: Mapping[int, int], values: Sequence[int], span: int) -> int:
        buffer = bytearray((span + 1) * width)
        for value in values:
            buffer[value * width : (value + 1) * width] = row[value].to_bytes(width, "little")
        return int.from_bytes(buffer, "little")

    top = min(cap, left_span + right_span)
    length = (top + 1) * width
    product = (packed(left, left_values, left_span) * packed(right, right_values, right_span)) & (
        (1 << (8 * length)) - 1
    )
    data = product.to_bytes(length, "little")
    out: dict[int, int] = {}
    for value in range(top + 1):
        count = int.from_bytes(data[value * width : (value + 1) * width], "little")
        if count:
            out[value] = count
    return out


# The costs the choice between the two products weighs: rough per-operation costs, measured on one
# machine. A pair of the pair-by-pair product costs a Python step, plus the multiplication of its two
# counts, which grows with the product of their bit lengths; the multiplication of two integers of
# B_long and B_short bytes about the unit times B_long * B_short ** 0.585, the shape of Karatsuba
# multiplication taken chunk by chunk along the longer integer; packing and unpacking about the slot
# cost per slot and per entry. The choice was checked on the 400 pairs the first draw from one
# determinized search space's cost table multiplied, rows of 1 to 4 000 entries and slots of 6 to 22
# bytes: walked about 15 s, packed 1.4 s, as chosen 1.5 s. On another machine the crossover moves, and
# near it either product costs about the same.
_SECONDS_PER_PAIR = 1.25e-7
_SECONDS_PER_PAIR_BIT_PRODUCT = 1e-12
_SECONDS_PER_PACKED_UNIT = 7.5e-10
_SECONDS_PER_PACKED_SLOT = 3e-7
# Below this many pairs the pair-by-pair product is too cheap to be worth weighing at all.
_PACK_FROM_PAIRS = 4096


def _product(left: Mapping[int, int], right: Mapping[int, int], cap: int) -> dict[int, int]:
    """Multiply two rows as polynomials in the cost, by the product an estimate of their costs prefers.

    Both are exact and give the same coefficients on rows of nonnegative whole-number costs and
    positive counts, which is what every row of a table is. The pair-by-pair product costs a Python
    step and a multiplication of two counts per pair of values whose sum stays within the cap; the
    packed one a multiplication of two integers as long as the rows' spans below the cap times the
    slot width, which the interpreter does in C, at a cost that grows much more slowly than the
    number of pairs. Dense wide rows are therefore packed; sparse ones, small ones, far-apart ones and
    ones whose sums mostly pass the cap are walked. The estimate is rough: near the crossover it may
    take the slower product, where the two cost about the same.

    Args:
        left (Mapping[int, int]): One row.
        right (Mapping[int, int]): The other row.
        cap (int): The largest cost value kept.

    Returns:
        dict[int, int]: ``(left * right)(a) = sum over a_1 + a_2 = a of left(a_1) right(a_2)`` for
            ``a <= cap``, with its zero entries left out.
    """
    if not left or not right:
        return {}
    if len(left) * len(right) < _PACK_FROM_PAIRS:
        return _dict_product(left, right, cap)
    left_values = [value for value in left if value <= cap]
    right_values = sorted(value for value in right if value <= cap)
    if not left_values or not right_values:
        return {}
    # The pairs the pair-by-pair product walks: those whose sum stays within the cap.
    pairs = sum(bisect_right(right_values, cap - value) for value in left_values)
    if pairs < _PACK_FROM_PAIRS:
        return _dict_product(left, right, cap)
    left_bits = max(left[value] for value in left_values).bit_length()
    right_bits = max(right[value] for value in right_values).bit_length()
    left_span, right_span = max(left_values), right_values[-1]
    width = (left_bits + right_bits + min(len(left_values), len(right_values)).bit_length() + 7) // 8
    longer, shorter = sorted(((left_span + 1) * width, (right_span + 1) * width), reverse=True)
    walked_seconds = pairs * (_SECONDS_PER_PAIR + _SECONDS_PER_PAIR_BIT_PRODUCT * left_bits * right_bits)
    packed_seconds = _SECONDS_PER_PACKED_UNIT * longer * shorter**0.585 + _SECONDS_PER_PACKED_SLOT * (
        left_span + right_span + len(left_values) + len(right_values)
    )
    if packed_seconds < walked_seconds:
        return _packed_product(left, right, cap)
    return _dict_product(left, right, cap)


@dataclass(frozen=True)
class CostTable(Generic[NT]):
    """``N_A(a)``: the branches rooted at a non-terminal whose term costs exactly ``a``, up to the cap.

    Attributes:
        space (SolutionSpace[NT, Any, Any]): The program the table was filled for. Its rows are
            indexed by that program's non-terminals, and read against another program they count
            something else.
        algebra (AdditiveCostAlgebra[Any]): The algebra the table was filled under. A search reads
            the table together with the algebra's per-clause costs, so the two must be the same.
        cost_cap (int): The largest cost value the table counts.
        counts (Mapping[NT, Mapping[int, int]]): Per non-terminal, its realized cost values up to the
            cap with their counts, ascending. A non-terminal without a term below the cap has an
            empty row, and one the program never mentions has none.
    """

    space: SolutionSpace[NT, Any, Any]
    algebra: AdditiveCostAlgebra[Any]
    cost_cap: int
    counts: Mapping[NT, Mapping[int, int]]

    def of(self, nonterminal: NT, cost: int) -> int:
        """Return ``N_A(a)``.

        Args:
            nonterminal (NT): The non-terminal ``A``.
            cost (int): The cost value ``a``.

        Returns:
            int: The number of branches, 0 outside the table.
        """
        row = self.counts.get(nonterminal)
        if row is None or not 0 <= cost <= self.cost_cap:
            return 0
        return row.get(cost, 0)

    def split_row(self, nonterminals: Sequence[NT]) -> Mapping[int, int]:
        """Return the number of ways to fill a tuple of holes, per total cost up to the cap.

        The holes are filled independently, so the row is the product of theirs. Rows are kept, so
        a tuple of holes is multiplied out once per table however often a search asks for it.

        Args:
            nonterminals (Sequence[NT]): The non-terminals of the holes.

        Returns:
            Mapping[int, int]: The realized total costs with their counts. ``{0: 1}`` on an empty
                tuple, the empty product: one way to fill no holes, at no cost.
        """
        holes_types = tuple(nonterminals)
        cache = self.__dict__.get("_split_rows")
        if cache is None:
            cache = {}
            object.__setattr__(self, "_split_rows", cache)
        cached = cache.get(holes_types)
        if cached is not None:
            return cached
        if not holes_types:
            return MappingProxyType({0: 1})
        # Built from the last hole backwards, one suffix at a time, each kept: a loop and not a
        # recursion, so the length of a tuple costs time and never the stack.
        start = len(holes_types) - 1
        while start > 0 and holes_types[start - 1 :] in cache:
            start -= 1
        suffix = holes_types[start:]
        row: Mapping[int, int] = cache.get(suffix) or self.counts.get(suffix[0]) or MappingProxyType({})
        if len(suffix) == 1:
            cache[suffix] = row
        for position in range(start - 1, -1, -1):
            suffix = holes_types[position:]
            row = MappingProxyType(_product(self.counts.get(suffix[0]) or {}, row, self.cost_cap))
            cache[suffix] = row
        return row

    def split_counts(self, nonterminals: Sequence[NT], total: int) -> int:
        """Return the number of ways to fill a tuple of holes at a total cost.

        Args:
            nonterminals (Sequence[NT]): The non-terminals of the holes.
            total (int): The cost to distribute over them.

        Returns:
            int: ``sum over a_1 + .. + a_k = total of prod_i N_{A_i}(a_i)``, 0 outside the cap.
        """
        if not 0 <= total <= self.cost_cap:
            return 0
        return self.split_row(nonterminals).get(total, 0)


# A clause as the fill reads it: its cost, the non-terminals of its holes, and the clause itself,
# which only an error message ever looks at.
_Clause = tuple[int, tuple[Any, ...], Any]


def _components(nonterminals: Sequence[NT], successors: Mapping[NT, Sequence[NT]]) -> list[list[NT]]:
    """Return the strongly connected components, every component after the ones it reaches.

    Tarjan's algorithm, iterative, since a program's dependency chains run deeper than Python's
    recursion limit. It emits a component once everything it reaches has been emitted, which is the
    children-first order the fill needs.

    Args:
        nonterminals (Sequence[NT]): The nodes, in a fixed order, so that the result does not
            depend on how a set happens to iterate.
        successors (Mapping[NT, Sequence[NT]]): The holes each non-terminal's clauses open.

    Returns:
        list[list[NT]]: The components, children first.
    """
    index: dict[NT, int] = {}
    low: dict[NT, int] = {}
    on_stack: set[NT] = set()
    stack: list[NT] = []
    components: list[list[NT]] = []
    for root in nonterminals:
        if root in index:
            continue
        index[root] = low[root] = len(index)
        stack.append(root)
        on_stack.add(root)
        work = [(root, iter(successors[root]))]
        while work:
            node, pending = work[-1]
            descended = False
            for successor in pending:
                if successor not in index:
                    index[successor] = low[successor] = len(index)
                    stack.append(successor)
                    on_stack.add(successor)
                    work.append((successor, iter(successors[successor])))
                    descended = True
                    break
                if successor in on_stack:
                    low[node] = min(low[node], index[successor])
            if descended:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component: list[NT] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(component)
    return components


def _zero_realizable(clauses: Mapping[NT, Sequence[_Clause]]) -> set[NT]:
    """Return the non-terminals that have a term of cost zero.

    The least fixed point of "a clause of cost zero all of whose holes have one", computed by counting
    down, per clause of cost zero, the distinct holes not yet known to have one.

    Args:
        clauses (Mapping[NT, Sequence[_Clause]]): Each non-terminal's clauses.

    Returns:
        set[NT]: The non-terminals with ``N_A(0) > 0``.
    """
    heads: list[NT] = []
    waiting: list[int] = []
    users: dict[NT, list[int]] = {}
    ready: list[NT] = []
    for nonterminal, head_clauses in clauses.items():
        for cost, hole_types, _rule in head_clauses:
            if cost != 0:
                continue
            distinct = set(hole_types)
            users_index = len(heads)
            heads.append(nonterminal)
            waiting.append(len(distinct))
            if not distinct:
                ready.append(nonterminal)
            for hole in distinct:
                users.setdefault(hole, []).append(users_index)
    realizable: set[NT] = set()
    while ready:
        nonterminal = ready.pop()
        if nonterminal in realizable:
            continue
        realizable.add(nonterminal)
        for index in users.get(nonterminal, ()):
            waiting[index] -= 1
            if waiting[index] == 0:
                ready.append(heads[index])
    return realizable


def _free_edges(hole_types: tuple[NT, ...], zero: set[NT]) -> list[int]:
    """Return the positions a clause of cost zero reads at the value it fills.

    A hole is read at that value exactly when every other hole of the clause can be filled at cost
    zero, which is when the split can give it the whole value.

    Args:
        hole_types (tuple[NT, ...]): The non-terminals of the clause's holes.
        zero (set[NT]): The non-terminals that have a term of cost zero.

    Returns:
        list[int]: The positions of the holes so read.
    """
    dear = [position for position, hole in enumerate(hole_types) if hole not in zero]
    if not dear:
        return list(range(len(hole_types)))
    if len(dear) == 1:
        return dear
    return []


def _zero_cost_order(members: Sequence[NT], clauses: Mapping[NT, Sequence[_Clause]], zero: set[NT]) -> list[NT]:
    """Order the members of a cyclic component so that a hole read at the value being filled comes first.

    Args:
        members (Sequence[NT]): The component, in a fixed order.
        clauses (Mapping[NT, Sequence[_Clause]]): Each non-terminal's clauses.
        zero (set[NT]): The non-terminals that have a term of cost zero.

    Returns:
        list[NT]: The members, each after every member one of its clauses of cost zero reads at the
            value it fills.

    Raises:
        ValueError: If such reads close a loop, which pumps without paying. The message names the
            loop, clause by clause, since which clauses those are is what a caller needs in order to
            price them.
    """
    inside = set(members)
    edges: dict[NT, list[tuple[NT, Any]]] = {
        member: [
            (hole_types[position], rule)
            for cost, hole_types, rule in clauses[member]
            if cost == 0
            for position in _free_edges(hole_types, zero)
            if hole_types[position] in inside
        ]
        for member in members
    }
    unvisited, open_, done = 0, 1, 2
    state = dict.fromkeys(members, unvisited)
    reached_by: dict[NT, tuple[NT, Any]] = {}
    order: list[NT] = []
    for root in members:
        if state[root] != unvisited:
            continue
        state[root] = open_
        work = [(root, iter(edges[root]))]
        while work:
            node, pending = work[-1]
            descended = False
            for hole, rule in pending:
                if state[hole] == unvisited:
                    state[hole] = open_
                    reached_by[hole] = (node, rule)
                    work.append((hole, iter(edges[hole])))
                    descended = True
                    break
                if state[hole] == open_:
                    cycle = [f"{node} <- {_named(rule.terminal)}(.. {hole} ..)"]
                    current = node
                    while current != hole:
                        parent, parent_rule = reached_by[current]
                        cycle.append(f"{parent} <- {_named(parent_rule.terminal)}(.. {current} ..)")
                        current = parent
                    listed = "; ".join(reversed(cycle))
                    msg = (
                        f"clauses of cost zero close a cycle whose other holes can be filled at cost zero too, so "
                        f"it pumps without paying and a cost value may be realized by infinitely many terms: "
                        f"{listed}. Give one of these clauses a positive cost, or count by size, which a size "
                        f"bound keeps finite"
                    )
                    raise ValueError(msg)
            if descended:
                continue
            work.pop()
            state[node] = done
            order.append(node)
    return order


def _acyclic_row(
    head_clauses: Sequence[_Clause],
    rows: Mapping[NT, Mapping[int, int]],
    tuple_row: Callable[[tuple[NT, ...]], Mapping[int, int]],
    cap: int,
) -> dict[int, int]:
    """Return the row of a non-terminal whose holes all have finished rows.

    The clauses that share their cost and their first hole are multiplied once: their tails are
    summed first, and the product distributes over the sum.

    Args:
        head_clauses (Sequence[_Clause]): The non-terminal's clauses.
        rows (Mapping[NT, Mapping[int, int]]): The finished rows.
        tuple_row (Callable): The product of the rows of a tuple of holes.
        cap (int): The largest cost value kept.

    Returns:
        dict[int, int]: The row.
    """
    row: dict[int, int] = {}
    grouped: dict[tuple[int, NT], dict[tuple[NT, ...], int]] = {}
    for cost, hole_types, _rule in head_clauses:
        if not hole_types:
            row[cost] = row.get(cost, 0) + 1
        elif len(hole_types) == 1:
            _add_shifted(row, rows[hole_types[0]], cost, cap)
        else:
            tails = grouped.setdefault((cost, hole_types[0]), {})
            tails[hole_types[1:]] = tails.get(hole_types[1:], 0) + 1
    for (cost, first), tails in grouped.items():
        room = cap - cost
        tail_sum: dict[int, int] = {}
        for tail, times in tails.items():
            _add_shifted(tail_sum, tuple_row(tail), 0, room, times)
        _add_shifted(row, _product(rows[first], tail_sum, room), cost, cap)
    return row


def _cyclic_rows(
    members: Sequence[NT],
    clauses: Mapping[NT, Sequence[_Clause]],
    rows: dict[NT, dict[int, int]],
    cap: int,
    zero: set[NT],
) -> None:
    """Fill the rows of a cyclic component by increasing cost value.

    A clause of positive cost reads its holes below the value it fills, and a clause of cost zero
    reads a hole at it only where the others can be filled at cost zero, from members the order puts
    first. A split of a tuple of holes at a total is kept once computed: every value it reads is final
    by then, since every value below the current one is, and at the current one it reads a member's
    value only where that value is multiplied by a nonzero count, which is where the order put the
    member first. Anywhere else the product is zero whatever the value read.

    Args:
        members (Sequence[NT]): The component.
        clauses (Mapping[NT, Sequence[_Clause]]): Each non-terminal's clauses.
        rows (dict[NT, dict[int, int]]): The rows, finished outside the component; filled here for
            its members.
        cap (int): The largest cost value kept.
        zero (set[NT]): The non-terminals that have a term of cost zero.
    """
    order = _zero_cost_order(members, clauses, zero)
    inside = set(members)
    realized: dict[NT, list[int]] = {}
    for nonterminal in members:
        rows[nonterminal] = {}
        realized[nonterminal] = []
    splits: dict[tuple[tuple[NT, ...], int], int] = {}

    def values_of(nonterminal: NT) -> Sequence[int]:
        """Return the cost values a non-terminal realizes so far, ascending.

        Args:
            nonterminal (NT): The non-terminal.

        Returns:
            Sequence[int]: Its realized values.
        """
        if nonterminal in inside:
            return realized[nonterminal]
        known = realized.get(nonterminal)
        if known is None:
            known = sorted(rows.get(nonterminal, {}))
            realized[nonterminal] = known
        return known

    def split(hole_types: tuple[NT, ...], total: int) -> int:
        """Return the number of ways to fill the holes at exactly the total cost.

        Args:
            hole_types (tuple[NT, ...]): The non-terminals of the holes.
            total (int): The cost to distribute.

        Returns:
            int: The count.
        """
        if not hole_types:
            return 1 if total == 0 else 0
        first = hole_types[0]
        if len(hole_types) == 1:
            return rows.get(first, {}).get(total, 0)
        key = (hole_types, total)
        known = splits.get(key)
        if known is not None:
            return known
        rest = hole_types[1:]
        first_row = rows.get(first, {})
        value = 0
        for cost in values_of(first):
            if cost > total:
                break
            value += first_row[cost] * split(rest, total - cost)
        splits[key] = value
        return value

    for level in range(cap + 1):
        for nonterminal in order:
            value = 0
            for cost, hole_types, _rule in clauses[nonterminal]:
                if cost <= level:
                    value += split(hole_types, level - cost)
            if value:
                rows[nonterminal][level] = value
                realized[nonterminal].append(level)


def cost_table(
    space: SolutionSpace[NT, T, G], algebra: AdditiveCostAlgebra[Any], cost_cap: int, *, check: bool = True
) -> CostTable[NT]:
    """Fill ``N_A(a)`` for every non-terminal of a program and every cost value up to a cap.

    Args:
        space (SolutionSpace[NT, T, G]): The synthesized program.
        algebra (AdditiveCostAlgebra[Any]): The additive cost algebra; its per-clause costs
            (:func:`~cosy.search.counting.rule_cost`) must be whole numbers.
        cost_cap (int): The largest cost value counted.
        check (bool): Whether to decide the decomposition hypothesis first. Leaving it on is the
            supported use. (Default value = True)

    Returns:
        CostTable[NT]: The filled table.

    Raises:
        ValueError: If the cap is not a nonnegative whole number; under ``check``, if a predicate of
            the program reads a hole; if a clause's cost is not a whole number; or if clauses of cost
            zero close a loop that pumps without paying, in which case the message names it.
    """
    cost_cap = _whole_cap(cost_cap)
    if check:
        decomposable_or_raise(space)

    nonterminals = list(space.nonterminals())
    known = set(nonterminals)
    clauses: dict[NT, list[_Clause]] = {}
    for nonterminal in nonterminals:
        kept: list[_Clause] = []
        for rule in space.get(nonterminal) or ():
            if not _admitted(rule):
                continue
            cost = _whole_rule_cost(rule, algebra)
            hole_types = tuple(
                argument.origin for argument in rule.arguments if isinstance(argument, NonTerminalArgument)
            )
            # Above the cap a clause contributes nothing, since no cost comes back down; and a hole
            # whose non-terminal the program never mentions has no terms at all.
            if cost > cost_cap or any(hole not in known for hole in hole_types):
                continue
            kept.append((cost, hole_types, rule))
        clauses[nonterminal] = kept

    successors = {
        nonterminal: list(
            dict.fromkeys(hole for _cost, hole_types, _rule in clauses[nonterminal] for hole in hole_types)
        )
        for nonterminal in nonterminals
    }
    rows: dict[NT, dict[int, int]] = {}
    tuple_rows: dict[tuple[NT, ...], Mapping[int, int]] = {}

    def tuple_row(hole_types: tuple[NT, ...]) -> Mapping[int, int]:
        """Return the product of the finished rows of a tuple of holes, kept once computed.

        Args:
            hole_types (tuple[NT, ...]): The non-terminals of the holes, at least one.

        Returns:
            Mapping[int, int]: The product, up to the cap.
        """
        if len(hole_types) == 1:
            return rows[hole_types[0]]
        known_row = tuple_rows.get(hole_types)
        if known_row is None:
            known_row = _product(rows[hole_types[0]], tuple_row(hole_types[1:]), cost_cap)
            tuple_rows[hole_types] = known_row
        return known_row

    zero: set[NT] | None = None
    for component in _components(nonterminals, successors):
        head = component[0]
        if len(component) == 1 and head not in successors[head]:
            rows[head] = _acyclic_row(clauses[head], rows, tuple_row, cost_cap)
        else:
            if zero is None:
                zero = _zero_realizable(clauses)
            _cyclic_rows(component, clauses, rows, cost_cap, zero)

    return CostTable(
        space=space,
        algebra=algebra,
        cost_cap=cost_cap,
        counts=MappingProxyType({nt: MappingProxyType(dict(sorted(row.items()))) for nt, row in rows.items()}),
    )


def _initial_cost_nodes(
    query: ResolutionQuery[NT, T, G], algebra: AdditiveCostAlgebra[Any]
) -> list[tuple[Goal[NT, T, G], int]]:
    """Return the children of the query's root with the cost of their partial inhabitants.

    :func:`~cosy.search.counting.initial_nodes` with the cost in the place of the size. A partial-term
    query starts from goals that already carry the prescribed symbols, and those are charged.

    Args:
        query (ResolutionQuery[NT, T, G]): The query.
        algebra (AdditiveCostAlgebra[Any]): The algebra.

    Returns:
        list[tuple[Goal[NT, T, G], int]]: The initial goals with their costs.
    """
    space = query.solution_space
    tree, pos = query.tree, query.pos
    if tree is not None and pos is not None:
        goals = space.goal_from_tree(query.start, tree, pos)
        return [(goal, _whole_term_cost(partial_inhabitant(goal), algebra)) for goal in goals]
    initial: list[tuple[Goal[NT, T, G], int]] = []
    for rule in space.get(query.start) or ():
        goal = Goal.from_rhs_rule(rule)
        if goal is not None:
            initial.append((goal, _whole_rule_cost(rule, algebra)))
    return initial


@dataclass(frozen=True)
class WeightedCostTable(Generic[NT, T, G]):
    """Random search weighted from a cost table: a prescribed distribution on an additive cost.

    The table form of random search (:class:`~cosy.search.sampling.WeightedTable`) with the cost of
    an additive cost algebra in the place of the term size. A node's branch counts depend on it only
    through the cost of its partial inhabitant and the multiset of its holes, so they are a lookup,
    and a node is expanded only when the frontier reaches it.

    Attributes:
        query (ResolutionQuery[NT, T, G]): The query being sampled from.
        algebra (AdditiveCostAlgebra[Any]): The algebra the table was filled under.
        cost_cap (int): The largest cost a streamed term may have.
        table (CostTable[NT]): ``N_A(a)`` over the program.
        root_counts (Mapping[int, int]): ``N_r(a)``, summed over the query's initial nodes.
        unit_weights (Mapping[int, float]): ``pi(a) / N_r(a)`` per realized cost value.
        log_unit_weights (Mapping[int, float]): The same in log space, and what the search uses.
        subgoal_selection (Callable | None): The computation rule; None selects the engine's
            deepest-first rule. It must select an open hole, as the engine's rule does: a node is
            weighed by its open holes, and an expanded position that ``Goal.subgoals`` still lists
            is not one.
    """

    query: ResolutionQuery[NT, T, G]
    algebra: AdditiveCostAlgebra[Any]
    cost_cap: int
    table: CostTable[NT]
    root_counts: Mapping[int, int]
    unit_weights: Mapping[int, float]
    log_unit_weights: Mapping[int, float]
    subgoal_selection: Callable[[Goal[NT, T, G]], tuple[Path, NonTerminalArgument[NT]]] | None = None

    @property
    def total(self) -> int:
        """Return the number of success branches of cost at most the cap.

        Returns:
            int: ``sum_a N_r(a)``.
        """
        return sum(self.root_counts.values())

    def _cache(self, name: str) -> dict[Any, Any]:
        """Return one of the derived caches over this frozen construction, created on first use.

        Args:
            name (str): The cache's attribute name.

        Returns:
            dict[Any, Any]: The cache.
        """
        cache = self.__dict__.get(name)
        if cache is None:
            cache = {}
            object.__setattr__(self, name, cache)
        return cache

    def holes_of(self, goal: Goal[NT, T, G]) -> tuple[NT, ...]:
        """Return the non-terminals of a goal's open holes, in a canonical order.

        Args:
            goal (Goal[NT, T, G]): The search node.

        Returns:
            tuple[NT, ...]: The hole types, sorted so that equal multisets give equal tuples.
        """
        rank = self._cache("_rank_cache")
        if not rank:
            rank.update({nonterminal: index for index, nonterminal in enumerate(self.table.counts)})
        return tuple(sorted(holes(goal).values(), key=lambda nt: (rank.get(nt, -1), id(nt))))

    def rule_cost_of(self, rule: RHSRule[NT, T, G]) -> int:
        """Return the whole-number cost a clause adds, computed once per clause.

        Args:
            rule (RHSRule[NT, T, G]): The clause.

        Returns:
            int: Its cost.
        """
        costs = self._cache("_rule_costs")
        cost = costs.get(id(rule))
        if cost is None:
            cost = _whole_rule_cost(rule, self.algebra)
            costs[id(rule)] = cost
        return cost

    def branch_counts_of(self, goal: Goal[NT, T, G] | None, cost: int) -> dict[int, int]:
        """Return ``B_n`` of a node, from the table alone.

        Args:
            goal (Goal[NT, T, G] | None): The search node, or None for the query's root.
            cost (int): The cost of its partial inhabitant.

        Returns:
            dict[int, int]: The branch counts, one entry per realized cost value up to the cap.
        """
        if goal is None:
            return dict(self.root_counts)
        return self._branch_counts_of_holes(() if goal.success else self.holes_of(goal), cost)

    def _branch_counts_of_holes(self, hole_types: tuple[NT, ...], cost: int) -> dict[int, int]:
        """Return ``B_n`` of a node given by the multiset of its holes and its cost so far.

        All the table needs of a node, which is why a child can be weighed before it is built. No
        holes is a success node: one branch, of its own cost.

        Args:
            hole_types (tuple[NT, ...]): The node's holes, in the canonical order of :meth:`holes_of`.
            cost (int): The cost of its partial inhabitant.

        Returns:
            dict[int, int]: The branch counts, one entry per realized cost value up to the cap.
        """
        if not hole_types:
            return {cost: 1} if cost <= self.cost_cap else {}
        row = self.table.split_row(hole_types)
        return {cost + value: count for value, count in row.items() if cost + value <= self.cost_cap}

    def log_weight_of(self, goal: Goal[NT, T, G] | None, cost: int) -> float:
        """Return the location of the Gumbel distribution a node's key follows.

        Memoised on the multiset of holes and the cost so far, which are all a node's weight
        depends on.

        Args:
            goal (Goal[NT, T, G] | None): The search node, or None for the query's root.
            cost (int): The cost of its partial inhabitant.

        Returns:
            float: ``log sum_a B_n(a) pi(a) / N_r(a)``, and ``-inf`` on a node with no completion
                within the cap.
        """
        if goal is None:
            return self._weight(None, -1, dict(self.root_counts))
        return self._log_weight_of_holes(() if goal.success else self.holes_of(goal), cost)

    def _log_weight_of_holes(self, hole_types: tuple[NT, ...], cost: int) -> float:
        """Return a node's ``log w`` from the multiset of its holes and its cost so far.

        Args:
            hole_types (tuple[NT, ...]): The node's holes, in canonical order; empty on a success node.
            cost (int): The cost of its partial inhabitant.

        Returns:
            float: ``log sum_a B_n(a) pi(a) / N_r(a)``, ``-inf`` without a completion within the cap.
        """
        known = self._cache("_weight_cache").get((hole_types, cost))
        if known is not None:
            return known
        return self._weight(hole_types, cost, self._branch_counts_of_holes(hole_types, cost))

    def _weight(self, hole_types: tuple[NT, ...] | None, cost: int, counts: Mapping[int, int]) -> float:
        """Return the log-sum-exp of branch counts under the unit weights, and keep it.

        Args:
            hole_types (tuple[NT, ...] | None): The key's holes; None for the root.
            cost (int): The key's cost; -1 for the root.
            counts (Mapping[int, int]): The node's branch counts.

        Returns:
            float: ``log sum_a B_n(a) pi(a) / N_r(a)``.
        """
        weights = self._cache("_weight_cache")
        key = (hole_types, cost)
        known = weights.get(key)
        if known is not None:
            return known
        value = log_sum_exp(
            [
                math.log(count) + self.log_unit_weights[total]
                for total, count in counts.items()
                if total in self.log_unit_weights
            ]
        )
        weights[key] = value
        return value

    def stream(self, rng: random.Random) -> Iterator[Tree[T]]:
        """Draw one stream from this weighted table.

        Args:
            rng (random.Random): The source of randomness.

        Yields:
            Tree[T]: The inhabitants of cost at most the cap, in decreasing key order.
        """
        for _, inhabitant in self.keyed_stream(rng):
            yield inhabitant

    def keyed_stream(self, rng: random.Random) -> Iterator[tuple[float, Tree[T]]]:
        """Draw one stream, keeping the key each inhabitant was streamed under.

        Args:
            rng (random.Random): The source of randomness.

        Yields:
            tuple[float, Tree[T]]: The key and the inhabitant, in decreasing key order.
        """
        if not self.root_counts:
            return
        select = deepest_first_subgoal if self.subgoal_selection is None else self.subgoal_selection

        rank = self._cache("_rank_cache")
        if not rank:
            rank.update({nonterminal: index for index, nonterminal in enumerate(self.table.counts)})
        space = self.query.solution_space

        def canonical(hole_types: list[NT]) -> tuple[NT, ...]:
            return tuple(sorted(hole_types, key=lambda nt: (rank.get(nt, -1), id(nt))))

        def expand(node: _Node) -> tuple[Tree[T] | None, Sequence[tuple[_Node, float]]]:
            """Expand one node on demand, and weigh its children from the table without building them.

            A child is pushed as its parent, the position and the clause, and weighed from the
            multiset of its holes and its cost so far, which the parent and the clause give without
            applying the clause; its goal is built when it is popped
            (:class:`~cosy.search.sampling.WeightedTable` does the same).

            Args:
                node (_Node): The goal (None at the root or while unbuilt), the cost of its partial
                    inhabitant, and, while unbuilt, the parent, position and clause that build it.

            Returns:
                tuple: The inhabitant (None on an inner node) and the retained children with their
                    ``log w``, in clause order.

            Raises:
                ValueError: If the engine refuses a clause the frontier had weighed, or if the
                    computation rule selects a position that is not an open hole.
            """
            goal, cost, pending = node
            if pending is not None:
                goal = _built(pending)
            if goal is not None and goal.success:
                return goal.grounded[()][1], ()
            kept: list[tuple[_Node, float]] = []
            if goal is None:
                for child, child_cost in _initial_cost_nodes(self.query, self.algebra):
                    if child_cost > self.cost_cap:
                        continue
                    log_weight = self.log_weight_of(child, child_cost)
                    if log_weight > -math.inf:
                        kept.append(((child, child_cost, None), log_weight))
                return None, kept
            position, argument = select(goal)
            open_holes = holes(goal)
            if position not in open_holes:
                msg = (
                    f"the computation rule selected position {position}, which is not an open hole: a table "
                    "weighs a node by its open holes, so the rule must select one, as deepest_first_subgoal does"
                )
                raise ValueError(msg)
            others = [hole_type for hole_position, hole_type in open_holes.items() if hole_position != position]
            for rule in space.get(argument.origin) or ():
                if not _admitted(rule):
                    continue
                child_cost = cost + self.rule_cost_of(rule)
                if child_cost > self.cost_cap:
                    continue
                opened = [arg.origin for arg in rule.arguments if isinstance(arg, NonTerminalArgument)]
                log_weight = self._log_weight_of_holes(canonical(others + opened), child_cost)
                if log_weight > -math.inf:
                    kept.append(((None, child_cost, (goal, position, rule)), log_weight))
            return None, kept

        yield from keyed_stream((None, 0, None), self.log_weight_of(None, 0), expand, rng)


def weighted_cost_table(
    query: ResolutionQuery[NT, T, G],
    algebra: AdditiveCostAlgebra[Any],
    distribution: Callable[[Any], float],
    cost_cap: int,
    *,
    subgoal_selection: Callable[[Goal[NT, T, G]], tuple[Path, NonTerminalArgument[NT]]] | None = None,
    table: CostTable[NT] | None = None,
) -> WeightedCostTable[NT, T, G]:
    """Build random search under an additive cost and a prescribed distribution on its values.

    Args:
        query (ResolutionQuery[NT, T, G]): The query to sample from, generator or partial-term.
        algebra (AdditiveCostAlgebra[Any]): The additive cost algebra, with whole-number costs.
        distribution (Callable[[Any], float]): ``pi``, evaluated on each cost value the query
            realizes up to the cap. It need not be normalized, but it must be positive on every one
            of them; above the cap it is never read.
        cost_cap (int): The largest cost a streamed term may have, a nonnegative whole number.
        subgoal_selection (Callable | None): The computation rule, which must select an open hole.
            (Default value = None)
        table (CostTable[NT] | None): A table already filled for this program under this algebra,
            to at least this cap. Passing one is how several queries share the cost of filling it;
            a table filled to a higher cap is read up to this one. (Default value = None)

    Returns:
        WeightedCostTable[NT, T, G]: The construction, ready to stream from.

    Raises:
        ValueError: If a predicate of the program reads a hole, if the cap is not a nonnegative whole
            number, if a cost is not a whole number, if clauses of cost zero close a loop that pumps
            without paying, if a table handed in was filled for another program, under another
            algebra or to a lower cap, or if ``distribution`` is not positive on some realized cost
            value up to the cap.
    """
    cost_cap = _whole_cap(cost_cap)
    decomposable_or_raise(query.solution_space)
    filled = cost_table(query.solution_space, algebra, cost_cap, check=False) if table is None else table
    if filled.space is not query.solution_space:
        msg = (
            "the table was filled for another program than the one the query asks; its rows are indexed "
            "by that program's non-terminals and would count something else here"
        )
        raise ValueError(msg)
    if filled.algebra is not algebra:
        msg = (
            "the table was filled under another algebra than the one the search charges clauses by; "
            "its counts would weigh the nodes by one cost and the search would bound them by another"
        )
        raise ValueError(msg)
    if filled.cost_cap < cost_cap:
        msg = (
            f"the table was filled to the cost cap {filled.cost_cap} but the search asks for {cost_cap}; "
            f"the missing values would read as zero and cut the space short"
        )
        raise ValueError(msg)

    probe = WeightedCostTable(
        query=query,
        algebra=algebra,
        cost_cap=cost_cap,
        table=filled,
        root_counts={},
        unit_weights={},
        log_unit_weights={},
        subgoal_selection=subgoal_selection,
    )
    root_counts: dict[int, int] = {}
    for goal, cost in _initial_cost_nodes(query, algebra):
        if cost > cost_cap:
            continue
        for value, count in probe.branch_counts_of(goal, cost).items():
            root_counts[value] = root_counts.get(value, 0) + count

    unit_weights, log_unit_weights = _spread(distribution, root_counts)
    return WeightedCostTable(
        query=query,
        algebra=algebra,
        cost_cap=cost_cap,
        table=filled,
        root_counts=root_counts,
        unit_weights=unit_weights,
        log_unit_weights=log_unit_weights,
        subgoal_selection=subgoal_selection,
    )
