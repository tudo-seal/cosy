"""Recognizable constraints: a predicate stated through the finite abstraction it factors through.

An external predicate is what makes a synthesized program impossible to count from the program
alone. The counting recursion reads a clause as "fill the holes independently and multiply the
ways", and a predicate that reads a hole breaks exactly that: the residual at the hole is a proper
subset of the hole's language, and a table indexed by the non-terminal cannot see the difference.
:func:`cosy.search.counting.decomposable_or_raise` is what says which clauses do it, and coupled
holes are the same failure with two arguments rather than one.

There is a class of predicates for which the failure is repairable, and the class is not "the
predicate reads only one hole". It is this:

    **(REC)** To a predicate ``P`` of arity ``k`` belongs a finite, bottom-up computable
    abstraction ``alpha`` from terms into a set ``Q``, and a relation ``R`` over ``Q^k``, with
    ``P(t_1, ..., t_k)`` holding exactly when ``(alpha(t_1), ..., alpha(t_k))`` is in ``R``.

That is the class of *recognizable tree relations*. (REC) does not forbid coupling. It permits
coupling, as long as the coupling factors through finitely many classes, which is what separates it
from the condition the counting recursion asks for, that no predicate read a hole at all. It
neither contains nor is contained in "reads only one hole": a predicate on a single hole is outside
(REC) where the property it reads has no finite abstraction, and a predicate on two holes is inside
it where the coupling does. Under (REC) the
predicate can be pushed into the non-terminals by a product construction
(:func:`cosy.search.determinize.determinize`), after which the program is predicate-free and every
clause decomposes again. Duchon, Flajolet, Louchard and Schaeffer (2004, pp. 589-590) do the same
thing by hand for one side condition, compiling it into the specification rather than filtering
with it, and reject naive filtering as exponential.

**A repository states ``alpha`` and ``R``, never ``P``.** ``P`` is *derived* here as ``R`` after
``alpha``, which is what makes (REC) hold by construction instead of by assertion. Stating both
separately would allow them to disagree, and a disagreement between them changes the sampled
distribution without leaving any other trace: the engine would keep the terms ``P`` admits while
the counting counted the terms ``R`` admits. A caller who genuinely wants a predicate outside
(REC) writes :meth:`cosy.core.SpecificationBuilder.constraint` and pays the tree form of counting.

``alpha`` is an algebra over the whole synthesis alphabet, terminals and literals alike, whose
carrier is finite. A literal occurs as a nullary symbol, so ``alpha(value, ())`` is its state.
Where the abstraction has nothing to say about literals, returning the value itself is the
identity choice, and it makes a relation read a literal exactly as the predicate would.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from cosy.core.tree import Tree

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["RecognizableConstraint", "StateRelation", "TreeAbstraction", "state_of"]

Q = TypeVar("Q")
"""The carrier of an abstraction: finite, hashable, and compared by equality."""

TreeAbstraction = Any
"""``alpha`` as a Sigma-algebra: ``(symbol, states of the arguments) -> state``.

A terminal receives the states of its arguments in clause order, a literal receives the empty
tuple. The alias is deliberately untyped: the symbol type of a solution space is whatever the
repository uses for combinators and for literal values at once, and pinning it here would state
something the type system could then be relied on for and that is not true.
"""

StateRelation = Any
"""``R`` on states: a substitution by variable name, exactly as a predicate receives it.

The named holes of the clause carry their abstraction's state instead of their term, and the
literals carry ``alpha(value, ())``, which under the identity choice on literals is the value
itself. Anonymous holes are absent, exactly as they are absent from the substitution a predicate
receives.
"""


def state_of(term: Tree[Any], abstraction: TreeAbstraction) -> Any:
    """Fold an abstraction bottom-up over a ground term.

    Iterative rather than recursive: a term of the size the table form exists to reach is deeper
    than the interpreter's recursion limit long before it is large enough to be interesting.

    Args:
        term (Tree[Any]): The ground term to abstract.
        abstraction (TreeAbstraction): ``alpha``.

    Returns:
        Any: ``alpha(term)``.
    """
    order: list[Tree[Any]] = []
    pending = [term]
    while pending:
        node = pending.pop()
        order.append(node)
        pending.extend(node.children)
    # Keyed on the subterms themselves and skipped where the key is already there, so equal
    # subterms are abstracted once however often they occur. ``Tree`` hashes structurally, which
    # is what makes that a saving on the repeated arguments a coupling predicate is usually about.
    # Reversing a pre-order puts every subterm after its own subterms, so the states a node reads
    # are present when it is reached, and a repeated subterm is reached at its last occurrence
    # first.
    states: dict[Tree[Any], Any] = {}
    for node in reversed(order):
        if node not in states:
            states[node] = abstraction(node.root, tuple(states[child] for child in node.children))
    return states[term]


@dataclass(frozen=True)
class RecognizableConstraint(Generic[Q]):
    """A predicate given as ``R`` after ``alpha``, which is the form the determinization consumes.

    The instance is itself the predicate the engine evaluates: calling it abstracts every term of
    the substitution and hands the states to ``R``. A repository that states a constraint this way
    therefore loses nothing, the engine and the checker and the tree form of counting all working
    unchanged, and it gains the option of :func:`cosy.search.determinize.determinize`, which
    removes the predicate altogether by carrying its states in the non-terminals.

    Attributes:
        abstraction (TreeAbstraction): ``alpha``, an algebra over the synthesis alphabet with a
            finite carrier. It is applied to every symbol the program writes, terminals and
            literal values alike.
        relation (StateRelation): ``R``, deciding on the states. It receives the substitution a
            predicate receives, with each term replaced by its state.
    """

    abstraction: TreeAbstraction
    relation: StateRelation

    def __call__(self, substitution: Mapping[str, Any]) -> bool:
        """Decide the predicate on a substitution of ground terms.

        A value is read as a term where it is a :class:`~cosy.core.tree.Tree` and as a literal
        otherwise. The substitution carries no record of which name is which, and the engine binds
        every named hole to a term, so the test is on the value. A repository whose literal values
        are themselves terms is the one case the test cannot place, and it should state such a
        value through :meth:`~cosy.core.SpecificationBuilder.constraint` instead.

        Args:
            substitution (Mapping[str, Any]): The clause's substitution, terms for the named holes
                and values for the literals.

        Returns:
            bool: ``R(alpha(t_1), ..., alpha(t_k))``.
        """
        return bool(
            self.relation(
                {
                    name: state_of(value, self.abstraction) if isinstance(value, Tree) else self.abstraction(value, ())
                    for name, value in substitution.items()
                }
            )
        )
