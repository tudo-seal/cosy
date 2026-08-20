"""Tests that intersection types and solution spaces survive pickling and copying.

`Constructor`, `Literal` and `Var` store themselves in their derived `organized` field. The default
pickling protocol restores such a value in two steps, allocating an empty instance and then handing
it its instance dictionary, which forces the self-referential set to be built while the instance is
still empty. Since a set insertion hashes its element and the hash of a frozen dataclass reads the
constructor fields, loading used to fail with a missing attribute while dumping always succeeded.
Deep copying failed the same way. Every test below therefore performs a full round trip and never
merely dumps a value.
"""

import copy
import pickle
from dataclasses import fields
from inspect import Parameter, isabstract, signature
from typing import TYPE_CHECKING

import pytest

from cosy.core.specification_builder import SpecificationBuilder
from cosy.core.synthesizer import Synthesizer
from cosy.core.types import (
    Arrow,
    Constructor,
    DataGroup,
    Intersection,
    Literal,
    Omega,
    Type,
    Var,
)

if TYPE_CHECKING:
    from cosy.core.solution_space import SolutionSpace
    from cosy.core.tree import Tree

TYPES_MODULE = Type.__module__

SAMPLE_TYPES: list[Type] = [
    Omega(),
    Constructor("a"),
    Constructor("a", Literal(3)),
    Constructor("a", Intersection(Constructor("b"), Constructor("c"))),
    Var("x"),
    Literal(3),
    Literal((True, False)),
    Arrow(Constructor("a"), Constructor("b")),
    Arrow(Var("x"), Arrow(Constructor("a"), Literal(1))),
    Intersection(Constructor("a"), Constructor("b")),
    Intersection(Arrow(Constructor("a"), Constructor("b")), Constructor("c", Var("y"))),
]

SELF_ORGANIZED_TYPES: list[Type] = [Constructor("a"), Literal(3), Var("x")]


def concrete_type_classes() -> set[type[Type]]:
    """Collect every concrete type class that is declared in the types module.

    Subclasses are collected recursively, so classes introduced in the middle of the hierarchy are
    covered as well. Restricting the result to classes declared in `cosy.core.types` keeps the guard
    deterministic: without it, a type subclass defined by any other test would leak into this one as
    soon as the suite runs randomized or in parallel. The price of the restriction is that a
    concrete type class declared anywhere else is not covered here, so a new concrete type belongs
    into `cosy.core.types` or needs a round trip test of its own.

    Returns:
        set[type[Type]]: The non-abstract subclasses of `Type` declared in `cosy.core.types`.
    """
    found: set[type[Type]] = set()
    seen: set[type[Type]] = set()
    pending: list[type[Type]] = list(Type.__subclasses__())
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(current.__subclasses__())
        if current.__module__ == TYPES_MODULE and not isabstract(current):
            found.add(current)
    return found


CONCRETE_TYPE_CLASSES = sorted(concrete_type_classes(), key=lambda type_class: type_class.__name__)


def deeply_nested_type() -> Type:
    """Build a type whose nesting goes through the safe combinators.

    `Type.curry` and `Type.intersect` are the constructors that respect subtyping, so a type built
    with them is the realistic shape a user hands to the synthesizer.

    Returns:
        Type: A nested type mixing arrows, intersections, constructors, variables and literals.
    """
    result = Type.intersect(
        [
            Constructor("result", Var("n")),
            Constructor("tag", Literal("deep")),
            Arrow(Constructor("in"), Constructor("out", Literal(7))),
        ]
    )
    curried = Type.curry(
        [
            Constructor("a"),
            Intersection(Constructor("b"), Var("m")),
            Arrow(Constructor("c"), Constructor("d")),
        ],
        result,
    )
    return Type.intersect([curried, Constructor("wrapper", curried), Var("z")])


def digit(value: int) -> str:
    """Turn a digit into its decimal representation.

    Args:
        value (int): The digit to represent.

    Returns:
        str: The decimal representation of `value`.
    """
    return str(value)


def shout(value: int) -> str:
    """Turn a digit into an emphatic decimal representation.

    Args:
        value (int): The digit to represent.

    Returns:
        str: The decimal representation of `value`, followed by an exclamation mark.
    """
    return f"{value}!"


def wrap(inner: str) -> str:
    """Wrap a representation.

    Args:
        inner (str): The representation to wrap.

    Returns:
        str: The wrapped representation.
    """
    return f"(wrap {inner})"


def merge(left: str, right: str) -> str:
    """Merge two representations.

    Args:
        left (str): The first representation.
        right (str): The second representation.

    Returns:
        str: The merged representation.
    """
    return f"(merge {left} {right})"


@pytest.fixture
def component_specifications():
    """Specify a small repository whose solution space contains four solutions.

    Returns:
        dict: The component specifications of the repository.
    """
    digits = DataGroup("digits", [0, 1])
    return {
        digit: SpecificationBuilder().parameter("value", digits).suffix(Constructor("digit", Var("value"))),
        shout: SpecificationBuilder().parameter("value", digits).suffix(Constructor("digit", Var("value"))),
        wrap: SpecificationBuilder()
        .parameter("value", digits)
        .argument("inner", Constructor("digit", Var("value")))
        .suffix(Constructor("wrapped", Var("value"))),
        merge: SpecificationBuilder()
        .argument("left", Constructor("wrapped", Literal(0)))
        .argument("right", Constructor("wrapped", Literal(1)))
        .suffix(Constructor("merged")),
    }


@pytest.fixture
def query():
    """Specify the query of the small repository.

    Returns:
        Type: The synthesis target.
    """
    return Constructor("merged")


@pytest.mark.parametrize("original", SAMPLE_TYPES, ids=str)
def test_pickle_roundtrip_preserves_type(original: Type) -> None:
    """Check that a type is restored with its derived fields intact.

    The stream must not carry the derived fields at all, since restoring them from the stream is
    exactly the two step reconstruction that used to fail.

    Args:
        original (Type): The type to send through a pickle round trip.
    """
    stream = pickle.dumps(original)
    assert b"organized" not in stream

    restored = pickle.loads(stream)

    assert type(restored) is type(original)
    assert restored == original
    assert hash(restored) == hash(original)
    assert restored.organized == original.organized
    assert restored.free_vars == original.free_vars


@pytest.mark.parametrize("original", SELF_ORGANIZED_TYPES, ids=str)
def test_roundtrip_keeps_a_type_inside_its_own_organized_field(original: Type) -> None:
    """Check that a type which organizes into itself refers to the copy, not to the original.

    Equality of types ignores `organized`, so comparing the two sets cannot tell a restored value
    that refers to itself from one that still refers to the original. Only identity can, and the
    broken variant is not hypothetical: before the fix, a shallow copy shared the set of the
    original and therefore contained the original instead of itself.

    Args:
        original (Type): The type to send through a pickle round trip and a deep copy.
    """
    assert any(part is original for part in original.organized)

    for restored in (pickle.loads(pickle.dumps(original)), copy.deepcopy(original)):
        assert restored.organized is not original.organized
        assert any(part is restored for part in restored.organized)
        assert all(part is not original for part in restored.organized)


def test_pickle_roundtrip_preserves_deeply_nested_type() -> None:
    """Check that a type built via `Type.curry` and `Type.intersect` survives a round trip."""
    original = deeply_nested_type()

    restored = pickle.loads(pickle.dumps(original))

    assert restored == original
    assert hash(restored) == hash(original)
    assert restored.organized == original.organized
    assert restored.free_vars == original.free_vars
    assert str(restored) == str(original)


def test_every_concrete_type_is_covered() -> None:
    """Check that the round trip parametrization covers every concrete type class.

    A type class that is not listed in `SAMPLE_TYPES` would silently go untested, so the set of
    sampled classes has to match the set of classes the types module declares.
    """
    assert {type(sample) for sample in SAMPLE_TYPES} == concrete_type_classes()


@pytest.mark.parametrize(
    "type_class",
    CONCRETE_TYPE_CLASSES,
    ids=[type_class.__name__ for type_class in CONCRETE_TYPE_CLASSES],
)
def test_replayed_arguments_match_the_constructor_signature(type_class: type[Type]) -> None:
    """Check that reconstruction may replay the constructor arguments positionally.

    `Type.__reduce__` replays the constructor with the values of all fields that carry `init=True`,
    in declaration order. That is only correct while those fields are exactly the parameters of the
    constructor and while every one of them accepts a positional value. A keyword-only field would
    end up in the wrong position, and an `InitVar` parameter would be dropped from the round trip
    without a word, because `dataclasses.fields` does not report it.

    Args:
        type_class (type[Type]): The type class to inspect.
    """
    parameters = list(signature(type_class).parameters.values())
    replayed = [field.name for field in fields(type_class) if field.init]

    assert [parameter.name for parameter in parameters] == replayed
    assert all(parameter.kind is Parameter.POSITIONAL_OR_KEYWORD for parameter in parameters)


@pytest.mark.parametrize("original", SAMPLE_TYPES, ids=str)
def test_deepcopy_preserves_type(original: Type) -> None:
    """Check that `copy.deepcopy` preserves a type.

    Deep copying uses `__reduce__` as well, so it failed with the same missing attribute before and
    is repaired by the same change.

    Args:
        original (Type): The type to copy.
    """
    copied = copy.deepcopy(original)

    assert type(copied) is type(original)
    assert copied == original
    assert hash(copied) == hash(original)
    assert copied.organized == original.organized
    assert copied.free_vars == original.free_vars


def test_pickle_roundtrip_preserves_solution_space(query, component_specifications) -> None:
    """Check that a solution space enumerates the same terms after a pickle round trip.

    Non-terminals of a solution space are types, so a solution space could not be stored at all
    before types became picklable. The limit that remains is the one every pickle has: combinators,
    predicates and literal value functions have to be picklable themselves, which is why the
    specifications below are built from module level functions rather than from lambdas.

    `SolutionSpace.enumerate_trees` promises no particular term order, so the comparison below is
    deliberately order independent. It compares the terms as strings first, because a failure then
    names the terms that differ instead of printing tree object addresses.

    Args:
        query (Type): The synthesis target.
        component_specifications (dict): The component specifications of the repository.
    """
    solution_space: SolutionSpace = Synthesizer(component_specifications).construct_solution_space(query)
    before: list[Tree] = list(solution_space.enumerate_trees(query))
    assert len(before) == 4

    restored: SolutionSpace = pickle.loads(pickle.dumps(solution_space))
    after: list[Tree] = list(restored.enumerate_trees(query))

    assert sorted(str(tree) for tree in after) == sorted(str(tree) for tree in before)
    assert set(after) == set(before)
