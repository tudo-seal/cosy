"""
Definition of intersection types `Type` and parameterized abstractions `Abstraction`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Type(ABC):
    organized: set[Type] = field(init=True, kw_only=True, compare=False)
    free_vars: set[str] = field(init=True, kw_only=True, compare=False)

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def subst(self, substitution: dict[str, Any]) -> Type:
        pass

    @staticmethod
    def intersect(types: Sequence[Type]) -> Type:
        if len(types) > 0:
            rtypes = reversed(types)
            result: Type = next(rtypes)
            for ty in rtypes:
                if ty.organized:
                    result = Intersection(ty, result)
            return result
        return Omega()

    def __pow__(self, other: Type) -> Type:
        return Arrow(self, other)

    def __and__(self, other: Type) -> Type:
        return Intersection(self, other)

    def __rmatmul__(self, name: str) -> Type:
        return Constructor(name, self)


class Group(ABC):
    name: str = field(init=False)

    @abstractmethod
    def __iter__(self):
        # enumeration logic
        pass

    @abstractmethod
    def __contains__(self, x):
        # default membership logic
        return x in self.__iter__()

    def __str__(self) -> str:
        return f"{self.name}"


class DataGroup(Group):
    # Group definition based on given data (e.g. a list, range, set, ...)
    def __init__(self, name: str, data: Iterable):
        self.name = name
        self._data = data

    def __iter__(self):
        return iter(self._data)

    def __contains__(self, x):
        return x in self._data


@dataclass(frozen=True)
class Omega(Type):
    organized: set[Type] = field(init=False, compare=False)
    free_vars: set[str] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        super().__init__(
            organized=set(),
            free_vars=set(),
        )

    def __str__(self) -> str:
        return "omega"

    def subst(self, _substitution: dict[str, Any]) -> Type:
        return self


@dataclass(frozen=True)
class Constructor(Type):
    name: str = field(init=True)
    arg: Type = field(default=Omega(), init=True)
    organized: set[Type] = field(init=False, compare=False)
    free_vars: set[str] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        super().__init__(
            organized={self}
            if len(self.arg.organized) <= 1
            else {Constructor(self.name, ap) for ap in self.arg.organized},
            free_vars=self.arg.free_vars,
        )

    def __str__(self) -> str:
        if self.arg == Omega():
            return str(self.name)
        return f"{self.name!s}({self.arg!s})"

    def subst(self, substitution: dict[str, Any]) -> Type:
        if not any(var in substitution for var in self.free_vars):
            return self
        return Constructor(self.name, self.arg.subst(substitution))


@dataclass(frozen=True)
class Arrow(Type):
    source: Type = field(init=True)
    target: Type = field(init=True)
    organized: set[Type] = field(init=False, compare=False)
    free_vars: set[str] = field(init=False, compare=False)

    def __new__(cls, _source: Type, target: Type):
        if isinstance(target, Omega):
            # if the target is omega, then the type is omega via subtyping
            return Omega()
        return object.__new__(cls)

    def __post_init__(self) -> None:
        super().__init__(
            organized={self}
            if len(self.target.organized) == 1
            else {Arrow(self.source, tp) for tp in self.target.organized},
            free_vars=set.union(self.source.free_vars, self.target.free_vars),
        )

    def __str__(self) -> str:
        return f"{self.source} -> {self.target}"

    def subst(self, substitution: dict[str, Any]) -> Type:
        if not any(var in substitution for var in self.free_vars):
            return self
        return Arrow(
            self.source.subst(substitution),
            self.target.subst(substitution),
        )


@dataclass(frozen=True)
class Intersection(Type):
    left: Type = field(init=True)
    right: Type = field(init=True)
    organized: set[Type] = field(init=False, compare=False)
    free_vars: set[str] = field(init=False, compare=False)

    def __new__(cls, left: Type, right: Type):
        if isinstance(left, Omega):
            return right
        if isinstance(right, Omega):
            return left

        instance = object.__new__(cls)
        object.__setattr__(instance, "left", left)
        object.__setattr__(instance, "right", right)
        object.__setattr__(instance, "organized", set.union(left.organized, right.organized))
        object.__setattr__(instance, "free_vars", set.union(left.free_vars, right.free_vars))
        return instance

    def __init__(self, _left: Type, _right: Type):
        pass

    def __str__(self) -> str:
        return f"{self.left} & {self.right}"

    def subst(self, substitution: dict[str, Any]) -> Type:
        if not any(var in substitution for var in self.free_vars):
            return self
        return Intersection(
            self.left.subst(substitution),
            self.right.subst(substitution),
        )


@dataclass(frozen=True)
class Literal(Type):
    value: Any  # has to be Hashable
    organized: set[Type] = field(init=False, compare=False)
    free_vars: set[str] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        super().__init__(
            organized={self},
            free_vars=set(),
        )

    def __str__(self) -> str:
        return f"[{self.value!s}]"

    def subst(self, _substitution: dict[str, Any]) -> Type:
        return self


@dataclass(frozen=True)
class Var(Type):
    name: str
    organized: set[Type] = field(init=False, compare=False)
    free_vars: set[str] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        super().__init__(
            organized={self},
            free_vars={self.name},
        )

    def __str__(self) -> str:
        return f"<{self.name!s}>"

    def subst(self, substitution: dict[str, Any]) -> Type:
        if self.name in substitution:
            return Literal(substitution[self.name])
        return self


@dataclass(frozen=True)
class Parameter(ABC):
    """Abstract base class for parameter specification."""

    name: str
    group: Group | Type

    def __str__(self) -> str:
        return f"<{self.name}, {self.group}>"


@dataclass(frozen=True)
class LiteralParameter(Parameter):
    """Specification of a literal parameter."""

    group: Group
    #  Specification of literal assignment from a collection
    values: Callable[[dict[str, Any]], Sequence[Any]] | None = field(default=None)


@dataclass(frozen=True)
class TermParameter(Parameter):
    """Specification of a term parameter."""

    group: Type


@dataclass(frozen=True)
class Predicate:
    constraint: Callable[[dict[str, Any]], bool]
    only_literals: bool

    def __str__(self) -> str:
        return f"[{self.constraint.__name__}, only literals]" if self.only_literals else f"[{self.constraint.__name__}]"


@dataclass(frozen=True)
class Implication:
    predicate: Predicate
    body: Abstraction | Implication | Type

    def __str__(self) -> str:
        return f"{self.predicate} => {self.body}"


@dataclass(frozen=True)
class Abstraction:
    """Abstraction of a term parameter or a literal parameter."""

    parameter: Parameter
    body: Abstraction | Implication | Type

    def __str__(self) -> str:
        return f"{self.parameter}.{self.body}"
