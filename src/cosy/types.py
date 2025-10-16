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
    is_omega: bool = field(init=True, kw_only=True, compare=False)
    size: int = field(init=True, kw_only=True, compare=False)
    organized: set[Type] = field(init=True, kw_only=True, compare=False)
    free_vars: set[str] = field(init=True, kw_only=True, compare=False)

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def _organized(self) -> set[Type]:
        pass

    @abstractmethod
    def _size(self) -> int:
        pass

    @abstractmethod
    def _is_omega(self) -> bool:
        pass

    @abstractmethod
    def _free_vars(self) -> set[str]:
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
                result = Intersection(ty, result)
            return result
        return Omega()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["is_omega"]
        del state["size"]
        del state["organized"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.__dict__["is_omega"] = self._is_omega()
        self.__dict__["size"] = self._size()
        self.__dict__["organized"] = self._organized()

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
    is_omega: bool = field(init=False, compare=False)
    size: int = field(init=False, compare=False)
    organized: set[Type] = field(init=False, compare=False)
    free_vars: set[str] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        super().__init__(
            is_omega=self._is_omega(),
            size=self._size(),
            organized=self._organized(),
            free_vars=self._free_vars(),
        )

    def _is_omega(self) -> bool:
        return True

    def _size(self) -> int:
        return 1

    def _organized(self) -> set[Type]:
        return set()

    def __str__(self) -> str:
        return "omega"

    def _free_vars(self) -> set[str]:
        return set()

    def subst(self, _substitution: dict[str, Any]) -> Type:
        return self


@dataclass(frozen=True)
class Constructor(Type):
    name: str = field(init=True)
    arg: Type = field(default=Omega(), init=True)
    is_omega: bool = field(init=False, compare=False)
    size: int = field(init=False, compare=False)
    organized: set[Type] = field(init=False, compare=False)
    free_vars: set[str] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        super().__init__(
            is_omega=self._is_omega(),
            size=self._size(),
            organized=self._organized(),
            free_vars=self._free_vars(),
        )

    def _is_omega(self) -> bool:
        return False

    def _size(self) -> int:
        return 1 + self.arg.size

    def _organized(self) -> set[Type]:
        if len(self.arg.organized) <= 1:
            return {self}
        return {Constructor(self.name, ap) for ap in self.arg.organized}

    def _free_vars(self) -> set[str]:
        return self.arg.free_vars

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
    is_omega: bool = field(init=False, compare=False)
    size: int = field(init=False, compare=False)
    organized: set[Type] = field(init=False, compare=False)
    free_vars: set[str] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        super().__init__(
            is_omega=self._is_omega(),
            size=self._size(),
            organized=self._organized(),
            free_vars=self._free_vars(),
        )

    def _is_omega(self) -> bool:
        return self.target.is_omega

    def _size(self) -> int:
        return 1 + self.source.size + self.target.size

    def _organized(self) -> set[Type]:
        if len(self.target.organized) == 0:
            return set()
        if len(self.target.organized) == 1:
            return {self}
        return {Arrow(self.source, tp) for tp in self.target.organized}

    def _free_vars(self) -> set[str]:
        return set.union(self.source.free_vars, self.target.free_vars)

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
    is_omega: bool = field(init=False, compare=False)
    size: int = field(init=False, compare=False)
    organized: set[Type] = field(init=False, compare=False)
    free_vars: set[str] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        super().__init__(
            is_omega=self._is_omega(),
            size=self._size(),
            organized=self._organized(),
            free_vars=self._free_vars(),
        )

    def _is_omega(self) -> bool:
        return self.left.is_omega and self.right.is_omega

    def _size(self) -> int:
        return 1 + self.left.size + self.right.size

    def _organized(self) -> set[Type]:
        return set.union(self.left.organized, self.right.organized)

    def _free_vars(self) -> set[str]:
        return set.union(self.left.free_vars, self.right.free_vars)

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
    is_omega: bool = field(init=False, compare=False)
    size: int = field(init=False, compare=False)
    organized: set[Type] = field(init=False, compare=False)
    free_vars: set[str] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        super().__init__(
            is_omega=self._is_omega(),
            size=self._size(),
            organized=self._organized(),
            free_vars=self._free_vars(),
        )

    def _is_omega(self) -> bool:
        return False

    def _size(self) -> int:
        return 1

    def _organized(self) -> set[Type]:
        return {self}

    def _free_vars(self) -> set[str]:
        return set()

    def __str__(self) -> str:
        return f"[{self.value!s}]"

    def subst(self, _substitution: dict[str, Any]) -> Type:
        return self


@dataclass(frozen=True)
class Var(Type):
    name: str
    is_omega: bool = field(init=False, compare=False)
    size: int = field(init=False, compare=False)
    organized: set[Type] = field(init=False, compare=False)
    free_vars: set[str] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        super().__init__(
            is_omega=self._is_omega(),
            size=self._size(),
            organized=self._organized(),
            free_vars=self._free_vars(),
        )

    def _is_omega(self) -> bool:
        return False

    def _size(self) -> int:
        return 1

    def _organized(self) -> set[Type]:
        return {self}

    def _free_vars(self) -> set[str]:
        return {self.name}

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
