"""Definition of intersection types `Type` and parameterized abstractions `Abstraction`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, fields


@dataclass(frozen=True)
class Type(ABC):
    """_summary_.

    Attributes:
        organized (set[Type]): _description_
        free_vars (set[str]): _description_
    """

    organized: set[Type] = field(init=True, kw_only=True, compare=False)
    free_vars: set[str] = field(init=True, kw_only=True, compare=False)

    @abstractmethod
    def __str__(self) -> str:
        """_summary_."""

    def __reduce__(self) -> tuple[type[Type], tuple[Any, ...]]:
        """Reconstruct through the constructor instead of through the instance dictionary.

        `Constructor`, `Literal` and `Var` store themselves in their derived `organized` field, so
        the default two-step pickling, which allocates an empty instance and then restores its
        dictionary, has to build that set while the instance is still empty. Inserting a value into
        a set hashes it, and the hash of a frozen dataclass reads fields that are only restored one
        step later, so unpickling fails with a missing attribute. Dumping always works, which is why
        the defect stayed hidden. Deep copying failed with the same missing attribute, because it
        uses the same protocol.

        Replaying the constructor keeps the derived fields out of the stream entirely and lets
        `__post_init__` rebuild them once the value is complete. The arguments to replay are exactly
        the fields the constructor takes, and those are exactly the ones that do not carry the
        cycle. They are replayed positionally in declaration order, so every concrete subclass has
        to redeclare `organized` and `free_vars` as `init=False` and must not add keyword-only or
        `InitVar` constructor parameters. The test suite checks that invariant for every concrete
        type class of this module.

        Returns:
            The class and the arguments that rebuild the instance.
        """
        return self.__class__, tuple(getattr(self, f.name) for f in fields(self) if f.init)

    @abstractmethod
    def subst(self, substitution: dict[str, Any]) -> Type:
        """_summary_.

        Args:
            substitution (dict[str, Any]): _description_
        """

    @staticmethod
    def intersect(types: Sequence[Type]) -> Type:
        """_summary_.

        Args:
            types (Sequence[Type]): _description_

        Returns:
            Type: _description_
        """
        result: Type = Omega()
        for ty in reversed(types):
            if not ty.organized:
                continue
            result = Intersection(ty, result) if result.organized else ty
        return result

    @staticmethod
    def curry(sources: Sequence[Type], target: Type) -> Type:
        """_summary_.

        Args:
            sources (Sequence[Type]): _description_
            target (Type): _description_

        Returns:
            Type: _description_
        """
        if isinstance(target, Omega):
            # if the target is omega, then the type is omega via subtyping
            return target
        result: Type = target
        for src in reversed(sources):
            result = Arrow(src, result)
        return result

    def __pow__(self, other: Type) -> Type:
        """_summary_.

        Args:
            other (Type): _description_

        Returns:
            Type: _description_
        """
        return Arrow(self, other)

    def __and__(self, other: Type) -> Type:
        """_summary_.

        Args:
            other (Type): _description_

        Returns:
            Type: _description_
        """
        return Intersection(self, other)

    def __rmatmul__(self, name: str) -> Type:
        """_summary_.

        Args:
            name (str): _description_

        Returns:
            Type: _description_
        """
        return Constructor(name, self)


class Group(ABC):
    """_summary_."""

    name: str = field(init=False)

    @abstractmethod
    def __iter__(self):
        # enumeration logic
        """_summary_."""

    @abstractmethod
    def __contains__(self, x: Any) -> bool:
        # default membership logic
        """_summary_.

        Args:
            x (Any): _description_

        Returns:
            bool: _description_
        """
        return x in self.__iter__()

    def __str__(self) -> str:
        """_summary_.

        Returns:
            str: _description_
        """
        return f"{self.name}"


class DataGroup(Group):
    # Group definition based on given data (e.g. a list, range, set, ...)
    """_summary_.

    Attributes:
        name (str): _description_
    """

    def __init__(self, name: str, data: Iterable):
        """_summary_.

        Args:
            name (str): _description_
            data (Iterable): _description_
        """
        self.name = name
        self._data = data

    def __iter__(self) -> Any:
        """_summary_.

        Returns:
            Any: _description_
        """
        return iter(self._data)

    def __contains__(self, x: Any) -> bool:
        """_summary_.

        Args:
            x (Any): _description_

        Returns:
            bool: _description_
        """
        return x in self._data


@dataclass(frozen=True)
class Omega(Type):
    """_summary_.

    Attributes:
        organized (set[Type]): _description_
        free_vars (set[str]): _description_
    """

    organized: set[Type] = field(init=False, compare=False)
    free_vars: set[str] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        """_summary_."""
        super().__init__(
            organized=set(),
            free_vars=set(),
        )

    def __str__(self) -> str:
        """_summary_.

        Returns:
            str: _description_
        """
        return "omega"

    def subst(self, _substitution: dict[str, Any]) -> Type:
        """_summary_.

        Returns:
            Type: _description_
        """
        return self


@dataclass(frozen=True)
class Constructor(Type):
    """_summary_.

    Attributes:
        name (str): _description_
        arg (Type): _description_
        organized (set[Type]): _description_
        free_vars (set[str]): _description_
    """

    name: str = field(init=True)
    arg: Type = field(default=Omega(), init=True)
    organized: set[Type] = field(init=False, compare=False)
    free_vars: set[str] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        """_summary_."""
        super().__init__(
            organized={self}
            if len(self.arg.organized) <= 1
            else {Constructor(self.name, ap) for ap in self.arg.organized},
            free_vars=self.arg.free_vars,
        )

    def __str__(self) -> str:
        """_summary_.

        Returns:
            str: _description_
        """
        if self.arg == Omega():
            return str(self.name)
        return f"{self.name!s}({self.arg!s})"

    def subst(self, substitution: dict[str, Any]) -> Type:
        """_summary_.

        Args:
            substitution (dict[str, Any]): _description_

        Returns:
            Type: _description_
        """
        return Constructor(self.name, self.arg.subst(substitution))


@dataclass(frozen=True)
class Arrow(Type):
    """_summary_.

    Attributes:
        source (Type): _description_
        target (Type): _description_
        organized (set[Type]): _description_
        free_vars (set[str]): _description_
    """

    source: Type = field(init=True)
    target: Type = field(init=True)
    organized: set[Type] = field(init=False, compare=False)
    free_vars: set[str] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        """_summary_.

        Raises:
            TypeError: _description_
        """
        if isinstance(self.target, Omega):
            # if the target is omega, then via subtyping the type is omega and not an Arrow type
            msg = "Arrow type creation with omega target is unsafe. Use Type.curry for safe arrow creation which respects subtyping."
            raise TypeError(msg)
        super().__init__(
            organized={self}
            if len(self.target.organized) == 1
            else {Arrow(self.source, tp) for tp in self.target.organized},
            free_vars=set.union(self.source.free_vars, self.target.free_vars),
        )

    def __str__(self) -> str:
        """_summary_.

        Returns:
            str: _description_
        """
        return f"{self.source} -> {self.target}"

    def subst(self, substitution: dict[str, Any]) -> Type:
        """_summary_.

        Args:
            substitution (dict[str, Any]): _description_

        Returns:
            Type: _description_
        """
        return Arrow(
            self.source.subst(substitution),
            self.target.subst(substitution),
        )


@dataclass(frozen=True)
class Intersection(Type):
    """_summary_.

    Attributes:
        left (Type): _description_
        right (Type): _description_
        organized (set[Type]): _description_
        free_vars (set[str]): _description_
    """

    left: Type = field(init=True)
    right: Type = field(init=True)
    organized: set[Type] = field(init=False, compare=False)
    free_vars: set[str] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        """_summary_.

        Raises:
            TypeError: _description_
            TypeError: _description_
        """
        if isinstance(self.left, Omega):
            # if the left is omega, then via subtyping the type is right and not necessarily an Intersection type
            msg = "Intersection type creation with omega left is unsafe. Use Type.intersect for safe intersection creation which respects subtyping."
            raise TypeError(msg)
        if isinstance(self.right, Omega):
            # if the right is omega, then via subtyping the type is left and not necessarily an Intersection type
            msg = "Intersection type creation with omega right is unsafe. Use Type.intersect for safe intersection creation which respects subtyping."
            raise TypeError(msg)
        super().__init__(
            organized=set.union(self.left.organized, self.right.organized),
            free_vars=set.union(self.left.free_vars, self.right.free_vars),
        )

    def __str__(self) -> str:
        """_summary_.

        Returns:
            str: _description_
        """
        return f"{self.left} & {self.right}"

    def subst(self, substitution: dict[str, Any]) -> Type:
        """_summary_.

        Args:
            substitution (dict[str, Any]): _description_

        Returns:
            Type: _description_
        """
        return Intersection(
            self.left.subst(substitution),
            self.right.subst(substitution),
        )


@dataclass(frozen=True)
class Literal(Type):
    """_summary_.

    Attributes:
        value (Any): _description_
        organized (set[Type]): _description_
        free_vars (set[str]): _description_
    """

    value: Any  # has to be Hashable
    organized: set[Type] = field(init=False, compare=False)
    free_vars: set[str] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        """_summary_."""
        super().__init__(
            organized={self},
            free_vars=set(),
        )

    def __str__(self) -> str:
        """_summary_.

        Returns:
            str: _description_
        """
        return f"[{self.value!s}]"

    def subst(self, _substitution: dict[str, Any]) -> Type:
        """_summary_.

        Returns:
            Type: _description_
        """
        return self


@dataclass(frozen=True)
class Var(Type):
    """_summary_.

    Attributes:
        name (str): _description_
        organized (set[Type]): _description_
        free_vars (set[str]): _description_
    """

    name: str
    organized: set[Type] = field(init=False, compare=False)
    free_vars: set[str] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        """_summary_."""
        super().__init__(
            organized={self},
            free_vars={self.name},
        )

    def __str__(self) -> str:
        """_summary_.

        Returns:
            str: _description_
        """
        return f"<{self.name!s}>"

    def subst(self, substitution: dict[str, Any]) -> Type:
        """_summary_.

        Args:
            substitution (dict[str, Any]): _description_

        Returns:
            Type: _description_

        Raises:
            ValueError: _description_
        """
        if self.name in substitution:
            return Literal(substitution[self.name])
        msg = f"Variable {self.name} not found in substitution."
        raise ValueError(msg)


@dataclass(frozen=True)
class Parameter(ABC):
    """Abstract base class for parameter specification."""

    name: str
    group: Group | Type

    def __str__(self) -> str:
        """_summary_.

        Returns:
            str: _description_
        """
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
    """_summary_.

    Attributes:
        constraint (Callable[[dict[str, Any]], bool]): _description_
        only_literals (bool): _description_
    """

    constraint: Callable[[dict[str, Any]], bool]
    only_literals: bool

    def __str__(self) -> str:
        """_summary_.

        Returns:
            str: _description_
        """
        return f"[{self.constraint.__name__}, only literals]" if self.only_literals else f"[{self.constraint.__name__}]"


@dataclass(frozen=True)
class Implication:
    """_summary_.

    Attributes:
        predicate (Predicate): _description_
        body (Abstraction | Implication | Type): _description_
    """

    predicate: Predicate
    body: Abstraction | Implication | Type

    def __str__(self) -> str:
        """_summary_.

        Returns:
            str: _description_
        """
        return f"{self.predicate} => {self.body}"


@dataclass(frozen=True)
class Abstraction:
    """Abstraction of a term parameter or a literal parameter."""

    parameter: Parameter
    body: Abstraction | Implication | Type

    def __str__(self) -> str:
        """_summary_.

        Returns:
            str: _description_
        """
        return f"{self.parameter}.{self.body}"
