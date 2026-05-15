# Literature
# [1] Van Der Rest, Cas, and Wouter Swierstra. "A completely unique account of enumeration."
#     Proceedings of the ACM on Programming Languages 6.ICFP (2022): 105.
import contextlib

# Here, the indexed type [1, Section 4] is the tree grammar, where indices are non-terminals.
# Uniqueness is guaranteed by python's set (instead of list) data structure.
from collections import deque
from collections.abc import Callable, Hashable, Sequence
from copy import copy
from functools import partial
from inspect import Parameter, _empty, _ParameterKind, signature
from typing import Any, Generic, TypeVar

T = TypeVar("T", bound=Hashable)

Path = tuple[int, ...]


class Tree(Generic[T]):
    """
    Please only use immutably.
    """

    root: T
    children: tuple["Tree[T]", ...]
    size: int
    _hash: int
    _positions: set[Path] | None = None
    _leaf_positions: set[Path] | None = None

    def __init__(self, root: T, children: Sequence["Tree[T]"] = ()) -> None:
        self.root = root
        self.children = tuple(children)
        self.size = 1 + sum(child.size for child in self.children)
        self._hash = hash((self.root, self.children))

    def __hash__(self) -> int:
        return self._hash

    def __lt__(self, other: "Tree[T]") -> bool:
        return self.size < other.size

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tree):
            return False
        return self.size == other.size and self.root == other.root and self.children == other.children

    def __rec_to_str__(self, *, outermost: bool) -> str:
        str_root = [f"{self.root!s}"]
        str_args = [f"{subtree.__rec_to_str__(outermost=False)}" for subtree in self.children]

        strings = str_root + str_args
        if not outermost and len(strings) > 1:
            return f"({' '.join(strings)})"
        return " ".join(strings)

    def __str__(self) -> str:
        return self.__rec_to_str__(outermost=True)

    def __copy__(self) -> "Tree[T]":
        children_copy = tuple(copy(child) for child in self.children)
        return Tree(
            root=self.root,
            children=children_copy,
        )

    def interpret(self, interpretation: dict[T, Any] | None = None) -> Any:
        """Recursively evaluate given term."""

        terms: deque[Tree[T]] = deque((self,))
        combinators: deque[tuple[T, int]] = deque()
        # decompose terms
        while terms:
            t = terms.pop()
            combinators.append((t.root, len(t.children)))
            terms.extend(reversed(t.children))
        results: deque[Any] = deque()

        # apply/call decomposed terms
        while combinators:
            (c, n) = combinators.pop()
            parameters_of_c: Sequence[Parameter] = []
            current_combinator: partial[Any] | T | Callable[..., Any] = (
                c if interpretation is None or c not in interpretation else interpretation[c]
            )

            if callable(current_combinator):
                try:
                    parameters_of_c = list(signature(current_combinator).parameters.values())
                except ValueError as exc:
                    msg = (
                        f"Interpretation of combinator {c} does not expose a signature. "
                        "If it's a built-in, you can simply wrap it in another function."
                    )
                    raise TypeError(msg) from exc

                if n == 0 and len(parameters_of_c) == 0:
                    current_combinator = current_combinator()

            arguments = deque(results.pop() for _ in range(n))

            while arguments:
                if not callable(current_combinator):
                    msg = (
                        f"Interpretation of combinator {c} is applied to {n} argument(s), "
                        f"but can only be applied to {n - len(arguments)}"
                    )
                    raise TypeError(msg)

                use_partial = False

                simple_arity = len(list(filter(lambda x: x.default == _empty, parameters_of_c)))
                default_arity = len(list(filter(lambda x: x.default != _empty, parameters_of_c)))

                # if any parameter is marked as var_args, we need to use all available arguments
                pop_all = any(x.kind == _ParameterKind.VAR_POSITIONAL for x in parameters_of_c)

                # If a var_args parameter is found, we need to subtract it from the normal parameters.
                # Note: python does only allow one parameter in the form of *arg
                if pop_all:
                    simple_arity -= 1

                # If a combinator needs more arguments than available, we need to use partial
                # application
                if simple_arity > len(arguments):
                    use_partial = True

                fixed_parameters: deque[Any] = deque(
                    arguments.popleft() for _ in range(min(simple_arity, len(arguments)))
                )

                var_parameters: deque[Any] = deque()
                if pop_all:
                    var_parameters.extend(arguments)
                    arguments = deque()

                default_parameters: deque[Any] = deque()
                for _ in range(default_arity):
                    with contextlib.suppress(IndexError):
                        default_parameters.append(arguments.popleft())

                if use_partial:
                    current_combinator = partial(
                        current_combinator,
                        *fixed_parameters,
                        *var_parameters,
                        *default_parameters,
                    )
                else:
                    current_combinator = current_combinator(*fixed_parameters, *var_parameters, *default_parameters)

            results.append(current_combinator)
        return results.pop()

    def positions(self) -> set[Path]:
        """Return all positions in the tree."""
        if self._positions is not None:
            return self._positions
        result: set[Path] = set()
        queue: deque[tuple[Tree[T], Path]] = deque([(self, ())])
        while queue:
            current, path = queue.popleft()
            result.add(path)
            for i, child in enumerate(current.children):
                queue.append((child, (*path, i)))
        self._positions = result
        return result

    def leaf_positions(self) -> set[Path]:
        """Return all leaf positions in the tree."""
        if self._leaf_positions is not None:
            return self._leaf_positions
        result: set[Path] = set()
        all_positions: set[Path] = self.positions()
        # leaf positions are all positions that are no prefix of another position
        # a prefix of a position is defined as follows: p is a prefix of q if p == q or p is a prefix of q[:-1]
        for pos in all_positions:
            if not any(pos != other and pos == other[: len(pos)] for other in all_positions):
                result.add(pos)
        self._leaf_positions = result
        return result

    def subtree_at(self, pos: Path) -> "Tree[T]":
        """Return subtree at given position."""
        if pos == ():
            return self
        for i, child in enumerate(self.children):
            if i == pos[0]:
                return copy(child.subtree_at(pos[1:]))
        msg = f"Path {pos} is not valid for this tree"
        raise IndexError(msg)

    def replace_subtree_at(self, pos: Path, tree: "Tree[T]") -> "Tree[T]":
        """Return replaced subtree at given position."""
        if pos == ():
            return tree

        # validate pos by attempting to access the subtree once (avoids materializing all positions)
        try:
            _ = self.subtree_at(pos) if pos != () else tree
        except IndexError:
            msg = f"Path {pos} is not valid for this tree"
            raise IndexError(msg) from BaseException

        new_tree = copy(self)

        # traverse the path to the subtree to replace
        current = new_tree
        for i in pos[:-1]:
            if i < 0 or i >= len(current.children):
                msg = "Invalid path."
                raise ValueError(msg)
            current = current.children[i]
        # replace the subtree at the given path
        insert = copy(tree)
        current.children = (*current.children[: pos[-1]], insert, *current.children[pos[-1] + 1 :])
        return new_tree
