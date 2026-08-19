"""_summary_."""

# Literature
# [1] Van Der Rest, Cas, and Wouter Swierstra. "A completely unique account of enumeration."
#     Proceedings of the ACM on Programming Languages 6.ICFP (2022): 105.
import contextlib

# Here, the indexed type [1, Section 4] is the tree grammar, where indices are non-terminals.
# Uniqueness is guaranteed by python's set (instead of list) data structure.
from collections import deque
from collections.abc import Callable, Hashable, Sequence
from functools import partial
from inspect import Parameter, _empty, _ParameterKind, signature
from typing import Any, Generic, TypeVar

T = TypeVar("T", bound=Hashable)

Path = tuple[int, ...]


class Tree(Generic[T]):
    """Please only use immutably."""

    root: T
    children: tuple["Tree[T]", ...]
    size: int
    _hash: int
    _positions: set[Path] | None = None
    _leaf_positions: set[Path] | None = None
    # tuple[interpretation id, reference to interpretation dict (avoid GC, see test), cached interpretation result]
    # Breaks for non-deterministic interpretations. The entry belongs to the node, and a node is
    # shared by every term built around it, so an interpretation dict that is changed in place --
    # same object, same id, different contents -- now reports its stale value through every one of
    # those terms rather than only through the one the node was first evaluated in.
    _interpreted: tuple[int, dict[T, Any] | None, Any] | None = None

    def __init__(self, root: T, children: Sequence["Tree[T]"] = ()) -> None:
        """_summary_.

        Args:
            root (T): _description_
            children (Sequence['Tree[T]']): _description_ (Default value = ())
        """
        self.root = root
        self.children = tuple(children)
        self.size = 1 + sum(child.size for child in self.children)
        self._hash = hash((self.root, self.children))

    def __hash__(self) -> int:
        """_summary_.

        Returns:
            int: _description_
        """
        return self._hash

    def __lt__(self, other: "Tree[T]") -> bool:
        """_summary_.

        Args:
            other (Tree[T]): _description_

        Returns:
            bool: _description_
        """
        return self.size < other.size

    def __eq__(self, other: object) -> bool:
        """_summary_.

        Args:
            other (object): _description_

        Returns:
            bool: _description_
        """
        if not isinstance(other, Tree):
            return False
        return self.size == other.size and self.root == other.root and self.children == other.children

    def __rec_to_str__(self, *, outermost: bool) -> str:
        """_summary_.

        Args:
            outermost (bool): _description_

        Returns:
            str: _description_
        """
        str_root = [f"{self.root!s}"]
        str_args = [f"{subtree.__rec_to_str__(outermost=False)}" for subtree in self.children]

        strings = str_root + str_args
        if not outermost and len(strings) > 1:
            return f"({' '.join(strings)})"
        return " ".join(strings)

    def __str__(self) -> str:
        """_summary_.

        Returns:
            str: _description_
        """
        return self.__rec_to_str__(outermost=True)

    def __copy__(self) -> "Tree[T]":
        """Return a new root over the same children.

        Shallow, because the nodes are immutable: nothing a caller can do to the copy can be
        observed through the original, so copying the children as well would only cost the size
        of the term.  The recursive version was ``deepcopy`` under another name, and it made
        every ``subtree_at`` a copy of the subtree it read.

        Returns:
            Tree[T]: A node equal to this one, sharing its children.
        """
        return Tree(root=self.root, children=self.children)

    def interpret(self, interpretation: dict[T, Any] | None = None) -> Any:
        """Recursively evaluate given term.

        Args:
            interpretation (dict[T, Any] | None): _description_ (Default value = None)

        Returns:
            Any: _description_

        Raises:
            TypeError: _description_
            TypeError: _description_
        """

        # if interpretation hasn't changed, skip interpreting, use cache
        evaluated = self._interpreted
        if evaluated is not None and evaluated[0] == id(interpretation):
            return evaluated[2]

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
        result = results.pop()

        # no cache hit (first seen or interpretation changed), overwrite cache
        self._interpreted = (id(interpretation), interpretation, result)
        return result

    def positions(self) -> set[Path]:
        """Return all positions in the tree.

        Returns:
            set[Path]: _description_
        """
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
        """Return all leaf positions in the tree.

        Returns:
            set[Path]: _description_
        """
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
        """Return the subtree at the given position.

        The node itself, not a copy of it.  The class is immutable, so sharing is safe, and it is
        what the rest of the class assumes: ``replace_subtree_at`` shares every node off the path
        it rebuilds.  Copying on the way back up made reading a position cost a copy of
        everything below it -- reading every position of a 2047-node term took 94206 copies.

        Iterative rather than recursive: terms grow to hundreds of nodes, and a chain that deep
        overflows a recursive descent.

        Args:
            pos (Path): The position, as a tuple of child indices.

        Returns:
            Tree[T]: The node at ``pos``.

        Raises:
            IndexError: If ``pos`` does not address a node of this tree.
        """
        current: Tree[T] = self
        for index in pos:
            if index < 0 or index >= len(current.children):
                msg = f"Path {pos} is not valid for this tree"
                raise IndexError(msg)
            current = current.children[index]
        return current

    def replace_subtree_at(self, pos: Path, tree: "Tree[T]") -> "Tree[T]":
        """Return a copy of this tree with the subtree at the given position replaced.

        Neither this tree nor the replacement is modified. The result shares every node that is not
        on the path from the root to ``pos`` with ``self``, and shares ``tree`` itself; only the
        nodes along that path are rebuilt.

        Rebuilding through ``__init__`` rather than mutating in place is what keeps ``size`` and
        ``_hash`` correct. Both are computed once at construction, so an in-place replacement left
        every ancestor of the replacement point reporting the values it had *before* the change --
        which broke the equality/hash contract of the class and, through it, every cache and every
        ``set`` keyed on trees.

        Args:
            pos (Path): Position of the subtree to replace, as a tuple of child indices.
            tree (Tree[T]): The replacement subtree.

        Returns:
            Tree[T]: The resulting tree. For ``pos == ()`` this is ``tree`` itself.

        Raises:
            IndexError: If ``pos`` does not address a node of this tree.
        """
        if pos == ():
            return tree

        # descend to the replacement point, remembering the nodes along the way
        path_nodes: list[Tree[T]] = [self]
        current: Tree[T] = self
        for index in pos:
            if index < 0 or index >= len(current.children):
                msg = f"Path {pos} is not valid for this tree"
                raise IndexError(msg)
            current = current.children[index]
            path_nodes.append(current)

        # rebuild bottom-up; everything off the path is shared rather than copied
        replacement = tree
        for depth in range(len(pos) - 1, -1, -1):
            parent = path_nodes[depth]
            index = pos[depth]
            replacement = Tree(
                parent.root,
                (*parent.children[:index], replacement, *parent.children[index + 1 :]),
            )
        return replacement
