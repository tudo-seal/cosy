"""_summary_."""

# Literature
# [1] Van Der Rest, Cas, and Wouter Swierstra. "A completely unique account of enumeration."
#     Proceedings of the ACM on Programming Languages 6.ICFP (2022): 105.
import contextlib

# Here, the indexed type [1, Section 4] is the tree grammar, where indices are non-terminals.
# Uniqueness is guaranteed by python's set (instead of list) data structure.
from collections import deque
from collections.abc import Callable, Hashable, Sequence
from functools import lru_cache, partial
from inspect import Parameter, _empty, _ParameterKind, signature
from typing import Any, Generic, TypeVar

T = TypeVar("T", bound=Hashable)

Path = tuple[int, ...]


# ``inspect.signature`` costs about 5.4 microseconds, and ``interpret`` asked it once per
# occurrence of a combinator rather than once per combinator.  A chain of five thousand nodes over
# two combinators paid five thousand times for two answers.  The memo is keyed on the combinator,
# which is what carries the signature.
#
# What the bound has to hold is the set of combinators evaluated together, which is the size of a
# component repository: 3 to 7 in the examples here, 4 in the benchmarks, and 24 to 49 in the
# largest algebras in use elsewhere.  An LRU that no longer holds its working set drops to a zero
# hit rate at once rather than declining, because every lookup evicts the entry the next one asks
# for.  Measured over 200 combinators, ``maxsize=128`` runs about ninety times slower than
# ``maxsize=1024``.  ``_parameters_cached.cache_info()`` reports a set that no longer fits.
#
# The bound also limits retention.  A caller that builds a fresh algebra per evaluation gets its
# hits within one call and none across calls, so an unbounded memo keeps every callable it has
# seen, with whatever those callables close over.  Measured on that pattern, it grew to 200000
# entries in 50000 evaluations and became slower than the bounded memo, since its table keeps
# being rebuilt.
#
# The memo holds metadata about a combinator, never the result of applying one.  Every combinator
# is still called on every evaluation, so an interpretation may have side effects and may answer
# differently each time.  See ``tests/test_interpretation_semantics.py``.
@lru_cache(maxsize=1024)
def _parameters_cached(combinator: Callable[..., Any]) -> tuple[Parameter, ...]:
    """Return the parameters of a callable, from a bounded memo.

    Args:
        combinator (Callable[..., Any]): The callable to inspect.

    Returns:
        tuple[Parameter, ...]: Its parameters, in declaration order.  A tuple rather than a list,
            because the memo hands out the object it stored.

    Raises:
        ValueError: If ``combinator`` exposes no signature.  ``interpret`` turns this into a
            ``TypeError`` naming the combinator.  A memo that swallowed it would answer wrongly
            instead of reporting the failure.
        TypeError: If ``combinator`` cannot be a memo key, raised by ``lru_cache`` before this
            body runs, or if ``signature`` cannot inspect it.  ``_parameters_of`` tells the two
            apart.
    """
    return tuple(signature(combinator).parameters.values())


def _parameters_of(combinator: Callable[..., Any]) -> tuple[Parameter, ...]:
    """Return the parameters of a callable, through the memo wherever that is possible.

    Args:
        combinator (Callable[..., Any]): The callable to inspect.

    Returns:
        tuple[Parameter, ...]: Its parameters, in declaration order.

    Raises:
        ValueError: If ``combinator`` exposes no signature, see ``_parameters_cached``.
        TypeError: If ``signature`` cannot inspect ``combinator``, for instance because it carries
            a ``__signature__`` that is not a signature.  The other ``TypeError``, the one an
            unhashable combinator raises on the memo key, is handled below rather than reported.
    """
    try:
        return _parameters_cached(combinator)
    except TypeError:
        # Two failures arrive as ``TypeError``, and retrying without the memo tells them apart.
        # An unhashable combinator fails on the cache key before the body runs, a value object
        # that defines ``__eq__`` and so has no ``__hash__`` being the common case.  Inspecting it
        # directly is a detour around the memo rather than a failure, and it is the only path the
        # memo adds, so it is tested.  A combinator that ``signature`` itself rejects raises again
        # here and reaches the caller, as it did before the memo existed.  Checking hashability up
        # front would hash every combinator a second time on the hot path, for a case that does
        # not occur in practice.
        return tuple(signature(combinator).parameters.values())


class Tree(Generic[T]):
    """Please only use immutably."""

    root: T
    children: tuple["Tree[T]", ...]
    size: int
    _hash: int
    _positions: frozenset[Path] | None = None
    _leaf_positions: frozenset[Path] | None = None

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

    # Writing is recursive, three ``save()`` levels per node here and four with the default
    # protocol.  On Python 3.10 and 3.11 that recursion is bounded by the interpreter's recursion
    # limit: measured from inside a pytest run, a chain of 318 nodes is the deepest term that can
    # be written, against 238 before.  From 3.12 on the separate C recursion limit binds instead,
    # at about 3325.  Terms do grow deeper than that, which is why ``subtree_at``, ``_walk`` and
    # ``interpret`` are iterative, so pickling is the one place in this class where depth is still
    # a limit.  The reduction below at least raises the ceiling by a third.
    def __reduce__(self) -> tuple[type["Tree[T]"], tuple[T, tuple["Tree[T]", ...]]]:
        """Reconstruct through the constructor instead of through the instance dictionary.

        Everything ``__init__`` computes (``size`` and ``_hash``) and everything filled on demand
        (the two position sets) follows from ``root`` and ``children``, so none of it has to be
        written.  The default protocol writes the instance dictionary, and therefore writes all of
        it, together with the ``__orig_class__`` that ``Tree[str](...)`` leaves on an instance.
        ``__copy__`` and ``replace_subtree_at`` already build their results out of ``root`` and
        ``children`` alone.  This makes the third way of producing a node agree with them.

        ``_hash`` is what makes this more than a question of size.  It is
        ``hash((root, children))``, and hashing a string is randomized per process, so a
        transported ``_hash`` is the writing process's answer to a question the reading process
        would answer differently.  A term read back from a file then compares *equal* to the same
        term built here and hashes *differently*: a ``set`` keeps both of them, and a ``dict``
        deduplicates neither, measured by writing under ``PYTHONHASHSEED=1`` and reading under
        ``PYTHONHASHSEED=2``.  Recomputing on load is what makes a term that arrived by stream
        interchangeable with one built in place.

        The position sets are the plain case: a second encoding of a structure the stream already
        carries.  A term of 2047 nodes with them filled at the root writes 19486 bytes this way
        instead of 116805, and writing it costs 0.67 milliseconds instead of 1.32 (CPython 3.13,
        pickle protocol 4, the default).  Filled at every node, which is what one pass over all
        positions of all subterms leaves behind, the old figure is 380857 bytes and this one is
        unchanged.

        ``copy.deepcopy`` reduces through here as well, so a deep copy now starts with cold caches
        too.  ``copy.copy`` continues to go through ``__copy__``.  The class is taken from
        ``self.__class__`` rather than named outright, because loading must not change what an
        object is.

        Returns:
            tuple[type[Tree[T]], tuple[T, tuple[Tree[T], ...]]]: The class and the constructor
                arguments that rebuild this node.  Shared children stay shared, because pickle
                memoizes the objects it has already written whichever way they are reduced.
        """
        return (self.__class__, (self.root, self.children))

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Rebuild from an instance dictionary rather than adopting it.

        Only a stream written before ``__reduce__`` existed reaches this method, since a reduction
        that hands back constructor arguments produces no state at all.  Such a stream carries the
        derived fields, and adopting them is what the reduction above avoids producing: the
        ``_hash`` in it was computed under the writing process's hash seed.  A stream old enough
        carries an interpretation result as well, which no longer has a field to be read into and
        is dropped here with the rest.  Taking ``root`` and ``children`` and computing the rest
        here makes that fix reach terms that were written before it existed, which matters because
        the terms anyone keeps on disk are the expensive ones, and leaves the position sets cold,
        where they belong.

        Args:
            state (dict[str, Any]): The instance dictionary of the node as it was written.
        """
        self.root = state["root"]
        self.children = state["children"]
        self.size = 1 + sum(child.size for child in self.children)
        self._hash = hash((self.root, self.children))

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
                    parameters_of_c = _parameters_of(current_combinator)
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

    def _walk(self) -> tuple[frozenset[Path], frozenset[Path]]:
        """Fill both position caches in one traversal.

        The leaves are read off the walk, a node without children being a leaf, rather than
        filtered out of the position set afterwards.  Filtering compares every position against
        every other, which is quadratic: a term of 32767 nodes took 84 seconds, and resolving a
        term at a position asks for the leaves of that term on every call.

        Returns:
            tuple[frozenset[Path], frozenset[Path]]: The positions and the leaf positions, in that
                order, as they were stored.
        """
        positions: set[Path] = set()
        leaves: set[Path] = set()
        queue: deque[tuple[Tree[T], Path]] = deque([(self, ())])
        while queue:
            current, path = queue.popleft()
            positions.add(path)
            if not current.children:
                leaves.add(path)
            for i, child in enumerate(current.children):
                queue.append((child, (*path, i)))
        self._positions = frozenset(positions)
        self._leaf_positions = frozenset(leaves)
        return self._positions, self._leaf_positions

    def positions(self) -> frozenset[Path]:
        """Return all positions in the tree.

        The cached set is handed out as it is, frozen rather than copied.  It belongs to the node,
        and a node is shared by every term built around it, so a caller who mutated what it got
        back would change what every one of those terms reports.  Freezing makes that impossible
        instead of merely inadvisable, and it costs nothing per call, where copying the set would
        be linear in the size of the term on every read.

        Returns:
            frozenset[Path]: The positions of every node.
        """
        cached = self._positions
        if cached is None:
            cached, _ = self._walk()
        return cached

    def leaf_positions(self) -> frozenset[Path]:
        """Return all leaf positions in the tree.

        Handed out frozen and uncopied for the same reason ``positions`` is: the cache belongs to
        the node, and the node is shared.

        Returns:
            frozenset[Path]: The positions without children.
        """
        cached = self._leaf_positions
        if cached is None:
            _, cached = self._walk()
        return cached

    def subtree_at(self, pos: Path) -> "Tree[T]":
        """Return the subtree at the given position.

        The node itself, not a copy of it.  The class is immutable, so sharing is safe, and it is
        what the rest of the class assumes: ``replace_subtree_at`` shares every node off the path
        it rebuilds.  Copying on the way back up made reading a position cost a copy of
        everything below it.  Reading every position of a 2047-node term took 94206 copies.

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
        on the path from the root to ``pos`` with ``self``, and shares ``tree`` itself.  Only the
        nodes along that path are rebuilt.

        Rebuilding through ``__init__`` rather than mutating in place is what keeps ``size`` and
        ``_hash`` correct. Both are computed once at construction, so an in-place replacement left
        every ancestor of the replacement point reporting the values it had *before* the change.
        That broke the equality and hash contract of the class and, through it, every cache and
        every ``set`` keyed on trees.

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

        # rebuild bottom-up, everything off the path is shared rather than copied
        replacement = tree
        for depth in range(len(pos) - 1, -1, -1):
            parent = path_nodes[depth]
            index = pos[depth]
            replacement = Tree(
                parent.root,
                (*parent.children[:index], replacement, *parent.children[index + 1 :]),
            )
        return replacement
