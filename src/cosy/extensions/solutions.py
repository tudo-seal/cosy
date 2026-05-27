"""_summary_."""

from collections.abc import Callable, Generator, Hashable, Iterable, Sequence
from typing import Any, Generic, TypeVar

from cosy.core.subtypes import Taxonomy
from cosy.core.tree import Tree
from cosy.core.types import Abstraction, Implication, Type
from cosy.extensions import visualize

T = TypeVar("T", bound=Hashable)


class Solutions(Generic[T]):
    """_summary_.

    Attributes:
        tree_generator (Iterable[Tree[T]]): _description_
    """

    def __init__(self, tree_generator: Iterable[Tree[T]]):
        """_summary_.

        Args:
            tree_generator (Iterable[Tree[T]]): _description_
        """
        self._generated_trees: list[Tree[T]] = []
        self.tree_generator = tree_generator
        # def caching_tree_generator():
        #     for tree in self._generated_trees:
        #         yield tree
        #     for tree in tree_generator:
        #         self._generated_trees.append(tree)
        #         yield tree
        #
        # self.trees = caching_tree_generator()

    def trees(self) -> Generator[Tree[T], None, None]:
        """_summary_.

        Yields:
            Tree[T]: _description_
        """
        for tree in self._generated_trees:
            yield tree
        for tree in self.tree_generator:
            self._generated_trees.append(tree)
            yield tree


class _MaestroSolutions(Solutions[T]):
    """_summary_.

    Attributes:
        component_interpretations (dict[T, Callable[..., Any]]): _description_
        named_components_with_specifications (Sequence[tuple[T, Callable, Abstraction | Implication | Type]]): _description_
        taxonomy (Taxonomy | None): _description_
        max_count (int | None): _description_
    """

    def __init__(
        self,
        trees: Iterable[Tree[T]],
        component_interpretations: dict[T, Callable[..., Any]],
        named_components_with_specifications: Sequence[tuple[T, Callable, Abstraction | Implication | Type]],
        max_count: int | None,
        taxonomy: Taxonomy | None,
    ):
        """_summary_.

        Args:
            trees (Iterable[Tree[T]]): _description_
            component_interpretations (dict[T, Callable[..., Any]]): _description_
            named_components_with_specifications (Sequence[tuple[T, Callable, Abstraction | Implication | Type]]): _description_
            max_count (int | None): _description_
            taxonomy (Taxonomy | None): _description_
        """
        super().__init__(trees)
        self.component_interpretations = component_interpretations
        self.named_components_with_specifications = named_components_with_specifications
        self.taxonomy = taxonomy
        self.max_count = max_count

    def __iter__(self) -> Generator[Any, None, None]:
        """_summary_.

        Yields:
            Any: _description_
        """
        for result in self.trees():
            yield result.interpret(interpretation=self.component_interpretations)

    def visualize(self, amount: int | None = None):
        """_summary_.

        Args:
            amount (int | None): _description_ (Default value = None)

        Raises:
            ValueError: _description_
        """
        if amount is None:
            if self.max_count is not None:
                amount = self.max_count
            else:
                msg = "No max_count provided to maestro.query(), so you must provide a specific amount of results to visualize"
                raise ValueError(msg)

        visualize.visualize(
            amount=amount,
            trees=self.trees(),
            named_components_with_specifications=self.named_components_with_specifications,
            taxonomy=self.taxonomy,
        )
