import itertools
import json
import os
import pathlib
import threading
from collections import deque
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import TypeAlias, TypeVar

from cosy.core.tree import Tree
from cosy.core.types import (
    Abstraction,
    Arrow,
    Constructor,
    Group,
    Implication,
    Intersection,
    LiteralParameter,
    Omega,
    Parameter,
    TermParameter,
    Type,
)

T = TypeVar("T", bound=Hashable)

Colour: TypeAlias = str

# See: https://eleanormaclure.wordpress.com/wp-content/uploads/2011/03/colour-coding.pdf
# https://gist.github.com/ollieglass/f6ddd781eeae1d24e391265432297538
# Kenneth Kelly: A Colour Alphabet and the Limits of Colour Coding
KELLY_COLOURS: list[Colour] = [
    "#F3C300",
    "#875692",
    "#F38400",
    "#A1CAF1",
    "#BE0032",
    "#C2B280",
    "#848482",
    "#008856",
    "#E68FAC",
    "#0067A5",
    "#F99379",
    "#604E97",
    "#F6A600",
    "#B3446C",
    "#DCD300",
    "#882D17",
    "#8DB600",
    "#654522",
    "#E25822",
    "#2B3D26",
    "#F2F3F4",
    "#222222",
]

ComponentSpecifications: TypeAlias = dict[T, tuple[Callable, (Abstraction | Implication | Type)]]


def collect_parameters(
    specification: Abstraction | Implication | Type,
) -> list[Parameter]:
    if isinstance(specification, Type):
        return []
    if isinstance(specification, Abstraction):
        return [*specification.parameter, *collect_parameters(specification.body)]
    if isinstance(specification, Implication):
        return collect_parameters(specification.body)
    raise TypeError


ConstructorName: TypeAlias = str


@dataclass
class SpecInfo:
    groups: set[Group]
    parameters: deque[Parameter]
    constructors: set[ConstructorName]


def collect_constructors(typ: Type) -> set[ConstructorName]:
    if isinstance(typ, Constructor):
        return {typ.name}
    if isinstance(typ, Arrow):
        return collect_constructors(typ.source).union(collect_constructors(typ.target))
    if isinstance(typ, Intersection):
        return collect_constructors(typ.left).union(collect_constructors(typ.right))
    if isinstance(typ, Omega):
        return set()
    msg = f"type {typ} should not appear here"
    raise TypeError(msg)


def inspect_spec(specification: Abstraction | Implication | Type) -> SpecInfo:
    if isinstance(specification, Type):
        return SpecInfo(
            groups=set(),
            parameters=deque(),
            constructors=collect_constructors(specification),
        )
    if isinstance(specification, Abstraction):
        collection = inspect_spec(specification.body)
        param = specification.parameter
        collection.parameters.appendleft(param)
        if isinstance(param, LiteralParameter):
            # TODO: Collect information about the possible values of the group
            collection.groups.add(param.group)
        elif isinstance(param, TermParameter):
            collection.constructors.update(collect_constructors(param.group))
        return collection
    if isinstance(specification, Implication):
        return inspect_spec(specification.body)
    raise TypeError


def inspect_specifications(
    component_specifications: ComponentSpecifications[T],
) -> dict[T, SpecInfo]:
    return {name: inspect_spec(spec) for name, (interpretation, spec) in component_specifications.items()}


def tree_to_dict(tree: Tree[T], component_specifications: ComponentSpecifications) -> dict:
    spec_info = inspect_specifications(component_specifications)
    all_constructors: set[ConstructorName] = set.union(*[i.constructors for i in spec_info.values()])
    color_map: dict[ConstructorName, Colour]
    if len(all_constructors) <= len(KELLY_COLOURS):
        color_map = dict(zip(all_constructors, KELLY_COLOURS, strict=False))
    else:
        # more than len(KELLY_COLOURS) different constructors. Defaulting them all to white
        color_map = dict.fromkeys(all_constructors, "#000000")
    result = {}
    stack: list[tuple[Tree[T], dict, Parameter | None, T | None]] = [(tree, result, None, None)]
    while len(stack) > 0:
        current_tree, the_dict, param, parent = stack.pop()
        parameters: list[Parameter] | None
        colors: list[Colour]
        interpretations = {name: interpretation for name, (interpretation, spec) in component_specifications.items()}
        root_is_combinator = current_tree.root in component_specifications
        if root_is_combinator:
            _interpretation, specification = component_specifications[current_tree.root]
            things = inspect_spec(specification)
            parameters: deque[Parameter] = things.parameters
            colors = [color_map[c] for c in things.constructors]
        else:
            parameters = None
            colors = ["#000000"]
        name = f"{param}: " if param is not None else ""
        combinator: str | None = current_tree.root.__name__ if callable(current_tree.root) else str(current_tree.root)
        if combinator is not None:
            name += combinator

        the_dict["parent"] = "" if parent is None else str(parent)
        the_dict["val"] = current_tree.interpret(interpretation=interpretations)
        the_dict["parameter"] = "" if param is None else str(param)
        the_dict["combinator"] = "" if (combinator is None or not root_is_combinator) else combinator
        the_dict["colors"] = colors
        the_dict["is_combinator"] = root_is_combinator
        children = []
        the_dict["children"] = children
        for i, c in enumerate(current_tree.children):
            child_dict = {}
            children.append(child_dict)
            stack.append((c, child_dict, (parameters[i] if parameters is not None else None), tree.root))
    return {
        "tree": result,
        "color_map": color_map,
    }


class MyServer(threading.Thread):
    def run(self):
        self.server = ThreadingHTTPServer(("localhost", 8000), SimpleHTTPRequestHandler)
        self.server.serve_forever()

    def stop(self):
        self.server.shutdown()


def visualize(
    amount: int,
    trees: Iterable[Tree[T]],
    named_components_with_specifications: Sequence[tuple[T, Callable, Abstraction | Implication | Type]],
):
    visualization_file_path = pathlib.Path(__file__).parent / "visualization/results.json"
    with open(visualization_file_path, "w", encoding="utf-8") as visualization_file:
        visualization_file.write(
            json.dumps(
                [
                    tree_to_dict(
                        tree,
                        component_specifications={n: (i, s) for n, i, s in named_components_with_specifications},
                    )
                    for tree in itertools.islice(trees, amount + 1)
                ]
            )
        )
    os.chdir(visualization_file_path.parent)
    server = MyServer()
    server.start()
    print(
        'Visualization server started. Please open "http://localhost:8000/collapsible_tree.html" to see the visualization.'
    )  # noqa: T201
    # webbrowser.open("http://localhost:8000/collapsible_tree.html", new=0, autoraise=True)
    input("Press enter to continue...")
    server.stop()
