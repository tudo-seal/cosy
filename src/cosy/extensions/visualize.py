import json
import os
import pathlib
import threading
import webbrowser
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
    "#F2F3F4",
    "#222222",
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


def tree_to_dict(tree: Tree[T], component_specifications: ComponentSpecifications):
    spec_info = inspect_specifications(component_specifications)
    all_constructors: set[ConstructorName] = set.union(*[i.constructors for i in spec_info.values()])
    color_map: dict[ConstructorName, Colour]
    if len(all_constructors) <= len(KELLY_COLOURS):
        color_map = dict(zip(all_constructors, KELLY_COLOURS, strict=False))
    else:
        # print(
        #     f"Encountered more than {len(KELLY_COLOURS)} different constructors. Defaulting them all to white!"
        # )
        color_map = dict.fromkeys(all_constructors, "#000000")

    def rec_tree_to_dict(
        tree: Tree[T],
        component_specifications: ComponentSpecifications[T],
        param: Parameter | None = None,
        parent: T | None = None,
    ) -> dict:
        parameters: list[Parameter] | None
        colors: list[Colour]
        interpretations = {name: interpretation for name, (interpretation, spec) in component_specifications.items()}
        if tree.root in component_specifications:
            _interpretation, specification = component_specifications[tree.root]
            things = inspect_spec(specification)
            parameters: deque[Parameter] = things.parameters
            colors = [color_map[c] for c in things.constructors]
        else:
            parameters = None
            colors = ["#000000"]
        name = f"{param}: " if param is not None else ""
        combinator: str | None = tree.root.__name__ if callable(tree.root) else str(tree.root)
        if combinator is not None:
            name += combinator
        children = [
            rec_tree_to_dict(
                c,
                component_specifications,
                parent=tree.root,
                param=parameters[i] if parameters is not None else None,
            )
            for i, c in enumerate(tree.children)
        ]
        return {
            "parent": "null" if parent is None else str(parent),
            "val": tree.interpret(interpretation=interpretations),
            "parameter": "null" if param is None else str(param),
            "combinator": "" if combinator is None else combinator,
            "edge_name": "null",  # "null" if parent is None else tree.interpret(),
            "children": children,
            "colors": colors,
        }

    return rec_tree_to_dict(tree, component_specifications)


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
        visualization_file.write("{\n")
        for i, tree in enumerate(trees):
            if i >= amount:
                break
            tree_dict = tree_to_dict(
                tree,
                component_specifications={n: (i, s) for n, i, s in named_components_with_specifications},
            )
            prefix = ",\n" if i != 0 else ""
            visualization_file.write(f'{prefix}"{i}": {json.dumps(tree_dict, indent=2, default=str)}')
        visualization_file.write("}")
    os.chdir(visualization_file_path.parent)
    server = MyServer()
    server.start()
    webbrowser.open("http://localhost:8000/collapsible_tree.html", new=0, autoraise=True)
    input("Press enter to exit...")
    server.stop()

    # httpd = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
    # httpd.serve_forever()
    # input("Press any key to exit...")
    # httpd.shutdown()
