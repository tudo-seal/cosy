from collections import deque
from typing import Sequence, TypeVar, Hashable, Callable, Any

from cosy.core import Constructor
from cosy.core.synthesizer import Specification
from cosy.core.types import Abstraction, Implication, Type, Var, LiteralParameter, Parameter, TermParameter, Group, \
    Predicate

T = TypeVar("T", bound=Hashable)

class AllGroup(Group):
    def __init__(self) -> None:
        self.name = "All-Group"

    def __iter__(self):
        raise NotImplementedError("Cannot iterate over the all group!")

    def __contains__(self, x: Any) -> bool:
        return True

ALL_GROUP = AllGroup()

class Hashabledict(dict):
    def __hash__(self):
        return hash(frozenset(self.items()))


def debug_note_repository(
        named_components_with_specifications: Sequence[tuple[T, Callable, Specification]]
) -> Sequence[tuple[T, Callable, Specification]]:
    result: list[tuple[T, Callable, Specification]] = []

    def modifiy_specification(combinator: T, call_map: list[bool], spec: Specification) -> Specification:
        if isinstance(spec, Abstraction):
            if isinstance(spec.parameter, LiteralParameter):
                call_map.append(True)
                return Abstraction(spec.parameter, modifiy_specification(combinator, call_map, spec.body))
            elif isinstance(spec.parameter, TermParameter):
                unpacking_param_name = "_DEBUG_argument_" + spec.parameter.name
                unpacking_param = LiteralParameter(
                    unpacking_param_name, ALL_GROUP,
                    lambda m: [None] if m["_DEBUG_arguments"] is None else [m["_DEBUG_arguments"][1][spec.parameter.name]])
                modified_param = TermParameter(spec.parameter.name,
                                               spec.parameter.group & Constructor("_DEBUG_Args", Var(unpacking_param_name)))
                call_map.append(False)
                call_map.append(True)
                return Abstraction(unpacking_param, Abstraction(modified_param, modifiy_specification(combinator, call_map, spec.body)))
            else:
                raise RuntimeError("Impossible case")
        elif isinstance(spec, Implication):
            return Specification(spec.predicate, modifiy_specification(combinator, call_map, spec.body))
        elif isinstance(spec, Type):
            return spec & Constructor("_DEBUG_Args", Var("_DEBUG_arguments"))
        else:
            raise RuntimeError("Impossible case")
    for combinator, interpretation, specification in named_components_with_specifications:
        call_map: list[bool] = [False]

        modified_spec = Abstraction(
            LiteralParameter("_DEBUG_arguments", ALL_GROUP),
            Implication(
                Predicate(
                    constraint=lambda m, c=combinator: m["_DEBUG_arguments"] is None or m["_DEBUG_arguments"][0] == "_DEBUG_argument_" + str(c),
                    only_literals=False
                ),
                modifiy_specification(combinator, call_map, specification)
            )
        )

        def modified_interpretation(*args,
                                    the_call_map: tuple[bool, ...] = tuple(call_map),
                                    the_interpretation = interpretation,
                                    ):
            assert len(args) == len(the_call_map)
            args_to_pass: list = [arg for arg, keep in zip(args, the_call_map) if keep]

            return the_interpretation(*args_to_pass)
        result.append((combinator, modified_interpretation, modified_spec))
    return result