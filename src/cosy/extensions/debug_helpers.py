from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, Union

from cosy.core import Constructor
from cosy.core.synthesizer import Specification
from cosy.core.types import Abstraction, Group, Implication, LiteralParameter, Predicate, TermParameter, Type, Var

T = TypeVar("T", bound=Hashable)

class AllGroup(Group):
    def __init__(self) -> None:
        self.name = "All-Group"

    def __iter__(self):
        raise NotImplementedError("Cannot iterate over the all group!")

    def __contains__(self, x: Any) -> bool:
        return True

ALL_GROUP = AllGroup()


def debug_value_name(argument: str) -> str:
    return "_DEBUG_values_" + argument

DEBUG_VALUES_ARGUMENT = "_DEBUG_values_args"
DEBUG_VALUES_CONSTRUCTOR = "_DEBUG_values_cons"


class HashableDefaultDict(defaultdict):
    def __hash__(self):
        return hash(frozenset(self.items()))

@dataclass(frozen=True)
class PartialTerm:
    combinator: str
    params_and_named_args: dict[str, Union[None, Any, 'PartialTerm']]
    unnamed_arguments: tuple['PartialTerm',...]

def partial_term_builder(
        combinator: str,
        *unnamed_arguments: 'PartialTerm',
        **params_and_named_args: Union[Any, None, 'PartialTerm']
):
    return PartialTerm(
        combinator=debug_value_name(combinator),
        unnamed_arguments=unnamed_arguments,
        params_and_named_args=HashableDefaultDict(lambda: None, params_and_named_args),
    )

class PartialTermBuilder:
    def __init__(self, combinator: str):
        self.combinator = combinator



def debug_note_repository(
        named_components_with_specifications: Sequence[tuple[T, Callable, Specification]]
) -> Sequence[tuple[T, Callable, Specification]]:
    result: list[tuple[T, Callable, Specification]] = []

    def modifiy_specification(combinator: T, call_map: list[bool], spec: Specification) -> Specification:
        if isinstance(spec, Abstraction):
            if isinstance(spec.parameter, LiteralParameter):
                values = spec.parameter.values
                modified_values: Callable[[dict[str, Any]], Sequence[Any]] | None
                if values is None:
                    modified_values = None
                else:
                    def modified_values(m: dict[str, PartialTerm]) -> Sequence[Any]:
                        partial_term = m[DEBUG_VALUES_ARGUMENT]
                        if partial_term is None:
                            return values(m)
                        forced_values = partial_term.params_and_named_args[spec.parameter.name]
                        forced_value_sequence: Sequence[Any]
                        if not isinstance(forced_values, Sequence) or isinstance(forced_values, str):
                            forced_value_sequence = [forced_values]
                        else:
                            forced_value_sequence = forced_values
                        if values is not None and not all(v in values for v in forced_value_sequence):
                            error_msg = (
                                f"Miss-formed partial term: {partial_term}\n"
                                f"The specified values for the parameter {spec.parameter.name} are not all legal"
                                f"values for this parameter. The following values are not legal candidates:\n"
                                f"{[v for v in forced_value_sequence if v not in values]}\n"
                                f"In the context: {m}"
                            )
                            raise ValueError(error_msg)
                        return forced_value_sequence

                modified_param = LiteralParameter(
                    name=spec.parameter.name,
                    group=spec.parameter.group,
                    values=modified_values
                )
                call_map.append(True)
                return Abstraction(modified_param, modifiy_specification(combinator, call_map, spec.body))
            if isinstance(spec.parameter, TermParameter):
                unpacking_param_name = debug_value_name(spec.parameter.name)
                def unpack(m: dict[str, PartialTerm]) -> Sequence[Any]:
                    partial_term = m[DEBUG_VALUES_ARGUMENT]
                    if partial_term is None:
                        return [None]
                    subterm = partial_term.params_and_named_args[spec.parameter.name]
                    if subterm is None or isinstance(subterm, PartialTerm):
                        return [subterm]
                    error_msg = (
                        f"Miss-formed partial term: {partial_term}\n"
                        f"For combinator {partial_term.combinator}, {spec.parameter.name} is the name of"
                        f"an argument, but the assigned value to this argument ({subterm}) is not a PartialTerm"
                    )
                    raise ValueError(error_msg)

                unpacking_param = LiteralParameter(
                    unpacking_param_name, ALL_GROUP,
                    unpack
                )
                modified_param = TermParameter(spec.parameter.name,
                                               spec.parameter.group & Constructor(DEBUG_VALUES_CONSTRUCTOR, Var(unpacking_param_name)))
                call_map.append(False)
                call_map.append(True)
                return Abstraction(unpacking_param, Abstraction(modified_param, modifiy_specification(combinator, call_map, spec.body)))
            raise RuntimeError("Impossible case")
        if isinstance(spec, Implication):
            return Specification(spec.predicate, modifiy_specification(combinator, call_map, spec.body))
        if isinstance(spec, Type):
            return spec & Constructor(DEBUG_VALUES_CONSTRUCTOR, Var(DEBUG_VALUES_ARGUMENT))
        raise RuntimeError("Impossible case")

    for combinator, interpretation, specification in named_components_with_specifications:
        call_map: list[bool] = [False]

        modified_spec = Abstraction(
            LiteralParameter(DEBUG_VALUES_ARGUMENT, ALL_GROUP),
            Implication(
                Predicate(
                    constraint=lambda m, c=combinator: m[DEBUG_VALUES_ARGUMENT] is None or m[DEBUG_VALUES_ARGUMENT].combinator == debug_value_name(str(c)),
                    only_literals=True
                ),
                modifiy_specification(combinator, call_map, specification)
            )
        )

        def modified_interpretation(*args,
                                    the_call_map: tuple[bool, ...] = tuple(call_map),
                                    the_interpretation = interpretation,
                                    ):
            assert len(args) == len(the_call_map)
            args_to_pass: list = [arg for arg, keep in zip(args, the_call_map, strict=False) if keep]

            return the_interpretation(*args_to_pass)
        result.append((combinator, modified_interpretation, modified_spec))
    return result
