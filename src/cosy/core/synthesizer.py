"""Synthesizer implementing Finite Combinatory Logic with Predicates.
It constructs a logic program via `constructSolutionSpace` from the following ingredients:
- collection of component specifications
- parameter space
- optional specification taxonomy
- target specification"""

from collections import deque
from collections.abc import (
    Callable,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    TypeVar,
)

from cosy.core.combinatorics import maximal_elements, minimal_covers
from cosy.core.solution_space import (
    Argument,
    ConstantArgument,
    NonTerminalArgument,
    RHSRule,
    SolutionSpace,
)
from cosy.core.subtypes import Subtypes, Taxonomy
from cosy.core.types import (
    Abstraction,
    Arrow,
    Group,
    Implication,
    Intersection,
    LiteralParameter,
    Parameter,
    Predicate,
    TermParameter,
    Type,
)

# type of components
C = TypeVar("C", bound=Hashable)

# type of component specifications
Specification = Abstraction | Implication | Type


@dataclass(frozen=True)
class MultiArrow:
    # type of shape arg1 -> arg2 -> ... -> argN -> target
    args: tuple[Type, ...]
    target: Type

    def __str__(self) -> str:
        if len(self.args) > 0:
            return f"{[str(a) for a in self.args]} -> {self.target!s}"
        return str(self.target)


@dataclass()
class SpecificationInfo:
    # container for auxiliary information about a specification
    prefix: list[LiteralParameter | TermParameter | Predicate]
    term_predicates: tuple[Callable[[dict[str, Any]], bool], ...]
    type: list[list[MultiArrow]]


class Synthesizer(Generic[C]):
    def __init__(
        self,
        component_specifications: Mapping[C, Specification],
        taxonomy: Taxonomy | None = None,
    ):
        self.repository: tuple[tuple[C, SpecificationInfo], ...] = tuple(
            (c, Synthesizer._function_types(c, ty)) for c, ty in component_specifications.items()
        )
        self.subtypes = Subtypes(taxonomy if taxonomy is not None else {})

    @staticmethod
    def _function_types(
        combinator: C,
        parameterized_type: Specification,
    ) -> SpecificationInfo:
        """Presents a type as a list of 0-ary, 1-ary, ..., n-ary function types."""

        def unary_function_types(ty: Type) -> Iterable[tuple[Type, Type]]:
            tys: deque[Type] = deque((ty,))
            while tys:
                match tys.pop():
                    case Arrow(src, tgt) if tgt.organized:
                        yield (src, tgt)
                    case Intersection(sigma, tau):
                        tys.extend((sigma, tau))

        prefix: list[LiteralParameter | TermParameter | Predicate] = []
        variables: set[str] = set()
        literal_variables: set[str] = set()
        while not isinstance(parameterized_type, Type):
            if isinstance(parameterized_type, Abstraction):
                param = parameterized_type.parameter
                if param.name in variables:
                    # check if parameter names are unique
                    msg = f"Duplicate name {param.name} in specification of combinator {combinator!s}."
                    raise ValueError(msg)
                variables.add(param.name)
                if isinstance(param, LiteralParameter):
                    prefix.append(param)
                    literal_variables.add(param.name)
                    if not isinstance(param.group.name, str):
                        msg = f"Group of literal parameter {param.name} in specification of combinator {combinator!s} has no name."
                        raise TypeError(msg)
                elif isinstance(param, TermParameter):
                    prefix.append(param)
                    for free_var in param.group.free_vars:
                        if free_var not in literal_variables:
                            # check if each parameter variable is abstracted
                            msg = (
                                f"Parameter {free_var} is not abstracted in specification of combinator {combinator!s}."
                            )
                            raise ValueError(msg)
                parameterized_type = parameterized_type.body
            elif isinstance(parameterized_type, Implication):
                prefix.append(parameterized_type.predicate)
                parameterized_type = parameterized_type.body
            else:
                msg = (
                    f"Specification of combinator {combinator!s} is neither an Abstraction, an Implication, nor a Type."
                )
                raise TypeError(msg)

        for free_var in parameterized_type.free_vars:
            if free_var not in literal_variables:
                # check if each parameter variable is abstracted
                msg = f"Parameter {free_var} is not abstracted in specification of combinator {combinator!s}."
                raise ValueError(msg)

        current: list[MultiArrow] = [MultiArrow((), parameterized_type)]

        multiarrows = []
        while len(current) != 0:
            multiarrows.append(current)
            current = [
                MultiArrow((*c.args, new_arg), new_tgt)
                for c in current
                for (new_arg, new_tgt) in unary_function_types(c.target)
            ]

        term_predicates: tuple[Callable[[dict[str, Any]], bool], ...] = tuple(
            p.constraint for p in prefix if isinstance(p, Predicate) and not p.only_literals
        )
        return SpecificationInfo(prefix, term_predicates, multiarrows)

    def _enumerate_substitutions(
        self,
        prefix: list[LiteralParameter | TermParameter | Predicate],
        substitution: dict[str, Any],
    ) -> Iterable[dict[str, Any]]:
        """Enumerate all substitutions for the given parameters fairly.
        Take initial_substitution with inferred literals into account."""

        stack: deque[tuple[dict[str, Any], int, Iterator[Any] | None]] = deque([(substitution, 0, None)])

        while stack:
            substitution, index, generator = stack.pop()
            if index >= len(prefix):
                # no more parameters to process
                yield substitution
                continue
            parameter = prefix[index]
            if isinstance(parameter, LiteralParameter):
                if generator is None:
                    if parameter.name in substitution:
                        value = substitution[parameter.name]
                        if parameter.values is not None and value not in parameter.values(substitution):
                            # the inferred value is not in the set of values
                            continue
                        if value not in parameter.group:
                            # the inferred value is not in the group
                            continue
                        stack.appendleft((substitution, index + 1, None))
                    elif parameter.values is not None:
                        stack.appendleft((substitution, index, iter(parameter.values(substitution))))
                    else:
                        try:
                            # consider all individual values of a group
                            stack.appendleft((substitution, index, iter(parameter.group)))
                        except TypeError as e:
                            msg = f"Group {parameter.group.name} is not iterable."
                            raise ValueError(msg) from e
                else:
                    try:
                        value = next(generator)
                    except StopIteration:
                        continue
                    if value in parameter.group:
                        stack.appendleft(({**substitution, parameter.name: value}, index + 1, None))
                    stack.appendleft((substitution, index, generator))

            elif isinstance(parameter, Predicate) and parameter.only_literals:
                if parameter.constraint(substitution):
                    # the predicate is satisfied
                    stack.appendleft((substitution, index + 1, None))
            else:
                stack.appendleft((substitution, index + 1, None))

    def _subqueries(
        self,
        nary_types: list[MultiArrow],
        paths: Iterable[Type],
        substitution: dict[str, Any],
    ) -> Sequence[list[Type]]:
        # does the target of a multi-arrow contain a given type?
        def target_contains(m: MultiArrow, t: Type) -> bool:
            return self.subtypes.check_subtype(m.target, t, substitution)

        # cover target using targets of multi-arrows in nary_types
        covers = minimal_covers(nary_types, paths, target_contains)
        if len(covers) == 0:
            return []

        # intersect arguments of multi-arrows at same positions
        def intersect_args(arg_tuples: Iterable[tuple[Type, ...]]) -> list[Type]:
            return [Type.intersect(arg_tuple) for arg_tuple in zip(*arg_tuples, strict=True)]

        intersected_args: Generator[list[Type]] = (intersect_args(m.args for m in ms) for ms in covers)

        # consider only maximal argument vectors
        def compare_args(args1, args2) -> bool:
            return all(
                map(
                    lambda a, b: self.subtypes.check_subtype(a, b, substitution),
                    args1,
                    args2,
                )
            )

        return maximal_elements(intersected_args, compare_args)

    def _necessary_substitution(
        self,
        paths: Iterable[Type],
        combinator_type: list[list[MultiArrow]],
    ) -> dict[str, Any] | None:
        """
        Computes a substitution that needs to be part of every substitution S such that
        S(combinator_type) <= paths.

        If no substitution can make this valid, None is returned.
        """

        result: dict[str, Any] = {}

        for path in paths:
            unique_substitution: dict[str, Any] | None = None
            is_unique = True

            for nary_types in combinator_type:
                for ty in nary_types:
                    substitution = self.subtypes.infer_substitution(ty.target, path)
                    if substitution is None:
                        continue
                    if unique_substitution is None:
                        unique_substitution = substitution
                    else:
                        unique_substitution = Subtypes.glb_substitutions(unique_substitution, substitution)
                        if unique_substitution is None:
                            is_unique = False
                            break
                        if not is_unique:
                            break
                if not is_unique:
                    break

            if not is_unique:
                continue  # substitution not unique substitution - skip
            if unique_substitution is None:
                return None  # no substitution for this path

            # merge consistent substitution
            unique_substitution = Subtypes.lub_substitutions(result, unique_substitution)
            if unique_substitution is None:
                return None  # conflict in necessary substitution
            result = unique_substitution

        return result

    def construct_solution_space_rules(self, *targets: Type) -> Generator[tuple[Type, RHSRule]]:
        """Generate logic program rules for the given target types."""

        # current target types
        stack: deque[tuple[Type, tuple[C, SpecificationInfo, Iterator] | None]] = deque(
            (target, None) for target in targets
        )
        seen: set[Type] = set()

        while stack:
            current_target, current_target_info = stack.pop()
            # if the target is omega, then the result is junk
            if not current_target.organized:
                msg = f"Target type {current_target} is omega."
                raise ValueError(msg)

            # target type was not initialized before
            if current_target not in seen or current_target_info is not None:
                if current_target_info is None:
                    seen.add(current_target)
                    # try each combinator
                    for combinator, specification_info in self.repository:
                        # Compute necessary substitutions
                        substitution = self._necessary_substitution(
                            current_target.organized,
                            specification_info.type,
                        )

                        # If there cannot be a suitable substitution, ignore this combinator
                        if substitution is None:
                            continue

                        # Keep necessary substitutions and enumerate the rest
                        selected_instantiations = self._enumerate_substitutions(specification_info.prefix, substitution)
                        stack.appendleft(
                            (
                                current_target,
                                (
                                    combinator,
                                    specification_info,
                                    iter(selected_instantiations),
                                ),
                            )
                        )
                else:
                    combinator, specification_info, selected_instantiations = current_target_info
                    instantiation = next(selected_instantiations, None)
                    if instantiation is not None:
                        stack.appendleft((current_target, current_target_info))
                        named_arguments: tuple[Argument, ...] | None = None

                        # and every arity of the combinator type
                        for nary_types in specification_info.type:
                            for subquery in self._subqueries(
                                nary_types,
                                current_target.organized,
                                instantiation,
                            ):
                                if named_arguments is None:  # do this only once for each instantiation
                                    named_arguments = tuple(
                                        ConstantArgument(
                                            param.name,
                                            instantiation[param.name],
                                            param.group,
                                        )
                                        if isinstance(param, LiteralParameter)
                                        else NonTerminalArgument(
                                            param.name,
                                            param.group.subst(instantiation),
                                        )
                                        for param in specification_info.prefix
                                        if isinstance(param, Parameter)
                                    )
                                    stack.extendleft(
                                        (argument.origin, None)
                                        for argument in named_arguments
                                        if isinstance(argument, NonTerminalArgument)
                                    )

                                anonymous_arguments: tuple[NonTerminalArgument, ...] = tuple(
                                    NonTerminalArgument(
                                        None,
                                        ty.subst(instantiation),
                                    )
                                    for ty in subquery
                                )
                                yield (
                                    current_target,
                                    RHSRule[Type, Any, Group](
                                        (*named_arguments, *anonymous_arguments),
                                        specification_info.term_predicates,
                                        combinator,
                                    ),
                                )
                                stack.extendleft((q.origin, None) for q in anonymous_arguments)

    def construct_solution_space(self, *targets: Type) -> SolutionSpace[Type, C, Group]:
        """Constructs a logic program in the current environment for the given target types."""

        solution_space: SolutionSpace[Type, C, Group] = SolutionSpace()
        for nt, rule in self.construct_solution_space_rules(*targets):
            solution_space.add_rule(nt, rule.terminal, rule.arguments, rule.predicates)

        return solution_space
