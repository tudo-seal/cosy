# pylint: disable=invalid-name
"""
This module provides a `SpecificationBuilder` class, which allows users to define specifications
in a declarative manner using a fluent interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from cosy.synthesizer import Specification

from typing import Any

from cosy.types import (
    Abstraction,
    Group,
    Implication,
    LiteralParameter,
    Predicate,
    TermParameter,
    Type,
)


class SpecificationBuilder:
    """
    A domain-specific language (SpecificationBuilder) to define component specifications.

    This class provides a interface for defining specifications in a declarative manner. It allows
    users to specify the name and group of each parameter, as well as filter.

    Examples:
        SpecificationBuilder()
            .Parameter("x", int)
            .Parameter("y", int, lambda vars: vars["x"] + 1)
            .Parameter("z", str)
            .ParameterConstraint(lambda vars: len(vars["z"]) == vars["x"] + vars["y"])
            .Suffix(<Type using Var("x"), Var("y") and Var("z")>)

        constructs a specification for a function with three parameters:
        - `x`: an integer
        - `y`: an integer, the value of which is `x` + 1
        - `z`: a string, whose length is equal to `x` + `y`
        The `Suffix` method specifies the function, which uses the variables `x`, `y`, and `z`.
    """

    def __init__(self) -> None:
        """
        Initialize the SpecificationBuilder object
        """

        self._result: Callable[[Specification], Specification] = lambda suffix: suffix

    def parameter(
        self,
        name: str,
        group: Group,
        candidates: Callable[[dict[str, Any]], Sequence[Any]] | None = None,
    ) -> SpecificationBuilder:
        """
        Introduce a new parameter variable.

        `group` is a Group, and an instance of this specification will be generated
        for each valid literal in the corresponding literal group.
        You can use this variable as Var(name) in all `Type`s, after the introduction
        and in all predicates.
        Optionally, you can specify a sequence of candidate values, that will be used to generate
        the literals. This sequence is parameterized by the values of previously
        defined literal variables. This is useful, if you want to restrict the values of a variable
        to a subset of the values in the corresponding literal group.

        :param name: The name of the new variable.
        :type name: str
        :param group: The group of the variable.
        :type group: Group
        :param candidates: Parameterized sequence of candidate values, that will be used to generate the literals.
        :type candidates: Callable[[dict[str, Any]], Sequence[Any]] | None
        :return: The SpecificationBuilder object.
        :rtype: SpecificationBuilder
        """

        def new_result(suffix: Specification, result=self._result) -> Specification:
            return result(Abstraction(LiteralParameter(name, group, candidates), suffix))

        self._result = new_result
        return self

    def argument(self, name: str, specification: Type) -> SpecificationBuilder:
        """
        Introduce a new variable.

        `group` is a `Type`, and an instance will be generated for each tree, satisfying
        the specification given by the type. Since this can only be done in the enumeration step,
        you can only use these variables in predicates, that themselves belong to variables whose `group` is a `Type`.

        :param name: The name of the new variable.
        :type name: str
        :param specification: The type of the variable.
        :type specification: Type
        :return: The SpecificationBuilder object.
        :rtype: SpecificationBuilder
        """

        def new_result(suffix: Specification, result=self._result) -> Specification:
            return result(Abstraction(TermParameter(name, specification), suffix))

        self._result = new_result
        return self

    def parameter_constraint(self, constraint: Callable[[Mapping[str, Any]], bool]) -> SpecificationBuilder:
        """
        Constraint on the previously defined parameter variables.

        :param constraint: A constraint deciding, if the currently chosen parameter values are valid.
            The values of variables are passed by a dictionary, where the keys are the names of the
            parameter variables and the values are the corresponding values.
        :type constraint: Callable[[Mapping[str, Any]], bool]
        :return: The SpecificationBuilder object.
        :rtype: SpecificationBuilder
        """

        def new_result(suffix: Specification, result=self._result) -> Specification:
            return result(Implication(Predicate(constraint, True), suffix))

        self._result = new_result
        return self

    def constraint(self, constraint: Callable[[Mapping[str, Any]], bool]) -> SpecificationBuilder:
        """
        Constraint on the previously defined parameter variables and argument variables.

        :param constraint: A constraint deciding, if the currently chosen values are valid.
            The values of variables are passed by a dictionary, where the keys are the names of the
            variables and the values are the corresponding values.
        :type constraint: Callable[[Mapping[str, Any]], bool]
        :return: The SpecificationBuilder object.
        :rtype: SpecificationBuilder
        """

        def new_result(suffix: Specification, result=self._result) -> Specification:
            return result(Implication(Predicate(constraint, False), suffix))

        self._result = new_result
        return self

    def suffix(self, suffix: Type) -> Specification:
        """
        Constructs the final specification wrapping the given `Type` `suffix`.

        :param suffix: The wrapped type.
        :type suffix: Type
        :return: The constructed specification.
        :rtype: Abstraction | Type
        """
        return self._result(suffix)
