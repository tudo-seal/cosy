"""Auxiliary functions for inspecting the specification.
The method `inspect` analyses the given  component specifications, parameter space, and taxonomy,
and provides info, warnings, and reports errors if any.
"""

import logging
from collections import deque
from collections.abc import Hashable, Mapping
from itertools import chain
from typing import TypeVar

from cosy.synthesizer import Specification, Taxonomy
from cosy.types import (
    Abstraction,
    Arrow,
    Constructor,
    Group,
    Implication,
    Intersection,
    LiteralParameter,
    TermParameter,
    Type,
)

# type of components
C = TypeVar("C", bound=Hashable)


class Inspector:
    """Inspector class for analyzing component specifications, parameter space, and taxonomy."""

    _logger: logging.Logger

    def __init__(self, logger=None):
        if logger is None:
            self._logger = logging.getLogger(__name__)
            self._logger.setLevel(logging.DEBUG)
            # handler = logging.StreamHandler()
            # handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
            # self._logger.addHandler(handler)
        else:
            self._logger = logger

    @staticmethod
    def _constructors(ty: Type) -> set[str]:
        """
        Get the constructors of a type.
        """
        constructors = set()
        stack: deque[Type] = deque([ty])
        while stack:
            match stack.pop():
                case Intersection(l, r):
                    stack.extend((l, r))
                case Arrow(src, tgt):
                    stack.extend((src, tgt))
                case Constructor(name, arg):
                    constructors.add(name)
                    stack.append(arg)

        return constructors

    def inspect(
        self,
        component_specifications: Mapping[C, Specification],
        taxonomy: Taxonomy | None = None,
    ):
        """
        Inspect the component specifications, parameter space, and taxonomy.
        A `ValueError` is raised if the specifications are not well-formed, which includes:
        - a component has two parameters/arguments with the same name (shadowing)
        - a parameter name is used in the specification of a component but not abstracted via a parameter
        - a group is used in a parameter but not defined in the parameter space

        An info is logged if:
        - a name is bound to different groups in different components
        - a parameter is abstracted but not used in the specification (caveat: constraints cannot be checked)
        - a group is not used in any component
        - a concept is used only in one component
        - a concept in the taxonomy is not used in any component
        """

        if taxonomy is None:
            taxonomy = {}

        all_groups: set[tuple[str, Group | Type]] = set()
        all_constructors: list[set[str]] = []

        for specification in component_specifications.values():
            prefix: list[LiteralParameter | TermParameter] = []
            # mapping from variable names to groups
            groups: dict[str, Group | Type] = {}
            # set of parameter names occurring in bodies
            parameter_names: set[str] = set()
            parameterized_type = specification
            # set of constructors occurring in the specification
            constructors: set[str] = set()

            while not isinstance(parameterized_type, Type):
                if isinstance(parameterized_type, Abstraction):
                    param = parameterized_type.parameter
                    if isinstance(param, LiteralParameter | TermParameter):
                        prefix.append(param)
                        if param.name in groups:
                            # check if parameter names are unique
                            msg = f"Duplicate name: {param.name}"
                            raise ValueError(msg)
                        groups[param.name] = param.group
                        for n, g in all_groups:
                            if n == param.name and g != param.group:
                                self._logger.info(
                                    "%s is used both as %s and %s",
                                    param.name,
                                    param.group,
                                    g,
                                )
                        all_groups.add((param.name, param.group))
                    if isinstance(param, TermParameter):
                        parameter_names.update(param.group.free_vars)
                        constructors.update(Inspector._constructors(param.group))
                    parameterized_type = parameterized_type.body
                elif isinstance(parameterized_type, Implication):
                    parameterized_type = parameterized_type.body

            parameter_names.update(parameterized_type.free_vars)
            constructors.update(Inspector._constructors(parameterized_type))
            all_constructors.append(constructors)
            # check if every variable in the body is abstracted
            for var in parameter_names:
                if var not in groups:
                    msg = f"Variable {var} is not abstracted via a parameter"
                    raise ValueError(msg)

            # check if every abstracted variable is used
            for var, group in groups.items():
                if isinstance(group, str) and var not in parameter_names:
                    self._logger.info("Variable %s is abstracted via a parameter but not used", var)

        # check is some constructor is used only in one component
        for constructors in all_constructors:
            for constructor in constructors:
                if sum(1 for cs in all_constructors if constructor in cs) == 1:
                    self._logger.info("Concept %s is used in only one component", constructor)

        # check if every concept in the taxonomy is used
        for name, subtypes in taxonomy.items():
            for c in chain([name], subtypes):
                if c not in set.union(*all_constructors):
                    self._logger.info("Concept %s is not used in any component", c)

        # further ideas:
        # check if each parameter of a non-iterable group without candidate values is in the codomain
