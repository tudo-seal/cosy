"""
This module provides a `Subtypes` class, which is used to check subtyping relationships
between types in the intersection type system.
"""

from collections import deque
from collections.abc import Mapping
from typing import Any

from cosy.types import Arrow, Constructor, Intersection, Literal, Type, Var

# a mapping from a concept to the set it its subconcepts
Taxonomy = Mapping[str, set[str]]


class Subtypes:
    def __init__(self, taxonomy: Taxonomy):
        self.taxonomy = self._transitive_closure(self._reflexive_closure(taxonomy))

    def _check_subtype_rec(
        self,
        subtypes: deque[Type],
        supertype: Type,
        substitutions: Mapping[str, Literal],
    ) -> bool:
        if not supertype.organized:
            return True
        match supertype:
            case Literal(value2):
                while subtypes:
                    match subtypes.pop():
                        case Literal(value1):
                            if value2 == value1:
                                return True
                        case Var(name1):
                            if substitutions[name1] == supertype.value:
                                return True
                        case Intersection(l, r):
                            subtypes.extend((l, r))
                return False
            case Constructor(name2, arg2):
                casted_constr: deque[Type] = deque()
                while subtypes:
                    match subtypes.pop():
                        case Constructor(name1, arg1):
                            if name2 == name1 or name2 in self.taxonomy.get(name1, {}):
                                casted_constr.append(arg1)
                        case Intersection(l, r):
                            subtypes.extend((l, r))
                return len(casted_constr) != 0 and self._check_subtype_rec(casted_constr, arg2, substitutions)
            case Arrow(src2, tgt2):
                casted_arr: deque[Type] = deque()
                while subtypes:
                    match subtypes.pop():
                        case Arrow(src1, tgt1):
                            if self._check_subtype_rec(deque((src2,)), src1, substitutions):
                                casted_arr.append(tgt1)
                        case Intersection(l, r):
                            subtypes.extend((l, r))
                return len(casted_arr) != 0 and self._check_subtype_rec(casted_arr, tgt2, substitutions)
            case Intersection(l, r):
                return self._check_subtype_rec(subtypes.copy(), l, substitutions) and self._check_subtype_rec(
                    subtypes, r, substitutions
                )
            case Var(name):
                while subtypes:
                    match subtypes.pop():
                        case Literal(value):
                            if substitutions[name] == value:
                                return True
                        case Intersection(l, r):
                            subtypes.extend((l, r))
                return False
            case _:
                msg = f"Unsupported type in check_subtype: {supertype}"
                raise TypeError(msg)

    def check_subtype(
        self,
        subtype: Type,
        supertype: Type,
        substitutions: Mapping[str, Literal],
    ) -> bool:
        """Decides whether subtype <= supertype with respect to intersection type subtyping."""

        return self._check_subtype_rec(deque((subtype,)), supertype, substitutions)

    def infer_substitution(self, subtype: Type, path: Type) -> dict[str, Any] | None:
        """Infers a unique substitution S such that S(subtype) <= path where path is closed. Returns None is no solution exists or multiple solutions exist. Does not respect groups."""

        if not subtype.organized:
            return None

        match subtype:
            case Literal(value1):
                match path:
                    case Literal(value2):
                        if value1 == value2:
                            return {}
            case Constructor(name1, arg1):
                match path:
                    case Constructor(name2, arg2):
                        if name2 == name1 or name2 in self.taxonomy.get(name1, {}):
                            if not arg2.organized:
                                return {}
                            return self.infer_substitution(arg1, arg2)
            case Arrow(src1, tgt1):
                match path:
                    case Arrow(src2, tgt2):
                        substitution = self.infer_substitution(tgt1, tgt2)
                        if substitution is None:
                            return None
                        if all(name in substitution for name in src1.free_vars):
                            if self.check_subtype(src2, src1, substitution):
                                return substitution
                            return None
                        return {}  # there are actual non-Ambiguous cases (relevant in practice?)
            case Intersection(l, r):
                substitution1 = self.infer_substitution(l, path)
                substitution2 = self.infer_substitution(r, path)
                if substitution1 is None:
                    return substitution2
                if substitution2 is None:
                    return substitution1
                if all(
                    (name in substitution2 and substitution2[name] == value for name, value in substitution1.items())
                ):
                    return substitution1  # substitution1 included in substitution2
                if all(
                    (name in substitution1 and substitution1[name] == value for name, value in substitution2.items())
                ):
                    return substitution2  # substitution2 included in substitution1
                return {}
            case Var(name):
                match path:
                    case Literal(value2):  # here a contains check in the group could be done
                        return {name: value2}
            case _:
                msg = f"Unsupported type in infer_substitution: {subtype}"
                raise TypeError(msg)
        return None

    @staticmethod
    def _reflexive_closure(env: Mapping[str, set[str]]) -> dict[str, set[str]]:
        all_types: set[str] = set(env.keys())
        for v in env.values():
            all_types.update(v)
        result: dict[str, set[str]] = {subtype: {subtype}.union(env.get(subtype, set())) for subtype in all_types}
        return result

    @staticmethod
    def _transitive_closure(env: Mapping[str, set[str]]) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {subtype: supertypes.copy() for (subtype, supertypes) in env.items()}
        has_changed = True

        while has_changed:
            has_changed = False
            for known_supertypes in result.values():
                for supertype in known_supertypes.copy():
                    to_add: set[str] = {
                        new_supertype for new_supertype in result[supertype] if new_supertype not in known_supertypes
                    }
                    if to_add:
                        has_changed = True
                    known_supertypes.update(to_add)

        return result
