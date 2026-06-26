##Constraints##
"""
Demonstrates constraints in CoSy.
"""

from typing import TYPE_CHECKING

from cosy.core.specification_builder import SpecificationBuilder
from cosy.core.types import Constructor, Type
from cosy.extensions.partial_terms import (
    partial_term_builder,
)
from cosy.maestro import Maestro

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from cosy.core.synthesizer import Specification


def herd_nil() -> list[str]:
    return []


def herd_cons(animal: str, tail: list[str]) -> list[str]:
    return [animal, *tail]


def main():
    named_components_with_specifications: Sequence[tuple[str, Callable, Specification]] = [
        (
            "Dog",
            lambda: "A Dog",
            SpecificationBuilder().suffix(Constructor("CDog") & Constructor("Walking")),
        ),
        (
            "Cat",
            lambda: "A Cat",
            SpecificationBuilder().suffix(Constructor("CCat") & Constructor("Walking")),
        ),
        (
            "Generic Animal",
            lambda: "A Generic Animal",
            SpecificationBuilder().suffix(Constructor("CAnimal") & Constructor("Walking")),
        ),
        # (
        #     "Wrap",
        #     lambda x: f"<{x}>",
        #     SpecificationBuilder()
        #     .argument("animal", Constructor("CAnimal") & Constructor("Walking"))
        #     .suffix(Constructor("CAnimal")),
        # ),
        (
            "Add Wings",
            # lambda color, animal: f"{animal} with {color} wings!",
            lambda animal: f"{animal} with wings!",
            SpecificationBuilder()
            # .parameter("wing_color", DataGroup("color", ["blue", "red"]))
            .argument("animal", Constructor("CAnimal") & Constructor("Walking"))
            .suffix(Constructor("CAnimal") & Constructor("Flying")),
        ),
        (
            "HerdNil",
            herd_nil,
            SpecificationBuilder().suffix(Constructor("Herd")),
        ),
        (
            "HerdCons",
            herd_cons,
            SpecificationBuilder()
            .argument("animal", Constructor("CAnimal"))
            .argument("tail", Constructor("Herd"))
            .suffix(Constructor("Herd")),
        ),
    ]
    taxonomy = {
        "CDog": {"CAnimal"},
        "CCat": {"CAnimal"},
    }

    b = partial_term_builder
    partial_term = b(
        "HerdCons",
        # animal=b("Add Wings"),
        animal=b("Add Wings", animal=None),
        tail=b("HerdNil"),
    )

    target: Type = Constructor("Herd")

    # Tell the Maestro about the component specifications
    maestro = Maestro(named_components_with_specifications, taxonomy=taxonomy)

    # Query the Maestro with the target, then print results
    results = maestro.query(target, max_count=40, partial_term=partial_term)
    for i, result in enumerate(results):
        print(f"{i}. -----------------")
        print(result)


if __name__ == "__main__":
    main()
