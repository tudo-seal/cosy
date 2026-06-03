##Constraints##
"""
Demonstrates constraints in CoSy.
"""

from cosy.core.specification_builder import SpecificationBuilder
from cosy.core.types import Constructor, Literal, Type, DataGroup
from cosy.extensions.debug_helpers import DEBUG_VALUES_CONSTRUCTOR, debug_note_repository, partial_term_builder
from cosy.maestro import Maestro


def herd_nil() -> list[str]:
    return []


def herd_cons(animal: str, tail: list[str]) -> list[str]:
    return [animal, *tail]


def main():
    named_components_with_specifications = [
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
        (
            "Wrap",
            lambda x: f"<{x}>",
            SpecificationBuilder()
            .argument("animal", Constructor("CAnimal") & Constructor("Walking"))
            .suffix(Constructor("CAnimal")),
        ),
        (
            "Add Wings",
            lambda color, animal: f"{animal} with {color} wings!",
            SpecificationBuilder()
            .parameter("wing_color", DataGroup("color", ["blue", "red"]))
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
    debug_repository = debug_note_repository(named_components_with_specifications)
    b = partial_term_builder
    partial_term = b(
        "HerdCons",
        # animal=b("Add Wings"),
        animal=b(
            "Add Wings",
            # animal=None,
        ),
        tail=b("HerdNil"),
    )
    temp1 = b(
            "Add Wings",
            animal=None,
        )
    temp2 = b(
            "Add Wings",
        )

    # Tell the Maestro about the component specifications
    maestro = Maestro(named_components_with_specifications, taxonomy=taxonomy)
    # maestro = Maestro(debug_repository, taxonomy=taxonomy)

    # Query for heavy strings
    target: Type = Constructor("Herd")
    # target: Type = Constructor("Herd") & (Constructor(DEBUG_VALUES_CONSTRUCTOR, Literal(partial_term)))

    # Query the Maestro with the target, then visualize and print results
    results = maestro.query(target, max_count=40, partial_term=partial_term)

    for i, result in enumerate(results):
        print(f"{i}. -----------------")
        print(result)
        pass
    # results.visualize()
    # print("Now printing all infinite results in order:")


if __name__ == "__main__":
    main()
