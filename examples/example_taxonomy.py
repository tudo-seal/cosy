##Constraints##
"""
Demonstrates constraints in CoSy.
"""

from cosy.core.specification_builder import SpecificationBuilder
from cosy.core.types import Constructor, Type, Literal
from cosy.extensions.debug_helpers import debug_note_repository, Hashabledict
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
            "Add Wings",
            lambda x: f"{x} with wings!",
            SpecificationBuilder()
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

    arguments = (
        "_DEBUG_argument_HerdCons", Hashabledict({
            "animal": None,
            "tail": None,
        })
    )
    # Tell the Maestro about the component specifications
    maestro = Maestro(debug_repository, taxonomy=taxonomy)

    # Query for heavy strings
    target: Type = Constructor("Herd") & (Constructor("_DEBUG_Args", Literal(arguments)))

    # Query the Maestro with the target, then visualize and print results
    results = maestro.query(target, max_count=40)

    for i, result in enumerate(results):
        print(f"{i}. -----------------")
        print(result)
    results.visualize()
    # print("Now printing all infinite results in order:")


if __name__ == "__main__":
    main()
