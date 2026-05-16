##Constraints##
"""
Demonstrates constraints in CoSy.
"""

from cosy.core.specification_builder import SpecificationBuilder
from cosy.core.types import Constructor, Group, Literal, Type, Var, DataGroup
from cosy.maestro import Maestro

def herd_nil(typ) -> list[str]:
    return []

def herd_cons(typ, animal: str, tail: list[str]) -> list[str]:
    return [animal] + tail

def main():

    types = DataGroup("types", ["Cat", "Dog", "Animal"])
    named_components_with_specifications = [
        (
            "Dog",
            lambda: "A Dog",
            SpecificationBuilder().suffix(Constructor("CDog")),
        ),
        (
            "Cat",
            lambda: "A Cat",
            SpecificationBuilder().suffix(Constructor("CCat")),
        ),
        (
            "Generic Animal",
            lambda: "A Generic Animal",
            SpecificationBuilder().suffix(Constructor("CAnimal")),
        ),
        (
            "WrapAnimal",
            lambda x: x,
            SpecificationBuilder()
            .argument("x", Constructor("CAnimal"))
            .suffix(Constructor("Typ", Literal("Animal"))),
        ),
        (
            "WrapDog",
            lambda x: x,
            SpecificationBuilder()
            .argument("x", Constructor("CDog"))
            .suffix(Constructor("Typ", Literal("Dog"))),
        ),
        (
            "WrapCat",
            lambda x: x,
            SpecificationBuilder()
            .argument("x", Constructor("CCat"))
            .suffix(Constructor("Typ", Literal("Cat"))),
        ),
        (
            "HerdNil",
            herd_nil,
            SpecificationBuilder()
            .parameter("typ", types)
            .suffix(Constructor("Herd", Var("typ"))),
        ),
        (
            "HerdCons",
            herd_cons,
            SpecificationBuilder()
            .parameter("typ", types)
            .argument("animal", Constructor("Typ", Var("typ")))
            .argument("tail", Constructor("Herd", Var("typ")))
            .suffix(Constructor("Herd", Var("typ"))),
        ),
    ]
    taxonomy = {
        "CDog": {"CAnimal"},
        "CCat": {"CAnimal"},
    }

    # Tell the Maestro about the component specifications
    maestro = Maestro(named_components_with_specifications, taxonomy=taxonomy)

    # Query for heavy strings
    target: Type = Constructor("Herd")

    # Query the Maestro with the target, then visualize and print results
    results = maestro.query(target, max_count=20)

    for (i, result) in enumerate(results):
        print(f"{i}. -----------------")
        print(result)
    results.visualize(amount=20)
    # print("Now printing all infinite results in order:")


if __name__ == "__main__":
    main()
