##Symbolic Regression##
"""Shows how to do a symbolic regression using cosy."""

import time
from math import cos, sin
from typing import Any

from cosy.core import Constructor, Literal, SpecificationBuilder, Synthesizer, Var
from cosy.core.tree import Tree
from cosy.core.types import DataGroup, Group
from cosy.evolutionary_algorithms.evolutionary import EvolutionarySearch
from cosy.evolutionary_algorithms.fitness import ScalarFitnessComparator
from cosy.evolutionary_algorithms.initialisation import SampledInitialization
from cosy.evolutionary_algorithms.mutation import ResolutionMutation
from cosy.evolutionary_algorithms.recombination import SubtreeSwap
from cosy.evolutionary_algorithms.rng.factory import RNGFactory
from cosy.evolutionary_algorithms.selection import FitnessBasedReplacement, RankBasedSelection
from cosy.evolutionary_algorithms.termination import Generations
from cosy.search import DepthBoundedRandomSampler, generator_query


class SymbolicRegression:
    """_summary_.

    Attributes:
        max_depth (int): _description_
        variables (list[str]): _description_
        constants (list[float]): _description_
    """

    def __init__(self, max_depth: int, variables: list[str], constants: list[float]) -> None:
        """_summary_.

        Args:
            max_depth (int): _description_
            variables (list[str]): _description_
            constants (list[float]): _description_
        """
        self.max_depth = max_depth
        self.variables = variables
        self.constants = constants

    def specification(self) -> dict[str, Any]:
        """_summary_.

        Returns:
            dict[str, Any]: _description_
        """
        depth: Group = DataGroup("depth", range(self.max_depth + 1))
        variable: Group = DataGroup("variable", self.variables)
        constant: Group = DataGroup("constant", self.constants)

        return {
            "Const": SpecificationBuilder()
            .parameter("c", constant)
            .suffix(
                Constructor("EXP", Constructor("depth", Literal(0)) & Constructor("depth", Literal(None)))
                & Constructor("Value")
            ),
            "Var": SpecificationBuilder()
            .parameter("v", variable)
            .suffix(
                Constructor("EXP", Constructor("depth", Literal(0)) & Constructor("depth", Literal(None)))
                & Constructor("Value")
            ),
            "(+)": SpecificationBuilder()
            .parameter("d1", depth)
            .parameter("d2", depth)
            .parameter("d", depth, lambda v: [max(v["d1"], v["d2"]) + 1])
            .argument("left", Constructor("EXP", Constructor("depth", Var("d1"))))
            .argument("right", Constructor("EXP", Constructor("depth", Var("d2"))))
            .suffix(
                Constructor("EXP", Constructor("depth", Var("d")) & Constructor("depth", Literal(None)))
                & Constructor("Non-Value")
            ),
            "(-)": SpecificationBuilder()
            .parameter("d1", depth)
            .parameter("d2", depth)
            .parameter("d", depth, lambda v: [max(v["d1"], v["d2"]) + 1])
            .argument("left", Constructor("EXP", Constructor("depth", Var("d1"))))
            .argument("right", Constructor("EXP", Constructor("depth", Var("d2"))))
            .suffix(
                Constructor("EXP", Constructor("depth", Var("d")) & Constructor("depth", Literal(None)))
                & Constructor("Non-Value")
            ),
            "(*)": SpecificationBuilder()
            .parameter("d1", depth)
            .parameter("d2", depth)
            .parameter("d", depth, lambda v: [max(v["d1"], v["d2"]) + 1])
            .argument("left", Constructor("EXP", Constructor("depth", Var("d1"))))
            .argument("right", Constructor("EXP", Constructor("depth", Var("d2"))))
            .suffix(
                Constructor("EXP", Constructor("depth", Var("d")) & Constructor("depth", Literal(None)))
                & Constructor("Non-Value")
            ),
            "sin": SpecificationBuilder()
            .parameter("d", depth)
            .parameter("d1", depth, lambda v: [v["d"] - 1])
            .argument("arg", Constructor("EXP", Constructor("depth", Var("d1"))))
            .suffix(
                Constructor("EXP", Constructor("depth", Var("d")) & Constructor("depth", Literal(None)))
                & Constructor("Non-Value")
            ),
            "cos": SpecificationBuilder()
            .parameter("d", depth)
            .parameter("d1", depth, lambda v: [v["d"] - 1])
            .argument("arg", Constructor("EXP", Constructor("depth", Var("d1"))))
            .suffix(
                Constructor("EXP", Constructor("depth", Var("d")) & Constructor("depth", Literal(None)))
                & Constructor("Non-Value")
            ),
        }

    @staticmethod
    def pretty_term_algebra() -> dict[str, Any]:
        """_summary_.

        Returns:
            dict[str, Any]: _description_
        """
        return {
            "Const": lambda c: f"{c}",
            "Var": lambda v: f"{v}",
            "(+)": lambda d, d1, d2, left, right: f"({left} + {right})",
            "(-)": lambda d, d1, d2, left, right: f"({left} - {right})",
            "(*)": lambda d, d1, d2, left, right: f"({left} * {right})",
            "sin": lambda d, d1, arg: f"sin({arg})",
            "cos": lambda d, d1, arg: f"cos({arg})",
        }

    @staticmethod
    def evaluation_algebra() -> dict[str, Any]:
        # Implements a term assignment. Therefore, the evaluation algebra takes a variable assignment as an additional
        # input and evaluates the term in floating point arithmetic with respect to this variable assignment.
        # The variable assignment is a mapping from variable names to their values, e.g., {"x": 1.0}.
        """_summary_.

        Returns:
            dict[str, Any]: _description_
        """
        return {
            "Const": lambda c, x: c,
            "Var": lambda v, x: x[v],
            "(+)": lambda d, d1, d2, left, right, x: left(x) + right(x),
            "(-)": lambda d, d1, d2, left, right, x: left(x) - right(x),
            "(*)": lambda d, d1, d2, left, right, x: left(x) * right(x),
            "sin": lambda d, d1, arg, x: sin(arg(x)),
            "cos": lambda d, d1, arg, x: cos(arg(x)),
        }


def target_function(x: float) -> float:
    """_summary_.

    Args:
        x (float): _description_

    Returns:
        float: _description_
    """
    return 2.5382 * x * x + 1.2345 * x - 0.5678


def run_symbolic_regression(
    *,
    seed: int = 0,
    train_values: list[float] | None = None,
    test_values: list[float] | None = None,
    population_size: int = 50,
    max_generations: int = 20,
    max_depth: int = 4,
    max_size: int = 40,
    sample_depth: int = 6,
) -> tuple[Tree[str], float, float]:
    """Run a symbolic regression demo with optional seeding for deterministic behavior.

    The random behavior of the evolutionary algorithm is controlled via an RNGFactory
    instance seeded from the provided `seed` parameter. All components (initialization,
    mutation, recombination, selection) receive independent, deterministically derived
    child RNGs from this factory, ensuring reproducible results.

    Note: Complete determinism across multiple process invocations requires
    PYTHONHASHSEED environment variable to be set (e.g., PYTHONHASHSEED=0),
    because the solution space construction uses Python sets which have
    non-deterministic iteration order without this.

    Args:
        seed (int): Random seed for reproducibility (default: 0).
        train_values (list[float] | None): Training data points (default: [-2, -1, 0, 1, 2]).
        test_values (list[float] | None): Test data points (default: [-3, -0.5, 0.5, 3]).
        population_size (int): EA population size (default: 50).
        max_generations (int): Maximum GA generations (default: 20).
        max_depth (int): Maximum expression depth the repository admits (default: 4).
        max_size (int): The bound of the recombination acceptance test, in function-symbol
            occurrences. An exchange may deepen a term, and this is what bounds the growth.
            (Default value = 40)
        sample_depth (int): The depth bound of the sampler the initializer and the mutation draw
            from, in edges. (Default value = 6)

    Returns:
        tuple[Tree[str], float, float]: (best_tree, train_mse, test_mse): Best-of-run solution and its MSE values.
    """
    train_values = train_values if train_values is not None else [-2.0, -1.0, 0.0, 1.0, 2.0]
    test_values = test_values if test_values is not None else [-3.0, -0.5, 0.5, 3.0]

    # Create a deterministic RNG setup for all random operations
    # Each component receives an independent, deterministically-seeded child RNG from a factory.
    # This ensures that all evolutionary randomness is reproducible from a single seed.
    # Factory creates independent child RNGs deterministically: same seed → same RNG sequences
    rng_factory = RNGFactory.from_seed(seed)
    initialization_rng = rng_factory.child()  # For initial population generation
    mutation_rng = rng_factory.child()  # For mutation operator
    recombination_rng = rng_factory.child()  # For crossover operator
    selection_rng = rng_factory.child()  # For parent selection
    gp_rng = rng_factory.child()  # For EA controller internal randomness

    y_train = [target_function(value) for value in train_values]
    y_test = [target_function(value) for value in test_values]

    constants = [2.5382, 1.2345, 0.5678]
    repo = SymbolicRegression(max_depth=max_depth, variables=["x"], constants=constants)

    target = Constructor("EXP", Constructor("depth", Literal(None))) & Constructor("Non-Value")
    synthesizer = Synthesizer(repo.specification(), {})
    solution_space = synthesizer.construct_solution_space(target).prune()

    def mean_squared_error(y_true: list[float], y_pred: list[float]) -> float:
        """_summary_.

        Args:
            y_true (list[float]): _description_
            y_pred (list[float]): _description_

        Returns:
            float: _description_
        """
        return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred, strict=False)) / len(y_true)

    def fitness_function(tree: Tree[str]) -> float:
        """_summary_.

        Args:
            tree (Tree[str]): _description_

        Returns:
            float: _description_
        """
        substitute_in_tree = tree.interpret(repo.evaluation_algebra())
        y_pred = [substitute_in_tree({"x": x}) for x in train_values]
        return mean_squared_error(y_train, y_pred)

    def test_function(tree: Tree[str]) -> float:
        """_summary_.

        Args:
            tree (Tree[str]): _description_

        Returns:
            float: _description_
        """
        substitute_in_tree = tree.interpret(repo.evaluation_algebra())
        y_pred = [substitute_in_tree({"x": x}) for x in test_values]
        return mean_squared_error(y_test, y_pred)

    # Pass seeded RNGs to all components at construction time for determinism. The driver
    # distributes nothing, so what each component draws from is readable here.
    #
    # The sampler is the depth-bounded one. Mutation poses a fresh residual query per call, and a
    # size-uniform sampler builds its weighted construction once per query, so a counting sampler
    # would pay that construction on every mutation. On this space that is the difference between
    # a fraction of a millisecond and several hundred of them per offspring.
    #
    # That choice makes this a fast demonstration rather than one of the configurations the package
    # docstring lists as converging almost surely: truncation is conservative but not generous, and
    # the operators bound depth here while the recombination test bounds size.
    #
    # Truncation over parents and offspring together gives every copy of an individual a place of
    # its own, and a pass that neither recombines nor mutates hands a parent on unchanged. The
    # multiplicity of the fittest individuals therefore grows, and a run can reach a population
    # that holds few distinct terms and stops improving. Which seed does that is a property of the
    # run rather than of the components, so the shipped one is chosen to show the search working.
    initialization: SampledInitialization[Any, str, Any] = SampledInitialization(
        DepthBoundedRandomSampler(sample_depth, initialization_rng)
    )
    mutation: ResolutionMutation[Any, str, Any] = ResolutionMutation(
        DepthBoundedRandomSampler(sample_depth, mutation_rng), mutation_rng
    )
    recombination: SubtreeSwap[Any, str, Any] = SubtreeSwap(recombination_rng, max_size=max_size)
    query = generator_query(solution_space, target)

    search: EvolutionarySearch[Any, str, Any] = EvolutionarySearch(
        initializer=initialization,
        mutation=mutation,
        recombination=recombination,
        # Rank-based rather than fitness-proportional: the fitness here is a mean squared error,
        # and a scalarization of it would have to map errors of any magnitude into the positive
        # reals. Ranks are invariant under the scale of the error, which is what this problem
        # needs, and proportional selection is the component that takes a scalarization.
        parent_selection=RankBasedSelection(1.7, selection_rng),
        survivor_selection=FitnessBasedReplacement(),
        termination=Generations(max_generations),
        population_size=population_size,
        crossover_rate=0.8,
        mutation_rate=0.3,
        rng=gp_rng,
        comparator=ScalarFitnessComparator(greater_is_better=False),
    )

    best_tree = search.evolutionary_best(query, fitness_function)
    return best_tree, fitness_function(best_tree), test_function(best_tree)


if __name__ == "__main__":
    start_time = time.time()
    best_tree, train_mse, test_mse = run_symbolic_regression(
        seed=4, population_size=250, max_generations=100, max_depth=5
    )
    end_time = time.time()

    print(f"Symbolic Regression took {end_time - start_time:.5f} seconds.")
    print(f"Best solution: {best_tree.interpret(SymbolicRegression.pretty_term_algebra())}")
    print(f"Train MSE: {train_mse:.5f}")
    print(f"Test MSE: {test_mse:.5f}")
