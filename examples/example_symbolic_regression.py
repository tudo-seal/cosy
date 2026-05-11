import random
import time
from math import cos, sin
from typing import Any

from cosy.core import Constructor, Literal, SpecificationBuilder, Synthesizer, Var
from cosy.core.tree import Tree
from cosy.core.types import DataGroup, Group
from cosy.evolutionary_algorithms.evolutionary import EAState, SimpleGeneticProgramming
from cosy.evolutionary_algorithms.fitness import ScalarFitnessComparator
from cosy.evolutionary_algorithms.initialisation import RandomLimitedDepthFirstInitialization
from cosy.evolutionary_algorithms.mutation import ResolutionMutation
from cosy.evolutionary_algorithms.recombination import Crossover
from cosy.evolutionary_algorithms.selection import AgeBasedReplacement, FitnessProportionalSelection, Selection
from cosy.rng.factory import RNGFactory


class SymbolicRegression:
    def __init__(self, max_depth: int, variables: list[str], constants: list[float]) -> None:
        self.max_depth = max_depth
        self.variables = variables
        self.constants = constants

    def specification(self) -> dict[str, Any]:
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
    return 2.5382 * x * x + 1.2345 * x - 0.5678


def run_symbolic_regression(
    *,
    seed: int = 0,
    train_values: list[float] | None = None,
    test_values: list[float] | None = None,
    population_size: int = 50,
    max_generations: int = 20,
    max_depth: int = 4,
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
        seed: Random seed for reproducibility (default: 0).
        train_values: Training data points (default: [-2, -1, 0, 1, 2]).
        test_values: Test data points (default: [-3, -0.5, 0.5, 3]).
        population_size: EA population size (default: 12).
        max_generations: Maximum GA generations (default: 6).
        max_depth: Maximum tree depth (default: 4).

    Returns:
        (best_tree, train_mse, test_mse): Best-of-run solution and its MSE values.
    """
    train_values = train_values if train_values is not None else [-2.0, -1.0, 0.0, 1.0, 2.0]
    test_values = test_values if test_values is not None else [-3.0, -0.5, 0.5, 3.0]

    # Create a deterministic RNG setup for all random operations
    # Each component receives an independent, deterministically-seeded child RNG from a factory.
    # This ensures that all evolutionary randomness is reproducible from a single seed.
    random.seed(seed)  # Seed module-level random for consistency

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
        return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred, strict=False)) / len(y_true)

    def fitness_function(tree: Tree[str]) -> float:
        substitute_in_tree = tree.interpret(repo.evaluation_algebra())
        y_pred = [substitute_in_tree({"x": x}) for x in train_values]
        return mean_squared_error(y_train, y_pred)

    def test_function(tree: Tree[str]) -> float:
        substitute_in_tree = tree.interpret(repo.evaluation_algebra())
        y_pred = [substitute_in_tree({"x": x}) for x in test_values]
        return mean_squared_error(y_test, y_pred)

    def termination(state: EAState[str]) -> bool:
        return state.generation >= max_generations

    # Pass seeded RNGs to all components at construction time for determinism
    initialization = RandomLimitedDepthFirstInitialization(solution_space, target, max_depth, rng=initialization_rng)
    mutation = ResolutionMutation(solution_space, target, max_depth, rng=mutation_rng)
    recombination = Crossover(solution_space, target, max_depth, rng=recombination_rng)
    parent_selection: Selection = FitnessProportionalSelection(rng=selection_rng)
    survivor_selection: Selection = AgeBasedReplacement()
    fitness_comparator = ScalarFitnessComparator(False)

    # All component RNGs are already specified, so disable distribute_rngs
    # Use gp_rng from factory for consistency
    gp = SimpleGeneticProgramming(
        solution_space,
        target,
        termination,
        initialization,
        mutation,
        recombination,
        parent_selection,
        survivor_selection,
        fitness_comparator,
        rng=gp_rng,
        distribute_rngs=False,
    )

    best_tree = gp.evolutionary_best(fitness_function, population_size, 0.2, 0.4, verbose=False)
    if best_tree is None:
        msg = "Symbolic regression demo did not produce a solution"
        raise RuntimeError(msg)

    return best_tree, fitness_function(best_tree), test_function(best_tree)


if __name__ == "__main__":
    start_time = time.time()
    best_tree, train_mse, test_mse = run_symbolic_regression(seed=0)
    end_time = time.time()

    print(f"Symbolic Regression took {end_time - start_time:.5f} seconds.")
    print(f"Best solution: {best_tree.interpret(SymbolicRegression.pretty_term_algebra())}")
    print(f"Train MSE: {train_mse:.5f}")
    print(f"Test MSE: {test_mse:.5f}")
