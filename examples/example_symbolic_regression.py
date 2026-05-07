from src.cosy.core.tree import Tree
from src.cosy.core.types import Group, DataGroup
from src.cosy.core import SpecificationBuilder, Synthesizer, Constructor, Literal, Var
from src.cosy.evolutionary_algorithms.evolutionary import SimpleGeneticProgramming, EAState
from src.cosy.evolutionary_algorithms.selection import FitnessProportionalSelection, FitnessBasedReplacement
from src.cosy.evolutionary_algorithms.initialisation import RandomLimitedDepthFirstInitialization
from src.cosy.evolutionary_algorithms.mutation import ResolutionMutation
from src.cosy.evolutionary_algorithms.recombination import Crossover
from src.cosy.evolutionary_algorithms.fitness import ScalarFitnessComparator

from math import sin, cos, log, exp
import time
import random


class Symbolic_Regression:

    def __init__(self, max_depth: int, variables: list[str], constants: list[float]) -> None:
        self.max_depth = max_depth
        self.variables = variables
        self.constants = constants

    def specification(self):
        depth: Group = DataGroup("depth", range(0, self.max_depth + 1))
        variable: Group = DataGroup("variable", self.variables)
        constant: Group = DataGroup("constant", self.constants)

        return {
            "Const": SpecificationBuilder()
            .parameter("c", constant)
            .suffix(Constructor("EXP", Constructor("depth", Literal(0)) & Constructor("depth", Literal(None)))),

            "Var": SpecificationBuilder()
            .parameter("v", variable)
            .suffix(Constructor("EXP", Constructor("depth", Literal(0)) & Constructor("depth", Literal(None)))),

            "(+)": SpecificationBuilder()
            .parameter("d", depth)
            .parameter("d1", depth)
            .parameter("d2", depth, lambda v: [v["d"] - 1 - v["d1"]])
            .argument("left", Constructor("EXP", Constructor("depth", Var("d1"))))
            .argument("right", Constructor("EXP", Constructor("depth", Var("d2"))))
            .suffix(Constructor("EXP", Constructor("depth", Var("d")) & Constructor("depth", Literal(None)))),

            "(-)": SpecificationBuilder()
            .parameter("d", depth)
            .parameter("d1", depth)
            .parameter("d2", depth, lambda v: [v["d"] - 1 - v["d1"]])
            .argument("left", Constructor("EXP", Constructor("depth", Var("d1"))))
            .argument("right", Constructor("EXP", Constructor("depth", Var("d2"))))
            .suffix(Constructor("EXP", Constructor("depth", Var("d")) & Constructor("depth", Literal(None)))),

            "(*)": SpecificationBuilder()
            .parameter("d", depth)
            .parameter("d1", depth)
            .parameter("d2", depth, lambda v: [v["d"] - 1 - v["d1"]])
            .argument("left", Constructor("EXP", Constructor("depth", Var("d1"))))
            .argument("right", Constructor("EXP", Constructor("depth", Var("d2"))))
            .suffix(Constructor("EXP", Constructor("depth", Var("d")) & Constructor("depth", Literal(None)))),

            "(/)": SpecificationBuilder()
            .parameter("d", depth)
            .parameter("d1", depth)
            .parameter("d2", depth, lambda v: [v["d"] - 1 - v["d1"]])
            .argument("left", Constructor("EXP", Constructor("depth", Var("d1"))))
            .argument("right", Constructor("EXP", Constructor("depth", Var("d2"))))
            .suffix(Constructor("EXP", Constructor("depth", Var("d")) & Constructor("depth", Literal(None)))),

            "sin": SpecificationBuilder()
            .parameter("d", depth)
            .parameter("d1", depth, lambda v: [v["d"] - 1])
            .argument("arg", Constructor("EXP", Constructor("depth", Var("d1"))))
            .suffix(Constructor("EXP", Constructor("depth", Var("d")) & Constructor("depth", Literal(None)))),

            "cos": SpecificationBuilder()
            .parameter("d", depth)
            .parameter("d1", depth, lambda v: [v["d"] - 1])
            .argument("arg", Constructor("EXP", Constructor("depth", Var("d1"))))
            .suffix(Constructor("EXP", Constructor("depth", Var("d")) & Constructor("depth", Literal(None)))),

            "log": SpecificationBuilder()
            .parameter("d", depth)
            .parameter("d1", depth, lambda v: [v["d"] - 1])
            .argument("arg", Constructor("EXP", Constructor("depth", Var("d1"))))
            .suffix(Constructor("EXP", Constructor("depth", Var("d")) & Constructor("depth", Literal(None)))),
        }

    def pretty_term_algebra(self):
        return {
            "Const": lambda c: f"Constant({c})",

            "Var": lambda v: f"Variable({v})",

            "(+)": lambda d, d1, d2, l, r: f"({l} + {r})",

            "(-)": lambda d, d1, d2, l, r: f"({l} - {r})",

            "(*)": lambda d, d1, d2, l, r: f"({l} * {r})",

            "(/)": lambda d, d1, d2, l, r: f"({l} / {r})",

            "sin": lambda d, d1, arg: f"sin({arg})",

            "cos": lambda d, d1, arg: f"cos({arg})",

            "log": lambda d, d1, arg: f"log({arg})",

        }

    def substitution_algebra(self):
        return {
            "Const": lambda c, x: c,

            "Var": lambda v, x: x[v],

            "(+)": lambda d, d1, d2, l, r, x: l(x) + r(x),

            "(-)": lambda d, d1, d2, l, r, x: l(x) - r(x),

            "(*)": lambda d, d1, d2, l, r, x: l(x) * r(x),

            "(/)": lambda d, d1, d2, l, r, x: l(x) / r(x),

            "sin": lambda d, d1, arg, x: sin(arg(x)),

            "cos": lambda d, d1, arg, x: cos(arg(x)),

            "log": lambda d, d1, arg, x: log(arg(x)),
        }


if __name__ == "__main__":

    def target_function(x : float) -> float:
        return 2.5382 * sin(1.2345 * x) + (0.1234 * (x ** 2) - 0.5678)

    train_data_size = 100
    test_data_size = 100

    train_data_bounds = (0.0, 10.0)
    test_data_bounds = (-5.0, 15.0)

    X_train = []
    y_train = []
    for _ in range(train_data_size):
        value = random.uniform(train_data_bounds[0], train_data_bounds[1])
        X_train.append(value)
        y_train.append(target_function(value))

    X_test = []
    y_test = []
    for _ in range(test_data_size):
        value = random.uniform(test_data_bounds[0], test_data_bounds[1])
        X_test.append(value)
        y_test.append(target_function(value))


    constants = [2.5382, 1.2345, 0.1234, 0.5678]

    for _ in range(5):
        constants.append(random.uniform(0.0, 10.0))

    repo = Symbolic_Regression(max_depth=6, variables=["x"], constants=constants)

    target = Constructor("EXP", Constructor("depth", Literal(None)))
    synthesizer = Synthesizer(repo.specification(), {})

    start_time = time.time()
    solution_space = synthesizer.construct_solution_space(target).prune()
    end_time = time.time()

    print(f"SolutionSpace construction took {end_time - start_time:.5f} seconds.")

    def mean_squared_error(y_true: list[float], y_pred: list[float]) -> float:
        return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

    def fitness_function(tree: Tree[str]) -> float:
        substitute_in_tree = tree.interpret(repo.substitution_algebra())
        y_pred = [substitute_in_tree({"x": x}) for x in X_train]
        return mean_squared_error(y_train, y_pred)

    def test_function(tree: Tree[str]) -> float:
        substitute_in_tree = tree.interpret(repo.substitution_algebra())
        y_pred = [substitute_in_tree({"x": x}) for x in X_test]
        return mean_squared_error(y_test, y_pred)

    max_generations = 50

    def termination(state: EAState[str]) -> bool:
        return state.generation >= max_generations

    initialization = RandomLimitedDepthFirstInitialization(solution_space, target, 7)
    mutation = ResolutionMutation(solution_space, target, 7)
    recombination = Crossover(solution_space, target)
    parent_selection = FitnessProportionalSelection()
    survivor_selection = FitnessBasedReplacement()
    fitness_comparator = ScalarFitnessComparator(False) # minimize MSE

    gp = SimpleGeneticProgramming(solution_space, target, termination, initialization, mutation, recombination,
                                  parent_selection, survivor_selection, fitness_comparator)

    start_time = time.time()
    best_tree = gp.evolutionary_best(fitness_function, 200, 0.05, 0.9)
    end_time = time.time()

    print(f"Symbolic Regression took {end_time - start_time:.5f} seconds.")

    print(f"Best solution: {best_tree.interpret(repo.pretty_term_algebra())}")
    print(f"Train MSE: {fitness_function(best_tree):.5f}")
    print(f"Test MSE: {test_function(best_tree):.5f}")

