from cosy.core.tree import Tree
from cosy.core.types import Group, DataGroup
from cosy.core import SpecificationBuilder, Synthesizer, Constructor, Literal, Var
from cosy.evolutionary_algorithms.evolutionary import SimpleGeneticProgramming, EAState
from cosy.evolutionary_algorithms.selection import FitnessProportionalSelection, FitnessBasedReplacement, TournamentSelection, AgeBasedReplacement, RankBasedSelection
from cosy.evolutionary_algorithms.initialisation import RandomLimitedDepthFirstInitialization
from cosy.evolutionary_algorithms.mutation import ResolutionMutation
from cosy.evolutionary_algorithms.recombination import Crossover
from cosy.evolutionary_algorithms.fitness import ScalarFitnessComparator

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
            .suffix(Constructor("EXP", Constructor("depth", Literal(0)) & Constructor("depth", Literal(None)))
                    & Constructor("Value")),

            "Var": SpecificationBuilder()
            .parameter("v", variable)
            .suffix(Constructor("EXP", Constructor("depth", Literal(0)) & Constructor("depth", Literal(None)))
                    & Constructor("Value")),

            "(+)": SpecificationBuilder()
            .parameter("d1", depth)
            .parameter("d2", depth)
            .parameter("d", depth, lambda v: [max(v["d1"], v["d2"]) + 1])
            .argument("left", Constructor("EXP", Constructor("depth", Var("d1"))))
            .argument("right", Constructor("EXP", Constructor("depth", Var("d2"))))
            .suffix(Constructor("EXP", Constructor("depth", Var("d")) & Constructor("depth", Literal(None)))
                    & Constructor("Non-Value")),

            "(-)": SpecificationBuilder()
            .parameter("d1", depth)
            .parameter("d2", depth)
            .parameter("d", depth, lambda v: [max(v["d1"], v["d2"]) + 1])
            .argument("left", Constructor("EXP", Constructor("depth", Var("d1"))))
            .argument("right", Constructor("EXP", Constructor("depth", Var("d2"))))
            .suffix(Constructor("EXP", Constructor("depth", Var("d")) & Constructor("depth", Literal(None)))
                    & Constructor("Non-Value")),

            "(*)": SpecificationBuilder()
            .parameter("d1", depth)
            .parameter("d2", depth)
            .parameter("d", depth, lambda v: [max(v["d1"], v["d2"]) + 1])
            .argument("left", Constructor("EXP", Constructor("depth", Var("d1"))))
            .argument("right", Constructor("EXP", Constructor("depth", Var("d2"))))
            .suffix(Constructor("EXP", Constructor("depth", Var("d")) & Constructor("depth", Literal(None)))
                    & Constructor("Non-Value")),

            "sin": SpecificationBuilder()
            .parameter("d", depth)
            .parameter("d1", depth, lambda v: [v["d"] - 1])
            .argument("arg", Constructor("EXP", Constructor("depth", Var("d1"))))
            .suffix(Constructor("EXP", Constructor("depth", Var("d")) & Constructor("depth", Literal(None)))
                    & Constructor("Non-Value")),

            "cos": SpecificationBuilder()
            .parameter("d", depth)
            .parameter("d1", depth, lambda v: [v["d"] - 1])
            .argument("arg", Constructor("EXP", Constructor("depth", Var("d1"))))
            .suffix(Constructor("EXP", Constructor("depth", Var("d")) & Constructor("depth", Literal(None)))
                    & Constructor("Non-Value")),
        }

    def pretty_term_algebra(self):
        return {
            "Const": lambda c: f"Constant({c})",

            "Var": lambda v: f"Variable({v})",

            "(+)": lambda d, d1, d2, l, r: f"({l} + {r})",

            "(-)": lambda d, d1, d2, l, r: f"({l} - {r})",

            "(*)": lambda d, d1, d2, l, r: f"({l} * {r})",

            "sin": lambda d, d1, arg: f"sin({arg})",

            "cos": lambda d, d1, arg: f"cos({arg})",


        }

    def substitution_algebra(self):
        return {
            "Const": lambda c, x: c,

            "Var": lambda v, x: x[v],

            "(+)": lambda d, d1, d2, l, r, x: l(x) + r(x),

            "(-)": lambda d, d1, d2, l, r, x: l(x) - r(x),

            "(*)": lambda d, d1, d2, l, r, x: l(x) * r(x),

            "sin": lambda d, d1, arg, x: sin(arg(x)),

            "cos": lambda d, d1, arg, x: cos(arg(x)),
        }


if __name__ == "__main__":

    def target_function(x: float) -> float:
        #return 2.5382 * sin(1.2345 * x) + (0.1234 * (x ** 2) - 0.5678)
        return 2.5382 * x**3 + 1.2345 * x**2 - 0.5678

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

    repo = Symbolic_Regression(max_depth=6, variables=["x"], constants=constants)

    target = Constructor("EXP", Constructor("depth", Literal(None))) & Constructor("Non-Value")
    synthesizer = Synthesizer(repo.specification(), {})

    start_time = time.time()
    solution_space = synthesizer.construct_solution_space(target).prune()
    end_time = time.time()

    print(f"SolutionSpace construction took {end_time - start_time:.5f} seconds.")

    SIZE_PENALTY_COEFF = 0.1

    def mean_squared_error(y_true: list[float], y_pred: list[float]) -> float:
        return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

    def fitness_function(tree: Tree[str]) -> float:
        substitute_in_tree = tree.interpret(repo.substitution_algebra())
        y_pred = [substitute_in_tree({"x": x}) for x in X_train]
        mse = mean_squared_error(y_train, y_pred)
        #mse += SIZE_PENALTY_COEFF * tree.size
        return mse

    def test_function(tree: Tree[str]) -> float:
        substitute_in_tree = tree.interpret(repo.substitution_algebra())
        y_pred = [substitute_in_tree({"x": x}) for x in X_test]
        return mean_squared_error(y_test, y_pred)

    max_generations = 200

    def termination(state: EAState[str]) -> bool:
        return state.generation >= max_generations

    initialization = RandomLimitedDepthFirstInitialization(solution_space, target, 6)
    mutation = ResolutionMutation(solution_space, target, 6)
    recombination = Crossover(solution_space, target, 6)
    parent_selection = FitnessProportionalSelection()
    survivor_selection = FitnessBasedReplacement()
    fitness_comparator = ScalarFitnessComparator(False)  # minimize MSE

    gp = SimpleGeneticProgramming(solution_space, target, termination, initialization, mutation, recombination,
                                  parent_selection, survivor_selection, fitness_comparator)

    start_time = time.time()
    best_tree = gp.evolutionary_best(fitness_function, 250, 0.1, 0.9)
    end_time = time.time()

    print(f"Symbolic Regression took {end_time - start_time:.5f} seconds.")

    print(f"Best solution: {best_tree.interpret(repo.pretty_term_algebra())}")
    print(f"Train MSE: {fitness_function(best_tree):.5f}")
    print(f"Test MSE: {test_function(best_tree):.5f}")
    #print(f"With size penalty: {SIZE_PENALTY_COEFF:.5f}")

