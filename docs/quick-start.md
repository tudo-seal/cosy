# Quick Start
Provides a guide and a toy example of how to create a repository and generate some results.
The toy example shows the computation of Fibonacci numbers by means of composition of components `fib_zero`, `fib_one`, and `fib_next`.

## 1. Define Component Specifications

Using domain-specific language provided by the `SpecificationBuilder` class define a mapping of components to respective specifications.

- The component `fib_zero` is specified by `Constructor("fib") & Constructor("at", Literal(0, "int"))`, which combines two properties.
  + `Constructor("fib")` means that `fib_zero` it is a Fibonacci number.
  + `Constructor("at", Literal(0))` means that `fib_zero` is associated with index `0`.
- The component `fib_one`, similarly to `fib_zero`, is a Fibonacci number and is associated with index `1`.
- The component `fib_next` has three parameters associated with the group `int`
  + `z` index of the constructed Fibonacci number
  + `y` index of the previous Fibonacci number, which is `z - 1`
  + `x` index of the Fibonacci number two indices prior, which is `z - 2`
  
  and two arguments
  + `f1` previous Fibonacci number
  + `f2` Fibonacci number two indices prior
 
  Given the above parameters and arguments the component `fib_next` computes a Fibonacci number and is associated with index `z`, specified by `Constructor("fib") & Constructor("at", Var("z")))`.

```
def fib_zero() -> int:
    return 0

def fib_one() -> int:
    return 1

def fib_next(_z: int, _y: int, _x: int, f1 : int, f2: int) -> int:
    return f1 + f2

component_specifications = {
    fib_zero: SpecificationBuilder()
          .suffix(Constructor("fib") & Constructor("at", Literal(0))),

    fib_one: SpecificationBuilder()
          .suffix(Constructor("fib") & Constructor("at", Literal(1))),

    fib_next: SpecificationBuilder()
              .parameter("z", "int")
              .parameter("y", "int", lambda vs: [vs["z"] - 1])
              .parameter("x", "int", lambda vs: [vs["z"] - 2])
              .argument("f1", Constructor("fib") & Constructor("at", Var("y")))
              .argument("f2", Constructor("fib") & Constructor("at", Var("x")))
              .suffix(Constructor("fib") & Constructor("at", Var("z"))),
}
```

## 2. Define Parameter Space

Define a mapping from parameter groups to parameter values.
In the toy example, indices less than `20` are considered.

```
parameter_space = {
    "int": list(range(0, 20))
}
```

## 3. Instantiate CoSy

Create an instance of `CoSy` by providing component specifications and the parameter space.

```
cosy = CoSy(component_specifications, parameter_space)
```

## 4. Specify a Query and Construct Solutions

Specify the query for which solutions should be found.
Solutions are found by means of instantiation and composition of the given components in the given parameter space.

### Arbitrary Fibonacci numbers

The following query `Constructor("fib")` describes arbitrary Fibonacci numbers at indices in the given parameter space.

```
query = Constructor("fib")
```

Using the `solve` method, iterate over and display solutions for the given query.

```
for solution in cosy.solve(query):
    print(solution)
```

### Fibonacci numbers at Specific Indices

The specification allows us to query Fibonacci numbers at specific indices.
For an index `i` the query `Constructor("fib") & Constructor("at", Literal(i))` describes the Fibonacci number at index `i`.
Using the `solve` method, construct and display this Fibonacci number.

```
for i in range(20):
    query = Constructor("fib") & Constructor("at", Literal(i))
    print(i, next(iter(cosy.solve(query))))
```