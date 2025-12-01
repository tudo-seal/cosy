# Quick Start
Provides a guide and a toy example of how to create a repository and generate some results.
The toy example shows the computation of Fibonacci numbers by means of result of components `fib_zero`, `fib_one`, and `fib_next`.

## 1. Define Component Specifications

Using domain-specific language provided by the `SpecificationBuilder` class define a triple of named components and respective specifications `(name, interpretation, specification)`.

- The component `fib_zero` is specified by `Constructor("fib") & Constructor("at", Literal(0, "int"))`, which combines two properties.
  + `Constructor("fib")` means that `fib_zero` it is a Fibonacci number.
  + `Constructor("at", Literal(0))` means that `fib_zero` is associated with index `0`.
- The component `fib_one`, similarly to `fib_zero`, is a Fibonacci number and is associated with index `1`.
- The component `fib_next` has three parameters associated with the DataGroup `int`. In the toy example, indices less than `20` are considered.
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

named_components_with_specifications = [
        (
            "fib_zero",
            fib_zero,
            SpecificationBuilder()
            .suffix(Constructor("fib") & Constructor("at", Literal(0))),
        ),
        (
            "fib_one",
            fib_one,
            SpecificationBuilder()
            .suffix(Constructor("fib") & Constructor("at", Literal(1))),
        ),
        (
            "fib_next",
            fib_next,
            SpecificationBuilder()
            .parameter("z", DataGroup("int", range(bound)))
            .parameter("y", DataGroup("int", range(bound)), lambda vs: [vs["z"] - 1])
            .parameter("x", DataGroup("int", range(bound)), lambda vs: [vs["z"] - 2])
            .argument("f1", Constructor("fib") & Constructor("at", Var("y")))
            .argument("f2", Constructor("fib") & Constructor("at", Var("x")))
            .suffix(Constructor("fib") & Constructor("at", Var("z"))),
        ),
    ]
```

## 2. Instantiate CoSy

Let the `Maestro` know what he is working with; by providing the named components with their specifications.

```
maestro = Maestro(component_specifications)
```

## 3. Specify a Target and Construct Results

Specify the target for which results should be found.
Results are found by means of instantiation and result of the given components in the given parameter space ().

### Arbitrary Fibonacci numbers

The following target `Constructor("fib")` describes arbitrary Fibonacci numbers at indices in the given parameter space.

```
target = Constructor("fib")
```

Using the `design` method, iterate over and display results for the given target.

```
for result in maestro.query(target):
    print(result)
```

### Fibonacci numbers at Specific Indices

The specification allows us to target Fibonacci numbers at specific indices.
For an index `i` the target `Constructor("fib") & Constructor("at", Literal(i))` describes the Fibonacci number at index `i`.
Using the `design` method, construct and display this Fibonacci number.

```
for i in range(20):
    target = Constructor("fib") & Constructor("at", Literal(i))
    print(i, next(iter(maestro.solve(target))))
```