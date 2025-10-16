# Constraints
There are certain aspects of specifications that are difficult to represent nominally.
Such aspects range from simple relationships between numeric parameters to complex performance characteristics of solutions.

The most general way to specify such aspects is via the `constraint` method provided by the `DSL`.

In the following example, we construct binary sequences that match a given regular expression.

First, we specify binary sequences, which are possibly empty sequences of `0`s and `1`s.

```
def empty() -> str:
    return ""

def zero(s: str) -> str:
    return s + "0"

def one(s: str) -> str:
    return s + "1"
```
with the following specifications: 
```
empty: DSL().suffix(Constructor("str")),

zero: DSL()
    .argument("s", Constructor("str"))
    .suffix(Constructor("str")),

one: DSL()
    .argument("s", Constructor("str"))
    .suffix(Constructor("str")),
```

Then, we specify when such sequences match a given regular expression using a `constraint`.
The component `fin` does not change a given sequence `s`.
However, in its specification, `s` matches the regular expression given by the parameter `r`.
This exposes a computed property (corresponding regular expression) as a nominal specification (parameter `r`).

```
def fin(_b: bool, s: str) -> str:
    return s
```
with specification: 
``` hl_lines="4"
fin: DSL()
    .parameter("r", "regular_expression")
    .argument("s", Constructor("str"))
    .constraint(lambda vs: re.fullmatch(vs["r"], vs["s"].interpret())) # (1)!
    .suffix(Constructor("matches", Var("r"))),
```

1. A parameter constraint is used to ensure that `s` matches the regular expression `r`. 

In the body of the constraint, parameters are given as their values and arguments are given as their tree representation, which can be interpreted.
Therefore, the value of the parameter `r` is `vs["r"]`.
The interpreted value of the argument `s` is `vs["s"].interpret()`

Using the above specifications, we can construct sequences that match the specified regular expressions:
```
class RegularExpression(Container): # (1)!
    def __contains__(self, value: object) -> bool:
        return isinstance(value, str)

parameter_space = {"regular_expression": RegularExpression()}
cosy = CoSy(component_specifications, parameter_space) # (2)!
query: Type = Constructor("matches", Literal("01+0")) # (3)!
 
for solution in cosy.solve(query): # (4)!
    print(solution)
```

1. For simplicity, regular expressions are assumed to be arbitrary strings. 
2. The CoSy instance with the component specifications and parameter space.
3. A Query for sequences matching the regular expression "01+0"
4. Solve the query and print all solutions. 

The above results in: `010`, `0110`, `01110`, `011110`, ...

### Remarks

- In the above example, we are free to change the particular regular expression in the `query`.

- The specialized `parameter_constraint` method provided by the `DSL` can speed up synthesis if the constraint *only* involves parameters (and not arguments).
