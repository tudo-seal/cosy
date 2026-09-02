"""The fitness layer: the partial order to compare in, and the scalarization beside it.

A fitness function maps into a **partially ordered set**, so ``compare`` is the primary interface
and it has four answers, ``INCOMPARABLE`` among them. The map into the reals sits on
:class:`Scalarization`, a component of its own, and no comparator carries one. On the Pareto
comparator such a map would be a weighted sum, which turns the Pareto front into one number and
chooses a point on it by the weights.

A scalarization is required to be strictly positive and monotone. Neither is enforceable for an
arbitrary callable, so both are pinned here for the instance the module ships, including the place
where positivity fails numerically, which is a limit worth knowing rather than patching.
"""

import math

import pytest

from cosy.core.tree import Tree
from cosy.evolutionary_algorithms import (
    Comparison,
    ExpScalarization,
    FitnessComparator,
    ParetoFitnessComparator,
    ScalarFitnessComparator,
    Scalarization,
    induced_fitness,
)
from tests.ea_fixtures import bi, lf, un


class _Float:
    """A float-like scalar with no length, standing in for ``numpy.float32``.

    Attributes:
        value (float): The wrapped number.
    """

    def __init__(self, value: float) -> None:
        """Wrap a number.

        Args:
            value (float): The number.
        """
        self.value = value

    def __float__(self) -> float:
        """Return the wrapped number.

        Returns:
            float: The number.
        """
        return self.value


# ---------------------------------------------------------------------------
# ScalarFitnessComparator: a total order
# ---------------------------------------------------------------------------


def test_comparison_follows_the_direction():
    """Greater is fitter under maximization, and the reverse under minimization."""
    assert ScalarFitnessComparator(greater_is_better=True).compare(2.0, 1.0) is Comparison.GREATER
    assert ScalarFitnessComparator(greater_is_better=False).compare(2.0, 1.0) is Comparison.LESS


def test_equal_values_tie_rather_than_being_incomparable():
    """A tie is a statement about two values, and incomparability is a refusal to make one."""
    assert ScalarFitnessComparator().compare(1.0, 1.0) is Comparison.EQUAL


def test_a_total_order_never_answers_incomparable():
    """Under a scalar comparator every pair is ranked."""
    comparator = ScalarFitnessComparator()
    values = [-math.inf, -2.0, 0.0, 1.5, math.inf]
    for first in values:
        for second in values:
            assert comparator.compare(first, second) is not Comparison.INCOMPARABLE


def test_comparison_is_antisymmetric():
    """Swapping the arguments swaps the verdict."""
    comparator = ScalarFitnessComparator()
    for first, second in [(1.0, 2.0), (2.0, 1.0), (3.0, 3.0)]:
        forward = comparator.compare(first, second)
        backward = comparator.compare(second, first)
        if forward is Comparison.EQUAL:
            assert backward is Comparison.EQUAL
        else:
            assert forward is not backward


def test_integers_and_one_element_sequences_are_scalars():
    """A fitness may arrive as an int or as a one-element vector."""
    comparator = ScalarFitnessComparator()
    assert comparator.compare(2, 1) is Comparison.GREATER
    assert comparator.compare([2.0], [1.0]) is Comparison.GREATER


def test_a_multi_objective_fitness_is_refused_rather_than_truncated():
    """Reading objective 0 and discarding the rest optimizes a different problem."""
    with pytest.raises(ValueError, match=r"two or more|2 objectives"):
        ScalarFitnessComparator().compare([1.0, 2.0], [1.0, 1.0])


def test_the_message_points_at_the_comparator_that_can_do_it():
    """The error names the way out."""
    with pytest.raises(ValueError, match="ParetoFitnessComparator"):
        ScalarFitnessComparator().compare([1.0, 2.0], 1.0)


def test_a_float_like_scalar_without_a_length_is_not_mistaken_for_a_vector():
    """The test is whether a value has a length, not whether it is a ``Sequence``."""
    assert ScalarFitnessComparator().compare(_Float(2.0), _Float(1.0)) is Comparison.GREATER


def test_nan_is_incomparable_rather_than_ranked():
    """A failed measurement stands in no order relation, not even to itself.

    The raw float comparisons answer ``LESS`` in **both** directions, which breaks antisymmetry,
    and a driver reading that finds nothing ever fitter than a failed measurement, so the first one
    to appear holds the best-so-far place for the rest of the run. Reporting incomparability is
    what the value actually is. Refusing it is then the driver's job, not the order's.
    """
    comparator = ScalarFitnessComparator()
    assert comparator.compare(math.nan, 1.0) is Comparison.INCOMPARABLE
    assert comparator.compare(1.0, math.nan) is Comparison.INCOMPARABLE


def test_nan_is_not_even_comparable_to_itself():
    """Reflexivity is what a partial order has and ``nan`` has not.

    This is the test the driver runs on every measured value: a fitness outside the codomain of
    the fitness function is refused there, naming the individual.
    """
    assert ScalarFitnessComparator().compare(math.nan, math.nan) is Comparison.INCOMPARABLE
    assert ParetoFitnessComparator().compare([1.0, math.nan], [1.0, math.nan]) is Comparison.INCOMPARABLE


# ---------------------------------------------------------------------------
# ParetoFitnessComparator: a genuine partial order
# ---------------------------------------------------------------------------


def test_dominance_in_every_objective():
    """Better everywhere is greater."""
    assert ParetoFitnessComparator().compare([2.0, 3.0], [1.0, 2.0]) is Comparison.GREATER
    assert ParetoFitnessComparator().compare([1.0, 2.0], [2.0, 3.0]) is Comparison.LESS


def test_at_least_as_good_everywhere_and_better_once():
    """The standard definition of dominance."""
    assert ParetoFitnessComparator().compare([2.0, 2.0], [2.0, 1.0]) is Comparison.GREATER


def test_a_trade_off_is_incomparable_rather_than_a_tie():
    """The point of the partial order: two vectors that trade objectives are not ranked.

    A comparator that summed the objectives would answer ``EQUAL`` here, which is a claim the order
    does not make.
    """
    assert ParetoFitnessComparator().compare([2.0, 1.0], [1.0, 2.0]) is Comparison.INCOMPARABLE


def test_identical_vectors_tie():
    """Agreement in every objective is a tie."""
    assert ParetoFitnessComparator().compare([1.0, 2.0], [1.0, 2.0]) is Comparison.EQUAL


def test_a_minimized_objective_reverses():
    """``maximize`` flags choose the direction per objective."""
    comparator = ParetoFitnessComparator(maximize=[True, False])
    assert comparator.compare([2.0, 1.0], [1.0, 2.0]) is Comparison.GREATER


def test_a_flag_count_that_does_not_match_is_refused():
    """A flag list of the wrong length is a configuration error."""
    with pytest.raises(ValueError, match="objectives"):
        ParetoFitnessComparator(maximize=[True]).compare([1.0, 2.0], [1.0, 1.0])


def test_the_pareto_comparator_carries_no_scalarization():
    """Summing objectives is not part of the order.

    Where numbers are genuinely needed, a :class:`Scalarization` is passed explicitly, so that
    choosing one is a visible decision rather than a default hidden in a comparator.
    """
    assert not hasattr(ParetoFitnessComparator(), "scalarize")
    assert not hasattr(ParetoFitnessComparator(), "sort_key")


def test_both_comparators_satisfy_the_protocol():
    """The component class is structural."""
    assert isinstance(ScalarFitnessComparator(), FitnessComparator)
    assert isinstance(ParetoFitnessComparator(), FitnessComparator)


# ---------------------------------------------------------------------------
# Scalarization
# ---------------------------------------------------------------------------


def test_the_exponential_is_strictly_positive():
    """A scalarization maps into the strictly positive reals, however bad the fitness."""
    scalarization = ExpScalarization()
    for value in [-50.0, -1.0, 0.0, 1.0, 50.0]:
        assert scalarization.scalarize(value) > 0.0


def test_the_exponential_is_monotone_in_the_fitness_order():
    """Monotonicity: a fitter value never scalarizes lower, in either direction."""
    maximizing = ExpScalarization(greater_is_better=True)
    assert maximizing.scalarize(1.0) < maximizing.scalarize(2.0)
    minimizing = ExpScalarization(greater_is_better=False)
    assert minimizing.scalarize(1.0) > minimizing.scalarize(2.0)


def test_a_constant_shift_of_the_fitness_leaves_the_ratios_untouched():
    """The property the exponential was chosen for.

    Raw values as roulette weights make every share depend on where the zero of the fitness sits,
    so a lift applied to bring negative values into range moves the shares with it. The exponential
    maps the *differences* to ratios, which a shift of the whole population leaves alone.
    """
    scalarization = ExpScalarization()
    ratio = scalarization.scalarize(3.0) / scalarization.scalarize(1.0)
    shifted = scalarization.scalarize(103.0) / scalarization.scalarize(101.0)
    assert math.isclose(ratio, shifted)


def test_the_exponential_underflows_below_the_float_range():
    """The numerical limit of positivity, stated rather than patched.

    Positivity holds in the reals, while ``math.exp`` returns 0.0 below about -745. A component
    that draws proportionally then sees a weight of zero, which is why it refuses such a value
    loudly instead of flooring it. The fix is to scale the fitness.
    """
    assert ExpScalarization().scalarize(-745.0) > 0.0
    assert ExpScalarization().scalarize(-746.0) == 0.0


def test_the_scale_brings_a_large_fitness_back_into_range():
    """The parameter that makes the advice in the docstring actionable.

    A mean squared error of 800 is an ordinary fitness. Under a minimizing order the exponential
    underflows on it, and the proportional draw refuses a weight of zero. Dividing by the scale
    keeps the order and moves the values into range.
    """
    plain = ExpScalarization(greater_is_better=False)
    assert plain.scalarize(800.0) == 0.0
    scaled = ExpScalarization(greater_is_better=False, scale=100.0)
    assert scaled.scalarize(800.0) > 0.0
    assert scaled.scalarize(700.0) > scaled.scalarize(800.0)


def test_a_scale_that_would_break_monotonicity_is_refused():
    """Zero divides and a negative scale reverses the order."""
    for scale in (0.0, -1.0):
        with pytest.raises(ValueError, match="strictly positive"):
            ExpScalarization(scale=scale)


def test_the_exponential_reports_overflow_rather_than_returning_infinity():
    """Overflow on a finite fitness is reported rather than passed on as infinity.

    A proportional draw reads infinity as an infinitely good individual and gives it the whole
    mass, so a finite fitness must not produce it.
    """
    with pytest.raises(ValueError, match="overflows"):
        ExpScalarization().scalarize(1e6)


def test_a_scale_that_overflows_the_division_is_reported_too():
    """A tiny scale makes the division overflow, not the exponential.

    A float division overflows to infinity without raising, so the quotient is already infinite
    when ``math.exp`` receives it. The refusal reads the result and catches this the same way.
    """
    with pytest.raises(ValueError, match="overflows"):
        ExpScalarization(scale=1e-320).scalarize(1e300)


def test_an_infinitely_good_fitness_stays_infinite():
    """An infinitely good fitness scalarizes to infinity under either direction of the order.

    Which value is infinitely good depends on the direction: ``inf`` under a maximizing order and
    ``-inf`` under a minimizing one. The other infinity is infinitely bad and scalarizes to zero,
    which the drawing component refuses.
    """
    assert ExpScalarization().scalarize(math.inf) == math.inf
    assert ExpScalarization(greater_is_better=False).scalarize(-math.inf) == math.inf


def test_a_failed_measurement_stays_a_failed_measurement():
    """``nan`` in, ``nan`` out: no substitute value."""
    assert math.isnan(ExpScalarization().scalarize(math.nan))


def test_the_scalarization_satisfies_the_protocol():
    """The component class is structural."""
    assert isinstance(ExpScalarization(), Scalarization)


def test_the_refusal_from_the_scalarization_names_a_scalarization():
    """The refusal from a scalarization names a scalarization as the way out.

    In a run, a two-objective fitness reaches ``ExpScalarization`` only after the run's comparator
    accepted it, and ``ScalarFitnessComparator`` does not. The comparator is therefore not what is
    missing, and pointing at one would send the caller back to a component that already works.
    """
    with pytest.raises(ValueError, match="Scalarization") as refusal:
        ExpScalarization().scalarize([1.0, 2.0])
    assert "ParetoFitnessComparator" not in str(refusal.value)


# ---------------------------------------------------------------------------
# Fitness algebras
# ---------------------------------------------------------------------------


def test_the_induced_fitness_is_the_fold():
    """The fitness induced by a fitness algebra is the fold into that algebra."""
    counting = {lf: 1, un: lambda child: child + 1, bi: lambda left, right: left + right + 1}
    individual = Tree(bi, (Tree(un, (Tree(lf, ()),)), Tree(lf, ())))
    assert induced_fitness(counting)(individual) == 4


def test_the_induced_fitness_defaults_to_the_symbols_own_meaning():
    """Without an interpretation the repository's own callables are the algebra."""
    individual = Tree(un, (Tree(lf, ()),))
    assert induced_fitness()(individual) == "u(.)"
