"""The Gumbel building blocks of random search.

Random search draws one key per inhabitant and streams by decreasing key, which makes every prefix
a sample without replacement. The keys are never drawn one per inhabitant, though: the greatest key
below a node is itself Gumbel with a location the branch counts compute, so the search draws node
keys top-down, each child conditioned on its parent's key. That conditioning is the single most
error-prone piece of the package, so this file pins the blocks *in isolation*, with no search space
and no counting, before the tree machinery uses them.

Exact invariants only. The distributional claims, which are the argmax frequency, the Gumbel(log W)
law of the maximum, and the Plackett-Luce law of the decreasing order, need statistics and are not
made here. What is checkable without statistics is the inverse-CDF identity, the conditioning
identity, and the order preservation the sampling guarantee relies on.
"""

import decimal
import math
import random

import pytest

from cosy.search.gumbel import _LOG2, _log1mexp, condition_on_maximum, gumbel_key, gumbel_noise


class FixedUniform(random.Random):
    """A generator whose ``random()`` replays a fixed sequence.

    Lets a test read the inverse-CDF identity off a known uniform draw instead of a statistic.
    """

    def __new__(cls, values) -> "FixedUniform":  # noqa: ARG004
        """Construct without letting the base class seed itself from the scripted draws.

        Up to Python 3.10 ``_random.Random.__new__`` forwards whatever the constructor received to
        ``seed()``, which hashes its argument. A sequence of draws is not hashable, so on 3.10 the
        class raises ``TypeError`` before ``__init__`` ever runs. From 3.11 on the arguments are no
        longer forwarded, which is why the difference shows on one row of the matrix alone.

        Args:
            values: The scripted draws. Consumed by ``__init__``, ignored here.

        Returns:
            FixedUniform: A fresh instance. Its inherited state is seeded from the default source
                and never read, since ``random`` is overridden.
        """
        return super().__new__(cls)

    def __init__(self, values) -> None:
        """Replay the given uniform draws in order.

        Args:
            values: The values ``random()`` returns, in order.
        """
        super().__init__()
        self._values = list(values)
        self._index = 0

    def random(self) -> float:
        """Return the next scripted uniform draw.

        Returns:
            float: The next value of the sequence.

        Raises:
            AssertionError: If the sequence is exhausted.
        """
        if self._index >= len(self._values):
            msg = "the scripted uniform sequence is exhausted"
            raise AssertionError(msg)
        value = self._values[self._index]
        self._index += 1
        return value


# ---------------------------------------------------------------------------
# The scripted generator
# ---------------------------------------------------------------------------


def test_the_scripted_generator_replays_its_sequence_and_then_stops():
    """The helper the tests below build on is constructible from a list, in order, and finite.

    Constructibility is what carries a version difference. ``FixedUniform`` inherits from
    ``random.Random``, whose ``__new__`` seeds from the constructor arguments up to Python 3.10,
    and a list of draws cannot be hashed. Without the ``__new__`` on the class, every test that
    scripts its draws fails on that one row of the matrix with a ``TypeError`` naming neither the
    helper nor the version.
    """
    rng = FixedUniform([0.25, 0.5])

    assert rng.random() == 0.25
    assert rng.random() == 0.5
    with pytest.raises(AssertionError, match="exhausted"):
        rng.random()


# ---------------------------------------------------------------------------
# The noise and the key
# ---------------------------------------------------------------------------


def test_gumbel_noise_is_the_inverse_cdf_of_a_uniform_draw():
    """``g = -log(-log(U))`` is the inverse of ``P(g <= x) = exp(-exp(-x))``.

    Dependency-free sampling from Gumbel(0) is exactly this transform of one uniform draw, so the
    identity is checkable against a scripted generator without any statistic.
    """
    draws = [0.25, 0.5, 0.75]
    rng = FixedUniform(draws)
    for uniform in draws:
        assert gumbel_noise(rng) == pytest.approx(-math.log(-math.log(uniform)))


def test_gumbel_noise_never_divides_by_a_zero_draw():
    """A uniform draw of exactly 0 has no image under the transform and must not reach it.

    ``random.Random.random()`` returns a value in [0, 1), so 0 is attainable and ``-log(0)`` would
    raise. The block redraws instead of substituting a value. A substituted key would bias the
    sample silently, which is the one failure mode a sampler must not have.
    """
    rng = FixedUniform([0.0, 0.0, 0.25])
    assert gumbel_noise(rng) == pytest.approx(-math.log(-math.log(0.25)))


def test_gumbel_key_shifts_the_location_by_the_log_weight():
    """``kappa(x) = log w(x) + g_x``, so the key follows Gumbel(log w(x))."""
    rng = FixedUniform([0.5, 0.5])
    noise = gumbel_noise(FixedUniform([0.5]))
    assert gumbel_key(math.log(3.0), rng) == pytest.approx(math.log(3.0) + noise)
    assert gumbel_key(0.0, rng) == pytest.approx(noise)


def test_gumbel_key_is_seed_deterministic():
    """Same seed, same keys, which is what makes every search built on the block reproducible."""
    first = [gumbel_key(math.log(2.0), random.Random(11)) for _ in range(3)]
    second = [gumbel_key(math.log(2.0), random.Random(11)) for _ in range(3)]
    assert first == second


# ---------------------------------------------------------------------------
# Top-down conditioning: the children keys given the parent key
# ---------------------------------------------------------------------------


def test_conditioning_makes_the_parent_key_the_maximum():
    """The shifted children attain the parent key exactly, and none exceeds it.

    This is the defining property of the construction. The greatest key below a node *is* the
    node's key, so the children of an expansion must have that key as their maximum, not
    approximately but as the value one of them takes.
    """
    rng = random.Random(3)
    parent = -0.75
    log_weights = [math.log(1.0), math.log(4.0), math.log(0.5)]
    shifted = condition_on_maximum(parent, log_weights, rng)
    assert max(shifted) == pytest.approx(parent)
    assert all(key <= parent + 1e-12 for key in shifted)


def test_conditioning_preserves_the_order_of_the_drawn_keys():
    """The shift is strictly increasing, so it permutes nothing.

    ``kappa -> -log(exp(-T) - exp(-Z) + exp(-kappa))`` is strictly monotone, which is why the
    top-down keys reproduce the ranking a per-inhabitant draw would have produced.

    Reading the unshifted draws off a second generator relies on the contract that the block draws
    one key per child, in the order of the weights, the same contract that makes a seeded search
    reproducible.
    """
    log_weights = [math.log(w) for w in (1.0, 2.0, 3.0, 4.0, 5.0)]
    unconditioned_rng = random.Random(5)
    unconditioned = [gumbel_key(weight, unconditioned_rng) for weight in log_weights]
    shifted = condition_on_maximum(-2.0, log_weights, random.Random(5))
    assert sorted(range(len(shifted)), key=lambda i: shifted[i]) == sorted(
        range(len(unconditioned)), key=lambda i: unconditioned[i]
    )


def test_conditioning_a_single_child_returns_the_parent_key():
    """One child carries the whole weight of its parent, so it inherits the key unchanged.

    A chain of unary expansions is the common case in a recursive space, and any drift here would
    accumulate along the chain.
    """
    rng = random.Random(7)
    shifted = condition_on_maximum(1.25, [math.log(2.0)], rng)
    assert shifted == [pytest.approx(1.25)]


def closed_form_conditioning(parent_key, drawn):
    """Shift the drawn keys onto ``parent_key`` by the closed form of the construction.

    The form the construction is defined by: with ``T`` the parent key and ``Z`` the maximum of the
    drawn keys, the conditioned key of a child is ``-log(exp(-T) - exp(-Z) + exp(-kappa))``. It is
    written here as the reference the implementation is bound to, and it is *not* what the
    implementation computes. The implementation uses an algebraically equal but numerically
    well-behaved rearrangement, and this test is what says the two agree.

    Args:
        parent_key (float): The key of the parent node.
        drawn (Sequence[float]): The unconditioned keys of the children.

    Returns:
        list[float]: One conditioned key per child.
    """
    maximum = max(drawn)
    return [-math.log(math.exp(-parent_key) - math.exp(-maximum) + math.exp(-key)) for key in drawn]


def test_conditioning_is_the_closed_form_of_the_construction():
    """The implementation computes the closed form, not merely something with the right maximum.

    The trap this closes: shifting every drawn key by the *same* additive constant
    ``parent_key - max(drawn)`` also has maximum ``parent_key``, also preserves the order, and also
    survives every structural assertion in this file, while producing the wrong conditional
    distribution. Only pinning the values themselves separates the two, and on moderate magnitudes
    the closed form is numerically harmless, so it can serve as the oracle.

    The draws are scripted, so the comparison is exact rather than statistical: the keys the
    implementation draws are ``log w_i + (-log(-log u_i))`` for the scripted ``u_i``.
    """
    log_weights = [math.log(w) for w in (1.0, 2.0, 7.0, 0.5)]
    uniforms = [0.1, 0.4, 0.75, 0.9]
    parent_key = 3.5

    drawn = [
        log_weight + (-math.log(-math.log(uniform))) for log_weight, uniform in zip(log_weights, uniforms, strict=True)
    ]
    expected = closed_form_conditioning(parent_key, drawn)

    shifted = condition_on_maximum(parent_key, log_weights, FixedUniform(uniforms))
    assert shifted == pytest.approx(expected, rel=1e-12)
    assert max(shifted) == pytest.approx(parent_key)


def test_conditioning_is_the_closed_form_for_far_apart_weights():
    """The agreement holds where the weights span many orders of magnitude.

    Same oracle, weights from ``1e-12`` to ``1e12``: the parent key stays moderate, so the closed
    form is still evaluable and the two must agree to the last digits.
    """
    log_weights = [math.log(10.0**exponent) for exponent in (-12, -6, 0, 6, 12)]
    uniforms = [0.05, 0.3, 0.5, 0.65, 0.95]
    parent_key = 30.0

    drawn = [
        log_weight + (-math.log(-math.log(uniform))) for log_weight, uniform in zip(log_weights, uniforms, strict=True)
    ]
    expected = closed_form_conditioning(parent_key, drawn)

    shifted = condition_on_maximum(parent_key, log_weights, FixedUniform(uniforms))
    assert shifted == pytest.approx(expected, rel=1e-12)


def test_conditioning_stays_usable_where_the_closed_form_overflows():
    """Below about ``-709`` the closed form overflows, and the implementation must not.

    ``exp(-T)`` is what breaks: at ``T = -740`` it is past the largest double. Keys that small are
    not exotic, since a key is ``log w + g`` and in a deep space with a wide bound the weight of a
    single inhabitant is easily ``1e-300`` and below, so the implementation evaluates the
    rearranged form, which never exponentiates a key on its own.

    The weights here are all of that order, which is the state the construction actually produces:
    a parent key sits just above the largest of its children's, never far below them. Given far
    *smaller* parent keys than children's locations, a state no expansion reaches, all children
    collapse onto the parent key in double precision, and that is the arithmetic rather than the
    implementation, since the true values then differ beyond the 300th digit.

    The magnitude is chosen to make the test discriminating. At ``-500`` with moderate weights the
    closed form still evaluates and every child collapses onto the parent key, so an implementation
    that simply returned ``[parent_key] * n`` would pass.
    """
    log_weights = [-750.0 + offset for offset in (-12.0, -6.0, 0.0, 6.0, 12.0)]
    uniforms = [0.05, 0.3, 0.5, 0.65, 0.95]

    drawn = [
        log_weight + (-math.log(-math.log(uniform))) for log_weight, uniform in zip(log_weights, uniforms, strict=True)
    ]
    parent_key = max(drawn) + 0.7  # what an expansion hands down: just above the children

    with pytest.raises(OverflowError):
        closed_form_conditioning(parent_key, drawn)

    shifted = condition_on_maximum(parent_key, log_weights, FixedUniform(uniforms))
    assert all(math.isfinite(key) for key in shifted)
    assert max(shifted) == pytest.approx(parent_key)
    assert len(set(shifted)) == len(shifted), "the children stay distinguishable"
    assert sorted(shifted, reverse=True) == [
        key for _, key in sorted(zip(drawn, shifted, strict=True), reverse=True)
    ], "the conditioning preserves the order of the draws"


@pytest.mark.parametrize(
    "log_weights",
    [[-math.inf], [-math.inf, -math.inf], [0.0, -math.inf], [math.inf], [math.nan]],
)
def test_conditioning_rejects_a_child_without_a_finite_weight(log_weights):
    """A weightless child is a broken contract, not a value to be repaired.

    The shift turns a ``-inf`` log-weight into a NaN key, and a NaN key compares false against
    everything: it would take an arbitrary place in the frontier instead of raising. The caller
    drops children that carry no key, so this is unreachable through the search, but the function
    is public API and a silent NaN is exactly the substituted measurement the repository forbids.

    Args:
        log_weights (list[float]): The non-finite log-weights under test.
    """
    with pytest.raises(ValueError, match="finite log-weight"):
        condition_on_maximum(0.0, log_weights, random.Random(3))


def test_conditioning_rejects_an_empty_expansion():
    """A node without children has no key to hand down, and the caller must not ask.

    Returning an empty list would let a caller treat a dead node as expanded.

    Raises:
        ValueError: Always, which is what the test asserts.
    """
    with pytest.raises(ValueError, match="at least one child"):
        condition_on_maximum(0.0, [], random.Random(1))


@pytest.mark.parametrize("value", [-1e-18, -1e-12, -1e-6, -0.1, -0.5, -_LOG2, -1.0, -5.0, -50.0, -700.0])
def test_log1mexp_holds_at_both_ends_of_its_range(value):
    """``log(1 - exp(v))`` computed accurately whether ``exp(v)`` is near 1 or near 0.

    The whole point of the two branches is accuracy. Near zero ``1 - exp(v)`` cancels, and only
    ``expm1`` keeps the digits. Far from zero ``exp(v)`` is tiny and ``log1p`` is what keeps them.
    Evaluating either branch over the other's range loses the value silently, so the reference here
    is computed in exact decimal arithmetic rather than by the other branch.

    Args:
        value (float): The non-positive argument under test.
    """
    with decimal.localcontext() as context:
        context.prec = 60
        exact = (decimal.Decimal(1) - decimal.Decimal(value).exp()).ln()

    assert _log1mexp(value) == pytest.approx(float(exact), rel=1e-12)


def test_log1mexp_is_minus_infinity_where_the_difference_vanishes():
    """At ``v = 0`` the argument of the logarithm is 0, which has no finite image."""
    assert _log1mexp(0.0) == -math.inf
    assert _log1mexp(1.0) == -math.inf


def test_conditioning_is_seed_deterministic():
    """Same seed and same weights, same conditioned keys."""
    log_weights = [math.log(w) for w in (1.0, 2.0, 3.0)]
    first = condition_on_maximum(-1.0, log_weights, random.Random(23))
    second = condition_on_maximum(-1.0, log_weights, random.Random(23))
    assert first == second
