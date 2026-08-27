"""Gumbel keys and their top-down conditioning: the randomness of random search.

Random search is a best-first search whose cost function is a random key per inhabitant. Two facts
make that work. A *Gumbel key* ``kappa(x) = log w(x) + g_x``, with independent ``g_x ~ Gumbel(0)``,
puts the greatest key on ``x`` with probability ``w(x) / W``, and listing a weighted set by
decreasing key is a sample drawn without replacement in proportion to ``w``. And the maximum of
such keys is itself Gumbel, with location ``log W``. The second fact is what lets the search avoid
drawing a key per inhabitant: it draws one key per *node*, from the location the branch counts
compute, and conditions the children of an expansion on the key their parent already carries. The
construction follows Maddison et al. and Kool et al.

The conditioning is the delicate part. Given the parent key ``T``, independent child keys
``kappa_i`` and their maximum ``Z``, the shifted keys are

    kappa~_i = -log( exp(-T) - exp(-Z) + exp(-kappa_i) ),

a strictly increasing map that sends ``Z`` to ``T`` and leaves the ranking of the children intact.
Evaluated in that form it cancels catastrophically as soon as the exponentials differ by orders of
magnitude, so :func:`condition_on_maximum` uses the algebraically equal but stable form of Kool
et al.

Everything here runs on ``random.Random`` alone. cosy carries no runtime dependency, so the Gumbel
draw is the inverse-CDF transform ``-log(-log(U))`` of one uniform draw rather than a library call.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import random
    from collections.abc import Sequence

__all__ = ["condition_on_maximum", "gumbel_key", "gumbel_noise"]

_LOG2 = math.log(2.0)


def gumbel_noise(rng: random.Random) -> float:
    """Draw one ``g ~ Gumbel(0)``.

    The Gumbel distribution with location 0 has the cumulative distribution
    ``P(g <= x) = exp(-exp(-x))``, so ``g = -log(-log(U))`` for ``U`` uniform on (0, 1) is a draw
    from it. ``random()`` returns a value in [0, 1), and 0 has no image under the transform, so a
    zero draw is *redrawn*. Substituting a value for it would bias the sample without any trace.

    Args:
        rng (random.Random): The source of randomness.

    Returns:
        float: One draw from Gumbel(0).
    """
    uniform = rng.random()
    while uniform == 0.0:
        uniform = rng.random()
    return -math.log(-math.log(uniform))


def gumbel_key(log_weight: float, rng: random.Random) -> float:
    """Draw the Gumbel key of an element of the given log-weight.

    The key is ``kappa(x) = log w(x) + g_x``. Adding a constant shifts the location, so the key
    follows ``Gumbel(log w(x))``. The argument is the *logarithm* of the weight because that is
    what the branch counts deliver: a node's weight is a sum over cost values whose terms underflow
    long before their logarithm does.

    Args:
        log_weight (float): ``log w(x)``, the location of the key's distribution.
        rng (random.Random): The source of randomness.

    Returns:
        float: One draw from ``Gumbel(log_weight)``.
    """
    return log_weight + gumbel_noise(rng)


def _log1mexp(value: float) -> float:
    """Return ``log(1 - exp(value))`` for ``value <= 0``, accurately at both ends.

    Args:
        value (float): A non-positive number.

    Returns:
        float: ``log(1 - exp(value))``, and ``-inf`` at ``value == 0``.
    """
    if value >= 0.0:
        return -math.inf
    if value > -_LOG2:
        return math.log(-math.expm1(value))
    return math.log1p(-math.exp(value))


def condition_on_maximum(parent_key: float, log_weights: Sequence[float], rng: random.Random) -> list[float]:
    """Draw the keys of one expansion's children, conditioned on their parent's key.

    A node's key is the greatest key among the inhabitants below it, so the children of an
    expansion must have exactly that key as their maximum. This draws one independent
    ``Gumbel(log_weights[i])`` per child and shifts the draws so that their maximum becomes
    ``parent_key``, which realizes the conditional distribution of the children keys given the
    parent key. The shift is strictly increasing, so the ranking of the unshifted draws survives.
    That is what makes the top-down keys reproduce the order a key-per-inhabitant draw would have
    produced.

    Args:
        parent_key (float): The key the parent node carries, and the maximum of the result.
        log_weights (Sequence[float]): One log-weight per child, the ``log`` of the total weight of
            the inhabitants below it. Children of weight 0 carry no key at all and must be dropped
            by the caller before this point.
        rng (random.Random): The source of randomness.

    Returns:
        list[float]: One key per child, in the order of ``log_weights``, with maximum
            ``parent_key``.

    Raises:
        ValueError: If ``log_weights`` is empty, since a node without children has no key to hand
            down and an empty result would read as a successfully expanded node, or if a log-weight
            is not finite.
    """
    if not log_weights:
        msg = "an expansion passes its key to at least one child"
        raise ValueError(msg)
    # A child of weight 0 has log-weight -inf, and the shift below turns that into a NaN key rather
    # than into an error: the comparisons a NaN key takes part in are all false, so it would sink to
    # an arbitrary place in the frontier and stay there. The caller drops those children, since a
    # node without inhabitants below it carries no key at all, so reaching this point with one is a
    # broken contract and not a value to be repaired.
    if any(not math.isfinite(log_weight) for log_weight in log_weights):
        msg = f"every child of an expansion carries a finite log-weight, but got {list(log_weights)}"
        raise ValueError(msg)

    drawn = [gumbel_key(log_weight, rng) for log_weight in log_weights]
    maximum = max(drawn)
    shifted: list[float] = []
    for key in drawn:
        # v is +inf-free by construction: key <= maximum, and key == maximum gives -inf, which the
        # two terms below turn into exactly parent_key, the child that held the maximum.
        difference = parent_key - key + _log1mexp(key - maximum)
        shifted.append(parent_key - max(difference, 0.0) - math.log1p(math.exp(-abs(difference))))
    return shifted
