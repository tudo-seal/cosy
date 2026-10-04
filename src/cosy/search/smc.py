"""Sequential Monte Carlo over the lazy search tree: particles guided by a construction's weights, corrected to its target.

Random search on estimated weights draws in proportion to the estimates, not to the target: where the estimates of
siblings do not add up to their parent's, the draw is off by as much (the saddle search, :func:`cosy.search.tilt.saddle_search`,
is one such construction). Sequential Monte Carlo keeps the estimates as a GUIDE and corrects for them with weights.

**A particle** starts at the root of the search tree and walks to a term, one expansion per step, choosing each child in
proportion to the guide's weight ``h(child)``; it carries a weight, multiplied at every step by ``sum h(children) / h(node)``,
and at the term by ``gamma(t) / h(t)``, ``gamma`` the construction's target weight of the term. Along a path the factors
telescope with the path's probability, ``prod h(next) / sum h(children)``: a particle's weight is

    w  =  gamma(t) / h(root) / q(path),

so the weighted particles follow ``gamma`` whatever the guide's errors, as importance sampling does, as long as the guide gives
a positive weight to every node above a term the target weighs, and the mean weight estimates ``sum_t gamma(t) / h(root)``
without bias. With an exact guide, ``h(n)`` the target mass below ``n``, every factor
is one and every particle weighs the same: the particles are then independent draws from the target.

**Resampling.** After every step the effective number of particles, ``(sum w)^2 / sum w^2``, is read; when it falls below a
share of the particles, they are resampled in proportion to their weights (``random.choices``) and their weights reset to
equal, the mean carried into the estimate of the normalizer. Particles that a guide leads into a dead end, a node with a
positive estimate and no child, weigh nothing from there on.

**One scale.** The guide's weight of a node is meant to estimate the target's mass below it, ``h(root)`` the target's whole
mass. Where the two are on different scales the weights stay right, but a particle that reaches its term early carries the
ratio of the scales against the particles still walking, the effective number collapses at that step, and a resampling then
spends the particles on the few that finished: consistent still, and wasteful. The constructions' own targets are on their
weights' scale.

**What it promises.** A weighted sample, consistent as the number of particles grows, the normalizer's estimate unbiased; not
an exact draw at a finite number, not without replacement, and the particles after a resampling share ancestors. Every
statement is about derivations, which are the terms under unambiguity.

**The constructions** it runs on hand it their search tree, the one their own random search walks (``expansion``), and the
target weight of a term (``log_target_of``): the cost table's and the tilt's weights are exact guides of their own targets, the
saddle search's an estimate of its target's.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, Any, Generic, Protocol

from cosy.core.solution_space import T
from cosy.search.sampling import log_sum_exp

if TYPE_CHECKING:
    import random
    from collections.abc import Callable, Sequence

    from cosy.core.tree import Tree

__all__ = ["Guided", "WeightedParticles", "sequential_monte_carlo"]


class Guided(Protocol[T]):
    """A construction whose search tree the particles walk and whose target they are corrected to."""

    def expansion(self) -> tuple[Any, float, Callable[[Any], tuple[Tree[T] | None, Sequence[tuple[Any, float]]]]]:
        """Return the root, its log weight, and the expansion that names a node's term or its children with their log weights."""
        ...

    def log_target_of(self, term: Tree[T]) -> float:
        """Return the log of the target's weight of a term, on the scale of the expansion's weights."""
        ...


def _effective(log_weights: Sequence[float]) -> float:
    """Return the effective number of particles, ``(sum w)^2 / sum w^2``, from their log weights; 0 when none weighs anything.

    Args:
        log_weights (Sequence[float]): The particles' log weights.

    Returns:
        float: The effective number.
    """
    top = max(log_weights, default=-math.inf)
    if top == -math.inf:
        return 0.0
    shifted = [math.exp(value - top) for value in log_weights]
    return math.fsum(shifted) ** 2 / math.fsum(value * value for value in shifted)


@dataclass(frozen=True)
class WeightedParticles(Generic[T]):
    """The particles of one run: their terms, weights, paths' probabilities, and how the run went.

    Attributes:
        terms (tuple[Tree[T] | None, ...]): Each particle's term; None for one that ended in a dead end.
        log_weights (tuple[float, ...]): Each particle's log weight since the last resampling, ``-inf`` for one that weighs
            nothing; their mean, times the means carried at each resampling, is the normalizer's estimate.
        log_proposals (tuple[float, ...]): Each particle's log probability of its path under the guide, its ancestors'
            included across resamplings.
        log_normalizer (float): The log of the estimate of ``sum_t gamma(t) / h(root)``; ``-inf`` when no particle weighs
            anything.
        resamplings (int): How often the particles were resampled.
        steps (int): The steps of the run, each one expansion of every particle not yet at its term.
        ess_trace (tuple[float, ...]): The effective number of particles after each step, before any resampling.
    """

    terms: tuple[Tree[T] | None, ...]
    log_weights: tuple[float, ...]
    log_proposals: tuple[float, ...]
    log_normalizer: float
    resamplings: int
    steps: int
    ess_trace: tuple[float, ...]

    @property
    def ess(self) -> float:
        """Return the effective number of the final particles.

        Returns:
            float: ``(sum w)^2 / sum w^2`` over the final weights; 0 when none weighs anything.
        """
        return _effective(self.log_weights)

    def draw(self, rng: random.Random, count: int) -> list[Tree[T]]:
        """Draw terms from the weighted particles, in proportion to their weights, with replacement.

        Args:
            rng (random.Random): The source of randomness.
            count (int): How many terms to draw, at least 0.

        Returns:
            list[Tree[T]]: The drawn terms.

        Raises:
            ValueError: If no particle weighs anything, or the count is negative.
        """
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            msg = f"the number of terms to draw is a whole number not below 0, not {count!r}"
            raise ValueError(msg)
        total = log_sum_exp(list(self.log_weights))
        if total == -math.inf:
            msg = "no particle reached a term that weighs anything, so there is nothing to draw"
            raise ValueError(msg)
        weights = [math.exp(value - total) for value in self.log_weights]
        chosen = rng.choices(range(len(weights)), weights=weights, k=count)
        return [self.terms[index] for index in chosen]  # type: ignore[misc]


def _pick(log_weights: Sequence[float], total: float, rng: random.Random) -> int:
    """Return the index of a child chosen in proportion to its weight.

    Args:
        log_weights (Sequence[float]): The children's log weights.
        total (float): Their log-sum-exp.
        rng (random.Random): The source of randomness.

    Returns:
        int: The index.
    """
    cumulative = list(accumulate(math.exp(value - total) for value in log_weights))
    return min(bisect_right(cumulative, rng.random() * cumulative[-1]), len(cumulative) - 1)


def sequential_monte_carlo(
    construction: Guided[T], rng: random.Random, particles: int, *, threshold: float = 0.5
) -> WeightedParticles[T]:
    """Run particles from the construction's root to its terms, guided by its weights, weighted to its target.

    Args:
        construction (Guided[T]): The construction: its expansion and its target weight of a term.
        rng (random.Random): The source of randomness.
        particles (int): The number of particles, at least one.
        threshold (float): The share of the particles below which the effective number triggers a resampling, from 0
            (never) to 1 (whenever the weights are uneven). (Default value = 0.5)

    Returns:
        WeightedParticles[T]: The run's particles.

    Raises:
        ValueError: If the number of particles is below one or the threshold outside ``[0, 1]``.
    """
    if isinstance(particles, bool) or not isinstance(particles, int) or particles < 1:
        msg = f"the particles must be a whole number of at least one, not {particles!r}"
        raise ValueError(msg)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        msg = f"the threshold is a share of the particles, a real number from 0 to 1, not {threshold!r}"
        raise ValueError(msg)
    root, root_log_weight, expand = construction.expansion()
    nodes: list[Any] = [root] * particles
    guides = [root_log_weight] * particles
    log_weights = [0.0 if root_log_weight > -math.inf else -math.inf] * particles
    log_proposals = [0.0] * particles
    terms: list[Tree[T] | None] = [None] * particles
    finished = [root_log_weight == -math.inf] * particles
    log_normalizer = 0.0
    resamplings = 0
    steps = 0
    ess_trace: list[float] = []
    while not all(finished):
        steps += 1
        for index in range(particles):
            if finished[index]:
                continue
            inhabitant, children = expand(nodes[index])
            if inhabitant is not None:
                terms[index] = inhabitant
                finished[index] = True
                log_weights[index] += construction.log_target_of(inhabitant) - guides[index]
                continue
            child_weights = [log_weight for _child, log_weight in children]
            total = log_sum_exp(child_weights)
            if total == -math.inf:
                finished[index] = True
                log_weights[index] = -math.inf
                continue
            chosen = _pick(child_weights, total, rng)
            log_weights[index] += total - guides[index]
            log_proposals[index] += child_weights[chosen] - total
            nodes[index] = children[chosen][0]
            guides[index] = child_weights[chosen]
        effective = _effective(log_weights)
        ess_trace.append(effective)
        # Equal weights give the number of particles up to rounding, a hair below it at many numbers: the comparison allows
        # for that, or a threshold of one would resample particles whose weights are all alike.
        if all(finished) or effective <= 0 or effective >= threshold * particles * (1 - 1e-9):
            continue
        mean = log_sum_exp(log_weights) - math.log(particles)
        top = max(log_weights)
        chosen_ones = rng.choices(
            range(particles), weights=[math.exp(value - top) for value in log_weights], k=particles
        )
        nodes = [nodes[index] for index in chosen_ones]
        guides = [guides[index] for index in chosen_ones]
        log_proposals = [log_proposals[index] for index in chosen_ones]
        terms = [terms[index] for index in chosen_ones]
        finished = [finished[index] for index in chosen_ones]
        log_weights = [0.0] * particles
        log_normalizer += mean
        resamplings += 1
    final = log_sum_exp(log_weights)
    log_normalizer = -math.inf if final == -math.inf else log_normalizer + final - math.log(particles)
    return WeightedParticles(
        terms=tuple(terms),
        log_weights=tuple(log_weights),
        log_proposals=tuple(log_proposals),
        log_normalizer=log_normalizer,
        resamplings=resamplings,
        steps=steps,
        ess_trace=tuple(ess_trace),
    )
