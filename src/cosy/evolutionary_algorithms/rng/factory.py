"""Small RNG factory helper used to derive independent RNG streams deterministically.

This utility provides a minimal abstraction to create child random.Random instances
from a master RNG (seeded deterministically). It is intentionally tiny and keeps
the project dependency-free.

Design Notes:
    - Each child RNG is seeded using self.master.randint(), ensuring deterministic
      and independent streams: same master seed always produces same child seeds.
    - Child RNGs are not shared; each call to child() / split() creates fresh
      random.Random instances.
    - This avoids tight coupling of random streams between components, making it
      safe to run components in different orders or concurrently with separate
      child RNGs.
"""

import random


class RNGFactory:
    """Factory for producing independent random.Random instances deterministically.

    RNGFactory manages a master random.Random and produces child RNGs via deterministic
    seeding. All children derived from the same master seed are guaranteed to be
    independent (different streams) and reproducible (same master seed → same children).

    Typical usage:
        # Create a factory from a seed
        factory = RNGFactory.from_seed(seed=42)

        # Get child RNGs for different components
        init_rng = factory.child()
        mutation_rng = factory.child()
        selection_rng = factory.child()

        # Or get multiple children at once
        rngs = factory.split(3)  # [child1, child2, child3]

        # For debugging: use named children
        init_rng = factory.child_named("initialization")
        mutation_rng = factory.child_named("mutation")
    """

    def __init__(self, master: random.Random | None = None):
        """Initialize the factory with a master RNG.

        Args:
            master: The master random.Random instance. If None, a new unseeded
                   random.Random() is created.
        """
        self.master = master if master is not None else random.Random()

    @classmethod
    def from_seed(cls, seed: int) -> "RNGFactory":
        """Create a factory seeded from an integer seed.

        Args:
            seed: Integer seed for reproducible child RNG streams.

        Returns:
            A new RNGFactory with a seeded master RNG.

        Example:
            factory = RNGFactory.from_seed(42)
            child1 = factory.child()  # Always produces same sequence for seed=42
        """
        return cls(random.Random(seed))

    @classmethod
    def from_random(cls, rng: random.Random) -> "RNGFactory":
        """Create a factory from an existing random.Random instance.

        Args:
            rng: An existing random.Random instance to use as the master.

        Returns:
            A new RNGFactory wrapping the provided RNG.
        """
        return cls(rng)

    def child(self) -> random.Random:
        """Return a new independent random.Random seeded from the master RNG.

        Each call produces a new, unseeded random.Random instance with a seed
        derived via self.master.randint(). The seed itself is drawn deterministically
        from the master, ensuring reproducibility.

        Returns:
            A fresh random.Random instance independent from any previous/future children.

        Note:
            The returned RNG is not seeded via random.seed(); it is initialized
            with a specific seed value. Calling this method multiple times with the
            same master produces distinct child RNGs (they do not share state).
        """
        return random.Random(self.master.randint(0, 2**32 - 1))

    def child_named(self, name: str) -> random.Random:
        """Return a new independent random.Random, optionally with a name for debugging.

        This method is identical to child() in behavior; the name parameter is provided
        for debugging and logging purposes only and does not affect the RNG's behaviour.

        Args:
            name: A descriptive label for the child RNG (e.g., "initialization", "mutation").
                 Used for debugging only; does not affect reproducibility.

        Returns:
            A fresh random.Random instance independent from any previous/future children.

        Example:
            factory = RNGFactory.from_seed(42)
            init_rng = factory.child_named("initialization")
            mut_rng = factory.child_named("mutation")
            # Both RNGs are independent and reproducible from seed=42.
        """
        # Name is for user/debugging purposes; actual seeding is deterministic.
        return self.child()

    def split(self, n: int) -> list[random.Random]:
        """Split the factory into n independent child RNGs.

        Produces n new random.Random instances in a single call, each seeded
        deterministically from the master RNG.

        Args:
            n: Number of child RNGs to produce. If n <= 0, returns an empty list.

        Returns:
            A list of n fresh, independent random.Random instances.

        Example:
            factory = RNGFactory.from_seed(42)
            init_rng, mut_rng, cross_rng, sel_rng = factory.split(4)
            # All RNGs are independent and reproducible from seed=42.

        Note:
            This is equivalent to calling child() n times, but more convenient
            and efficient when you need multiple RNGs at once.
        """
        if n <= 0:
            return []
        return [self.child() for _ in range(n)]

    def __call__(self) -> random.Random:
        """Alias for child(); allows factory to be used as a callable.

        Returns:
            A fresh random.Random instance, same as child().

        Example:
            factory = RNGFactory.from_seed(42)
            rng = factory()  # Equivalent to factory.child()
        """
        return self.child()
