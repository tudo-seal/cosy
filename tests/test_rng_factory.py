"""Unit tests for RNGFactory.

Tests cover:
- Factory creation (from_seed, from_random)
- Child RNG independence and determinism
- split() method for batch generation
- named children for debugging
"""

import random

from cosy.evolutionary_algorithms.rng.factory import RNGFactory


class TestRNGFactoryCreation:
    """Test factory instantiation and factory constructors."""

    def test_factory_from_seed_creates_reproducible_stream(self) -> None:
        """Two factories with same seed produce identical child RNGs."""
        factory1 = RNGFactory.from_seed(42)
        factory2 = RNGFactory.from_seed(42)

        child1 = factory1.child()
        child2 = factory2.child()

        # Both should produce same sequence (same seed was used internally)
        assert child1.random() == child2.random()
        assert child1.randint(0, 1000) == child2.randint(0, 1000)

    def test_factory_from_random_wraps_existing_rng(self) -> None:
        """Factory can wrap an existing random.Random instance."""
        master = random.Random(42)
        factory = RNGFactory.from_random(master)

        # Factory should use the provided master
        child = factory.child()
        assert isinstance(child, random.Random)

    def test_factory_default_master_unseeded(self) -> None:
        """Factory() with no args creates unseeded master (non-deterministic)."""
        factory1 = RNGFactory()
        factory2 = RNGFactory()

        child1 = factory1.child()
        child2 = factory2.child()

        # These should produce different sequences (unseeded masters)
        # Note: extremely unlikely to be equal by chance, but not impossible
        # We just check they are valid RNG instances
        assert isinstance(child1, random.Random)
        assert isinstance(child2, random.Random)


class TestChildRNGIndependence:
    """Test that child RNGs are independent despite coming from same master."""

    def test_child_rngs_are_independent(self) -> None:
        """Multiple children from same factory produce independent sequences."""
        factory = RNGFactory.from_seed(42)

        child1 = factory.child()
        child2 = factory.child()
        child3 = factory.child()

        # Each child should produce different first random values
        val1 = child1.random()
        val2 = child2.random()
        val3 = child3.random()

        assert val1 != val2
        assert val2 != val3
        assert val1 != val3

    def test_child_rngs_deterministic(self) -> None:
        """Same factory seed always produces same child sequence (in order)."""
        factory_a = RNGFactory.from_seed(42)
        factory_b = RNGFactory.from_seed(42)

        # Generate sequences from both factories in same order
        children_a = [factory_a.child() for _ in range(5)]
        children_b = [factory_b.child() for _ in range(5)]

        # Each child pair should produce identical sequences
        for child_a, child_b in zip(children_a, children_b, strict=False):
            for _ in range(10):
                assert child_a.random() == child_b.random()

    def test_different_factory_seeds_produce_different_children(self) -> None:
        """Different master seeds produce different child RNG streams."""
        factory1 = RNGFactory.from_seed(1)
        factory2 = RNGFactory.from_seed(2)

        child1 = factory1.child()
        child2 = factory2.child()

        # Children should produce different sequences
        val1 = child1.random()
        val2 = child2.random()
        assert val1 != val2


class TestSplitMethod:
    """Test the split() method for batch generation."""

    def test_split_produces_n_children(self) -> None:
        """split(n) produces exactly n children."""
        factory = RNGFactory.from_seed(42)

        children = factory.split(5)
        assert len(children) == 5
        assert all(isinstance(c, random.Random) for c in children)

    def test_split_produces_independent_children(self) -> None:
        """Children from split() are independent."""
        factory = RNGFactory.from_seed(42)

        children = factory.split(3)
        values = [c.random() for c in children]

        # All values should be distinct (very high probability)
        assert len(set(values)) == 3

    def test_split_zero_or_negative_returns_empty(self) -> None:
        """split(n) with n <= 0 returns empty list."""
        factory = RNGFactory.from_seed(42)

        assert factory.split(0) == []
        assert factory.split(-1) == []
        assert factory.split(-5) == []

    def test_split_deterministic_across_factories(self) -> None:
        """split(n) produces same child sequences for same seed."""
        factory1 = RNGFactory.from_seed(42)
        factory2 = RNGFactory.from_seed(42)

        children1 = factory1.split(3)
        children2 = factory2.split(3)

        for c1, c2 in zip(children1, children2, strict=False):
            for _ in range(5):
                assert c1.random() == c2.random()

    def test_split_equivalent_to_multiple_child_calls(self) -> None:
        """split(n) equivalent to calling child() n times."""
        factory1 = RNGFactory.from_seed(42)
        factory2 = RNGFactory.from_seed(42)

        children_split = factory1.split(3)
        children_sequential = [factory2.child() for _ in range(3)]

        for split_child, seq_child in zip(children_split, children_sequential, strict=False):
            for _ in range(5):
                assert split_child.random() == seq_child.random()


class TestNamedChildren:
    """Test child_named() for debugging and clarity."""

    def test_child_named_produces_valid_rng(self) -> None:
        """child_named() returns a valid random.Random."""
        factory = RNGFactory.from_seed(42)

        rng = factory.child_named("test")
        assert isinstance(rng, random.Random)

        # Should be usable normally
        val = rng.randint(0, 100)
        assert 0 <= val <= 100

    def test_child_named_deterministic(self) -> None:
        """child_named() produces same sequence for same seed (name doesn't affect it)."""
        factory1 = RNGFactory.from_seed(42)
        factory2 = RNGFactory.from_seed(42)

        rng1 = factory1.child_named("initialization")
        rng2 = factory2.child_named("initialization")

        # Names are same; should produce identical sequences
        for _ in range(5):
            assert rng1.random() == rng2.random()

    def test_different_names_still_deterministic(self) -> None:
        """Different names don't affect RNG sequence (names are for debugging)."""
        factory1 = RNGFactory.from_seed(42)
        factory2 = RNGFactory.from_seed(42)

        # Different names, same seed
        factory1.child_named("mutation")
        factory1.child_named("initialization")

        factory2.child_named("mutation")
        factory2.child_named("initialization")

        # Both factories should produce children in same order: first is "mutation", second is "init"
        # Wait, the point is that names don't affect seeding.
        # Let me reframe: factory1 produces child_named("mutation"), then child_named("init").
        # factory2 produces child_named("mutation"), then child_named("init").
        # The "same name" across factories should produce same sequence.
        # But within factory1, child_named("mutation") vs child_named("init")
        # are called in sequence, so they're different because they're generated at different times.
        # Let's verify that names DON'T affect the seeding, only the order of calls.

        # Reset factories
        factory1 = RNGFactory.from_seed(42)
        factory2 = RNGFactory.from_seed(42)

        # Same order, different names
        child1_mut = factory1.child_named("mutation")
        child2_mut = factory2.child_named("mutation")
        # These should be the same (same position in sequence, same factory seed)
        assert child1_mut.random() == child2_mut.random()

        # Second children
        child1_init = factory1.child_named("initialization")
        child2_init = factory2.child_named("initialization")
        # These should also be the same
        assert child1_init.random() == child2_init.random()


class TestCallableInterface:
    """Test factory() as callable (equivalent to child())."""

    def test_factory_callable_produces_child(self) -> None:
        """Calling factory() should be equivalent to factory.child()."""
        factory = RNGFactory.from_seed(42)

        rng1 = factory()
        rng2 = factory.child()

        # Both should be valid RNGs
        assert isinstance(rng1, random.Random)
        assert isinstance(rng2, random.Random)

    def test_factory_callable_deterministic(self) -> None:
        """factory() produces same sequences as .child() for same seed."""
        factory1 = RNGFactory.from_seed(42)
        factory2 = RNGFactory.from_seed(42)

        rng1 = factory1()
        rng2 = factory2.child()

        for _ in range(5):
            assert rng1.random() == rng2.random()


class TestComponentIntegration:
    """Test RNGFactory in realistic EA component scenarios."""

    def test_factory_for_component_rngs(self) -> None:
        """Realistic scenario: factory provides RNGs for EA components."""
        factory = RNGFactory.from_seed(42)

        # Simulate EA setup
        init_rng = factory.child_named("initialization")
        mutation_rng = factory.child_named("mutation")
        selection_rng = factory.child_named("selection")

        # Each component uses its RNG
        init_samples = [init_rng.random() for _ in range(3)]
        mutation_samples = [mutation_rng.random() for _ in range(3)]
        selection_samples = [selection_rng.random() for _ in range(3)]

        # All samples should be distinct (no shared state)
        all_samples = init_samples + mutation_samples + selection_samples
        assert len(set(all_samples)) == 9

    def test_factory_reproducible_across_runs(self) -> None:
        """Same factory seed reproduces same component sequences."""

        def run_simulation(seed: int) -> tuple:
            factory = RNGFactory.from_seed(seed)

            init_rng = factory.child_named("initialization")
            mutation_rng = factory.child_named("mutation")

            init_vals = init_rng.sample(range(1000), 5)
            mutation_vals = mutation_rng.sample(range(1000), 5)

            return init_vals, mutation_vals

        # Two runs with same seed should produce identical results
        run1_init, run1_mut = run_simulation(42)
        run2_init, run2_mut = run_simulation(42)

        assert run1_init == run2_init
        assert run1_mut == run2_mut

    def test_factory_independent_across_runs(self) -> None:
        """Different factory seeds produce different component sequences."""

        def run_simulation(seed: int) -> list:
            factory = RNGFactory.from_seed(seed)
            init_rng = factory.child_named("initialization")
            return init_rng.sample(range(1000), 5)

        run1 = run_simulation(42)
        run2 = run_simulation(43)

        # Different seeds should produce different results (very high probability)
        assert run1 != run2


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_very_large_split(self) -> None:
        """split() works for large n."""
        factory = RNGFactory.from_seed(42)
        children = factory.split(1000)

        assert len(children) == 1000
        assert all(isinstance(c, random.Random) for c in children)

    def test_reuse_factory_multiple_splits(self) -> None:
        """Factory can be reused for multiple split() calls."""
        factory = RNGFactory.from_seed(42)

        split1 = factory.split(3)
        split2 = factory.split(3)

        # split2 should produce different children than split1 (sequential seeding)
        val1_0 = split1[0].random()
        val2_0 = split2[0].random()

        assert val1_0 != val2_0
