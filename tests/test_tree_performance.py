"""What the tree operations cost, written as invariants rather than as timings.

``Tree`` is immutable: every operation that changes a term has to build a new one and share
whatever it did not change, because ``size`` and ``_hash`` are computed once at construction and
the equality and hash contract depends on them being right.  What a replacement costs is then the
depth of the position it replaces rather than the size of the term it replaces into.

The tests below count operations instead of measuring time.  A counted operation says the same
thing on a loaded machine as on an idle one, whereas a wall-clock bound would only say how busy
the machine running the suite happens to be.
"""

from cosy.core.tree import Tree

# ---------------------------------------------------------------------------
# replace_subtree_at -- rebuild the path, share the rest
# ---------------------------------------------------------------------------


def test_replace_subtree_at_shares_everything_off_the_path() -> None:
    """Only the nodes between the root and the replaced position are new.

    This is the contract the docstring states, and the reason a replacement is cheap: an offspring
    assembled by crossover costs the depth of the crossover point, not the size of the parent.  It
    is also what keeps the per-node interpretation cache useful across a generation -- a rebuilt
    node starts out with an empty one, so a copying implementation would make every individual
    look new to ``interpret`` even where nothing about it changed.
    """
    tree = Tree("f", (Tree("g", (Tree("x"), Tree("w"))), Tree("y")))
    replacement = Tree("z")

    replaced = tree.replace_subtree_at((0, 0), replacement)

    assert replaced.children[1] is tree.children[1]
    assert replaced.children[0].children[1] is tree.children[0].children[1]
    assert replaced.children[0].children[0] is replacement
    assert replaced is not tree
    assert replaced.children[0] is not tree.children[0]
