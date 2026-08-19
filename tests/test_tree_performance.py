"""What the tree operations cost, written as invariants rather than as timings.

``Tree`` is immutable: every operation that changes a term builds a new one and shares whatever it
did not change (``replace_subtree_at`` is written that way, and the equality and hash contract
depends on it).  Two operations were written before that was true and still defended against a
mutation that cannot happen:

* ``subtree_at`` copied the subtree it descended into, once per level, so reading a position cost
  a full copy of everything below it.  Its callers then held a clone, which quietly defeated the
  sharing the rest of the class is built on.
* ``__copy__`` recursed into every child, so a "copy" of a term was a copy of the whole term.

The tests below count operations instead of measuring time.  A counted operation says the same
thing on a loaded machine as on an idle one, whereas a wall-clock bound would only say how busy
the machine running the suite happens to be.
"""

from copy import copy

from cosy.core.tree import Tree


def chain(depth: int) -> Tree[str]:
    """Build a unary chain of the given depth.

    Args:
        depth (int): The number of unary nodes above the leaf.

    Returns:
        Tree[str]: The chain.
    """
    node: Tree[str] = Tree("leaf", ())
    for _ in range(depth):
        node = Tree("f", (node,))
    return node


def shared_layers(depth: int) -> Tree[str]:
    """Build a complete binary tree of the given depth out of one node object per level.

    Both children of every level are the same object, so the result is a term of
    ``2**(depth+1) - 1`` *positions* held in only ``depth + 1`` node objects.  That is the sharing
    this module is about, and it is what makes the large cases below cheap to build: nothing here
    depends on the two subtrees being distinct, and every test reasons in positions.

    Args:
        depth (int): The number of levels above the leaves.

    Returns:
        Tree[str]: The tree, with ``2**(depth+1) - 1`` positions.
    """
    node: Tree[str] = Tree("leaf", ())
    for _ in range(depth):
        node = Tree("g", (node, node))
    return node


# ---------------------------------------------------------------------------
# subtree_at -- reading a position must not build anything
# ---------------------------------------------------------------------------


def test_subtree_at_returns_the_node_itself() -> None:
    """The subtree at a position is the node that sits there, not a clone of it.

    Sharing is the contract of the class: ``replace_subtree_at`` shares every node off the path
    it rebuilds, and a reader that hands out clones makes that sharing unobservable -- and turns
    every read into a copy of the subtree.
    """
    tree = shared_layers(3)
    assert tree.subtree_at((0,)) is tree.children[0]
    assert tree.subtree_at((1, 0)) is tree.children[1].children[0]
    assert tree.subtree_at(()) is tree


def test_a_deep_chain_can_be_read_to_the_bottom() -> None:
    """Reading the bottom of a chain works at any depth a sampler can produce.

    Terms grow to hundreds of nodes, and the recursions that build them materialize partial terms
    along the way, so a term deeper than the interpreter's recursion limit is not a pathological
    case here.
    """
    deep = chain(5000)
    bottom = tuple(0 for _ in range(5000))
    assert deep.subtree_at(bottom).root == "leaf"


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


# ---------------------------------------------------------------------------
# __copy__ -- shallow, because the nodes are immutable
# ---------------------------------------------------------------------------


def test_copy_shares_the_children() -> None:
    """A shallow copy of an immutable node shares its children.

    ``copy`` on an immutable structure is a new root over the same children; the recursive version
    is ``deepcopy`` under another name, and nothing in this code base needs it.
    """
    tree = shared_layers(3)
    duplicate = copy(tree)
    assert duplicate == tree
    assert duplicate is not tree
    for original_child, copied_child in zip(tree.children, duplicate.children, strict=True):
        assert copied_child is original_child
