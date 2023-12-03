# for same issue as https://github.com/tiangolo/typer/issues/348,
# still need to use Optional["BinaryTree"]
from typing import Optional


class BinaryTreeNode:
    # binary tree node, known as MetaSymbol in Huffman coding
    value: Optional[int]
    left: Optional["BinaryTreeNode"]
    right: Optional["BinaryTreeNode"]

    def __init__(
        self,
        value: Optional[int] = None,
        left: Optional["BinaryTreeNode"] = None,
        right: Optional["BinaryTreeNode"] = None,
    ):
        self.left = left
        self.right = right
        self.value = value

    # #TODO: serialize tree
