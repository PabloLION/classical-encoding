# for same issue as https://github.com/tiangolo/typer/issues/348,
# still need to use Optional["BinaryTree"]
from typing import Optional


class BinaryTree:
    # binary tree node, known as MetaSymbol in Huffman coding
    value: Optional[int]
    left: Optional["BinaryTree"]
    right: Optional["BinaryTree"]

    def __init__(
        self,
        value: Optional[int] = None,
        left: Optional["BinaryTree"] = None,
        right: Optional["BinaryTree"] = None,
    ):
        self.left = left
        self.right = right
        self.value = value

    # #TODO: serialize tree
