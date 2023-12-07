"""
This is the implementation of FGK algorithm, which is the first and easier
adaptive Huffman coding. Vitter's algorithm is more complicated and efficient
but not necessary for this project.
"""
from typing import Callable, Iterator
from classical_encoding.helper.logger import logger
from classical_encoding.helper.byte_tool import BytePacker
from classical_encoding.helper.data_class import Bits, ByteSource
from classical_encoding.helper.data_structure import (
    ExtendedRestrictedFastOrderedList as OrderedList,
    NullableSwappableNode as MetaSymbol,
)

NYT = 256  # Not Yet Transmitted. One byte cannot hold; int for typing.

Byte = int  # Literal[0,...,255]


class AdaptiveHuffmanTree:
    # basically a ExtendedRestrictedFastOrderedList[NullableSwappableNode[int]]
    _root: MetaSymbol[int]  # the root of the huffman tree, for debugging

    @property
    def root(self) -> MetaSymbol[int]:
        return self._root

    _nyt_node: MetaSymbol[int]  # the Not Yet Transmitted node

    @property
    def nyt_node(self) -> MetaSymbol[int]:
        return self._nyt_node

    __dict: dict[int, MetaSymbol[int]]  # huffman dict, symbol->node, for new symbol
    # #TODO: this dict should be write-once

    __list: OrderedList[MetaSymbol[int]]  # weight is managed by ordered_list

    def __init__(self, first_symbol: Byte, nyt_value: int = NYT) -> None:
        self.__initialize_all_attributes(nyt_value)  #  without first symbol
        symbol_node = self.add_new_symbol(first_symbol)  # add the first symbol
        # update the outdated attributes, only needed for the first symbol.
        self._root = self.nyt_node.parent
        self.__dict = {nyt_value: self.nyt_node, first_symbol: symbol_node}

    def __initialize_all_attributes(self, nyt_value: int = NYT) -> None:
        """Initialize all attributes of the adaptive huffman tree"""
        self._nyt_node, _ = MetaSymbol.make_root(nyt_value)
        self._root = self.nyt_node
        self.__list = OrderedList()
        self.__list.new_item(self.nyt_node)
        self.__dict = {nyt_value: self.nyt_node}

    @classmethod
    def _init_without_first_symbol(cls, nyt_value: int = NYT) -> "AdaptiveHuffmanTree":
        """
        An alternative constructor for initializing an empty tree.
        Normally, we initialize the tree with the first symbol, but sometimes
        we need to initialize the tree without the first symbol.
        """
        self = cls.__new__(cls)
        self.__initialize_all_attributes(nyt_value)
        logger.error(f"empty tree created, {self._root=}")
        return self

    def add_new_symbol(self, symbol: Byte) -> MetaSymbol[int]:
        """
        Add a new symbol to the tree and adjust the ordered list for the tree.
        Return the new symbol node.
        Note that the new symbol node and the NYT node share the same parent
        extended meta symbol. And all three nodes are updated, start the update
        from the parent of extended meta symbol if needed
        """
        symbol_node = self._nyt_node.extend(symbol)
        extended_meta_symbol = self._nyt_node.parent
        # #TODO: this is not performant, for safe operations, we can add ..
        # #TODO+ two nodes with weight 1 directly, NYT still has weight 0
        self.__list.new_item(extended_meta_symbol)
        self.__list.add_one(extended_meta_symbol)
        # order: parent extended_meta_symbol comes before child symbol_node
        self.__list.new_item(symbol_node)
        self.__list.add_one(symbol_node)
        self.__dict[symbol] = symbol_node
        return symbol_node

    def update_huffman_tree(self, starting_node: MetaSymbol[int]) -> MetaSymbol[int]:
        """
        Update the huffman tree and the ordered list from the starting node.
        Return the root of the tree.
        """

        curr = starting_node
        # we have curr and result here.

        # #NOTE: CANNOT use par here because par will change
        # in the swap_with_subtree method
        # curr, par = par, par.parent
        while not curr.is_root:
            # #NOTE:!! par.is_dummy_root != curr.is_root:
            first_same_weight = self.__list.get_first_same_weight(curr)
            if first_same_weight != curr:
                logger.debug(f"swap {curr=} with {first_same_weight=}")
                curr.swap_with_subtree(first_same_weight)
            self.__list.add_one(curr)
            curr = curr.parent

        self.__list.add_one(curr)
        assert curr.is_root
        return curr

    def get_nyt_path(self) -> Bits:
        """Get the path of the NYT node"""
        return Bits.from_int1s(self._nyt_node.get_path())

    def __len__(self) -> int:
        raise NotImplementedError(
            "The length of a AdaptiveHuffmanTree is not well defined "
            + "between counting only the leaves vs all the intermediate nodes "
            + "in the tree. Use property `n_leaf` and `n_node` instead."
        )

    @property
    def n_leaf(self) -> int:
        return len(self.__dict)

    @property
    def n_node(self) -> int:
        return len(self.__list)

    def __contains__(self, symbol: int) -> bool:
        return symbol in self.__dict

    def __getitem__(self, symbol: int) -> MetaSymbol[int]:
        return self.__dict[symbol]


def format_bools(bits: list) -> str:
    # helper for debugging
    return "".join([str(int(b)) for b in bits])


class AdaptiveHuffman:
    """Gather functions for adaptive Huffman coding algorithm"""

    @staticmethod
    def encode_bytes(
        bytes: Iterator[Byte],  # #TODO: also accept bytes
        tree_state_check: Callable[[str], bool] | None = None,
    ) -> Bits:
        """Encode a sequence of bytes."""
        # The tree_state_check is the easiest and cleanest way I come up with
        # to test the tree state during encoding.
        # Considered Designs:
        # 1. Return the tree state after the whole encoding process
        #   - cannot see what's happening if there are exceptions in the middle
        #       of the encoding process.
        #   - will make the return type complicated when no check is needed.
        # 2. Return a generator that yields the tree state after each byte and
        #    return the encoded bits in the end.
        #   - will make the return type hard to define as only the last element
        #       is bits, and the rest are tree states.
        #   - every time we call the function, we need to be careful about the
        #      return type.
        # 3. Write another function for testing the tree state.
        #   - repeated code would also require us to change both functions when
        #       we change the algorithm.
        # 4. Use a global or class variable to store the tree state.
        #   - not modularized.
        # 5. Pass a list to compare the tree state.
        #   - not flexible.
        #   - reduces the readability of the code.
        first_byte = next(bytes)
        encoded_bits = Bits.from_int(first_byte, 8)
        huffman_tree = AdaptiveHuffmanTree(first_byte, nyt_value=NYT)
        for byte in bytes:
            encoded_bits += AdaptiveHuffman.encode_byte(huffman_tree, byte)
            if tree_state_check is not None:
                tree_state_check(str(huffman_tree.root))
        return encoded_bits

    @staticmethod
    def encode_byte(
        huffman_tree: AdaptiveHuffmanTree, symbol: int, check_byte_range: bool = True
    ) -> Bits:
        """Encode a byte.
        We get the result relatively early, but the main job is to update the
        tree and the ordered list.
        Args:
            byte (int): the byte to be encoded
        Returns:
            Bits: encoded data
        """
        if check_byte_range and (symbol < 0 or symbol > 255):
            raise ValueError("byte must be in range [0, 255].")
        if huffman_tree.n_leaf <= 1:  # tree has only one leaf: NYT node
            raise ValueError("huffman tree not initialized with first symbol")
        logger.debug(f"begin {symbol=:3d} =0b{symbol:08b}")

        if symbol not in huffman_tree:
            encoded = huffman_tree.get_nyt_path() + Bits.from_int8(symbol)
            # encoded should use the NYT node's path before adding the new symbol
            _ = huffman_tree.add_new_symbol(symbol)
            curr = huffman_tree.nyt_node.parent.parent  # update tree from here
        else:
            curr = huffman_tree[symbol]
            encoded = curr.get_bits_path()

        curr = huffman_tree.update_huffman_tree(curr)

        logger.info(f"done encoding {symbol=:3d} ==0b_ {symbol:08b}, {encoded=}")
        logger.debug(f"new nyt_node path: {huffman_tree.get_nyt_path()}")
        logger.debug(f"new tree: {huffman_tree.root}")
        return encoded

    @staticmethod
    def decode_bits(bits: Iterator[bool]) -> bytes:
        """Decode a byte.
        Args:
            bits (Bits): the bits to be decoded
        Returns:
            int: decoded byte
        """
        first_symbol = Bits.from_bools([next(bits) for _ in range(8)]).as_int()
        decoded_bytes = bytearray([first_symbol])
        huffman_tree = AdaptiveHuffmanTree(first_symbol, nyt_value=NYT)
        curr = huffman_tree.root  # root is needed in decoding
        seen_bits = []  # for debugging

        for b in bits:  # #TODO: kill indent with next(iter)
            seen_bits.append(b)
            # find the leaf node
            assert curr is not None  # for type checking
            if not curr.is_leaf:
                curr = curr.get_child(1 if b else 0)
            assert curr is not None  # for type checking
            if not curr.is_leaf:
                continue

            # now we are facing a leaf node

            if curr == huffman_tree.nyt_node:
                byte_content = [next(bits) for _ in range(8)]
                seen_bits.extend(byte_content)
                symbol = Bits.from_bools(byte_content).as_int()
                # do more afterwards
                logger.info(
                    f"new byte {symbol=:3d} ==0b_ {symbol:08b} in {format_bools(seen_bits)}"
                )
                huffman_tree.add_new_symbol(symbol)
                curr = huffman_tree.nyt_node.parent.parent  # update tree from here

            else:  # decoding a known byte
                symbol = curr.value
                assert symbol is not None, f"{curr.is_leaf=} but its value is None"

            curr = huffman_tree.update_huffman_tree(curr)

            if not curr.is_root:
                logger.warning(f"{curr=} is not root")
            decoded_bytes.append(symbol)
            logger.info(
                f"done decoding seen_bits={format_bools(seen_bits)} to {symbol=:3d} ==0b_ {symbol:08b}"
            )
            logger.debug(f"new nyt_node path: {huffman_tree.get_nyt_path()}")
            logger.debug(f"new tree: {huffman_tree.root}")
            seen_bits = []

        return bytes(decoded_bytes)


def adaptive_huffman_encoding(source: ByteSource) -> tuple[bytearray, Bits]:
    """Adaptive Huffman encoding algorithm
    Args:
        source (ByteSource): the source to be encoded
    Returns:
        Bits: encoded data
    """
    packer = BytePacker()
    packer.pack_bits(AdaptiveHuffman.encode_bytes(iter(source.data)))
    packer.flush(source.end_symbol)  # #FIX: wrong here
    return (packer.packed, source.end_symbol)


def test_unit_adaptive_huffman_coding_no_packer(
    source: bytes, expected_tree_status: list[str] | None = None
):
    if expected_tree_status is None:
        encoded = AdaptiveHuffman.encode_bytes(iter(source))
    else:
        it = iter(expected_tree_status)

        def tree_state_check(tree_state: str) -> bool:
            expected = next(it)  # if `it` is exhausted, raise StopIteration
            assert tree_state == expected, f"{tree_state=} != {expected=}"
            return True

        encoded = AdaptiveHuffman.encode_bytes(iter(source), tree_state_check)
        if (n := next(it, None)) is not None:
            raise ValueError(
                f"expected_tree_status should be completely consumed, but get next {n}"
            )

    print(f"encoded test passed with {encoded=}")
    decoded = AdaptiveHuffman.decode_bits(iter(encoded.as_bools()))
    assert source == decoded, f"{source=} != {decoded=}"


def test_adaptive_huffman_coding_no_packer():
    logger.setLevel("INFO")

    source = b"abcddbb"
    expected_tree_status = [
        # TODO: deserialize the tree
        # skip the tree initialized with one symbol "T[ROOT]None:(T256:(,),T97:(,))",
        "T[ROOT]None:(TNone:(T256:(,),T98:(,)),T97:(,))",
        "T[ROOT]None:(T97:(,),TNone:(TNone:(T256:(,),T99:(,)),T98:(,)))",
        "T[ROOT]None:(TNone:(TNone:(T256:(,),T100:(,)),T99:(,)),TNone:(T97:(,),T98:(,)))",
        "T[ROOT]None:(TNone:(TNone:(T256:(,),T98:(,)),T99:(,)),TNone:(T97:(,),T100:(,)))",
        "T[ROOT]None:(TNone:(TNone:(T256:(,),T97:(,)),T99:(,)),TNone:(T98:(,),T100:(,)))",
        "T[ROOT]None:(T98:(,),TNone:(TNone:(TNone:(T256:(,),T97:(,)),T99:(,)),T100:(,)))",
    ]

    test_unit_adaptive_huffman_coding_no_packer(source, expected_tree_status)
    source = b"abracadabra"
    test_unit_adaptive_huffman_coding_no_packer(source)

    # TODO: move to test util
    from random import randint

    n_test_case = 0
    n_passed = 0
    for _ in range(n_test_case):
        source_len = randint(1_000, 10_000)
        source = bytes([randint(0, 255) for _ in range(source_len)])
        try:
            test_unit_adaptive_huffman_coding_no_packer(source)
        except Exception as e:
            logger.critical(f"failed with {source=} with error {e=}")
            with open("failed_source.binary", "wb") as f:
                f.write(source)
        else:
            n_passed += 1
    print(f"{n_passed=} / {n_test_case=}")


def test_adaptive_huffman_encoding_with_packer(source: bytes):
    byte_source = ByteSource(b"abracadabra", Bits.from_bools([True] * 16))
    transmitted, end_symbol = adaptive_huffman_encoding(byte_source)
    unpacked_encoded_bits = BytePacker.unpack_bytes_to_bits(transmitted, end_symbol)
    decoded = AdaptiveHuffman.decode_bits(unpacked_encoded_bits)
    assert source == decoded, f"{source=} != {decoded=}"


if __name__ == "__main__":
    from sys import set_int_max_str_digits

    set_int_max_str_digits(8000)  # TODO: if Bits too long, do not serialize.

    # edge cases:
    # with open("failed_source.binary", "rb") as f:
    #     source = f.read()
    # logger.setLevel("DEBUG")
    # source = b"abcd abcd "
    # source = b"abc abc "
    # source = b"ab ab "
    # source = b"a a "
    # test_unit_adaptive_huffman_coding_no_packer(source)

    test_adaptive_huffman_coding_no_packer()
