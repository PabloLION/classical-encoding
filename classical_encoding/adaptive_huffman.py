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
    root: MetaSymbol[int]  # the root of the huffman tree, for debugging
    nyt_node: MetaSymbol[int]  # the Not Yet Transmitted node
    huffman_dict: dict[int, MetaSymbol[int]]  # symbol->node, check if symbol new
    ordered_list: OrderedList[MetaSymbol[int]]  # weight is managed by ordered_list

    def __init__(self, first_symbol: Byte, nyt_value: int = NYT) -> None:
        self.nyt_node, _ = MetaSymbol.make_root(nyt_value)
        self.symbol_node = self.nyt_node.extend(first_symbol)
        self.root = self.nyt_node.parent

        self.ordered_list = OrderedList()
        self.ordered_list.new_item(self.nyt_node)
        # #TODO: 9:start: reuse this code
        self.ordered_list.new_item(self.root)
        self.ordered_list.add_one(self.root)
        # this order is important, root comes before symbol_node as its parent
        self.ordered_list.new_item(self.symbol_node)
        self.ordered_list.add_one(self.symbol_node)
        # #TODO: 9:end: reuse this code

        self.huffman_dict = {NYT: self.nyt_node, first_symbol: self.symbol_node}

    @classmethod
    def __init_without_first_symbol__(
        cls, nyt_value: int = NYT
    ) -> "AdaptiveHuffmanTree":
        """
        An alternative constructor for initializing an empty tree.
        Normally, we initialize the tree with the first symbol, but sometimes
        we need to initialize the tree without the first symbol.
        """
        self = cls.__new__(cls)
        self.nyt_node, _ = MetaSymbol.make_root(nyt_value)
        self.root = self.nyt_node
        self.ordered_list = OrderedList()
        self.ordered_list.new_item(self.nyt_node)
        self.huffman_dict = {nyt_value: self.nyt_node}
        logger.error(f"empty tree created, {self.root=}")
        return self

    def add_new_symbol(self, symbol: Byte) -> MetaSymbol[int]:
        """
        Add a new symbol to the tree and adjust the ordered list for the tree.
        Return the new symbol node.
        """
        symbol_node = self.nyt_node.extend(symbol)
        extended_meta_symbol = self.nyt_node.parent
        # #TODO: 9:start: reuse this code
        # #TODO: this is not performant, for safe operations, we can add ..
        # #TODO+ two nodes with weight 1 directly, NYT still has weight 0
        self.ordered_list.new_item(extended_meta_symbol)
        self.ordered_list.add_one(extended_meta_symbol)
        # order: parent extended_meta_symbol comes before child symbol_node
        self.ordered_list.new_item(symbol_node)
        self.ordered_list.add_one(symbol_node)
        # #TODO: 9:end: reuse this code
        return symbol_node

    def update_huffman_tree(self, starting_node: MetaSymbol[int]) -> MetaSymbol[int]:
        # returning byte_node, new curr
        curr = starting_node
        # we have curr and result here.

        # #NOTE: CANNOT use par here because par will change
        # in the swap_with_subtree method
        # curr, par = par, par.parent
        while not curr.is_root:
            # #NOTE:!! par.is_dummy_root != curr.is_root:
            first_same_weight = self.ordered_list.get_first_same_weight(curr)
            if first_same_weight != curr:
                logger.debug(f"swap {curr=} with {first_same_weight=}")
                curr.swap_with_subtree(first_same_weight)
            self.ordered_list.add_one(curr)
            curr = curr.parent

        self.ordered_list.add_one(curr)
        assert curr.is_root
        return curr

    def __contains__(self, symbol: int) -> bool:
        return symbol in self.huffman_dict


def format_bits_list(bits: list) -> str:
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
        huffman_tree: AdaptiveHuffmanTree, symbol: int, byte_range_check: bool = True
    ) -> Bits:
        """Encode a byte.
        We get the result relatively early, but the main job is to update the
        tree and the ordered list.
        Args:
            byte (int): the byte to be encoded
        Returns:
            Bits: encoded data
        """
        if byte_range_check and (symbol < 0 or symbol > 255):
            raise ValueError("byte must be in range [0, 255].")
        assert (
            len(huffman_tree.huffman_dict) > 1
        ), "huffman tree initialized without symbol"

        logger.debug(f"begin {symbol=:3d} =0b{symbol:08b}")
        is_new_byte = symbol not in huffman_tree

        if is_new_byte:
            encoded_symbol = Bits.from_int1s(
                huffman_tree.nyt_node.get_path()
            ) + Bits.from_int(symbol, 8)
            # encoded should use the NYT node's path before adding the new symbol
            byte_node = huffman_tree.add_new_symbol(symbol)
            curr = huffman_tree.nyt_node.parent.parent  # update tree from here
            huffman_tree.huffman_dict[symbol] = byte_node

        else:
            curr = huffman_tree.huffman_dict[symbol]
            encoded_symbol = Bits.from_int1s(curr.get_path())

        curr = huffman_tree.update_huffman_tree(curr)

        logger.info(f"done encoding {symbol=:3d} ==0b_ {symbol:08b}, {encoded_symbol=}")
        logger.debug(f"new nyt_node path: {huffman_tree.nyt_node.get_path()}")
        logger.debug(f"new tree: {huffman_tree.root}")
        return encoded_symbol

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
            is_new_byte = curr.value == NYT

            if is_new_byte:
                byte_content = [next(bits) for _ in range(8)]
                seen_bits.extend(byte_content)
                symbol = Bits.from_bools(byte_content).as_int()
                # do more afterwards
                logger.info(
                    f"new byte {symbol=:3d} =0b{symbol:08b} found in {format_bits_list(seen_bits)} "
                )
                huffman_tree.add_new_symbol(symbol)
                curr = huffman_tree.nyt_node.parent.parent  # update tree from here

            else:  # decoding a known byte
                symbol = curr.value
                assert symbol is not None, f"{curr.is_leaf=} but its value is None"

            curr = huffman_tree.update_huffman_tree(curr)

            if not curr.is_root:
                # #FIX: the result is not consistent here
                logger.warning(f"{curr=} is not root")
            decoded_bytes.append(symbol)
            logger.info(
                f"done decoding seen_bits={format_bits_list(seen_bits)} to {symbol=:3d} ==0b_ {symbol:08b}"
            )
            logger.debug(f"new nyt_node path: {huffman_tree.nyt_node.get_path()}")
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
