"""
This is the implementation of FGK algorithm, which is the first and easier
adaptive Huffman coding. Vitter's algorithm is more complicated and efficient
but not necessary for this project.
"""
from typing import Iterator
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
    root: MetaSymbol[int]  # the root of the huffman tree # #TODO: need?
    nyt_node: MetaSymbol[int]  # the Not Yet Transmitted node
    huffman_dict: dict[int, MetaSymbol[int]]  # symbol->node, check if symbol new
    ordered_list: OrderedList[MetaSymbol[int]]  # weight is managed by ordered_list

    def __init__(
        self, first_symbol: Byte | None = None, nyt_value: int | None = None
    ) -> None:
        if not nyt_value:
            nyt_value = NYT

        self.nyt_node, _ = MetaSymbol.make_root(nyt_value)
        if not first_symbol:
            self.root = self.nyt_node
            self.ordered_list = OrderedList()
            self.ordered_list.new_item(self.nyt_node)
            self.huffman_dict = {NYT: self.nyt_node}
            logger.error(f"empty tree created, {self.root=}")
            return

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

    def add_new_symbol_and_return_nyt_parent(
        self, symbol: Byte
    ) -> tuple[MetaSymbol[int], MetaSymbol[int]]:
        nyt_node = self.nyt_node
        symbol_node = nyt_node.extend(symbol)
        curr = nyt_node.parent
        # #TODO: 9:start: reuse this code
        # #TODO: this is not performant, for safe operations, we can add ..
        # #TODO+ two nodes with weight 1 directly, NYT still has weight 0
        self.ordered_list.new_item(curr)
        self.ordered_list.add_one(curr)
        # this order is important, curr comes before byte_node as its parent
        self.ordered_list.new_item(symbol_node)
        self.ordered_list.add_one(symbol_node)
        # #TODO: 9:end: reuse this code
        curr = curr.parent  # finished adjusting new byte node and its parent
        return symbol_node, curr

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
        # curr is the root
        return curr

    def __contains__(self, symbol: int) -> bool:
        return symbol in self.huffman_dict


class AdaptiveHuffmanEncoder:
    huffman_tree: AdaptiveHuffmanTree

    def __init__(self) -> None:
        self.huffman_tree = AdaptiveHuffmanTree()

    def encode_byte(self, symbol: int, byte_range_check: bool = True) -> Bits:
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

        logger.debug(f"begin {symbol=:3d} =0b{symbol:08b}")
        if len(self.huffman_tree.huffman_dict) == 1:  # reading first symbol
            self.huffman_tree = AdaptiveHuffmanTree(symbol, nyt_value=NYT)
            return Bits.from_int(symbol, 8)

        is_new_byte = symbol not in self.huffman_tree

        if is_new_byte:
            encoded_symbol = Bits.from_int1s(
                self.huffman_tree.nyt_node.get_path()
            ) + Bits.from_int(symbol, 8)
            # result is not byte_node.path in the next method, but current NYT node's path
            byte_node, curr = self.huffman_tree.add_new_symbol_and_return_nyt_parent(
                symbol
            )
            self.huffman_tree.huffman_dict[symbol] = byte_node

        else:
            curr = self.huffman_tree.huffman_dict[symbol]
            encoded_symbol = Bits.from_int1s(curr.get_path())

        curr = self.huffman_tree.update_huffman_tree(curr)

        logger.info(f"done encoding {symbol=:3d} ==0b_ {symbol:08b}, {encoded_symbol=}")
        logger.debug(f"new nyt_node path: {self.huffman_tree.nyt_node.get_path()}")
        logger.debug(f"new tree: {self.huffman_tree.root}")
        return encoded_symbol


def format_bits_list(bits: list) -> str:
    return "".join([str(int(b)) for b in bits])


class AdaptiveHuffmanDecoder:
    huffman_tree: AdaptiveHuffmanTree

    def __init__(self) -> None:
        self.huffman_tree = AdaptiveHuffmanTree()

    def decode_bits(self, bits: Iterator[bool]) -> bytes:
        """Decode a byte.
        Args:
            bits (Bits): the bits to be decoded
        Returns:
            int: decoded byte
        """
        first_symbol = Bits.from_bools([next(bits) for _ in range(8)]).as_int()
        decoded_bytes = bytearray([first_symbol])
        self.huffman_tree = AdaptiveHuffmanTree(first_symbol, nyt_value=NYT)
        curr = self.huffman_tree.root  # root is needed in decoding
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
                seen_bits.pop()  # remove the last bit, the current `b`
                seen_bits.extend(byte_content)
                symbol = Bits.from_bools(byte_content).as_int()
                # do more afterwards
                logger.debug(
                    f"new byte begin for {format_bits_list(seen_bits)} {symbol=:3d} =0b{symbol:08b}"
                )
                _, curr = self.huffman_tree.add_new_symbol_and_return_nyt_parent(symbol)
            else:  # decoding a known byte
                symbol = curr.value
                assert symbol is not None, f"{curr.is_leaf=} but its value is None"

            curr = self.huffman_tree.update_huffman_tree(curr)

            if not curr.is_root:
                # #FIX: the result is not consistent here
                logger.warning(f"{curr=} is not root")
            decoded_bytes.append(symbol)
            logger.info(
                f"done decoding seen_bits={format_bits_list(seen_bits)} to {symbol=:3d} ==0b_ {symbol:08b}"
            )
            logger.debug(f"new nyt_node path: {self.huffman_tree.nyt_node.get_path()}")
            logger.debug(f"new tree: {self.huffman_tree.root}")
            seen_bits = []

        return bytes(decoded_bytes)


def adaptive_huffman_encoding(source: ByteSource) -> tuple[bytearray, Bits]:
    """Adaptive Huffman encoding algorithm
    Args:
        source (ByteSource): the source to be encoded
    Returns:
        Bits: encoded data
    """
    encoder = AdaptiveHuffmanEncoder()
    packer = BytePacker()
    for byte in source.data:
        packer.pack_bits(encoder.encode_byte(byte))
    packer.flush(source.end_symbol)  # #FIX: wrong here
    return (packer.packed, source.end_symbol)


def test_unit_adaptive_huffman_coding_no_packer(
    source: bytes, expected_tree_status: list[str] | None = None
):
    encoder = AdaptiveHuffmanEncoder()
    encoded = Bits(0, 0)
    if expected_tree_status is None:
        for byte in source:
            encoded += encoder.encode_byte(byte)
    else:
        for byte, tree_state in zip(source, expected_tree_status, strict=True):
            encoded += encoder.encode_byte(byte)
            assert (
                str(encoder.huffman_tree.root) == tree_state
            ), f"{tree_state=} != {encoder.huffman_tree.root=}"
    print(f"encoded test passed with {encoded=}")
    decoder = AdaptiveHuffmanDecoder()
    decoded = decoder.decode_bits(iter(encoded.as_bools()))
    assert source == decoded, f"{source=} != {decoded=}"


def test_adaptive_huffman_coding_no_packer():
    logger.setLevel("ERROR")

    source = b"abcddbb"
    expected_tree_status = [
        # TODO: deserialize the tree
        "T[ROOT]None:(T256:(,),T97:(,))",
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
    decoder = AdaptiveHuffmanDecoder()
    unpacked_encoded_bits = BytePacker.unpack_bytes_to_bits(transmitted, end_symbol)
    decoded = decoder.decode_bits(unpacked_encoded_bits)
    assert source == decoded, f"{source=} != {decoded=}"


if __name__ == "__main__":
    from sys import set_int_max_str_digits

    set_int_max_str_digits(8000)

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
