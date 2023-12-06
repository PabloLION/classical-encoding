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
    BIRTH_ORDER_LEFT,
    BIRTH_ORDER_RIGHT,
    ExtendedRestrictedFastOrderedList as OrderedList,
    NullableSwappableNode as MetaSymbol,
)

NYT = 256  # Not Yet Transmitted. One byte cannot hold; int for typing.

Byte = int  # Literal[0,...,255]


class AdaptiveHuffmanEncoder:
    nyt_node: MetaSymbol[int]  # the Not Yet Transmitted node
    huffman_dict: dict[int, MetaSymbol[int]]  # byte -> node
    ordered_list: OrderedList[MetaSymbol[int]]  # weight is managed by ordered_list
    root: MetaSymbol[int]  # the root of the huffman tree # #TODO: need?
    # __weight_change: list[int] # maybe we do not want to update weight every

    def __init__(self) -> None:
        self.root, _ = MetaSymbol.make_root(NYT)
        self.nyt_node = self.root
        self.ordered_list = OrderedList()
        self.ordered_list.new_item(self.nyt_node)
        self.huffman_dict = {NYT: self.nyt_node}

    def encode_byte(self, byte: int, byte_range_check: bool = True) -> Bits:
        """Encode a byte.
        We get the result relatively early, but the main job is to update the
        tree and the ordered list.
        Args:
            byte (int): the byte to be encoded
        Returns:
            Bits: encoded data
        """
        if byte_range_check and (byte < 0 or byte > 255):
            raise ValueError("byte must be in range [0, 255].")
        logger.debug(f"begin {byte=:3d} =0b{byte:08b}")

        is_new_byte = byte not in self.huffman_dict
        result = Bits(0, 0)  # will be overwritten

        if is_new_byte:
            result = Bits.from_int1s(self.nyt_node.get_path()) + Bits.from_int(byte, 8)
            # result is not byte_node.path in the next method, but current NYT node's path

        # #TODO :9begin:reuse this block
        if is_new_byte:
            byte_node = self.nyt_node.extend(byte)
            self.huffman_dict[byte] = byte_node
            curr = self.nyt_node.parent
            if curr.is_root:  # #TODO: do we need "self.root"?
                logger.info("changing root for the first new byte")
                self.root = curr  # #TODO: only first symbol needs this
            # #TODO: this is not performant, for safe operations, we can add ..
            # #TODO+ two nodes with weight 1 directly, NYT still has weight 0
            self.ordered_list.new_item(curr)
            self.ordered_list.add_one(curr)
            # this order is important, curr comes before byte_node as its parent
            self.ordered_list.new_item(byte_node)
            self.ordered_list.add_one(byte_node)
            curr = curr.parent  # finished adjusting new byte node and its parent
        else:
            curr = self.huffman_dict[byte]
            result = Bits.from_int1s(curr.get_path())

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

        if not curr.is_dummy_root:
            self.ordered_list.add_one(curr)
        else:
            curr = curr.right
            logger.info(
                f"{curr=} is dummy root, should happen only at the first new byte"
            )
        # curr is the root
        # #TODO: 9end:reuse this block

        logger.info(f"done encoding {byte=:3d} ==0b_ {byte:08b}, {result=}")
        logger.debug(f"new nyt_node path: {self.nyt_node.get_path()}")
        logger.debug(f"new tree: {self.root}")
        return result


def format_bits_list(bits: list) -> str:
    return "".join([str(int(b)) for b in bits])


class AdaptiveHuffmanDecoder:
    nyt_node: MetaSymbol[int]
    root: MetaSymbol[int]
    ordered_list: OrderedList[MetaSymbol[int]]  # to restore the tree

    def __init__(self) -> None:
        self.root, _ = MetaSymbol.make_root(NYT)
        self.nyt_node = self.root
        self.ordered_list = OrderedList()

    def decode_bits(self, bits: Iterator[bool]) -> bytes:
        """Decode a byte.
        Args:
            bits (Bits): the bits to be decoded
        Returns:
            int: decoded byte
        """
        decoded = bytearray()
        curr = self.root  # root is needed in decoding
        seen_bits = []  # for debugging
        assert curr.is_root and not curr.is_dummy_root
        bits = iter([False] + list(bits))  # add a 0 at the beginning for first byte

        for b in bits:  # #TODO: kill indent with next(iter)
            seen_bits.append(b)
            # find the leaf node
            assert curr is not None  # for type checker
            if not curr.is_leaf:
                if b == BIRTH_ORDER_LEFT:
                    curr = curr.left
                elif b == BIRTH_ORDER_RIGHT:
                    curr = curr.right
                else:
                    raise ValueError(f"unknown bit {b=}")

            assert curr is not None  # for type checker
            if not curr.is_leaf:
                continue

            # now we are facing a leaf node
            is_new_byte = curr.value == NYT
            byte = 0  # will be overwritten

            if is_new_byte:
                byte_content = [next(bits) for _ in range(8)]
                seen_bits.pop()  # remove the last bit, the current `b`
                seen_bits.extend(byte_content)
                byte = Bits.from_bools(byte_content).as_int()

            # #TODO: 9begin:reuse this block
            if is_new_byte:
                logger.debug(
                    f"new byte begin for {format_bits_list(seen_bits)} {byte=:3d} =0b{byte:08b}"
                )
                byte_node = self.nyt_node.extend(byte)
                curr = self.nyt_node.parent
                if curr.is_root:  # #TODO: do we need "self.root"?
                    logger.info("changing root for the first new byte")
                    self.root = curr  # #TODO: only first symbol needs this
                # #TODO: this is not performant, for safe operations, we can add ..
                # #TODO+ two nodes with weight 1 directly, NYT still has weight 0
                self.ordered_list.new_item(curr)
                self.ordered_list.add_one(curr)
                # this order is important, curr comes before byte_node as its parent
                self.ordered_list.new_item(byte_node)
                self.ordered_list.add_one(byte_node)
                curr = curr.parent  # finished adjusting new byte node and its parent
            else:  # decoding a known byte
                byte = curr.value
                assert byte is not None, f"{curr.is_leaf=} but its value is None"

            # #NOTE: CANNOT use par here because par will change
            # in the swap_with_subtree method
            # curr, par = par, par.parent
            while not curr.is_root:  # #FIX: is dummy root
                # #NOTE:!! par.is_dummy_root != curr.is_root:
                first_same_weight = self.ordered_list.get_first_same_weight(curr)
                if first_same_weight != curr:
                    logger.debug(f"swap {curr=} with {first_same_weight=}")
                    curr.swap_with_subtree(first_same_weight)
                self.ordered_list.add_one(curr)
                curr = curr.parent

            if not curr.is_dummy_root:
                self.ordered_list.add_one(curr)
            else:
                curr = curr.right
                logger.info(
                    f"{curr=} is dummy root, should happen only at the first new byte"
                )
            # #TODO :9end:reuse this block

            if not curr.is_root:
                # #FIX: the result is not consistent here
                logger.warning(f"{curr=} is not root")
            decoded.append(byte)
            logger.info(
                f"done decoding seen_bits={format_bits_list(seen_bits)} to {byte=:3d} ==0b_ {byte:08b}"
            )
            logger.debug(f"new nyt_node path: {self.nyt_node.get_path()}")
            logger.debug(f"new tree: {self.root}")
            seen_bits = []

        return bytes(decoded)


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
            assert str(encoder.root) == tree_state
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

    n_test_case = 10
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

    # with open("failed_source.binary", "rb") as f:
    #     source = f.read()
    logger.setLevel("DEBUG")
    source = b"abcd abcd "
    source = b"abc abc "
    source = b"ab ab "
    source = b"a a "
    test_unit_adaptive_huffman_coding_no_packer(source)

    # test_adaptive_huffman_coding_no_packer()
