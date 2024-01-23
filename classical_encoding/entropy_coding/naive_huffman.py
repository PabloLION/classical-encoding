from itertools import zip_longest
from typing import Counter

from classical_encoding.helper.data_class import Bits, ByteSource
from classical_encoding.helper.byte_tool import BetterBytePacker, BytePacker
from classical_encoding.helper.data_structure import BinaryTreeNode as MetaSymbol
from classical_encoding.helper.typing import Bytes


def encoded_symbol(end_symbol: Bits, huffman_dict: dict[int, tuple[int, int]]) -> Bits:
    return Bits.from_int(*(huffman_dict[end_symbol.as_int()]))


def gen_huffman_tree(source: ByteSource) -> MetaSymbol:
    # get frequency dict and legit end symbol
    freq = Counter(source.data)  # prob is not needed, freq is enough
    freq[source.end_symbol.as_int()] = 1

    # huffman tree process
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    symbol_freq = [(MetaSymbol(b), f) for b, f in sorted_freq]
    while len(symbol_freq) > 1:
        l_tree, l_freq = symbol_freq.pop()
        r_tree, r_freq = symbol_freq.pop()
        symbol_freq.append((MetaSymbol(None, l_tree, r_tree), l_freq + r_freq))
        symbol_freq.sort(key=lambda x: x[1], reverse=True)
    return symbol_freq[0][0]


def gen_huffman_tree_from_bytes(data: Bytes) -> MetaSymbol:
    # get frequency dict and legit end symbol
    freq = Counter(data)  # prob is not needed, freq is enough

    # huffman tree process
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    symbol_freq = [(MetaSymbol(b), f) for b, f in sorted_freq]
    while len(symbol_freq) > 1:
        l_tree, l_freq = symbol_freq.pop()
        r_tree, r_freq = symbol_freq.pop()
        symbol_freq.append((MetaSymbol(None, l_tree, r_tree), l_freq + r_freq))
        symbol_freq.sort(key=lambda x: x[1], reverse=True)
    return symbol_freq[0][0]


def show_huffman_dict(huffman_dict: dict[int, tuple[int, int]]):
    # This is to test if the leading 1 is necessary
    # turns out it is necessary, otherwise the length of the path will be wrong
    # because huffman tree can have both leading "right"s and leading "left"s
    # at the same time. Prefix-free: no whole code word that is a prefix of any
    # other code word. So like "00, 01, 10, 11" is also a prefix code.
    for symbol, (encoded, path_length) in huffman_dict.items():
        path = encoded + (1 << path_length)
        npl = max(1, encoded.bit_length())
        if npl != path_length:
            print(f"{npl=} != {path_length=}, for {encoded=:3d}, {path=:b}")
        print(f"{symbol=:3d}, {encoded=:016b}, {path_length=}")


def dict_from_huffman_tree(tree: MetaSymbol) -> dict[int, tuple[int, int]]:
    """
    Returns a dict of {symbol: (encoded, path_length)}
    """
    # left is 1, right is 0
    huffman_dict: dict[int, tuple[int, int]] = dict()

    def dfs(node: MetaSymbol, path: int = 1):  # add a leading 1 for length
        if node.value is not None:
            if node.value in huffman_dict:
                raise ValueError("Invalid huffman tree")
            path_length = path.bit_length() - 1  # first 1 is not path
            encoded = path - (1 << (path_length))  # remove leading 1
            huffman_dict[node.value] = (encoded, path_length)
        else:
            dfs(node.left, (path << 1) + 1) if node.left else None
            dfs(node.right, (path << 1)) if node.right else None

    dfs(tree)
    # cannot avoid adding the leading 1, tried with next line:
    # show_huffman_dict(huffman_dict)
    return huffman_dict


def naive_huffman_encode(source: ByteSource) -> tuple[MetaSymbol, bytes]:
    ## regarding every byte as a symbol, there will be at most 256 symbols
    tree = gen_huffman_tree(source)
    d = dict_from_huffman_tree(tree)
    packer = BytePacker()
    for byte in source.data:
        packer.pack_int(*(d[byte]))
    packer.flush(encoded_symbol(source.end_symbol, d))  # flush() adds EOT and padding
    return tree, packer.packed


def naive_huffman_encode_to_bytes(data_bytes: Bytes) -> Bytes:
    ## regarding every byte as a symbol, there will be at most 256 symbols
    tree = gen_huffman_tree_from_bytes(data_bytes)
    # print(f"encode serialized tree: {(tree.serialize())=}")
    d = dict_from_huffman_tree(tree)
    packer = []
    tree_bytes_with_length = tree.serialize()
    packer.extend(tree_bytes_with_length)

    packer_bits = []
    for byte in data_bytes:
        encoded_bits, path_length = d[byte]
        packer_bits.extend(Bits.from_int(encoded_bits, path_length))

    # pack the data_bits to bytes
    valid_bits_count = len(packer_bits)
    # print(f"encoded bits count: {valid_bits_count}")
    padding_false_count = 8 - (valid_bits_count % 8 or 8)
    packer_bits.extend([False] * padding_false_count)

    packed_bits = packer_bits
    assert len(packed_bits) % 8 == 0, "packed_bits should be multiple of 8"

    # make valid_bits_count to be the first 4 bytes
    assert valid_bits_count < 2**32, f"{valid_bits_count=} is too big for 4 bytes"
    first_4_bytes = Bits.from_int(valid_bits_count, 32).as_bytes()
    assert len(first_4_bytes) == 4, "first_4_bytes should be 4 bytes"
    packer.extend(first_4_bytes)

    for idx in range(0, len(packed_bits), 8):
        byte = packed_bits[idx : idx + 8]
        packer.append(int("".join(map(str, map(int, byte))), 2))

    return packer


def naive_huffman_decode_from_bytes(data_bytes: Bytes) -> Bytes:
    decoded = list()
    tree_length = int.from_bytes(data_bytes[:2], "big")
    tree = MetaSymbol.deserialize(data_bytes[: 2 + tree_length])
    node = tree

    data_bytes = data_bytes[2 + tree_length :]
    valid_data_bits_count = int.from_bytes(data_bytes[:4], "big")
    # print(f"decoded bits count: {valid_data_bits_count}")

    data_bytes = data_bytes[4:]
    bits = []
    for byte in data_bytes:
        bits.extend(Bits.from_int(byte, 8))
    bits = bits[:valid_data_bits_count]

    for bit in bits:
        if not node:  # and the for-in encoded continues
            raise ValueError("Invalid encoded data")
        node = node.left if bit else node.right
        if node is None:  # and the for-in encoded continues
            raise ValueError("Invalid encoded data")
        if node.value is not None:
            decoded.append(node.value)
            node = tree

    # assert next((b for b in bits if b), 2) == 2, "bits left should be padded 0"
    return decoded


def naive_huffman_decode(tree: MetaSymbol, encoded: bytes, end_symbol: Bits) -> bytes:
    # left is 1, right is 0
    decoded = bytearray()
    node = tree
    bits = BytePacker.unpack_bytes_to_bits(encoded, end_symbol)
    for bit in bits:
        if not node:  # and the for-in encoded continues
            raise ValueError("Invalid encoded data")
        node = node.left if bit else node.right
        if node is None:  # and the for-in encoded continues
            raise ValueError("Invalid encoded data")
        if node.value is not None:
            decoded.append(node.value)
            node = tree
            if node.value == end_symbol:
                break
    assert next((b for b in bits if b), 2) == 2, "bits left should be padded 0"
    return bytes(decoded)


def test_encode_decode_naive_huffman(source: ByteSource):
    tree, encoded = naive_huffman_encode(source)
    decoded = naive_huffman_decode(
        tree, encoded, encoded_symbol(source.end_symbol, dict_from_huffman_tree(tree))
    )

    if source.data != decoded:
        print("index, source binary, decoded binary, source, decoded")
        for i, (s, d) in enumerate(zip_longest(source.data, decoded, fillvalue=111)):
            if s != d:
                print(f" {i:4d},    {s:08b},    {d:08b},     {s:3d},     {d:3d}")
        if len(source.data) != len(decoded):
            raise ValueError(
                f"source length {len(source.data)} != decoded length {len(decoded)}"
            )
        raise ValueError("naive huffman test failed")
    print("naive huffman test passed")


def test_encode_decode_naive_huffman_bytes(data_bytes: Bytes):
    encoded_bytes = naive_huffman_encode_to_bytes(data_bytes)
    decoded = naive_huffman_decode_from_bytes(encoded_bytes)

    if data_bytes != decoded:
        print("index, source binary, decoded binary, source, decoded")
        for i, (s, d) in enumerate(zip_longest(data_bytes, decoded, fillvalue=111)):
            if s != d:
                print(f" {i:4d},    {s:08b},    {d:08b},     {s:3d},     {d:3d}")
        if len(data_bytes) != len(decoded):
            raise ValueError(
                f"source length {len(data_bytes)} != decoded length {len(decoded)}"
            )
        raise ValueError("naive huffman test failed")
    print("naive huffman test passed")


def test_naive_huffman():
    # generate a random source of random length between 100 bytes and 1000 bytes
    from random import randint

    sources = (
        [ByteSource(bytes([4] * 10 + [3] * 10 + [2] * 20 + [1] * 80 + [0] * 320))]
        + [
            ByteSource(bytes([randint(0, 255) for _ in range(9, 16)]))
            for _ in range(100)
        ]
        + [
            ByteSource(bytes([randint(0, 255) for _ in range(randint(100, 1000))]))
            for _ in range(100)
        ]
    )
    for source in sources:
        test_encode_decode_naive_huffman(source)


def test_naive_huffman_bytes():
    # generate a random source of random length between 100 bytes and 1000 bytes
    from random import randint

    data_bytes_collection = (
        [bytes([4] * 10 + [3] * 10 + [2] * 20 + [1] * 80 + [0] * 320)]
        + [bytes([randint(0, 255) for _ in range(9, 16)]) for _ in range(100)]
        + [
            bytes([randint(0, 255) for _ in range(randint(100, 1000))])
            for _ in range(100)
        ]
    )
    for data_bytes in data_bytes_collection:
        test_encode_decode_naive_huffman_bytes(data_bytes)


if __name__ == "__main__":
    test_naive_huffman_bytes()
    # test_naive_huffman()
