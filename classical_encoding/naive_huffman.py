from itertools import zip_longest
from typing import Counter

from classical_encoding.helper.basic_class import Bits, ByteSource
from classical_encoding.helper.byte_tool import BytePacker
from classical_encoding.helper.tree import BinaryTree as MetaSymbol


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


def dict_from_huffman_tree(tree: MetaSymbol) -> dict[int, tuple[int, int]]:
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


if __name__ == "__main__":
    test_naive_huffman()
