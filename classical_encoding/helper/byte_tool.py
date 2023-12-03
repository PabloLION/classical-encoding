from collections import deque
from typing import Iterator
from typing_extensions import deprecated

from classical_encoding.helper.basic_class import Bits


@deprecated("Use ``Bits.as_bytes()`` instead")
def int8_to_bytes(bits: Bits) -> bytearray:
    """
    Deprecated.
    Convert a list of bits to a bytearray.
    """
    assert len(bits) % 8 == 0
    return bytearray(
        sum(bit << (7 - i) for i, bit in enumerate(bits[j : j + 8]))  # Big Endian
        for j in range(0, len(bits), 8)
    )


class BytePacker:
    bits: list[bool]
    packed: bytearray

    def __init__(self):
        self.bits = list()
        self.packed = bytearray()

    def pack_bits(self, bits: Bits) -> None:
        self.bits.extend(bits)
        if len(self.bits) >= 8:
            group_until = len(self.bits) - len(self.bits) % 8
            new_bytes = Bits.from_bools(self.bits[:group_until]).as_bytes()
            self.packed.extend(new_bytes)
            self.bits = self.bits[group_until:]

    def pack_int(self, n: int, bit_length: int) -> None:
        # optional `bit_length` doesn't make sense, huffman tree can have
        # both leading "right"s and leading "left"s at the same time

        return self.pack_bits(Bits.from_int(n, bit_length))

    # on end of packing, call this method to flush the remaining bits
    def flush(self, end_symbol: Bits | None = None) -> None:
        """Flush the remaining bits, add the end symbol and pad 0s"""
        if end_symbol:
            self.bits.extend(end_symbol)
        n_padding = 8 - len(self.bits) % 8
        if n_padding != 8:  # hard to read (len(self.bits)-1)%8 +1
            self.bits.extend([False] * (n_padding))
        self.packed.extend(Bits.from_bools(self.bits).as_bytes())
        self.bits = list()
        # Also considered this way to show it's ending:
        # 1. end_symbol after packed data
        # 2. after end_symbol, next byte indicates how many digits are valid
        # 3. write the remaining bits to the next byte, and pad 0s to the end

    @staticmethod
    def unpack_bytes_to_bits(
        byte_list: bytes, end_symbol: Bits | None = None
    ) -> Iterator[bool]:
        """Convert a byte to a list of bits, if end_symbol is given,
        the last few bits will be checked for the end_symbol, so the end_symbol
        should not be all 0s.
        """
        if not end_symbol:  # the bytes are exactly the bits we need
            for byte in byte_list:
                yield from (bool(byte >> i & 1) for i in range(7, -1, -1))
        else:
            if set(end_symbol) == {False}:
                raise ValueError("end symbol should not be all 0s")
            # doing the end symbol check from beginning to end is complicated
            # for not online algorithm, checking the last few bytes is enough
            n_checking = (len(end_symbol) - 2) // 8 + 2
            # this n_checking is not accurate. The exact plan is like this:
            # 0 1 2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 ...
            # 0 8 16 16 16 16 16 16 16 16 24 24 24 24 24 24 24 24 32 32 ...
            for byte in byte_list[:-n_checking]:
                yield from (bool(byte >> i & 1) for i in range(7, -1, -1))
            # finding the end symbol in the last few bits is not heavy, use
            # easier implementation (so KMP is not necessary)
            end_symbol_deque = deque(end_symbol)
            last_bits = list(BytePacker.unpack_bytes_to_bits(byte_list[-n_checking:]))
            matching_bits = deque(maxlen=len(end_symbol))
            for count, bit in enumerate(reversed(last_bits)):
                matching_bits.appendleft(bit)
                if matching_bits == end_symbol_deque:
                    break
            else:
                print(f"{end_symbol_deque=}, {last_bits=}")
                raise ValueError("end symbol not found")
            yield from last_bits[:~count]


def test_packer_with_bytes():
    from random import randint

    packer = BytePacker()
    source = bytearray(randint(0, 255) for _ in range(randint(100, 1000)))
    for byte in source:
        packer.pack_int(byte, 8)
    packer.flush()  # without EOT end symbol

    packed = packer.packed
    unpacked = bytearray(packed)
    if len(source) != len(unpacked):
        raise ValueError(
            f"source length {len(source)} != unpacked length {len(unpacked)}"
        )
    if source != unpacked:
        print("index, source binary, unpacked binary, source, unpacked")
        for i in range(len(source)):
            if source[i] != unpacked[i]:
                print(
                    f" {i:4d},    {source[i]:08b},    {unpacked[i]:08b},     {source[i]:3d},     {unpacked[i]:3d}"
                )
        raise ValueError("packer test failed")
    print("packer with bytes test passed")


def test_packer_with_bits():
    from random import randint

    packer = BytePacker()
    source = Bits.from_int1s(randint(0, 1) for _ in range(randint(100, 1000)))
    packer.pack_bits(source)
    end_symbol = Bits.from_bools([False] * 8)
    while set(end_symbol) == {False}:
        end_symbol = Bits.from_int1s(randint(0, 1) for _ in range(randint(4, 8)))
    packer.flush(end_symbol)
    packed = packer.packed
    unpacked = Bits.from_bools(BytePacker.unpack_bytes_to_bits(packed, end_symbol))
    if len(source) != len(unpacked):
        raise ValueError(
            f"source length {len(source)} != unpacked length {len(unpacked)}"
        )
    if source != unpacked:
        print("index, source binary, unpacked binary, source, unpacked")
        for i in range(len(source)):
            if source[i] != unpacked[i]:
                print(
                    f" {i:4d},    {source[i]:08b},    {unpacked[i]:08b},     {source[i]:3d},     {unpacked[i]:3d}"
                )
        raise ValueError("packer test failed")
    print("packer with bits test passed")


if __name__ == "__main__":
    for _ in range(100):
        test_packer_with_bytes()
        test_packer_with_bits()


@deprecated("Use ``Bits.from_int()`` instead")
def int_to_bits(n: int, bit_length: int) -> Bits:
    """Convert an integer to a list of bits"""
    if n.bit_length() > bit_length:
        raise ValueError("bit length is not enough to hold the integer")
    return Bits.from_bools(bool(n >> i & 1) for i in range(bit_length - 1, -1, -1))


@deprecated("Use ``Bits.as_int()`` instead")
def bits_to_int(bits: Bits) -> tuple[int, int]:
    """Convert a list of bits to an integer"""
    return sum(bit << (len(bits) - 1 - i) for i, bit in enumerate(bits)), len(bits)
