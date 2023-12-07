from typing import (
    Iterable,
    Iterator,
    Sequence,
    overload,
    Collection,
)

from classical_encoding.helper.constant import STABLE_EOT


class Bits(Sequence[bool]):
    """
    A class to represent a sequence of bits. Use it as `tuple[bool, ...]` with
    some extra methods.
    """

    _data: int
    _length: int
    _seq: tuple[bool, ...]

    @property
    def data(self) -> int:
        return self._data

    @property
    def length(self) -> int:
        return self._length

    @property
    def seq(self) -> tuple[bool, ...]:
        if not hasattr(self, "_seq"):
            self._seq = tuple(
                bool(self.data >> i & 1) for i in range(self.length - 1, -1, -1)
            )
        return self._seq

    def __init__(self, data: int, length: int):
        self._data = data
        self._length = length

    @classmethod
    def from_int(cls, n: int, bit_length: int) -> "Bits":
        # optional `bit_length` doesn't make sense, huffman tree can have
        # both leading "right"s and leading "left"s at the same time
        return Bits(n, bit_length)

    @classmethod
    def from_int8(cls, n: int) -> "Bits":
        return Bits(n, 8)

    @classmethod
    def from_collection(cls, c: Collection[int | bool]) -> "Bits":
        return Bits(sum(bit << (len(c) - 1 - i) for i, bit in enumerate(c)), len(c))

    @classmethod
    def from_bools(cls, bits: Iterable[bool]) -> "Bits":
        # enough type gymnastics, cannot fix with all the following types
        # Collection[bool] and Iterable[bool] > Sequence[bool] > list[bool]
        return cls.from_collection(tuple(bits))

    @classmethod
    def from_int1s(cls, ints: Iterable[int]) -> "Bits":
        # enough type gymnastics, cannot fix with all the following types
        # Collection[int] and Iterable[int] > Sequence[int] > list[int]
        return cls.from_collection(tuple(ints))

    def as_bools(self) -> tuple[bool, ...]:
        return self.seq

    def as_int(self) -> int:
        return self.data

    def as_bytes(self) -> bytes:
        """Convert a Bits to a bytes"""
        assert len(self) % 8 == 0
        return bytes(
            sum(bit << (7 - i) for i, bit in enumerate(self[j : j + 8]))  # Big Endian
            for j in range(0, len(self), 8)
        )

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Bits):
            return isinstance(__value, Bits) and self._data == __value._data
        if isinstance(__value, Collection):
            return self._seq == __value
        return NotImplemented

    @overload
    def __getitem__(self, item: int) -> bool:
        ...

    @overload
    def __getitem__(self, item: slice) -> tuple[bool, ...]:
        ...

    def __getitem__(self, item: int | slice) -> bool | tuple[bool, ...]:
        return self.seq[item]

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[bool]:
        return iter(self.seq)

    def __add__(self, other: "Bits") -> "Bits":  # __radd__ is not needed
        return Bits(self.data << other.length | other.data, self.length + other.length)

    def __repr__(self) -> str:
        return f"Bits({''.join('1' if bit else '0' for bit in self.seq)}={self.length}bit{self.data})"


class ByteSource:  # NamedTuple is less readable and less flexible
    """
    Represent a source of bytes. Use it as `bytes` with some extra attributes.
    """

    data: bytes
    _end_symbol: Bits

    @property
    def end_symbol(self) -> Bits:
        return self._end_symbol

    def __init__(self, data: bytes, end_symbol: Bits | None = None):
        self.data = data
        if end_symbol:
            self._end_symbol = end_symbol
        elif len(alphabet := set(data)) < 256:
            self._end_symbol = Bits(next(i for i in range(256) if i not in alphabet), 8)
        else:
            self._end_symbol = Bits(STABLE_EOT, STABLE_EOT.bit_length())
