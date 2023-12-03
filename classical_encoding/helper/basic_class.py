from functools import cache
from typing import (
    Iterable,
    Iterator,
    Sequence,
    TypeVar,
    overload,
    Collection,
)

from classical_encoding.helper.constant import STABLE_EOT

T = TypeVar("T", bound=Sequence)


class Bits(Sequence[bool]):
    _data: int
    _length: int
    _seq: Sequence[bool]

    @property
    def length(self) -> int:
        return self._length

    def __init__(self, data: int, length: int):
        self._data = data
        self._length = length
        self._seq = [bool(data >> i & 1) for i in range(length - 1, -1, -1)]

    @classmethod
    def from_int(cls, n: int, length: int) -> "Bits":
        return Bits(n, length)

    @classmethod
    def from_bools(cls, bits: Iterable[bool]) -> "Bits":
        return cls.from_int1s([int(bit) for bit in bits])

    @classmethod
    def from_int1s(cls, ints: Iterable[int]) -> "Bits":
        _ints = tuple(ints)  # enough type gymnastics, all the following cannot fix
        # Collection[int] and Iterable[int] > Sequence[int] > list[int]
        return Bits(
            sum(bit << (len(_ints) - 1 - i) for i, bit in enumerate(_ints)), len(_ints)
        )

    def as_seq(self) -> Sequence[bool]:
        return self._seq

    def as_int(self) -> int:
        return self._data

    def as_bytes(self) -> bytes:
        """Convert a Bits to a bytearray"""
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
    def __getitem__(self, item: slice) -> Sequence[bool]:
        ...

    def __getitem__(self, item: int | slice) -> bool | Sequence[bool]:
        return self._seq[item]

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[bool]:
        return iter(self._seq)


class Source:  # NamedTuple is less readable and flexible
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
