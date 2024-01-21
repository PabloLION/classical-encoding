from typing import Collection, Sequence, TypeVar


Byte = int  # in range(256), effectively, Byte must be int
Bytes = Sequence[Byte]  # assume the length is known
Symbol = TypeVar("Symbol")  # Symbol is the type of the data to be compressed
Symbols = Sequence[Symbol]  # assume the length is known
