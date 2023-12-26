"""
asymmetric numeral systems
https://medium.com/@bredelet/understanding-ans-coding-through-examples-d1bebfc7e076
"""

from bisect import bisect_right
from itertools import accumulate


Symbol = int  # symbols are bytes
Index = int  # size_t
Pos = int  # real int, the position of the symbol in the distribution


def find_symbol_idx_pos_by_r(r: int, dist: tuple[int, ...]) -> tuple[Index, Pos]:
    ps = tuple(accumulate(dist))  # prefix sum
    if r > ps[-1]:  #  ps[-1] == sum(dist)
        raise ValueError("r is too large")
    idx = bisect_right(ps, r)
    pos = r - (ps[idx - 1] if idx else 0)
    return idx, pos


def test_find_symbol_idx_pos_by_r():
    dist = (3, 3, 2)
    for r, expected_idx, expected_pos in (  # type: ignore
        (0, 0, 0),
        (1, 0, 1),
        (2, 0, 2),
        (3, 1, 0),
        (4, 1, 1),
        (5, 1, 2),
        (6, 2, 0),
        (7, 2, 1),
    ):
        idx, pos = find_symbol_idx_pos_by_r(r, dist)
        assert idx == expected_idx, f"{idx} != {expected_idx} for {r}"
        assert pos == expected_pos, f"{pos} != {expected_pos} for {r}"
    print("test_find_symbol_by_r passed")


def decode_from_int(
    encoded: int, dist: tuple[int, ...], message_length: int
) -> list[Index]:
    """decode a list of symbols from an integer and the distribution"""
    symbols = [0] * message_length  # the decoded symbols
    x = encoded
    for i in range(message_length):
        m = sum(dist)  # M in the article
        d, r = divmod(x, m)
        idx_sym, pos_sym = find_symbol_idx_pos_by_r(r, dist)
        x = d * dist[idx_sym] + pos_sym
        symbols[~i] = idx_sym
    return symbols


def encode_to_int(symbols: list[Index], dist: tuple[int, ...]) -> int:
    """encode a list of (indices of) symbols to an integer with the
    given distribution"""
    x = 0
    m = sum(dist)  # M in the article
    i = [0] + list(accumulate(dist))  # prefix sum, I in the article

    for idx_sym in symbols:
        freq_sym = dist[idx_sym]
        d, r = divmod(x, freq_sym)
        x = d * m + i[idx_sym] + r
        print(x)
    return x


def test_decode_from_int():
    assert decode_from_int(0, (3, 3, 2), 3) == [0, 0, 0]
    ABACCACBC = [0, 1, 0, 2, 2, 0, 2, 1, 2]
    decoded = decode_from_int(17910, (3, 3, 2), 9)
    assert decoded == ABACCACBC, f"{decoded} != {ABACCACBC}"
    print("test_decode_from_int passed")


def test_encode_to_int():
    # assert encode_to_int([0, 0, 0], (3, 3, 2)) == 29
    ABACCACBC = [0, 1, 0, 2, 2, 0, 2, 1, 2]
    assert encode_to_int(ABACCACBC, (3, 3, 2)) == 17910
    print("test_encode_to_int passed")


if __name__ == "__main__":
    test_find_symbol_idx_pos_by_r()
    test_decode_from_int()
    test_encode_to_int()
else:
    raise ImportError("This script should not be imported")
