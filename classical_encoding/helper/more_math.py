from math import ceil


def ceil_div(n: int, d: int) -> int:
    # like ceil(n / d), n: numerator, d: denominator
    return -(-n // d)


def test_ceil_div():
    from random import randint

    for _ in range(100000):
        n = randint(-100000, 100000)
        d = randint(-10000, 10000)
        if d == 0:
            continue
        if ceil_div(n, d) != ceil(n / d):
            raise ValueError(f"ceil_div({n}, {d}) != -(-{n} // {d})")
    print("ceil_div test passed")


if __name__ == "__main__":
    test_ceil_div()
