from math import copysign

from classical_encoding.helper.more_math import ceil_div


class UniformScaleQuantizerForSignedWithDeadZone:
    @staticmethod
    def quantizer(x: int, q_step: int):
        return int(copysign((abs(x) // q_step), x))

    @staticmethod
    def dequantizer(y: int, q_step: int) -> int:
        return y * q_step + int(copysign(abs(y) // q_step, y))


class UniformScaleQuantizer:
    """
    Uniform scale quantizer. Quantizes input values into a finite set of
    values with uniform step size.
    Example: (q_step = 3)
        source:      0, 1, 2, 3, 4, 5, 6, 7, 8, 9,  10, 11, 12, 13, 14, 15, 16
        quantized:   0, 0, 0, 1, 1, 1, 2, 2, 2, 3,  3, 3,  4,  4,  4,  5,  5,
        dequantized: 1, 1, 1, 4, 4, 4, 7, 7, 7, 10, 10, 10, 13, 13, 13, 16, 16
    """

    q_step: int

    def __init__(self, q_step: int) -> None:
        self.q_step = q_step

    def quantize(self, x: int) -> int:
        assert 0 <= x < 256, f"Input value must be in range [0, 255], got {x}"
        return x // self.q_step

    def dequantize(self, y: int) -> int:
        return max(min(255, y * self.q_step + self.q_step // 2), 0)


def test_uniform_scale_quantizer():
    step = 3
    peak_absolute_errors = step // 2
    quantizer = UniformScaleQuantizer(step)
    for x in range(256):
        reconstructed = quantizer.dequantize(quantizer.quantize(x))
        assert (
            abs(reconstructed - x) <= peak_absolute_errors
        ), f"Quantization error for {x} is too big, got {reconstructed=}"

    assert quantizer.quantize(0) == 0
    assert quantizer.quantize(1) == 0
    assert quantizer.quantize(2) == 0
    assert quantizer.quantize(100) == 33
    assert quantizer.quantize(255) == 85
    assert quantizer.dequantize(0) == 1
    assert quantizer.dequantize(1) == 4
    assert quantizer.dequantize(2) == 7
    assert quantizer.dequantize(33) == 100
    assert (
        quantizer.dequantize(85) == 255
    ), f"{quantizer.dequantize(85)=} should not be 256"
    print("uniform_scale_quantizer tests passed")


if __name__ == "__main__":
    test_uniform_scale_quantizer()
    print("All tests passed")
