from math import copysign


class UniformScaleQuantizer:
    @staticmethod
    def quantizer(x: int, q_step: int):
        return int(copysign((abs(x) // q_step), x))

    @staticmethod
    def dequantizer(y: int, q_step: int) -> int:
        return y * q_step + int(copysign(abs(y) // q_step, y))
