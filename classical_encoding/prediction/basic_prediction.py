from classical_encoding.helper.typing import Bytes


class NaivePrediction1D:
    @staticmethod
    def prediction_extract(data: Bytes) -> Bytes:
        # 1D case:
        # the prediction used is always 0
        # with basic 1D predictor P[x]=I[x-1]
        # SOURCE     1 1 5 8 X
        # PREDICTION 0 1 1 5 8
        # RESIDUAL   1 0 4 3 2
        prediction = [0] + list(data[:-1])  # only used once
        residual = [x - p for x, p in zip(data, prediction)]
        return residual

    @staticmethod
    def prediction_restore(residual: Bytes) -> Bytes:
        # the prediction used is always 0
        # reverse the prediction_extract
        data = []
        for i, r in enumerate(residual):
            new_byte = r + data[i - 1] if i > 0 else r
            data.append(new_byte)
        return data
