from typing import Collection, Sequence, TypeVar


Byte = int


def prediction_extract(data: Sequence[Byte]) -> Sequence[Byte]:
    # 1D case:
    # the prediction used is always 0
    # with basic 1D predictor P[x]=I[x-1]
    # SOURCE     1 1 5 8 X
    # PREDICTION 0 1 1 5 8
    # RESIDUAL   1 0 4 3 2
    prediction = [0] + list(data[:-1])
    residual = [x - p for x, p in zip(data, prediction)]
    return residual


def prediction_restore(residual: Sequence[Byte]) -> Sequence[Byte]:
    # the prediction used is always 0
    # reverse the prediction_extract
    prediction = [0]
    data = []
    for i, r in enumerate(residual):
        new_byte = r + prediction[i]
        data.append(new_byte)
        prediction.append(new_byte)
    data = [x + p for x, p in zip(residual, prediction)]
    return data