from typing import Collection, Sequence, TypeVar


Byte = int


def prediction_extract(data: Sequence[Byte]) -> Sequence[Byte]:
    # the prediction used is always 0
    # with basic 1D predictor P[x]=I[x-1]
    # SOURCE     1 1 5 8 X
    # PREDICTION 0 1 1 5 8
    # RESIDUAL   1 1 5 8 (X-8)
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


def test_prediction():
    # this is a lossless transformation so the data is restored
    data = [1, 1, 5, 8, 10]
    residual = prediction_extract(data)
    assert residual == [1, 0, 4, 3, 2]
    data_restored = prediction_restore(residual)
    assert data_restored == data, f"{data_restored=} != {data=}"
    print("test_prediction OK")


if __name__ == "__main__":
    test_prediction()
