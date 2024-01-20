from classical_encoding.prediction.basic_prediction import (
    prediction_extract,
    prediction_restore,
)


def test_prediction():
    # this is a lossless transformation so the data is restored
    data = [1, 1, 5, 8, 10]
    residual = prediction_extract(data)
    assert residual == [1, 0, 4, 3, 2]
    data_restored = prediction_restore(residual)
    assert data_restored == data, f"{data_restored=} != {data=}"
    print("test_prediction new OK")


# The original data should be restored
# Input data should not be changed
