from classical_encoding.compression_pipeline.classical_pipeline import (
    CompressionPipeline,
)
from classical_encoding.prediction.basic_prediction import NaivePrediction1D


def test_naive_prediction_1d():
    # this is a lossless transformation so the data is restored
    data = [1, 1, 5, 8, 10]
    residual = NaivePrediction1D.prediction_extract(data)
    assert residual == [1, 0, 4, 3, 2]
    data_restored = NaivePrediction1D.prediction_restore(residual)
    assert data_restored == data, f"{data_restored=} != {data=}"


def test_naive_prediction_1d_purity():
    # Input data should not be changed
    data = [1, 1, 5, 8, 10]
    data_copy = data.copy()
    residual = NaivePrediction1D.prediction_extract(data)
    residual_copy = residual[:]  # shallow copy
    assert residual == [1, 0, 4, 3, 2]
    data_restored = NaivePrediction1D.prediction_restore(residual)
    assert data_restored == data, f"{data_restored=} != {data=}"
    assert data == data_copy, f"{data=} != {data_copy=}"
    assert residual == residual_copy, f"{residual=} != {residual_copy=}"


def test_naive_prediction_1d_with_pipeline():
    pipeline = CompressionPipeline(
        prediction_extract=NaivePrediction1D.prediction_extract,
        prediction_restore=NaivePrediction1D.prediction_restore,
    )

    data = [1, 1, 5, 8, 10]
    residual = pipeline.prediction_extract(data)
    assert residual == [1, 0, 4, 3, 2]
    data_restored = pipeline.prediction_restore(residual)
    assert data_restored == data, f"{data_restored=} != {data=}"
    pipeline._check(data)
    print("test_prediction_with_pipeline OK")
