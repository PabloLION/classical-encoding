from datetime import datetime
import json
from time import time
from typing import Any, Callable, Generic, NamedTuple, Optional

from matplotlib.pylab import f
from classical_encoding import OUTPUT_FOLDER
from classical_encoding.metrics.print_metric import calculate_metrics
from classical_encoding.metrics.rate_distortion_plot import (
    plot_rate_distortion,
)
from classical_encoding.helper.typing import Byte, Bytes, Symbol, Symbols, Metrics
from classical_encoding.helper.logger import logger

type Quantize = Callable[[Bytes], Bytes]
type Dequantize = Callable[[Bytes], Bytes]
type PredictionExtract = Callable[[Any], Bytes]  # #TODO: Any
type PredictionRestore = Callable[[Any], Bytes]  # #TODO: Any
# ECC: error correction code
type ECCIntegrate = Callable[[Bytes], Bytes]  # data -> data_with_ecc
type ECCExtract = Callable[[Bytes], Bytes]  # data_with_ecc -> data
type TransmissionSend = Callable[[Bytes], Any]  # #TODO: Any
type TransmissionReceive = Callable[[Any], Bytes]  # #TODO: Any
type EntropyEncode = Callable[[Bytes], Bytes]
type EntropyDecode = Callable[[Bytes], Bytes]
# input the data before and after compression, output the metrics
type CompressionMetrics = Callable[[Bytes, Bytes, Bytes], Metrics]


def identity[T](x: T) -> T:
    return x


(
    fake_quantizer,
    fake_dequantizer,
    fake_entropy_encoder,
    fake_entropy_decoder,
    fake_error_correction_integrate,
    fake_error_correction_extract,
    fake_transmission_send,
    fake_transmission_receive,
) = (identity,) * 8


def fake_prediction_extract(data: Symbols) -> Symbols:
    # the prediction used is always 0
    # with basic 1D predictor P[x]=I[x-1]
    # SOURCE     1 1 5 8 X
    # PREDICTION 0 1 1 5 8
    # RESIDUAL   1 0 4 3 2

    # with basic 2D predictor P[x,y]=mean(I[x-1,y-1], I[x-1,y], I[x,y-1])

    # here the prediction is always 0
    # SOURCE     1 1 5 8 X
    # PREDICTION 0 0 0 0 0
    # RESIDUAL   1 1 5 8 X
    return data


def fake_prediction_restore(data: Symbols) -> Symbols:
    # the prediction used is always 0
    # reverse the prediction_extract
    return data


class CompressionPipeline[Symbol]:
    quantize: Quantize
    dequantize: Dequantize
    prediction_extract: PredictionExtract
    prediction_restore: PredictionRestore
    entropy_encode: EntropyEncode
    entropy_decode: EntropyDecode
    error_correction_integrate: ECCIntegrate
    error_correction_extract: ECCExtract
    transmission_send: TransmissionSend
    transmission_receive: TransmissionReceive
    compression_metrics: CompressionMetrics
    metrics: list[Metrics]

    def __init__(
        self,
        *,
        quantize: Quantize = fake_quantizer,
        dequantize: Dequantize = fake_dequantizer,
        prediction_extract: PredictionExtract = fake_prediction_extract,
        prediction_restore: PredictionRestore = fake_prediction_restore,
        entropy_encode: EntropyEncode = fake_entropy_encoder,
        entropy_decode: EntropyDecode = fake_entropy_decoder,
        ecc_integrate: ECCIntegrate = fake_error_correction_integrate,
        ecc_extract: ECCExtract = fake_error_correction_extract,
        transmission_send: TransmissionSend = fake_transmission_send,
        transmission_receive: TransmissionReceive = fake_transmission_receive,
        # compression_metrics: CompressionMetrics = lambda _raw, _transmitted: None,
        compression_metrics: CompressionMetrics = calculate_metrics,
    ):
        self.quantize = quantize
        self.dequantize = dequantize
        self.prediction_extract = prediction_extract
        self.prediction_restore = prediction_restore
        self.entropy_encode = entropy_encode
        self.entropy_decode = entropy_decode
        self.error_correction_integrate = ecc_integrate
        self.error_correction_extract = ecc_extract
        self.transmission_send = transmission_send
        self.transmission_receive = transmission_receive
        self.compression_metrics = compression_metrics
        self.metrics = []

    def sender_pipeline(self, data: Symbols) -> Symbols:
        quantization_index = self.quantize(data)
        logger.debug(f"first 10 bytes of quantization_index: {quantization_index[:10]}")
        prediction_residual = self.prediction_extract(quantization_index)
        logger.debug(
            f"first 10 bytes of prediction_residual: {prediction_residual[:10]}"
        )
        entropy_encoded = self.entropy_encode(prediction_residual)
        logger.debug(
            f"encoded first 10 bytes of entropy_encoded: {entropy_encoded[:10]}"
        )
        encoded_with_ecc = self.error_correction_integrate(entropy_encoded)
        return encoded_with_ecc  # transmitted data

    def receiver_pipeline(self, encoded_with_ecc: Symbols):
        # Prediction -> Quantize -> Entropy -> ECC -> Transmission
        # Transmission -> ECC -> EntropyDecode -> Dequantize -> Prediction
        entropy_encoded = self.error_correction_extract(encoded_with_ecc)
        logger.debug(
            f"decoding first 10 bytes of entropy_encoded: {entropy_encoded[:10]}"
        )
        entropy_decoded = self.entropy_decode(entropy_encoded)
        logger.debug(
            f"decoded first 10 bytes of entropy_decoded: {entropy_decoded[:10]}"
        )
        quantization_index = self.prediction_restore(entropy_decoded)
        reconstructed = self.dequantize(quantization_index)
        return reconstructed

    def run(self, raw_data: Symbols):
        transmitted = self.sender_pipeline(raw_data)
        reconstructed = self.receiver_pipeline(transmitted)
        metrics = self.compression_metrics(raw_data, transmitted, reconstructed)
        self.metrics.append(metrics)
        return reconstructed, metrics

    def _check(self, raw_data: Symbols):
        # #TODO: rename: test_with_data
        reconstructed, _metrics = self.run(raw_data)
        return reconstructed == raw_data

    def show_metrics_result(self):
        plot_rate_distortion(self.metrics)
        for m in self.metrics:
            logger.info(m)
            # or do something else with m

    def dump_metrics_result(self):
        execution_time_stamp = time()
        dt_object = datetime.fromtimestamp(execution_time_stamp)
        formatted_time = dt_object.strftime("%Y%m%d-%H%M%S")
        json.dump(
            self.metrics, open(OUTPUT_FOLDER / f"metrics_{formatted_time}.json", "w")
        )


def test_default_pipeline():
    pipeline = CompressionPipeline[Byte]()
    data = b"Hello World!"
    reconstructed, metrics = pipeline.run(data)
    assert reconstructed == data
    print("test_default_pipeline passed")


if __name__ == "__main__":
    test_default_pipeline()
