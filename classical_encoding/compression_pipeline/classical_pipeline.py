from typing import Any, Callable, Generic
from classical_encoding.metrics.print_metric import calculate_metrics
from metrics.rate_distortion_plot import plot_rate_distortion
from classical_encoding.helper.typing import Bytes, Symbol, Symbols


type Quantize = Callable[[Bytes], Bytes]
type Dequantize = Callable[[Bytes], Bytes]
type PredictionExtract = Callable[[Any], Bytes]  # #TODO: Any
type PredictionRestore = Callable[[Any], Bytes]  # #TODO: Any
type ECCIntegrate = Callable[[Bytes], Bytes]  # data -> data_with_ecc
type ECCExtract = Callable[[Bytes], Bytes]  # data_with_ecc -> data
type TransmissionSend = Callable[[Bytes], Any]  # #TODO: Any
type TransmissionReceive = Callable[[Any], Bytes]  # #TODO: Any
type EntropyEncode = Callable[[Bytes], Bytes]
type EntropyDecode = Callable[[Bytes], Bytes]
type CompressionMetrics = Callable[[Bytes, Bytes], Any]
# input the data before and after compression, output the metrics


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
) = (identity,) * 7


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

def my_compression_metrics(raw_data: Symbols, transmitted: Symbols, reconstructed: Symbols) -> dict:
    plot_rate_distortion(raw_data, transmitted)
    return calculate_metrics(raw_data, reconstructed)


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
        compression_metrics: CompressionMetrics = my_compression_metrics
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

    def sender_pipeline(self, data: Symbols) -> Symbols:
        quantization_index = self.quantize(data)
        prediction_residual = self.prediction_extract(quantization_index)
        entropy_encoded = self.entropy_encode(prediction_residual)
        encoded_with_ecc = self.error_correction_integrate(entropy_encoded)
        return encoded_with_ecc  # transmitted data

    def receiver_pipeline(self, encoded_with_ecc: Symbols):
        # Prediction -> Quantize -> Entropy -> ECC -> Transmission
        # Transmission -> ECC -> EntropyDecode -> Dequantize -> Prediction
        encoded = self.error_correction_extract(encoded_with_ecc)
        entropy_decoded = self.entropy_decode(encoded)
        prediction_residual = self.dequantize(entropy_decoded)
        reconstructed = self.prediction_restore(prediction_residual)
        return reconstructed

    def run(self, raw_data: Symbols):
        transmitted = self.sender_pipeline(raw_data)
        reconstructed = self.receiver_pipeline(transmitted)
        metrics = self.compression_metrics(raw_data, transmitted, reconstructed)
        return reconstructed, metrics

    def _check(self, raw_data: Symbols):
        reconstructed, _metrics = self.run(raw_data)
        return reconstructed == raw_data


def test_default_pipeline():
    pipeline = CompressionPipeline[int]()
    data = b"Hello World!"
    reconstructed, metrics = pipeline.run(data)
    assert reconstructed == data
    assert metrics is None
    print("test_default_pipeline passed")


if __name__ == "__main__":
    test_default_pipeline()
