if __name__ != "__main__":
    raise ImportError(f"Script {__file__} should not be imported as a module")


from time import time
import numpy
from classical_encoding import RAW_DATASET_FOLDER
from classical_encoding.compression_pipeline.classical_pipeline import (
    CompressionPipeline,
    Quantize,
)
from classical_encoding.entropy_coding.naive_huffman import (
    naive_huffman_decode_from_bytes,
    naive_huffman_encode_to_bytes,
)
from classical_encoding.helper.typing import Byte
from classical_encoding.metrics.print_metric import calculate_metrics
from classical_encoding.prediction.basic_prediction import (
    DifferentialPulseCodeModulation2D,
)
from classical_encoding.quantization.uniform_scale_quantization import (
    UniformScaleQuantizer,
)
from classical_encoding.helper.logger import logger

# Params for the pipeline
IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_BANDS = 1000, 800, 3
dtype_in, dtype_safe = numpy.uint8, numpy.int16
QUANTIZATION_STEP = 7


quantizer = UniformScaleQuantizer(q_step=QUANTIZATION_STEP)
prediction = DifferentialPulseCodeModulation2D(
    IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_BANDS, numpy.uint8, numpy.int16
)

pipeline = CompressionPipeline[Byte](
    quantize=quantizer.quantize,
    dequantize=quantizer.dequantize,
    prediction_extract=prediction.extract,
    prediction_restore=prediction.restore,
    entropy_encode=naive_huffman_encode_to_bytes,
    entropy_decode=naive_huffman_decode_from_bytes,
    # ecc_integrate = # not implemented
    # ecc_extract = # not implemented
    # transmission_send = # not implemented
    # transmission_receive = # not implemented
    compression_metrics=calculate_metrics,
)

t = time()
finished = 0
file_path_list = list(RAW_DATASET_FOLDER.iterdir())
total = len(file_path_list)
interested_img_range = slice(20, 80)
interested_img_count = len(file_path_list[interested_img_range])

logger.info(f"running test on {interested_img_count} images")
for img in file_path_list[interested_img_range]:
    try:
        logger.debug(f"Testing image {finished}/{interested_img_count} at {img}")
        img_buffer = img.read_bytes()
        img_as_list = numpy.frombuffer(img_buffer, dtype=dtype_in).tolist()
        decompressed_img, _ = pipeline.run(img_as_list)
        # compressed_img = pipeline.sender_pipeline(img_list)
        # decompressed_img = pipeline.receiver_pipeline(compressed_img)

        for i, d in zip(img_as_list, decompressed_img, strict=True):
            error = abs(i - d)
            assert (
                error <= quantizer.peak_absolute_errors
            ), f"error {error} too big for {i} and {d}"

        logger.info(
            f"image {finished+1}/{interested_img_count} at {img} passed in {time() - t} second"
        )
        finished += 1
    except Exception as e:
        raise Exception(f"{img} failed")
    print(f"{img} passed")

pipeline.dump_metrics_result()
pipeline.show_metrics_result()
print("All tests passed")
