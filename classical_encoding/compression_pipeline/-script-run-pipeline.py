if __name__ != "__main__":
    raise ImportError(f"Script {__file__} should not be imported as a module")


from time import time
import numpy
from classical_encoding import RAW_DATASET_FOLDER, entropy_coding
from classical_encoding.compression_pipeline.classical_pipeline import (
    CompressionPipeline,
)
from classical_encoding.entropy_coding.adaptive_huffman import AdaptiveHuffman
from classical_encoding.helper.typing import Byte
from classical_encoding.prediction.basic_prediction import NaiveImagePrediction2D
from classical_encoding.quantization.uniform_scale_quantization import (
    UniformScaleQuantizer,
)

# Params for the pipeline
IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_BANDS = 1000, 800, 3
dtype_in, dtype_safe = numpy.uint8, numpy.int16


quantizer = UniformScaleQuantizer(q_step=3)
prediction = NaiveImagePrediction2D(
    IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_BANDS, numpy.uint8, numpy.int16
)
entropy_coding = AdaptiveHuffman()

pipeline = CompressionPipeline[Byte](
    quantize=quantizer.quantize,
    dequantize=quantizer.dequantize,
    prediction_extract=prediction.extract,
    prediction_restore=prediction.restore,
    entropy_encode=entropy_coding.encode,
    entropy_decode=entropy_coding.decode,
    # ecc_integrate = # not implemented
    # ecc_extract = # not implemented
    # transmission_send = # not implemented
    # transmission_receive = # not implemented
    # compression_metrics = @JJSUN
)

t = time()
finished = 0
total = len(list(RAW_DATASET_FOLDER.iterdir()))

for img in RAW_DATASET_FOLDER.iterdir():
    try:
        print(f"Testing image {finished}/{total} at {img}")
        img_buffer = img.read_bytes()
        img_list = numpy.frombuffer(img_buffer, dtype=dtype_in).tolist()
        compressed_img = pipeline.sender_pipeline(img_list)
        decompressed_img = pipeline.receiver_pipeline(compressed_img)
        assert img_list == decompressed_img
        print(f"image {finished}/{total} at {img} passed in {time() - t} second")
        finished += 1
    except Exception as e:
        raise Exception(f"{img} failed")
    print(f"{img} passed")

print("All tests passed")
