from pathlib import Path
from time import time
from typing import Any
from unittest import result
import numpy
from classical_encoding import (
    RAW_DATASET_FOLDER,
    TEST_RAW_IMAGE_PATH,
    TEST_RESULT_FOLDER,
)
from classical_encoding.compression_pipeline.classical_pipeline import (
    CompressionPipeline,
)
from classical_encoding.helper.test_util import save_as_png
from classical_encoding.helper.typing import Bytes


class NumpyUtils:
    @staticmethod
    def int16_to_uint8_bytes(int16_arr: numpy.ndarray) -> Bytes:
        # #TODO: assert range
        high = (int16_arr >> 8).astype(numpy.uint8)
        low = (int16_arr & 0xFF).astype(numpy.uint8)
        # output array will be [h0,l0,h1,l1,...]
        return numpy.stack((high, low), axis=-1).flatten().tolist()

    @staticmethod
    def uint8_bytes_to_int16_ndarray(
        uint8_bytes: Bytes,
    ) -> numpy.ndarray[numpy.int16, Any]:  # safe Any
        uint8_ndarray = numpy.array(uint8_bytes)
        high = uint8_ndarray[::2].astype(numpy.int16)
        low = uint8_ndarray[1::2].astype(numpy.int16)
        # #TODO: assert range
        return ((high << 8) + low).astype(numpy.int16)


def test_int16_bytes_conversion():
    int16_1d_array = numpy.array(
        [1, 2, 3, 128, 256, 512, 513, 515, -1, -16, -512, -128], dtype=numpy.int16
    )
    uint8_bytes = NumpyUtils.int16_to_uint8_bytes(int16_1d_array)
    expected_str = "0,1,0,2,0,3,0,128,1,0,2,0,2,1,2,3,255,255,255,240,254,0,255,128"
    exp = list(map(int, expected_str.split(",")))
    assert uint8_bytes == exp, f"expected {exp} but got {uint8_bytes}"
    reconstructed_int16_1d_array = NumpyUtils.uint8_bytes_to_int16_ndarray(uint8_bytes)
    assert numpy.all(reconstructed_int16_1d_array == int16_1d_array)


if __name__ == "__main__":
    test_int16_bytes_conversion()


class DifferentialPulseCodeModulation1D:
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


class DifferentialPulseCodeModulation2D:
    """
    Naive 2D image prediction

    Attributes:
        height: image height
        width: image width
        n_band: number of bands (e.g. 1 for grayscale, 3 for RGB)
        dtype_in: data type of the raw image
        dtype_safe: safe data type for prediction

    #TODO:
        U2: support other dtype_safe
    """

    height: int
    width: int
    n_band: int
    dtype_safe: type  # to avoid overflow and underflow
    dtype_raw: type  # data type of the raw image

    def __len__(self) -> int:
        return self.height * self.width * self.n_band

    def __init__(
        self,
        image_width: int,
        image_height: int,
        n_band: int,
        dtype_raw: type,
        dtype_safe: type,
    ):
        self.width = image_width
        self.height = image_height
        self.n_band = n_band
        self.dtype_raw = dtype_raw
        self.dtype_safe = dtype_safe  # to avoid overflow and underflow
        # #TODO: use dynamic dtype_safe from self.dtype for data overflow and underflow
        # #TODO: U2: support other dtype_residual

        # temporary assert for dtype
        assert self.dtype_raw == numpy.uint8, "only dtype_raw=numpy.uint8 is supported"
        assert (
            self.dtype_safe == numpy.int16
        ), "only dtype_safe=numpy.int16 is supported"

    def extract_ndarray(self, raw_data: Bytes) -> numpy.ndarray:
        img = self.to_ndarray(raw_data, self.dtype_raw)
        # pad the image with #000 on the top and left
        padded = numpy.pad(img, ((1, 0), (1, 0), (0, 0)), "constant").astype(
            self.dtype_safe
        )
        pred = (
            padded[1:, :-1, :] // 3 + padded[:-1, 1:, :] // 3 + padded[:-1, :-1, :] // 3
        )
        return img - pred

    def restore_ndarray(self, residual_3d: numpy.ndarray) -> Bytes:
        residual_3d = residual_3d.astype(self.dtype_safe)
        # pad the image with #000 on the top and left
        restored = numpy.zeros(
            (self.height + 1, self.width + 1, self.n_band), dtype=self.dtype_safe
        )
        for h_idx in range(self.height):
            for w_idx in range(self.width):
                for b_idx in range(self.n_band):
                    n00 = restored[h_idx, w_idx, b_idx]
                    n01 = restored[h_idx, w_idx + 1, b_idx]
                    n10 = restored[h_idx + 1, w_idx, b_idx]
                    prediction = n00 // 3 + n01 // 3 + n10 // 3  # at (i+1, j+1, c)
                    value = residual_3d[h_idx, w_idx, b_idx] + prediction
                    restored[h_idx + 1, w_idx + 1, b_idx] = value

                    assert (
                        0 <= value < 256
                    ), f"{value=} at ({h_idx}, {w_idx}, {b_idx}), but restored data should be 8 bits"

        # #TODO: use can_cast to check if the restored data is 8 bits
        # #FIX: use another file

        # remove the padded #000 on the top and left and convert to uint8
        return self.to_bytes(restored[1:, 1:, :], self.dtype_raw)

    def extract(self, raw_data: Bytes) -> Bytes:
        residual_3d = self.extract_ndarray(raw_data)
        return self.to_bytes(residual_3d, self.dtype_safe)

    def restore(self, residual: Bytes) -> Bytes:
        int16_residual = NumpyUtils.uint8_bytes_to_int16_ndarray(residual)
        residual_3d = self.to_ndarray(int16_residual, self.dtype_safe)
        return self.restore_ndarray(residual_3d)

    def to_ndarray(self, data: Bytes | numpy.ndarray, dtype: type) -> numpy.ndarray:
        """
        Safely convert Bytes of `self.dtype` to ndarray of length `len(self)`,
        type `self.dtype_safe` and shape (self.height, self.width, self.n_band)
        """
        self._assert_size(data)
        # #TODO: U2: support other types
        self._assert_array_item_can_cast(data, dtype)
        ndarray = numpy.array(data, dtype).reshape(
            (self.height, self.width, self.n_band)
        )
        return ndarray

    def to_bytes(self, data: numpy.ndarray, dtype: type) -> Bytes:
        if dtype == numpy.uint8:
            return self.uint8_to_bytes(data, dtype)
        elif dtype == numpy.int16:
            return self.int16_to_bytes(data, dtype)
        else:
            raise NotImplementedError(f"unsupported dtype={dtype}")

    def uint8_to_bytes(self, data: numpy.ndarray, dtype: type) -> Bytes:
        self._assert_array_item_can_cast(data, dtype)
        return data.astype(dtype).flatten().tolist()

    def int16_to_bytes(self, data: numpy.ndarray, dtype: type) -> Bytes:
        """
        Safely convert ndarray of shape (self.height, self.width, self.n_band),
        type `self.dtype` to Bytes of length `len(self)`
        """
        self._assert_array_item_can_cast(data, dtype)
        return NumpyUtils.int16_to_uint8_bytes(data.astype(dtype).flatten())

        # use ">" to force big-endian, ">u2" for uint16-BE
        # assert self.dtype_safe == numpy.uint16  # #TODO: U2: support other types
        # return data.astype(self.dtype_safe).tobytes()

    def _assert_size[T: Bytes | numpy.ndarray](self, data: T) -> T:
        msg = f"{len(data)=} != {len(self) } == {self.height=} * {self.width=} * {self.n_band=}"
        assert len(data) == len(self), msg
        return data

    def _assert_array_item_can_cast(self, data: numpy.ndarray | Bytes, dtype: type):
        """
        Check if every item in the array can be cast to `dtype`
        #TODO: extract to a supportive file for numpy
        """
        it = (
            iter(data)
            if isinstance(data, list) or isinstance(data, tuple)
            else numpy.nditer(data)
        )
        for item in it:
            assert numpy.can_cast(item, dtype), f"{item} cannot be cast to {dtype}"
        return data


def test_naive_image_prediction():
    dtype_in = numpy.uint8  # data type of the raw image
    dtype_safe = numpy.int16  # safe data type for prediction
    image_height, image_width, n_band = 1000, 800, 3

    prediction = DifferentialPulseCodeModulation2D(
        image_height, image_width, n_band, dtype_in, dtype_safe
    )

    buffer = TEST_RAW_IMAGE_PATH.read_bytes()  # good
    img_list_int = list(buffer)

    # #TODO: remove raw_img
    raw_img = numpy.frombuffer(buffer, dtype=dtype_in)
    # same as with open(TEST_RAW_IMAGE_PATH, "rb") as file:
    #     raw_img = numpy.frombuffer(file.read(), dtype=dtype_in)

    raw_data = raw_img.tolist()
    a = numpy.array(raw_data, dtype_safe)
    print(a.size, a.shape, a.dtype)
    print(numpy.all(0 <= a) and numpy.all(a < 256))  # true
    # output:
    # 2400000 (2400000,) int16
    # True
    # False

    residual_3d = prediction.extract_ndarray(raw_data)
    restored_from_3d = prediction.restore_ndarray(residual_3d)
    print(len(restored_from_3d), len(img_list_int))
    assert restored_from_3d == img_list_int
    # OK until here

    residual = prediction.extract(raw_data)
    restored = prediction.restore(residual)
    source_image_ndarry = prediction.to_ndarray(img_list_int, dtype_in)
    restored_image = prediction.to_ndarray(restored, dtype_in)
    assert restored == img_list_int
    save_as_png(source_image_ndarry, TEST_RESULT_FOLDER / "raw.png")
    save_as_png(restored_image, TEST_RESULT_FOLDER / "restored.png")
    print(
        f"test passed: from {TEST_RAW_IMAGE_PATH}, generated raw.png and restored.png in {TEST_RESULT_FOLDER}"
    )


def test_naive_image_prediction_with_pipeline(file_bytes: Bytes):
    dtype_in, dtype_safe = numpy.uint8, numpy.int16
    image_height, image_width, n_band = 1000, 800, 3
    prediction = DifferentialPulseCodeModulation2D(
        image_height, image_width, n_band, dtype_in, dtype_safe
    )

    pipeline = CompressionPipeline(
        prediction_extract=prediction.extract,
        prediction_restore=prediction.restore,
    )

    pipeline._check(file_bytes)
    print("image prediction with pipeline test passed")


def test_naive_image_prediction_with_pipeline_given_path(path: Path):
    file_buffer = path.read_bytes()
    file_bytes = list(file_buffer)
    test_naive_image_prediction_with_pipeline(file_bytes)


def test_all_image_with_time():
    counter = 0
    current_time = time()
    for img in RAW_DATASET_FOLDER.iterdir():
        try:
            test_naive_image_prediction_with_pipeline_given_path(img)
        except Exception as e:
            print(f"{img} failed")
            raise e
        print(
            f"{img} passed in {time() - current_time:.2f}s at {time()}, total passed {counter} of 610"
        )
        counter += 1

    print("All tests passed")


if __name__ == "__main__":
    # test_naive_image_prediction()
    test_naive_image_prediction_with_pipeline_given_path(TEST_RAW_IMAGE_PATH)
    # test_all_image_with_time() # works but very slow (610*9s=1h30m)
