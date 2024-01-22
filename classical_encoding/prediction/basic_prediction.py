from functools import cache
import numpy
from classical_encoding import TEST_RAW_IMAGE_PATH
from classical_encoding.helper.test_util import save_as_png
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


class NaiveImagePrediction2D:
    """
    Naive 2D image prediction

    Attributes:
        height: image height
        width: image width
        n_band: number of bands (e.g. 1 for grayscale, 3 for RGB)
        dtype_in: data type of the raw image
        dtype_safe: safe data type for prediction
        dtype_residual: data type of the residual bytes, only uint16 is supported

    #TODO:
        U2: support other dtype_residual
    """

    height: int
    width: int
    n_band: int
    dtype_safe: type  # to avoid overflow and underflow
    dtype_raw: type  # data type of the raw image
    dtype_residual: type  # data type of the residual bytes

    def __len__(self) -> int:
        return self.height * self.width * self.n_band

    def __init__(
        self,
        image_width: int,
        image_height: int,
        n_band: int,
        dtype_raw: type,
        dtype_safe: type,
        dtype_residual: type,
    ):
        self.width = image_width
        self.height = image_height
        self.n_band = n_band
        self.dtype_raw = dtype_raw
        self.dtype_safe = dtype_safe  # to avoid overflow and underflow
        # #TODO: use dynamic dtype_safe from self.dtype for data overflow and underflow
        self.dtype_residual = dtype_residual  # also consider the endianess
        # #TODO: U2: support other dtype_residual

        # temporary assert for dtype
        assert self.dtype_raw == numpy.uint8, "only dtype_raw=numpy.uint8 is supported"
        assert (
            self.dtype_safe == numpy.int16
        ), "only dtype_safe=numpy.int16 is supported"
        assert (
            self.dtype_residual == numpy.int16
        ), "only dtype_residual=numpy.uint16 is supported"

    def extract(self, raw_data: Bytes) -> Bytes:
        img = self.to_ndarray(raw_data, self.dtype_raw)
        # pad the image with #000 on the top and left
        padded = numpy.pad(img, ((1, 0), (1, 0), (0, 0)), "constant")
        pred = (padded[1:, :-1, :] + padded[:-1, 1:, :] + padded[:-1, :-1, :]) // 3
        return self.to_bytes(img - pred, self.dtype_residual)

    def restore(self, residual: Bytes) -> Bytes:
        residual_2d = self.to_ndarray(residual, self.dtype_residual)
        # pad the image with #000 on the top and left
        restored = numpy.zeros_like(
            (self.height + 1, self.width + 1, self.n_band), dtype=numpy.int16
        )
        for h_idx in range(self.height):
            for w_idx in range(self.width):
                for b_idx in range(self.n_band):
                    n00 = restored[h_idx, w_idx, b_idx]
                    n01 = restored[h_idx, w_idx + 1, b_idx]
                    n10 = restored[h_idx + 1, w_idx, b_idx]
                    prediction = (n00 + n01 + n10) // 3  # at (i+1, j+1, c)
                    restored[h_idx + 1, w_idx + 1, b_idx] = (
                        residual_2d[h_idx, w_idx, b_idx] + prediction
                    )
        assert numpy.all(
            0 <= restored < 256
        ), "restored data should be 8 bits"  # #FIX: use another file
        # remove the padded #000 on the top and left and convert to uint8
        return self._assert_size(
            self.to_bytes(restored[1:, 1:, :], self.dtype_residual)
        )

    def to_ndarray(self, data: Bytes, dtype: type) -> numpy.ndarray:
        """
        Safely convert Bytes of `self.dtype` to ndarray of length `len(self)`,
        type `self.dtype_safe` and shape (self.height, self.width, self.n_band)
        """
        assert self._assert_size(data)
        self._assert_array_item_can_cast(data, dtype)
        ndarray = numpy.array(data, dtype).reshape(
            (self.height, self.width, self.n_band)
        )
        return ndarray

    def to_bytes(self, data: numpy.ndarray, dtype: type) -> Bytes:
        """
        Safely convert ndarray of shape (self.height, self.width, self.n_band),
        type `self.dtype` to Bytes of length `len(self)`
        """
        self._assert_array_item_can_cast(data, dtype)
        return data.flatten().tolist()

        # use ">" to force big-endian, ">u2" for uint16
        # assert self.dtype_residual == numpy.uint16  # #TODO: U2: support other types
        # return data.astype(self.dtype_residual).tobytes()

    def _assert_size(self, data: Bytes):
        msg = f"{len(data)} != {len(self)} == {self.height} * {self.width} * {self.n_band}"
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
    dtype_residual = (
        numpy.uint16
    )  # data type of the residual bytes, only uint16 is supported

    prediction = NaiveImagePrediction2D(
        1000, 800, 3, dtype_in, dtype_safe, dtype_residual
    )

    buffer = TEST_RAW_IMAGE_PATH.read_bytes()  # good
    raw_img = numpy.frombuffer(buffer, dtype=dtype_in)
    with open(TEST_RAW_IMAGE_PATH, "rb") as file:
        raw_img2 = numpy.frombuffer(file.read(), dtype=dtype_in)
    assert numpy.all(raw_img == raw_img2)

    raw_data = raw_img.tolist()
    a = numpy.array(raw_data, dtype_safe)
    print(a.size, a.shape, a.dtype)
    print(numpy.all(0 <= a) and numpy.all(a < 256))
    print(numpy.can_cast(a, numpy.uint8))
    residual = prediction.extract(raw_data)
    restored = prediction.restore(residual)
    save_as_png(raw_img, "raw.png")
    save_as_png(restored, "restored.png")
    assert restored == TEST_RAW_IMAGE_PATH.read_bytes()


if __name__ == "__main__":
    test_naive_image_prediction()
