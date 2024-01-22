import numpy as np

from classical_encoding import TEST_RAW_IMAGE_PATH


# Parameters
IMAGE_WIDTH = 1000  # 替换
height = 800  # 替换为图像高度
channels = 3  # 替换为图像通道数


def prediction_extract(data_2d):
    # pad the image with #000 on the top and left
    data_2d = data_2d.astype(np.int16)
    padded = np.pad(data_2d, ((1, 0), (1, 0), (0, 0)), "constant")
    prediction = (padded[1:, :-1, :] + padded[:-1, 1:, :] + padded[:-1, :-1, :]) // 3
    residue = padded - prediction
    residue = residue.astype(np.int16)


def prediction(data_2d: np.ndarray):
    """
    Extract prediction residual from a 2D data
    """

    assert len(data_2d.shape) == 3, "data must be 2D and with a color band"
    data_2d = data_2d.astype(np.int16)
    residue = np.zeros_like(data_2d, dtype=np.int16)
    for c in range(data_2d.shape[2]):
        for i in range(1, data_2d.shape[0]):
            for j in range(1, data_2d.shape[1]):
                average = (
                    data_2d[i - 1, j, c]
                    + data_2d[i, j - 1, c]
                    + data_2d[i - 1, j - 1, c]
                ) // 3
                residue[i, j, c] = data_2d[i, j, c] - average
    residue[0, :, :] = data_2d[0, :, :]
    residue[:, 0, :] = data_2d[:, 0, :]
    return residue


def prediction_restore(residual_2d):
    reconstructed = np.zeros_like(residual_2d, dtype=np.int16)
    for c in range(reconstructed.shape[2]):
        reconstructed[:, :, c][0, :] = residual_2d[:, :, c][0, :]
        reconstructed[:, :, c][:, 0] = residual_2d[:, :, c][:, 0]

        for i in range(1, reconstructed.shape[0]):
            for j in range(1, reconstructed.shape[1]):
                average = (
                    reconstructed[i - 1, j, c]
                    + reconstructed[i, j - 1, c]
                    + reconstructed[i - 1, j - 1, c]
                ) // 3
                reconstructed[i, j, c] = average + residual_2d[i, j, c]

    reconstructed = reconstructed.astype(np.uint8)
    return reconstructed


if __name__ == "__main__":
    raw_image_path = TEST_RAW_IMAGE_PATH

    try:
        with open(raw_image_path, "rb") as file:
            raw_data: np.ndarray = np.frombuffer(file.read(), dtype=np.uint8).reshape(
                (height, IMAGE_WIDTH, channels)
            )
    except FileNotFoundError:
        print(f"test file at {raw_image_path} not found.")
        exit(1)

    predicted_image = prediction(raw_data)
    predicted_image2 = prediction_extract(raw_data)
    assert np.all(predicted_image == predicted_image2)

    reconstructed_image = prediction_restore(predicted_image)
    print(raw_data.shape, reconstructed_image.shape)
    print(np.all(raw_data == reconstructed_image))

    def save_as_raw(image_array, output_path):
        with open(output_path, "wb") as file:
            file.write(image_array.tobytes())

    # save_as_raw(predicted_image, 'predicted_image.raw')
    save_as_raw(reconstructed_image, "reconstructed_image.raw")
