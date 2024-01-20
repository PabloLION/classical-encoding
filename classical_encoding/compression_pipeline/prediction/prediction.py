import numpy as np
from PIL import Image


def save_as_raw(image_array, output_path):
    with open(output_path, "wb") as file:
        file.write(image_array.tobytes())

def prediction(original_image):
    original_image = original_image.astype(np.int16)

    prediction_residual = np.zeros_like(original_image, dtype=np.int16)
    for c in range(original_image.shape[2]):
        for i in range(1, original_image.shape[0]):
            for j in range(1, original_image.shape[1]):
                average = (original_image[i - 1, j, c] + original_image[i, j - 1, c] + original_image[
                    i - 1, j - 1, c]) // 3
                prediction_residual[i, j, c] = original_image[i, j, c] - average

        prediction_residual[:, :, c][0, :] = original_image[:, :, c][0, :]
        prediction_residual[:, :, c][:, 0] = original_image[:, :, c][:, 0]

    return prediction_residual


def remove_prediction(prediction_residual):
    reconstructed = np.zeros_like(prediction_residual, dtype=np.int16)
    for c in range(reconstructed.shape[2]):
        reconstructed[:, :, c][0, :] = prediction_residual[:, :, c][0, :]
        reconstructed[:, :, c][:, 0] = prediction_residual[:, :, c][:, 0]

        for i in range(1, reconstructed.shape[0]):
            for j in range(1, reconstructed.shape[1]):
                average = (reconstructed[i - 1, j, c] + reconstructed[i, j - 1, c] + reconstructed[
                    i - 1, j - 1, c]) // 3
                reconstructed[i, j, c] = average + prediction_residual[i, j, c]

    reconstructed = reconstructed.astype(np.uint8)
    return reconstructed


# 读取RAW文件为NumPy数组
raw_image_path = "img_0000.raw"
width = 1000  # 替换为图像宽度
height = 800  # 替换为图像高度
channels = 3  # 替换为图像通道数

with open(raw_image_path, "rb") as file:
    raw_data = np.frombuffer(file.read(), dtype=np.uint8).reshape((height, width, channels))

# 对图像进行RGB通道的像素预测
predicted_image = prediction(raw_data)

reconstructed_image = remove_prediction(predicted_image)
print(raw_data.shape,reconstructed_image.shape)
print(np.all(raw_data == reconstructed_image))

# print(raw_data)
print(raw_data[:, :, 0])
print(raw_data[:, :, 1])
print(raw_data[:, :, 2])
print('----------------------------------------')
# print(predicted_image)
print(predicted_image[:, :, 0])
print(predicted_image[:, :, 1])
print(predicted_image[:, :, 2])
print('----------------------------------------')
# print(reconstructed_image)
print(reconstructed_image[:, :, 0])
print(reconstructed_image[:, :, 1])
print(reconstructed_image[:, :, 2])

# save_as_raw(predicted_image, 'predicted_image.raw')
save_as_raw(reconstructed_image, 'reconstructed_image.raw')
