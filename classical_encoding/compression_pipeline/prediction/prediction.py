import numpy as np
from PIL import Image


def save_as_raw(image_array, output_path):
    with open(output_path, "wb") as file:
        file.write(image_array.tobytes())


def prediction(original_image):
    original_image = original_image.astype(np.int16)
    # predicted_image = np.zeros_like(original_image, dtype=np.int16)

    s0, s1, s2 = original_image.shape
    new_shape = (s0 + 1, s1 + 1, s2)
    predicted_image = np.zeros(new_shape, dtype=np.int16)
    for i in range(1, s0 + 1):
        for j in range(s1):
            for c in range(s2):
                average_value = (original_image[i - 1, j, c] + original_image[i, j - 1, c] + original_image[
                    i - 1, j - 1, c]) / 3
                # average_value = (original_image[i, j + 1, c] + original_image[i + 1, j, c] + original_image[
                #     i, j, c]) / 3
                predicted_image[i, j, c] = original_image[i, j, c] - average_value
                # predicted_value[i, j, c] = average_value

    return predicted_image


def remove_prediction(prediction_image):
    # original_image = np.zeros_like(prediction_image, dtype=np.uint8)
    s0, s1, s2 = prediction_image.shape
    new_shape = (s0 + 1, s1 + 1, s2)
    original_image = np.zeros(new_shape, dtype=np.int16)

    for i in range(prediction_image.shape[0]):
        for j in range(prediction_image.shape[1]):
            for c in range(prediction_image.shape[2]):
                average_value = (original_image[i, j + 1, c] + original_image[i + 1, j, c] + original_image[
                    i, j, c]) / 3
                original_image[i + 1, j + 1, c] = prediction_image[i, j, c] + average_value

    original_image = original_image[1:, 1:, :].astype(np.uint8)
    return original_image


# 读取RAW文件为NumPy数组
raw_image_path = "img_0000.raw"
width = 1000  # 替换为图像宽度
height = 800  # 替换为图像高度
channels = 3  # 替换为图像通道数

with open(raw_image_path, "rb") as file:
    raw_data = np.frombuffer(file.read(), dtype=np.uint8).reshape((height, width, channels))

# 对图像进行RGB通道的像素预测
predicted_image = prediction(raw_data)

original_image = remove_prediction(predicted_image)
print(np.all(raw_data == original_image))
exit()

print(raw_data)
print(raw_data[:, :, 0])
print(raw_data[:, :, 1])
print(raw_data[:, :, 2])
print('----------------------------------------')
print(predicted_image)
print(predicted_image[:, :, 0])
print(predicted_image[:, :, 1])
print(predicted_image[:, :, 2])

save_as_raw(predicted_image, 'predicted_image.raw')
save_as_raw(original_image, 'original_image.raw')
