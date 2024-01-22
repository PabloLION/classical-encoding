import numpy as np


def read_data(file_path, dtype):
    data = np.fromfile(file_path, dtype=dtype)
    return data


def calculate_mse(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    return mse


def calculate_psnr(mse, max_pixel_value=255.0):
    if mse == 0:
        return float("inf")
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr


def calculate_metrics(original_file, reconstructed_file, dtype):
    original_data = read_data(original_file, dtype)
    reconstructed_data = read_data(reconstructed_file, dtype)

    mse = calculate_mse(original_data, reconstructed_data)

    psnr = calculate_psnr(mse)

    print(f"MSE: {mse}, PSNR: {psnr}")
    return mse, psnr


# # 示例用法
# if __name__ == "__main__":
#     original_file = 'mandrill.raw'
#     reconstructed_file = 'mandrill_reconstructed.raw'
#     dtype = np.dtype('>u1')  # 数据类型

#     calculate_metrics(original_file, reconstructed_file, dtype)
