import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def calculate_mse(image1, image2):
    return np.mean((np.array(image1) - np.array(image2)) ** 2)

def calculate_psnr(mse, max_pixel_value=255.0):
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

def calculate_bps(compressed_image_size, num_samples):
    return (compressed_image_size * 8) / num_samples

def rate_distortion(original_image_path, reconstructed_files):
    original_image = Image.open(original_image_path)
    num_samples = np.prod(original_image.size)

    distortions = []
    bitrates = []

    for rec_file in reconstructed_files:
        compressed_image_size = os.path.getsize(rec_file)
        compressed_image = Image.open(rec_file)

        mse = calculate_mse(original_image, compressed_image)
        psnr = calculate_psnr(mse)
        bps = calculate_bps(compressed_image_size, num_samples)
        distortions.append(psnr)
        bitrates.append(bps)

    plt.figure(figsize=(10, 5))
    plt.plot(bitrates, distortions, 'bo-')
    plt.xlabel('Bitrate (bps)')
    plt.ylabel('Distortion (MSE)')
    plt.title('Rate-Distortion Plot')
    plt.grid(True)
    plt.show()

# if __name__ == "__main__":
#     original_image_path = 'cat.jpg'
#     reconstructed_files = ['cat_compressed_10.jpg', 'cat_compressed_20.jpg', 'cat_compressed_30.jpg', ...]  # 示例文件名
#     analyze_jpeg_compression(original_image_path, reconstructed_files)
