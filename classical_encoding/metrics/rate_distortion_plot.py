from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt


class Metrics(NamedTuple):
    psnr: float
    mse: float
    bps: float


def calculate_metrics(original, transmitted, reconstructed) -> Metrics:
    # create a `Metrics` Metrics(calc_psnr, mse, bps)
    ...


def calculate_bps(compressed_size_bytes, num_samples):
    return (compressed_size_bytes * 8) / num_samples


def calculate_mse(orig_data, comp_data):
    return np.mean((orig_data - comp_data) ** 2)


def calculate_psnr(mse, max_pixel_value=255):
    if mse == 0:
        return float("inf")
    return 20 * np.log10(max_pixel_value / np.sqrt(mse))


def process_and_plot(bps, psnr):
    plt.figure()
    plt.plot(bps, psnr, "bo")
    plt.xlabel("Bits Per Sample (bps)")
    plt.ylabel("Peak Signal-to-Noise Ratio (PSNR)")
    plt.title("PSNR vs. Bits Per Sample")
    plt.grid(True)
    plt.show()


def calculate_and_plot_rate_distortion(raw_data: bytes, transmitted: bytes):
    # 将字节数据转换为NumPy数组 #TODO 需要确认
    raw_data_np = np.frombuffer(raw_data, dtype=np.uint8)
    transmitted_np = np.frombuffer(transmitted, dtype=np.uint8)

    # 假设每个字节代表一个样本
    num_samples = len(raw_data)  # 样本总量

    # 计算bps和PSNR
    bps = calculate_bps(len(transmitted), num_samples)
    mse = calculate_mse(raw_data_np, transmitted_np)
    psnr = calculate_psnr(mse)

    # 调用绘图函数
    plot_rate_distortion(bps, psnr)


def plot_rate_distortion(bps_list, psnr_list):
    plt.figure()
    plt.plot(bps_list, psnr_list, "bo")
    plt.xlabel("Bits Per Sample (bps)")
    plt.ylabel("Peak Signal-to-Noise Ratio (PSNR)")
    plt.title("PSNR vs. Bits Per Sample")
    plt.grid(True)
    plt.show()
