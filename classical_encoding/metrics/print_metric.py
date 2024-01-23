from typing import NamedTuple
import numpy as np

from classical_encoding.helper.typing import Metrics


def calculate_mse(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    return mse


def calculate_psnr(mse, max_pixel_value=255.0):
    if mse == 0:
        return float("inf")
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr


def calculate_bps(original_data, transmitted):
    return (transmitted * 8) / original_data


def calculate_metrics(original_file, transmitted, reconstructed_file) -> Metrics:
    original_data = len(original_file)
    transmitted_data = len(transmitted)
    reconstructed_data = len(reconstructed_file)

    bps = calculate_bps(original_data, transmitted_data)
    mse = calculate_mse(original_data, reconstructed_data)
    psnr = calculate_psnr(mse)
    return Metrics(psnr, mse, bps)
