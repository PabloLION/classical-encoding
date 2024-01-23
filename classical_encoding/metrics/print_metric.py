from datetime import datetime
import json
from pathlib import Path
from re import M
from time import time
from typing import NamedTuple
import numpy as np
from classical_encoding import OUTPUT_FOLDER

from classical_encoding.helper.typing import Bytes, Metrics


def calculate_mse(original: Bytes, reconstructed: Bytes) -> float:
    mse = float(np.mean((np.array(original) - np.array(reconstructed)) ** 2))
    return mse


def calculate_psnr(mse: float, max_pixel_value=255.0) -> float:
    if mse == 0:
        return float("inf")
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr


def calculate_bps(original_data: Bytes, transmitted: Bytes) -> float:
    return float(len(transmitted) * 8 / len(original_data))


def calculate_metrics(
    original_file: Bytes, transmitted: Bytes, reconstructed_file: Bytes
) -> Metrics:
    bps = calculate_bps(original_file, transmitted)
    mse = calculate_mse(original_file, reconstructed_file)
    max_pixel_value = max(original_file)
    psnr = calculate_psnr(mse, max_pixel_value)
    return Metrics(psnr, mse, bps)


class MetricsHandle:
    metrics: list[Metrics]

    def __init__(self):
        self.metrics = []

    def append(self, metrics: Metrics):
        self.metrics.append(metrics)

    def load_extend_metrics(self, path: Path | str):
        self.metrics = json.load(open(path, "r"))
        for m in self.metrics:
            print(m)
            self.append(Metrics(*m))

    def dump_metrics(self, path: Path | str | None = None):
        if not path:
            execution_time_stamp = time()
            dt_object = datetime.fromtimestamp(execution_time_stamp)
            formatted_time = dt_object.strftime("%Y%m%d-%H%M%S")
            path = OUTPUT_FOLDER / f"metrics_{formatted_time}.json"
        assert path
        json.dump(self.metrics, open(path, "w"))

    def __iter__(self):
        return iter(self.metrics)
