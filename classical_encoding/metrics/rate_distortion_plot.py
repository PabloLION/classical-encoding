from typing import NamedTuple
import matplotlib.pyplot as plt
from classical_encoding.helper.typing import Metrics
from classical_encoding.metrics.print_metric import MetricsHandle


def plot_rate_distortion(metrics: list[Metrics] | MetricsHandle):
    bps_list = [m.bps for m in metrics]
    psnr_list = [m.psnr for m in metrics]
    plt.figure()
    plt.plot(bps_list, psnr_list, "bo")
    plt.xlabel("Bits Per Sample (bps)")
    plt.ylabel("Peak Signal-to-Noise Ratio (PSNR)")
    plt.title("PSNR vs. Bits Per Sample")
    plt.grid(True)
    plt.show()
