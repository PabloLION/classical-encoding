from typing import NamedTuple
import matplotlib.pyplot as plt

class Metrics(NamedTuple):
    psnr: float
    mse: float
    bps: float

def plot_rate_distortion(metrics: list[Metrics]):
    bps_list= [m.bps for m in metrics]
    psnr_list= [m.bps for m in metrics]
    plt.figure()
    plt.plot(bps_list, psnr_list, "bo")
    plt.xlabel("Bits Per Sample (bps)")
    plt.ylabel("Peak Signal-to-Noise Ratio (PSNR)")
    plt.title("PSNR vs. Bits Per Sample")
    plt.grid(True)
    plt.show()
