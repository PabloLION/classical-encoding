from itertools import zip_longest
from pathlib import Path
from typing import Iterable
from PIL import Image
import numpy


def compare_and_show_diff(target: Iterable, source: Iterable) -> bool:
    """Compare two iterables and show the differences"""
    for i, (t, s) in enumerate(zip_longest(target, source)):
        if t != s:
            print(f"diff at {i}: {t} != {s}")
            return False
    return True


def save_as_png(image_ndarray: numpy.ndarray, output_path: Path | str):
    image = Image.fromarray(image_ndarray)
    image.save(output_path)
