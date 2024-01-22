from itertools import zip_longest
from typing import Iterable
from PIL import Image


def compare_and_show_diff(target: Iterable, source: Iterable) -> bool:
    """Compare two iterables and show the differences"""
    for i, (t, s) in enumerate(zip_longest(target, source)):
        if t != s:
            print(f"diff at {i}: {t} != {s}")
            return False
    return True


def save_as_png(image_ndarray, output_path):
    image = Image.fromarray(image_ndarray)
    image.save(output_path)
