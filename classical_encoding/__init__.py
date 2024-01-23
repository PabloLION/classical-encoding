from os import listdir
from pathlib import Path
from subprocess import run


PACKAGE_ROOT = Path(__file__).parent

# test data
TEST_RAW_IMAGE_PATH = PACKAGE_ROOT / "test_data" / "img_0000.raw"

TEST_RESULT_FOLDER = PACKAGE_ROOT / "test_result"
TEST_RESULT_FOLDER.mkdir(exist_ok=True)

# dataset

DATASET_FOLDER = PACKAGE_ROOT / "dataset"
if not DATASET_FOLDER.exists():
    DATASET_FOLDER.mkdir()
if not DATASET_FOLDER.is_dir():
    raise ValueError("Dataset folder is not a directory.")
if not listdir(DATASET_FOLDER):
    print("Downloading dataset from GitHub...")
    # download dataset from github
    dataset_url = r"https://github.com/LuckyRxy/dataset/"
    commit_hash = "3c1fb40f91baa35b64cc80be05ccfa1284c1cb82"
    # # clone and pull dataset
    # run(f"git clone {dataset_url} {DATASET_FOLDER}")
    # run(f"git pull {dataset_url} {DATASET_FOLDER}")
    # # checkout to the commit
    # run(f"git checkout {commit_hash}")

    run(["git", "clone", dataset_url, str(DATASET_FOLDER)], check=True)
    run(["git", "-C", str(DATASET_FOLDER), "pull", dataset_url], check=True)
    # checkout to the commit
    run(["git", "-C", str(DATASET_FOLDER), "checkout", commit_hash], check=True)
RAW_DATASET_FOLDER = DATASET_FOLDER / "raw"
assert RAW_DATASET_FOLDER.exists()
assert RAW_DATASET_FOLDER.is_dir()
assert len(listdir(RAW_DATASET_FOLDER)) == 610  # we have 610 raw images

# output
OUTPUT_FOLDER = PACKAGE_ROOT / "compression_pipeline" / "test-result"
OUTPUT_FOLDER.mkdir(exist_ok=True)
