from pathlib import Path


PACKAGE_ROOT = Path(__file__).parent

# test data
TEST_RAW_IMAGE_PATH = PACKAGE_ROOT / "test_data" / "img_0000.raw"

TEST_RESULT_FOLDER = PACKAGE_ROOT / "test_result"
TEST_RESULT_FOLDER.mkdir(exist_ok=True)
