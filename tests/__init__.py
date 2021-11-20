import sys
from pathlib import Path

SRC_DIR = str(Path(__file__).absolute().parent.parent.joinpath("src"))
sys.path.append(SRC_DIR)
