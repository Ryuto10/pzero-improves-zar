# coding=utf-8

import gzip
from pathlib import Path
from typing import Generator

from logzero import logger


def read_lines(file_path: str, print_log: bool = True) -> Generator[str, None, None]:
    assert Path(file_path).exists(), f"Not found: {file_path}"
    if print_log:
        logger.info(f"Load: {file_path}")

    if file_path.endswith(".gzip") or file_path.endswith(".gz"):
        with gzip.open(filename=file_path, mode="rt", encoding="utf_8") as fi:
            for line in fi:
                yield line.rstrip("\n")

    else:
        with open(file_path) as fi:
            for line in fi:
                yield line.rstrip("\n")


def count_n_lines(file_path: str) -> int:
    for idx, _ in enumerate(read_lines(file_path, print_log=False), 1):
        pass
    return idx
