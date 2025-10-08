# binpacking/core/logging_setup.py
from __future__ import annotations
import logging
from pathlib import Path
import csv
from datetime import datetime

def setup_logging(log_dir: Path, name: str = "binpacking") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # File handler
    fh = logging.FileHandler(log_dir / f"{name}.log")
    fh.setLevel(logging.INFO)
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    # Avoid duplicate handlers on reruns
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

class CSVWriter:
    """
    Minimal CSV writer for experiment rows (e.g., seed, N, M_off, etc.).
    """
    def __init__(self, path: Path, headers: list[str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.headers = headers
        if not path.exists():
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()

    def write_row(self, row: dict) -> None:
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(row)

def run_stamp() -> str:
    """
    Timestamp string for run folders.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")
