"""
Helper functions for ModelArena.
"""
import time
import logging
from typing import Any, Union


# Configure standard logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

def get_logger(name: str) -> logging.Logger:
    """Returns a pre-configured logger."""
    return logging.getLogger(f"ModelArena.{name}")


def format_bytes(size_bytes: int) -> str:
    """Formats raw byte count into human readable size string."""
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    i = 0
    size = float(size_bytes)
    while size >= 1024 and i < len(units) - 1:
        size /= 1024.0
        i += 1
    return f"{size:.2f} {units[i]}"


def format_metric(val: Union[float, int, str], precision: int = 4) -> str:
    """Safely formats metric numbers to standard decimal precision."""
    if isinstance(val, (float, np.floating)):
        if np.isnan(val):
            return "N/A"
        return f"{val:.{precision}f}"
    elif isinstance(val, (int, np.integer)):
        return str(val)
    return str(val)


class Timer:
    """Context manager and stopwatch for timing model execution."""
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()

    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time
