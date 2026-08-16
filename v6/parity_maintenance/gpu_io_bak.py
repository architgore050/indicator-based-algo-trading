from __future__ import annotations

import logging
import os
import tempfile
from typing import TYPE_CHECKING

import pandas as pd

# ---------------------------------------------------------------------------
# GPU Backend Detection (lazy, with graceful fallback — mirrors definitions.py)
# ---------------------------------------------------------------------------

_cupy_available: bool = False
_cudf_available: bool = False
_cp = None
_cudf = None

try:
    import cupy as _cp  # noqa: F401
    _cupy_available = True
except ImportError:
    pass

try:
    import cudf as _cudf  # noqa: F401
    _cudf_available = True
except ImportError:
    pass

if TYPE_CHECKING:
    import cudf as cudf_type

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def gpu_io_available() -> bool:
    """Return True if both cudf and cupy are available for GPU I/O."""
    return _cupy_available and _cudf_available


def load_csv_to_gpu(path: str) -> "cudf.DataFrame":
    """Load CSV from disk -> pandas read -> copy_to_device() into cudf.

    For large files (>1M rows), uses chunked reading to avoid CPU memory spikes.
    Returns a cudf DataFrame on GPU.

    Raises
    ------
    RuntimeError
        If cudf is not available.
    """
    if _cudf is None:
        raise RuntimeError(
            "cudf is not available. Install RAPIDS (cudf) to use GPU I/O."
        )

    total_rows = _count_csv_rows(path)
    logger.info("Loading '%s' -> GPU (estimated %d rows)", path, total_rows)

    if total_rows > 1_000_000:
        # Chunked read to avoid CPU memory spike
        chunks: list["cudf.DataFrame"] = []
        for chunk in pd.read_csv(path, chunksize=500_000):
            chunks.append(cudf.from_pandas(chunk))
        df = cudf.concat(chunks, ignore_index=True)
    else:
        pdf = pd.read_csv(path)
        df = cudf.from_pandas(pdf)

    logger.info(
        "Loaded %d rows x %d cols to GPU (%s)",
        len(df),
        len(df.columns),
        type(df).__name__,
    )
    return df


def gpu_to_csv(df: "cudf.DataFrame", path: str, chunk_size: int = 500000) -> None:
    """Write cudf DataFrame to CSV via .to_pandas().to_csv().

    For large DataFrames, writes in chunks to avoid OOM during GPU->CPU transfer.
    Uses mode='w' for first chunk, mode='a' for subsequent chunks (no header repeat).

    Parameters
    ----------
    df : cudf.DataFrame
        GPU-backed DataFrame to write.
    path : str
        Destination CSV file path.
    chunk_size : int
        Number of rows per chunk. Defaults to 500,000.
    """
    if not _cudf_available:
        raise RuntimeError(
            "cudf is not available. Install RAPIDS (cudf) to use GPU I/O."
        )

    total = len(df)
    logger.info("Writing %d rows to '%s' (chunk_size=%d)", total, path, chunk_size)

    offset = 0
    written = 0

    while offset < total:
        end = min(offset + chunk_size, total)
        chunk = df.iloc[offset:end]
        pdf_chunk = chunk.to_pandas()

        mode = "w" if offset == 0 else "a"
        header = offset == 0
        pdf_chunk.to_csv(path, mode=mode, header=header, index=False)

        written += len(pdf_chunk)
        offset = end
        logger.debug("Wrote chunk: %d/%d rows", written, total)

    logger.info("Finished writing %d rows to '%s'", written, path)


def gpu_to_csv_safe(df: "cudf.DataFrame", path: str, chunk_size: int = 500000) -> None:
    """Atomic write: gpu_to_csv to temp file, then os.replace for atomicity.

    If writing fails, the destination file is never left in a partial state.

    Parameters
    ----------
    df : cudf.DataFrame
        GPU-backed DataFrame to write.
    path : str
        Destination CSV file path.
    chunk_size : int
        Number of rows per chunk. Defaults to 500,000.
    """
    dir_name = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", prefix=".gpu_io_", dir=dir_name)
    os.close(fd)

    try:
        gpu_to_csv(df, tmp_path, chunk_size=chunk_size)
        os.replace(tmp_path, path)
        logger.info("Atomic write complete: '%s' -> '%s'", tmp_path, path)
    except BaseException:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _count_csv_rows(path: str) -> int:
    """Count data rows in a CSV (excludes header). Fast, no data loading."""
    with open(path, "r") as f:
        header = f.readline()  # skip header
        count = sum(1 for _ in f)
    return count
