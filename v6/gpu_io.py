from __future__ import annotations

import logging
import os
import tempfile

import cudf
import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(path: str, skiprows: int = 0, nrows: int = None) -> cudf.DataFrame:
    """Load CSV via pandas, normalize columns, convert to cudf. ONE conversion.

    Parameters
    ----------
    path : str
        CSV file path.
    skiprows : int
        Number of data rows to skip (header always read).
    nrows : int
        Maximum number of data rows to read.
    """
    read_kwargs: dict = {}
    if skiprows > 0:
        read_kwargs["skiprows"] = range(1, skiprows + 1)
    if nrows is not None:
        read_kwargs["nrows"] = nrows
    pdf = pd.read_csv(path, **read_kwargs)
    pdf.columns = [col.strip().lower() for col in pdf.columns]
    if "timestamp" in pdf.columns:
        pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], utc=True).dt.tz_localize(None)
    return cudf.from_pandas(pdf)


def save_csv(df: cudf.DataFrame, path: str, chunk_size: int = 500000) -> None:
    """Write cudf DataFrame to CSV via chunked to_pandas().to_csv().

    Parameters
    ----------
    df : cudf.DataFrame
        GPU-backed DataFrame to write.
    path : str
        Destination CSV file path.
    chunk_size : int
        Number of rows per chunk. Defaults to 500,000.
    """
    total = len(df)
    logger.info("Writing %d rows to '%s' (chunk_size=%d)", total, path, chunk_size)

    offset = 0
    written = 0

    while offset < total:
        end = min(offset + chunk_size, total)
        # Use pyarrow bridge (to_arrow().to_pandas()) because cudf.to_pandas()
        # fails in this WSL environment (numba CUDA bindings cannot enumerate GPUs)
        chunk = df.iloc[offset:end].to_arrow().to_pandas()

        mode = "w" if offset == 0 else "a"
        header = offset == 0
        chunk.to_csv(path, mode=mode, header=header, index=False)

        written += len(chunk)
        offset = end
        logger.debug("Wrote chunk: %d/%d rows", written, total)

    logger.info("Finished writing %d rows to '%s'", written, path)


def save_csv_safe(df: cudf.DataFrame, path: str, chunk_size: int = 500000) -> None:
    """Atomic write: save_csv to temp file, then os.replace for atomicity.

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
        save_csv(df, tmp_path, chunk_size=chunk_size)
        os.replace(tmp_path, path)
        logger.info("Atomic write complete: '%s' -> '%s'", tmp_path, path)
    except BaseException:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
