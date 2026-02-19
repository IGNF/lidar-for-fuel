"""
LiDAR file validation utility.
"""
import logging
import os

import pdal

logger = logging.getLogger(__name__)


def check_lidar_file(input_file: str, spatial_ref: str) -> pdal.Pipeline:
    """
    Validate and load a LiDAR file (.las or .laz).

    Args:
        input_file: Path to .las or .laz file.
        spatial_ref (str): spatial reference to use when reading las file.

    Returns:
        pdal.Pipeline: Loaded and validated PDAL Pipeline.

    Raises:
        ImportError: If the `pdal` library is not installed.
        ValueError: If the input path is not a non-empty string, or if the file
            extension is not `.las` or `.laz`.
        FileNotFoundError: If the input file does not exist at the given path.
        IOError: If PDAL fails to read the file due to I/O or data corruption issues.
    """
    if pdal is None:
        raise ImportError("Install pdal: pip install pdal")

    # Path validation
    if not isinstance(input_file, str) or not input_file.strip():
        raise ValueError("Path must be a non-empty string")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")

    ext = os.path.splitext(input_file)[1].lower()
    if ext not in (".las", ".laz"):
        raise ValueError(f"Unsupported extension: {ext}")

    # Read with pdal
    try:
        pipeline = pdal.Pipeline() | pdal.Reader.las(filename=input_file, override_srs=spatial_ref, nosrs=True)
        arrays = pipeline.arrays
        if not arrays:
            raise IOError("No points read from file")
    except Exception as e:
        raise IOError(f"PDAL read error: {str(e)}") from e

    num_points = len(arrays[0])

    if num_points == 0:
        logger.warning("Empty file: %s", input_file)

    logger.info("✓ Valid LiDAR: %s (%s points)", input_file, num_points)

    return pipeline
