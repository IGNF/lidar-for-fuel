"""
LiDAR file validation utility.
"""

import os
import logging

try:
    import laspy
    from laspy.errors import LaspyException
except Exception:
    laspy = None
    LaspyException = Exception

logger = logging.getLogger(__name__)


def check_lidar_file(input_file: str) -> "laspy.LasData":
    """
    Validate and load a LiDAR file (.las or .laz).

    Args:
        input_file: Path to .las or .laz file.

    Returns:
        laspy.LasData: Loaded and validated LAS data.

    Raises:
        ImportError: If the `laspy` library is not installed, or if a LAZ backend
        (such as `lazrs`) is required but missing to decompress LAZ files.
        ValueError : If the input path is not a non-empty string, or if the file
        extension is not `.las` or `.laz`.
        FileNotFoundError: If the input file does not exist at the given path.
        IOError: If `laspy` fails to read the file due to I/O or data corruption issues.
    """
    if laspy is None:
        raise ImportError("Install laspy: pip install laspy")

    # Path validation
    if not isinstance(input_file, str) or not input_file.strip():
        raise ValueError("Path must be a non-empty string")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")

    ext = os.path.splitext(input_file)[1].lower()
    if ext not in (".las", ".laz"):
        raise ValueError(f"Unsupported extension: {ext}")

    # Read with LAZ backend handling
    try:
        las = laspy.read(input_file)
    except LaspyException as e:
        error_msg = str(e)
        if any(kw in error_msg for kw in ["No LazBackend", "cannot decompress", "LazBackend"]):
            raise ImportError(
                f"LAZ error: {error_msg}\nInstall: pip install 'laspy[lazrs]' or 'lazrs'"
            ) from e
        raise IOError(f"laspy read error: {error_msg}") from e

    # Conformity check
    num_points = len(las.points)
    header_count = getattr(las.header, "point_count", None)
    if header_count and header_count != num_points:
        logger.warning("Header points mismatch: %s vs %s", header_count, num_points)
    
    if num_points == 0:
        logger.warning("Empty file: %s", input_file)

    logger.info("✓ Valid LiDAR: %s (%s points)", input_file, num_points)
    return las
