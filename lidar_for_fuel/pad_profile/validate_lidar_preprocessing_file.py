"""
LiDAR file validation utility for the PAD PROFILE pipeline.

Extends the base LAS/LAZ validation with a check that the four extra dimensions
required for the PAD PROFILE processing (h_abg, X_sensor, Y_sensor, Z_sensor)
are present in the file. Only the file header is read so that large tiles are
never loaded into memory during validation.
"""
import logging
import os

import laspy

logger = logging.getLogger(__name__)

REQUIRED_EXTRA_DIMS = {"h_abg", "X_sensor", "Y_sensor", "Z_sensor"}


def check_lidar_file(input_file: str) -> None:
    """Validate a LiDAR file (.las or .laz) for the PAD PROFILE pipeline.

    In addition to the base checks (path, extension, existence), this function
    verifies that the file contains the four extra dimensions needed by the PAD
    PROFILE processing (h_abg, X_sensor, Y_sensor, Z_sensor).

    Only the file header is read (via laspy) so that large tiles are never
    loaded into memory.

    Args:
        input_file: Path to .las or .laz file.

    Raises:
        ValueError: If the input path is not a non-empty string, if the file
            extension is not `.las` or `.laz`, or if one or more required extra
            dimensions (h_abg, X_sensor, Y_sensor, Z_sensor) are missing.
        FileNotFoundError: If the input file does not exist at the given path.
    """
    if not isinstance(input_file, str) or not input_file.strip():
        raise ValueError("Path must be a non-empty string")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")

    ext = os.path.splitext(input_file)[1].lower()
    if ext not in (".las", ".laz"):
        raise ValueError(f"Unsupported extension: {ext}")

    # Read only the header — no point data is loaded
    with laspy.open(input_file) as reader:
        present_extra_dims = set(reader.header.point_format.extra_dimension_names)

    missing = REQUIRED_EXTRA_DIMS - present_extra_dims
    if missing:
        raise ValueError(
            f"Missing required extra dimensions in {input_file}: {sorted(missing)}. "
            f"Expected: {sorted(REQUIRED_EXTRA_DIMS)}."
        )

    logger.info("Valid LiDAR header for PAD PROFIL: %s", input_file)
