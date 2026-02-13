import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from lidar_for_fuel.pretreatment.filter_deviation_day_in_pointcloud import (
    filter_deviation_day,
)

logging.basicConfig(level=logging.INFO)
TMP_PATH = Path("./tmp/test_deviation_day")


def setup_module(module):
    """Setup: clean and create tmp directory."""
    if TMP_PATH.is_dir():
        shutil.rmtree(TMP_PATH)
    os.makedirs(TMP_PATH)


def test_filter_deviation_day():
    """Unit test: 10 points (2 out-of-date), filter and verify."""
    # GPS reference
    ref_dt = datetime(2023, 1, 1, 12, 0, 0)

    # Dates order: indices 0-4:2023-01-15, 5:16, 6:20(OUT), 7:14, 8:10(OUT), 9:16
    dates = [
        datetime(2023, 1, 15, 12, 0),  # 0 mode
        datetime(2023, 1, 15, 12, 0),  # 1
        datetime(2023, 1, 15, 12, 0),  # 2
        datetime(2023, 1, 15, 12, 0),  # 3
        datetime(2023, 1, 15, 12, 0),  # 4
        datetime(2023, 1, 16, 12, 0),  # 5 D+1 ✓
        datetime(2023, 1, 20, 12, 0),  # 6 OUT D+5 ✗
        datetime(2023, 1, 14, 12, 0),  # 7 D-1 ✓
        datetime(2023, 1, 10, 12, 0),  # 8 OUT D-5 ✗
        datetime(2023, 1, 16, 12, 0),  # 9 D+1 ✓
    ]

    gps_times = np.array([(d - ref_dt).total_seconds() for d in dates])
    n_points = len(gps_times)

    # Input point cloud (PDAL dimensions, fixed X for reproducible test)
    ins = {
        "X": np.arange(n_points) * 100.0,
        "Y": np.arange(n_points) * 100.0,
        "Z": np.arange(n_points) * 10.0,
        "gps_time": gps_times,
        "Intensity": np.arange(n_points, dtype=np.uint16),  # fixed, no random
        "ReturnNumber": np.ones(n_points, dtype=np.uint8),
    }

    outs = {k: np.zeros_like(v) for k, v in ins.items()}

    pdalargs = json.dumps({"deviation_day": 2, "gpstime_ref": "2023-01-01 12:00:00"})

    print("=== TEST filter_deviation_day ===")
    print(f"Initial points: {n_points}")
    unique_dates = np.unique([(ref_dt + timedelta(seconds=float(t))).date() for t in ins["gps_time"]])
    print("Unique dates:", unique_dates)
    print("Out-of-date indices: 6(2023-01-20), 8(2023-01-10)")

    # Run filter
    success = filter_deviation_day(ins, outs, pdalargs)

    print(f"Result: success={success}")
    print(f"Points kept: {len(outs['X'])} (expected: 8/10)")
    print("X after (kept indices 0-5,7,9):", outs["X"])
    kept_dates = np.unique([(ref_dt + timedelta(seconds=float(t))).date() for t in outs["gps_time"]])
    print("Dates kept (expected 14/15/16):", kept_dates)

    # Correct assertions (indices 6,8 removed → X: 0,100,200,300,400,500,700,900)
    assert success is True, "Function must return True"
    assert len(outs["X"]) == 8, f"Error: {len(outs['X'])} points instead of 8"
    expected_x = np.array([0, 100, 200, 300, 400, 500, 700, 900])
    assert np.allclose(outs["X"], expected_x), f"Expected {expected_x}, got {outs['X']}"
    print("TEST FULLY PASSED!")
