import os
import shutil
from pathlib import Path
import laspy
import numpy as np


from lidarforfuel.fPCpretreatment.pointcloud.convert_lidar_gpstime_to_date import convert_gpstime_to_time 

TMP_PATH = Path("./tmp/check_lidar")
SAMPLE_LAS = "./data/pointcloud/test_data_0000_0000_LA93_IGN69.laz" 
TMP_PATH = Path("./tmp/convert_gpstime")
SAMPLE_LAS = "./data/pointcloud/test_data_0000_0000_LA93_IGN69.laz"  

def setup_module(module):
    """Clean and recreate tmp directory before tests."""
    if TMP_PATH.is_dir():
        shutil.rmtree(TMP_PATH)
    os.makedirs(TMP_PATH)

def test_convert_gpstime_to_time_return_format_okay():
    """Test function returns valid numpy array of datetimes."""
    # Charge ton sample LAS
    las = laspy.read(SAMPLE_LAS)
    
    result = convert_gpstime_to_time(las, "2020-01-01 00:00:00")
    
    assert isinstance(result, np.ndarray) is True
    assert result.dtype == np.dtype('O')  # object array pour datetimes
    assert len(result) == len(las.points)

