# COMMONS


def give_name_resolution_raster(size):
    """
    Give a resolution from raster

    Args:
        size (int): raster cell size

    Return:
        _size(str): resolution from raster for output's name
    """
    size_cm = size * 100
    if int(size) == float(size):
        _size = f"_{int(size)}M"
    elif int(size_cm) == float(size_cm):
        _size = f"_{int(size_cm)}CM"
    else:
        raise ValueError(
            f"Cell size is subcentimetric ({size}m) i.e raster resolution is "
            + "too high : output name not implemented for this case"
        )

    return _size
