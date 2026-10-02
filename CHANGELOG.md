# v0.2.0
- Get a buffer around the input pointcloud in order to compensate the difference between las tiling grid and the output raster grid 
- Finish PAD profil computation
- Export result PAD profile in 9 rasters: 
        - Plant Area Density by stratum of the low-strata bands. (0.5m strata)
        - Plant Area Density by stratum of the main profile, cover-corrected when possible (1m strata)
        - Point count per class
        - Cumulative entering-ray count for each stratum of the main profile.
        - Vegetation/ground hit count for each stratum of the main profile.
        - cos theta
        - pl factor
        - Canopy cover fraction by stratum (2m strata)
        - Points acquisition dates (min, max, maj)

# v0.1.1
- Preprocessing: Add "check_las" decorator to return an error when the output las cannot be read
- [Work in progress] Add PAD profile computation:
    - Add check las on preprocessing
    - Add function "compute cos theta"
    - Add function "build vertical strata"
    - Add function "compute Ni and N"
    - Add function "calculate the fractions of incoming rays intercepted for each "NRD" stratum"
    - Add function "compute Gap Fraction"
    - Add function "calculate PAD profile"


# v0.1.0
- Add function "check lidar data"
- Add function "filter by deviation day"
- Add function "filter by dimension / values"
- Add function "detect and remove outliers"
- Add function "download DTM LIDAR HD from Geoplateforme"
- Add function "normalize height"
- Add function "add trajectory"
- Update version for cicd_deploy.yml

# v0.0.2
- Add folder "configs" in Dockerfile

# v0.0.1
- Initialized GitHub repository.
- Implemented continuous integration pipeline to automatically build Docker image on each version tag.
- Introduced "main_pretreatment" function that validates input LiDAR tiles (validate_lidar_file.py).