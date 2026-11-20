# not yet released
- [Work in progress] Add fuel metrics computation (port of R "ffuelmetrics2"):
    - Add function "read PAD heights" from the PAD band names
    - Add function "prepare PAD profile" (invalid values set to 0, optional 3-strata smoothing)
    - Add function "compute CBD" (PAD x FMA, capped at bd_max)
    - Add function "read FMA" as a single value or from an FMA map
    - Add function "select default ladder threshold" (threshold closest to the median)
    - Add function "compute canopy metrics" (CTH, CDH, CBH, CBH2, CBHlow, CanBD*, CBDcan)
    - Add function "compute ladder metrics" (LadderBot, LadderTop, LadderLoad, SFLeq0, SFLeq_intot, ProfType, CFL, FSG for each ladder threshold)
    - Add function "fuel metrics core": fuel loads (CLFL, FL_k1, FL_k2), default ladder metrics (LFL, CFL, FSG, ProfType) and all metrics with the R band names and order
    - Assumed divergence from R (decision D2 of docs/epic-3-fuel-metrics.md): a pixel with no data (PAD all NaN) gets NaN on all the fuel metrics, instead of the metrics of a null profile (zero loads, CBH2 = canopy height threshold, ProfType 1) given by R
- Config: align the "fuel_metrics" section with the code parameter names

# v0.3.0
- Fix outliers : Classify statistical outliers using height above ground without removing points, and assign in classe "18"
- Fix for downloading DTM : The output grid is aligned on the native DTM grid and extends half a pixel beyond the tile bounds
- use ign-pdal-tools v1.18.1 to get logs when checking output las
- Add metadata by LIDAR tile : metadata and version for LidarForFuel

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