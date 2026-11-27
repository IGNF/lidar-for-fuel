# not yet released
- Add fuel metrics computation (step 3):
    - Add function "read FMA" as a single value or from an FMA map; a value or a map value must be in [0.05, 0.5] kg/m²
    - Add function "read PAD heights" from the PAD band names
    - Add function "prepare PAD profile" (invalid values set to 0, optional 3-strata smoothing)
    - Add function "cap PAD": the PAD bands read from pad_profile_1m are brought back into [0, pad_max] (values above pad_max and +inf set to pad_max, negative values set to 0); R does not cap
    - Add function "compute CBD" (PAD x FMA, capped at bd_max)
    - Add function "select default ladder threshold" (threshold closest to the median)
    - Add function "compute canopy metrics" (CTH, CDH, CBH, CBH2, CBHlow, CanBD*, CBDcan, CBD_cbh, CBD_cbh2, CBD_cbhlow)
    - Add function "compute ladder metrics" (LadderBot, LadderTop, LadderLoad, SFLeq0, SFLeq_intot, ProfType, CFL, FSG for each ladder threshold)
    - Add function "fuel metrics core": fuel loads (CLFL, FL_k1, FL_k2), default ladder metrics (LFL, CFL, FSG, ProfType) and all metrics with the band names and order; a pixel with no data (PAD all NaN, or no FMA) gets NaN on all metrics, instead of the zero loads and ProfType 1 of R.
    - Add function "export fuel metrics": reads a pad_profile_1m raster and the FMA (value or map), caps the PAD on read, computes the fuel metrics pixel by pixel and writes {stem}_fuel_metrics.tif on the same grid (one float32 band per metric); logs per tile the no-data pixels, the partly NaN profiles set to 0 and the PAD values capped; a FMA error names the PAD raster and the FMA parameter

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