# v0.2.0
- Get a buffer around the input pointcloud in order to compensate the difference between las tiling grid and the output raster grid 
- Export result PAD profile in 9 rasters: 
        - PAD_{dz_low}_{min_layer}: Plant Area Density for that stratum of the
                low-strata band. One key per stratum.
        - PAD_{dz}_{min_layer}: Plant Area Density for that stratum of the main
                profile, cover-corrected when possible. One key per stratum.
        - Class_{code}: point count for each tracked LAS classification code,
                after the temporal filter, before the vegetation/ground subsetting.
        - Total: point count of any classification, after the temporal filter.
        - N_{dz}_{min_layer}, Ni_{dz}_{min_layer}: cumulative entering-ray count
                and vegetation/ground hit count for that stratum of the main profile.
        - Cover_h_pad: canopy cover fraction above `height_cover`, or `NaN` if `use_cover=False`.
        - Cover_2: canopy cover fraction above 2m.
        - Cover_4: canopy cover fraction above 4m.
        - Cover_6: canopy cover fraction above 6m.
        - cos_theta: scan angle factor (1.0 if `scanning_angle=False`).
        - pl_factor: correction factor for beam path length, `1 / cos_theta`.
        - Date_maj: Unix time (seconds) of the modal acquisition day for the
                points in the pixel/plot -- the center of the ±deviation_days
                temporal window.
        - Date_min, Date_max: Unix time (seconds) of the lower/upper bound of
                that ±deviation_days temporal window (`Date_maj` -/+ `deviation_days`).

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