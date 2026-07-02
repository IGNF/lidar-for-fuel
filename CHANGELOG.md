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
- ADD function "add trajectory"
- Update version for cicc_deploy.yml

# v0.0.2
- Add folder "configs" in Dockerfile

# v0.0.1
- Initialized GitHub repository. 
- Implemented continuous integration pipeline to automatically build Docker image on each version tag.  
- Introduced "main_pretreatment" function that validates input LiDAR tiles (validate_lidar_file.py).