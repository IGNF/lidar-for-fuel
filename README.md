# LidarForFuel industrialiazation

This repo contains the code [lidarForFuel](https://github.com/oliviermartin7/LidarForFuel) written with R and developped by Olivier Martin : that code be re-written to accommodate specified changes to industralization.


## Content

lidarforfuel aims to compute fuel metrics from airborne LiDAR data and map them at a large scale. Currently, two R functions have been developed: 1) fPCpretreatment: pretreatment of a point cloud and 2) fCBDprofile_fuelmetrics: computing fuel metrics. These functions can be used either at the plot scale for specific analyses on small areas or at a large scale using a catalog of LiDAR tiles from the lidR package.

![Illustration summarising the global approach!](img/readme_1_general.png)

It is important to note that the function fCBDprofile_fuelmetrics for computing fuel metrics/profile needs as entry a pretreated point cloud obtained with the fPCpretreatment.



# Installation / Usage

This library can be used in different ways:
* directly from sources: `make install` creates a mamba environment with the required dependencies
* installed with `pip` from pypi: ` pip install lidarforfuel`
* used in a docker container: see documentation [Dockerfile](Dockerfile)

## Project tree


* `.github/`: folder containing issue templates and GitHub Actions;
* `.vscode/`: folder containing a VS Code configuration for the project;
* `doc/`: folder containing documentation .md files (e.g., install.md);
* `img/`: folder containing images;
* `tests/`: scripts and instructions for running tests;
* `README.md`: this file

## Contacts 


|Nom|Prénom|mail|fonction|
|---|---|---|---|
|   |   |   |   |
|   |   |   |   |
|   |   |   |   |
