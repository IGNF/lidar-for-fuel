FROM mambaorg/micromamba:latest 

COPY environment.yml /environment.yml

# Using USER root seems to fix permission issues when building mamba environment with pip packages
USER root
RUN micromamba env create -n lidarforfuel -f /environment.yml && \
    micromamba clean --all --yes

ENV PATH=$PATH:/opt/conda/envs/lidarforfuel/bin/
ENV PROJ_LIB=/opt/conda/envs/lidarforfuel/share/proj/

WORKDIR /lidarforfuel
RUN mkdir tmp
COPY lidar_for_fuel lidar_for_fuel
COPY test test
COPY data data

