# For launching fuel_metrics (step 3), on the PAD rasters written by step 2 (example_launch_pad_profile.sh)
# Every *_pad_profile_1m.tif of io.input_dir is processed; add io.input_filename=<name>_pad_profile_1m.tif
# to process a single file. The parameters are in the "fuel_metrics" section of configs/config.yaml.
python -m lidar_for_fuel.main_fuel_metrics \
  io.input_dir=./tmp/ \
  io.output_dir=./tmp/fuel_metrics/
