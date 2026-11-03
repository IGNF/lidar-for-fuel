# For launching add_version
python -m lidar_for_fuel.main_add_version \
  io.input_dir=./data/pointcloud/ \
  io.input_filename=test_semis_2024_0751_6690_LA93_IGN69.laz  \
  add_version.output_dir=./tmp/ \
  add_version.output_filename=test_semis_2024_0751_6690_LA93_IGN69_add_version.gpkg