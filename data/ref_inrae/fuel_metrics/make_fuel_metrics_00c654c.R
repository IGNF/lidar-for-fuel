# Régénère la référence R de l'étape 3 (métriques de combustible) avec lidarforfuel Dev 00c654c
# (1.0.0.9014, 23/11/2026) : `ffuelmetrics2Rast` transmet `CBHThresholdRatio` et `smooth_pad`,
# et calcule `CBD_cbh`, `CBD_cbh2` et `CBD_cbhlow`.
#
# Entrée : le PAD R de la dalle de référence (bandes PAD_1_*) et les paramètres `ffuelmetrics2`
# de la config INRAE. Sortie : 512500_6854997_fuel_metrics_00c654c.tif, à côté de ce script.
# L'ancienne référence (512500_6854997_fuel_metrics.tif, R 8fcccd2, ratio 1/3 effectif,
# CBD_cbh* à NaN) est conservée pour l'historique.
#
# Utilisation, avec 00c654c installé dans une bibliothèque séparée :
#   git -C <dépôt LidarForFuel> archive 00c654c | tar -x -C <src>
#   R CMD INSTALL --library=<lib> <src>
#   R_LIBS=<lib> Rscript data/ref_inrae/fuel_metrics/make_fuel_metrics_00c654c.R

# Le paquet installé depuis `git archive` ne garde pas le commit : la version 1.0.0.9014 est le
# seul contrôle possible. Elle correspond à 00c654c au 23/11/2026 ; un commit Dev ultérieur qui
# ne change pas la version passerait ce contrôle : installer exactement 00c654c.
stopifnot(as.character(packageVersion("lidarforfuel")) == "1.0.0.9014")

script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
if (length(script) != 1) stop("Lancer ce script avec Rscript (il se repère par --file=).")
here <- dirname(normalizePath(script))
ref_dir <- dirname(here)

cfg <- yaml::read_yaml(file.path(ref_dir, "2026-09-28_fuel_metrics_zones-pilotes.yml"))$ffuelmetrics2

pad <- terra::rast(file.path(ref_dir, "pad_profile", "512500_6854997_pad_1m.tif"))
heights <- lidarforfuel::parse_pad_heights(names(pad))
pad <- pad[[heights$idx]]

metrics <- lidarforfuel::ffuelmetrics2Rast(
  PADval = pad,
  FMAcan = cfg$FMAcan,
  FMAshrub = cfg$FMAshrub,
  smooth_pad = cfg$smooth_pad,
  zval = heights$z_bottom,
  dz = cfg$dz,
  LadderFuelBDthresholds = unlist(cfg$LadderFuelBDthresholds),
  CanopyHeightThreshold = cfg$CanopyHeightThreshold,
  CanopyBDTreshFrac = cfg$CanopyBDTreshFrac,
  CanopyTopBDThresholds = unlist(cfg$CanopyTopBDThresholds),
  CBHThresholdRatio = cfg$CBHThresholdRatio,
  bdmax = cfg$bdmax,
  M = cfg$M
)
stopifnot(terra::nlyr(metrics) == 45)

terra::writeRaster(
  metrics,
  file.path(here, "512500_6854997_fuel_metrics_00c654c.tif"),
  datatype = "FLT4S",
  gdal = "COMPRESS=DEFLATE",
  overwrite = TRUE
)
