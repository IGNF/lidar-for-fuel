---
title: 'Épique 3 — Métriques de combustible : portage de ffuelmetrics2 (R) dans lidar-for-fuel'
type: 'epic'
created: '2026-10-22'
status: 'draft'
updated: '2026-11-13'
author: 'Mary (analyste BMAD) pour Malvina Dupays'
sources:
  - 'LidarForFuel (R), branche Dev, commit 8fcccd2 (lidarforfuel 1.0.0.9013, 12/11/2026), qui intègre la PR #48 mergée (261b91e). Pour le calcul, R/ffuelmetrics2.R y est identique à 7c7d35e + PR #48 (1e2cc00), la version analysée auparavant — R/ffuelmetrics2.R, tests/testthat/test-ffuelmetrics2.R, Code_examples/Processing_Pipeline_example_V1.0.R'
  - 'Sortie de référence R : zone_pilote_test/HE_IGN_Galaxy/R/10_10_2026/{pad_1m.tif, fuel_metrics.tif, 20261011T022901_config.yaml} (lidarforfuel 1.0.0.9012)'
  - 'lidar-for-fuel (Python), branche dev 9a71d7f (v0.3.0) — lidar_for_fuel/pad_profile/*, configs/config.yaml'
  - 'Prototype de parité : proto-ffuelmetrics2-parite-2026-10-22.py (dossier de recherche local, hors dépôt)'
  - 'Brief étape 3 (04/11/2026) : _bmad-output/planning-artifacts/briefs/brief-lidar-for-fuel-2026-11-04/{brief.md, addendum.md, note-inrae-reunion-mi-novembre.md}'
  - 'Sprint Change Proposal du 05/11/2026 : _bmad-output/planning-artifacts/sprint-change-proposal-2026-11-05.md'
---

# Épique 3 — Métriques de combustible (portage de `ffuelmetrics2`)

## Description

**Objectif.** Ajouter à lidar-for-fuel la troisième et dernière étape de la chaîne lidarforfuel, **« ffuelmetrics2Rast »**. Elle transforme le raster de profils PAD à 1 m (sortie de l'étape 2) en un raster de **métriques de combustible** par pixel de 10 m : charges, hauteurs de base et de sommet du houppier, combustible en échelle, type de profil. Il y a 45 métriques avec la configuration par défaut.

<img width="1883" height="958" alt="Image" src="https://github.com/user-attachments/assets/b6623f19-d434-4d95-839f-e354b4908c1f" />

**Pour qui.** L'IGN produit ces métriques sur toute la France (LiDAR HD). Les chercheurs de l'INRAE les transforment en **cartes de combustible et de risque d'incendie**, que les SDIS et l'ONF utilisent. L'étape 3 fournit donc la matière première de ces cartes, pas la carte finale.

**Contexte.** Il s'agit du portage en Python du code R `ffuelmetrics2` (LidarForFuel, INRAE). Les étapes 1 (prétraitement) et 2 (profils PAD) sont déjà portées et validées par l'INRAE. Sans l'étape 3, la chaîne nationale produit des profils PAD, mais aucune des métriques que l'INRAE exploite.

| Étape | R (lidarforfuel) | Python (lidar-for-fuel) | État |
|---|---|---|---|
| 1. Prétraitement | `fPCpretreatment` | `main_preprocessing.py` | fait |
| 2. Profils PAD par pixel | `pad_metrics` + `lidR::pixel_metrics` | `main_pad_profile.py` → `{stem}_pad_profile_1m.tif` (+ 8 autres rasters) | fait |
| **3. Métriques de combustible** | `ffuelmetrics2` / `ffuelmetrics2Rast` | **à créer : `main_fuel_metrics.py`** | **cette épique** |

L'étape 3 est un **calcul pur, pixel par pixel**. Elle ne lit aucun nuage de points, seulement le profil PAD vertical de chaque pixel. Elle est donc indépendante des écarts de PAD entre IGN et INRAE à l'étape 2 : on peut la valider seule, directement contre la sortie R.

**Référence R.** La branche Dev de LidarForFuel au commit `8fcccd2` (lidarforfuel 1.0.0.9013, 12/11/2026), qui intègre la PR #48 (décision D7).

**Calendrier.**
- **mi-novembre 2026** : réunion avec l'INRAE. On y présente les décisions D2 à D5 et une tolérance chiffrée par métrique (story 3.8) ;
- **mi-décembre 2026** : code de l'étape 3 prêt ;
- **janvier 2027** : lancement de la production nationale, après validation par l'INRAE.

Les décisions encore ouvertes (D2 à D5 : pixels sans donnée, bandes `CBD_cbh*`, `cbh_threshold_ratio`, seuil de 1 m codé en dur) sont en annexe, au §6. Elles sont reprises dans la note pour l'INRAE.

### Grille de sortie

L'étape 3 **ne choisit pas sa grille** : elle écrit sur la grille exacte du raster PAD d'entrée (même transform, même CRS, même taille). C'est l'étape 2 qui cale les rasters PAD sur la grille de la FMA CosiaFrance, `CosiaFrance_V16.tif`.

**Propriétés de la grille de référence**, lues dans `CosiaFrance_V16.tif` le 13/11/2026 :

| Propriété | Valeur |
|---|---|
| CRS | EPSG:2154 (RGF93 / Lambert-93) |
| Résolution | 10,0 m × 10,0 m |
| Dimensions | 106 595 lignes × 114 541 colonnes |
| Origine (coin haut-gauche) | X = 98 039,69 m, Y = 7 111 486,70 m |
| Emprise | left = 98 039,69 ; bottom = 6 045 536,70 ; right = 1 243 449,69 ; top = 7 111 486,70 |
| NoData | −128 (int8). C'est un raster de classes FMA, que l'on ne réutilise pas en sortie. |

> Les données LiDAR prétraitées sont déjà en EPSG:2154 : **aucune reprojection n'est nécessaire**.
>
> **Point d'attention : décalage de 6 cm.** La config Python cale les rasters PAD sur l'origine **arrondie** (98 039,75 ; 7 111 486,75), et la sortie R tombe sur cette même grille. L'origine réelle du fichier CosiaFrance est (98 039,6885 ; 7 111 486,6972) : il y a donc un décalage de 6,1 cm en X et de 5,3 cm en Y. Ce décalage est sans effet sur cette épique, car la FMA y est un scalaire de configuration (D6). Il faudra le régler avant d'utiliser la FMA CosiaFrance pixel par pixel.

### Données en entrée

#### 1. Profil PAD (Plant Area Density) — seule entrée de calcul

**Type de donnée :** raster GeoTIFF, un fichier par dalle
**Nom du raster :** `{stem}_pad_profile_1m.tif`
**Nombre de canaux :** 60
**Pas du raster :** 10 m
**Résolution verticale :** 1 m
**Volume de la donnée estimé :** environ 2,4 Mo par dalle de 1 km² non compressée (60 canaux × 100 × 100 pixels × 4 octets, float32)

| Nom canaux | Format | Valeurs limites à contrôler | Définition |
| --- | --- | --- | --- |
| PAD_1_0 | Décimal (float32, nodata NaN) | 0–5 (plafonné à 5) | PAD entre 0 et 1 m (m²/m³). 1er chiffre = résolution verticale ; 2e chiffre = hauteur du bas de la strate. |
| PAD_1_1 | Décimal | 0–5 (plafonné à 5) | PAD entre 1 et 2 m (m²/m³) |
| … | Décimal | 0–5 (plafonné à 5) | … |
| PAD_1_i | Décimal | 0–5 (plafonné à 5) | PAD entre i et i+1 m (m²/m³) |
| … | Décimal | 0–5 (plafonné à 5) | … |
| PAD_1_59 | Décimal | 0–5 (plafonné à 5) | PAD entre 59 et 60 m (m²/m³) |

Le nombre de canaux suit la config de l'étape 2 (`nlayers: 60`). L'étape 3 lit les hauteurs dans les noms de canaux : elle ne suppose pas qu'il y en a 60.

#### 2. Paramètres de configuration

Une nouvelle section `fuel_metrics` de la config contient les FMA de canopée et d'arbuste, les seuils d'échelle et de sommet de canopée, `canopy_height_threshold`, `cbh_threshold_ratio`, `bd_max` et `m_perrakis`. Le détail des valeurs est en annexe, au §3.1.

#### Donnée disponible mais non utilisée dans cette épique : PAD de la strate basse

`{stem}_pad_sb_0.5m.tif` (4 canaux `PAD_0.5_0` à `PAD_0.5_1.5`, pas de 10 m, résolution verticale 0,5 m, plafonné à 5) est produit par l'étape 2. R ne l'utilise pas dans `ffuelmetrics2`. Il ne servira qu'à la future charge de surface SFL corrigée par quantile mapping, **hors périmètre** (§10).

### Données en sortie

**Type de donnée :** raster GeoTIFF, un fichier par dalle
**Nom du raster :** `{stem}_fuel_metrics.tif`
**Nombre de canaux :** 19 + 8 × (nombre de seuils d'échelle) + (nombre de seuils de sommet de canopée), soit **45** avec la configuration par défaut
**Pas du raster :** 10 m, sur la grille du raster PAD d'entrée
**Format :** float32, nodata = NaN, canaux nommés et ordonnés comme en R
**Volume de la donnée estimé :** environ 1,8 Mo par dalle de 1 km² non compressée (45 × 100 × 100 × 4 octets). Avec environ 550 000 dalles, cela fait de l'ordre de 1 To à l'échelle nationale, avant compression.

| Nom canaux | Format | Valeurs limites à contrôler | Définition |
| --- | --- | --- | --- |
| CLFL | Décimal | ≥ 0 | Charge de canopée et de combustible en échelle au-dessus de `canopy_height_threshold` (kg/m²) |
| FL_k1, FL_k2 | Décimal | ≥ 0 | Charge de la 1re et de la 2e strate, non corrigée (kg/m²) |
| LadderBot_s, LadderTop_s | Décimal | 0–60 m | Bas et haut du combustible en échelle, pour chaque seuil d'échelle `s` (0p010, 0p020, 0p050) (m) |
| LadderLoad_s | Décimal | ≥ 0 | Charge de combustible en échelle pour le seuil `s` (kg/m²) |
| SFLeq0_s, SFLeq_intot_s | Décimal | ≥ 0 | Charge de surface équivalente selon Perrakis (2025), formule simple et formule intégrée (kg/m²) |
| ProfType_s | Entier stocké en float32 | {1, 2, 3, 4} | Type de profil pour le seuil `s` : 1 pas de canopée, 2 canopée sans échelle, 3 échelle continue depuis la surface, 4 échelle suspendue |
| CFL_s | Décimal | ≥ 0 | Charge de canopée au-dessus de LadderTop pour le seuil `s` (kg/m²) |
| FSG_s | Décimal | ≥ 0 | Écart entre strates (Fuel Strata Gap) pour le seuil `s` (m) |
| CTH_0p0012, CTH_0p0100 | Décimal | 0–60 m | Hauteur du sommet de canopée, pour chaque seuil de densité (m) |
| CDH | Décimal | 0–60 m | Hauteur dominante, plafonnée par `CTH_0p0100` (m) |
| CBH | Décimal | 0–60 m | Base du houppier (m) |
| CBH2 | Décimal | ≥ `canopy_height_threshold` (1 m), ≤ 60 m | Base du houppier « continue », métrique principale (m) |
| FSG | Décimal | ≥ 0 | Copie de `FSG_0p020` (seuil par défaut) (m) |
| CBHlow | Décimal | 0–60 m | Base du houppier recherchée dans la partie basse, entre 1 m et CTH/3 (m) |
| CanBDmax, CanBDmin, CanBDlowmax, CanBDlowmin | Décimal | 0 – `bd_max` (1) | Extrêmes de densité CBD qui servent à trouver CBH et CBHlow (kg/m³) |
| CBDcan | Décimal | 0 – `bd_max` (1) | Densité CBD moyenne entre CBH2 et min(CDH, CTH) (kg/m³) |
| CBD_cbh, CBD_cbh2, CBD_cbhlow | Décimal | 0 – `bd_max` (1) | Densité CBD à la hauteur CBH, CBH2 ou CBHlow. **Toujours NaN en R** (constat A1, décision D3) (kg/m³) |
| LFL, CFL, ProfType | Décimal, décimal, entier | ≥ 0, ≥ 0, {1, 2, 3, 4} | Copies des valeurs du seuil par défaut (`LadderLoad_0p020`, `CFL_0p020`, `ProfType_0p020`) |

Avec la configuration par défaut, les 3 seuils d'échelle × 8 métriques donnent les canaux 4 à 27. Les noms suivent le format R : `0.01` → `0p010` pour les seuils d'échelle, `0.01` → `0p0100` pour les seuils CTH. Le seuil par défaut est celui de la liste le plus proche de la médiane (PR #48). L'ordre exact et les conventions de hauteur (bas, centre ou haut de strate) sont en annexe, au §3.2.

### Pipeline général de l'étape 3

Le schéma suit l'ordre des stories. À l'exécution, c'est le point d'entrée (3.7) qui appelle les autres étapes. La story 3.1 ne calcule rien : elle fournit le jeu de référence R qui sert aux tests de toutes les stories suivantes.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ENTRÉES                                                                 │
│  • {stem}_pad_profile_1m.tif (sortie de l'étape 2) : bandes PAD_1_0 …    │
│    PAD_1_59, float32, nodata NaN, PAD plafonné à [0, 5], grille 10 m     │
│  • Config, section fuel_metrics : fma_can, fma_shrub, smooth_pad,        │
│    seuils d'échelle et de CTH, canopy_height_threshold,                  │
│    cbh_threshold_ratio, bd_max, m_perrakis                               │
│  • Sortie de référence R (pad_1m.tif + fuel_metrics.tif, zone pilote)    │
└──────────────────────────┬───────────────────────────────────────────────┘
                           │
                    ╔══════▼══════╗
                    ║   US-3.1    ║  Préparation : jeu de référence R
                    ╚══════╤══════╝  (fenêtre PAD + 45 métriques attendues)
                           │  Jeu de test pour le TDD des stories suivantes
                    ╔══════▼══════╗
                    ║   US-3.2    ║  Lecture des strates (dz, h_bottom),
                    ║             ║  nettoyage, lissage, profil CBD   (étape 1)
                    ╚══════╤══════╝
                           │  Profil CBD + h_bottom, dz
                    ╔══════▼══════╗
                    ║   US-3.3    ║  Métriques de canopée : CTH, CBH, CBH2,
                    ║             ║  CBHlow, CDH, CBDcan         (étapes 2 à 6)
                    ╚══════╤══════╝
                           │  Hauteurs de canopée + CBD recalculée selon CBH2
                    ╔══════▼══════╗
                    ║   US-3.4    ║  Charges, combustible en échelle,
                    ║             ║  assemblage : fuel_metrics_core (étapes 7 à 9)
                    ╚══════╤══════╝  → parité stricte avec R
                           │  Dict de métriques par pixel (45 avec la config par défaut)
                    ╔══════▼══════╗
                    ║   US-3.5    ║  Corrections assumées D2, D3, D5
                    ║             ║  (après les réponses de l'INRAE)
                    ╚══════╤══════╝
                           │  Dict de métriques corrigé
                    ╔══════▼══════╗
                    ║   US-3.6    ║  Application au raster PAD (boucle pixels)
                    ║             ║  & export GeoTIFF via _write_geotiff
                    ╚══════╤══════╝
                           │  {stem}_fuel_metrics.tif pour une dalle
                    ╔══════▼══════╗
                    ║   US-3.7    ║  Point d'entrée main_fuel_metrics (Hydra) :
                    ║             ║  fichier ou dossier, config, contrôle
                    ╚══════╤══════╝  d'équivalence R, doc
                           │
┌──────────────────────────▼───────────────────────────────────────────────┐
│  SORTIES                                                                 │
│  • {stem}_fuel_metrics.tif : 19 + 8 × n_échelle + n_CTH bandes (45 par   │
│    défaut), float32, nodata NaN, noms et ordre de R, même grille que     │
│    le raster PAD d'entrée                                                │
│  • Logs par dalle : pixels entièrement NaN, profils corrigés à 0,        │
│    pixels traités                                                        │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  VALIDATION (hors chaîne de calcul)                                      │
│  • US-3.8 : mesure rapide chaîne complète Python vs R (tolérances Q6),   │
│    passe 1 avec le prototype, passe 2 avec main_fuel_metrics             │
│  • US-3.9 (reportable) : étape 3 dans lff-compare                        │
└──────────────────────────────────────────────────────────────────────────┘
```

### Stories

| Story | Titre | Effort | Remarque |
|---|---|---|---|
| 3.1 | Jeu de données de référence R | Facile | Préalable au TDD |
| 3.2 | Lecture des strates et préparation du profil CBD | Facile | |
| 3.3 | Métriques de canopée (CTH, CBH, CBH2, CBHlow, CDH, CBDcan) | Normal | |
| 3.4 | Charges, combustible en échelle et assemblage (parité stricte avec R) | Normal | |
| 3.5 | Corrections assumées par rapport à R | Facile | Attend les réponses de l'INRAE (D2, D3, D5) |
| 3.6 | Application au raster PAD et export GeoTIFF | Normal | |
| 3.7 | Point d'entrée `main_fuel_metrics`, configuration et documentation | Facile | |
| 3.8 | Mesure rapide des écarts sur la chaîne complète | Facile | **Obligatoire** : passe 1 avant mi-novembre, passe 2 mi-décembre |
| 3.9 | Étape 3 dans l'outil de comparaison `lff-compare` | Normal | **Reportable** après janvier 2027 |

Le détail de chaque story (description, effort, critères d'acceptation) est en annexe, au §9. Le périmètre reste ouvert : des stories pourront être ajoutées ou retirées selon les réponses de l'INRAE et les retours de validation.

**Priorités en cas de retard.** Le développement est assuré par une seule personne ; les PR sont relues par d'autres. Si le calendrier glisse, on reporte d'abord la story 3.9. Les stories 3.1 à 3.8 ne sont pas négociables.

## Estimation de l'Effort

**Normal.**

L'algorithme est maîtrisé : un prototype Python reproduit déjà la sortie R avec 0 écart sur 40 401 pixels × 45 métriques de la zone pilote. Le calcul est léger (environ 3 s par dalle) et ne demande aucune optimisation. L'incertitude porte sur les arbitrages de l'INRAE (D2 à D5, tolérances Q6) et sur le calendrier de validation, qui tombe pendant les fêtes.

## Critères d'acceptation

- **AC-E.1 — Parité de l'étape 3 seule.** **Étant donné** le PAD produit par R sur la zone pilote HE_IGN_Galaxy (`pad_1m.tif` du 10/10/2026), **quand** on lui applique l'étape 3 Python, **alors** chacun des 45 canaux est identique à `fuel_metrics.tif` de R, à l'arrondi float32 près (tolérance relative 10⁻⁴), sauf les divergences assumées D2, D3 et D5, qui sont listées dans le `CHANGELOG.md`.
- **AC-E.2 — Écart maîtrisé sur la chaîne complète.** **Étant donné** le PAD produit par lidar-for-fuel sur la zone pilote, **quand** on compare les métriques de l'étape 3 Python à celles de la chaîne R, **alors** chaque métrique respecte la tolérance convenue avec l'INRAE (Q6, ordre de grandeur 5 %).
- **AC-E.3 — Format de sortie.** **Étant donné** une dalle `{stem}_pad_profile_1m.tif`, **quand** l'étape 3 s'exécute, **alors** elle écrit `{stem}_fuel_metrics.tif` : un GeoTIFF float32, nodata = NaN, sur la même grille que l'entrée (transform, CRS, taille), avec les noms et l'ordre des canaux de R.
- **AC-E.4 — Canaux pilotés par la configuration.** **Étant donné** une configuration avec n seuils d'échelle et m seuils de sommet de canopée, **quand** l'étape 3 s'exécute, **alors** la sortie a 19 + 8n + m canaux, nommés d'après les seuils configurés. Les canaux `LFL`, `CFL`, `FSG` et `ProfType` prennent les valeurs du seuil le plus proche de la médiane.
- **AC-E.5 — Pixels sans donnée.** **Étant donné** un pixel dont tout le profil PAD est NaN, **quand** l'étape 3 s'exécute, **alors** le pixel reçoit la valeur décidée avec l'INRAE (recommandation IGN : NaN sur tous les canaux, D2) et n'apparaît pas comme « sans combustible ». Le nombre de ces pixels est journalisé par dalle.
- **AC-E.6 — Robustesse à l'échelle nationale.** **Étant donné** une dalle atypique (eau, bord de chantier, dalle entièrement vide), **quand** l'étape 3 s'exécute, **alors** elle ne s'interrompt pas, écrit sa sortie et journalise les pixels vides et les profils corrigés.
- **AC-E.7 — Lancement comme les étapes 1 et 2.** **Étant donné** un fichier ou un dossier de rasters PAD et la configuration, **quand** on lance `main_fuel_metrics`, **alors** on obtient une sortie par dalle. Un avertissement est émis si la condition d'équivalence avec R n'est pas remplie : `5 × min(fma_can, fma_shrub) ≥ bd_max` et `smooth_pad = False`.
- **AC-E.8 — Validation par l'INRAE.** **Étant donné** le code final de la mi-décembre, **quand** l'INRAE examine le test de parité (AC-E.1) et la mesure des écarts refaite sur la version finale (story 3.8, passe 2), **alors** elle donne son accord avant le lancement de la production en janvier 2027.

---

# Annexes techniques

Les annexes gardent la numérotation 3 à 11, à laquelle renvoient les stories et la note INRAE.

## 3. Contrat d'entrée et de sortie

### 3.1 Entrée

- `{stem}_pad_profile_1m.tif`, produit par `export_raster` ([lidar_for_fuel/pad_profile/export_raster.py](../lidar_for_fuel/pad_profile/export_raster.py)) :
  - bandes `PAD_1_0` … `PAD_1_59`, au format `PAD_{dz}_{h_bas}`, où `h_bas` est la hauteur du bas de la strate au‑dessus du sol ;
  - float32, nodata = NaN ;
  - PAD **plafonné à [0, 5]** au moment de l'export ;
  - grille CosiaFrance à 10 m, EPSG:2154, calée sur l'origine arrondie de la config (98039.75, 7111486.75), soit à 6 cm de l'origine réelle du fichier CosiaFrance (voir « Grille de sortie »).
- Paramètres : voir le tableau ci‑dessous et la proposition de config en story 3.7.

| Paramètre R | Nom Python proposé | Valeur (config INRAE zones pilotes) | Rôle |
|---|---|---|---|
| `FMAcan` | `fma_can` | 0.2543576 kg/m² | masse surfacique du combustible de canopée |
| `FMAshrub` | `fma_shrub` | 0.2543576 kg/m² | masse surfacique du combustible arbustif |
| `smooth_pad` | `smooth_pad` | `False` | moyenne mobile sur 3 strates |
| `LadderFuelBDthresholds` | `ladder_fuel_bd_thresholds` | [0.01, 0.02, 0.05] kg/m³ | seuils de densité pour le combustible en échelle |
| `CanopyHeightThreshold` | `canopy_height_threshold` | 1 m | limite entre combustible de surface et canopée/échelle |
| `CanopyBDTreshFrac` | `canopy_bd_thresh_frac` | 0.3 | fraction entre CBDmin et CBDmax pour définir le seuil CBH |
| `CanopyTopBDThresholds` | `canopy_top_bd_thresholds` | [0.0012, 0.01] kg/m³ | seuils de densité pour le sommet de canopée |
| `CBHThresholdRatio` | `cbh_threshold_ratio` | **1/3 exact**, et non 0.333 (voir D4) | fraction de CTH au‑dessus de laquelle on cherche la base du houppier dominant |
| `bdmax` | `bd_max` | 1.0 kg/m³ | plafond de la densité CBD |
| `M` | `m_perrakis` | 2.5 | facteur M de Perrakis (2025) |

> **Condition d'équivalence avec R.** Le PAD Python est plafonné à 5, alors que le PAD R ne l'est pas (maximum observé : 7,01). Les résultats restent identiques tant que `5 × min(fma_can, fma_shrub) ≥ bd_max`, car la densité CBD est de toute façon plafonnée à `bd_max`. C'est vrai avec la configuration actuelle : 5 × 0,254 = 1,27 ≥ 1. C'est aussi vrai seulement si `smooth_pad = False`, car le lissage porterait sur des valeurs déjà plafonnées. La story 3.7 doit vérifier cette condition au démarrage et émettre un avertissement sinon.

### 3.2 Sortie

`{stem}_fuel_metrics.tif` : une seule GeoTIFF, **dans l'ordre exact de R**, avec les noms de bandes R, en float32, nodata = NaN. Elle a la même grille (transform, CRS, taille) que le raster PAD d'entrée.

Le nombre de bandes dépend de la configuration : **19 + 8 × n_échelle + n_CTH**, où `n_échelle` est le nombre de seuils `ladder_fuel_bd_thresholds` distincts et `n_CTH` le nombre de seuils `canopy_top_bd_thresholds`. Avec la configuration par défaut (3 seuils d'échelle, 2 seuils CTH), on obtient **45 bandes**. Les numéros du tableau ci‑dessous correspondent à ce cas.

Les noms de bandes sont dérivés des seuils **réellement configurés**, comme le fait la PR #48 dans `ffuelmetrics2Rast` (ligne 362). Avant cette PR, R nommait les bandes à partir de ses seuils par défaut : avec une autre configuration, les noms étaient donc faux (A11).

| # | Bande | Unité | Définition | Référence de hauteur |
|---|---|---|---|---|
| 1 | `CLFL` | kg/m² | charge de canopée + échelle au‑dessus de `canopy_height_threshold` | — |
| 2-3 | `FL_k1`, `FL_k2` | kg/m² | charge de la 1ʳᵉ et de la 2ᵉ strate (non corrigée, sous‑estimée) | — |
| 4-27 | pour chaque seuil `s` ∈ {`0p010`, `0p020`, `0p050`} : `LadderBot_s`, `LadderTop_s`, `LadderLoad_s`, `SFLeq0_s`, `SFLeq_intot_s`, `ProfType_s`, `CFL_s`, `FSG_s` | m, m, kg/m², kg/m², kg/m², classe 1-4, kg/m², m | bas et haut du combustible en échelle ; charge entre 1 m et LadderTop ; surface équivalente selon Perrakis (formule simple et formule intégrée) ; type de profil ; charge de canopée au‑dessus de LadderTop ; écart entre strates (Fuel Strata Gap) | LadderBot = **bas** de strate, LadderTop = **haut** de strate |
| 28-29 | `CTH_0p0012`, `CTH_0p0100` | m | hauteur du sommet de canopée, c'est‑à‑dire la strate la plus haute dont la CBD dépasse le seuil | **centre** de strate |
| 30 | `CDH` | m | hauteur dominante, plafonnée par `CTH_0p0100` (= min(CDH, CTH)) | **centre** de strate |
| 31 | `CBH` | m | base du houppier : première strate au‑dessus du seuil | **bas** de strate |
| 32 | `CBH2` | m | base du houppier « continue », métrique principale | **bas** de strate |
| 33 | `FSG` | m | copie de `FSG_0p020` (seuil médian) | — |
| 34 | `CBHlow` | m | base du houppier recherchée dans la partie basse, entre 1 m et CTH/3 | **bas** de strate |
| 35-38 | `CanBDmax`, `CanBDmin`, `CanBDlowmax`, `CanBDlowmin` | kg/m³ | extrêmes de densité CBD utilisés pour trouver CBH et CBHlow | — |
| 39 | `CBDcan` | kg/m³ | densité CBD **moyenne** entre CBH2 et min(CDH, CTH) | — |
| 40-42 | `CBD_cbh`, `CBD_cbh2`, `CBD_cbhlow` | kg/m³ | densité CBD à la hauteur CBH, CBH2 ou CBHlow. **Toujours NaN en R** (voir A1) | — |
| 43-45 | `LFL`, `CFL`, `ProfType` | kg/m², kg/m², classe | copies de `LadderLoad_0p020`, `CFL_0p020` et `ProfType_0p020` | — |

**Les codes `ProfType`.** `ProfType` décrit la forme du profil vertical d'un pixel, c'est‑à‑dire la façon dont le combustible se répartit entre trois zones :
- **la surface**, de 0 m à `canopy_height_threshold` (CHT, 1 m) ;
- **la canopée dominante**, au‑dessus de `CBH2` ;
- **la zone intermédiaire**, entre CHT et `CBH2`. On y cherche le **combustible en échelle**, qui peut faire monter le feu de la surface jusqu'aux houppiers.

Le code dépend de l'existence de strates assez denses (CBD ≥ seuil `t`) dans la zone intermédiaire, et de la hauteur où commence la première d'entre elles (étape 8 de la section 4). Hauteurs données à titre d'illustration :

```
h (m)  Type 1            Type 2            Type 3            Type 4
       pas de canopée    canopée seule     échelle continue  échelle suspendue
       CBH2 = 1 m        pas d'échelle     LadderBot = 1 m   LadderBot > 1 m

   20                    ████ canopée      ████ canopée      ████ canopée
   15                    ──── CBH2         ──── CBH2         ──── CBH2
   10                                      ▓▓▓▓ échelle      ▓▓▓▓ échelle
    5                    (vide)            ▓▓▓▓              ▓▓▓▓
    2                                      ▓▓▓▓              (vide)
   1   ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
    0  ░░░░ surface      ░░░░ surface      ░░░░ surface      ░░░░ surface

░ surface (0 à 1 m)   ▓ combustible en échelle (CBD ≥ seuil)
█ canopée dominante   ┄ canopy_height_threshold (1 m)
```

- **1, pas de strate dominante.** Aucune base de houppier n'a été trouvée : `CBH2` est retombé sur CHT (étape 3C). La zone intermédiaire « CHT < h ≤ CBH2 » est alors vide, donc il ne peut pas y avoir de combustible en échelle.
- **2, canopée sans échelle.** `CBH2` > CHT, mais aucune strate de la zone intermédiaire n'atteint le seuil. La surface et la canopée sont séparées par un vide.
- **3, échelle continue depuis la surface.** Le premier bloc de combustible en échelle commence à CHT (`LadderBot` ≤ CHT). Il touche directement la strate de surface : c'est le profil le plus favorable à la montée du feu.
- **4, échelle suspendue.** Le combustible en échelle commence au‑dessus de CHT (`LadderBot` > CHT). Un vide le sépare de la surface, ce qui fait trois strates distinctes.

À noter :
- Le combustible en échelle n'est cherché qu'au‑dessus de CHT. « Continu depuis la surface » (type 3) signifie donc « qui commence à CHT », et non « qui descend jusqu'au sol ».
- La strate de surface des types 2 et 4 est une zone de référence. Elle est toujours comptée, même si la strate 0–1 m est vide : sa présence n'est pas testée.
- Il y a un `ProfType` par seuil (`ProfType_0p010`, `ProfType_0p020`, `ProfType_0p050`). Un même pixel peut être de type 3 avec le seuil 0.01 et de type 2 avec le seuil 0.05. La bande `ProfType` est celle du seuil médian (étape 9).
- En R, le bas de la zone de recherche est `h > 1` codé en dur, et non CHT (constat A2). Avec CHT = 1 m, le résultat est le même.

Quelques remarques sur ces sorties :
- Ces conventions de hauteur mélangent des centres et des bords de strate. Ce n'est pas une erreur du portage : c'est le comportement de R, qu'il faut **reproduire** puis documenter dans les docstrings.
- Les noms de suffixe suivent le format `sprintf` de R : `"%0.3f"` pour les seuils d'échelle (0.01 → `0p010`) et `"%0.4f"` pour les seuils CTH (0.01 → `0p0100`).
- Le test R vérifie bien 45 sorties avec les seuils par défaut (`expect_length(metrics, 45)`).

## 4. Algorithme de référence (à porter tel quel)

**Notations.**
- `h_bas[i]` est la hauteur du bas de la strate i, lue dans le nom de bande, et `dz` l'épaisseur des strates.
- `h[i] = h_bas[i] + dz/2` est le centre de la strate. C'est ce que le code R appelle `zval` **après** le décalage de la ligne 106.
- `CHT` = `canopy_height_threshold`.
- On note les indices à partir de 0, comme en Python. R les compte à partir de 1.

> Conformément à la convention du dépôt (mémoire projet *z vs h_abg*), `z` désigne en Python l'altitude brute. Les hauteurs au‑dessus du sol de cette étape se nomment donc `h_*` (`h_bottom`, `h_center`), et non `z*`.

1. **Préparation**
   - PAD non fini, NaN ou négatif → 0.
   - Si `smooth_pad` : moyenne glissante centrée sur 3 strates. Aux bords, la moyenne porte sur les 2 valeurs disponibles (comportement de `slider::slide_dbl(.before=1,.after=1)`, `.complete=FALSE`).
   - `CBD[i] = min(PAD[i] × (fma_can si h[i] > CHT sinon fma_shrub), bd_max)`.
2. **Sommet de canopée.** Pour chaque seuil `t` de `canopy_top_bd_thresholds` : `CTH_t = max(h[i] | CBD[i] ≥ t)`, ou NaN si aucune strate ne dépasse le seuil. On pose ensuite `CTH = CTH_{max(seuils)}`, soit le seuil 0.01.
3. **Base du houppier**
   - **3A, canopée haute** (seulement si CTH n'est pas NaN) :
     - `I = {i | h[i] > max(CHT, CTH × ratio) et h[i] ≤ CTH}`. Si I n'est pas vide :
     - `CanBDmax = max CBD[I]`, et `idmax` = **premier** indice de I qui atteint ce maximum.
     - `I2 = {i | h[i] > CHT et h[i] < h[idmax]}`. Si I2 n'est pas vide : `CanBDmin = min CBD[I2]`.
     - Si en plus `CanBDmin < CanBDmax` :
       - `idmin` = premier indice de I2 qui atteint ce minimum ;
       - seuil `S = CanBDmin × (1 − frac) + frac × CanBDmax` ;
       - `CBH = min(h[i] | CBD[i] ≥ S, h[i] > max(CHT, h[idmin]), h[i] ≤ h[idmax]) − dz/2` ;
       - si CBH existe : on parcourt `k` de `idmax` jusqu'à `idmin+1` en descendant, et au premier `k` tel que `CBD[k−1] < S`, on pose `CBH2 = h[k] − dz/2` et on s'arrête.
     - ⚠️ CanBDmax et CanBDmin sont renseignés en sortie **même si** CBH n'est pas trouvé.
   - **3B, canopée basse** :
     - `I = {i | h[i] > CHT et h[i] ≤ CTH × ratio}` ;
     - même logique qu'en 3A, ce qui donne `CanBDlowmax`, `CanBDlowmin` et `CBHlow = min(...) − dz/2` ;
     - **différence avec 3A** : il n'y a pas de test `min < max`.
   - **3C** : si CBH2 est NaN, `CBH2 ← CBHlow` ; s'il est encore NaN, `CBH2 ← CHT`.
4. **Recalcul de la densité CBD selon CBH2** : `CBD[i] = min(PAD[i] × (fma_can si h[i] > CBH2 sinon fma_shrub), bd_max)`. Ce recalcul se fait à partir du PAD préparé à l'étape 1, éventuellement lissé.
5. **Hauteur dominante CDH**
   - `CDH = max(h[i] | CBD[i] ≥ 0.1 × CanBDmax)`, à condition que CanBDmax ne soit pas NaN ;
   - sinon `CDH = CTH` ;
   - puis `CDH = min(CDH, CTH)`. **NaN se propage** : si CTH est NaN, CDH vaut NaN.
6. **Densité moyenne CBDcan**
   - `haut = min(CDH, CTH)`. Si `haut` est NaN, **CBDcan vaut NaN** : c'est la sémantique `any(NA)` de R. La référence le confirme : `CBDcan` a 39 830 pixels valides, exactement autant que `CTH_0p0100`.
   - Sinon, `CBDcan = moyenne(CBD[i] | CBH2 ≤ h[i] ≤ haut)`, ou 0 si cet ensemble est vide.
7. **Charges**
   - `CLFL = Σ CBD[h > CHT] × dz` ;
   - `FL_k1 = CBD[0] × dz` et `FL_k2 = CBD[1] × dz`. Ces deux valeurs supposent que la première strate commence à 0 m.
8. **Combustible en échelle**, pour chaque seuil `t` de `sorted(set(ladder_fuel_bd_thresholds))`
   - Initialisation : `LadderBot = LadderTop = CHT`, `Load = SFLeq0 = SFLeq_int = 0`, `ProfType = 1` si `CBH2 == CHT`, sinon 2.
   - `A = {CBD ≥ t et h > 1 et h ≤ CBH2}`. Le `1` est codé en dur en R (voir A2).
   - Si A n'est pas vide :
     - `i0` = premier indice de A, et `LadderBot = h[i0] − dz/2` ;
     - `j` avance tant que la strate est dans A : `LadderTop = h[j_dernier] + dz/2` ;
     - `ProfType = 3` si `LadderBot ≤ CHT`, sinon 4 ;
     - `K = {CHT < h ≤ LadderTop}` et `Load = Σ CBD[K] × dz` ;
     - `SFLeq0 = M × Load × CBH2 / (CBH2 − LadderTop/2)` ;
     - `SFLeq_int = M × Σ(CBD[K] × CBH2 / (CBH2 − h[K])) × dz`.
   - Dans tous les cas : `CFL = Σ CBD[h > LadderTop] × dz` et `FSG = CBH2 − LadderTop + dz/2`.
9. **Valeurs par défaut** : on prend les sorties du seuil **le plus proche de la médiane** pour remplir `LFL`, `CFL`, `FSG` et `ProfType`. C'est la règle de la PR #48, ligne 283 : `seuils[which.min(abs(seuils − median(seuils)))]`.
   - Avec un nombre impair de seuils, c'est la médiane elle‑même : 0.02 pour la configuration par défaut.
   - Avec un nombre pair, il y a deux seuils à égale distance de la médiane. `which.min` prend le premier rencontré dans **l'ordre de la liste fournie**, qui n'est pas forcément l'ordre trié. Par exemple, `[0.01, 0.02, 0.05, 0.1]` donne 0.02, alors que `[0.05, 0.02, 0.01, 0.1]` donne 0.05. Le Python doit reproduire cette règle : `np.argmin(np.abs(t − np.median(t)))` sur la liste de la config, sans la trier. `np.argmin` renvoie lui aussi le premier indice en cas d'égalité.
   - L'étape 8 calcule les métriques d'échelle sur `sorted(set(...))`, mais le suffixe par défaut se cherche dans la liste **brute**.
   - Puis on assemble les sorties dans l'ordre de la section 3.2.

## 5. Constats sur le code R

| ID | Constat | Preuve | Impact |
|---|---|---|---|
| **A1** | `CBD_cbh`, `CBD_cbh2` et `CBD_cbhlow` valent **toujours NA**. Le code cherche `zval == CBH`, mais `zval` est un centre de strate (x,5) alors que CBH est un bord (entier), depuis la modification du 12/05/2026. | `fuel_metrics.tif` : 0 pixel valide sur 40 401 pour ces 3 bandes. | Trois bandes vides livrées. |
| **A2** | Dans la recherche du combustible en échelle, la condition `zval > 1` est **codée en dur** au lieu d'utiliser `CanopyHeightThreshold`. | `ffuelmetrics2.R:253` | Aucun impact avec CHT = 1. Faux si CHT vaut 2 ou 3 m, alors que la documentation annonce « typiquement 1 à 3 m ». |
| **A3** | `ffuelmetrics2Rast` ne transmet **ni `smooth_pad` ni `CBHThresholdRatio`** à la fonction de calcul. En mode raster, R utilise donc toujours `smooth_pad = FALSE` et un ratio de 1/3, quelle que soit la config. | Prototype : 0 écart avec un ratio de 1/3 ; avec 0.333 (la valeur de la config), 209 pixels changent de CBH2 (jusqu'à 14 m) et 5 083 de CanBDlowmax. | La config INRAE 0.333 est **ignorée sans avertissement**. |
| **A4** | Les valeurs par défaut diffèrent entre la fonction scalaire et la fonction raster : FMA 0.25 contre 1, `CanopyBDTreshFrac` 0.3 contre 0.1, ratio 1/3 contre 1/5 (ce dernier n'est pas transmis). | `ffuelmetrics2.R:70-81` contre `:315-326` | Risque de confusion. En Python, une seule source de valeurs par défaut : la config. |
| **A5** | Un pixel sans donnée (PAD entièrement NaN) est converti en profil nul. Il reçoit alors `CLFL = 0`, `ProfType = 1`, `CBH2 = 1`… | 223 pixels entièrement NaN dans `pad_1m.tif` ; tous ont `CLFL = 0` dans `fuel_metrics.tif`. | **Faux signal « pas de combustible »** sur les zones sans mesure : bords de chantier, pixels écartés par le contrôle qualité. |
| **A6** | Un PAD partiellement NaN est mis à 0 sans le signaler. La variable `invalid` est calculée puis jamais utilisée. | `:122` ; 1 pixel concerné sur la zone pilote. | Faible. Compter ces pixels et le journaliser. |
| **A7** | Variables mortes : `ztmp`, `zval_1mplus`, `invalid`, `k <- 3`. | `:111-112`, `:122`, `:133` | Ne pas les porter. |
| **A8** | La documentation de `zval` (« pour PAD_1_10, zval devrait être 10.5 ») contredit l'usage réel (`z_bottom` = 10, puis +0,5·dz dans le code). | `:54` contre `parse_pad_heights` + `:106` | L'usage réel fait foi : le paramètre attend le **bas** de strate. |
| **A9** | Si le nombre de seuils d'échelle est pair, la médiane n'est pas un des seuils, et la recherche par nom échoue. **Corrigé par la PR #48** (ligne 283, mergée le 12/11/2026) : on prend le seuil le plus proche de la médiane (voir §4, étape 9). | `:286-292` (`7c7d35e`) ; PR #48 `e3e63fe`, merge `261b91e` | Le Python suit la PR #48. Il n'y a pas de `ValueError`. |
| **A10** | Unités de `fuel_mass_area()` : la documentation annonce un LMA en g/m² (valeur par défaut 140, ce qui donne 2,30 kg/m²). Or la valeur de la config, 0.2543576, correspond à `fuel_mass_area(LMA = 0.140)`, c'est‑à‑dire un LMA en **kg/m²**. | Calcul : 1 / (0,51 / (591/500) + 0,49 / 0,14) = 0,254358 | Hors périmètre, car on fournit la FMA en scalaire. À signaler à l'INRAE (la branche R `doc-fix-lma-units` existe). |
| **A11** | Les noms de bandes de `ffuelmetrics2Rast` viennent des seuils par défaut de `ffuelmetrics2`, et non de ceux passés en paramètre. **Corrigé par la PR #48** (ligne 362, mergée le 12/11/2026). | `:365` (`7c7d35e`) ; PR #48 `e3e63fe`, merge `261b91e` | Le Python dérive les noms des seuils configurés (§3.2). |

**Version R de référence : Dev `8fcccd2` (lidarforfuel 1.0.0.9013), qui intègre la PR #48.**
- **Merge de la PR #48, vérifié le 13/11/2026** (`git fetch origin Dev`). La PR a été mergée le 12/11/2026 (`261b91e`, commits `e3e63fe` + `ac17099`), puis `8fcccd2` a passé la version à 1.0.0.9013. Le diff `7c7d35e..8fcccd2` sur `R/ffuelmetrics2.R` ne contient **que les 2 lignes de la PR #48** (283 et 362), identiques à `1e2cc00`. La référence figée « `7c7d35e` + PR #48 » et Dev `8fcccd2` sont donc **identiques pour le calcul**, et rien ne change pour le portage.
- Les autres changements de Dev depuis `7c7d35e` ne touchent pas l'étape 3 : `pad_metrics.R` appelle désormais `lidarforfuel::.pad_metrics`, un simple préfixe d'espace de noms pour le parallélisme ; `get_traj` accepte un chunk de catalogue ; la config d'exemple a été renommée `inst/extdata/Config_example.yml`, avec `keep_N: true` pour le PAD ; l'exemple de pipeline montre l'étape 3 parallélisée dalle par dalle (`ffuelmetrics2` sur `SpatRaster`, `zval = z_bottom`, ce qui confirme A8). `tests/testthat/test-ffuelmetrics2.R` n'a pas changé.
- L'analyse initiale portait sur `e72b8f5`. Le diff `e72b8f5..7c7d35e` sur `R/ffuelmetrics2.R` a été vérifié le 05/11/2026 : il ne contient que de la documentation, des espaces et le retrait de `as_tibble` dans la signature de `ffuelmetrics2Rast`, qui le passait de toute façon en dur à `FALSE`. Les deux versions sont **équivalentes pour le calcul**.
- Au 05/11/2026, la tête de Dev (`188b4ab`) ne modifiait plus ce fichier. La PR #48 l'a modifié ensuite, comme décrit plus haut.
- La référence du 10/10/2026 (1.0.0.9012) a été produite sans la PR #48. Elle reste valide, car elle utilise 3 seuils (impair) et les seuils par défaut (noms corrects).
- La parité totale du prototype montre que `ffuelmetrics2` n'a pas changé entre ces versions.

## 6. Décisions à valider

| ID | Question | Recommandation (Mary) | Alternative | Note INRAE |
|---|---|---|---|---|
| **D1** | Fidélité du portage | **D'abord la parité stricte avec R** (story 3.4, test de référence), **ensuite** des corrections explicites et testées (story 3.5). Chaque divergence est listée dans la docstring et dans le CHANGELOG. | Corriger directement pendant le portage, mais on perd alors la preuve de parité. | — (choix IGN) |
| **D2** | Pixels sans donnée (A5) | Mettre **NaN sur toutes les bandes** quand le PAD du pixel est entièrement NaN. | Garder le comportement R (zéros). | **Q1 (prioritaire)** |
| **D3** | `CBD_cbh*` (A1) | Les **calculer correctement**, c'est‑à‑dire prendre la CBD de la strate dont le bas vaut CBH (`h_bottom == CBH`), et signaler le bug à l'INRAE. | Les garder à NaN (parité), ou retirer les 3 bandes (on aurait alors 42 bandes au lieu de 45). | Q2 (Q7 peut la simplifier) |
| **D4** | Valeur de `cbh_threshold_ratio` (A3) | Mettre **`0.3333333333333333`** dans la config (= 1/3 en double précision), car c'est la valeur que R utilise réellement, et transmettre **tous** les paramètres. | Mettre 0.333 (la valeur affichée par l'INRAE), mais on diverge alors de la production R. | Q4 |
| **D5** | `zval > 1` codé en dur (A2) | Utiliser `canopy_height_threshold`. Les résultats ne changent pas avec la config actuelle. | Garder 1 en dur. | Q3 |
| **D6** | FMA | Scalaire dans la config pour cette épique. Une FMA en raster (par essence ou par pixel) est différée. | — | — (hors périmètre, brief) |
| **D7** | Version R de référence | **Dev `8fcccd2` (1.0.0.9013)**. Elle contient la PR #48, mergée le 12/11/2026, et équivaut pour le calcul à l'ancienne référence « `7c7d35e` + PR #48 ». Les évolutions ultérieures de R ne sont pas suivies : l'INRAE prévient l'IGN de tout changement. | — (choix confirmé par le merge) | Q5 — **résolue** |

> **Mise à jour du 13/11/2026** : la PR #48 a été mergée le 12/11/2026. **D7 est confirmée**, et Q5 n'est plus une question pour la réunion.
>
> **Statut au 05/11/2026** : D1, D6 et D7 sont retenues côté IGN (D7 est à confirmer par l'INRAE, Q5). D2, D3, D4 et D5 modifient des résultats ou des choix scientifiques. Elles seront **présentées à la réunion de mi‑novembre** (note `note-inrae-reunion-mi-novembre.md`), avec des réponses attendues d'Olivier Martin et François Pimont. La story 3.5 attend ces réponses ; les stories 3.1 à 3.4 n'en dépendent pas.
>
> Trois questions de la note ne sont pas des décisions de portage, mais conditionnent la validation :
> - **Q6**, la tolérance par métrique sur la chaîne complète : on la chiffre avec la story 3.8 ;
> - **Q7**, les bandes réellement exploitées par l'INRAE : on oriente la validation selon la réponse ;
> - **Q8**, qui valide à l'INRAE et à quelle date : la validation tombe entre mi‑décembre et janvier, pendant les fêtes.

## 7. Pièges du portage R → Python

1. **Indices.** R compte à partir de 1, Python à partir de 0 :
   - `CBDval[1]` → `cbd[0]` ;
   - la boucle `for (k in rev(seq.int(idmin + 1, idmax)))` qui teste `CBDval[k - 1]` devient `for k in range(idmax, idmin, -1)` qui teste `cbd[k - 1]`.
2. **Premier indice.** `which(x == max)[1]` en R correspond à `np.flatnonzero(x == m)[0]` ou à `np.argmax(x == m)` en Python. Prendre le **premier** élément, pas le dernier.
3. **NaN.** En R, `which()` ignore les NA. En numpy, une comparaison avec NaN vaut `False`, ce qui donne le même résultat pour la sélection d'indices. En revanche, `any(NA)` vaut NA en R : il faut **traiter explicitement** le cas `CTH` NaN pour `CDH` et `CBDcan` (étapes 5 et 6). De même, `min(x, NA)` vaut NA en R, alors que `min()` en Python ne propage pas NaN de façon fiable.
4. **Bornes en virgule flottante.** Il faut **multiplier** `CTH * ratio`, comme R, et non diviser. Avec un ratio de 1/3, `CTH × ratio` tombe exactement sur un centre de strate (x,5) quand `CTH = 3k + 1,5`. Le choix entre `>` et `>=` décide alors du résultat. Il faut garder exactement les comparateurs de R (section 4).
5. **Égalité `CBH2 == CHT`** pour initialiser `ProfType`. Elle est exacte tant que `dz` vaut 0,5 ou 1. Utiliser `np.isclose` pour la robustesse : cela n'a aucun effet sur la parité.
6. **Noms de bandes.** `f"{t:0.3f}".replace(".", "p")` pour les seuils d'échelle et `f"{t:0.4f}"` pour les seuils CTH. Le nom `SFLeq_intot` (et non `SFLeq_int`) est conservé.
7. **Lissage.** Moyenne sur 3 strates avec fenêtre partielle aux bords, par exemple `pd.Series(pad).rolling(3, center=True, min_periods=1).mean()`. **Ne pas** utiliser `np.convolve(mode="same")`, qui divise toujours par 3.
8. **Recalcul de la densité CBD** à l'étape 4 : il part du **PAD** (éventuellement lissé), et non de la CBD de l'étape 1.

## 8. Données de référence et preuve de faisabilité

- **Référence R** : `zone_pilote_test/HE_IGN_Galaxy/R/10_10_2026/`.
  - `pad_1m.tif` : 201 × 201 pixels de 10 m, 186 bandes dont `PAD_1_0` à `PAD_1_59`.
  - `fuel_metrics.tif` : 45 bandes.
  - La grille R (origine 512489.75, 6856496.75) est **alignée** sur la grille CosiaFrance de la config Python (98039.75, 7111486.75), puisque les deux tombent en x9,75 et x6,75 modulo 10.
  - Configuration : `20261011T022901_config.yaml`.
- **Prototype** : `proto-ffuelmetrics2-parite-2026-10-22.py` (dossier de recherche local, hors dépôt).
  - C'est un portage littéral de la section 4, sans les décisions D2 à D5.
  - Résultat : **0 écart sur 40 401 × 45 valeurs** (tolérance relative 10⁻⁴, écart maximal observé 5,6·10⁻⁶).
  - Il sert de **preuve, pas de code à copier** : il ne respecte ni le TDD ni les conventions du dépôt.
- **Performance** : 0,27 ms par pixel en boucle Python, soit environ 3 s par dalle de 1 km (10 000 pixels). Ni vectorisation ni parallélisation ne sont nécessaires dans cette épique.

---

## 9. Stories

> **Modèle suivi.** Chaque story comporte :
> - une **Description** du point de vue de l'utilisateur, sans détails d'implémentation : pour qui, ce qu'on cherche à faire, et l'objectif final dans le contexte de la chaîne ;
> - une **Estimation de l'effort**, grossière : Non spécifié, Trivial, Facile, Normal ou Difficile ;
> - des **Critères d'acceptation** testables, au format « Étant donné…, quand…, alors… ».
>
> Les détails techniques (dépendances, fichiers pressentis) sont regroupés à la fin de chaque story, dans « Informations complémentaires ». Les noms de fichiers sont des **propositions** : Amelia les valide dans la spec de chaque story, en suivant l'idiome du dépôt (une responsabilité par module, arguments explicites, docstrings Google, tests `test/<module>/test_<fichier>.py`). Elle doit aussi suivre la consigne mémoire « adapter les fonctions existantes plutôt que d'en créer des parallèles », par exemple réutiliser `_write_geotiff` et `_format_num`.

### Story 3.1 — Jeu de données de référence R

#### Description

**En tant que** développeuse, **je veux** un petit jeu de profils PAD accompagnés des métriques que R a calculées pour eux, versionné dans le dépôt, **pour pouvoir** développer chaque story en TDD contre la sortie R.

- **Pour qui** : la développeuse de l'étape 3.
- **Ce qu'on cherche à faire** : disposer, dès le départ, de la « bonne réponse » pour un échantillon de pixels. On extrait une petite fenêtre de la sortie R de la zone pilote : les profils PAD en entrée et les 45 métriques attendues en sortie. Cette fenêtre doit couvrir tous les cas particuliers de l'algorithme.
- **Objectif final** : chaque story suivante écrit d'abord un test qui compare le Python à ces valeurs R, puis le code qui le fait passer. Ce jeu sert aussi de preuve de parité pour la validation par l'INRAE.

#### Estimation de l'effort

Facile.

#### Critères d'acceptation

- **AC-3.1.1** — **Étant donné** la sortie R de la zone pilote HE_IGN_Galaxy (`pad_1m.tif`, `fuel_metrics.tif`), **quand** on extrait le jeu de référence, **alors** `data/fuel_metrics/` contient :
  - une fenêtre de `pad_1m.tif` avec uniquement les bandes `PAD_1_*` ;
  - la même fenêtre de `fuel_metrics.tif` (45 bandes) ;
  - ces deux fichiers alignés au pixel près et pesant moins de 1 Mo au total.
- **AC-3.1.2** — **Étant donné** le jeu de référence extrait, **quand** on lance le test de couverture, **alors** il trouve au moins un pixel pour chacun de ces cas :
  - PAD entièrement NaN ;
  - `CTH_0p0100` NaN ;
  - `ProfType` = 1, 2, 3 et 4 ;
  - `CBH` NaN avec `CBHlow` défini ;
  - `CBH2` issu de 3A.

  Si aucune fenêtre unique ne couvre tous les cas, on ajoute un jeu de profils sélectionnés, sous forme de CSV ou de petit raster de 1 × N pixels.
- **AC-3.1.3** — **Étant donné** le dossier `data/fuel_metrics/`, **quand** on ouvre son `README.md`, **alors** on y trouve la provenance : version lidarforfuel 1.0.0.9012, configuration, date du traitement et fenêtre extraite.
- **AC-3.1.4** — **Étant donné** le jeu de référence, **quand** un test appelle le helper de lecture, **alors** il obtient `(pad[n_pixels, n_strates], h_bottom, dz, attendu{nom: valeurs})`.

#### Informations complémentaires

**Dépendances** : aucune.

### Story 3.2 — Lecture des strates et préparation du profil CBD

#### Description

**En tant que** développeuse, **je veux** des fonctions qui lisent les noms de bandes PAD et transforment un profil PAD en profil CBD, **pour pouvoir** réaliser l'étape 1 de l'algorithme et fournir aux étapes suivantes (2 à 4) le profil CBD, les hauteurs de strates et `dz`.

- **Pour qui** : la développeuse de l'étape 3. Cette story est la couche d'entrée du calcul, sur laquelle s'appuient les stories 3.3 et 3.4.
- **Ce qu'on cherche à faire** :
  1. **Lire les noms de bandes.** Le raster PAD contient des bandes nommées `PAD_1_0`, `PAD_1_1`…, mêlées à d'autres bandes (`Cover_2`, `N_1_3`…). On ne garde que les bandes `PAD_*`. Leur nom donne l'épaisseur des strates et la hauteur du bas de chaque strate : `PAD_1_10` signifie `dz = 1` m et `h_bottom = 10` m au‑dessus du sol.
  2. **Passer du PAD à la CBD.** Le **PAD** (*Plant Area Density*, en m²/m³) mesure la quantité de végétation dans chaque strate. La **CBD** (*Canopy Bulk Density*, en kg/m³) mesure la masse de combustible. On passe de l'un à l'autre en multipliant par la masse surfacique du combustible (FMA) : celle de la canopée au‑dessus d'une hauteur limite, celle des arbustes en dessous. Avant ce calcul, les valeurs invalides (NaN, inf, négatives) sont remplacées par 0, et le profil peut être lissé sur 3 strates.
  3. **Choisir le seuil d'échelle par défaut**, parmi ceux de la config. C'est lui qui servira à remplir les bandes `LFL`, `CFL`, `FSG` et `ProfType`.
- **Objectif final** : toutes les métriques de combustible se calculent à partir de ce profil CBD. Une entrée propre et validée ici évite de propager des erreurs dans les 45 métriques.

#### Estimation de l'effort

Facile.

#### Critères d'acceptation

- **AC-3.2.1** — **Étant donné** les noms `PAD_1_0` … `PAD_1_59` mêlés à d'autres bandes (`Cover_2`, `N_1_3`…), **quand** on lit les strates, **alors** on obtient les indices des bandes PAD, `dz = 1` et `h_bottom = [0 … 59]`, triés numériquement. **Étant donné** le nom `PAD_0.5_1.5`, **alors** on obtient `dz = 0.5` et `h_bottom = 1.5`.
- **AC-3.2.2** — **Étant donné** des bandes PAD d'épaisseurs différentes (par exemple `PAD_1_0` et `PAD_0.5_1`), **quand** on lit les strates, **alors** une `ValueError` est levée.
- **AC-3.2.3** — **Étant donné** un profil PAD qui contient des `NaN`, des `inf` ou des valeurs négatives, **quand** on prépare le profil, **alors** ces valeurs sont remplacées par 0, et le nombre de valeurs corrigées est journalisé en debug.
- **AC-3.2.4** — **Étant donné** le profil `[0, 3, 0, 0]` et `smooth_pad=True`, **quand** on prépare le profil, **alors** on obtient `[1.5, 1, 1, 0]` : moyenne centrée sur 3 strates, et moyenne sur les 2 valeurs disponibles aux bords.
- **AC-3.2.5** — **Étant donné** un profil PAD préparé, `fma_can`, `fma_shrub`, `bd_max` et une hauteur limite, **quand** on calcule la CBD, **alors** elle vaut `min(PAD × FMA, bd_max)`, avec `fma_can` pour les strates au‑dessus de la limite et `fma_shrub` pour les autres. La limite est un paramètre de la fonction : CHT à l'étape 1, `CBH2` à l'étape 4.
- **AC-3.2.6** — **Étant donné** une liste de seuils d'échelle, **quand** on choisit le seuil par défaut, **alors** on obtient le seuil le plus proche de la médiane, et en cas d'égalité le premier de la liste **dans l'ordre de la config** (règle de la PR #48, A9, §4 étape 9) :
  - `[0.01, 0.02, 0.05]` → 0.02 ;
  - `[0.01, 0.02, 0.05, 0.1]` → 0.02 ;
  - `[0.05, 0.02, 0.01, 0.1]` → 0.05 ;
  - `[0.02]` → 0.02.

  Un nombre pair de seuils ne lève pas d'erreur. **Étant donné** une liste vide, **alors** une `ValueError` est levée.

#### Informations complémentaires

**Périmètre** : étape 1 de la section 4, lecture de `PAD_{dz}_{h_bas}` (équivalent de `parse_pad_heights` en R, `R/pad_metrics.R:286`) et validation des paramètres. **Fichiers pressentis** : `lidar_for_fuel/fuel_metrics/prepare_cbd_profile.py`, `test/fuel_metrics/test_prepare_cbd_profile.py`. **Dépendances** : aucune.

### Story 3.3 — Métriques de canopée (CTH, CBH, CBH2, CBHlow, CDH, CBDcan)

#### Description

**En tant qu'** utilisatrice des cartes de combustible, **je veux** les hauteurs et densités de la canopée calculées comme en R, **pour pouvoir** caractériser la strate dominante de chaque pixel.

- **Pour qui** : les utilisateurs des métriques (INRAE, modélisation du comportement du feu), et l'IGN qui les produit.
- **Ce qu'on cherche à faire** : décrire la canopée à partir du profil CBD de chaque pixel :
  - **CTH**, la hauteur du sommet de la canopée, pour deux seuils de densité ;
  - **CBH**, **CBH2** et **CBHlow**, la base du houppier selon trois définitions. CBH2, la base « continue », est la métrique principale ;
  - **CDH**, la hauteur dominante ;
  - **CBDcan**, la densité moyenne de combustible dans la canopée ;
  - les densités extrêmes `CanBD*` qui servent à trouver ces hauteurs.
- **Objectif final** : la base et la densité du houppier conditionnent le passage d'un feu de surface à un feu de cimes. Ces valeurs doivent être identiques à celles de R pour que l'INRAE valide la chaîne Python. Elles servent aussi d'entrée à la story 3.4, en particulier CBH2.

#### Estimation de l'effort

Normal.

#### Critères d'acceptation

- **AC-3.3.1** — **Étant donné** des profils synthétiques écrits à la main pour chaque branche de l'algorithme, **quand** on calcule les métriques de canopée, **alors** chaque métrique a la valeur calculée à la main. Les branches couvertes sont :
  - 3A normal ;
  - 3A avec `CanBDmin == CanBDmax`, qui ne donne pas de CBH ;
  - repli sur CBHlow ;
  - repli sur `CHT` ;
  - profil nul ;
  - profil dont le maximum est atteint sur plusieurs strates, où l'on doit prendre la **première**.
- **AC-3.3.2** — **Étant donné** un profil sans aucune strate de CBD ≥ 0.01, **quand** on calcule les métriques de canopée, **alors** `CTH_0p0100`, `CDH` et `CBDcan` valent NaN, `CBH2 = CHT`, et `CTH_0p0012` est renseigné s'il existe une densité ≥ 0.0012 (piège 3).
- **AC-3.3.3** — **Étant donné** le jeu de référence 3.1, **quand** on calcule les métriques de canopée de chaque pixel, **alors** `CTH_0p0012`, `CTH_0p0100`, `CDH`, `CBH`, `CBH2`, `CBHlow`, `CanBDmax`, `CanBDmin`, `CanBDlowmax`, `CanBDlowmin` et `CBDcan` sont égales à celles de R : `assert_allclose(rtol=1e-5, atol=1e-6)`, avec égalité des NaN.
- **AC-3.3.4** — **Étant donné** `ratio = 1/3` et un profil où `CTH = 4.5`, **quand** on cherche la base du houppier en 3A, **alors** la strate de centre 1,5 m est exclue de la recherche. Ce test de non‑régression vérifie que les bornes en virgule flottante restent celles de R (piège 4).

#### Informations complémentaires

**Périmètre** : étapes 2 à 6 de la section 4. Elles renvoient `CTH_*`, `CBH`, `CBH2`, `CBHlow`, `CanBD*`, `CDH`, `CBDcan` et la CBD recalculée (étape 4), dont la story 3.4 a besoin. **Fichiers pressentis** : `lidar_for_fuel/fuel_metrics/compute_canopy_metrics.py` et son test. **Dépendances** : 3.1, 3.2.

### Story 3.4 — Charges, combustible en échelle et assemblage des métriques (parité stricte avec R)

#### Description

**En tant qu'** utilisatrice des cartes de combustible, **je veux** obtenir pour un profil vertical l'ensemble des métriques de combustible, dans l'ordre et avec les noms de R (45 avec la config par défaut), **pour pouvoir** disposer exactement de ce que produit `ffuelmetrics2`.

- **Pour qui** : les utilisateurs des métriques (INRAE), et l'IGN qui doit prouver la parité avec R.
- **Ce qu'on cherche à faire** : compléter les métriques de canopée (story 3.3) avec :
  - les **charges de combustible** : `CLFL` au‑dessus de 1 m, `FL_k1` et `FL_k2` dans les deux premières strates ;
  - pour chaque seuil de densité, le **combustible en échelle** : sa hauteur de bas et de haut, sa charge, la charge de surface équivalente selon Perrakis, la charge de canopée au‑dessus (`CFL`) et l'écart entre strates (`FSG`) ;
  - le **type de profil** `ProfType` (voir §3.2) ;
  - puis **assembler** toutes les métriques dans l'ordre de R.
- **Objectif final** : c'est le cœur du portage, et la preuve que le Python reproduit R à l'identique. Cette story n'applique **aucune** correction des défauts de R : les corrections viennent ensuite, en story 3.5, et restent ainsi mesurables. La même fonction pourra servir plus tard au calcul sur une placette.

#### Estimation de l'effort

Normal.

#### Critères d'acceptation

- **AC-3.4.1** — **Étant donné** la config par défaut, **quand** on calcule les métriques d'un profil, **alors** on obtient exactement les 45 noms de la section 3.2, dans le même ordre. **Étant donné** `ladder_fuel_bd_thresholds = [0.01, 0.02, 0.05, 0.1]`, **alors** on obtient 19 + 8 × 4 + 2 = **53** clés, dont `LadderBot_0p100` … `FSG_0p100`.
- **AC-3.4.2** — **Étant donné** des profils synthétiques écrits à la main pour chaque cas de combustible en échelle, **quand** on calcule les métriques, **alors** chacune a la valeur calculée à la main :
  - pas de combustible en échelle → ProfType 1 ou 2, `LadderBot = LadderTop = CHT`, `Load = 0` ;
  - échelle continue depuis la surface → ProfType 3 ;
  - échelle suspendue → ProfType 4 ;
  - échelle interrompue, où seul le premier bloc continu compte.
- **AC-3.4.3** — **Étant donné** un profil avec du combustible en échelle, **quand** on calcule les charges, **alors** `SFLeq0` et `SFLeq_intot` suivent les formules de l'étape 8, `FSG = CBH2 − LadderTop + dz/2`, et `CFL` est la somme des charges au‑dessus de `LadderTop`.
- **AC-3.4.4** — **Étant donné** un profil quelconque, **quand** on calcule les métriques, **alors** `LFL`, `CFL`, `FSG` et `ProfType` sont égaux aux valeurs du seuil médian (`LadderLoad_0p020`, `CFL_0p020`, `FSG_0p020`, `ProfType_0p020` avec la config par défaut).
- **AC-3.4.5 (test de parité R)** — **Étant donné** tout le jeu de référence 3.1, **quand** on calcule les métriques de chaque pixel, **alors** les 45 métriques sont égales à celles de R (`rtol=1e-5, atol=1e-6`, NaN égaux). Cela vaut aussi pour les pixels entièrement NaN (zéros de R) et pour `CBD_cbh*` (NaN de R).

#### Informations complémentaires

**Périmètre** : étapes 7 à 9 de la section 4, plus l'assemblage, en composition avec 3.2 et 3.3. Signature proposée : `fuel_metrics_core(pad, h_bottom, dz, **params) -> dict`. **Fichiers pressentis** : `lidar_for_fuel/fuel_metrics/compute_ladder_metrics.py`, `lidar_for_fuel/fuel_metrics/calculate_fuel_metrics.py` (`fuel_metrics_core`) et leurs tests. **Dépendances** : 3.3.

### Story 3.5 — Corrections assumées par rapport à R

#### Description

**En tant qu'** utilisatrice des cartes de combustible, **je veux** que les défauts identifiés dans le code R soient corrigés de façon explicite et tracée, **pour pouvoir** utiliser des cartes sans faux zéros ni bandes vides.

- **Pour qui** : les utilisateurs des métriques (INRAE), et l'équipe de production IGN qui livre les cartes.
- **Ce qu'on cherche à faire** : corriger trois défauts de R, selon les décisions prises avec l'INRAE (section 6) :
  - **D2** : un pixel sans donnée ne doit plus apparaître comme « sans combustible » (charges nulles), mais comme « sans valeur » ;
  - **D3** : les trois bandes `CBD_cbh*`, toujours vides en R à cause d'un bug, doivent être renseignées ;
  - **D5** : la recherche du combustible en échelle doit respecter `canopy_height_threshold` au lieu d'une valeur de 1 m codée en dur.
- **Objectif final** : livrer en production des cartes justes, tout en gardant la trace de chaque écart avec R pour que l'INRAE puisse le vérifier.

#### Estimation de l'effort

Facile.

#### Critères d'acceptation

> **Préalable** : décisions D2, D3 et D5 validées avec l'INRAE (section 6). Ces critères sont à adapter selon la décision retenue.

- **AC-3.5.1 (D2)** — **Étant donné** un pixel dont le PAD est entièrement NaN, **quand** on calcule ses métriques, **alors** toutes valent NaN.
- **AC-3.5.2 (D3)** — **Étant donné** un profil où CBH, CBH2 ou CBHlow est défini, **quand** on calcule ses métriques, **alors** `CBD_cbh`, `CBD_cbh2` ou `CBD_cbhlow` vaut la densité CBD, **après l'étape 4**, de la strate dont le bas vaut cette hauteur. **Étant donné** une de ces hauteurs à NaN, **alors** la densité correspondante vaut NaN.
- **AC-3.5.3 (D5)** — **Étant donné** `canopy_height_threshold = 2`, **quand** on cherche le combustible en échelle, **alors** la recherche utilise `h > canopy_height_threshold` et le résultat diffère de celui de R. **Étant donné** `canopy_height_threshold = 1`, **alors** le résultat est identique à celui de R.
- **AC-3.5.4** — **Étant donné** les corrections en place, **quand** on lance le test de parité 3.4, **alors** il reste vert sur toutes les autres métriques. Les divergences assumées sont **exclues explicitement par leur nom** dans ce test, avec un commentaire qui renvoie à cette épique, et non par une tolérance.
- **AC-3.5.5** — **Étant donné** une divergence assumée avec R, **quand** on lit la docstring de `fuel_metrics_core` et le `CHANGELOG.md`, **alors** elle y est listée.

#### Informations complémentaires

**Dépendances** : 3.4, et les réponses de l'INRAE (réunion de mi‑novembre).

### Story 3.6 — Application au raster PAD et export GeoTIFF

#### Description

**En tant qu'** utilisatrice des cartes de combustible, **je veux** calculer les métriques sur toute une dalle à partir de son raster PAD `{stem}_pad_profile_1m.tif`, **pour pouvoir** obtenir la carte `{stem}_fuel_metrics.tif`.

- **Pour qui** : les utilisateurs des cartes (INRAE, acteurs de la prévention des incendies), et l'équipe de production IGN.
- **Ce qu'on cherche à faire** : appliquer le calcul de la story 3.4 à chaque pixel d'une dalle, puis écrire une carte multi‑bandes, une bande par métrique, sur la même grille que le raster PAD. Le calcul doit passer sur toutes les dalles, y compris atypiques (eau, bord de chantier, sol nu), et dire ce qu'il a corrigé.
- **Objectif final** : produire une carte de combustible par dalle LiDAR HD, ce qui permet la production nationale à partir de janvier 2027.

#### Estimation de l'effort

Normal.

#### Critères d'acceptation

- **AC-3.6.1** — **Étant donné** un raster PAD, **quand** on calcule et exporte les métriques, **alors** le fichier écrit a exactement la même transform, le même CRS et la même taille que le raster d'entrée. Il contient une bande float32 par métrique (45 avec la config par défaut, voir §3.2), nodata NaN, et les descriptions de bandes sont les noms renvoyés par `fuel_metrics_core`, dans l'ordre.
- **AC-3.6.2** — **Étant donné** un raster PAD Python (60 bandes PAD) ou un raster PAD R (186 bandes), **quand** on calcule les métriques, **alors** les bandes PAD sont trouvées **par leur nom** (story 3.2), et non par leur position, et les deux rasters sont acceptés.
- **AC-3.6.3** — **Étant donné** la fenêtre de référence 3.1, **quand** on produit le raster de métriques, **alors** il est égal au raster R, à l'exception des divergences assumées en 3.5.
- **AC-3.6.4** — **Étant donné** un pixel dont le PAD est entièrement NaN, **quand** on produit le raster, **alors** ce pixel vaut NaN sur toutes les bandes (D2).
- **AC-3.6.5** — **Étant donné** le code d'export, **quand** on le relit, **alors** l'écriture passe par `_write_geotiff` de `pad_profile/export_raster.py`, éventuellement déplacé dans `commons/` s'il est partagé. On n'écrit pas de nouvel outil d'écriture.
- **AC-3.6.6** — **Étant donné** un raster de 100 × 100 pixels sur un poste de développement, **quand** on lance le calcul, **alors** il prend moins de 10 s. C'est une indication, pas un test bloquant en CI.
- **AC-3.6.7** — **Étant donné** une dalle atypique, **quand** on lance le calcul, **alors** il passe **sans exception** (critère de succès 4 du brief) :
  - une dalle dont **tous** les pixels sont NaN (eau, bord de chantier) produit un raster de la même grille, entièrement NaN sur toutes les bandes, avec un `logger.warning` ;
  - une dalle sans aucune strate au‑dessus de `canopy_height_threshold` (sol nu) donne des métriques finies, sans division par zéro ni `RuntimeWarning` numpy non maîtrisé.

  Les deux cas sont testés sur de petits rasters synthétiques.
- **AC-3.6.8** — **Étant donné** une dalle traitée, **quand** le calcul se termine, **alors** un `logger.info` donne, pour que ce qui a été corrigé soit **signalé et non masqué** :
  - le nombre de pixels entièrement NaN, mis à NaN en sortie (D2) ;
  - le nombre de profils partiellement NaN, inf ou négatifs, corrigés à 0 (A6, AC‑3.2.3) ;
  - le nombre de pixels traités.
- **AC-3.6.9** — **Étant donné** un raster d'entrée **sans aucune bande `PAD_*`**, **quand** on lance le calcul, **alors** une `ValueError` explicite donne le nom du fichier. Ce n'est pas une dalle atypique mais une erreur d'entrée : on ne produit pas de sortie vide en silence.

#### Informations complémentaires

**Fichiers pressentis** : `lidar_for_fuel/fuel_metrics/export_fuel_metrics.py` et son test. **Dépendances** : 3.4 (3.5 pour AC-3.6.4).

### Story 3.7 — Point d'entrée `main_fuel_metrics`, configuration et documentation

#### Description

**En tant qu'** opératrice de production, **je veux** lancer l'étape 3 comme les étapes 1 et 2, sur un fichier ou sur tout un dossier, **pour pouvoir** enchaîner toute la chaîne lidar-for-fuel.

- **Pour qui** : l'équipe qui lance les traitements (IGN, et à terme la production nationale par la GPAO).
- **Ce qu'on cherche à faire** : une commande unique, configurée comme les étapes précédentes, qui traite un fichier ou tout un dossier de rasters PAD et écrit les cartes de métriques. Les paramètres scientifiques sont dans la config, avec les valeurs de l'INRAE par défaut. Une documentation et un script d'exemple permettent à d'autres personnes de lancer l'étape sans aide.
- **Objectif final** : rendre l'étape 3 utilisable en production, pour la validation de l'INRAE puis la production nationale de janvier 2027.

#### Estimation de l'effort

Facile.

#### Critères d'acceptation

- **AC-3.7.1** — **Étant donné** `configs/config.yaml`, **quand** on l'ouvre, **alors** il contient une section `fuel_metrics:` avec les paramètres du tableau 3.1, nommés en snake_case. Les valeurs par défaut sont celles de la config INRAE, avec `cbh_threshold_ratio: 0.3333333333333333` (D4).
- **AC-3.7.2** — **Étant donné** un dossier qui contient des fichiers `*_pad_profile_1m.tif`, **quand** on lance `python -m lidar_for_fuel.main_fuel_metrics io.input_dir=… io.output_dir=…`, **alors** tous les fichiers du dossier sont traités, ou seulement `io.input_filename` s'il est renseigné, et un fichier `{stem}_fuel_metrics.tif` est écrit pour chacun. **Étant donné** un `input_dir` ou un `output_dir` invalide, **alors** les messages d'erreur sont identiques à ceux de `main_pad_profile.py`.
- **AC-3.7.3** — **Étant donné** une config où `smooth_pad` est vrai ou bien `5 × min(FMA) < bd_max`, **quand** on lance le traitement, **alors** un avertissement est journalisé au démarrage : la condition d'équivalence avec R de la section 3.1 n'est pas remplie.
- **AC-3.7.4** — **Étant donné** la story livrée, **quand** on consulte le dépôt, **alors** on y trouve :
  - `scripts/example_launch_fuel_metrics.sh` ;
  - le README mis à jour (étape 3, contenu de la sortie, tableau des 45 bandes ou lien vers ce tableau) ;
  - `CHANGELOG.md` et `_version.py` mis à jour.
- **AC-3.7.5** — **Étant donné** la fenêtre de référence, **quand** le test d'intégration lance `main` (via `hydra.compose` ou un appel direct, comme `test_main_pad_profile.py`), **alors** le fichier est produit avec 45 bandes (config par défaut).

#### Informations complémentaires

**Dépendances** : 3.6.

### Story 3.8 (obligatoire) — Mesure rapide des écarts sur la chaîne complète

#### Description

**En tant qu'** analyste, **je veux** chiffrer rapidement l'écart entre les métriques de la chaîne Python complète et celles de la chaîne R sur la dalle `Semis_2024_0513_6856`, **pour pouvoir** proposer à l'INRAE une tolérance par métrique (Q6) à la réunion de mi‑novembre, puis fournir un élément de validation en décembre.

- **Pour qui** : l'analyste IGN (Malvina), et l'INRAE qui doit fixer les tolérances et valider la chaîne.
- **Ce qu'on cherche à faire** : comparer, pixel par pixel, les cartes de métriques produites par Python et par R sur une même dalle, et résumer les écarts métrique par métrique. La mesure se fait **deux fois** :
  - **passe 1, avant la mi‑novembre 2026** : l'étape 3 est calculée par le **prototype**, appliqué au PAD de lidar‑for‑fuel **v0.3.0** ;
  - **passe 2, mi‑décembre 2026** : même mesure, avec cette fois `main_fuel_metrics` (story 3.7), dans sa version finale.
- **Objectif final** : arriver à la réunion avec des chiffres plutôt qu'avec des intuitions, et montrer d'où viennent les écarts. Comme l'étape 3 seule ne crée aucun écart, tout écart mesuré vient de l'étape 2 (le PAD).

#### Estimation de l'effort

Facile.

#### Critères d'acceptation

- **AC-3.8.1** — **Étant donné** les rasters de métriques Python et R d'une même dalle, **quand** on les compare, **alors** la comparaison porte sur l'intersection de leurs emprises, les deux grilles étant alignées. Les pixels valides d'un côté et NaN de l'autre sont **comptés à part** et ne sont pas mêlés aux écarts.
- **AC-3.8.2** — **Étant donné** la comparaison faite, **quand** on lit les résultats, **alors** on obtient, pour chaque métrique ou chaque groupe de la note INRAE (Q6) :
  - l'écart absolu moyen ;
  - le 99ᵉ centile de l'écart absolu ;
  - la part de pixels dans une tolérance candidate ;
  - pour `ProfType*`, le taux d'accord de classe et la matrice de confusion.
- **AC-3.8.3** — **Étant donné** les résultats de la passe 1, **quand** on prépare la réunion, **alors** le tableau Q6 de `note-inrae-reunion-mi-novembre.md` est rempli : colonne « Écart mesuré », plus une proposition de seuil par métrique.
- **AC-3.8.4** — **Étant donné** le prototype appliqué au PAD R, **quand** on compare sa sortie à celle de R, **alors** l'écart est nul. L'écart mesuré sur la chaîne complète vient donc **entièrement du PAD**. Le rapport le dit, et vérifie l'hypothèse de l'addendum selon laquelle l'écart sur la strate 0–1 m touche surtout `FL_k1`.
- **AC-3.8.5** — **Étant donné** une passe terminée, **quand** on ouvre `research/mesure-rapide-etape3-<date>.md`, **alors** on y trouve la version de lidar‑for‑fuel, le commit, la configuration et les chiffres de cette passe.

#### Informations complémentaires

C'est un **script ponctuel**, sans tests ni CLI, rangé dans `_bmad-output/planning-artifacts/research/` du dépôt lidar-for-fuel. Prototype utilisé en passe 1 : `proto-ffuelmetrics2-parite-2026-10-22.py`. **Dépendances** : aucune pour la passe 1 (prototype et sorties v0.3.0) ; 3.7 pour la passe 2. **Responsable suggérée** : Malvina, avec Mary en appui.

### Story 3.9 (reportable) — Étape 3 dans l'outil de comparaison `lff-compare`

#### Description

**En tant qu'** analyste, **je veux** que `lff-compare` compare aussi les métriques de combustible, **pour pouvoir** rejouer la comparaison IGN/INRAE complète sur d'autres zones, avec un rapport homogène avec ceux des étapes 1 et 2.

- **Pour qui** : les analystes IGN et INRAE qui comparent les deux chaînes.
- **Ce qu'on cherche à faire** : transformer la mesure ponctuelle de la story 3.8 en un outil réutilisable, intégré à `lff-compare`. Il doit distinguer l'écart dû à l'étape 3 elle‑même de l'écart hérité du PAD.
- **Objectif final** : pouvoir vérifier la chaîne sur de nouvelles zones, ou après une évolution du code R ou Python, sans réécrire de script.

> **Premier élément reporté en cas de retard** (brief, « Priorités »). On peut la livrer après janvier 2027.

#### Estimation de l'effort

Normal.

#### Critères d'acceptation

- **AC-3.9.1** — **Étant donné** le dépôt `lff_compare`, **quand** on y ajoute l'étape 3, **alors** un module `lff_compare/fuel_metrics/` (`compare.py`, `summary.py`, `report.py`) est créé sur le modèle de `pad_prod_file/`. Il réutilise `commons/` (config Hydra, découverte des fichiers, rapports, figures, PDF), sans dupliquer d'outils.
- **AC-3.9.2** — **Étant donné** une dalle avec ses sorties R et Python, **quand** on lance la comparaison, **alors** deux modes sont proposés :
  - **A**, l'étape 3 Python appliquée au PAD R : c'est la parité, avec un écart attendu nul hors des divergences assumées de la story 3.5 ;
  - **B**, la chaîne Python complète comparée à la chaîne R.

  Le rapport donne aussi B − A, c'est‑à‑dire l'écart qui vient du PAD.
- **AC-3.9.3** — **Étant donné** une comparaison terminée, **quand** on ouvre le rapport, **alors** il commence par une synthèse chiffrée, comme les rapports existants, reprend les indicateurs de l'AC-3.8.2 et ajoute des cartes d'écart sur CBH2 et CFL. `synthesis.py` intègre l'étape 3.
- **AC-3.9.4** — **Étant donné** le module livré, **quand** on lance sa suite de tests, **alors** on trouve des tests rapides sur données synthétiques et un test de non‑régression `-m slow` sur la dalle `0513_6856`, selon les conventions du dépôt `lff-compare`.

#### Informations complémentaires

**Dépendances** : 3.7 (pour le mode A et une entrée stable). **Dépôt** : `lff_compare`, et non `lidar-for-fuel`.

---

## 10. Hors périmètre (différé)

- **FMA en raster**, par pixel ou par essence : R accepte `FMAcan`/`FMAshrub` sous forme de `SpatRaster`.
- **Graphique `plot_profile`** (contrôle visuel d'un profil).
- **Calcul au niveau d'une placette** à partir d'un nuage (`cloud_metrics`). La fonction 1‑D de la story 3.4 le permet déjà ; il n'y a juste pas de point d'entrée dédié.
- **Charge de surface SFL** calculée à partir du profil à 0,5 m (`pad_sb_0.5m`) et **correction par quantile mapping** de `FL_k1`/`FL_k2`, annoncée dans les commentaires R mais pas implémentée.
- **Vectorisation ou parallélisation dans une dalle** : inutile à 3 s par dalle. À l'échelle nationale, il faut environ 550 000 dalles × 3 s ≈ 460 h CPU. Ce volume se parallélise **dalle par dalle** par la GPAO, sans travail dans cette épique. L'estimation est à recaler sur le nombre réel de dalles LiDAR HD.
- **Veille sur la branche Dev de R** : comparer Dev à la version figée (`8fcccd2`, D7) et signaler les changements dans `R/ffuelmetrics2.R` et `R/pad_metrics.R`. C'est une idée retenue pour plus tard ; en attendant, l'INRAE s'engage à prévenir l'IGN.
- **Version de lidar‑for‑fuel dans les rasters de sortie** : depuis la PR #43, elle est inscrite dans les GPKG. Il faudrait l'étendre aux rasters de métriques, utile si l'INRAE fait évoluer les formules pendant la production.

**Actions hors stories (Malvina)** :
- **Note de décision pour l'INRAE** (`note-inrae-reunion-mi-novembre.md`) : la compléter avec les chiffres de la story 3.8, passe 1, et la présenter à la réunion de mi‑novembre. Profiter de la réunion pour caler la date et la personne qui valide (Q8).
- **Remonter A1 à A11 à l'INRAE** sous forme d'issue sur `oliviermartin7/LidarForFuel`. A9 et A11 sont réglés par le merge de la PR #48 (12/11/2026) : il reste A1 à A8 et A10 à remonter.

## 11. Ordre de réalisation

```
              ┌─> 3.8 passe 1 (prototype + PAD v0.3.0) ──> note INRAE ──> [réunion mi‑nov.]
              │                                                                 │ D2–D5, Q6    
3.1 ─┬─> 3.2 ─> 3.3 ─> 3.4 ─┬─> 3.5* ─> 3.6 ─> 3.7 ─> 3.8 passe 2 ─> [mi‑déc. : code prêt]
     └──────────────────────┘    ▲                                        │
                                 └── réponses INRAE ──────────────────────┘
                                                              ──> [validation INRAE] ─> [prod. janv. 2027]
                                               3.9 (lff-compare) : après 3.7, reportable après janvier
```

- **\*** La story 3.5 attend les réponses de l'INRAE sur D2, D3 et D5. On peut faire 3.6 avant 3.5, puis ajouter AC-3.6.4 ensuite.
- La story **3.8, passe 1**, n'a **aucune dépendance**. Elle passe **en premier**, en parallèle de 3.1, puisqu'elle doit être prête avant la réunion de mi‑novembre.

| Jalon | Date | Contenu |
|---|---|---|
| J1 | avant la réunion de mi‑novembre 2026 | 3.8 passe 1 faite, tableau Q6 de la note rempli |
| J2 | réunion de mi‑novembre 2026 | décisions D2 à D5 et tolérances Q6 ; validateur et date de validation (Q8) |
| J3 | mi‑décembre 2026 | 3.1 à 3.7 mergées, 3.8 passe 2 faite |
| J4 | entre mi‑décembre 2026 et janvier 2027 | validation par l'INRAE (test de parité 3.4/3.5 et 3.8 passe 2) |
| J5 | janvier 2027 | lancement de la production nationale |

Pour lancer le développement : `bmad-build` sur chaque story, dans l'ordre ci‑dessus, avec cette épique en contexte.
