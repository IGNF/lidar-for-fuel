# Epic

> Épique détaillée (contrat d'entrée et de sortie, algorithme, constats A1 à A11, décisions D1 à D7, critères d'acceptation de chaque story) : [`docs/epic-3-fuel-metrics.md`](https://github.com/IGNF/lidar-for-fuel/blob/feat/ffuelmetrics2/docs/epic-3-fuel-metrics.md).

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

Les décisions encore ouvertes (D2 à D5 : pixels sans donnée, bandes `CBD_cbh*`, `cbh_threshold_ratio`, seuil de 1 m codé en dur) sont dans l'épique détaillée, au §6. Elles sont reprises dans la note pour l'INRAE.

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

Une nouvelle section `fuel_metrics` de la config contient les FMA de canopée et d'arbuste, les seuils d'échelle et de sommet de canopée, `canopy_height_threshold`, `cbh_threshold_ratio`, `bd_max` et `m_perrakis`. Le détail des valeurs est dans l'épique détaillée, au §3.1.

#### Donnée disponible mais non utilisée dans cette épique : PAD de la strate basse

`{stem}_pad_sb_0.5m.tif` (4 canaux `PAD_0.5_0` à `PAD_0.5_1.5`, pas de 10 m, résolution verticale 0,5 m, plafonné à 5) est produit par l'étape 2. R ne l'utilise pas dans `ffuelmetrics2`. Il ne servira qu'à la future charge de surface SFL corrigée par quantile mapping, **hors périmètre** (épique détaillée, §10).

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

Avec la configuration par défaut, les 3 seuils d'échelle × 8 métriques donnent les canaux 4 à 27. Les noms suivent le format R : `0.01` → `0p010` pour les seuils d'échelle, `0.01` → `0p0100` pour les seuils CTH. Le seuil par défaut est celui de la liste le plus proche de la médiane (PR #48). L'ordre exact et les conventions de hauteur (bas, centre ou haut de strate) sont dans l'épique détaillée, au §3.2.

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

Le détail de chaque story (description, effort, critères d'acceptation) est dans l'épique détaillée, au §9. Le périmètre reste ouvert : des stories pourront être ajoutées ou retirées selon les réponses de l'INRAE et les retours de validation.

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
