---
title: 'Story 3.6 — Application au raster PAD et export GeoTIFF'
type: 'feature'
created: '2026-11-24'
status: 'in-progress'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: '6eb5c094f03a8a0462e885bfa0f7cd07bb4b4c1a'
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-3-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem :** `fuel_metrics_core` calcule les métriques d'un seul profil. Rien ne lit encore une dalle `{stem}_pad_profile_1m.tif` pour écrire la carte `{stem}_fuel_metrics.tif`, et rien n'appelle `cap_pad` (AC-3.2.8). La production nationale a besoin de cette carte par dalle.

**Approach :** un module `export_fuel_metrics.py` en deux couches :
- une fonction sur tableaux, qui reçoit les bandes PAD, leurs noms et les grilles de FMA, plafonne le PAD (`cap_pad`), applique `fuel_metrics_core` pixel par pixel et journalise ce qui a été corrigé ;
- une fonction sur fichier, qui lit le raster PAD et la FMA (valeur ou carte), appelle la première et écrit le GeoTIFF par `_write_geotiff`.

Critères de l'épique : AC-3.6.1 à AC-3.6.9 (`docs/epic-3-fuel-metrics.md`, story 3.6).

**Décision (24/11/2026), AC-3.6.7 sol nu : on suit R.** Il ne doit y avoir ni exception ni `RuntimeWarning`, et les charges, l'échelle et `ProfType` doivent être finies. Les métriques de houppier (`CBH`, `CBHlow`, `CanBD*`, `CBD_cbh*`, et `CTH_*`, `CDH`, `CBDcan` pour un profil nul) restent NaN, comme dans R. L'AC-3.6.7 est reformulée dans l'épique. Pas de nouvelle divergence.

## Boundaries & Constraints

**Always :**
- Même grille en sortie qu'en entrée (transform, CRS, taille). Les bandes sont en float32, avec nodata NaN, et portent les noms et l'ordre de `fuel_metrics_core`.
- Les bandes PAD sont trouvées par leur nom. `cap_pad` est appelé juste après la lecture.
- Le calcul ne s'interrompt jamais sur une dalle atypique. En revanche, une erreur d'entrée (une bande non `PAD_*`, aucune bande) lève une `ValueError` qui donne le nom du fichier.
- Le log de fin de dalle est en info et compte :
  - les pixels entièrement NaN ;
  - les pixels sans FMA ;
  - les valeurs ramenées dans [0, 5], en distinguant les valeurs > 5 finies, les +inf et les négatifs (report de la revue de `cap_pad`) ;
  - les profils partiellement NaN mis à 0 (A6) ;
  - les pixels traités.

**Never :**
- Ne pas modifier `fuel_metrics_core` ni sa signature : les tests de parité de l'étape 3 seule (AC-E.1) l'appellent.
- Ne pas écrire de nouvel outil d'écriture GeoTIFF.
- Pas de point d'entrée Hydra ni de lecture de config : c'est la story 3.7.
- Pas de vectorisation ni de parallélisation.

## I/O & Edge-Case Matrix

| Scénario | Entrée | Résultat attendu | Erreurs |
|---|---|---|---|
| Dalle de référence | PAD R (bandes `PAD_1_*`), FMA 0,2543576 | égale à `fuel_metrics_00c654c.tif` hors pixels D2 | — |
| Pixel sans donnée | PAD tout NaN, ou FMA NaN | NaN sur toutes les bandes (D2) | — |
| Dalle vide | tous les pixels NaN | raster de même grille, tout NaN ; `logger.warning` | pas d'exception |
| Sol nu | aucune strate au-dessus de CHT | pas d'exception ni de `RuntimeWarning` ; charges, échelle et `ProfType` finies ; houppier NaN comme R | — |
| PAD hors [0, 5] | 7, +inf, −1 | plafonné ; comptes distincts dans le log | — |
| Bandes invalides | une bande non `PAD_*`, ou aucune | — | `ValueError` avec le chemin du fichier |

</frozen-after-approval>

## Code Map

- `lidar_for_fuel/fuel_metrics/calculate_fuel_metrics.py` -- `fuel_metrics_core(pad, h_bottom, dz, fma_can, fma_shrub, smooth_pad, ladder_fuel_bd_thresholds, canopy_height_threshold, canopy_bd_thresh_frac, canopy_top_bd_thresholds, cbh_threshold_ratio, bd_max, m_perrakis) -> dict[str, float]`. Les clés, dans l'ordre, sont les noms des bandes. Elle gère déjà D2 (PAD tout NaN ou FMA NaN). Elle ignore le compte A6 de `prepare_pad_profile` : on le compte dans l'appelant (profil ni tout NaN ni sans NaN). À ne pas modifier.
- `lidar_for_fuel/fuel_metrics/prepare_cbd_profile.py` :
  - `read_pad_heights(names)` lève `ValueError` sur un nom non PAD ou sur une liste vide : on intercepte l'erreur pour y ajouter le chemin du fichier ;
  - `cap_pad(pad, names)` renvoie `(capped, n)` ;
  - `read_fma(fma, transform, shape, crs)` renvoie une grille float64 et accepte une valeur ou un chemin de carte.
- `lidar_for_fuel/pad_profile/export_raster.py` -- `_write_geotiff(path, bands, transform, srid)` : à réutiliser sur place, sans le déplacer. On élargit seulement l'annotation `srid` à `str | CRS`, pour passer `src.crs` : le CRS de la référence R est un `LOCAL_CS` sans EPSG.
- `test/fuel_metrics/test_calculate_fuel_metrics.py` -- `_INRAE_PARAMS`, `_REF_PAD_RASTER_1M`, `_REF_FUEL_METRICS`, et le masque D2 de `test_fuel_metrics_parity_with_r`, à reprendre. Les deux rasters de référence font 25 × 26 pixels, transform (10, 512489.75, 6854996.75), avec 25 pixels D2. Le PAD R a 186 bandes, dont 60 `PAD_1_*` (max 2,04).
- Environnement : lancer pytest, black, isort et flake8 dans l'env conda `lidar_for_fuel` (`conda run -n lidar_for_fuel …`), jamais par `uv run`, qui laisse `uv.lock` et `.venv` dans le dépôt. Ne pas commiter.
- Performance mesurée : 0,48 ms par pixel avec `fuel_metrics_core`, soit environ 4,8 s pour 100 × 100 pixels (AC-3.6.6).

## Tasks & Acceptance

**Execution :**
- [x] `test/fuel_metrics/test_export_fuel_metrics.py` -- TDD :
  - parité fichier : écrire les `PAD_1_*` de la référence dans `tmp_path`, exporter, puis comparer la grille, les noms, le float32 et les valeurs à la référence avec le masque D2 (AC-3.6.1, AC-3.6.3) ;
  - cas de la matrice sur de petits tableaux synthétiques ;
  - texte exact du log de fin de dalle ;
  - carte FMA avec un pixel NoData.
- [x] `lidar_for_fuel/fuel_metrics/export_fuel_metrics.py` -- `compute_fuel_metrics_raster` (tableaux) et `export_fuel_metrics(pad_path, output_dir, fma_can, fma_shrub, …) -> Path`, qui écrit `{stem}_fuel_metrics.tif` (le suffixe `_pad_profile_1m` est retiré du nom d'entrée). Paramètres explicites, docstrings Google.
- [x] `lidar_for_fuel/pad_profile/export_raster.py` -- annotation `srid: str | CRS`.
- [x] `docs/epic-3-fuel-metrics.md`, `CHANGELOG.md`, `deferred-work.md` -- AC-3.6.8 (comptes distincts), AC-3.6.7 reformulée (on suit R), entrée CHANGELOG, clôture de l'entrée « Deferred from: code review of spec-3-2-pad-cap-on-read ».

**Acceptance Criteria :**
- Étant donné un raster PAD de 100 × 100 pixels, quand on lance le calcul, alors il prend moins de 10 s (mesure indicative, sans test bloquant).
- Étant donné la suite complète, quand on la lance, alors elle est verte, et black, isort (`--profile black`) et flake8 ne signalent rien sur les fichiers touchés.

### Review Findings

Revue du 25/11/2026 (2ᵉ passage, PR #62 : `6eb5c09..0a9d881`) : Blind Hunter (BH), Edge Case Hunter (ECH), Verification Gap (VG), Acceptance Auditor (AA).

- [ ] [Review][Patch] (medium, VG) Aucun test ne détecte une inversion de `fma_can` et `fma_shrub` dans le chemin raster ou fichier : tous les tests d'export passent la même FMA aux deux. Ajouter un test avec `fma_can ≠ fma_shrub` (ex. 0,3 / 0,1), comparé à `fuel_metrics_core` [lidar_for_fuel/fuel_metrics/export_fuel_metrics.py:116]
- [ ] [Review][Patch] (low, AA+BH) Test sol nu, cas « surface fuel only » : l'ensemble exact des bandes NaN n'est pas vérifié (seulement `⊆ _CROWN`). Il doit valoir CBH, CBHlow, CanBD*, CBD_cbh*. La boucle `for name in ["CLFL", …]` est redondante avec la ligne précédente [test/fuel_metrics/test_export_fuel_metrics.py:257]
- [ ] [Review][Patch] (low, BH) Une erreur de `read_fma` (carte non alignée, valeurs hors [0,05 ; 0,5] sur la fenêtre de la dalle, CRS) nomme la carte FMA nationale mais pas la dalle PAD. La préfixer par `PAD raster {path}:` comme pour les bandes, et compléter `Raises:` (valeurs hors plage) [lidar_for_fuel/fuel_metrics/export_fuel_metrics.py:221]
- [ ] [Review][Patch] (low, BH) L'id de test « no PAD band » est trompeur : `[""]` passe par la branche « Band '' is not a PAD band », pas par « no band » (inatteignable par un GeoTIFF). Renommer l'id (ex. « unnamed band ») [test/fuel_metrics/test_export_fuel_metrics.py:331]
- [ ] [Review][Patch] (low, BH) `deferred-work.md` : l'entrée « valider la FMA donnée par valeur » est rangée sous le titre de la revue de spec-3-2-pad-cap-on-read et double l'entrée existante « FMA > 0 et finies (story 3.7) ». La fusionner avec cette dernière [_bmad-output/implementation-artifacts/deferred-work.md:81]
- [ ] [Review][Patch] (low, BH) Annotation `band_names: Sequence[str]`, alors que `export_fuel_metrics` passe `src.descriptions`, qui peut contenir `None`. Aligner sur `read_pad_heights` (`Sequence[str | None]`) [lidar_for_fuel/fuel_metrics/export_fuel_metrics.py:36]
- [x] [Review][Defer] (low, VG) `pad_max` n'est jamais passé avec une valeur autre que 5 : un oubli de transmission resterait invisible [lidar_for_fuel/fuel_metrics/export_fuel_metrics.py:231] — deferred: `pad_max` n'est pas encore exposé dans la config ; le test vient avec la story 3.7
- [x] [Review][Defer] (low, ECH) Une carte FMA contenant ±inf passe le contrôle de plage de `read_fma` (filtré par `isfinite`) et donne des CBD fausses, sans log [lidar_for_fuel/fuel_metrics/prepare_cbd_profile.py:197] — deferred: pré-existant (`read_fma`, story 3.2) ; à traiter avec la validation des FMA en 3.7
- [x] [Review][Defer] (low, ECH) FMA donnée par valeur inf, nulle ou négative, acceptée sans erreur [lidar_for_fuel/fuel_metrics/export_fuel_metrics.py:221] — deferred: pré-existant et déjà reporté (« FMA > 0 et finies », story 3.7)
- [x] [Review][Defer] (low, AA+BH) Le bilan de fin de dalle et l'avertissement « no pixel with data » ne nomment pas la dalle [lidar_for_fuel/fuel_metrics/export_fuel_metrics.py:137] — deferred: le point d'entrée 3.7 journalise la dalle avant le calcul, comme `main_pad_profile` (« Export 8 rasters for tile : %s »)
- [x] [Review][Defer] (low, AA) `epic-3-context.md` décrit des bandes PAD « mêlées à d'autres bandes », alors que le raster Python ne contient que des `PAD_*` et que l'AC-3.6.9 rejette toute autre bande [_bmad-output/implementation-artifacts/epic-3-context.md:36] — deferred: fichier de contexte agent, à corriger à la main ou à régénérer

**Rejected :**
- (ECH) `compute_fuel_metrics_raster` appelé avec un tableau masqué perd le masque : low, seul `export_fuel_metrics` l'appelle et il remplit le masque par NaN ; ce serait un garde pour un état non démontré.
- (ECH) Le test de parité ne pourrait pas détecter un plafonnement (et le n° 12 du triage précédent se réfute à tort) : low. Avec 5 × 0,254 ≥ `bd_max`, le plafond ne change pas les métriques, et le PAD de la dalle reste ≤ 2,04 ; la correction modifierait la spec.
- (AA) Le log ne donne pas le total des valeurs plafonnées : false. L'AC-3.6.8 réécrite demande les trois comptes distincts, dont le total est la somme ; `cap_pad` journalise déjà le total.
- (AA) Dalle vide et sol nu testés seulement sur tableaux : low. Le chemin fichier n'a aucune branche propre à ces cas et il est couvert par le test de la dalle de référence.
- (AA) Pixel hors de la carte FMA non testé par le chemin fichier : false. C'est une transmission à `read_fma`, déjà testée (`test_read_fma_map_nodata_and_outside_the_map_are_nan`).
- (AA) Performance mesurée sur la fonction tableaux seulement : low. L'AC-3.6.6 est indicative, et la lecture de 60 bandes de 50 × 50 pixels est négligeable.
- (BH) Dalle à une seule strate → `FL_k2` NaN : low, inatteignable en production (`nlayers: 60` dans `configs/config.yaml`), et documenté dans `fuel_metrics_core`.
- (BH) « A6 » désigne deux choses, et la divergence +inf manque dans le CHANGELOG : false. Le CHANGELOG (« cap PAD ») la documente déjà comme divergence assumée ; la renommer modifierait l'épique.
- (BH) « processed » compte tous les pixels : false. C'est la définition de la spec (Implementation Notes).
- (BH) `review_loop_iteration` et le Spec Change Log ne reflètent pas la revue précédente : la correction modifierait la spec sous revue.

## Implementation Notes

- Log counts are disjoint: « without FMA » counts only the pixels with a PAD (not all NaN) but `fma_can` or `fma_shrub` NaN; « partly NaN » (A6) counts only the pixels that are computed (not D2); « processed » = all the pixels of the tile. The `logger.warning` fires when no pixel is computed (all-NaN PAD or no FMA everywhere).
- PAD counts (> `pad_max` finite, +inf, negative incl. −inf) are taken before `cap_pad`; `cap_pad` keeps its own per-band info log.
- `export_fuel_metrics` checks the band names (`read_pad_heights`) before reading the data, and re-raises the `ValueError` prefixed with `PAD raster {path}:`. `compute_fuel_metrics_raster` parses the names again (cheap) to get `dz` / `h_bottom`.
- `pad_max` (default `PAD_MAX`) is a parameter of both functions, passed on to `cap_pad`; its config exposure stays in 3.7.
- Measured: 4.25 s for 100 × 100 pixels (ref tile repeated, `compute_fuel_metrics_raster`).

## Spec Change Log

## Review Triage Log

Revue du 25/11/2026 : Blind Hunter (BH), Edge Case Hunter (ECH), Verification Gap (VG).

| # | Source | Remarque | Verdict | Preuve et suite |
|---|---|---|---|---|
| 1 | VG, BH | Avertissement « no pixel with data » jamais testé pour une dalle avec PAD mais sans FMA nulle part | medium | Pré-vérifié : remplacer la condition par `all_nan.all()` ne fait échouer aucun test. Or c'est le cas d'une dalle hors de la carte FMA. **Patch** : test dédié. |
| 2 | VG | Comptes du log qui ne doivent pas se recouper (PAD tout NaN et sans FMA ; partiellement NaN et sans FMA) jamais testés | low | Pré-vérifié : retirer `~all_nan` ou `~no_fma` ne fait échouer aucun test. **Patch** : test des comptes exacts. |
| 3 | BH, ECH | `np.broadcast_to` accepte une FMA (cols,) ou (1, cols), alors que la docstring promet une `ValueError` | low | Vérifié : une FMA de forme (3,) passe sur une grille 2 × 3. **Patch** : exiger la forme exacte (rows, cols). |
| 4 | BH | AC-3.6.8 réécrite en une phrase trop longue | low | **Patch** : liste des trois comptes, date en fin de puce. |
| 5 | ECH | FMA donnée par valeur sans contrôle de plage (≤ 0 accepté) | low | Pré-existant (`read_fma`, « no check here ») et pas causé par la 3.6. **Defer** vers la 3.7 (validation de la config). |
| 6 | BH | Deux logs info par dalle (celui de `cap_pad` et le bilan) | low | Rejeté : le log bande par bande est le comportement voulu de l'AC-3.2.8, et l'AC-3.6.8 n'interdit pas une ligne de plus ; le supprimer obligerait à ajouter un paramètre à `cap_pad`. |
| 7 | BH | Le compte de `cap_pad` est ignoré et peut dériver des trois comptes | false | Mêmes masques (`< PAD_MIN`, `> pad_max`) : finies > max + +inf + négatives = compte de `cap_pad`, exactement. |
| 8 | BH, ECH | Grille vide (0 ligne ou 0 colonne) → `UnboundLocalError` | low | Rejeté : inatteignable, rasterio n'ouvre pas de raster de taille nulle ; l'échec est bruyant. |
| 9 | ECH | PAD non 3-D → erreur de déballage peu claire | low | Rejeté : inatteignable, `src.read()` renvoie toujours un tableau 3-D. |
| 10 | ECH | `pad_max` = +inf ou < 0 : log faux ou PAD faux | low | Rejeté : `pad_max` n'est pas exposé ; sa validation est déjà reportée en 3.7 (`deferred-work.md`). |
| 11 | BH | `_write_geotiff` reste privé dans `pad_profile` ; `srid` devrait se nommer `crs` | false | Choix de la spec (Code Map) et permis par l'AC-3.6.5 (« éventuellement ») ; aucun dommage nommé. |
| 12 | BH | Le test de parité suppose sans le vérifier que le PAD de référence reste ≤ 5 | false | Si un plafond changeait une valeur, la comparaison avec R échouerait bruyamment : elle ne peut pas passer à tort. |
| 13 | BH | Tests manquants : `pad_max`, `smooth_pad=True`, nom sans suffixe, carte FMA mal alignée, écrasement | low | Rejeté : simples transmissions à des fonctions déjà testées (`fuel_metrics_core`, `cap_pad`, `read_fma`) ; le cas sans FMA est le n° 1. |
| 14 | BH | Performance non vérifiée sur le vrai chemin | false | Mesurée à l'implémentation : 4,25 s pour 100 × 100 par `compute_fuel_metrics_raster` ; l'AC-3.6.6 exclut un test bloquant. |
| 15 | BH | Même carte FMA lue deux fois | low | Rejeté : une fenêtre de 50 × 50 pixels, coût négligeable ; l'éviter ajoute une branche. |
| 16 | BH, ECH | Un nodata ≠ NaN dans le PAD (ex. 0) masquerait des zéros réels | low | Rejeté : le contrat d'entrée (§3.1) est float32 avec nodata NaN, ce que l'étape 2 écrit toujours. |
| 17 | BH | PAD sans CRS accepté en silence | low | Rejeté : l'étape 2 écrit toujours un CRS ; il faudrait ajouter un garde. |
| 18 | BH | Étape non branchée (pas de `main_fuel_metrics`, pas de `pad_max` en config) | false | Exclu par l'intention (« Pas de point d'entrée Hydra ni de lecture de config : c'est la story 3.7 »). |
| 19 | BH | CHANGELOG « above 5 » codé en dur | false | `pad_max` n'est pas réglable par l'utilisateur : 5 est exact. |
| 20 | BH | La spec n'est pas dans le diff | false | Le diff de revue a été construit sans elle ; elle sera commitée avec le code. |

## Verification

**Commands :**
- `conda run -n lidar_for_fuel python -m pytest -q test/fuel_metrics` -- attendu : tout est vert.
- `conda run -n lidar_for_fuel python -m pytest -q` -- attendu : suite complète verte.
