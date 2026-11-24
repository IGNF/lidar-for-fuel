---
title: 'Story 3.5 (révision) — Alignement sur R 00c654c et réponses INRAE du 23/11/2026'
type: 'feature'
created: '2026-11-23'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: 'b0f550de31f8ee6d9f46632ae9da000f07e06dbb'
context:
  - '{project-root}/docs/epic-3-fuel-metrics.md'
  - '{project-root}/_bmad-output/implementation-artifacts/epic-3-context.md'
  - '{project-root}/_bmad-output/implementation-artifacts/spec-3-5-empty-pixels-nan.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem :** L'INRAE a répondu le 23/11/2026 et publié R Dev `00c654c` (1.0.0.9014). `ffuelmetrics2Rast` transmet maintenant `CBHThresholdRatio` et `smooth_pad`, et `CBD_cbh`, `CBD_cbh2` et `CBD_cbhlow` sont calculées. Le Python garde un ratio de 1/3, des bandes `CBD_cbh*` à NaN et une référence R obsolète. Un pixel sans FMA n'est pas non plus traité.

**Approach :** Décisions validées par l'utilisatrice le 23/11/2026 :
- **D4 fermée** : `cbh_threshold_ratio` = 0.333 ;
- **D3 rouverte** : `CBD_cbh*` calculées comme R `00c654c` ;
- **D2 étendue** : un pixel sans FMA vaut NaN sur toutes les métriques. D2 reste une divergence tant que R ne l'applique pas, ce que l'INRAE a promis (Q1 bis) ;
- **Q6 bis** : une tolérance de 10⁻⁴ est confirmée ;
- **D7 → `00c654c`** : la référence R est régénérée en local, et l'ancienne est gardée ;
- **Q10** : le retrait du plafond du PAD à 5 se fera dans une branche séparée. Ici, on ne met à jour que la documentation.

## Boundaries & Constraints

**Always :**
- Parité stricte avec R `00c654c`, y compris dans ses choix :
  - `CBD_cbh*` est lue dans la **première** CBD, celle dont la FMA est coupée à CHT, et non après le recalcul selon CBH2 ;
  - on prend le premier indice de la strate dont le centre vaut `CBH + 0.5*dz` ;
  - `CBD_cbh2` n'est rempli que si CBH2 vient de 3A, et reste NaN quand CBH2 vient de CBHlow ou de CHT.
- Noms et ordre des 45 clés inchangés. Tolérance des tests de parité inchangée (`rtol=1e-5, atol=1e-6`, plus stricte que 10⁻⁴).
- Sans FMA, c'est-à-dire si `fma_can` **ou** `fma_shrub` vaut NaN : toutes les métriques sont NaN, comme pour un PAD entièrement NaN.

**Never :**
- Toucher à l'étape 2 (le plafond `raster_clip` d'`export_raster.py`) : c'est le travail de Q10.
- Modifier ou supprimer l'ancienne référence `512500_6854997_fuel_metrics.tif`.
- Écrire dans le dépôt R de l'utilisatrice, ou dans sa bibliothèque R : on installe `00c654c` dans le scratchpad.

## I/O & Edge-Case Matrix

| Scénario | Entrée | Résultat attendu |
|---|---|---|
| CBH trouvé en 3A | profil à houppier | `CBD_cbh` = CBD (1er calcul) de la strate de bas CBH |
| CBH2 trouvé en 3A | idem | `CBD_cbh2` = CBD (1er calcul) de la strate de bas CBH2 |
| CBH2 issu de CBHlow ou de CHT | pas de base en 3A | `CBD_cbh2` NaN ; `CBD_cbhlow` renseigné si CBHlow existe |
| FMA différentes | `fma_can` ≠ `fma_shrub`, CBH2 > CHT | `CBD_cbh*` utilisent la FMA coupée à CHT (et non à CBH2) |
| Pas de CTH | profil nul | les 3 bandes NaN |
| FMA NaN | `fma_can` ou `fma_shrub` = NaN, PAD valide | 45 métriques NaN, mêmes clés |

</frozen-after-approval>

## Code Map

- `lidar_for_fuel/fuel_metrics/compute_canopy_metrics.py` -- `compute_canopy_metrics` : `cbd` (1er calcul) est disponible entre l'étape 3 et le recalcul de l'étape 4. `upper[3]` est le CBH2 de 3A, avant les replis de 3C. On y ajoute les 3 clés `CBD_cbh*` au dict renvoyé. Ne pas toucher à `_crown_base`.
- `lidar_for_fuel/fuel_metrics/calculate_fuel_metrics.py` -- `fuel_metrics_core` : remplacer les 3 `np.nan` par les valeurs de `canopy` ; étendre `no_data` à une FMA NaN ; docstrings du module et de la fonction (A1 corrigé, A3 corrigé dans R, plafond à 5 en attente de Q10).
- `lidar_for_fuel/fuel_metrics/prepare_cbd_profile.py` -- `read_fma` renvoie déjà NaN hors carte ou sur NoData, et bloque la dalle hors de [0,05 ; 0,5] (Q9.2.ii confirmé). À ne pas modifier.
- `configs/config.yaml` -- `cbh_threshold_ratio: 0.333` et son commentaire.
- `test/fuel_metrics/test_compute_canopy_metrics.py`, `test/fuel_metrics/test_calculate_fuel_metrics.py` -- `_INRAE_PARAMS` (ratio 1/3), `_REF_FUEL_METRICS`, les attendus des profils écrits à la main, avec `CBD_cbh*` NaN à la ligne 81 (attendus calculés par R), et le test de parité avec le masque D2 (l. 344).
- `data/ref_inrae/pad_profile/512500_6854997_pad_1m.tif` -- entrée R : 26 × 25 pixels, 186 bandes, dont `PAD_1_*`.
- `data/ref_inrae/2026-09-28_fuel_metrics_zones-pilotes.yml` -- paramètres `ffuelmetrics2` de l'INRAE (ratio 0.333).
- R `00c654c` -- `~/Malvina/LidarForFuel/git/LidarForFuel` (déjà récupéré). Rscript 4.5.3 et terra 1.9.50 sont disponibles.

## Tasks & Acceptance

**Execution :**
- [x] `data/ref_inrae/fuel_metrics/make_fuel_metrics_00c654c.R` -- script de provenance : lire `PAD_1_*`, appeler `ffuelmetrics2Rast` avec les paramètres du yml, écrire `512500_6854997_fuel_metrics_00c654c.tif`. On l'exécute avec `00c654c` installé dans le scratchpad (`git archive` puis `R CMD INSTALL --library`). Il sert aussi à recalculer, avec R, les attendus des profils écrits à la main.
- [x] `test/fuel_metrics/*` -- TDD : parité sur la nouvelle référence avec un ratio de 0.333, `CBD_cbh*` comprises ; un garde-fou vérifie que la dalle contient des `CBD_cbh*` finies et des NaN ; cas de la matrice ; attendus R recalculés.
- [x] `compute_canopy_metrics.py`, `calculate_fuel_metrics.py` -- code et docstrings.
- [x] `configs/config.yaml` -- 0.333.
- [x] `docs/epic-3-fuel-metrics.md`, `epic-3-context.md` -- report des réponses : D2, D3, D4, D7, Q6 bis, Q8, Q9 et Q10. On met à jour A1 et A3, l'AC-3.5.2 (rétabli), les AC-3.7.1, AC-3.7.3, AC-E.1, AC-E.5 et AC-3.8.3 bis, ainsi que les §10 et §11. On note aussi la correction de `classify_ground` (#49), qui touche l'étape 1 et sera traitée en 3.8.
- [x] `CHANGELOG.md`, `deferred-work.md` -- entrées : Q10 (branche séparée) ; retrait du masque D2 quand R passera aux NaN.

**Acceptance Criteria :**
- Étant donné la référence `00c654c`, quand on lance la parité sur l'étape 3, alors les 45 bandes concordent sur tous les pixels avec donnée. Les pixels D2 sont exclus par un masque commenté.
- Étant donné `configs/config.yaml`, quand on l'ouvre, alors `cbh_threshold_ratio` vaut 0.333.

## Implementation Notes

**Décisions** (invisibles pour l'utilisatrice) :
- **Installation de R `00c654c`** : `git archive 00c654c` puis `R CMD INSTALL --library` dans le scratchpad, en 1.0.0.9014. Le dépôt R n'a été que `fetch`é et la bibliothèque R de l'utilisatrice (1.0.0.9011) n'a pas été touchée. Le script vérifie la version installée.
- **Comparaison exacte `h == bas + 0.5*dz`**, comme dans R. Elle est exacte pour dz = 1 et dz = 0,5. Avec un dz non dyadique, R et le Python donneraient tous deux NaN par erreur d'arrondi : on garde la parité. C'est l'entrée de `deferred-work.md` (3.3), qui proposait de passer par l'indice : elle est close.
- **FMA NaN** : on calcule sur un profil nul (PAD à 0, FMA à 1) puis on renvoie `dict.fromkeys(..., nan)`. Le calcul avec une FMA NaN aurait planté dans `_crown_base`, avec un `IndexError` sur `cbd == nan`. Pour un PAD entièrement NaN, le résultat est identique à avant, sauf le log debug d'A6, qui ne compte plus ces pixels.
- **Profils écrits à la main du test `calculate`** : le ratio y reste à 1/3, la valeur des calculs R existants. Avec 0.333, la strate de 2,5 m passerait en 3A. Seules les `CBD_cbh*` ont été recalculées par R `00c654c`. Les autres valeurs, recalculées elles aussi, sont identiques.
- **Ancienne référence** : elle est gardée et n'est plus utilisée par les tests. Comparée à la nouvelle, elle ne diffère que par l'effet du ratio et par les `CBD_cbh*` ; les charges diffèrent au plus de 1,2·10⁻⁷ en relatif (arrondi float32).

**Réalisé** :
- `data/ref_inrae/fuel_metrics/make_fuel_metrics_00c654c.R` et `512500_6854997_fuel_metrics_00c654c.tif` : 26 × 25 pixels, 45 bandes, même grille et mêmes noms que l'ancienne. La dalle a 449 `CBD_cbh` et `CBD_cbh2` finies, 274 `CBD_cbhlow` finies et 25 pixels vides (encore des zéros en R).
- `compute_canopy_metrics` : nouvelle fonction `_cbd_at_bottom` et 3 clés après `CBDcan`. `fuel_metrics_core` : les 3 valeurs, D2 étendu à la FMA, docstrings (A1 et A3 corrigés dans R ; le plafond à 5 attend Q10).
- `configs/config.yaml` : 0.333. Épique (D2, D3, D4, D7, Q6 bis, Q8, Q9, Q10, A1, A3, A4, §4, §10, §11, AC-E.1, E.2, E.5, E.7, E.8, 3.5.x, 3.6.4, 3.6.8, 3.7.1, 3.7.3, 3.8.3 bis), `epic-3-context.md`, `CHANGELOG.md` et `deferred-work.md` : 2 entrées closes, 3 ajoutées (Q10, masque D2, #49).

**Vérification** :
- TDD : 14 tests rouges avant le code, puis 94 verts dans `test/fuel_metrics`.
- Mutations, toutes détectées :
  - CBD prise après le recalcul selon CBH2 : 9 échecs ;
  - comparaison sur le bas au lieu du centre (bug A1) : 8 échecs ;
  - FMA NaN non traitée : 2 échecs.
- Suite complète : 250 tests réussis. black, isort (`--profile black`) et flake8 ne signalent rien sur les 4 fichiers Python.

## Spec Change Log

## Review Triage Log

Revue du 24/11/2026 : Blind Hunter (13 remarques), Edge Case Hunter (3), Verification Gap (aucun trou). Le scratchpad avait été vidé ; les relecteurs ont été relancés sur le diff régénéré.

| # | Source | Remarque | Verdict | Justification et suite donnée |
|---|---|---|---|---|
| 1 | BH | L'AC-3.4.5 de l'épique décrit encore la parité avec `CBD_cbh*` NaN et l'ancienne référence | low | **Corrigé** : note qui renvoie à l'AC-3.5.4 (référence `00c654c`). |
| 2 | BH | `epic-3-context.md` dit que `fuel_metrics.tif` a été « régénéré » alors qu'un nouveau fichier a été ajouté | low | **Corrigé** : les deux fichiers sont nommés, avec leur rôle. |
| 3 | BH | Le schéma de l'épique dit « sans plafond », alors que le plafond est toujours là (Q10 différée) | low | **Corrigé** : « plafonné à 5 (à retirer, Q10) ». |
| 4 | BH | Ligne A4 : faute (« et » pour « est ») et phrase contradictoire | low | **Corrigé**. Le reste d'A4 (FMA et `CanopyBDTreshFrac` par défaut) reste ouvert, la ligne de clôture est juste. |
| 5 | BH | Rien ne suit le comptage des pixels sans FMA exigé par l'AC-3.6.8 | false | C'est l'AC-3.6.8 elle-même qui le suit ; la 3.6 compte sur le raster (`np.isnan` de la FMA lue), comme pour le PAD (entrée existante de `deferred-work.md`). |
| 6 | BH | La spec est citée, mais elle n'est pas suivie par git et absente du diff | false | La spec est le fichier des affirmations, volontairement hors du diff ; elle sera commitée avec le changement. |
| 7 | BH | Rien ne prouve les écarts « ratio et `CBD_cbh*` seulement, 10⁻⁷ », ni les « 25 pixels vides » | false | Mesures faites pendant l'implémentation (notes ci-dessus) et refaites de façon indépendante par Edge Case Hunter ; les 25 pixels sont vérifiés par le test de parité (`no_data.any()`, CLFL = 0). Les 223 de A5 portent sur la zone pilote. |
| 8 | BH + ECH | Le script R ne fixe que la version, ne marche pas avec `source()` et ne contrôle pas la sortie | low | **Corrigé** : commentaire sur la limite du contrôle de version (pas de commit après `git archive`), `stop()` explicite hors Rscript, `stopifnot(nlyr == 45)`. Relancé : sortie identique au bit près. |
| 9 | BH | L'ancienne référence n'est plus utilisée : la supprimer ou la documenter | false | Garder l'ancienne est une décision de l'utilisatrice (23/11/2026) ; le rôle de chaque fichier est écrit dans l'épique (§5), dans le contexte et dans le script. |
| 10 | BH | Un pixel vide calcule toutes les métriques sur un profil nul | false | C'était déjà le cas avec D2 (revue 3.5, n° 10 : 0,27 ms par pixel, le coût de R) ; la FMA NaN ne fait qu'ajouter ces pixels. |
| 11 | BH | `_cbd_at_bottom` compare des flottants au lieu de passer par l'indice suggéré dans `deferred-work.md` | false | Choix voulu, pour la parité avec R (notes ci-dessus). Edge Case Hunter a vérifié l'aller-retour exact pour dz = 0,1, 0,2, 0,25, 0,3, 0,5 et 1. |
| 12 | BH | Le test FMA NaN ne combine pas les deux FMA, ni une FMA NaN avec un PAD NaN, ni le lissage | low | Rejeté : la condition est un `or`, ces cas n'ajoutent rien. Un PAD en partie NaN avec une FMA valide est déjà couvert (`one_finite_pad_value`). |
| 13 | BH | Les profils écrits à la main gardent 1/3 ; il n'y a aucun cas R sur la frontière 0.333 | false | 0.333 est validé par la parité sur la dalle (Verification Gap : 1/3 fait échouer 38 bandes), et la frontière `CTH × ratio` a déjà son test dédié. |
| 14 | ECH | FMA de la config inf, nulle ou négative : D2 ne s'applique pas | low | Différé (`deferred-work.md`, 3.7) : c'est le contrôle de la config, qui existait avant cette story. |
| 15 | ECH | FMA de la config NaN : dalle entièrement NaN sans log | low | Même cause que le n° 14, même entrée différée. |
| 16 | ECH | Script R lancé avec `source()` : chemin vide | low | Regroupé avec le n° 8, corrigé. |

## Verification

**Commands :**
- `conda run -n lidar_for_fuel python -m pytest test/fuel_metrics` -- tous verts
- `conda run -n lidar_for_fuel python -m pytest` -- suite complète verte
- `conda run -n lidar_for_fuel black --check`, `isort --profile black --check`, `flake8` sur les fichiers modifiés -- propres
