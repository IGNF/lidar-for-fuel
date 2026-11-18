---
title: 'Story 3.3 — Métriques de canopée (CTH, CBH, CBH2, CBHlow, CDH, CBDcan)'
type: 'feature'
created: '2026-11-17'
status: 'done'
route: 'oneshot'
review_loop_iteration: 0
context:
  - '{project-root}/docs/epic-3-fuel-metrics.md'
  - '{project-root}/_bmad-output/implementation-artifacts/epic-3-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem :** Le portage de `ffuelmetrics2` dispose de la couche d'entrée (story 3.2) mais pas encore des métriques de canopée. Il faut calculer, pour un profil vertical, la hauteur du sommet de canopée, les trois bases de houppier, la hauteur dominante et la densité moyenne de la canopée, exactement comme R. La story 3.4 s'appuie sur ces valeurs, en particulier sur CBH2 et sur la CBD recalculée.

**Approach :** Créer le module `lidar_for_fuel/fuel_metrics/compute_canopy_metrics.py`, développé en TDD dans `test/fuel_metrics/test_compute_canopy_metrics.py`. Il porte les étapes 2 à 6 du §4 de l'épique (sommet de canopée, base du houppier 3A/3B/3C, recalcul de la CBD selon CBH2, CDH, CBDcan). Le comportement doit être strictement celui de R (pièges §7). Les critères d'acceptation sont AC-3.3.1 à AC-3.3.4 de l'épique.

</frozen-after-approval>

## Implementation Notes

**Décisions** (choix internes, invisibles dans le résultat pour l'utilisatrice) :

- **Branche** : `feat/ffuelmetrics2_canopy_metrics`, créée depuis `feat/ffuelmetrics2`, comme pour la 3.2.
- **Fonction publique** `compute_canopy_metrics(pad, h_center, dz, fma_can, fma_shrub, bd_max, canopy_height_threshold, canopy_bd_thresh_frac, canopy_top_bd_thresholds, cbh_threshold_ratio) -> tuple[dict[str, float], np.ndarray]`.
  - Entrée : le PAD **préparé** par `prepare_pad_profile`. La fonction calcule elle-même la CBD de l'étape 1 (limite = CHT) et celle de l'étape 4 (limite = CBH2) avec `compute_cbd`, car l'étape 4 repart du PAD (piège 8).
  - Sortie : un dict indexé par les **noms de bandes R** (`CTH_0p0012`, `CTH_0p0100`, `CDH`, `CBH`, `CBH2`, `CBHlow`, `CanBDmax`, `CanBDmin`, `CanBDlowmax`, `CanBDlowmin`, `CBDcan`), pour que l'assemblage de la 3.4 soit une simple lecture, et la CBD recalculée.
  - Noms CTH : `f"CTH_{t:0.4f}".replace(".", "p")`, dans l'ordre de la config. `CTH` de référence = celui du **plus grand** seuil.
- **Portage littéral** des comparateurs de R, `CTH * ratio` multiplié, premier indice en cas d'égalité (`np.flatnonzero(...)[0]`), boucle CBH2 `for k in range(idmax, idmin, -1)` sur `cbd[k - 1] < S`.
- **CTH NaN** traité explicitement : 3A et 3B sont sautées (en Python, `max(1, nan)` vaut 1, contrairement à R). `CDH = min(CDH, CTH)` propage NaN.
- **CBDcan quand `haut` est NaN** : on reproduit `any(inCan)` de R au cas près. NaN s'il existe une strate `h ≥ CBH2` (`TRUE & NA`), 0 sinon (tout `FALSE`). Le second cas ne survient pas avec un profil de 60 m.
- **Moyenne de CBDcan** avec `np.mean` (float64). R calcule en `long double` : l'écart est d'au plus 1 ulp, sans effet sur la sortie float32 ni sur un seuil (CBDcan n'est comparée à rien ensuite). On n'ajoute donc pas de seconde moyenne « façon R ».
- **Hors périmètre** : `CBD_cbh*` (toujours NaN en R, assemblés par la 3.4, corrigés plus tard selon D3), charges et combustible en échelle (3.4), pixels vides (3.5), config (3.7).
- **Parité (AC-3.3.3)** sur `data/ref_inrae` (25 × 26 pixels) : bandes `PAD_*` du raster R, paramètres de la config INRAE (`FMA = 0.2543576`, `bd_max = 1`, `CHT = 1`, `frac = 0.3`, seuils CTH `[0.0012, 0.01]`, pas de lissage) mais **ratio = 1/3**, la valeur réellement utilisée par R en mode raster (A3).
- **Structure** : une fonction privée `_crown_base` sert à 3A et à 3B (seule différence : le test `min < max`). Elle calcule aussi un CBH2 en 3B, que l'appelant ignore, car R ne fait pas la descente vers une base continue dans la canopée basse.

**Fichiers créés** :
- `lidar_for_fuel/fuel_metrics/compute_canopy_metrics.py` ;
- `test/fuel_metrics/test_compute_canopy_metrics.py`, 19 tests après la revue : 6 profils écrits à la main (AC-3.3.1), AC-3.3.2, AC-3.3.4, recalcul de la CBD selon CBH2, noms CTH pilotés par la config, parité R, et un garde-fou qui vérifie que la dalle de référence contient des valeurs et des NaN pour `CBH`, `CBHlow`, `CanBDmin` et `CanBDlowmin`.

**Vérification** :
- Parité R : **0 écart** sur les 11 bandes × 650 pixels.
- Le test de parité est discriminant : avec ratio 0.333, 145 pixels diffèrent sur `CanBDlowmax` (et 7 sur CBH2) ; avec `frac = 0.1`, 288 pixels diffèrent sur CBH2.
- `pytest test/fuel_metrics` : 51 réussis avant la revue, 55 après. Suite complète : 206 réussis, hors `test_download_dtm_native_grid` (timeout réseau sur la Géoplateforme, sans lien avec la story).
- black, isort (`--profile black`, comme le pre-commit) et flake8 sont propres.

## Review Triage Log

Revue « Blind Hunter » (11 remarques).

| # | Remarque du relecteur | Verdict | Justification et suite donnée |
|---|---|---|---|
| 1 | Seuils CTH en double ou de même nom `%0.4f` : fusion silencieuse dans le dict, et CTH de référence = dernier au lieu du premier (R) | low | Réel, mais il faut une config anormale pour le déclencher. **Reporté** en story 3.7 (validation de la config), comme le point 1 de la 3.2. |
| 2 | Pas de contrôle des longueurs, des seuils vides, ni de l'ordre croissant de `h_center` | false | Longueurs différentes et liste vide lèvent déjà une erreur explicite (broadcast numpy, `max()` vide). L'ordre croissant est garanti par `read_pad_heights`. |
| 3 | `np.mean` au lieu de `mean(na.rm = TRUE)` : une FMA NaN (pixel hors carte) donne CBDcan NaN, alors que R donne une valeur | medium (non vérifié sur données réelles) | Réel en théorie, mais R n'est pas cohérent non plus avec une CBD NaN (`max()` sans `na.rm` en 3A/3B). La règle pour les pixels sans FMA est une question ouverte (Q9). **Reporté** en story 3.6. |
| 4 | La parité R ne teste pas l'étape 4 (FMA identiques sur la dalle) ni la CBD renvoyée | low | Réel. **Corrigé** : test avec des valeurs calculées par Rscript (`FMAcan = 0.3`, `FMAshrub = 0.15`), 11 métriques identiques à 1e-12 près, et CBD renvoyée vérifiée. |
| 5 | La tolérance `rtol=1e-5, atol=1e-6` est plus lâche que la parité mesurée | false | C'est la tolérance fixée par l'AC-3.3.3. L'écart mesuré (0) est noté dans la vérification. |
| 6 | Le test de parité dérive ses paramètres de `_HAND_PARAMS` | low | Réel, et la correction est simple. **Corrigé** : constante `_INRAE_PARAMS`, reliée à la config de référence. |
| 7 | Le plafond de CDH (CDH < CTH, CDH > CTH plafonné) n'est pas testé à la main | low | Réel. **Corrigé** : deux tests. |
| 8 | CBH trouvé sans CBH2 avec `frac = 0` : non documenté | low | Réel, et c'est le comportement de R. **Corrigé** : docstring de `_crown_base`. La validation de `frac` relève de la config (3.7). |
| 9 | La branche « CBDcan = 0 quand le sommet est NaN et qu'aucune strate n'atteint CBH2 » n'est pas testée | low | Réel. **Corrigé** : test sur un profil d'une strate. |
| 10 | Le test de l'AC-3.3.2 ne vérifie que 5 métriques sur 11 | low | Réel. **Corrigé** : comparaison des 11 métriques. |
| 11 | `top` recalcule `min(CDH, CTH)` ; la docstring du module parle des étapes 2 à 6 alors que l'étape 1 est faite | low | Réel. **Corrigé** : `top = cdh` et docstring « étapes 1 à 6 ». Le test de l'étape 1 avec des FMA différentes est couvert par le point 4. Le CBH2 calculé puis ignoré en 3B est un choix documenté (voir Structure). |

## Revue à trois relecteurs (17/11/2026, demande de l'utilisatrice)

Relecture de `compute_canopy_metrics` selon trois angles : fidélité à R, cas limites et calcul numérique, conception et contrat. Deux relecteurs ont comparé la fonction à R par Rscript sur 4 000 et 3 000 profils aléatoires (`dz` 0,5 et 1, CHT, `frac`, ratios et seuils variés, égalités dues au plafond `bd_max`). Résultat : **0 écart** sur 10 des 11 métriques ; CBDcan diffère de 1 ulp en float64 dans environ 15 % des cas. Temps mesuré : environ 0,1 ms par pixel.

| # | Remarque | Verdict | Suite donnée |
|---|---|---|---|
| 1 | CBH2 renvoyé en `int` quand CHT est un entier de la config | low | **Corrigé** : `cht = float(...)`. |
| 2 | Le commentaire de la garde « CTH NaN » est inexact : les masques sont de toute façon vides (`h <= nan`) | low | **Corrigé** : le commentaire est juste, la garde est gardée pour la lisibilité. |
| 3 | Le drapeau `require_min_below_max` n'a aucun effet en 3B (les strates sous le premier max y sont toujours < max) ; la vraie différence, la descente vers CBH2, est calculée puis jetée | low | **Corrigé** (choix (b) de l'utilisatrice) : un drapeau `upper_canopy` commande le test `min < max` et la descente ; CBH2 vaut NaN en 3B. Vérifié par les deux bancs R : 0 écart. |
| 4 | Dans `_crown_base`, `bd_max` a deux sens (plafond de config et max local) | low | **Corrigé** : renommé en `cbd_max` / `cbd_min`. |
| 5 | Docstring : « CTH NaN donne CBDcan NaN » trop absolu ; alias `top` inutile ; 3A ne dit pas que CBH2 suppose CBH | low | **Corrigé**. |
| 6 | Aucun test avec `dz = 0.5`, ni pour `frac = 0` | low | **Corrigé** : deux tests, avec des valeurs calculées par Rscript. |
| 7 | CBDcan : `np.mean` au lieu de la moyenne `long double` de R (1 ulp), et sans `na.rm` | low | **Gardé** (choix (a) de l'utilisatrice) en attendant la réponse de l'INRAE à Q6 bis ; l'écart est noté dans un commentaire du code. Le cas `na.rm` rejoint le report sur les FMA NaN (story 3.6). |
| 8 | Seuils CTH de même nom : le dict garde le dernier, R le premier | low | Déjà **reporté** en story 3.7. |
| 9 | `IndexError` dans `_crown_base` si une CBD NaN tombe dans la zone de recherche | low | Inatteignable aujourd'hui (PAD nettoyé ; une FMA NaN vide les zones). Rejoint le report FMA NaN de la story 3.6. |
| 10 | `CBD_cbh*` sont lues par R sur la CBD de l'**étape 1**, que la fonction ne renvoie pas | — | **Reporté** en 3.4/3.5 (voir deferred-work). |
| 11 | Pour la 3.4 : `**params` lèverait `TypeError`, `h_center` à calculer une fois, ordre des clés à décider | — | **Reporté** en 3.4 (voir deferred-work). |

**Tests** : 57 réussis dans `test/fuel_metrics`.
