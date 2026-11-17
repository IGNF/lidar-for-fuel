---
title: 'Story 3.2 — Lecture des strates et préparation du profil CBD'
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

**Problem :** L'étape 3 (portage de `ffuelmetrics2`) a besoin d'une couche d'entrée. Elle doit lire les bandes `PAD_{dz}_{h_bas}` d'un raster PAD (où elles sont mêlées à d'autres bandes) et transformer un profil PAD vertical en profil CBD, comme le fait l'étape 1 de l'algorithme R. Elle doit aussi choisir le seuil d'échelle par défaut selon la règle de la PR #48. Les stories 3.3 et 3.4 s'appuient sur cette couche.

**Approach :** Créer le module `lidar_for_fuel/fuel_metrics/prepare_cbd_profile.py` avec quatre fonctions pures, développées en TDD dans `test/fuel_metrics/test_prepare_cbd_profile.py` :
1. lecture des strates PAD à partir des noms de bandes ;
2. nettoyage et lissage optionnel du profil PAD ;
3. calcul de la CBD avec une hauteur limite en paramètre ;
4. choix du seuil d'échelle par défaut.

Le comportement doit être strictement celui de R (épique 3, §4 étape 1, pièges §7). Les critères d'acceptation sont AC-3.2.1 à AC-3.2.6 de l'épique.

</frozen-after-approval>

## Implementation Notes

**Décisions** (choix internes, invisibles dans le résultat pour l'utilisatrice) :

- **`parse_pad_strata(band_names)`**
  - Renvoie `(indices, dz, h_bottom)` : les indices (0‑based) des bandes `PAD_*` et les `h_bottom` triés numériquement, avec `dz` scalaire.
  - Regex stricte `^PAD_(nombre)_(nombre)$` : `Cover_2`, `N_1_3` et `Ni_1_3` sont ignorées.
  - Lève une `ValueError` si les bandes PAD ont des `dz` différents (AC-3.2.2), et aussi si aucune bande `PAD_*` n'est trouvée. Le message avec le nom de fichier (AC-3.6.9) sera ajouté par la story 3.6, qui connaît le fichier.
  - Écart volontaire avec R : `parse_pad_heights` ne trie pas et ne vérifie pas `dz`. C'est l'épique qui demande ce tri et cette vérification.
- **`prepare_pad_profile(pad, smooth_pad)`**
  - Profil 1‑D. Renvoie `(pad_propre, n_corrigées)`.
  - Les valeurs non finies ou négatives passent à 0 **avant** le lissage, comme en R.
  - Le nombre de corrections est journalisé en debug, et renvoyé pour que la story 3.6 l'agrège (AC-3.6.8).
  - Le lissage reproduit `slider::slide_dbl(.before=1,.after=1)` : `pd.Series.rolling(3, center=True, min_periods=1).mean()`.
  - L'entrée n'est pas modifiée.
- **`compute_cbd(pad, h_center, height_limit, fma_can, fma_shrub, bd_max)`**
  - Calcule `min(PAD × FMA, bd_max)`, avec `fma_can` si `h_center > height_limit` (comparaison stricte, comme `zval > CanopyHeightThreshold` en R) et `fma_shrub` sinon.
  - `h_center = h_bottom + dz/2`, calculé par l'appelant.
- **`select_default_ladder_threshold(thresholds)`**
  - Calcule `t[np.argmin(|t − median(t)|)]` sur la liste de la config **non triée**.
  - Lève une `ValueError` si la liste est vide.
- **Hors périmètre** : pas de mise à NaN des pixels vides (D2, story 3.5), et pas de config ni de CHANGELOG (story 3.7).
- **Conventions** : pas de `__init__.py`, comme les autres paquets ; docstrings Google ; hauteurs nommées `h_*`.

**Surprise : l'exemple 2 de l'AC-3.2.6 de l'épique est faux.** L'épique attend `[0.01, 0.02, 0.05, 0.1]` → 0.02. En double précision, ce n'est pas une égalité : |0.05 − 0.035| = 0.01499999… < |0.02 − 0.035| = 0.0150000…03.
- Vérifié avec `Rscript` : R renvoie **0.05** pour cette liste comme pour `[0.05, 0.02, 0.01, 0.1]`.
- numpy donne le même résultat. Le code suit R (parité, D1/Q6) et le test attend 0.05.
- La règle « premier dans l'ordre de la config en cas d'égalité » est testée sur une vraie égalité, avec des valeurs exactes en binaire : `[0.25, 0.5, 0.75, 1.0]` → 0.5 et `[0.75, 0.5, 0.25, 1.0]` → 0.75. R donne les mêmes résultats.
- Il faudra corriger l'exemple de l'AC-3.2.6 et de l'étape 9 du §4 dans l'épique (modification de spec, laissée à l'utilisatrice).

**Fichiers créés** :
- `lidar_for_fuel/fuel_metrics/prepare_cbd_profile.py` ;
- `test/fuel_metrics/test_prepare_cbd_profile.py`, 20 tests, dont un sur les noms de bandes réels du raster R `data/ref_inrae/pad_profile/512500_6854997_pad_1m.tif`.

**Vérification** :
- `pytest test/fuel_metrics` : 20 tests réussis ;
- suite complète : 175 réussis et 1 échec, `test_download_dtm_native_grid` (timeout réseau sur la Géoplateforme, sans lien avec la story) ;
- black, isort et flake8 sont propres.

**Après la revue** (voir le journal ci-dessous) :
- `parse_pad_strata` refuse maintenant un `dz` nul, les doublons et les trous.
- Le lissage n'utilise plus pandas. Il reproduit `mean()` de R au bit près (accumulation en `long double` et passe de correction). Vérifié contre `slider::slide_dbl` : 0 écart sur 192 valeurs, alors que l'implémentation précédente avec `pd.rolling` différait d'1 ulp sur environ 1 valeur sur 4.
- Résultat : 25 tests réussis.

## Review Triage Log

| # | Remarque du relecteur | Verdict | Justification et suite donnée |
|---|---|---|---|
| 1 | Aucune validation des paramètres (FMA, `bd_max`, seuils NaN) | low | Réel, mais ces valeurs viennent de la config, que la story 3.7 lit et contrôle. **Reporté** dans deferred-work. |
| 2 | `parse_pad_strata` accepte doublons, trous et `dz = 0` | medium | Réel : l'entrée est un raster externe (R ou Python, AC-3.6.2), et un profil irrégulier fausserait en silence les étapes 3.3 et 3.4. **Corrigé** : contrôle de contiguïté et de `dz > 0`, avec tests. Le contrôle « départ à 0 m » est **reporté** à la story 3.4 (il ne faut pas diverger de R si `z0 ≠ 0`). |
| 3 | La docstring de `select_default_ladder_threshold` laisse croire qu'un nombre pair de seuils implique une égalité | low | Réel, et la correction est simple. **Corrigé**. |
| 4 | L'épique contredit le test de l'AC-3.2.6 | medium | Réel, mais c'est la spec qui est fausse, pas le code (R renvoie 0.05). **Reporté**, car la modification de l'épique revient à l'utilisatrice. |
| 5 | Le `rolling` de pandas n'est pas exactement la moyenne de chaque fenêtre | medium | Réel : écart d'1 ulp, aussi par rapport à R. **Corrigé** : on reproduit `mean()` de R en `long double`, sans aucun écart sur 192 valeurs. Un test est ajouté avec des valeurs calculées par R. |
| 6 | pandas, dépendance non déclarée | low | Résolu par le point 5 : pandas n'est plus importé. |
| 7 | Fonctions 1‑D non vérifiées ; `height_limit` typé en scalaire | false | Le contrat est un profil par appel : `fuel_metrics_core(pad, h_bottom, dz)` en 3.4, avec une boucle par pixel acceptée en 3.6. Ajouter des contrôles de forme reviendrait à garder des chemins qui ne sont jamais atteints. |
| 8 | `compute_cbd` ne vérifie pas que `pad` et `h_center` ont la même longueur | low | Avec des longueurs 1‑D différentes, numpy lève déjà une erreur de broadcast. Rejeté. |
| 9 | Un log debug par profil inonde les logs | false | Ce log est exigé par l'AC-3.2.3, et il n'est émis qu'au niveau debug. Rejeté. |
| 10 | Le test du log est trop faible (`"5" in caplog.text`) | low | Réel, et la correction est simple. **Corrigé** : on vérifie le niveau et le message exact. |
| 11 | Tests manquants : raster R à 0,5 m, bandes `None`, doublons | low | **Corrigé** : test paramétré sur les deux rasters R, `None` mêlé aux bandes PAD, doublons, trous et `dz` nul. Le test en 2‑D est rejeté (voir le point 7). La frontière 1,5 m est déjà couverte par `test_compute_cbd_height_limit_is_strict`. |
| 12 | La docstring du module ne parle que de l'étape 1 | low | **Corrigé** : elle mentionne maintenant les étapes 1, 4 et 9. |

## Changement après livraison (17/11/2026, demande de l'utilisatrice)

- **Décision.** `parse_pad_strata` est remplacée par `read_pad_heights(band_names) -> (dz, h_bottom)`.
- **Pourquoi.** L'entrée de l'étape 3 est toujours `{stem}_pad_profile_1m.tif`. L'étape 2 Python écrit 8 rasters distincts, et celui-ci ne contient que les bandes `PAD_1_*`, déjà triées. Le filtrage des bandes mêlées et le renvoi d'indices, hérités de `parse_pad_heights` en R, n'ont donc pas d'utilité.
- **Ce que fait la nouvelle fonction.**
  - Elle exige que toutes les bandes soient des `PAD_{dz}_{h}` et lève une `ValueError` sinon.
  - Elle garde l'ordre des bandes, sans retri.
  - Elle vérifie que `dz` est unique et non nul, et que les strates sont contiguës et croissantes.
- **Rasters R.** Le raster R (186 bandes) n'est plus accepté directement : les tests de parité en extraient les bandes `PAD_*`.
- **Épique amendée.** AC-3.2.1, AC-3.2.2, le périmètre de la story 3.2, AC-3.6.2 et AC-3.6.9.
- **Tests.** 27 tests réussis.
