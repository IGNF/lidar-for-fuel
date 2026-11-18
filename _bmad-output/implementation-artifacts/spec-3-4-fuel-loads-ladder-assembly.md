---
title: 'Story 3.4 — Charges, combustible en échelle et assemblage des métriques (parité stricte avec R)'
type: 'feature'
created: '2026-11-18'
status: 'done'
route: 'oneshot'
review_loop_iteration: 0
context:
  - '{project-root}/docs/epic-3-fuel-metrics.md'
  - '{project-root}/_bmad-output/implementation-artifacts/epic-3-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem :** Le portage de `ffuelmetrics2` calcule le profil CBD (story 3.2) et les métriques de canopée (story 3.3), mais pas encore les charges, le combustible en échelle, le type de profil ni l'assemblage final. Sans eux, on ne peut pas produire pour un profil les 45 métriques de R, ni prouver la parité avec R sur le jeu de référence.

**Approach :** Créer en TDD `lidar_for_fuel/fuel_metrics/compute_ladder_metrics.py` (étape 8 du §4 de l'épique) et `lidar_for_fuel/fuel_metrics/calculate_fuel_metrics.py` (`fuel_metrics_core` : préparation 3.2, canopée 3.3, charges de l'étape 7, échelle, valeurs par défaut de l'étape 9 et assemblage dans l'ordre de R du §3.2). Aucune correction des défauts de R (story 3.5). Les critères d'acceptation sont AC-3.4.1 à AC-3.4.5 de l'épique.

</frozen-after-approval>

## Implementation Notes

**Décisions** (choix internes, invisibles dans le résultat pour l'utilisatrice) :

- **Branche** : `feat/ffuelmetrics2_fuel_loads`, créée depuis `feat/ffuelmetrics2` (`69ef8e9`), comme pour la 3.2 et la 3.3.
- **`compute_ladder_metrics(cbd, h_center, dz, cbh2, canopy_height_threshold, ladder_fuel_bd_thresholds, m_perrakis) -> dict[str, float]`**, calculé sur la CBD **recalculée** (étape 4, renvoyée par `compute_canopy_metrics`).
  - Une passe par seuil de `sorted(set(...))` donne 8 clés, dans l'ordre de R : `LadderBot_s`, `LadderTop_s`, `LadderLoad_s`, `SFLeq0_s`, `SFLeq_intot_s`, `ProfType_s`, `CFL_s`, `FSG_s`. Le suffixe est `f"{t:0.3f}".replace(".", "p")`.
  - Le bas de la zone de recherche est une constante nommée `LADDER_MIN_HEIGHT = 1.0` (`zval > 1` codé en dur en R, voulu par l'INRAE, D5 refusée), indépendante de CHT. La charge `K` part, elle, de CHT (`zlow`), comme R.
  - Seul le premier bloc continu de A compte (`while above[j]`).
  - Pour initialiser `ProfType`, on teste `CBH2 == CHT` avec `np.isclose` (piège 5).
  - Formules de Perrakis calculées en flottants numpy, pour donner `inf`/NaN comme R au lieu de `ZeroDivisionError`. Ce cas est inatteignable avec des strates régulières qui commencent au sol.
- **`fuel_metrics_core(pad, h_bottom, dz, fma_can, fma_shrub, smooth_pad, ladder_fuel_bd_thresholds, canopy_height_threshold, canopy_bd_thresh_frac, canopy_top_bd_thresholds, cbh_threshold_ratio, bd_max, m_perrakis) -> dict[str, float]`**.
  - Arguments explicites, nommés **comme la section `fuel_metrics` de la config** : la 3.7 pourra passer cette section telle quelle, et une clé inconnue lèvera une `TypeError`. C'est le point reporté de la 3.3.
  - Entrée : le PAD **brut** du pixel. Elle appelle `prepare_pad_profile`, puis calcule `h_center = h_bottom + dz/2` une seule fois.
  - Le dict de la 3.3 est relu par nom, et c'est `fuel_metrics_core` qui fixe l'ordre du contrat : `CLFL`, `FL_k1`, `FL_k2`, échelles, CTH, `CDH`, `CBH`, `CBH2`, `FSG`, `CBHlow`, `CanBD*`, `CBDcan`, `CBD_cbh*`, `LFL`, `CFL`, `ProfType`. L'ordre du dict de la 3.3 n'est pas modifié.
- **Charges (étape 7)** sur la CBD de l'étape 4 : `CLFL = Σ CBD[h > CHT]·dz`, `FL_k1 = CBD[0]·dz`, et `FL_k2 = CBD[1]·dz`, ou NaN s'il n'y a qu'une strate (le `NA` de R).
- **`h_bottom[0] ≠ 0`** (point reporté de la 3.2) : **aucun contrôle**, comme R. La docstring précise que `FL_k1` et `FL_k2` sont les deux premières strates, quelle que soit leur hauteur. Un éventuel contrôle relève de la lecture du raster (3.6).
- **Seuil par défaut (étape 9)** : `select_default_ladder_threshold` (PR #48, liste brute non triée), relu dans le dict des échelles.
- **`CBD_cbh`, `CBD_cbh2`, `CBD_cbhlow`** : NaN, comme R (A1). La correction se fera en 3.5 (D3).
- **Pixels entièrement NaN** : le profil est mis à zéro par la préparation, comme R (zéros, `ProfType = 1`, `CBH2 = CHT`). Le NaN viendra en 3.5 (D2).
- **Sommes** avec `np.sum` (float64). R accumule en `long double` : l'écart est d'au plus quelques ulp, invisible en float32 et dans la tolérance de l'AC-3.4.5.
- **Hors périmètre** : corrections D2 et D3 (3.5), boucle sur le raster, FMA en carte et export (3.6), validation de la config et avertissement d'équivalence (3.7).

**Fichiers prévus** :
- `lidar_for_fuel/fuel_metrics/compute_ladder_metrics.py` et `test/fuel_metrics/test_compute_ladder_metrics.py`. Profils écrits à la main, vérifiés par Rscript : types 1 et 2 sans échelle, type 3, type 4, échelle interrompue, formules `SFLeq0` / `SFLeq_intot` / `FSG` / `CFL`, seuils non triés ou en double, seuil de 1 m indépendant de CHT.
- `lidar_for_fuel/fuel_metrics/calculate_fuel_metrics.py` et `test/fuel_metrics/test_calculate_fuel_metrics.py`. Tests :
  - AC-3.4.1 : les 45 noms comparés aux `descriptions` du raster R, et 53 clés avec 4 seuils ;
  - charges ;
  - AC-3.4.4 ;
  - parité des 45 bandes sur `data/ref_inrae` (650 pixels, dont 25 entièrement NaN ; les 4 types sont présents pour chaque seuil).
- Paramètres de la parité : config INRAE (`FMA = 0.2543576`, `bd_max = 1`, CHT = 1, `frac = 0.3`, seuils d'échelle `[0.01, 0.02, 0.05]`, seuils CTH `[0.0012, 0.01]`, M = 2.5, pas de lissage) et ratio = 1/3 (A3).

**Code à réutiliser, sans le modifier** :
- `prepare_cbd_profile.py` : `prepare_pad_profile` (l. 151), `select_default_ladder_threshold` (l. 279) ;
- `compute_canopy_metrics.py` : `compute_canopy_metrics`, qui renvoie `(metrics, cbd_step4)`.

Source R : `/home/MDupays/Malvina/LidarForFuel/git/LidarForFuel/R/ffuelmetrics2.R`, lignes 228 à 303 (checkout `e72b8f5`, équivalent pour le calcul ; seule la ligne 283 diffère, et elle est déjà portée).

**Réalisé** :
- Les fichiers sont ceux prévus. Fonction publique en plus : `ladder_suffix(t)`, que `fuel_metrics_core` utilise pour retrouver les bandes du seuil par défaut, ce qui évite de dupliquer le format `%0.3f`.
- Tests : 9 pour l'échelle et 9 pour `fuel_metrics_core`. Le profil de bout en bout (`FMAcan = 1`, `FMAshrub = 0.5`) a été calculé par Rscript : ses 45 valeurs sont égales à 1e-12 près, avec les types 3, 3 et 2 selon le seuil. Les charges et l'échelle partent donc bien de la CBD de l'étape 4.

**Vérification** :
- Parité R (AC-3.4.5) : **0 écart** sur les 45 bandes × 650 pixels, y compris les 25 pixels entièrement NaN.
- Banc R hors dépôt, sur le fichier R de `8fcccd2` extrait par `git show` : **0 écart** à `rtol = 1e-12` sur 3 000 profils aléatoires (123 360 valeurs). Les profils mêlent `dz` 0,5 et 1, de 1 à 60 strates, FMA distinctes, CHT de 0,5, 1 et 2, de 1 à 4 seuils d'échelle (parfois non triés), des ratios 1/3, 0,2 et 0,333, `bd_max` de 0,5 et 1, et M de 2,5 et 3.
- Mutations : remplacer le seuil de 1 m par CHT, ou ne plus arrêter le bloc continu, fait échouer les tests écrits à la main. Remplacer `h <= CBH2` par `<` ne change rien, car un centre de strate ne peut pas égaler une base sur une grille régulière.
- `pytest test/fuel_metrics` : 75 réussis. Suite complète : 231 réussis, hors `test_download_dtm_native_grid` (réseau). black, isort (`--profile black`) et flake8 propres.

## Review Triage Log

Revue « Blind Hunter » (13 remarques). Après les corrections : 77 réussis dans `test/fuel_metrics`, flake8 propre.

| # | Remarque du relecteur | Verdict | Justification et suite donnée |
|---|---|---|---|
| 1 | Deux seuils d'échelle distincts de même suffixe `%0.3f` fusionnent (37 clés au lieu de 45) | low | Réel, mais il faut une config anormale pour le déclencher, comme pour les CTH de la 3.3. **Reporté** en story 3.7 (validation de la config). |
| 2 | La docstring du module annonce une liste de corrections absente | low | Réel. **Corrigé** : elle renvoie à la liste des défauts conservés dans `fuel_metrics_core` et à la story 3.5. A2 est un choix voulu (pas un défaut), déjà documenté avec `LADDER_MIN_HEIGHT`. A3 relève de l'appelant (ratio de la config). |
| 3 | Avec CHT < 1, une échelle continue est toujours de type 4, et ce n'est pas documenté | low | Réel, et c'est le comportement de R (A2). **Corrigé** : docstring de `compute_ladder_metrics`. Le banc R couvre CHT = 0,5. |
| 4, 5 | Les tests sans échelle et le test de `CLFL` n'utilisent que CHT = 1, égal à `LADDER_MIN_HEIGHT` | low | Réel. **Corrigé** : un test sans échelle avec CHT = 2, et un test de `CLFL` avec CHT = 2 (valeur R : 0,63). Les mutations « initialiser à 1 m » et « `CLFL` au‑dessus de 1 m » échouent désormais. |
| 6 | Le passage de la section `fuel_metrics` de la config n'est pas testé | low | Réel. Le relier au test de parité le ferait dépendre des valeurs par défaut de la config (D4 ouverte). **Reporté** en story 3.7 (point d'entrée). |
| 7 | Le test de parité ne vérifie pas que la dalle couvre les 4 types | low | Réel, et la correction est simple. **Corrigé** : assertion sur les `ProfType_*` de la référence (types 1 à 4 présents). |
| 8 | Le compte des valeurs de PAD corrigées est jeté | low | Réel. Le compte se fait plus simplement sur le raster. **Reporté** en story 3.6 (A6, AC-E.5). |
| 9 | Un profil vide ou des longueurs différentes donnent des erreurs peu claires | false | `read_pad_heights` refuse un raster sans bande PAD, et `h_bottom` vient des mêmes bandes que le PAD. L'état n'est pas atteignable, et l'erreur est de toute façon bruyante. |
| 10 | Une FMA NaN donne des métriques incohérentes | low | Déjà **reporté** en story 3.6 (Q9, report de la 3.3). |
| 11 | `np.isclose` diffère du `==` de R, sans être documenté | low | Rejeté. C'est le piège 5 de l'épique, et le commentaire le dit. CBH2 vaut exactement CHT (repli) ou un bas de strate : aucun cas ne sépare `==` de `isclose`. |
| 12 | Le banc R et les mutations ne sont pas dans le dépôt | low | Rejeté. Rscript ne tourne pas dans la CI, comme pour la 3.3. Le test de parité et le profil calculé par R restent dans le dépôt. |
| 13 | Statut du spec, CHANGELOG, « seuils distincts » dans la formule | low | Le statut est mis à jour à la finalisation. Le CHANGELOG n'a d'entrée ni pour la 3.2 ni pour la 3.3 : il revient à la 3.7 (documentation, AC-3.5.5). **Corrigé** : « distinct » dans la formule de `fuel_metrics_core`. |
