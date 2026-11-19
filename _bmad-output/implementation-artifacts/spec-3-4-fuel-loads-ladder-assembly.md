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
| 11 | `np.isclose` diffère du `==` de R, sans être documenté | low | Rejeté. C'est le piège 5 de l'épique, et le commentaire le dit. CBH2 vaut exactement CHT (repli) ou un bas de strate : aucun cas ne sépare `==` de `isclose`. *Le 18/11, la revue de code a cru à tort que les deux comparaisons différaient pour les `dz` de 0,1 à 0,4. La deuxième revue (19/11) a confirmé ce rejet : elles sont équivalentes. Le code garde le `==` de R, choisi par l'utilisatrice.* |
| 12 | Le banc R et les mutations ne sont pas dans le dépôt | low | Rejeté. Rscript ne tourne pas dans la CI, comme pour la 3.3. Le test de parité et le profil calculé par R restent dans le dépôt. |
| 13 | Statut du spec, CHANGELOG, « seuils distincts » dans la formule | low | Le statut est mis à jour à la finalisation. Le CHANGELOG n'a d'entrée ni pour la 3.2 ni pour la 3.3 : il revient à la 3.7 (documentation, AC-3.5.5). **Corrigé** : « distinct » dans la formule de `fuel_metrics_core`. |

### Review Findings

Revue de code du 18/11/2026 : `feat/ffuelmetrics2...feat/ffuelmetrics2_fuel_loads` (`8b82ee0`, `43f9bdb`), par 4 relecteurs (Blind Hunter, Edge Case Hunter, Verification Gap, Acceptance Auditor), avec l'épique en contexte. 24 remarques.

Corrections appliquées le 19/11/2026. Les valeurs attendues des nouveaux tests ont été calculées par Rscript (8fcccd2), et elles sont identiques au portage à `rtol = 1e-12` sur toutes les bandes. Les 6 mutations visées font chacune échouer un test : `>=` sur CHT dans `CLFL` et dans `K`, `smooth_pad` non transmis, `bd_max` non transmis, et seuil par défaut cherché dans `sorted(set())` (`isclose` au lieu de `==` est en fait un mutant équivalent, voir la deuxième revue). Résultats : `pytest test/fuel_metrics` 82 réussis, suite complète 238 réussis, black, isort et flake8 propres sur les fichiers modifiés.

- [x] [Review][Patch] Initialiser `ProfType` avec `CBH2 == CHT`, comme R, au lieu de `np.isclose`. Décision de l'utilisatrice du 18/11/2026 : écriture de R (D1). L'écart supposé pour les `dz` de 0,1 à 0,4 n'existe pas (voir la deuxième revue) : les deux sont équivalents [lidar_for_fuel/fuel_metrics/compute_ladder_metrics.py:94]
- [x] [Review][Patch] Le cas « centre de strate = CHT » n'est pas testé : `>=` au lieu de `>` dans `CLFL` et dans `K` laisse les 77 tests verts [lidar_for_fuel/fuel_metrics/calculate_fuel_metrics.py:106, compute_ladder_metrics.py:113]
- [x] [Review][Patch] Le passage de `smooth_pad` n'est pas testé : `prepare_pad_profile(pad, False)` laisse tout vert [lidar_for_fuel/fuel_metrics/calculate_fuel_metrics.py:93]
- [x] [Review][Patch] Le passage de `bd_max` n'est pas testé : aucun profil n'atteint le plafond [lidar_for_fuel/fuel_metrics/calculate_fuel_metrics.py:97]
- [x] [Review][Patch] Rien ne vérifie que le seuil par défaut est cherché dans la liste brute (`sorted(set())` laisse tout vert), et le cas pair ne distingue pas 0p050 de 0p100 (blocs identiques) [test/fuel_metrics/test_calculate_fuel_metrics.py:168]
- [x] [Review][Patch] Arguments positionnels vers `compute_canopy_metrics` (10) et `compute_ladder_metrics` (7), dans un ordre différent de la signature appelante : passer par mots-clés [lidar_for_fuel/fuel_metrics/calculate_fuel_metrics.py:95-114]
- [x] [Review][Patch] Docstring de `fuel_metrics_core` : préciser la condition de parité avec le raster R (`smooth_pad = False`, ratio = 1/3, A3) et la mise à zéro silencieuse des valeurs invalides (A6) [lidar_for_fuel/fuel_metrics/calculate_fuel_metrics.py:57]
- [x] [Review][Patch] Docstring trop affirmative : « toujours de type 4 quand CHT < 1 m » n'est vrai que pour des bas de strate qui tombent sur 1 m (dz 0,5 ou 1) [lidar_for_fuel/fuel_metrics/compute_ladder_metrics.py:38]
- [x] [Review][Patch] Le test du pixel vide ne vérifie que 6 métriques sur 45 : comparer tout le dict à celui d'un profil nul [test/fuel_metrics/test_calculate_fuel_metrics.py:187]
- [x] [Review][Patch] Commentaire de `_HAND_R_METRICS` inexact (« one ladder of each kind » : seuls les types 3 et 2 y figurent) [test/fuel_metrics/test_calculate_fuel_metrics.py:37]
- [x] [Review][Patch] Le cas `no_stratum_above_0p01`, déplacé dans `_HAND_CASES`, n'a pas de commentaire et a quitté sa section [test/fuel_metrics/test_compute_canopy_metrics.py]
- [x] [Review][Defer] L'épique écrit « `LadderLoad` entre 1 m et LadderTop » et « `CLFL` au‑dessus de 1 m », alors que R et le code partent de CHT [docs/epic-3-fuel-metrics.md:§3.2, story 3.4] — deferred: la correction touche l'épique, qui revient à l'utilisatrice
- [x] [Review][Defer] « La 3.7 pourra passer la section `fuel_metrics` telle quelle » est faux quand une FMA est une carte (D6) : `fuel_metrics_core` attend une valeur par pixel [lidar_for_fuel/fuel_metrics/calculate_fuel_metrics.py:23-24] — deferred: le contrat relève des stories 3.6 et 3.7

**Rejetées** :
- `false` : la refonte de `test_compute_canopy_metrics.py` sort du spec. Elle a été demandée par l'utilisatrice, dans un commit séparé, et garde les mêmes cas.
- `false` : l'AC-3.4.4 n'est vérifié directement que sur un profil. La parité couvre les 650 pixels, puisque `LFL` ≡ `LadderLoad_0p020` en R et que les deux bandes sont comparées.
- `false` : un profil vide donne une `IndexError`. L'échec est bruyant sur un état inatteignable (`read_pad_heights` exige des bandes PAD).
- `false` : `deferred-work.md` oublierait les listes de seuils vides ou NaN. L'entrée de la 3.2 couvre déjà « seuils non vides, finis et positifs » (story 3.7).
- `low` : PAD et `h_bottom` de longueurs différentes passent sans erreur (le `stopifnot` de R n'est pas porté). Les deux viennent des mêmes bandes du raster, et la garde ne protégerait qu'un usage futur (placette).
- `low` : `compute_ladder_metrics` avec `cbh2` NaN donne `ProfType = 2`. CBH2 n'est jamais NaN, ni en R (repli sur CHT) ni via `fuel_metrics_core`.
- `low` : seuils d'échelle NaN, et test des noms sans la config. Déjà reportés en 3.7 (validation de la config, passage de la section).
- rejet (correction du spec en revue) : les comptes de tests du spec sont périmés (9/9 et 75 au lieu de 10/10 et 77), ainsi que `review_loop_iteration`, la liste des fichiers et les numéros de lignes R (235 à 303 ; ligne 286).

#### Deuxième revue (19/11/2026)

Revue des corrections non commitées (diff `HEAD` = `43f9bdb`, `lidar_for_fuel/` et `test/`), par les 4 mêmes relecteurs, avec l'épique en contexte. 12 remarques. Verification Gap : aucun trou.

Corrections appliquées le 19/11/2026. Les nouvelles valeurs attendues viennent de Rscript (8fcccd2) ou, pour le cas d'échelle à 1 m, des formules de l'étape 8 comme les autres cas écrits à la main. Les mutations visées font chacune échouer un test : `h >= 1 m`, `CFL` ou `FSG` par défaut pris sur un autre seuil, et valeurs négatives conservées. Résultats : `pytest test/fuel_metrics` 83 réussis, black, isort et flake8 propres sur les fichiers modifiés.

- [x] [Review][Patch] `==` et `np.isclose` sont équivalents pour `ProfType` : un CBH2 trouvé vaut `h[k] − dz/2` avec `k > i_min` et `h[i_min] > CHT`, donc CBH2 ≥ CHT + dz/2 (CBHlow aussi) ; sinon il vaut CHT exactement. Vérifié sur 40 000 profils aléatoires (dz 0,1 à 1, grilles décalées) : écart minimal dz/2, aucun cas où `isclose` diffère de `==`. Le constat de la première revue (« faux pour les dz de 0,1 à 0,4 ») est donc erroné. Corriger le commentaire (`==` garde l'écriture de R, sans effet sur la parité), supprimer le test sur une entrée inatteignable, et corriger l'entrée de `deferred-work.md` : le piège 5 de l'épique est juste [lidar_for_fuel/fuel_metrics/compute_ladder_metrics.py:98-100, test/fuel_metrics/test_compute_ladder_metrics.py:166]
- [x] [Review][Patch] Le paragraphe A3 de `fuel_metrics_core` dit « whatever the configuration » sans dire que c'est à l'appelant de passer ces valeurs, et omet la condition `5 × min(fma_can, fma_shrub) ≥ bd_max` avec le PAD Python plafonné à 5 (épique §3.1) [lidar_for_fuel/fuel_metrics/calculate_fuel_metrics.py:63-66]
- [x] [Review][Patch] La puce A6 dit « without any warning », alors que `prepare_pad_profile` journalise en debug et que le compte est jeté ; aucun test de `fuel_metrics_core` ne vérifie qu'un profil partiellement NaN, inf ou négatif donne les métriques du profil mis à 0 [lidar_for_fuel/fuel_metrics/calculate_fuel_metrics.py:59-60]
- [x] [Review][Patch] La borne `h > LADDER_MIN_HEIGHT` (1 m) n'est pas testée à sa limite : aucun centre de strate ne tombe sur 1 m, et `>=` laisse tout vert [lidar_for_fuel/fuel_metrics/compute_ladder_metrics.py:102]
- [x] [Review][Patch] Commentaire du cas `even_number_of_thresholds` inexact : « the 4 ladders differ », alors que 0p010 et 0p020 ont la même charge (0,265) ; seul 0p050 se distingue des trois autres [test/fuel_metrics/test_calculate_fuel_metrics.py:200]
- [x] [Review][Patch] Le cas `exact_tie_keeps_the_order_given` cite « R: LFL = 0.9 » sans le vérifier : ajouter la valeur R de `LFL` à chaque cas [test/fuel_metrics/test_calculate_fuel_metrics.py:207]
- [x] [Review][Patch] Le seuil par défaut n'est distingué des autres que par `LFL` : `CFL` et `FSG` pourraient venir d'un autre seuil sans être vus ; étendre le contrôle d'unicité à ces deux bandes (`ProfType` ne peut pas l'être : 3 pour 0p500 et 0p750) [test/fuel_metrics/test_calculate_fuel_metrics.py:219]
- [x] [Review][Patch] Le test `smooth_pad` / `bd_max` ne précise pas l'appel R qui a produit ses valeurs (`zval = 0:7`, `dz = 1`, FMA 1 et 0,5, frac 0,3, ratio 1/3) [test/fuel_metrics/test_calculate_fuel_metrics.py:190]
- [x] [Review][Patch] Le test CHT = 0,5 n'affirme ni `CBH2` (5) ni `LadderBot_0p020` (1 > CHT), sur lesquels repose son commentaire « suspended » [test/fuel_metrics/test_calculate_fuel_metrics.py:160]

**Rejetées** :
- rejet (correction du spec en revue) : la note des décisions (ligne 32) parle encore de `np.isclose`.
- `false` (en partie) : comparer les 45 bandes dans le test `smooth_pad` / `bd_max`. Ces deux paramètres n'entrent que dans `prepare_pad_profile` et `compute_canopy_metrics` ; une bande modifiée suffit à prouver la transmission, et la parité couvre le reste. Seule la documentation de l'appel R est retenue (ci-dessus).
- `false` (en partie) : avec le cas pair, un mauvais choix entre 0,01 et 0,02 passerait inaperçu. Le test exige 0p050, dont la charge (0,225) diffère de celle des trois autres : tout autre choix échoue. Seul le commentaire est retenu (ci-dessus).

#### Troisième revue (19/11/2026)

Revue du cumul des deux tours de corrections (diff `HEAD` = `43f9bdb`, `lidar_for_fuel/` et `test/`), par les 4 mêmes relecteurs, avec l'épique en contexte. 21 remarques, 20 après regroupement. Verification Gap : aucun trou. Aucune violation d'AC.

Corrections appliquées le 19/11/2026. Les valeurs des nouveaux cas (ratio 0,8, frac 0,8, M = 3, seuils répétés) ont été calculées par Rscript (8fcccd2) : identiques au portage à `rtol = 1e-12` sur toutes les bandes. Les mutations visées (ratio, frac ou M codés en dur dans `fuel_metrics_core`, doublons retirés avant la médiane) font chacune échouer un test. Résultats : `pytest test/fuel_metrics` 87 réussis, suite complète 243 réussis, black, isort et flake8 propres sur les fichiers modifiés.

- [x] [Review][Patch] `ProfType` = 1 suppose que `cbh2` est exactement la valeur de CHT renvoyée par `compute_canopy_metrics` en repli ; un appel direct avec un CBH2 en float32 ou recalculé (`0.1 + 0.2`) donne 2. Documenter cette précondition sur l'argument `cbh2` [lidar_for_fuel/fuel_metrics/compute_ladder_metrics.py:51]
- [x] [Review][Patch] Le commentaire de `==` oublie le repli sur CBHlow (lui aussi strictement au-dessus de CHT + dz/2), dit « at least » au lieu de « strictement », et sa phrase sur la strate est obscure [lidar_for_fuel/fuel_metrics/compute_ladder_metrics.py:98-101]
- [x] [Review][Patch] Docstring : « `ProfType` = 1 if CBH2 = CHT (no crown base found) » est faux avec `frac = 0`, où CBH est trouvé mais CBH2 se replie sur CHT ; écrire « CBH2 fell back on CHT (neither CBH2 nor CBHlow found) » [lidar_for_fuel/fuel_metrics/compute_ladder_metrics.py:43-44]
- [x] [Review][Patch] Docstring A3 : « only if `5 × min(fma) >= bd_max` » présente comme nécessaire une condition suffisante (un profil sans PAD au-dessus de 5 donne le même résultat que R quelle que soit la FMA) [lidar_for_fuel/fuel_metrics/calculate_fuel_metrics.py:68-69]
- [x] [Review][Patch] Docstring A3 : « the PAD of step 2 » se confond avec l'étape 2 de la même docstring (les charges) ; nommer l'étape `pad_profile` [lidar_for_fuel/fuel_metrics/calculate_fuel_metrics.py:67]
- [x] [Review][Patch] `cbh_threshold_ratio`, `canopy_bd_thresh_frac` et `m_perrakis` ne sont jamais testés via `fuel_metrics_core` avec une autre valeur que celle des fixtures : une valeur codée en dur passerait. Ajouter un cas R par paramètre au test « passed on » (ratio 0,8 → CBHlow 5 ; frac 0,8 → CBH2 6 ; M 3 → SFLeq0_0p020 0,09375) [test/fuel_metrics/test_calculate_fuel_metrics.py:180]
- [x] [Review][Patch] Le seuil par défaut avec des seuils répétés n'est pas testé : la médiane de R porte sur la liste brute, doublons compris (`[0.01, 0.01, 0.02, 0.05]` → 0,01, vérifié avec Rscript et en Python) ; un dédoublonnage avant la médiane passerait [test/fuel_metrics/test_prepare_cbd_profile.py:250]
- [x] [Review][Patch] Le commentaire de `r_lfl` dit « same call as _HAND_R_METRICS », alors que deux cas utilisent un autre profil : préciser que `PADval` est le `pad` du cas [test/fuel_metrics/test_calculate_fuel_metrics.py:245]
- [x] [Review][Patch] Le contrôle d'unicité filtre par `suffix not in name` : un suffixe contenu dans un autre (`0p010` dans `10p010`) exclurait à tort une bande ; comparer au nom exact [test/fuel_metrics/test_calculate_fuel_metrics.py:250]
- [x] [Review][Patch] Le test CHT = 0,5 dit « Values computed by R » sans l'appel R, et n'affirme pas `LadderTop_0p020`, `CFL_0p020` ni `FSG_0p020` (R : 2, 0,63, 3,5) [test/fuel_metrics/test_calculate_fuel_metrics.py:163]
- [x] [Review][Patch] L'entrée « piège 5 » de `deferred-work.md` se contredit (« préciser » puis « rien n'est à corriger ») ; or le piège dit l'égalité exacte « tant que `dz` vaut 0,5 ou 1 », ce qui est trop restrictif : elle l'est quel que soit `dz` [_bmad-output/implementation-artifacts/deferred-work.md:49-50]
- [x] [Review][Patch] L'en-tête du module de tests d'échelle annonce des strates de 1 m et CHT = 1 m, alors que des cas utilisent dz 0,5 et 2 et CHT 2 [test/fuel_metrics/test_compute_ladder_metrics.py:8]

**Rejetées** :
- déjà reporté : suffixes de seuils d'échelle en collision (`0.0101` et `0.0104` → `0p010`). L'entrée de `deferred-work.md` (story 3.7) existe depuis la première revue.
- déjà reporté : le log A6 est en debug, alors que l'épique demande de signaler les corrections. Le compte et le log visible relèvent de la story 3.6 (entrée de `deferred-work.md`).
- rejet (correction du spec en revue) : la note des décisions (ligne 32) parle encore de `np.isclose`.
- `false` : un profil PAD vide donne une `IndexError`. Déjà rejeté à la première revue : l'état n'est pas atteignable (`read_pad_heights` exige des bandes PAD), et l'échec est bruyant.
- `false` : avec une valeur par défaut NaN, `nan not in others` passerait toujours. Aucun des trois cas n'a de `LFL`, `CFL` ou `FSG` NaN, et `LFL` est en plus comparé à la valeur R.
- `false` : la condition `5 × min(fma) >= bd_max` ne serait suivie nulle part. L'épique (§3.1) confie à la story 3.7 sa vérification au démarrage, avec un avertissement.
- `false` : le test des valeurs invalides ne couvre pas le lissage. `test_prepare_pad_profile_smoothing_after_cleaning` vérifie déjà l'ordre nettoyage puis lissage, et `test_prepare_pad_profile_replaces_invalid_values_by_zero` vérifie le log (`caplog`), `-inf` compris.
- `low` : rendre les arguments de `compute_canopy_metrics` et `compute_ladder_metrics` obligatoirement nommés (`*`). Les appels de `fuel_metrics_core` passent déjà par mots-clés, et le changement toucherait les signatures et tous les appelants des tests, sans défaut démontré.
