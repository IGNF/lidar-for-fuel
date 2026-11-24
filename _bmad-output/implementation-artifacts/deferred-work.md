# Travail reporté

- source_spec: `_bmad-output/implementation-artifacts/spec-3-2-prepare-cbd-profile.md`
  summary: Corriger l'exemple `[0.01, 0.02, 0.05, 0.1] → 0.02` de l'épique 3 (AC-3.2.6 et §4 étape 9) en `→ 0.05`.
  evidence: Vérifié avec Rscript, `which.min(abs(s - median(s)))` renvoie 0.05 pour les deux ordres de liste, car il n'y a pas d'égalité exacte en double précision. Le code et les tests suivent R. Comme l'épique est une spec, sa correction revient à l'utilisatrice.
- source_spec: `_bmad-output/implementation-artifacts/spec-3-2-prepare-cbd-profile.md`
  summary: Valider les paramètres scientifiques de la section `fuel_metrics` de la config au démarrage : FMA > 0 et finies, `bd_max` > 0, seuils non vides, finis et positifs (story 3.7).
  evidence: Le périmètre de la story 3.2 mentionne une « validation des paramètres ». Pourtant, `select_default_ladder_threshold([nan, …])` renvoie le premier seuil sans erreur, et `compute_cbd` accepte des FMA négatives. Ces valeurs viennent de la config, que la story 3.7 lit et contrôle déjà (avertissement d'équivalence AC-3.7.3).
- source_spec: `_bmad-output/implementation-artifacts/spec-3-2-prepare-cbd-profile.md`
  summary: Décider en story 3.4 quoi faire si le premier `h_bottom` n'est pas 0, puisque `FL_k1` et `FL_k2` supposent que la première strate commence au sol.
  evidence: `parse_pad_strata` vérifie que le profil est régulier (même `dz`, ni doublon ni trou), mais pas son point de départ. R calcule `FL_k1 = CBD[1]·dz` sans le vérifier. Imposer 0 dès la story 3.2 risquerait de diverger de R avec un `z0` non nul.
- source_spec: `_bmad-output/implementation-artifacts/spec-3-2-prepare-cbd-profile.md`
  summary: FAIT le 17/11/2026 : l'exemple de l'épique est corrigé (AC-3.2.6 et §4 étape 9), il donne maintenant `[0.01, 0.02, 0.05, 0.1]` → 0.05, et deux cas d'égalité exacte sont ajoutés.
  evidence: Il remplace le premier élément de cette liste.
- source_spec: `_bmad-output/implementation-artifacts/spec-3-2-prepare-cbd-profile.md`
  summary: Story 3.7 (AC-3.7.3, AC-E.7) — revoir la condition d'équivalence avec R `5 × min(fma_can, fma_shrub) ≥ bd_max` quand une FMA est la carte : elle doit porter sur le minimum de la carte, ou être évaluée par pixel.
  evidence: La carte FMA va de 0,05 à 0,5 kg/m². Pour un pixel où FMA < 0,2, on a 5 × FMA < 1 = `bd_max`. Le PAD Python plafonné à 5 donne alors une CBD plus faible que le PAD R non plafonné (maximum observé 7,01). Il y a donc un écart avec R sur ces pixels si PAD > 5, en contradiction avec Q6, qui demande un écart nul. Avec la valeur scalaire 0,254 de la config, la condition reste vraie.
- source_spec: `_bmad-output/implementation-artifacts/spec-3-3-canopy-metrics.md`
  summary: Story 3.7 — refuser les seuils `canopy_top_bd_thresholds` en double ou de même nom au format `%0.4f` (ex. 0.01 et 0.01004 → `CTH_0p0100`).
  evidence: `compute_canopy_metrics` range les CTH dans un dict ; deux seuils de même nom fusionnent (10 métriques au lieu de 11, nombre de bandes faux) et le CTH de référence prend la dernière valeur, alors que R prend la première. Ne survient qu'avec une config anormale : contrôle à faire à la lecture de la config.
- source_spec: `_bmad-output/implementation-artifacts/spec-3-3-canopy-metrics.md`
  summary: **Traité le 23/11/2026 (spec-3-5-r-00c654c-alignment : NaN sur toutes les métriques, réponse INRAE à Q9).** Story 3.6 — décider du traitement des pixels sans FMA (NaN renvoyé par `read_fma`) avant le calcul des métriques (Q9 posée à l'INRAE).
  evidence: Une FMA NaN rend la CBD NaN ; les métriques divergent alors de R (ex. `CBDcan` NaN avec `np.mean`, 0.2 avec `mean(na.rm = TRUE)` de R), et R lui-même n'est pas cohérent (`max()` sans `na.rm` en 3A/3B). Sévérité medium non vérifiée sur données réelles ; à trancher avec la réponse de l'INRAE à Q9 (probablement NaN sur toutes les bandes, comme D2).
- source_spec: `_bmad-output/implementation-artifacts/spec-3-3-canopy-metrics.md`
  summary: **Traité le 23/11/2026 (spec-3-5-r-00c654c-alignment, R Dev 00c654c).** ~~Sans objet depuis le 20/11/2026 (D3 : bandes gardées à NaN, comme R).~~ Story 3.4/3.5 (D3) — calculer `CBD_cbh`, `CBD_cbh2` et `CBD_cbhlow` dans `compute_canopy_metrics`, à partir de la CBD de l'étape 1 et des indices de strate (NaN en 3.4 pour la parité, corrigées sur place en 3.5).
  evidence: R lit ces bandes sur la CBD de l'étape 1, avant le recalcul selon CBH2 (`ffuelmetrics2.R` lignes 179, 186 et 204), et ne remplit `CBD_cbh2` que si CBH2 vient de 3A. La fonction ne renvoie que la CBD de l'étape 4 ; les deux diffèrent dès que `fma_can ≠ fma_shrub`. Passer par l'indice évite l'égalité de flottants `h_bottom == CBH`. À confirmer avec la correction R attendue (Q2 bis).
- source_spec: `_bmad-output/implementation-artifacts/spec-3-3-canopy-metrics.md`
  summary: Story 3.4 — donner à `fuel_metrics_core` des arguments explicites (et non `**params` transmis tel quel), calculer `h_center` une seule fois, et décider si l'ordre des clés du dict fait partie du contrat.
  evidence: `compute_canopy_metrics` refuse `ladder_fuel_bd_thresholds`, `m_perrakis` et `smooth_pad` (`TypeError`). R intercale `FSG` entre `CBH2` et `CBHlow`, alors que `_assert_metrics` fige aujourd'hui l'ordre du dict de la 3.3.
- source_spec: `_bmad-output/implementation-artifacts/spec-3-4-fuel-loads-ladder-assembly.md`
  summary: Story 3.7 — refuser les seuils `ladder_fuel_bd_thresholds` distincts mais de même nom au format `%0.3f` (ex. 0.0101 et 0.0102 → `0p010`).
  evidence: `compute_ladder_metrics` range les 8 métriques de chaque seuil dans un dict ; deux seuils de même suffixe fusionnent (37 clés au lieu de 45, sans erreur) et le seuil par défaut peut désigner le bloc écrasé. Même cas que les seuils CTH de la 3.3 ; ne survient qu'avec une config anormale, à contrôler à la lecture de la config.
- source_spec: `_bmad-output/implementation-artifacts/spec-3-4-fuel-loads-ladder-assembly.md`
  summary: Story 3.7 — tester que la section `fuel_metrics` de `configs/config.yaml` se passe telle quelle à `fuel_metrics_core` (`**cfg.fuel_metrics`).
  evidence: Les arguments de `fuel_metrics_core` portent les noms des clés de la config, ce qui est vérifié à la main. Pourtant, aucun test ne protège ce contrat : une clé renommée ne serait détectée qu'à l'exécution. Le test de parité garde ses propres paramètres, pour ne pas dépendre des valeurs par défaut de la config (D4 ouverte).
- source_spec: `_bmad-output/implementation-artifacts/spec-3-4-fuel-loads-ladder-assembly.md`
  summary: Story 3.6 — compter et journaliser par dalle les pixels entièrement NaN et les valeurs de PAD corrigées (A6, AC-E.5), car `fuel_metrics_core` ignore le compte renvoyé par `prepare_pad_profile`.
  evidence: `fuel_metrics_core` renvoie seulement le dict des métriques, ce qui garde un contrat simple pour l'assemblage. Le compte se fait plus simplement en un seul calcul sur le raster PAD (`np.isnan(pad).all(0)`, `~np.isfinite | < 0`), au moment où la 3.6 le lit.

## Deferred from: code review of spec-3-4-fuel-loads-ladder-assembly (2026-11-18)

- source_spec: `_bmad-output/implementation-artifacts/spec-3-4-fuel-loads-ladder-assembly.md`
  summary: Corriger dans l'épique 3 (§3.2, ligne de `LadderLoad`, et description de la story 3.4 pour `CLFL`) « entre 1 m et LadderTop » / « au‑dessus de 1 m » en « au‑dessus de CHT ».
  evidence: R (`zlow <- CanopyHeightThreshold`, `ffuelmetrics2.R` l. 110, 236, 264), le §4 (étapes 7 et 8) et le code partent de CHT. Le texte ne coïncide avec le code que si CHT = 1 m. Comme l'épique est une spec, sa correction revient à l'utilisatrice.
- source_spec: `_bmad-output/implementation-artifacts/spec-3-4-fuel-loads-ladder-assembly.md`
  summary: Stories 3.6/3.7 — `fuel_metrics_core` attend une FMA par pixel. Avec une carte FMA (D6), la section `fuel_metrics` ne peut pas lui être passée telle quelle : il faut d'abord la résoudre avec `read_fma`, puis passer la valeur du pixel.
  evidence: Le spec 3.4 (et l'entrée « tester que la section `fuel_metrics` se passe telle quelle ») suppose des FMA scalaires. Avec un chemin de carte, `compute_cbd` recevrait une chaîne. Le contrat de la boucle pixel (3.6) doit séparer les paramètres scalaires des FMA lues par pixel.
- source_spec: `_bmad-output/implementation-artifacts/spec-3-4-fuel-loads-ladder-assembly.md`
  summary: (Facultatif) Élargir le piège 5 de l'épique 3 (§7) : l'égalité `CBH2 == CHT` est exacte quel que soit `dz`, pas seulement « tant que `dz` vaut 0,5 ou 1 ». `np.isclose` et `==` sont donc équivalents ; le code garde le `==` de R.
  evidence: Revues de code du 18 et du 19/11/2026. CBH2 vaut CHT exactement en repli ; sinon c'est le CBH2 ou le CBHlow trouvé, strictement au-dessus de CHT + dz/2 (vérifié sur 40 000 profils aléatoires, dz de 0,1 à 1). La recommandation du piège (« aucun effet sur la parité ») reste juste ; seule la portée annoncée est trop restrictive. Le choix de `==` est une décision de l'utilisatrice du 18/11/2026 (écriture de R).

## Deferred from: spec-3-5-r-00c654c-alignment (2026-11-23)

- source_spec: `_bmad-output/implementation-artifacts/spec-3-5-r-00c654c-alignment.md`
  summary: Branche séparée (étape 2) — retirer le plafond du PAD à 5 (`raster_clip` de `lidar_for_fuel/pad_profile/export_raster.py`, bandes `pad_profile_1m` et `pad_sb_0.5m`) et ses tests, puis retirer l'avertissement d'équivalence (AC-3.7.3) et la condition `5 × min(FMA) ≥ bd_max` de l'épique et de la docstring de `fuel_metrics_core`.
  evidence: Réponse INRAE du 23/11/2026 à Q10 : « retire le plafond de python ». Le plafond éventuel sera un post-traitement des rasters PAD avant l'étape 3, à discuter en point technique. Décision de l'utilisatrice : branche séparée, car c'est l'étape 2.
- source_spec: `_bmad-output/implementation-artifacts/spec-3-5-r-00c654c-alignment.md`
  summary: Quand R mettra les pixels sans donnée à NaN (Q1 bis), régénérer la référence avec `make_fuel_metrics_*.R` sur cette version, retirer le masque D2 du test de parité et ne plus lister D2 comme divergence (docstring, CHANGELOG, épique).
  evidence: L'INRAE a répondu « OUI » le 23/11/2026, mais R Dev 00c654c donne encore le profil nul (CLFL = 0, ProfType = 1) sur les 25 pixels vides de la dalle de référence.
- source_spec: `_bmad-output/implementation-artifacts/spec-3-5-r-00c654c-alignment.md`
  summary: Story 3.8 — R Dev 00c654c corrige aussi `fPCpretreatment` (`las <- lidR::classify_ground(...)`, issue #49) : le reclassement du sol n'était pas appliqué avant. Vérifier l'effet sur l'étape 1 et sur la chaîne R de référence de la comparaison.
  evidence: `git show 00c654c -- R/fPCpretreatment.R`. L'écart nul est attendu sur la chaîne complète (Q6 bis, 23/11/2026).
- source_spec: `_bmad-output/implementation-artifacts/spec-3-5-r-00c654c-alignment.md`
  summary: Story 3.7 — valider à la lecture de la config les FMA données en valeur (`fma_can`, `fma_shrub`) : finies et dans [0,05 ; 0,5] comme la carte. Une valeur NaN de la config rend aujourd'hui toute la dalle NaN sans message, et une valeur inf ou ≤ 0 donne des métriques en partie NaN sans que D2 s'applique.
  evidence: Revue du 24/11/2026 (Edge Case Hunter). `read_fma` contrôle la plage de la carte mais renvoie la valeur de la config sans contrôle (`np.full(shape, float(fma))`). C'était déjà le cas avant cette story ; une config anormale reste nécessaire pour le déclencher.

