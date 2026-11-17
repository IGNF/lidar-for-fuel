# Epic 3 Context: Métriques de combustible (portage de `ffuelmetrics2`)

<!-- Compiled from planning artifacts. Edit freely. Regenerate with compile-epic-context if planning docs change. -->

## Goal

Ajouter à lidar-for-fuel la 3ᵉ et dernière étape de la chaîne LidarForFuel : transformer le raster de profils PAD à 1 m (sortie de l'étape 2) en un raster de métriques de combustible par pixel de 10 m. Ces métriques sont les charges, les hauteurs de base et de sommet du houppier, le combustible en échelle et le type de profil. On en compte 45 avec la config par défaut. Il s'agit d'un portage Python fidèle du code R `ffuelmetrics2` de l'INRAE. Ce code est la condition pour démarrer la production nationale en janvier 2027 : l'INRAE produit ses cartes de combustible et de risque d'incendie à partir de ces métriques. Le code doit être prêt mi‑décembre 2026 et validé par l'INRAE avant la production.

## Stories

- Story 3.1 : Jeu de données de référence R (fait)
- Story 3.2 : Lecture des strates et préparation du profil CBD
- Story 3.3 : Métriques de canopée (CTH, CBH, CBH2, CBHlow, CDH, CBDcan)
- Story 3.4 : Charges, combustible en échelle et assemblage (parité stricte avec R)
- Story 3.5 : Corrections assumées par rapport à R
- Story 3.6 : Application au raster PAD et export GeoTIFF
- Story 3.7 : Point d'entrée `main_fuel_metrics`, configuration et documentation
- Story 3.8 : Mesure et explication des écarts sur la chaîne complète (obligatoire)
- Story 3.9 : Étape 3 dans `lff-compare` (reportable)

## Requirements & Constraints

- **Parité stricte avec R d'abord, corrections ensuite.** Le Python reproduit R à l'arrondi float32 près. Les tests de parité utilisent `rtol=1e-5, atol=1e-6`, avec égalité des NaN. Les corrections assumées viennent dans une story séparée, et chaque divergence est listée dans la docstring et dans le `CHANGELOG.md`.
- **Écart nul attendu par l'INRAE** à entrées et paramètres identiques. Il n'y a pas de tolérance en pourcentage.
- **Pixels sans donnée** (PAD entièrement NaN) : NaN sur toutes les bandes. Cette décision est acceptée par l'INRAE, et la correction est appliquée après la parité.
- **Le seuil `h > 1` codé en dur** dans la recherche du combustible en échelle est voulu par l'INRAE. Il devient une constante nommée, indépendante de `canopy_height_threshold`.
- **`cbh_threshold_ratio`** : la décision est encore ouverte. En attendant, on utilise 1/3 exact (`0.3333333333333333`), la valeur que R calcule réellement en mode raster, et on n'utilise **pas** 0.333.
- **Robustesse nationale.** Le traitement ne s'interrompt jamais sur une dalle atypique (eau, bord de chantier, dalle vide). Ce qui est corrigé est signalé dans les logs, pas masqué.
- **Performance.** Environ 3 s par dalle en boucle Python pixel par pixel. Aucune vectorisation ni parallélisation n'est requise.
- **Nombre de bandes piloté par la config** : 19 + 8·n_seuils_échelle + n_seuils_CTH. Ne jamais coder 45 en dur.

## Technical Decisions

- **Version R de référence** : branche Dev `8fcccd2` (lidarforfuel 1.0.0.9013, qui inclut la PR #48). Elle sera rebasée quand l'INRAE aura corrigé les bandes `CBD_cbh*`.
- **Entrée** : `{stem}_pad_profile_1m.tif`, bandes nommées `PAD_{dz}_{h_bas}` (ex. `PAD_1_10`, `PAD_0.5_1.5`), mêlées à d'autres bandes. Valeurs en float32, nodata NaN, PAD plafonné à [0, 5]. Les bandes PAD se trouvent **par leur nom**, jamais par leur position. Le raster R a 186 bandes, le raster Python 60.
- **Hauteurs** : `h_bottom` est le bas de strate, lu dans le nom de bande, et `h = h_bottom + dz/2` est le centre de strate (le `zval` de R). En Python, les hauteurs au‑dessus du sol se nomment `h_*`, jamais `z*` (`z` désigne l'altitude brute).
- **Préparation du profil** :
  - les valeurs non finies ou négatives passent à 0 ;
  - lissage optionnel par moyenne mobile centrée sur 3 strates, avec une fenêtre partielle aux bords (moyenne des valeurs disponibles, pas une division par 3) ;
  - CBD = min(PAD × FMA, bd_max), avec `fma_can` au‑dessus d'une hauteur limite et `fma_shrub` en dessous. La limite vaut CHT au premier calcul et CBH2 au recalcul.
- **Seuil d'échelle par défaut** (règle de la PR #48) : le seuil le plus proche de la médiane, pris sur la liste de la config **non triée**. En cas d'égalité, on prend le premier dans l'ordre de la config (`np.argmin`). Un nombre pair de seuils ne lève pas d'erreur.
- **Pièges du portage** :
  - les indices R commencent à 1, ceux de Python à 0 ;
  - `which(...)[1]` en R correspond au **premier** indice en Python ;
  - les NaN se propagent comme le `any(NA)` de R ;
  - il faut multiplier `CTH * ratio`, comme R, et non diviser ;
  - garder exactement les comparateurs de R ;
  - noms de bandes : `f"{t:0.3f}"` pour les seuils d'échelle et `f"{t:0.4f}"` pour les seuils CTH, avec `.` remplacé par `p`.
- **Code** :
  - package `lidar_for_fuel/fuel_metrics/`, tests dans `test/fuel_metrics/` ;
  - un module par responsabilité, des arguments explicites, des docstrings Google ;
  - adapter les fonctions existantes plutôt que d'en créer des parallèles (réutiliser `_write_geotiff` et `_format_num` de `pad_profile/export_raster.py`) ;
  - point d'entrée Hydra comme `main_pad_profile.py`.
- **Condition d'équivalence avec R** : `5 × min(fma_can, fma_shrub) ≥ bd_max` et `smooth_pad = False`. Le point d'entrée émet un avertissement si elle n'est pas remplie.

## Cross-Story Dependencies

- La 3.1 fournit les données de référence R : `data/ref_inrae/` contient le PAD R, `fuel_metrics.tif` à 45 bandes et la config INRAE, tous issus de la même dalle de 500 m. Les rasters à 10 m ont un débord normal d'une colonne et d'une ligne. Ces données servent au TDD de toutes les stories suivantes.
- 3.2 → 3.3 → 3.4 forment le calcul 1‑D par profil.
- La 3.5 suit la 3.4. La 3.6 suit la 3.4 ; elle a besoin de la 3.5 pour le NaN des pixels vides.
- La 3.7 suit la 3.6. La passe 2 de la 3.8 suit la 3.7 ; sa passe 1 est indépendante.
- La 3.9 vit dans le dépôt `lff_compare` et suit la 3.7.
