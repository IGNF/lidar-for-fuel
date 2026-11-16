---
title: "Brief — Étape 3 lidar-for-fuel : métriques de combustible (portage de ffuelmetrics2)"
status: final
created: 2026-11-04
updated: 2026-11-13
---

# Brief — Étape 3 : métriques de combustible

L'IGN doit porter en Python l'étape 3 de LidarForFuel (`ffuelmetrics2`) d'ici la mi-décembre 2026. C'est la condition pour que la production nationale démarre en janvier 2027 avec les métriques de combustible.

Le détail du « comment » est dans l'épique 3 : contrat d'entrée et de sortie, algorithme, décisions D1 à D6, stories. Les détails techniques (PR #48, écarts de PAD, pistes pour l'outil de comparaison) sont dans `addendum.md`.

## Pourquoi ce portage

L'INRAE a développé en R la chaîne LidarForFuel. Elle transforme un nuage de points LiDAR aérien en métriques de combustible forestier, en trois étapes : prétraitement, profils verticaux de densité de végétation (PAD), puis métriques de combustible.

L'IGN porte cette chaîne en Python pour l'exécuter sur son infrastructure de production, à l'échelle nationale, sur l'ensemble des dalles LiDAR HD. Les étapes 1 et 2 sont portées par l'IGN et validées par l'INRAE. L'étape 3 reste à faire.

**Chaîne de valeur.** L'IGN produit les métriques. Les chercheurs de l'INRAE les transforment en cartes de combustible et de risque d'incendie. Les services d'incendie et de secours (SDIS) et l'ONF utilisent ces cartes.

Sans l'étape 3, la chaîne nationale produit des profils PAD mais aucune des métriques que l'INRAE exploite.

## Ce qui est livré

1. **L'étape 3 de la chaîne Python.** Elle part du raster des profils PAD produit par l'étape 2 : des pixels de 10 m, découpés en strates verticales de 1 m. Pour chaque pixel, elle calcule les 45 métriques de `ffuelmetrics2` : charges, hauteurs de base et de sommet du houppier, combustible en échelle, type de profil. Elle écrit le résultat dans un GeoTIFF de 45 bandes, sur la même grille que l'entrée. Elle se lance comme les étapes 1 et 2.
2. **Une mesure rapide des écarts entre Python et R** sur la zone pilote. C'est un script ponctuel. Il applique l'étape 3 au PAD produit par lidar-for-fuel v0.3.0, puis compare le résultat aux métriques de la chaîne R complète. Il sert deux fois :
   - **avant la mi-novembre**, pour chiffrer la proposition de tolérance ;
   - **en décembre**, refait sur la version finale, comme élément de validation pour l'INRAE.
3. **Une note de décision pour l'INRAE**, à présenter à la réunion de mi-novembre 2026. Elle rassemble les questions ouvertes listées plus bas, avec une proposition de tolérance chiffrée par métrique.
4. **Un outil de comparaison complet**, propre et réutilisable sur d'autres zones. Il couvre l'étape 3 seule et la chaîne complète, avec un rapport et des cartes d'écart. Il est reportable après janvier (voir « Priorités en cas de retard »).

## Critères de succès

L'étape 3 est « prête pour la production » quand les quatre conditions suivantes sont réunies :

1. **Parité de l'étape 3 seule.** Avec en entrée le PAD produit par R sur la zone pilote, les 45 métriques Python sont identiques à celles de R, à l'arrondi float32 près, en dehors des divergences assumées et documentées. Un prototype a déjà atteint 0 écart sur 40 401 pixels × 45 bandes.
2. **Écart maîtrisé sur la chaîne complète.** Sur la zone pilote, l'écart entre les métriques de la chaîne Python complète et celles de la chaîne R reste sous les seuils convenus avec l'INRAE. L'ordre de grandeur indicatif est de 5 %. Les seuils sont toutefois définis métrique par métrique (voir Q6) et proposés à la mi-novembre à partir de la mesure rapide.
3. **Validation par l'INRAE** avant le lancement de la production en janvier 2027. Elle s'appuie sur le test de parité de l'étape 3 et sur la mesure rapide refaite sur la version finale.
4. **Robustesse à l'échelle nationale.** L'étape ne s'interrompt ni sur les dalles atypiques (eau, bords de chantier) ni sur les pixels sans donnée. Elle signale ce qu'elle a corrigé au lieu de le masquer. Le volume de calcul n'est pas un enjeu : environ 3 s par dalle.

## Périmètre

**Référence R.** La branche Dev de LidarForFuel au commit `8fcccd2` (lidarforfuel 1.0.0.9013, 12/11/2026), qui intègre la PR #48, mergée le 12/11/2026. Les évolutions ultérieures de R ne sont pas suivies dans cette livraison. L'INRAE s'engage à prévenir l'IGN de tout changement.

**Inclus (en plus des livrables ci-dessus).**
- Portage fidèle de `ffuelmetrics2`, avec le comportement de la PR #48 (nombre de seuils pair, noms de bandes).
- Paramètres en configuration.
- Jeu de référence et tests de parité.
- Documentation des divergences assumées avec R.

**Exclus pour cette livraison.**
- FMA par pixel ou par essence.
- Graphique de profil.
- Calcul par placette.
- Charge de surface corrigée par quantile mapping.
- Détection automatique des changements sur la branche Dev de R. C'est une idée retenue pour plus tard.

**Priorités en cas de retard.** Le développement est assuré par une seule personne ; les PR sont relues par d'autres. Si le calendrier glisse, l'outil de comparaison complet est reporté en premier, après janvier. Le portage, les tests de parité et la mesure rapide ne sont pas négociables.

## Risques et questions ouvertes

Ces points constituent la trame de la note pour l'INRAE de mi-novembre. Les codes D2 à D5 renvoient aux décisions de l'épique 3.

| # | Sujet | Enjeu et décision attendue |
|---|---|---|
| Q1 | **Pixels sans donnée (D2).** R leur attribue une charge nulle et un profil « sans combustible ». | **Risque majeur.** À l'échelle nationale, les pixels sans donnée apparaîtraient comme sans combustible sur les cartes de risque transmises aux SDIS. L'IGN propose de leur attribuer NaN. |
| Q2 | **Bandes `CBD_cbh`, `CBD_cbh2` et `CBD_cbhlow` toujours vides en R (D3)**, à cause d'un bug de comparaison de hauteurs. | Les corriger, les laisser vides pour rester identique à R, ou les retirer ? |
| Q3 | **Seuil de 1 m codé en dur dans la recherche du combustible en échelle (D5)**, au lieu du paramètre `CanopyHeightThreshold`. | Sans effet avec la configuration actuelle, mais faux dès que `CanopyHeightThreshold` vaut 2 ou 3 m. |
| Q4 | **`CBHThresholdRatio` (D4).** La configuration affiche 0.333, mais R utilise 1/3 en mode raster. | Entre 0.333 et 1/3, CBH2 diffère jusqu'à 14 m sur une partie des pixels. Il faut confirmer que 1/3 est la valeur voulue. |
| Q5 | ~~**PR #48 non mergée.**~~ **Résolue le 12/11/2026** : la PR est mergée telle quelle. | Le Python suivait déjà la PR #48. Il n'y a plus de risque de divergence avec le R officiel. |
| Q6 | **Tolérance sur la chaîne complète.** L'IGN propose des seuils chiffrés par métrique, à partir de la mesure rapide ; l'INRAE les amende. | Un pourcentage relatif unique n'a pas de sens pour une classe comme `ProfType`, ni pour une charge nulle. Il faut un critère par métrique : écart moyen, 99ᵉ centile, part de pixels dans la tolérance, accord sur le type de profil. |
| Q7 | **Usage réel des 45 bandes.** Lesquelles l'INRAE exploite-t-il vraiment ? | La réponse oriente les priorités de validation et peut simplifier Q2. |

**Risque de calendrier.** La validation par l'INRAE tombe entre la mi-décembre et janvier, pendant les fêtes. Il faut la caler dès la réunion de mi-novembre.
