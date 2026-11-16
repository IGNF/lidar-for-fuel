---
title: "Addendum — Brief étape 3 (métriques de combustible)"
created: 2026-11-04
updated: 2026-11-13
---

# Addendum au brief étape 3 : détails pour le PRD et la spec

## Référence : épique 3

L'épique 3 donne le contrat d'entrée et de sortie, l'algorithme, les constats A1 à A10, les décisions D1 à D6 et les stories 3.1 à 3.8 :
`docs/epic-3-fuel-metrics.md`, versionné dans le dépôt (22/10/2026, mise à jour le 13/11/2026). Ancienne copie : `/home/MDupays/Malvina/LidarForFuel/bmad/planning-artifacts/epic-3-fuel-metrics.md`.

## Écarts à reporter dans l'épique 3

- **Story 3.8** (comparaison de la chaîne complète) : à découper en deux.
  - Une mesure rapide, **obligatoire**. C'est un script ponctuel, lancé avant la réunion de mi-novembre puis refait sur la version finale en décembre. Avec le test de parité de l'étape 3, elle sert à la validation par l'INRAE.
  - Un outil complet, **reportable** après janvier. C'est le premier élément reporté en cas de retard.
- **AC-3.2.6** : l'épique prévoit une `ValueError` quand le nombre de seuils est pair, ce qui est contraire à la PR #48. Il faut s'aligner sur la PR #48 (ligne 283, voir la section « PR #48 »).
- **Noms de bandes** : les dériver des seuils réellement configurés, comme le fait la PR #48 dans `ffuelmetrics2Rast` (ligne 362).
- **Commit de base de l'épique.** L'épique cite `e72b8f5`, alors que le dernier commit Dev sur `ffuelmetrics2.R` est `7c7d35e` (10/10/2026, correction de documentation). Il faut vérifier que les deux états sont équivalents.

## PR #48 (oliviermartin7/LidarForFuel)

- Auteur : jmmonnet. Ouverte le 20/10/2026 avec un seul commit, `1e2cc00`, de +2/−2 lignes dans `R/ffuelmetrics2.R`.
- **Mergée le 12/11/2026** (`261b91e`, commits `e3e63fe` + `ac17099`), puis version 1.0.0.9013 et NEWS (`8fcccd2`). Les deux lignes mergées sont identiques à celles de `1e2cc00` : rien ne change pour le portage.
- **Ligne 283.** Le suffixe par défaut devient le seuil de `LadderFuelBDthresholds` le plus proche de la médiane. En pseudo-code : `which.min(abs(seuils - median(seuils)))`.
- **Ligne 362.** Le gabarit de noms de `ffuelmetrics2Rast` reçoit maintenant `LadderFuelBDthresholds` et `CanopyTopBDThresholds`.
- **Impact sur le jeu de référence du 10/10/2026 : aucun.** Ce jeu utilise les 3 seuils par défaut, un nombre impair.

## État des écarts PAD entre Python et R (étapes 1 et 2)

- Les écarts portent surtout sur les **strates basses, de 0 à 1 m**.
- Deux correctifs sont intégrés dans lidar-for-fuel v0.3.0 :
  - les points aberrants sont détectés sur le Z normalisé et non plus sur Zref ;
  - le téléchargement du MNT tient compte du décalage de grille.
- Écarts attendus : minimes.
- **Sensibilité de l'étape 3.** Ce point est tiré de l'analyse de l'algorithme de l'épique, avec `CanopyHeightThreshold` (CHT) = 1 m :
  - la strate 0–1 m entre directement dans `FL_k1` ;
  - elle est exclue de `CLFL` (h > CHT) ;
  - elle est aussi exclue de la recherche du combustible en échelle (h > 1, seuil codé en dur : voir Q3 du brief) ;
  - un écart limité à 0–1 m devrait donc toucher surtout `FL_k1` ;
  - c'est à vérifier par la mesure rapide.

## Outil de comparaison : pistes

- Deux modes :
  - **A** : l'étape 3 Python appliquée au PAD R, pour la parité stricte (critère de succès 1) ;
  - **B** : la chaîne Python complète comparée à la chaîne R (critère 2). C'est ce que fait la mesure rapide.
- La différence B − A isole l'écart qui vient du PAD.
- Indicateurs par métrique :
  - écart absolu moyen ;
  - 99ᵉ centile ;
  - part des pixels dans la tolérance ;
  - accord sur `ProfType` ;
  - cartes d'écart sur CBH2 et CFL.
- Les grilles étant alignées, la comparaison porte sur leur zone commune (l'intersection des emprises).

## Volume de calcul à l'échelle nationale (estimation)

Environ 3 s par dalle (mesure de l'épique) multipliées par environ 550 000 dalles de 1 km² donnent à peu près 460 h CPU. Le calcul se parallélise dalle par dalle. Ce chiffre est à recaler sur le nombre réel de dalles LiDAR HD.

## Idées différées

- **Fonction de veille.** Comparer la branche Dev de R à la version figée (Dev `8fcccd2`, qui contient la PR #48) et signaler les changements dans `R/ffuelmetrics2.R` et `R/pad_metrics.R`.
- **Versionnement des sorties de production.** La version de lidar-for-fuel est déjà inscrite dans les GPKG depuis la PR #43. Il faudrait l'étendre aux rasters de métriques si l'INRAE fait évoluer les formules en cours de production.
