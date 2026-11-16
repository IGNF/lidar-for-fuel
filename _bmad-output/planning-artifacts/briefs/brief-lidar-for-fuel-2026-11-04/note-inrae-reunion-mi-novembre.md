---
title: "Note IGN → INRAE — Étape 3 (ffuelmetrics2) : décisions à prendre"
status: draft
created: 2026-11-04
destinataires: Olivier Martin, François Pimont (INRAE)
réunion: mi-novembre 2026
---

# Étape 3 de lidar-for-fuel : décisions à prendre avant la production

## Contexte

L'IGN porte en Python la fonction `ffuelmetrics2` (étape 3 de LidarForFuel). L'objectif est un lancement de la production nationale LiDAR HD en **janvier 2027**, avec un code prêt à la **mi-décembre 2026**.

**Où en est le portage.** Un prototype Python reproduit à l'identique la sortie R de la zone pilote HE_IGN_Galaxy. Pour ce test, on lui donne le PAD produit par R (traitement du 10/10/2026, lidarforfuel 1.0.0.9012). Il obtient 0 écart sur 40 401 pixels × 45 métriques ; l'écart maximal, 5,6·10⁻⁶, correspond à l'arrondi float32.

**Version R de référence retenue par l'IGN.** La branche Dev au commit `8fcccd2` (lidarforfuel 1.0.0.9013), qui intègre la PR #48 (auteur GitHub : jmmonnet), mergée le 12/11/2026. L'IGN ne suivra pas d'autre évolution de R sans être prévenu par l'INRAE.

En portant le code, l'IGN a relevé plusieurs comportements du code R qui demandent votre arbitrage. Pour chacun, nous proposons une recommandation.

## Décisions demandées

### Q1. Pixels sans donnée : faut-il des zéros ou NaN ? **(prioritaire)**

**Constat.** Quand le profil PAD d'un pixel est entièrement vide (NaN), R le convertit en profil nul. Le pixel reçoit alors `CLFL = 0`, `ProfType = 1` et `CBH2 = 1`, c'est-à-dire « pas de combustible ». C'est le cas des 223 pixels sans mesure de la zone pilote.

**Pourquoi c'est important.** En production nationale, les zones sans mesure (bords de chantier, eau, pixels écartés par le contrôle qualité) apparaîtraient sur les cartes de risque comme des zones **sans combustible** et non **sans mesure**. Or ces cartes sont destinées aux SDIS et à l'ONF.

**Recommandation IGN.** Mettre NaN sur les 45 métriques quand le PAD du pixel est entièrement vide.

### Q2. Bandes `CBD_cbh`, `CBD_cbh2`, `CBD_cbhlow` toujours vides

**Constat.** Ces trois bandes n'ont aucun pixel valide sur les 40 401 de la zone pilote. Le code cherche `zval == CBH`. Or, depuis la modification du 12/05/2026, `zval` est un centre de strate (x,5) alors que CBH est un bord de strate (valeur entière). L'égalité n'est donc jamais vraie.

**Recommandation IGN.** Les calculer correctement : prendre la densité CBD (recalculée après CBH2) de la strate dont le **bas** vaut CBH, CBH2 ou CBHlow.

**Alternatives** : les garder vides pour rester identique au R, ou les retirer (42 bandes au lieu de 45).

### Q3. Seuil de 1 m codé en dur dans la recherche du combustible en échelle

**Constat.** `ffuelmetrics2.R`, ligne 253 : la condition est `zval > 1` au lieu d'utiliser `CanopyHeightThreshold`. Il n'y a aucun effet avec la configuration actuelle (seuil à 1 m), mais le résultat est faux si ce seuil vaut 2 ou 3 m, alors que la documentation annonce « typiquement 1 à 3 m ».

**Recommandation IGN.** Utiliser `CanopyHeightThreshold`.

### Q4. Valeur de `CBHThresholdRatio` : 0.333 ou 1/3 ?

**Constat.** `ffuelmetrics2Rast` ne transmet ni `CBHThresholdRatio` ni `smooth_pad` à la fonction de calcul. En mode raster, R utilise donc toujours **1/3** et `smooth_pad = FALSE`, quelle que soit la configuration. Si l'on applique la valeur affichée dans la configuration (0.333), le résultat change sur la zone pilote :
- 209 pixels changent de CBH2, avec des écarts allant jusqu'à 14 m ;
- 5 083 pixels changent de CanBDlowmax.

**Recommandation IGN.** Fixer explicitement la valeur à 1/3 (celle que R utilise réellement) et transmettre tous les paramètres. Merci de confirmer que 1/3 est bien la valeur voulue.

### Q5. PR #48 (seuils d'échelle et noms de bandes) — résolue

**Constat.** La PR #48 fait deux choses :
- quand le nombre de seuils d'échelle est pair, elle choisit le seuil le plus proche de la médiane ;
- elle dérive les noms de bandes de `ffuelmetrics2Rast` des seuils réellement passés.

Elle est sans effet sur la référence du 10/10/2026, qui utilise 3 seuils par défaut.

**Résolue le 12/11/2026 : la PR #48 a été mergée telle quelle.** Merci. Le Python s'aligne sur ce comportement, et aucune décision n'est plus demandée sur ce point.

### Q6. Tolérance acceptée entre la chaîne Python et la chaîne R

**Contexte.** Sur la chaîne complète (étapes 1 à 3 en Python), les métriques ne seront pas identiques à celles de R, car le PAD Python diffère légèrement du PAD R. Les écarts se concentrent sur les strates de 0 à 1 m. Ils devraient être réduits dans lidar-for-fuel v0.3.0, qui intègre deux correctifs :
- les points aberrants sont détectés sur la hauteur normalisée ;
- le MNT est téléchargé en tenant compte du décalage de grille.

**Proposition IGN.** Une tolérance par métrique plutôt qu'un pourcentage unique. Un écart relatif n'a pas de sens pour une classe comme `ProfType`, ni pour une charge qui vaut 0. L'ordre de grandeur visé est de 5 %.

> **À compléter avant la réunion** avec les résultats de la mesure rapide (étape 3 Python appliquée au PAD Python v0.3.0, comparée à `fuel_metrics.tif` de l'INRAE).

| Métrique | Indicateur | Écart mesuré | Seuil proposé |
|---|---|---|---|
| CBH2, CTH, CDH (m) | écart absolu moyen ; 99ᵉ centile | _à mesurer_ | _à proposer_ |
| CFL, LFL, CLFL, FL_k1, FL_k2 (kg/m²) | écart absolu moyen ; part de pixels dans la tolérance | _à mesurer_ | _à proposer_ |
| CBDcan (kg/m³) | écart absolu moyen | _à mesurer_ | _à proposer_ |
| ProfType (classe) | taux d'accord | _à mesurer_ | _à proposer_ |

**Question.** Ces indicateurs et ces seuils vous conviennent-ils pour valider la chaîne Python ?

### Q7. Quelles métriques utilisez-vous réellement ?

Parmi les 45 bandes produites, lesquelles exploitez-vous pour les cartes de combustible et de risque ? Votre réponse nous permettra de concentrer la validation sur ces métriques, et elle peut simplifier Q2.

### Q8. Calendrier de validation

Le code sera prêt à la mi-décembre. La validation par l'INRAE s'appuiera sur deux éléments :
- le test de parité de l'étape 3 (PAD R en entrée, sortie identique au R) ;
- la mesure des écarts sur la chaîne complète, refaite sur la version finale.

Qui, à l'INRAE, validera, et à quelle date ? La période des fêtes laisse peu de marge avant le lancement de janvier.

## Pour information (sans décision demandée)

- **Pixels partiellement vides.** Un PAD partiellement NaN est mis à 0 sans être signalé (variable `invalid` calculée mais jamais utilisée). Un seul pixel est concerné sur la zone pilote. L'IGN journalisera ces corrections.
- **PAD plafonné à 5 côté Python.** Ce plafond n'existe pas en R (maximum observé : 7,01). Le résultat reste identique tant que 5 × min(FMAcan, FMAshrub) ≥ bdmax et que `smooth_pad` est désactivé. C'est le cas avec la configuration actuelle (5 × 0,254 = 1,27 ≥ 1). L'IGN émettra un avertissement si cette condition n'est plus remplie.
- **Valeurs par défaut différentes** entre `ffuelmetrics2` et `ffuelmetrics2Rast` (FMA, `CanopyBDTreshFrac`, ratio). En Python, toutes les valeurs viendront d'une seule source : la configuration.
- **Unités de `fuel_mass_area()`.** La documentation annonce un LMA en g/m², mais la valeur de la configuration (0.2543576) correspond à un LMA de 0,140 en **kg/m²**. La branche R `doc-fix-lma-units` traite peut-être ce point.
