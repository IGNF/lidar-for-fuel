---
title: "Sprint Change Proposal — Épique 3 (métriques de combustible) : alignement sur le brief du 04/11/2026"
date: 2026-11-05
author: MDupays (workflow bmad-correct-course)
status: approuvé (2026-11-05) — appliqué à l'épique
mode: incrémental (12 propositions, toutes approuvées)
classification: Modérée
artefact modifié: /home/MDupays/Malvina/LidarForFuel/bmad/planning-artifacts/epic-3-fuel-metrics.md
---

# Sprint Change Proposal — Épique 3 : alignement sur le brief du 04/11/2026

## 1. Résumé du problème

L'épique 3 (portage de `ffuelmetrics2`) a été écrite le 22/10/2026. Le brief du 04/11/2026, son addendum et la note pour l'INRAE apportent depuis des éléments que l'épique ne prend pas en compte :

- **un calendrier** : réunion avec l'INRAE à la mi‑novembre, code prêt à la mi‑décembre, production nationale en janvier 2027 ;
- **une référence R figée** : la branche Dev plus la PR #48 (non mergée). Cette PR contredit l'AC-3.2.6 et change la façon de nommer les bandes ;
- **un nouveau livrable obligatoire** : une mesure rapide des écarts sur la chaîne complète, avant la mi‑novembre ;
- **un ordre de report** : en cas de retard, l'outil de comparaison complet passe après janvier ;
- **un critère de robustesse** à l'échelle nationale, sans critère d'acceptation correspondant dans l'épique.

**Type de changement** : nouvelle exigence venue des parties prenantes (brief), plus un écart d'implémentation de référence (PR #48).

**Preuves vérifiées le 05/11/2026** (clone local `LidarForFuel`, `git fetch origin Dev` et `pull/48/head`) :

| Point | Constat |
|---|---|
| `e72b8f5..7c7d35e` sur `R/ffuelmetrics2.R` | 5 lignes : documentation, espaces, retrait de `as_tibble` dans la signature de `ffuelmetrics2Rast` (il était de toute façon passé en dur à `FALSE`). **Les deux versions sont équivalentes pour le calcul.** La tête de Dev (`188b4ab`) ne touche plus ce fichier. |
| PR #48, ligne 283 | `which.min(abs(seuils − median(seuils)))` sur la liste **non triée**. En cas d'égalité, c'est le premier seuil dans l'ordre de la config qui l'emporte. |
| PR #48, ligne 362 | Les noms de bandes viennent des seuils passés en paramètre. **Le nombre de bandes vaut donc 19 + 8·n_échelle + n_CTH** (45 avec la config par défaut). |
| lidar‑for‑fuel `dev` v0.3.0 (`9a71d7f`) | Grille CosiaFrance (98039.75, 7111486.75) et PAD plafonné à [0, 5] : le contrat d'entrée §3.1 reste valide. |
| Dépôt `lff_compare` | Le paquet existe déjà pour les étapes 1 et 2 (Hydra, rapports, synthèse). L'outil complet doit l'étendre, pas le dupliquer. |
| `main_pad_profile.py` | Aucune exception n'est rattrapée par dalle : la robustesse doit venir du calcul lui‑même. |

## 2. Analyse d'impact

### Épiques

Seule l'épique 3 est concernée. Elle reste réalisable en l'état ; seuls son découpage et son ordre changent. Aucune autre épique n'est touchée.

### Stories

| Story | Impact |
|---|---|
| 3.1 | Aucun |
| 3.2 | AC-3.2.6 réécrit pour suivre la PR #48 |
| 3.3 | Aucun |
| 3.4 | Titre, énoncé et AC-3.4.1 : nombre de bandes dérivé de la config, avec un test à 53 clés |
| 3.5 | Aucun sur le contenu ; dépend explicitement des réponses de la réunion de mi‑novembre |
| 3.6 | AC-3.6.1 (nombre de bandes) ; **nouveaux** AC-3.6.7 à 3.6.9 (robustesse, journal des corrections, erreur d'entrée) |
| 3.7 | AC-3.7.5 précisé (45 bandes avec la config par défaut) |
| 3.8 | **Réécrite** : mesure rapide obligatoire, en deux passes, la première sans dépendance (prototype) |
| 3.9 | **Nouvelle** : étape 3 dans `lff-compare`, reportable après janvier |

### Autres artefacts

- **PRD** : il n'y en a pas. Le brief en tient lieu, et il est déjà à jour.
- **Architecture / UX** : il n'y en a pas, ce qui n'a pas d'impact.
- **Note INRAE** : pas de modification. Son tableau Q6 sera rempli par la story 3.8, passe 1.
- **Dépôt `lff_compare`** : il accueillera la story 3.9, plus tard.

### Impact technique

Le code n'est pas encore commencé : il n'y a ni retour arrière ni migration. La story 3.8, passe 1, demande de lancer les étapes 1 et 2 de lidar‑for‑fuel v0.3.0 sur `Semis_2024_0513_6856`, si ce n'est pas déjà fait.

## 3. Approche recommandée

**Ajustement direct** de l'épique : on modifie et on ajoute des stories dans le plan existant. Pas de retour arrière, pas de remise en cause du périmètre.

- **Justification** : le brief ne change pas le « quoi » de l'épique (portage fidèle, parité, 3.1 à 3.7). Il ajoute du calendrier, une référence R plus précise et un livrable de mesure. Tout cela entre dans la structure existante.
- **Effort** : environ une demi‑journée pour mettre l'épique à jour. La nouvelle story 3.8 est de taille S, la 3.9 de taille M (reportable).
- **Risque principal** : la passe 1 de la story 3.8 doit être faite **dans les 10 jours** (avant la mi‑novembre). Elle passe donc en premier, avant la story 3.1.
- **Risque secondaire** : la PR #48 pourrait être refusée ou modifiée par l'INRAE (Q5). L'impact serait limité aux AC-3.2.6 et 3.4.1.

## 4. Propositions de modification détaillées

Toutes portent sur `epic-3-fuel-metrics.md`.

| # | Section | Changement |
|---|---|---|
| P1 | Frontmatter | R = Dev `7c7d35e` + PR #48 `1e2cc00` ; Python = `9a71d7f` (v0.3.0) ; ajout de `updated: 2026-11-05` et du brief aux sources |
| P2 | §1 Synthèse | Ajout du calendrier (mi‑novembre, mi‑décembre, janvier 2027) ; 4ᵉ point à décider (PR #48) ; découpage 3.1 à 3.7, plus 3.8 obligatoire et 3.9 reportable ; priorités en cas de retard |
| P3 | §3.2 Sortie | Le nombre de bandes devient 19 + 8·n_échelle + n_CTH (45 par défaut) ; noms dérivés des seuils configurés (PR #48, ligne 362) ; le test R « 45 » est précisé « avec les seuils par défaut » |
| P4 | §4 étape 9 | Seuil par défaut = le plus proche de la médiane, avec départage par l'ordre de la config (`np.argmin` sur la liste non triée) ; exemples `[0.01,0.02,0.05,0.1]` → 0.02 et `[0.05,0.02,0.01,0.1]` → 0.05 |
| P5 | §5 Constats | A9 marqué « corrigé par la PR #48 » ; **nouveau constat A11** (noms de bandes de `ffuelmetrics2Rast`) ; paragraphe de version réécrit avec l'équivalence `e72b8f5`/`7c7d35e` vérifiée |
| P6 | §6 Décisions | Colonne « Note INRAE » (D2↔Q1, D3↔Q2, D4↔Q4, D5↔Q3) ; **nouvelle décision D7** (référence R = Dev + PR #48, ↔Q5) ; encadré de statut au 05/11 ; Q6, Q7 et Q8 rattachées à la validation |
| P7 | AC-3.2.6 | Ancien : « un nombre pair de seuils → `ValueError` ». Nouveau : règle de la PR #48, avec 4 cas de test, et `ValueError` seulement pour une liste vide |
| P8 | 3.4, 3.6, 3.7 | Ancien : « 45 » en dur. Nouveau : nombre dérivé de la config, avec un test à 4 seuils donnant 53 clés (AC-3.4.1) ; AC-3.4.5 reste à 45 (référence R) |
| P9 | Story 3.8 | Ancienne : 3.8 optionnelle, dépendant de 3.7. Nouvelle : **3.8 obligatoire**, mesure rapide en script ponctuel, passe 1 avec le prototype sans dépendance, passe 2 avec la version finale en décembre, AC-3.8.1 à 3.8.5. **Nouvelle 3.9** : module `lff_compare/fuel_metrics/`, modes A et B, AC-3.9.1 à 3.9.4, reportable. Mention de `fix/biais_pad_profile` retirée (branche abandonnée) |
| P10 | Story 3.6 | Nouveaux AC-3.6.7 (dalle entièrement NaN ou sol nu : pas d'exception), AC-3.6.8 (journal des corrections par dalle), AC-3.6.9 (aucune bande `PAD_*` → `ValueError`) |
| P11 | §10 Hors périmètre | Ajouts : volume national (environ 460 h CPU, parallélisé dalle par dalle), veille sur Dev, version dans les rasters ; « Actions hors stories » : note INRAE et issue A1 à A11 |
| P12 | §11 Ordre | Graphe à deux voies (3.8 passe 1 en parallèle, dès le départ) ; tableau des jalons J1 à J5 ; `bmad-build` remplace `bmad-create-story`, `bmad-dev-story` et `bmad-quick-dev` |

Le texte complet, ancien et nouveau, de chaque proposition a été validé en séance. Il sera reporté tel quel dans l'épique.

## 5. Passage de relais

**Classification : Modérée.** On réorganise le backlog (une story réécrite, une story créée, l'ordre change), sans replanification de fond.

| Qui | Responsabilité |
|---|---|
| Développeuse (Malvina, `bmad-build`) | 1. Mettre à jour `epic-3-fuel-metrics.md` avec P1 à P12. 2. **Faire tout de suite la story 3.8, passe 1.** 3. Enchaîner sur les stories 3.1 à 3.4. |
| Malvina (hors stories) | Compléter la note INRAE avec les chiffres de la passe 1, puis la présenter à la mi‑novembre (D2 à D5, D7, Q6, Q8) |
| INRAE (O. Martin, F. Pimont) | Répondre aux questions Q1 à Q8 et décider du sort de la PR #48 |

**Critères de succès de la correction** :
- l'épique ne contient plus de contradiction avec le brief (PR #48, « 45 » en dur, story 3.8) ;
- le tableau Q6 de la note est rempli avant la réunion ;
- les jalons J1 à J5 apparaissent dans l'épique, et la story 3.9 y est marquée comme premier élément reportable.
