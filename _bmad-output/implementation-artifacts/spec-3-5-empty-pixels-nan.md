---
title: 'Story 3.5 — Corrections assumées par rapport à R : pixels sans donnée à NaN (D2)'
type: 'feature'
created: '2026-11-20'
status: 'done'
route: 'oneshot'
review_loop_iteration: 0
context:
  - '{project-root}/docs/epic-3-fuel-metrics.md'
  - '{project-root}/_bmad-output/implementation-artifacts/epic-3-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem :** `fuel_metrics_core` reproduit R jusque dans ses défauts. Un pixel dont le PAD est entièrement NaN (eau, bord de chantier, pixel écarté) reçoit les métriques d'un profil nul : charges à 0, `CBH2` = CHT, `ProfType` = 1 (A5). Sur les cartes, ces zones sans mesure apparaissent donc « sans combustible ».

**Approach :** Appliquer en TDD la décision D2, acceptée par l'INRAE le 16/11/2026 : pour un PAD entièrement NaN, toutes les métriques valent NaN. Les noms et l'ordre des clés ne changent pas. Le test de parité avec R reste vert sur tous les autres pixels, et la divergence est listée dans la docstring de `fuel_metrics_core` et dans le `CHANGELOG.md`. Les critères d'acceptation sont AC-3.5.1, AC-3.5.4 et AC-3.5.5 de l'épique. D3 (bandes `CBD_cbh*` gardées à NaN, comme R) et D5 sont hors périmètre : elles ont été retirées de la story.

</frozen-after-approval>

## Implementation Notes

**Décisions** (choix internes, invisibles dans le résultat pour l'utilisatrice) :

- **Branche** : `feat/ffuelmetrics2_corrections`, créée depuis `feat/ffuelmetrics2` (`d6ac376`).
- **Où** : dans `fuel_metrics_core` ([lidar_for_fuel/fuel_metrics/calculate_fuel_metrics.py](../../lidar_for_fuel/fuel_metrics/calculate_fuel_metrics.py)), pour que le raster (3.6, AC-3.6.4) et un futur calcul par placette en profitent sans code de plus.
- **Critère « sans donnée »** : `np.isnan(pad).all()` sur le PAD **brut**, avant la préparation. Un profil partiellement NaN, inf ou négatif reste traité comme en R : valeurs mises à 0 (A6). Un profil entièrement inf n'est pas « sans donnée ».
- **Comment** : on calcule les métriques comme aujourd'hui, puis on renvoie `dict.fromkeys(metrics, np.nan)`. Les clés et leur ordre restent ceux du contrat, quelle que soit la config, sans dupliquer la liste des noms. Le coût n'est payé que par les pixels vides.
- **Parité (AC-3.5.4)** : D2 porte sur des pixels, pas sur des bandes. Le test de parité exclut donc explicitement les pixels entièrement NaN, avec un masque et un commentaire qui nomme D2. Sur ces pixels, il vérifie que le Python donne NaN sur toutes les bandes, là où R donne ses zéros.
- **Docstrings** : la puce A5 passe des « défauts conservés » à une rubrique « divergence assumée » (D2). La docstring du module ne dit plus « portage strict sans correction ». Aucune référence de story n'est ajoutée dans le code.
- **Hors périmètre** : comptage et journalisation des pixels vides par dalle (AC-3.6.8, AC-E.5), traitement d'une FMA NaN (Q9, 3.6), bandes `CBD_cbh*` (D3, gardées à NaN).

**Fichiers prévus** :
- `lidar_for_fuel/fuel_metrics/calculate_fuel_metrics.py` : retour NaN et docstrings.
- `test/fuel_metrics/test_calculate_fuel_metrics.py` : `test_fuel_metrics_all_nan_profile_gives_the_zeros_of_r` est remplacé par un test D2 (toutes les métriques NaN, mêmes clés que R, y compris avec 4 seuils d'échelle). Un cas vérifie qu'un seul PAD fini suffit pour ne pas être « sans donnée ». Le test de parité masque les pixels D2.
- `CHANGELOG.md` : une ligne sous « not yet released ».

**Réalisé** :
- Les fichiers sont ceux prévus. `fuel_metrics_core` teste `np.isnan(pad).all()` sur le PAD brut, calcule les métriques comme avant (sur le profil mis à zéro), puis renvoie `dict.fromkeys(metrics, np.nan)` pour un pixel sans donnée.
- Tests : le test « zéros de R » est remplacé par un test D2 paramétré (3 et 4 seuils d'échelle : mêmes clés que pour un profil valide, toutes NaN et de type `float`). Un nouveau test vérifie qu'une seule valeur finie suffit à calculer les métriques (A6). Le test de parité masque les 25 pixels sans donnée de la référence : il vérifie que R y donne `CLFL` = 0 et que le Python y donne NaN sur les 45 bandes.

**Vérification** :
- TDD : les 3 tests modifiés échouaient avant le changement de code (3 échecs, 17 réussis).
- Mutations : renvoyer les métriques sans le NaN fait échouer le test D2 et la parité ; tester `np.isnan(pad).any()` au lieu de `.all()` fait échouer le test « une valeur finie suffit ».
- Parité R (AC-3.5.4) : 0 écart sur les 45 bandes × 625 pixels avec donnée.
- `pytest test/fuel_metrics` : 89 réussis. Suite complète : 245 réussis. black, isort (`--profile black`) et flake8 propres sur les fichiers modifiés.

## Review Triage Log

Revue « Blind Hunter » (10 remarques). Après les corrections : `pytest test/fuel_metrics` 89 réussis, black, isort et flake8 propres.

| # | Remarque du relecteur | Verdict | Justification et suite donnée |
|---|---|---|---|
| 1 | Pour un pixel vide, `prepare_pad_profile` journalise « N valeurs mises à 0 » et renvoie ce compte : le log serait trompeur et la 3.6 risquerait de compter ces pixels comme corrections A6 | low | Rejeté. Le log est en debug et reste exact (le profil est bien mis à 0 avant le calcul jeté). `fuel_metrics_core` ne renvoie pas ce compte : la 3.6 compte sur le raster (entrée existante de `deferred-work.md`). |
| 2 | Un profil sans aucune valeur finie mais pas entièrement NaN (inf, négatifs) garde les métriques du profil nul | false | Inatteignable en production : l'étape 2 plafonne le PAD à [0, 5] (`np.clip` dans `export_raster.py`), ce qui ne laisse que des NaN comme valeurs non valides. L'épique définit D2 par « PAD entièrement NaN ». |
| 3 | AC-3.5.4 : le commentaire du test de parité ne renvoie pas à l'épique ; la divergence est exclue par masque de pixels, et non « par leur nom » | low | **Corrigé** : le commentaire cite `docs/epic-3-fuel-metrics.md §6`. Le masque est voulu, car D2 porte sur des pixels (voir les décisions) ; le texte de l'AC appartient à l'épique, et c'est à l'utilisatrice de le modifier. |
| 4 | `np.isnan(pad)` est appelé avant la conversion en float : `TypeError` sur un tableau objet | false | Inatteignable : le PAD vient d'un raster float32, et l'échec serait bruyant. |
| 5 | Le test de parité ne vérifie le profil nul de R que par `CLFL`, et la dalle n'a pas de pixel partiellement NaN | low | **Corrigé** : il vérifie aussi `CBH2` = CHT et `ProfType` = 1 dans la référence R, et le commentaire dit que le cas partiellement NaN est couvert par les profils écrits à la main. |
| 6 | Le test D2 ne fait varier que le nombre de seuils d'échelle, pas les seuils CTH ni les doublons | false | Les clés NaN viennent par construction du dict calculé (`dict.fromkeys(metrics)`), quelle que soit la config ; le test compare déjà ces clés à celles d'un profil valide. |
| 7 | Docstring du module : « both » est ambigu, et le pluriel est faux (une seule divergence) | low | **Corrigé**. |
| 8 | CHANGELOG : `CBH2` = CHT est omis, et le code D2 n'est pas défini | low | **Corrigé** : le profil nul est complet, et l'entrée renvoie à l'épique. |
| 9 | Statut de la spec encore `in-progress` ; épique et `deferred-work.md` non mis à jour | false | Le statut passe à `done` à la finalisation. Q1 bis et AC-3.6.8 sont déjà suivis dans l'épique. |
| 10 | La note « le coût n'est payé que par les pixels vides » serait trompeuse : calcul jeté sur les dalles très vides | false | La note est exacte : seuls les pixels vides font un calcul inutile, ce qui ne coûte pas plus qu'avant D2 (0,27 ms par pixel). |
