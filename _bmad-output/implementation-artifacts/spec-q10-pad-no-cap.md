---
title: 'Q10 — Retrait du plafond du PAD à 5 (étape 2)'
type: 'bugfix'
created: '2026-11-24'
status: 'done'
route: 'oneshot'
review_loop_iteration: 0
context:
  - '{project-root}/docs/epic-3-fuel-metrics.md'
  - '{project-root}/_bmad-output/implementation-artifacts/epic-3-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem :** l'étape 2 plafonne le PAD à [0, 5] avant d'écrire `pad_profile_1m` et `pad_sb_0.5m` (`raster_clip` d'`export_raster.py`), alors que R `pad_metrics` ne plafonne pas. Sur la zone pilote, R va jusqu'à 7,01 pour le PAD 1 m et jusqu'à 11,83 pour le PAD 0,5 m. Les rasters PAD diffèrent donc de R, et à l'étape 3 la CBD diffère aussi dès que la FMA est inférieure à 0,2.

**Approach :** l'INRAE a répondu le 23/11/2026 à Q10 : « retire le plafond de python ». On écrit donc le PAD brut, comme R. Un éventuel plafond sera un post-traitement des rasters PAD, à discuter en point technique : il est hors périmètre. La condition d'équivalence `5 × min(FMA) ≥ bd_max` et l'AC-3.7.3 de l'épique n'ont alors plus lieu d'être, et la docstring de `fuel_metrics_core` est mise à jour en conséquence. Le traitement est fait dans la branche `fix/pad_profile_no_cap`, créée depuis `feat/ffuelmetrics2`.

</frozen-after-approval>

## Implementation Notes

**Décisions** :
- **Les deux bornes sont retirées** (`raster_clip` passe de `(0.0, 5.0)` à `None` pour `pad_profile_1m` et `pad_sb_0.5m`), et pas seulement la borne haute : R ne borne pas du tout. Le PAD est ≥ 0 tant que `ground_margin < dz` (−log d'une fraction ≤ 1). La correction de Laplace de `compute_nrd` garde Gf > 0 dans la formule simple. Avec la couverture, un NRD égal à la couverture donnerait inf et un NRD supérieur NaN, comme en R. Un inf n'est plus ramené à 5 : l'étape 3 le met à 0, comme R (A6). Sur la zone pilote, R n'a ni inf ni valeur négative ; il dépasse 5 sur 2 valeurs à 1 m (max 7,01) et sur 5 valeurs à 0,5 m (max 11,83).
- **La couverture garde sa borne** `(0.0, 1.0)` : Q10 ne porte que sur le PAD. Le paramètre `clip` de `_band_array` reste donc utilisé ; seul l'exemple de sa docstring change.
- **Formatage** : black (une ligne vide après la docstring du module `export_raster.py`) et isort (un import de `test_export_raster.py`) n'étaient pas respectés avant ce changement. Ils sont appliqués parce que le hook pre-commit bloque sinon.

**Réalisé** :
- `export_raster.py` : PAD écrit brut.
- `test_export_raster.py` : le test « plafonne à 5 » devient `test_export_raster_writes_raw_pad_values` (42 à 1 m, 11,83 à 0,5 m), et l'assertion `<= 5` du test sur dalle réelle est retirée.
- `test_main_pad_profile.py` : l'assertion `<= 5` est retirée et `>= 0` est gardée, avec un commentaire mis à jour.
- Docstring de `fuel_metrics_core`.
- Épique : tableau d'entrée, `pad_sb_0.5m`, schéma, §3.1, note sur la condition d'équivalence marquée historique, AC-E.7, AC-3.7.3 retiré, §10.
- `epic-3-context.md`, CHANGELOG, et l'entrée de `deferred-work.md`, marquée traitée.

**Vérification** :
- TDD : le nouveau test échouait avant le changement (1 échec, 20 réussis).
- `pytest test/pad_profile` : 106 réussis. Suite complète : 250 réussis.
- black, isort et flake8 ne signalent rien sur les 4 fichiers Python.

## Review Triage Log

Revue Blind Hunter du 24/11/2026 : 13 remarques. Après les corrections, la suite complète passe (250 tests), et black et flake8 ne signalent rien.

| # | Remarque | Verdict | Justification et suite donnée |
|---|---|---|---|
| 1 | Deux entrées de `deferred-work.md` (spec 3.2) reposent encore sur le plafond ou sur l'AC-3.7.3 | low | **Corrigé** : la condition d'équivalence est marquée « sans objet », et la validation de la config est rattachée à sa lecture (3.7). |
| 2 | Le tri n° 2 de la spec 3.5 (« profil inf ou négatif inatteignable grâce au plafond ») n'est plus valable | low | Rejeté. Un profil entièrement inf ou négatif demande `ground_margin ≥ dz` ou NRD ≥ couverture sur toutes les strates. R le traite de la même façon (profil nul) : la parité est gardée. |
| 3 | « PAD ≥ 0 par construction » est trop fort (`ground_margin > dz`, formule de couverture, inf impossible dans la formule simple) | low | **Corrigé** : condition `ground_margin < dz` dans l'épique et dans le commentaire du test ; la note de la spec est rectifiée (Laplace, inf ou NaN seulement avec la couverture). |
| 4 | La borne de la couverture `(0, 1)` n'est plus testée | low | **Corrigé** : `Cover_2` = 1,5 → 1, `Cover_4` = −0,2 → 0 dans le test d'export. |
| 5 | Le nouveau test ne vérifie pas le retrait de la borne basse | low | **Corrigé** : `PAD_1_1` = −0,5 est écrit tel quel. |
| 6 | Le test sur dalle réelle n'a plus de contrôle de valeur | low | Rejeté : `test_main_pad_profile` garde `nanmin >= 0` sur une dalle réelle. |
| 7 | Aucun test de parité du PAD avec R | false | Hors de l'intention : la parité de l'étape 2 se mesure en story 3.8 (AC-E.2). |
| 8 | Plus rien ne signale les PAD > 5 | low | Différé (`deferred-work.md`) : c'est une évolution utile pour le point technique, pas une régression (R ne les signale pas). |
| 9 | « 0–5 (plafonné à 5) » reste dans `epic-3-issue-github.md` et l'issue GitHub ; rien ne suit la correction de la spécification INRAE | low | Rejeté : `epic-3-issue-github.md` est l'instantané de l'issue au 13/11. L'issue GitHub et la spécification INRAE ne relèvent pas du dépôt (signalé à l'utilisatrice). |
| 10 | Statut incohérent (spec `in-progress`, documents qui disent « fait ») | false | La spec passe à `done` à la finalisation, dans le même commit que les documents. |
| 11 | Le CHANGELOG ne dit pas que les PAD des versions ≤ v0.3.0 sont plafonnés | low | **Corrigé**. |
| 12 | Style du test (commentaires plutôt que docstring, préfixe `PAD_0.5_` en dur, `StopIteration`) | low | Rejeté : les tests voisins mêlent les deux styles (l'ancien test n'avait pas de docstring). Un échec de `next` reste un échec net du test. |
| 13 | La note historique du §3.1 est au présent | low | **Corrigé** : passée au passé. |
