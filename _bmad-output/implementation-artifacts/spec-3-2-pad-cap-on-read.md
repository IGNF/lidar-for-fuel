---
title: 'Plafond du PAD à [0, 5] à la lecture du raster pad_profile_1m (étape 3)'
type: 'feature'
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

**Problem :** depuis Q10 (24/11/2026), l'étape 2 écrit le PAD brut, comme R : `pad_profile_1m` peut contenir des valeurs au-dessus de 5 (jusqu'à 7,01 sur la zone pilote), voire inf ou négatives. La spécification PAD de l'INRAE donne pourtant « valeurs limites à contrôler : 0–5 (plafonné à 5) », et l'INRAE a indiqué qu'un éventuel plafond serait un post-traitement des rasters PAD avant l'étape 3.

**Approach :** ajouter dans la couche d'entrée de l'étape 3 une fonction qui, à la lecture des bandes de `pad_profile_1m`, vérifie que chaque bande est dans [0, 5] et ramène dans cet intervalle les valeurs qui en sortent, comme le faisait la sortie de l'étape 2 avant Q10. Le travail est fait dans une nouvelle branche, créée depuis `feat/ffuelmetrics2`.

</frozen-after-approval>

## Implementation Notes

**Décisions** :
- **Même plafond qu'avant Q10** : `np.clip(pad, 0, 5)`, comme `_band_array` d'`export_raster.py` avant `4fcd708`. NaN reste NaN : un pixel entièrement NaN reste un pixel sans donnée (D2). +inf devient 5 et −inf 0.
- **Emplacement** : `cap_pad` dans `prepare_cbd_profile.py` (couche d'entrée, à côté de `read_pad_heights` et `read_fma`). Le lecteur du raster (story 3.6) n'existe pas encore : la fonction sera appelée par lui, juste après la lecture des bandes. L'AC-3.6.2 de l'épique le précise.
- **Paramètre** : `pad_max`, 5 par défaut (constante `PAD_MAX`). Le PAD minimal est 0. L'exposition de `pad_max` dans la config, et la possibilité de désactiver le plafond, relèvent de la story 3.7 (lecture de la config, qui n'existe pas encore).
- **Signalé, pas masqué** : un `logger.info` donne, par bande, le nombre de valeurs ramenées dans [0, 5]. Rien n'est journalisé si aucune valeur n'est corrigée.
- **Écart avec R** : R ne plafonne pas. Avec la FMA de la config (0,254) et `smooth_pad = False`, le plafond ne change pas les métriques, car 5 × 0,254 ≥ `bd_max` = 1. Il les change si une FMA < 0,2 (carte FMA) ou si `smooth_pad = True`. Il change aussi les pixels où le PAD est +inf : 5 au lieu de 0 (A6). Les tests de parité appellent `fuel_metrics_core` sans ce plafond.

**Réalisé** :
- `prepare_cbd_profile.py` : constantes `PAD_MIN`, `PAD_MAX` et fonction `cap_pad(pad, band_names, pad_max=PAD_MAX)`.
- `test_prepare_cbd_profile.py` : 5 tests (plafond et log par bande, raster valide inchangé, `pad_max`, noms de bandes incohérents, dalle de référence R inchangée).
- Épique : AC-3.2.8, AC-3.6.2 (appel juste après la lecture), AC-3.6.8 (compte de `cap_pad`), AC-E.1, AC-E.7, condition d'équivalence du §3.1 (de nouveau valable pour ce plafond), §3.1 entrée, §6, §10. `epic-3-context.md`, CHANGELOG, docstring de `fuel_metrics_core`.

**Surprise** : la dalle de référence R ne dépasse pas 2,04 (le 7,01 de la zone pilote est sur d'autres dalles). Le test sur cette dalle vérifie donc qu'elle passe sans changement : le plafond ne touche pas la parité testée.

**Vérification** :
- TDD : les tests échouaient avant l'implémentation (`ImportError`).
- Suite complète : 255 réussis. black, isort et flake8 ne signalent rien.

## Review Triage Log

Revue Blind Hunter du 24/11/2026 : 13 remarques.

| # | Remarque | Verdict | Justification et suite donnée |
|---|---|---|---|
| 1 | Plafond actif par défaut, sans interrupteur, alors que l'INRAE fait de R la référence | low | Différé (`deferred-work.md`, story 3.7) : le plafond à 5 par défaut est la demande de l'utilisatrice ; l'interrupteur (`pad_max: null`) relève de la lecture de la config, qui n'existe pas encore. |
| 2 | Le plafond n'est pas présenté comme une divergence assumée | medium | **Corrigé** : AC-3.2.8, note du §3.1 et CHANGELOG le disent, à faire valider par l'INRAE. |
| 3 | AC-E.7, note d'équivalence du §3.1 et AC-3.7.3 disent qu'il n'y a plus de plafond | low | **Corrigé** pour AC-E.7 et le §3.1 ; le retour de l'avertissement (AC-3.7.3) est différé avec le n° 1. L'ancienne entrée de `deferred-work.md` n'est pas modifiée (règle du workflow), la nouvelle la complète. |
| 4 | Docstring de `fuel_metrics_core` incomplète | low | **Corrigé** : elle renvoie à `cap_pad`. |
| 5 | +inf → 5 contredit la note « inf mis à 0 » de la spec Q10 | low | **Corrigé** dans la documentation (§3.1, AC-3.2.8) : c'est le comportement d'avant Q10, demandé, signalé comme divergence. |
| 6 | Le log de `prepare_pad_profile` ne verra plus les inf ni les négatifs | low | **Corrigé** : AC-3.6.8 compte les corrections de `cap_pad` à part. |
| 7 | `pad_max` négatif ou NaN non validé | low | Rejeté pour l'instant : seule la valeur par défaut est utilisée ; la validation est différée avec la config (n° 1). |
| 8 | « same dtype » faux pour une entrée entière ; entrée 0-d | low | **Corrigé** pour la docstring (« for a float raster ») ; l'entrée 0-d n'est pas atteignable (raster lu par rasterio). |
| 9 | La borne basse double celle de `prepare_pad_profile` | false | Le plafond [0, 5] est la demande (« comme le faisait l'étape 2 ») ; les deux comptes sont maintenant distingués (n° 6). |
| 10 | Le log mélange > 5, inf et négatifs, sans nom de dalle | low | Rejeté : le compte par bande suffit à repérer les cas ; le nom de dalle est ajouté par l'appelant (story 3.6). |
| 11 | Rien n'appelle encore `cap_pad` | false | Le lecteur (story 3.6) n'existe pas ; l'AC-3.6.2 impose l'appel juste après la lecture. |
| 12 | Tests manquants (`pad_max` invalide, effet sur les métriques, raster Python) | low | Rejeté : l'effet sur les métriques passe par `compute_cbd`, déjà testé ; un test sur raster relève de la 3.6. |
| 13 | Spec incomplète, « vérifie » sans contrôle bloquant | false | La spec est complétée à la finalisation ; « vérifier » = repérer, corriger et journaliser, comme demandé (« post-traiter »). |
