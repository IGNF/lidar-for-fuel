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

### Review Findings

Revue de code du 24/11/2026 (PR #61, `db82e10..ee5e973`) : Blind Hunter, Edge Case Hunter, Verification Gap (aucun manque) et Acceptance Auditor.

- [x] [Review][Defer] Un seul compte pour trois cas dans `cap_pad` [lidar_for_fuel/fuel_metrics/prepare_cbd_profile.py:118] — deferred: le log par dalle naît en 3.6 : l'appelant comptera les +inf avant `cap_pad`, sans changer sa signature d'ici là. `cap_pad` additionne les valeurs juste au-dessus de 5 (post-traitement normal), les +inf (anomalie A6, qui fait passer la CBD de 0 à min(5 × FMA, `bd_max`)) et les négatifs. L'AC-3.6.8 réécrite ne compte plus à part les inf et les négatifs : ils sont noyés dans le compte du plafond. Le n° 10 du premier triage a rejeté le mélange dans le log, mais pas la perte du compte A6 dans l'AC-3.6.8.
- [x] [Review][Patch] La condition d'équivalence du §3.1 est encore dite « sans objet » à plusieurs endroits [docs/epic-3-fuel-metrics.md:273] — titre de la note du §3.1 (« historique, sans objet depuis le 24/11/2026 ») ; début de l'AC-E.7 (l. 240) ; motif de l'AC-3.7.3 (l. 743, « n'a plus lieu d'être ») ; puce barrée de `epic-3-context.md` (l. 55, « sans objet… pas d'avertissement au démarrage »). Celui qui écrira la story 3.7 lira des consignes contradictoires.
- [x] [Review][Patch] Le plafond n'est pas listé comme divergence assumée là où l'épique l'exige [lidar_for_fuel/fuel_metrics/calculate_fuel_metrics.py:55] — la section « Assumed divergence from R » de `fuel_metrics_core` ne cite que D2 (AC-3.5.5) ; l'AC-3.6.3 (l. 705) n'exempte que « les divergences assumées en 3.5 », alors que l'AC-3.6.2 met `cap_pad` dans le chemin du raster. Le test de parité de la 3.6 n'a donc pas d'exemption documentée pour le plafond.
- [x] [Review][Patch] « Un PAD +inf donne CBD = `bd_max` » est faux si FMA < 0,2 [docs/epic-3-fuel-metrics.md:579] — CBD = min(5 × FMA, `bd_max`) (`compute_cbd`) ; même formule au §3.1 (l. 273, « donc CBD = `bd_max` »).
- [x] [Review][Patch] Date de la réponse INRAE fausse dans la docstring de `cap_pad` [lidar_for_fuel/fuel_metrics/prepare_cbd_profile.py:88] — « answer … (24/11/2026) » ; l'INRAE a répondu le 23/11 (`export_raster.py`, épique), le plafond a été retiré le 24/11.
- [x] [Review][Patch] `cap_pad` renvoie du float64 pour un raster float32 si `pad_max` est un scalaire NumPy [lidar_for_fuel/fuel_metrics/prepare_cbd_profile.py:131] — vérifié sous NumPy 2.4.3 : `np.clip(float32, 0.0, np.float64(5))` → float64, contre la docstring (« same dtype for a float raster »). Correction : `.astype(pad.dtype, copy=False)`.
- [x] [Review][Patch] Commentaire mal recoupé après la retouche [test/pad_profile/test_main_pad_profile.py:130] — une ligne de 117 caractères suivie du fragment `# values <= 1).`
- [x] [Review][Patch] Double ligne vide avant la nouvelle entrée [_bmad-output/implementation-artifacts/deferred-work.md:73]

**Rejeté** :
- `pad_max` NaN ou négatif (tout NaN, ou PAD négatif partout) — low : seule la valeur par défaut est utilisée, aucun appelant ; la validation est déjà dans `deferred-work.md` (story 3.7).
- `pad_max = None` (mode « sans plafond » prévu) → `TypeError` — low : ce mode n'existe pas encore, et l'erreur est bruyante.
- Entrée non 3-D ou 0-d non contrôlée — low : inatteignable, `src.read(indexes)` renvoie toujours (bandes, lignes, colonnes) ; il faudrait ajouter un garde.
- Raster sans bande PAD → erreur de `reshape` peu claire — low : `read_pad_heights` lève d'abord « The PAD raster has no band », et l'AC-3.6.9 impose une erreur explicite.
- Tableau masqué (`read(masked=True)`) dont le masque serait perdu — low : inatteignable, le nodata est NaN et aucune lecture masquée n'est prévue.
- `None.startswith` dans le test sur la dalle de référence — false : toutes les bandes de la dalle ont une description, et le test passe.
- CHANGELOG sans −inf ni NaN, et AC-E.1 « seule D2 » — false : −inf est une valeur négative, donc couverte ; l'AC-E.1 limite le test à `fuel_metrics_core` sans le plafond.
- CHANGELOG « the cap is now applied when the fuel metrics read » alors que rien n'appelle `cap_pad` — false : l'étape 3 n'est pas encore exécutable, donc aucun utilisateur ne s'en sert.
- « Q10 » retiré des commentaires du code — choix voulu (commit `29df3ab`).
- Pas de test `cap_pad` → `compute_cbd` sur un pixel +inf — low : chaque fonction est testée et l'enchaînement est trivial ; un tel test relève de l'appelant (3.6).
- Tests de cas limites et annotation `Sequence[str]` (noms `None`) — low : les noms sont validés avant par `read_pad_heights`, qui rejette `None`.
- Le test sur la dalle de référence serait une tautologie — false : il affirme seulement que la parité testée sur cette dalle n'est pas touchée, ce qu'il prouve.
- `PAD_MIN` en dur, et mode `pad_max: null` futur — false : ce mode n'existe pas ; l'entrée différée de la 3.7 le couvre.
- Pas de renvoi croisé dans `deferred-work.md` vers l'entrée Q10 — low : la nouvelle entrée la complète explicitement (n° 3 du premier triage).
- « Réalisé » cite le §6 alors que le diff modifie la Description — rejeté : la correction porterait sur la spec revue ; la puce Q10 du §6 est le compte rendu daté de la réponse INRAE du 23/11.
- Tableaux d'entrée « ≥ 0, non plafonné (Q10) » — false : ils décrivent le contenu du fichier, qui n'est pas plafonné.
- `epic-3-context.md`, « les valeurs non finies ou négatives passent à 0 » — false : c'est toujours ce que fait `prepare_pad_profile`.
