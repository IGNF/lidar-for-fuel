---
title: 'Story 3.7 — Point d''entrée main_fuel_metrics, configuration et documentation'
type: 'feature'
created: '2026-11-25'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: '6696e3033f8b71a87746d9fa7a9b8c3249175966'
context:
  - '{project-root}/_bmad-output/implementation-artifacts/epic-3-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem :** l'étape 3 n'a qu'une API Python (`export_fuel_metrics`). On ne peut pas encore la lancer comme les étapes 1 et 2, en ligne de commande sur un fichier ou sur un dossier, avec ses paramètres dans `configs/config.yaml`. Une config anormale (FMA hors plage, seuils en double, `pad_max` absurde) produit aujourd'hui des cartes fausses sans erreur.

**Approach :** un point d'entrée Hydra `lidar_for_fuel/main_fuel_metrics.py`, calqué sur `main_pad_profile.py`, qui passe la section `fuel_metrics` à `export_fuel_metrics`. Les paramètres sont validés au démarrage, avant la première dalle. On y ajoute le script d'exemple, le README et le CHANGELOG. Critères de l'épique : AC-3.7.1, 3.7.2, 3.7.4, 3.7.5 (l'AC-3.7.3 est retirée). On solde au passage les reports vers la 3.7 de `deferred-work.md`.

**Décisions (25/11/2026) :**
- **Version** : `_version.py` ne change pas dans cette story. La montée de version se fera dans un commit séparé à la fin de l'épique, après la 3.8. L'AC-3.7.4 est amendée dans l'épique (`_version.py` retiré, avec la date).
- **`pad_max`** : un nombre fini > 0, 5 par défaut. `null` est refusé, il n'y a pas d'interrupteur pour couper le plafond, et `cap_pad` ne change pas.
- **Taille** : la spec dépasse la cible de 1600 tokens. On la garde entière, car elle ne sert qu'un objectif.

## Boundaries & Constraints

**Always :**
- Messages d'erreur sur `input_dir` et `output_dir` identiques à ceux de `main_pad_profile.py` (AC-3.7.2).
- Seuls les fichiers `*_pad_profile_1m.tif` du dossier sont traités : l'étape 2 écrit 8 rasters par dalle dans le même dossier. `io.input_filename`, s'il est renseigné, est traité seul.
- La dalle est journalisée avant le calcul (`Processing tile : …`), comme à l'étape 2.
- Une erreur de config lève une `ValueError` qui nomme la clé, avant le traitement de toute dalle.
- FMA donnée par valeur : finie et dans [`FMA_MAP_MIN`, `FMA_MAP_MAX`] = [0,05 ; 0,5], comme la carte. Une carte FMA contenant ±inf est hors plage.
- Nombre de bandes jamais codé en dur, sauf dans le test d'intégration de la config par défaut (45).

**Never :**
- Ne pas modifier `fuel_metrics_core`, ni la signature de `compute_fuel_metrics_raster`, ni le calcul.
- Pas de parallélisation, ni de reprise sur erreur : comme l'étape 2, une dalle en erreur arrête le lancement.
- Pas d'avertissement d'équivalence avec R (ancien AC-3.7.3) : à décider avec l'INRAE, il reste reporté.
- Ne pas traiter ici les 6 actions de la revue de la 3.6 (spec 3.6, `in-progress`).

## I/O & Edge-Case Matrix

| Scénario | Entrée | Résultat attendu | Erreurs |
|---|---|---|---|
| Dossier | `io.input_dir` avec 2 `*_pad_profile_1m.tif` et d'autres rasters de l'étape 2 | 2 fichiers `{stem}_fuel_metrics.tif` | — |
| Un fichier | `io.input_filename` renseigné | 1 fichier | — |
| `input_dir` absent ou nul | — | — | messages de `main_pad_profile.py` |
| `output_dir` nul | — | — | message de `main_pad_profile.py` |
| FMA valeur | NaN, inf, 0, −0,1, 0,6 | — | `ValueError` (`fma_can` ou `fma_shrub`) |
| Carte FMA | un pixel +inf | — | `ValueError` « outside [0.05, 0.5] » |
| Seuils | vides, non finis, ≤ 0, ou deux seuils de même nom de bande (`0.01`/`0.01004` en CTH, `0.0101`/`0.0102` en échelle) | — | `ValueError` qui nomme la clé |
| `bd_max`, `pad_max` | ≤ 0, NaN, inf, `null` | — | `ValueError` |

</frozen-after-approval>

## Code Map

- `lidar_for_fuel/main_pad_profile.py` -- modèle à suivre : `@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")`, `logging.basicConfig`, contrôles `io.input_dir` et `io.output_dir`, `main_on_one_tile`, boucle sur `os.listdir`. À ne pas modifier.
- `lidar_for_fuel/fuel_metrics/export_fuel_metrics.py` -- `export_fuel_metrics(pad_path, output_dir, fma_can, fma_shrub, smooth_pad, ladder_fuel_bd_thresholds, canopy_height_threshold, canopy_bd_thresh_frac, canopy_top_bd_thresholds, cbh_threshold_ratio, bd_max, m_perrakis, pad_max=PAD_MAX) -> Path`. Les clés de `fuel_metrics` portent déjà ces noms ; seul `pad_max` manque.
- `lidar_for_fuel/fuel_metrics/prepare_cbd_profile.py` -- `read_fma` (l. 134) : la valeur est renvoyée sans contrôle (l. 167, « no check here ») ; la carte est contrôlée sur `values[np.isfinite(values)]` (l. 197), ce qui laisse passer ±inf. À adapter sur place : contrôle de la valeur, et `~np.isnan` sur la carte. `PAD_MAX`, `FMA_MAP_MIN`, `FMA_MAP_MAX` (l. 29-34). `cap_pad` (l. 85).
- `lidar_for_fuel/fuel_metrics/compute_ladder_metrics.py:79` `ladder_suffix` et `compute_canopy_metrics.py:173` `_cth_name` -- noms de bande des seuils. Le contrôle des doublons les réutilise (ensemble des noms plus petit que la liste ⇒ erreur).
- `configs/config.yaml` -- section `fuel_metrics:` déjà conforme à l'AC-3.7.1 ; il faut ajouter `pad_max: 5`. `io.input_filename`, `io.input_dir`, `io.output_dir` existent.
- `test/fuel_metrics/test_export_fuel_metrics.py` -- `_PARAMS`, `_REF_PAD_RASTER_1M`, écriture des `PAD_1_*` de la référence dans `tmp_path` (`test_export_fuel_metrics_reference_tile_equals_r`) : à reprendre pour l'intégration. Le raster R (186 bandes) n'est pas une entrée valide.
- `test/fuel_metrics/test_prepare_cbd_profile.py:259-312` -- tests de `read_fma` à compléter.
- `scripts/example_launch_pad_profile.sh`, `README.md` (aucune section par étape aujourd'hui ; table des bandes : `docs/epic-3-fuel-metrics.md` §3.2), `CHANGELOG.md` (`# not yet released`), `lidar_for_fuel/_version.py` (`0.3.0`).
- Hydra 1.3.2 : un `main` décoré par `@hydra.main` accepte une `DictConfig` passée directement. On l'obtient par `hydra.initialize(config_path=…)` + `compose(overrides=[…])`.
- Environnement : `conda run -n lidar_for_fuel …` pour pytest, black, isort et flake8, jamais `uv run`. Pas d'attribution IA dans les commits.

## Tasks & Acceptance

**Execution :**
- [x] `test/fuel_metrics/test_main_fuel_metrics.py` -- TDD (comme `test/pad_profile/test_main_pad_profile.py`) :
  - intégration : 2 tuiles `PAD_1_*` de référence et 1 raster étranger dans `tmp_path`, `main(compose(...))` avec la config par défaut, puis 2 sorties à 45 bandes (AC-3.7.5) ;
  - `io.input_filename` traité seul ;
  - messages d'erreur sur `io` ;
  - les clés de `fuel_metrics` sont exactement les paramètres d'`export_fuel_metrics` (report de la 3.4).
- [x] `lidar_for_fuel/main_fuel_metrics.py` -- `main` Hydra, filtre `*_pad_profile_1m.tif`, validation des paramètres au démarrage, appel `export_fuel_metrics(path, output_dir, **params)` avec les listes converties (`OmegaConf.to_container`).
- [x] `lidar_for_fuel/fuel_metrics/validate_fuel_metrics_params.py` -- `validate_fuel_metrics_params(**params)` : les cas de la matrice, sauf les FMA ; tests dans `test/fuel_metrics/test_validate_fuel_metrics_params.py`.
- [x] `lidar_for_fuel/fuel_metrics/prepare_cbd_profile.py` + tests -- `read_fma` : valeur contrôlée (finie, dans la plage), carte ±inf hors plage.
- [x] `test/fuel_metrics/test_export_fuel_metrics.py` -- `pad_max=2.0` passé par `export_fuel_metrics` : comptes et valeurs plafonnées (report de la revue 3.6).
- [x] `configs/config.yaml` -- `pad_max: 5`.
- [x] `scripts/example_launch_fuel_metrics.sh`, `README.md` (lancement des étapes 2 et 3, contenu de la sortie, lien vers le tableau des bandes), `CHANGELOG.md` (sous `# not yet released`) -- AC-3.7.4 ; `_version.py` n'est pas modifié.
- [x] `docs/epic-3-fuel-metrics.md` -- AC-3.7.4 amendée : `_version.py` barré, monté à la fin de l'épique (25/11/2026).
- [x] `deferred-work.md` -- marquer « Traité » les reports soldés ; ceux sur `pad_max: null` et sur l'avertissement d'équivalence restent ouverts (`null` refusé le 25/11).

**Acceptance Criteria :**
- Étant donné la config par défaut, quand on lance `python -m lidar_for_fuel.main_fuel_metrics io.input_dir=… io.output_dir=…`, alors les paramètres sont validés une fois, chaque dalle est journalisée puis écrite, et le code de sortie est 0.
- Étant donné la suite complète, quand on la lance, alors elle est verte, et black, isort et flake8 ne signalent rien sur les fichiers touchés.

## Implementation Notes

## Spec Change Log

## Review Triage Log

Revue du 26/11/2026 : Blind Hunter (BH), Edge Case Hunter (ECH), Verification Gap (VG).

| # | Source | Remarque | Verdict | Preuve et suite |
|---|---|---|---|---|
| 1 | BH, ECH, VG | `fma_can: null` (ou une liste) → `TypeError` de `float()` dans `read_fma`, non intercepté, qui ne nomme pas la clé | low | Reproduit par VG. C'est contraire à la règle Always « `ValueError` qui nomme la clé ». **Patch** : intercepter aussi `TypeError` dans `main`. |
| 2 | BH, ECH | Chemin de carte FMA absent ou mal écrit (ou nombre entre guillemets) : non contrôlé au démarrage, `RasterioIOError` à la première dalle, sans la clé | low | La boucle saute toute `str`. **Patch** : vérifier au démarrage que le fichier existe, `ValueError` qui nomme la clé. |
| 3 | BH, ECH | Dossier sans `*_pad_profile_1m.tif` : sortie 0, aucun fichier, aucun message | low | Cas atteignable (mauvais dossier, étape 2 pas encore lancée). **Patch** : `logger.warning` si aucun fichier ne correspond. |
| 4 | VG, BH | Rien ne vérifie que `main` transmet un `pad_max` autre que 5 (ni les autres paramètres) à `export_fuel_metrics` | medium | Pré-vérifié : le test d'intégration tourne avec 5, le test des clés ne regarde que les noms. **Patch** : test `main` avec `fuel_metrics.pad_max=2` et contrôle du log de fin de dalle. |
| 5 | BH | README : on ne dit pas que `io.input_filename` est un nom de fichier dans `io.input_dir` | low | **Patch** : une phrase. |
| 6 | ECH | README : « The parameters are checked before the first tile » est faux pour les valeurs d'une carte FMA (contrôlées dalle par dalle) | low | **Patch** : préciser. |
| 7 | ECH | Carte FMA hors plage seulement sur une dalle tardive : arrêt en cours de lancement, et l'erreur ne nomme ni la clé ni la dalle | low | L'arrêt est voulu (Never : pas de reprise). Le nom de la dalle est déjà l'action n° 3 de la revue 3.6 (préfixe `PAD raster {path}:`). **Defer**. |
| 8 | BH | Validation de la FMA éclatée (`read_fma` sur une grille 1 × 1 fictive dans `main`) | low | Rejeté : pas de dommage nommé ; un helper partagé serait un refactor. |
| 9 | BH | `pad_max` obligatoire dans le validateur, alors qu'il a une valeur par défaut dans l'exporteur : `TypeError` si la clé manque | low | Rejeté : la config du dépôt contient la clé (test des clés), et l'erreur nomme `pad_max`. |
| 10 | BH, ECH | Autres paramètres non contrôlés (`canopy_bd_thresh_frac`, `cbh_threshold_ratio`, CHT, `m_perrakis`, `smooth_pad` en chaîne, seuils > `bd_max`) | low | Rejeté : hors de la matrice figée, peu probable, et l'ajout demande de nouveaux gardes. |
| 11 | BH | Le CLI réel n'est pas testé ; le script n'a ni shebang ni mode exécutable | low | Rejeté : `example_launch_pad_profile.sh` a la même forme. Le CLI a été lancé à la main (code de sortie 0). |
| 12 | BH | Le changement de `read_fma` n'est pas marqué comme cassant | low | Rejeté : section `not yet released`, et aucun appelant externe. |
| 13 | BH | Implementation Notes vides dans la spec | — | Rejeté : la correction modifierait la spec de ce build. |
| 14 | ECH | Clé inconnue dans `fuel_metrics` (`+fuel_metrics.foo=1`) acceptée, puis `TypeError` à la première dalle | low | Rejeté : il faut un `+` explicite avec Hydra ; c'est peu probable, et l'erreur est bruyante. |
| 15 | ECH | `io.output_dir=""` → `FileNotFoundError` brute de `os.makedirs` | false | Même comportement que `main_pad_profile.py`, ce que la spec impose (messages identiques). |

## Verification

**Commands :**
- `conda run -n lidar_for_fuel python -m pytest -q test/fuel_metrics` -- attendu : tout est vert.
- `conda run -n lidar_for_fuel python -m pytest -q` -- attendu : suite complète verte.
- `conda run -n lidar_for_fuel bash scripts/example_launch_fuel_metrics.sh` -- attendu : un `*_fuel_metrics.tif` écrit (sur des données de test).
