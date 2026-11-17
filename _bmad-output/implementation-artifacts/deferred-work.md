# Travail reporté

- source_spec: `_bmad-output/implementation-artifacts/spec-3-2-prepare-cbd-profile.md`
  summary: Corriger l'exemple `[0.01, 0.02, 0.05, 0.1] → 0.02` de l'épique 3 (AC-3.2.6 et §4 étape 9) en `→ 0.05`.
  evidence: Vérifié avec Rscript, `which.min(abs(s - median(s)))` renvoie 0.05 pour les deux ordres de liste, car il n'y a pas d'égalité exacte en double précision. Le code et les tests suivent R. Comme l'épique est une spec, sa correction revient à l'utilisatrice.
- source_spec: `_bmad-output/implementation-artifacts/spec-3-2-prepare-cbd-profile.md`
  summary: Valider les paramètres scientifiques de la section `fuel_metrics` de la config au démarrage : FMA > 0 et finies, `bd_max` > 0, seuils non vides, finis et positifs (story 3.7).
  evidence: Le périmètre de la story 3.2 mentionne une « validation des paramètres ». Pourtant, `select_default_ladder_threshold([nan, …])` renvoie le premier seuil sans erreur, et `compute_cbd` accepte des FMA négatives. Ces valeurs viennent de la config, que la story 3.7 lit et contrôle déjà (avertissement d'équivalence AC-3.7.3).
- source_spec: `_bmad-output/implementation-artifacts/spec-3-2-prepare-cbd-profile.md`
  summary: Décider en story 3.4 quoi faire si le premier `h_bottom` n'est pas 0, puisque `FL_k1` et `FL_k2` supposent que la première strate commence au sol.
  evidence: `parse_pad_strata` vérifie que le profil est régulier (même `dz`, ni doublon ni trou), mais pas son point de départ. R calcule `FL_k1 = CBD[1]·dz` sans le vérifier. Imposer 0 dès la story 3.2 risquerait de diverger de R avec un `z0` non nul.
- source_spec: `_bmad-output/implementation-artifacts/spec-3-2-prepare-cbd-profile.md`
  summary: FAIT le 17/11/2026 : l'exemple de l'épique est corrigé (AC-3.2.6 et §4 étape 9), il donne maintenant `[0.01, 0.02, 0.05, 0.1]` → 0.05, et deux cas d'égalité exacte sont ajoutés.
  evidence: Il remplace le premier élément de cette liste.
- source_spec: `_bmad-output/implementation-artifacts/spec-3-2-prepare-cbd-profile.md`
  summary: Story 3.7 (AC-3.7.3, AC-E.7) — revoir la condition d'équivalence avec R `5 × min(fma_can, fma_shrub) ≥ bd_max` quand une FMA est la carte : elle doit porter sur le minimum de la carte, ou être évaluée par pixel.
  evidence: La carte FMA va de 0,05 à 0,5 kg/m². Pour un pixel où FMA < 0,2, on a 5 × FMA < 1 = `bd_max`. Le PAD Python plafonné à 5 donne alors une CBD plus faible que le PAD R non plafonné (maximum observé 7,01). Il y a donc un écart avec R sur ces pixels si PAD > 5, en contradiction avec Q6, qui demande un écart nul. Avec la valeur scalaire 0,254 de la config, la condition reste vraie.
