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
