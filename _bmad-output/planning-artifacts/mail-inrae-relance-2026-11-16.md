**Objet :** LidarForFuel – ffuelmetrics2 : suite à vos réponses, quelques précisions (dont un point bloquant sur CBHThresholdRatio) et carte FMA

Bonjour,

Merci beaucoup pour vos réponses rapides. Voici ce que nous en retenons, puis les quelques points qu'il nous reste à préciser.

**Ce que nous retenons**

- **Q1** : NaN sur les 45 métriques quand le PAD du pixel est entièrement vide.
- **Q2** : vous corrigez les bandes CBD_cbh, CBD_cbh2 et CBD_cbhlow côté R. Le Python suivra votre correction.
- **Q3** : le seuil de 1 m dans la recherche du combustible en échelle est voulu. Nous le gardons tel quel.
- **Q5** : la PR #48 est mergée. Notre référence R est Dev `8fcccd2`.
- **Q6** : l'écart attendu est nul entre R et Python, à entrées et paramètres identiques.
- **Q7** : les 45 métriques sont calculées pour l'instant. La sélection des variables viendra plus tard.

**Point bloquant : CBHThresholdRatio (Q4)**

Vous indiquez que la valeur voulue est 0.333, celle du fichier de config. Le problème est que **R n'utilise pas cette valeur en mode raster**. `ffuelmetrics2Rast` ne transmet pas `CBHThresholdRatio` à `ffuelmetrics2`, qui applique donc sa valeur par défaut, 1/3. C'est aussi le cas de `smooth_pad`, qui reste toujours à FALSE, quelle que soit la config.

Ce point compte, car il est incompatible avec l'écart nul attendu (Q6). Sur la zone pilote HE_IGN_Galaxy, si on passe de 1/3 à 0.333 :
- CBH2 change sur **209 pixels**, avec des écarts allant jusqu'à **14 m** ;
- CanBDlowmax change sur **5 083 pixels**.

Les deux options possibles :
1. **Vous corrigez `ffuelmetrics2Rast`** pour qu'il transmette `CBHThresholdRatio` et `smooth_pad`. La valeur de référence devient alors 0.333, et le Python l'utilise.
2. **R reste en l'état.** Le Python prend alors 1/3, la valeur que R calcule réellement, pour garantir l'écart nul.

→ **Quelle option retenez-vous ?** En attendant votre réponse, nous développons avec 1/3 : c'est ce qui reproduit la production R actuelle. Le paramètre est de toute façon exposé dans la config.

**Autres précisions**

- **Q1 bis, NaN dans R** : allez-vous aussi mettre les pixels sans donnée à NaN dans R ? Sinon, ces pixels (223 sur la zone pilote) feront toujours un écart entre R et Python. Nous les exclurons alors explicitement de la comparaison.
- **Q2 bis, correction des CBD_cbh\*** : pouvez-vous nous indiquer le commit de la correction quand il sera fait ? Nous voulons vérifier deux choses :
  - la densité est bien prise après le recalcul de la CBD selon CBH2 ;
  - elle est prise sur la strate dont le **bas** vaut CBH, CBH2 ou CBHlow.

  Cette correction fera évoluer notre version R de référence.
- **Q6 bis, définition de « écart 0 »** :
  - Sur l'étape 3 seule, appliquée à votre PAD, notre prototype reproduit R sur 40 401 pixels × 45 métriques. L'écart relatif maximal est de 5,6·10⁻⁶, dû à l'arrondi float32. Considérez-vous que cela vaut 0, ou attendez-vous des valeurs identiques au bit près ?
  - Comprenons-nous bien que l'exigence d'écart nul vaut aussi pour la chaîne complète, étapes 1 et 2 comprises, à MNT, nuage et trajectoires identiques ?
- **Q3, pour la réunion technique** (non bloquant) : avec `CanopyHeightThreshold` = 1, il n'y a aucun effet. Mais si ce seuil passe à 2 ou 3 m, LadderBot peut se situer entre 1 m et CHT, alors que LadderLoad ne compte que les strates au-dessus de CHT. Est-ce le comportement attendu ?
- **Q8, validation** : qui validera la chaîne Python côté INRAE, et à quelle date ? Le code sera prêt mi-décembre, et la production nationale démarre en janvier 2027. La validation tombera donc probablement pendant les fêtes.
- **Q9, carte FMA** (nouveau point) : comme R accepte `FMAcan` et `FMAshrub` en `SpatRaster`, le Python accepte maintenant, pour chacun, soit une valeur, soit la carte FMA (1 bande, 10 m, valeurs de 0,05 à 0,5 kg/m²). Quatre questions :
  1. **Usage en production** : à partir de quand la carte remplacera-t-elle la valeur 0.2543576 de la config ? La même carte sert-elle pour `FMAcan` et pour `FMAshrub` ?
  2. **Plafond du PAD** : avec la carte, le plafond du PAD à 5 crée un écart avec R sur les pixels dont la FMA est inférieure à 0,2. Voir Q10 ci-dessous.
  3. **Pixels sans FMA** (NoData, ou hors de la carte) : faut-il mettre NaN sur les 45 métriques, comme pour Q1, ou utiliser une valeur par défaut (0.2543576) ? Et si une valeur de la carte sort de [0,05 ; 0,5], faut-il bloquer la dalle (c'est notre choix actuel) ou seulement le signaler ?
  4. **Grille** : la carte est-elle calée sur l'origine réelle de CosiaFrance (98 039,69 ; 7 111 486,70) ? Nos rasters PAD, comme la sortie R, sont calés sur l'origine arrondie (98 039,75 ; 7 111 486,75), soit un décalage de 6 cm. Nous prenons pour chaque pixel PAD le pixel FMA le plus proche, ce qui revient à la même valeur.
- **Q10, plafond du PAD à 5 : votre spécification et le code R divergent** (nouveau point) :
  - **La spécification** des profils PAD que vous nous avez transmise indique, pour chaque bande `PAD_*` : « valeurs limites à contrôler : 0–5 (plafonné à 5) ». Le Python applique donc ce plafond à l'étape 2 : toute valeur de PAD supérieure à 5 est écrite à 5.
  - **Le code R**, lui, ne plafonne pas le PAD (`pad_metrics`) : nous avons observé jusqu'à 7,01 sur la zone pilote.

  Conséquences pour l'écart nul attendu (Q6) :
  - **à l'étape 2**, les rasters PAD de R et du Python diffèrent sur toutes les strates où le PAD dépasse 5 ;
  - **à l'étape 3**, il n'y a aucun effet avec la FMA actuelle (0.2543576), car 5 × 0,254 ≥ `bdmax` = 1 : la CBD est plafonnée à 1 dans les deux cas. En revanche, avec la carte FMA (Q9), l'écart apparaît dès que la FMA est inférieure à 0,2, par exemple 0,14 pour *Fraxinus* ou *Castanea* (Martin-Ducup et al. 2025). Ce seuil vaut `bdmax` / 5 = 1 / 5 : en dessous, un PAD plafonné à 5 ne suffit plus pour que la CBD atteigne `bdmax`, alors que le PAD non plafonné de R peut l'atteindre. Par exemple, avec un PAD de 6 et une FMA de 0,14, R donne une CBD de 0,84 et le Python 0,70.

  → **Laquelle des deux fait référence ?**
  1. **La spécification** : le PAD est plafonné à 5, et vous ajoutez ce plafond dans R.
  2. **Le code R** : le PAD n'est pas plafonné, nous retirons le plafond du Python et la spécification est corrigée.
  3. **« À contrôler »** signifie seulement signaler les valeurs hors de [0 ; 5], sans les modifier.

Merci encore. Nous sommes disponibles pour en parler en réunion technique si c'est plus simple.

Bien cordialement,

Malvina
