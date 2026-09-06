# Prédiction de notes d'examen — Kaggle Playground Series 2026

Projet de modélisation statistique (ENSIIE). Régression sur données tabulaires :
prédire la note finale d'étudiants à partir de variables comportementales.

**Résultat : RMSE de 8.869 (baseline linéaire) à 8.667 — top 20 % du classement.**

---

## Les données

630 000 observations, douze variables explicatives, une cible continue.

| Type | Variables |
|---|---|
| Numériques | `age`, `study_hours`, `class_attendance`, `sleep_hours` |
| Catégorielles ordinales | `sleep_quality`, `exam_difficulty`, `facility_rating` |
| Catégorielles nominales | `gender`, `course`, `internet_access`, `study_method` |
| Cible | `exam_score` |

Statistiques descriptives sur l'ensemble d'entraînement :

| Variable | Min | Médiane | Max | Moyenne | Écart-type |
|---|---|---|---|---|---|
| `exam_score` | 19.6 | 62.6 | 100.0 | 62.5 | 18.9 |
| `study_hours` | 0.08 | 4.0 | 7.9 | 4.0 | 2.4 |
| `class_attendance` | 40.6 | 72.6 | 99.4 | 72.0 | 17.4 |
| `sleep_hours` | 4.1 | 7.1 | 9.9 | 7.1 | 1.7 |
| `age` | 17 | 21 | 24 | 20.6 | 2.3 |

Les données sont **synthétiques**, générées par un modèle de deep learning
entraîné sur un jeu réel. Elles sont donc exceptionnellement propres — aucune
valeur manquante, aucun doublon — tout en conservant des interactions non
triviales entre variables. La difficulté du projet n'était pas le nettoyage,
mais la modélisation.

Métrique d'évaluation : **RMSE**, en points de note.

---

## Exploration

### Audit de qualité

- Zéro valeur manquante, zéro doublon sur 630 000 lignes
- Comparaison des modalités catégorielles entre `train` et `test` : aucune
  modalité fantôme, c'est-à-dire présente en test mais absente de l'entraînement
- Vérification du domaine de définition de la cible

### Un plafonnement à 100

**2,5 % des observations valent exactement 100** — un artefact du plafonnement de
la génération synthétique. La cible est bornée dans `[0, 100]`.

Conséquence directe : les modèles linéaires n'étant pas contraints par nature,
ils peuvent prédire au-delà de ces bornes. Toutes les prédictions sont donc
**recadrées** dans l'intervalle avant soumission.

### Le prédicteur dominant

Corrélations avec `exam_score` :

| Variable | Corrélation |
|---|---|
| `study_hours` | **0.762** |
| `class_attendance` | 0.361 |
| `sleep_hours` | 0.167 |
| `age` | 0.010 |

Les heures d'étude dominent nettement. L'âge n'apporte rien.

---

## Prétraitement (R)

### Encodage différencié selon la nature de la variable

| Type | Traitement | Justification |
|---|---|---|
| Ordinales | Encodage numérique croissant — `poor` = 1, `average` = 2, `good` = 3 | Préserve la hiérarchie réelle entre modalités |
| Nominales | One-hot | Évite d'introduire un ordre artificiel entre catégories |

Le one-hot est appliqué sur la **concaténation** de `train` et `test`, pour
garantir un nombre de colonnes identique des deux côtés.

### Prévention des fuites de données

Les paramètres de standardisation (µ, σ) sont calculés **exclusivement sur
l'ensemble d'entraînement**, puis appliqués tels quels au test. C'est ce qui
garantit que le modèle ne reçoit aucune information anticipée sur les données
d'évaluation.

---

## Progression des modèles

| Modèle | RMSE (Kaggle) |
|---|---|
| Régression linéaire (baseline) | 8.869 |
| Elastic Net | 8.870 |
| Lasso | 8.870 |
| Ridge | 8.956 |
| Arbre de décision | 9.614 |
| Random Forest | 8.911 |
| SVR (cuML sur GPU, échantillon 100k) | 9.100 |
| LightGBM | 8.773 |
| CatBoost | 8.780 |
| XGBoost | 8.724 |
| Moyenne de 2 XGBoost (graines différentes) | 8.715 |
| Stacking de 7 modèles | 8.704 |
| **+ Feature engineering** | **8.683** |
| Stacking final + Optuna | **8.667** |

---

## Quatre résultats qui méritent explication

### 1. La régularisation n'apporte rien

Ridge, Lasso et Elastic Net n'améliorent pas le modèle naïf — Ridge le dégrade
même légèrement.

Ce n'est pas un échec d'implémentation. La régularisation réduit la **variance**
de l'estimateur au prix d'un peu de biais, ce qui est utile lorsqu'on dispose de
beaucoup de variables pour peu d'observations. Ici c'est exactement l'inverse :
630 000 lignes pour douze variables. L'estimateur des moindres carrés a déjà une
variance quasi nulle et une stabilité extrême.

> Il n'y a pas de variance à réduire. Pénaliser n'ajoute qu'un biais inutile.

### 2. L'arbre de décision est pire que le modèle linéaire

9.614 contre 8.869, ce qui surprend au premier abord.

L'explication tient à la nature de la cible. Un arbre prédit par **paliers
constants** — une valeur unique par feuille — produisant une fonction en escalier
mal adaptée à une distribution de notes continue et lisse. La droite de
régression, malgré sa rigidité, colle mieux.

### 3. Le SVR échoue par sacrifice de données

Le SVR a une complexité d'entraînement entre O(n²) et O(n³). Sur 630 000 lignes,
l'entraînement CPU sous R saturait la mémoire.

Le passage à **cuML** (RAPIDS, exécution GPU) avec un encodage `float32` n'a pas
suffi : il a fallu sous-échantillonner à **100 000 lignes**, soit 16 % des données.

Le résultat de 9.100 ne mesure donc pas la faiblesse du SVR en soi, mais le coût
d'avoir renoncé à 84 % de l'information disponible.

### 4. Le feature engineering bat l'architecture ⭐

**C'est le résultat le plus important du projet.**

Un simple XGBoost enrichi de 13 variables construites à la main (8.683) surpasse
un stacking de 7 modèles entraîné sur données brutes (8.704).

Les variables ajoutées traduisent des intuitions métier :

**Interactions** (6) — l'effet d'une variable dépend du niveau d'une autre :
`study_x_sleep_quality`, `study_x_attendance`, `sleep_x_sleep_quality`,
`study_x_difficulty`, `attendance_x_internet`, et `triple_effort`
(le produit des trois facteurs d'effort).

**Termes quadratiques** (3) — rendements décroissants, étudier deux fois plus ne
rapporte pas deux fois plus : `study_hours_sq`, `class_attendance_sq`,
`sleep_quality_sq`.

**Ratios** (2) — `study_sleep_ratio` (effort rapporté à la récupération) et
`attendance_study_ratio` (assiduité rapportée au travail personnel).

**Scores composites** (2) — `total_learning` (temps d'apprentissage total) et
`env_score` (accès internet binarisé plus qualité des infrastructures).

> La qualité des données prime sur la sophistication du modèle.

---

## Le stacking, et pourquoi retirer un modèle faible dégrade le résultat

### Architecture

Sept modèles de niveau 0 : trois XGBoost avec des graines différentes (42, 2024,
1337), deux LightGBM (42, 2024), un CatBoost sur GPU, et un perceptron
multicouche construit sous TensorFlow/Keras — trois couches denses (256, 128, 64)
avec batch normalization et dropout (0.2 puis 0.1), entraîné sur 50 époques.

Méta-apprenant de niveau 1 : une **régression Ridge** (α = 1.0). Le choix d'un
modèle linéaire pénalisé est délibéré : les sept prédictions sont fortement
corrélées entre elles puisqu'elles visent la même cible, et Ridge gère cette
multicolinéarité en répartissant les poids sans surpondérer un modèle.

### Les prédictions out-of-fold sont indispensables

Si le méta-modèle était entraîné sur des prédictions produites par les modèles de
base **sur leurs propres données d'entraînement**, ces prédictions seraient
artificiellement excellentes. Le méta-modèle en conclurait que ces modèles ne se
trompent jamais et leur accorderait une confiance aveugle — qui s'effondrerait en
test, sur des données réellement nouvelles.

On découpe donc en 5 plis : les modèles s'entraînent sur 4 plis et prédisent sur
le cinquième. Chaque prédiction provient ainsi d'un modèle qui n'a jamais vu
l'observation concernée.

### L'expérience la plus instructive du projet ⭐

L'analyse des poids attribués par Ridge a révélé un comportement inattendu :
**CatBoost recevait un poids négatif** (−0.204) et le MLP un poids quasi nul
(−0.006).

Hypothèse naturelle : ces deux modèles apportent du bruit, autant les retirer.

La version 2 du stacking les a donc supprimés, ne gardant que six modèles à base
d'arbres. Résultat : le score s'est **dégradé**, 8.679 contre 8.675.

**L'explication.** La force d'un ensemble ne réside pas dans la performance
individuelle de ses membres, mais dans la **diversité de leurs erreurs**. Bien que
moins précis isolément, CatBoost et le MLP se trompaient *différemment* des
autres. En les retirant, on a augmenté la corrélation globale des prédictions —
XGBoost et LightGBM fonctionnant de façon très similaire — et privé Ridge des
nuances dont il avait besoin pour corriger les erreurs résiduelles.

Un poids négatif ne signifie donc pas « mauvais modèle » : il signifie que le
méta-apprenant s'en sert comme **terme de correction**.

Détail élégant : la somme des poids finaux vaut 1.0005. Ridge a spontanément
reconstruit une moyenne pondérée.

### Une variante testée sans succès

La version 3 remplaçait Ridge par un méta-XGBoost fortement bridé
(`max_depth = 3`, `min_child_weight = 10`, `gamma = 1.0`). Score : 8.678 — mieux
que la v2, mais toujours en dessous de Ridge. Un méta-apprenant arborescent tend
à surajuster les prédictions out-of-fold, là où une régression pénalisée se
contente de lisser.

---

## Optimisation bayésienne

La configuration finale utilise **Optuna** plutôt qu'une recherche sur grille.
Contrairement à cette dernière qui teste toutes les combinaisons aveuglément,
Optuna mémorise les essais précédents pour orienter les suivants vers les zones
prometteuses de l'espace des hyperparamètres.

Deux études de 20 essais chacune, évaluées en validation croisée à 3 plis pour
limiter le coût. Paramètres retenus pour XGBoost :

```python
learning_rate    = 0.049
max_depth        = 5
subsample        = 0.854
colsample_bytree = 0.843
gamma            = 0.994
tree_method      = "hist"
```

Le `tree_method="hist"` était nécessaire pour traiter 630 000 lignes dans un
temps raisonnable : il binne les variables continues en histogrammes, ce qui
transforme la recherche de split en un simple parcours de bins.

---

## Structure du dépôt

```
src/R/
  script.r                exploration, prétraitement, modèles linéaires (R)

src/python/
  decision_tree.py        arbre de décision
  random_forest.py        forêt aléatoire
  svr_gpu.py              SVR sur GPU via cuML
  lightgbm_gpu.py         LightGBM
  catboost_gpu.py         CatBoost
  xgb.py                  XGBoost
  2_xgb_mean.py           moyenne de deux XGBoost (graines différentes)
  feature_engineering.py  création des 13 variables
  xgb_fe.py               XGBoost sur données enrichies
  xgb_kfold_fe.py         validation croisée 5 plis
  2_xgb_mean_fe.py        moyenne de graines + FE
  stacking.py             stacking sur données brutes
  stacking_fe.py          stacking v1 avec FE
  stacking_fe_2.py        v2 — sans CatBoost ni MLP
  stacking_fe_3.py        v3 — méta-apprenant XGBoost
  stacking_fe_4.py        v4 — Optuna, configuration finale

submissions/              fichiers de soumission Kaggle
data/                     train et test (non versionnés)
```

Chaque script produit sa propre soumission, ce qui permet de retracer la
progression pas à pas.

---

## Limites connues

**`study_sleep_ratio` est numériquement instable.** Cette variable divise par
`sleep_hours`, qui a été standardisée et se trouve donc centrée sur zéro. Le
dénominateur peut s'approcher arbitrairement de zéro, produisant des valeurs qui
explosent et changent de signe de part et d'autre. La constante ε = 10⁻³ ajoutée
n'y change rien. Il aurait fallu calculer les ratios **avant** standardisation.

**Le RMSE out-of-fold du stacking est optimiste.** Le méta-modèle Ridge est
ajusté puis évalué sur la même matrice de prédictions, ce qui rend le 8.7113
in-sample. Seul le score Kaggle constitue une mesure honnête.

**Fuite mineure sur le MLP.** Le `StandardScaler` est ajusté sur l'ensemble des
données avant le découpage en plis, donc il « voit » les plis de validation.
L'impact est faible sur 630 000 lignes — moyenne et écart-type varient peu — mais
c'est méthodologiquement incorrect.

---

## Reproduire

```bash
# Partie R : exploration, prétraitement, modèles linéaires
Rscript src/R/script.r

# Partie Python : arbres, stacking, Optuna
git checkout branch_A
pip install -r requirements.txt
python src/python/stacking_fe_4.py
```

---

## Ce que le projet illustre

**La volumétrie dicte l'outil.** Sur 630 000 observations, les algorithmes à
complexité quadratique deviennent inutilisables et les arbres classiques trop
lents. La solution passe par le gradient boosting optimisé par histogrammes.

**La donnée prime sur l'algorithme.** Le levier le plus rentable n'a pas été
l'empilement de sept modèles, mais treize variables construites à partir
d'intuitions métier.

**La diversité vaut mieux que la performance individuelle.** Retirer les modèles
les moins précis a dégradé l'ensemble.

---

## Équipe

Projet réalisé en groupe de quatre dans le cadre de l'UE Modélisation
Statistique. Contribution personnelle : exploration des données, prétraitement en
R, modèles linéaires et régularisés.
