library(caret)
library(glmnet)

#Etude de données

train <- read.csv("../../data/train.csv")
test <- read.csv("../../data/test.csv")

print(paste("Train dimensions:", nrow(train), "lignes,", ncol(train), "colonnes"))
print(paste("Test dimensions:", nrow(test), "lignes,", ncol(test), "colonnes"))

#voir le type de variable
str(train)

#voir les premières lignes de train
head(train)

#résumé statistique complet
summary(train)

#visualisation simple de exam_score
hist(train$exam_score, 
     main = "Distribution des notes (Exam Score)", 
     xlab = "Note", 
     ylab = "Fréquence",
     col = "lightblue", 
     border = "black",
     breaks = 30) #breaks définit le nombre de barres

#chercher les valeurs manquantes
valeurs_manquantes <- colSums(is.na(train))

#afficher celles qui ont un des problèmes
print("Valeurs manquantes par colonne :")
print(valeurs_manquantes[valeurs_manquantes > 0])

#recherche des doublons
nb_doublons <- sum(duplicated(train))
print(paste("Nombre de lignes dupliquées :", nb_doublons))

#vérification de la cohérence
print(paste("Note min :", min(train$exam_score)))
print(paste("Note max :", max(train$exam_score)))

#TRAITEMENT DE DONNEES

#on convertit les variables textuelles en variables numériques

#1. les variables ordinales

#Fonction pour mapper les textes vers des chiffres
convert_ordinal <- function(df) {
  #Sleep Quality: Poor -> 1, Average -> 2, Good -> 3
  df$sleep_quality <- as.numeric(factor(df$sleep_quality, 
                                        levels = c("poor", "average", "good"), 
                                        ordered = TRUE))
  
  #Exam Difficulty: Easy -> 1, Moderate -> 2, Hard -> 3
  df$exam_difficulty <- as.numeric(factor(df$exam_difficulty, 
                                          levels = c("easy", "moderate", "hard"), 
                                          ordered = TRUE))
  
  #Facility Rating: Low -> 1, Medium -> 2, High -> 3
  df$facility_rating <- as.numeric(factor(df$facility_rating, 
                                          levels = c("low", "medium", "high"), 
                                          ordered = TRUE))
  return(df)
}

#On applique la transformation sur train et test
train <- convert_ordinal(train)
test <- convert_ordinal(test)

#2.Les variables nominales (Gender, Course, Study_method, Internet_access)

#On transforme simplement les chaines de caractères en "Facteurs"
cols_nominales <- c("gender", "course", "study_method", "internet_access")

for(col in cols_nominales) {
  train[[col]] <- as.factor(train[[col]])
  test[[col]] <- as.factor(test[[col]])
}

#Vérifier si toutes les variables sont factor ou num
str(train)

#NORMALISATION (Centrer-Réduire)

#On liste les colonnes purement numériques
vars_a_normaliser <- c("age", "study_hours", "class_attendance", "sleep_hours")

# On applique la fonction scale()
train[vars_a_normaliser] <- scale(train[vars_a_normaliser])
test[vars_a_normaliser]  <- scale(test[vars_a_normaliser])

#Vérification rapide (la moyenne de study_hours doit être proche de 0)
print("Vérification normalisation: Moyenne study_hours:")
print(mean(train$study_hours))

#MODELISATION NAÏVE AVEC VALIDATION CROISÉE

# number = 5 (On coupe les données en 5 parts égales)
control <- trainControl(method = "cv", number = 5)

#on entraîne le modèle
set.seed(42) # Pour que les découpes soient toujours les mêmes
model_naif_cv <- train(exam_score ~ . -id,  # On exclutl'ID
                       data = train,        # On lui donne TOUT le dataset (il gère la coupe)
                       method = "lm",       # lm = Linear Model
                       trControl = control)

#résultats
print(model_naif_cv)

#on extrait le RMSE moyen précis
rmse_naif <- model_naif_cv$results$RMSE
print(paste("RMSE Moyen (Naïf - Cross Validation) :", round(rmse_naif, 4)))

#on calcule le R²

#on calcule la Somme des Carrés Résiduelle (RSS)
#c'est la somme de vos erreurs au carré (ce que le modèle n'a pas compris)
rss <- sum((predictions_val - val_set$exam_score)^2)

#on calcule la Somme Totale des Carrés (TSS)
#c'est la variance totale des vraies notes (l'écart par rapport à la moyenne générale)
tss <- sum((val_set$exam_score - mean(val_set$exam_score))^2)

# R² = 1 - (Erreur / Variance Totale)
r_squared <- 1 - (rss / tss) #0.7772

print(paste("Score R² (Validation) :", round(r_squared, 4)))

#on affiche les coeff des variables dans la formule
final_model_naif <- model_naif_cv$finalModel
print(coefficients(final_model_naif))

#GÉNÉRATION DU PREMIER RENDU KAGGLE

#prédiction avec le modèle naïf
predictions_test <- predict(model_naif_cv, newdata = test)

# Un modèle linéaire peut parfois prédire -5 ou 105. On recadre entre 0 et 100.
predictions_test <- ifelse(predictions_test < 0, 0, predictions_test)
predictions_test <- ifelse(predictions_test > 100, 100, predictions_test)

#on crée le dataframe à rendre
submission <- data.frame(
  id = test$id,
  exam_score = predictions_test
)

# 4. Sauvegarde en CSV
write.csv(submission, "../../submissions/submission_naif.csv", row.names = FALSE) ## SCORE DU MODELE NAIF: 8.86871

# On applique des pénalités sur les coefficients

# Fonctions utilitaires
get_metrics <- function(model) {
  best <- model$results[which.min(model$results$RMSE), ]
  return(c(RMSE = best$RMSE, Rsquared = best$Rsquared))
}

generate_submission <- function(model, filename) {
  preds <- predict(model, newdata = test)
  preds <- ifelse(preds < 0, 0, preds)
  preds <- ifelse(preds > 100, 100, preds)
  
  submission <- data.frame(id = test$id, exam_score = preds)
  write.csv(submission, filename, row.names = FALSE)
  print(paste("Fichier généré :", filename))
}

# (Le modèle Naïf est déjà dans 'model_naif_cv')

# RIDGE (Alpha = 0)
print("--- Entraînement RIDGE ---")
grid_ridge <- expand.grid(alpha = 0, lambda = seq(0.001, 1, length = 20))

set.seed(42)
model_ridge <- train(exam_score ~ . -id, 
                     data = train, 
                     method = "glmnet", 
                     trControl = control, 
                     tuneGrid = grid_ridge)

generate_submission(model_ridge, "../../submissions/submission_ridge.csv") ##SCORE KAGGLE: 8.95649


# LASSO (Alpha = 1)
print("--- Entraînement LASSO ---")
grid_lasso <- expand.grid(alpha = 1, lambda = seq(0.0001, 0.1, length = 20))

set.seed(42)
model_lasso <- train(exam_score ~ . -id, 
                     data = train, 
                     method = "glmnet", 
                     trControl = control, 
                     tuneGrid = grid_lasso)

generate_submission(model_lasso, "../../submissions/submission_lasso.csv") ##SCORE KAGGLE: 8.86969


# ELASTIC NET (Alpha variable)
print("--- Entraînement ELASTIC NET ---")
# On teste des alpha entre 0.1 et 0.9 (le mix) et plusieurs lambdas
grid_enet <- expand.grid(alpha = seq(0.1, 0.9, length = 5), 
                         lambda = seq(0.001, 0.5, length = 10))

set.seed(42)
model_enet <- train(exam_score ~ . -id, 
                    data = train, 
                    method = "glmnet", 
                    trControl = control, 
                    tuneGrid = grid_enet)

generate_submission(model_enet, "../../submissions/submission_enet.csv") ##SCORE KAGGLE: 8.86965


# Podium

metrics_naif <- get_metrics(model_naif_cv)
metrics_ridge <- get_metrics(model_ridge)
metrics_lasso <- get_metrics(model_lasso)
metrics_enet  <- get_metrics(model_enet)

results_table <- data.frame(
  Modele = c("Naïf", "Ridge", "Lasso", "ElasticNet"),
  RMSE = c(metrics_naif["RMSE"], metrics_ridge["RMSE"], metrics_lasso["RMSE"], metrics_enet["RMSE"]),
  R2 = c(metrics_naif["Rsquared"], metrics_ridge["Rsquared"], metrics_lasso["Rsquared"], metrics_enet["Rsquared"])
)

print("PODIUM PAR RMSE (Du plus précis au moins précis)")
print(results_table[order(results_table$RMSE), ])

print("PODIUM PAR R² (Du meilleur au pire)")
print(results_table[order(-results_table$R2), ])

# Afficher les meilleurs paramètres pour Elastic Net
print("Meilleurs paramètres Elastic Net :")
print(model_enet$bestTune)

#L'analyse :

# La pénalité ne sert à rien : Lasso et Elastic Net essaient de supprimer des variables,
# mais comme on a beaucoup de données (630 000 lignes) pour peu de variables, le modèle Naïf 
# était déjà très stable. Il n'y avait pas de variables à couper.

# La limite mathématique : Tous ces modèles supposent que la relation est une ligne droite 
# (Y=aX+b). Or, dans la vraie vie, ce n'est pas toujours le cas (par exemple, étudier 20h par 
# jour ne donne pas forcément 2 fois meilleure note que 10h, à cause de la fatigue).

# SVR

## 1. Création d'un échantillon pour ne pas bloquer ton PC
#set.seed(42)
#train_small <- train[sample(nrow(train), 20000), ]
#
## 2. Paramétrage de la validation croisée (plus légère pour le test)
#control_svr <- trainControl(method = "cv", number = 3)
#
## 3. Entraînement du SVR (Noyau Radial)
## Note : C'est ici que le modèle cherche des relations non-linéaires complexes
#print("--- Entraînement SVR Radial (Échantillon) ---")
#model_svr <- train(exam_score ~ . -id, 
#                   data = train_small, 
#                   method = "svmRadial", 
#                   trControl = control_svr,
#                   preProcess = c("center", "scale"), # Crucial pour SVR
#                   tuneLength = 3) # Teste 3 combinaisons de C et Sigma
#
#print(model_svr)

#LE SVR EST TROP DEMANDANT PUISQUE LES DONNEES SONT TROP GRANDES (complexité O(n²))