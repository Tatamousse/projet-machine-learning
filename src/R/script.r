library(caret)
library(glmnet)

#Etude de données

train <- read.csv("../../data/train.csv")
test <- read.csv("../../data/test.csv")

print(paste("Train dimensions:", nrow(train), "lignes,", ncol(train), "colonnes"))
print(paste("Test dimensions:", nrow(test), "lignes,", ncol(test), "colonnes"))

#voir le type de variable
str(train)
str(test)

#voir les premières lignes de train
head(train)
head(test)

#résumé statistique complet
summary(train)
summary(test)

#visualisation simple de exam_score
hist(train$exam_score, 
     main = "Distribution des notes (Exam Score)", 
     xlab = "Note", 
     ylab = "Fréquence",
     col = "lightblue", 
     border = "black",
     breaks = 30) #breaks définit le nombre de barres


#BOXPLOT
#configuration pour afficher 4 graphiques en même temps
par(mfrow = c(2, 2))

#boxplot pour les variables numériques principales
boxplot(train$study_hours, main = "Heures d'étude", col = "orange", horizontal = TRUE)
boxplot(train$class_attendance, main = "Présence en classe", col = "lightgreen", horizontal = TRUE)
boxplot(train$sleep_hours, main = "Heures de sommeil", col = "orchid", horizontal = TRUE)
boxplot(train$age, main = "Âge", col = "gold", horizontal = TRUE)

# Réinitialiser l'affichage
par(mfrow = c(1, 1))


#VALEURS MANQUANTES OU DOUBLONS
#chercher les valeurs manquantes
valeurs_manquantes_train <- colSums(is.na(train))

#afficher celles qui ont un des problèmes
print("Valeurs manquantes par colonne :")
print(valeurs_manquantes_train[valeurs_manquantes_train > 0])

#recherche des doublons
nb_doublons_train <- sum(duplicated(train))
print(paste("Nombre de lignes dupliquées (train.csv):", nb_doublons_train))

#vérification de la cohérence
print(paste("Note min :", min(train$exam_score)))
print(paste("Note max :", max(train$exam_score)))

# AUDIT DES MODALITÉS POUR LES VARIABLES TEXTE

cat("\n TRAIN \n")

vars_char_train <- names(train)[sapply(train, is.character)]

for (col in vars_char_train) {
  cat("\n==============================\n")
  cat("Variable :", col, "\n")
  cat("Nombre de modalités :", length(unique(train[[col]])), "\n\n")
  
  print(sort(unique(train[[col]])))
}

cat("\n==============================\n")

cat("\n TEST \n")

vars_char_test <- names(test)[sapply(test, is.character)]

for (col in vars_char_test) {
  cat("\n==============================\n")
  cat("Variable :", col, "\n")
  cat("Nombre de modalités :", length(unique(test[[col]])), "\n\n")
  
  print(sort(unique(test[[col]])))
}

# AUDIT OK : toutes les variables catégorielles sont propres et cohérentes
# train / test (aucune modalité fantôme détectée)








#TRAITEMENT DE DONNEES

#on convertit les variables textuelles en variables numériques

#1. les variables ordinales

#Fonction pour mapper les textes vers des chiffres
convert_categorical <- function(df) {
  
  # ---------- ORDINALES ----------
  
  # sleep_quality : poor < average < good
  df$sleep_quality <- as.numeric(factor(
    df$sleep_quality,
    levels = c("poor", "average", "good"),
    ordered = TRUE
  ))
  
  # exam_difficulty : easy < moderate < hard
  df$exam_difficulty <- as.numeric(factor(
    df$exam_difficulty,
    levels = c("easy", "moderate", "hard"),
    ordered = TRUE
  ))
  
  # facility_rating : low < medium < high
  df$facility_rating <- as.numeric(factor(
    df$facility_rating,
    levels = c("low", "medium", "high"),
    ordered = TRUE
  ))
  # ---------- NOMINALES ----------
  
  df$gender <- factor(
    df$gender,
    levels = c("female", "male", "other")
  )
  
  df$course <- factor(
    df$course,
    levels = c("ba", "b.com", "b.sc", "b.tech", "bba", "bca", "diploma")
  )
  
  df$internet_access <- factor(
    df$internet_access,
    levels = c("no", "yes")
  )
  
  df$study_method <- factor(
    df$study_method,
    levels = c(
      "coaching",
      "group study",
      "mixed",
      "online videos",
      "self-study"
    )
  )
  
  return(df)
}
#On applique la transformation sur train et test
train <- convert_categorical(train)
test <- convert_categorical(test)


#Vérifier si toutes les variables sont factor ou num
str(train)
str(test)

#NORMALISATION (Centrer-Réduire)

#On liste les colonnes purement numériques
vars_a_normaliser <- c("age", "study_hours", "class_attendance", "sleep_hours")

# Calcul des moyennes et écarts-types SUR LE TRAIN
means <- sapply(train[vars_a_normaliser], mean)
sds   <- sapply(train[vars_a_normaliser], sd)

# Normalisation du train
train[vars_a_normaliser] <- sweep(train[vars_a_normaliser], 2, means, "-")
train[vars_a_normaliser] <- sweep(train[vars_a_normaliser], 2, sds, "/")

# Normalisation du test AVEC les stats du train
test[vars_a_normaliser] <- sweep(test[vars_a_normaliser], 2, means, "-")
test[vars_a_normaliser] <- sweep(test[vars_a_normaliser], 2, sds, "/")

#Vérification rapide (la moyenne de study_hours doit être proche de 0)
print("Vérification normalisation: \n Moyenne study_hours:")
print(mean(train$study_hours))
print("écart type study_hours:")
print(sd(train$study_hours)) 

# EXPORT DES DONNÉES POST-TRAITEMENT

write.csv(
  train,
  "../../data/train_processed.csv",
  row.names = FALSE
)

write.csv(
  test,
  "../../data/test_processed.csv",
  row.names = FALSE
)

cat("Données post-traitées exportées pour Python\n")









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

r2_naif <- model_naif_cv$results$Rsquared
print(paste("R² Moyen (Naïf - Cross Validation) :", round(r2_naif, 4)))

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
grid_ridge <- expand.grid(
  alpha = 0,
  lambda = 10^seq(-4, 2, length = 50)
)

set.seed(42)
model_ridge <- train(exam_score ~ . -id, 
                     data = train, 
                     method = "glmnet", 
                     trControl = control, 
                     tuneGrid = grid_ridge)

generate_submission(model_ridge, "../../submissions/submission_ridge.csv") ##SCORE KAGGLE: 8.95649


# LASSO (Alpha = 1)
print("--- Entraînement LASSO ---")
grid_lasso <- expand.grid(
  alpha = 1,
  lambda = 10^seq(-4, 2, length = 50)
)

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
grid_enet <- expand.grid(
  alpha  = seq(0, 1, by = 0.1),
  lambda = 10^seq(-4, 2, length = 50)
)

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