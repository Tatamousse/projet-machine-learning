import numpy as np
import pandas as pd

# =============================================================================
# FEATURE ENGINEERING
# À exécuter après le script R (qui génère train_processed.csv / test_processed.csv)
# Génère train_fe.csv et test_fe.csv à utiliser dans tous tes modèles
# =============================================================================

train = pd.read_csv("../../data/train_processed.csv")
test  = pd.read_csv("../../data/test_processed.csv")

print("Train shape (avant FE):", train.shape)
print("Test shape  (avant FE):", test.shape)


def add_features(df):
    """
    Ajoute toutes les nouvelles features sur un dataframe.
    Fonctionne sur train ET test (pas besoin de exam_score).
    """

    # Étudier beaucoup ET dormir bien
    df["study_x_sleep_quality"] = df["study_hours"] * df["sleep_quality"]

    # Étudier beaucoup ET aller en cours
    df["study_x_attendance"] = df["study_hours"] * df["class_attendance"]

    # Quantité de sommeil × qualité du sommeil
    df["sleep_x_sleep_quality"] = df["sleep_hours"] * df["sleep_quality"]

    # Présence en cours × accès internet (booléen 0/1 après encoding)
    # On anticipe que internet_access sera encodé 0/1 plus tard
    # On le fait directement ici sur la colonne texte -> binaire temporaire
    internet_bin = (df["internet_access"].astype(str).str.lower() == "yes").astype(int)
    df["attendance_x_internet"] = df["class_attendance"] * internet_bin

    # Difficulté de l'examen × heures d'étude (un examen dur nécessite plus de prépa)
    df["study_x_difficulty"] = df["study_hours"] * df["exam_difficulty"]

    # Rapport effort / récupération
    # Attention : sleep_hours peut être 0 après normalisation -> on ajoute une constante
    df["study_sleep_ratio"] = df["study_hours"] / (df["sleep_hours"] + 1e-3)

    # Présence relative à l'effort d'étude
    df["attendance_study_ratio"] = df["class_attendance"] / (df["study_hours"] + 1e-3)


    # "Temps total d'apprentissage" = étude perso + cours
    df["total_learning"] = df["study_hours"] + df["class_attendance"]

    # Score d'environnement favorable (accès ressources + qualité infra)
    df["env_score"] = internet_bin + df["facility_rating"]


    # Rendements décroissants : étudier 20h n'est pas 2× mieux qu'étudier 10h
    df["study_hours_sq"]      = df["study_hours"] ** 2
    df["class_attendance_sq"] = df["class_attendance"] ** 2
    df["sleep_quality_sq"]    = df["sleep_quality"] ** 2


    # Le "profil idéal" : étude × présence × qualité de sommeil
    df["triple_effort"] = df["study_hours"] * df["class_attendance"] * df["sleep_quality"]

    return df


# Application sur train et test
train = add_features(train)
test  = add_features(test)

print("\nTrain shape (après FE):", train.shape)
print("Test shape  (après FE):", test.shape)

# Affichage des nouvelles colonnes créées
new_cols = [
    "study_x_sleep_quality", "study_x_attendance", "sleep_x_sleep_quality",
    "attendance_x_internet", "study_x_difficulty",
    "study_sleep_ratio", "attendance_study_ratio",
    "total_learning", "env_score",
    "study_hours_sq", "class_attendance_sq", "sleep_quality_sq",
    "triple_effort"
]

print("\nNouvelles features créées :", len(new_cols))
for col in new_cols:
    print(f"  - {col}")

# Vérification : pas de NaN introduits
nan_train = train[new_cols].isna().sum().sum()
nan_test  = test[new_cols].isna().sum().sum()
print(f"\nNaN dans train après FE : {nan_train}")
print(f"NaN dans test  après FE : {nan_test}")

# Export
train.to_csv("../../data/train_fe.csv", index=False)
test.to_csv("../../data/test_fe.csv",   index=False)

print("\ntrain_fe.csv et test_fe.csv générés avec succès !")