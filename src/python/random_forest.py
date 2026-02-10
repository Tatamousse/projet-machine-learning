import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Chargement des données processées
train = pd.read_csv("../../data/train_processed.csv")
test  = pd.read_csv("../../data/test_processed.csv")

print("Train shape:", train.shape)
print("Test shape :", test.shape)

# 2. One-Hot Encoding (train + test)
full = pd.concat([train.drop(columns=["exam_score"]), test], axis=0, ignore_index=True)
full_encoded = pd.get_dummies(full, drop_first=True)

X = full_encoded.iloc[:len(train)].drop(columns=["id"])
X_test = full_encoded.iloc[len(train):].drop(columns=["id"])
y = train["exam_score"]

# 3. Split train / validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Modèle Random Forest (RÉGULARISÉ)
rf = RandomForestRegressor(
    n_estimators=200,        # nombre d'arbres
    max_depth=15,            # clé contre l’overfitting
    min_samples_split=50,
    min_samples_leaf=20,
    n_jobs=-1,               # utilise tous les coeurs CPU
    random_state=42
)

rf.fit(X_train, y_train)

# 5. Évaluation
val_preds = rf.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
r2   = r2_score(y_val, val_preds)
print("RMSE validation :", rmse) #RMSE=8.9402
print("R² validation   :", r2) #r²=0.7752

# 6. Entraînement final sur tout le train
rf.fit(X, y)

# 7. Prédiction test + export Kaggle
test_preds = rf.predict(X_test)
test_preds = np.clip(test_preds, 0, 100)

submission = pd.DataFrame({
    "id": test["id"],
    "exam_score": test_preds
})

submission.to_csv("../../submissions/submission_random_forest.csv", index=False)
print("submission_random_forest.csv généré") # SCORE KAGGLE: 8.91147
