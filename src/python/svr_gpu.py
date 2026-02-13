import numpy as np
import pandas as pd

from cuml.svm import SVR
from cuml.preprocessing import StandardScaler

# 1. Chargement des données
train = pd.read_csv("../../data/train_processed.csv")
test  = pd.read_csv("../../data/test_processed.csv")

print("Train shape:", train.shape)
print("Test shape :", test.shape)

# 2. One-Hot Encoding sécurisé (toutes les colonnes catégorielles deviennent numériques)
full = pd.concat([train.drop(columns=["exam_score"]), test], axis=0, ignore_index=True)
full_encoded = pd.get_dummies(full, drop_first=True)

X = full_encoded.iloc[:len(train)]
X_test = full_encoded.iloc[len(train):]
y = train["exam_score"]

# 3. Standardisation (OBLIGATOIRE pour SVR GPU)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values.astype(np.float32))
X_test_scaled = scaler.transform(X_test.values.astype(np.float32))

# 4. Entraînement final SVR GPU avec les meilleurs paramètres
final_model = SVR(
    kernel="rbf",
    C=3,
    epsilon=0.2,
    gamma="scale"
)
final_model.fit(X_scaled, y)

# 5. Prédiction test + export Kaggle
test_preds = final_model.predict(X_test_scaled)
test_preds = np.clip(test_preds, 0, 100)

submission = pd.DataFrame({
    "id": test["id"],
    "exam_score": test_preds
})
submission.to_csv("../../submissions/submission_svr_gpu_final.csv", index=False)

print("\nsubmission_svr_gpu_final.csv généré")
