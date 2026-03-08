import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from cuml.svm import SVR
from cuml.preprocessing import StandardScaler

#Chargement des données
train_full = pd.read_csv("../../data/train_processed.csv")
train = train_full.sample(n=100000, random_state=42)
test  = pd.read_csv("../../data/test_processed.csv")

print("Train shape:", train.shape)
print("Test shape :", test.shape)

#One-Hot Encoding sécurisé
full = pd.concat([train.drop(columns=["exam_score"]), test], axis=0, ignore_index=True)
full_encoded = pd.get_dummies(full, drop_first=True)

X = full_encoded.iloc[:len(train)]
X_test = full_encoded.iloc[len(train):]
y = train["exam_score"]

#Standardisation (OBLIGATOIRE pour SVR GPU)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values.astype(np.float32))
X_test_scaled = scaler.transform(X_test.values.astype(np.float32))

#Split train / validation
X_train_s, X_val_s, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

#Entraînement
final_model = SVR(
    kernel="rbf",
    C=3,
    epsilon=0.2,
    gamma="scale"
)
final_model.fit(X_train_s, y_train)

#Évaluation
val_preds = final_model.predict(X_val_s)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"RMSE validation : {rmse:.4f}")

#Entraînement final sur tout le train (100k) pour la soumission
final_model.fit(X_scaled, y)

#Prédiction test + export Kaggle
test_preds = final_model.predict(X_test_scaled)
test_preds = np.clip(test_preds, 0, 100)

submission = pd.DataFrame({
    "id": test["id"],
    "exam_score": test_preds
})
submission.to_csv("../../submissions/submission_svr_gpu_final.csv", index=False)
print("\nsubmission_svr_gpu_final.csv généré")  # SCORE KAGGLE: 9.1