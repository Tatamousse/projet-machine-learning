import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Chargement des données processées
train = pd.read_csv("../../data/train_processed.csv")
test  = pd.read_csv("../../data/test_processed.csv")

print("Train shape:", train.shape)
print("Test shape :", test.shape)

# 2. One-Hot Encoding sécurisé (train + test)
full = pd.concat(
    [train.drop(columns=["exam_score"]), test],
    axis=0,
    ignore_index=True
)

full_encoded = pd.get_dummies(full, drop_first=True)

X = full_encoded.iloc[:len(train)].drop(columns=["id"])
X_test = full_encoded.iloc[len(train):].drop(columns=["id"])
y = train["exam_score"]

# 3. Split train / validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val, label=y_val)
dtest  = xgb.DMatrix(X_test)

# 4. Paramètres XGBoost (CPU hist – RAPIDE & STABLE)
params = {
    "objective": "reg:squarederror",
    "tree_method": "hist",   # compatible partout
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

evals = [(dtrain, "train"), (dval, "val")]

# 5. Entraînement avec Early Stopping
model = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=50
)

# 6. Évaluation
val_preds = model.predict(dval)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
r2   = r2_score(y_val, val_preds)
print("RMSE validation :", rmse) #RMSE=8.7558
print("R² validation   :", r2) #r²=0.7844

# 7. Entraînement final sur TOUT le train
dtrain_full = xgb.DMatrix(X, label=y)

model_final = xgb.train(
    params,
    dtrain_full,
    num_boost_round=model.best_iteration
)

# 8. Prédiction test + export Kaggle
test_preds = model_final.predict(dtest)
test_preds = np.clip(test_preds, 0, 100)

submission = pd.DataFrame({
    "id": test["id"],
    "exam_score": test_preds
})

submission.to_csv("../../submissions/submission_xgb.csv", index=False)
print("submission_xgb.csv généré") # SCORE KAGGLE: 8.72408
