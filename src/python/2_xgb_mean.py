import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Chargement des données
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

# 4. Paramètres XGBoost (bons + stables)
base_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",  
    "learning_rate": 0.06,
    "max_depth": 7,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 5,
    "gamma": 0.3
}

# 5. Entraînement XGB #1
params_1 = base_params.copy()
params_1["seed"] = 42

model_1 = xgb.train(
    params_1,
    dtrain,
    num_boost_round=1200,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=50,
    verbose_eval=100
)

preds_val_1 = model_1.predict(dval)

# 6. Entraînement XGB #2 (seed différent)
params_2 = base_params.copy()
params_2["seed"] = 2024

model_2 = xgb.train(
    params_2,
    dtrain,
    num_boost_round=1200,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=50,
    verbose_eval=100
)

preds_val_2 = model_2.predict(dval)

# 7. Évaluation de la moyenne
val_preds_avg = 0.5 * preds_val_1 + 0.5 * preds_val_2

rmse = np.sqrt(mean_squared_error(y_val, val_preds_avg))
r2   = r2_score(y_val, val_preds_avg)

print("\n ENSEMBLE 2 XGB")
print("Validation RMSE :", rmse) #RMSE=8.7420
print("Validation R²   :", r2) #r²=0.7851

# 8. Entraînement final sur tout le train
dtrain_full = xgb.DMatrix(X, label=y)

model_1_full = xgb.train(
    params_1,
    dtrain_full,
    num_boost_round=model_1.best_iteration
)

model_2_full = xgb.train(
    params_2,
    dtrain_full,
    num_boost_round=model_2.best_iteration
)

# 9. Prédictions test + moyenne
test_preds_1 = model_1_full.predict(dtest)
test_preds_2 = model_2_full.predict(dtest)

test_preds_avg = 0.5 * test_preds_1 + 0.5 * test_preds_2
test_preds_avg = np.clip(test_preds_avg, 0, 100)

submission = pd.DataFrame({
    "id": test["id"],
    "exam_score": test_preds_avg
})

submission.to_csv("../../submissions/submission_2_xgb_mean.csv", index=False) 
print("\n submission_2_xgb_mean.csv généré") # SCORE KAGGLE: 8.71549
