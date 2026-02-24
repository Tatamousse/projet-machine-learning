import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# 1. Chargement des données
train = pd.read_csv("../../data/train_fe.csv")
test  = pd.read_csv("../../data/test_fe.csv")

print("Train shape:", train.shape)
print("Test shape :", test.shape)

# 2. One-Hot Encoding (train + test)
full = pd.concat(
    [train.drop(columns=["exam_score"]), test],
    axis=0,
    ignore_index=True
)

full_encoded = pd.get_dummies(full, drop_first=True)

X = full_encoded.iloc[:len(train)].drop(columns=["id"])
X_test = full_encoded.iloc[len(train):].drop(columns=["id"])
y = train["exam_score"].values

dtest = xgb.DMatrix(X_test)

# 3. Paramètres XGBoost
params = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "eval_metric": "rmse",
    "learning_rate": 0.04903521267559334,
    "max_depth": 5,
    "subsample": 0.8539559833318924,
    "colsample_bytree": 0.8428445166723066,
    "min_child_weight": 6,
    "gamma": 0.9942170069541861,
    "seed": 42
}

# 4. K-Fold (5 folds)
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))        # prédictions out-of-fold sur le train
test_preds = np.zeros(len(X_test))  # moyenne des prédictions test sur les 5 folds
best_iterations = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n--- Fold {fold + 1} / {N_FOLDS} ---")

    dtrain = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx])
    dval   = xgb.DMatrix(X.iloc[val_idx], label=y[val_idx])

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=200
    )

    oof_preds[val_idx] = model.predict(dval)
    test_preds += model.predict(dtest) / N_FOLDS
    best_iterations.append(model.best_iteration)

    fold_rmse = np.sqrt(mean_squared_error(y[val_idx], oof_preds[val_idx]))
    print(f"Fold {fold + 1} RMSE : {fold_rmse:.4f}")

# 5. Score OOF global
oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"\nOOF RMSE (5-Fold) : {oof_rmse:.4f}") #8.7278
print(f"Best iterations par fold : {best_iterations}")
print(f"Moyenne best_iteration : {int(np.mean(best_iterations))}")

# 6. Export Kaggle
test_preds = np.clip(test_preds, 0, 100)

submission = pd.DataFrame({
    "id": test["id"],
    "exam_score": test_preds
})

submission.to_csv("../../submissions/submission_xgb_kfold_fe.csv", index=False)
print("\nsubmission_xgb_kfold_fe.csv généré") #SCORE KAGGLE: 8.68213