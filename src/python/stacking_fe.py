import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Chargement des données
train = pd.read_csv("../../data/train_fe.csv")
test  = pd.read_csv("../../data/test_fe.csv")

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
y = train["exam_score"].values

# 3. K-Fold Cross Validation
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = []
test_preds = []
model_names = []

# 4. Modèles XGBoost (3 seeds)
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "device": "cuda",        
    "learning_rate": 0.049,
    "max_depth": 5,
    "subsample": 0.854,
    "colsample_bytree": 0.843,
    "min_child_weight": 6,
    "gamma": 0.994
}

for seed in [42, 2024, 1337]:
    print(f"\n=== XGBoost seed={seed} ===")
    oof = np.zeros(len(X))
    test_fold_preds = np.zeros(len(X_test))

    for fold, (tr, val) in enumerate(kf.split(X)):
        dtrain = xgb.DMatrix(X.iloc[tr], label=y[tr])
        dval   = xgb.DMatrix(X.iloc[val], label=y[val])
        dtest  = xgb.DMatrix(X_test)

        params = xgb_params.copy()
        params["seed"] = seed

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        oof[val] = model.predict(dval)
        test_fold_preds += model.predict(dtest) / N_FOLDS

        fold_rmse = np.sqrt(mean_squared_error(y[val], oof[val]))
        print(f"  Fold {fold+1} RMSE: {fold_rmse:.4f}")

    oof_rmse = np.sqrt(mean_squared_error(y, oof))
    print(f"  OOF RMSE: {oof_rmse:.4f}")

    oof_preds.append(oof)
    test_preds.append(test_fold_preds)
    model_names.append(f"xgb_seed_{seed}")

# 5. Modèles LightGBM (2 seeds)
lgb_params = dict(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.85,
)

for seed in [42, 2024]:
    print(f"\n=== LightGBM seed={seed} ===")
    oof = np.zeros(len(X))
    test_fold_preds = np.zeros(len(X_test))

    for fold, (tr, val) in enumerate(kf.split(X)):
        model = lgb.LGBMRegressor(**lgb_params, random_state=seed)

        model.fit(
            X.iloc[tr], y[tr],
            eval_set=[(X.iloc[val], y[val])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(False)],
        )

        oof[val] = model.predict(X.iloc[val])
        test_fold_preds += model.predict(X_test) / N_FOLDS

        fold_rmse = np.sqrt(mean_squared_error(y[val], oof[val]))
        print(f"  Fold {fold+1} RMSE: {fold_rmse:.4f}")

    oof_rmse = np.sqrt(mean_squared_error(y, oof))
    print(f"  OOF RMSE: {oof_rmse:.4f}")

    oof_preds.append(oof)
    test_preds.append(test_fold_preds)
    model_names.append(f"lgb_seed_{seed}")

# 6. CatBoost
print("\n=== CatBoost ===")
oof = np.zeros(len(X))
test_fold_preds = np.zeros(len(X_test))

for fold, (tr, val) in enumerate(kf.split(X)):
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=7,
        loss_function="RMSE",
        task_type="GPU",
        early_stopping_rounds=50,
        random_seed=42,
        verbose=False
    )

    model.fit(X.iloc[tr], y[tr], eval_set=(X.iloc[val], y[val]))
    oof[val] = model.predict(X.iloc[val])
    test_fold_preds += model.predict(X_test) / N_FOLDS

    fold_rmse = np.sqrt(mean_squared_error(y[val], oof[val]))
    print(f"  Fold {fold+1} RMSE: {fold_rmse:.4f}")

oof_rmse = np.sqrt(mean_squared_error(y, oof))
print(f"  OOF RMSE: {oof_rmse:.4f}")

oof_preds.append(oof)
test_preds.append(test_fold_preds)
model_names.append("catboost")

# 7. MLP
print("\n=== MLP ===")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

oof = np.zeros(len(X))
test_fold_preds = np.zeros(len(X_test))

def build_mlp(input_dim):
    model = models.Sequential([
        layers.Dense(256, activation="relu", input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

for fold, (tr, val) in enumerate(kf.split(X_scaled)):
    model = build_mlp(X_scaled.shape[1])

    model.fit(
        X_scaled[tr], y[tr],
        validation_data=(X_scaled[val], y[val]),
        epochs=50,
        batch_size=512,
        verbose=0
    )

    oof[val] = model.predict(X_scaled[val], verbose=0).ravel()
    test_fold_preds += model.predict(X_test_scaled, verbose=0).ravel() / N_FOLDS

    fold_rmse = np.sqrt(mean_squared_error(y[val], oof[val]))
    print(f"  Fold {fold+1} RMSE: {fold_rmse:.4f}")

oof_rmse = np.sqrt(mean_squared_error(y, oof))
print(f"  OOF RMSE: {oof_rmse:.4f}")

oof_preds.append(oof)
test_preds.append(test_fold_preds)
model_names.append("mlp")

# 8. Résumé des OOF par modèle
print("\n=== OOF RMSE par modèle ===")
for name, oof in zip(model_names, oof_preds):
    rmse = np.sqrt(mean_squared_error(y, oof))
    print(f"  {name:20s} : {rmse:.4f}")

# 9. Ridge sur les prédictions OOF
oof_stack = np.column_stack(oof_preds)
test_stack = np.column_stack(test_preds)

ridge = Ridge(alpha=1.0)
ridge.fit(oof_stack, y)

print("\nPoids Ridge par modèle :")
for name, coef in zip(model_names, ridge.coef_):
    print(f"  {name:20s} : {coef:.4f}")

oof_final = ridge.predict(oof_stack)
test_final = ridge.predict(test_stack)

# 10. RMSE OOF final
stacking_rmse = np.sqrt(mean_squared_error(y, oof_final))
print(f"\nSTACKING OOF RMSE : {stacking_rmse:.4f}") #RMSE = 8.7155

# 11. Export Kaggle
test_final = np.clip(test_final, 0, 100)

submission = pd.DataFrame({
    "id": test["id"],
    "exam_score": test_final
})

submission.to_csv("../../submissions/submission_stacked_fe.csv", index=False)
print("submission_stacked_fe.csv généré") #SCORE KAGGLE : 8.67454

"""
=== OOF RMSE par modèle ===
  xgb_seed_42          : 8.7267
  xgb_seed_2024        : 8.7262
  xgb_seed_1337        : 8.7268
  lgb_seed_42          : 8.7344
  lgb_seed_2024        : 8.7371
  catboost             : 8.7656
  mlp                  : 8.8806

Poids Ridge par modèle :
  xgb_seed_42          : 0.2439
  xgb_seed_2024        : 0.2707
  xgb_seed_1337        : 0.2471
  lgb_seed_42          : 0.2672
  lgb_seed_2024        : 0.1818
  catboost             : -0.2041
  mlp                  : -0.0060
"""