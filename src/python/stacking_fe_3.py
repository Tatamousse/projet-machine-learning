import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

import tensorflow as tf
from tensorflow.keras import layers, models

# TensorFlow : allocation mémoire progressive
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# 1. Chargement des données FE
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

# 3. K-Fold
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = []
test_preds = []
model_names = []

# ── NIVEAU 0 ──────────────────────────────────────────────────────────────────

# 4. XGBoost (3 seeds) — GPU
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

        params = {**xgb_params, "seed": seed}

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

# 5. LightGBM (2 seeds) — CPU
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

# 6. CatBoost — GPU
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

# 7. MLP — CPU
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

# ── RÉSUMÉ NIVEAU 0 ───────────────────────────────────────────────────────────

print("\n=== OOF RMSE par modèle (niveau 0) ===")
for name, oof in zip(model_names, oof_preds):
    rmse = np.sqrt(mean_squared_error(y, oof))
    print(f"  {name:20s} : {rmse:.4f}")

# ── NIVEAU 1 : XGBoost méta-learner ──────────────────────────────────────────

print("\n=== Méta-learner XGBoost (niveau 1) ===")

oof_stack  = np.column_stack(oof_preds)   # (n_train, 7)
test_stack = np.column_stack(test_preds)  # (n_test,  7)

# Volontairement léger pour éviter l'overfitting sur les OOF
meta_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "device": "cuda",
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "gamma": 1.0,
    "seed": 42
}

meta_oof        = np.zeros(len(oof_stack))
meta_test_preds = np.zeros(len(test_stack))

for fold, (tr, val) in enumerate(kf.split(oof_stack)):
    dmeta_train = xgb.DMatrix(oof_stack[tr], label=y[tr])
    dmeta_val   = xgb.DMatrix(oof_stack[val], label=y[val])
    dmeta_test  = xgb.DMatrix(test_stack)

    meta_model = xgb.train(
        meta_params,
        dmeta_train,
        num_boost_round=500,
        evals=[(dmeta_val, "val")],
        early_stopping_rounds=30,
        verbose_eval=False
    )

    meta_oof[val] = meta_model.predict(dmeta_val)
    meta_test_preds += meta_model.predict(dmeta_test) / N_FOLDS

    fold_rmse = np.sqrt(mean_squared_error(y[val], meta_oof[val]))
    print(f"  Fold {fold+1} RMSE: {fold_rmse:.4f}")

meta_rmse = np.sqrt(mean_squared_error(y, meta_oof))
print(f"\nSTACKING OOF RMSE (méta XGBoost) : {meta_rmse:.4f}") #RMSE=8.7189

# ── EXPORT ────────────────────────────────────────────────────────────────────

meta_test_preds = np.clip(meta_test_preds, 0, 100)

submission = pd.DataFrame({
    "id": test["id"],
    "exam_score": meta_test_preds
})

submission.to_csv("../../submissions/submission_stacked_fe_v3.csv", index=False)
print("submission_stacked_fe_v3.csv généré") #SCORE KAGGLE : 8.67756

"""
=== OOF RMSE par modèle (niveau 0) ===
  xgb_seed_42          : 8.7267
  xgb_seed_2024        : 8.7262
  xgb_seed_1337        : 8.7268
  lgb_seed_42          : 8.7344
  lgb_seed_2024        : 8.7371
  catboost             : 8.7654
  mlp                  : 9.3277

=== Méta-learner XGBoost (niveau 1) ===
  Fold 1 RMSE: 8.7086
  Fold 2 RMSE: 8.7149
  Fold 3 RMSE: 8.7094
  Fold 4 RMSE: 8.7199
  Fold 5 RMSE: 8.7416
"""