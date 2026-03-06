import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models

# Silence optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# TensorFlow : allocation mémoire progressive
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# 1. Chargement des données
train = pd.read_csv("../../data/train_fe.csv")
test  = pd.read_csv("../../data/test_fe.csv")

print("Train shape:", train.shape)
print("Test shape :", test.shape)

# 2. One-Hot Encoding
full = pd.concat(
    [train.drop(columns=["exam_score"]), test],
    axis=0,
    ignore_index=True
)
full_encoded = pd.get_dummies(full, drop_first=True)

X = full_encoded.iloc[:len(train)].drop(columns=["id"])
X_test = full_encoded.iloc[len(train):].drop(columns=["id"])
y = train["exam_score"].values

# K-Fold pour tuning (3 folds) et stacking (5 folds)
kf_tune  = KFold(n_splits=3, shuffle=True, random_state=42)
kf_stack = KFold(n_splits=5, shuffle=True, random_state=42)

# PHASE 1 : OPTUNA — Tuning XGBoost

print("\n" + "="*60)
print("PHASE 1 : Tuning XGBoost (20 trials)...")
print("="*60)

def objective_xgb(trial):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cuda",
        "seed": 42,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
    }
    oof = np.zeros(len(X))
    for tr, val in kf_tune.split(X):
        model = xgb.train(
            params,
            xgb.DMatrix(X.iloc[tr], label=y[tr]),
            num_boost_round=2000,
            evals=[(xgb.DMatrix(X.iloc[val], label=y[val]), "val")],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        oof[val] = model.predict(xgb.DMatrix(X.iloc[val]))
    return np.sqrt(mean_squared_error(y, oof))

study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(objective_xgb, n_trials=20, show_progress_bar=True)

best_xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "device": "cuda",
    **study_xgb.best_params
}

print(f"\nMeilleur RMSE XGBoost (3-Fold) : {study_xgb.best_value:.4f}")
print("Meilleurs params :", study_xgb.best_params)

# PHASE 2 : OPTUNA — Tuning LightGBM

print("\n" + "="*60)
print("PHASE 2 : Tuning LightGBM (20 trials)...")
print("="*60)

def objective_lgb(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "random_state": 42,
        "n_estimators": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
    }
    oof = np.zeros(len(X))
    for tr, val in kf_tune.split(X):
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X.iloc[tr], y[tr],
            eval_set=[(X.iloc[val], y[val])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(False)],
        )
        oof[val] = model.predict(X.iloc[val])
    return np.sqrt(mean_squared_error(y, oof))

study_lgb = optuna.create_study(direction="minimize")
study_lgb.optimize(objective_lgb, n_trials=20, show_progress_bar=True)

best_lgb_params = {
    "n_estimators": 2000,
    **study_lgb.best_params
}

print(f"\nMeilleur RMSE LightGBM (3-Fold) : {study_lgb.best_value:.4f}")
print("Meilleurs params :", study_lgb.best_params)

# PHASE 3 : STACKING FINAL (5-Fold)

print("\n" + "="*60)
print("PHASE 3 : Stacking final (5-Fold)...")
print("="*60)

oof_preds  = []
test_preds = []
model_names = []

# XGBoost (3 seeds) — params Optuna
for seed in [42, 2024, 1337]:
    print(f"\n=== XGBoost seed={seed} ===")
    oof = np.zeros(len(X))
    test_fold_preds = np.zeros(len(X_test))

    for fold, (tr, val) in enumerate(kf_stack.split(X)):
        params = {**best_xgb_params, "seed": seed}
        model = xgb.train(
            params,
            xgb.DMatrix(X.iloc[tr], label=y[tr]),
            num_boost_round=2000,
            evals=[(xgb.DMatrix(X.iloc[val], label=y[val]), "val")],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        oof[val] = model.predict(xgb.DMatrix(X.iloc[val]))
        test_fold_preds += model.predict(xgb.DMatrix(X_test)) / 5

        print(f"  Fold {fold+1} RMSE: {np.sqrt(mean_squared_error(y[val], oof[val])):.4f}")

    print(f"  OOF RMSE: {np.sqrt(mean_squared_error(y, oof)):.4f}")
    oof_preds.append(oof)
    test_preds.append(test_fold_preds)
    model_names.append(f"xgb_seed_{seed}")

# LightGBM (2 seeds) — params Optuna
for seed in [42, 2024]:
    print(f"\n=== LightGBM seed={seed} ===")
    oof = np.zeros(len(X))
    test_fold_preds = np.zeros(len(X_test))

    for fold, (tr, val) in enumerate(kf_stack.split(X)):
        model = lgb.LGBMRegressor(**best_lgb_params, random_state=seed)
        model.fit(
            X.iloc[tr], y[tr],
            eval_set=[(X.iloc[val], y[val])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(False)],
        )
        oof[val] = model.predict(X.iloc[val])
        test_fold_preds += model.predict(X_test) / 5

        print(f"  Fold {fold+1} RMSE: {np.sqrt(mean_squared_error(y[val], oof[val])):.4f}")

    print(f"  OOF RMSE: {np.sqrt(mean_squared_error(y, oof)):.4f}")
    oof_preds.append(oof)
    test_preds.append(test_fold_preds)
    model_names.append(f"lgb_seed_{seed}")

# CatBoost — GPU
print("\n=== CatBoost ===")
oof = np.zeros(len(X))
test_fold_preds = np.zeros(len(X_test))

for fold, (tr, val) in enumerate(kf_stack.split(X)):
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
    test_fold_preds += model.predict(X_test) / 5

    print(f"  Fold {fold+1} RMSE: {np.sqrt(mean_squared_error(y[val], oof[val])):.4f}")

print(f"  OOF RMSE: {np.sqrt(mean_squared_error(y, oof)):.4f}")
oof_preds.append(oof)
test_preds.append(test_fold_preds)
model_names.append("catboost")

# MLP — GPU
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

for fold, (tr, val) in enumerate(kf_stack.split(X_scaled)):
    model = build_mlp(X_scaled.shape[1])
    model.fit(
        X_scaled[tr], y[tr],
        validation_data=(X_scaled[val], y[val]),
        epochs=50,
        batch_size=512,
        verbose=0
    )
    oof[val] = model.predict(X_scaled[val], verbose=0).ravel()
    test_fold_preds += model.predict(X_test_scaled, verbose=0).ravel() / 5

    print(f"  Fold {fold+1} RMSE: {np.sqrt(mean_squared_error(y[val], oof[val])):.4f}")

print(f"  OOF RMSE: {np.sqrt(mean_squared_error(y, oof)):.4f}")
oof_preds.append(oof)
test_preds.append(test_fold_preds)
model_names.append("mlp")

# PHASE 4 : MÉTA-LEARNER Ridge

print("\n=== OOF RMSE par modèle ===")
for name, oof in zip(model_names, oof_preds):
    print(f"  {name:20s} : {np.sqrt(mean_squared_error(y, oof)):.4f}")

oof_stack  = np.column_stack(oof_preds)
test_stack = np.column_stack(test_preds)

ridge = Ridge(alpha=1.0)
ridge.fit(oof_stack, y)

print("\nPoids Ridge par modèle :")
for name, coef in zip(model_names, ridge.coef_):
    print(f"  {name:20s} : {coef:.4f}")

oof_final  = ridge.predict(oof_stack)
test_final = ridge.predict(test_stack)

stacking_rmse = np.sqrt(mean_squared_error(y, oof_final))
print(f"\nSTACKING OOF RMSE : {stacking_rmse:.4f}") # RMSE=8.7113

# Export
test_final = np.clip(test_final, 0, 100)
submission = pd.DataFrame({"id": test["id"], "exam_score": test_final})
submission.to_csv("../../submissions/submission_stacked_fe_v4.csv", index=False)
print("submission_stacked_fe_v4.csv généré") # SCORE KAGGLE : 8.66743

"""
=== OOF RMSE par modèle ===
  xgb_seed_42          : 8.7243
  xgb_seed_2024        : 8.7248
  xgb_seed_1337        : 8.7251
  lgb_seed_42          : 8.7227
  lgb_seed_2024        : 8.7245
  catboost             : 8.7654
  mlp                  : 8.8737

Poids Ridge par modèle :
  xgb_seed_42          : 0.2195
  xgb_seed_2024        : 0.1985
  xgb_seed_1337        : 0.1668
  lgb_seed_42          : 0.3308
  lgb_seed_2024        : 0.2651
  catboost             : -0.2122
  mlp                  : 0.0320
"""