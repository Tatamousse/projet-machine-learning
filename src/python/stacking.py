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
y = train["exam_score"].values

# 3. K-Fold Cross Validation
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = []     # Out-of-fold predictions (train)
test_preds = []    # Test predictions
model_names = []

# 4. Modèles XGBoost (3 seeds différents)
xgb_params = {
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

for seed in [42, 2024, 1337]:
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
            num_boost_round=1200,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        oof[val] = model.predict(dval)
        test_fold_preds += model.predict(dtest) / N_FOLDS

    oof_preds.append(oof)
    test_preds.append(test_fold_preds)
    model_names.append(f"xgb_seed_{seed}")

# 5. Modèles LightGBM (2 seeds)
for seed in [42, 2024]:
    oof = np.zeros(len(X))
    test_fold_preds = np.zeros(len(X_test))

    for tr, val in kf.split(X):
        model = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed
        )

        model.fit(
            X.iloc[tr], y[tr],
            eval_set=[(X.iloc[val], y[val])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50)],
        )

        oof[val] = model.predict(X.iloc[val])
        test_fold_preds += model.predict(X_test) / N_FOLDS

    oof_preds.append(oof)
    test_preds.append(test_fold_preds)
    model_names.append(f"lgb_seed_{seed}")

# 6. Modèle CatBoost
oof = np.zeros(len(X))
test_fold_preds = np.zeros(len(X_test))

for tr, val in kf.split(X):
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=7,
        loss_function="RMSE",
        random_seed=42,
        verbose=False
    )

    model.fit(X.iloc[tr], y[tr])
    oof[val] = model.predict(X.iloc[val])
    test_fold_preds += model.predict(X_test) / N_FOLDS

oof_preds.append(oof)
test_preds.append(test_fold_preds)
model_names.append("catboost")

# 7. Préparation des données pour le MLP (scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

oof = np.zeros(len(X))
test_fold_preds = np.zeros(len(X_test))

# 8. Modèle MLP (NN)
def build_mlp(input_dim):
    model = models.Sequential([
        layers.Dense(256, activation="relu", input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

for tr, val in kf.split(X_scaled):
    model = build_mlp(X_scaled.shape[1])

    model.fit(
        X_scaled[tr], y[tr],
        validation_data=(X_scaled[val], y[val]),
        epochs=50,
        batch_size=512,
        verbose=0
    )

    oof[val] = model.predict(X_scaled[val]).ravel()
    test_fold_preds += model.predict(X_test_scaled).ravel() / N_FOLDS

oof_preds.append(oof)
test_preds.append(test_fold_preds)
model_names.append("mlp")

# 9. Stacking : Ridge Regression sur les prédictions OOF
oof_stack = np.column_stack(oof_preds)
test_stack = np.column_stack(test_preds)

ridge = Ridge(alpha=1.0)
ridge.fit(oof_stack, y)

oof_final = ridge.predict(oof_stack)
test_final = ridge.predict(test_stack)

# 10. Évaluation OOF du stacking
rmse = np.sqrt(mean_squared_error(y, oof_final))
print("STACKING RMSE:", rmse) # RMSE=8.7392

# 11. Génération du fichier de soumission Kaggle
test_final = np.clip(test_final, 0, 100)

submission = pd.DataFrame({
    "id": test["id"],
    "exam_score": test_final
})

submission.to_csv("../../submissions/submission_stacked.csv", index=False) # SCORE KAGGLE: 8.70398 (temps d'exécution: 1h20)
print("submission_stacked.csv généré")
