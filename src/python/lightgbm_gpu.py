import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error

# 1. Chargement des données
train = pd.read_csv("../../data/train_processed.csv")
test  = pd.read_csv("../../data/test_processed.csv")
print("Train shape:", train.shape)
print("Test shape :", test.shape)

# 2. One-Hot Encoding sécurisé
full = pd.concat([train.drop(columns=["exam_score"]), test], axis=0, ignore_index=True)
full_encoded = pd.get_dummies(full, drop_first=True)

X = full_encoded.iloc[:len(train)]
X_test = full_encoded.iloc[len(train):]
y = train["exam_score"]

# 3. Split train / validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Grille de paramètres
param_grid = {
    "max_depth": [5, 7, 9],
    "learning_rate": [0.03, 0.05, 0.07],
    "reg_lambda": [0.1, 1, 3],
}

best_rmse = np.inf
best_params = None
best_model = None

# 5. Recherche des meilleurs paramètres
for params in ParameterGrid(param_grid):
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        reg_lambda=params["reg_lambda"],
        objective="regression",
        device="gpu",
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
    )

    preds_val = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds_val))
    print(f"{params} -> RMSE: {rmse:.4f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_params = params
        best_model = model

print("\nBEST PARAMS:", best_params)
print("BEST RMSE on validation:", best_rmse)

# 6. Entraînement final sur tout le train
final_model = lgb.LGBMRegressor(
    n_estimators=2000,
    max_depth=best_params["max_depth"],
    learning_rate=best_params["learning_rate"],
    reg_lambda=best_params["reg_lambda"],
    objective="regression",
    device="gpu",
    random_state=42
)

final_model.fit(
    X, y,
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
)

# 7. Prédiction test + export
test_preds = final_model.predict(X_test)
test_preds = np.clip(test_preds, 0, 100)

submission = pd.DataFrame({
    "id": test["id"],
    "exam_score": test_preds
})

submission.to_csv("../../submissions/submission_lgbm_gpu.csv", index=False)
print("\nsubmission_lgbm_gpu.csv généré")