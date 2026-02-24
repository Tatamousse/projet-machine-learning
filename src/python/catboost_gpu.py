import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error

# 1. Chargement des données
train = pd.read_csv("../../data/train_processed.csv")
test  = pd.read_csv("../../data/test_processed.csv")

print("Train shape:", train.shape)
print("Test shape :", test.shape)

# 2. One-Hot Encoding sécurisé (train + test)
full = pd.concat([train.drop(columns=["exam_score"]), test], axis=0, ignore_index=True)
full_encoded = pd.get_dummies(full, drop_first=True)

X = full_encoded.iloc[:len(train)]
X_test = full_encoded.iloc[len(train):]
y = train["exam_score"]

# 3. Split train / validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Définir la grille de paramètres
param_grid = {
    "depth": [5, 7, 9],
    "learning_rate": [0.03, 0.05, 0.07],
    "l2_leaf_reg": [1, 3, 5],
}

best_rmse = np.inf
best_params = None
best_model = None

# 5. Recherche des meilleurs paramètres
for params in ParameterGrid(param_grid):
    model = CatBoostRegressor(
        iterations=2000,
        depth=params["depth"],
        learning_rate=params["learning_rate"],
        l2_leaf_reg=params["l2_leaf_reg"],
        loss_function="RMSE",
        task_type="GPU",
        early_stopping_rounds=50,
        verbose=False,
        random_seed=42
    )
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    preds_val = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds_val))
    
    print(f"{params} -> RMSE: {rmse:.4f}")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_params = params
        best_model = model

print("\nBEST PARAMS:", best_params)
print("BEST RMSE on validation:", best_rmse)

# 6. Entraînement final sur tout le train avec les meilleurs paramètres
final_model = CatBoostRegressor(
    iterations=2000,
    depth=best_params["depth"],
    learning_rate=best_params["learning_rate"],
    l2_leaf_reg=best_params["l2_leaf_reg"],
    loss_function="RMSE",
    task_type="GPU",
    early_stopping_rounds=50,
    verbose=100,
    random_seed=42
)

final_model.fit(X, y)

# 7. Prédiction test + export Kaggle
test_preds = final_model.predict(X_test)
test_preds = np.clip(test_preds, 0, 100)

submission = pd.DataFrame({
    "id": test["id"],
    "exam_score": test_preds
})

submission.to_csv("../../submissions/submission_catboost_gpu.csv", index=False)
print("\nsubmission_catboost_gpu.csv généré") # SCORE KAGGLE: 8.7798