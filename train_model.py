import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("plantfolio_final_ml_dataset.csv")

categorical_cols = [
    "Plant_Name",
    "Season",
    "Humidity_Preference",
    "Light_Preference",
    "Drainage_Preference",
    "Watering_Method"
]
numerical_cols = [
    "Temperature_Preference_Min_C",
    "Temperature_Preference_Max_C"
]
target_col = "Watering_Frequency_days"

X = df[categorical_cols + numerical_cols]
y = df[target_col]
groups = df["Plant_Name"]

# ==========================================
# 1. Conventional Train/Test Split (80/20)
# ==========================================
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y, test_size=0.2, random_state=42
)

encoder_c = OneHotEncoder(categories='auto', handle_unknown='ignore', sparse_output=False)
X_train_c_enc = encoder_c.fit_transform(X_train_c[categorical_cols])
X_train_c_final = np.hstack([X_train_c[numerical_cols].values, X_train_c_enc])

X_test_c_enc = encoder_c.transform(X_test_c[categorical_cols])
X_test_c_final = np.hstack([X_test_c[numerical_cols].values, X_test_c_enc])

rf_c = RandomForestRegressor(n_estimators=100, random_state=42)
rf_c.fit(X_train_c_final, y_train_c)

preds_c = rf_c.predict(X_test_c_final)
mae_c = mean_absolute_error(y_test_c, preds_c)
rmse_c = root_mean_squared_error(y_test_c, preds_c)
r2_c = r2_score(y_test_c, preds_c)

print(f"=== Conventional 80/20 Split (Random Shuffle) ===")
print(f"Train samples: {len(y_train_c)}, Test samples: {len(y_test_c)}")
print(f"MAE:  {mae_c:.4f} days")
print(f"RMSE: {rmse_c:.4f} days")
print(f"R^2:  {r2_c:.4f}")

# ==========================================
# 2. Plant-Grouped Split (GroupShuffleSplit by Plant_Name)
# ==========================================
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train_g, X_test_g = X.iloc[train_idx], X.iloc[test_idx]
y_train_g, y_test_g = y.iloc[train_idx], y.iloc[test_idx]

# Check leakage
train_plants = set(X_train_g["Plant_Name"])
test_plants = set(X_test_g["Plant_Name"])
overlap = train_plants.intersection(test_plants)
assert len(overlap) == 0, f"Plant leakage detected! Overlap: {overlap}"

encoder_g = OneHotEncoder(categories='auto', handle_unknown='ignore', sparse_output=False)
X_train_g_enc = encoder_g.fit_transform(X_train_g[categorical_cols])
X_train_g_final = np.hstack([X_train_g[numerical_cols].values, X_train_g_enc])

X_test_g_enc = encoder_g.transform(X_test_g[categorical_cols])
X_test_g_final = np.hstack([X_test_g[numerical_cols].values, X_test_g_enc])

rf_g = RandomForestRegressor(n_estimators=100, random_state=42)
rf_g.fit(X_train_g_final, y_train_g)

preds_g = rf_g.predict(X_test_g_final)
mae_g = mean_absolute_error(y_test_g, preds_g)
rmse_g = root_mean_squared_error(y_test_g, preds_g)
r2_g = r2_score(y_test_g, preds_g)

print(f"\n=== Plant-Grouped 80/20 Split (No Plant Leakage) ===")
print(f"Train plants: {len(train_plants)}, Test plants: {len(test_plants)} (Overlap: {len(overlap)})")
print(f"Train samples: {len(y_train_g)}, Test samples: {len(y_test_g)}")
print(f"MAE:  {mae_g:.4f} days")
print(f"RMSE: {rmse_g:.4f} days")
print(f"R^2:  {r2_g:.4f}")

# ==========================================
# 3. Evaluate Legacy Model Artifact
# ==========================================
legacy_model = joblib.load("plant_watering_model.pkl")
legacy_encoder = joblib.load("plant_encoder.pkl")

# Predict on conventional test set using legacy artifacts
legacy_enc_c = legacy_encoder.transform(X_test_c[categorical_cols])
legacy_X_c = np.hstack([X_test_c[numerical_cols].values, legacy_enc_c])
legacy_preds_c = legacy_model.predict(legacy_X_c)

print(f"\n=== Legacy Model Artifact Evaluation on Test Set ===")
print(f"MAE:  {mean_absolute_error(y_test_c, legacy_preds_c):.4f} days")
print(f"RMSE: {root_mean_squared_error(y_test_c, legacy_preds_c):.4f} days")
print(f"R^2:  {r2_score(y_test_c, legacy_preds_c):.4f}")

