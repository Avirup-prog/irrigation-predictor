# ============================================================
#  Crop Irrigation Need Predictor
#  SDG 2 - Zero Hunger | SDG 15 - Life on Land
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
#  STEP 1 — LOAD DATASET
# ============================================================

df = pd.read_csv("Crop_recommendation.csv")

print("=" * 50)
print("STEP 1: Dataset Loaded")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")


# ============================================================
#  STEP 2 — ENGINEER IRRIGATION LABEL
#
#  Logic: A crop NEEDS irrigation (1) when:
#    - rainfall is low  (below 25th percentile → < 65 mm)
#    - temperature is high (above 75th percentile → > 28°C)
#    - OR humidity is low  (below 25th percentile → < 60%)
#
#  Otherwise: no irrigation needed (0)
# ============================================================

def needs_irrigation(row):
    low_rainfall   = row["rainfall"]    < 65.0
    high_temp      = row["temperature"] > 28.0
    low_humidity   = row["humidity"]    < 60.0
    if low_rainfall and (high_temp or low_humidity):
        return 1
    return 0

df["irrigation_needed"] = df.apply(needs_irrigation, axis=1)

print("\n" + "=" * 50)
print("STEP 2: Irrigation Label Engineered")
print("=" * 50)
print(df["irrigation_needed"].value_counts())
print(f"\n0 = No irrigation needed")
print(f"1 = Irrigation needed")


# ============================================================
#  STEP 3 — FEATURE ENGINEERING
# ============================================================

# Input features (X) — all numeric columns except the label
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

X = df[FEATURES]
y = df["irrigation_needed"]

print("\n" + "=" * 50)
print("STEP 3: Features and Target Set")
print("=" * 50)
print(f"Features : {FEATURES}")
print(f"Target   : irrigation_needed")
print(f"X shape  : {X.shape}")
print(f"y shape  : {y.shape}")


# ============================================================
#  STEP 4 — TRAIN / TEST SPLIT & SCALING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 80% train, 20% test
    random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n" + "=" * 50)
print("STEP 4: Data Split and Scaled")
print("=" * 50)
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")


# ============================================================
#  STEP 5 — TRAIN 3 MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"      : DecisionTreeClassifier(random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

print("\n" + "=" * 50)
print("STEP 5: Training Models")
print("=" * 50)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred   = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        "model"   : model,
        "y_pred"  : y_pred,
        "accuracy": accuracy
    }
    print(f"\n{name}")
    print(f"  Accuracy : {accuracy * 100:.2f}%")
    print(f"  Report   :\n{classification_report(y_test, y_pred)}")


# ============================================================
#  STEP 6 — COMPARE MODELS (BAR CHART)
# ============================================================

print("\n" + "=" * 50)
print("STEP 6: Model Comparison Chart")
print("=" * 50)

names      = list(results.keys())
accuracies = [results[n]["accuracy"] * 100 for n in names]

plt.figure(figsize=(8, 5))
bars = plt.bar(names, accuracies, color=["#5DCAA5", "#7F77DD", "#1D9E75"])
plt.ylim(60, 100)
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison — Irrigation Predictor")
for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{acc:.2f}%",
        ha="center", fontsize=11
    )
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()
print("Chart saved as model_comparison.png")


# ============================================================
#  STEP 7 — CONFUSION MATRIX (BEST MODEL)
# ============================================================

best_name  = max(results, key=lambda n: results[n]["accuracy"])
best_model = results[best_name]["model"]
best_pred  = results[best_name]["y_pred"]

print("\n" + "=" * 50)
print(f"STEP 7: Confusion Matrix — Best Model ({best_name})")
print("=" * 50)

cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True, fmt="d",
    cmap="Greens",
    xticklabels=["No Irrigation", "Needs Irrigation"],
    yticklabels=["No Irrigation", "Needs Irrigation"]
)
plt.title(f"Confusion Matrix — {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix saved as confusion_matrix.png")


# ============================================================
#  STEP 8 — FEATURE IMPORTANCE (RANDOM FOREST)
# ============================================================

print("\n" + "=" * 50)
print("STEP 8: Feature Importance (Random Forest)")
print("=" * 50)

rf_model    = results["Random Forest"]["model"]
importances = rf_model.feature_importances_
feat_df     = pd.DataFrame({
    "Feature"   : FEATURES,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print(feat_df.to_string(index=False))

plt.figure(figsize=(8, 5))
sns.barplot(
    data=feat_df,
    x="Importance", y="Feature",
    palette="Greens_r"
)
plt.title("Feature Importance — Random Forest")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("Feature importance chart saved as feature_importance.png")


# ============================================================
#  STEP 9 — SAVE BEST MODEL & SCALER
# ============================================================

os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/irrigation_model.pkl")
joblib.dump(scaler,     "model/scaler.pkl")

print("\n" + "=" * 50)
print("STEP 9: Model Saved")
print("=" * 50)
print(f"Best model : {best_name}")
print(f"Saved to   : model/irrigation_model.pkl")
print(f"Scaler     : model/scaler.pkl")


# ============================================================
#  STEP 10 — QUICK PREDICTION TEST
# ============================================================

print("\n" + "=" * 50)
print("STEP 10: Quick Prediction Test")
print("=" * 50)

sample = pd.DataFrame([{
    "N"          : 20,
    "P"          : 30,
    "K"          : 20,
    "temperature": 35.0,
    "humidity"   : 45.0,
    "ph"         : 6.5,
    "rainfall"   : 40.0
}])

sample_scaled  = scaler.transform(sample)
prediction     = best_model.predict(sample_scaled)[0]
label          = "Irrigation NEEDED" if prediction == 1 else "No Irrigation Needed"

print(f"Sample input : {sample.iloc[0].to_dict()}")
print(f"Prediction   : {label}")

print("\n" + "=" * 50)
print("ALL STEPS COMPLETE!")
print("Output files: model_comparison.png, confusion_matrix.png,")
print("              feature_importance.png, model/irrigation_model.pkl")
print("=" * 50)
