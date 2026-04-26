# =============================================================================
# KNN on Diabetes Dataset
# Course: Computational Science for Computer Science
# Topic: Machine Learning – K-Nearest Neighbors
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# =============================================================================
# PART 1: DATA UNDERSTANDING
# =============================================================================
print("=" * 60)
print("PART 1: DATA UNDERSTANDING")
print("=" * 60)

feature_descriptions = {
    "Pregnancies":              "Number of times the patient has been pregnant.",
    "Glucose":                  "Plasma glucose concentration (2-hour oral glucose tolerance test). Higher = diabetes risk.",
    "BloodPressure":            "Diastolic blood pressure (mm Hg).",
    "SkinThickness":            "Triceps skin fold thickness (mm), used to estimate body fat.",
    "Insulin":                  "2-hour serum insulin (mu U/ml).",
    "BMI":                      "Body Mass Index (weight kg / height m²), indicator of body fatness.",
    "DiabetesPedigreeFunction": "Score based on family history indicating genetic risk of diabetes.",
    "Age":                      "Patient's age in years.",
    "Outcome":                  "Target class: 0 = Non-diabetic, 1 = Diabetic.",
}

print("\nFeature Descriptions:")
for feat, desc in feature_descriptions.items():
    print(f"  {feat}: {desc}")

print("\nMost Important Features for Prediction:")
print("  Glucose, BMI, Age, Insulin, DiabetesPedigreeFunction")

print("\nFeatures with Likely Missing/Zero Values:")
print("  Insulin, SkinThickness, BloodPressure, Glucose, BMI")

# =============================================================================
# PART 2: DATA PREPROCESSING
# =============================================================================
print("\n" + "=" * 60)
print("PART 2: DATA PREPROCESSING")
print("=" * 60)

# Load dataset
url = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
try:
    df = pd.read_csv(url)
    print("\nDataset loaded from URL.")
except Exception:
    print("Could not load from URL. Please place 'diabetes.csv' in the same directory.")
    import sys; sys.exit(1)

print(f"\nOriginal dataset shape: {df.shape}")

# Features that cannot be biologically zero
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

print("\nZero-value counts before cleaning:")
for col in zero_cols:
    zeros = (df[col] == 0).sum()
    print(f"  {col}: {zeros} zeros ({zeros/len(df)*100:.1f}%)")

# --- Before stats ---
print("\nMean values BEFORE preprocessing:")
before_means = df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].mean()
for col, val in before_means.items():
    print(f"  {col}: {val:.2f}")

# Handle missing data: remove rows where any zero-column is 0
df_clean = df.copy()
df_clean = df_clean[(df_clean[zero_cols] != 0).all(axis=1)]

print(f"\nDataset shape after removing invalid rows: {df_clean.shape}")
print(f"  Rows removed: {len(df) - len(df_clean)}")

# --- After stats ---
print("\nMean values AFTER preprocessing:")
after_means = df_clean[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].mean()
for col, val in after_means.items():
    print(f"  {col}: {val:.2f}")

# Before/after comparison table
print("\nBefore vs After Preprocessing (Mean):")
print(f"{'Feature':<30} {'Before':>10} {'After':>10}")
print("-" * 52)
for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    print(f"  {col:<28} {before_means[col]:>10.2f} {after_means[col]:>10.2f}")

# Separate features and target
X = df_clean.drop("Outcome", axis=1).values
y = df_clean["Outcome"].values

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature scaling applied: Standardization (zero mean, unit variance)")
print(f"  Sample scaled row[0]: {X_scaled[0].round(3)}")

# =============================================================================
# PART 3: KNN IMPLEMENTATION
# =============================================================================
print("\n" + "=" * 60)
print("PART 3: KNN IMPLEMENTATION")
print("=" * 60)

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# --- Manual Euclidean Distance Computation ---
print("\n--- Manual Euclidean Distance Computation ---")
print("Test Instance X: [Pregnancies=1, Glucose=103, BP=80, SkinThick=11,")
print("                   Insulin=82, BMI=19.4, DPF=0.49, Age=22]")

# Raw (unscaled) training samples from the document
manual_samples = np.array([
    [1,  89,  66, 23,  94, 28.1, 0.167, 21],
    [0, 137,  40, 35, 168, 43.1, 2.288, 33],
    [3, 126,  88, 41, 235, 39.3, 0.704, 27],
    [3,  78,  50, 32,  88, 31.0, 0.248, 26],
    [2, 197,  70, 45, 543, 30.5, 0.158, 53],
    [1, 189,  60, 23, 846, 30.1, 0.398, 59],
    [5, 166,  72, 19, 175, 25.8, 0.587, 51],
    [0, 118,  84, 47, 230, 45.8, 0.551, 31],
    [1, 103,  30, 38,  83, 43.3, 0.183, 33],
    [1, 115,  70, 30,  96, 34.6, 0.529, 32],
])
manual_outcomes = [0, 1, 0, 1, 1, 1, 1, 1, 0, 1]
test_instance   = np.array([1, 103, 80, 11, 82, 19.4, 0.49, 22])

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

print(f"\n{'ID':<4} {'Distance':>10}  {'Outcome'}")
print("-" * 30)
distances = []
for i, (sample, outcome) in enumerate(zip(manual_samples, manual_outcomes), 1):
    d = euclidean_distance(test_instance, sample)
    distances.append((i, d, outcome))
    print(f"S{i:<3} {d:>10.2f}  {outcome}")

# Sort by distance
distances_sorted = sorted(distances, key=lambda x: x[1])
print("\nSorted (nearest first):")
print(f"{'Rank':<5} {'ID':<5} {'Distance':>10}  {'Outcome'}")
print("-" * 35)
for rank, (sid, d, outcome) in enumerate(distances_sorted, 1):
    print(f"{rank:<5} S{sid:<4} {d:>10.2f}  {outcome}")

# K=3 prediction
k3_neighbors = distances_sorted[:3]
k3_votes = [n[2] for n in k3_neighbors]
k3_pred = max(set(k3_votes), key=k3_votes.count)
print(f"\nK=3 Nearest Neighbors: {[(f'S{n[0]}', n[2]) for n in k3_neighbors]}")
print(f"  Votes → Class 0: {k3_votes.count(0)}, Class 1: {k3_votes.count(1)}")
print(f"  Predicted Outcome: {k3_pred} ({'Diabetic' if k3_pred == 1 else 'Non-Diabetic'})")

# =============================================================================
# PART 4: MODEL EVALUATION
# =============================================================================
print("\n" + "=" * 60)
print("PART 4: MODEL EVALUATION")
print("=" * 60)

k_values   = [3, 5, 7]
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"\n--- K = {k} ---")
    print(f"  Accuracy: {acc * 100:.2f}%")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:\n{cm}")

best_k   = k_values[np.argmax(accuracies)]
best_acc = max(accuracies)
print(f"\nBest K: {best_k}  (Accuracy: {best_acc * 100:.2f}%)")

print("\nEvaluation Answers:")
print(f"  1. Best K = {best_k}. Smaller K is too noisy; larger K washes out local patterns.")
print("  2. Performance changes because K controls the decision boundary smoothness.")
print("  3. K too small → overfitting (sensitive to outliers).")
print("     K too large → underfitting (predicts the majority class for most instances).")

# =============================================================================
# BONUS: VISUALIZATION — Accuracy vs K
# =============================================================================
print("\n" + "=" * 60)
print("BONUS: VISUALIZATION")
print("=" * 60)

# Extended K range for a richer plot
k_range     = range(1, 21)
acc_list    = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc_list.append(accuracy_score(y_test, knn.predict(X_test)))

plt.figure(figsize=(10, 5))
plt.plot(k_range, acc_list, marker="o", color="steelblue", linewidth=2)
plt.axvline(x=best_k, color="red", linestyle="--", label=f"Best K={best_k}")
plt.title("KNN Accuracy vs K Value (Diabetes Dataset)")
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.xticks(list(k_range))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_vs_k.png", dpi=150)
plt.show()
print("  Saved: accuracy_vs_k.png")

# Confusion Matrix Plot (best K)
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred_best = knn_best.predict(X_test)

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_best,
    display_labels=["Non-Diabetic", "Diabetic"],
    cmap="Blues", ax=ax
)
ax.set_title(f"Confusion Matrix (K={best_k})")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("  Saved: confusion_matrix.png")

# =============================================================================
# BONUS: COMPARISON — KNN vs Logistic Regression
# =============================================================================
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))

print("\n--- KNN vs Logistic Regression ---")
print(f"  KNN  (K={best_k}) Accuracy: {best_acc * 100:.2f}%")
print(f"  Logistic Regression Accuracy: {lr_acc * 100:.2f}%")

plt.figure(figsize=(6, 4))
bars = plt.bar(
    [f"KNN (K={best_k})", "Logistic Regression"],
    [best_acc * 100, lr_acc * 100],
    color=["steelblue", "darkorange"],
    edgecolor="black"
)
for bar, val in zip(bars, [best_acc * 100, lr_acc * 100]):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 3,
             f"{val:.1f}%", ha="center", va="top", color="white", fontweight="bold")
plt.ylim(0, 100)
plt.title("Model Comparison: KNN vs Logistic Regression")
plt.ylabel("Accuracy (%)")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()
print("  Saved: model_comparison.png")

print("\n" + "=" * 60)
print("DONE. All parts completed successfully.")
print("=" * 60)