import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. CREATE OUTPUT FOLDERS
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ==========================================
# 2. LOAD CLEANED DATASET
# ==========================================

file_path = "dataset/cleaned_phishing_dataset.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET ==========")
print("Shape:", df.shape)


# ==========================================
# 3. SELECT FEATURES
# ==========================================

features = [
    "ranking",
    "mld_res",
    "mld.ps_res",
    "card_rem",
    "ratio_Rrem",
    "ratio_Arem",
    "jaccard_RR",
    "jaccard_RA",
    "jaccard_AR",
    "jaccard_AA",
    "jaccard_ARrd",
    "jaccard_ARrem"
]

X = df[features]
y = df["label"]


# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 5. CREATE RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 6. TRAIN MODEL
# ==========================================

print("\n========== TRAINING ==========")

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 7. PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 8. PERFORMANCE
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")


# ==========================================
# 9. CLASSIFICATION REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Phishing"]
    )
)


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Phishing"]
)

disp.plot()

plt.title("Phishing Website Detection - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300
)

plt.close()


# ==========================================
# 11. FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=True
)

print("\n========== FEATURE IMPORTANCE ==========")
print(
    importance.sort_values(
        by="Importance",
        ascending=False
    ).to_string(index=False)
)

plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png",
    dpi=300
)

plt.close()


# ==========================================
# 12. SAVE MODEL
# ==========================================

model_file = "models/phishing_random_forest.pkl"

joblib.dump(model, model_file)

print("\n========== MODEL SAVED ==========")
print(model_file)


# ==========================================
# 13. SAVE FEATURE LIST
# ==========================================

feature_file = "models/features.pkl"

joblib.dump(features, feature_file)

print("Feature list saved:")
print(feature_file)


# ==========================================
# 14. FINAL
# ==========================================

print("\n==========================================")
print("PROJECT TRAINING COMPLETED SUCCESSFULLY")
print("==========================================")