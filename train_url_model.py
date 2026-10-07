import os
import pandas as pd
import joblib

from url_features import extract_features

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. CREATE MODEL FOLDER
# ==========================================

os.makedirs("models", exist_ok=True)


# ==========================================
# 2. LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv(
    "dataset/cleaned_phishing_dataset.csv"
)

print("\n========== DATASET ==========")
print("Rows:", len(df))


# ==========================================
# 3. EXTRACT URL FEATURES
# ==========================================

print("\n========== EXTRACTING URL FEATURES ==========")

feature_data = []

for i, url in enumerate(df["domain"]):

    features = extract_features(str(url))

    feature_data.append(features)

    if (i + 1) % 10000 == 0:
        print(f"Processed {i + 1} URLs")


X = pd.DataFrame(feature_data)

y = df["label"]


print("\nFeature matrix shape:", X.shape)


# ==========================================
# 4. DISPLAY FEATURES
# ==========================================

print("\n========== FEATURES ==========")
print(X.columns.tolist())


# ==========================================
# 5. TRAIN-TEST SPLIT
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
# 6. CREATE RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)


# ==========================================
# 7. TRAIN
# ==========================================

print("\n========== TRAINING URL MODEL ==========")

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 8. PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 9. EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========== URL MODEL PERFORMANCE ==========")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")


# ==========================================
# 10. CLASSIFICATION REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Legitimate",
            "Phishing"
        ]
    )
)


# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

print("\n========== CONFUSION MATRIX ==========")

print(confusion_matrix(y_test, y_pred))


# ==========================================
# 12. FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")

print(
    importance.to_string(index=False)
)


# ==========================================
# 13. SAVE MODEL
# ==========================================

model_path = "models/url_phishing_random_forest.pkl"

joblib.dump(
    model,
    model_path
)

print("\n========== MODEL SAVED ==========")
print(model_path)


# ==========================================
# 14. SAVE FEATURE NAMES
# ==========================================

feature_path = "models/url_features.pkl"

joblib.dump(
    list(X.columns),
    feature_path
)

print("Feature list saved:")
print(feature_path)


# ==========================================
# 15. COMPLETE
# ==========================================

print("\n==========================================")
print("URL-BASED MODEL TRAINING COMPLETED")
print("==========================================")