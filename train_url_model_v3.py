import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from url_features_v2 import extract_features


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/cleaned_phishing_dataset.csv")

df["domain"] = df["domain"].astype(str)

print("\n========== DATASET ==========")
print("Rows:", len(df))


# ==========================================
# EXTRACT FEATURES
# ==========================================

print("\n========== EXTRACTING FEATURES ==========")

X = np.array(
    [extract_features(url) for url in df["domain"]]
)

y = df["label"].astype(int).values


print("Feature matrix shape:", X.shape)


# ==========================================
# EXTRACT HOSTNAMES FOR GROUPED SPLIT
# ==========================================

from urllib.parse import urlparse


def get_hostname(url):

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        hostname = urlparse(url).hostname

        if hostname:
            return hostname.lower()

    except Exception:
        pass

    return ""


df["hostname"] = df["domain"].apply(get_hostname)

valid = df["hostname"] != ""

X = X[valid]
y = y[valid]

groups = df.loc[valid, "hostname"].values


print("Unique hostnames:", len(np.unique(groups)))


# ==========================================
# GROUPED TRAIN / TEST SPLIT
# ==========================================

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X, y, groups=groups)
)

X_train = X[train_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]


print("\n========== GROUPED SPLIT ==========")

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ==========================================
# TRAIN RANDOM FOREST
# ==========================================

print("\n========== TRAINING MODEL ==========")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# EVALUATION
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("\n========== MODEL RESULTS ==========")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")


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


print("\n========== CONFUSION MATRIX ==========")

print(confusion_matrix(y_test, y_pred))


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

feature_names = [
    "url_len",
    "domain_len",
    "path_len",
    "query_len",
    "is_ip",
    "subdomain_count",
    "letter_count",
    "digit_count",
    "special_count",
    "dot_count",
    "dash_count",
    "underscore_count",
    "slash_count",
    "question_count",
    "equal_count",
    "ampersand_count",
    "percent_count",
    "at_count",
    "digit_ratio",
    "letter_ratio",
    "special_ratio",
    "is_https",
    "entropy",
    "suspicious_keyword_count",
    "brand_count",
    "has_login_path",
    "has_sensitive_query"
]

importance = model.feature_importances_

feature_importance = sorted(
    zip(feature_names, importance),
    key=lambda x: x[1],
    reverse=True
)

print("\n========== FEATURE IMPORTANCE ==========")

for name, value in feature_importance:
    print(f"{name:30s} {value:.6f}")


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/url_features_v3_random_forest.pkl"
)

joblib.dump(
    feature_names,
    "models/url_features_v3_names.pkl"
)

print("\n========== MODEL SAVED ==========")

print("models/url_features_v3_random_forest.pkl")
print("models/url_features_v3_names.pkl")

print("\nDONE!")