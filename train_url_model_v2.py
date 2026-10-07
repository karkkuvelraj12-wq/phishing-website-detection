import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv(
    "dataset/cleaned_phishing_dataset.csv"
)

print("\n========== DATASET ==========")
print("Rows:", len(df))


# ==========================================
# 2. PREPARE URL TEXT
# ==========================================

urls = df["domain"].astype(str)
labels = df["label"].astype(int)


# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    urls,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

print("\n========== DATA SPLIT ==========")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. CHARACTER TF-IDF
# ==========================================

print("\n========== TF-IDF ==========")

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=150000,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("Training matrix:", X_train_tfidf.shape)
print("Testing matrix:", X_test_tfidf.shape)


# ==========================================
# 5. LOGISTIC REGRESSION
# ==========================================

print("\n========== TRAINING ==========")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)

print("Training completed!")


# ==========================================
# 6. PREDICTION
# ==========================================

y_pred = model.predict(X_test_tfidf)


# ==========================================
# 7. PERFORMANCE
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
# 8. CLASSIFICATION REPORT
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
# 9. CONFUSION MATRIX
# ==========================================

print("\n========== CONFUSION MATRIX ==========")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================
# 10. SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/url_tfidf_logistic_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/url_tfidf_vectorizer.pkl"
)


# ==========================================
# 11. SUCCESS
# ==========================================

print("\n========== MODEL SAVED ==========")

print(
    "models/url_tfidf_logistic_model.pkl"
)

print(
    "models/url_tfidf_vectorizer.pkl"
)

print("\n==========================================")
print("URL TF-IDF MODEL COMPLETED")
print("==========================================")