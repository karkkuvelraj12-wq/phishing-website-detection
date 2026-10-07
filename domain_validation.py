import pandas as pd
import joblib

from urllib.parse import urlparse
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv("dataset/cleaned_phishing_dataset.csv")
df["domain"] = df["domain"].astype(str)


# ==============================
# EXTRACT HOSTNAME
# ==============================

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

df = df[df["hostname"] != ""].copy()


# ==============================
# DATASET INFORMATION
# ==============================

print("\n========== DATASET ==========")

print("Rows:", len(df))
print("Unique hostnames:", df["hostname"].nunique())


# ==============================
# GROUPED TRAIN/TEST SPLIT
# ==============================

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(
        df["domain"],
        df["label"],
        groups=df["hostname"]
    )
)

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]


print("\n========== GROUPED SPLIT ==========")

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

print(
    "Training hostnames:",
    train_df["hostname"].nunique()
)

print(
    "Testing hostnames:",
    test_df["hostname"].nunique()
)


# ==============================
# TF-IDF
# ==============================

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=150000,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(
    train_df["domain"]
)

X_test = vectorizer.transform(
    test_df["domain"]
)

y_train = train_df["label"]
y_test = test_df["label"]


# ==============================
# TRAIN MODEL
# ==============================

print("\n========== TRAINING ==========")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

print("Training completed!")


# ==============================
# EVALUATION
# ==============================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("\n========== GROUPED VALIDATION ==========")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")


print("\n========== CONFUSION MATRIX ==========")

print(confusion_matrix(y_test, y_pred))


# ==============================
# SAVE GROUPED MODEL
# ==============================

joblib.dump(
    model,
    "models/grouped_url_tfidf_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/grouped_url_tfidf_vectorizer.pkl"
)

print("\n========== MODEL SAVED ==========")

print("models/grouped_url_tfidf_model.pkl")
print("models/grouped_url_tfidf_vectorizer.pkl")

print("\nDONE!")