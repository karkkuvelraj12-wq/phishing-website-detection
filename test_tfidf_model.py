import joblib


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(
    "models/url_tfidf_logistic_model.pkl"
)

vectorizer = joblib.load(
    "models/url_tfidf_vectorizer.pkl"
)


# ==========================================
# TEST URLs
# ==========================================

test_urls = [
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.amazon.com",
    "https://www.wikipedia.org",
    "https://www.github.com",
    "https://www.linkedin.com",
    "https://www.youtube.com",
    "https://www.amazon.in",
    "https://www.amrita.edu"
]


# ==========================================
# TEST MODEL
# ==========================================

print("\n========== TF-IDF MODEL STRESS TEST ==========\n")

for url in test_urls:

    X = vectorizer.transform([url])

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    if prediction == 1:

        result = "PHISHING"
        confidence = probabilities[1] * 100

    else:

        result = "LEGITIMATE"
        confidence = probabilities[0] * 100

    print(f"URL        : {url}")
    print(f"Prediction : {result}")
    print(f"Confidence : {confidence:.2f}%")
    print("-" * 60)