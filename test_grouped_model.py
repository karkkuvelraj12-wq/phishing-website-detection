import joblib

# Load the GROUPED model
model = joblib.load("models/grouped_url_tfidf_model.pkl")
vectorizer = joblib.load("models/grouped_url_tfidf_vectorizer.pkl")

test_urls = [
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.amazon.com",
    "https://www.wikipedia.org",
    "https://github.com",
    "https://www.linkedin.com",
    "https://www.youtube.com",
    "https://www.amazon.in",
    "https://www.amrita.edu"
]

print("\n========== GROUPED MODEL LEGITIMATE TEST ==========\n")

X = vectorizer.transform(test_urls)

predictions = model.predict(X)
probabilities = model.predict_proba(X)

for url, prediction, probability in zip(
    test_urls,
    predictions,
    probabilities
):

    result = "PHISHING" if prediction == 1 else "LEGITIMATE"
    confidence = max(probability) * 100

    print("URL        :", url)
    print("Prediction :", result)
    print(f"Confidence : {confidence:.2f}%")
    print("-" * 60)

print("\n========== TEST COMPLETED ==========")