import joblib

model = joblib.load("models/grouped_url_tfidf_model.pkl")
vectorizer = joblib.load("models/grouped_url_tfidf_vectorizer.pkl")

test_urls = [
    "http://paypal-login-security-example.com/login",
    "http://apple-id-verification-example.com/login",
    "http://secure-update-account-example.com/verify/password",
    "http://amazon-account-security-example.com/login",
    "http://microsoft-verification-example.com/signin",
    "http://google-account-verify-example.com/login",
    "http://bank-secure-login-example.com/verify",
    "http://facebook-security-update-example.com/login"
]

print("\n========== PHISHING URL STRESS TEST ==========\n")

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