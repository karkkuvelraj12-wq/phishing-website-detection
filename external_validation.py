import joblib

# Load the grouped-validation model
model = joblib.load("models/grouped_url_tfidf_model.pkl")
vectorizer = joblib.load("models/grouped_url_tfidf_vectorizer.pkl")

# External validation URLs.
# These are URL strings only — DO NOT open them.
test_data = [

    # =========================
    # LEGITIMATE
    # =========================

    ("https://www.google.com", 0),
    ("https://www.microsoft.com", 0),
    ("https://www.apple.com", 0),
    ("https://www.amazon.com", 0),
    ("https://www.wikipedia.org", 0),
    ("https://github.com", 0),
    ("https://www.linkedin.com", 0),
    ("https://www.youtube.com", 0),
    ("https://www.amazon.in", 0),
    ("https://www.amrita.edu", 0),
    ("https://www.python.org", 0),
    ("https://www.mozilla.org", 0),
    ("https://www.nasa.gov", 0),
    ("https://www.ibm.com", 0),
    ("https://www.oracle.com", 0),
    ("https://www.adobe.com", 0),
    ("https://www.netflix.com", 0),
    ("https://www.reddit.com", 0),
    ("https://www.stackoverflow.com", 0),
    ("https://www.w3.org", 0),

    # =========================
    # SYNTHETIC PHISHING
    # =========================

    ("http://paypal-login-security-example.com/login", 1),
    ("http://apple-id-verification-example.com/login", 1),
    ("http://secure-update-account-example.com/verify/password", 1),
    ("http://amazon-account-security-example.com/login", 1),
    ("http://microsoft-verification-example.com/signin", 1),
    ("http://google-account-verify-example.com/login", 1),
    ("http://bank-secure-login-example.com/verify", 1),
    ("http://facebook-security-update-example.com/login", 1),
    ("http://secure-paypal-account-example.com/verify", 1),
    ("http://amazon-security-check-example.com/password", 1),
]

urls = [item[0] for item in test_data]
actual = [item[1] for item in test_data]

# Transform URLs
X = vectorizer.transform(urls)

# Predict
predicted = model.predict(X)
probabilities = model.predict_proba(X)

print("\n========== EXTERNAL VALIDATION ==========\n")

correct = 0

for url, real_label, prediction, probability in zip(
    urls,
    actual,
    predicted,
    probabilities
):

    actual_name = "PHISHING" if real_label == 1 else "LEGITIMATE"
    predicted_name = "PHISHING" if prediction == 1 else "LEGITIMATE"

    confidence = max(probability) * 100

    if real_label == prediction:
        status = "CORRECT"
        correct += 1
    else:
        status = "WRONG"

    print("URL        :", url)
    print("Actual     :", actual_name)
    print("Prediction :", predicted_name)
    print(f"Confidence : {confidence:.2f}%")
    print("Result     :", status)
    print("-" * 65)


# Overall result
total = len(test_data)
accuracy = correct / total * 100

print("\n========== EXTERNAL RESULT ==========")
print(f"Correct : {correct}/{total}")
print(f"Accuracy: {accuracy:.2f}%")

print("\n========== TEST COMPLETED ==========")