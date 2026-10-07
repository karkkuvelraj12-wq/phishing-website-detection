import joblib


# ==========================================
# LOAD TF-IDF MODEL
# ==========================================

model = joblib.load(
    "models/url_tfidf_logistic_model.pkl"
)

vectorizer = joblib.load(
    "models/url_tfidf_vectorizer.pkl"
)


# ==========================================
# KNOWN LEGITIMATE DOMAINS
# These are used only as a sanity check.
# They are NOT part of the model's test metrics.
# ==========================================

legitimate_urls = [
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.amazon.com",
    "https://www.wikipedia.org",
    "https://github.com",
    "https://www.linkedin.com",
    "https://www.youtube.com",
    "https://www.amrita.edu",
    "https://www.nasa.gov",
    "https://www.ibm.com",
    "https://www.oracle.com",
    "https://www.python.org",
    "https://www.mozilla.org",
    "https://www.adobe.com"
]


# ==========================================
# SUSPICIOUS-LOOKING TEST STRINGS
#
# These are synthetic examples.
# DO NOT OPEN THEM.
# They are NOT claimed to be real phishing sites.
# ==========================================

suspicious_urls = [
    "http://paypal-login-secure-example.com/verify/account",
    "http://secure-bank-account-example.com/login/password",
    "http://microsoft-verification-example.com/signin",
    "http://amazon-account-example.com/confirm/payment",
    "http://apple-id-verification-example.com/login",
    "http://secure-update-account-example.com/verify/password"
]


# ==========================================
# FUNCTION
# ==========================================

def test_urls(urls, category):

    print(f"\n========== {category} ==========\n")

    correct = 0

    for url in urls:

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
        print("-" * 65)


# ==========================================
# RUN TESTS
# ==========================================

test_urls(
    legitimate_urls,
    "LEGITIMATE DOMAIN SANITY TEST"
)

test_urls(
    suspicious_urls,
    "SYNTHETIC SUSPICIOUS-URL TEST"
)

print("\n========== TEST COMPLETED ==========")