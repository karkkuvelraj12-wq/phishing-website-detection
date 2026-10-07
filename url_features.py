import re
import math
from urllib.parse import urlparse


SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "secure", "account",
    "update", "banking", "confirm", "password",
    "payment", "support", "authentication"
]


def shannon_entropy(text):
    if not text:
        return 0

    probabilities = [
        text.count(char) / len(text)
        for char in set(text)
    ]

    return -sum(
        p * math.log2(p)
        for p in probabilities
    )


def extract_features(url):

    # Add scheme if user does not provide one
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    features = {}

    # Basic lengths
    features["url_len"] = len(url)
    features["dom_len"] = len(hostname)
    features["path_len"] = len(path)
    features["query_len"] = len(query)

    # IP address detection
    features["is_ip"] = int(
        bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", hostname))
    )

    # Subdomains
    parts = hostname.split(".")
    features["subdom_cnt"] = max(len(parts) - 2, 0)

    # Character counts
    features["letter_cnt"] = sum(c.isalpha() for c in url)
    features["digit_cnt"] = sum(c.isdigit() for c in url)

    special_chars = "!@#$%^&*()[]{};:',\"<>|\\"
    features["special_cnt"] = sum(
        c in special_chars for c in url
    )

    features["eq_cnt"] = url.count("=")
    features["qm_cnt"] = url.count("?")
    features["amp_cnt"] = url.count("&")
    features["dot_cnt"] = url.count(".")
    features["dash_cnt"] = url.count("-")
    features["under_cnt"] = url.count("_")
    features["slash_cnt"] = url.count("/")

    # Ratios
    length = max(len(url), 1)

    features["letter_ratio"] = (
        features["letter_cnt"] / length
    )

    features["digit_ratio"] = (
        features["digit_cnt"] / length
    )

    features["spec_ratio"] = (
        features["special_cnt"] / length
    )

    # HTTPS
    features["is_https"] = int(
        parsed.scheme == "https"
    )

    # Entropy
    features["entropy"] = shannon_entropy(url)

    # Suspicious keyword count
    lower_url = url.lower()

    features["suspicious_keyword_count"] = sum(
        keyword in lower_url
        for keyword in SUSPICIOUS_KEYWORDS
    )

    return features