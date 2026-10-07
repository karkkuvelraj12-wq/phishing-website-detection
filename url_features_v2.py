import re
import math
from urllib.parse import urlparse

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "verification",
    "secure", "security", "account", "update",
    "password", "passwd", "payment", "bank",
    "banking", "confirm", "authentication",
    "authorize", "wallet", "recover"
]


def shannon_entropy(text):
    if not text:
        return 0.0

    probabilities = [
        text.count(char) / len(text)
        for char in set(text)
    ]

    return -sum(
        p * math.log2(p)
        for p in probabilities
    )


def extract_features(url):

    original_url = url

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    full_url = original_url.lower()

    # Basic counts
    url_len = len(original_url)
    domain_len = len(hostname)
    path_len = len(path)
    query_len = len(query)

    digit_count = sum(c.isdigit() for c in original_url)
    letter_count = sum(c.isalpha() for c in original_url)

    special_count = sum(
        not c.isalnum()
        for c in original_url
    )

    # Character counts
    dot_count = original_url.count(".")
    dash_count = original_url.count("-")
    underscore_count = original_url.count("_")
    slash_count = original_url.count("/")
    question_count = original_url.count("?")
    equal_count = original_url.count("=")
    ampersand_count = original_url.count("&")
    percent_count = original_url.count("%")
    at_count = original_url.count("@")

    # Hostname structure
    subdomain_count = max(
        0,
        len(hostname.split(".")) - 2
    )

    # IP address detection
    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    is_ip = int(bool(re.match(ip_pattern, hostname)))

    # Protocol
    is_https = int(parsed.scheme == "https")

    # Suspicious keywords
    suspicious_keyword_count = sum(
        1 for word in SUSPICIOUS_WORDS
        if word in full_url
    )

    # Brand impersonation indicators
    brand_words = [
        "paypal", "apple", "amazon",
        "google", "microsoft",
        "facebook", "instagram",
        "linkedin", "netflix",
        "bank"
    ]

    brand_count = sum(
        1 for word in brand_words
        if word in hostname.lower()
    )

    # Ratios
    total_len = max(len(original_url), 1)

    digit_ratio = digit_count / total_len
    letter_ratio = letter_count / total_len
    special_ratio = special_count / total_len

    # Entropy
    entropy = shannon_entropy(original_url)

    # Path/query indicators
    has_login_path = int(
        any(
            word in path.lower()
            for word in ["login", "signin", "verify", "account"]
        )
    )

    has_sensitive_query = int(
        any(
            word in query.lower()
            for word in [
                "password",
                "passwd",
                "token",
                "auth",
                "login"
            ]
        )
    )

    return [
        url_len,
        domain_len,
        path_len,
        query_len,
        is_ip,
        subdomain_count,
        letter_count,
        digit_count,
        special_count,
        dot_count,
        dash_count,
        underscore_count,
        slash_count,
        question_count,
        equal_count,
        ampersand_count,
        percent_count,
        at_count,
        digit_ratio,
        letter_ratio,
        special_ratio,
        is_https,
        entropy,
        suspicious_keyword_count,
        brand_count,
        has_login_path,
        has_sensitive_query
    ]