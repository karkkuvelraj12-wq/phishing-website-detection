from url_features import extract_features

url = "https://www.google.com"

features = extract_features(url)

print("\n========== URL ==========")
print(url)

print("\n========== EXTRACTED FEATURES ==========")

for name, value in features.items():
    print(f"{name:30} : {value}")