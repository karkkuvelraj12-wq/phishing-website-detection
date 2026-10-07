import tkinter as tk
from tkinter import messagebox
import joblib


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = "models/grouped_url_tfidf_model.pkl"
VECTORIZER_PATH = "models/grouped_url_tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# ==========================================
# ANALYZE URL
# ==========================================

def analyze_url():

    url = url_entry.get().strip()

    if not url:
        messagebox.showwarning(
            "Input Required",
            "Please enter a URL."
        )
        return

    try:
        # Convert URL into TF-IDF representation
        X = vectorizer.transform([url])

        # Prediction
        prediction = model.predict(X)[0]

        # Probability
        probabilities = model.predict_proba(X)[0]
        confidence = max(probabilities) * 100

        # ==================================
        # THREE-LEVEL DECISION
        # ==================================

        if confidence < 70:

            result_label.config(
                text="⚠  UNCERTAIN",
                fg="#d97706"
            )

            explanation_label.config(
                text="The model is not sufficiently confident.\n"
                     "Treat this URL with caution."
            )

        elif prediction == 1:

            result_label.config(
                text="⚠  PHISHING URL",
                fg="#dc2626"
            )

            explanation_label.config(
                text="The URL shows patterns associated with phishing."
            )

        else:

            result_label.config(
                text="✓  LEGITIMATE URL",
                fg="#15803d"
            )

            explanation_label.config(
                text="The URL is classified as legitimate by the model."
            )

        confidence_label.config(
            text=f"Model Confidence: {confidence:.2f}%"
        )

        status_label.config(
            text="Analysis completed successfully."
        )

    except Exception as e:

        messagebox.showerror(
            "Analysis Error",
            f"Unable to analyze the URL.\n\n{e}"
        )


# ==========================================
# CLEAR
# ==========================================

def clear_url():

    url_entry.delete(0, tk.END)

    result_label.config(
        text="Enter a URL to analyze",
        fg="black"
    )

    confidence_label.config(
        text="Model Confidence: --"
    )

    explanation_label.config(
        text="Results will appear here."
    )

    status_label.config(
        text="Ready for analysis."
    )


# ==========================================
# MAIN WINDOW
# ==========================================

root = tk.Tk()

root.title("Phishing Website Detection")
root.geometry("850x650")

root.resizable(False, False)

root.configure(bg="#f4f6f8")


# ==========================================
# TITLE
# ==========================================

title_label = tk.Label(
    root,
    text="PHISHING WEBSITE DETECTION",
    font=("Arial", 26, "bold"),
    bg="#f4f6f8"
)

title_label.pack(pady=(35, 8))


subtitle_label = tk.Label(
    root,
    text="Machine Learning Based URL Analysis",
    font=("Arial", 13),
    bg="#f4f6f8"
)

subtitle_label.pack(pady=(0, 35))


# ==========================================
# INPUT SECTION
# ==========================================

input_frame = tk.Frame(
    root,
    bg="#f4f6f8"
)

input_frame.pack()


input_label = tk.Label(
    input_frame,
    text="Enter Website URL",
    font=("Arial", 13, "bold"),
    bg="#f4f6f8"
)

input_label.pack(pady=(0, 10))


url_entry = tk.Entry(
    input_frame,
    width=75,
    font=("Arial", 12),
    relief=tk.SOLID,
    borderwidth=1
)

url_entry.pack(
    ipady=9
)


# ==========================================
# BUTTON SECTION
# ==========================================

button_frame = tk.Frame(
    root,
    bg="#f4f6f8"
)

button_frame.pack(pady=18)


analyze_button = tk.Button(
    button_frame,
    text="ANALYZE URL",
    font=("Arial", 11, "bold"),
    command=analyze_url,
    padx=25,
    pady=10,
    cursor="hand2"
)

analyze_button.pack(
    side=tk.LEFT,
    padx=8
)


clear_button = tk.Button(
    button_frame,
    text="CLEAR",
    font=("Arial", 11, "bold"),
    command=clear_url,
    padx=25,
    pady=10,
    cursor="hand2"
)

clear_button.pack(
    side=tk.LEFT,
    padx=8
)


# ==========================================
# RESULT CARD
# ==========================================

result_frame = tk.Frame(
    root,
    bg="white",
    relief=tk.GROOVE,
    borderwidth=2,
    width=750,
    height=230
)

result_frame.pack(
    pady=20,
    padx=50
)

result_frame.pack_propagate(False)


result_label = tk.Label(
    result_frame,
    text="Enter a URL to analyze",
    font=("Arial", 22, "bold"),
    bg="white"
)

result_label.pack(
    pady=(30, 12)
)


confidence_label = tk.Label(
    result_frame,
    text="Model Confidence: --",
    font=("Arial", 14),
    bg="white"
)

confidence_label.pack(
    pady=5
)


explanation_label = tk.Label(
    result_frame,
    text="Results will appear here.",
    font=("Arial", 11),
    bg="white",
    justify=tk.CENTER
)

explanation_label.pack(
    pady=8
)


status_label = tk.Label(
    result_frame,
    text="Ready for analysis.",
    font=("Arial", 10),
    bg="white"
)

status_label.pack(
    pady=5
)


# ==========================================
# INFORMATION SECTION
# ==========================================

info_label = tk.Label(
    root,
    text=(
        "Model: Character-level TF-IDF + Logistic Regression\n"
        "Validation: Grouped by hostname\n"
        "Note: This tool analyzes the URL string only."
    ),
    font=("Arial", 10),
    bg="#f4f6f8",
    justify=tk.CENTER
)

info_label.pack(
    pady=15
)


# ==========================================
# FOOTER
# ==========================================

footer_label = tk.Label(
    root,
    text="Phishing Website Detection | AI & Data Science Mini Project",
    font=("Arial", 9),
    bg="#f4f6f8"
)

footer_label.pack(
    side=tk.BOTTOM,
    pady=12
)


# ==========================================
# RUN APPLICATION
# ==========================================

root.mainloop()