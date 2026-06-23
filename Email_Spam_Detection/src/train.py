import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from preprocess import transform_text


# ------------------------------
# Create folders automatically
# ------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ------------------------------
# Load Dataset
# ------------------------------

df = pd.read_csv(
    "data/spam.csv",
    encoding="latin1"
)

df = df.iloc[:, 0:2]

df.columns = [
    "label",
    "message"
]


# ------------------------------
# Encode Labels
# ------------------------------

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


# ------------------------------
# Text Preprocessing
# ------------------------------

df["message"] = df["message"].apply(transform_text)


# ------------------------------
# TF-IDF
# ------------------------------

vectorizer = TfidfVectorizer(
    max_features=3000
)

X = vectorizer.fit_transform(
    df["message"]
)

y = df["label"]


# ------------------------------
# Train Test Split
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# ------------------------------
# Model
# ------------------------------

model = MultinomialNB()

model.fit(X_train, y_train)


# ------------------------------
# Prediction
# ------------------------------

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)


# ------------------------------
# Console Output
# ------------------------------

print("\n")

print("=" * 60)

print("EMAIL SPAM DETECTION")

print("=" * 60)

print()

print(f"Accuracy : {accuracy*100:.2f}%")

print()

print(
    classification_report(
        y_test,
        prediction
    )
)

print("=" * 60)


# ------------------------------
# Save Model
# ------------------------------

joblib.dump(
    model,
    "models/spam_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/vectorizer.pkl"
)


# ------------------------------
# Confusion Matrix
# ------------------------------

cm = confusion_matrix(
    y_test,
    prediction
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Ham",
        "Spam"
    ]
)

fig, ax = plt.subplots(
    figsize=(6, 6)
)

disp.plot(
    ax=ax,
    colorbar=False
)

plt.title(
    "Confusion Matrix"
)

plt.savefig(
    "outputs/confusion_matrix.png"
)

plt.close()


# ------------------------------
# Accuracy Chart
# ------------------------------

fig, ax = plt.subplots(
    figsize=(5, 4)
)

ax.bar(
    ["Accuracy"],
    [accuracy * 100]
)

ax.set_ylim(0, 100)

ax.set_ylabel("Percentage")

ax.set_title("Model Accuracy")

plt.savefig("outputs/accuracy_graph.png")

plt.close()


print()

print("Model saved successfully.")

print("Vectorizer saved successfully.")

print()

print("Generated:")

print("outputs/confusion_matrix.png")

print("outputs/accuracy_graph.png")

print()

print("Training Completed Successfully!")
