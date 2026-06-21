import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


def train_model(data_path):

    # Load dataset
    df = pd.read_csv(data_path)

    # Remove Id column
    if "Id" in df.columns:
        df.drop("Id", axis=1, inplace=True)

    # ==============================
    # Correlation Matrix
    # ==============================
    print("\n" + "=" * 50)
    print("CORRELATION MATRIX")
    print("=" * 50)

    print(df.drop("Species", axis=1).corr())

    # Features and Target
    X = df.drop("Species", axis=1)
    y = df["Species"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # ==============================
    # Confusion Matrix
    # ==============================
    cm = confusion_matrix(y_test, predictions)

    print("\n" + "=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)

    print(cm)

    # Save Confusion Matrix Image
    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    print("\nConfusion Matrix image saved:")
    print("outputs/confusion_matrix.png")

    # ==============================
    # Classification Report
    # ==============================
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    # ==============================
    # Accuracy
    # ==============================
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"\nAccuracy : {accuracy * 100:.2f}%")

    # ==============================
    # Save Model
    # ==============================
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        "models/iris_model.pkl"
    )

    print("\nModel saved successfully.")
    print("models/iris_model.pkl")

    return model