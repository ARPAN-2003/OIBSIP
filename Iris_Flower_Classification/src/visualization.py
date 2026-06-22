import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_visualizations(data_path):
    
    # Create outputs folder if not exists
    os.makedirs("outputs", exist_ok=True)

    # Load dataset
    df = pd.read_csv(data_path)

    # Remove Id column if present
    if "Id" in df.columns:
        df = df.drop("Id", axis=1)

    print("Generating visualizations...")

    # -------------------------
    # 1. Pair Plot
    # -------------------------
    sns.pairplot(df, hue="Species")
    plt.savefig("outputs/pairplot.png")
    plt.close()

    # -------------------------
    # 2. Correlation Heatmap
    # -------------------------
    plt.figure(figsize=(8, 6))

    numeric_df = df.select_dtypes(include=["number"])

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Feature Correlation Heatmap")
    plt.savefig("outputs/heatmap.png")
    plt.close()

    # -------------------------
    # 3. Species Count Plot
    # -------------------------
    plt.figure(figsize=(6, 4))

    sns.countplot(
        x="Species",
        data=df
    )

    plt.title("Species Distribution")
    plt.savefig("outputs/species_distribution.png")
    plt.close()

    print("Visualizations saved successfully.")