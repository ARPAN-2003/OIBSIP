import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def correlation_heatmap(df):

    numeric_df = df.select_dtypes(include=["number"])

    plt.figure(figsize=(8,6))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        "outputs/correlation_heatmap.png"
    )

    plt.close()