import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def unemployment_by_state(state_data):

    plt.figure(figsize=(12,8))

    state_data.head(15).plot(kind="bar")

    plt.title("Top States with Highest Unemployment")
    plt.ylabel("Unemployment Rate (%)")

    plt.tight_layout()

    plt.savefig(
        "outputs/unemployment_by_state.png"
    )

    plt.close()


def unemployment_by_region(region_data):

    plt.figure(figsize=(8,5))

    region_data.plot(
        kind="bar",
        color=["skyblue","orange"]
    )

    plt.title("Urban vs Rural Unemployment")

    plt.tight_layout()

    plt.savefig(
        "outputs/unemployment_by_region.png"
    )

    plt.close()


def labour_chart(data):

    plt.figure(figsize=(12,6))

    data.head(15).plot(kind="bar")

    plt.title(
        "Labour Participation Rate"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/labour_participation.png"
    )

    plt.close()


def correlation_heatmap(df):

    plt.figure(figsize=(8,6))

    numeric_df = df.select_dtypes(include='number')

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/heatmap.png"
    )

    plt.close()