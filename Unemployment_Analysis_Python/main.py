from src.data_loader import load_dataset

from src.analysis import (
    state_wise_unemployment,
    region_wise_employment,
    labour_participation
)

from src.visualization import (
    unemployment_by_state,
    unemployment_by_region,
    labour_chart,
    correlation_heatmap
)

from src.report import (
    generate_report
)


DATA_PATH = (
    "data/Unemployment in India.csv"
)


def main():

    df = load_dataset(DATA_PATH)

    print("\nDataset Shape:")
    print(df.shape)

    print("\nFirst 5 Rows:")
    print(df.head())

    state_data = state_wise_unemployment(df)

    region_data = region_wise_employment(df)

    labour_data = labour_participation(df)

    unemployment_by_state(
        state_data
    )

    unemployment_by_region(
        region_data
    )

    labour_chart(
        labour_data
    )

    correlation_heatmap(df)

    generate_report(
        state_data,
        labour_data
    )

    print(
        "\nAnalysis Complete."
    )

    print(
        "Check outputs folder."
    )


if __name__ == "__main__":
    main()