def generate_report(
        state_data,
        labour_data):

    with open(
        "outputs/summary_report.txt",
        "w"
    ) as file:

        file.write(
            "UNEMPLOYMENT ANALYSIS REPORT\n"
        )

        file.write(
            "="*50 + "\n\n"
        )

        file.write(
            "Top 5 States with Highest Unemployment:\n\n"
        )

        file.write(
            str(state_data.head())
        )

        file.write("\n\n")

        file.write(
            "Top 5 Labour Participation States:\n\n"
        )

        file.write(
            str(labour_data.head())
        )