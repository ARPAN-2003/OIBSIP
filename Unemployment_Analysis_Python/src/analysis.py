def state_wise_unemployment(df):

    result = (
        df.groupby("Region")["Estimated Unemployment Rate (%)"]
        .mean()
        .sort_values(ascending=False)
    )

    return result


def region_wise_employment(df):

    result = (
        df.groupby("Area")["Estimated Unemployment Rate (%)"]
        .mean()
    )

    return result


def labour_participation(df):

    result = (
        df.groupby("Region")["Estimated Labour Participation Rate (%)"]
        .mean()
        .sort_values(ascending=False)
    )

    return result