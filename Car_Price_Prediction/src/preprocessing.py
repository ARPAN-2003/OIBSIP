import pandas as pd

def preprocess_data(filepath):

    df = pd.read_csv(filepath)

    df.drop_duplicates(inplace=True)

    df.dropna(inplace=True)

    return df