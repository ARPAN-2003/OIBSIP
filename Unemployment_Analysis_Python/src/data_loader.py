import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    return df