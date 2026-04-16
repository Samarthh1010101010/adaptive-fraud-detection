import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)

    X = df.drop("Class", axis=1).values
    y = df["Class"].values

    return train_test_split(X, y, test_size=0.3, random_state=42)
