import pandas as pd

def separate_xy(dataframe):
    X = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, -1].values
    return [X, y]

def load(datapath):
    dataset = pd.read_csv(datapath, header=0)
    print(dataset.shape)
    [X, y] = separate_xy(dataset)
    return [X, y]