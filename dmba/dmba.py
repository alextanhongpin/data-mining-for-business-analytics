import pandas as pd
import os
import matplotlib.pylab as plt

# plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams['figure.dpi'] = 96

dirpath = "./datasets/dmba"


def load_data(name: str, **kwargs):
    fullpath = os.path.join(dirpath, name)
    df = pd.read_csv(fullpath, **kwargs)
    normalize_columns(df)

    return df


def load_boston_housing():
    """
    crim: per capita crime rate by town.
    zn: proportion of residential land zoned for lots over 25,000 sq.ft.
    indus: proportion of non-retail business acres per town.
    chas: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
    nox: nitrogen oxides concentration (parts per 10 million).
    rm: average number of rooms per dwelling.
    age: proportion of owner-occupied units built prior to 1940.
    dis: weighted mean of distances to five Boston employment centres.
    rad: index of accessibility to radial highways.
    tax: full-value property-tax rate per \$10,000.
    ptratio: pupil-teacher ratio by town.
    black: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
    lstat: lower status of the population (percent).
    cat_medv: median value of owner-occupied homes in \$1001s.
    """
    df = pd.read_csv("./datasets/dmba/BostonHousing.csv")
    df = df.rename(columns={"CAT. MEDV": "CAT_MEDV"})
    normalize_columns(df)

    return df


def normalize_columns(df):
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
