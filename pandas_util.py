import pandas as pd


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
    
# https://stackoverflow.com/questions/42699243/how-to-build-a-lift-chart-a-k-a-gains-chart-in-python
def gains_chart(gains, color='C0', label=None, ax=None, figsize=None):
    """ Create a gains chart using predicted values
    Input:
        gains: must be sorted by probability
        color (optional): color of graph
        ax (optional): axis for matplotlib graph
        figsize (optional): size of matplotlib graph
    """
    nTotal = len(gains)  # number of records
    nActual = gains.sum()  # number of desired records

    # get cumulative sum of gains and convert to percentage
    cumGains = pd.concat([pd.Series([0]), gains.cumsum()])  # Note the additional 0 at the front
    gains_df = pd.DataFrame({'records': list(range(len(gains) + 1)), 'cumGains': cumGains})

    ax = gains_df.plot(x='records', y='cumGains', color=color, label=label, legend=False,
                       ax=ax, figsize=figsize)

    # Add line for random gain
    ax.plot([0, nTotal], [0, nActual], linestyle='--', color='k')
    ax.set_xlabel('# records')
    ax.set_ylabel('# cumulative gains')
    return ax

def lift_chart(predicted, title='Decile Lift Chart', labelBars=True, ax=None, figsize=None):
    """ Create a lift chart using predicted values
    Input:
        predictions: must be sorted by probability
        ax (optional): axis for matplotlib graph
        title (optional): set to None to suppress title
        labelBars (optional): set to False to avoid mean response labels on bar chart
    """
    # group the sorted predictions into 10 roughly equal groups and calculate the mean
    groups = [int(10 * i / len(predicted)) for i in range(len(predicted))]
    meanPercentile = predicted.groupby(groups).mean()
    # divide by the mean prediction to get the mean response
    meanResponse = meanPercentile / predicted.mean()
    meanResponse.index = (meanResponse.index + 1) * 10

    ax = meanResponse.plot.bar(color='C0', ax=ax, figsize=figsize)
    ax.set_ylim(0, 1.12 * meanResponse.max() if labelBars else None)
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Lift')
    if title:
        ax.set_title(title)

    if labelBars:
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}', (p.get_x(), p.get_height() + 0.1))
    return ax
