import math
import numpy as np
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error


def adjusted_r2_score(y_true, y_pred, model):
    """ calculate adjusted R2
    Input:
        y_true: actual values
        y_pred: predicted values
        model: predictive model
    """
    n = len(y_pred)
    p = len(model.coef_)
    if p >= n - 1:
        return 0
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def AIC_score(y_true, y_pred, model=None, df=None):
    """ calculate Akaike Information Criterion (AIC)
    Input:
        y_true: actual values
        y_pred: predicted values
        model (optional): predictive model
        df (optional): degrees of freedom of model
    One of model or df is requried
    """
    if df is None and model is None:
        raise ValueError('You need to provide either model or df')
    n = len(y_pred)
    p = len(model.coef_) + 1 if df is None else df
    resid = np.array(y_true) - np.array(y_pred)
    sse = np.sum(resid ** 2)
    constant = n + n * np.log(2 * np.pi)
    return n * math.log(sse / n) + constant + 2 * (p + 1)


def BIC_score(y_true, y_pred, model=None, df=None):
    """ calculate Schwartz's Bayesian Information Criterion (AIC)
    Input:
        y_true: actual values
        y_pred: predicted values
        model: predictive model
        df (optional): degrees of freedom of model
    """
    aic = AIC_score(y_true, y_pred, model=model, df=df)
    p = len(model.coef_) + 1 if df is None else df
    n = len(y_pred)
    return aic - 2 * (p + 1) + math.log(n) * (p + 1)


def classification_summary(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print()
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print()
    print("Classification report:")
    print(classification_report(y_true, y_pred))


def regression_summary(y_true, y_pred):
    print('RMSE:', mean_squared_error(y_true, y_pred))
    print('MAE:', mean_absolute_error(y_true, y_pred))
