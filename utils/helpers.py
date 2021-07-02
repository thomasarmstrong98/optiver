from typing import Union

import numpy as np
import pandas as pd


def log_return(prices: Union[np.array, pd.Series]):
    return np.log(prices).diff()


def realized_volatility(log_returns: Union[np.array, pd.Series]):
    return np.sqrt(np.sum(log_returns ** 2))


def rmspe(y_true: Union[np.array, pd.Series], y_pred: Union[np.array, pd.Series]):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))


def linear_time_weighted_average(data: Union[np.array, pd.Series], beta: float):
    """
    Takes in some :param data: that is at linear intervals and then computes an average
    in which the later values have linearly more weighting than previous. We give
    the first value a weighting of 1, the following of 1 + :param beta:, and the final
    value gets a rating of 1 + :param beta:*(len(data)-1).
    :param data: any numerical data
    :param beta: slope parameter
    :return: np.double
    """

    weights = [1 + (w - 1) * beta for w in np.arange(len(data))]
    return np.sum(weights * data) / np.sum(weights)