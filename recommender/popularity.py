"""
Contains recommender systems algorithms that are based on popularity
"""
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.preprocessing import MinMaxScaler


def simple(csv: str, operation_column: str, ascending: Optional[bool] = False):
    """
    Sorts given operation column
    :param csv: csv file path location
    :param operation_column: column to operate on
    :param ascending: sort
    :return: dataframe sorted on given operation column
    """
    dataframe = pd.read_csv(csv)
    return dataframe.sort_values(operation_column, ascending=ascending)


def weighted_average(
    csv: str,
    operation_column: list,
    ascending: Optional[bool] = False,
    quantile: Optional[bool] = 0.70,
):
    """
    # see algorithm on http://trailerpark.weebly.com/imdb-rating.html
    :param csv: csv file path location
    :param operation_column: list of two columns [counts, average] respectively
    :param ascending: sort
    :param quantile: values at the given quantile on made `weighted_average` column
    :return: dataframe with new column `weighted_average`
    """
    dataframe = pd.read_csv(csv)
    count, average = operation_column
    R = dataframe[average]
    v = dataframe[count]
    C = dataframe[average].mean()
    m = dataframe[count].quantile(quantile)
    weighted = ((R * v) + (C * m)) / (v + m)
    dataframe["weighted_average"] = weighted
    return dataframe.sort_values("weighted_average", ascending=ascending)


def scale(dataframe: pd.DataFrame, dictionary: dict):
    """
    Scales and adds each column separately to the range 0-1 with each importance
    :param dataframe: pandas dataframe
    :param dictionary: column_name and its importance(in percent 0.10...) as dictionary
    :return: dataframe where `score` is final result
    """
    dataframe_cop = dataframe.copy()
    scaler = MinMaxScaler()
    cols = list(dictionary.keys())
    scaler_fit = scaler.fit_transform(dataframe_cop[cols])
    scaled_df = pd.DataFrame(scaler_fit, columns=cols)
    dataframe_cop.drop(cols, axis="columns", inplace=True)
    dataframe_cop[cols] = scaled_df[cols]
    score = np.zeros((len(dataframe_cop)))
    for i in dictionary.items():
        score += (dataframe_cop[i[0]] * i[1]).values
    dataframe_cop["score"] = score
    return dataframe_cop
