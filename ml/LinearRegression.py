import abc
import copy
import graphviz
import math

from numpy import ndarray

from ml.Model import Model
import numpy as np
import pandas as pd
from typing import Tuple, Any
import uuid


class LinearRegressionClassifier(Model, abc.ABC):
    def __int__(self, bias_weight: np.array = None):
        self._bias_weight = bias_weight
        self._info = {'cost': []}

    def fit(self, df: pd.DataFrame, heuristic: str = 'bgd', bias_weight: np.array = None, learning_rate: float = 0.01, error_tolerance: float = 1e-6):
        """
        :param df: dataframe
        :param heuristic: [default = 'bgd', 'sgd']
        :param bias_weight:
        :param learning_rate:
        :param error_tolerance:
        :return:
        """

        # add bias
        # df.insert(0, column='__bias_lrc', value=np.ones(len(df)), allow_duplicates=True)
        bias_weight = np.random.rand(df.columns.size-1) if bias_weight is None else bias_weight
        self.__int__(bias_weight=bias_weight)

        calc_gradient = None
        if heuristic == 'bgd':
            calc_gradient = self._batch_gradient_descent
        elif heuristic == 'sgd':
            calc_gradient = self._stochastic_gradient_descent
        else:
            raise Exception("heuristic: [default = 'bgd', 'sgd']")

        diff = 1
        while error_tolerance < diff:
            grad = calc_gradient(df)
            prev = copy.deepcopy(self._bias_weight)
            self._bias_weight -= (learning_rate * grad)
            diff = np.linalg.norm(prev-self._bias_weight)
            self._info['cost'].append(self._cost(df))
            print(f'\r{100.*error_tolerance/diff:.4f}%', end='')

        # df.drop('__bias_lrc', axis=1, inplace=True)

    def predict(self, input_features):
        return np.dot(self._bias_weight, input_features)

    def predict_batch(self, batch_input_features) -> np.ndarray:
        return np.dot(batch_input_features, self._bias_weight)

    def _batch_gradient_descent(self, df: pd.DataFrame) -> np.array:
        grad = np.zeros(len(self._bias_weight))
        for idx, row in df.iterrows():
            grad -= (row[-1] - np.dot(row[:-1].values, self._bias_weight)) * row[:-1].values

        return grad

    def _stochastic_gradient_descent(self, df: pd.DataFrame, index: int = None) -> np.array:
        row = df.iloc[np.random.randint(low=0, high=len(df)) if index is None else index, :]
        grad = -1 * (row[-1] - np.dot(row[:-1].values, self._bias_weight)) * row[:-1].values
        return grad

    def _cost(self, df: pd.DataFrame) -> float:
        cost = 0.
        for idx, row in df.iterrows():
            cost += pow(row[-1] - np.dot(row[:-1].values, self._bias_weight), 2)

        return 0.5*cost
