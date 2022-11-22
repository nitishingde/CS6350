import abc
import copy
from ml.Model import Model
import numpy as np
import pandas as pd
from typing import Callable


class SVMClassifier(Model, abc.ABC):

    def __int__(self, features: int = 0):
        self._weights = np.zeros(features)
        self._bias = 1.

    def fit(self, data_frame: pd.DataFrame, epochs: int = 10, C: float = 0.1, scheduler: Callable[[float], float] = None):
        """
        SVM Binary Classifier
        :param data_frame: data frame where the last column should be label with {-1, 1}
        :param epochs: number of iterations
        :param C: hyperparameter
        :param scheduler: function which accepts epoch parameter 't', and has parameters 'ynot' and 'a' baked in.
        :return: SVM model classifier
        """
        self.__int__(features=data_frame.columns.size-1)

        n = len(data_frame)
        for t in range(epochs):
            yt = scheduler(t)
            df = data_frame.sample(frac=1)
            for idx, row in df.iterrows():
                input_features = row[:-1].to_numpy()
                prediction = np.dot(self._weights, input_features) + self._bias
                yi = row[-1]
                if prediction * yi <= 1:
                    self._weights -= (yt*input_features - yt*C*n*yi*input_features)
                    self._bias += yt*C*n*yi*self._bias
                else:
                    self._weights -= yt*self._weights
                    # bias = bias

            print(f'\r{100. * t / epochs:.2f}%', end='')
        print('\r100.00%', end='')

    def predict(self, input_features):
        return np.sign(np.dot(self._weights, input_features) + self._bias)

    def predict_batch(self, batch_input_features: pd.DataFrame) -> pd.Series:
        predictions = pd.Series([None] * len(batch_input_features))
        for idx, input_features in batch_input_features.iterrows():
            predictions[idx] = self.predict(input_features)

        return predictions
