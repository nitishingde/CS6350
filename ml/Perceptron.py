import abc
import copy
from ml.Model import Model
import numpy as np
import pandas as pd


class StandardPerceptronClassifier(Model, abc.ABC):
    def __int__(self, features: int = 0):
        self._weights = np.zeros(features)
        self._bias = 1.

    def fit(self, data_frame: pd.DataFrame, epochs: int = 10, learning_rate: int = 0.1):
        self.__int__(features=data_frame.columns.size-1)

        for epoch in range(epochs):
            df = data_frame.sample(frac=1)
            for idx, row in df.iterrows():
                input_features = row[:-1].to_numpy()
                prediction = np.sign(np.dot(self._weights, input_features) + self._bias)
                if prediction != row[-1]:
                    self._weights += learning_rate * row[-1] * input_features
                    self._bias += learning_rate * row[-1] * self._bias
            print(f'\r{100. * epoch / epochs:.2f}%', end='')
        print('\r100.00%', end='')

    def predict(self, input_features):
        return np.sign(np.dot(self._weights, input_features) + self._bias)

    def predict_batch(self, batch_input_features: pd.DataFrame) -> pd.Series:
        predictions = pd.Series([None] * len(batch_input_features))
        for idx, input_features in batch_input_features.iterrows():
            predictions[idx] = self.predict(input_features)

        return predictions


class VotedPerceptronClassifier(Model, abc.ABC):
    class Vote:
        def __init__(self, ndim):
            self.weights = np.zeros(ndim)
            self.bias = 1.
            self.c = 1

        def __str__(self):
            return f'weights={self.weights}, bias={self.bias}, c={self.c}'

        def __repr__(self):
            return self.__str__()

    def __int__(self):
        self._votes = []

    def fit(self, data_frame: pd.DataFrame, epochs: int = 10, learning_rate: int = 0.1):
        self.__int__()

        vote = self.Vote(data_frame.columns.size-1)

        for epoch in range(epochs):
            df = data_frame.sample(frac=1)
            for idx, row in df.iterrows():
                input_features = row[:-1].to_numpy()
                prediction = np.sign(np.dot(vote.weights, input_features) + vote.bias)
                if prediction != row[-1]:
                    self._votes.append(copy.deepcopy(vote))
                    vote.weights += learning_rate * row[-1] * input_features
                    vote.bias += learning_rate * row[-1] * vote.bias
                    vote.c = 1
                else:
                    vote.c += 1
            print(f'\r{100. * epoch / epochs:.2f}%', end='')
        self._votes.append(vote)
        print('\r100.00%', end='')

    def predict(self, input_features):
        prediction = 0
        for vote in self._votes:
            prediction += vote.c*np.sign(np.dot(vote.weights, input_features) + vote.bias)

        return np.sign(prediction)

    def predict_batch(self, batch_input_features: pd.DataFrame) -> pd.Series:
        predictions = pd.Series([None] * len(batch_input_features))
        for idx, input_features in batch_input_features.iterrows():
            predictions[idx] = self.predict(input_features)

        return predictions


class AveragePerceptronClassifier(Model, abc.ABC):
    def __int__(self, features: int = 0):
        self._avg_weights = np.zeros(features)
        self._avg_bias = 1.

    def fit(self, data_frame: pd.DataFrame, epochs: int = 10, learning_rate: int = 0.1):
        self.__int__(data_frame.columns.size-1)

        weights = np.zeros(data_frame.columns.size-1)
        bias = 1.
        for epoch in range(epochs):
            df = data_frame.sample(frac=1)
            for idx, row in df.iterrows():
                input_features = row[:-1].to_numpy()
                prediction = np.sign(np.dot(weights, input_features) + bias)
                if prediction != row[-1]:
                    weights += learning_rate * row[-1] * input_features
                    bias += learning_rate * row[-1] * bias
                self._avg_weights += weights
                self._avg_bias += bias
            print(f'\r{100. * epoch / epochs:.2f}%', end='')
        print('\r100.00%', end='')

    def predict(self, input_features):
        return np.sign(np.dot(self._avg_weights, input_features) + self._avg_bias)

    def predict_batch(self, batch_input_features: pd.DataFrame) -> pd.Series:
        predictions = pd.Series([None] * len(batch_input_features))
        for idx, input_features in batch_input_features.iterrows():
            predictions[idx] = self.predict(input_features)

        return predictions
