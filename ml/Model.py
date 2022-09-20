import abc
import pandas as pd


class Model(metaclass=abc.ABCMeta):
    def __int__(self):
        pass

    @abc.abstractmethod
    def fit(self, data_frame: pd.DataFrame):
        pass

    @abc.abstractmethod
    def predict(self, input_features):
        pass

    @abc.abstractmethod
    def predict_batch(self, batch_input_features) -> pd.Series:
        pass
