import abc
import math
from ml.Model import Model
from ml.DecisionTree import DecisionTreeClassifier
import numpy as np
import pandas as pd


class BaggingClassifier(Model, abc.ABC):
    def __init__(self, T: int = 0):
        self.hypothesises = [DecisionTreeClassifier() for _ in range(T)]

    def fit(self, df: pd.DataFrame, numerical_attributes: dict = None, T: int = 0, label: str = None):
        self.__init__(T)
        for t in range(T):
            print(f'{(100.0*t)/T}%', end='\r')
            new_df = df.sample(n=len(df), replace=True).drop_duplicates(ignore_index=True)
            model = self.hypothesises[t]
            model.fit(df=new_df, numerical_attributes=numerical_attributes, heuristic='entropy')

    def predict(self, input_features, T: int = 0):
        prediction = 0
        for i in range(T):
            prediction += self.hypothesises[i].predict(input_features)

        return np.sign(prediction)

    def predict_batch(self, batch_input_features, T: int = 0) -> pd.Series:
        predictions = pd.Series([None] * len(batch_input_features))
        for row in range(len(batch_input_features)):
            predictions[row] = self.predict(batch_input_features.iloc[row, :], T=T)

        return predictions
