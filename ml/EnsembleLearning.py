import abc
import math
from ml.Model import Model
from ml.DecisionTree import DecisionTreeClassifier
import numpy as np
import pandas as pd


class AdaBoostClassifier(Model, abc.ABC):
    def __init__(self, T: int = 1):
        self._T = T
        self._alphas = np.zeros(T, dtype=np.float)
        self._hypothesises = [DecisionTreeClassifier() for _ in range(T)]

    def fit(self, df: pd.DataFrame, numerical_attributes: dict = None, T: int = 0, label: str = None, df_test: pd.DataFrame = None):
        self.__init__(T)
        m = len(df)
        weights = np.ones(m, dtype=np.float)/m
        weight_col = '__weights_elc'

        for t in range(T):
            print(f'{(100.0*t)/T}%', end='\r')
            new_df = df.copy(deep=True)
            new_df.insert(loc=df.columns.size-1, column=weight_col, value=weights, allow_duplicates=True)
            model = self._hypothesises[t]
            model.fit(new_df, numerical_attributes=numerical_attributes, weighted_attribute=weight_col, heuristic='entropy', max_depth=2)
            model.gen_tree()

            error = 0
            predictions = np.ones(len(df))
            for idx, row in new_df.iterrows():
                if model.predict(row) != row[label]:
                    error += row[weight_col]
                    predictions[idx] = 0

            self._alphas[t] = 0.5 * math.log((1 - error) / error)

            for i in range(len(weights)):
                if predictions[i]:
                    weights[i] *= math.exp(-self._alphas[t])
                else:
                    weights[i] *= math.exp(self._alphas[t])

            weights /= weights.sum()

    def predict(self, input_features, T: int = None):
        ans = 0
        T = self._T if T is None else T
        for i in range(T):
            alpha, model = self._alphas[i], self._hypothesises[i]
            ans += alpha*model.predict(input_features)

        return np.sign(ans)

    def predict_batch(self, batch_input_features, T: int = None) -> pd.Series:
        predictions = pd.Series([None] * len(batch_input_features))
        for row in range(len(batch_input_features)):
            predictions[row] = self.predict(batch_input_features.iloc[row, :], T=T)

        return predictions


class BaggingClassifier(Model, abc.ABC):
    def __init__(self, T: int = 0):
        self.hypothesises = [DecisionTreeClassifier() for _ in range(T)]

    def fit(self, df: pd.DataFrame, numerical_attributes: dict = None, T: int = 0, m: int = None, replace: bool = True, label: str = None):
        """
        :param df: dataframe needs to have column names and last column should be the label
        :param numerical_attributes: dict, numerical attributes as keys and corresponding aggregate function [mean, median]
        :param T: heuristic count
        :param m: sample count
        :param replace: allow or disallow sampling of the same row more than once.
        :param label: classifier label column name
        :return:
        """

        m = len(df) if m is None else m
        self.__init__(T)

        for t in range(T):
            print(f'{(100.0*t)/T}%', end='\r')
            new_df = df.sample(n=m, replace=replace).drop_duplicates(ignore_index=True)
            model = self.hypothesises[t]
            model.fit(df=new_df, numerical_attributes=numerical_attributes, heuristic='entropy')
        print('100%', end='\r')

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
