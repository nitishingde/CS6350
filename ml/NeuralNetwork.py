import abc
from ml.Model import Model
import numpy as np
import pandas as pd


class NN2Classifier(Model, abc.ABC):
    def __init__(self, n: int = 4, width: int = 5, initializer=np.random.standard_normal):
        super().__init__()
        self.w1 = initializer((width, n))      # (5, 4)
        self.w2 = initializer((width, width))  # (5, 5)
        self.w3 = initializer((1, width))      # (1, 5)

        self.b1 = initializer((1, width))  # (1, 5)
        self.b2 = initializer((1, width))  # (1, 5)
        self.b3 = initializer((1, 1))      # (1, 1)

    def fit(self, data_frame: pd.DataFrame, width: int = 5, y0: float = 0.001, d: int = 8, epochs: int = 100, initializer=np.random.standard_normal):
        self.__init__(n=data_frame.columns.size-1, width=width, initializer=initializer)

        for t in range(epochs):
            yt = y0/(1+y0*t/d)
            for _ in range(len(data_frame)):
                df_row = data_frame.sample(n=1, replace=False)
                x_train, y_train = df_row.iloc[:, :data_frame.columns.size-1], df_row.iloc[:, data_frame.columns.size-1].to_numpy()
                # forward pass
                z1 = np.dot(x_train, self.w1.transpose()) + self.b1
                a1 = self._sigmoid(z1)
                z2 = np.dot(a1, self.w2.transpose()) + self.b2
                a2 = self._sigmoid(z2)
                y = np.dot(a2, self.w3.transpose()) + self.b3
                L = 0.5 * np.square((y - y_train))

                # backpropagation, find all gradients
                dldy = y - y_train
                dldb3 = float(dldy.sum())
                dldw3 = np.dot(dldy.transpose(), a2)

                dldz2 = np.dot(dldy, self.w3) * a2 * (1 - a2)
                # dldz2 = dldy * self.w3.transpose() * a2 * (1 - a2)
                dldb2 = dldz2.sum(axis=0)
                dldw2 = np.dot(dldz2.transpose(), a1)

                dldz1 = np.dot(dldz2, self.w2) * a1 * (1 - a1)
                dldb1 = dldz1.sum(axis=0)
                dldw1 = np.dot(dldz1.transpose(), x_train)

                # update all weights, biases after finding all the gradients
                self.b3 = self.b3 - yt * dldb3
                self.w3 = self.w3 - yt * dldw3
                self.b2 = self.b2 - yt * dldb2
                self.w2 = self.w2 - yt * dldw2
                self.b1 = self.b1 - yt * dldb1
                self.w1 = self.w1 - yt * dldw1

    def predict(self, input_features):
        z1 = np.dot(input_features, self.w1.transpose()) + self.b1
        a1 = self._sigmoid(z1)
        z2 = np.dot(a1, self.w2.transpose()) + self.b2
        a2 = self._sigmoid(z2)
        return int(np.round(np.dot(a2, self.w3.transpose()) + self.b3))

    def predict_batch(self, batch_input_features) -> pd.Series:
        predictions = pd.Series([None] * len(batch_input_features))
        for idx, input_features in batch_input_features.iterrows():
            predictions[idx] = self.predict(input_features)

        return predictions

    @staticmethod
    def _sigmoid(z):
        return 1./(1+np.exp(-z))
