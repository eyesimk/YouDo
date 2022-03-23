import pandas as pd
from sklearn import datasets
import numpy as np

cal_housing = datasets.fetch_california_housing()
X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
X = X["MedInc"]
y = cal_housing.target


# df = pd.DataFrame(dict(MedInc=X['MedInc'], Price=cal_housing.target))


class LinearRegression:

    def __init__(self):
        self.weight = 0
        self.bias = 0
        self.history = {
            "w": [],
            "b": [],
            "loss": []
        }

    def fit(self, X, y, iterations=1000, learning_rate=0.01):
        theta = 4
        for iteration in range(iterations):
            pred = self.weight * X + self.bias
            dif = y - pred

            dif = dif.apply(lambda x: theta if x > theta else x)

            # mean squared error
            mse = sum(dif ** 2) / len(X)

            # gradient of loss function: G f(w,b) = [df(w,b)/dw, df(w,b)/db]
            gradient = [
                sum(2 * (-X) * dif) / len(X),
                sum(2 * -1 * dif) / len(X)
            ]

            self.weight -= gradient[0] * learning_rate
            self.bias -= gradient[1] * learning_rate

            self.history["w"].append(self.weight)
            self.history["b"].append(self.bias)
            self.history["loss"].append(mse)

    def predict(self, d):
        pred = self.weight * d + self.bias
        return pred


model1 = LinearRegression()
model1.fit(X, y)

loss = model1.history["loss"]

loss = pd.DataFrame(loss)

preds1 = model1.predict(X)


class LinearRegression_regularized:

    def __init__(self):
        self.weight = 0
        self.bias = 0
        self.history = {
            "w": [],
            "b": [],
            "loss": []
        }

    def fit(self, X, y, lam, iterations=1000, learning_rate=0.01):
        lam = 0.5
        theta = 4
        for iteration in range(iterations):
            pred = self.weight * X + self.bias
            dif = y - pred

            dif = dif.apply(lambda x: theta if x > theta else x)

            # loss function
            reg_loss = (sum(dif ** 2) + lam * (self.weight ** 2 + self.bias ** 2)) / len(X)

            # gradient of loss function: G f(w,b) = [df(w,b)/dw, df(w,b)/db]
            gradient = [

                (sum(2 * (-X) * dif) + 2 * lam * self.weight) / len(X),
                (sum(2 * -1 * dif) + 2 * lam * self.bias) / len(X)
            ]

            self.weight -= gradient[0] * learning_rate
            self.bias -= gradient[1] * learning_rate

            self.history["w"].append(self.weight)
            self.history["b"].append(self.bias)
            self.history["loss"].append(reg_loss)

    def predict(self, d):
        pred = self.weight * d + self.bias
        return pred


model = LinearRegression_regularized()
model.fit(X, y, 5)

loss = model.history["loss"]

loss = pd.DataFrame(loss)

loss.plot()

preds2 = model.predict(X)
print(model.weight, model.bias)
print(max(preds2 - y))

