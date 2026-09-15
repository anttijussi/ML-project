import numpy as np
import pandas as pd
import sklearn as scikit_learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

datared = pd.read_csv('wine+quality/winequality-red.csv', sep=';')
datawhite = pd.read_csv('wine+quality/winequality-white.csv', sep=';')

y_red = datared["quality"]
y_white = datawhite["quality"]

X_red = datared.drop(columns = "quality")
X_white = datawhite.drop(columns = "quality")

reg = LinearRegression()
reg.fit(X_red, y_red)
y_red_pred = reg.predict(X_red)
tr_error = mean_squared_error(y_red, y_red_pred)
print(tr_error)