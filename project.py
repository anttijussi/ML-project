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


# Red wine
reg1 = LinearRegression()
reg1.fit(X_red, y_red)
y_red_pred = reg1.predict(X_red)
tr_error_red = mean_squared_error(y_red, y_red_pred)
print("RMS error in red: ", tr_error_red)

# White wine
reg2 = LinearRegression()
reg2.fit(X_white, y_white)
y_white_pred = reg2.predict(X_white)
tr_error_white = mean_squared_error(y_white, y_white_pred)
print("RMS error in white: ", tr_error_white)