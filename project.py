import numpy as np
import pandas as pd
import sklearn as scikit_learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Reading CSV's into pandas
datared = pd.read_csv('wine+quality/winequality-red.csv', sep=';')
datawhite = pd.read_csv('wine+quality/winequality-white.csv', sep=';')

# Extracting labels
y_red = datared["quality"]
y_white = datawhite["quality"]

# Min and max of labels
print("Min and max rating of red wines: ", y_red.min(), y_red.max())
print("Min and max rating of white wines: ", y_white.min(), y_white.max())

# Mean of labels
print("Mean of rating for red wines: ", y_red.mean())
print("Mean of rating for white wines: ", y_white.mean())

# Extracting features
X_red = datared.drop(columns = "quality")
X_white = datawhite.drop(columns = "quality")

# Splitting red wine into train, test and validation sets (70/15/15)
X_red_train, X_red_left, y_red_train, y_red_left = train_test_split(X_red, y_red, test_size=0.3, random_state=42)
X_red_val, X_red_test, y_red_val, y_red_test = train_test_split(X_red_left, y_red_left, test_size=0.5, random_state=42)

# Same for white wine
X_white_train, X_white_left, y_white_train, y_white_left = train_test_split(X_white, y_white, test_size=0.3, random_state=42)
X_white_val, X_white_test, y_white_val, y_white_test = train_test_split(X_white_left, y_white_left, test_size=0.5, random_state=42)

# Red wine
reg1 = LinearRegression()
reg1.fit(X_red_train, y_red_train)
y_red_pred = reg1.predict(X_red_val)
tr_error_red = mean_squared_error(y_red_val, y_red_pred)
rms_red = np.sqrt(tr_error_red)
print("RMS error in red: ", rms_red)

# White wine
reg2 = LinearRegression()
reg2.fit(X_white_train, y_white_train)
y_white_pred = reg2.predict(X_white_val)
tr_error_white = mean_squared_error(y_white_val, y_white_pred)
rms_white = np.sqrt(tr_error_white)
print("RMS error in white: ", rms_white)