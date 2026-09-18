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

# Amount of data points/labels
print("Amount of red wine data points:", y_red.count())
print("Amount of white wine data points:", y_white.count())

# Min and max of labels
print("Min and max rating of red wines:", y_red.min(), y_red.max())
print("Min and max rating of white wines:" , y_white.min(), y_white.max())

# Mean of labels
print("Mean of rating for red wines:", y_red.mean())
print("Mean of rating for white wines:", y_white.mean())
print("\n")

# Extracting features
X_red = datared.drop(columns = "quality")
X_white = datawhite.drop(columns = "quality")

# Splitting red wine into train, test and validation sets (70/15/15)
X_red_train, X_red_left, y_red_train, y_red_left = train_test_split(X_red, y_red, test_size=0.3, random_state=42)
X_red_val, X_red_test, y_red_val, y_red_test = train_test_split(X_red_left, y_red_left, test_size=0.5, random_state=42)

# Same for white wine
X_white_train, X_white_left, y_white_train, y_white_left = train_test_split(X_white, y_white, test_size=0.3, random_state=42)
X_white_val, X_white_test, y_white_val, y_white_test = train_test_split(X_white_left, y_white_left, test_size=0.5, random_state=42)


print("RESULTS:")
# Red wine
reg1 = LinearRegression()
reg1.fit(X_red_train, y_red_train)
y_red_pred = reg1.predict(X_red_val)
tr_error_red = mean_squared_error(y_red_val, y_red_pred)
rms_red = np.sqrt(tr_error_red)
print("RMS error in red:", rms_red)

# White wine
reg2 = LinearRegression()
reg2.fit(X_white_train, y_white_train)
y_white_pred = reg2.predict(X_white_val)
tr_error_white = mean_squared_error(y_white_val, y_white_pred)
rms_white = np.sqrt(tr_error_white)
print("RMS error in white:", rms_white)


# White wine prediction figure
plt.figure(figsize=(8, 6))

# Predictions as a scatter plot
plt.scatter(y_white_val, y_white_pred, alpha=0.5, color='red', label='Predictions')

# Target predictions
plt.plot([3, 8], [3, 8], color='black', linestyle='--', label='Ideal predictions')

plt.title('Predicted vs actual quality of white wines')
plt.xlabel('Actual quality (validation data))')
plt.ylabel('Predicted quality')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

plt.savefig("white_wine_pred.png")

# Red wine prediction figure
plt.figure(figsize=(8, 6))

# Predictions as a scatter plot
plt.scatter(y_red_val, y_red_pred, alpha=0.5, color='red', label='Predictions')

# Target predictions
plt.plot([3, 8], [3, 8], color='black', linestyle='--', label='Ideal predictions')

plt.title('Predicted vs actual quality of red wines')
plt.xlabel('Actual quality (validation data))')
plt.ylabel('Predicted quality')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

plt.savefig("red_wine_pred.png")