# Fixing matplotlib backend
import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import pandas as pd
import sklearn as scikit_learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Acquiring data
datared = pd.read_csv('wine+quality/winequality-red.csv', sep=';')
datawhite = pd.read_csv('wine+quality/winequality-white.csv', sep=';')

# Combining the two dataframes and creating a new binary feature for the color
datared["is_red"] = 1
datawhite["is_red"] = 0
newdata = pd.concat([datared, datawhite], ignore_index=True)

# Fetchinq the desired features and labels
y = newdata["quality"]
X = newdata.drop(columns = ["quality", "density", "total sulfur dioxide", "citric acid"])

# Splitting the data into training, validation and testing data
X_train, X_left, y_train, y_left = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_left, y_left, test_size=0.5, random_state=42)

#Standardising the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Applying linear regression
reg = LinearRegression()
reg.fit(X_train_scaled, y_train)
y_pred_reg = reg.predict(X_val_scaled)
rms_reg = np.sqrt(mean_squared_error(y_val, y_pred_reg))
print("RMS error with linear regression: ", rms_reg)

# Applying random forest
forest = RandomForestRegressor(random_state=42)
forest.fit(X_train_scaled, y_train)
y_pred_forest = forest.predict(X_val_scaled)
rms_forest = np.sqrt(mean_squared_error(y_val, y_pred_forest))
print("RMS error with random forest: ", rms_forest)

# Seemingly random forest is the more accurate model. Let us calculate the final RMS with the testing data
X_test_scaled = scaler.transform(X_test)
y_pred_forest_final = forest.predict(X_test_scaled)
rms_forest_final = np.sqrt(mean_squared_error(y_test, y_pred_forest_final))
print("RMS error for random forest with testing data: ", rms_forest_final)

# Linear regression visualisation
plt.figure(figsize=(8, 6))

# Predictions as a scatter plot
plt.scatter(y_val, y_pred_reg, alpha=0.5, color='red', label='Predictions')

# Target predictions
plt.plot([3, 8], [3, 8], color='black', linestyle='--', label='Ideal predictions')

plt.title('Predicted vs actual quality')
plt.xlabel('Actual quality (validation data))')
plt.ylabel('Predicted quality')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

plt.show()



# Random forest visualisation
plt.figure(figsize=(8, 6))

# Predictions as a scatter plot
plt.scatter(y_val, y_pred_forest, alpha=0.5, color='red', label='Predictions')

# Target predictions
plt.plot([3, 8], [3, 8], color='black', linestyle='--', label='Ideal predictions')

plt.title('Predicted vs actual quality')
plt.xlabel('Actual quality (validation data))')
plt.ylabel('Predicted quality')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

plt.show()