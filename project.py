import numpy as np
import pandas as pd
import sklearn as scikit_learn


datared = pd.read_csv('wine+quality/winequality-red.csv', sep=';')
datawhite = pd.read_csv('wine+quality/winequality-white.csv', sep=';')

print(datared.head(5))