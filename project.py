import numpy as np
import pandas as pd
import sklearn as scikit_learn


datared = pd.read_csv('winequality-red.csv')
datawhite = pd.read_csv('winequality-white.csv')

datared.head(5)