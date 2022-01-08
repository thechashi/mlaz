# Data Preprocessing 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Taking care fo missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer() # axis = 0 column, axis = 1 rows
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

