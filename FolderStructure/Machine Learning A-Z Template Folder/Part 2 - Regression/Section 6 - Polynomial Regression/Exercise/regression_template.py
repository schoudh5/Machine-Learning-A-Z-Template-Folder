#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# TODO : splitting the dataset and test set if needed

# TODO : apply feature scaling if needed
# create the regression here

# fitting the regression model to the dataset


# predicting a new result with Polynomial regression
y_pred = regressor.predict((6.5))

# visualizing the regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff(Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
# visualizing the regression results for higher resolution and smoother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()







