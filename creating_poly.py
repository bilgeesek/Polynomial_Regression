# Polynomial Regression

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Polynomial Regression model on the entire dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# You can set the degree as you wish
poly_regression = PolynomialFeatures(degree=5)
X_poly = poly_regression.fit_transform(X)
lin_regression = LinearRegression()
lin_regression.fit(X_poly, y)

# Visualising the Polynomial Regression result
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[7.5]]))
