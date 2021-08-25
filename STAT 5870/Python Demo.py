# Run a Cell
print("Hello, World!")

# You may use Python as a pocket calculator
1 + 2
1/2 + 1/2
(1 + 2) * 2

3 ** 2
3 ** (1/2) 

# Data Analysis using Python
# Generate random data set
import numpy as np

np.random.seed(1)
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.rand(100, 1)

# Draw a scatter plot
import matplotlib.pyplot as plt

plt.scatter(x, y)

# Fit a simple linear regression model
from sklearn.linear_model import LinearRegression

# Model initialization
regression_model = LinearRegression()
# Fit simple linear regression model
regression_model.fit(x, y)
# See estimated slope and intercept coefficients
print("Model slope:", regression_model.coef_)
print("Model intercept:", regression_model.intercept_)

# Prediction
y_predicted = regression_model.predict(x)
y_predicted
# Mean Squared Error (MSE)
mse = np.sum((y_predicted - y) ** 2)
print("Mean Squared Error:", mse)
