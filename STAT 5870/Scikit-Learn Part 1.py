import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# The steps in using the Scikit-Learn are as follows:
#
# 1. Choose a class of model by importing the appropriate estimator class from Scikit-Learn.
# 2. Choose model hyperparameters by instantiating this class with desired values.
# 3. Arrange data into a features matrix and target vector.
# 4. Fit the model to your data by calling the fit() method of the model instance.
# 5. Apply the model to new data:
#    - For supervised learning, we predict the labels for unknown data using the predict() method.
#    - For unsupervised learning, we transform or infer properties of the data using the transform() or predict() method.


# Supervised learning example: Simple Linear Regression
# Regression Problem
#
# np.random.rand(d0, d1, ...)
# return array with given shape with random samples from a uniform distribution over [0, 1).
np.random.seed(seed=0)
x = 10 * np.random.rand(50)
y = -1 + 2 * x + np.random.normal(0, 1, 50)
plt.scatter(x, y)

# 1. Choose a class of model.
# import linear regression class
from sklearn.linear_model import LinearRegression

# 2. Choose model hyperparameters.
# instantiate the LinearRegression class and specify fit_intercept hyperparameter
model = LinearRegression(fit_intercept = True)
model

# 3. Arragne data into a featrues matrix and target vector.
x.shape
y.shape

X = x[:, np.newaxis]
X.shape

y = y[:, np.newaxis]
y.shape

# 4. Fit the model to your data.
# this can be done with the fit() method
model.fit(X, y)

# all model parameters that were learned during the fit() process have trailing underscores
model.coef_ # estimated slope coefficient
model.intercept_ # estimated intercept coefficient 

# 5. Predict labels for unknown data.
# this can be done using the predict() method
#
# np.linspace(start, stop, num)
# note the parameter stop is inclusive.
# return evenly spaced numbers over a specified interval.
xfit = np.linspace(-1, 11, 50)
Xfit = xfit[:, np.newaxis]

yfit = model.predict(Xfit)

plt.scatter(x, y)
plt.scatter(xfit, yfit)

plt.scatter(x, y)
plt.plot(xfit, yfit)


# Supervised learning example: Discriminant Analysis
# Classification Problem
#
iris = pd.read_csv("iris.csv")
X_iris = iris.drop("species", axis=1)
X_iris.head()

y_iris = iris["species"]

# Split the data into a training set and test set.
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, test_size = 0.5, random_state = 1)

Xtrain.head()
Xtrain.shape

ytrain[0:5]

Xtest.head()
Xtest.shape

ytest[0:5]

# Import linear discriminant analysis class
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis()
# Fit the model to your training data.
model.fit(Xtrain, ytrain)
# Predict labels for test data
ypred = model.predict(Xtest)

# Find the fraction of predicted labels that match their true value
from sklearn.metrics import accuracy_score
accuracy_score(ypred, ytest)

# Test error rate
1 - accuracy_score(ypred, ytest)

# Obtain the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(ypred, ytest)


# Import quadratic discriminant analysis class
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
model = QuadraticDiscriminantAnalysis()
# Fit the model to your training data.
model.fit(Xtrain, ytrain)
# Predict labels for test data
ypred = model.predict(Xtest)

# Find the fraction of predicted labels that match their true value
from sklearn.metrics import accuracy_score
accuracy_score(ypred, ytest)

# Test error rate
1 - accuracy_score(ypred, ytest)

# Obtain the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(ypred, ytest)


# Unsupervised learning example: PCA
# Dimensionality Reduction
#
from sklearn.decomposition import PCA
model = PCA(n_components = 2)
model.fit(X_iris)
X_twoPC = model.transform(X_iris)
X_twoPC

iris["PC1"] = X_twoPC[:, 0]
iris["PC2"] = X_twoPC[:, 1]
sns.lmplot(x = "PC1", y = "PC2", hue = "species", data = iris, fit_reg = False)


# Unsupervised learning example: K-means clustering
# Clustering Problem
#
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 3)
model.fit(X_iris)
ypred = model.predict(X_iris)

iris["cluster"] = ypred
sns.lmplot(x = "PC1", y = "PC2", hue = "species", col = "cluster", data = iris, fit_reg = False)


