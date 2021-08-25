import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Model Validation
iris = pd.read_csv("iris.csv")
X_iris = iris.drop("species", axis=1)
y_iris = iris["species"]

# Model validation the wrong way
#
# KNN (k-nearest neighbors) classifier with n_neighbors
# KNN with k = 5
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 5)

model.fit(X_iris, y_iris)
ypred = model.predict(X_iris)

# Train error rate
from sklearn.metrics import accuracy_score
1 - accuracy_score(ypred, y_iris)

# KNN with k = 2
model = KNeighborsClassifier(n_neighbors = 2)

model.fit(X_iris, y_iris)
ypred = model.predict(X_iris)

# Train error rate
1 - accuracy_score(ypred, y_iris)


# Model validation the right way: Holdout sets (validation set approach)
#
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, test_size = 0.5, random_state = 2)

# KNN (k-nearest neighbors) classifier with n_neighbors
# KNN with k = 5
model = KNeighborsClassifier(n_neighbors = 5)

model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

# Test error rate
1 - accuracy_score(ypred, ytest)

# KNN with k = 2
model = KNeighborsClassifier(n_neighbors = 2)

model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

# Test error rate
1 - accuracy_score(ypred, ytest)


# Model validation the right way: Cross-Validation
#
from sklearn.model_selection import cross_val_score

# KNN (k-nearest neighbors) classifier with n_neighbors
# KNN with k = 5
model = KNeighborsClassifier(n_neighbors = 5)

# 5-folds cross validation
cross_val_score(model, X_iris, y_iris, cv = 5)
# estimated test error rate
1 - np.mean(cross_val_score(model, X_iris, y_iris, cv = 5))

# 10-folds cross validation 
cross_val_score(model, X_iris, y_iris, cv = 10)
# estimated test error rate
1 - np.mean(cross_val_score(model, X_iris, y_iris, cv = 10))


# Feature Engineering
# Example: Using Titanic Data Set
titanic = sns.load_dataset("titanic")
titanic.shape
titanic.head()

X_titanic = titanic[["pclass", "sex", "age", "fare"]].copy()
X_titanic.head()

y_titanic = titanic["survived"].copy()

X_titanic.dtypes

X_titanic["pclass"] = X_titanic["pclass"].astype(str)

X_titanic.dtypes

# Categorical Features
# Using Pandas get dummies method
X_titanic = pd.get_dummies(X_titanic, prefix_sep = "_", drop_first = True)
X_titanic.head()

# Missing Data Imputation
y_titanic.isnull().any()

X_titanic.isnull().any()
X_titanic[X_titanic["age"].isnull()]

X_titanic.head(n=7)

# Simple imputation approachs
# Method 1: mean
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(strategy = "mean")

X_titanic_imp_mean = imp_mean.fit_transform(X_titanic)

X_titanic_final = pd.DataFrame(X_titanic_imp_mean, columns = X_titanic.columns)
X_titanic_final.head(n=7)

# Method 2: median
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(strategy = "median")

X_titanic_imp_mean = imp_mean.fit_transform(X_titanic)

X_titanic_final = pd.DataFrame(X_titanic_imp_mean, columns = X_titanic.columns)
X_titanic_final.head(n=7)

# Method 3: most frequent value
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(strategy = "most_frequent")

X_titanic_imp_mean = imp_mean.fit_transform(X_titanic)

X_titanic_final = pd.DataFrame(X_titanic_imp_mean, columns = X_titanic.columns)
X_titanic_final.head(n=7)

# Method 4: fixed value
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(strategy = "constant", fill_value = 0)

X_titanic_imp_mean = imp_mean.fit_transform(X_titanic)

X_titanic_final = pd.DataFrame(X_titanic_imp_mean, columns = X_titanic.columns)
X_titanic_final.head(n=7)


