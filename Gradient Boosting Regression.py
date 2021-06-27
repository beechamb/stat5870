import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)

X.head()

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.25, random_state = 1)

from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(
        max_depth = 2,
        n_estimators = 10,
        learning_rate = 1,
        loss = "ls",
        random_state = 1
)

model.fit(Xtrain, ytrain)

ypred = model.predict(Xtest)

ytest[0:6]
ypred[0:6]

from sklearn.metrics import mean_squared_error
errors = mean_squared_error(ytest, ypred)
errors

