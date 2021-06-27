#Week 7 Homework (Homework 6)
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#Problem 1
wine = pd.read_csv("winequality_red.csv")

#Part 1
#Splitting the dataset
y_wine = wine["quality"]
x_wine = wine.drop("quality", axis = 1)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x_wine, y_wine,
                                                test_size = 0.5, random_state = 1)
#Part 2
#Model 1
from sklearn.ensemble import GradientBoostingRegressor
model1 = GradientBoostingRegressor(
        max_depth = 2,
        n_estimators = 10,
        learning_rate = 1,
        loss = "ls",
        random_state = 1
)

model1.fit(xtrain, ytrain)

ypred1 = model1.predict(xtest)


from sklearn.metrics import mean_squared_error
errors1 = mean_squared_error(ytest, ypred1)
errors1

#Model 2
model2 = GradientBoostingRegressor(
        max_depth = 2,
        n_estimators = 15,
        learning_rate = 1,
        loss = "ls",
        random_state = 1
)

model2.fit(xtrain, ytrain)

ypred2 = model2.predict(xtest)

errors2 = mean_squared_error(ytest, ypred2)
errors2

#Model 3
model3 = GradientBoostingRegressor(
        max_depth = 2,
        n_estimators = 10,
        learning_rate = 6,
        loss = "ls",
        random_state = 1
)

model3.fit(xtrain, ytrain)

ypred3 = model3.predict(xtest)

errors3 = mean_squared_error(ytest, ypred3)
errors3


#Model 4
model4 = GradientBoostingRegressor(
        max_depth = 2,
        n_estimators = 5,
        learning_rate = 1,
        loss = "ls",
        random_state = 1
)

model4.fit(xtrain, ytrain)

ypred4 = model4.predict(xtest)

errors4 = mean_squared_error(ytest, ypred4)
errors4

#Model 5
model5 = GradientBoostingRegressor(
        max_depth = 2,
        n_estimators = 5,
        learning_rate = 0.5,
        loss = "ls",
        random_state = 1
)

model5.fit(xtrain, ytrain)

ypred5 = model5.predict(xtest)

errors5 = mean_squared_error(ytest, ypred5)
errors5

#It appears that as the n_estimators parameter increases, the test mean squared 
#error becomes "worse", ~0.4543 in model 1 (with a lower n_estimators: 10) as 
#opposed to ~0.4639 in model 2 (with a higher n_estimators: 15). Though, this 
#does not appear to be a very influential parameter. A higher learning rate appears 
#to make the model "worse", and it appears this parameter has a high effect on the 
#mean squared error of the model. When I increased the learning rate to 6 in 
#model 3 as opposed to 1 in model 1, the mean squared error increased by an 
#enormous amount ~17504013034859! The best performing model I have found in this 
#homework assignment appears to be model 5, which has a mean squared error of
# ~0.4287, a lower n_estimators (5) and a lower learning rate (0.5).