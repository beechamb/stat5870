import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Week 6 Homework
diabetes = pd.read_csv("diabetes.csv")
diabetes.head()

#Part 1
#splitting dataset
y_diabetes = diabetes["Outcome"]
x_diabetes = diabetes.drop("Outcome", axis = 1)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x_diabetes, y_diabetes,
                                                test_size = 0.5, random_state = 1)

#Part 2
#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
modelLDA = LinearDiscriminantAnalysis()

modelLDA.fit(xtrain, ytrain)

ypredLDA = modelLDA.predict(xtest)

#QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
modelQDA = QuadraticDiscriminantAnalysis()

modelQDA.fit(xtrain, ytrain)

ypredQDA = modelQDA.predict(xtest)


#KNN (k=5)
from sklearn.neighbors import KNeighborsClassifier
modelKNN5 = KNeighborsClassifier(n_neighbors = 5)

modelKNN5.fit(xtrain, ytrain)
ypredKNN5 = modelKNN5.predict(xtest)

#KNN (k=15)
modelKNN15 = KNeighborsClassifier(n_neighbors = 15)

modelKNN15.fit(xtrain,ytrain)
ypredKNN15 = modelKNN15.predict(xtest)

#Test Error Rates
from sklearn.metrics import accuracy_score

#test error LDA
TELDA = 1 - accuracy_score(ypredLDA, ytest)
TELDA
#test error QDA
TEQDA = 1 - accuracy_score(ypredQDA, ytest)
TEQDA
#test error KNN (k=5)
TEKNN5 = 1 - accuracy_score(ypredKNN5, ytest)
TEKNN5
#test error KNN (k=15)
TEKNN15 = 1 - accuracy_score(ypredKNN15, ytest)
TEKNN15

errors = [TELDA, TEQDA, TEKNN5, TEKNN15]
names = ["LDA", "QDA", "KNN5", "KNN15"]

plt.scatter(names, errors)
plt.title("Models and their Test Error Rates")
plt.xlabel("Model")
plt.ylabel("Test Error")

#LDA has the lowest test error by 0.04, and therefor would be considered the "best"
#model. Another interesting point to consider is that when we increased the number of
#neighbors in the KNN model, the test error went down from 0.315 (k = 5) to 0.284 (k = 15),
#so in this case, the more complicated model with a higher variance was a better model.