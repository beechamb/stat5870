## Import NumPy Package ##
import numpy as np

## Import Pandas Package ##
import pandas as pd

## Import Matplotlib Package ##
import matplotlib.pyplot as plt

## Import Seaborn Package
import seaborn as sns

## Seaborn Package ##
# Load iris data set
# Five variables: sepal_length, sepal_width, petal_length, petal_width, species
# Three levels in species variable: setosa, versicolor, virginica
iris = pd.read_csv("iris.csv")
iris.columns

## Scatterplot and Matrix of scatterplots ##
# Basic scatterplot with seaborn
sns.regplot(x = "sepal_length", y = "sepal_width", data = iris, fit_reg = False)
# or 
plt.scatter("sepal_length", "sepal_width", data = iris)

sns.regplot(x = "sepal_length", y = "sepal_width", data = iris )

# Use categorical variable to color scatterplot
sns.lmplot(x = "sepal_length", y = "sepal_width", data = iris, hue = "species", fit_reg = False, legend = False)

sns.lmplot(x = "sepal_length", y = "sepal_width", data = iris, hue='species', fit_reg = False)

# Correlogram or Correlation matrix
sns.pairplot(iris)

sns.pairplot(iris, hue = "species", markers = ["o", "s", "d"])

sns.pairplot(iris, hue = "species", markers = ["o", "s", "d"], palette = "Set2")

sns.pairplot(iris, hue = "species", markers = ["o", "s", "d"], palette = "Blues")


## Histogram and Density Plot ##
sns.distplot(iris["sepal_length"], bins = 20)
 
sns.distplot(iris["sepal_length"], bins = 20, kde = False)

sns.distplot(iris["sepal_length"], bins = 20, hist = False )

#two histograms on the same plot
sns.distplot(iris["sepal_length"], color = "skyblue", label = "Sepal Length")
sns.distplot(iris["sepal_width"], color = "orange", label = "Sepal Width")
plt.legend()


## Boxplot ##
# one numerical variable only
sns.boxplot(y = "sepal_length", data = iris)

# one nemrical variable, and several groups (side-by-side boxplot)
sns.boxplot(x = "species", y = "sepal_length", data = iris, palette = "Set2")

my_pal = {"versicolor": "skyblue", "setosa": "orange", "virginica": "lightgreen"}
sns.boxplot(x = "species", y = "sepal_length", data = iris, palette = my_pal)

# grouped boxplot
titanic = sns.load_dataset("titanic")
titanic.head()

sns.boxplot(x = "class", y = "age", hue = "sex", data = titanic, palette="Set1")


## Bar plot ##
sns.countplot(x = "sex", data = titanic)
sns.countplot(x = "class", data = titanic)

# grouped bar plot 
sns.countplot(x = "class", hue = "sex", data = titanic)


