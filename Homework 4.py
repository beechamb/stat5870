#Homework 4
#Importing packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#problem 1
#part 1
#generate 100 random samples, normal distribution, mean 2, standard deviation 3
np.random.seed(seed=0)
arr1 = np.random.normal(2,3,100)
arr1

#part 2
#100 random samples, normal distribution, mean 0, standard deviation 1
np.random.seed(seed=0)
arr2 = np.random.normal(0,1,100)
arr2

#part 3
#create data frame using the two arrays from 1 and 2, set cols as x1 and x2
df1 = pd.DataFrame({"x1":arr1, "x2":arr2})
df1

#part 4
#Draw a plot that contains both histograms for two variables in the data frame in 3
plt.subplot(2, 2, 1)
plt.hist(df1["x1"], bins = 40, alpha = 0.3, color = "blue")
plt.subplot(2, 2, 2)
plt.hist(df1["x2"], bins = 40, alpha = 0.3, color = "green")

#part 5
#Obtain a 5 times 5 two-dimensional array of random integers from 0 to 9.
np.random.seed(seed=0)
arr3 = np.random.randint(0, 10, (5,5))
arr3

#part 6
#Find mean values for each row from the two-dimensional array in 5.
arr3.mean(axis = 1)

#problem 2
#load dataset
titanic = sns.load_dataset("titanic")

#part 1
#How many observations are in the titanic dataset?
len(titanic.index)
#there are 891 observations

#part 2
#Find all variable names.
titanic.columns

#part 3
#Check whether there are any missing values in the dataset.
titanic.isna()

#part 4
#Remove observations with missing values from the dataset.
cleantitanic = titanic.dropna()
cleantitanic

#part 5
#How many observations left in the dataset?
len(cleantitanic.index)
#182 observations are left

#part 6
#Draw the boxplot of age variable.
sns.boxplot(y = "age", data = cleantitanic)

#part 7
#Draw the scatterplot between age (x-axis) and fare (y-axis).
sns.regplot(x = "age", y = "fare", data = cleantitanic, fit_reg = False)

#part 8
#Update the scatter plot in 7. such that the new scatter plot satises followings:
#If the observation is female, then the observation will be plotted in orange color,
#otherwise it will be plotted in lightblue color.
my_pal = {"female": "orange", "male": "lightblue"}
sns.lmplot(x = "age", y = "fare", data = cleantitanic, hue = "sex", fit_reg = False, palette = my_pal)

#part 9
#Draw the side-by-side boxplot to visualize the fare dierence between male and female.
sns.boxplot(x = "sex", y = "fare", data = cleantitanic, palette = "Set2")

#part 10
#Find the mean age in each class.
grouped = cleantitanic.groupby("class")
grouped["age"].mean()

#part 11
#Find a subset with person's age less than or equal to 20.
cleantitanic[cleantitanic.age <= 20]

#part 12
#Find a subset with female person whose age is greater than 50.
cleantitanic[(cleantitanic.sex == "female") & (cleantitanic.age > 50)]

#part 13
#Find a subset containing only sex, age, and fare variables.
cleantitanic[["sex", "age", "fare"]]
