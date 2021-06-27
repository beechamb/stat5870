import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#################
# Python Basics #
#################
 
## Flow Control ##
# if else statement
x = 5
if x > 0:
    print("The x is positive.")
else: # for the rest of cases
    print("The x is negative.")

x = -3    
if x > 0:
    print("The x is positive.")
elif x == 0:
    print("The x is zero.")
else:
    print("The x is negative.")
   
# for loop
# if you do need to iterate over a sequence of numbers use range() function.
# range() returns a range abject, which is nothing but a sequence of integers.    

l = [2, 4, 6, 8, 10]
for i in range(3):
    print(l[i])
    
l = [1, 0, 1, 1, 0]
x = np.zeros(5)
for i in range(5):
    if l[i] == 0:
        x[i] = np.random.normal(0, 1, 1)
    else:
        x[i] = np.random.binomial(5, 0.5, 1)        
x
        
## User-Defined Functions ##
def square(x):
    return x * x

square(3)
square(4)

def square(x):
    x2 = x * x
    print("{} times {} equals {}." .format(x, x, x2))
square(3)
square(4)


#########
# NumPy #
#########

# np.zeros(shpae, dtype)
# return a new array of given shape and type, filled with all zeros.
x = np.zeros(5) 
x = np.zeros(5, dtype = int)
x = np.zeros((5, 1), dtype = int)
x = np.zeros((5, 5), dtype = int)
x
# np.arange(start, stop, step)
# return evenly spaced values within a given interval.
# note that the parameter start is inclusive, while the parameter stop is exclusive.
# the parameter step determines spacing between values.
np.arange(0, 10)
np.arange(10) # you can skip start if it is 0.
np.arange(0, 10, 2)

# np.linspace(start, stop, num)
# note the parameter stop is inclusive.
# return evenly spaced numbers over a specified interval.
np.linspace(0, 1, 5)

# np.random.normal(loc, scale, size)
# return random samples from a normal distribution with specified mean and standard deviation.
np.random.normal(0, 1, 10)
np.random.normal(0, 1, (3,3))

# np.random.binomial(n, p, size)
# np.random.uniform(low, high, size)

# np.random.randint(low, high, size)
# return random integers from "low" (inclusive) to "high" (exclusive).
np.random.randint(0, 10, 5)

# np.random.choice(a, size, replace)
# return random samples from a given 1-D array
np.random.choice(np.arange(1, 21), 10, replace = False)

# np.random.seed(seed)
# seed the generator to make the random numbers reproducible
np.random.seed(seed=0)
np.random.uniform(0, 1, 10)


## Aggregations on NumPy Arrays ##
# 1D array
x = np.random.rand(10)
x

np.sum(x) # x.sum()
np.mean(x) # x.mean()
np.var(x)
np.std(x)
np.min(x)
np.max(x)
np.argmin(x)
np.argmax(x)

## Computation on NumPy Arrays ##
np.exp(x) # can't do x.exp()
np.log(x)
np.abs(x)

x1 = np.array([1, 3, 5])
x2 = np.array([2, 4, 6])
np.add(x1, x2)
np.multiply(x1, x2)


##########
# Pandas #
##########

## Pandas DataFrame Object ##
# A DataFrame is a two-dimensional array with both flexible row indices and flexible column names.
# Create a DataFrame object from 2D array
np.random.seed(seed=0)
data = np.random.randint(0, 100, (5,3))
data

df = pd.DataFrame(data, columns=["X1","X2","X3"])
df

df.values
df.columns
df.index

# Create a DataFrame object from multiple 1D array
a = np.array([1, 3, 5])
b = np.array([2, 4, 6])

df = pd.DataFrame({"X1": a, "X2": b})
df

df.shape
df.head()
df.describe()

## Pandas DataFrame Indexing Conventions ##
# indexing refers to columns
df["X1"]

# slicing refers to rows
df[0:2] #not inclusive

# direct masking refers to rows
df[df["X1"] >= 3]
df[df.X1 >= 3]

## Please Review the Example in Pandas Part 3 ##
### Example: US States Data ###

## GroupBy: Split, Apply, Combine ##


##############
# Matplotlib #
##############

# Simple line plots #
x = np.linspace(0, 10, 1000)
plt.plot(x, np.sin(x), color = "green") # solid green

# Multiple subplots #
df=pd.DataFrame({"x": range(1,101), 
                 "y": np.random.randn(100)*15+range(1,101), 
                 "z": (np.random.randn(100)*15+range(1,101))*2})
 
plt.subplot(1, 3, 1)
plt.scatter("x", "y", data=df, marker="o", alpha=0.4)
plt.subplot(1, 3, 2)
plt.scatter("x", "z", data=df, marker="o", color="grey", alpha=0.3)
plt.subplot(1, 3, 3)
plt.scatter("x", "z", data=df, marker="o", color="orange", alpha=0.3)

# Advanced subplots #
# 2 rows and 2 columns
# The first plot is on row 1, and is spread all along the 2 columns
plt.subplot2grid((2, 2), (0, 0), colspan=2)
plt.scatter("x", "y", data=df, marker="o", alpha=0.4)
# The second plot is on row 2, spread on first 1 column
plt.subplot2grid((2, 2), (1, 0), colspan=1)
plt.scatter("x", "z", data=df, marker="o", color="grey", alpha=0.3)
# The last plot is on row 2, spread on remining 1 column
plt.subplot2grid((2, 2), (1, 1), colspan=1)
plt.scatter("x", "z", data=df, marker="o", color="orange", alpha=0.3)


###########
# Seaborn #
###########

iris = pd.read_csv("iris.csv")

# Scatter plots #
sns.regplot(x = "sepal_length", y = "sepal_width", data = iris)
sns.regplot(x = "sepal_length", y = "sepal_width", data = iris, fit_reg = False)

# Scatter plot with differnt colors for different groups #
sns.lmplot(x = "sepal_length", y = "sepal_width", hue='species', data = iris, fit_reg = False)

# Histogram #
sns.distplot(iris["sepal_length"], bins = 20)
sns.distplot(iris.sepal_length, bins = 20, kde = False)

# Boxplot and side-by-side boxplot #
sns.boxplot(y = "sepal_length", data = iris)
sns.boxplot(x = "species", y = "sepal_length", data = iris, palette = "Set2")

# Bar plot and grouped bar plot #
titanic = sns.load_dataset("titanic")
sns.countplot(x = "species", data = iris)
sns.countplot(x = "class", hue = "sex", data = titanic)

