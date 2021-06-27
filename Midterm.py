import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Problem 1
#coin tossing simulation
#a
S_CoinTossing = np.array(["H","T"])

#b
np.random.seed(seed=0)
CoinTossing10 = np.random.choice(S_CoinTossing, 10, replace = True)
CoinTossing10

#proportion of H
(np.sum(CoinTossing10 == "H"))/(len(CoinTossing10))
#proportion = 0.2

#c
np.random.seed(seed=0)
CoinTossing100 = np.random.choice(S_CoinTossing, 100, replace = True)
CoinTossing100

#proportion of H
(np.sum(CoinTossing100 == "H"))/(len(CoinTossing100))
#proportion is 0.44

#d
np.random.seed(seed=0)
CoinTossing10000 = np.random.choice(S_CoinTossing, 10000, replace = True)
CoinTossing10000

#proportion of H
(np.sum(CoinTossing10000 == "H"))/(len(CoinTossing10000))
#proportion is 0.4915

#e (see attached word document)

#dice rolling simulation
#f
def roll(n):
    l = np.array([1,2,3,4,5,6])
    return(np.random.choice(l,n,replace = True))

#g
np.random.seed(seed=0)
arr1 = roll(10)
arr1
#barplot
sns.countplot(x = arr1)

#h
np.random.seed(seed=0)
arr2 = roll(100)
arr2
#barplot
sns.countplot(x = arr2)

#i
np.random.seed(seed=0)
arr3 = roll(10000)
arr3
#barplot
sns.countplot(x = arr3)

#j (see attached word document)

#Problem 2
#a
iris = pd.read_csv("iris.csv")

#b
from scipy.stats import pearsonr
pearsonr(iris.sepal_width, iris.sepal_length)[0]
#There appears to be a weak negative linear correlation ofabout  -0.1094 
#between sepal_width and sepal_length.
#(also see attached word document)

#c
sns.regplot(x = "sepal_width", y = "sepal_length", data = iris)

#d
sns.lmplot(x = "sepal_width", y = "sepal_length", data = iris, hue = "species", legend = False)

#e
subset1 = iris[iris["species"] == "setosa"]
subset2 = iris[iris["species"] == "versicolor"]
subset3 = iris[iris["species"] == "virginica"]

cor1 = pearsonr(subset1.sepal_width, subset1.sepal_length)[0]
cor2 = pearsonr(subset2.sepal_width, subset2.sepal_length)[0]
cor3 = pearsonr(subset3.sepal_width, subset3.sepal_length)[0]

cor1
cor2
cor3

#f (see attached word document)

#Problem 3
#a
unemp = pd.read_csv("unemp.csv")

#b
unemp.shape[0] #885548 observations in the dataset

#c
unemp.head(n=10)

#d
unemp.isnull().any() #there are no missing values

#e
grouped = unemp.groupby("Year")["Rate"].mean()
#mean unemployment rates every year
grouped

#f
df = pd.DataFrame(grouped, columns = ["Rate"])
df
plt.plot(df.index, df.Rate)
plt.xlabel("Year")
plt.ylabel("Mean Unemployment Rate")

#g
unemp_MI_Jan_2016 = unemp.loc[(unemp["State"] == "Michigan") & (unemp["Year"] == 2016) & (unemp["Month"] == "January")]
unemp_MI_Jan_2016

#h
sorted_unempMI_Jan2016 = unemp_MI_Jan_2016.sort_values("Rate", ascending = True)
sorted_unempMI_Jan2016.head()

#i
unemp_Kzoo_MI_2010_2015 = unemp.loc[(unemp["State"] == "Michigan") & (unemp["County"] == "Kalamazoo County") & (unemp["Year"] >= 2010) & (unemp["Year"] <= 2015)]
unemp_Kzoo_MI_2010_2015

#j
sns.boxplot(x = "Year", y = "Rate", data = unemp_Kzoo_MI_2010_2015, palette = "Set2")
