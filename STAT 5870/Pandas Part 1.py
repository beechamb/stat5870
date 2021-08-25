## Import NumPy Package ##
import numpy as np
?np
?np.random.randint

## Import Pandas Package ##
import pandas as pd
?pd

## Pandas Series Object ##
# A Series is a one-dimensional array of indexed data.
# Create a Series object from a list
data = pd.Series([0.25, 0.5, 0.75, 1])
data

# Create a Series object from 1D array
data = pd.Series(np.array([0.25, 0.5, 0.75, 1]))
data

data.values
data.index

data = pd.Series([0.25, 0.5, 0.75, 1], index=["a","b","c","d"])
data["b"]

data = pd.Series([0.25, 0.5, 0.75, 1], index=[2,5,3,7])
data[5]

data = pd.Series(5, index=[10,20,30])
data

# Create a Series object from a dictionary
population_dict = {"California": 38332521,
                   "Texas": 26448193,
                   "New York": 19651127,
                   "Florida": 19552860,
                   "Illinois": 12882135}
population_dict

# By default a series will be created where the index is drawn from the keys.
population = pd.Series(population_dict) 
population
population["New York"]
population["California":"New York"]


## Pandas DataFrame Object ##
# A DataFrame is a two-dimensional array with both flexible row indices and flexible column names.
# Create a DataFrame object from 2D array
np.random.seed(seed=0)
data = np.random.randint(0, 100, (5,3))
data

df = pd.DataFrame(data, columns=["X1","X2","X3"])
df

df = pd.DataFrame(data, index=["a","b","c","d","e"], columns=["X1","X2","X3"])
df

df.values #only take out vals, in the 2d array
df.index #only gives index values for each row
df.columns #only gives column names

df["X1"]
df["a"] # Error!

# Create a DataFrame object from sigle Series object
data = pd.DataFrame(population, columns=["population"])
data

data.values
data.index
data.columns

# Create a DataFrame object from a dictionary of Series object
area_dict = {"California": 423967, 
             "Texas": 695662, 
             "New York": 141297, 
             "Florida": 170312, 
             "Illinois": 149995}
area_dict 

area = pd.Series(area_dict)

states = pd.DataFrame({"population": population, "area": area})
states

states.values
states.index
states.columns

states["population"]
states["California"] # Error!


## Data Indexing and Selection ##
# Data Selection in Series #
data = pd.Series([0.25, 0.5, 0.75, 1], index=["a","b","c","d"])
data

data["b"]
data.b

# we can extend a Series by assigning to a new index value
data["e"] = 1.25
data

# slicing by explicit index
data["a":"c"] # the final index is included

# slicing by implicit index
data[0:3] # the final index is excluded

# masking
data[(data > 0.3) & (data < 0.8)]

# fancy indexing
data[["a", "e"]]

# Indexers: loc and iloc
# These slicing and indexing conventions can be a source of confusion. For example,
data = pd.Series([0.25, 0.50, 0.75], index=[1,3,5])
data[1] # explicit index when indexing
data[1:3] # implicit index when slicing

# the loc attribute allows indexing and slicing that always references the explicit index
data.loc[1]
data.loc[1:3]

# the iloc attribute allows indexing and slicing that always references the implicit Python-style index
data.iloc[1]
data.iloc[1:3]


# Data Selection in DataFrame #
data = pd.DataFrame({"population": population, "area": area})
data

data["population"]
data.population

# we can add a new column to the DataFrame by assigning to a new column name
data["density"] = data["population"]/data["area"]

# something to be careful
data[0] # Error! It won't take out first row like in 2D array

data.values
data.values[0] # passing a single index to an array accesses a row

data["area"] # passing a single "index" to a DataFrame accesses a column
data["Texas"] # Error!

# Indexers: loc and iloc
# loc: explicit indexing
data.loc["New York"]
data.loc[:"New York"]
data.loc[:,:"area"]
data.loc[:"New York", :"area"] 

# iloc: implicit indexing
data.iloc[2]
data.iloc[:3] # first three rows
data.iloc[:,:2] # first two columns
data.iloc[:3, :2] # first three rows and first two columns

# other usages
data.loc[data["density"] > 100]
data.loc[data.density > 100]
data.loc[:,["population", "density"]]
data.loc[data.density > 100, ["population", "density"]]

data.loc["New York", "density"] = 139

# Additional indexing conventions
# indexing refers to columns
data["density"]
data["California"] # Error!

# slicing refers to rows
data["California":"New York"]
data["population":"area"] # Error!
data[1:3]

# direct masking operations are also interpreted row-wise
data[data.density > 100]


