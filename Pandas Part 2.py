## Import NumPy Package ##
import numpy as np

## Import Pandas Package ##
import pandas as pd

## Operating on Data in Pandas ##
np.random.seed(seed=0)
# define a Series
ser = pd.Series(np.random.randint(1, 10, 4))
ser

# define a DataFrame
df = pd.DataFrame(np.random.randint(1, 10, (3,4)), columns = ["X1","X2","X3","X4"])
df

# We can use NumPy operation functions
np.exp(ser)

np.log(df)
np.mean(df)
np.mean(df, axis = 0)
np.mean(df, axis = 1)
np.min(df)

# Index alignment for Series
area = pd.Series({"Alaska": 1723337,
                  "Texas": 695662,
                  "California": 423967})

population = pd.Series({"California": 38332521,
                        "Texas": 26448193,
                        "New York": 19651127})

# The resutling array contains the union of indices of the two input arrays.
# Any item for which one or the other does not have an entry is marked with NaN, or "Not a Number".
population / area

top3states = pd.DataFrame({"population": population, "area": area})
top3states

# Any missing values are filled in with NaN by default.
A = pd.Series([2, 4, 6], index = [0, 1, 2])
A
B = pd.Series([1, 3, 5], index = [1, 2, 3])
B

A + B

# We can modify the fill value.
A.add(B, fill_value=0)

# Index alignment in DataFrame
np.random.seed(seed=0)
A = pd.DataFrame(np.random.randint(0, 20, (2,2)), columns=["X1","X2"])
A

B = pd.DataFrame(np.random.randint(0, 10, (3,3)), columns=["X2","X1","X3"])
B

A + B # note that indices are aligned correctly irrespective of their order in the two objects.

fill = np.mean(A.values)
A.add(B, fill_value=fill)


## Handling Missing Data ## 
# None: Pythonic missing data
val1 = np.array([1, None, 3, 4])
val1
val1 + 1 # Error!

# NaN: Missing numerical data
val2 = np.array([1, np.nan, 3, 4])
val2
val2 + 1

np.sum(val2)
np.min(val2)
np.max(val2)

np.nansum(val2)
np.nanmin(val2)
np.nanmax(val2)

# NaN and None in Pandas
# Pandas is built to handle the two of them nearly interchangeably
pd.Series([1, None, 2, np.nan]) 

# Operating on null values
# (a) Detecting null values
data = pd.Series(np.array([1, np.nan, "Hello", None]))
data.isna()
data.notna()

data[data.notna()]

# (b) Dropping null values
df = pd.DataFrame([[1, np.nan, 2],
                  [2, 3, 5],
                  [np.nan, 4, 6]], columns=["X1","X2","X3"])
df

df.dropna() # by default, dropna() will drop all rows in which any null value is present
df.dropna(axis="columns") # drops all columns containing a null value

df["X4"] = np.nan # adding another column with all NaN
df

df.dropna(axis = "columns", how = "all")
df.dropna(axis = "rows", thresh = 3)

# (c) Filling null values
data = pd.Series([1, np.nan, 2, None, 3])
data

data.fillna(0)
data.fillna(method = "ffill")
data.fillna(method = "bfill")

data = pd.Series([1, np.nan, np.nan, 2, None, 3])
data

data.fillna(0)
data.fillna(method = "ffill")
data.fillna(method = "bfill")

data = pd.Series([np.nan, np.nan, np.nan, 2, None, 3])
data

data.fillna(method = "ffill")

df = pd.DataFrame([[1, np.nan, 2],
                  [2, 3, 5],
                  [np.nan, 4, 6]], columns = ["X1", "X2", "X3"])
df

df.fillna(0)
df.fillna(method = "ffill")
df.fillna(pd.Series([1,2], index = ["X1", "X2"]))


## Combining Datasets: Concat and Append ##
# pd.concat() can be used for simple cocatenations
# simple cocatenation
ser1 = pd.Series(["A", "B", "C"], index = [1, 2, 3])
ser2 = pd.Series(["D", "E", "F"], index = [4, 5, 6])
ser1
ser2

pd.concat([ser1, ser2])

data1 = np.array([["A1", "B1"],["A2", "B2"]])
df1 = pd.DataFrame(data1, index = [1, 2], columns = ["A", "B"])
data2 = np.array([["A3", "B3"],["A4", "B4"]])
df2 = pd.DataFrame(data2, index = [3, 4], columns = ["A", "B"])
df1
df2

pd.concat([df1, df2])

data3 = np.array([["A0", "B0"],["A1", "B1"]])
df3 = pd.DataFrame(data3, index = [1, 2], columns = ["A", "B"])
data4 = np.array([["C0", "D0"],["C1", "D1"]])
df4 = pd.DataFrame(data4, index = [1, 2], columns = ["C", "D"])
df3
df4

pd.concat([df3, df4], axis = 1)

# Duplicate indicies
data5 = np.array([["A0", "B0"],["A1", "B1"]])
df5 = pd.DataFrame(data5, index = [1, 2], columns = ["A", "B"])
data6 = np.array([["A2", "B2"],["A3", "B3"]])
df6 = pd.DataFrame(data6, index = [1, 2], columns = ["A", "B"])
df5
df6

pd.concat([df5, df6])

# Catching the repeats as an error
pd.concat([df5, df6], verify_integrity = True)

# ignoring the index
pd.concat([df5, df6], ignore_index = True)

# Concatenation with joins
data7 = np.array([["A1", "B1", "C1"],["A2", "B2", "C2"]])
df7 = pd.DataFrame(data7, index = [1, 2], columns = ["A", "B", "C"])
data8 = np.array([["B3", "C3", "D3"],["B4", "C4", "D4"]])
df8 = pd.DataFrame(data8, index = [3, 4], columns = ["B", "C", "D"])
df7
df8

# By default, the join is a union of the input columns
pd.concat([df7, df8])

# We can change this to an intersection of the columns
pd.concat([df7, df8], join = "inner")


