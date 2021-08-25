## Import NumPy Package ##
import numpy as np

## Import Pandas Package ##
import pandas as pd

## Combining Datasets: Merge and Join ##
## Categories of Joins
# One-to-one joins
df1 = pd.DataFrame({"employee": ["Bob", "Jake", "Lisa", "Sue"],
                    "group": ["Accounting", "Engineering", "Engineering", "HR"]})
df2 = pd.DataFrame({"employee": ["Lisa", "Bob", "Jake", "Sue"],
                    "hire_date": [2004, 2008, 2012, 2014]})
df1
df2
pd.concat([df1, df2])

df3 = pd.merge(df1, df2)
df3

# Many-to-one joins
df4 = pd.DataFrame({"group": ["Accounting", "Engineering", "HR"],
                    "supervisor": ["Carly", "Guido", "Steve"]})
df4
df3
pd.merge(df3, df4)

# Many-to-many joins
df5 = pd.DataFrame({"group": ["Accounting", "Accounting", "Engineering", "Engineering", "HR", "HR"],
                    "skills": ["math", "spreadsheets", "coding", "linux", "spreadsheets", "organization"]})
df1
df5
pd.merge(df1, df5)


## Specification of the Merge Key
# The 'on' keyword
df1
df2
pd.merge(df1, df2, on = "employee")

# The left_on and right_on keywords
df3 = pd.DataFrame({"name": ["Bob", "Jake", "Lisa", "Sue"],
                    "salary": [70000, 80000, 120000, 90000]})
df1
df3
pd.merge(df1, df3) # Error!

#useful for two different col names in two different datasets
#keeps both columns
new_df3 = pd.merge(df1, df3, left_on = "employee", right_on = "name")
new_df3

#drop column from dataframe afterwards (col axis = 1)
new_df3.drop("name", axis = 1)


## Specifying Set Arithmetic for Joins
df6 = pd.DataFrame({"name": ["Peter", "Paul", "Mary"],
                    "food": ["fish", "beans", "bread"]})
df7 = pd.DataFrame({"name": ["Mary", "Joeseph"],
                    "drink": ["wine", "beer"]})
df6
df7
pd.merge(df6, df7) # default setting is how = "inner"
pd.merge(df6, df7, how = "inner") # use intersection of keys from both data frames
#name appears in either dataset, name is kept with outer
pd.merge(df6, df7, how = "outer") # use union of keys from both data frames
pd.merge(df6, df7, how = "left") # use only keys from left data frame
pd.merge(df6, df7, how = "right") # use only keys from right data frame


## Overlapping Column Names
df8 = pd.DataFrame({"name": ["Bob", "Jake", "Lisa", "Sue"],
                    "rank": [1, 2, 3, 4]})
df9 = pd.DataFrame({"name": ["Bob", "Jake", "Lisa", "Sue"],
                    "rank": [3, 1, 4, 2]})
df8
df9
pd.merge(df8, df9, on = "name")
pd.merge(df8, df9, on = "name", suffixes = ["_L", "_R"])



### Example: US States Data ###
## To set working directory
import os
os.getcwd()
os.chdir('/Users/kevinlee/Dropbox/WMU/TEACHING/STAT 5870 Big Data Analysis Using Python (Summer 2020)/Week 4/Lecture')

# Load .csv file
pop = pd.read_csv("state-population.csv")
areas = pd.read_csv("state-areas.csv")
abbrevs = pd.read_csv("state-abbrevs.csv")

# Load .txt file
# pd.read_fwf("filename.txt")
# or
# pd.read_csv("filename.txt", sep=" ")

# Load .xlsx file
# pd.read_excel("filename.xlsx")

pop.head() # checks first five observations 
pop.head(n=10) # to check first ten observations

areas.head()
areas.tail() # checks last five observations

abbrevs.head()

## Given this information, 
## we want to rank US states and territories by their 2010 population density.

# start with a many-to-one merge that will give us the full state name within the population DataFrame.
#specify common columns we are using
#outer keeps union of two dataframes, so we do not lose any information
merged = pd.merge(pop, abbrevs, how = "outer", left_on = "state/region", right_on = "abbreviation")
merged.head()

merged = merged.drop("abbreviation", axis = 1)
merged.head()

# check whether there were any mismatches here
merged.isnull().any()

merged[merged["population"].isnull()]

merged[merged["state"].isnull()]
merged.loc[merged["state"].isnull(), "state/region"].unique()

#combine state abbrev and state
merged.loc[merged["state/region"] == "PR", "state"] = "Puerto Rico"
merged.loc[merged["state/region"] == "USA", "state"] = "United States"
merged.isnull().any()

# merge the result with the area data
final = pd.merge(merged, areas, on = "state", how = "left")
final.head()

# check for nulls to see if there were any mismatches
final.isnull().any()

#missing value from part where there wasn't whole united states areas
#these values can be dropped
final[final["area (sq. mi)"].isnull()]
final.loc[final["area (sq. mi)"].isnull(), "state/region"].unique()

cleaned_data = final.dropna() #remove rows if there is at least one missing val
cleaned_data.head()

cleaned_data.isnull().any()

#copy year 2010 because that is the year we are looking for
data2010 = cleaned_data.loc[(cleaned_data["year"] == 2010) & (cleaned_data["ages"] == "total")].copy()
data2010.head()

# now let's calculate density
data2010["density"] = data2010["population"]/data2010["area (sq. mi)"]
data2010.head()

density2010 = data2010.loc[:,["state", "density"]]
density2010.head()

density2010_2 = data2010[["state", "density"]]
density2010_2.head()

# sort the DataFrame based on density
density2010.sort_values("density", ascending = False)


## Aggregation ##
## Simple Aggregation in Pandas ##
data2010.mean()

data2010["density"].mean()
data2010.density.mean()
round(data2010.density.mean(), 2) # we round it to 2 places.

# Generates descriptive statistics that summarize the variable, exlcuding NaN.
data2010.describe()


## To show all columns of data frame in console window
data2010.head()

pd.options.display.max_columns = None
data2010.head()

# to go back to the default setting
pd.options.display.max_columns = 6
data2010.head()
