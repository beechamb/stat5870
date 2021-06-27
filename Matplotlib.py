## Import NumPy Package ##
import numpy as np

## Import Pandas Package ##
import pandas as pd

## Import Matplotlib Package ##
import matplotlib.pyplot as plt

## Simple Line Plots ##
x = np.linspace(0, 10, 1000)
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

y = 2*x
plt.plot(x, y)

# Adjusting the Plot: Line Colors
plt.plot(x, np.sin(x-0), color = "lightblue") # specify color by name
plt.plot(x, np.sin(x-0), color = "orange") 

# Adjusting the Plot: Line Styles
plt.plot(x, x+0, linestyle = "solid")
plt.plot(x, x+1, linestyle = "dashed")
plt.plot(x, x+2, linestyle = "dashdot")
plt.plot(x, x+3, linestyle = "dotted")

# Adjusting the Plot: Axes Limits
plt.plot(x, np.sin(x))

plt.plot(x, np.sin(x))
plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5)

plt.plot(x, np.sin(x))
plt.axis([-3, 13, -2, 2])

plt.plot(x, np.sin(x))
plt.axis("tight")

plt.plot(x, np.sin(x))
plt.axis("equal")

# Adjusting the Plot: Labeling Plots
plt.plot(x, np.sin(x))
plt.title("A Sine Curve")
plt.xlabel("x")
plt.ylabel("sin(x)")

# Adjusting the Plot: Legend
plt.plot(x, np.sin(x), color = "lightblue", label = "sin(x)")
plt.plot(x, np.cos(x), color = "orange", label = "cos(x)")
plt.axis("equal")
plt.legend()


## Simple Scatter Plots ##
x = np.linspace(0, 10, 30)
y = np.sin(x)

# Adjusting the Plot: Marker style
plt.plot(x, y, "o", color = "black")
plt.plot(x, y, ".", color = "black")
plt.plot(x, y, "x", color = "black")
plt.plot(x, y, "+", color = "black")
plt.plot(x, y, "s", color = "black")
plt.plot(x, y, "d", color = "black")


# scatter plots with plt.scatter
plt.scatter(x, y, marker = "o")

np.random.seed(seed=0)
x = np.random.normal(size = 100)
y = np.random.normal(size = 100)

plt.scatter(x, y, c = "orange", s = 50, alpha = 0.3)

plt.axvline(x=0, color = "lightblue", linestyle = "dashed") # to add vertical line
plt.axhline(y=1, color = "lightgreen") # to add horizontal line


## Histograms ##
np.random.seed(seed=0)
data = np.random.normal(size = 1000)
plt.hist(data)

x1 = np.random.normal(0, 0.8, 1000)
x2 = np.random.normal(-2, 1, 1000)
x3 = np.random.normal(3, 2, 1000)

plt.hist(x1, bins = 40, alpha = 0.3, color = "blue")
plt.hist(x2, bins = 40, alpha = 0.3, color = "red")
plt.hist(x3, bins = 40, alpha = 0.3, color = "green")


## Multiple Subplots ## 
x = np.linspace(0, 10, 1000)

plt.subplot(2, 3, 1)
plt.plot(x, np.sin(x-0), color = "blue")
plt.subplot(2, 3, 2)
plt.plot(x, np.sin(x-1), color = "green")
plt.subplot(2, 3, 3)
plt.plot(x, np.sin(x-2), color = "red") 
plt.subplot(2, 3, 4)
plt.plot(x, np.sin(x-3), color = "pink") 
plt.subplot(2, 3, 5)
plt.plot(x, np.sin(x-4), color = "orange")
plt.subplot(2, 3, 6)
plt.plot(x, np.sin(x-5), color = "skyblue") 

# adujst spacing between plots
plt.subplots_adjust(hspace = 0.2, wspace = 0.4)
plt.subplot(2, 3, 1)
plt.plot(x, np.sin(x-0), color = "blue")
plt.subplot(2, 3, 2)
plt.plot(x, np.sin(x-1), color = "green")
plt.subplot(2, 3, 3)
plt.plot(x, np.sin(x-2), color = "red") 
plt.subplot(2, 3, 4)
plt.plot(x, np.sin(x-3), color = "pink") 
plt.subplot(2, 3, 5)
plt.plot(x, np.sin(x-4), color = "orange")
plt.subplot(2, 3, 6)
plt.plot(x, np.sin(x-5), color = "skyblue") 


## Advanced subplots ##
df=pd.DataFrame({"x": range(1,101), 
                 "y": np.random.randn(100)*15+range(1,101), 
                 "z": (np.random.randn(100)*15+range(1,101))*2})

# Example 1
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

# Example 2
# 2 rows and 4 columns
# The first plot is on row 1, and is spread all along the 4 columns
plt.subplot2grid((2, 4), (0, 0), colspan=4)
plt.scatter("x", "y", data=df, marker="o", alpha=0.4)
# The second plot is on row 2, spread on 3 columns
plt.subplot2grid((2, 4), (1, 0), colspan=3)
plt.scatter("x", "z", data=df, marker="o", color="grey", alpha=0.3)
# The last plot is on row 1, spread on 1 column
plt.subplot2grid((2, 4), (1, 3), colspan=1)
plt.scatter("x", "z", data=df, marker="o", color="orange", alpha=0.3)
 
#Example 3
# 2 rows and 2 columns
plt.subplot2grid((2, 2), (0, 0), colspan=1)
plt.scatter("x", "y", data=df, marker="o", alpha=0.4)
plt.subplot2grid((2, 2), (1, 0), colspan=1)
plt.scatter("x", "z", data=df, marker="o", color="grey", alpha=0.3)
plt.subplot2grid((2, 2), (0, 1), rowspan=2)
plt.scatter("x", "z", data=df, marker="o", color="orange", alpha=0.3)


## Stylesheets ##
x = np.linspace(0, 10, 1000)
np.random.seed(0)
data = np.random.normal(size=1000)

plt.subplot(1, 2, 1)
plt.hist(data)
plt.subplot(1, 2, 2)
plt.plot(x, np.sin(x))

plt.style.available

with plt.style.context("seaborn-whitegrid"):
    plt.subplot(1, 2, 1)
    plt.hist(data)
    plt.subplot(1, 2, 2)
    plt.plot(x, np.sin(x))

with plt.style.context("seaborn-dark"):
    plt.subplot(1, 2, 1)
    plt.hist(data)
    plt.subplot(1, 2, 2)
    plt.plot(x, np.sin(x))

with plt.style.context("dark_background"):
    plt.subplot(1, 2, 1)
    plt.hist(data)
    plt.subplot(1, 2, 2)
    plt.plot(x, np.sin(x))

with plt.style.context("ggplot"):
    plt.subplot(1, 2, 1)
    plt.hist(data)
    plt.subplot(1, 2, 2)
    plt.plot(x, np.sin(x))
    

