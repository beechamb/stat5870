## Import NumPy Package ##
import numpy as np

## NumPy Array Slicing ##
# subarrays as "views" rather than "copies" of the array data
x = np.array([[1,2,3],[4,5,6],[7,8,9]])
x_sub = x[:2, :2]
x_sub

x_sub[0,0] = 99
x_sub
x

# creating copies of arrays
x_sub_copy = x[:2, :2].copy()
x_sub_copy = np.copy(x[:2,:2])
x_sub_copy

x_sub_copy[0,0] = 33
x_sub_copy
x


## NumPy Array Fancy Indexing ##
# 1D array
np.random.seed(seed=1)
x = np.random.randint(0, 100, 10)

ind = np.array([3, 5, 7, 9])
x[ind]

ind = np.array([[3,5],[7,9]])
x[ind]

# 2D array
x = np.arange(12)
x = x.reshape((3,4))

row = np.array([0,1,2])
col = np.array([2,1,3])

x[row, col]


## Computation on NumPy Arrays ##
# +, np.add
# -, np.subtract
# *, np.multiply
# /, np.divide
# //, np.floor_divide
# **, np.power
# %, np.mod

# 1D array computation
y = [1, 2, 3, 4, 5]
y + 1 # Error!
y * 2

x = np.array([1, 2, 3, 4, 5])
x + 1
x * 2

x = np.array([1, 2, 3])
x ** 3
np.power(x, 3) # x^3
np.power(3, x) # 3^x

x = np.array([-3, -2, -1, 0, 1, 2, 3])
np.absolute(x)
np.abs(x)

x = np.array([1, 2, 3])
np.exp(x)
np.log(x) # natural log (ln)
np.log10(x) # log with base 10

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 3, 5, 7, 9])
x + y
np.add(x, y)
x * y
np.multiply(x, y)

y = np.array([1, 3, 5])
x + y # Error!

# 2D array (Matrix) multiplication
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])
np.matmul(x, y)
np.dot(x, y)

x = np.array([[1, 3, 5]])
y = np.array([2, 4, 6])
y = y[:,np.newaxis]
np.matmul(x, y)
np.matmul(y, x)

# 2D array (Matrix) transpose
z = np.array([[1, 2, 3], [4, 5, 6]])
z.transpose()
np.transpose(z)


## Aggregations on NumPy Arrays ##
# 1D array
x = np.random.rand(10)

x.sum()
x.mean()
x.var()
x.std()
x.min()
x.max()
x.argmin()
x.argmax()

# 2D array
m = np.random.randint(0,10,(3,4))

m.sum()
# axis = 0 means that the first axis will be collapsed
# for 2D arrays, this means that values within each column will be aggregated.
m.sum(axis=0)
m.min(axis=0)

# axis = 1 means that the second axis will be collapsed
# for 2D arrays, this means that values within each row will be aggregated.
m.sum(axis=1)
m.min(axis=1)


## Comparisons on NumPy Arrays ##
x = np.array([1, 2, 3, 4, 5])

x < 3
x == 3
(x < 3) | (x == 3)
(x < 3) & (x == 3)
~(x < 3) # x >= 3

x[x < 3]
np.where(x > 2)
np.where(x > 2, 1, 0)
np.where(x > 2, x, 10*x)

# counting entries
x = np.array([1, 2, 3, 4, 5])
np.sum(x < 4)

np.any(x > 4)
np.any(x < 0)

np.all(x < 6)
np.all(x < 4)


## Sorting NumPy Arrays ##
# 1D array
x = np.random.choice(np.arange(10), 5, replace=True)
np.sort(x)

