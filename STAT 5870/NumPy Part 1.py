## Import NumPy Package ##
import numpy as np

## Create NumPy Arrays from Python Lists ##
# one-dimensional NumPy array 
x = np.array([1, 2, 3, 4, 5])

# two-dimensional NumPy array
y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # 3 times 3 matrix (3 rows and 3 colums)
z = np.array([[1, 2, 3, 4, 5]]) # 1 times 5 matrix (1 row and 5 colums)
z

## NumPy Array Attributes ##
x.ndim # number of dimensions
x.shape # number of elements along each array dimension
x.size # number of all elements in the array

y.ndim
y.shape
y.size

z.ndim
z.shape
z.size


## Create NumPy Arrays with special functions ##
# np.zeros(shape, dtype)
# return a new array of given shape and type, filled with all zeros.
x = np.zeros(5) 
x.dtype

x = np.zeros(5, dtype=int)
x.dtype

x = np.zeros((5, 1), dtype=int)
x = np.zeros((5, 5), dtype=int)

# np.ones(shape, dtype)
# retun a new array of given shape and type, filled with all ones.
np.ones(5)
np.ones((5,5))

# np.full(shape, fill_value, dtype)
# return a new array of given shape and type, filled with "fill_value".
np.full(5, 3.14)
np.full((5,5), 3.14)

# np.eye(N, dtype)
# return the N times N identity matrix.
np.eye(3)

# np.diag()
# construct the diagonal matrix with diagonal entries specified.
np.diag((1, 3, 5))
# extract diagonal components
y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.diag(y)

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


## Create Arrays with Random Numbers ##
# np.random.uniform(low, high, size)
# return random samples from a uniform distribution with an interval [low, high].
np.random.uniform(0, 1, 5)
np.random.uniform(0, 1, (2,2))

# np.random.normal(loc, scale, size)
# return random samples from a normal distribution with specified mean (loc) and standard deviation (scale).
np.random.normal(0, 1, 5)
np.random.normal(0, 1, (2,2))

# np.random.randint(low, high, size)
# return random integers from "low" (inclusive) to "high" (exclusive).
np.random.randint(0, 10, 5)
np.random.randint(0, 10, (2,2))

# np.random.choice(a, size, replace)
# return random samples from a given 1-D array
x = np.array([-1, 3.5, 5.2, 7, 9.1])
np.random.choice(x, 2, replace = False)
np.random.choice(np.arange(20), (2,5), replace = False)

# np.random.rand(d0, d1, ...)
# return array with given shape with random samples from a uniform distribution over [0, 1).
np.random.rand(5)
np.random.rand(5,1)
np.random.rand(2,2)

# np.random.seed(seed)
# seed generator to make the random numbers reproducible
np.random.seed(seed=0)
np.random.randint(0, 10, (3,3))


## NumPy Array Indexing and Slicing ##
# 1D array indexing and slicing
np.random.seed(seed=1)
x = np.random.randint(10, 100, 10)
x[0]
x[-1]
x[0:3]
x[:3]
x[5:10]
x[5:]
x[:]
x[::2] # stepping by 2
x[:5:2]
x[::-1] # equivalent to the reverse of x

# 2D array indexing and slicing
np.random.seed(seed=1)
x = np.random.randint(0, 100, (5,5))
x[0,0] # when retrieving a particular element in a 2D array, row index first, followed by the column index.
x[0,1] 
x[:,0] # first column
x[:,:3] # first three columns
x[0,:] # first row (or x[0])
x[:3,:] # first three rows (or x[:3])


## Arrary Concatenation ##
# np.concatenate()
# in NumPy, axis 0 refers to the row axis, while axis 1 refers to the column axis.
# one-dimensional array
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

z = np.concatenate((x,y))
z = np.concatenate((x,y), axis=0) # it is not stacking!

# two-dimensional array
x = np.array([[1, 2, 3]])
y = np.array([[4, 5, 6]])

z = np.concatenate((x,y), axis=0)
z = np.concatenate((x,y), axis=1) 

x = np.array([[1, 2, 3,], [4, 5, 6]])

z = np.concatenate((x,x), axis=0)
z = np.concatenate((x,x), axis=1)


## Array Reshaping ##
# 1D array to 2D array
x = np.array([1, 2, 3, 4, 5, 6])
y = x.reshape((2,3))
z = x.reshape((2,-1)) # -1 is placeholder
z = x.reshape((-1,2))

# 2D to 1D
x = np.array([[1, 2, 3], [4, 5, 6]])
z = x.reshape(-1)

# instead of reshaping a 1D array into a 2D array, we can add new axis.
x = np.array([1, 2, 3, 4, 5, 6])
y = x[:,np.newaxis]
z = x.reshape(-1,1)

y = x[np.newaxis,:]
z = x.reshape(1,-1)


