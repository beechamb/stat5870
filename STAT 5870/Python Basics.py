### Basic Types ###
## Numbers
# Integer
x = 5
type(x)
print(x, type(x))

# Floating point number
x = 3.141592
type(x)
print(x, type(x))

## Strings
a = "How are you?"
type(a)
print(a, type(a))

a = "She said, "How are you?""
print(a) # Error!

a = "She said, \"How are you?\""
print(a)

a = 'She said, "How are you?"'
print(a)

# The len function is a built-in function of Python, 
# which is used for getting the length of a list of any type.
len(a) 

# String additions and multiplications
a = "hello"
b = "world"
a + b 
a * 3
(a + b) *3

# \t: tab, \n: new line
a = "STAT\t5870"
print(a)
a = "STAT5870\nWelcome!"
print(a)


### Indexing and slicing ###
a = "Western_Michigan_University"
a
len(a)

## Indexing
# Python index starts from 0, increments by 1,
# and ends at the len(a)-1
a[0]
a[1]
a[26]

# Python also indexes the arrays backwards,
# using negative numbers.
a[-1]
a[-2]
a[-27]
    
## Slicing
# You can use a colon to apply a range filter.
# Note that a[i:j] will return a string 
# starting with a[i] and ending with a[j-1], not a[j].
a[0:3]
a[0:7]
a[:7] # you can skip the starting index 0, if it starts from 0.
a[7:27]
a[7:] # you can skip the ending index, if it ends to the end.


### Data Structures ###
## Lists
x = [1, 2, 3]
x
type(x)
print(x, type(x))

y = ["a", "b", "c"]
y
print(y, type(y))

# List elements do not have to be of the same type.
z = [1, 2, 3, "a", "b", "c"]
z
len(z)

# List creation shortcuts
x * 5
x + y # returns a new copy of list

# Indexing and slicing of lists
# Same as that of strings
z[0]
z[:3]

# Sorting lists
l = [1, 6, 3, 4, 2, 5]
l.sort()
l

l = [1, 6, 3, 4, 2, 5]
l.sort(reverse = True)
l

l = [1, 6, 3, 4, 2, 5]
sorted(l) # returns a new copy
sorted(l, reverse = True)

# Removing 
l = [1, 2, 3, 4, 5]
l.pop() # remove the last item
l

l.pop(0) # remove the first item
l

# Aggregating
l = [1, 2, 3, 4, 5]
min(l)
max(l)
sum(l)


## Dictionaries
colleges = {"CAS": "College of Arts and Sciences", "CEAS": "College of Engineering and Applied Sciences"}
colleges
type(colleges)

colleges["CAS"]
colleges["CEAS"]

colleges.keys()
colleges.values()

colleges["HCB"] = "Haworth College of Business"
colleges


### Operators ###
## Arithmetic operators
# Python supports all types of arithmetic operators
(1 + 2) * 3

# Powers
2 ** 10 # 2 to the power of 10

# Division
5 / 2 # true division
5 // 2 # floor division
5 % 2 # remainder division

# Negation
x = 1
-x

# Short cut for math operation and assignment
a = 1
a = a + 1

a = 1
a += 1

b = 2
b = b * 3

b = 2
b *= 3

## Comparison operators
# Python supports all types of comparison operators
# <, strictly less than
# <=, less than or equal to
# >, strictrly greater than
# >=, greater than or equal to
# ==, equal
# !=, not equal
a = 1
b = 3
a == b
a != b
a > b
a < b


### Flow Control ###
## Branches
x = -3
if x > 0:
    print("The x is positive.")

if x > 0:
    print("The x is positive.")
else: # for the rest of cases
    print("The x is negative.")
    
if x > 0:
    print("The x is positive.")
elif x == 0:
    print("The x is zero.")
else:
    print("The x is negative.")
    
if x > 0:
    print("The x is positive.")
else:
    if x == 0: # if statements can be nested
        print("The x is zero.")
    else:
        print("The x is negative.")
     
## Loops
l = ["a", "b", "c"]
for i in l:
    print(i)
    
# If you do need to iterate over a sequence of numbers use range() function.
# range() returns a range abject, which is nothing but a sequence of integers.    
# range(n) = [0, 1, 2, ..., n-1]
for i in range(3):
    print(l[i])
     
for i in range(1,3):
    print(l[i])
        
l1 = ["a", "b", "c"]
l2 = ["x", "y", "z"]
for i in l1: # outer for loop
    for j in l2: # inner for loop
        print(i + j)

l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in l:
    if i == 5:
        break # exit the loop
    else:
        print(i)
        
l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in l:
    if i % 2 == 0: # if i is an even number
        continue # move on to the next iteration of the loop
    else:
        print(i)   
        
        
### User-Defined Functions ###
# In below example, 
# square is the function name,
# x is the argument,
# x * x is the return value of this function.
def square(x):
    return x * x

square(3)
square(4)


