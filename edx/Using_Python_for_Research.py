# Using Python for Research; Harvard University





# Week 0: Introductions and Self-Assessment/Introduction and Welcome

# In a scientific context, we care deeply about the transparency
# and replicability of results.
# That's the only way how science accumulates.
# You may be working with sensitive data sets that you cannot share with anybody
# else.

# For one, Python is a very clear and powerful programming language.
# Python also has a very elegant syntax.
# Python has a large standard library, which supports many common programming tasks.
# If, at some point, you find that Python is not enough
# you can always extend Python using other modules written in other languages.










# Week 1: Basics of Python 3/Part 1: Objects and Methods

# 1.1.1: Python Basics

# Python is interpreted language that means you can run them without compiling.
# Python has two modes. Interactive mode and standart (script) mode.
# The interactive mode is meant for experimenting your code one line or one expression at a time.
# In contrast, the standard mode is ideal for running your programs from start to finish.

# Python is high level language.






# 1.1.2: Objects

# Python contains many data types as part of the core language.
# All data in a Python program is represented by objects and by relationships between objects.

# The value of some objects can change in the course of program execution.
# Objects whose value can change are said to be mutable objects,
# whereas objects whose value is unchangeable after they've been created
# are called immutable.

# Python also contains building functions that
# can be used by all Python programs.
# The Python library consists of all core elements, such as data types
# and built-in functions
# but the bulk of the Python library consists of modules.

# Each object in Python has three characteristics. These characteristics are called 
# object type, object value, and object identity. 
# Object type tells Python what kind of an object it's dealing with. 
# A type could be a number, or a string, or a list, or something else.
# Object value is the data value that is contained by the object. 
# This could be a specific number, for example.
# Finally, you can think of object identity as an identity number for the object. 
# Each distinct object in the computer's memory
# will have its own identity number.

# Most Python objects have either data or functions or both associated with them. 
# These are known as attributes.
# The name of the attribute follows the name of the object.
# And these two are separated by a dot in between them.

# The two types of attributes are called either data attributes or methods.
# A data attribute is a value that is attached to a specific object.
# In contrast, a method is a function that is attached to an object.

object.date_attribute
object.method() == object.function()

# Object type always determines the kind of operations that it supports.
# In other words, depending on the type of the object, different methods 
# may be available to you as a programmer.

# Finally, an instance is one occurrence of an object. 
# For example, you could have two strings.
# They may have different values stored in them, but they nevertheless 
# support the same set of methods.



# Python has immutable objects (e.g., strings and tuples) and mutable 
# objects (e.g., dictionaries and lists). 
# What does it mean for an object to be immutable?
# Its contents cannot be modified by the programmer after the object has been created.

# What is the difference between methods and data attributes of objects?
# Methods are functions associated with objects, whereas data attributes are data associated with objects.

a = [1, 2]; b = a; b[0] = 0;
id(a); id(b)
c = 1; d = c; d = 2 * d
id(c); id(d)
c = '12'; d = c; d = d[:1]
id(c); id(d)








# 1.1.3: Modules and Methods

# Python modules are libraries of code and you can import Python 
# modules using the import statements.

# The module comes with several functions.
# Shouldn't be with several data attributes?
import math
math.pi

# The math module also comes with several functions, or methods.
math.sqrt(10)
math.sin(math.pi / 2)

from math import pi
pi


# Well namespace is a container of names shared by objects that typically go together.
import math
import numpy as np
math.sqrt
np.sqrt
math.sqrt(2)
np.sqrt(2)
math.sqrt([2, 3, 4]) # doesn't work with math.sqrt()
np.sqrt([2, 3, 4])


# What exactly happens when you run the Python import statement?
# Three things happen.
# The first thing that happens is Python creates
# a new namespace for all the objects which are defined in the new module.
# So in abstract sense, this is our new namespace.
# That's the first step.
# The second step that Python does is it executes the code of the module
# and it runs it within this newly created namespace.
# The third thing that happens is Python creates
# a name-- let's say np for numpy-- and this name references this new namespace
# object.
# So when you call np.sqrt function, Python is using the sqrt function within the numpy namespace.


# If you wanted to know what is the type of this object
name = 'Amy'
type(name)
# find out what other methods that are available.
# dir function, to get a directory of the methods.
# In this case, I know that name was a string, so instead of typing name, 
# I can just type str and Python will give me the same exact list of methods.
# directory of attributes
dir(name)
dir(str)
help(name.upper)
help(str.upper)
help(str)
name.upper
name.upper()
help(name.upper())

# Suppose that math.sqrt and numpy.sqrt had identical behavior. 
# Are they the same function?
# No. Because they belong to different namespaces, Python treats 
# them separately, regardless of their behavior.









# 1.1.4: Numbers and Basic Calculations

# Numbers are one type of object in Python.
# And Python, in fact, provides three different numeric types.
# These are called integers, floating point numbers, and complex numbers.
# Python has unlimited precision for integers.

# If we use integer division (floor division), Python is going to give us an answer of 2.
15 / 7
15 // 7

# The interactive mode in Python provides a very useful operation which is the underscore operator.
# And the value of the underscore operator is always the last object that Python has returned to you.

15 / 2.3
_ * 2.3

import math
math.factorial(5)







# 1.1.5: Random Choice

import random
random.choice([1, 2, 3])
random.choice(['aa', 'bb', 'cc'])

import math
math.factorial(6)

from random import choice
choice([2, 44, 55, 66])








# 1.1.6: Expressions and Booleans

# Expression is a combination of objects and operators that computes a value.
type(True)

# boolean type values: True, False;
# Operations involving logic, so-called boolean operations, 
# take in one or more boolean object and then
# they return one boolean object back to you. There are only three 
# boolean operations, which are "or", "and", and "not".

True or False
True and True
not False

2 < 4
2 <= 2
2 == 2
2 != 2


# These two comparisons are used to test whether two objects are the one and the same. is, is not.
# Notice that this is different than asking if the contents of two objects are the same. ==, !=
# What is the difference between == and is?
# == tests whether objects have the same value, whereas is tests whether objects have the same identity.

[2, 3] == [2, 3]
[2, 3] == [3, 3]
[2, 3] is [2, 3]
[2, 3] is not [2, 3]

# Python takes the second number, which is number 2, an integer-- 
# it turns that into a floating point number.

2.00 == 2

True == True
True is True

# Consider the expression True and not False is True
# True

a = [1, 2]
b = [1, 2]
c = a
d = a[:]

a == b
a == c
a == d

a is b
a is c
a is d

# Identity
id(a)
id(b)
id(c)
id(d)


























# Week 1: Basics of Python 3/Part 2: Sequence Objects

# Sequences
# In Python, a sequence is a collection of objects ordered by their position.
# In Python, there are three basic sequences, which are lists, tuples, and so-called "range objects".
# But Python also has additional sequence types for representing things like strings.
# ... any sequence data type will support the common sequence operations. 
# generic sequence functions/operations eg. +, len().
# in addition, these different types will have their own methods available 
# for performing specific operations.
# Sequences are called "sequences" because the objects that they contain form a sequence.

# The first, fundamental aspect to understand about sequences is that indexing starts at 0. 
# ... or we can use a negative index, which is counting positions from right to left. 
# In that case, we have to use the negative 1 for the very last object in our sequence.

# : slices, slicing
# some_list[lower:upper]

# Consider the following tuple and index (1,2,3)[-0:0]. What will this return?
# ()













# 1.2.2: Lists

# Lists are mutable sequences of objects of any type.
# whereas lists are sequences of any type of Python objects.
# It is common practice for a list to hold objects of just one type, although this is not strictly a requirement.

a = [1]
a.append(2)
a.reverse()
b = [3, 4]
# Concacenate lists
a + b
# Add elements of two lists
c = [a[i] + b[i] for i in range(len(a))]

# But see, Python doesn't return any object to me. This is because list methods are what are called in-place methods.
# They modify the original list.
# It operates on the list that I have, meaning the original list. It moves the last object first and the first object last.
c.reverse()
list(reversed(c))

# list methods are what are called in-place methods. They modify the original list.
c.sort(reverse=True)
names = ['Zofia', 'Bill', 'Alan']
names.sort()

# It will construct this new list using the objects
sorted(c, reverse=True)
sorted(names)

# When we're using the list method sort, we're taking the existing list and reordering the objects within that list.
# If we're using the sorted function, we're actually asking Python to construct a completely new list.
# It will construct this new list using the objects in the previous list in such a way 
# that the objects in the new list will appear in a sorted order.

# Finally, if you wanted to find out how many objects our list contains, we can use a generic sequence function, len.

# What do list methods such as reverse and sort return?
# They return nothing because they are in-place methods, meaning they alter the content of the original list.











# 1.2.3: Tuples

# Tuples are immutable sequences typically used to store heterogeneous data.
# The best way to view tuples is as a single object that consists of several different parts.

# contatenate tuples
(1, 2) + (3, 4)

# packing and unpacking tuples.
# Packing a tuple, or tuple packing.
a = 12.34
b = 23.45
coordinate = (a, b)
type(coordinate)

# unpacking a tuple
(a1, b1) = coordinate
a1

coordinatesXY = [
    (55, 56), 
    (12, 40)
    ]

type(coordinatesXY)
for (x, y) in coordinatesXY:
    print(x, y)

a = (2)
type(a)
a = tuple([2])
type(a)
b = (2,)
type(b)

x = (1, 2, 3)
x.count(3)
sum(x)

# Which of the following prints type tuple?
type((2,))











# 1.2.4: Ranges

# Ranges are immutable sequences of integers, and they are commonly used in for loops.
# Ranges take up less memory than lists because they do not hold all the numbers simultaneously.
# To store a range object, Python is only storing three different numbers, 
# which are the starting number, the stopping number, and its step size.

range(5)
list(range(5))
list(range(1, 6))
list(range(1, 13, 2))












# 1.2.5: Strings

# Strings are immutable sequences of characters.
# Strings are sequences of individual characters,

S = 'Python'
S[0]
S[-1]
S[:3]
S[-3:]
S[:-3]
'y' in 'Python'
'Y' in 'Python'

# polymorphism means that what an operator does depends on the type of objects it is being applied to.
# concatenation
'2' + '2'
3 * S
'eight equals ' + str(8)

# directory of attributes
dir(str)
# this help with ? doesn't work
str.replace?
help(str.replace)

name = 'A Alan'
name.replace('A', 'a')
names = name.split(' ')
' '.join([name.upper() for name in names])
name
help(str.join)


from string import digits
digits
list(range(10))

'22'.isdigit()
dir(str)
help(str)
help(str.isdigit)













# 1.2.6: Sets

# Sets are unordered collections of distinct hashable objects.
# cannot be indexed. So the objects inside sets don't have locations.
# In practice, what that means is you can use sets for immutable objects
# like numbers and strings, but not for mutable objects like lists and dictionaries.
# elements can never be duplicated. all of the objects inside a set
# are always going to be unique or distinct.

# There are two types of sets. One type of set is called just "a set".
# And the other type of set is called "a frozen set".
# Frozen set is immutable, normal set is mutable
# One of the key ideas about sets is that they cannot be indexed. So the objects inside sets don't have locations.
# elements can never be duplicated.

# A set is an unordered collection of items. Every set element is unique 
# (no duplicates) and must be immutable (cannot be changed).
# However, a set itself is mutable. We can add or remove items from it.

ids = set()
ids = {1, 3, 5}
type(ids)
ids = set((1, 3, 5))
ids = set([1, 3, 5])
len(ids)
ids.add(10)
ids.add(1)
# removes element from set
ids.pop()

ids = set(range(10))
males = {1, 3, 5, 6, 7}
type(males)

# Set operations
# difference
females = ids - males
ids.difference(males)
# OR, union
ids | males
males.union(females)
# AND (ampersand), intersection
females & males
females.intersection(males)
# XOR, symmetric difference
females ^ males
females.symmetric_difference(males)

# implication =>
# p => q = ~p v q
not females or males

word = "antidisestablishmentarianism"
set(word)
word.count('a')
# how many distinctive letters in a word
len(set(word))

from collections import Counter
Counter(word)

# issubset
males.issubset(ids)
# in doesn't work
males in ids
'A' in 'ABC'














# 1.2.7: Dictionaries

# Dictionaries are mappings from key objects to value objects.
# Dictionaries consists of Key:Value pairs, where the keys must be immutable and the values can be anything.
# Dictionaries themselves are mutable
# they are not sequences, and therefore do not maintain any type of left-right order.

age = {}
type(age)
age = dict()

age = {'Tim': 28, 'Jim': 35, 'Pam': 40}
# view object, As we modify the dictionary, the content of the view object will change automatically.
# This is the nature of views objects, their content is dynamically updated as you modify your dictionary.
names = age.keys()
type(names)
ages = age.values()
type(ages)
age['Tim']
age["Tim"] += 2

age.keys()
type(age.keys())
age.values()
type(age.values())
age.items()

age.update({'Tom': 50})
age['Tom2'] = 50

age[0] = 1
age[0]

# Test membership in dictionary.
'Tim' in age
'Tim' in age.keys()




# Why may you not edit the key "Tim" to "Tom"?
# Dictionary keys are not mutable.

# Which of the following data structures may be used as keys in a dict?
# Strings, Tuples, not Lists






























# Week 1: Basics of Python 3/Part 3: Manipulating Objects
# 1.3.1 Dynamic Typing

# We're going to talk about type checking, but let's first talk about types.
# What does a type mean the context of a computer?
# So if we take a look at a computer's memory,
# instead of strings or numbers or dictionaries,
# all we have is a sequence of zeroes and ones.
# And the question is, how should a program interpret
# these sequences of zeros and ones?
# What a type does is two things.
# First, it tells a program, you should be reading these sequences
# in chunks of, let's say, 32 bits.
# So that's the first thing.
# The second thing that it tells computer is, what does this number
# here, this sequence of bits, represent?
# Does it represent a floating-point number, or a character,
# or a piece of music, or something else?

# Typing or assigning data types refers to the set of rules
# that the language uses to ensure that the part of the program receiving
# the data knows how to correctly interpret that data.
# Some languages are statically typed, like C or C++,
# and other languages are dynamically typed, like Python.
# Static typing means that type checking is performed during compile time, 
# whereas dynamic typing means that type checking is performed at run time.

# There are three important concepts here-- variable, object, and reference.
# So when you assign variables to objects in Python,
# the following three things happen.
# First, if we type x equals 3, the first thing Python will do
# is create the object, in this case, number 3.
# The second thing Python will do is it will create the variable name, x.
# And the third thing is Python will insert a reference from the name
# of the variable to the actual object.

# Variable names and objects are stored in different parts of computer's memory.
# A key point to remember here is that variable names always link to objects, never to other variables.
# A variable is, therefore, a reference to the given object.
# In other words, a name is possibly one out of many names
# that are attached to that object.

# When assigning objects, it's useful to keep in mind the distinction between mutable and immutable objects.
# Remember, mutable objects, like lists and dictionaries, can be modified at any point of program execution.
# In contrast, immutable objects, like numbers and strings, cannot be altered after they've been created in the program.

# dynamic typing illustration
# immutable
"""
Let's work through these three lines of code.
When we're first asking Python to run the first line, x is equal to 3,
remember the following steps take place.
Python first creates the object, 3, then it creates the variable name, x,
and, finally it inserts reference from the variable name to the object itself.
When we go and look at the second line, y
is equal to x. The object x, in this case number 3, already exists.
The next step Python does is it creates a new variable name, which
is equal to y, and then it inserts a reference to the object
that x, variable name x, is currently referencing.
Remember, a variable cannot reference another variable.
A variable can only reference an object.
We then will want the third line, which is y equals y minus 1.
Python first looks at the object here, which is our number 3.
But we know that numbers are immutable.
Therefore, in order to do the subtraction,
a new object has to be created, which in this case is the number 2.
We already have the variable name y in our computer's memory.
So the final thing Python does is, it removes this reference
and inserts a reference from y to the object 2.
So once we've run all the these three lines of code,
x will eventually be equal to 3 and y will be equal to 2.
"""

x = 3
y = x
y -= 1
x

# mutable
"""
Let's then look at dynamic typing for mutable objects.
The behavior for mutable objects, like lists and dictionaries,
looks different, although it really follows the same logic
as for immutable objects.
To illustrate the concepts, I've written down three lines of code.
Let's start from the first one, L1 is equal to a list which
consists of the numbers 2, 3, and 4.
Let's think about what happens as this line is run.
First, Python creates the object-- list 2, 3, 4.
The second step, Python creates the variable name L1.
And third, because of the assignment, L1 will reference this list.
If we look at the second step, L2 is equal to L1,
the object L1 which currently references the list already exists.
Therefore, Python only needs to do two things.
The first thing is it creates the variable name L2.
Because L2 cannot reference L1, which is another variable,
it must reference the object that L1 references.
Therefore, L2 essentially becomes a synonym for the very same object.
When we look at the third line, L1 application 0 equals 24.
In this case, what happens is we are using the name
L1 to reference this object, and we're modifying
the content of the number at location 0 from 2 to 24.
After this modification, the content of the list is going to be 24, 3, and 4.
By looking at the code here, your first impression
might have been that you have two lists, L1 and L2,
and only list L1 gets modified.
However, if you understand how dynamic typing works in Python,
you have realized that we only have two names that
reference the very same object.
The last line, L1 at location 0 equals 24
would have been identical to typing L2 at location 0 equals 24.
This is again for the reason that both of these variable names, L1 and L2,
reference the very same object.


Each object in Python has a type, value, and an identity.
Mutable objects in Python can be identical in content
and yet be actually different objects.
Let's illustrate this point with a simple example.
I'm going to define a list L which has the elements 1, 2, and 3.
And I'm also going to find another list, which I'm going to call M,
and it also has the elements 1, 2, and 3.
I can compare these lists using two equal signs.
So I am asking, is L equal to M?
When we're comparing two lists, the actual comparison
is carried out element-wise.
So this 0-th element in L is compared with the 0-th element of M,
and so forth.
In this case, the content of these two lists is identical.
Therefore, when I ask, is L equal to M, Python returns "True".
But there is another way how we can compare these two objects.
We can also ask, is L the same object as M?
And in this case, Python returns "False".
What's happening here?
We can use the id function to obtain the identity of an object.
And the number returned by Python corresponds to the object's location
in memory.
So if I type id of L, Python returns a number.
If I type id of M, Python returns another number.
You can think of these numbers as some type of social security
numbers for these different objects.
So typing L is M is really the same as typing,
is the identity of L equal to the identity of M?
The main point here is that mutable objects can be identical in content,
yet be different objects.
Consider a situation where we have defined a list, L, again
consisting of the numbers 1, 2, and 3.
What if I wanted to create a copy of that list?
Remember, if I type M is equal to L, in that case
M is just another name for the same list, L.
But what if I wanted to create a completely new object that
has identical content to L?
The easiest way to do this is the following.
I can ask Python to create a new list, and then assign
that list object to a new variable.
This is the syntax.
I already have my list object, L, in the computer's memory.
By typing list parentheses L, Python will
create a completely new object for me
but the content of that list is going to be identical to the content of list L.
In this case, if I ask, M is L, I get a false.
That's because these two objects are distinct.
But if I ask, is M equal to L in content,
the answer is going to be true.
Another way to create a copy of a list is to use the slicing syntax.
So I could have said something like, M is equal to L, putting square brackets,
and taking every single element from the list L.
This results in the creation of a new list object which is then
assigned to the variable name M.
"""

L1 = [1, 2, 3]
L2 = L1
L1[0] = 24
L2 == L1
L2 is L1
L3 = L1[:]
L4 = list(L1)

id(L1)
id(L2)
id(L3)
id(L4)












# 1.3.2: Copies

# There are two types of copies that are available.
# A shallow copy constructs a new compound object and then insert its references into it to the original object.
# a deep copy constructs a new compound object and then recursively inserts copies into it of the original objects.

import copy
x = [1,[2]]
y = copy.copy(x)
z = copy.deepcopy(x)
y is z


"""
Let's clarify this with a diagram.
Let's think about a situation where we have an object x,
and we have references to other objects embedded in our object x.
I'm going to call these objects a and b.
If we construct a shallow copy, this is what happens.
A copy of the object x is going to be created.
Let's called is x prime.
But then we will insert references in x to the original objects a and b.
We call this type of copy a shallow copy of x.
Let's then think about what happens if we have a deep copy.
We will create a copy of x.
Let's call that x prime prime.
And now we will also create copies of a and b.
I'm going to call these a prime prime and b prime prime.
These are their own objects.
And into my new copy of x, I'm going to insert references
to the copies of a and b.
This object is called a deep copy of the object x.
"""














# 1.3.3: Statements

# Statements are used to compute values, assign values, and modify attributes, among many other things.
# The return statement is used to return values from a function.
# import statement, which is used to import modules.
# the pass statement is used to do nothing in situations where we need a placeholder for syntactical reasons.

# Compound statements contain groups of other statements, and they affect or control 
# the execution of those other statements in some way.
# Compound statements typically span multiple lines.
# A compound statement consists of one or more clauses, where a clause consist of a header and a block or a suite of code.
# The close headers of a particular compound statement start with a keyword, end with a colon, and are
# all at the same indentation level.
# A block or a suite of code of each clause, however, must be indented to indicate that it forms a group of statements
# that logically fall under that header.

# Here's an example of compound statement with one clause.
# Line 1 here is the header line.
# And lines 2 and 3 form the block of code.
# Let's look at this compound statement in a little bit more detail.
# On line 1, we first ask, if x is greater than y followed by a colon.
# If this is true, if x really is greater than y,
# then Python will run lines 2 and 3.
# On line 2, we're calculating the difference as x minus y.
# And on line 3, we are printing out the message, x is greater than y.
# Regardless of what happens with the comparison,
# line 4 will always get printed.
if x > y:
    diff = x - y
    print('x is greater than y')
print("Foo")

if False:
    print("False!")
elif True:
    print("Now True!")
else:
    print("Finally True!")











# 1.3.4: For and While Loops

# The For Loop is a sequence iteration that assigns items in sequence
# to target one at a time and runs the block of code for each item.
# Unless the loop is terminated early with the break statement, the block of code
# is run as many times as there are items in the sequence.

for x in range(10):
    print(x)


names = ['Tom', 'Anna', 'Sam']
for name in names:
    print(name)

for i in range(len(names)):
    print(names[i])

age = {'Tim': 28, 'Jim': 35, 'Pam': 40}
for name in age.keys():
    print(name, age[name])

for name in age:
    print(name, age[name])

for (key, val) in age.items():
    print(key, val)

for name in sorted(age):
    print(name, age[name])

for name in sorted(age, reverse=True):
    print(name, age[name])


bears = {"Grizzly": "angry", "Brown": "friendly", "Polar": "friendly"}
for bear in bears:
    if bears[bear] == 'friendly':
        print('Hello, '+bear+' bear!')
    else:
        print('odd')


n = 11
not any([n%i == 0 for i in range(2, n)])


n = 100
number_of_times = 0
while n >= 1:
    n //= 2
    number_of_times += 1
print(number_of_times)










# 1.3.5: List Comprehensions
# One is list comprehensions are very fast.
# The second reason is list comprehensions are very elegant.

[i**2 for i in range(10)]

sum([i for i in range(10) if i%2 == 1])
















# 1.3.6: Reading and Writing Files

# changing current working directory if needed
from os import getcwd, chdir
getcwd()
chdir('/home/ukasz/Documents/IT/Python/')

# reading
path = 'edx/'

# open(filename) generates a file object.
for line in open(path+'input.txt'):
    print(line)


# Now, at the end of the line, although you don't see it,
# we always have a line break character.
# So if we have a Python string where line is equal to first,
# although you don't see this character, it's
# going to cause extra line breaks in subsequent processing of your text.
# There is a way to remove this character using the rstrip() method.
# If you call line.rstrip(), Python extracts the line feed character
# and leaves you with the first part of the string.
for line in open(path+'input.txt'):
    line = line.rstrip()
    print(line)

for line in open(path+'input.txt'):
    line = line.rstrip().split(' ')
    print(line)


# writing
# We indicate this by providing that second argument as a string,
# and the content of the string is simply "w".
# What this does is it creates a file object
# for writing to a file named output.txt.
# The next step for us is to capture this file object in some variable.
# However, we have to add an extra character, which
# is the line break character that we extracted
# when we were reading the file.
F = open(path+'output.txt', 'w')
F.write('Python\n')
F.close()


F2 = open(path+'output2.txt', 'w+')
F2.write('Hello\nWolrd\n')
F2.close()

lines = []
for line in open(path+'output2.txt'):
    lines.append(line.strip())
print(lines)











# 1.3.7: Introduction to Functions
# Functions are devices for grouping statements so that they can be 
# easily run more than once in a program.
# Functions maximize code reuse and minimize code redundancy.
# Functions enable dividing larger tasks into smaller chunks, 
# an approach that is called procedural decomposition.

def add(a, b):
    mysum = a + b
    return mysum

add(2, 3)
add

# Arguments to Python functions are matched by position.
# An argument is an object that is passed to a function as its 
# input when the function is called.
# A parameter, in contrast, is a variable that is used in the function 
# definition to refer to that argument.

# mutalbe object
def modify(mylist):
    mylist[0] *= 10

L = [1, 3, 5, 7, 9]
modify(L)
L


def modify2(mylist):
    mylist[0] *= 10
    return(mylist)

L = [1, 3, 5, 7, 9]
modify2(L)
M = modify2(L)
M is L

id(M)
id(L)













# 1.3.8: Writing Simple Functions

# Let's try writing a simple function in Python.
# To do this, I've split my screen into two parts.
# At the top, I have my interactive mode.
# And at the bottom, I have my editor mode.
# I've chosen to write the function in the editor,
# because functions that involve more than two or three lines
# are just easier to deal with in the editor mode.


# lists intersection
def intersect(s1, s2):
    res = []
    for x in s1:
        if x in s2:
            res.append(x)
    return res

intersect([1, 2, 3, 4], [3, 4, 5, 6])

def intersect2(s1, s2):
    return [x for x in s1 if x in s2]

intersect2([1, 2, 3, 4], [3, 4, 5, 6])

# create string from character set with specified length
import random
def password(length):
    characters = 'abcdef' + '012356789'
    return ''.join([random.choice(characters) for _ in range(length)])

password(5)




''.join([str(elem) for elem in list(range(5))])
type('') == str
4 in 'ae'

def is_vowel(letter):
    if type(letter) == int:
        letter = str(letter)
    if letter in "aeiouy":
        return(True)
    else:
        return(False)

is_vowel(4)


def factorial(n):
    if n == 0:
        return 1
    else:
        N = 1
        for i in range(1, n+1):
            N *= i
        return(N)

factorial(5)







# 1.3.9: Common Mistakes and Errors


















# Homework: Week 1
# Week 1 Homework: Exercise 1

import string

alphabet = string.ascii_letters




sentence = 'Jim quickly realized that the beautiful gowns are expensive'

count_letters = {}

# from collections import Counter
# Counter(sentence)

for letter in sentence:
    if letter in alphabet:
        count_letters[letter] = count_letters.get(letter, 0) + 1

list(count_letters.values())[7]
# count_letters.get('ą', 0)




def counter(input_string):
    count_letters = {}
    for letter in input_string:
        if letter in alphabet:
            count_letters[letter] = count_letters.get(letter, 0) + 1
    return count_letters

counter(sentence)




address = """Four score and seven years ago our fathers brought forth on this continent, a new nation, 
conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a 
great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. 
We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final 
resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper 
that we should do this. But, in a larger sense, we can not dedicate -- we can not consecrate -- we can not hallow -- 
this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add 
or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. 
It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so 
nobly advanced. It is rather for us to be here dedicated to the great task remaining before us -- that from these honored 
dead we take increased devotion to that cause for which they gave the last full measure of devotion -- that we here 
highly resolve that these dead shall not have died in vain -- that this nation, under God, shall have a new birth of 
freedom -- and that government of the people, by the people, for the people, shall not perish from the earth."""   

address_count = counter(address)
address_count['h']




max(address_count, key=address_count.get)
# address_count[max(address_count, key=address_count.get)]



# Week 1 Homework: Exercise 2

import math

math.pi / 4




import random

random.seed(1) # Fixes the see of the random number generator.

def rand():
   return random.uniform(-1, 1)

rand()




def distance(x, y):
   return math.sqrt((y[0] - x[0])**2 + (y[1] - x[1])**2)

distance((0, 0), (1, 1))




def in_circle(x, origin = [0,0]):
   if distance(x, origin) < 1:
      return True
   else:
      return False

in_circle((1, 1))




random.seed(1) 

inside = [in_circle((rand(), rand())) for _ in range(10000)]
sum(inside) / len(inside)




math.pi/4 - (sum(inside)/len(inside))



# keyword=1
def moving_window_average(x, n_neighbors=1):
    n = len(x)
    width = n_neighbors*2 + 1
    x = [x[0]]*n_neighbors + x + [x[-1]]*n_neighbors
    # To complete the function,
    # return a list of the mean of values from i to i+width for all values i from 0 to n-1.
    return [sum(x[i:i+width])/len((x[i:i+width])) for i in range(n)]

x = [0,10,5,3,1,5]
print(moving_window_average(x, 1))




random.seed(1) # This line fixes the value called by your function,
               # and is used for answer-checking.
    
random_numbers = [random.random() for _ in range(10000)]
# moving_window_average(random_numbers, 5)[9]
Y = [moving_window_average(random_numbers, i) for i in range(1, 10)]
Y[4][9]




ranges = [max(y) - min(y) for y in Y]
ranges












































# Week 2: Python Libraries and Concepts Used in Research/Week 2 Overview

# 2.1.1: Scope Rules

# But how does Python know which update function to call
# or which variable x to use?
# The answer is that it uses so-called "scope rules"
# to make this determination.
# It searches for the object, layer by layer,
# moving from inner layers towards outer layers,
# and it uses the first update function or the first x variable that it finds.

# You can memorize this scope rule by the acronym LEGB.
# LEGB Local, Enclosing Function, Global, Built-in

# In other words, local is the current function you're in.
# Enclosing function is the function that called the current function, if any.
# Global refers to the module in which the function was defined.
# And built-in refers to Python's built-in namespace.

# An argument is an object that is passed to a function as its input when the function is called.
# A parameter, in contrast, is a variable that is used in the function definition to refer to that argument.


def increment(n):
    n += 1
    print(n)

n = 1
increment(n)
print(n)

2; 1


def increment(n):
    n += 1
    return n

n = 1
while n < 10:
    n = increment(n)
print(n)





# 2.1.2: Classes and Object-Oriented Programming

# Our emphasis has been and will be on functions and functional programming,
# but it's also helpful to know at least something about classes
# and object-oriented programming.
# In general, an object consists of both internal data and methods
# that perform operations on the data.
# you can create a new type of object known as a class.

# Inheritance means that you can define a new object type, a new class, 
# that inherits properties from an existing object type.
# When a class is created via inheritance, the new class
# inherits the attributes defined by its base class, the class it is inheriting
# attributes from-- in this case, a list.
# The so-called derived class, in this case "MyList",
# can then both redefine any of the inherited attributes, and in addition
# it can also add its own new attributes.




x = [2, 5, 9, 11, 10, 2, 7]
x.sort()
min(x)
max(x)
x.remove(10)


# The functions defined inside a class are known as "instance methods"
# because they operate on an instance of the class.
# By convention, the name of the class instance is called "self",
# and it is always passed as the first argument
# to the functions defined as part of a class.



class MyList(list):
    def remove_min(self):
        self.remove(min(self))
    def remove_max(self):
        self.remove(max(self))
    def append_sum(self):
        self.append(sum(self))
    def remove_nth(self, i):
        del self[i]

y = MyList(x)

type(x)
type(y)
dir(x)
dir(y)

y.remove_min()
y.remove_max()
y.remove_nth(-1)
y.append_sum()


class NewList(list):
    def remove_max(self):
        self.remove(max(self))
    def append_sum(self):
        self.append(sum(self))

x = NewList([1,2,3])
while max(x) < 10:
    x.remove_max()
    x.append_sum()

print(x)





















# Week 2: Python Libraries and Concepts Used in Research/Part 2: NumPy

# 2.2.1: Introduction to NumPy Arrays

# NumPy is a Python module designed for scientific computation.
# Unlike dynamically growing Python lists, NumPy arrays have a size that is fixed when they are constructed.
# Elements of NumPy arrays are also all of the same data

import numpy as np
np.version.version

np.zeros(5) + np.ones(5)
np.zeros((5, 3))
# we assume that lower case variables are vectors or one-dimensional arrays and upper case variables are
# matrices, or two-dimensional arrays.
np.array([[1, 2], [3, 4]]).transpose()

x = np.array([1, 2, 3])
y = np.array([2, 4, 6])
X = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
Y = np.array([
    [2, 4, 6],
    [8, 10, 12]
])
X.shape
X.size
X.transpose()
np.transpose(X)






# 2.2.2: Slicing NumPy Arrays

x[2]
x[:2]
x + y

X[:, 1]
list(X[:, 1])
# firs row, all the same
X[1]
X[1,]
X[1, :]

[2, 4] + [6, 8]
np.array([2, 4]) + np.array([6, 8])






# 2.2.3: Indexing NumPy Arrays

ind = [1, 2]
y[ind]
y[[1, 2]]
y[1:3]

np_ind = np.array([1, 2])
y[np_ind]

# we can have an array consisting of true and false, which are two Boolean elements.
# Boolean array
# We can use the Boolean array, also called a logical array, to index another vector.
y > 4
x[x > 2]


# When you slice an array using the colon operator, you get a view of the object.
# This means that if you modify it, the original array will also be modified.
# This is in contrast with what happens when you index an array, in which case
# what is returned to you is a copy of the original data.
# In summary, for all cases of indexed arrays, what is returned
# is a copy of the original data, not a view as one gets for slices.

# slicing - object view
x1 = x[1:3]
x1[0] = 5
x
# x vector has been changed too

# indexing - copy of the original data
x1 = x[[1, 2]]
x1[0] = 10



a = np.array([1,2])
b = np.array([3,4,5])

# Consider again the above code, as well as the following:
c = b[1:]
b[a] is c
# False The is comparison operator tests if two objects are the same exact object --- not if 
# they have the same exact values. When testing values, you could try b[a] == c or all(b[a] == c).










# 2.2.4: Building and Examining NumPy Arrays

import numpy as np
np.linspace(1, 100, 10)
np.linspace(10, 100, 10)
np.logspace(1, 2, 10)
plt.plot(np.logspace(1, 2, 10))
np.logspace(np.log10(10), np.log10(100), 10)
np.logspace(np.log10(250), np.log10(500), 10)

X = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
# You can check the shape of an array using shape.
X.shape
# You can check the number of elements of an array with size.
X.size

# Notice that you don't have parentheses following the shape
# or size in the above examples.
# This is because shape and size are data attributes, not methods of the arrays.


# Sometimes we need to examine whether any or all elements of an array
# fulfill some logical condition.
x = np.random.random(10)
np.any(x > 0.9)
np.all(x >= 0.1)


# x%i == 0 tests if x has a remainder when divided by i. 
# If this is not true for all values strictly between 1 and x, it must be prime!
not np.any([x%i == 0 for i in range(2, 20)])














# 2.3.1: Introduction to Matplotlib and Pyplot

# Matplotlib is a Python plotting library that produces publication-quality figures.
# Pyplot is a collection of functions that make matplotlib work like Matlab,
# Pyplot provides what is sometimes called a state machine interface to matplotlib library.

import matplotlib.pyplot as plt
import numpy as np

# If for some reason you'd like to suppress the printing of that object,
# in the IPython Shell you can add a semi-colon at the end of the line
# doesn' work
plt.plot([0, 1, 4, 9, 16])
plt.show()

x = np.linspace(0, 10, 20)
y1 = x**2
y2 = x**1.5
plt.plot(x, y1)
plt.show()

# a keyword argument is an argument which is supplied to the function by explicitly naming each parameter
# and specifying its value.

plt.plot([0, 1, 2], [0, 1, 4], 'rd-')

# In this case, I'm requesting plt to use blue, to use circles, and to use a solid line.
plt.plot(x, y1, 'bo-')
plt.plot(x, y1, 'bo-', linewidth=2, markersize=4, label='First')
plt.plot(x, y2, 'gs-', linewidth=2, markersize=12, label='Second')
plt.show()






# 2.3.2: Customizing Your Plots

plt.xlabel('$X$')
plt.ylabel('$Y$')
# plt.axis[(xmin, xmax, ymin, ymax)]
plt.axis([-.5, 10.5, -5, 105])
plt.legend(loc='upper left')
# plt.savefig("myplot.png")
# plt.savefig("myplot.pdf")
plt.show()








# 2.3.3: Plotting Using Logarithmic Axes

# To understand why logarithmic plots are sometimes useful, let's
# take a quick look at the underlying math.
# Consider a function of the form y is equal to x the power of alpha.
# y is equal to x to the power of alpha.
# If alpha is equal to 1, that corresponds to a line that goes through the origin.
# Alpha equal to half gives you the square root, and alpha equal to 2
# gives us a parabola.
# Let's see what we can do with this equation.
# We're first going to take logs of both sides, which gives us
# log of y is equal to log of x to alpha.
# You may remember that in the logarithm, we can pull the exponent here
# to the front of the expression.
# Let's do just that, in which case we end up with alpha being at the front.
# So our alpha ends up being over here.
# In this case, what we have is log y is equal to alpha times log of x.
# We can now think about plotting this on a different set of axes,
# where instead of using our original y, we have log transformed the axes.
# I'm going to call this y prime.
# I'm going to do the same for my x.
# My alpha stays put.
# So my log x becomes x prime.
# So you'll see that on these new axes, we have a much simpler equation.
# Y prime is equal to alpha times x prime.
# So let's look at this as a plot.

# The shape of this function is going to be simply a line that looks like this.
# In this case, alpha is going to be given by the slope of this line.
# So the lesson here is that functions of the form y is equal to x to power alpha
# show up as straight lines on a loglog() plot.
# The exponent alpha is given by the slope of the line.


# y = x^a
# log y = log x^a
# log y = a log x
# y' = a x'


# So the lesson here is that functions of the form y is equal to x to power alpha
# show up as straight lines on a loglog() plot.
# The exponent alpha is given by the slope of the line.


# x = np.linspace(0, 10, 20)
# even spacing on x axis
x = np.logspace(-1, 1, 40)
y1 = x**2
y2 = x**1.5
plt.loglog(x, y1, "bo-", linewidth=2, markersize=4, label="First")
plt.loglog(x, y2, "gs-", linewidth=2, markersize=4, label="Second")
plt.xlabel("$X$")
plt.ylabel("$Y$")
plt.axis([-.5, 10.5, -5, 105])
plt.legend(loc="upper left")
plt.show()



x = np.logspace(0,1,10)
y = x**2
plt.loglog(x,y,"bo-")
plt.show()










# 2.3.4: Generating Histograms

# Histograms
import matplotlib.pyplot as plt
import numpy as np

x = np.random.normal(size=1000)
plt.hist(x, density=True)
# When we set density to be true, the histogram, in this case on the y-axis,
# instead of having the number of observations that fall in each bin,
# we have the proportion of observations that fall in each bin.
# That's what it means for a histogram to be normalized.
# we have specified 20 bins between the numbers minus 5 and plus 5.
# to have one bin we need to specify
# the start point-- the start location of the bin
# and the end location of that bin.
plt.hist(x, density=True, bins=np.linspace(-5, 5, 21))
plt.hist(x, density=True, bins=20)
plt.show()


# Gamma distribution
x = np.random.gamma(2, 3, 100000)
plt.figure()
#The first integer describes the number of subplot rows, 
# the second integer describes the number of subplot columns, 
# and the third integer describes the location index of the subplot to be created, 
# where indices are laid out along the rows and columns in the same order as reading 
# Latin characters on a page.
plt.subplot(231)
plt.hist(x, bins=30)
plt.subplot(234)
plt.hist(x, bins=30, density=True)
plt.subplot(233)
plt.hist(x, bins=30, cumulative=True)
plt.subplot(236)
plt.hist(x, bins=30, density=True, cumulative=True, histtype="step")
# In this case, we have created just one figure
# with four panels where each type of histogram appears in its own subplot.
plt.show()



































# Week 2: Python Libraries and Concepts Used in Research/Part 4: Randomness and Time

# Video 2.4.1: Simulating Randomness

import random
random.choice(["H", "T"])
random.choice([0, 1])
random.choice(range(1, 7))

random.choice([random.choice(range(1, i)) for i in [6, 8, 10]])
random.choice(range(1, random.choice([6, 8, 10]) + 1))


# do not repeat numbers
random.sample(range(10), 10)
sum(random.sample(range(10), 10)) == sum(range(10))

# picks may repeat
sum(random.choice(range(10)) for _ in range(10))













# 2.4.2: Examples Involving Randomness

import random
import numpy as np
import matplotlib.pyplot as plt
rolls = [random.choice(range(1, 7)) for _ in range(100)]
plt.hist(rolls, density=True, bins=np.linspace(.5, 6.5, 7), ec='black')
plt.show()

# The law of large numbers, which is a theorem of probability,
# tells us that we should expect more or less the same number of 1s and 2s
# all the way to the 6s because they each have the same probability.

# y = x1 + x2 + x3 + ... + xnn

x = sum([random.choice(range(1, 7)) for _ in range(10)])
y = [sum([random.choice(range(1, 7)) for _ in range(10)]) for _ in range(1000000)]
plt.hist(y, density=True, bins=np.linspace(5, 65, 25), ec='black')
# plt.hist(y, density=True, ec="black")
plt.show()

# The so-called central limit theorem, or CLT,
# states that the sum of a large number of random variables
# regardless of their distribution will approximately
# follow a normal distribution.

# What is the law of large numbers with respect to histograms?
# We expect the histogram of a sample to better reflect the distribution 
# as the sample size increases.

# What is the Central Limit Theorem?
# The distribution of the sum of many random variables is approximately normal.









# 2.4.3: Using the NumPy Random Module

import numpy as np

# standard uniform distribution
np.random.random()
np.random.random(5)
# generate a matrix with rand
np.random.random((2, 5))
# Let's then look at the normal distribution. 
# It requires the mean and the standard deviation as its input parameters.
# standard normal distribution, mean = 0, standard deviation = 1
np.random.normal(0, 1, 5)
np.random.normal(0, 1, (2, 5))

# generate random integers
X = np.random.randint(1, 7, (1000000, 10))
X.shape
# summing over all of the rows of the array.
np.sum(X, axis=0)
# summing over all of the columns.
np.sum(X, axis=1)
# sum over all columns
X.sum(1)
X.sum(1,)


help(np.sum)
Y = np.sum(X, axis=1)
plt.hist(Y, density=True, bins=np.linspace(5, 65, 25), ec="black")
plt.hist(Y, density=True, ec="black")
plt.show()











# 2.4.4: Measuring Time
import time
import matplotlib.pyplot as plt
import random
import numpy as np

start_time = time.time()
y = [sum([random.choice(range(1, 7)) for _ in range(10)]) for _ in range(1000000)]
# plt.hist(y, density=True, bins=np.linspace(5, 65, 25), ec="black")
plt.hist(y, density=True, ec="black")
time.time() - start_time
plt.show()

start_time = time.time()
X = np.random.randint(1, 7, (1000000, 10))
Y = np.sum(X, axis=1)
# plt.hist(Y, density=True, bins=np.linspace(5, 65, 25), ec="black")
plt.hist(Y, density=True, ec="black")
time.time() - start_time
plt.show()












# 2.4.5: Random Walks

# This is a good point to introduce random walks.
# Random walks have many uses.
# They can be used to model random movements of molecules,
# but they can also be used to model spatial trajectories of people,
# the kind we might be able to measure using GPS or similar technologies.
# There are many different kinds of random walks, and properties of random walks
# are central to many areas in physics and mathematics.

import numpy as np
import matplotlib.pyplot as plt

# random displacements
delta_X = np.random.normal(0, 1, (2, 10000))
plt.plot(delta_X[0], delta_X[1], 'go')
plt.show()

# cumulative sum
# We're going to be using axis equals 1 because we
# would like to take the cumulative sum over the columns of this array.
# position a any time
X = np.cumsum(delta_X, axis=1)
plt.plot(X[0], X[1], "ro-")
plt.show()
# X_0 = np.array([[0] + list(np.cumsum(delta_X[0])), [0] + list(np.cumsum(delta_X[1]))])
X_0 = np.array([[0], [0]])
X_with0 = np.concatenate((X_0, X), axis=1)
plt.plot(X_with0[0], X_with0[1], "ro-")
plt.show()


# How are the displacements, the individual steps, in the random walk related to one another?
# Any two consecutive displacements are independent of one another.

# What does np.concatenate do?
help(np.concatenate)
# Takes an iterable of np.arrays as arguments, and binds them along the axis argument.
















# Week 2: Python Libraries and Concepts Used in Research/Homework: Week 2

import numpy as np

def create_board():
    return np.zeros((3, 3), dtype=int)

create_board()




def place(board, player, position):
    if (player in (1, 2) and 
        board[position] == 0):
        board[position] = player
    return board

place(create_board(), 1, (0, 0))




def possibilities(board):
    return [(i, j) for i in range(board.shape[0]) for j in range(board.shape[1]) if board[(i, j)] == 0]
    # return np.where(board == 0, board, 100)

place_1 = place(create_board(), 1, (0, 0))
possibilities(place_1)
# possibilities(create_board())




import random

random.seed(1)

def random_place(board, player):
    choice = random.choice(possibilities(board))
    return place(board, player, choice)

place_1 = place(create_board(), 1, (0, 0))
random_place(place_1, 2)
# random_place(create_board(), 1)




def play_random_game(board):
    random.seed(1)
    for _ in range(3):
        for player in [1, 2]:
            random_place(board, player)
    return board

play_random_game(create_board())




import numpy as np

def row_win(board, player):
    # return np.any([np.all([True if (0, j) == player else False for j in range(3)]) for i in range(3)])
    # return np.any([np.all([board[(i, j)] == player for j in range(3)]) for i in range(3)])
    # return any([all([board[(i, j)] == player for i in range(3)]) for j in range(3)])
    return np.any(np.all(board == player, axis=1)) # this checks if any row contains all positions equal to player.
     

row_win(play_random_game(create_board()), 1)




def col_win(board, player):
    return np.any(np.all(board == player, axis=0))
     
row_win(play_random_game(create_board()), 1)




board = play_random_game(create_board())
# board[1,1] = 2

def diag_win(board, player):
    return (np.all(board.diagonal() == player) or
            np.all(np.fliplr(board).diagonal() == player))

diag_win(board, 2)




def evaluate(board):
    winner = 0
    for player in [1, 2]:
        if (row_win(board, player) or 
            col_win(board, player) or 
            diag_win(board, player)):
            winner = player
    if np.all(board != 0) and winner == 0:
        winner = -1
    return winner

evaluate(board)




def evaluate2(board):
    if (row_win(board, 1) or
        col_win(board, 1) or
        diag_win(board, 1)):
        return 1
    elif (row_win(board, 2) or
          col_win(board, 2) or
          diag_win(board, 2)):
        return 2
    elif np.any(board == 0):
        return 0
    else:
        return -1

evaluate2(board)





random.seed(1)
def play_game(R):
    result = []
    for _ in range(R):
        board = create_board()
        for player in [1, 2, 1, 2, 1, 2, 1, 2, 1]:
            random_place(board, player)
            eval = evaluate(board)
            if eval in [1, 2]:
                break
        result.append(eval)
    return result

results = play_game(1000)
results.count(1)

random.seed(1)
def play_game2():
    board = create_board()
    winner = 0
    while winner == 0:
        for player in [1, 2]:
            random_place(board, player)
            winner = evaluate(board)
            if winner != 0:
                break
    return winner

results = [play_game2() for i in range(1000)]
results.count(1)




random.seed(1)

def play_strategic_game(R):
    result = []
    for _ in range(R):
        board = create_board()
        board[1, 1] = 1
        for player in [2, 1, 2, 1, 2, 1, 2, 1]:
            random_place(board, player)
            eval = evaluate(board)
            if eval in [1, 2]:
                break
        result.append(eval)
    return result

strategic_results = play_strategic_game(1000)
strategic_results.count(1)

random.seed(1)
def play_strategic_game2():
    board, winner = create_board(), 0
    board[1, 1] = 1
    while winner == 0:
        for player in [2,1]:
            random_place(board, player)
            winner = evaluate(board)
            if winner != 0:
                break
    return winner

results = [play_strategic_game2() for i in range(1000)]
results.count(1)

























# 3.1.1: Introduction to DNA Translation

"""
Life depends on the ability of cells to store, retrieve, and translate
genetic instructions.
These instructions are needed to make and maintain living organisms.
For a long time, it was not clear what molecules
were able to copy and transmit genetic information.
We now know that this information is carried by the deoxyribonucleic acid
or DNA in all living things.
DNA is a discrete code physically present
in almost every cell of an organism.
We can think of DNA as a one dimensional string of characters
with four characters to choose from.
These characters are A, C, G, and T. They
stand for the first letters with the four nucleotides used to construct DNA.
The full names of these nucleotides are adenine, cytosine, guanine,
and thymine.
Each unique three character sequence of nucleotides,
sometimes called a nucleotide triplet, corresponds to one amino acid.
The sequence of amino acids is unique for each type of protein
and all proteins are built from the same set of just 20 amino acids
for all living things.
Protein molecules dominate the behavior of the cell
serving as structural supports, chemical catalysts, molecular motors, and so on.
The so called central dogma of molecular biology
describes the flow of genetic information in a biological system.
Instructions in the DNA are first transcribed into RNA
and the RNA is then translated into proteins.
We can think of DNA, when read as sequences of three letters,
as a dictionary of life.
"""

# What is the central dogma of molecular biology that describes the basic flow of genetic information?
# DNA -> RNA -> Protein










# 3.1.2: Downloading DNA Data

# ncbi nuceleotide NM_207618.2
# DNA sequance dna.txt nad protein sequence protein.txt




# 3.1.3: Importing DNA Data Into Python

import os
os.getcwd()
os.chdir('/home/ukasz/Documents/IT/Python/')
path = '/edx/translation/'

# read one line at a time
for line in open(os.getcwd()+path+'dna.txt'):
    print(line.strip())

f = open(os.getcwd()+path+'dna.txt')
seq = f.read()
# print with \n
seq
# print without \n
print(seq)
f.close()

# remove \n linebreaks
seq = seq.replace('\n', '').replace('\r', '')



seq
print(seq)

# delete '\n'
''.join((''.join(seq.split('\r'))).split('\n'))
seq.replace('\n', '').replace('\r', '')
dir(str)
seq = seq.replace('\n', '').replace('\r', '')






# 3.1.4: Translating the DNA Sequence

# We've created a dictionary called table where
# the keys are strings corresponding to codons or nucleotide triples.
# And their values are also strings which correspond to common one-letter symbols
# used for the different amino acids.

import os
os.getcwd()
os.chdir('/home/ukasz/Documents/IT/Python/')

table2 = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

type(table2)

table2['ATA']

len(seq) % 3



path = '/edx/translation/'

import json
with open(os.getcwd()+path+'table.json', 'r') as F2:
    table = F2.read()

type(table)
table_js = json.loads(table)

table_js['ATA']
type(table_js)

def DNA_to_aminoacid(seq):
    """
    Translate a string containing a nucleotide sequence into a string containing the corresponding sequence of amino acids. 
    Nucleotides are translated in triplets using the table dictionary; each amino acid 4 is encoded with a string of length 1.
    """
    
    protein = ''
    for i in range(0, len(seq), 3):
        # protein += table_js[seq[i:i+3]]
        protein += table2[seq[i:i+3]]
    return protein


DNA_to_aminoacid('ATAGGCAAA')



    




# 3.1.5: Comparing Your Translation

def read_seq(input):
    """Reads and removes special characters from input file"""

    with open(input, 'r') as f:
        seq = f.read()
    seq = seq.replace('\n', '').replace('\r', '')
    return seq

read_seq(os.getcwd()+path+'dna.txt')
prt = read_seq(os.getcwd()+path+'protein.txt')
dna = read_seq(os.getcwd()+path+'dna.txt')


help(DNA_to_aminoacid)
DNA_to_aminoacid('ATAGGCAAA')
DNA_to_aminoacid('atgtctact'.upper())
DNA_to_aminoacid(seq)
DNA_to_aminoacid(seq[:len(seq) - 2])

# you will see two numbers next to it, 21 and 938.
# These are the locations of the gene where
# the coding sequence starts and ends.
DNA_to_aminoacid(seq[20:938])
prt
# At the very end of a protein coding sequence,
# nature places what's called a stop codon.
# There are three stop codons, and their function
# is to tell someone reading the sequence that this
# is where you should stop reading.
# It's almost like an end of paragraph sign.
DNA_to_aminoacid(seq[20:935]) == prt 
DNA_to_aminoacid(seq[20:938])[:-1] == prt 
DNA_to_aminoacid(seq[20:23])

seq[935:938]
DNA_to_aminoacid(seq[935:938])
seq[68:71]


def DNA_to_aminoacid2(seq):
    # return ''.join([table2[seq[i:i+3]] for i in range(0, len(seq), 3)])
    return ''.join([table_js[seq[i:i+3]] for i in range(0, len(seq), 3)])

DNA_to_aminoacid2('ATAGGCAAA')
DNA_to_aminoacid2(seq[:len(seq)-2])
























# Week 3: Case Studies Part 1/Homework: Case Study 1
import string

alphabet = ' ' + string.ascii_lowercase
positions = {alphabet[i]:i for i in range(len(alphabet))}

message = "hi my name is caesar"
encoded_message = ""
key = 3

def encode(message, key):
    return ''.join([alphabet[((alphabet.index(letter)+key) % 27)] for letter in message])
    # return ''.join([alphabet[(positions[letter]+key) % 27] for letter in message])

encoded_message = encode(message, key)
encoded_message


















# Week 3: Case Studies Part 1/Case Study 2: Language Processing

# 3.2.1: Introduction to Language Processing

# Project Gutenberg is the oldest digital library of books.




# 3.2.2: Counting Words

text = 'Some sample of some text. Another sample.'
text = "This comprehension check is to check for comprehension."

def count_words(text):
    '''Count the number of words provied string. Skip punctuation.'''

    text = text.lower()
    for punctuation in ['.', ',', ';', ':', '"', "'", '\n', '!', '?', '(', ')']:
        text = text.replace(punctuation, '')

    word_counts = {}
    for word in text.split():
        word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts

count_words(text)


from collections import Counter

def count_words_fast(text):

    '''Count the number of words provied string. Skip punctuation. With from collections import Counter'''

    text = text.lower()
    for punctuation in ['.', ',', ';', ':', '"', "'", '\n', '!', '?', '(', ')']:
        text = text.replace(punctuation, '')
    return Counter(text.split())

count_words_fast(text)

count_words(text) == count_words_fast(text)








# 3.2.3: Reading in a Book

# Character encoding refers to the process how
# computer encodes certain characters.
# In this case, we'll use what is called UTF-8 encoding, which
# is the dominant character encoding for the web.

import os
os.getcwd()
# os.chdir('/home/ukasz/Documents/IT/Python')
path = '/edx/Language_Processing/Books/English/shakespeare'


def read_book(title):
    """
    Read a book from a file and return it as a string.
    """

    with open(title, 'r', encoding='utf8') as current_file:
        text = current_file.read()
        text = text.replace('\n', '').replace('\r', '')
    return text

text = read_book(os.getcwd()+path+'/'+'Romeo and Juliet.txt')

len(text)
text.find("What's in a name?")
text[42757:42757+1000]


# Open url in browser.
import webbrowser
webbrowser.open("https://www.gutenberg.org/cache/epub/2261/pg2261.txt")


import urllib.request
# from https://www.kite.com/python/answers/how-to-read-a-text-file-from-a-url-in-python
url = "https://www.gutenberg.org/cache/epub/2261/pg2261.txt"
file = urllib.request.urlopen(url)

for line in file:
	decoded_line = line.decode("utf-8")
	print(decoded_line)


def read_book_url(title):
    """
    Read a book from url and return it as a string.
    """
    decoded_line = ''
    for line in urllib.request.urlopen(title):
        decoded_line += line.decode("utf8")

    text = decoded_line
    text = text.replace("\n", "").replace("\r", "")
    return text

read_book_url("https://www.gutenberg.org/cache/epub/2261/pg2261.txt")


# urllib.request.urlopen doesn't work with with
import urllib.request
def read_book_url2(title):
    """
    Read a book from url and return it as a string.
    """

    with urllib.request.urlopen(title) as decoded_line:
        text = decoded_line("utf8")
        text = text.replace("\n", "").replace("\r", "")
    return text

read_book_url2("https://www.gutenberg.org/cache/epub/2261/pg2261.txt")

len(read_book(os.getcwd()+path+'/'+'Romeo and Juliet.txt'))
len(read_book_url("https://www.gutenberg.org/cache/epub/2261/pg2261.txt"))
read_book(os.getcwd()+path+'/'+'Romeo and Juliet.txt')[0]
read_book_url("https://www.gutenberg.org/cache/epub/2261/pg2261.txt")[1]

romeo_book = read_book_url("https://www.gutenberg.org/cache/epub/2261/pg2261.txt")[1:]
ind = romeo_book.find("By any other")
romeo_book[ind - 50:ind + 10]









# 3.2.4: Computing Word Frequency Statistics

def word_stats(word_counts):
    """Returns number of unique words and word frequencies."""

    num_uniq = len(word_counts)
    counts = word_counts.values()
    return (num_uniq, counts)

word_count = count_words_fast(text)
word_stats(word_count)
num_uniq, counts = word_stats(word_count)
# how many words in total
sum(counts)


romeo_book_de = read_book_url("https://www.gutenberg.org/cache/epub/6996/pg6996.txt")[1:]
num_uniq_de, counts_de = word_stats(count_words_fast(romeo_book_de))
romeo_book_pl = read_book_url("https://www.gutenberg.org/files/27062/27062-0.txt")[1:]
num_uniq_pl, counts_pl = word_stats(count_words_fast(romeo_book_pl))









# 3.2.5: Reading Multiple Files

import os
os.getcwd()
# os.chdir('/home/ukasz/Documents/IT/Python/')
path = '/edx/Language_Processing/Books'
os.listdir(os.getcwd()+path)

for language in os.listdir(os.getcwd()+path):
    for author in os.listdir(os.getcwd()+path+'/'+language):
        for title in os.listdir(os.getcwd()+path+'/'+language+'/'+author):
            input_file = os.getcwd()+path+'/'+language+'/'+author+'/'+title
            print(input_file)
            text = read_book(input_file)
            (num_uniq, counts) = word_stats(count_words_fast(text))


# pandas is a library that provides additional data structures and data
# analysis functionalities for Python.
# It's especially useful for manipulating numerical tables and time series data.
# pandas gets its name from the term panel data
# used to refer to multi-dimensional structured data sets.

import pandas as pd

# package / libraray version
pd.__version__


table = pd.DataFrame(columns=('Name', 'Age')) # create an empty table
table.loc[1] = 'James', 25 # add elements
table.loc[2] = 'Jess', 22
table2 = pd.DataFrame([['Tom', 40]], columns=('Name', 'Age'))
table.append(table2)
table
table.columns # column names
table.Name  # select column
table.Name
table[['Name', 'Age']]


df = pd.DataFrame([[1, 2], [3, 4]],
                  columns=('A', 'B'),
                  index=(0, 1))  # create table with data
df.loc[2] = 5, 6
df.append(pd.DataFrame([[7, 8]], columns=('A', 'B')))
df2 = pd.DataFrame([[7, 8]], columns=('A', 'B'))
df.append(df2)
df.append(df2, ignore_index=True)


stats = pd.DataFrame(columns=('language', 'author', 'title', 'length', 'unique'))
title_num = 1

import os
os.getcwd()
# os.chdir('/home/ukasz/Documents/IT/Python')
path = '/edx/Language_Processing/Books'
os.listdir(os.getcwd()+path)

for language in os.listdir(os.getcwd()+path):
    for author in os.listdir(os.getcwd()+path+'/'+language):
        for title in os.listdir(os.getcwd()+path+'/'+language+'/'+author):
            input_file = os.getcwd()+path+'/'+language+'/'+author+'/'+title
            print(input_file)
            text = read_book(input_file)
            (num_uniq, counts) = word_stats(count_words_fast(text))
            stats.loc[title_num] = language, author.capitalize(), title[:-4], sum(counts), num_uniq
            title_num += 1

stats
stats.head()
stats.tail()








# 3.2.6: Plotting Book Statistics

stats.length
stats['length']
stats.unique

import matplotlib.pyplot as plt

plt.plot(stats.length, stats.unique, 'bo')
plt.loglog(stats.length, stats.unique, 'bo')
plt.show()
stats[stats.language == 'English']

# plt.figure(figsize=(10, 10))
subset = stats[stats.language == 'English']
plt.loglog(subset.length, subset.unique, 'o', label='English', color='crimson')
subset = stats[stats.language == 'French']
plt.loglog(subset.length, subset.unique, 'o', label='French', color='forestgreen')
subset = stats[stats.language == 'German']
plt.loglog(subset.length, subset.unique, 'o', label='German', color='orange')
subset = stats[stats.language == 'Portuguese']
plt.loglog(subset.length, subset.unique, 'o', label='Portuguese', color='blueviolet')
plt.legend(loc='lower right')
plt.xlabel('Book length')
plt.ylabel('Number of unique words')
plt.savefig(os.getcwd()+'/edx/Language_Processing'+'/'+'book_statistics_log.png')
plt.show()

with open(path+'books_statistics.csv', 'w+') as F2:
    F2.write(stats.to_csv(index=False))


F2 = open(path+'books_statistics.csv', 'w+')
F2.write(stats.to_csv(index=False))
F2.close()

















# Course/Week 3: Case Studies Part 1/Homework: Case Study 2

import os
import pandas as pd
import numpy as np
from collections import Counter

def count_words_fast(text):
    text = text.lower()
    skips = [".", ",", ";", ":", "'", '"', "\n", "!", "?", "(", ")"]
    for ch in skips:
        text = text.replace(ch, "")
    word_counts = Counter(text.split(" "))
    return word_counts

def word_stats(word_counts):
    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique, counts)




# path = '/home/ukasz/Documents/IT/Python/edx/Language_Processing/hamlets.csv'
# hamlets = pd.read_csv(path, index_col=0)
hamlets = pd.read_csv('https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@hamlets.csv', index_col=0)
hamlets




language, text = hamlets.iloc[0]

# create DataFrame from Counter or dict
data = pd.DataFrame(dict(count_words_fast(text)).items(), columns=('word', 'count'))
# data = pd.DataFrame({
#    'word': count_words_fast(text).keys(),
#    'count': count_words_fast(text).values()
#})

data




# data["length"] = list(map(len, data["word"]))
# data["length"] = data["word"].apply(lambda x: len(x))

# create data in column based on another column data
data['length'] = data['word'].apply(len)
# below doesn't work
# data.length = len(data.word)

# data['frequency'] = data['count'].apply(lambda x: 'unique' if x == 1 else ('frequent' if x > 10 else 'infrequent'))
# create conditional data in column
data.loc[data['count'] > 10, 'frequency'] = 'frequent'
data.loc[data['count'] <= 10, 'frequency'] = 'infrequent'
data.loc[data['count'] == 1, 'frequency'] = 'unique'


data
# len(data[data.frequency == 'unique'])




sub_data = pd.DataFrame({
    "language": language,
    "frequency": ('frequent', 'infrequent', 'unique'),
    # "mean_word_length": data.groupby(by='frequency')['length'].mean(),
    "mean_word_length": data.groupby('frequency').length.mean(),
    #"num_words": data.groupby('frequency')['word'].count()
    "num_words": data.groupby(by='frequency').size()
})

print(sub_data)




def summarize_text(language, text):
    counted_text = count_words_fast(text)

    data = pd.DataFrame({
        "word": list(counted_text.keys()),
        "count": list(counted_text.values())
    })
    
    data.loc[data['count'] > 10,  'frequency'] = 'frequent'
    data.loc[data['count'] <= 10, 'frequency'] = 'infrequent'
    data.loc[data['count'] == 1,  'frequency'] = 'unique'
    
    data['length'] = data['word'].apply(len)
    
    sub_data = pd.DataFrame({
        "language": language,
        "frequency": ['frequent', 'infrequent', 'unique'],
        "mean_word_length": data.groupby('frequency').length.mean(),
        "num_words": data.groupby(by = 'frequency').size()
    })
    
    return(sub_data)

# print(summarize_text(language, text))


summarize_text(language, text)

def grouped_data_fun(hamlets):
    sum_data = pd.DataFrame(columns=('language', 'frequency', 'mean_word_length', 'num_words'))
    for i in range(3):
        (language, text) = hamlets.iloc[i]
        # print(summarize_text(language, text))
        sum_data = sum_data.append(summarize_text(language, text))
    return sum_data

print(grouped_data_fun(hamlets))

grouped_data = grouped_data_fun(hamlets)




import matplotlib.pyplot as plt

colors = {"Portuguese": "green", "English": "blue", "German": "red"}
markers = {"frequent": "o", "infrequent": "s", "unique": "^"}

for i in range(grouped_data.shape[0]):
    row = grouped_data.iloc[i]
    plt.plot(row.mean_word_length, row.num_words,
        marker = markers[row.frequency],
        color = colors[row.language],
        markersize = 10
    )

color_legend = []
marker_legend = []
for color in colors:
    color_legend.append(
        plt.plot([], [],
        color=colors[color],
        marker='o',
        label=color,
        markersize=10,
        linestyle='None')
    )
for marker in markers:
    marker_legend.append(
        plt.plot([], [],
        color='k',
        marker=markers[marker],
        label=marker,
        markersize=10, 
        linestyle='None')
    )

plt.legend(numpoints=1, loc='upper left')

plt.xlabel('Mean Word Length')
plt.ylabel('Number of Words')
# plt.savefig(os.getcwd()+'/'+'edx/Language_Processing/grouped_data.png')
plt.show()



















# Week 3: Case Studies Part 1/Case Study 3: Introduction to Classification

# 3.3.1: Introduction to kNN Classification


# Statistical learning refers to a collection
# of mathematical and computation tools to understand data.
# In what is often called supervised learning,
# the goal is to estimate or predict an output based on one or more inputs.
# The inputs have many names, like predictors, independent variables,
# features, and variables being called common.
# The output or outputs are often called response variables,
# or dependent variables.
# If the response is quantitative(ilośćiowy)-- say, a number that measures weight or height,
# we call these problems regression problems.
# If the response is qualitative(jakiściowy)-- say, yes or no, or blue or green,
# we call these problems classification problems.
# This case study deals with one specific approach to classification.
# The goal is to set up a classifier such that when
# it's presented with a new observation whose category is not known,
# it will attempt to assign that observation to a category, or a class,
# based on the observations for which it does know the true category.
# This specific method is known as the k-Nearest Neighbors classifier,
# or kNN for short.
# Given a positive integer k, say 5, and a new data point,
# it first identifies those k points in the data that are nearest to the point
# and classifies the new data point as belonging to the most common class
# among those k neighbors.

# Let's first set up a co-ordinate system.
# We have measured two different features and data.
# Let's call this feature x1, which could be someone's weight,
# and this feature is x2, which could be someone's height.
# We're given a set of points.
# Some of them are blue, like these points over here,
# and another set of points that are red, like these points over here.
# The color of the point tells us the category to which
# that particular observation belongs.
# The goal behind classification is the following:
# Imagine we get a new data point, say, the black point over here,
# whose class is not known.
# Should we, as scientists, point to the blue category or the red category?
# What kNN does is the following:
# It first identifies some of the closest neighbors around it,
# and it assigns the black point to the category
# to which a majority of the points around it belong.
# In this case, we have three points.
# They're all blue.
# And this point gets assigned blue.

# How does the k-Nearest Neighbors classifier classify observations?
# According to the most common class among the nearest k neighbors






# 3.3.2: Finding the Distance Between Two Points

import numpy as np

def distance(p1, p2):
    """Find the distance between two points."""
    
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))

p1 = np.array([1, 1])
p2 = np.array([4, 4])
distance(p1, p2)








# 3.3.3: Majority Vote


# For building our KNN classifier, we need to be
# able to compute, what is sometimes called, majority vote.
# This means that given an array or a sequence of votes,
# which could be say numbers 1, 2, and 3, we
# need to determine how many times each occurs,
# and then find the most common element.
# So for example, if we had two 1s, three 2s, and one 3,
# the majority vote would be 2.
# Note that while we need to count the number of times each vote occurs
# in the sequence, we will not be returning
# the counts themselves, but instead the observation
# corresponding to the highest count.

from collections import Counter
import random

dict(Counter(votes))

# def majority_vote(votes):
#     """Return most common element."""
# 
#     winners = []
#     vote_dict = dict(Counter(votes))
#     for vote, count in vote_dict.items():
#         if count == max(vote_dict.values()):
#             winners.append(vote)
#     return random.choice(winners)

def majority_vote(votes):
    """Return most common element"""

    vote_dict = dict(Counter(votes))
    return random.choice([vote for vote, count in vote_dict.items() if count == max(vote_dict.values())])

votes = [1, 1, 2, 2, 2, 3, 3, 3]
winner = majority_vote(votes)

# As you may remember, the most commonly occurring element in a sequence
# is called mode in statistics.

import scipy.stats as ss
def majority_vote_short(votes):
    """Return most the common element in votes using mode."""

    mode, _ = ss.mstats.mode(np.array(votes))
    return int(mode)

majority_vote_mode(votes)







# 3.3.4: Finding Nearest Neighbors

import numpy as np

points = np.array([[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]])
p = np.array([2.5, 2])

import matplotlib.pyplot as plt
plt.plot(points[:, 0], points[:, 1], 'ro')
plt.plot(p[0], p[1], 'bo')
# plt.axis(0, 3.5, 0, 3.5)
plt.show()

for point in np.array(list([p]) + list(points)):
    plt.plot(point[0], point[1], 'bo')
plt.show()



# def find_nearest_neighbors(p, points, k=5):
#     """Find the k nearest neighbors of point p and return their indices."""
# 
#     distances = np.zeros(points.shape[0])
#     for i in range(len(distances)):
#         distances[i] = distance(p, points[i])
#     return np.argsort(distances)[:k]

def find_nearest_neighbors(p, points, k=5):
    """Find k nearest neighbors"""

    # np.argsort returns to indices that would sort the given array.
    # sort by index
    return np.argsort([distance(p, point) for point in points])[:k]

ind = find_nearest_neighbors(p, points, 2)
points[ind]


# We'll call that function knn predict, and it
# takes in three parameters-- p, the new point we'd like to classify,
# points, our existing data or our training data, and the value for k.

# So far what we have is the code that enables
# us to find those points, those k points that are nearest to p.
# However, to be able to make a class prediction,
# we also need to know the classes to which these k points belong to.
def knn_predict(p, points, outcomes, k=5):
    ind = find_nearest_neighbors(p, points, k)
    return majority_vote(outcomes[ind])

# these are the classes of the points that are
# specified in the input variable points.
# class labels, outcomes
outcomes = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])

knn_predict(np.array([2.5, 2.7]), points, outcomes, 2)
knn_predict(np.array([1.0, 2.7]), points, outcomes, 2)







# 3.3.5: Generating Synthetic Data

# We're going to write a function that generates two end data points, where
# the first end points are from class 0, and the second end points
# are from class 1.
# These data are known as synthetic data because we
# will be generating these data with the help of the computer.
# In this case, we'll generate predictors from two
# bivariate normal distributions, where the first distribution gives rise
# to observations belonging to class 0, and the second gives rise
# to observations belonging to class 1.
# The word, bivariate, just means 2 variables, like x and y.
# If it were generating say, just the x variables, then we'd
# be dealing with univariate data.

import scipy.stats as ss
ss.norm(0, 1).rvs((5, 2))

import numpy as np
np.random.normal(0, 1, (5, 2))

# concatenate arrays as lists
np.array(list(np.random.normal(size=(5, 2))) + list(np.random.normal(1, 1, size=(5, 2))))
np.repeat((0, 1), 5)
# And then we need to specify the axis of concatenation, which is in this case, equal to 0.
# That's because we're concatenating along the rows of these arrays.

def generate_synth_data(n=50):
    """Generate two sets of points from bivariate normal distributions."""

    points = np.concatenate((np.random.normal(size=(n, 2)), np.random.normal(1, 1, (n, 2))), axis=0)
    outcomes = np.concatenate((np.zeros(n, dtype=int), np.ones(n, dtype=int)))
    return(points, outcomes)

points, outcomes = generate_synth_data(20)

import matplotlib.pyplot as plt
plt.plot(points[:20, 0], points[:20, 1], 'ro')
plt.plot(points[20:, 0], points[20:, 1], 'bo')
plt.show()







# 3.3.6: Making a Prediction Grid

# Our next task is to plot a prediction grid.
# This means that once we've observed our data,
# we can examine some part of the predictor space
# and compute the class prediction for each point in the grid using the knn
# classifier.
# So instead of finding out how our classifier might classify a given
# point, we can ask how it classifies all points that
# belong to a rectangular region of the predictor space.

import numpy as np

def make_prediction_grid(predictors, outcomes, limits, h, k):
    """Classify each point on the prediction grid."""

    (x_min, x_max, y_min, y_max) = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    # Meshgrid generates two 2-dimensional arrays.
    # it returns matrices, the first containing the x values for each grid point and 
    # the second containing the y values for each grid point.
    xx, yy = np.meshgrid(xs, ys)

    prediction_grid = np.zeros(xx.shape, dtype=int)
    # We'll then be looping over all of our x values, all of our y values,
    # asking our knn predict algorithm or a function to predict the class label
    # 0 or 1 for each point.
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x, y])
            prediction_grid[j, i] = knn_predict(p, predictors, outcomes, k)
    return (xx, yy, prediction_grid)

# np.meshgrid(np.array([6, 2]), np.array([3, 9, 5, 6]))

xvalues = np.array([0, 1, 2]);
yvalues = np.array([2, 3, 4]);
np.meshgrid(xvalues, yvalues)
xx, yy = np.meshgrid(xvalues, yvalues)


# What does np.arange do?
# Creates regularly spaced values between the first and second argument, 
# with spacing given in the third argument

# What does enumerate do?
# Takes an iterable and returns a new iterable with tuples as elements, 
# where the first index of each tuple is the index of the tuple in the iterable.






# 3.3.7: Plotting the Prediction Grid

import numpy as np
import math
from collections import Counter
import random
import matplotlib.pyplot as plt
import os

def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """Plot KNN predictions for every point on the grid."""

    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap(['hotpink', 'lightskyblue', 'yellowgreen'])
    observation_colormap = ListedColormap(['red', 'blue', 'green'])
    # plt.figure(figsize =(10, 10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap=background_colormap, alpha=0.5, shading='auto')
    plt.scatter(predictors[:, 0], predictors [:, 1], c=outcomes, cmap=observation_colormap, s=50)
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.xticks(())
    plt.yticks(())
    plt.xlim(np.min(xx), np.max(xx))
    plt.ylim(np.min(yy), np.max(yy))
    plt.savefig(os.getcwd()+'/'+'edx/knn/'+filename)
    plt.show()


predictors, outcomes = generate_synth_data()
predictors.shape
outcomes.shape

k = 50
filename = 'knn_synth_50.png'
limits = (-3, 4, -3, 4)
h = 0.1
(xx, yy, prediction_grid) = make_prediction_grid(
    predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)

# Looking at the plot here for k equals 50,
# we can see that the decision boundary is pretty smooth.
# In contrast, if you look at the plot on the right, where k is equal to 5,
# you'll see that the shape of the decision boundary is more complicated.
# It seems that you might be able to find a value of k that maximizes
# the accuracy of the predictions.
# But that's somewhat short sighted.
# This is because what you really care about is not
# how well your method performs on the training data set,
# the data set we've used so far.
# But rather how well it performs on a future dataset you haven't yet seen.
# It turns out that using a value for k that's too large or too small
# is not optimal.
# A phenomenon that is known as the bias-variance tradeoff.
# This suggests that some intermediate values of k might be best.
# We will not talk more about it here,
# but for this application, using k equal to 5 is a reasonable choice.








# 3.3.8: Applying the kNN Method

# SciKitLearn is an open source machine learning library for Python.

# We'll be applying both the SciKitLearn and our homemade classifier
# to a classic data set created by Ron Fisher in 1933.
# It consists of 150 different iris flowers.
# 50 from each of three different species.
# For each flower, we have the following covariates: sepal length, sepal width,
# petal length, and petal width.

from sklearn import datasets
import sklearn.datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
iris['data']
iris.data
predictors = iris.data[:, :2]
# outcomes vector
outcomes = iris.target

plt.plot(predictors[outcomes == 0][:, 0], predictors[outcomes == 0][:, 1], "ro")
plt.plot(predictors[outcomes == 1][:, 0], predictors[outcomes == 1][:, 1], "go")
plt.plot(predictors[outcomes == 2][:, 0], predictors[outcomes == 2][:, 1], "bo")
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()

# prediction grid plot
k = 5; filename = "iris_grid.png"; limits = (4, 8, 1.5, 4.5); h = 0.1
xx, yy, prediction_grid = make_prediction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn
# And we'll fit our model using the fit method.
knn.fit(predictors, outcomes)
# Finally, we can do the predictions.
sk_predictions = knn.predict(predictors)
sk_predictions.shape

# Now these are the predictions to our own homemade knn algorithm did.
my_predictions = np.array([knn_predict(p, predictors, outcomes, 5) for p in predictors])
# how often do the sk_predictions predictions agree with my predictions.
np.mean(sk_predictions == my_predictions)

# We can also ask how frequently do my predictions
# and SciKit predictions agree with the actual observed outcomes.
np.mean(sk_predictions == outcomes)
np.mean(my_predictions == outcomes)
# using SciKit, the actual observed outcomes for the data
# points that we have observed, agree with the predictions of the SciKit library
# 83% of the time.
# In this case, our homemade predicter is actually somewhat better.
# We're correct approximately 85% of the time.

# How often do the predictions from the homemade and scikit-learn kNN classifiers 
# accurately predict the class of the data in the iris dataset described in Video 3.3.8?
# 85%





# Week 3: Case Studies Part 1/Homework: Case Study 3
Repaet thit case studies. knn





















# Week 4: Case Studies Part 2/Case Study 4: Classifying Whiskies

# 4.1.1: Getting Started with Pandas

# Pandas is a Python library that provides data structures and functions
# for working with structured data, primarily tabular data.
# Pandas is built on top of NumPy and some familiarity with NumPy
# makes Pandas easier to use and understand.

# Series is a one-dimensional array-like object,
# and Data Frame is a two-dimensional array-like object.

import pandas as pd

pd.Series([6, 3, 8, 6])
x = pd.Series([6, 3, 8, 6], index=['q', 'w', 'e', 'r'])
x['q']
x.q
x[['q', 'w']]
x.index
x.reindex(sorted(x.index))

age = {"Tim": 29, "Jim": 31, "Pam": 27, "Sam": 35}
x = pd.Series(age)



data = {"name": ['Tim', 'Jim', 'Pam', 'Sam'],
        "age": [29, 31, 27, 35],
        "Zip": ['02115', '02130', '67700', '00100']
        }
x = pd.DataFrame(data, columns=['age', 'Zip'], index=data['name'])

# We can retrieve a column by using dictionary-like notation
# or we can specify the name of the column as an attribute of the Data Frame.
x['age']
x.age
x.index.tolist()
sorted(x.index)
x.reindex(sorted(x.index))


# Series and Data Frame objects support arithmetic operations like addition.
x = pd.Series([6, 3, 8, 6], index=['q', 'w', 'e', 'r'])
y = pd.Series([7, 3, 5, 2], index=['e', 'q', 'r', 't'])
x + y
# NaN not a number










# 4.1.2: Loading and Inspecting Data

import pandas as pd
import numpy as np
import os

os.getcwd()
path = '/edx/whiskies'
# os.chdir('/home/ukasz/Documents/IT/Python'+'/edx/whiskies')


# whisky = pd.read_csv(os.getcwd()+path+'/'+'whiskies.txt', index_col=0)
whisky = pd.read_csv(os.getcwd()+path+'/'+'whiskies.txt')
whisky['Region'] = pd.read_csv(os.getcwd()+path+'/'+'regions.txt')
whisky.head()
whisky.tail()
whisky.iloc[5:10, :5]
whisky.columns

# Extract the flavors columns
flavors = whisky.iloc[:, 2:14]








# 4.1.3: Exploring Correlations


# Let's find out about correlations of different flavor attributes.
# In other words, we'd like to learn whether whiskies
# that score high on, say, sweetness also score high on the honey attribute.
# We'll be using the core function to compute correlations
# across the columns of a data frame.
# There are many different kinds of correlations,
# and by default, the function uses what is
# called Pearson correlation which estimates
# linear correlations in the data.
# In other words, if you have measured attributes for two variables,
# let's call them x and y the Pearson correlation coefficient
# between x and y approaches plus 1 as the points in the xy scatterplot approach
# a straight upward line.
# But what is the interpretation of a correlation
# coefficient in this specific context?
# A large positive correlation coefficient indicates
# that the two flavor attributes in question
# tend to either increase or decrease together.
# In other words, if one of them has a high score
# we would expect the other, on average, also to have a high score.


import matplotlib.pyplot as plt

corr_flavors = pd.DataFrame.corr(flavors)
# plt.figure(figsize=(10, 10))
# use the pseudo color, or P color
# function to plot the contents of the correlation matrix.
plt.pcolor(corr_flavors, cmap='jet')
plt.colorbar()
# plt.savefig(os.getcwd()+path+'/'+'corr_flavors.png')
plt.show()


# We can also look at the correlation among whiskies across flavors.
# To do this, we first need to transpose our table.
# Since these whiskies are made by different distilleries,
# we can also think of this as the correlation
# between different distilleries in terms of the flavor profiles of the whiskies
# that they produce.
corr_whisky = pd.DataFrame.corr(flavors.transpose())
plt.pcolor(corr_whisky, cmap='jet')
# I'd like to modify my plot slightly by making the axes tied so that they only
# cover the range for which I have data.
plt.axis('tight')
plt.colorbar()
# plt.savefig(os.getcwd()+path+'/'+'corr_whisky.png')
plt.show()






# 4.1.4: Clustering Whiskies By Flavor Profile

# Next we're going to cluster whiskeys based on their flavor profiles.
# We'll do this using a clustering method from the scikit-learn machine learning
# module.
# The specific method we'll be using is called spectral co-clustering.
# One way to think about spectral co-clustering method
# is to consider a list of words and a list of documents,
# which is the context in which the method was first introduced.
# We can represent the problem as a graph, where on the left we have words
# and on the right, we have documents.
# The goal is to find clusters that consist
# of sets of words and sets of documents that often go together.

# We can first represent this graph as what
# is called an adjacency matrix, where the rows correspond to words
# and the columns correspond to documents.
# Any given element of this matrix represents
# the number of times a given word appears in the given document.
# We can then take this matrix, manipulate it in certain ways,
# and find an approximate solution to the stated clustering problem,
# in terms of eigenvalues and eigenvectors of this modified matrix.
# We will not go into the details here, but the term spectral
# refers to the use of eigenvalues and eigenvectors of some matrix,
# and this is the meaning of the term spectral in spectral co-clustering.


# Since that whiskeys in the dataset come from six different regions,
# we're going to ask the clustering algorithm to find six blocks.

from sklearn.cluster._bicluster import SpectralCoclustering

model = SpectralCoclustering(n_clusters=6, random_state=0)
model.fit(corr_whisky)
model.rows_
# Each row in this array identifies a cluster, here ranging from 0 to 5,
# and each column identifies a row in the correlation matrix,
# here ranging from 0 to 85.
# If we sum all of the columns of this array,
# we can find out how many observations belong to each cluster.
np.sum(model.rows_, axis=1)
# Remember, axis 0 is rows, axis equal to 1 is columns.
# The output tells us how many whiskeys belong to a cluster 0,
# cluster 1, cluster 2, and so on.
# For example, here, 19 whiskeys belong to cluster number 2.
# If instead we sum all of the rows, we can find out how many
# clusters belong to each observation.
# Because each observation belongs in just one of the six clusters,
# the answer should be 1 for all of them.
np.sum(model.rows_, axis=0)
model.row_labels_
# Observation number 0 belongs to cluster number 5,
# observation number 1 belongs to cluster number 2, and so on.
# All of the entries in the array have to be numbers between 0 and 5
# because we specified 6 clusters.







# 4.1.5: Comparing Correlation Matrices

# Let's draw the clusters as groups that we just
# discovered in our whisky DataFrame.
# Let's also rename the indices to match the sorting.

whisky.head()
# We first extract the group labels from the model
# and append them to the whisky table.
# We also specify their index explicitly.
whisky['Group'] = pd.Series(model.row_labels_, index=whisky.index)
list(whisky.index)
# We then reorder the rows in increasing order by group labels.
# These are the group labels that we discovered
# using spectral co-clustering.
whisky = whisky.iloc[np.argsort(model.row_labels_)]
whisky = whisky.reset_index(drop=True)
# We have now reshuffled the rows and columns of the table.
# So let's also recalculate the correlation matrix.
# The reason for this step is that when we calculate the correlations,
# what pandas returns is a DataFrame.
# What I'd like to have is a NumPy array; hence the conversion.
correlations = np.array(pd.DataFrame.corr(whisky.iloc[:, 2:14].transpose()))

import matplotlib.pyplot as plt

plt.subplot(121)
plt.pcolor(corr_whisky, cmap='jet')
plt.title('Original')
plt.axis('tight')
plt.subplot(122)
plt.pcolor(correlations, cmap='jet')
plt.title('Rearranged')
plt.axis('tight')
# plt.savefig(os.getcwd()+path+'/'+'correlations.png')
plt.show()

# We asked the spectral co-clustering method
# to identify six different groups of whiskies.
# If you follow the diagonal line on the right from the bottom-left corner
# to the top-right corner, you'll be able to see visually
# those six blocks of whiskies.
# Based on this, we would expect whiskies that
# belong to the same block to be similar in their flavor
# in terms of their smokiness, in terms of their honey flavor, and so on.

import pandas as pd
data = pd.Series([1, 2, 3, 4])
data = data.iloc[[3, 0, 1, 2]]
# alters the order of appearance, but leaves the indices the same.

data = data.reset_index(drop=True)
# The 0th index of the data has been reordered to index 3 of the original, which is 4.
data[0]


from itertools import repeat
list(map(pow, range(10), 2))
list(map(pow, range(10), repeat(3)))
np.array(range(10)) ** 2

[1, 2] * 5
np.array([1, 2]) * 5
list(range(1, 5)) * 5






# Week 4: Case Studies Part 2/Homework: Case Study 4

from sklearn.cluster._bicluster import SpectralCoclustering
import numpy as np
import pandas as pd

whisky = pd.read_csv("https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@whiskies.csv", index_col=0)
whisky.info()
correlations = pd.DataFrame.corr(whisky.iloc[:, 2:14].transpose())
correlations = np.array(correlations)
correlations




# First, we import a tool to allow text to pop up on a plot when the cursor
# hovers over it.  Also, we import a data structure used to store arguments
# of what to plot in Bokeh.  Finally, we will use numpy for this section as well!

from bokeh.models import HoverTool, ColumnDataSource

# Let's plot a simple 5x5 grid of squares, alternating between two colors.
plot_values = [1, 2, 3, 4, 5]
plot_colors = ['#0173b2', '#de8f05']

# How do we tell Bokeh to plot each point in a grid?  Let's use a function that
# finds each combination of values from 1-5.
from itertools import product

grid = list(product(plot_values, plot_values))
print(grid)

# print(list(product(range(1, 6), range(1, 6))))





# The first value is the x coordinate, and the second value is the y coordinate.
# Let's store these in separate lists.
from itertools import repeat

xs, ys = zip(*grid)
print(xs)
print(ys)

print(list(repeat([1, 2], 3)))
# print(list(range(1, 6)) * 5)
list(range(1, 6)) * 5

print(sum(map(lambda x: [x] * 3, range(1, 6)), []))

list_of_lists= zip(*repeat(range(1, 6), 3))
print([val for sublist in list_of_lists for val in sublist])

list_of_lists= zip(*repeat(range(1, 6), 3))
print(list(np.array(list(list_of_lists)).flatten()))

np.repeat(range(1, 6), 3)




# Now we will make a list of colors, alternating between red and blue.

colors = [plot_colors[i%2] for i in range(len(grid))]
print(colors)




# Finally, let's determine the strength of transparency (alpha) for each point,
# where 0 is completely transparent.

alphas = np.linspace(0, 1, len(grid))

# Bokeh likes each of these to be stored in a special dataframe, called
# ColumnDataSource. Let's store our coordinates, colors, and alpha values.

source = ColumnDataSource(
    data = {
        "x": xs,
        "y": ys,
        "colors": colors,
        "alphas": alphas,
    }
)
# We are ready to make our interactive Bokeh plot!
from bokeh.plotting import figure, output_file, show

output_file('Basic_Example.html', title='Basic Example')
fig = figure(tools='hover')
fig.rect('x', 'y', 0.9, 0.9, source=source, color='colors', alpha='alphas')
hover = fig.select(dict(type=HoverTool))
hover.tooltips = {
    "Value": "@x, @y",
    }
show(fig)




regions = ["Speyside", "Highlands", "Lowlands", "Islands", "Campbelltown", "Islay"]
cluster_colors = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161']

region_colors = dict(zip(regions, cluster_colors))
group_colors = dict(zip(range(6), cluster_colors))
print(region_colors)
print(group_colors)





distilleries = list(whisky.Distillery)
correlation_colors = []
for i in range(len(distilleries)):
    for j in range(len(distilleries)):
        if correlations[i, j] < 0.7:                      # if low correlation,
            correlation_colors.append('white')         # just use white.
        else:                                          # otherwise,
            if distilleries[i] == distilleries[j]:                  # if the groups match,
                correlation_colors.append(cluster_colors[whisky.Group[i]]) # color them by their mutual group.
            else:                                      # otherwise
                correlation_colors.append('lightgray') # color them lightgray.


correlation_colors2 = ['white' if correlations[i, j] < 0.7 else cluster_colors[whisky.Group[i]] 
                            if distilleries[i] == distilleries[j] else 'lightgray' 
                                for i in range(len(distilleries)) for j in range(len(distilleries))]
correlation_colors == correlation_colors2




source = ColumnDataSource(
    data = {
        "x": np.repeat(distilleries,len(distilleries)),
        "y": list(distilleries)*len(distilleries),
        "colors": correlation_colors,
        "correlations": correlations.flatten(),
    }
)

output_file('Whisky_Correlations.html', title='Whisky Correlations')
fig = figure(title='Whisky Correlations',
             x_axis_location='above',
             x_range=list(reversed(distilleries)),
             y_range=distilleries,
             tools='hover,box_zoom,reset'
            )
fig.grid.grid_line_color = None
fig.axis.axis_line_color = None
fig.axis.major_tick_line_color = None
fig.axis.major_label_text_font_size = '5pt'
fig.xaxis.major_label_orientation = np.pi / 3
fig.rect('x', 'y', .9, .9, source=source,
     color='colors', alpha='correlations')
hover = fig.select(dict(type=HoverTool))
hover.tooltips = {
    "Whiskies": "@x, @y",
    "Correlation": "@correlations",
}
show(fig)




points = [(0,0), (1,2), (3,1)]
xs, ys = zip(*points)
colors = ['#0173b2', '#de8f05', '#029e73']

output_file("Spatial_Example.html", title="Regional Example")
location_source = ColumnDataSource(
    data={
        "x": xs,
        "y": ys,
        "colors": colors,
    }
)

fig = figure(title='Regional Example',
             x_axis_location='above',
             tools='hover, save'
            )
# fig.plot_width = 300
# fig.plot_height = 380
fig.circle('x',
           'y',
           size=10,
           source=location_source,
           color='colors',
           line_color=None
           )

hover = fig.select(dict(type = HoverTool))
hover.tooltips = {
    "Location": "(@x, @y)"
}
show(fig)





def location_plot(title, colors):
    output_file(title+'.html')
    location_source = ColumnDataSource(
        data={
            "x": whisky[' Latitude'],
            "y": whisky[' Longitude'],
            "colors": colors,
            "regions": whisky.Region,
            "distilleries": whisky.Distillery
        }
    )
    
    fig = figure(title=title,
                 x_axis_location='above',
                 tools='hover, save'
                 )
    # fig.plot_width = 400
    # fig.plot_height = 500
    fig.circle('x',
               'y',
               size=9,
               source=location_source,
               color='colors',
               line_color=None
               )
    hover = fig.select(dict(type=HoverTool))
    hover.tooltips = {
        "Distillery": "@distilleries",
        "Location": "(@x, @y)"
    }
    show(fig)

region_cols = [region_colors[i] for i in whisky.Region]
location_plot('Whisky_Locations_and_Regions', region_cols)




region_cols = [region_colors[i] for i in whisky.Region]
classification_cols = [group_colors[i] for i in whisky.Group]

location_plot("Whisky_Locations_and_Regions", region_cols)
location_plot("Whisky_Locations_and_Groups", classification_cols)

























# Week 4: Case Studies Part 2/Case Study 5: Bird Migration

# 4.2.1: Introduction to GPS Tracking of Birds

import pandas as pd

birddata = pd.read_csv('https://courses.edx.org/assets/courseware/v1/6184eb0f87c7b58db1a5c336e436ed09/asset-v1:HarvardX+PH526x+2T2021+type@asset+block/bird_tracking.csv')
birddata.info()
pd.set_option('display.max_columns', None)
birddata.head()



# 4.2.2: Simple Data Visualizations

import matplotlib.pyplot as plt
import numpy as np
import os

os.getcwd()
path = 'edx/GPS'

birddata.bird_name == 'Eric'
x = birddata.longitude[birddata.bird_name == 'Eric']
y = birddata.latitude[birddata.bird_name == 'Eric']
plt.plot(x, y, '.')
plt.savefig(os.getcwd()+'/'+path+'/'+'flight_trajectory.png')
plt.show()

bird_names = pd.unique(birddata.bird_name)
for bird_name in bird_names:
    x = birddata.longitude[birddata.bird_name == bird_name]
    y = birddata.latitude[birddata.bird_name == bird_name]
    plt.plot(x, y, '.', label=bird_name)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc='upper left')
# plt.savefig(os.getcwd()+'/'+path+'/'+'3traj.png')
plt.show()






# 4.2.3: Examining Flight Speed

birddata.columns
speed = birddata.speed_2d[birddata.bird_name == 'Eric']
plt.hist(speed)
plt.hist(speed[:10])
plt.show()
# non number objects
np.isnan(speed)
any(np.isnan(speed))
np.isnan(speed).any()
# numeber of NaN's
np.sum(np.isnan(speed))
# remove Nan's
speed[-np.isnan(speed)]
list(-np.isnan(speed)) == list(~np.isnan(speed))

# NaN rows
speed[np.isnan(speed) == True]
len(speed[np.isnan(speed) == True])
speed[-np.isnan(speed) == True]
not(False)

# bitwise not operator with and without parenthesis
list(speed[~(np.isnan(speed) == True)]) == list(speed[~np.isnan(speed) == True])

# hist without NaN
plt.hist(speed[~np.isnan(speed) == True], bins=np.linspace(0, 30, 20), density=True)
# plt.axis([0, 25, 0, .4])
plt.axis('tight')
plt.xlabel('2D speed (m/s)')
plt.ylabel('Frequency')
plt.savefig(os.getcwd()+'/'+path+'/'+'hist.png')
plt.show()

# hist using pandas
birddata.speed_2d.plot(kind='hist', range=[0, 30])
plt.xlabel('2D speed (m/s)')
plt.ylabel('Frequency')
plt.show()





# 4.2.4: Using Datetime

birddata.date_time
import datetime

# how much time has leapsed, time delta object
time1 = datetime.datetime.today()
type(time1)
datetime.datetime.today() - time1
type(datetime.datetime.today() - time1)

# Here is the field following second is UTC,
# which stands for coordinated universal time,
# which is an offset that is expressed in hours.
# In this case those entries are always 0, something you can check easily.

date_str = birddata.date_time[0]
type(date_str)
# turn string to datetime object
datetime.datetime.strptime(date_str[:-3], '%Y-%m-%d %H:%M:%S')
timestamps = [datetime.datetime.strptime(birddata.date_time[i][:-3], \
'%Y-%m-%d %H:%M:%S') for i in range(len(birddata))]
timestamps[:3]
birddata['timestamp'] = pd.Series(timestamps, index=birddata.index)
birddata['timestamp'] = timestamps
list(birddata.index)[10]
birddata.head()

# drop a column
birddata.drop(['timestamp'], axis=1)
birddata.drop(['timestamp'], axis='columns')
birddata.drop(columns=['timestamp'])

# apply timestamp without for loop and pd.Series
birddata['timestamp'] = birddata['date_time'].apply(
    lambda x: datetime.datetime.strptime(x[:-3], '%Y-%m-%d %H:%M:%S')
)


# arythmetic with timestamps
# how much time has passed between two points
birddata.timestamp[4] - birddata.timestamp[3]
times = birddata.timestamp[birddata.bird_name == 'Eric']
elapsed_time = [time - times[0] for time in times]
# If you look at the entry at say 1000, we know that in this case
# 12 days, 2 hours, and 2 minutes have passed
# between observation located at 0, and observation located at index 1000.
elapsed_time[1000]


# But how can we measure time in certain units, like hours or days?
elapsed_days = np.array(elapsed_time) / datetime.timedelta(days=1)
# how many days have passed
elapsed_time[1000] / datetime.timedelta(days=1)
elapsed_time[1000] / datetime.timedelta(hours=1)


plt.plot(np.array(elapsed_time) / datetime.timedelta(days=1))
plt.xlabel('Observation')
plt.ylabel('Elapsed time (days)')
# plt.savefig(os.getcwd()+'/'+path+'/'+'timeplot.png')
plt.show()







# 4.2.5: Calculating Daily Mean Speed

len(elapsed_days)
len(birddata.speed_2d)
plt.plot(birddata.timestamp, birddata.speed_2d)
# shows speed at each timestamp not each day
plt.show()


data_eric = birddata[birddata.bird_name == 'Eric']
data_eric.timestamp[len(data_eric)-1] - data_eric.timestamp[0]

current_day = data_eric.timestamp[0].date()
avg_day_speed = []
day_speed_list = []
i = 0
while i <= len(data_eric):
    while data_eric.timestamp[i].date() == current_day:
        if not np.isnan(data_eric.speed_2d[i]):
            day_speed_list.append(birddata.speed_2d[i])
        i += 1
    avg_day_speed.append(np.mean(day_speed_list))
    day_speed_list = []
    current_day = birddata.timestamp[i].date()

plt.plot(avg_day_speed)
plt.xlabel('Day')
plt.ylabel('Mean speed (m/s)')
plt.show()


next_day = 1
inds = []
daily_mean_speeed = []
for (i, t) in enumerate(elapsed_days):
    if t < next_day:
        inds.append(i)
    else:
        daily_mean_speeed.append(np.mean(birddata[birddata.bird_name == 'Eric'].speed_2d[inds]))
        next_day += 1
        inds = []

plt.plot(daily_mean_speeed)
plt.xlabel('Day')
plt.ylabel('Mean speed (m/s)')
# plt.savefig(os.getcwd()+'/'+path+'/'+'daily_mean_speed.png')
plt.show()

avg_day_speed == daily_mean_speeed
avg_day_speed[1]
daily_mean_speeed[1]

daystamps = [birddata.timestamp[i].date() for i in range(len(birddata))]
type(daystamps)
birddata['daystamp'] = pd.Series(daystamps, index=birddata.index)

data_eric.timestamp[data_eric.daystamp == data_eric.daystamp[0]]
np.mean(data_eric.speed_2d[data_eric.daystamp == data_eric.daystamp[0]])
daily_mean_speeed[0]










# 4.2.6: Using the Cartopy Library

# Cartopy provides cartographic tools for Python

# conda install -c conda-forge cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# To move forward, we need to specify a specific projection
proj = ccrs.Mercator()

# plt.figure(figsize=(10, 10))
# ax.set_extent((-25.0, 20.0, 52.0, 10.0))
ax = plt.axes(projection=proj)
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
for name in bird_names:
    x = birddata.longitude[birddata.bird_name == name]
    y = birddata.latitude[birddata.bird_name == name]
    ax.plot(x, y, '.', transform=ccrs.Geodetic(), label=name)
plt.legend(loc='upper left')
# plt.savefig(os.getcwd()+'/'+path+'/'+'map.png')
plt.show()

# In this case, we can see the flight trajectories as before,
# but in this case, we've used an actual cartographic projection.
# That means that these correspond to an actual map.

# We've just looked at some very basics of how to visualize
# bird flights obtained from GPS data.


birddata.bird_name == 'Eric'
birddata['bird_name'] == 'Eric'
birddata[birddata['bird_name'] == 'Eric']








# Week 4: Case Studies Part 2/Homework: Case Study 5


import pandas as pd
import numpy as np
birddata = pd.read_csv("https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@bird_tracking.csv", index_col=0)
birddata.head()




# First, use `groupby()` to group the data by "bird_name".
grouped_birds = birddata.groupby('bird_name')

# Now calculate the mean of `speed_2d` using the `mean()` function.
mean_speeds = grouped_birds.speed_2d.mean()
mean_speeds = birddata.groupby('bird_name').speed_2d.mean()

# Find the mean `altitude` for each bird.
mean_altitudes = grouped_birds.altitude.mean()



import datetime as dt

# Convert birddata.date_time to the `pd.datetime` format.
birddata.date_time = pd.to_datetime(birddata.date_time)

# Create a new column of day of observation
birddata['date'] = birddata.date_time.dt.date

# Use `groupby()` to group the data by date.
grouped_bydates = birddata.groupby('date')

# Find the mean `altitude` for each date.
mean_altitudes_perday = grouped_bydates.altitude.mean()

mean_altitudes_perday



# Use `groupby()` to group the data by bird and date.
grouped_birdday = birddata.groupby(['bird_name', 'date'])
grouped_daybird = birddata.groupby(['date', 'bird_name'])

# Find the mean `altitude` for each bird and date.
mean_altitudes_perday = grouped_birdday.altitude.mean()
# grouped_daybird.altitude.mean()
mean_altitudes_perday



import matplotlib.pyplot as plt

eric_daily_speed  = grouped_birdday.speed_2d.mean().Eric
sanne_daily_speed = grouped_birdday.speed_2d.mean().Sanne
nico_daily_speed  = grouped_birdday.speed_2d.mean().Nico

# eric_daily_speed.plot(label='Eric')
plt.plot(eric_daily_supeed, label='Eric')
sanne_daily_speed.plot(label='Sanne')
nico_daily_speed.plot(label='Nico')
plt.plot(eric_daily_speed)
plt.legend(loc='upper left')
plt.show()






















# Week 4: Case Studies Part 2/Case Study 6: Social Network Analysis

# 4.3.1: Introduction to Network Analysis

# Many systems of scientific and societal interest
# consist of a large number of interacting components.
# The structure of these systems can be represented
# as networks where network nodes represent the components,
# and network edges, the interactions between the components.
# Network analysis can be used to study how pathogens, behaviors,
# and information spread in social networks,
# having important implications for our understanding of epidemics
# and the planning of effective interventions.
# In a biological context, at a molecular level,
# network analysis can be applied to gene regulation networks, signal
# transduction networks, protein interaction networks, and much,
# much more.

# we should really distinguish between the terms network and graph.
# Network refers to the real world object, such as a road
# network, whereas a graph refers to its abstract mathematical representation.
# It's useful to be aware of this distinction,

# Graphs consist of nodes, also called vertices, and links, also called edges.
# Mathematically, a graph is a collection of vertices and edges
# where each edge corresponds to a pair of vertices.
# When we visualize graphs, we typically draw vertices as circles and edges
# as lines connecting the circles.

# The degree of a vertex is the number of entries connected to it.
# A path is a sequence of unique vertices, such
# that any two vertices in the sequence are connected by an edge.
# The length of a path is defined as the number of edges in that path.
# In this case, we're specifically talking about the shortest paths,
# a path that comprises the minimum number of steps for us to go from vertex a
# to vertex b.

# A vertex v is said to be reachable from vertex u
# if there exists a path from u to v, that is, if there is a way
# to get from u to v.
# A graph is said to be connected if every vertex is reachable
# from every other vertex, that is, if there
# is a path from every vertex to every other vertex.
# Any component is connected when considered on its own,
# but there are no edges between the nodes that belong to different components.
# The size of a component is defined as the number of nodes in the component.

# 4.3.2: Basics of NetworkX

import networkx as nx

# We can then create an instance of an undirected graph using that Graph function.
G = nx.Graph()
# adding nodes
G.add_node(1)
G.add_nodes_from([2, 3])
G.add_nodes_from(['u', 'v'])
G.nodes()
# adding edges
G.add_edge(1, 2)
G.add_edge('u', 'v')
G.add_edges_from([(1, 3), (1, 4), (1, 5), (1, 6)])
G.add_edge('u', 'w')
G.nodes()
G.edges()
# removing nodes and edges
G.remove_node(2)
G.remove_nodes_from([4, 5])
G.remove_edge(1, 3)
G.remove_edges_from([(1, 2), ('u', 'w')])
G.number_of_nodes()
G.number_of_edges()






# 4.3.3: Graph Visualization

import matplotlib.pyplot as plt
import os

os.getcwd()
# chdir('/home/ukasz/Documents/IT/Python')
path = 'edx/Network_Analysis'


G = nx.karate_club_graph()
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
# plt.savefig(os.getcwd()+'/'+path+'/'+'karate_graph.png')
plt.show()


# Networkx stores the degrees of nodes in a dictionary where
# the keys are node IDs and the values are their associated degrees.

G.degree()
G.degree()[33]
G.degree(33)
# Note that although both are called degree,
# Python distinguishes between the two based on their call signature.
# The former has no arguments, whereas the latter has one argument.
# The ID of the node whose degree we are querying.
# This is how Python knows whether to return a dictionary
# or whether to use a function to look up the degree of the given node.
G.degree()[33] is G.degree(33)
G.number_of_nodes(); G.number_of_edges()








# 4.3.4: Random Graphs

# The simplest possible random graph model is the so-called Erdos-Renyi,
# also known as the ER graph model.
# This family of random graphs has two parameters, capital N and lowercase p.
# Here the capital N is the number of nodes in the graph,
# and p is the probability for any pair of nodes to be connected by an edge.

from scipy.stats import bernoulli

# We'll be using the rvs method to generate one single realization
# in this case of a Bernoulli random variable.
bernoulli.rvs(0.2)
N = 20
p = 0.2

random.seed(1)
def er_graph(N, p):
    """Generate an ER graph."""

    G = nx.Graph()
    G.add_nodes_from(range(N))
    # G.nodes()
    for node1 in G.nodes():
        for node2 in G.nodes():
            # Just to be clear here, the first p is the name of the keyword argument
            # that we are providing.
            # The second p here is the actual value of p, in this case 0.2.
            if (bernoulli.rvs(p=p) and
                node1 < node2):
                G.add_edge(node1, node2)
    # G.edges()
    # G.number_of_nodes()
    # G.number_of_edges()
    return G

nx.draw(er_graph(50, .08), node_size=40, node_color='red')
# plt.savefig(os.getcwd()+'/'+path+'/'+'er1.png')
plt.show()


from itertools import combinations
list(combinations(range(3), 2))

def er_graph_comb(N, p):
    """Generate an ER graph."""

    G = nx.Graph()
    G.add_nodes_from(range(N))
    # G.nodes()
    for node in list(combinations(range(N), 2)):
        if bernoulli.rvs(p=p):
            G.add_edge(node[0], node[1])
    return G

nx.draw(er_graph_comb(50, .08), node_size=40, node_color='red')
# plt.savefig(os.getcwd()+'/'+path+'/'+'er1.png')
plt.show()









# 4.3.5: Plotting the Degree Distribution

nx.er_graph(10, 1)

G.degree()
list(dict(G.degree()).keys())
list(dict(G.degree()).values())
dict(G.degree()).values()

# I first want to turn that into a list.
# I turn this into a list because G.degree.values gives me
# a view object to the values.
# I actually want to create a copy of that and turn it into a list.
# That's why I include the list surrounding the G.degree values object.


def plot_degree_dist(G):
    plt.hist([v for k, v in G.degree()], histtype='step', label='name')
    plt.xlabel('Degree $k$')
    plt.ylabel('$P(k)$')
    plt.title('Degree distribution')
    # plt.savefig(os.getcwd()+'/'+path+'/'+'hist1.png')
    

G1 = er_graph(500, .08)
plot_degree_dist(G1)
G2 = er_graph(500, .08)
plot_degree_dist(G2)
G3 = er_graph(500, .08)
plot_degree_dist(G3)
plt.show()








# 4.3.6: Descriptive Statistics of Empirical Social Networks

# The structure of connections in a network
# can be captured in what is known as the Adjacency matrix of the network.
# If we have n nodes, this is n by n matrix,
# where entry ij is one if node i and node j have a tie between them.
# Otherwise, that entry is equal to zero.

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

path = 'edx/Network_Analysis'
A1 = np.loadtxt(os.getcwd()+'/'+path+'/'+'/''adj_allVillageRelationships_vilno_1.csv', delimiter=',')
A2 = np.loadtxt(os.getcwd()+'/'+path+'/'+'/''adj_allVillageRelationships_vilno_2.csv', delimiter=',')
# Our next step will be to convert the adjacency matrices to graph objects.
G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)
np.mean([v for v in G1.degree().values()])
def basic_net_stats(G):
    print('Number of nodes %d' % G.number_of_nodes())
    print('Number of edges {}'.format(G.number_of_edges()))
    print('Average degree {:.2f}'.format(np.mean([v for v in dict(G.degree()).values()])))
    # print('Average degree %.2f' % np.mean([v for k, v in G.degree()]))
    
basic_net_stats(G1)
basic_net_stats(G2)

plot_degree_dist(G1)
plot_degree_dist(G2)
# plt.legend(loc='upper right')
# plt.savefig(os.getcwd()+'/'+path+'/'+'village_hist.png')
plt.show()

# Notice how the degree distributions look quite different from what
# we observed for the ER networks.
# It seems that most people have relatively few connections,
# whereas a small fraction of people have a large number of connections.
# This distribution doesn't look at all symmetric,
# and its tail extends quite far to the right.
# This suggests that the ER graphs are likely not good models
# for real world social networks.
# In practice, we can use ER graphs as a kind of reference graph
# by comparing their properties to those of empirical social networks.
# More sophisticated network models are able to capture
# many of the properties that are shown by real world networks.








# 4.3.7: Finding the Largest Connected Component

# In most networks, most nodes are connected to each other
# as part of a single connected component.
# That is for each pair of nodes in this component,
# there exists a set of edges that create a path between them.
# Let's now find out how large the largest connected component
# is in our two graphs.

# deprecated and removed
nx.connected_component_subgraphs(G1)

gen = (G1.subgraph(c) for c in nx.connected_components(G1))
# Generator functions do not return a single object
# but instead, they can be used to generate a sequence of objects
# using the next method.
# To get to an actual component, we can use the next method.
type(gen)
g = gen.__next__()
type(g)
# number of nodes
g.number_of_nodes()
len(g)
len(gen.__next__())
# And what Python is telling us is that the next subsequent component
# has three nodes in it.
# And we could, in principle, run these a few times
# until we run out of components.

# When we're running the line length of generator next,
# Python is going over the graph one component at a time.
# So for example, in this case, we might have
# five components or as many as 25 components in our graphs.
# Each of these components has some size associated
# with it, which again is the number of nodes
# that make up that given component.
# When we run this line, Python is telling us
# that there is a component in that graph that has size 2.
# In other words, there is a component that consists of only two nodes.
# We can keep running this and until eventually we'll
# have run out of components.

# number of nodes
len(G1)
G1.number_of_nodes()


# First, we call connected components subgraphs on G1.
# We provide that as input, the max function.
# And then we provide a key, which is equal to len, in this case.
# The object is G1 underscore LCC-- LCC for Largest Connected Component.
G1_LCC = max((G1.subgraph(c) for c in nx.connected_components(G1)), key=len)
G2_LCC = max((G2.subgraph(c) for c in nx.connected_components(G2)), key=len)
len(G1_LCC)
G1_LCC.number_of_nodes()
G2_LCC.number_of_nodes()

# This is the number of nodes in the largest connected component.
# We can divide that by the number of nodes in the graph itself.
G1_LCC.number_of_nodes() / G1.number_of_nodes()
G2_LCC.number_of_nodes() / G2.number_of_nodes()
# And in this case, we see that 97.9% of all of the nodes of graph G1
# are contained in the largest connected component.
nx.draw(G1_LCC, node_color='red', edge_color='gray', node_size=20)
plt.savefig(os.getcwd()+'/'+path+'/'+'village1.png')
nx.draw(G2_LCC, node_color='green', node_size=20)
plt.savefig(os.getcwd()+'/'+path+'/'+'village2.png')
plt.show()

# The visualization algorithm that we have used
# is stochastic, meaning that if you run it several times,
# you will always get a somewhat different graph layout.

# However, in most visualizations, you should
# find that the largest connected component of G2
# appears to consist of two separate groups.
# These groups are called network communities.
# And the idea is that a community is a group
# of nodes that are densely connected to other nodes in the group,
# but only sparsely connected nodes outside of that group.










# Week 4: Case Studies Part 2/Homework: Case Study 6

# Homophily is a network characteristic.  Homophily occurs when nodes 
# that are neighbors in a network also share a characteristic more often 
# than nodes that are not network neighbors.  In this case study, we will investigate 
# homophily of several characteristics of individuals connected in social networks in rural India.


from collections import Counter
import numpy as np

def marginal_prob(chars):
    # color_dict = OrderedDict()
    # for color in chars.values():
    #    color_dict[color] = color_dict.get(color, 0) + 1
    #return {color: count / sum(color_dict.values()) for color, count in color_dict.items()}
    frequencies = dict(Counter(chars.values()))
    sum_frequencies = sum(frequencies.values())
    return {char: freq / sum_frequencies for char, freq in frequencies.items()}
        
def chance_homophily(chars):
    marginal_probs = marginal_prob(chars)
    return np.sum(np.square(list(marginal_probs.values())))

favorite_colors = {
    "ankit":  "red",
    "xiaoyu": "blue",
    "mary":   "blue"
}

color_homophily = chance_homophily(favorite_colors)
print(color_homophily)



import pandas as pd

df  = pd.read_csv("https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@individual_characteristics.csv", low_memory=False, index_col=0)
df1 = df[df.village == 1]
df2 = df[df.village == 2]

df1.head()



sex1 = df1.set_index('pid').resp_gend.to_dict()
caste1 = df1.set_index('pid').caste.to_dict()
religion1 = df1.set_index('pid').religion.to_dict()

sex2 = df2.set_index('pid')['resp_gend'].to_dict()
caste2 = df2.set_index('pid')['caste'].to_dict()
religion2 = df2.set_index('pid')['religion'].to_dict()

caste2[202802]



print("Village 1 chance of same sex:", chance_homophily(sex1))
print("Village 1 chance of same caste:", chance_homophily(caste1))
print("Village 1 chance of same religion:", chance_homophily(religion1))

print("Village 2 chance of same sex:", chance_homophily(sex2))
print("Village 2 chance of same caste:", chance_homophily(caste2))
print("Village 2 chance of same religion:", chance_homophily(religion2))



def homophily(G, chars, IDs):
    """
    Given a network G, a dict of characteristics chars for node IDs,
    and dict of node IDs for each node in the network,
    find the homophily of the network.
    """
    num_same_ties = 0
    num_ties = 0
    for (n1, n2) in G.edges():
        if (IDs[n1] in chars and
            IDs[n2] in chars):
            if G.has_edge(n1, n2):
                num_ties += 1 # Should `num_ties` be incremented?  What about `num_same_ties`?
                if chars[IDs[n1]] == chars[IDs[n2]]:
                    num_same_ties += 1 # Should `num_ties` be incremented?  What about `num_same_ties`?
    return (num_same_ties / num_ties)



data_filepath1 = "https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@key_vilno_1.csv"
data_filepath2 = "https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@key_vilno_2.csv"

data_1 = pd.read_csv(data_filepath1, index_col=0)
# data_1['0'].to_dict()[100]
data_1['0'][100]



import networkx as nx

A1 = np.array(pd.read_csv("https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@adj_allVillageRelationships_vilno1.csv", index_col=0))
A2 = np.array(pd.read_csv("https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@adj_allVillageRelationships_vilno2.csv", index_col=0))
G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)

pid1 = pd.read_csv(data_filepath1, dtype=int)['0'].to_dict()
pid2 = pd.read_csv(data_filepath2, dtype=int)['0'].to_dict()


print("Village 1 observed proportion of same sex:", homophily(G1, sex1, pid1))
print("Village 1 observed proportion of same caste:", homophily(G1, caste1, pid1))
print("Village 1 observed proportion of same religion:", homophily(G1, religion1, pid1))
print("Village 2 observed proportion of same sex:", homophily(G2, sex2, pid2))
print("Village 2 observed proportion of same caste:", homophily(G2, caste2, pid2))
print("Village 2 observed proportion of same religion:", homophily(G2, religion2, pid2))





















# Week 5: Statistical Learning/Week 5 Overview

# 5.1.1: Introduction to Statistical Learning


# Statistical learning can be divided into two categories which
# are called supervised learning and unsupervised learning.
# Supervised learning refers to a collection of techniques and algorithms
# that, when given a set of example inputs and example outputs,
# learn to associate the inputs with the outputs.
# The outputs usually need to be provided by a supervisor, which
# could be a human or another algorithm, and this is where the name comes from.
# Unsupervised learning refers to a collection of techniques and algorithms
# that are given inputs only and there are no outputs.
# The goal of supervised learning is to learn relationships and structure
# from such data.

# In this case study, we will learn the basics
# of supervised statistical learning.
# We will look at regression, which refers to problems
# that have continuous outputs, and we will also
# look at classification, which refers to problems
# that have categorical outputs like 0 or 1, blue or green, and so on.

# In statistics, variables can be either quantitative or qualitative.
# Quantitative variables take on numerical values, such as income,
# whereas qualitative variables take on values in a given category,
# such as male or female.
# The range of quantitative variables depends on what they measure
# and what units are used.
# For example, we might measure annual income
# in dollars or thousands of dollars.
# Similarly, qualitative variables can have two or more categories or classes.
# The male-female example refers to a qualitative variable
# with two categories

# For example, for income we might specify that a household with an annual income
# less than $30, 000 a year is a low income household
# a household with an income between $30, 000 and $100, 000
# is a middle income household
# and a household with an annual income
# exceeding $100, 000 a year a high income household.

# Methods in supervised learning are divided into two groups
# based on whether the output variable, also called the outcome,
# is quantitative or qualitative.
# If the outcome is quantitative, we talk about regression problems,
# whereas if the outcome is qualitative, we talk about classification problems.
# Note that this division into regression and classification problems
# is made based on the nature of the output, not the inputs,
# and it's common for both regression and classification problems
# to involve a mixture of quantitative and qualitative inputs.

# In both problems, we have some input variable X and an output variable Y,
# and we seek some function f of X for predicting Y, given values of the input
# X. What the best prediction is depends on the so-called loss function,
# which is a way of quantifying how far our predictions for Y for a given
# value of X are from the true observed values of Y. This
# is the subject of statistical decision theory which is outside of our scope,
# but we will state the relevant results here.

# First, in a regression setting, by far the most common loss function
# is the so-called squared error loss.
# And in that case, the best value to predict for a given X
# is a conditional expectation, or a conditional average,
# of Y given X. So what that means is that what we should predict
# is the average of all values of Y that correspond to a given value of X.

# Second, in a classification setting, we most often
# use the so-called 0-1 loss function, and in that case,
# the best classification for a given X is obtained
# by classifying observation of the class with the highest
# conditional probability given X. In other words, for a given value of X,
# we compute the probability of each class and we then
# assign the observation to the class with the highest probability.

# So let's write down what is it we're actually estimating.
# In this case, we are estimating, in the regression setting,
# a conditional expectation of Y given a specific value for x.
# E(Y|X=x)
# So this is simply saying that this is a conditional mean, a conditional average
# taking over all points that share at the value of x.
# This is your regression function.
# So if we repeat this for all values of x, we will get a line like this.
# And typically we might call this f of x.

# In a classification setting, we would want to estimate two probabilities.
# These are conditional probabilities.
# So the first of them would be the probability
# P(Y=0|X=x)
# P(Y=1|X=x)
# that the random variable Y is equal to 0 given the value of x.
# And the second probability is for Y equal to 1
# given that the random variable, which is the large X, is equal to the small x.
# And whichever of these two probabilities is largest,
# that's going to be our prediction for a given value of x.

# What is the difference between supervised and unsupervised learning?
# Supervised learning matches inputs and outputs, whereas unsupervised learning discovers structure for inputs only.

# What is the difference between regression and classification?
# Regression results in continuous outputs, whereas classification results in categorical outputs.

# What is the difference between least squares loss and 0-1 loss?
# Least squares loss is used to estimate the expected value of outputs, 
# whereas 0-1 loss is used to estimate the probability of outputs.






# 5.1.2: Generating Example Regression Data

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

# generate datapoints
# linear regresion
n = 100
beta_0 = 5
beta_1 = 2
np.random.seed(1)
# generate x variables
# x = 10 * np.random.random(100)
x = 10 * ss.uniform.rvs(size=n)
# deterministic part of the model + random noise
y = beta_0 + beta_1*x + ss.norm.rvs(loc=0, scale=1, size=n)
# plt.figure()
plt.plot(x, y, 'o', ms=5)
xx = np.array([0, 10])
plt.plot(xx, beta_0 + beta_1*xx)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
np.mean(x); np.mean(y)

# So what we did was we first sampled 100 points from the interval 0 to 10.
# Then we computed the corresponding y values.
# And then we added a little bit of normally distributed
# or Gaussian noise all of those points.
# And that's why the points are being scattered around the line.









# 5.1.3: Simple Linear Regression

# In simple linear regression, the goal
# is to predict a quantitative response Y on the basis of a single predictor
# variable X. It assumes the following relationship between the random
# variables X and Y.
# Y = B0 + B1*X + E
# Our random variable, which is capital Y, is going to be given to us by some
# parameter beta 0 plus some other parameter-- let's call it beta 1--
# times a random variable X plus epsilon, which is some error term.
# Note that the capital letters here always correspond to random variables.

# Once we have used training data to produce estimates,
# beta 0 hat and beta 1 hat, for the model coefficients,
# we can predict future values of Y.
# So I'm going to write this underneath here.
# Our predicted value is going to be a lowercase y with a hat on top.
# We have our beta 0 hat to indicate that this has been estimated from data.
# Plus, we have beta 1 hat, again estimated from data.
# And then we have our lowercase x over here.
# So here, just to be clear, y hat indicates
# a prediction, or a specific value, of the random variable Y
# on the basis of a specific value where X (the uppercase X)
# is equal to a lowercase x.
# Notice the hats on top of the betas.
# They indicate that these are parameter estimates,
# meaning that their parameter values that have been estimated using data.

# To estimate the unknown coefficients in the model, we must use data.
# Let's say that our data consists of n observation
# pairs, where each pair consists of a measurement of x
# and a measurement of y.
# We can write these n observations as follows.
# We can take our first pair, x1, y1.
# This is our first data point.
# Then we can take our second data point, x2, y2, and so on, all the way to xn,
# yn.
# These are observed data.
# The most common approach to estimating the model parameters
# involves minimizing the least squares criterion.
# We first define the i-th residual as follows.
# We typically use lowercase e for the residual.
# ei = yi - y^i
# So the residual for the i-th observation is
# going to be given by the data point yi minus our predicted value
# for that same data point.
# So e sub i here is the difference between the i-th
# observed response value and the i-th response value predicted by the model.
# From this, we can define the residual sum of squares, RSS, as follows.
# I'll create a bit of space here.
# RSS is equal to the first residual squared
# plus the second residual squared plus the n-th residual squared.
# The least squares estimates of beta 0 and beta 1--
# these guys here-- are those values of beta 0
# and beta 1 that minimize the RSS criterion.
# RSS = sum(ei^2)

# What is the difference between  (capital letter) and  (lowercase letter)?
# Y is a random variable, whereas y is a particular value.


def compute_rss(y_estimate, y):
  # return np.sum(np.power(y-y_estimate, 2))
  return np.sum(np.square(y-y_estimate))

def estimate_y(x, b_0, b_1):
  return b_0 + b_1*x

rss = compute_rss(estimate_y(x, beta_0, beta_1), y)











# 5.1.4: Least Squares Estimation in Code

# We can use matrix calculus to obtain a formula, a closed form
# solution for obtaining the least-squares estimates,
# and that is how some software packages do the estimation.
# Other software packages use different methods,
# such as the method of gradient descent, which is a numerical optimization
# method used for complex models.

slopes = np.arange(-10, 15, 0.01)
rss = [np.sum((y - beta_0 - slope*x)**2) for slope in slopes]
# Now, this gives us the deviation from the value predicted by the model.
# So we want to make sure to square this.
# And then we want to add these terms up.
min(rss)
# create a variable which is ind_min,
# and this will be used to find the index within the rss list that
# gives me the lowest value for rss.
ind_min = np.argmin(rss)
slopes[ind_min]
print('Estimate for that slope: ', slopes[ind_min])
print('RSS for the lope: ', rss[ind_min])
plt.plot(slopes, rss)
plt.xlabel('Slope')
plt.ylabel('RSS')
plt.show()

# We're plotting on the x axis different slope values,
# and on the y-axis different RSS values.


# Note that in this case, the estimated value of the parameter (2.0) coincides 
# with the true value of the parameter. Generally, we do not know the underlying 
# true value, but here it is known to us because we generated the data ourselves. 
# In practical settings, the estimated parameter value may not always match the true value.












# Simple Linear Regression in Code

import statsmodels.api as sm

# We're going to define a mod object by saying sm.OLS.
# OLS stands for ordinary least squares.
# And we need an argument inside the parentheses.
# The first is y, which is our y-values.
# The second one is x, our predictor values.
mod = sm.OLS(y, x)
# we first fit that.
est = mod.fit()
# gives us a summary of the model of the fitted model object.
est.summary()
# In this case, we've actually fitted a slightly different model,
# a model that has a slope, but no intercept,
# meaning that there is no constant term in the model.
# This means that the line is forced to go through 0, which
# is why the slope is artificially large.

# Let's now try to fit a slightly different model, a model that includes
# the constant, the intercept term.
# I'm now going to be building another variable, which
# is capital X, which is the same as x, but includes one column of 1s.
X = sm.add_constant(x)
# Then we take the model.
# We define a new model object.
# I'm going to call that mod.
# We continue to do OLS.
# We have our outcome y, but now we need the capital X, the one
# that has the constants added to it.
# We'll estimate this model-- we'll fit the model by saying mod.fit,
mod = sm.OLS(y, X)
est = mod.fit()
est.summary()
# In this case, we can see we have two predictors in the model.
# We have the constant term, which has been estimated as 5.2,
# and then we have a coefficient to go with the predictor x1.
# The intercept 5.2 is the value of the outcome y
# when all predictors, here just x1, are set to 0.
# The slope has a somewhat different interpretation,
# which is that an increase of 1 in the value of x1
# is associated with an increase of 1.97 in the value of the outcome y.

# Imagine generating a new sample of data and feeding the model
# to this new sample.
# In this case, we would get a somewhat different estimate
# for intercept and slope.
# If we repeated this process many times, always generating a new data set,
# and fitting the model to this new data set,
# we would end up with a distribution of estimates for the intercept
# and another distribution of estimates for the slope.
# These distributions are known as sampling distributions of the parameter
# estimates.

# Let's focus on the sampling distribution of the slope estimates.
# The standard deviation of this distribution
# is known as the standard error of the slope.
# The smaller the standard error, the more precisely it's
# being estimated, meaning that the more we've
# been able to learn about its value from the data.

# We can take the value of the standard error, multiply it by 2,
# or 1.96 to be exact, and add this number to the point estimate of the intercept.
# Similarly, we can subtract this number from the point estimate.
# Doing this gives us the 95% confidence interval, which
# is also shown as part of the output.
# (1.9685 +- 1.96*.031)*.95

# The output summary includes the so-called r-squared statistic,
# which is the proportion of variance explained.
# And because it's a proportion, it's always between 0 and 1.
# But what does variance explained actually mean?
# Before we fit our model, we can compute what
# is called the total sum of squares, or TSS, which
# is defined as the sum of the squared differences between outcome
# yi and the mean outcome.
# Now after we've created the model, we compute a similar quantity
# called the residual sum of squares, RSS, which
# is defined as the sum of the squared differences between the outcome yi
# and the outcome predicted by the model yi hat.
# If the model is useful at all, we would expect
# the RSS is to be smaller than the TSS.
# The r-squared statistic takes the difference between TSS and RSS,
# and then divides that quantity by TSS.
# A number near 0, therefore, indicates that the model did not
# explain much of the variability in the response or the outcome.
# Larger values are better, but what values of r-squared are considered good
# always depends on the application context.
# R^2 = (TSS-RSS) / TSS

# If the true intercept were negative but the regression model did not include an intercept term, 
# what would that imply for the estimated slope?
# The estimated slope would likely be lower than the true slope.

# What does an estimated intercept term correspond to?
# The estimated outcome when the input is set to zero.

# What does an estimated slope term correspond to?
# The change in the estimated output when the input changes by one unit.

# You could create several datasets using different seed values and estimate the slope from each. 
# These parameters will follow some distribution.
# What is the name used for this distribution?
# The sampling distribution of the parameter estimates.

# If the R^2 value is high, this indicates
# a good fit: the residual sum of squares is low compared to the total sum of squares.












# 5.1.6: Multiple Linear Regression

# In multiple linear regression,
# the goal is to predict a quantitative or a scalar valued
# response, Y, on the basis of several predictor variables.
# And the model takes the following form.
# We're still going to be using a capital Y for our output or outcome.
# And Y is equal to beta 0 plus beta 1 times X1 plus beta 2 times
# X2 and so on until beta p times Xp.
# And then we also have the error term here.
# In this case, I put these little bars at the bottom and top of X to highlight
# the fact that these are capital X's.
# So these are all random variables.
# Multiple linear regression is very similar to simple linear regression,
# but it's worth taking a moment to make sure we
# know how to interpret the coefficients.
# In general, consider a predictor Xk and the parameter beta
# k associated with that predictor.
# A unit change the value of xk is associated with a change beta hat k
# in the value of the outcome y, while keeping all other predictors fixed.
# If the values of the predictors are correlated,
# it may not be possible to change the value of one predictor
# and keep the others fixed.
# So one therefore always needs to be careful with interpretation
# of model results.
# Y = B0 + B1*X1 + B2*X2 + ... + Bn*Xn + e


# Consider a multiple regression model with two inputs. The model predictions for the output y are given by
# y^ = b0^ + x1*b1^ + x2b2^
# b1 and b2 have been estimated from data. If we assume that , and .
# What is the interpretation of b1?
# The change in the predicted outcome if x1 is increased by 1, holding x2 constant.


# Consider the model and parameters in Question 1. For a given expected output prediction , 
# what would be the expected change in the prediction value if you increased  by 1, and decreased  by 3?
# 8










# 5.1.7: scikit-learn for Linear Regression

# Scikit-learn is an open source,
# machine learning library for Python.
# It's one of the most prominent Python libraries for machine learning.
# And it is widely used in both industry and academia.
# Scikit-learn depends on two other Python packages, NumPy and SciPy,

import numpy as np
import scipy.stats as ss

# sample size
n = 500
beta_0 = 5
beta_1 = 2
beta_2 = -1
# So we're going to have three predictors in the model, one constant and two covariates.
np.random.seed(1)
# rvs = random variables realizations
x_1 = 10 * ss.uniform.rvs(size=n)
x_2 = 10 * ss.uniform.rvs(size=n)
# outcome deterministic + noise
y = beta_0 + beta_1*x_1 + beta_2*x_2 + ss.norm.rvs(loc=0, scale=1, size=n)
X = np.stack([x_1, x_2], axis=1)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], y, c=y)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
plt.show()
# Therefore, the true underlying surface is a plane in three dimensional space.


from sklearn.linear_model import LinearRegression
# Then we'll define an object called lm, which is short for a linear model.
# It's a linear regression.
lm = LinearRegression(fit_intercept=True)
# As the next step, we'll fit this model, lm.fit.
# We provide our x's and our y's.
lm.fit(X, y)
# We can extract the coefficients from the model.
# So we can type lm.intercept, which gives us
# the value of the estimated intercept.
lm.intercept_
# And we can access the coefficients by typing lm.coef,
lm.coef_
# We can also try to predict the value of the outcome for some value of x.
# So I'm going to create a variable X_0.
# And in this case, because we have two predictors in our model,
# we need to provide a value for x1 and x2.
# So say we're interested in the value of the outcome when x1 is equal to 2,
# x2 is equal to 4.
X_0 = np.array([2, 4])
# We can now take our model object, lm, and we can use the predict method.
lm.predict([X_0])
# Reshape your data, either using x.reshape(-1, 1) 
# if your data has a single feature, which is not our case here,
# or x.reshape(1, -1) if it contains a single sample.
lm.predict(X_0.reshape(1, -1))
# So for this particular input value, 2, 4, the values of x1 and x2,
# our predicted output is 5.07.
lm.score(X, y)
# We can also find out the score, the r-squared statistic,
# how well the model works.
# We again, call the model object lm.
# We call the score function.
# And this takes in two inputs.
# The first one is the X matrix, and the second one is the outcome.
# So what happens under the hood is the following.
# The model takes the input values X. It generates a prediction y
# hat, produced by the model.
# And it compares the y hat with the true outcome values y in the training set.
# And in this case, we have a very high r-squared value, 0.98.
# This would be unusually high in most applications.










# 5.1.8: Assessing Model Accuracy

# To evaluate the performance of a regression model,
# we need to somehow quantify how well the predictions of the model
# agree with the observed data.
# In the regression setting, the most commonly used measure
# is the mean squared error, or MSE.
# It's defined in the following way.
# MSE = 1/n sum (yi - f^(xi))^2
# MSE is given by an average.
# We're averaging over n data points.
# Our indexing variable is i, which goes from 1 to n.
# We take yi, which is the observed outcome.
# From that, we subtract our prediction at the corresponding value xi,
# and we square the difference.
# This is the definition of MSE.
# If we compute the MSE using the training data, the data that
# was used to train the model, then we usually call it the &amp;quot;training MSE.&amp;quot;
# But we don't usually care how well the model works on the training data.
# But instead, we'd like the model to perform well
# on previously unseen test data.
# And when we compute the MSE using the test data set,
# we call it the &amp;quot;test MSE.&amp;quot;
# So far, we've talked about model accuracy in the context of regression.

# To evaluate how well a classifier performs, as we'll see shortly,
# we compute the training error rate, which
# is the proportion of errors the classifier makes
# when applied to training data.
# Analogously to the regression setting, we also
# have test error rate, which is the proportion
# of errors or misclassifications the classifier makes
# when applied to test data.

# We can obtain estimates of test error by dividing our data set into two parts--
# the training data and the test data.
# And we use the training data only to train the model.
# Once the model has been fitted, we can now
# test its accuracy using the test data, which was not
# in any way used to train the model.


from sklearn.model_selection import train_test_split
# We call train_test_split using our X variable, which are our predictors,
# our y, and this is our outcome.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.5, random_state=1)
# Now we can fit the model, the linear model, using the training data.
# So our linear model lm is going to be equal to LinearRegression.
lm = LinearRegression(fit_intercept=True)
# So it's important to realize that the first argument here corresponds
# to the matrix of covariates, or predictors, in the training data set.
# The second argument corresponds to the outcomes in the training data set.
lm.fit(X_train, y_train)
# We can now test how well the model performs.
# We want to provide the test matrix, the test covariates.
# And the second input we need is the test outputs.
lm.score(X_test, y_test)
# So let's think about this, what happens?
# The first argument gives the predictors, or covariates, to the linear model.
# Based on that, it makes predictions.
# Let's call them y hats.
# Then the y_test vector, which we provide as a second argument,
# provides the true outcomes for the corresponding x values.
# And this is how lm.score knows how to compute the accuracy of the model.
# We can run this, and in this case, we get a very high r-squared value.


# Statistical models generally perform best
# when their capacity is appropriate for the complexity of the modeling task.
# Typically, more flexible models require estimating
# a greater number of parameters.
# Models that are too flexible can lead to overfitting,
# which means that the model starts to follow the noise in the data
# too closely.
# In the extreme case, the model can memorize the data points rather
# learn the structure of the data.
# The problem with this is that it generalizes poorly to unseen data.
# In contrast, if the model is too simple, it can underfit the data, in which case
# the model is not sufficiently flexible to learn the structure in the data.

# When evaluating the performance of a model in a 
# regression setting on test data, which measure is most appropriate?
# Test MSE

# When evaluating the performance of a model in a classification setting on 
# test data, which measure is most appropriate?
# Test error rate

# What is the primary motivation for splitting our model into training and testing data?
# By evaluating how our model fits on unseen data, we can see how generalizable it is.
















# Week 5: Statistical Learning/Part 2: Logistic Regression

# 5.2.1: Generating Example Classification Data

# We're next going to look at the classification problem,
# and we'll start by generating data from two
# separate two-dimensional normal distributions.

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
# %matplotlib notebook

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

h = 1
sd = 1
n = 50

def gen_data(n, h, sd1, sd2):
    x1 = ss.norm.rvs(-h, sd1, n)
    y1 = ss.norm.rvs(0, sd1, n)
    x2 = ss.norm.rvs(h, sd2, n)
    y2 = ss.norm.rvs(0, sd2, n)
    return (x1, y1, x2, y2)

(x1, y1, x2, y2) = gen_data(n, 1, 1, 1.5)
(x1, y1, x2, y2) = gen_data(n, 1.5, 1, 1.5)
(x1, y1, x2, y2) = gen_data(n, 0, 1, 1)
(x1, y1, x2, y2) = gen_data(n, 1, 2, 2.5)
(x1, y1, x2, y2) = gen_data(n, 10, 100, 100)
(x1, y1, x2, y2) = gen_data(n, 20, .5, .5)



def plot_data(x1, y1, x2, y2):
    plt.plot(x1, y1, 'o', ms=5)
    plt.plot(x2, y2, 'o', ms=5)
    plt.xlabel('$X_1$')
    plt.ylabel('$X^2$')
    plt.show()

plot_data(x1, y1, x2, y2)


# As we saw on the white board, the group of points on the left
# are the observations coming from class 1, and the orange points on the right
# are the observations coming from class 2.
# You'll see that the centers of these clouds of data points
# are more or less symmetrically located around X1 is equal to 0.
# But you should also see that the cloud on the right, the orange cloud,
# is broader than the blue cloud.
# And that's because we used to greater value for its standard deviation.









# 5.2.2: Logistic Regression

# In regression, our outcome variable was continuous.
# Now our goal is to predict a categorical outcome, such as blue or orange,
# or 0 or 1, so we're dealing with a classification problem.
# There are many different classifiers, techniques
# that carry out the classification task.
# But here we'll use one of the most basic ones, called logistic regression.
# The name of this technique is a little bit confusing,
# since it has the word regression as part of the name.
# But despite this name, it's a binary classifier,
# meaning that it is applied to classification settings
# where we have two categories.
# Our goal is to model the conditional probability
# that the outcome belongs to a particular class conditional on the values
# of the predictors.
# We can also call these conditional class probabilities.
# If we have only two classes, we can code the responses, the class labels,
# using 0s and 1s.
# So the outcome, y, is always either 0 or 1.

# We can write p of x as a shorthand for the probability
# p(x) = P(Y=1|X)
# that y is equal to 1 given the value of x.
# So this is a conditional probability.

# What is one of the problems with using linear regression to predict probabilities?
# Linear regression may predict values outside of the interval between 0 and 1.


def prob_to_odds(p):
    if p <= 0 or p >= 1:
        print("Probabilities must be between 0 and 1.")
    return p / (1-p)

prob_to_odds(0.8)







# 5.2.3: Logistic Regression in Code

from sklearn.linear_model import LogisticRegression

# Then second, we can set up or instantiate a classifier object.
# So we'll call LogisticRegression.
# And this simply generates the model object.
# The next step for us is to call the fit function on the clf object.
clf = LogisticRegression()
# Let's start with the y.
# These are the outcomes.
# And in this case, we have observations from two classes, 1 and 2.
# So we might have a bunch of 1's followed by a bunch of 2's.
# This would be our outcome vector y.
# The x is going to be a matrix where the different rows correspond
# to different observations.
# In this case, we have n observations or data points.
# And the columns here correspond to the values of the covariates,
# or the predictors.
X = np.vstack((np.vstack((x1, y1)).T, np.vstack((x2, y2)).T))
X.shape
n = 1000
y = np.hstack((np.repeat(1, n), np.repeat(2, n)))
y.shape
# same with concatenate and np.ones
# y = np.concatenate((np.ones(10, dtype=int), (np.ones(10, dtype=int) + 1)))

# So next thing we want to do is generate our test and training datasets.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.5, random_state=1)
X_train.shape
# Our final step is to fit the classifier.
# So we called our classifier clf.
# It supports the fit method.
# We train it with the training data.
# The first argument is the X data, the predictors,
# and the second one is the values of the outcomes.
clf.fit(X_train, y_train)
# One thing we can do is we can now find out what is the score--
# in other words, how well does the classifier perform.
clf.score(X_test, y_test)
# Remember, in the classification context, we're
# modeling conditional probabilities.
# So we can ask scikit-learn to give us estimates of these probabilities.
# To compute the estimated class probabilities,
# we will use the predict_proba function.
# And the argument has to be a np.array which has two covariates because we
# trained the model with two covariates.
# We can try to put into values minus 2 and 0.
clf.predict_proba(np.array([-2, 0]))
clf.predict_proba(np.array([[-2, 0]]))
clf.predict_proba(np.array([-2, 0]).reshape(1, -1))
# What the output is telling us is that there
# is a 0.97 probability that this particular test
# point belongs to class 1.
# And there is a 0.02 or 0.03 probability that it belongs to class 2.

# In addition to estimating these conditional probabilities,
# we can also ask our classifier to make a prediction.
clf.predict(np.array([[-2, 0]]))
# And in this case, we saw previously that the predicted probability
# was 0.97 for class 1.


# If you have data and want to train a model, which method would you use?
# clf.fit()

# If you want to compute the accuracy of your model, which method would you use?
# clf.score()

# If you want to estimate the probability of a data point being in each class, which method would you use?
# clf.predict_proba()

# If you want to know to which class your model would assign a new data point, which method would you use?
# clf.predict()








# 5.2.4: Computing Predictive Probabilities Across the Grid

# What we want to do next is to compute these probabilities
# at every point of the x1-x2 grid.

# So when we call meshgrid, we get two matrices at the output.
# The first output matrix is going to give us
# the x1 coordinate at all of these grid points,
# and the second one is going to give us the x2 coordinate at the same grid
# points.


def plot_probs(ax, clf, class_no):
    xx1, xx2 = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1))
    # np.ravel flattens the matrix
    # probs = clf.predict_proba(np.stack((xx1.ravel(), xx2.ravel()), axis=1))
    probs = clf.predict_proba(np.vstack((xx1.ravel(), xx2.ravel())).T)

    Z = probs[:, class_no]
    Z = Z.reshape(xx1.shape)
    CS = ax.contourf(xx1, xx2, Z)
    # cbar = plt.colorbar(CS)
    plt.colorbar(CS)
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")
    return probs


# plt.figure(figsize=(5,8))
ax = plt.subplot(211)
plot_probs(ax, clf, 0)
# The first time we call a plot probes function,
# we're estimating the probabilities that these different observations belong
# to class 0, the first class.
plt.title("Pred. prob for class 1")
ax = plt.subplot(212)
plot_probs(ax, clf, 1)
plt.title("Pred. prob for class 2")
plt.show()

# So in this case, blue corresponds to 0, meaning low probability, and yellow
# corresponds to a high probability.
# What this means is that if we are somewhere here
# in this part of the plot, it's very likely
# that this particular observation has a class probability 1 for class 1.
# And if we continue to move further to the right, the probability,
# the conditional probability that a given observation will belong to class 1
# will go to 0.














# Week 5: Statistical Learning/Part 3: Random Forest

# 5.3.1: Tree-Based Methods for Regression and Classification

"""
Random forest is a powerful method
for regression and classification.
We will next cover the conceptual foundations of random forests,
but we need to start from a simpler method first.
These simpler methods are called tree-based methods.
The random forest method makes use of several trees
when making its prediction, and since in graph theory, a collection of trees
is called a forest, this is where the random forest method gets its name.
In other words, to make a prediction, the random forest
considers the predictions of several trees.
It would not, however, be useful to have many identical trees
because all these trees would presumably give you the same prediction.
This is why the trees in the forest are randomized
in a way we'll come back to shortly.
Tree-based methods can be used for regression and classification.
These methods involve dividing the predictor space
into simpler regions using straight lines.
So we first take the entire predictor space and divide it into two regions.
We now in turn look at each of these two smaller regions,
divide them into yet smaller regions, and so on,
continuing until we hit some stopping criteria.
So the way we divide the predictor space into smaller regions
is recursive in nature.
To make a prediction for a previously unseen test observation,
we find the region of the predictor space where the test observation falls.
In the regression setting, we return the mean
of the outcomes of the training observations in that particular region,
whereas in a classification setting we return
the mode, the most common element of the outcomes of the training
observations in that region.
When we use lines to divide the predictor space into regions,
these lines must be aligned with the directions
of the axes of the predictor space.
And because of this constraint, we can summarize the splitting rules
in a tree.
This is also why these methods are known as decision tree methods.
In higher dimensions, these lines become planes,
so we end up dividing the predictor space into high-dimensional rectangles
or boxes.
How do we decide where to make these cuts?
The basic idea is that we'd like to carve out
regions in the predictor space that are maximally
homogeneous in terms of their outcomes.
Remember, we'll ultimately use the mean or the mode
of the outcomes falling in a given region as our predicted outcome
for an unseen observation.
So we can minimize error by finding maximally homogeneous regions
in the predictor space.
Whenever we make a split, we consider all predictors from x1 to xp,
and for each predictor, we consider all possible cut points.
We choose the predictor - cut point combination such
that the resulting division of the predictor space
has the lowest value of some criterion, usually called a loss function,
that we're trying to minimize.
In regression, this loss function is usually
RSS, the residual sum of squares.
In classification, two measures are commonly used,
called the Gini index and the cross-entropy.
You can find their definitions online, but the basic idea
is, again, to make cuts using a predictor
- cut point combination that makes the classes within each region
as homogeneous as possible.

Let's look at decision trees on the white board.
Let's think about a classification problem
where we have two covariates, or predictors, again, x1 and x2,
and we have a bunch of observations in this space.
Because it's a classification problem, each point
will have a class associated with it, either number 1 or number 2.
Let's add a couple of more points here.
The idea with decision trees is the following.
We are trying to split this space, the predictor space,
by forming regions that are maximally homogeneous.
So let's say we happen to have many 2's here and just one 1 there.
One option would be for us to split the predictor space here.
If our x1 ranges from 0 to 10, this might correspond
to a value of x1 is equal to 6.
What does the tree representation look like?
So at the top, we start from all of our data.
We split the predictor space into two regions.
In this region, x1 is greater than 6.
So it corresponds to the right branch.
And this part here, this region corresponds to a situation
where x1 is less than 6.
Once we've now split the space into two, we can proceed to make a further cut.
So for example, perhaps in this region over here,
which corresponds to this branch here, we
might choose to make a cut here for a given value of x2.
In the tree, we then introduce another branch.
Let's say that the value of x2 here happens to be 8.
And in this case, the cut is based on the value of x2.
So we go to the right.
If the value of x2 is greater than or equal to 8,
otherwise we go to the left branch, and so on.
So we continue to build these trees.
We continue splitting these regions by introducing cuts
in the predictor space.
We stop once we meet some stopping criterion.
At that point, we should have regions here
that consist mostly of 2's, and 1's, say 2's here and 1's over here.
How does a decision tree make a prediction?
Let's say we have observed a data point here,
which corresponds to a specific value of x2 and a specific value of x1.
If this is a classification problem, which it has been so far,
we first find the region where this data point falls.
In this case, it's this particular region here
that we're now highlighting in green.
Because it's a classification problem, we find the mode of these observations
here.
They all seem to belong to class two.
And that's why our prediction for this point is going to be equal to two.
If we were dealing with the regression problem,
we wouldn't have 1's and 2's as the outcomes,
but we would have some other measurements, like BMIs or incomes
or something like that.
To make a prediction in that setting, we would proceed just as before.
But instead of returning the mode, we would return the mean
of the observations in that region.
"""


# The goal of a tree-based method is typically to split up the predictor or feature space such that:
# data within each region are as similar as possible.

# For classification, how does a decision tree make a prediction for a new data point?
# It returns the mode of the outcomes of the training data points in the predictor space where 
# the new data point falls.

# For regression, how does a decision tree make a prediction for a new data point?
# It returns the mean of the outcomes of the training data points in the predictor space where 
# the new data point falls.






# 5.3.2: Random Forest Predictions

"""
We will next aggregate several trees
to form a forest of trees.
The prediction of the random forest combines information
from the predictions of the individual trees.
In the regression setting, the prediction of the random forest
is the mean of the predictions of the individual trees.
In a classification setting, the prediction of the random forest
is the mode of the predictions of the individual trees.
Random forests introduce two types of randomness to decision trees.
The first type has to do with introducing randomness to the data,
so that each tree is fit to a somewhat different dataset.
The second type of randomness has to do with which
predictors are considered when making a split at any point in a given tree.
These two steps have the implication of decorrelating
the trees, which ultimately gives us a more reliable method.
The first type of randomness, randomness in data,
is due to bootstrap aggregation, which is often called bagging.
Bootstrap is a re-sampling method, which involves repeatedly drawing samples
from a training set and refitting a model on each sample.
If we have n observations in our training data set,
we form a bootstrap dataset by randomly selecting n observations
with replacement from the original dataset.
Because the sampling is performed with replacement,
the same observation can occur multiple times in the bootstrap data.
We can perform this process multiple times,
and we'll likely get a somewhat different data set every time.
So bagging, in the context of decision trees,
means that we draw a number of bootstrap datasets and fit each to a tree.
The second type of randomness, randomness
in how we split the predictor space, happens as follows.
Normally with decision trees, we consider each predictor-cut point
combination when making a cut into predictor space.
In contrast, in random forest, each time we consider a split,
we don't look at all predictors, but instead
draw a small random sample of the predictors.
And now we're only allowed to use these predictors when making a split.
Each time we make a split, we take a new sample of predictors.
This sounds really strange, but it's actually a very effective trick.
Let's consider a simple example.
We start from some dataset having 1,000 observations,
and we have 9 predictors from x1 through x9.
We want to build, say, 50 trees.
So let's randomize the data first.
We first draw 50 bootstrap samples from the original data
and dedicate a separate tree for each dataset.
We then fit the trees one by one.
Starting from the first tree and the first cut,
we first determine which predictors to use.
If we're allowed to use, say, three predictors when making a cut,
we might be allowed to use x3, x7, and, say, x8 for the first cut.
We make the best cut we can given the data and these three predictors,
and we then move the second cut in the first tree.
This time we might be allowed to use predictors x1, x5, and x7.
And again, we find the best cut.
We proceed until we fit the first three, meaning until we fit
whatever stopping criterion we have.
We then continue the same way until we get all of the trees in the forest.
To make a prediction using a random forest,
we identify the region in the predicted space
separately for each tree, where the test observation happens to fall.
Based on this, we next have each tree make a separate prediction.
And we then combine the predictions of the individual trees
to form the prediction of the forest.
So this is how random forests work.
If you're not familiar with them from before,
you may want to read more about them.
Fortunately, using random forests in sklearn is easy.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

# Random forests get their name by introducing randomness to decision trees in two ways, 
# once at the data level and once at the predictor level.
# How is randomness at the data level introduced?
# Each tree gets a bootstrapped random sample of training data.

# How is randomness at the predictor level introduced?
# Each tree gets a bootstrapped random sample of training data.

# In a classification setting, how does a random forest make predictions?
# Each tree makes a prediction and the mode of these predictions is the prediction of the forest.

# In a regression setting, how does a random forest make predictions?
# Each tree makes a prediction and the mean of these predictions is the prediction of the forest.







# Week 5: Statistical Learning/Homework: Case Study 7, Part 1

# https://www.themoviedb.org/?language=en
# https://www.kaggle.com/tmdb/tmdb-movie-metadata

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

df = pd.read_csv("https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@movie_data.csv", index_col=0)

df.head(3)



# df.loc[df.revenue > df.budget, 'profitable'] = 1
# df.loc[df.revenue <= df.budget, 'profitable'] = 0
df['profitable'] = (df.revenue > df.budget).astype(int)
# df.profitable = df.profitable.astype(int)

regression_target = 'revenue'
classification_target = 'profitable'

# len(df.loc[df.profitable == 1])
# value_count() is like counter
df.profitable.value_counts()



df = df.replace([np.inf, -np.inf], np.nan)
# df = df.dropna(how="any")
df = df.dropna()



df = df.reset_index(drop=True)
genres = set()

for row in range(df.shape[0]):
    for genre in df.genres.loc[row].split(','):
        genres.add(genre.strip())

for genre in sorted(genres):
    df[genre] = df['genres'].str.contains(genre).astype(int)

df.head()




regression_target = 'revenue'
classification_target = 'profitable'

continuous_covariates = ['budget', 'popularity',
                         'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + \
    [regression_target, classification_target]
plotting_variables = ['budget', 'popularity', regression_target]

axes = pd.plotting.scatter_matrix(df[plotting_variables],
                                  alpha=0.15,
                                  color=(0, 0, 0),
                                  hist_kwds={"color": (0, 0, 0)},
                                  facecolor=(1, 0, 0)
                                  )
# plt.style.use('ggplot')
plt.show()

# determine the skew.

df[outcomes_and_continuous_covariates].skew()



ex6_cov = ['budget', 'popularity', 'runtime', 'vote_count', 'revenue']
for col in ex6_cov:
    df[col] = df[col].apply(lambda x: np.log10(x+1))

df[ex6_cov].skew()


# df.to_csv('movies_clean.csv')










# Week 5: Statistical Learning/Homework: Case Study 7, Part 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")

# EDIT THIS CODE TO LOAD THE SAVED DF FROM THE LAST HOMEWORK
path = 'edx/Network_Analysis'
df = pd.read_csv(os.getcwd()+'/'+path+'/'+'movies_clean.csv')



# Define all covariates and outcomes from `df`.
regression_target = 'revenue'
classification_target = 'profitable'
all_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average', 'Action', 'Adventure', 'Fantasy',
                  'Science Fiction', 'Crime', 'Drama', 'Thriller', 'Animation', 'Family', 'Western', 'Comedy', 'Romance',
                  'Horror', 'Mystery', 'War', 'History', 'Music', 'Documentary', 'TV Movie', 'Foreign']

regression_outcome = df[regression_target]
classification_outcome = df[classification_target]
covariates = df[all_covariates]

# Instantiate all regression models and classifiers.
linear_regression = LinearRegression()
logistic_regression = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)



def correlation(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return r2_score(y, predictions)
    
def accuracy(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    return accuracy_score(y, predictions)



# Determine the cross-validated correlation for linear and random forest models.
linear_regression_scores = cross_val_score(
    logistic_regression, covariates, classification_outcome, cv=10, scoring=correlation)
forest_regression_scores = cross_val_score(
    forest_classifier, covariates, classification_outcome, cv=10, scoring=correlation)

# Plot Results
plt.axes().set_aspect('equal', 'box')
plt.scatter(linear_regression_scores, forest_regression_scores)
plt.plot((0, 1), (0, 1), 'k-')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Linear Regression Score")
plt.ylabel("Forest Regression Score")

# Show the plot.
plt.show()




# Determine the cross-validated accuracy for logistic and random forest models.
logistic_regression_scores = cross_val_score(
    logistic_regression, covariates, classification_outcome, cv=10, scoring=accuracy)
forest_classification_scores = cross_val_score(
    forest_classifier, covariates, classification_outcome, cv=10, scoring=accuracy)

# Plot Results
plt.axes().set_aspect('equal', 'box')
plt.scatter(logistic_regression_scores, forest_classification_scores)
plt.plot((0, 1), (0, 1), 'k-')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Linear Classification Score")
plt.ylabel("Forest Classification Score")

plt.show()



positive_revenue_df = df[df.revenue > 0]

# Replace the dataframe in the following code, and run.

regression_outcome = positive_revenue_df[regression_target]
classification_outcome = positive_revenue_df[classification_target]
covariates = positive_revenue_df[all_covariates]

# Reinstantiate all regression models and classifiers.
linear_regression = LinearRegression()
logistic_regression = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
linear_regression_scores = cross_val_score(linear_regression, covariates, regression_outcome, cv=10, scoring=correlation)
forest_regression_scores = cross_val_score(forest_regression, covariates, regression_outcome, cv=10, scoring=correlation)
logistic_regression_scores = cross_val_score(logistic_regression, covariates, classification_outcome, cv=10, scoring=accuracy)
forest_classification_scores = cross_val_score(forest_classifier, covariates, classification_outcome, cv=10, scoring=accuracy)

np.mean(forest_regression_scores)




# Determine the cross-validated correlation for linear and random forest models.

# Plot Results
plt.axes().set_aspect('equal', 'box')
plt.scatter(linear_regression_scores, forest_regression_scores)
plt.plot((0, 1), (0, 1), 'k-')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Linear Regression Score")
plt.ylabel("Forest Regression Score")

# Show the plot.
plt.show()

# Print the importance of each covariate in the random forest regression.
forest_regression.fit(positive_revenue_df[all_covariates], positive_revenue_df[regression_target])    
sorted(list(zip(all_covariates, forest_regression.feature_importances_)), key=lambda tup: tup[1])




# Determine the cross-validated accuracy for logistic and random forest models.
logistic_regression_scores = cross_val_score(
    logistic_regression, covariates, classification_outcome, cv=10, scoring=correlation)
forest_classification_scores = cross_val_score(
    forest_classifier, covariates, classification_outcome, cv=10, scoring=correlation)

# Plot Results
plt.axes().set_aspect('equal', 'box')
plt.scatter(logistic_regression_scores, forest_classification_scores)
plt.plot((0, 1), (0, 1), 'k-')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Linear Classification Score")
plt.ylabel("Forest Classification Score")

# Show the plot.
plt.show()

# Print the importance of each covariate in the random forest classification.
forest_classifier.fit(
    positive_revenue_df[all_covariates], positive_revenue_df[classification_target])
sorted(list(zip(all_covariates, forest_classifier.feature_importances_)),
       key=lambda tup: tup[1], reverse=True)









