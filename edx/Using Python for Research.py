# Using Python for Researc; Harvard University


# All data in a Python program is represented by objects and by relationships between objects.

# Each object in Python has three characteristics. These characteristics are called object type, object value, 
# and object identity. 
# A type could be a number, or a string, or a list, or something else.

# The name of the attribute follows the name of the object.
# And these two are separated by a dot in between them.

# The two types of attributes are called either data attributes or methods.
object.attribute
object.function()

# Python modules are libraries of code 

# Well namespace is a container of names shared by objects that typically go together.

name = 'Amy'
type(name)
# dir function, to get a directory of the methods.
dir(name)
dir(str)
help(name.upper)
help(str)
name.upper()
## help(name.upper())
15 / 2.3
_ * 2.3

from math import factorial, sqrt
import random
import re
from typing import OrderedDict
from cartopy.crs import Projection
from matplotlib.pyplot import figure, legend, table
from networkx.algorithms.components.connected import connected_components
from networkx.convert import to_networkx_graph
import numpy

from numpy.core.defchararray import array, count, mod
from numpy.core.fromnumeric import argsort, reshape, shape

from hackerrank import H, average
factorial(6)

from random import choice
choice([2, 44, 55, 66])

# Expression is a combination of objects and operators that computes a value.

# boolean type values: True, Fales; boolean operations: "or", "and", and "not".

# These two comparisons are used to test whether two objects are the one and the same. is, is not.
# Notice that this is different than asking if the contents of two objects are the same. ==, !=
# What is the difference between == and is?
# == tests whether objects have the same value, whereas is tests whether objects have the same identity.

2.00 == 2

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

id(a)
id(b)
id(c)
id(d)


# Sequences
# In Python, a sequence is a collection of objects ordered by their position.
# lists, tuples, and so-called "range objects".
# But Python also has additional sequence types for representing things like strings.
# any sequence data type will support the common sequence operations. + generic sequence functions.
# in addition, these different types will have their own methods available for performing specific operations.
# : slices

# Lists are mutable sequences of objects of any type.
a = [1, 2]
b = [3, 4]
a + b
c = [a[i] + b[i] for i in range(len(a))]
c.reverse()
list(reversed(c))

# list methods are what are called in-place methods. They modify the original list.
c.sort(reverse=True)

# It will construct this new list using the objects
sorted(c, reverse=True)


# Tuples are immutable sequences typically used to store heterogeneous data.
coordinatesXY = [(55, 56), (12, 40)]
for x, y in coordinatesXY:
    print(x, y)

type(tuple((2)))
a = tuple([2])
type(a)
b = (2,)
type(b)

x = (1, 2, 3)
x.count(3)
sum(x)


# Ranges are immutable sequences of integers,
# Ranges take up less memory than lists because they do not hold all the numbers simultaneously.


# Strings are immutable sequences of characters.
"y" in "Python"

# polymorphism means that what an operator does depends on the type of objects it is being applied to.
"2" + "2"
"eight equals " + str(8)

dir(str)
str.replace?
help(str.replace)

name = "A Alan"
name.replace("A", "a")
names = name.split(" ")
" ".join([i.upper() for i in names])

from string import digits
digits

"22".isdigit()
help(str.isdigit)


# Sets are unordered collections of distinct hashable objects.
# In practice, what that means is you can use sets for immutable objects
# like numbers and strings, but not for mutable objects like lists and dictionaries.

# Frozen set is immutable
# One of the key ideas about sets is that they cannot be indexed. So the objects inside sets don't have locations.
# elements can never be duplicated.

ids = set([1, 3, 5])
ids.add(10)
ids.pop()

ids = set(range(10))
males = {1, 3, 5, 6, 7}

ids | males

word = "antidisestablishmentarianism"
word.count('a')
len(set(word))

from collections import Counter
Counter(word)

males.issubset(ids)

males in ids


# Dictionaries are mappings from key objects to value objects.
# Dictionaries consists of Key:Value pairs, where the keys must be immutable and the values can be anything.

age = {}
age = dict()

age = {"Tim": 29, "Jim": 31}
names = age.keys()
age["Tim"]
age["Tim"] += 1

age.keys()
type(age.keys())
age.values()
age.items()

age.update({"Tom": 50})
age["Tom2"] = 50
names

ages = age.values()

"Tim" in age
age[0] = 1




# Dynamic Typing
# Some languages are statically typed, like C or C++,
# and other languages are dynamically typed, like Python.

# Static typing means that type checking is performed during compile time, 
# whereas dynamic typing means that type checking is performed at run time.

# There are three important concepts here-- variable, object, and reference.
# A key point to remember here is that variable names always link to objects, never to other variables.
# Remember, mutable objects, like lists and dictionaries, can be modified at any point of program execution.
# In contrast, immutable objects, like numbers and strings, cannot be altered after they've been created in the program.

# immutable
x = 3
y = x
y -= 1
x

# mutable
# And third, because of the assignment, L1 will reference this list.
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



# Copies
# A shallow copy constructs a new compound object and then insert its references into it to the original object.
# a deep copy constructs a new compound object and then recursively inserts copies into it of the original objects.

import copy
x = [1,[2]]
y = copy.copy(x)
z = copy.deepcopy(x)
y is z


# Statements
# Statements are used to compute values, assign values, and modify attributes, among many other things.
# The return statement is used to return values from a function.
# import statement, which is used to import modules.
# the pass statement is used to do nothing in situations where we need a placeholder for syntactical reasons.

# Compound statements contain groups of other statements,
# A compound statement consists of one or more clauses, where a clause consist of a header and a block or a suite of code.
# The close headers of a particular compound statement start with a keyword, end with a colon, and are
# all at the same indentation level.
# A block or a suite of code of each clause, however, must be indented to indicate that it forms a group of statements
# that logically fall under that header.



# For and While Loops
n = 100
n //= 2

n=100
number_of_times = 0
while n >= 1:
    n //= 2
    number_of_times += 1
print(number_of_times)


# List Comprehensions
# One is list comprehensions are very fast.
# The second reason is list comprehensions are very elegant.

sum([i for i in range(10) if i%2 == 1])



# Reading and Writing Files
# reading
path = "/home/ukasz/Documents/Programowanie/Python/edx/"
path2 = "edx/"

for line in open(path2+"input.txt"):
    print(line)

for line in open(path2+"input.txt"):
    line = line.rstrip().split()
    print(line)

# writing
F = open(path2+"output.txt", "w")
F.write("Python\n")
F.close()

# print working directory
from os import getcwd, chdir
getcwd()
chdir('/home/ukasz/Documents/Programowanie/Python/edx')

F2 = open(path2+"output2.txt", "w+")
F2.write("Hello\nWolrd\n")
F2.close()
lines = []
for line in open(path2+"output2.txt"):
    lines.append(line.strip())
print(lines)


# Functions
# Functions are devices for grouping statements so that they can be easily run more than once in a program.
# Functions enable dividing larger tasks into smaller chunks, an approach that is called procedural decomposition.

"".join([str(elem) for elem in list(range(5))])
type("") == str
4 in 'ae'

def is_vowel(letter):
    if type(letter) == int:
        letter = str(letter)
    if letter in "aeiouy":
        return(True)
    else:
        return(False)

is_vowel(4)


# Week 1 Homework: Exercise 1
from string import ascii_letters
ascii_letters

sentence = 'Jim quickly realized that the beautiful gowns are expensive'
from collections import Counter
Counter(sentence).most_common
sorted(Counter(sentence).items())[2:][8]

address = """Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this. But, in a larger sense, we can not dedicate -- we can not consecrate -- we can not hallow -- this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us -- that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion -- that we here highly resolve that these dead shall not have died in vain -- that this nation, under God, shall have a new birth of freedom -- and that government of the people, by the people, for the people, shall not perish from the earth."""
address.count('h')
Counter(address).most_common

from math import pi
round(pi/4, 6)

from random import uniform, seed
seed(1)
uniform(-1, 1)

def in_circle(x, origin = [0,0]):

def gen_data(length):
    seed(1)
    points = [uniform(0, 1) for _ in range(length)]
    return points 

gen_data(5)

from statistics import mean
def aver(entry, nei):
    seed(1)
    points = [uniform(0, 1) for _ in range(20)]
    return mean(points[entry - nei:entry + nei + 1])


aver(10, 5)







# Scope rules
# LEGB Local, Enclosing Function, Global, Built-in

# An argument is an object that is passed to a function as its input when the function is called.
# A parameter, in contrast, is a variable that is used in the function definition to refer to that argument.


# Classes and Object-Oriented Programming
x = [5, 2, 9, 11, 10, 2, 7]

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

type(y)
dir(x)
dir(y)

y.remove_min()
y.remove_nth(1)
y.append_sum()



# Introduction to NumPy Arrays
import numpy as np
np.zeros(5) + np.ones(5)
np.zeros((5, 3))
# we assume that lower case variables are vectors or one-dimensional arrays and upper case variables are
# matrices, or two-dimensional arrays.
np.array([[1, 2], [3, 4]]).transpose()

x = np.array([1, 2, 3])
y = np.array([2, 4, 6])
X = np.array([[1, 2, 3], [4, 5, 6]])
Y = np.array([[2, 4, 6], [8, 10, 12]])
X.shape
X.size

list(X[:, 1])
X[1, ]

ind = [1, 2]
y[ind]
y[[1, 2]]
ind_np = np.array([1, 2])
y[ind_np]

y > 4
x[x > 2]

# When you slice an array using the colon operator, you get a view of the object.
# This means that if you modify it, the original array will also be modified.
# This is in contrast with what happens when you index an array, in which case
# what is returned to you is a copy of the original data.
# In summary, for all cases of indexed arrays, what is returned
# is a copy of the original data, not a view as one gets for slices.

a = np.array([1,2])
b = np.array([3,4,5])

c = b[1:]
b[a] is c
all(b[a] == c)

np.linspace(1, 100, 10)
np.logspace(1, 2, 10)
np.logspace(np.log10(250), np.log10(500), 10)

X = np.array([[1, 2, 3], [4, 5, 6]])
X.shape
X.size

x = np.random.random(10)
np.any(x > 0.9)
np.all(x >= 0.1)




# Matplotlib and Pyplot
import matplotlib.pyplot as plt
import numpy as np

plt.plot([0, 1, 4, 9, 16]);
plt.show()

x = np.linspace(0, 10, 20)
y1 = x**2
y2 = x**1.5
plt.plot(x, y1)
plt.show()

# a keyword argument is an argument which is supplied to the function by explicitly naming each parameter
# and specifying its value.

plt.plot([0,1,2],[0,1,4],"rd-")

plt.plot(x, y1, "bo-", linewidth=2, markersize=4, label="First")
plt.plot(x, y2, "gs-", linewidth=2, markersize=4, label="Second")
plt.xlabel("$X$")
plt.ylabel("$Y$")
plt.axis([-.5, 10.5, -5, 105])
plt.legend(loc="upper left")
# plt.savefig("myplot.png")
# plt.savefig("myplot.pdf")
plt.show()


# So the lesson here is that functions of the form y is equal to x to power alpha
# show up as straight lines on a loglog() plot.
# The exponent alpha is given by the slope of the line.


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



# Histograms
import matplotlib.pyplot as plt
import numpy as np

x = np.random.normal(size=1000)
plt.hist(x, density=True, bins=np.linspace(-5, 5, 21))
plt.show()


x = np.random.gamma(2, 3, 100000)
plt.figure()
plt.subplot(231)
plt.hist(x, bins=30)
plt.subplot(232)
plt.hist(x, bins=30, density=True)
plt.subplot(234)
plt.hist(x, bins=30, cumulative=True)
plt.subplot(235)
plt.hist(x, bins=30, density=True, cumulative=True, histtype="step")
plt.show()





# Simulating Randomness
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

import matplotlib.pyplot as plt
rolls = [random.choice(range(1, 7)) for _ in range(100)]
plt.hist(rolls, density=True, bins=np.linspace(.5, 6.5, 7), ec="black")
plt.show()

x = sum([random.choice(range(1, 7)) for _ in range(10)])
y = [sum([random.choice(range(1, 7)) for _ in range(10)]) for _ in range(10000)]
plt.hist(y, density=True, bins=np.linspace(5, 65, 25), ec="black")
plt.show()

# generate a matrix with rand
np.random.random((2, 5))
np.random.normal(0, 1, 5)
np.random.normal(0, 1, (2, 5))

X = np.random.randint(1, 7, (1000000, 10))
# X.shape
# X.sum(1,)

# help(sum)
# help(np.sum)
Y = np.sum(X, axis=1)
plt.hist(Y, density=True, bins=np.linspace(5, 65, 25), ec="black")
plt.show()



# Measuring Time
import time
import matplotlib.pyplot as plt
import random
import numpy as np

start_time = time.time()
y = [sum([random.choice(range(1, 7)) for _ in range(10)]) for _ in range(1000000)]
plt.hist(y, density=True, bins=np.linspace(5, 65, 25), ec="black")
plt.show()
time.time() - start_time

start_time = time.time()
X = np.random.randint(1, 7, (1000000, 10))
Y = np.sum(X, axis=1)
plt.hist(Y, density=True, bins=np.linspace(5, 65, 25), ec="black")
time.time() - start_time
plt.show()



# Random Walks
import numpy as np
import matplotlib.pyplot as plt

delta_X = np.random.normal(0, 1, (2, 10000))
X = np.cumsum(delta_X, axis=1)
# X_0 = np.array([[0] + list(np.cumsum(delta_X[0])), [0] + list(np.cumsum(delta_X[1]))])
X_0 = np.concatenate(([[0], [0]], X), axis=1)
plt.plot(X_0[0], X_0[1], "rv-")
plt.show()

x = ['ab', 'cd']
for i in x:
    i = i.upper()
print(x)

for i in range(2):
    x[i] = x[i].upper()
print(x)






# Using Python for Research Homework: Week 2

import numpy as np

def create_board():
    board = np.zeros((3, 3), dtype=int)
    return board

board = create_board()



def place(board, player, position):
    if player in [1, 2] and board[position] == 0:
        board[position] = player
    return board

place_1 = place(board, 1, (0, 0))



def possibilities(board):
    return [(i, j) for i in range(3) for j in range(3) if board[(i, j)] == 0]
    # return np.where(board == 0, board, 100)

possibilities(place_1)



import random 
# random.seed(1)

def random_place(board, player):
    choice = random.choice(possibilities(board))
    # board[choice] = player
    place(board, player, choice)
    return board

random_place(place_1, 2)



# random.seed(1)
board = create_board()

def run_game(board):
    for i in range(1, 7):
        if i%2 != 0:
            random_place(board, 1)
        if i%2 == 0:
            random_place(board, 2)
    # random_place(board, 2) if i%2 == 0 else random_place(board, 1) for i in range(1, 7)
    return board

game_played = run_game(board)


import numpy as np

def row_win(board, player):
    # return any([all([board[(i, j)] == player for i in range(3)]) for j in range(3)])
    return np.any(np.all(board == player, axis=1))
     

row_win(game_played, 1)



def col_win(board, player):
    # return any([all([board[(i, j)] == player for i in range(3)]) for j in range(3)])
    return np.any(np.all(board == player, axis=0))
     

row_win(game_played, 1)



# board[1,1] = 2
def diag_win(board, player):
    return np.all(board.diagonal() == player) or np.all(np.fliplr(board).diagonal() == player)
    #return board

diag_win(game_played, 2)



def evaluate(board):
    winner = 0
    for player in [1, 2]:
        # add your code here!
        if row_win(board, player) or col_win(board, player) or diag_win(board, player):
            winner = player
        pass
    if np.all(board != 0) and winner == 0:
        winner = -1
    return winner

evaluate(board)



# random.seed(1)
def play_game(games):

    results = []
    for _ in range(games):
        board = create_board()
        bo = board
        for _ in range(4):
            for player in [1, 2]:
                random_place(board, player)
        random_place(board, 1)
        results.append(evaluate(board))

    # return results.count(1)
    return results.count(1)

random.seed(1)
game(1000)






# ncbi nuceleotide NM_207618.2
path = "edx/translation/"

for line in open(path+"dna.txt"):
    print(line.strip())

import os
os.getcwd()
os.chdir(path)

def read_seq(input):
    '''Reads and removes special characters from input file'''
    with open(input, "r") as F:
        seq = F.read()
    seq = seq.replace("\n", "").replace("\r", "")
    return seq

read_seq("dna.txt")
prt = read_seq("protein.txt")
dna = read_seq("dna.txt")


seq
print(seq)

# delete "\n"
"".join(seq.split("\n"))
seq.replace("\n", "")
dir(str)
seq = seq.replace("\n", "").replace("\r", "")


import json
with open("table.json", "r") as F2:
    table = F2.read()

type(table)
table_js = json.loads(table)


table_js["ATA"]
type(table_js)

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

def DNA_to_aminoacid(seq):
    """
    Translates DNA to aminoacid.
    """

    protein = ''
    if True: #len(seq)%3 == 0:
        # for i in range(len(seq)//3):
        #     print(seq[3*i:3 * (i+1)])
        for i in range(0, len(seq), 3):
            protein += table_js[seq[i:i+3]]
    return protein

help(DNA_to_aminoacid)
DNA_to_aminoacid('ATAGGCAAA')
DNA_to_aminoacid(seq)
DNA_to_aminoacid(soeq[:len(seq) - 2])

DNA_to_aminoacid(dna[20:935]) == prt 
DNA_to_aminoacid(dna[20:938])[:-1] == prt 
DNA_to_aminoacid(dna[20:23])

dna[935:938]
DNA_to_aminoacid(dna[935:938])
dna[68:71]


# Using Python for Research Homework: Week 3, Case Study 1 A cipher

import string

alphabet = " " + string.ascii_lowercase
positions = {alphabet[i]:i for i in range(len(alphabet))}

message = "hi my name is caesar"
key = 1

def encode(message, key):
    encoded_message = ""
    for i in message:
        val = positions[i]
        encoded_message += alphabet[(val + key)%27]
    return encoded_message

encode(message, key)



# Introduction to Language Processing
from collections import Counter

text = 'Some sample of some text. Another sample.'
text = "This comprehension check is to check for comprehension."
def count_words_al(text):
    text = text.replace('.', '').lower()
    return Counter(text.split())

count_words_al(text)


def count_words(text):
    '''Count the number of words provied string. Skip punctuation.'''

    text = text.lower()

    for punctuation in ['.', ',', ';', ':', '"', "'"]:
        text = text.replace(punctuation, '')

    word_counts = {}
    for word in text.split():
        word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts

count_words(text)


def count_words_fast(text):
    '''Count the number of words provied string. Skip punctuation. With from collections import Counter'''
    text = text.lower()
    for punctuation in [".", ",", ";", ":", "'", '"', "\n", "!", "?", "(", ")"]:
        text = text.replace(punctuation, '')
    return Counter(text.split())

count_words_fast(text)


import os
os.getcwd()
os.chdir("/home/ukasz/Documents/Programowanie/Python/")
path = 'edx/Language_Processing/'


def read_book(title_path):
    """
    Read a book from a file and return it as a string.
    """
    with open(path+title_path, "r", encoding="utf8") as current_file:
    # with webbrowser.open(title_path) as current_file:
        text = current_file.read()
        text = text.replace("\n", "").replace("\r", "")
    return text


# read_book("https://www.gutenberg.org/cache/epub/2261/pg2261.txt")
read_book("romeo.txt")



import webbrowser
def read_book_web(title_path):
    """Open url in browser."""
    with webbrowser.open(title_path) as current_file:
        text = current_file.read()
        text = text.replace("\n", "").replace("\r", "")
    return text

read_book_web("https://www.gutenberg.org/cache/epub/2261/pg2261.txt")


# from https://www.kite.com/python/answers/how-to-read-a-text-file-from-a-url-in-python
url = "https://www.gutenberg.org/cache/epub/2261/pg2261.txt"
file = urllib.request.urlopen(url)

for line in file:
	decoded_line = line.decode("utf-8")
	print(decoded_line)


import urllib.request
def read_book_url(title_path):
    """
    Read a book from url and return it as a string.
    """
    decoded_line = ''
    for line in urllib.request.urlopen(title_path):
        decoded_line += line.decode("utf8")

    text = decoded_line
    text = text.replace("\n", "").replace("\r", "")
    return text

read_book_url("https://www.gutenberg.org/cache/epub/2261/pg2261.txt")


# urllib.request.urlopen doesn't work with with
import urllib.request
def read_book_url2(title_path):
    """
    Read a book from url and return it as a string.
    """

    with urllib.request.urlopen(title_path) as decoded_line:
        text = decoded_line("utf8")
        text = text.replace("\n", "").replace("\r", "")
    return text

read_book_url2("https://www.gutenberg.org/cache/epub/2261/pg2261.txt")

len(read_book("romeo.txt"))
len(read_book_url("https://www.gutenberg.org/cache/epub/2261/pg2261.txt"))
read_book("romeo.txt")[0]
read_book_url("https://www.gutenberg.org/cache/epub/2261/pg2261.txt")[1]

romeo_book = read_book_url("https://www.gutenberg.org/cache/epub/2261/pg2261.txt")[1:]
ind = romeo_book.find("By any other")
romeo_book[ind - 50:ind + 10]



def word_stats(word_counts):
    """Returns number of unique words and word frequencies"""
    num_uniq = len(word_counts)
    counts = word_counts.values()
    return (num_uniq, counts)

word_stats(count_words_fast(romeo_book))

romeo_book_de = read_book_url("https://www.gutenberg.org/cache/epub/6996/pg6996.txt")[1:]
word_stats(count_words_fast(romeo_book_de))
romeo_book_pl = read_book_url("https://www.gutenberg.org/files/27062/27062-0.txt")[1:]
word_stats(count_words_fast(romeo_book_pl))



import pandas as pd

table = pd.DataFrame(columns=("Name", "Age"))
table.loc[1] = "James", 25
table.loc[2] = "Jess", 22
table.append[3] = 'Tom', 40
table
table.columns
table.Name
table["Name"]

df0 = pd.DataFrame(columns=('A', 'B'))
df = pd.DataFrame([[1, 2], [3, 4]], columns=('A', 'B'), index=(1, 2))
df2 = pd.DataFrame([[5, 6]], columns=('A', 'B'))
df0.append(df, ignore_index=True)
df.append(df2, ignore_index=True)



import os
os.getcwd()
path = 'edx/Language_Processing/'
import pandas as pd

stats = pd.DataFrame(columns=("language", "author", "title", "length", "unique"))
title_num = 1


for language in os.listdir("edx/Language_Processing"):
    for author in os.listdir("edx/Language_Processing/"+language):
        for title in os.listdir("edx/Language_Processing/"+language+"/"+author):
            input_file = language+"/"+author+"/"+title
            print(input_file)
            text = read_book(input_file)
            (num_uniq, counts) = word_stats(count_words_fast(text))
            stats.loc[title_num] = language, author.capitalize(), title[:-4], sum(counts), num_uniq
            title_num += 1

stats.head()


os.listdir(os.getcwd()+"/edx/Language_Processing")
stats.length
stats.unique

import matplotlib.pyplot as plt

plt.plot(stats.length, stats.unique, "bo")
plt.loglog(stats.length, stats.unique, "bo")
plt.show()
stats[stats.language == "French"]

plt.figure(figsize=(10, 10))
subset = stats[stats.language == "English"]
plt.loglog(subset.length, subset.unique, "o", label="English", color="crimson")
subset = stats[stats.language == "French"]
plt.loglog(subset.length, subset.unique, "o", label="French", color="forestgreen")
subset = stats[stats.language == "German"]
plt.loglog(subset.length, subset.unique, "o", label="German", color="orange")
subset = stats[stats.language == "Portuguese"]
plt.loglog(subset.length, subset.unique, "o", label="Portuguese", color="blueviolet")
plt.legend()
plt.xlabel("Book length")
plt.ylabel("Number of unique words")
plt.savefig(path+"book_statistics.png")
plt.show()

with open(path+"books_statistics.csv", "w+") as F2:
    F2.write(stats.to_csv(index=False))


F2 = open(path+"books_statistics.csv", "w+")
F2.write(stats.to_csv(index=False))
F2.close()




# Homework: Week 3, Case Study 2
# def count_words_fast(text):
# def word_stats(word_counts):

import pandas as pd
from collections import Counter
path = '/home/ukasz/Documents/Programowanie/Python/edx/Language_Processing/hamlets.csv'
# hamlets = pd.read_csv(path, index_col=False)
hamlets = pd.read_csv("https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@hamlets.csv", index_col=0)
print(hamlets)

language, text = hamlets.iloc[0]

# data = pd.DataFrame(dict(count_words_fast(text)).items(), columns=("word", "count"))
data = pd.DataFrame({
    "word": count_words_fast(text).keys(),
    "count": count_words_fast(text).values()
})

print(data)


# data["length"] = list(map(len, data["word"]))
# data["length"] = data["word"].apply(lambda x: len(x))
data["length"] = data["word"].apply(len)
# below doesn't work
# data.length = len(data.word)

# data["frequency"] = data["count"].apply(lambda x: "unique" if x == 1 else ("frequent" if x > 10 else "infrequent"))
data.loc[data["count"] > 10, "frequency"] = "frequent"
data.loc[data["count"] <= 10, "frequency"] = "infrequent"
data.loc[data["count"] == 1, "frequency"] = "unique"


sub_data = pd.DataFrame({
    "language": language,
    "frequency": ("frequent", "infrequent", "unique"),
    "mean_word_length": data.groupby(by="frequency")["length"].mean(),
    # "mean_count": data.groupby(by="frequency")["count"].mean(),
    #"num_words": data.groupby("frequency")["word"].count()
    "num_words": data.groupby(by="frequency").size()
})

print(sub_data)


def summarize_text(language, text):
    counted_text = count_words_fast(text)

    data = pd.DataFrame({
        "word": list(counted_text.keys()),
        "count": list(counted_text.values())
    })
    
    data.loc[data["count"] > 10,  "frequency"] = "frequent"
    data.loc[data["count"] <= 10, "frequency"] = "infrequent"
    data.loc[data["count"] == 1,  "frequency"] = "unique"
    
    data["length"] = data["word"].apply(len)
    
    sub_data = pd.DataFrame({
        "language": language,
        "frequency": ["frequent","infrequent","unique"],
        "mean_word_length": data.groupby(by = "frequency")["length"].mean(),
        "num_words": data.groupby(by = "frequency").size()
    })
    
    return(sub_data)

# print(summarize_text(language, text))
    
# write your code here!
summarize_text(language, text)

def grouped_data_fun(hamlets):
    sum_data = pd.DataFrame(columns=("language", "frequency", "mean_word_length", "num_words"))
    for i in range(3):
        language, text = hamlets.iloc[i]
        # print(summarize_text(language, text))
        sum_data = sum_data.append(summarize_text(language, text))
    return sum_data

print(grouped_data_fun(hamlets))

grouped_data = grouped_data_fun(hamlets)



colors = {"Portuguese": "green", "English": "blue", "German": "red"}
markers = {"frequent": "o","infrequent": "s", "unique": "^"}
import matplotlib.pyplot as plt
for i in range(grouped_data.shape[0]):
    row = grouped_data.iloc[i]
    plt.plot(row.mean_word_length, row.num_words,
        marker=markers[row.frequency],
        color = colors[row.language],
        markersize = 10
    )

color_legend = []
marker_legend = []
for color in colors:
    color_legend.append(
        plt.plot([], [],
        color=colors[color],
        marker="o",
        label = color, markersize = 10, linestyle="None")
    )
for marker in markers:
    marker_legend.append(
        plt.plot([], [],
        color="k",
        marker=markers[marker],
        label = marker, markersize = 10, linestyle="None")
    )
plt.legend(numpoints=1, loc = "upper left")

plt.xlabel("Mean Word Length")
plt.ylabel("Number of Words")
plt.show()














# Introduction to kNN Classification
# In what is often called supervised learning, the goal is to estimate or predict an output based on one or more inputs.
# The inputs have many names, like predictors, independent variables, features, and variables being called common.
# The output or outputs are often called response variables, or dependent variables.
# If the response is quantitative(ilośćiowy)-- say, a number that measures weight or height, we call these problems regression problems.
# If the response is qualitative(jakościowy)-- say, yes or no, or blue or green, we call these problems classification problems.

import numpy as np

def distance(p1, p2):
    """Find the distance between two points"""
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))

p1 = np.array([1, 1])
p2 = np.array([4, 4])
distance(p1, p2)



from collections import Counter
import random

def majority_vote(votes):
    """Return most common element"""
    winners = []
    vote_dict = dict(Counter(votes))
    for vote, count in vote_dict.items():
        if count == max(vote_dict.values()):
            winners.append(vote)
    return random.choice(winners)


def majority_vote(votes):
    vote_dict = dict(Counter(votes))
    max_vote_dict = max(vote_dict.values())
    return random.choice([vote for vote, count in vote_dict.items() if count == max_vote_dict])


# start_time = time.time()
votes = [1, 1, 2, 2, 2, 3, 3, 3]
# votes = [random.randint(0, 5) for _ in range(1000000)]
winner = majority_vote(votes)
winner
# time.time() - start_time


from scipy import stats
def majority_vote_mode(votes):
    """Return most common element using mode"""
    mode, _ = stats.mode(np.array(votes))
    return int(mode)

majority_vote_mode(votes)



points = np.array([[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]])
p = np.array([2.5, 2])

def find_nearest_neighbors(p, points, k=5):
    """Find k nearest neighbors"""
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    return np.argsort(distances)[:k]

def find_nearest_neighbors(p, points, k=5):
    """Find k nearest neighbors"""
    return np.argsort([distance(p, point) for point in points])[:k]

ind = find_nearest_neighbors(p, points, 3)
print(points[ind])



import matplotlib.pyplot as plt
for point in np.array(list([p]) + list(points)):
    plt.plot(point[0], point[1], "ro")


plt.plot(points[:, 0], points[:, 1], "ro")
plt.plot(p[0], p[1], "bo")
plt.axis(0, 3.5, 0, 3.5)
plt.show()


def knn_predict(p, points, outcomes, k=5):
    ind = find_nearest_neighbors(p, points, k)
    return majority_vote(outcomes[ind])


outcomes = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])

knn_predict(np.array([2.5, 2.7]), points, outcomes, 2)
knn_predict(np.array([1.0, 2.7]), points, outcomes, 2)



stats.norm().rvs((5, 2))
np.array(list(np.random.normal(size=(5, 2))) + list(np.random.normal(1, 1, size=(5, 2))))
np.repeat(0, 5)

def generate_synth_data(n=50):
    """Generate two sets of points from bivariate normal distribution"""
    points = np.concatenate((np.random.normal(size=(n, 2)), np.random.normal(1, 1, size=(n, 2))))
    outcomes = np.concatenate((np.zeros(n, dtype=int), np.ones(n, dtype=int)))
    return(points, outcomes)

points, outcomes = generate_synth_data(20)

import matplotlib.pyplot as plt
plt.plot(points[:20, 0], points[:20, 1], "ro")
plt.plot(points[20:, 0], points[20:, 1], "bo")
plt.show()



def make_prediction_grid(predictors, outcomes, limits, h, k):
    x_min, x_max, y_min, y_max = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)

    prediction_grid = np.zeros(xx.shape, dtype=int)
    """Classify each point on the prediction grid."""
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x, y])
            prediction_grid[j, i] = knn_predict(p, predictors, outcomes, k)
    return (xx, yy, prediction_grid)

# np.meshgrid(np.array([6, 2]), np.array([3, 9, 5, 6]))

xvalues = np.array([0, 1, 2, 3, 4]);
yvalues = np.array([0, 1, 2, 3, 4]);
np.meshgrid(xvalues, yvalues)
xx, yy = np.meshgrid(xvalues, yvalues)



def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5, shading='auto')
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    # plt.savefig("edx/knn/"+filename)
    plt.show()



predictors, outcomes = generate_synth_data()
predictors.shape
outcomes.shape

k = 5; filename = "knn_synth_5.png"; limits = (-3, 4, -3, 4); h = 0.1
xx, yy, prediction_grid = make_prediction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)



from sklearn import datasets
iris = datasets.load_iris()
iris["data"]
predictors = iris.data[:, :2]
outcomes = iris.target

plt.plot(predictors[outcomes == 0][:, 0], predictors[outcomes == 0][:, 1], "ro")
plt.plot(predictors[outcomes == 1][:, 0], predictors[outcomes == 1][:, 1], "go")
plt.plot(predictors[outcomes == 2][:, 0], predictors[outcomes == 2][:, 1], "bo")
plt.show()

k = 5; filename = "iris_grid.png"; limits = (4, 8, 1.5, 4.5); h = 0.1
xx, yy, prediction_grid = make_prediction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(predictors, outcomes)
sk_predictions = knn.predict(predictors)

my_predictions = np.array([knn_predict(p, predictors, outcomes, 5) for p in predictors])
np.mean(sk_predictions == my_predictions)

np.mean(sk_predictions == outcomes)
np.mean(my_predictions == outcomes)




# Homework: Week 3, Case Study 3




















# Week 4 Classifying Whiskies
import pandas as pd

x = pd.Series([6, 3, 8, 6], index=['q', 'w', 'e', 'r'])
x['q']
x[['q', 'w']]
x.index
x.reindex(sorted(x.index))

age = {"Tim": 29, "Jim": 31, "Pam": 27, "Sam": 35}
x = pd.Series(age)

y = pd.Series([7, 3, 5, 2], index=['e', 'q', 'r', 't'])
x+y
# NaN not a number


data = {'name': ['Tim', 'Jim', 'Pam', 'Sam'],
        'age': [29, 31, 27, 35],
        'Zip': ['02115', '02130', '67700', '00100']}
x = pd.DataFrame(data, columns=['name', 'age', 'Zip'])
x['name']
x.name



import pandas as pd
import numpy as np
import os

os.getcwd()
os.chdir('/home/ukasz/Documents/Programowanie/Python'+'/edx/whiskies')

whisky = pd.read_csv('whiskies.txt')
whisky['Region'] = pd.read_csv('regions.txt')
whisky.head()
whisky.tail()
whisky.iloc[5:10, 0:5]
whisky.columns

flavors = whisky.iloc[:, 2:14]
corr_flavors = pd.DataFrame.corr(flavors)
import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 10))
plt.pcolor(corr_flavors, cmap='jet')
plt.colorbar()
plt.show()


corr_whisky = pd.DataFrame.corr(flavors.transpose())
plt.pcolor(corr_whisky, cmap='jet')
plt.axis('tight')
plt.colorbar()
plt.show()


from sklearn.cluster._bicluster import SpectralCoclustering

model = SpectralCoclustering(n_clusters=6, random_state=0)
model.fit(corr_whisky)
model.rows_
np.sum(model.rows_, axis=1)
np.sum(model.rows_, axis=0)
model.row_labels_



whisky.head()
whisky['Group'] = pd.Series(model.row_labels_, index=whisky.index)
whisky = whisky.iloc[np.argsort(model.row_labels_)]
whisky = whisky.reset_index(drop=True)
correlations = np.array(pd.DataFrame.corr(whisky.iloc[:, 2:14].transpose()))

plt.subplot(121)
plt.pcolor(corr_whisky, cmap='jet')
plt.title('Original')
plt.axis('tight')
plt.subplot(122)
plt.pcolor(correlations, cmap='jet')
plt.title('Rearranged')
plt.axis('tight')
plt.show()

from itertools import repeat
list(map(pow, range(10), 2))
list(map(pow, range(10), repeat(3)))

[1, 2] * 5
list(range(1, 5)) * 5









# GPS
import pandas as pd

birddata = pd.read_csv('https://courses.edx.org/assets/courseware/v1/6184eb0f87c7b58db1a5c336e436ed09/asset-v1:HarvardX+PH526x+2T2021+type@asset+block/bird_tracking.csv')
birddata.info()
pd.set_option('display.max_columns', None)
birddata.head()

import matplotlib.pyplot as plt
import numpy as np

x, y = birddata.longitude[birddata.bird_name == 'Eric'], birddata.latitude[birddata.bird_name == 'Eric']
plt.plot(x, y, '.')
plt.show()

bird_names = pd.unique(birddata.bird_name)
for bird_name in bird_names:
    x, y = birddata.longitude[birddata.bird_name == bird_name], birddata.latitude[birddata.bird_name == bird_name]
    plt.plot(x, y, '.', label=bird_name)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc='lower right')
plt.show()


birddata.columns
speed = birddata.speed_2d[birddata.bird_name == 'Eric']
plt.hist(speed)
plt.hist(speed[:10])
plt.show()
np.isnan(speed)
any(np.isnan(speed))
np.isnan(speed).any()
np.sum(np.isnan(speed))
speed[-np.isnan(speed)]

plt.hist(speed[~np.isnan(speed)], bins=np.linspace(0, 30, 20), density=True)
# plt.axis([0, 25, 0, .4])
plt.axis('tight')
plt.xlabel('2D speed (m/s)')
plt.ylabel('Frequency')
plt.show()

birddata.speed_2d.plot(kind='hist', range=[0, 30])
plt.show()



birddata.date_time
import datetime

time1 = datetime.datetime.today()
time2 = datetime.datetime.today()
time2 - time1

date_str = birddata.date_time[0]
type(date_str)
datetime.datetime.strptime(date_str[:-3], '%Y-%m-%d %H:%M:%S')
timestamps = [datetime.datetime.strptime(birddata.date_time[i][:-3], \
'%Y-%m-%d %H:%M:%S') for i in range(len(birddata))]
timestamps[:3]
birddata['timestamp'] = pd.Series(timestamps, index=birddata.index)
birddata.head()
birddata.timestamp[4] - birddata.timestamp[3]

times = birddata.timestamp[birddata.bird_name == 'Eric']
elapsed_time = [time - times[0] for time in times]
elapsed_days = np.array(elapsed_time) / datetime.timedelta(days=1)
elapsed_time[1000] / datetime.timedelta(days=1)
elapsed_time[1000] / datetime.timedelta(days=1)

plt.plot(np.array(elapsed_time) / datetime.timedelta(hours=1))
plt.xlabel('Observation')
plt.ylabel('Elapsed time (days)')
plt.show()



len(elapsed_days)
len(birddata.speed_2d)
plt.plot(birddata.timestamp, birddata.speed_2d)
plt.show()
birddata.timestamp[100].date() - birddata.timestamp[10].date()
birddata[birddata.bird_name == 'Eric'].groupby(['timestamp']).mean()

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
plt.show()

avg_day_speed == daily_mean_speeed
avg_day_speed[1]
daily_mean_speeed[1]

daystamps = [birddata.timestamp[i].date() for i in range(len(birddata))]
birddata['daystamp'] = pd.Series(daystamps, index=birddata.index)

data_eric.timestamp[data_eric.daystamp == data_eric.daystamp[0]]
np.mean(data_eric.speed_2d[data_eric.daystamp == data_eric.daystamp[0]])
daily_mean_speeed[0]



conda install -c conda-forge cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

proj = ccrs.Mercator()

# plt.figure(figsize=(10, 10))
# ax.set_extent((-25.0, 20.0, 52.0, 10.0))
ax = plt.axes(projection=proj)
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
for name in bird_names:
    ix = birddata.bird_name == name
    x, y = birddata.longitude[ix], birddata.latitude[ix]
    ax.plot(x, y, '.', transform=ccrs.Geodetic(), label=name)
plt.legend(loc='upper left')
plt.show()


birddata.bird_name == 'Eric'
birddata['bird_name'] == 'Eric'
birddata[birddata['bird_name'] == 'Eric']



import datetime
















# Network Analysis
# Graphs consist of nodes, also called vertices, and links, also called edges.
import networkx as nx

G = nx.Graph()
G.add_node(1)
G.add_nodes_from([2, 3])
G.add_nodes_from(['u', 'v'])
G.nodes()
G.add_edge(1, 2)
G.add_edge('u', 'v')
G.add_edges_from([(1, 3), (1, 4), (1, 5), (1, 6)])
G.add_edge('u', 'w')
G.nodes()
G.edges()
G.remove_node(2)
G.remove_nodes_from([4, 5])
G.remove_edge(1, 3)
G.remove_edges_from([(1, 2), ('u', 'w')])
G.number_of_nodes()
G.number_of_edges()


G = nx.karate_club_graph()
import matplotlib.pyplot as plt
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
plt.show()
G.degree()[33] is G.degree(33)
G.number_of_nodes(); G.number_of_edges()

from scipy.stats import bernoulli
bernoulli.rvs(0.2)
N = 20
p = 0.2

def er_graph(N, p):
    """Generate an ER graph."""
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.nodes()
    for node1 in G.nodes():
        for node2 in G.nodes():
            if (bernoulli.rvs(p=p) and\
                node1 < node2):
                G.add_edge(node1, node2)
    # G.edges()
    # G.number_of_nodes()
    # G.number_of_edges()
    return G

nx.draw(er_graph(50, .08), node_size=40, node_color='gray')
plt.show()

nx.er_graph(10, 1)

list(dict(G.degree()).values())
def plot_degree_dist(G):
    plt.hist([v for k, v in G.degree()], histtype='step', label='name')
    plt.xlabel('Degree $k$')
    plt.ylabel('$P(k)$')
    plt.title('Degree distribution')
    

G1 = er_graph(500, .08)
plot_degree_dist(G1)
G2 = er_graph(500, .08)
plot_degree_dist(G2)
G3 = er_graph(500, .08)
plot_degree_dist(G3)
G4 = er_graph(100, .03)
plot_degree_dist(G4)

plt.show()


import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
A1 = np.loadtxt(os.getcwd()+'/edx/Network_Analysis'+'/adj_allVillageRelationships_vilno_1.csv', delimiter=',')
A2 = np.loadtxt(os.getcwd()+'/edx/Network_Analysis'+'/adj_allVillageRelationships_vilno_2.csv', delimiter=',')
G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)

def basic_net_stats(G):
    print('Number of nodes %d' % G.number_of_nodes())
    print('Number of edges {}'.format(G.number_of_edges()))
    print('Average degree %.2f' % np.mean([v for k, v in G.degree()]))
    
basic_net_stats(G1)
basic_net_stats(G2)

plot_degree_dist(G1)
plot_degree_dist(G2)
plt.legend(loc='lower right')
plt.show()


nx.nx.connected_component_subgraphs(G1)
gen = (G1.subgraph(c) for c in nx.connected_components(G1))
# Generator functions do not return a single object but instead, they can be used to generate a sequence of objects
# using the next method.
type(gen)
g = gen.__next__()
type(g)
g.number_of_nodes()
len(g)
len(gen.__next__())
len(G1)
G1.number_of_nodes()
G1_LCC = max((G1.subgraph(c) for c in nx.connected_components(G1)), key=len)
G2_LCC = max((G2.subgraph(c) for c in nx.connected_components(G2)), key=len)
len(G1_LCC)
G1_LCC.number_of_nodes()
G1_LCC.number_of_nodes() / G1.number_of_nodes()
G2_LCC.number_of_nodes() / G2.number_of_nodes()
nx.draw(G1_LCC, node_color='red', edge_color='gray', node_size=20)
nx.draw(G2_LCC, node_color='red', edge_color='gray', node_size=20)
plt.show()














# Statistical Learning
# Quantitative variables take on numerical values, such as income,
# whereas qualitative variables take on values in a given category,
# such as male or female.
# If the outcome is quantitative, we talk about regression problems,
# whereas if the outcome is qualitative, we talk about classification problems.
# What the best prediction is depends on the so-called loss function,
# which is a way of quantifying how far our predictions for Y for a given
# value of X are from the true observed values of Y.
# In a regression setting, by far the most common loss function
# is the so-called squared error loss.

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

n = 100
beta_0 = 5
beta_1 = 2
np.random.seed(1)
x = 10 * ss.uniform.rvs(size=n)
y = beta_0 + beta_1*x + ss.norm.rvs(loc=0, scale=1, size=n)
# plt.figure()
plt.plot(x, y, 'o', ms=5)
xx = np.array([0, 10])
plt.plot(xx, beta_0 + beta_1*xx)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
np.mean(x); np.mean(y)



# In simple linear regression, the goal is to predict a quantitative response Y 
# on the basis of a single predictor variable X.
# Note that the capital letters here always correspond to random variables.
# RSS - residual sum of squares

def compute_rss(y_estimate, y):
  return np.sum(np.power(y-y_estimate, 2))
def estimate_y(x, b_0, b_1):
  return b_0 + b_1*x
rss = compute_rss(estimate_y(x, beta_0, beta_1), y)



# We can use matrix calculus to obtain a formula, a closed form
# solution for obtaining the least-squares estimates,
# and that is how some software packages do the estimation.
# Other software packages use different methods,
# such as the method of gradient descent, which is a numerical optimization
# method used for complex models.

slopes = np.arange(-10, 15, 0.01)
rss = [np.sum((y - beta_0 - slope*x)**2) for slope in slopes]
ind_min = np.argmin(rss)
print('Estimate for that slope: ', slopes[ind_min])
print('RSS for the lope: ', rss[ind_min])
plt.plot(slopes, rss)
plt.xlabel('Slope')
plt.ylabel('RSS')
plt.show()




import statsmodels.api as sm
mod = sm.OLS(y, x)
est = mod.fit()
est.summary()
X = sm.add_constant(x)
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

# Before we fit our model, we can compute what
# is called the total sum of squares, or TSS, which
# is defined as the sum of the squared differences between outcome
# yi and the mean outcome.
# We compute a similar quantity
# called the residual sum of squares, RSS, which
# is defined as the sum of the squared differences between the outcome yi



n = 500
beta_0 = 5
beta_1 = 2
beta_2 = -1
np.random.seed(1)
x_1 = 10 * ss.uniform.rvs(size=n)
x_2 = 10 * ss.uniform.rvs(size=n)
y = beta_0 + beta_1*x_1 + beta_2*x_2 + ss.norm.rvs(loc=0, scale=1, size=n)
X = np.stack([x_1, x_2], axis=1)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], y, c=y)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
plt.show()

from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept=True)
lm.fit(X, y)
lm.intercept_
lm.coef_
X_0 = np.array([2, 4])
lm.predict([X_0])
lm.predict(X_0.reshape(1, -1))
lm.score(X, y)




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.5, random_state=1)
lm = LinearRegression(fit_intercept=True)
lm.fit(X_train, y_train)
lm.score(X_test, y_test)




# Week 5, Case Study Part 1
https://www.themoviedb.org/?language=en
https://www.kaggle.com/tmdb/tmdb-movie-metadata
import pandas as pd
import numpy as np
df = pd.read_csv("https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@movie_data.csv", index_col=0)
df.head()
df.where(df == np.inf)










# Logistic Regression
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
%matplotlib notebook

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

(x1, y1, x2, y2) = gen_data(50, 1, 1, 1.5 )
(x1, y1, x2, y2) = gen_data(1000, 1.5, 1, 1.5 )

def plot_data(x1, y1, x2, y2):
    plt.plot(x1, y1, 'o')
    plt.plot(x2, y2, 'o')
    plt.xlabel('$X_1$')
    plt.ylabel('$X^2$')
    plt.show()

plot_data(x1, y1, x2, y2)



# conditional class probabilities:
# Our goal is to model the conditional probability
# that the outcome belongs to a particular class conditional on the values
# of the predictors.
# can say that logistic regression is a linear model that models probabilities
# on a non-linear scale.

def prob_to_odds(p):
    if p <= 0 or p >= 1:
        print("Probabilities must be between 0 and 1.")
    return p / (1-p)

prob_to_odds(0.8)



from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
X = np.vstack((np.vstack((x1, y1)).T, np.vstack((x2, y2)).T))
n = 1000
y = np.hstack((np.repeat(1, n), np.repeat(2, n)))
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.5, random_state=1)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
clf.predict_proba(np.array([[-2, 0]]))
clf.predict(np.array([[-2, 0]]))

from math import log, sqrt
log(4, 2)
sqrt(8, 2)
np.power(44, (1/2)) == 2 * sqrt(11)

np.stack((xx1.ravel(), xx2.ravel()), axis=1).shape
Z.reshape((100, 100))

def plot_probs(ax, clf, class_no):
    xx1, xx2 = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1))
    probs = clf.predict_proba(np.stack((xx1.ravel(), xx2.ravel()), axis=1))
    Z = probs[:, class_no]
    Z = Z.reshape(xx1.shape)
    CS = ax.contourf(xx1, xx2, Z)
    cbar = plt.colorbar(CS)
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")


# plt.figure(figsize=(5,8))
ax = plt.subplot(211)
plot_probs(ax, clf, 0)
plt.title("Pred. prob for class 1")
ax = plt.subplot(212)
plot_probs(ax, clf, 1)
plt.title("Pred. prob for class 2");
plt.show()




# Random Forest
# The random forest method makes use of several trees when making its prediction
