# help

dir(str)  # returns a list of attributes and methods belonging to an object
help(str)
help(str.isdigit)
type(str) # type of an object
id(str) # identity of an object







# print

from pprint import pprint
pprint(dir(str))


print('Germany', end = '\n')
print('ABC' + ' ' + str(99))
print('ABC', ' ', str(99))

print('number: %d, name: %s' % (99, 'Ukasz'))

print('Ukasz'[1:]+ '100')
print('Ukasz'[1:], '100')

for x in '234':
   print('value :', x)

age = {'Tim': 28, 'Jim': 35, 'Pam': 40}
print('Students names : %s' % list(age.keys()))
print('Students names : {}'.format(list(age.keys())))

for key in age:
    print(key, age[key])

for (key, val) in age.items():
    print(key, val)

for key in age.keys():
   print('{}: {}'.format(key, age[key]))

for key in age.keys():
    print('%s: %d' % (key, age[key]))

for key in age.keys():
   print(key, ':', age[key])

for key in age:
    print(': '.join((key, str(age[key]))))


# r, R - raw string suppresses actual meaning of escape characters.
print('\n')
print(r'\n')
print(R'\n')
print('/n')


# Format
print('welcome {} too'.format('you'))
print('welcome {n1} {n2} 2'.format(n1='you', n2='too'))
print('welcome {1} {0}'.format('you', 'too'))
print('The binary to decimal value is : {:d}'.format(0b11))
print('The binary value is : {:b}'.format(500))
print('The scientific value is : {:e}'.format(40))
print('The scientific value is : {:E}'.format(40))
print('The value is : {:.3f}'.format(40))
print('The value is : {:n}'.format(500.00))
print('The value is : {:.2%}'.format(1.80))
print('The value is {:_}'.format(1000000))
print('The value is {:,}'.format(1000000))
print('The value is: {:5}'.format(40))
print('The value is: {:05}'.format(40))
print('test {:5} test {:5}'.format(1, 2))
print('The value is: {}'.format(-40))
print('The value is: {:-}'.format(-40))
print('The value is: {:+}'.format(40))
print('The value is: {:=}'.format(-40))
print('The value {:^10} is positive value'.format(40))
print('The value {:<10} is positive value'.format(40))
print('The value {:>10} is positive value'.format(40))

print('welcome %s too' % ('you'))
print('welcome %s %s' % ('you', 'too'))


# Using class with format()
class Class_format():
   msg1 = 'you'
   msg2 = 'too'

print('tak, to {c.msg1} {c.msg2}'.format(c=Class_format))


# Using dictionary with format()
my_dict = {'msg1': 'you', 'msg2': 'too'}
print('tak, to {m[msg1]} {m[msg2]}'.format(m=my_dict))
my_dict = {'msg1': [], 'msg2': []}
my_dict['msg1'].append('yes')
my_dict['msg1'].append('you')
my_dict['msg2'].append('too')










# List

a = list([2, 3])
a = [2, 3]
a.remove(3) # remove chosen value
del a[0] # remove value at position
a.pop(0)  # remove value at position and print it
a.append(2)
a.append(1)
a.reverse()
b = [3, 4]
a + b # concacenate lists
c = [a[i] + b[i] for i in range(len(a))] # add lists
sum(c)
list(reversed(c))
c.sort(reverse=False)
sorted(c)
len(c)
c.index(5) # returns index of the value
[1, 2, 3].index(3)
c.insert(0, 1) # insters an element at i-th position # prepend

# to sort the list by length of the elements
str1 = ['cat', 'mammal', 'goat', 'is', '']
str1.sort()
str1.sort(key=len)
sorted(str1, key=len)

# removes '' and ' ' from string list
list(filter(str.strip, str1))

[1, 1, 2].count(1) # count chosen elements in list
[None, True, False].count(True)

'foobar'[::-1]
'foobar'[-1::-1]
'foobar'.startswith('foo')







# Tuples

a = 12.34
b = 23.45
coordinate = (a, b) # tuple packing
(a1, b1) = coordinate   # tuple unpacking

a = (2)
type(a)
a = tuple([2])
type(a)
b = (2,)
type(b)

x = (1, 2, 3)
len(x)
x.count(3)
sum(x)
x[0]





# Strings

S = 'Python'
S[0]
S[-1]
S[:3]
S[-3:]
S[:-3]
'y' in 'Python'
'Y' in 'Python'

'2' + '2'
S * 3
'eight equals ' + str(8)

name = 'A Alan'
name.replace('A', 'a')
seq = seq.replace('\n', '').replace('\r', '')
name.split('l')
names = name.split(' ')
';'.join('ukasz')
' '.join([name.title() for name in names])
''.join([random.choice(string.ascii_lowercase) for _ in range(5)])
name.find('l') # same as index but returns -1 of none
name.index('l') # same as find but sends ValueError it str not found


'22'.isdigit()
'str'.isalpha()
'ukasz'.upper()
'ukasz'.capitalize()
'ukasz'.title()
'ukasz'.count('a')
'ukasz'.count('k', 0, 1)

# Strip
'  2 2  '.strip()
'  2 2  '.rstrip()
str1 = ' abc '
str2 = '*abc*'
print(str1.strip())
print(str2.strip('*'))
print(str1.strip('c a'))


# Find
str1 = ' abc '
str1.find('c')
str1.find('c', 1, 5)
mystring = 'Meet Guru99 Tutorials Site. Best site for Python Tutorials!'
mystring.find('Tutorials')
mystring.find('Tutorials', 20)
mystring.rfind('Tutorials')
mystring.find('foo') # returns -1 if nothing found
mystring.index('Tutorials') # returns error if nothing found












# Sets

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
ids.remove(1)

ids = set(range(10))
males = {1, 3, 5, 6, 7}

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

word = 'antidisestablishmentarianism'
set(word)
word.count('a')
# how many distinctive letters in a word
len(set(word))


# issubset
males.issubset(ids)
# in doesn't work
males in ids
'A' in 'ABC'





# Dictionaries

age = {}
type(age)
age = dict()

age = {'Tim': 28,
       'Jim': 35,
       'Pam': 40
       }
names = age.keys()
ages = age.values()
age.items()
age.get('Tim')
age.get('Pim', 0) # dict.get() returns 0 if key not in dict
age['Tim']
age['Tim'] += 2

age['Tom2'] = 50
age.update({'Tom': 50})

# remove elements, all
del age['Tom2']
age.pop('Tom2')
age.clear() # remove all the elements from the dictionary.
del age # delete, remove dictionary


# lists as values
age.update({'Tom': []})
age['Tom'].append(50)
age['Tom'].append(51)

# int as a key
age[0] = 1
age[0]

'Tim' in age
'Tim' in age.keys()

{**age, **{'Sam': 55}}
{*age, *{'Sam': 55}}



age_copy = age.copy() # creates copy which has another id
age_copy['Tim'] = 100

sentence = 'Jim quickly realized that the beautiful gowns are expensive'

def counter(input_string):
    count_letters = {}
    for letter in input_string:
        if letter in string.ascii_letters:
            count_letters[letter] = count_letters.get(letter, 0) + 1
    return count_letters

counter(sentence)










# max

address_count = {'J': 1, 'i': 5, 'm': 1, 'q': 1, 'u': 3, 'c': 1,
                 'k': 1, 'l': 3, 'y': 1, 'r': 2, 'e': 8, 'a': 4,
                 'z': 1, 'd': 1, 't': 4, 'h': 2, 'b': 1, 'f': 1, 'g': 1,
                 'o': 1, 'w': 1, 'n': 2, 's': 2, 'x': 1, 'p': 1, 'v': 1}

max(address_count, key=address_count.get)
address_count[max(address_count, key=address_count.get)]

# select votes with max count form dict
[vote for vote, count in address_count.items() if count == max(address_count.values())]


# find mode
from collections import Counter
word = "antidisestablishmentarianism"
Counter(word)
[(vote, count) for vote, count in dict(Counter(word)).items() if count == max(dict(Counter(word)).values())]

arr = [3, 3, 3, 3, -1]
max(set(arr), key=arr.count)






# sorted

def sort_dict(input_dict):
    address_count_desc = {}
    for key in sorted(input_dict, key=input_dict.get, reverse=True):
        address_count_desc[key] = address_count[key]
    return address_count_desc

sort_dict(address_count)





# sort

a = [(1, 2), (4, 1), (9, 10), (13, -3)]

a.sort(key=lambda x: x[1])

sorted(a, key=lambda x: x[1])






# sum

import numpy as np

X = np.random.randint(1, 7, (100, 10))
X.shape

# summing over all of the rows of the array.
np.sum(X, axis=0)
# summing over all of the columns.
np.sum(X, axis=1)

# sum over all columns
X.sum(1)
X.sum(1,)






# repeat

from itertools import repeat
import numpy as np

np.repeat([1, 2], 3)
np.repeat([[1, 2], [3, 4]], 2)
np.repeat([[1, 2], [3, 4]], 2, axis=0)
np.repeat([[1, 2], [3, 4]], 2, axis=1)

[1, 2] * 3
list(range(1, 3)) * 3
np.array([1, 2]) * 5

list(repeat([1, 2], 3))

sum(map(lambda x: [x] * 3, range(1, 6)), [])

list(map(pow, range(10), repeat(3)))







# chain

from itertools import chain

# Merging two range into one
frange = chain(range(10), range(10, 20, 1))
list(frange)

list(range(10)) + list(range(10, 20, 1))


a_list = [[1, 2], [3, 4], [5, 6]]
print(list(chain(*a_list)))
print(list(chain.from_iterable(a_list)))




# reduce

from functools import reduce

reduce((lambda x, y: x + y), [1, 2, 3, 4])
reduce((lambda x, y: x - y), [4, 3, 2, 1])






# from collections import defaultdict

from collections import defaultdict

colors = (
   ('first', 'blue'),
   ('second', 'red'),
   ('third', 'green'),
   ('first', 'orange')
)

colors_list = defaultdict(list)
for key, val in colors:
    colors_list[key].append(val)

dict(colors_list)
colors_list.get('first')


from collections import defaultdict
tree = lambda: defaultdict(tree)
some_dict = tree()
some_dict['colours']['favourite'] = "yellow"

import json
print(json.dumps(some_dict))








# from collections import deque

from collections import deque

d = deque()
d.append(1)
d.append('1')
len(d)
d[0]
d[-1]
d.popleft()
d.remove('1')

d = deque(range(5), maxlen=5) # max elements in q
d.appendleft(-1) # add element from left
d.extend([4, 5, 6]) # add list










# enumerate

for i, char in enumerate(('a', 'b', 'c'), -3):
    print(i, char)

for i in enumerate('foo'):
    print(i)

for i in enumerate({'a': 'PHP', 'b': 'JAVA', 'c': 'PYTHON', 'd': 'NODEJS'}.items()):
    print(i)

my_list = ['apple', 'banana', 'grapes', 'pear']
list(enumerate(my_list, 1)) # enumerate a tuple







# round

round_num = 15.456
round(round_num, 2)





# Range

for i in range(15, 5, -1):
   print(i, end =" ")

print(range(ord('a'), ord('c')))
print(chr(97))

ord('u')
chr(117)
print(u'\u0050')


def abc(c_start, c_stop):
    for i in range(ord(c_start), ord(c_stop)):
        yield chr(i)
list(abc('a', 't'))






# isinstance()

isinstance(51, int)
isinstance(5.0, int) # it's int, but object is float
isinstance(5.0, float)
isinstance('Hello World', str)
isinstance({1, 2, 3, 4, 5}, set)
isinstance((1, 2, 3, 4, 5), tuple)
isinstance([1, 2, 3, 4, 5], list)
isinstance({'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd'}, dict)

(5.0).is_integer()

class MyClass:
   message = 'Hello World'
class1 = MyClass()
isinstance(class1, MyClass)














# import math

import math

math.pi
math.sqrt(16)
math.sin(math.pi / 2)
math.factorial(5)








# import numpy as np

import numpy as np

np.version.version

np.zeros(5) + np.ones(5)
np.zeros((5, 3))

np.array([[1, 2], [3, 4]]).transpose()
np.array([[1, 2], [3, 4]]).T

x = np.array([1, 2, 3])
y = np.array([2, 4, 6])
X = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

X.shape
X.size
X.dtype # data type
X.transpose()
np.transpose(X)
X.reshape(3, 2)
X.flatten()
np.unique(X).tolist()
X.tolist()
np.where(x == 1) # works like index or find

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

ind = [1, 2]
y[ind]
y[[1, 2]]
y[1:3]

y > 4
x[x > 2]

np.arange(0, 100, 20)   # Return evenly spaced values within a given interval. (step, spacing)
np.linspace(0, 100, 11) # Return evenly spaced numbers over a specified interval. num
np.meshgrid(np.arange(0, 100, 20), np.linspace(0, 100, 11))

z = np.random.random(3)
np.random.random((2, 3))
np.random.choice([1, '2', 'tree'])
np.random.seed(1)
np.random.uniform(-1, 1)
np.random.normal(size=(2, 3))
np.random.normal(0, 1, (2, 3))
np.random.randint(8, 10, (2, 3))


np.random.sample(3) # works as np.random.random(), not like random.sample()

np.any(z > 0.9)
np.all(z >= 0.1)

np.all(np.array([[1, 2], [1, 2]]) == 1, axis=0)
np.all(np.array([[1, 2], [1, 2]]) == 1, axis=1)

np.cumsum([1, 2, 3])
np.subtract.accumulate([1, 2, 3])
np.power([1, 2, 3], 2)
np.square([1, 2, 3])
np.sqrt([9, 16, 25])
np.sum([3, 4, 10])
np.prod([3, 4, 10])
np.product([3, 4, 10])

np.min([9, 16, 25])
np.max([9, 16, 25])
np.mean([9, 16, 25])
np.median([9, 16, 25])
np.std([9, 16, 25])

l = np.array([1, 2])
m = np.array([3, 4])
# a1*b1 + a2*b2 
np.dot(l, m)

n = np.array([[1, 2], [3, 4]])
o = np.array([[5, 6], [7, 8]])
# 1*5 + 2*7 = 19
np.matmul(n, o)

np.linalg.det(n)


x1 = np.array([[1, 2], [3, 4]])
x2 = np.array([[5, 6], [7, 8]])
np.concatenate((x1, x2), axis=0)
np.concatenate((x1, x2), axis=1)
np.concatenate((x1, x2), axis=None)
np.concatenate(([[1, 2]], [[5, 6]]), axis=1)

np.stack((x1, x2))
np.stack((x1, x2), axis=1)
np.vstack((x1, x2))
np.hstack((x1, x2))







# array diagonals
np.fliplr([[1, 2], [3, 4]]).diagonal()


# np.argsort returns to indices that would sort the given array.
# sort by index
np.argsort([3, 2, 1])


np.round([12.3456, 45.1], 2)






# import scipy.stats as ss
import scipy.stats as ss
from scipy import stats as ss


word = "antidisestablishmentarianism"
ss.mode([char for char in word])


ss.norm(0, 1).rvs((5, 2)) # generate normal distribution array; rvs = random variables realizations
ss.norm.rvs(0, 1, (5, 2))

ss.uniform.rvs(size=3)










# SciPy

import numpy as np
from scipy import io as sio
array_ones = np.ones((4, 4))
# to do mathlab
sio.savemat('example.mat', {'ar': array_ones})
data = sio.loadmat('example.mat', struct_as_record=True)
data['ar']


from scipy.special import comb
# find combinations of 5, 2 values using comb(N, k)
comb(5, 2, exact=True, repetition=False)
comb(5, 2, exact=True, repetition=True)


from scipy import linalg
import numpy as np
two_d_array = np.array([[4, 5], [3, 2]])
linalg.det(two_d_array)
linalg.inv(two_d_array)
eg_val, eg_vect = linalg.eig(two_d_array)
#get eigenvalues
eg_val
#get eigenvectors
eg_vect


from matplotlib import pyplot as plt
import numpy as np
#Frequency in terms of Hertz
t = np.linspace(0, 2, 1000, endpoint=True)
a = np.sin(10 * np.pi * t)
# figure, axis = plt.subplots()
# axis.plot(t, a)
# axis.set_xlabel('$Time (s)$')
# axis.set_ylabel('$Signal amplitude$')
plt.xlabel('Time (s)')
plt.ylabel('Signal amplitude')
plt.title('Sin(x)')
plt.plot(t, a, label='Sin(x)')
plt.legend(loc='upper left')
plt.show()








# CSV

# If you want to read the file, pass in r
# If you want to read and write the file, pass in r+
# If you want to overwrite the file, pass in w
# If you want to append to the file, pass in a

# In general, if the format is written by humans, it tends to be text mode. jpg image files are not generally written by humans (and are indeed not readable by humans), and you should therefore open them in binary mode by adding a b to the mode string (if you’re following the opening example, the correct mode would be rb). If you open something in text mode (i.e. add a t, or nothing apart from r/r+/w/a), you must also know which encoding to use. For a computer, all files are just bytes, not characters.

# encoding='utf-8-sig'

# better use pandas
import os

import csv
dire = '/home/ukasz/Documents/IT/Python/'
with open(os.getcwd()+'/'+'data.csv', mode = 'rt', encoding='utf-8-sig') as f:
    for row in csv.reader(f):
        print(row)

import csv
dire = '/home/ukasz/Documents/IT/Python/'
reader = csv.DictReader(open(os.getcwd()+'/'+'data.csv', encoding='utf-8-sig'))
for raw in reader:
    print(raw)

with open(dire+'writeData.csv', mode='w') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"',
                        quoting=csv.QUOTE_MINIMAL)
    #way to write to csv file
    writer.writerow(
        ['Programming language', 'Designed by', 'Appeared', 'Extension'])
    writer.writerow(['Python', 'Guido van Rossum', '1991', '.py'])
    writer.writerow(['Java', 'James Gosling', '1995', '.java'])
    writer.writerow(['C++', 'Bjarne Stroustrup', '1985', '.cpp'])










# import random

import random

random.seed(1) # Fixes the see of the random number generator.
random.choice([1, '2', 'tree'])
random.sample(range(10), 3)
random.random()
random.uniform(-1, 1)






# Counter

from collections import Counter

word = "antidisestablishmentarianism"
Counter(word)
list(Counter(word).elements()) # list of all letters
Counter(word).most_common(2) # the most repetitive elements

Counter([5, 5, 6, 6, 7]).most_common(2) # the most frequent elements in list

counter1 = Counter({'x': 4, 'y': 2, 'z': -2})
counter2 = Counter({'x1': -12, 'y': 5, 'z':4 })
#Addition
counter1 + counter2 # only the values that are positive will be returned.

#Subtraction
counter4 = counter1 - counter2 # all -ve numbers are excluded.For example z will be z = -2-4=-6, since it is -ve value it is not shown in the output

#Intersection
counter5 = counter1 & counter2 # it will give all common positive minimum values from counter1 and counter2

#Union
counter6 = counter1 | counter2 # it will give positive max values from counter1 and counter2

counter1 = Counter({'x': 5, 'y': 12, 'z': -2, 'x1':0})
counter2 = Counter({'x': 2, 'y':5})
counter1.subtract(counter2)

counter1 = Counter({'x': 5, 'y': 12, 'z': -2, 'x1':0})
counter2 = Counter({'x': 2, 'y':5})
counter1.update(counter2)
counter1 + counter2

counter1 =  Counter({'x': 5, 'y': 12, 'z': -2, 'x1':0})
counter1['y'] = 20
counter1['y1'] = 1
counter1['y']







# import os

import os
os.system('cls' if os.name == 'nt' else 'clear')

from os import getcwd, chdir
import os

getcwd()
chdir('/home/ukasz/Documents/IT/Python/')
path = 'edx'

lines = []
for line in open(getcwd()+'/'+path+'/'+'input.txt'): # read a file
    lines.append(line.strip())
lines

f = open(getcwd()+'/'+path+'/'+'input.txt')
for i in f.readlines():
   print(i.strip())
f.close()


with open(getcwd()+'/'+path+'/'+'input.txt') as f: # read with 'with'
    table = f.read()
table
print(table)

path = '/edx/translation/'
# f = open(os.getcwd()+path+'dna.txt')
# f.close()
with open(os.getcwd()+path+'dna.txt') as f:
    seq = f.read()

# print with \n
seq
# print without \n
print(seq)

f = open(getcwd()+'/'+'edx'+'/'+'input.txt', 'w') # write a file
f.write('Hello\nWorld\nnew\n')
f.close()

with open(getcwd()+'/'+'edx'+'/'+'input.txt', 'a') as f: # append to a file
    f.write('Python\n')

dire = '/home/ukasz/Documents/IT/Python/'
with open(dire+'delme.txt', 'w') as f:
    for i in range(2):
        f.write('This is line {}\r\n'.format(i + 1))

with open(dire+'delme.txt', 'a') as f:
    for i in range(2):
        f.write('Appended line %d\r\n' % (i + 1))




# os.chdir('/home/ukasz/Documents/IT/Python/')
path = '/edx/Language_Processing/Books'
os.listdir(os.getcwd()+path) # lists directory catalogs

for language in os.listdir(os.getcwd()+path):
    for author in os.listdir(os.getcwd()+path+'/'+language):
        for title in os.listdir(os.getcwd()+path+'/'+language+'/'+author):
            input_file = os.getcwd()+path+'/'+language+'/'+author+'/'+title
            print(input_file)
            


# Check If File or Directory Exists

from os import path

dire = '/edx/translation/'
print('File exists: ' + str(path.exists(os.getcwd()+'/'+dire+'dna.txt')))
print('directory exists: ' + str(path.exists(os.getcwd()+'/'+dire)))

print('Is it File?: ' + str(path.isfile(os.getcwd()+'/'+dire+'dna.txt')))
print('Is it File?: ' + str(path.isfile(os.getcwd()+'/'+dire)))

print ('Is it Directory?: ' + str(path.isdir(os.getcwd()+'/'+dire+'dna.txt')))
print ('Is it Directory?: ' + str(path.isdir(os.getcwd()+'/'+dire)))

import pathlib
file = pathlib.Path(os.getcwd()+'/'+dire+'dna.txt')
file = pathlib.Path(os.getcwd()+'/'+dire)
if file.exists():
    print('File/directory exists')
else:
    print("File/directory doesn't exist")




# import shutil

# backup a file
import os
import shutil
from os import path

dire = '/edx/translation/'
if path.exists(os.getcwd()+'/'+dire+'dna.txt'):
    src = path.realpath(os.getcwd()+'/'+dire+'dna.txt')
    head, tail = path.split(src)
    print(src)
    print('patch: ', head)
    print('file: ', tail)
    dst = src+'.bak'
    shutil.copy(src, dst)
    # copy over the permissions
    shutil.copystat(src, dst)


from os import path
import datetime
# from datetime import date, time, timedelta
import time

# Get the modification time
dire = '/edx/translation/'
t = path.getmtime(os.getcwd()+'/'+dire+'dna.txt')
print(t)
print(datetime.time.ctime(path.getmtime(os.getcwd()+'/'+dire+'dna.txt')))
print(datetime.datetime.fromtimestamp(t))
print(datetime.datetime.fromtimestamp(path.getmtime(os.getcwd()+'/'+dire+'dna.txt')))


# Rename

import os
import shutil
from os import path

dire = '/edx/translation/'
if path.exists(os.getcwd()+'/'+dire+'dna.txt.bak'):
    # get the path to the file in the current directory
    src = path.realpath(os.getcwd()+'/'+dire+'dna.txt.bak')
    # rename the original file
    os.rename(src, os.getcwd()+'/'+dire+'delme.txt')














# Zip & UnZip

first_name = ['Joe','Earnst','Thomas','Martin','Charles']
last_name = ['Schmoe','Ehlmann','Fischer','Walter','Rogan','Green']
age = [23, 65, 11, 36, 83]
list(zip(first_name, last_name,  age))

for f, l, a in zip(first_name, last_name,  age):
    print('{} {} is {} years old'.format(f, l, a))

# unzip
full_name_list = [('Joe', 'Schmoe', 23),
                  ('Earnst', 'Ehlmann', 65),
                  ('Thomas', 'Fischer', 11),
                  ('Martin', 'Walter', 36),
                  ('Charles', 'Rogan', 83),
                 ]

list(zip(*full_name_list))
(*full_name_list)


import os
import shutil
from zipfile import ZipFile
from os import path
from shutil import make_archive

# Check if file exists
dire = '/edx/translation/'
if path.exists(os.getcwd()+'/'+dire+'dna.txt'):
    # get the path to the file in the current directory
    src = path.realpath(os.getcwd()+'/'+dire+'dna.txt')
    # now put things into a ZIP archive
    root_dir, tail = path.split(src)
    shutil.make_archive(os.getcwd()+'/'+dire+'delme_archive', 'zip', root_dir)

# more fine-grained control over ZIP files
with ZipFile(os.getcwd()+'/'+dire+'delme_archive.zip', 'w') as newzip:
    newzip.write(os.getcwd()+'/'+'pep8')




# *
# *someting - skips outer braces, feeding function with (possibly) more than one argument






# generators

# simple generator with range(10)
def gen_fun():
    for i in range(10):
        yield i

for i in gen_fun():
    print(i)


# Fibonacci generator
def fib(n):
    a = b = 1
    for i in range(n):
        yield(a)
        a, b = b, a + b

# print all values
for i in fib(10):
    print(i)

# use next() to access the next element of a sequence
fib_10 = fib(10)
print(next(fib_10))


# iterate a string
iter_text = iter('some_text')
print(next(iter_text))








# import pandas as pd

import pandas as pd

pd.__version__ # package / libraray version
pd.set_option('display.max_columns', None)


pd.Series([6, 3, 8, 6])
x = pd.Series([6, 3, 8, 6], index=['q', 'w', 'e', 'r'])
x['q']
x.q
x[['q', 'w']]
x.index
x.reindex(sorted(x.index)) # sort by index

x.unique()
pd.unique(x)
x.tolist()

age = {'Tim': 29, 'Jim': 31, 'Pam': 27, 'Sam': 35}
x = pd.Series(age)


table = pd.DataFrame(columns=('Name', 'Age')) # create an empty table
table.loc[1] = 'James', 25 # add elements
table.loc[2] = 'Jess', 22
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

data = {"name": ['Tim', 'Jim', 'Pam', 'Sam'],
        "age": [29, 31, 27, 35],
        "Zip": ['02115', '02130', '67700', '00100']
        }
x = pd.DataFrame(data, columns=['age', 'Zip'], index=data['name'])
x.loc['Tim']
x.age
x['age']
x.index.tolist()
x.reindex(sorted(x.index))



# read csv
# path = '/home/ukasz/Documents/IT/Python/edx/Language_Processing/hamlets.csv'
# hamlets = pd.read_csv(path, index_col=0)
hamlets = pd.read_csv('https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@hamlets.csv', index_col=0)
hamlets.info()

language, text = hamlets.loc[1] # Access a group of rows and columns by label(s) or a boolean array.
language, text = hamlets.iloc[0] # Purely integer-location based indexing for selection by position.


dire = '/home/ukasz/Documents/IT/Python/'
result = pd.read_csv(os.getcwd()+'/'+'data.csv')


C = {
    'Programming language': ['Python', 'Java', 'C++'],
    'Designed by': ['Guido van Rossum', 'James Gosling', 'Bjarne Stroustrup'],
    'Appeared': ['1991', '1995', '1985'],
    'Extension': ['.py', '.java', '.cpp'],
}
df = pd.DataFrame(C, columns=['Programming language',
                  'Designed by', 'Appeared', 'Extension'])

df.to_csv(os.getcwd()+'/'+'pandaresult.csv',
          index=None, header=True)



path = '/edx/whiskies'
whisky = pd.read_csv(os.getcwd()+path+'/'+'whiskies.txt')
whisky.info()
whisky.head()
whisky.loc[0]
whisky['Region'] = pd.read_csv(os.getcwd()+path+'/'+'regions.txt')
whisky.head()
whisky.tail()
whisky.iloc[5:10, :5]
whisky.columns
flavors = whisky.iloc[:, 2:14]
whisky.iloc[np.argsort(whisky.Body)][::-1] # descending sort
whisky_sorted = whisky.iloc[np.lexsort((whisky['Sweetness'], whisky['Body']))]
whisky_sorted = whisky_sorted.reset_index(drop=True) # reset index and drops old index
correlations = np.array(pd.DataFrame.corr(whisky.iloc[:, 2:14].transpose()))

# drop a column
whisky.drop(['Region'], axis=1)
whisky.drop(['Region'], axis='columns')
whisky.drop(columns=['Region'])



import matplotlib.pyplot as plt

plt.pcolor(correlations, cmap='jet')
plt.title('Rearranged')
plt.axis('tight')
# plt.savefig(os.getcwd()+path+'/'+'correlations.png')
plt.show()


plt.plot(whisky.Sweetness[whisky.Body == 2], whisky.Honey[whisky.Body == 2], '.', label='whiskey body = 2')
plt.xlabel('$Swetness$')
plt.ylabel('$Honey$')
plt.legend(loc='upper left')
plt.show()
plt.hist(whisky.Sweetness[whisky.Body == 2])


word = "antidisestablishmentarianism"
# create DataFrame from Counter or dict with 'columns'
data = pd.DataFrame(dict(Counter(word)).items(), columns=('letter', 'count'))
# data.set_index('letter') # set index column as other column
# create DataFrame from Counter or dict column by column
data = pd.DataFrame({'letter': dict(Counter(word)).keys(),
                    'count': dict(Counter(word)).values()})


# create data in column based on another column data
data['length'] = data['letter'].apply(len)
data['length'] = data['count'].apply(lambda x: x + 1)

# data['length'] = data['word'].apply(lambda x: len(x))
# data['length'] = list(map(len, data['word']))


# create column based on condition from another column
data.loc[data['count'] > 2, 'freq'] = 'high'
data.loc[data['count'] < 3, 'freq'] = 'low'

# create column based on condition data from another columns
data['freq2'] = (data['count'] > data['length']).astype(int)


# groupby
data.groupby('freq').mean()
data.groupby('freq').length.mean()
data.groupby('freq').size()
groupby_freq = data.groupby('freq') # object groupby 'freq'
groupby_freq.mean()

data.freq.value_counts()

pd.DataFrame({'freq': ('high', 'low'),
              'mean_length': data.groupby('freq').length.mean(),
              'size': data.groupby('freq').size(),
              'length_size': data.groupby('freq').letter.max()
              })






# filter

codespeedy_list = ['hey','there','','whats','','up']
# filter None
list(filter(None, codespeedy_list))
list(filter(bool, codespeedy_list))
list(filter(len, codespeedy_list))

# removes '' and ' ' from string list
str1 = ['cat', 'mammal', 'goat', 'is', '', ' ']
list(filter(str.strip, str1))

# find ditits string
list(filter(str.isdigit, '1is2'))


# sort elements by int in substrings
def order(sentence):
    return ' '.join(sorted(sentence.split(), key=lambda x: int(''.join(filter(str.isdigit, x)))))
order('is2 Thi1s T4est 3a')

def order(sentence):
    return ' '.join(sorted(sentence.split(), key=lambda x: sorted(x)))

# use function as a key in sorted
def digits_first_fun(x):
    return int(''.join(filter(str.isdigit, x)))
digits_first_fun('1is2')

def order(sentence):
    return ' '.join(sorted(sentence.split(), key=digits_first_fun))
order('is2 Thi1s T4est 3a')


# filter < 0
list(filter(lambda x: x < 0, range(-5, 5)))








# bool

# use not str % 2
if not x % 2 is <==> to if x % 2 == 0
2 % 2 # is 0; not 0 is 1
not 2 % 2





# None

import numpy as np

None_list = np.array([None, True, False])
# select only not None's
Not_None_list = None_list[None_list != None]
# count True's in list; filter None
np.sum(Not_None_list)




# NaN

import numpy as np
import pandas as pd

NaN_list = pd.Series([1, np.NaN, 3])
# remove np.Nan's
np.array(NaN_list[~np.isnan(NaN_list)], dtype=int)
np.isnan(NaN_list)
np.isnan(NaN_list).any()
np.any(np.isnan(NaN_list))
# removes NaN's
NaN_list[-np.isnan(NaN_list) == True]



df = df.replace([np.inf, -np.inf], np.nan)
# df = df.dropna(how="any")
df = df.dropna()

# removing np.Nan's without Pandas doesn't work
Nan_list = [1, 2, np.NaN]
~np.isnan(Nan_list)
# removing Nan's doesn't work that way, need to use Pandas
Nan_list[~np.isnan(Nan_list)]




# import string

import string 

string.digits
string.ascii_lowercase
string.ascii_letters

'str'.isalpha()



# import mathplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np


# Plots
plt.plot([0, 1, 4, 9, 16])
plt.plot(np.linspace(0, 10, 20) ** 1.5)
plt.plot(np.linspace(0, 10, 20), np.linspace(0, 10, 20) ** 1.5)
plt.plot([0, 1, 2], [0, 1, 4], 'rd-')
plt.plot(np.linspace(0, 10, 20) ** 1.5, 'bo-', linewidth=2, markersize=4, label='First')
plt.plot(np.linspace(0, 10, 20) ** 2, 'gs-', linewidth=2, markersize=12, label='Second')
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.title('$x^a$'+' plot')
plt.legend(loc='upper left')
# plt.axis([-.5, 10.5, -5, 105])
# plt.savefig("myplot.png")
# plt.savefig("myplot.pdf")

plt.show()

# Histograms
plt.hist(np.random.normal(size=10000), density=True, ec='black')
plt.hist(np.random.normal(size=10000), density=True, bins=np.linspace(-5, 5, 21))
plt.hist(np.random.normal(size=10000), density=True, bins=21, histtype='step')
plt.show()


x = np.random.gamma(2, 3, 100000)
# plt.figure()
plt.subplot(231)
plt.hist(x, bins=30)
plt.subplot(234)
plt.hist(x, bins=30, density=True)
plt.subplot(233)
plt.hist(x, bins=30, cumulative=True)
plt.subplot(236)
plt.hist(x, bins=30, density=True, cumulative=True, histtype="step")
plt.show()

help(plt.figure)







# import time

import time
import matplotlib.pyplot as plt
import random
import numpy as np

start_time = time.time()
X = np.random.randint(1, 7, (1000000, 10))
Y = np.sum(X, axis=1)
plt.hist(Y, density=True, ec="black")
time.time() - start_time
plt.show()



time.sleep(1.5)
print('This message will be printed after a wait of 5 seconds')




# import timeit

import timeit
timeit.timeit('foo = 10 * 5')
timeit.timeit(stmt='a=10; b=10; sum = a + b')








# import datetime

import datetime

time1 = datetime.datetime.today()
datetime.datetime.today() - time1

date_str = '2013-08-15 00:18:08+00'
datetime.datetime.strptime(date_str[:-3], '%Y-%m-%d %H:%M:%S')


(datetime.datetime.today() - time1) / datetime.timedelta(seconds=1)
(datetime.datetime.today() - time1) / datetime.timedelta(hours=1)
(datetime.datetime.today() - time1) / datetime.timedelta(days=1)

time1.date()
time1.time()











# import json

import json

path = '/edx/translation/'

with open(os.getcwd()+path+'table.json', 'r') as F2:
    table = F2.read()

type(table)
table_js = json.loads(table)
type(table_js)
table_js['ATA']


x = {
    "name": "Ken",
    "age": 45,
    "married": True,
    "children": ("Alice", "Bob"),
    "pets": ["Dog"],
    "cars": [
        {"model": "Audi A1", "mpg": 15.1},
        {"model": "Jeep Compass", "mpg": 18.1}
    ]
}
# sorting result in asscending order by keys:
type(x)
# dict to string
sorted_string = json.dumps(x, indent=4, sort_keys=True)
type(sorted_string)
print(sorted_string)

# save .json dict -> str
# here we create new data_file.json file with write mode using file i/o operation
dire = '/home/ukasz/Documents/IT/Python/'
with open(os.getcwd()+'/'+'json_file.json', 'w') as f:
    person_data = {  "person":  { "name":  "Kenn", "sex":  "male", "age":  28}}
    # write json data into file
    json.dump(person_data, f, indent=4, sort_keys=True)

# dict to string
json.dumps(person_data, indent=4, sort_keys=True)


# json data string
person_data = '{"person": {"name": "Kenn", "sex": "male", "age": 28}}'
type(person_data)
# Decoding or converting JSON format in dictionary using loads()
dict_obj = json.loads(person_data)
# check type of dict_obj
type(dict_obj)
# get human object details
dict_obj.get('person')
dict_obj['person']['name']

# read str -> dict
# File I/O Open function for read data from JSON File
dire = '/home/ukasz/Documents/IT/Python/'
with open(os.getcwd()+'/'+'json_file.json') as f:
    # store file data in object
    data = json.load(f)
type(data)
data


# Create a List that contains dictionary
lst = ['a', 'b', 'c', {"4": 5, "6": 7}]
# separator used for compact representation of JSON.
# Use of ',' to identify list items
# Use of ':' to identify key and value in dictionary
compact_obj = json.dumps(lst, indent=4, separators=(',', ': '))
print(compact_obj)


dic = {"a": 4, "b": 5}
# To format the code use of indent and 4 shows number of space and use of separator is not
# necessary but standard way to write code of particular function.
formatted_obj = json.dumps(dic, indent=4, separators=(',', ': '))
print(formatted_obj)
formatted_obj = json.dumps(dic, indent=4)
print(formatted_obj)










# from sklearn import datasets

from sklearn import datasets
import sklearn.datasets as datasets
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



# Unicode

import numpy as np

py_unicode = np.array([U'\u0050', u'\u0059', u'\u0054',
                       u'\u0048', u'\u004F', u'\u004E'
                       ])
''.join(py_unicode)

print(u'\u0050')

ord('u')
chr(117)







# if __name__ == '__main__':

if __name__ == '__main__':
   print('ala')






# break & continue statements

my_list = ['Siya', 'Tiya', 'Guru', 'Daksh', 'Riya', 'Guru']

for i, elem in enumerate(my_list):
   print(i, elem)
   if elem == 'Guru':
       print('Found break')
       break
       print('Hidden')
   print('Też nie see')

i = 0
while True:
    print(my_list[i])
    if (my_list[i] == 'Guru'):
        print('Found the name Guru')
        break
        print('After break statement')
    i += 1
print('After while-loop exit')

for i in range(4):
    for j in range(4):
        if j == 2:
            break
        print('number is :', i, j)

for i in range(5):
    if i == 2:
        continue
    print(i)

i = 0
while i < 5:
    if i == 2:
        i += 1
        continue
    print(i)
    i += 1

for i in range(4):
    pass





# yield statement
# Yield
# The yield keyword in python works like a return with the only difference is that 
# instead of returning a value, it gives back a generator object to the caller.
# Python3 Yield keyword returns a generator to the caller and the execution of the 
# code starts only when the generator is iterated.


sequence = [10, 2, 8, 7, 5, 4, 3, 11, 0, 1]

def filter4(arg):
    for i in arg:
        if i > 4:
            yield(i)

list(filter4(sequence))


def generator():
    yield 'H'
    yield 'E'
    yield 'L'
    yield 'L'
    yield 'O'

print(''.join(list(generator())))

for i in generator():
    print(i, end='')

lett = generator()
next(lett)


# function that filters vowels
def filter_vowels(letter):
    vowels = ['a', 'e', 'i', 'o', 'u']
    if (letter in vowels):
        return True
    else:
        return False

# filter works as yield
filtered_vowels = filter(filter_vowels, letters)

print('The filtered vowels are:')
for vowel in filtered_vowels:
    print(vowel)
next(filtered_vowels)
list(filtered_vowels)













# lambda function

# lambdas in filter()
sequence = [10, 2, 8, 7, 5, 4, 3, 11, 0, 1]
list(filter(lambda x: x > 4, sequence))

filtered_result = map(lambda x: x * x, sequence)
list(filtered_result)

list_to_round = [2.6743, 3.63526, 4.2325, 5.9687967, 6.3265, 7.6988, 8.232, 9.6907]
list(map(round, list_to_round))
list(map(lambda x: round(x, 2), list_to_round))
list(map(np.round, list_to_round))
list(map(lambda x: np.round(x, 2), list_to_round))

list(map(lambda s: s.upper(), 'qw zx'))


def myMapFunc(list1, list2):
   return list1 + list2
my_list1 = [2, 3, 4, 5, 6, 7, 8, 9]
my_list2 = [4, 8, 12, 16, 20, 24, 28]
list(map(myMapFunc, my_list1, my_list2))




# map
# map(function_to_apply, list_of_inputs)

list(map(lambda x: x ** 2 if not x % 2 else x ** 3, range(1, 11)))

# tuple of functions as inputs
def add(x):
    return x + x
def multiply(x):
    return x * x

for i in range(5):
    print(list((map(lambda x: x(i), (add, multiply)))))









# RegEx

import re
re.findall(r'^\w+', 'education is fun')
re.findall(r'\w+$', 'education is fun')
re.split(r'\s','we are splitting the words')
re.split(r's','split the swords')

# re.match() function of re in Python will search the regular expression pattern and return the first occurrence. 
# The Python RegEx Match method checks for a match only at the beginning of the string. 
# So, if a match is found in the first line, it returns the match object. 
# But if a match is found in some other line, the Python RegEx Match function returns null.

# re.search() function will search the regular expression pattern and return the first occurrence.
# Unlike Python re.match(), it will check all lines of the input string.
# The Python re.search() function returns a match object when the pattern is found and “null” if the pattern is not found

# findall() module is used to search for “all” occurrences that match a given pattern.
# In contrast, search() module will only return the first occurrence that matches the specified pattern.
# findall() will iterate over all the lines of the file and will return all non-overlapping matches of pattern in a single step.

list = ['guru99 get', 'guru99 give', 'guru Selenium']
for element in list:
    z = re.match(r'(g\w+)\W(g\w+)', element)
    if z:
        print(z.groups())


patterns = ['software testing', 'guru99']
text = 'software testing is fun?'
for pattern in patterns:
    print('Looking for "%s" in "%s" ->' % (pattern, text), end=' ')
    if re.search(pattern, text):
        print('found a match!')
        print((re.search(pattern, text)).group())
    else:
        print('no match')

re.search(patterns[0], text).group()
re.match(patterns[0], text).group()
re.findall(patterns[0], text)

abc = ['guru99@google.com', 'careerguru99@hotmail.com', 'users@yahoomail.com']
for i in abc:
    mat = re.match(r'[\w\.-]+@[\w\.-]+\.[\w]+', i)
    # re.search works the same
    if mat:
        print(i)
        print(mat.start())
        print(mat.end())
        print(mat.group(0))
        print(mat.groups())

abc = 'guru99@google.com, careerguru99@hotmail.com, users@yahoomail.com'
emails = re.findall(r'[\w\.-]+@[\w\.-]+\.[\w]+', abc)

xx = """guru99 
careerguru99	u
selenium"""
re.findall(r'^\w', xx)
re.findall(r'^\w', xx, re.MULTILINE)


# Lets use a regular expression to match a date string. Ignore
# the output since we are just testing if the regex matches.
regex = r'([a-zA-Z]+) (\d+)'
if re.search(regex, 'June 24'):
    # Indeed, the expression '([a-zA-Z]+) (\d+)' matches the date string
    
    # If we want, we can use the MatchObject's start() and end() methods 
    # to retrieve where the pattern matches in the input string, and the 
    # group() method to get all the matches and captured groups.

    match = re.search(regex, 'June 24')
    
    # This will print [0, 7), since it matches at the beginning and end of the 
    # string
    print('Match at index %s, %s' % (match.start(), match.end()))
    
    # The groups contain the matched values. In particular:
    #    match.group(0) always returns the fully matched string
    #    match.group(1), match.group(2), ... will return the capture
    #            groups in order from left to right in the input string
    #    match.group() is equivalent to match.group(0)
    
    # So this will print 'June 24'
    print('Full match: %s' % (match.group(0)))
    # So this will print 'June'
    print('Month: %s' % (match.group(1)))
    # So this will print '24'
    print('Day: %s' % (match.group(2)))
else:
    # If re.search() does not match, then None is returned
    print('The regex pattern does not match. :(')

# Lets use a regular expression to match a few date strings.
regex = r'[a-zA-Z]+ \d+'
matches = re.findall(regex, 'June 24, August 9, Dec 12')
for match in matches:
    print('Full match: %s' % (match))
print(matches)

# To capture the specific months of each date we can use the following pattern
regex = r'([a-zA-Z]+) \d+'
matches = re.findall(regex, 'June 24, August 9, Dec 12')
for match in matches:
    print('Match month: %s' % (match))

# If we need the exact positions of each match
regex = r'([a-zA-Z]+) \d+'
matches = re.finditer(regex, 'June 24, August 9, Dec 12')
for match in matches:
    # which corresponds with the start and end of each match in the input string
    print('Match at index: %s, %s' % (match.start(), match.end()))

# Lets try and reverse the order of the day and month in a date 
# string. Notice how the replacement string also contains metacharacters
# (the back references to the captured groups) so we use a raw 
# string for that as well.
regex = r'([a-zA-Z]+) (\d+)'

# This will reorder the string and print:
#   24 of June, 9 of August, 12 of Dec
re.sub(regex, r'\2 of \1', 'June 24, August 9, Dec 12')

# Lets create a pattern and extract some information with it
regex = re.compile(r'(\w+) World')
result = regex.search('Hello World is the easiest')
if result:
    print(result.start(), result.end())

# This will print:
#   Hello
#   Bonjour
# for each of the captured groups that matched
for result in regex.findall('Hello World, Bonjour World'):
    print(result)

# This will substitute 'World' with 'Earth' and print:
#   Hello Earth
print(regex.sub(r'\1 Earth', 'Hello World'))








# Exceptions

# no file to open
try:
    file = open('test.txt', 'rb')
except IOError as e:
    print('An IOError occurred. {}'.format(e.args[-1]))
    # raise e


# putt all the exceptions which are likely to occur in a tuple
try:
    file = open('test.txt', 'rb')
except (IOError, EOFError) as e:
    print('An error occurred. {}'.format(e.args[-1]))


# handle individual exceptions in separate except blocks
try:
    file = open('test.txt', 'rb')
except IOError as e:
    print('An error occurred.')
    # raise e
except EOFError as e:
    print('An EOF error occurred.')
    # raise e

# trapping ALL exceptions:
try:
    file = open('test.txt', 'rb')
except Exception as e:
    # Some logging if you want
    raise e

# finally runs whatever except occurs or not
try:
    file = open('test.txt', 'rb')
except IOError as e:
    print('An IOError occurred. {}'.format(e.args[-1]))
finally:
    print("This would be printed whether or not an exception occurred!")


try:
    print('I am sure no exception is going to occur!')
except Exception:
    print('exception')
else:
    # any code that should only run if no exception occurs in the try,
    # but for which exceptions should NOT be caught
    print('This would only run if no exception occurs. And an error here '
          'would NOT be caught.')
finally:
    print('This would be printed in every case.')








# function as an argument

def b(function, arg):
    function(arg)

def a(function, inner_fun, *arg):
    function(inner_fun, *arg)

a(b, print, 'Hello')



# *args; are tuples

def args_fun(*arg):
    # return type(arg) 
    return arg
args_fun(1, 2)

def args_fun2(arg1, arg2):
    print('arg1:', arg1)
    print('arg2: '+ str(arg2))
args = ('one', 2)
args_fun2(*args)

# **kwargs; are dict's

def kwargs_fun(**kwargs):
    # return type(kwargs)
    return kwargs
kwargs_fun(name='no_name', sth='no')

def args_fun2(arg1, arg2):
    print('arg1:', arg1)
    print('arg2: '+ str(arg2))
    print(type(arg2))
kwargs = {'arg2': 2, 'arg1': 'one'}
args_fun2(**kwargs)

def args_fun3(**kwarg):
    # print(type(kwarg))
    print(kwarg)
kwargs = {'arg2': 2, 'arg1': 'one'}
args_fun3(**kwargs)












# Instance & Class variables

# Instance variables are for data which is unique to every object
# Class variables are for data shared between different instances of a class

class Cal(object):
    # pi is a class variable
    pi = 3.142

    def __init__(self, radius):
        # self.radius is an instance variable
        self.radius = radius

    def area(self):
        return self.pi * self.radius**2

a = Cal(32)
a.area()
a.pi
Cal(32).area()

a.pi = 5

b = Cal(44)
b.area()
b.pi


# wrong usage of mutable class variables
class SuperClass(object):
    superpowers = []

    def __init__(self, name):
        self.name = name

    def add_superpower(self, power):
        self.superpowers.append(power)

foo = SuperClass('foo')
foo.name
bar = SuperClass('bar')
bar.name
foo.add_superpower('fly')
bar.superpowers


# __x__ Magic Methods; commonly called dunder (double underscore)
# __init__ class initializer

class GetTest(object):

    def __init__(self, name):
        self.name = name
        print('Greetings {} !!'.format(name))

    def another_method(self):
        print('I am another method which is not automatically called')

a = GetTest('luk')
a.another_method()
a.name
GetTest('luk').another_method()


# __getitem__ # allows its instances to use the [] (indexer) operator.
class GetTest(object): 

    def __init__(self):
        self.info = {
            'name': 'Luk',
            'country': 'Poland',
            'number': 123456789
        }

    def __getitem__(self, i):
        return self.info[i]

GetTest()['name']
GetTest().__getitem__('name')







# Classes and Object-Oriented Programming

# Constructors
# A constructor is a class function that instantiates an object to 
# predefined values. It begins with a double underscore (_). 
# It __init__() method

class User(object):
    name = ''
    day = 15

    def __init__(self, name, years):
        self.name = name
        self.age = years

    def sayHello(self):
        print('Welcome to Guru99, ' + self.name)

User1 = User('Alex')
User1.sayHello()

User('Ukasz').sayHello()
User('Ukasz', 38).name
User('Ukasz', 38).age
me = User('Ukasz', 38)
me.day


class MyList(list):
    def remove_min(self):
        self.remove(min(self))

    def remove_max(self):
        self.remove(max(self))

    def append_sum(self):
        self.append(sum(self))

    def remove_nth(self, i):
        del self[i]

x = [1, 2, 3, 4, 5]
y = MyList(x)

type(x)
type(y)
dir(x)
dir(y)

y.remove_min()
y.remove_max()
y.remove_nth(-1)
y.append_sum()



# Python OOPs: Class, Object, Inheritance and Constructor 
# A Class in Python is a logical grouping of data and functions. 
# It gives the freedom to create data structures that contains arbitrary 
# content and hence easily accessible.

class mClass(object):
    def method1(self):
        print('self from met1')

    def method2(self, arg):
        print('self from met2 ' + arg)

def main():
    c = mClass()
    c.method1()
    c.method2('+ some arg')

if __name__=='__main__':
    main()

type(mClass)
del mClass

# Inheritance is a feature used in object-oriented programming; 
# it refers to defining a new class with less or no modification to 
# an existing class. The new class is called derived class and from one 
# which it inherits is called the base. Python supports inheritance; 
# it also supports multiple inheritances. A class can inherit attributes 
# and behavior methods from another class called subclass or heir class.

class myClass():
    def method1(self):
        print('met1')

class childClass(myClass):
    # def method1(self):
        # myClass.method1(self)
        # print('childClass Method1')

    def method2(self):
        print('childClass method2')

def main1():
    # exercise the class methods
    c2 = childClass()
    c2.method1()
    c2.method2()

if __name__ == '__main__':
    main1()







# decorators

"""What is a decorator in Python?
A decorator in Python is a function that takes another function as its argument, 
and returns yet another function . Decorators can be extremely useful as they allow the 
extension of an existing function, without any modification to the original function source code

Decorators are a significant part of Python. In simple words: they are functions which modify the functionality of other functions. They help to make our code shorter and more Pythonic. 
"""

# function within function
def hi(name="yasoob"):
    def greet():
        return "now you are in the greet() function"

    def welcome():
        return "now you are in the welcome() function"

    if name == "yasoob":
        return greet
    else:
        return welcome

a = hi()
print(a)
print(hi())
print(hi()())


# function as an argument
def hi():
    return "hi yasoob!"

def doSomethingBeforeHi(func):
    print("I am doing some boring work before executing hi()")
    print(func())

doSomethingBeforeHi(hi)


# first decorator
def f_without():
    print('I have no decorators')

f_without()

def decor(a_func):
    def fun_in_decor():
        print('You think so?')
        a_func()
        print('Look again')
    return fun_in_decor

f_without = decor(f_without)
f_without()


# decorator with @
def f_without():
    print('I have no decorators')
f_without()

def decor(a_func):
    def fun_in_decor():
        print('You think so?')
        a_func()
        print('Look again')
    return fun_in_decor

@decor
def f_without():
    print('I have no decorators')
f_without()

# returns name of a func inside a decor
print(f_without.__name__)


from functools import wraps

def new_decor(a_func):
    @wraps(a_func) # it's for proper name of a func; a_func instead of decorated
    def fun_in_decor():
        print('You think so?')
        a_func()
        print('Look again')
    return fun_in_decor

@ new_decor
def f_without():
    print('I have no decorators')
f_without()

# returns proper name of a func
print(f_without.__name__)


from functools import wraps
def decorator_name(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not can_run:
            return "Function will not run"
        return f(*args, **kwargs)
    return decorated

@decorator_name
def func():
    return("Function is running")

can_run = True
print(func())
# Output: Function is running

can_run = False
print(func())
# Output: Function will not run

print(func.__name__)


# example 1
from functools import wraps

def decor(add_fun):
    @wraps(add_fun)
    def inner_fun(*args, **kwargs):
        print('name: ', add_fun.__name__, **kwargs)
        return add_fun(*args, **kwargs)
    return inner_fun    

@decor
def add_fun(x):
    """Docstring"""
    return x ** x

add_fun(3)
help(add_fun)








# Mutation

# adds elements to a list despite fact its target=[] in parameters
# In Python the default arguments are evaluated once when the function is defined, not each time the function is called.
def add(x, target=[]):
    target.append(x)
    return target

add(1)

# creates new list everytime function is called
def add(x, target=None):
    if target == None:
        target = []
    target.append(x)
    return target

add(1)









"""A Singleton pattern in python is a design pattern that allows you to create just one instance of a class, throughout the lifetime of a program. Using a singleton pattern has many benefits. A few of them are:
https: // python-patterns.guide/gang-of-four/singleton/

"""





# Virtual Environment
# Virtualenv is a tool which allows us to make isolated python environments. 
# Different applications need different versions of Python/packages


pip install virtualenv # install
virtualenv --system-site-packages mycoolproject # virtualenv to have access to your systems site-packages
virtualenv --no-site-packages projekt1 # makes clean virtualenv
source myproject/bin/activate # activates that isolated environment
deactivate # turn off env








