# help
from scipy.fftpack import shift
from sympy import intersection


dir(str)  # returns a list of attributes and methods belonging to an object
help(str)
help(str.isdigit)
type(str)  # <class 'type'> # type of an object
id(str)  # identity of an object





# slice
'abc'[::-1]  # 'cba' # reversed string
''.join(reversed('abc'))  # 'cba' # same
[1, 2, 3][::-1]  # [3, 2, 1] # reversed list





# Strings
'welcome '+ "you" + ' too ' + "2"  # 'welcome you too 2'
'welcome {0} too {1}'.format('you', '2')  # 'welcome you too 2'
'welcome {a} too {b}'.format(a='you', b=2)  # 'welcome you too 2'
'welcome %s too %d' % ('you', 2)  # 'welcome you too 2'
first_str, sec_str = "you", "2"
f'welcome {first_str} too {sec_str}'  # 'welcome you too 2'

arg = 'you'
arg2 = 2
'welcome {0} too {1}'.format(arg, arg2)
'welcome %s too %d' % (arg, arg2)
f'welcome {arg} too {arg2}'

'2' + '2'  # '22'
'S' * 3  # 'SSS'
'equals = ' + str(8)  # 'equals = 8'

'abca'.find('a')  # 0 # same as index but returns -1 if none
'abca'.find('a', 1, 2)  # -1 # find with start, stop
'abca'.rfind('a')  # 3
'abca'.index('b')  # 1 # same as find but sends ValueError it str not found

'abca'.count('a')  # 1
'abc'.count('a', 1, 1)  # 0 # count with start, stop

'A Alan'.replace('A', 'a')  # 'a alan' # replace 'A' with 'a'
# 'GGTCAG' # remove new line and carret return
'GGT\nCAG\r'.replace('\n', '').replace('\r', '')
print('GGT\nCAG\r')

'A Alan'.split('l')  # ['A A', 'an']
'A  Alan'.split()  # ['A', 'Alan'] # remove additional spaces
'A  Alan'.split(' ')  # ['A', '', 'Alan'] # capture additional spaces
list(iter('abcd'))  # ['a', 'b', 'c', 'd'] # generator from string
{x:int(y) for x, _, y in ['0+1', '1-2']}  # {'0': 1, '1':2}

';'.join(str(123))  # '1;2;3'
';'.join('abcd')  # 'a;b;c;d'
' '.join(chr(i) for i in range(97, 100))  # 'a b c'

'abc'.isalpha()  # True # string.ascii_letters
'abc'.islower()  # True # string.ascii_lowercase
'abc'.isupper()  # False # string.ascii_uppercase
'abc'.isnumeric()  # False
'abc123'.isalnum()  # True
'abc123'.isalpha()  # False
'123'.isdigit()  # True
'123'.isalnum()  # True
'123'.isdecimal()  # True
'123'.isnumeric()  # True
'1010'.isnumeric()  # True
'{:b}'.format(10).isnumeric()  # True

'abc'.upper()  # 'ABC'
'ABc'.lower()  # 'abc'
'abc def'.capitalize()  # 'Abc def'
'abc def'.title()  # 'Abc Def'
" ".join(map(str.capitalize, "abc def".split()))  # 'Abc Def'
"abc't def".title()  # 'Abc'T Def'
import string
string.capwords("abc't def")  # 'Abc't Def'

'  1 2  '.strip()  # '1 2' # removes leading and trailing whitespaces
'  1 2  '.rstrip()  # '  1 2' # removes trailing whitespaces
'* 1 2 *'.rstrip('*')  # '* 1 2 ' # removes trailing asterisk
'abc'.strip('a c')  # 'b'

('abc' * 2).center(10)  # '  abcabc  '
('abc' * 2).ljust(10)  # 'abcabc    '
('abc' * 2).rjust(10)  # '    abcabc'

'foobar'.startswith('foo')  # True
'foobar'.endswith('bar')  # True


# slice
S = 'Python'
S[0]  # 'P' # first element
S[-1]  # 'n' # last element

S[:2]  # 'Py' # ends on the 2nd elem exclusive # first two elements counting from left
S[4:]  # 'on' # starts from the 4th elem # last two elements counting from left
S[:-4]  # 'Py' # ends on the 4th elem from right exclusive # first two elements counting from right
S[-2:]  # 'on' # starts from the 2nd elem from right # last two elements counting from right

S[:2:-1]  # 'noh' # starts form the end, ends on the 2nd elem exclusive reversed # last three elements counting from left reversed
S[4::-1]  # 'ohtyP' # starts from the 4th elem, ends on the start reversed # first five elements counting from left reversed
S[:-4:-1]  # 'noh' # starts form the end, ends on the 4th elem from right exclusive reversed # last three elements counting from right reversed
S[-2::-1]  # 'ohtyP' # starts from the 2nd elem from right, ends on the start reversed # first five elements counting from right reversed

[start counting from +left/-right:
 stop counting from +left/-right:
 step counting from +left/-right:
]

# format
'a{}b'.format(['#', '!'])  # "a['#', '!']b"
'{:d}'.format(0b100)  # '4' # binary to decimal
'{:d}'.format(0x1c5)  # '453' # hexagonal to decimal
'{:b}'.format(8)  # '1000' # decimal to binary
'{:.3f}'.format(40)  # '40.000' # float with precision
'{:n}'.format(500.00)  # '500' # remove trailing zeros
'{:.2%}'.format(1.80)  # '180.00%' # to % with precision
'{:,}'.format(1000000)  # '1,000,000' # set the separator # '   40' # set the string length and insert leading whitespaces
'{:5}'.format(40)  # '   40'
'40'.rjust(5)  # '   40'
'{:05}'.format(40)  # '00040' # set the string length and insert leading zeros
'40'.rjust(5, '0')  # '00040'
"{:^{}}".format("*" * 3, 5)  # " *** "
"{} {}".format("*" * 3, "5")  # "*** 5"
'{:^5}'.format("*")  # "  *  "
'{:^5}'.format(40)  # ' 40  ' # set the string length and center
"***".center(5)  # " *** "
'40'.center(5)  # '  40 '
'{:<5}'.format(40)  # '40   ' # set the string length and adjust left
'40'.ljust(5)  # '40   '
'{:>5}'.format(40)  # '   40' # set the string length and adjust right
'40'.rjust(5)
'{:.2e}'.format(40)  # 4.00e+01' # scientific form
'{:.2E}'.format(40)  # 4.00E+01' # scientific form

# Using class with format()
class Class_format():
   msg1 = 'you'
   msg2 = 'too'

'welcome {c.msg1} {c.msg2}'.format(c=Class_format)


# Using dictionary with format()
format_dict = {'msg1': 'you', 'msg2': 'too'} # string as value
type(format_dict['msg1'])
print('welcome {m[msg1]} {m[msg2]}'.format(m=format_dict))
format_dict = {'msg1': [], 'msg2': []} # list as value
format_dict['msg1'].append('yes')
format_dict['msg1'].append('you')
format_dict['msg2'].append('too')





# binary octal decimal hexadecimal

# 256 = 0:255 = FF = byte = 8 bits
bin(255)  # '0b11111111'  # dec to bin
'{:b}'.format(255)  # '11111111'  # dec to bin
f'{255:016b}'  # '0000000011111111' # dec to bin with leading zeros
hex(255)  # '0xff' # dec to hex
'{:x}'.format(255)  # 'ff' # dec to hex
oct(255)  # '0o377' # dec to ocx
'{:o}'.format(255)  # 377  # dec to ocx
int(bin(255), 2)  # 255 # dec to bin to dec
int('11111111', 2)  # 255 # bin to dec
int('0b11111111', 2)  # 255 # bin to dec
'{:d}'.format(0b11111111)  # '255'  # bin to dec
int('0xff', 16)  # 255 # hex to dec
hex(0b11111111)  # '0xff' # bin to hex
print(0b11111111)  # 255 # binary int to decimal
hex(int('11111111', 2))  # '0xff'# bin to dec to hex
bin(0xff)  # '0b11111111' # hex to bin





# comprehension
[chr(j) * i for i in range(1, 3) for j in range(97, 100)]  # ['a', 'b', 'c', 'aa', 'bb', 'cc']
sum(int(i) for i in str(169))  # 16 # comprehension generator sum
sum(i in 'aeiou' for i in 'abracadabra')  # 5





# bool
'a' in 'aeiou'  # True
'a' in ['a', 'e', 'i', 'o', 'u'] # True
True or 1/0  # True
not p or not q <==> not (p and q)
p => q <==> (not p) and q

list(True if i in 'ab' else False for i in 'abc')  # [True, True, False]
list(i in 'ab' for i in 'abc')  # [True, True, False]
list(True for i in 'abc' if i in 'ab')  # [True, True]
list(i > 5 for i in range(10) if not i % 2)  # [False, False, False, True, True]




# bitwise operators
https://realpython.com/python-bitwise-operators/
"""
Operator	Example	Meaning
&	a & b	Bitwise AND
|	a | b	Bitwise OR
^	a ^ b	Bitwise XOR (exclusive OR)
~	~a	    Bitwise NOT
<<	a << n	Bitwise left shift
>>	a >> n	Bitwise right shift
"""

AND, intersection
156 & 52  # 20
0b10011100 & 0b110100  # 20
bin(20)  # '10100'

OR, disjunction
156 | 52  # 188
0b10011100 | 0b110100  # 188
bin(188)  # '10111100'

XOR, 
156 ^ 52  # 168
0b10011100 ^ 0b110100  # 168
bin(168)  # '10101000'

NOT, complement
~ 156  # -157
~ 0b10011100  # -157
Need to use binary number representations
~ 156 & 255  # 99
~ 0b10011100 & 255  # 99

Left Shift
0b100111  # 39
0b100111 << 1  # 78
0b100111 << 2  # 156
0b100111 << 3  # 312
39 << 3  # 312
Fixed-length mask
39 << 3 & 255  # 56
int(bin(312)[3:], 2)  # 56

Right shift
0b10011101 >> 1  # 78
0b10011101 >> 2  # 39
0b10011101 >> 3  # 19

4 & 1  # 0
100 & 1  # 0
5 & 1  # 1
101 & 1  # 1

0b101 & 0b101  # 5
0b100 | 0b101  # 5
0b100 ^ 0b101  # 1

int('0011100', 2)
0.1 + 0.2
a = 256
b = 256
id(a)
id(b)
a is b
a = 257
b = 257
id(a)
id(b)
a is b

11 << 1  # 11 * 2
11 >> 1  # 11 // 2
2 ^ 3  # 10 xor 11 -> 01 -> 1





# print

from pprint import pprint
pprint(dir(str)) # dir in tree form

print('int = ', 5, 10, sep='aaa', end='zzz\n')  # int = aaa5aaa10zzz
print('abc' + ' ' + str(99))  # abc 99
print('abc', ' ', str(99))  # abc   99

# r, R - raw string suppresses actual meaning of escape characters.
print('a\n')
print(r'a\n')  # new line suppressed
print(R'a\n')  # new line suppressed
print('a\\n')  # new line suppressed
print('a/n')

for x in '234':
   print('value:', x)

age = {'Tim': 28, 'Jim': 35, 'Pam': 40}
for key in age:
    print(key, age[key])





# List

[1, 2] + [3]  # [1, 2, 3] # concatenate lists
list([2, 3])  # [2, 3] # create a list
a = [2, 3, 4, 3]  # [2, 3, 4, 3]  # create a list

a.remove(3) # removes the first occurrence of the chosen value
del a[0] # removes value at position
a.pop(0)  # removes value at position and print it

a.append(5) # 
a.insert(0, 1) # insters an element at i-th position # prepend

a.reverse()

[10, 20, 3, 3].index(3) # 2 # the first index of the value
[10, 5, 10].count(10) # count chosen elements in list
[None, True, False].count(True)

# to sort the list by length of the elements
animal_lst = ['cat', 'mammal', 'goat', 'is', '']
animal_lst.sort()
animal_lst.sort(key=len)
sorted(animal_lst, key=len)

a = [1, 2]
b = [3, 4]
c = [a[i] + b[i] for i in range(len(a))] # add lists
sum(c)
len(c)
list(reversed(c))
c.sort(reverse=False)
sorted(c)

[1, 2, 3].index(max([1, 2, 3])) # find index of the max element in array





# Tuples

a = 12.34
b = 23.45
coordinate = (a, b) # tuple packing
(a1, b1) = coordinate   # tuple unpacking

x = (1, 2, 3)
x.count(3)  # 1
len(x)  # 3
sum(x)  # 6
x[0]  # 1

# list of tuples
for (i, j, k) in [(1, 2, 3), (4, 5, 6)]:
    print(i, j, k)

# make one element tuple
a = (2)
type(a)  # <class 'int'>
a = tuple([2])
type(a)  # <class 'tuple'>
b = (2,)
type(b)  # <class 'tuple'>






# Sets

s = set()
s = {1, 3, 5}
type(s)
s = set((1, 3, 5))
s = set([1, 3, 5])
len(s)

s.add(10)
s.add(1)
s.pop()  # removes and shows first element from set
s.remove(5)  # removes chosen value with error if missing
s.discard(5)  # removes chosen value without an error if missing


A = {0, 1, 2, 3, 4}
B = {0, 2, 4, 6, 8}

# Set operations
A - B  # difference
A.difference(B)
A | B  # OR, union
A.union(B)
A & B  # AND (ampersand), intersection
A.intersection(B)
A ^ B  # XOR, symmetric difference
A.symmetric_difference(B)

# implication =>
# p => q = ~p v q
not A or B

word = 'antidisestablishmentarianism'
set(word)  # {'l', 'b', 'a', 's', 't', 'e', 'i', 'r', 'n', 'd', 'h', 'm'}
len(set(word))  # 12 # how many distinctive letters in a word


# https://realpython.com/python-sets/
A.issubset({0, 1, 2, 3, 4})  # issubset
A <= {0, 1, 2, 3, 4}
# issuperset
{0, 1, 2, 3, 4}.issuperset(A)
{0, 1, 2, 3, 4} >= A

A in {0, 1, 2, 3, 4}  # in doesn't work with sets
'A' in 'ABC'  # True





# Dictionaries

age = {}
age = dict()
type(age)  # <class 'dict'>

s_dict = {'Tim': 28,
          'Jim': 35,
          'Pam': 40
          }

names = s_dict.keys()  # dict_keys(['Tim', 'Jim', 'Pam'])
ages = s_dict.values()  # dict_values([28, 35, 40])
s_dict.items()  # dict_items([('Tim', 28), ('Jim', 35), ('Pam', 40)])
s_dict.get('Tim')  # 28
s_dict.get('Pim', 0)  # 0 # dict.get() returns 0 if key not in dict
s_dict['Tim']  # 28
s_dict['Tim'] += 2  # 30

s_dict['Tom2'] = 50
s_dict.update({'Tom': 50})

# remove elements, all
del s_dict['Tom2']
s_dict.pop('Tom')
s_dict.clear() # remove all the elements from the dictionary.
del s_dict # delete, remove dictionary


# lists as values
s_dict.update({'Tom': []})  # []
s_dict['Tom'].append(0)  # [0]
s_dict['Tom'].append(1)  # [0, 1]

# int as a key
s_dict[0] = 1  # {0: 1}
s_dict[0]

'Tim' in s_dict
'Tim' in s_dict.keys()

{**s_dict} == s_dict
{**s_dict, **{'Sam': 55}}
{*s_dict, *{'Sam': 55}}



age_copy = s_dict.copy() # creates copy which has another id
age_copy['Tim'] = 100

sentence = 'Jim quickly realized that the beautiful gowns are expensive'

import string
def counter(input_string):
    count_letters = {}
    for letter in input_string:
        if letter in string.ascii_letters:
            count_letters[letter] = count_letters.get(letter, 0) + 1
    return count_letters

counter(sentence)


s_dict = {'Tim': 28,
          'Jim': 35,
          'Pam': 40
          }
'Students names: {}'.format(list(s_dict.keys()))  # "Students names: ['Tim', 'Jim', 'Pam']"
'Students names: %s' % list(s_dict.keys())  # "Students names: ['Tim', 'Jim', 'Pam']"
f'Students names: {list(s_dict.keys())}'  # "Students names: ['Tim', 'Jim', 'Pam']"


for key in s_dict:
    print(key, s_dict[key])

for (key, val) in s_dict.items():
    print(key, val)

for key in s_dict.keys():
   print('{}: {}'.format(key, s_dict[key]))

for key in s_dict.keys():
    print('%s: %d' % (key, s_dict[key]))

for key in s_dict.keys():
    print(f'{key}: {s_dict[key]}')

for key in s_dict.keys():
   print(key + ':', s_dict[key])

for (key, val) in s_dict.items():
    print(key + ": " + str(val) )

for key in s_dict:
    print(': '.join((key, str(s_dict[key]))))





# min
# max

# from array
arr = [3, 10, 3, 3, 3, 9, 9]
min(set(arr), key=arr.count)  # 10 # find the first least frequent element in the list
arr.index(max(arr))  # find index of the max element in array
min([3, 9], key=arr.count)  # 9 # find the least frequent element in the 'key' list that's in input list

arr = ['notnot', 'some', 'random', 'text', '']
max(arr, key=len)  # first longest element in list
list(filter(lambda x: len(x) == len(max(arr, key=len)), arr)) # longest elements in list


# find min/max in list, dict, string
arr = [3, 10, 3, 3, 3, 11]
arr = "antidisestablishmentarianism"
min(arr, key=arr.count) # find the first least frequent element in the list
max(arr, key=arr.count) # find the first most frequent element in the list


from collections import Counter
cn_arr = Counter(arr)
min(cn_arr, key=cn_arr.get)  # 10 # find the key with the lowest value
cn_arr[min(cn_arr, key=cn_arr.get)]  # find the value of the key with the lowest value
[key for key, val in cn_arr.items() if val == min(cn_arr.values())]  # find all keys with the lowest value
{key: val for key, val in cn_arr.items() if val == min(cn_arr.values())}  # find all key: val with the lowest value


goals = [(5, 1), (2, 2), (2, 1)]
min((a - b, -b, i) for i, (a, b) in enumerate(goals))[-1] # min of sorted indexes order by (increasing, decreasing)





# sorted

# sort dict by count
from collections import Counter
arr = [3, 10, 3, 3, 3, 10, 6]
arr = "antidisestablishmentarianism"
count_dict = Counter(arr)
sorted(count_dict, key=count_dict.get)  # sort the keys by the values

# sort elements by int in substrings
def order(sentence):
    return ' '.join(sorted(sentence.split(), key=lambda x: int(''.join(filter(str.isdigit, x)))))
order('is2 Thi1s 23as T4est 3a')  # 'Thi1s is2 3a T4est 23as'

# sort elements by the lowest int in substrings
def order(sentence):
    return ' '.join(sorted(sentence.split(), key=lambda x: sorted(x)))
order('is2 Thi1s 23as T4est 3a')  # 'Thi1s 23as is2 3a T4est'

sorted(sorted('is2 Thi1s T4est 3a'.split(), key=lambda x: -len(x)), key=lambda x: sorted(x)) # sort within sort

# use function as a key in sorted
def digits_first_fun(x):
    return int(''.join(filter(str.isdigit, x)))
digits_first_fun('1is2')

def order(sentence):
    return ' '.join(sorted(sentence.split(), key=digits_first_fun))
order('is2 Thi1s T4est 3a')

to_sort = [[12, 'tall', 'blue', 1],
           [20, 'short', 'red', 9],
           [4, 'tall', 'blue', 13]
           ]
sorted(to_sort, key=lambda x: (x[1], x[0])) # sort by two columns

import operator
sorted(to_sort, key=operator.itemgetter(1, 0)) # sort by two columns with itemgetter

goals = [(5, 1), (2, 2), (2, 1)]
sorted((a-b, -b, i) for i, (a, b) in enumerate(goals))[0][-1] # sorted indexes order by (increasing, decreasing)
sorted(range(len(goals)), key=lambda x: (goals[x][0], -goals[x][1]))[0] # sorted indexes order by (increasing, decreasing)

sorted(('8 3 -5 42 -1 0 0 -9 4 7 4 -4').split(), key=int)  # sort chars with int key.


# sort

a = [(1, 2), (4, 1), (9, 10), (13, -3)]
a.sort(key=lambda x: x[1])
sorted(a, key=lambda x: x[1])





# sum

sum([True, False, True, False])  # 2 # counts True

import numpy as np
X = [[1, 2], [2, 3], [4, 5]]
np.sum(X, axis=0)  # summing over all of the rows, number of the columns stays
np.sum(X, axis=1) # summing over all of the columns, number of the rows stays

sum(int(i) for i in str(169))  # comprehension generator sum

sum([[1, 1, 1], [2, 2, 2]], [])  # [1, 1, 1, 2, 2, 2]




# product

from itertools import product

np.product([1, 2, 4])  # 8

list(product([1, 2, 4], [8, 5, 7], [0, 1]))  # [(1, 8, 0), (1, 8, 1), (1, 5, 0), (1, 5, 1), (1, 7, 0), (1, 7, 1), (2, 8, 0), (2, 8, 1), (2, 5, 0), (2, 5, 1), (2, 7, 0), (2, 7, 1), (4, 8, 0), (4, 8, 1), (4, 5, 0), (4, 5, 1), (4, 7, 0), (4, 7, 1)]
list(product(*([1, 2, 4], [8, 5, 7], [0, 1])))  # [(1, 8, 0), (1, 8, 1), (1, 5, 0), (1, 5, 1), (1, 7, 0), (1, 7, 1), (2, 8, 0), (2, 8, 1), (2, 5, 0), (2, 5, 1), (2, 7, 0), (2, 7, 1), (4, 8, 0), (4, 8, 1), (4, 5, 0), (4, 5, 1), (4, 7, 0), (4, 7, 1)]
list(product(*[['1', '2', '4'], ['8', '5', '7'], ['0', '1']]))  # [('1', '8', '0'), ('1', '8', '1'), ('1', '5', '0'), ('1', '5', '1'), ('1', '7', '0'), ('1', '7', '1'), ('2', '8', '0'), ('2', '8', '1'), ('2', '5', '0'), ('2', '5', '1'), ('2', '7', '0'), ('2', '7', '1'), ('4', '8', '0'), ('4', '8', '1'), ('4', '5', '0'), ('4', '5', '1'), ('4', '7', '0'), ('4', '7', '1')]
list(product('124', '857', '01'))  # [('1', '8', '0'), ('1', '8', '1'), ('1', '5', '0'), ('1', '5', '1'), ('1', '7', '0'), ('1', '7', '1'), ('2', '8', '0'), ('2', '8', '1'), ('2', '5', '0'), ('2', '5', '1'), ('2', '7', '0'), ('2', '7', '1'), ('4', '8', '0'), ('4', '8', '1'), ('4', '5', '0'), ('4', '5', '1'), ('4', '7', '0'), ('4', '7', '1')]
[i+j+k for i, j, k in list(product('124', '857', '01'))] # ['180', '181', '150', '151', '170', '171', '280', '281', '250', '251', '270', '271', '480', '481', '450', '451', '470', '471']




# repeat

from itertools import repeat
import numpy as np

[1, 2] * 3  # [1, 2, 1, 2, 1, 2]
list(range(1, 3)) * 3  # [1, 2, 1, 2, 1, 2]

np.repeat([1, 2], 3)  # [1, 1, 1, 2, 2, 2]


np.array([1, 2]) * 3  # [3, 6]

np.repeat([[1, 2], [3, 4]], 3)  # [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
np.repeat([[1, 2], [3, 4]], 3, axis=0)  # [[1, 2], [1, 2], [1, 2], [3, 4], [3, 4], [3, 4]]
np.repeat([[1, 2], [3, 4]], 3, axis=1)  # [[1, 1, 1, 2, 2, 2], [3, 3, 3, 4, 4, 4]]

list(map(lambda x: [x] * 3, range(1, 3)))  # [[1, 1, 1], [2, 2, 2]]
list(repeat([1, 2], 3))  # [[1, 2], [1, 2], [1, 2]]

list(map(pow, range(5), repeat(3)))  # [0, 1, 8, 27, 64] # TypeError: 'int' object is not iterable
list(map(pow, range(5), [3] * 5))  # [0, 1, 8, 27, 64]





# chain

from itertools import chain

list(chain(range(2), range(2, 4, 1)))  # [0, 1, 2, 3] # Merging two range into one

a_list = [[1, 2], [3, 4], [5, 6]]
list(chain(*a_list))  # [1, 2, 3, 4, 5, 6]
list(chain.from_iterable(a_list))  # [1, 2, 3, 4, 5, 6]





# groupby

from itertools import groupby

{k: list(v) for k, v in groupby('AAAABBBCCDAABBB')} # groups consecutive keys





# reduce

from functools import reduce

reduce((lambda x, y: x + y), [1, 2, 3, 4], 0)  # 10 # starts sum with 0
reduce((lambda x, y: x - y), [4, 3, 2, 1], 8)  # -1 
reduce((lambda x, y: x * y), [1, 2, 3, 4], 1)  # 24

import numpy as np
np.product([1, 2, 3, 4])




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
# defaultdict(<class 'list'>, {'first': ['blue', 'orange'], 'second': ['red'], 'third': ['green']})

dict(colors_list)
colors_list.get('first')
colors_list['first']


tree = lambda: defaultdict(tree)
some_dict = tree()
some_dict['colours']['favorite'] = 'yellow'

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

for i, char in enumerate([10, 11, 12], -3):
    print(i, char)

for i in enumerate('foo'):
    print(i)

for i in enumerate({'a': 'PHP', 'b': 'JAVA', 'c': 'PYTHON', 'd': 'NODEJS'}.items()):
    print(i)

list(enumerate(['apple', 'banana', 'grapes', 'pear'], 10))  # [(10, 'apple'), (11, 'banana'), (12, 'grapes'), (13, 'pear')] # enumerate a tuple





# round

round(15.456, 2)  # 15.46
round(15.456, 0)  # 15.0
int(15.456)  # 15





# Range

for i in range(15, 5, -1):
    print(i, end =' ')
print(end='\n')

range(ord('a'), ord('c'))





# Unicode

py_unicode = [U'\u0050', u'\u0059', u'\u0054',
              u'\u0048', u'\u004F', u'\u004E'
              ]
''.join(py_unicode)


# https://en.wikipedia.org/wiki/Bracket
u'\u0050' # P
u'\u003C' # <
u'\u003E' # >
u'\u0028' # (
u'\u0029' # )
u'\u005B' # [
u'\u005D' # ]
u'\u007B' # {
u'\u007D' # }
u'\u2264' # ≤
u'\u0061' # a


ord('u')
chr(117)

import html
html.unescape('&pound;682m')





# Chr

chr(97)
ord('a')

def abc(c_start, c_stop):
    for i in range(ord(c_start), ord(c_stop)):
        yield chr(i)
list(abc('a', 't'))





# type()
# isinstance()

type('Hello World')  # <class 'str'>
isinstance('Hello World', str)  # True
type(51)  # <class 'int'>
isinstance(51, int)  # True
type(5.0)  # <class 'float'>
type(5)  # <class 'int'>
type(5) == int  # True
isinstance(5.0, int)  # False # it's int, but object is float
isinstance(5.0, float) # True
(5.0).is_integer()  # True # checks if float is an int
(5).is_integer()  # 'int' object has no attribute 'is_integer'
isinstance({1, 2, 3, 4, 5}, set)  # True
isinstance((1, 2, 3, 4, 5), tuple)  # True
isinstance([1, 2, 3, 4, 5], list)  # True
isinstance({'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd'}, dict)  # True
"5".isdigit()  # True


class MyClass:
   message = 'Hello World'
class1 = MyClass()
isinstance(class1, MyClass)






# Counter

from collections import Counter

word = "antidisestablishmentarianism"
list(Counter(word).keys())  # ['a', 'n', 't', 'i', 'd', 's', 'e', 'b', 'l', 'h', 'm', 'r']
list(Counter(word).values())  # [4, 3, 3, 5, 1, 4, 2, 1, 1, 1, 2, 1]
list(Counter(word).elements())  # ['a', 'a', 'a', 'a', 'n', 'n', 'n', 't', 't', 't', 'i', 'i', 'i', 'i', 'i', 'd', 's', 's', 's', 's', 'e', 'e', 'b', 'l', 'h', 'm', 'm', 'r']
list(Counter(word).items())  # [('a', 4), ('n', 3), ('t', 3), ('i', 5), ('d', 1), ('s', 4), ('e', 2), ('b', 1), ('l', 1), ('h', 1), ('m', 2), ('r', 1)]
{k: v for k, v in Counter(word).items() if v > 2}  # {'a': 4, 'n': 3, 't': 3, 'i': 5, 's': 4} # filter pairs with val > 2
Counter(word).most_common(2) # the most repetitive/frequent elements # powtarzające, najczęstsze

# counter1 = Counter({'a': 4, 'b': 2, 'c': -2, 'd': 1})
counter1 = Counter({'a': 4, 'b': 2, 'c': -2, 'e': 0})
counter2 = Counter({'d': -12, 'b': 5, 'c': 4})

# Union # keys: or, value: max, positive 
counter1 | counter2  # Counter({'b': 5, 'a': 4, 'c': 4}) # positive max values from counter1 and counter2

# Or # 
counter1 or counter2  # Counter({'a': 4, 'b': 2, 'e': 0, 'c': -2}) # When using the or operator in Python between two objects, it doesn't combine these objects. Instead, it returns the first operand if it is truthy, or the second operand otherwise. 

# Addition # key: or, value: addition, positive
counter1 + counter2  # Counter({'b': 7, 'a': 4, 'c': 2}) # returns positive sum

# Intersection # key: and, val: min, positive
counter1 & counter2  # Counter({'b': 2}) # common positive min values from counter1 and counter2

# And
counter1 and counter2  # Counter({'b': 5, 'c': 4, 'd': -12}) # When using and (and is a boolean operator), it returns the second operand if the first operand is truthy; otherwise, it returns the first operand. In the context of Counter objects, both counter1 and counter2 are truthy, so counter1 and counter2 returns counter2.

# Subtraction # key: or, val: subtraction, positive
counter1 - counter2  # Counter({'d': 12, 'a': 4}) # returns positive subtraction

# .subtract like '-' with negative
counter1.subtract(counter2)  # Counter({'d': 12, 'a': 4, 'b': -3, 'c': -6})
counter1 = Counter({'a': 4, 'b': 2, 'c': -2, 'e': 0})

# .update like '+' with negative
counter1.update(counter2)  # Counter({'b': 7, 'a': 4, 'c': 2, 'd': -12})
counter1 = Counter({'a': 4, 'b': 2, 'c': -2, 'e': 0})





# lambda
string_to_number = lambda s: int(s)
cuboid = lambda l, w, d: l * w * d
cuboid(1, 2, 3)  # 6
cuboid = lambda _, __, ___: _*__*___
goals = lambda *x: sum(x)
goals(1, 2, 3)  # 6





# filter
# lambda function

# removes None and '', leaves ' '
foo = ['hey', 'there', '', 'whats', ' ', 'up', None]
list(filter(None, foo))  # ['hey', 'there', 'whats', ' ', 'up']
list(filter(bool, foo))  # ['hey', 'there', 'whats', ' ', 'up']

# removes '', leaves ' '
foo = ['hey', 'there', '', 'whats', ' ', 'up']
list(filter(len, foo))  # ['hey', 'there', 'whats', ' ', 'up'] # None has no len()
list(filter(lambda x: len(x), foo))  # ['hey', 'there', 'whats', ' ', 'up'] # None has no len()

# removes '' and ' '
foo = ['cat', 'mammal', 'goat', 'is', '', ' ']
list(filter(str.strip, foo))

''.join(filter(str.isdigit, '1is2')) # find digits in string
from string import digits
''.join(filter(lambda x: x in digits, '1is2')) # find digits in string
''.join(filter(lambda x: x.isdigit(), '1is2')) # find digits in string

''.join(filter(str.isalpha, '1is2')) # find letters in string
''.join(filter(str.islower, '1is2')) # find lowercase in string

# filter < 0
list(filter(lambda x: x < 0, range(-5, 5)))  # [-5, -4, -3, -2, -1]
def less_than_0(x):
    return True if x < 0 else False 
list(filter(less_than_0, range(-5, 5)))  # [-5, -4, -3, -2, -1]


# function that filters vowels
def filter_vowels(letter):
    vowels = ['a', 'e', 'i', 'o', 'u']
    if letter in vowels:
        return True
    else:
        return False

# filter works as yield
letters = 'abcdefghij'
filtered_vowels = filter(filter_vowels, letters)
filtered_vowels = filter(lambda x: True if x in "aeoiu" else False, letters)

for vowel in filtered_vowels:
    print(vowel)
next(filtered_vowels)
list(filtered_vowels)


sequence = [10, 2, 8, 7, 5, 4, 3, 11, 0, 1]
list(filter(lambda x: x > 4, sequence))  # [10, 8, 7, 5, 11]
list(map(lambda x: x ** 2, sequence))  # [100, 4, 64, 49, 25, 16, 9, 121, 0, 1]
filtered_result = map(lambda x: x ** 2 if not x % 2 else None, sequence) # functions return None
list(filtered_result)  # [100, 4, 64, None, None, 16, None, None, 0, None]
list(filter(None, filtered_result))  # [100, 4, 64, 16]

list_to_round = [2.6743, 3.63526, 4.2325, 5.9687967, 6.3265, 7.6988, 8.232, 9.6907]
list(map(round, list_to_round))  # [3, 4, 4, 6, 6, 8, 8, 10]
list(map(lambda x: round(x, 2), list_to_round))  # [2.67, 3.64, 4.23, 5.97, 6.33, 7.7, 8.23, 9.69]
list(map(np.round, list_to_round))  # [3.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 10.0]
list(map(lambda x: np.round(x, 2), list_to_round))  # [2.67, 3.64, 4.23, 5.97, 6.33, 7.7, 8.23, 9.69]

''.join(map(lambda s: s.upper(), 'ab cd'))  # 'AB CD'
"".join(map(str.upper, 'ab cd'))  # 'AB CD'


def myMapFunc(list1, list2):
   return list1 + list2
list(map(myMapFunc, [1, 2, 3], [4, 5, 6]))  # [5, 7, 9]




# map

list(map(lambda x: x ** 2 if not x % 2 else x ** 3, range(1, 11)))
[x ** 2 if not x % 2 else x ** 3 for x in range(1, 11)]

sum(map(lambda x: x[0] - x[1], [[10, 0], [3, 5], [5, 8]]))


# tuple of functions as inputs
def add(x):
    return x + x
def multiply(x):
    return x * x

for i in range(5):
    print(list(map(lambda x: x(i), (add, multiply))))

list(map(pow, [1, 2, 3], [1, 2, 3]))





# bool

bool(0)  # False
bool(None)  #  False

bool(1)  # True
bool(256)  # True

# use not str % 2
if not x % 2 is <==> if x % 2 == 0
not 2 % 2 # not 0 => 1





# None

import numpy as np

None_list = [None, True, False, True]
Not_None_list = None_list[None_list != None]  # select only True, not work
np.sum(Not_None_list)  # count True's in list





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
filter(~np.isnan(Nan_list), Nan_list)
Nan_list[~np.isnan(Nan_list)]





# Zip & UnZip

first_name = ['Joe','Earnst','Thomas']
last_name = ['Schmoe','Ehlmann','Fischer','Walter']
age = [23, 65, 11]
list(zip(first_name, last_name, age))

for f, l, a in zip(first_name, last_name, age):
    print('{} {} is {} years old'.format(f, l, a))

# unzip
people_list = (('Joe', 'Schmoe', 23),
               ('Earnst', 'Ehlmann', 65),
               ('Thomas', 'Fischer', 11)
               )

list(zip(*people_list))





# import string

import string 

string.digits  # '0123456789' .isnumeric()
string.hexdigits  # '0123456789abcdefABCDEF'
string.ascii_lowercase  # 'abcdefghijklmnopqrstuvwxyz' # .islower()
string.ascii_uppercase  # 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' # .isupper()
string.ascii_letters  # 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' # .isalpha()
string.punctuation  # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
string.capwords('some test')  # 'Some Test' # .upper





# yield statement
# generators

# The yield keyword in python works like a return with the only difference is that 
# instead of returning a value, it gives back a generator object to the caller.
# Python3 Yield keyword returns a generator to the caller and the execution of the 
# code starts only when the generator is iterated.


# Fibonacci generator
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# print all values
for i in fib(10):
    print(i)

# use next() to access the next element of a sequence
fib_10 = fib(10)
next(fib_10)

# iterate a string
iter_text = iter('some_text')
next(iter_text)


sequence = [10, 2, 8, 7, 5, 4, 3, 11, 0, 1]

def filter4(arg):
    for i in arg:
        if i > 4:
            yield i

list(filter4(sequence))
filtered_seq = filter4(sequence)
next(filtered_seq)


def hello_gen():
    yield 'H'
    yield 'E'
    yield 'L'
    yield 'L'
    yield 'O'

''.join(list(hello_gen()))

lett = hello_gen()
next(lett)





# import random

import random

random.seed(1) # Fixes the see of the random number generator.
random.choice([1, '2', 'tree'])
random.sample(range(10), 3)
random.random()
random.uniform(-1, 1)





# import math
# better numpy

import math
math.pi  # 3.141592653589793
math.sqrt(16)  # 4.0
math.sin(math.pi / 2)  # 1.0
math.factorial(5)  # 120

import numpy as np
np.pi  # 3.141592653589793
np.sqrt(16)  # 4.0
np.sin(np.pi / 2)  # 1.0
np.math.factorial(5)  # 120, deprecated





# import numpy as np

import numpy as np

np.version.version

np.zeros(5) + np.ones(5)  # array([1., 1., 1., 1., 1.])
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

x[2]  # 3
x[:2]  # [1, 2]
x + y  # [3, 6, 9]

X[:, 1]  # [2, 5] # all rows, 1st column
X[1]  # 1st row, all the same
X[1,]
X[1, :]

[1, 2] + [3, 4]  # [1, 2, 3, 4]
np.array([1, 2]) + np.array([3, 4])  # [4, 6]

ind = [1, 2]
y[ind]  # [4, 6]
y[[1, 2]]  # [4, 6]
y[1:3]  # [4, 6]

y > 4  # [False, False, True]
x[x > 2]  # [3]

np.arange(0, 50, 10)  # [0, 10, 20, 30, 40] # Return evenly spaced values within a given interval. (step, spacing)
np.linspace(0, 40, 5)  # [0, 10, 20, 30, 40] # Return evenly spaced numbers over a specified interval. num
np.meshgrid(np.arange(0, 100, 20), np.linspace(0, 100, 11))

np.random.random(3)  # [0.07208041, 0.25077184, 0.66814752]
z = np.random.random(3)
np.random.random((2, 3))  # [[0.03312195, 0.50714327, 0.5094406 ], [0.88311309, 0.29393605, 0.65023533]]
np.random.choice([1, '2', 'tree'])
np.random.seed(1)
np.random.uniform(-1, 1)
np.random.normal(size=(2, 3))
np.random.normal(0, 1, (2, 3))
np.random.randint(8, 10, (2, 3))


np.random.sample(3) # works as np.random.random(), not like random.sample()

np.any(z > 0.9)  # False
np.all(z >= 0.1)  # True

np.all(np.array([[1, 2], [1, 2]]) == 1, axis=0)  # [True, False]
np.all(np.array([[1, 2], [1, 2]]) == 1, axis=1)  # [False, False]

np.cumsum([1, 2, 3])  # [1, 3, 6]
# accumulate
np.add.accumulate([1, 2, 3])  # [1, 3, 6]
np.multiply.accumulate([1, 2, 3])  # [1, 2, 6]
np.subtract.accumulate([1, 2, 3])  # [1, -1, -4]
# reduce
np.add.reduce([1, 2, 3])  # 6
np.multiply.reduce([1, 2, 3])  # 6
np.subtract.reduce([1, 2, 3])  # -4

np.power([1, 2, 3], 2)  # [1, 4, 9]
np.power([1, 2, 3], [1, 2, 3])  # [1, 4, 27]
np.square([1, 2, 3])  # [1, 4, 9]
np.sqrt([9, 16, 25, 5])  # [3, 4, 25, 2.23]
np.sum([1, 2, 3], axis=0)  # 17 # more in # sum
np.prod([1, 2, 3])  # 6
np.product([1, 2, 3]) # for multiple list product see itertools.product # reduce(lambda)


np.min([9, 16, 25])  # 9
np.max([9, 16, 25])  # 25
np.mean([9, 16, 25])  # 16.6
np.median([9, 16, 25])  # 16.0
np.std([9, 16, 25])  # 6.54

np.round([17.05, 15.49], 0)  # 17.0
np.ceil(17.05)  # 18.0
np.floor(17.05)  # 17.0

l = np.array([1, 2])
m = np.array([3, 4])
# a1*b1 + a2*b2 
np.dot(l, m)  # 11

n = np.array([[1, 2], [3, 4]])
o = np.array([[5, 6], [7, 8]])
# 1*5 + 2*7 = 19
np.matmul(n, o)

np.linalg.det(n)  # -2


x1 = np.array([[1, 2], [3, 4]])
x2 = np.array([[5, 6], [7, 8]])
np.concatenate((x1, x2), axis=0)  # [[1, 2], [3, 4], [5, 6], [7, 8]])
np.concatenate((x1, x2), axis=1)  # [[1, 2, 5, 6], [3, 4, 7, 8]])
np.concatenate((x1, x2), axis=None)  # [1, 2, 3, 4, 5, 6, 7, 8]
np.concatenate(([[1, 2]], [[3, 4]]), axis=0) # [[1, 2], [3, 4]]
np.concatenate(([[1, 2]], [[3, 4]]), axis=1) # [1, 2, 3, 4]

np.stack((x1, x2))  # [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
np.stack((x1, x2), axis=1)  # [[[1, 2], [5, 6]], [[3, 4], [7, 8]]]
np.vstack((x1, x2))  # [[1, 2], [3, 4], [5, 6], [7, 8]]
np.hstack((x1, x2))  # [[1, 2, 3, 4], [5, 6, 7, 8]]

np.where(np.array([1, 2, 3]) == 3)  # array([2]) # index, find

# array diagonals
np.array([[1, 2], [3, 4]]).diagonal()  # [1, 4]
np.fliplr([[1, 2], [3, 4]]).diagonal()  # [2, 3]


# np.argsort returns to indices that would sort the given array.
# sort by index
np.argsort([3, 2, 1])  # [2, 1, 0]





# import time

import time

start_time = time.time()
time.time() - start_time

time.sleep(5)  # wait with output
print('This message will be printed after a wait of 5 seconds')





# import timeit

import timeit

timeit.timeit('foo = 10 * 5')
timeit.timeit(stmt='a=10; b=10; sum = a + b')

timeit.timeit(lambda: sum(i in "aeiou" for i in test_str), number=100000)
timeit.timeit(lambda: sum(True for i in test_str if i in "aeiou"), number=100000)
timeit.timeit(lambda: sum(map(lambda x: True if x in 'aeiou' else False, test_str)), number=100000)




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





# if __name__ == '__main__':

if __name__ == '__main__':
   print('abc')





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





# eval

eval('1' '+' '2') # evaluates a string





# RegEx

import re

r'some string' # raw string; means ignore all the escape characters (backslashes) in the string
print('some\ntext')
print(r'some\ntext')


"""
Pattern Description
\d	Matches any decimal digit; this is equivalent to the class [0-9].
\D	Matches any non-digit character
\s	Matches any whitespace character
\S	Matches any non-whitespace character
\w	Matches any alphanumeric character
\W	Matches any non-alphanumeric character.
Metacharacters
.	Matches with any single character except newline ‘\n'.
?	match 0 or 1 occurrence of the pattern to its left
+	1 or more occurrences of the pattern to its left
*	0 or more occurrences of the pattern to its left
^  	It represents the pattern present at the beginning of the string.
$	It represents the pattern present at the end of the string. 
\b	boundary between word and non-word. /B is opposite of /b
[..]	Matches any single character in a square bracket
\	It is used for special meaning characters like . to match a period or + for plus sign.
{n,m}	Matches at least n and at most m occurrences of preceding
a| b	Matches either a or b

MetaCharacters	Description
\	Used to drop the special meaning of character following it
[]	Represent a character class
^	Matches the beginning
$	Matches the end
.	Matches any character except newline
|	Means OR (Matches with any of the characters separated by it.
?	Matches zero or one occurrence
*	Any number of occurrences (including 0 occurrences)
+	One or more occurrences
{}	Indicate the number of occurrences of a preceding regex to match.
()	Enclose a group of Regex

List of special sequences 
Special Sequence	Description	Examples
\A	Matches if the string begins with the given character	\Afor 	for geeks
for the world
\b	Matches if the word begins or ends with the given character. \b(string) will check for the beginning of the word and (string)\b will check for the ending of the word.	\bge	geeks
get
\B	It is the opposite of the \b i.e. the string should not start or end with the given regex.	\Bge	together
forge
\d	Matches any decimal digit, this is equivalent to the set class [0-9]	\d	123
gee1
\D	Matches any non-digit character, this is equivalent to the set class [^0-9]	\D	geeks
geek1
\s	Matches any whitespace character.	\s	gee ks
a bc a
\S	Matches any non-whitespace character	\S	a bd
abcd
\w	Matches any alphanumeric character, this is equivalent to the class [a-zA-Z0-9_].	\w	123
geeKs4
\W	Matches any non-alphanumeric character.	\W	>$
gee<>
\Z	Matches if the string ends with the given regex	ab\Z	abcdab
abababab


"""

import re

# re.match() matches at the beginning
re.match(r"[:;][-~]?[\)D]", ";-) ';~)'")  # <re.Match object; span=(0, 3), match=';-)'>
re.match(r'abc', 'abc def')  # <re.Match object; span=(0, 3), match='abc'>
re.match(r'def', 'abc def')  # None
re.match(r'abc', 'abc def').start()  # 0
re.match(r'abc', 'abc def').end()  # 3
re.match(r'abc', 'abc def').span()  # (0, 3)
re.match(r'abc', 'abc def').group()  # 'abc'
re.match(r'abc', 'abc def').string  # 'abc def' # It returns a string passed into the function
re.match(r'^[;:][~-]?[)D]$', ':)').span()  # (0, 2)
re.match(r'\d+', '507f1f77bcf86cd799439016').group()  # 507
re.match(r'[0-9a-f]{24}$', '507f1f77bcf86cd799439016').group()  # '507f1f77bcf86cd799439016'
re.match(r'[\w\.-]+@[\w\.-]+\.[\w]+', 'someone@mail.com').group()  # 'someone@mail.com'
re.match(r'(https?://)?(www\.)?(?P<domain>[\w-]+)\..*$', 'https://www.codewars.com.com').group('domain')  # codewars
re.match(r'.*?IV', 'IV')
re.match(r'(\w+)\s(\w+)', 'Hello World').group()  # 'Hello World'
re.match(r'(\w+)\s(\w+)', 'Hello World').group(0)  # 'Hello World'
re.match(r'(\w+)\s(\w+)', 'Hello World').group(1)  # 'Hello'
re.match(r'(\w+)\s(\w+)', 'Hello World').group(2)  # 'World'
re.match(r'(\w+)\s(\w+)', 'Hello World').groups()  # ('Hello', 'World')

# re.search() function searches for a specified pattern anywhere in the given string and stops the search on the first occurrence.
re.search(r"[:;][-~]?[\)D]", "* ;-) ';~)'")  # <re.Match object; span=(2, 5), match=';-)'>
re.search(r'abc', 'abc def abc')  # <re.Match object; span=(0, 3), match='abc'>
re.search(r'def', 'abc def abc')  # <re.Match object; span=(4, 7), match='def'>
re.search(r'[^abc\s]', 'abc def')  # <re.Match object; span=(4, 5), match='d'> # ^ invert character class
re.search(r'def', 'abc def abc').group()  # 'def'
re.search(r'([a-zA-Z]+) (\d+)', 'June 24').group()  # 'June 24'

# re.findall() The object returns a list of all occurrences.
re.findall(r"[:;][-~]?[\)D]", "* ;-) ';~)'")  # [';-)', ';~)']
re.findall('abc', 'abc def abc')  # ['abc', 'abc']
re.findall('\d+', 'my number is 123 and not 456, and definitely not 789')  # ['123', '456', '789']
re.findall(r'^\w+', 'education is fun')  # ['education]
re.findall(r'\w+', 'education is fun')  # ['education', 'is', 'fun']
re.findall(r'\w+$', 'education is fun')  # ['fun']
re.findall(r'\d', '12ia22')  # ['1', '2', '2', '2']
re.findall(r'\d+', '12ia22')  # ['12', '22']
re.findall(r'[;:][~-]?[)D]', ' '.join([':)', ';(', ';}', ':-D']))  # [':)', ':-D']
re.findall(r'^\w', 'abc\ndef\nabc')  # ['a'] # search only first line
re.findall(r'^\w', 'abc\ndef\nabc', re.MULTILINE)  # ['a', 'd', 'a'] # search all lines
re.findall(r'[a-zA-Z]+\s\d+', 'June 24, August 9, Dec 12')  # ['June 24', 'August 9', 'Dec 12']
re.findall(r'([a-zA-Z]+)\s\d+', 'June 24, August 9, Dec 12')  # ['June', 'August', 'Dec']

# re.finditer() returns an iterator object of all matches in the target string
for i in re.finditer('abc', 'abc def abc'):
    print(i.span())  # (0, 3)\n(8, 11)
for i in re.finditer(r'([a-zA-Z]+)\s\d+', 'June 24, August 9, Dec 12'):
    print(i.span())  # (0, 7)\n(9, 17)\n(19, 25)

# re.split() splits on the pattern
re.split(r'\s','split the swords', 2)  # ['split', 'the', 'swords']  # stops after 2nd split
re.split(r's', 'split the swords')  # ['', 'plit the ', 'word', '']
re.split(r'[.,]', '100,000.00')  # ['100', '000', '00']
re.split(r'\D', '100,000.00')  # ['100', '000', '00']

# re.compile() makes a pattern
patt = re.compile(r'[aeoiu]')  # create a pattern
patt.match('igloo')  # <re.Match object; span=(0, 1), match='i'>
patt.match('not')  # None
patt2 = re.compile(r'(\w+) World')  # create a pattern
patt2.sub(r'\1 Earth', 'Hello World')  # 'Hello Earth'


# re.sub() replaces the pattern with string
re.sub(r'\n', '\n\r', 'abc\ndef\nabc', 1)  # 'abc\n\rdef\nabc' # steps afters 1st occurrence
re.sub(r'([A-Z])', r' \1', 'helloWorld')  # 'hello World' # break up came casing
re.sub(r' *?[#!].*', '', 'June #24\nAugust 9\nDe!c 12')  # 'June\nAugust 9\nDe' # why space ?
re.sub(r' *?["#", "!"].*', '', 'June #24\nAugust 9\nDe!c 12') # why space ?
re.sub(r'([a-zA-Z]+)\s(\d+)', r'\2 of \1', 'June 24, August 9, Dec 12')  # '24 of June, 9 of August, 12 of Dec'















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

datetime.datetime.fromtimestamp(1350508407) # 


# Get the modification time
dire = '/edx/translation/'
t = path.getmtime(os.getcwd()+'/'+dire+'dna.txt')
type(t)
print(datetime.time.ctime(path.getmtime(os.getcwd()+'/'+dire+'dna.txt')))
print(datetime.datetime.fromtimestamp(t))
print(datetime.datetime.fromtimestamp(path.getmtime(os.getcwd()+'/'+dire+'dna.txt')))

datetime.fromtimestamp(int("507F1f77bcf86cd799439011"[:8], 16))


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









# PDF
from PyPDF2 import PdfReader

reader = PdfReader('/home/ukasz/Documents/Nokia Academy/Entrance/Entrance_NA17_lecture.pdf')
numbe_of_pages = len(reader.pages)
page = reader.pages[0]
text = page.extract_text()
print(text)


# https://realpython.com/pdf-python/
# pdf_merging.py

from PyPDF2 import PdfFileReader, PdfFileWriter

def merge_pdfs(paths, output):
    pdf_writer = PdfFileWriter()

    for path in paths:
        pdf_reader = PdfFileReader(path)
        for page in range(pdf_reader.getNumPages()):
            # Add each page to the writer object
            pdf_writer.addPage(pdf_reader.getPage(page))

    # Write out the merged PDF
    with open(output, 'wb') as out:
        pdf_writer.write(out)


if __name__ == '__main__':
    paths = ['/home/ukasz/Documents/Nokia Academy/Entrance/Entrance_NA17_lecture.pdf',
             '/home/ukasz/Documents/Nokia Academy/Entrance/Entrance_NA18_final.pdf'
             ]
    merge_pdfs(paths, output='/home/ukasz/Documents/Nokia Academy/Entrance/Entrance.pdf')

if __name__ == '__main__':
    paths = ['/home/ukasz/Documents/Nokia Academy/Week1/IP_Basics_10052022.pdf',
             '/home/ukasz/Documents/Nokia Academy/Week1/NA Wro 18 I&V - 01. Fundamental_measusrements.pdf',
             '/home/ukasz/Documents/Nokia Academy/Week1/NA Wro 18 I&V - 02. Fundamentals - Env, Distortions, Multiple Access.pdf',
             '/home/ukasz/Documents/Nokia Academy/Week1/NA Wro 18 I&V - 03. LTE - Architecture.pdf'
             ]
    merge_pdfs(paths, output='/home/ukasz/Documents/Nokia Academy/Week1.pdf')

if __name__ == '__main__':
    paths = ['/home/ukasz/Documents/Nokia Academy/Week2/NA Wro 18 I&V - 06. 5G - 01. Introduction_to_5G.pdf',
             '/home/ukasz/Documents/Nokia Academy/Week2/NA_ENG_Physical_Layer_LysienL_ApiyoA.pdf',
             '/home/ukasz/Documents/Nokia Academy/Week2/1. MCC MNC.pdf',
             '/home/ukasz/Documents/Nokia Academy/Week2/2. EMM ECM States and TAU Process.pdf',
             '/home/ukasz/Documents/Nokia Academy/Week2/3. NA18_Testing_1.pdf',
             '/home/ukasz/Documents/Nokia Academy/Week2/4. NA Wro 18 I&V - 02. Fundamentals - testing in Nokia 2022.pdf'
             ]
    merge_pdfs(paths, output='/home/ukasz/Documents/Nokia Academy/Week2.pdf')

if __name__ == '__main__':
    paths = ['/home/ukasz/Documents/Nokia Academy/Week3/NA Wro 18 I&V - Mobility_management.pdf',
             '/home/ukasz/Documents/Nokia Academy/Week3/NA Wro 18 I&V - Mobility_management II.pdf',
             '/home/ukasz/Documents/Nokia Academy/Week3/NA Wro 18 I&V - 14. 5G - 01. 5G_architecture_and_configurations.pdf',
             '/home/ukasz/Documents/Nokia Academy/Week3/NA Wro 18 I&V - 14. 5G - 02. Introduction to 5G Nokia Solutions.pdf',
             '/home/ukasz/Documents/Nokia Academy/Week3/Labolatory_03.06_notes.pdf',
             ]
    merge_pdfs(paths, output='/home/ukasz/Documents/Nokia Academy/Week3.pdf')






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
    def fun_in_decor(x):
        print('You think so?')
        a_func(x)
        print('Look again')
    return fun_in_decor

@ new_decor
def f_without(x):
    print('I have no decorators')
f_without('a')

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






# Higher-order function
# https://en.wikipedia.org/wiki/Higher-order_function
def twice(f):
    def result(x):
        return f(f(x))
    return result

plus_three = lambda i: i + 3
g = twice(plus_three)
g(7)

@twice
def g(i):
    return i + 3
g(7)


