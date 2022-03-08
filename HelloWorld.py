# Almost everything in Python is an object, with its properties and methods.
# A Class is like an object constructor, or a "blueprint" for creating objects.

# Parameters to functions are references to objects, which are passed by value. 
# When you pass a variable to a function, python passes the reference to the object 
# to which the variable refers (the value).



# Guru99 tutorial

## Python Programming Basics for Beginners
# Print

print("USA")
print("UK")
print(2 * "\n")
print("Canada")
print("\n\n")
print("Germany", end = "\n")
print("France", end = " @# ")
print("Japan")
print("Poland")

print(8 * "\n")
print("\n\n\n")

a = 100
print(a)
b = 99
print('ABC' + str(b))



# Python Variable Types: Local & Global

def someFunction():
    a='Twoja Stata'
    print(a)
someFunction()
print(a)

def someFunction():
    global a
    print(a)
    a = 101
someFunction()
print(a)


print(a)
del (a, b)
print(a)




## Python Data Structure
# Tuple
# To perform different task, tuple allows you to use many built-in functions like all(), 
# any(), enumerate(), max(), min(), sorted(), len(), tuple(), etc.

tup1 = ('Robert', 'Carlos', '1965', 'Terminator 1995', 'Actor', 'Florida')
tup2 = (1, 2, 3, 4, 5, 6, 7)
print(tup2[1:4])

x = ('Guru99', 20, 'Education')  # tuple packing
(company, emp, profile) = x  # tuple unpacking
print(company)
print(emp)
print(profile)

(key_1, key_2) = ('val_1', 'val_2')
key_2


a = (5, 6)
b = (6, 4)
if a > b:
   print('a is bigger')
else:
   print('b is bigger')



x = ('a', 'b', 'c', 'd', 'e')
x[2:4]
y = max((1, 3, 5, 2))
y



# Dictionary
a = {'x': 100, 'y': 200}
a.items()
a.keys()
a.values()
del(a)

Dict = {'Tim': 18,
        'Charlie': 12,
        'Tiffany': 22,
        'Robert': 25
        }
Dict['Tiffany']
Dict.get('Tiffany')
Boys = {'Tim': 18, 'Charlie': 12, 'Robert': 25}
Girls = {'Tiffany': 22}

studentX = Boys.copy()
id(Boys)
id(studentX)
studentX
Boys.update({'Sarah': 9})
Dict
del Dict['Sarah']
Dict.pop('Sarah')
print(Dict)

print('Students Name: %s' % list(Dict.items()))

Dict = {'Tim': 18, 'Charlie': 12, 'Tiffany': 22, 'Robert': 25}
Boys = {'Tim': 18, 'Charlie': 12, 'Robert': 25}

for key in Dict.keys():
   if key in Boys.keys():
      print(True)
   else:
      print(False)

students = list(Dict.keys())
students.sort()
print(students)
for student in students:
   print(': '.join((student, str(Dict[student]))))

for student in Dict.keys():
   print(student,':', Dict[student])

for student in Dict.keys():
   print('{}: {}'.format(student, Dict[student]))


print('Length : %d' % len(Dict))
print(len(Students))
print('variable Type: %s' % type(Dict))
print('printable string:%s' % str(Dict))

Dict_copy = Dict.copy()
print(Dict_copy)
Dict_copy.update({'Tim': 80})
print(Dict_copy)

Here is the list of all Dictionary Methods
Method	Description	Syntax
copy()	Copy the entire dictionary to new dictionary	dict.copy()
update()	Update a dictionary by adding a new entry or a key-value pair to anexisting entry or by deleting an existing entry.	Dict.update([other])
items()	Returns a list of tuple pairs (Keys, Value) in the dictionary.	dictionary.items()
sort()	You can sort the elements	dictionary.sort()
len()	Gives the number of pairs in the dictionary.	len(dict)
Str()	Make a dictionary into a printable string format	Str(dict)


my_dict1 = {'username': 'XYZ', 'email': 'xyz@gmail.com', 'location': 'Mumbai'}
my_dict2 = {'firstName': 'Nick', 'lastName': 'Price'}
my_dict1.update(my_dict2)
print(my_dict1)
my_dict = {**my_dict1, **my_dict2}
{*my_dict1, *my_dict2}
'email' in my_dict



# Dictionary
my_dict = {'Name': [], 'Address': [], 'Age': []}
print(my_dict['Name'])
my_dict['Name'].append('Guru')
my_dict['Name'] = 'G'
print(my_dict['Name'])
my_dict['Address'].append('Mumbai')
my_dict['Age'].append(30)
print(my_dict)
print('username:', my_dict['Name'])


del my_dict['Name']
print(my_dict)
del my_dict
my_dict.clear()
my_dict.pop('Name')
print(my_dict)
my_dict['name'] = 'Nick'
print(my_dict)
my_dict.update({'name2': 'Nick2'})

my_dict1['my_dict1'] = my_dict
print(my_dict1)

Important built-in methods on a dictionary:
Method	Description
clear()	It will remove all the elements from the dictionary.
append()	It is a built-in function in Python that helps to update the values for the keys in the dictionary.
update()	The update() method will help us to merge one dictionary with another.
pop()	Removes the element from the dictionary.




# Operators
# Arithmetic Operators
x = 4
y = 5
print(x + y)

# Comparison Operators
print('x > y is', x != y)

# Assignment Operators
x += 6
print("x + 6 =", x)

# Logical Operators or Bitwise Operators
a = True
b = False
print('a and b is', a and b)

# Membership Operators
lis1 = [1, 2, 3, 4, 5]
if (y in lis1):
   print('Yes')
else:
   print('No')

if (y+100 not in lis1):
   print('%d + 100 is not in %s' % (y, str(lis1)))


# Identity Operators
is, is not

if x is not y:
   print(x, y)

Operators (Decreasing order of precedence)	Meaning
**	Exponent
*, /, //, %	Multiplication, Division, Floor division, Modulus
+, –	Addition, Subtraction
<= < > >=	Comparison operators
= %= /= //= -= += *= **=	Assignment Operators
is is not	Identity operators
in not in	Membership operators
not or and	Logical operators

print(9 // 4)
print(9 % 4)
print(9 ** 4)




# Arrays

# Following tables show the type codes:

# Type code	Python type	C Type	Min size(bytes)
'u'	Unicode character	Py_UNICODE	2
'b'	Int	Signed char	1
'B'	Int	Unsigned char	1
'h'	Int	Signed short	2
'l'	Int	Signed long	4
'L'	Int	Unsigned long	4
'q'	Int	Signed long long	8
'Q'	Int	Unsigned long long	8
'H'	Int	Unsigned short	2
'f'	Float	Float	4
'd'	Float	Double	8
'i'	Int	Signed int	2
'I'	Int	Unsigned int	2


import array as myarray
from array import array
balance = myarray.array('i', [300, 200, 100, 500])
print(balance[1])
print(balance[1:3])

balance.insert(2, 150)
print(balance)
balance[0] = 5000
print(balance)

balance.pop(0)
del balance[0]
balance.remove(5000)
print(balance)
print(balance + balance)

# element = my_list.clear()
# print(element)
# print(my_list)

print(balance.index(100))
balance.reverse()
print(balance)
print(balance.count(200))

# Unicode
p = array('u',[u'\u0050',u'\u0059',u'\u0054',u'\u0048',u'\u004F',u'\u004E'])
print(p)
q = (p.tounicode)()
print(q)

for x in balance:
   print(x)


abc = myarray.array('d', [2.5, 4.9, 6.7])
print('Array first element is:', abc[0])
print('Array last element is:', abc[-1])
print(abc.index(2.5))

print(abc + abc)

abc = myarray.array('q', [3, 9, 6, 5, 20, 13, 19, 22, 30, 25])
print(abc[2:-1])



# Python Conditional Loops
# If

def main():
   x, y = 8, 9
   if x < y:
       st = 'x is less than y'
   elif x == y:
       st = 'x is same as y'
   else:
       st = 'x is greater than y'
   print(st)

if __name__ == '__main__':
   main()


def main():
   x, y = 10, 8
   st = 'x is less than y' if (x < y) else 'x is greater than or equal to y'
   print(st)

if __name__ == '__main__':
   main()

total = 100
#country = 'US'
country = 'AU'
if country == 'US':
    if total <= 50:
        print('Shipping Cost is  $50')
elif total <= 100:
        print('Shipping Cost is $25')
elif total <= 150:
	    print('Shipping Costs $5')
else:
        print('FREE')
if country == 'AU': 
	  if total <= 50:
	    print('Shipping Cost is  $100')
else:
	    print('FREE')


def SwitchExample(argument):
    switcher = {
        0: ' This is Case Zero ',
        1: ' This is Case One ',
        2: ' This is Case Two ',
    }
    return switcher.get(argument, 'nothing')
    # return switcher[argument]


if __name__ == '__main__':
    argument = 0
    print(SwitchExample(argument))




# For & While

x = 0
while x < 4:
   print(x)
   x += 1

for x in range(2, 4):
   print(x)

for x in enumerate(range(2, 4)):
   print(x)

Months = ['Jan', 'Feb', 'Mar', 'April', 'May', 'June']
for m in Months:
   print(m)

for x in range(10, 20):
   # if x==15: break
   if (x % 2 == 0): continue
   print(x)

for i, m in enumerate(Months):
   print(i, m)

for x in '234':
   print('value :', x)






# Break & Continue

my_list = ['Siya', 'Tiya', 'Guru', 'Daksh', 'Riya', 'Guru']
for i in range(len(my_list)):
   print(i, my_list[i])
   if my_list[i] == 'Guru':
       print('Found break')
       break
       print('Hidden')
   print('Też nie see')

print(len(my_list))

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

def my_func():
   print('hell pass hospital')
   pass

my_func()


# Pass statement in for-loop
test = "Guru"
for i in test:
    if i == 'r':
        print('Pass executed')
        pass
    print(i)
print('\n')









# Python OOPs: Class, Object, Inheritance and Constructor 
# A Class in Python is a logical grouping of data and functions. 
# It gives the freedom to create data structures that contains arbitrary 
# content and hence easily accessible.

class mClass():
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
    def method1(self):
        myClass.method1(self)
        # print('childClass Method1')

    def method2(self):
        print('childClass method2')

def main1():
    # exercise the class methods
    c2 = childClass()
    c2.method1()
    # c2.method2()

if __name__ == '__main__':
    main1()


# Constructors
# A constructor is a class function that instantiates an object to 
# predefined values. It begins with a double underscore (_). 
# It __init__() method

class User():
    name = ''

    def __init__(self, name):
        self.name = name

    def sayHello(self):
        print('Welcome to Guru99, ' + self.name)

User1 = User('Alex')
User1.sayHello()

User('Ukasz').sayHello()




# Strings
# In Python everything is object and string are an object too.

var1 = 'Guru99!'
var2 = 'Software Testing'
print('var[0]:', var1[0])
print('var2[1:5]:', var2[1:5])
print('u' in var1)
print('8' in var1)
print('\n')
# r, R - raw string suppresses actual meaning of escape characters.
print(r'\n')
print('/n')
print(R'/n')

name = 'guru'
number = 99
print('%s %d' % (name, number))
print(name, '', number)
print(name + str(number))
print(name + ' ' + str(number))

print(name * 2)
print(number * 2)
print(name[1:3]+ name)
print(name[1:3], name)

oldstring = 'nie lubię Cię'
newstring = oldstring.replace('nie', '')
print(newstring)
# Strings are immutable.
oldstring.replace('nie lubię Cię', 'lubię')
print(oldstring)

x = 'Guru99'
x.replace('Guru99','Python')
print(x)

print(name.upper())
print(name.capitalize())
print(':'.join(name))
print(':'.join((name, str(number))))
print(name.join(str(number)))
print('AB'.join(name))
str1 = '12345'
print(''.join(str1))
print(''.join(reversed(str1)))
word = 'abc def ghi'
print(word.split(' '))
print(word.split('d'))




# Strip
str1 = ' abc '
str2 = '*abc*'
str3 = 'abc'
print(str1.strip())
print(str2.strip('*'))
print(str1.strip('c a'))




# Count
print(str1.count('a'))
print('abcda'.count('a', 2, ))




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
class MyClass1():
   msg1 = 'twoja'
   msg2 = 'stara'

print('tak, to {c.msg1} {c.msg2}'.format(c=MyClass1))

# Using dictionary with format()
my_dict = {'msg1': "twoja", 'msg2': "stara"}
print('tak, to {m[msg1]} {m[msg2]}'.format(m=my_dict))
my_dict = {'msg1': [], 'msg2': []}
my_dict['msg1'].append('twoja')
my_dict['msg1'].append('nowa')
my_dict['msg2'].append('stara')



# Length
print((str1))
print(len(str1))
print(len(my_dict))
list1 = ["Tim", "Charlie", "Tiffany", "Robert"]
print(len(list1))
Tup = ('Jan', 'feb', 'march')
print("The length of the tuple is", len(Tup))
arr1 = ['Tim', 'Charlie', 'Tiffany', 'Robert']
print("The length of the Array is", len(arr1))




# Find
print('text', str1.find('c'))
print('text', str1.find('c', 1, 5))
mystring = 'Meet Guru99 Tutorials Site.Best site for Python Tutorials!'
print('The position of Tutorials using find() : ', mystring.find('Tutorials'))
print('The position of Tutorials using find() : ', mystring.find('Tutorials', 20))
print('The position of Tutorials using rfind() : ', mystring.rfind('Tutorials'))
print('The position of Tutorials using index() : ', mystring.index('Tutorials'))


my_string = 'test string test, test string testing, test string test string'
startIndex = 0
count = 0
for i in range(len(my_string)):
    k = my_string.find('test', startIndex)
    if k != -1:
        startIndex = k+1
        count += 1
    else:
        break
print('The total count of substring test is:', count)

my_string = 'test string test, test string testing, test string test string'
startIndex = 0
count = 0
i = 0
while True:
    k = my_string.find('test', startIndex)
    if k != -1:
        startIndex = k + 1
        count += 1
    else:
        break
    i += 1
print(count)




# Functions
# Main Function & Method Example: Understand __main__
# Python main function is a starting point of any program. When the program is run, 
# the python interpreter runs the code sequentially. Main function is executed only 
# when it is run as a Python program. It will not run the main function if it imported 
# as a module.

def main1():
    print('hello')

if __name__ == '__main__':
    main1()

main1()



# A Function in Python is a piece of code which runs when it is referenced.
def sq(x = 5):
    return x * x
sq()


def multi(x, y=0):
    print('x =', x)
    print('y =', y)
    return x * y

multi(y=2, x=4)


def fun2(*args):
    # print(args)
    return args
fun2(1, 2, 3, 4, 5)


# Lambda Functions
# A Lambda Function in Python programming is an anonymous function or a function
# having no name. It is a small and restricted function having no more 
# than one line.

adder = lambda x, y=2: x + y
adder(1)

# What a lambda returns
string = 'some kind of a useless lambda'
print(lambda string: print(string))

# What a lambda returns #2
string = 'some kind of a useless lambda'
(lambda str: print(str))(string)


# A REGULAR FUNCTION
def guru(funct, *args):
   funct(*args)

def printer_one(arg):
   return print(arg)

def printer_two(arg):
   print(arg)

# CALL A REGULAR FUNCTION
guru(printer_one, 'printer 1 REGULAR CALL')
guru(printer_two, 'printer 2 REGULAR CALL \n')

# CALL A REGULAR FUNCTION THRU A LAMBDA
guru(lambda: printer_one('printer 1 LAMBDA CALL'))
guru(lambda: printer_two('printer 2 LAMBDA CALL \n'))


(lambda x: x + x)(2)

# lambdas in filter()
sequence = [10, 2, 8, 7, 5, 4, 3, 11, 0, 1]
filtered_result = filter(lambda x: x > 4, sequence)
print(list(filtered_result))
print(filtered_result)


# lambdas in map()
def filter2(arg):
    for i in arg:
        if i > 4:
            print(i)
filter2(sequence)

sequence = [10, 2, 8, 7, 5, 4, 3, 11, 0, 1]
def filter3(arg):
    lis = []
    # it = 0
    for i in arg:
        if i > 4:
            lis.append(i)
            # it += 1
    return lis
filter3(sequence)

def filter4(arg):
    for i in arg:
        if i > 4:
            yield(i)
list(filter4(sequence))


filtered_result = map(lambda x: x * x, sequence)
print(list(filtered_result))


# lambdas in reduce()
from functools import reduce
sequences = [1, 2, 3, 4, 5]
product = reduce(lambda x, y: x * y, sequences)
product




# Abs

int_num = -25
float_num = -10.50
complex_num = (3 + 10j)
print('The absolute value of an integer number is:', abs(int_num))
print('The absolute value of a float number is:', abs(float_num))
print('The magnitude of the complex number is:', abs(complex_num))




# Round()
import random

def truncate(num):
   return int(num * 1000) / 1000


arr = [random.uniform(0.01, 0.05) for _ in range(100)]
sum_num = 0
sum_trun = 0
for i in arr:
   sum_num += i
   sum_trun = truncate(sum_trun + i)

print('Testing by using truncating upto 3 decimal places')
print('The original sum is = ', sum_num)
print('The total using truncate = ', sum_trun)
print('The difference from original - truncate = ', sum_num - sum_trun)
print('\n\n')
print('Testing by using round() upto 3 decimal places')
sum_num1 = 0
sum_truncate1 = 0
for i in arr:
   sum_num1 = sum_num1 + i
   sum_truncate1 = round(sum_truncate1 + i, 3)

print('The original sum is =', sum_num1)
print('The total using round = ', sum_truncate1)
print('The difference from original - round =', sum_num1 - sum_truncate1)


import numpy as np
arr = [-0.341111, -1.455098989, 4.232323, -0.3432326, 7.626632, 5.122323]
arr1 = np.round(arr, 2)
print(arr1)


import decimal
round_num = 15.456
final_val = round(round_num, 2)
# Using decimal module
final_val1 = decimal.Decimal(round_num).quantize(decimal.Decimal('0.00'), rounding=decimal.ROUND_CEILING)
final_val2 = decimal.Decimal(round_num).quantize(decimal.Decimal('0.00'), rounding=decimal.ROUND_DOWN)
final_val3 = decimal.Decimal(round_num).quantize(decimal.Decimal('0.00'), rounding=decimal.ROUND_FLOOR)
final_val4 = decimal.Decimal(round_num).quantize(decimal.Decimal('0.00'), rounding=decimal.ROUND_HALF_DOWN)
final_val5 = decimal.Decimal(round_num).quantize(decimal.Decimal('0.00'), rounding=decimal.ROUND_HALF_EVEN)
final_val6 = decimal.Decimal(round_num).quantize(decimal.Decimal('0.00'), rounding=decimal.ROUND_HALF_UP)
final_val7 = decimal.Decimal(round_num).quantize(decimal.Decimal('0.00'), rounding=decimal.ROUND_UP)

print('Using round()', final_val)
print('Using Decimal - ROUND_CEILING ', final_val1)
print('Using Decimal - ROUND_DOWN ', final_val2)
print('Using Decimal - ROUND_FLOOR ', final_val3)
print('Using Decimal - ROUND_HALF_DOWN ', final_val4)
print('Using Decimal - ROUND_HALF_EVEN ', final_val5)
print('Using Decimal - ROUND_HALF_UP ', final_val6)
print('Using Decimal - ROUND_UP ', final_val7)




# Range
for i in range(3, 10, 2):
   print(i, end=' ')

for i in range(15, 5, -1):
   print(i, end =' ')

arr_list = ['Mysql', 'Mongodb', 'PostgreSQL', 'Firebase']
for i in arr_list:
   print(i, end=' ')

print(list(range(10)))

print(range(ord('a'), ord('c')))
print(chr(97))

def abc(c_start, c_stop):
   for i in range(ord(c_start), ord(c_stop)):
      yield chr(i)
print(list(abc('a', 't')))

def range1(x):
   for i in range(x):
      yield i
print(list(range1(5)))

def range2(x):
   ak = []
   for i in range(x):
      ak.append(i)
   return ak
print(list(range2(5)))

startvalue = range(5)[0]
print('The first element in range is = ', startvalue)
secondvalue = range(5)[1]
print('The second element in range is = ', secondvalue)
lastvalue = range(5)[-1]
print('The first element in range is = ', lastvalue)


from itertools import chain

print('Merging two range into one')
frange = chain(range(10), range(10, 20, 1))
list(frange)


import numpy as np 
for i in np.arange(10):
   print(i, end =' ')  

import numpy as np 
for i in np.arange(0.5, 1.5, 0.2):
   print(round(i, 1), end ='\n') 




# Map

def square(x):
   return x * x

my_list = [2, 3, 4, 5, 6, 7, 8, 9]
up_list = map(square, my_list)
print(list(up_list))

list(map(lambda x: x**2, my_list))

my_list = [2.6743, 3.63526, 4.2325, 5.9687967, 6.3265, 7.6988, 8.232, 9.6907]
updated_list = list(map(round, my_list))


def upfun(s):
    return s.upper()

my_str = 'welcome to guru99 tutorials1!'
updated_list1 = list(map(upfun, my_str))
''.join(updated_list1)

updated_list1 = map(lambda s: s.upper(), my_str)


my_tuple = ('php', 'java', 'python', 'c++', 'c')
updated_list = map(upfun, my_tuple)
print(list(updated_list))

my_list = [2, 3, 4, 5, 6, 7, 8, 9]
updated_list = map(lambda x: x + 10, my_list)
print(list(updated_list))


def myMapFunc(list1, list2):
   return list1 + list2

my_list1 = [2, 3, 4, 5, 6, 7, 8, 9]
my_list2 = [4, 8, 12, 16, 20, 24, 28]

updated_list = map(myMapFunc, my_list1, my_list2)
list(updated_list)


def myMapFunc(list1, tuple1):
   return list1 + '_' + tuple1

my_list = ['a', 'b', 'b', 'd', 'e']
my_tuple = ('PHP', 'Java', 'Python', 'C++', 'C')
list(map(myMapFunc, my_list, my_tuple))

list(map(lambda x, y: x + '_' + y, my_list, my_tuple))




# Timeit"

import timeit
timeit.timeit('foo = 10 * 5')
timeit.timeit(stmt='a=10; b=10; sum = a + b')

import timeit
import_module = 'import random'
testcode = '''
def test():
   return random.randint(10, 100)
'''
print(timeit.repeat(stmt=testcode, setup=import_module))
print(timeit.timeit(stmt=testcode, setup=import_module))

>python -m timeit -s 'text="hello world"'




# Yield
# The yield keyword in python works like a return with the only difference is that 
# instead of returning a value, it gives back a generator object to the caller.
# Python3 Yield keyword returns a generator to the caller and the execution of the 
# code starts only when the generator is iterated.

def testyield():
   yield 'Welcome to Guru99 Python Tutorials'
output = testyield()
for i in output:
   print(i)
list(testyield())[0]
print(next(testyield()))


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
// wyświetla tylko pierwszą
next(generator())


# Normal function
def normal_test():
   return 'Hello World'

# Generator function
def generator_test():
   yield 'Hello World'

print(normal_test())  # call to normal function
print(generator_test())  # call to generator function
for i in generator_test():
   print(i)

print(list(generator_test()))
print(next(generator_test()))


def even(n):
   for i in range(n):
         if i % 2 == 0:
            yield i

print(list(even(10)))
for i in even(10):
   print(i)

num = even(10)
print(next(num))
print(next(num))
print(next(num))

print(list(num))
print(list(num))


def fibf(n):
   c1, c2 = 0, 1
   for i in range(n):
      yield c1
      c3 = c1 + c2
      c1 = c2
      c2 = c3

fib = fibf(7)
list(fib)


def test(n):
   return n * n

def getSquare(n):
   for i in range(n):
      yield test(i)

sq = getSquare(10)
print(next(sq))
print(list(sq))





# Queue

import queue
q1 = queue.Queue(2)
q1.empty()
q1.put(10)
q1.put(5)
q1.full()
q1.qsize()
item1 = q1.get()
print('The item removed from the queue is ', item1)
item1 = q1.get()
print('The item removed from the queue is ', item1)


import queue
q1 = queue.Queue()
for i in range(5):
   q1.put(i)

while not q1.empty():
   print("er", q1.get())


q2 = queue.LifoQueue()
for i in range(5):
   q2.put(i)

while not q2.empty():
   print("sdf", q2.get(), end=" ")

import queue
q1 = queue.Queue()
q1.put(11)
q1.put(5)
q1.put(4)
q1.put(21)
q1.put(3)
q1.put(10)
print(list(q1.queue))

# using bubble sort on the queue
n = q1.qsize()
for i in range(n):
   x = q1.get()  # the element is removed
   # print('x ', str(i), ' ', str(x))
   for j in range(n - 1):
      y = q1.get()  # the element is removed
      # print('y ', str(j), ' ', str(y))
      if x > y:
         q1.put(y)  # the smaller one is put at the start of the queue
      else:
         q1.put(x)  # the smaller one is put at the start of the queue
         x = y  # the greater one is replaced with x and compared again with next element
   print(list(q1.queue))
   q1.put(x)

while q1.empty() == False:
   print(q1.get(), end=" ")


import queue
q1 = queue.Queue()
q1.put(11)
q1.put(5)
q1.put(4)
q1.put(21)
q1.put(3)
q1.put(10)

def reverseQueue(q1src, q2dest):
   buffer = q1src.get()
   if (q1src.empty() == False):
      reverseQueue(q1src, q2dest)  # using recursion
      q2dest.put(buffer)
   return q2dest

q2dest = queue.Queue()
qReversed = reverseQueue(q1, q2dest)

while (qReversed.empty() == False):
   print(qReversed.get(), end=" ")









# Counter
# Python Counter is a container that will hold the count of each of the elements present in the container. 
# The counter is a sub-class available inside the dictionary class. Using the Python Counter tool, you can count 
# the key-value pairs in an object, also called a hash table object.
from collections import Counter
list1 = ['x','y','z','x','x','x','y', 'z']
Counter(list1)

my_str = "Welcome to Guru99 Tutorials!"
Counter(my_str)

dict1 =  {'x': 4, 'y': 2, 'z': 2, 'z': 2}
Counter(dict1)

tuple1 = ('x', 'y', 'z', 'x', 'x', 'x', 'y', 'z')
Counter(tuple1)

count1 = Counter()
count1.update('Welcome to Guru99 Tutorials!')
print(count1)
count1.update('Some txt? uuu')
print(count1)

print('%s: %d' % ('u', count1['u']))
count1['u']
for char in 'Guru':
   print(char, count1[char])


from collections import Counter
dict1 =  {'x': 4, 'y': 2, 'z': 2}
del dict1['x']
Counter(dict1)


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


counter1 =  Counter({'x': 5, 'y': 2, 'z': -2, 'x1':0})
elements1 = counter1.elements()
list(counter1.elements())


counter1.most_common(2)


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












# Enumerate

my_list = ['D', 'B', 'C', 'A']
tuple(enumerate(my_list))

for i in enumerate(my_list):
   print(i)

for i in enumerate(my_list, -5):
   print(i)

my_tuple = ('E', 'B', 'C', 'D', 'A')
for i in enumerate(my_tuple):
   print(i)

str1 = 'foo'
for i in enumerate(str1):
   print(i)

my_dict = {'a': 'PHP', 'b': 'JAVA', 'c': 'PYTHON', 'd': 'NODEJS'}
for i in enumerate(my_dict.items()):
    print(i)






# time sleep():

import time
print('Welcome to guru99 Python Tutorials')
time.sleep(1.5)
print('This message will be printed after a wait of 5 seconds')


import time
print('Code Execution Started')
def display():
    print('Welcome to Guru99 Tutorials')
    time.sleep(5)

display()
print('Function Execution Delayed')


import asyncio
print('Code Execution Started')
async def display():
   await asyncio.sleep(5)
   print('Welcome to Guru99 Tutorials')

asyncio.run(display())


from threading import Event
print('Code Execution Started')
def display():
   print('Welcome to Guru99 Tutorials')

Event().wait(5)
display()


from threading import Timer
print('Code Execution Started')
def display2():
   print('Welcome to Guru99 Tutorials')

t = Timer(5, display2)
t.start()








# type() and isinstance()

str_list = 'Welcome to Guru99'
age = 50
pi = 3.14
c_num = 3j+10
my_list = ['A', 'B', 'C', 'D']
my_tuple = ('A', 'B', 'C', 'D')
my_dict = {'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd'}
my_set = {'A', 'B', 'C', 'D'}

print('The type is : ', type(str_list))
print('The type is : ', type(age))
print('The type is : ', type(pi))
print('The type is : ', type(c_num))
print('The type is : ', type(my_list))
print('The type is : ', type(my_tuple))
print('The type is : ', type(my_dict))
print('The type is : ', type(my_set))

class test:
    s = 'testing'
t = test()
print(type(t))

class MyClass:
    x = 'Hello World'
    y = 50

t1 = type('NewClass', (MyClass, ), dict(x = 'Hello World2', y = 60))
print(type(t1))
print(vars(t1))
type(MyClass)
vars(MyClass)


age = isinstance(51, int)
print('age is integer :', age)
message = isinstance('Hello World', str)
print('message is a string:', message)
my_set = isinstance({1,2,3,4,5}, set)
print('my_set is a set:', my_set)
my_tuple = isinstance((1,2,3,4,5), tuple)
print('my_tuple is a set:', my_tuple)
my_list = isinstance([1,2,3,4,5], list)
print('my_list is a list:', my_list)
my_dict = isinstance({'A':'a', 'B':'b', 'C':'c', 'D':'d'}, dict)
print('my_dict is a dict:', my_dict)

class MyClass:
    message = 'Hello World'

class1 = MyClass()
print('_class is a instance of MyClass() : ', isinstance(class1, MyClass))




# File Handling
# Create, Open, Append, Read, Write

dire = '/home/ukasz/Documents/IT/Python/'

f = open(dire+'delme.txt', 'w+')
for i in range(2):
    f.write('This is line %d\r\n' % (i+1))
f.close()

f = open(dire+'delme.txt', 'w+')
for i in range(3):
    f.write('This is line %d\n' % (i+1))
f.close()

f = open(dire+'delme.txt', 'w+')
for i in range(4):
    f.write('This is line %d\r' % (i+1))
f.close()

f = open(dire+'delme.txt', 'a+')
for i in range(2):
    f.write('Appended line %d\r\n' % (i+1))
f.close()


f = open(dire+'delme.txt', 'r')
if f.mode == 'r':
   contents = f.read()
   print(contents)
f.close()

f = open(dire+'delme.txt', 'r')
print(f.read())
f.close()

f = open(dire+'delme.txt', 'r')
opened_file = f.read()
f.close()
print(opened_file)

f = open(dire+'delme.txt', 'r')
for i in f.readlines():
   print(i.strip())
f.close()

f = open(dire+'delme.txt', 'r')
print(f.readlines())
f.close()

Following are the various File Modes in Python:

# Mode	Description
# 'r'	This is the default mode. It Opens file for reading.
# 'w'	This Mode Opens file for writing.
# If file does not exist, it creates a new file.
# If file exists it truncates the file.
# 'x'	Creates a new file. If file already exists, the operation fails.
# 'a'	Open file in append mode.
# If file does not exist, it creates a new file.
# 't'	This is the default mode. It opens in text mode.
# 'b'	This opens in binary mode.
# '+'	This will open a file for reading and writing (updating)






# Check If File or Directory Exists

# import os.path
from os import path

dire = '/home/ukasz/Documents/IT/Python/'
print('File exists:' + str(path.exists(dire+'delme.txt')))
print('directory exists:' + str(path.exists(dire+'PycharmProjects')))

print('Is it File?' + str(path.isfile(dire+'delme.txt')))
print('Is it File?' + str(path.isfile(dire+'myDirectory')))

print ('Is it Directory?' + str(path.isdir(dire+'delme')))
print ('Is it Directory?' + str(path.isdir(dire+'PycharmProjects')))


import pathlib
file = pathlib.Path(dire+'delme.txt')
file = pathlib.Path(dire+'PycharmProjects')
if file.exists():
    print('File/directory exists')
else:
    print("File/directory doesn't exist")






# COPY File using shutil.copy(), shutil.copystat()

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









# Zip

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











# Exception Handling: Try, Catch, Finally
# tu trzeba wrócić
'''
coś tu lipa jest
try:
   {
   catch (ArrayIndexOutOfBoundsException e) {
   System.err.printin("Caught first " + e.getMessage()); } catch (IOException e) {
   System.err.printin("Caught second " + e.getMessage());}
   }

try:
   raise KeyboardInterrupt
finally:
   print('welcome, world!')
Output
Welcome, world!
KeyboardInterrupt
'''




# readline()

dire = "/home/ukasz/Documents/Programowanie/Python/"
f = open(dire+"demo.txt","w+")
for i in range(5):
   f.write("Testing - %d line \r\n" % (i+1))
f.close()


f = open(dire+'demo.txt', 'r')
if f.mode == 'r':
   contents = f.read()
   print(contents)
f.close()


myfile = open(dire+"demo.txt", "r")
myline = myfile.readline()
print(myline)
myfile.close()


myfile = open(dire+"demo.txt", "r")
myline = myfile.readline(10)
print(myline)
myfile.close()


myfile = open(dire+"demo.txt", "r")
myline = myfile.readline()
while myline:
   print(myline)
   myline = myfile.readline()
myfile.close()


myfile = open(dire+"demo.txt", "r")
mylist = myfile.readlines()
print(mylist)
myfile.close()


myfile = open(dire+"demo.txt", "r")
for line in myfile:
   print(line)
myfile.close()

# tak nie zadziała
myfile = open(dire+"demo.txt", "r")
print(myfile)
myfile.close()

myfile = open(dire+"demo.txt", "r")
while myfile:
   line  = myfile.readline()
   print(line)
   if line == "":
       break
myfile.close()




# Data Science

# from somewhere else NumPy
adminq

import numpy as np
myPythonList = [1, 9, 8, 3]
numpy_array_from_list = np.array(myPythonList)
a = np.array([1, 9, 8, 3])
a.shape
a.dtype

b  = np.array([1.1, 2.0, 3.2])
print(b.dtype)

c = np.array([[1, 2, 3], [4, 5, 6]])
print(c.shape)
print(c)

d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(d.shape)

np.zeros((2, 3))
np.zeros((2, 3), dtype=np.int16)
np.ones((2, 3))

c.reshape(3, 2)
d.reshape(3, 2, 2)
d.flatten()

f = np.array([1, 2, 3])
g = np.array([4, 5, 6])
np.hstack((f, g))
np.vstack((f, g))


import numpy as np
normal_array = np.random.normal(5, .5, 1000)
import matplotlib.pyplot as plt
plt.hist(normal_array)
plt.show()

np.ones((4, 4))
h = np.matrix(np.ones((4, 4)))
np.asarray(np.matrix(np.ones((4, 4))))
np.array(h)[2] = 2
np.asarray(h)[2] = 2

np.arange(1, 11, 2)
np.linspace(1.0, 5.0, 10, endpoint= False)
np.logspace(3.0, 4.0, 4)

j = np.array([1,2,3], dtype=np.complex128)
j.itemsize

k = np.array([1,2,3], dtype=np.int16)
k.itemsize

c[0]
c[:, 0]
c[1, :2]

normal_array

np.min(normal_array)
np.max(normal_array)
np.mean(normal_array)
np.median(normal_array)
np.std(normal_array)

l = np.array([1, 2])
m = np.array([3, 4])
np.dot(l, m)

n = [[1,2],[3,4]] 
o = [[5,6],[7,8]] 
### 1*5+2*7 = 19
np.matmul(n, o)

np.linalg.det(n)








# SciPy

import numpy as np
from scipy import io as sio
array_ones = np.ones((4, 4))
# to do mathlab
sio.savemat('example.mat', {'ar': array_ones})
data = sio.loadmat('example.mat', struct_as_record=True)
data['ar']


from scipy.special import cbrt
#Find cubic root of 27 & 64 using cbrt() function
cb = cbrt([27, 64])
#print value of cb
print(cb)


from scipy.special import exp10
#define exp10 function and pass value in its
exp = exp10([1, 10, 20])
print(exp)
for i in exp:
   print("The value is  : {:n}".format(i))


from scipy.special import comb
# find combinations of 5, 2 values using comb(N, k)
comb(5, 2, exact=True, repetition=False)
comb(5, 2, exact=True, repetition=True)

from scipy.special import perm
# import scipy.special as ss
# find permutation of 5, 2 using perm (N, k) function
perm(5, 2, exact = True)


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


# matplotlib inline
from matplotlib import pyplot as plt
import numpy as np
#Frequency in terms of Hertz
#Sample rate
t = np.linspace(0, 2, 1000, endpoint=True)
a = np.sin(10 * np.pi * t)
# figure, axis = plt.subplots()
# axis.plot(t, a)
# axis.set_xlabel('$Time (s)$')
# axis.set_ylabel('$Signal amplitude$')
plt.xlabel('Time (s)')
plt.ylabel('Signal amplitude')
plt.plot(t, a)
plt.show()



from scipy import fftpack
A = fftpack.fft(a)
frequency = fftpack.fftfreq(len(a)) * fre_samp
figure, axis = plt.subplots()

axis.stem(frequency, np.abs(A))
axis.set_xlabel('Frequency in Hz')
axis.set_ylabel('Frequency Spectrum Magnitude')
axis.set_xlim(-fre_samp / 2, fre_samp/ 2)
axis.set_ylim(-5, 110)
plt.show()


# matplotlib inline
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np

def function(a):
   return a*2 + 20 * np.sin(a)
plt.plot(a, function(a))
plt.show()
# use BFGS algorithm for optimization
optimize.fmin_bfgs(function, 0)


x = np.arange(0, 2*np.pi, 0.1)   # start,stop,step
y = np.sin(x)
plt.plot(x, y, label='Sin(x)')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Sin(x)')
plt.legend(loc='upper left')
plt.show()


import numpy as np
from scipy.optimize import minimize
#define function f(x)
def f(x):
   return .4*(1 - x[0])**2
optimize.minimize(f, [2, -1], method="Nelder-Mead")


from scipy import misc
from matplotlib import pyplot as plt
import numpy as np
#get face image of panda from misc package
panda = misc.face()
#plot or show image of face
plt.imshow(panda)
plt.show()


#Flip Down using scipy misc.face image  
flip_down = np.flipud(misc.face())
plt.imshow(flip_down)
plt.show()

from scipy import ndimage, misc
from matplotlib import pyplot as plt
panda = misc.face()
#rotatation function of scipy for image – image rotated 135 degree
panda_rotate = ndimage.rotate(panda, 135)
plt.imshow(panda_rotate)
plt.show()


from scipy import integrate
# take f(x) function as f
f = lambda x : x**2
#single integration with a = 0 & b = 1
integrate.quad(f, 0 , 1)


from scipy import integrate
import numpy as np
#import square root function from math lib
from math import sqrt
# set  fuction f(x)
f = lambda x, y : 64 *x*y
# lower limit of second integral
p = lambda y : 0
# upper limit of first integral
q = lambda y : sqrt(1 - 2*y**2)
# perform double integration
integration = integrate.dblquad(f , 0 , 2/4, p, q)
print(integration)
integrate.dblquad(f , 0 , 2/4, 0, q)

from scipy import integrate
f = lambda y, x: x*y**2
integrate.dblquad(f, 0, 2, lambda x: 0, lambda x: 1)
integrate.dblquad(f, 0, 2, 0, 1)









# CSV
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


import pandas
dire = '/home/ukasz/Documents/IT/Python/'
result = pandas.read_csv(os.getcwd()+'/'+'data.csv')
print(result)


C = {
    'Programming language': ['Python', 'Java', 'C++'],
    'Designed by': ['Guido van Rossum', 'James Gosling', 'Bjarne Stroustrup'],
    'Appeared': ['1991', '1995', '1985'],
    'Extension': ['.py', '.java', '.cpp'],
}
df = pd.DataFrame(C, columns=['Programming language',
                  'Designed by', 'Appeared', 'Extension'])
# here you have to write path, where result file will be stored
df.to_csv(os.getcwd()+'/'+'pandaresult.csv',
                       index=None, header=True)






# JSON

import json

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

# save .json
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


dic = { 'a': 4, 'b': 5 }
# To format the code use of indent and 4 shows number of space and use of separator is not 
# necessary but standard way to write code of particular function.
formatted_obj = json.dumps(dic, indent=4, separators=(',', ':'))
print(formatted_obj)
formatted_obj = json.dumps(dic, indent=4)
print(formatted_obj)


# create function to check instance is complex or not
def complex_encode(object):
    # check using isinstance method
    if isinstance(object, complex):
        return [object.real, object.imag]
    # raised error using exception handling if object is not complex
    raise TypeError(repr(object) + " is not JSON serialized")

# perform json encoding by passing parameter
complex_obj = json.dumps(4 + 5j, default=complex_encode)
print(complex_obj)


import json
# function check JSON string contains complex object
def is_complex(objct):
   if '__complex__' in objct:
      return complex(objct['real'], objct['img'])
   return objct

# use of json loads method with object_hook for check object complex or not
complex_object = json.loads('{"__complex__": true, "real": 4, "img": 5}', object_hook=is_complex)
# here we not passed complex object so it's convert into dictionary
simple_object = json.loads('{"real": 6, "img": 7}', object_hook=is_complex)
print("Complex_object......", complex_object)
print("Without_complex_object......", simple_object)


# import JSONEncoder class from json
from json.encoder import JSONEncoder
colour_dict = { "colour": ["red", "yellow", "green" ]}
# directly called encode method of JSON
JSONEncoder().encode(colour_dict)
json.dumps(colour_dict)


import json
# import JSONDecoder class from json
from json.decoder import JSONDecoder
colour_string = '{ "colour": ["red", "yellow"]}'
# directly called decode method of JSON
JSONDecoder().decode(colour_string)
json.loads(colour_string)


import json
import requests
# get JSON string data from CityBike NYC using web requests library
json_response = requests.get("https://feeds.citibikenyc.com/stations/stations.json")
# check type of json_response object
print(type(json_response.text))
# load data in loads() function of json library
bike_dict = json.loads(json_response.text)
#check type of news_dict
print(type(bike_dict))
# now get stationBeanList key data from dict
print(bike_dict['stationBeanList'][0])
print((bike_dict.get('stationBeanList')[0])['id'])


import json
dire = "/home/ukasz/Documents/Programowanie/Python/"
#File I/O Open function for read data from JSON File
data = {} #Define Empty Dictionary Object
try:
   with open(dire+'json_file.json') as file_object:
      data = json.load(file_object)
      print(data)
except ValueError:
   print("Bad JSON file format, Change JSON File")


import json
# pass float Infinite value
infinite_json = json.dumps(float('inf'))
# check infinite json type
print(infinite_json)
print(type(infinite_json))
json_nan = json.dumps(float('nan'))
print(json_nan)
# pass json_string as Infinity
infinite = json.loads('Infinity')
print(infinite)
# check type of Infinity
print(type(infinite))


import json
repeat_pair = '{"a":  1, "a":  2, "a":  3}'
json.loads(repeat_pair)

echo '{"name" : "Kings Authur" }' | python -m json.tool






# MySQL

# Installing MySQL
sudo apt update
sudo apt upgrade
sudo apt install mysql-server
# sudo mysql_secure_installation
# pip3 install mysql-connector
pip install mysql-connector-python

sudo mysql
SHOW VARIABLES LIKE 'validate_password%';
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '<at_least_8_password>';
mysql -u root -p

import mysql.connector
db_connection = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="q!@#q!@#"
)
print(db_connection)


import mysql.connector
db_connection = mysql.connector.connect(
   host="localhost",
   port=3306,
   user="root",
   passwd="<password>",
   db="myflixdb"
)
print(db_connection)


# creating database_cursor to perform SQL operation
db_cursor = db_connection.cursor()
# executing cursor with execute method and pass SQL query
db_cursor.execute("DROP DATABASE IF EXISTS my_first_db2")
db_cursor.execute("CREATE DATABASE guru_db")
# get list of all databases
db_cursor.execute("SHOW DATABASES")
#print all databases
for db in db_cursor:
   print(db)

db_cursor.execute("USE guru_db")

import mysql.connector
db_connection = mysql.connector.connect(
   host="localhost",
   port=3306,
   user="root",
   passwd="<password>",
   database="my_first_db"
  )

db_cursor = db_connection.cursor()
#Here creating database table as student'
db_cursor.execute("CREATE TABLE student2 (id INT, name VARCHAR(255))")
#Get database table'
db_cursor.execute("SHOW TABLES")
for table in db_cursor:
	print(table)


#Here creating database table as employee with primary key
db_cursor.execute("CREATE TABLE employee(id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), salary INT(6))")
#Get database table
db_cursor.execute("SHOW TABLES")
for table in db_cursor:
   print(table)


#Here we modify existing column id
db_cursor.execute("ALTER TABLE student MODIFY id INT PRIMARY KEY")

student_sql_query = "INSERT INTO student(id,name) VALUES(01, 'John')"
employee_sql_query = " INSERT INTO employee (id, name, salary) VALUES (01, 'John', 10000)"
#Execute cursor and pass query as well as student data
db_cursor.execute(student_sql_query)
#Execute cursor and pass query of employee and data of employee
db_cursor.execute(employee_sql_query)
db_connection.commit()
print(db_cursor.rowcount, "Record Inserted")




# Facebook

sudo apt install default-jre
sudo apt install default-jdk
sudo snap install --classic eclipse
pip install selenium

# dużo zachodu
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
# Step 1) Open Firefox 
browser = webdriver.Firefox()
# Step 2) Navigate to Facebook
browser.get("http://www.facebook.com")
# Step 3) Search & Enter the Email or Phone field & Enter Password
username = browser.find_element_by_id("email")
password = browser.find_element_by_id("pass")
submit   = browser.find_element_by_id("loginbutton")
username.send_keys("you@email.com")
password.send_keys("yourpassword")
# Step 4) Click Login
submit.click()




# Matrix: Transpose, Multiplication, NumPy Arrays

M1 = [[8, 14, -6], [12, 7, 4], [-11, 3, 21], [3, 4, 5]]
matrix_length = len(M1)
#To read the last element from each row.
for i in range(matrix_length):
   print(M1[i][-1])

# tak nie działa, w np.array działa
print(M1[1, 1])

#To print the rows in the Matrix
for i in range(matrix_length):
   print(M1[i])


M1 = [[8, 14, -6],
      [12, 7, 4],
      [-11, 3, 21]]

M2 = [[3, 16, -6],
      [9, 7, -4],
      [-1, 3, 13]]

M3 = [[0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]]

# To Add M1 and M2 matrices
for i in range(len(M1)):
   for j in range(len(M1[0])):
      M3[i][j] = M1[i][j] + M2[i][j]

M3


import numpy as np
M1 = np.array([[5, -10, 15], [3, -6, 9], [-4, 8, 12]])
print(M1)


M1 = np.array([[3, 6, 9], [5, -10, 15], [-7, 14, 21]])
M2 = np.array([[9, -18, 27], [11, 22, 33], [13, -26, 39]])
M3 = M1 + M2  
print(M3)


M1 = np.array([[3, 6], [5, -10]])
M2 = np.array([[9, -18], [11, 22]])
M3 = M1.dot(M2)  
print(M3)


M1 = np.array([[3, 6, 9], [5, -10, 15], [4,8,12]])
M2 = M1.transpose()
print(M2)


arr = np.array([2, 4, 6, 8, 10, 12, 14, 16])
print(arr[3:6]) # will print the elements from 3 to 5
print(arr[:5]) # will print the elements from 0 to 4
print(arr[2:]) # will print the elements from 2 to length of the array.
print(arr[-5:-1]) # will print from the end i.e. -5 to -2
print(arr[:-1]) # will print from end i.e. 0 to -2


M1 = np.array([[2, 4, 6, 8, 10],
               [3, 6, 9, -12, -15],
               [4, 8, 12, 16, -20],
               [5, -10, 15, -20, 25]])


print(M1[1:3, 1:4]) # For 1:3, it will give first and second row.
#The columns will be taken from first to third.


print(M1[:2,])
print(M1[:3,:2])

for i in range(len(M1)):
   print(M1[i, -1])
print(M1[:, -1])

print(M1[-1, -1])




# List: Comprehension, Append, Sort, Length, Reverse

list1 = ['physics', 'chemistry', 'mathematics']
list1[0] = 'biology'
print(list1)

list1 = [3, 5, 7, 8, 9, 20]
list1.remove(3)
list1.pop(0)
del list1[0]
print(list1)

list_1 = [3, 5, 7, 8, 9, 20]
list_1.append(3.33)
print(list_1)

len(list1)
max(list1)
reverse(list1)
list1.sort(reverse=True)

animals = ('cat', 'dog', 'fish', 'cow')
print(list(animals))

animals = ['cat', 'dog', 'fish', 'cow', 'goat']
fish_index = animals.index('fish')
print(fish_index)

sum_of_values = sum(list_1 )
print(sum_of_values)

list1.sort()
print(list1)

# to sort the list by length of the elements
str1 = ['cat', 'mammal', 'goat', 'is']
str1.sort()
#sort_by_length = str1.sort(key = len)
str1.sort(key=len)

list2 = [10, 20, 30, 40, 50, 60, 70]
for elem in list2:
   elem = elem + 5
   print(elem)

list2 = [10, 20, 30, 40, 50, 60, 70]
for elem in list2[:3]:
   print(elem)
   list2.remove(elem)
print(list2)

list2 = [10, 20, 30, 40, 50, 60, 70]
for elem in range(len(list2[:3])):
   print(elem, end='_')
   list2.pop(0)
print(list2)

list2 = [10, 20, 30, 40, 50, 60, 70]
for elem in enumerate(list2[:3]):
   print(elem)
   list2.pop(0)
print(list2)

list2 = [10, 20, 30, 40, 50, 60, 70]
new_list = []	
for elem in list2[2:]:
   new_list.append(elem)
   print(elem)
new_list


list_of_squres = []
for i in range(1, 10):
   list_of_squres.append(i ** 2)
print(list_of_squres) 

list_of_squres_2 = [i ** 2 for i in range(1, 10)]   
print(list_of_squres_2)

from statistics import mean   
list2 = [10, 20, 30, 40, 50, 60, 70]
print(mean(list2))

from numpy import mean
list2 = [10, 20, 30, 40, 50, 60, 70]
print(mean(list2))

list1 = [2, 3, 4, 3, 10, 3, 5, 6, 3]
elm_count = list1.count(3)
print('The count of element: 3 is ', elm_count)

list1 = ['red', 'green', 'blue', 'orange', 'green', 'gray', 'green']
color_count = list1.count('green')
print('The count of color: green is ', color_count)


my_list = [1, 1, 2, 3, 2, 2, 4, 5, 6, 2, 1]
temp_list = []
for i in my_list:
   if i not in temp_list:
      temp_list.append(i)
print(temp_list)

list(set(my_list))

my_list = [1, 1, 2, 3, 2, 2, 4, 5, 6, 2, 1]
temp_list = []
[temp_list.append(i) for i in my_list if i not in temp_list]
print(temp_list)

my_list = [1, 1, 2, 3, 2, 2, 4, 5, 6, 2, 1]
my_final_list = set(my_list)
print(my_final_list)

import numpy as np
my_list = [1, 2, 2, 3, 1, 4, 5, 1, 2, 6]
myFinalList = np.unique(my_list).tolist()
print(myFinalList)

import pandas as pd
my_list = [1, 2, 2, 3, 1, 4, 5, 1, 2, 6]
myFinalList = pd.unique(my_list).tolist()
print(myFinalList)

my_list = [1, 2, 2, 3, 1, 4, 5, 1, 2, 6]
my_finallist = []
for j, i in enumerate(my_list):
   if i not in my_list[:j]:
      my_finallist.append(i)
print(my_finallist)

my_list = [1, 2, 2, 3, 1, 4, 5, 1, 2, 6]
my_finallist = [i for j, i in enumerate(my_list) if i not in my_list[:j]] 
print(my_finallist)

my_list = ['A', 'B', 'C', 'D', 'E', 'F']
print('The index of element C is ', my_list.index('C'))
print('The index of element F is ', my_list.index('F'))


my_list = ['Guru', 'Siya', 'Tiya', 'Guru', 'Daksh', 'Riya', 'Guru']
all_indexes = []
for i in range(len(my_list)):
    if my_list[i] == 'Guru':
        all_indexes.append(i)
print('Originallist ', my_list)
print('Indexes for element Guru : ', all_indexes)


my_list = ['Guru', 'Siya', 'Tiya', 'Guru', 'Daksh', 'Riya', 'Guru']
result = []
elementindex = -1
while True:
    try:
        elementindex = my_list.index('Guru', elementindex + 1)
        result.append(elementindex)
    except ValueError:
        break
print('OriginalList is ', my_list)
print('The index for element Guru is ', result)


my_list = ['Guru', 'Siya', 'Tiya', 'Guru', 'Daksh', 'Riya', 'Guru'] 
print('Originallist ', my_list)
all_indexes = [i for (i, a) in enumerate(my_list) if a == 'Guru']
print('Indexes for element Guru : ', all_indexes)

# list of letters
letters = ['a', 'b', 'd', 'e', 'i', 'j', 'o']

# function that filters vowels
def filter_vowels(letter):
    vowels = ['a', 'e', 'i', 'o', 'u']
    if (letter in vowels):
        return True
    else:
        return False
filtered_vowels = filter(filter_vowels, letters)

print('The filtered vowels are:')
for vowel in filtered_vowels:
    print(vowel)
list(filtered_vowels)

my_list = ['Guru', 'Siya', 'Tiya', 'Guru', 'Daksh', 'Riya', 'Guru'] 
print('Originallist', my_list)
all_indexes = list(filter(lambda i: my_list[i] == 'Guru', range(len(my_list)))) 
print('Indexes for element Guru : ', all_indexes)


import numpy as np
my_list = ['Guru', 'Siya', 'Tiya', 'Guru', 'Daksh', 'Riya', 'Guru'] 
np_array = np.array(my_list)
item_index = np.where(np_array == 'Guru')[0]
print('Originallist', my_list)
print('Indexes for element Guru :', item_index)


from more_itertools import locate
my_list = ['Guru', 'Siya', 'Tiya', 'Guru', 'Daksh', 'Riya', 'Guru'] 
print('Originallist : ', my_list)
print('Indexes for element Guru :', list(locate(my_list, lambda x: x == 'Guru'))) 


"""
Built-in Functions
FUNCTION	DESCRIPTION
Round()	Rounds off the number passed as an argument to a specified number of digits and returns the floating point value
Min()	return minimum element of a given list
Max()	return maximum element of a given list
len()	Returns the length of the list
Enumerate()	This built-in function generates both the values and indexes of items in an iterable, so we don't need to count manually
Filter()	tests if each element of a list true or not
Lambda	An expression that can appear in places where a def (for creating functions) is not syntactic, inside a list literal or a function's call arguments
Map()	returns a list of the results after applying the given function to each item of a given iterable
Accumulate()	apply a particular function passed in its argument to all of the list elements returns a list containing the intermediate results
Sum()	Returns the sum of all the numbers in the list
Cmp()	This is used for comparing two lists and returns 1 if the first list is greater than the second list.
Insert	Insert element to list at particular position
List Methods
FUNCTION	DESCRIPTION
Append()	Adds a new item to the end of the list
Clear()	Removes all items from the list
Copy()	Returns a copy of the original list
Extend()	Add many items to the end of the list
Count()	Returns the number of occurrences of a particular item in a list
Index()	Returns the index of a specific element of a list
Pop()	Deletes item from the list at particular index (delete by position)
Remove()	Deletes specified item from the list (delete by value)
Reverse()	In-place reversal method which reverses the order of the elements of the list
"""





# RegEx

"""
dentifiers	Modifiers	White space characters	Escape required
\d= any number (a digit)	\d represents a digit.Ex: \d{1,5} it will declare digit between 1,5 like 424,444,545 etc.	
\n = new line	. + * ? [] $ ^ () {} | \
\D= anything but a number (a non-digit)	+ = matches 1 or more	
\s= space	
\s = spac (tab,space,newline etc.)	
? = matches 0 or 1	
\t =tab	
\S= anything but a space	* = 0 or more	
\e = escape	
\w = letters ( Match alphanumeric character, including “_”)	$ match end of a string	
\r = carriage return	
\W =anything but letters ( Matches a non-alphanumeric character excluding “_”)	^ match start of a string	
\f= form feed	
. = anything but letters (periods)	| matches either or x/y	—————–	
\b = any character except for new line	[] = range or “variance”	—————-	
\.	{x} = this amount of preceding code	—————–
"""

import re
xx = 'education is fun'
re.findall(r'^\w+', xx)
print(r1)

import re
xx = 'education is fun'
r1 = re.findall(r'^\w+', xx)
print((re.split(r'\s','we are splitting the words')))
print((re.split(r's','split the swords')))


# re.match() function of re in Python will search the regular expression pattern and return the first occurrence. 
# The Python RegEx Match method checks for a match only at the beginning of the string. 
# So, if a match is found in the first line, it returns the match object. 
# But if a match is found in some other line, the Python RegEx Match function returns null.

import re
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


# re.search() function will search the regular expression pattern and return the first occurrence. 
# Unlike Python re.match(), it will check all lines of the input string. 
# The Python re.search() function returns a match object when the pattern is found and “null” if the pattern is not found

# findall() module is used to search for “all” occurrences that match a given pattern. 
# In contrast, search() module will only return the first occurrence that matches the specified pattern. 
# findall() will iterate over all the lines of the file and will return all non-overlapping matches of pattern in a single step.

abc = ['guru99@google.com', 'careerguru99@hotmail.com', 'users@yahoomail.com']
for i in abc:
    mat = re.search(r'[\w\.-]+@[\w\.-]+\.[\w]+', i)
    if mat:
        print(i)
        print(mat.start())
        print(mat.end())
        print(mat.group(0))
        print(mat.groups())


abc = 'guru99@google.com, careerguru99@hotmail.com, users@yahoomail.com'
emails = re.findall(r'[\w\.-]+@[\w\.-]+\.[\w]+', abc)
print(emails)
for email in emails:
   print(email)


import re
xx = """guru99 
careerguru99	u
selenium"""
k1 = re.findall(r"^\w", xx)
k2 = re.findall(r"^\w", xx, re.MULTILINE)
print(k1)
print(k2)


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
for result in regex.findall("Hello World, Bonjour World"):
   print(result)

# This will substitute "World" with "Earth" and print:
#   Hello Earth
print(regex.sub(r"\1 Earth", "Hello World"))









# Calendar

import calendar
# Create a plain text calendar
c = calendar.TextCalendar(calendar.THURSDAY)
str = c.formatmonth(2025, 1, 0, 0)
print(str)

# Create an HTML formatted calendar
hc = calendar.HTMLCalendar(calendar.THURSDAY)
str = hc.formatmonth(2025, 1)
print(str)

# loop over the days of a month
# zeroes indicate that the day of the week is in a next month or overlapping month
for i in c.itermonthdays(2025, 4):
   print(i)
   
# The calendar can give info based on local such a names of days and months (full and abbreviated forms)
for name in calendar.month_name:
   print(name)

for day in calendar.day_name:
   print(day)

# calculate days based on a rule: For instance an audit day on the second Monday of every month
# Figure out what days that would be for each month, we can use the script as shown here
for month in range(1, 13):
# It retrieves a list of weeks that represent the month
   mycal = calendar.monthcalendar(2025, month)
# The first MONDAY has to be within the first two weeks
   week1 = mycal[0]
   week2 = mycal[1]
   if week1[calendar.MONDAY] != 0:
      auditday = week1[calendar.MONDAY]
   else:
   # if the first MONDAY isn't in the first week, it must be in the second week
      auditday = week2[calendar.MONDAY]
   print("%10s %2d" % (calendar.month_name[month], auditday))
        













#PyTest
# By default pytest only identifies the file names starting with test_ or ending with _test 
# as the test files. We can explicitly mention other filenames though (explained later). 
# Pytest requires the test method names to start with “test.” 
# All other method names will be ignored even if we explicitly ask to run those methods.

# In Python Pytest, if an assertion fails in a test method, then that 
# method execution is stopped there. The remaining code in that test 
# method is not executed, and Pytest assertions will continue with the next test method.

# By default pytest only identifies the file names starting with test_ or ending
# with _test as the test files. We can explicitly mention other filenames
# though (explained later). Pytest requires the test method names to
# start with “test.” All other method names will be ignored even if we
# explicitly ask to run those methods.

# is this a typo? You can register custom marks to avoid this warning - for details,
# create file in that folder
cat > pytest.ini

[pytest]
filterwarnings =
    ignore::UserWarning


# run test_sample1.py test
> py.test test_sample1.py -v


# @pytest.mark.set1
def test_file1_method1():
	x = 5
	y = 6
	assert x + 1 == y, 'some text 2'
	assert x == y, 'test failed because x=' + str(x) + ' y=' + str(y)
	assert x == y, 'some text 1'
	assert 1 == 2, 'one is two'

# @pytest.mark.set2
def test_file1_method2():
	x = 5
	y = 6
	assert x + 1 == y, 'some sext 3'

@pytest.mark.set1
def test_file2_method1():
	x = 5
	y = 6
	assert x + 1 == y, 'test failed'
	assert x == y, 'test failed because x=' + str(x) + ' y=' + str(y)
	assert 1 == 2, 'jeden jest dwa'

@pytest.mark.set1
def test_file2_method2():
	x = 5
	y = 6
	assert x + 1 == y, 'test failed'


# run all tests in folder
> py.test
# run a specific test
> py.test test_sample1.py
> py.test -k method1 -v
-k <expression> is used to represent the substring to match
-v increases the verbosity
> py.test -k method -v # will run all the four methods, starts with 'method'
# Run tests by markers
pytest -m set1 -v

# Run Tests in Parallel with Pytest
pip install pytest-xdist
conda install -c anaconda jedi
conda update conda
pip list
conda install pytest=6.2.2
conda update flask
conda search -f <package_name> 

py.test -n 4

import pytest
@pytest.fixture
def supply_AA_BB_CC():
	aa = 25
	bb = 35
	cc = 45
	return [aa, bb, cc]


import pytest

def test_comparewithAA(supply_AA_BB_CC):
	zz = 35
	assert supply_AA_BB_CC[0] == zz, 'aa and zz comparison failed'


def test_comparewithBB(supply_AA_BB_CC):
	zz = 35
	assert supply_AA_BB_CC[1] == zz, 'bb and zz comparison failed'


def test_comparewithCC(supply_AA_BB_CC):
	zz = 35
	assert supply_AA_BB_CC[2] == zz, 'cc and zz comparison failed'

> pytest test_basic_fixture.py -v

# pytest will look for the fixture in the test file first and if not found it will look 
# in the conftest.py
# conftest.py A fixture method can be accessed across multiple test files by defining 
# it in conftest.py file.

> pytest -k comparewith -v



# The purpose of parameterizing a test is to run a test against multiple sets of arguments. 
# We can do this by @pytest.mark.parametrize.

# We will see this with the below PyTest example. Here we will pass 3 arguments 
# to a test method. This test method will add the first 2 arguments and 
# compare it with the 3rd argument.

import pytest


@pytest.mark.parametrize('input1, input2, output', [(5, 5, 10), (3, 5, 12)])
def test_add(input1, input2, output):
	assert input1 + input2 == output, 'failed'

> pytest -k test_add -v

# Pytest Xfail / Skip Tests
'''
The xfailed test will be executed, but it will not be counted as part failed or passed tests. 
There will be no traceback displayed if that test fails. We can xfail tests using
@pytest.mark.xfail.

Skipping a test means that the test will not be executed. We can skip tests using
@pytest.mark.skip.
'''

import pytest


@pytest.mark.skip
def test_add_1():
	assert 100 + 200 == 400, 'failed'


@pytest.mark.skip
def test_add_2():
	assert 100 + 200 == 300, 'failed'


@pytest.mark.xfail
def test_add_3():
	assert 15 + 13 == 28, 'failed'


@pytest.mark.xfail
def test_add_4():
	assert 15 + 13 == 100, 'failed'


def test_add_5():
	assert 3 + 2 == 5, 'failed'


def test_add_6():
	assert 3 + 2 == 6, 'failed'

'''
test_add_1 and test_add_2 are skipped and will not be executed.
test_add_3 and test_add_4 are xfailed. These tests will be executed and will be part of xfailed(on test failure) or xpassed(on test pass) tests. There won’t be any traceback for failures.
test_add_5 and test_add_6 will be executed and test_add_6 will report failure with traceback while the test_add_5 passes
'''

# Results XML
py.test test_sample1.py -v --junitxml='result.xml'


# Fake data
https://reqres.in/

import pytest
import requests
import json
@pytest.mark.parametrize('userid, firstname',[(1,'George'),(2,'Janet')])
def test_list_valid_user(supply_url,userid,firstname):
	url = supply_url + '/users/' + str(userid)
	resp = requests.get(url)
	j = json.loads(resp.text)
	assert resp.status_code == 200, resp.text
	assert j['data']['id'] == userid, resp.text
	assert j['data']['first_name'] == firstname, resp.text

def test_list_invaliduser(supply_url):
	url = supply_url + '/users/50'
	resp = requests.get(url)
	assert resp.status_code == 404, resp.text


pytest -k test_list -v
pytest -k test_login -v
--disable-pytest-warnings


import pytest
import requests
import json
def tst_list_valid_user(supply_url, userid, firstname):
    url = supply_url + '/users/' + str(userid)
    resp = requests.get(url)
    j = json.loads(resp.text)
    return([j['data']['last_name'], j['data']['first_name']])

tst_list_valid_user('https://reqres.in/api', 1, 'George')







# Urllib.Request and urlopen()

# read the data from the URL and print it
import urllib.request
# open a connection to a URL using urllib
# webUrl  = urllib.request.urlopen('https://www.youtube.com/user/guru99com')
webUrl  = urllib.request.urlopen('https://www.google.com/')
#get the result code and print it
print("result code: " + str(webUrl.getcode()))
# read the data from the URL and print it
data = webUrl.read()
print(data)







# Read xml file example(Minidom, ElementTree)

import xml.dom.minidom
# use the parse() function to load and parse an XML file
doc = xml.dom.minidom.parse("Myxml.xml")
# print out the document node and the name of the first child tag
print (doc.nodeName)
print (doc.firstChild.tagName)
# get a list of XML tags from the document and print each one
expertise = doc.getElementsByTagName("expertise")
print ("%d expertise:" % expertise.length)
for skill in expertise:
   print (skill.getAttribute("name"))
# create a new XML tag and add it into the document
newexpertise = doc.createElement("expertise")
newexpertise.setAttribute("name", "BigData")
doc.firstChild.appendChild(newexpertise)
print (" ")
expertise = doc.getElementsByTagName("expertise")
print ("%d expertise:" % expertise.length)
for skill in expertise:
   print (skill.getAttribute("name"))








# PyQt5

import sys
from PyQt5.QtWidgets import QApplication, QWidget
app = QApplication(sys.argv)
w = QWidget()
w.resize(300,300)
w.setWindowTitle("majn tajtle")
w.show()
sys.exit(app.exec_())



import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QLineEdit, QMessageBox, QComboBox, QMenuBar

def dialog():
   mbox = QMessageBox()
   mbox.setWindowTitle("dialog title")
   mbox.setText("setText")
   mbox.setDetailedText("setDetailedText")
   mbox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)        
   mbox.exec_()

if __name__ == "__main__":
   app = QApplication(sys.argv)
   w = QWidget()
   w.resize(300,300)
   w.setWindowTitle('majn tajtle')
   
   label = QLabel(w)
   label.setText("QLabel")
   label.move(100,130)
   label.show()

   btn = QPushButton(w)
   btn.setText('Button text'')
   btn.move(110,150)
   btn.show()
   btn.clicked.connect(dialog)
   
   line1 = QLineEdit(w)
   line1.move(100, 180)
   line1.setText("line text")
   line1.show()

   rad1 = QRadioButton(w)
   rad1.move(100, 210)
   # rad1.setChecked(True)
   rad1.show()

   drop1 = QComboBox(w)
   drop1.addItems(["item one", "item two", "item three"])

   bar1 = QMenuBar(w)
   #bar1.move(100, 230)
   bar1.show()

   w.show()
   sys.exit(app.exec_())


import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout

if __name__ == "__main__":
   app = QApplication([])
   w = QWidget()
   w.setWindowTitle("Musketeers")

   btn1 = QPushButton("Athos")
   btn2 = QPushButton("Porthos")
   btn3 = QPushButton("Aramis")

   hbox = QHBoxLayout(w)
   # hbox = QVBoxLayout(w)

   hbox.addWidget(btn1)
   hbox.addWidget(btn2)
   hbox.addWidget(btn3)
   w.show()
   sys.exit(app.exec_())


import sys
from PyQt5.QtWidgets import QPushButton, QApplication, QWidget, QGridLayout

if __name__ == "__main__":
   app = QApplication([])
   w = QWidget()
   grid = QGridLayout(w)
   for i in range(3):
      for j in range(3):
         grid.addWidget(QPushButton("Button"), i, j)
   w.show()
   sys.exit(app.exec_())


import sys
from PyQt5.QtWidgets import QPushButton, QApplication, QWidget, QGridLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette

if __name__ == "__main__":
   app = QApplication([])
   # app.setStyle("Fusion")

   qp = QPalette()
   qp.setColor(QPalette.ButtonText, Qt.green)
   qp.setColor(QPalette.Window, Qt.gray)
   qp.setColor(QPalette.Button, Qt.red)
   app.setPalette(qp)   

   w = QWidget()
   w.setWindowTitle("Buttons")
   grid = QGridLayout(w)

   grid.addWidget(QPushButton("Button one"), 0, 0)
   grid.addWidget(QPushButton("Button two"), 0, 1)
   grid.addWidget(QPushButton("Button three"), 0, 2)
   grid.addWidget(QPushButton("Button four"), 1, 1)
   grid.addWidget(QPushButton("Button five"), 2, 0, 10, 3)

   w.show()
   sys.exit(app.exec_())







# Multithreading

import time
import _thread

def thread_test(name, wait):
   i = 0
   while i <= 3:
      time.sleep(wait)
      print("Running %s\n" %name)
      i = i + 1
   print("%s has finished execution" %name)

if __name__ == "__main__":
   _thread.start_new_thread(thread_test, ("First Thread", 1))
   _thread.start_new_thread(thread_test, ("Second Thread", 2))
   _thread.start_new_thread(thread_test, ("Third Thread", 3))



import time
import threading

class threadtester (threading.Thread):
   def __init__(self, id, name, i):
      threading.Thread.__init__(self)
      self.id = id
      self.name = name
      self.i = i
      
   def run(self):
      thread_test(self.name, self.i, 5)
      print ("%s has finished execution " %self.name)

def thread_test(name, wait, i):
   while i:
      time.sleep(wait)
      print ("Running %s \n" %name)
      i = i - 1

if __name__=="__main__":
   thread1 = threadtester(1, "First Thread", 1)
   thread2 = threadtester(2, "Second Thread", 2)
   thread3 = threadtester(3, "Third Thread", 3)
   thread1.start()
   thread2.start()
   thread3.start()
   thread1.join()
   thread2.join()
   thread3.join()



import threading
lock = threading.Lock()
def first_function():
   for i in range(5):
      lock.acquire()
      print ('lock acquired')
      print ('Executing the first funcion')
      lock.release()

def second_function():
   for i in range(5):
      lock.acquire()
      print ('lock acquired')
      print ('Executing the second funcion')
      lock.release()

if __name__=="__main__":
   thread_one = threading.Thread(target=first_function)
   thread_two = threading.Thread(target=second_function)
   thread_one.start()
   thread_two.start()
   thread_one.join()
   thread_two.join()











# modules
# A module is a file with python code. The code can be in the form of variables, functions, or 
# class defined. The filename becomes the module name.

import sys
print(sys.path)

def disp_message():
   return "Moduł działa"

import os
sys.path.insert(0, os.path.abspath('/home/ukasz/Documents/Programowanie/Python/modtest/'))

import sys
import module1
print(module1.disp_message())
print(sys.path)



import os
sys.path.insert(0, os.path.abspath('/home/ukasz/Documents/Programowanie/Python/myproj/'))
import sys
import Car
car_det = Car.Car("BMW","Z5", 2020)
print(car_det.brand_name)
print(car_det.car_details())
print(car_det.get_Car_brand())
print(car_det.get_Car_model())


class Car:
	brand_name = "BMW"
	model = "Z4"
	manu_year = "2020"

	def __init__(self, brand_name, model, manu_year):
		self.brand_name = brand_name
		self.model = model
		self.manu_year = manu_year

	def car_details(self):
		print("Car brand is ", self.brand_name)
		print("Car model is ", self.model)
		print("Car manufacture year is ", self.manu_year)
					
	def get_Car_brand(self):
		print("Car brand is ", self.brand_name)

	def get_Car_model(self):
		print("Car model is ", self.model)

Car("BMW", "Z4", 2020).get_Car_model()



my_name = "nejm"
my_address = "Kantry"

def disp_message():
	return "Moduł działa"
	
def display_message1():
	return "All about Python!"

# A package is a directory with all modules defined inside it.

import os
import sys
sys.path.insert(0, os.path.abspath('/home/ukasz/Documents/Programowanie/Python/modtest/'))

from module1 import disp_message, my_name
print(disp_message())
print(my_name)



import test
print(test.display_message())

from test import * # nie robić tak
print(display_message())

import json
print(dir(json))



def mod1_func1():
print("Welcome to Module1 function1")

def mod1_func2():
print("Welcome to Module1 function2")

def mod1_func3():
print("Welcome to Module1 function3")


import mypackage.module1 as mod1
print(mod1.mod1_func1())
print(mod1.mod1_func2())
print(mod1.mod1_func2())


# Import module
from mypackage import module1
print(module1.mod1_func1())

from mypackage import module1 as inner_module1
print(inner_module1.mod1_func1())

import mypackage.module1
print(module1.mod1_func1())

import mypackage.module1 as inner_module1
print(inner_module1.mod1_func1())


# Import function
from mypackage.module1 import mod1_func1
print(mod1_func1())

from mypackage.module1 import mod1_func1 as inner_mod1_func1
print(inner_mod1_func1())

#import mypackage.module1.mod1_func1
#print(mod1_func1())

#import mypackage.module1.mod1_func1 as inner_mod1_func1
#print(inner_mod1_func1())


import sys
print(sys.path)



#   a = list(x)
#   b = dict(y)
#   complex_function(a, b)
 

def complex_function3(objects, dict):
    list = []
    i = 0
    while i<len(objects):
        x = objects[i]
        i = i+1
        if not(x == None or x in dict.keys()):
            list.append(x)
    x = set(list)
    return x


def complex_function(objects, my_dict):
    return {el for el in objects if el is not None and el not in my_dict}


A Singleton pattern in python is a design pattern that allows you to create just one instance of a class, throughout the lifetime of a program. Using a singleton pattern has many benefits. A few of them are:
https: // python-patterns.guide/gang-of-four/singleton/

What is a decorator in Python?
A decorator in Python is a function that takes another function as its argument, and returns yet another function . Decorators can be extremely useful as they allow the extension of an existing function, without any modification to the original function source code
