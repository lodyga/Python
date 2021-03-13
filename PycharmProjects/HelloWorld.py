Skip to content
Search or jump to…

Pull requests
Issues
Marketplace
Explore
 
@lodyga 
lodyga
/
pycharmGIT
1
00
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
pycharmGIT/naukaPythonLinux.py /
@lodyga
lodyga Update naukaPythonLinux.py
Latest commit 3fa698c on 30 Dec 2020
 History
 1 contributor
1526 lines (1116 sloc)  29.3 KB
 
print("USA")
print("UK")
print(2 * "\n")
print("Canada")
print("\n\n")
print("Germany", end='!')
print("France", end=' @ ')
print("Japan")
print("Poland")

a = 100
print(a)
b = 99
print('ABC' + str(b))


def somefunction():
   global a
   print(a)
   a = 'nie 100'


somefunction()
print(a)
del (a, b)

tup1 = ('Robert', 'Carlos', '1965', 'Terminator 1995', 'Actor', 'Florida')
tup2 = (1, 2, 3, 4, 5, 6, 7)
print(tup2[1:4])
x = ("Guru99", 20, "Education")  # tuple packing
(company, emp, profile) = x  # tuple unpacking
print(company)
print(emp)
print(profile)

a = (5, 6)
b = (6, 4)
if a > b:
   print("a is bigger")
else:
   print("b is bigger")

a = {'x': 100, 'y': 200}
b = list(a.items())
print(a)
print(b)

x = ("a", "b", "c", "d", "e")
print(x[2:4])
y = max(1, 3, 5, 2)
print(y)

Dict = {'Tim': 18, 'Charlie': 12, 'Tiffany': 22, 'Robert': 25}
print(Dict['Tiffany'])
Boys = {'Tim': 18, 'Charlie': 12, 'Robert': 25}
Girls = {'Tiffany': 22}
studentX = Boys.copy()
print(Boys)
print(studentX)
Dict.update({"Sarah": 9})
print(Dict)
del Dict['Charlie']
print(Dict)
Dict = {'Tim': 18, 'Charlie': 12, 'Tiffany': 22, 'Robert': 25}
for key in Dict.keys():
   if key in Boys.keys():
       print(True)
   else:
       print(False)

Students = list(Dict.keys())
Students.sort()
print(Students)
for S in Students:
   print(":".join((S, str(Dict[S]))))

for S in Students:
   print(S)

print("Length : %d" % len(Dict))
print(len(Students))
print("variable Type: %s" % type(Dict))
print("printable string:%s" % str(Dict))
print("sdf %s" % str(Dict))

my_dict1 = {"username": "XYZ", "email": "xyz@gmail.com", "location": "Mumbai"}
my_dict2 = {"firstName": "Nick", "lastName": "Price"}
my_dict1.update(my_dict2)
print(my_dict1)
my_dict = {**my_dict1, **my_dict2}
print(my_dict)
print("email" in my_dict)
print("location" in my_dict)
print("test" in my_dict)

# Dictionary

my_dict = {"Name": [], "Address": [], "Age": []}
print(my_dict['Name'])
my_dict["Name"].append("Guru")
print(my_dict['Name'])
my_dict["Address"].append("Mumbai")
my_dict["Age"].append(30)
print(my_dict)
print("username:", my_dict['Name'])

# del my_dict['Name']
# print(my_dict)
# del my_dict
# your_dict.clear()
# my_dict.pop("Name")
# print(my_dict)
my_dict['name'] = 'Nick'
print(my_dict)

my_dict1["my_dict1"] = my_dict
print(my_dict1)

# Operators

x = 4
y = 5
print(x + y)
print('x>y is', x != y)
print("x>y is", x != y)
x += 6
print("x+6=", x)
a = True
b = False
print('a and b is', a and b)

lis1 = [1, 2, 3, 4, 5]
if y in lis1:
   print('Yes')
else:
   print('No')
print(9 // 4)
print(9 % 4)

# Arrays

import array

balance = array.array('i', [300, 200, 100])
print(balance[1])
balance.insert(2, 150)
print(balance)
balance[0] = 5000
print(balance)
# balance.pop(0)
# del balance[0]
balance.remove(5000)
print(balance)
print(balance.index(100))
balance.reverse()
print(balance)
print(balance.count(200))
for x in balance:
   print(x)

import array as myarray

abc = myarray.array('d', [2.5, 4.9, 6.7])
print('Array first element is:', abc[0])
print('Array last element is:', abc[-1])
print(abc.index(2.5))

print(abc + abc)

abc = myarray.array('q', [3, 9, 6, 5, 20, 13, 19, 22, 30, 25])
print(abc[2:-1])


# If

def main():
   x, y = 8, 8
   if x < y:
       st = "x is less than y"
   elif x == y:
       st = "x is same as y"
   else:
       st = "x is greater than y"
   print(st)


if __name__ == "__main__":
   main()


def main():
   x, y = 10, 8
   st = "x is less than y" if (x < y) else "x is greater than or equal to y"
   print(st)


if __name__ == "__main__":
   main()

total = 100
# country = "US"
country = "AU"
if country == "US":
   if total <= 50:
       print("Shipping Cost is  $50")
   elif total <= 100:
       print("Shipping Cost is $25")
   elif total <= 150:
       print("Shipping Costs $5")
else:
   print("FREE")
if country == "AU":
   if total <= 50:
       print("Shipping Cost is  $100")
   else:
       print("FREE")

# For & While

x = 0
while x < 4:
   print(x)
   x += 1
print("\n")
for x in range(2, 4):
   print(x)
print("\n")
for x in '234':
   print('wartosc :', x)
print('\n')
for x in enumerate(range(2, 4)):
   print(x)
print('\n')
Months = ["Jan", "Feb", "Mar", "April", "May", "June"]
for m in Months:
   print(m)

for x in range(10, 20):
   # if x==15: break
   if (x % 5 == 0): continue
   print(x)
print('\n')
for i, m in enumerate(Months):
   print(i, m)

# Break % Continue
print('\n')
my_list = ['Siya', 'Tiya', 'Guru', 'Daksh', 'Riya', 'Guru']
for i in range(len(my_list)):
   print(i, my_list[i])
   if my_list[i] == 'Guru':
       print('Found break')
       break
       print('Hidden')
   print('Też nie see')
print('See')
print('\n')
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
print('\n')

for i in range(4):
   for j in range(4):
       if j == 2:
           break
       print('number is :', i, j)

for i in range(5):
   if i == 2:
       continue
   print(i)
print('\n')
i = 0
while i < 5:
   if i == 2:
       i += 1
       continue
   print(i)
   i += 1


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


# Class, Object

# Example file for working with classes
class myClass():
   def method1(self):
       print("Guru99")


class childClass(myClass):
   def method1(self):
       # myClass.method1(self)
       print("childClass Method1")

   def method2(self):
       print("childClass method2")


def main():
   # exercise the class methods
   c2 = childClass()
   c2.method1()
   # c2.method2()


if __name__ == "__main__":
   main()
print('\n')


# Constructors

class User:
   name = ""

   def __init__(self, name):
       self.name = name

   def sayHello(self):
       print("Welcome to Guru99, " + self.name)


User1 = User("Alex")
User1.sayHello()

# String

var1 = 'Guru99!'
var2 = 'Software Testing'
print('var[0]:', var1[0])
print('var2[1:5]:', var2[1:5])
print('u' in var1)
print('8' in var1)
print('\n')
print(r'\n')
print('/n')
print(R'/n')

name = 'guru'
number = 99
print('%s %d' % (name, number))
print((name + ' ' + str(number)))
print(name * 2)
print(number * 2)
print(name[1:3] + name)
oldstring = 'nie lubię Cię'
newstring = oldstring.replace('nie', '')
print(newstring)
print(name.upper())
print(name.capitalize())
print(':'.join(name))
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
print(str1.strip("c"))
print(str3.strip("c"))

# Count
print(str1.count('a'))
print(str1.count('a', 2, ))

# Format
print('welcome {} too'.format('you'))
print('welcome {n1} {n2} 2'.format(n1='you', n2='too'))
print('welcome {0} {1} 3'.format('you', 'too'))
print('welcome {} {} 4'.format('you', 'too'))
print('welcome {1} {0} 5'.format('you', 'too'))
print("The binary to decimal value is : {:d}".format(0b0011))
print("The binary value is : {:b}".format(500))
print("The scientific value is : {:e}".format(40))
print("The value is  : {:.3f}".format(40))
print("The value is  : {:n}".format(500.00))
print("The value is  : {:.2%}".format(1.80))
print("The value is   {:_}".format(1000000))
print("The value is: {:5}".format(40))
print("The value is: {}".format(-40))
print("The value is: {:-}".format(-40))
print("The value is: {:+}".format(40))
print("The value is: {:=}".format(-40))
print("The value {:^10} is positive value".format(40))
print("The value {:<10} is positive value".format(40))


class MyClass1:
   msg1 = 'twoja'
   msg2 = 'stara'


print('tak to {c.msg1}1 {c.msg2}2'.format(c=MyClass1))
my_dict = {'msg1': "twoja", 'msg2': "stara"}
print('{m[msg1]} {m[msg2]}'.format(m=my_dict))
print('test {:5} tset {:5}'.format(1, 2))

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
print('text', str1.find("c"))
print('text', str1.find("c", 1, 5))
mystring = "Meet Guru99 Tutorials Site.Best site for Python Tutorials!"
print("The position of Tutorials using find() : ", mystring.find("Tutorials"))
print("The position of Tutorials using rfind() : ", mystring.rfind("Tutorials"))
print("The position of Tutorials using index() : ", mystring.index("Tutorials"))

startIndex = 0
count = 0


# for i in range(len(mystring)):
#    k = my_string.find('T', startIndex)
#    if(k != -1):
#        startIndex = k+1
#        count += 1
#        k = 0


# print("The total count of substring test is: ")

# Main Function & Method Example: Understand __main__

def main1():
   print('hello')


if __name__ == "__main__":
   main1()

print('tak')
print(__name__)

# import MainFunction

print("done")


# Functions

def func1():
   print("tekst")


func1()


def sq(x=5):
   return x * x


print(sq())


def multi(x, y=0):
   print("x=", x)
   print("y=", y)
   return x * y


print(multi(y=2, x=4))


def func2(*args):
   print(args)


func2(1, 2, 3, 4, 5)

# Lambda Functions

adder = lambda x, y=2: x + y
print(adder(1))

# What a lambda returns
string = 'some kind of a useless lambda'
print(lambda string: print(string))

# What a lambda returns #2
x = "some kind of a useless lambda"
(lambda x: print(x))(x)


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

sequence = [10, 2, 8, 7, 5, 4, 3, 11, 0, 1]
filtered_result = filter(lambda x: x > 4, sequence)
print(list(filtered_result))
print(filtered_result)


def filter2(arg):
   for i in range(len(arg)):
       if arg[i] > 4:
           print(arg[i])


filter2(sequence)

filtered_result = map(lambda x: x * x, sequence)
print(list(filtered_result))

from functools import reduce

sequences = [1, 2, 3, 4, 5]
product = reduce(lambda x, y: x * y, sequences)
print(product)

# Abs

int_num = -25
float_num = -10.50
complex_num = (3 + 10j)
print("The absolute value of an integer number is:", abs(int_num))
print("The absolute value of a float number is:", abs(float_num))
print("The magnitude of the complex number is:", abs(complex_num))

import random


def truncate(num):
   return int(num * 1000) / 1000


arr = [random.uniform(0.01, 0.05) for _ in range(100)]
sum_num = 0
sum_trun = 0
for i in arr:
   sum_num += i
   sum_trun = truncate(sum_trun + i)

print("Testing by using truncating upto 3 decimal places")
print("The original sum is = ", sum_num)
print("The total using truncate = ", sum_trun)
print("The difference from original - truncate = ", sum_num - sum_trun)
print("\n\n")
print("Testing by using round() upto 3 decimal places")
sum_num1 = 0
sum_truncate1 = 0
for i in arr:
   sum_num1 = sum_num1 + i
   sum_truncate1 = round(sum_truncate1 + i, 3)

print("The original sum is =", sum_num1)
print("The total using round = ", sum_truncate1)
print("The difference from original - round =", sum_num1 - sum_truncate1)

# import numpy as np
# arr = [-0.341111, 1.455098989, 4.232323, -0.3432326, 7.626632, 5.122323]
# arr1 = np.round(arr, 2)
# print(arr1)

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

print("Using round()", final_val)
print("Using Decimal - ROUND_CEILING ", final_val1)
print("Using Decimal - ROUND_DOWN ", final_val2)
print("Using Decimal - ROUND_FLOOR ", final_val3)
print("Using Decimal - ROUND_HALF_DOWN ", final_val4)
print("Using Decimal - ROUND_HALF_EVEN ", final_val5)
print("Using Decimal - ROUND_HALF_UP ", final_val6)
print("Using Decimal - ROUND_UP ", final_val7)

print("\n")
print("# Range")
for i in range(3, 10, 2):
   print(i, end=" ")

arr_list = ['Mysql', 'Mongodb', 'PostgreSQL', 'Firebase']
for i in range(len(arr_list)):
   print(arr_list[i], end=" ")

print(list(range(10)))

print(range(ord('a'), ord('c')))
print(chr(97))


def abc(c_start, c_stop):
   for i in range(ord(c_start), ord(c_stop)):
       yield chr(i)


print(list(abc("a", "t")))


def range1(x):
   for i in range(x):
       yield i


print(list(range1(5)))

startvalue = range(5)[0]
print("The first element in range is = ", startvalue)

secondvalue = range(5)[1]
print("The second element in range is = ", secondvalue)

lastvalue = range(5)[-1]
print("The first element in range is = ", lastvalue)

from itertools import chain

print("Merging two range into one")
frange = chain(range(10), range(10, 20, 1))
print(list(frange))

print("\n")
print("# Map")


def square(x):
   return x * x


my_list = [2, 3, 4, 5, 6, 7, 8, 9]
up_list = map(square, my_list)
print(list(up_list))

my_list = [2.6743, 3.63526, 4.2325, 5.9687967, 6.3265, 7.6988, 8.232, 9.6907]
updated_list = map(round, my_list)
print(updated_list)
print(list(updated_list))
for i in updated_list:
   print(i, end="")

print("\n")


def upfun(s):
   return s.upper()


my_str = "welcome to guru99 tutorials1!"
updated_list1 = map(upfun, my_str)

print(list(updated_list1))
print(list(updated_list1))  # dlaczego

for i in updated_list1:
   print(i, end="")

print("\n")

my_str = "welcome to guru99 tutorials2!"
updated_list2 = map(upfun, my_str)
for i in updated_list2:
   print(i, end="")

print(list(updated_list2))

my_tuple = ('php', 'java', 'python', 'c++', 'c')

updated_list = map(upfun, my_tuple)
print(updated_list)
print(list(updated_list))

my_list = [2, 3, 4, 5, 6, 7, 8, 9]
updated_list = map(lambda x: x + 10, my_list)
print(list(updated_list))


def myMapFunc(list1, tuple1):
   return list1 + "_" + tuple1


my_list = ['a', 'b', 'b', 'd', 'e']
my_tuple = ('PHP', 'Java', 'Python', 'C++', 'C')

updated_list = map(myMapFunc, my_list, my_tuple)
print(list(updated_list))

print("\n")
print("# Timeit")

import timeit

print(timeit.timeit("sadf = 10 * 5"))
print("The time taken is ", timeit.timeit(stmt='a=10;b=10;sum=a+b'))

import timeit

import_module = "import random"
testcode = '''
def test():
   return random.randint(10, 100)
'''
# print(timeit.repeat(stmt=testcode, setup=import_module))


print("\n")
print("# Yield")


def testyield():
   yield "Welcome to Guru99 Python Tutorials"


output = testyield()
for i in output:
   print(i)


def generator():
   yield "H"
   yield "E"
   yield "L"
   yield "L"
   yield "O"


print(list(generator()))

for i in generator():
   print(i)


# Normal function
def normal_test():
   return "Hello World"


# Generator function
def generator_test():
   yield "Hello World"


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
print(list(fib))
print(list(fib))


def test(n):
   return n * n


def getSquare(n):
   for i in range(n):
       yield test(i)


sq = getSquare(10)
print(list(sq))



print("\n")
print("# Queue")

import queue

q1 = queue.Queue()
q1.put(10)
q1.put(5)
print(q1.full())
item1 = q1.get()
print('The item removed from the queue is ', item1)
item1 = q1.get()
print('The item removed from the queue is ', item1)

for i in range(5):
   q1.put(i)

while not q1.empty():
   print("er", q1.get())

q2 = queue.LifoQueue()
for i in range(5):
   q2.put(i)

while not q2.empty():
   print("sdf", q2.get(), end=" ")
print("\n")

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
   print('x ', str(i), ' ', str(x))
   for j in range(n - 1):
       y = q1.get()  # the element is removed
       print('y ', str(j), ' ', str(y))
       if x > y:
           q1.put(y)  # the smaller one is put at the start of the queue

       else:
           q1.put(x)  # the smaller one is put at the start of the queue
           x = y  # the greater one is replaced with x and compared again with next element
       print(list(q1.queue))
   q1.put(x)

while (q1.empty() == False):
   print(q1.get(), end=" ")
print("\n")

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
print("\n")



print("\n")
print("# Counter")

from collections import Counter
list1 = ['x','y','z','x','x','x','y', 'z']
print(Counter(list1))

my_str = "Welcome to Guru99 Tutorials!"
print(Counter(my_str))

dict1 =  {'x': 4, 'y': 2, 'z': 2, 'z': 2}
print(Counter(dict1))

tuple1 = ('x','y','z','x','x','x','y','z')
print(Counter(tuple1))

_count = Counter()
_count.update('Welcome to Guru99 Tutorials!')
print(_count)
print("%s : %d" % ("u", _count["u"]))
for char in "Guru":
   print(char, _count[char])


print(dict1)
del dict1["x"]
print(dict1)
print(Counter(dict1))
print("\n")

counter1 =  Counter({'x': 4, 'y': 2, 'z': -2})

counter2 = Counter({'x1': -12, 'y': 5, 'z':4 })

#Addition
counter3 = counter1 + counter2 # only the values that are positive will be returned.

print(counter3)

#Subtraction
counter4 = counter1 - counter2 # all -ve numbers are excluded.For example z will be z = -2-4=-6, since it is -ve value it is not shown in the output
print(counter4)


#Intersection
counter5 = counter1 & counter2 # it will give all common positive minimum values from counter1 and counter2

print(counter5)

#Union
counter6 = counter1 | counter2 # it will give positive max values from counter1 and counter2

print(counter6)

counter1 =  Counter({'x': 5, 'y': 2, 'z': -2, 'x1':0})
_elements = counter1.elements()
for i in _elements:
   print(i)

common_element = counter1.most_common(2)
print(common_element)
print(counter1.most_common())


counter1 = Counter({'x': 5, 'y': 12, 'z': -2, 'x1':0})
counter2 = Counter({'x': 2, 'y':5})

counter1.subtract(counter2)
print(counter1)

counter1 = Counter({'x': 5, 'y': 12, 'z': -2, 'x1':0})
counter2 = Counter({'x': 2, 'y':5})
counter1.update(counter2)
print(counter1)

counter1 =  Counter({'x': 5, 'y': 12, 'z': -2, 'x1':0})
counter1['y'] = 20
counter1['y1'] = 1
print(counter1)
print(counter1['y'])



print("\n")
print("# Enumerate")

my_list = ['D', 'B', 'C', 'A']
en_l = enumerate(my_list)
print(list(en_l))

for i in enumerate(my_list):
   print(i)

for i in enumerate(my_list, 10):
   print(i)

for i in range(len(my_list)):
   print(my_list[i])

my_tuple = ("E", "B", "C", "D", "A")
for i in enumerate(my_tuple):
 print(i)

for i in enumerate(str1):
 print(i)

for i in enumerate(counter1):
 print(i)



print("\n")
print("# time sleep():")

import time
print("Welcome to guru99 Python Tutorials")
time.sleep(.00005)
print("This message will be printed after a wait of 5 seconds")

import asyncio
print('Code Execution Started')
async def display():
   await asyncio.sleep(.0005)
   print('Welcome to Guru99 Tutorials')

asyncio.run(display())

from threading import Event

print('Code Execution Started')
def display():
   print('Welcome to Guru99 Tutorials')

Event().wait(.00005)
display()

from threading import Timer
print('Code Execution Started')
def display():
   print('Welcome to Guru99 Tutorials')

t = Timer(.000005, display)
#t.start()



print("\n")
print("# type() and isinstance()")

str_list = "Welcome to Guru99"
age = 50
pi = 3.14
c_num = 3j+10
my_list = ["A", "B", "C", "D"]
my_tuple = ("A", "B", "C", "D")
my_dict = {"A": "a", "B": "b", "C": "c", "D": "d"}
my_set = {'A', 'B', 'C', 'D'}

print("The type is : ", type(str_list))
print("The type is : ", type(age))
print("The type is : ", type(pi))
print("The type is : ", type(c_num))
print("The type is : ", type(my_list))
print("The type is : ", type(my_tuple))
print("The type is : ", type(my_dict))
print("The type is : ", type(my_set))
print('\n')

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

age = isinstance(51, int)
print('age is integer :', age)
message = isinstance("Hello World",str)
print("message is a string:", message)
my_set = isinstance({1,2,3,4,5},set)
print("my_set is a set:", my_set)
class MyClass:
   _message = "Hello World"

_class = MyClass()
print("_class is a instance of MyClass() : ", isinstance(_class,MyClass))



print("\n")
print("# File Handling: Create, Open, Append, Read, Write")

f = open("guru99.txt","w+")
for i in range(2):
    f.write("This is line %d\r\n" % (i+1))

f.close()

f = open('guru99.txt', 'a+')

for i in range(2):
   f.write('Appended line %d\r\n' % (i+1))

f.close()

f = open('guru99.txt', 'r')
if f.mode == 'r':
   contents = f.read()
   print(contents)

f.close()

f = open('guru99.txt', 'r')
f1 = f.readlines()
for i in f1:
   print(i)

f.close()



print("\n")
print("# Check If File or Directory Exists")

import os.path
from os import path

def main():
  print("File exists:"+str(path.exists('guru99.txt')))
  print("File exists:" + str(path.exists('career.guru99.txt')))
  print("directory exists:" + str(path.exists('myDirectory')))

if __name__== "__main__":
  main()

def main():
   print("Is it File?" + str(path.isfile('guru99.txt')))
   print("Is it File?" + str(path.isfile('myDirectory')))

if __name__ == '__main__':
   main()

def main():

  print ("Is it Directory?" + str(path.isdir('guru99.txt')))
  print ("Is it Directory?" + str(path.isdir('myDirectory')))

if __name__== "__main__":
  main()


import pathlib
file = pathlib.Path('guru99.txt')
if file.exists():
   print("File exist")
else:
   print("File not exist")



print("\n")
print("# COPY File using shutil.copy(), shutil.copystat()")

import  os
import shutil
from os import path

if path.exists("guru99.txt"):
   src = path.realpath('guru99.txt')
   head, tail = path.split(src)
   print(src)
   print('patch :', head)
   print('file :', tail)
   dst = src+'.bak'
   shutil.copy(src, dst)
   # copy over the permissions
   shutil.copystat(src, dst)

import datetime
from datetime import date, time, timedelta
import time

def main():


   # Get the modification time
   t = path.getmtime("guru99.txt.bak")
   print(time.ctime(t))
   print(datetime.datetime.fromtimestamp(t))


if __name__ == "__main__":
   main()



print("\n")
print("# Rename")

import os
import shutil
from os import path


def main():
   # make a duplicate of an existing file
   if path.exists("guru99.txt"):
       # get the path to the file in the current directory
       src = path.realpath("guru99.txt");

       # rename the original file
       os.rename('guru99.txt', 'career.guru99.txt')


#if __name__ == "__main__":
   main()



print("\n")
print("# Zip")

import os
import shutil
from zipfile import ZipFile
from os import path
from shutil import make_archive

# Check if file exists
if path.exists("guru99.txt"):
   # get the path to the file in the current directory
   src = path.realpath("guru99.txt")
   # now put things into a ZIP archive
   root_dir, tail = path.split(src)
   shutil.make_archive("guru99 archive","zip",root_dir)
   # more fine-grained control over ZIP files
   with ZipFile("testguru99.zip", "w") as newzip:
       newzip.write("guru99.txt")
       newzip.write("guru99.txt.bak")



print("\n")
print("# Try, Catch, Finally")

# try:
#    {
#    catch (ArrayIndexOutOfBoundsException e) {
#    System.err.printin("Caught first " + e.getMessage()); } catch (IOException e) {
#    System.err.printin("Caught second " + e.getMessage());}
#    }

#try:
#    raise KeyboardInterrupt
#finally:
#    print('welcome, world!')
#Output
#Welcome, world!
#KeyboardInterrupt



print("\n")
print("# readline()")

f = open("demo.txt","w+")
for i in range(5):
    f.write("Testing - %d line \r" % (i+1))

f.close()

f = open('demo.txt', 'r')
if f.mode == 'r':
   contents = f.read()
   print(contents)

f.close()

myfile = open("demo.txt", "r")
myline = myfile.readline()
print(myline)
myfile.close()

myfile = open("demo.txt", "r")
myline = myfile.readline(11)
print(myline)
myfile.close()

myfile = open("demo.txt", "r")
myline = myfile.readline()
while myline:
   print(myline)
   myline = myfile.readline()
myfile.close()


myfile = open("demo.txt", "r")
mylist = myfile.readlines()
print(mylist)
myfile.close()

myfile = open("demo.txt", "r")
for line in myfile:
   print(line)
myfile.close()

myfile = open("demo.txt", "r")
while myfile:
   line  = myfile.readline()
   print(line)
   if line == "":
       break
myfile.close()



print("\n")
print("# SciPy")

# Python -m pip install --user numpy scipy
# sudo apt install python3-pip
# pip3 install numpy scipy
# sudo pip3 install -U numpy
# conda install -c anaconda numpy
# /home/ukasz/PycharmProjects/pythonProject/.git/


import numpy as np
from scipy import io as sio
array = np.ones((4, 4))
sio.savemat('example.mat', {'ar': array})
data = sio.loadmat('example.mat', struct_as_record=True)
print(data['ar'])

from scipy.special import cbrt
#Find cubic root of 27 & 64 using cbrt() function
cb = cbrt([27, 64])
#print value of cb
print(cb)

from scipy.special import exp10
#define exp10 function and pass value in its
exp = exp10([1, 10, 20])
print(exp)

from scipy.special import comb
#find combinations of 5, 2 values using comb(N, k)
com = comb(5, 2, exact=False, repetition=False)
print(com)

from scipy.special import perm
#find permutation of 5, 2 using perm (N, k) function
per = perm(5, 2, exact = False)
print(per)

from scipy import linalg
import numpy as np
two_d_array = np.array([[4, 5], [3, 2]])
print(two_d_array)
print(linalg.det(two_d_array))
print(linalg.inv(two_d_array))
eg_val, eg_vect = linalg.eig(two_d_array )
print(eg_val)
#get eigenvectors
print(eg_vect)


# tkinter
# sudo apt-get install python3-tk
# matplotlib inline
from matplotlib import pyplot as plt
import numpy as np
#Frequency in terms of Hertz
fre = 5
#Sample rate
fre_samp = 50
t = np.linspace(0, 2, 2 * fre_samp, endpoint = False )
a = np.sin(fre * 2 * np.pi * t)
figure, axis = plt.subplots()
axis.plot(t, a)
axis.set_xlabel ('Time (s)')
axis.set_ylabel ('Signal amplitude')
#plt.show()


from scipy import fftpack
A = fftpack.fft(a)
frequency = fftpack.fftfreq(len(a)) * fre_samp
figure, axis = plt.subplots()

axis.stem(frequency, np.abs(A))
axis.set_xlabel('Frequency in Hz')
axis.set_ylabel('Frequency Spectrum Magnitude')
axis.set_xlim(-fre_samp / 2, fre_samp/ 2)
axis.set_ylim(-5, 110)
#plt.show()

# matplotlib inline
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np

def function(a):
       return   a*2 + 20 * np.sin(a)
plt.plot(a, function(a))
#plt.show()
#use BFGS algorithm for optimization
# optimize.fmin_bfgs(function, 0)

from scipy import misc
from matplotlib import pyplot as plt
import numpy as np
#get face image of panda from misc package
panda = misc.face()
#plot or show image of face
plt.imshow( panda )
#plt.show()


from scipy import integrate
# take f(x) function as f
f = lambda x : x**2
#single integration with a = 0 & b = 1
integration = integrate.quad(f, 0 , 1)
print(integration)

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
integration = integrate.dblquad(f , 0 , 2/4,  p, q)
print(integration)



print("\n")
print("# CSV")

#import necessary modules
import csv
with open('data.csv', 'rt') as f:
  data = csv.reader(f)
  for row in data:
        print(row)

reader = csv.DictReader(open("data.csv"))
for raw in reader:
    print(raw)


with open('writeData.csv', mode='w') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    #way to write to csv file
    writer.writerow(['Programming language', 'Designed by', 'Appeared', 'Extension'])
    writer.writerow(['Python', 'Guido van Rossum', '1991', '.py'])
    writer.writerow(['Java', 'James Gosling', '1995', '.java'])
    writer.writerow(['C++', 'Bjarne Stroustrup', '1985', '.cpp'])


from pandas import DataFrame




© 2021 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
