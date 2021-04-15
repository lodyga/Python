print("USA")
print("UK")
print(2 * "\n")
print("Canada")
print("\n\n")
print("Germany", end = "!")
print("France", end = " @# ")
print("Japan")
print("Poland")

print(8 * "\n")
print("\n\n\n")

a = 100
print(a)
b = 99
print('ABC' + str(b))


def someFunction():
    a='Twoja Stata'
    print(a)
someFunction()
print(a)

def someFunction():
    global a
    print(a)
someFunction()
print(a)


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



# Dictionary
Dict = {'Tim': 18, 'Charlie': 12, 'Tiffany': 22, 'Robert': 25}
print(Dict['Tiffany'])
# to samo 
print(Dict.get("Tim"))
Boys = {'Tim': 18, 'Charlie': 12, 'Robert': 25}
Girls = {'Tiffany': 22}
studentX = Boys.copy()
print(Boys)
print(studentX)
Dict.update({"Sarah": 9})
print(Dict)
del Dict['Charlie']
print(Dict)

print("Students Name: %s" % list(Dict.items()))

Dict = {'Tim': 18, 'Charlie': 12, 'Tiffany': 22, 'Robert': 25}
Boys = {'Tim': 18, 'Charlie': 12, 'Robert': 25}
for key in Dict.keys():
   if key in Boys.keys():
       print(True)
   else:
       print(False)

Students = list(Dict.keys())
Students.sort()
print(Students)
for S in Students:
   print(": ".join((S, str(Dict[S]))))

for S in Students:
   print(S)

print("Length : %d" % len(Dict))
print(len(Students))
print("variable Type: %s" % type(Dict))
print("printable string:%s" % str(Dict))
print("sdf %s" % str(Dict))

Dict_copy = Dict.copy()
print(Dict_copy)
Dict_copy.update({"Tim": 80})
print(Dict_copy)


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
# to to samo co
# my_dict.update({"name": "Nick"})

my_dict1["my_dict1"] = my_dict
print(my_dict1)



# Operators

x = 4
y = 5
print(x + y)
print('x>y is', x != y)
x += 6
print("x+6=", x)
a = True
b = False
print('a and b is', a and b)

lis1 = [1, 2, 3, 4, 5]
if (y in lis1):
   print('Yes')
else:
   print('No')
print(9 // 4)
print(9 % 4)
print(9 ** 4)




# Arrays

import array as myarray
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

element = my_list.clear()
print(element)
print(my_list)

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
#country = "US"
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


def SwitchExample(argument):
    switcher = {
        0: " This is Case Zero ",
        1: " This is Case One ",
        2: " This is Case Two ",
    }
    return switcher.get(argument, "nothing")
    # return switcher[argument]


if __name__ == "__main__":
    argument = 0
    print (SwitchExample(argument))










# For & While

x = 0
while x < 4:
   print(x)
   x += 1

for x in range(2, 4):
   print(x)

for x in enumerate(range(2, 4)):
   print(x)

Months = ["Jan", "Feb", "Mar", "April", "May", "June"]
for m in Months:
   print(m)

for x in range(10, 20):
   # if x==15: break
   if (x % 2 == 0): continue
   print(x)

for i, m in enumerate(Months):
   print(i, m)

for x in '234':
   print('wartosc :', x)






# Break % Continue

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







# Class, Object

# Example file for working with classes
class myClass():
   def method1(self):
      print("Guru99")
        
   def method2(self, someString):    
      print("Software Testing:" + someString)

def main():           
   # exercise the class methods
   c = myClass()
   c.method1()
   c.method2(" Testing is fun")
  
if __name__== "__main__":
   main()


# Example file for working with classes
class myClass():
   def method1(self):
      print("Guru99")

class childClass(myClass):
   #def method1(self):
      # myClass.method1(self)
      #print("childClass Method1")

   def method2(self):
      print("childClass method2")

def main1():
   # exercise the class methods
   c2 = childClass()
   c2.method1()
   # c2.method2()

if __name__ == "__main__":
   main1()


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
print(name, "", number)
print(name + str(number))
print(name + " " + str(number))

print(name * 2)
print(number * 2)
print(name[1:3]+ name)
print(name[1:3], name)

oldstring = 'nie lubię Cię'
newstring = oldstring.replace('nie', '')
print(newstring)
print(name.upper())
print(name.capitalize())
print(':'.join(name))
print(":".join((name, str(number))))
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
print(str1.strip("c a"))




# Count
print(str1.count('a'))
print("abcda".count('a', 2, ))






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


# Using class with format()
class MyClass1():
   msg1 = 'twoja'
   msg2 = 'stara'

print('tak to {c.msg1}1 {c.msg2}2'.format(c = MyClass1))


# Using dictionary with format()
my_dict = {'msg1': "twoja", 'msg2': "stara"}
print('{m[msg1]}1 {m[msg2]}2'.format(m = my_dict))
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
print("The position of Tutorials using find() : ", mystring.find("Tutorials", 20))
print("The position of Tutorials using rfind() : ", mystring.rfind("Tutorials"))
print("The position of Tutorials using index() : ", mystring.index("Tutorials"))


my_string = "test string test, test string testing, test string test string"
startIndex = 0
count = 0
for i in range(len(mystring)):
   k = my_string.find("test", startIndex)
   if(k != -1):
      startIndex = k+1
      count += 1
      # k = 0
print("The total count of substring test is:", count)








# Main Function & Method Example: Understand __main__
def main1():
   print('hello')

if __name__ == "__main__":
   main1()

print('tak')
print(__name__)



# How Function Return Value?
def sq(x = 5):
   return x * x
print(sq())


def multi(x, y=0):
   print("x =", x)
   print("y =", y)
   return x * y
print(multi(y=2, x=4))


def fun2(*args):
   print(args)
   return args
fun2(1, 2, 3, 4, 5)






# Lambda Functions

adder = lambda x, y=2: x + y
print(adder(1))

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


sequence = [10, 2, 8, 7, 5, 4, 3, 11, 0, 1]
filtered_result = filter(lambda x: x > 4, sequence)
print(list(filtered_result))
print(filtered_result)


def filter2(arg):
   for i in range(len(arg)):
      if arg[i] > 4:
         print(arg[i])
filter2(sequence)


sequence = [10, 2, 8, 7, 5, 4, 3, 11, 0, 1]
def filter3(arg):
   lis = []
   it = 0
   for i in range(len(arg)):
      if arg[i] > 4:
         lis.append(arg[i])
         it += 1
   return lis
filter3(sequence)


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


import numpy as np
arr = [-0.341111, 1.455098989, 4.232323, -0.3432326, 7.626632, 5.122323]
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

print("Using round()", final_val)
print("Using Decimal - ROUND_CEILING ", final_val1)
print("Using Decimal - ROUND_DOWN ", final_val2)
print("Using Decimal - ROUND_FLOOR ", final_val3)
print("Using Decimal - ROUND_HALF_DOWN ", final_val4)
print("Using Decimal - ROUND_HALF_EVEN ", final_val5)
print("Using Decimal - ROUND_HALF_UP ", final_val6)
print("Using Decimal - ROUND_UP ", final_val7)








# Range
for i in range(3, 10, 2):
   print(i, end=" ")

for i in range(15, 5, -1):
    print(i, end =" ")

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



import numpy as np 
for i in np.arange(10):
   print(i, end =" ")  

import numpy as np 
for  i in np.arange(0.5, 1.5, 0.2):
   print(i, end =" ") 







# Map

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


def upfun(s):
   return s.upper()
my_str = "welcome to guru99 tutorials1!"
updated_list1 = map(upfun, my_str)
print(list(updated_list1))

for i in updated_list1:
   print(i, end="")


my_tuple = ('php', 'java', 'python', 'c++', 'c')
updated_list = map(upfun, my_tuple)
print(list(updated_list))

my_list = [2, 3, 4, 5, 6, 7, 8, 9]
updated_list = map(lambda x: x + 10, my_list)
print(list(updated_list))


def myMapFunc(list1, list2):
   return list1+list2

my_list1 = [2,3,4,5,6,7,8,9]
my_list2 = [4,8,12,16,20,24,28]

updated_list = map(myMapFunc, my_list1,my_list2)
print(list(updated_list))


def myMapFunc(list1, tuple1):
   return list1+"_"+tuple1

my_list = ['a','b', 'b', 'd', 'e']
my_tuple = ('PHP','Java','Python','C++','C')

updated_list = map(myMapFunc, my_list, my_tuple)
print(list(updated_list))










# Timeit"

import timeit
print(timeit.timeit("sadf = 10 * 5"))
print("The time taken is ", timeit.timeit(stmt='a=10;b=10;sum=a+b'))

import timeit
import_module = "import random"
testcode = '''
def test():
   return random.randint(10, 100)
'''
print(timeit.repeat(stmt=testcode, setup=import_module))










# Yield

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


def test(n):
   return n * n

def getSquare(n):
   for i in range(n):
      yield test(i)

sq = getSquare(10)
print(list(sq))










# Queue

import queue
q1 = queue.Queue()
q1.put(10)
q1.put(5)
print(q1.full())
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
print(_count["u"])
for char in "Guru":
   print(char, _count[char])


from collections import Counter
dict1 =  {'x': 4, 'y': 2, 'z': 2}
del dict1["x"]
print(Counter(dict1))


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
print(list(counter1.elements()))
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











# Enumerate

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

str1 = "dup"
for i in enumerate(str1):
   print(i)













# time sleep():

import time
print("Welcome to guru99 Python Tutorials")
time.sleep(.5)
print("This message will be printed after a wait of 5 seconds")


import time
print('Code Execution Started')
def display():
   print('Welcome to Guru99 Tutorials')
   time.sleep(5)

display()
print('Function Execution Delayed')


# coś nie działa
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


# coś nie działa
from threading import Timer
print('Code Execution Started')
def display2():
   print('Welcome to Guru99 Tutorials')

t = Timer(5, display2)
t.start()








# type() and isinstance()

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
message = isinstance("Hello World", str)
print("message is a string:", message)
my_set = isinstance({1,2,3,4,5}, set)
print("my_set is a set:", my_set)
my_tuple = isinstance((1,2,3,4,5), tuple)
print("my_tuple is a set:", my_tuple)
my_list = isinstance([1,2,3,4,5],list)
print("my_list is a list:", my_list)
my_dict = isinstance({"A":"a", "B":"b", "C":"c", "D":"d"},dict)
print("my_dict is a dict:", my_dict)

class MyClass:
   _message = "Hello World"

_class = MyClass()
print("_class is a instance of MyClass() : ", isinstance(_class,MyClass))









# File Handling: Create, Open, Append, Read, Write

dir = "/home/ukasz/Documents/Programowanie/Python/"

f = open(dir+"guru99.txt","w+")
for i in range(2):
   f.write("This is line %d\r\n" % (i+1))
f.close()


f = open(dir+"guru99.txt", 'a+')
for i in range(2):
   f.write('Appended line %d\r\n' % (i+1))
f.close()


f = open(dir+"guru99.txt", 'r')
if f.mode == 'r':
   contents = f.read()
   print(contents)
f.close()


f = open(dir+"guru99.txt", 'r')
f1 = f.readlines()
for i in f1:
   print(i)
f.close()










# Check If File or Directory Exists

import os.path
from os import path

dir = "/home/ukasz/Documents/Programowanie/Python/"
print("File exists:" + str(path.exists(dir+'guru99.txt')))
print("File exists:" + str(path.exists(dir+'career.guru99.txt')))
print("directory exists:" + str(path.exists(dir+'PycharmProjects')))

print("Is it File?" + str(path.isfile(dir+'guru99.txt')))
print("Is it File?" + str(path.isfile(dir+'myDirectory')))

print ("Is it Directory?" + str(path.isdir(dir+'guru99.txt')))
print ("Is it Directory?" + str(path.isdir(dir+'PycharmProjects')))


import pathlib
file = pathlib.Path(dir+'guru99.txt')
if file.exists():
   print("File exist")
else:
   print("File not exist")











# COPY File using shutil.copy(), shutil.copystat()

import os
import shutil
from os import path

dir = "/home/ukasz/Documents/Programowanie/Python/"
if path.exists(dir+"guru99.txt"):
   src = path.realpath(dir+'guru99.txt')
   head, tail = path.split(src)
   print(src)
   print('patch :', head)
   print('file :', tail)
   dst = src+'.bak'
   shutil.copy(src, dst)
   # copy over the permissions
   shutil.copystat(src, dst)


from os import path
import datetime
from datetime import date, time, timedelta
import time

# Get the modification time
dir = "/home/ukasz/Documents/Programowanie/Python/"
t = path.getmtime(dir+"guru99.txt.bak")
print(t)
print(time.ctime(t))
print(datetime.datetime.fromtimestamp(t))










# Rename

import os
import shutil
from os import path


# make a duplicate of an existing file
dir = "/home/ukasz/Documents/Programowanie/Python/"
if path.exists(dir+"guru99.txt"):
   # get the path to the file in the current directory
   src = path.realpath(dir+"guru99.txt")
   # rename the original file
   os.rename(dir+'guru99.txt', dir+'career.guru99.txt')












# Zip

import os
import shutil
from zipfile import ZipFile
from os import path
from shutil import make_archive

# Check if file exists
dir = "/home/ukasz/Documents/Programowanie/Python/"
if path.exists(dir+"guru99.txt"):
   # get the path to the file in the current directory
   src = path.realpath(dir+"guru99.txt")
   # now put things into a ZIP archive
   root_dir, tail = path.split(src)
   shutil.make_archive(dir+"guru99_archive","zip",root_dir)
   # more fine-grained control over ZIP files
   with ZipFile(dir+"testguru99.zip", "w") as newzip:
      newzip.write(dir+"guru99.txt")
      newzip.write(dir+"guru99.txt.bak")










# Try, Catch, Finally

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

dir = "/home/ukasz/Documents/Programowanie/Python/"
f = open(dir+"demo.txt","w+")
for i in range(5):
   f.write("Testing - %d line \r\n" % (i+1))
f.close()


f = open(dir+'demo.txt', 'r')
if f.mode == 'r':
   contents = f.read()
   print(contents)
f.close()


myfile = open(dir+"demo.txt", "r")
myline = myfile.readline()
print(myline)
myfile.close()


myfile = open(dir+"demo.txt", "r")
myline = myfile.readline(10)
print(myline)
myfile.close()


myfile = open(dir+"demo.txt", "r")
myline = myfile.readline()
while myline:
   print(myline)
   myline = myfile.readline()
myfile.close()


myfile = open(dir+"demo.txt", "r")
mylist = myfile.readlines()
print(mylist)
myfile.close()


myfile = open(dir+"demo.txt", "r")
for line in myfile:
   print(line)
myfile.close()


myfile = open(dir+"demo.txt", "r")
while myfile:
   line  = myfile.readline()
   print(line)
   if line == "":
       break
myfile.close()












# SciPy

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
#get eigenvalues
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

panda = misc.face()
#rotatation function of scipy for image – image rotated 135 degree
panda_rotate = ndimage.rotate(panda, 135)
plt.imshow(panda_rotate)
plt.show()


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











# CSV

#import necessary modules
import csv
dir = "/home/ukasz/Documents/Programowanie/Python/"
with open(dir+'data.csv', mode = 'rt', encoding='utf-8-sig') as f:
   data = csv.reader(f)
   for row in data:
      print(row)


import csv
dir = "/home/ukasz/Documents/Programowanie/Python/"
reader = csv.DictReader(open(dir+"data.csv", encoding='utf-8-sig'))
for raw in reader:
   print(raw)


with open(dir+'writeData.csv', mode='w') as file:
   writer = csv.writer(file, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
   #way to write to csv file
   writer.writerow(['Programming language', 'Designed by', 'Appeared', 'Extension'])
   writer.writerow(['Python', 'Guido van Rossum', '1991', '.py'])
   writer.writerow(['Java', 'James Gosling', '1995', '.java'])
   writer.writerow(['C++', 'Bjarne Stroustrup', '1985', '.cpp'])


#import necessary modules
import pandas
result = pandas.read_csv(dir+'data.csv')
print(result)


from pandas import DataFrame
C = {
   'Programming language': ['Python','Java', 'C++'],
   'Designed by': ['Guido van Rossum', 'James Gosling', 'Bjarne Stroustrup'],
   'Appeared': ['1991', '1995', '1985'],
   'Extension': ['.py', '.java', '.cpp'],
}
df = DataFrame(C, columns = ['Programming language', 'Designed by', 'Appeared', 'Extension'])
# here you have to write path, where result file will be stored
export_csv = df.to_csv ('pandaresult.csv', index=None, header=True) 
print(df)












# JSON

import json
x = {
   "name": "Ken",
   "age": 45,
   "married": True,
   "children": ("Alice", "Bob"),
   "pets": ['Dog'],
   "cars": [
      {"model": "Audi A1", "mpg": 15.1},
      {"model": "Zeep Compass", "mpg": 18.1}
   ]
}
# sorting result in asscending order by keys:
print(type(x))
print(x)
sorted_string = json.dumps(x, indent=4, sort_keys=True)
print(sorted_string)


# here we create new data_file.json file with write mode using file i/o operation
dir = "/home/ukasz/Documents/Programowanie/Python/"
with open(dir+'json_file.json', "w") as file_write:
   person_data = {  "person":  { "name":  "Kenn",  "sex":  "male",  "age":  28}}
   # write json data into file
   print(json.dump(person_data, file_write))


import json # json data string
person_data = '{"person": {"name": "Kenn", "sex": "male", "age": 28}}'
print(type(person_data))
# Decoding or converting JSON format in dictionary using loads()
dict_obj = json.loads(person_data)
# check type of dict_obj
print("Type of dict_obj", type(dict_obj))
# get human object details
print("Person......",  dict_obj.get('person'))


import json
#File I/O Open function for read data from JSON File
dir = "/home/ukasz/Documents/Programowanie/Python/"
with open(dir+'json_file.json') as file_object:
   # store file data in object
   data = json.load(file_object)
   print(type(file_object))
   print(type(data))
   print(data)


import json
# Create a List that contains dictionary
lst = ['a', 'b', 'c', {'4': 5, '6': 7}]
# separator used for compact representation of JSON.
# Use of ',' to identify list items
# Use of ':' to identify key and value in dictionary
compact_obj = json.dumps(lst, separators=(',', ':'))
print(compact_obj)


import json
dic = { 'a': 4, 'b': 5 }
''' To format the code use of indent and 4 shows number of space and use of separator is not necessary but standard way to write code of particular function. '''
formatted_obj = json.dumps(dic, indent=4, separators=(',', ':'))
print(formatted_obj)


import json
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
print(JSONEncoder().encode(colour_dict))


import json
# import JSONDecoder class from json
from json.decoder import JSONDecoder
colour_string = '{ "colour": ["red", "yellow"]}'
# directly called decode method of JSON
JSONDecoder().decode(colour_string)


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
dir = "/home/ukasz/Documents/Programowanie/Python/"
#File I/O Open function for read data from JSON File
data = {} #Define Empty Dictionary Object
try:
   with open(dir+'json_file.json') as file_object:
    data = json.load(file_object)
    print(data)
except ValueError:
   print("Bad JSON file format,  Change JSON File")


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
   port=3306,
   user="root",
   passwd="<password>",
   db="myflixdb"
)
print(db_connection)


import mysql.connector
db_connection = mysql.connector.connect(
   host="localhost",
   port=3306,
   user="root",
   passwd="<password>"
)
# creating database_cursor to perform SQL operation
db_cursor = db_connection.cursor()
# executing cursor with execute method and pass SQL query
db_cursor.execute("CREATE DATABASE my_first_db")
# get list of all databases
db_cursor.execute("SHOW DATABASES")
#print all databases
for db in db_cursor:
   print(db)


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
db_cursor.execute("CREATE TABLE student (id INT, name VARCHAR(255))")
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











# Matrix

M1 = [[8, 14, -6], [12, 7, 4], [-11, 3, 21], [3, 4, 5]]
matrix_length = len(M1)
#To read the last element from each row.
for i in range(matrix_length):
    print(M1[i][-1])

#To print the rows in the Matrix
for i in range(matrix_length):
    print(M1[i])


M1 = [[8, 14, -6], 
      [12,7,4], 
      [-11,3,21]]
    
M2 = [[3, 16, -6],
           [9,7,-4], 
           [-1,3,13]]

M3  = [[0,0,0],
       [0,0,0],
       [0,0,0]]

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


arr = np.array([2,4,6,8,10,12,14,16])
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


print(M1[:2,]) # This will print f
print(M1[:3,:2])








# List: Comprehension, Apend, Sort, Length, Reverse

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

animals = ("cat", "dog", "fish", "cow")
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

list1 = [2,3,4,3,10,3,5,6,3]
elm_count = list1.count(3)
print('The count of element: 3 is ', elm_count)

list1 = ['red', 'green', 'blue', 'orange', 'green', 'gray', 'green']
color_count = list1.count('green')
print('The count of color: green is ', color_count)


my_list = [1,1,2,3,2,2,4,5,6,2,1]
temp_list = []
for i in my_list:
    if i not in temp_list:
        temp_list.append(i)
print(temp_list)

my_list = [1,1,2,3,2,2,4,5,6,2,1]
temp_list = []
[temp_list.append(i) for i in my_list if i not in temp_list]
print(temp_list)

my_list = [1,1,2,3,2,2,4,5,6,2,1]
my_final_list = set(my_list)
print(my_final_list)

import numpy as np
my_list = [1,2,2,3,1,4,5,1,2,6]
myFinalList = np.unique(my_list).tolist()
print(myFinalList)

import pandas as pd
my_list = [1,2,2,3,1,4,5,1,2,6]
myFinalList = pd.unique(my_list).tolist()
print(myFinalList)

my_list = [1,2,2,3,1,4,5,1,2,6]
my_finallist = []
for j, i in enumerate(my_list):
   if i not in my_list[:j]:
      my_finallist.append(i)
print(my_finallist)

my_list = [1,2,2,3,1,4,5,1,2,6]
my_finallist = [i for j, i in enumerate(my_list) if i not in my_list[:j]] 
print(my_finallist)

my_list = ['A', 'B', 'C', 'D', 'E', 'F']
print("The index of element C is ", my_list.index('C'))
print("The index of element F is ", my_list.index('F'))


my_list = ['Guru', 'Siya', 'Tiya', 'Guru', 'Daksh', 'Riya', 'Guru'] 
all_indexes = [] 
for i in range(len(my_list)) : 
   if my_list[i] == 'Guru' : 
      all_indexes.append(i)
print("Originallist ", my_list)
print("Indexes for element Guru : ", all_indexes)


my_list = ['Guru', 'Siya', 'Tiya', 'Guru', 'Daksh', 'Riya', 'Guru'] 
result = []
elementindex = -1
while True:
   try:
      elementindex = my_list.index('Guru', elementindex+1)
      result.append(elementindex)
   except ValueError:
      break
print("OriginalList is ", my_list)
print("The index for element Guru is ", result)


my_list = ['Guru', 'Siya', 'Tiya', 'Guru', 'Daksh', 'Riya', 'Guru'] 
print("Originallist ", my_list)
all_indexes = [a for a in range(len(my_list)) if my_list[a] == 'Guru']
print("Indexes for element Guru : ", all_indexes)

# list of letters
letters = ['a', 'b', 'd', 'e', 'i', 'j', 'o']

# function that filters vowels
def filter_vowels(letter):
   vowels = ['a', 'e', 'i', 'o', 'u']
   if(letter in vowels):
      return True
   else:
      return False
filtered_vowels = filter(filter_vowels, letters)

print('The filtered vowels are:')
for vowel in filtered_vowels:
   print(vowel)


my_list = ['Guru', 'Siya', 'Tiya', 'Guru', 'Daksh', 'Riya', 'Guru'] 
print("Originallist ", my_list)
all_indexes = list(filter(lambda i: my_list[i] == 'Guru', range(len(my_list)))) 
print("Indexes for element Guru : ", all_indexes)


import numpy as np
my_list = ['Guru', 'Siya', 'Tiya', 'Guru', 'Daksh', 'Riya', 'Guru'] 
np_array = np.array(my_list)
item_index = np.where(np_array == 'Guru')[0]
print("Originallist", my_list)
print("Indexes for element Guru :", item_index)


from more_itertools import locate
my_list = ['Guru', 'Siya', 'Tiya', 'Guru', 'Daksh', 'Riya', 'Guru'] 
print("Originallist : ", my_list)
print("Indexes for element Guru :", list(locate(my_list, lambda x: x == 'Guru'))) 








# RegEx

import re
xx = "guru99,education is fun"
r1 = re.findall(r"^\w+", xx)
print(r1)


import re
xx = "guru99,education is fun"
r1 = re.findall(r"^\w+", xx)
print((re.split(r'\s','we are splitting the words')))
print((re.split(r's','split the swords')))


import re
list = ["guru99 get", "guru99 give", "guru Selenium"]
for element in list:
   z = re.match(r"(g\w+)\W(g\w+)", element)
   if z:
      print((z.groups()))


patterns = ['software testing', 'guru99']
text = 'software testing is fun?'
for pattern in patterns:
   print('Looking for "%s" in "%s" ->' % (pattern, text), end=' ')
   if re.search(pattern, text):
      print('found a match!')
   else:
      print('no match')


abc = 'guru99@google.com, careerguru99@hotmail.com, users@yahoomail.com'
emails = re.findall(r'[\w\.-]+@[\w\.-]+', abc)
for email in emails:
   print(email)

abc = ['guru99@google.com, careerguru99@hotmail.com, users@yahoomail.com']
for i in abc:
   if re.search(r'[\w\.-]+@[\w\.-]+', i):
      print(i)


import re
xx = """guru99 
careerguru99	
selenium"""
k1 = re.findall(r"^\w", xx)
k2 = re.findall(r"^\w", xx, re.MULTILINE)
print(k1)
print(k2)


import re
# Lets use a regular expression to match a date string. Ignore
# the output since we are just testing if the regex matches.
regex = r"([a-zA-Z]+) (\d+)"
if re.search(regex, "June 24"):
   # Indeed, the expression "([a-zA-Z]+) (\d+)" matches the date string
   
   # If we want, we can use the MatchObject's start() and end() methods 
   # to retrieve where the pattern matches in the input string, and the 
   # group() method to get all the matches and captured groups.
   match = re.search(regex, "June 24")
   
   # This will print [0, 7), since it matches at the beginning and end of the 
   # string
   print("Match at index %s, %s" % (match.start(), match.end()))
   
   # The groups contain the matched values.  In particular:
   #    match.group(0) always returns the fully matched string
   #    match.group(1), match.group(2), ... will return the capture
   #            groups in order from left to right in the input string
   #    match.group() is equivalent to match.group(0)
   
   # So this will print "June 24"
   print("Full match: %s" % (match.group(0)))
   # So this will print "June"
   print("Month: %s" % (match.group(1)))
   # So this will print "24"
   print("Day: %s" % (match.group(2)))
else:
   # If re.search() does not match, then None is returned
   print("The regex pattern does not match. :(")


import re
# Lets use a regular expression to match a few date strings.
regex = r"[a-zA-Z]+ \d+"
matches = re.findall(regex, "June 24, August 9, Dec 12")
for match in matches:
   # This will print:
   #   June 24
   #   August 9
   #   Dec 12
   print("Full match: %s" % (match))

# To capture the specific months of each date we can use the following pattern
regex = r"([a-zA-Z]+) \d+"
matches = re.findall(regex, "June 24, August 9, Dec 12")
for match in matches:
   # This will now print:
   #   June
   #   August
   #   Dec
   print("Match month: %s" % (match))

# If we need the exact positions of each match
regex = r"([a-zA-Z]+) \d+"
matches = re.finditer(regex, "June 24, August 9, Dec 12")
for match in matches:
   # This will now print:
   #   0 7
   #   9 17
   #   19 25
   # which corresponds with the start and end of each match in the input string
   print("Match at index: %s, %s" % (match.start(), match.end()))
   

import re
# Lets try and reverse the order of the day and month in a date 
# string. Notice how the replacement string also contains metacharacters
# (the back references to the captured groups) so we use a raw 
# string for that as well.
regex = r"([a-zA-Z]+) (\d+)"

# This will reorder the string and print:
#   24 of June, 9 of August, 12 of Dec
print(re.sub(regex, r"\2 of \1", "June 24, August 9, Dec 12"))
   
   
import re
# Lets create a pattern and extract some information with it
regex = re.compile(r"(\w+) World")
result = regex.search("Hello World is the easiest")
if result:
    # This will print:
    #   0 11
    # for the start and end of the match
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

import pytest
def test_file1_method1():
	x=5
	y=6
	assert x+1 == y,"test failed1"
	assert x == y,"test failed2 because x=" + str(x) + " y=" + str(y)
def test_file1_method2():
	x=5
	y=6
	assert x+1 == y,"test failed3" 

def test_file2_method1():
	x=5
	y=6
	assert x+1 == y,"test failed"
	assert x == y,"test failed because x=" + str(x) + " y=" + str(y)
def test_file2_method2():
	x=5
	y=6
	assert x+1 == y,"test failed"


py.test
py.test test_sample1.py
py.test -k method1 -v
-k <expression> is used to represent the substring to match
-v increases the verbosity

pytest -m set1 -v

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

def test_comparewithAA(supply_AA_BB_CC):
	zz = 35
	assert supply_AA_BB_CC[0] == zz,"aa and zz comparison failed"

def test_comparewithBB(supply_AA_BB_CC):
	zz = 35
	assert supply_AA_BB_CC[1] == zz,"bb and zz comparison failed"

def test_comparewithCC(supply_AA_BB_CC):
	zz = 35
	assert supply_AA_BB_CC[2] == zz,"cc and zz comparison failed"

pytest test_basic_fixture.py -v

# conftest.py A fixture method can be accessed across multiple test files by defining it in conftest.py file.

pytest -k test_comparewith -v


import pytest
@pytest.mark.parametrize("input1, input2, output", [(5, 5, 10), (3, 5, 12)])
def test_add(input1, input2, output):
	assert input1 + input2 == output, "failed"

pytest -k test_add -v


import pytest
@pytest.mark.skip
def test_add_1():
	assert 100+200 == 400, "failed"

@pytest.mark.skip
def test_add_2():
	assert 100+200 == 300, "failed"

@pytest.mark.xfail
def test_add_3():
	assert 15+13 == 28, "failed"

@pytest.mark.xfail
def test_add_4():
	assert 15+13 == 100, "failed"

def test_add_5():
	assert 3+2 == 5, "failed"

def test_add_6():
	assert 3+2 == 6, "failed"


py.test test_sample1.py -v --junitxml="result.xml"

https://reqres.in/.

import pytest
import requests
import json
@pytest.mark.parametrize("userid, firstname",[(1,"George"),(2,"Janet")])
def test_list_valid_user(supply_url,userid,firstname):
	url = supply_url + "/users/" + str(userid)
	resp = requests.get(url)
	j = json.loads(resp.text)
	assert resp.status_code == 200, resp.text
	assert j['data']['id'] == userid, resp.text
	assert j['data']['first_name'] == firstname, resp.text

def test_list_invaliduser(supply_url):
	url = supply_url + "/users/50"
	resp = requests.get(url)
	assert resp.status_code == 404, resp.text


pytest -k test_list -v
pytest -k test_login -v



import pytest
import requests
import json
def tst_list_valid_user(supply_url, userid, firstname):
    url = supply_url + "/users/" + str(userid)
    resp = requests.get(url)
    j = json.loads(resp.text)
    return([j["data"]["last_name"], j["data"]["first_name"]])

tst_list_valid_user("https://reqres.in/api", 1, "George")







# Urllib.Request and urlopen()

# read the data from the URL and print it
import urllib.request
# open a connection to a URL using urllib
webUrl  = urllib.request.urlopen('https://www.youtube.com/user/guru99com')
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





