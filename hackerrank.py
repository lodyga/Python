# 1 Say "Hello, World!" With Python
print("Hello, World!")



# 2 Python If-Else
if __name__ == '__main__':
    n = int(input().strip())
    if (n % 2 == 1) or 6 <= n <= 20:
        print("Weird")
    else:
        print("Not Weird")



# 3 Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())

    print(a + b, end="\n")
    print(a - b, end="\n")
    print(a * b)



# 4 Divison
if __name__ == '__main__':
    a = int(input())
    b = int(input())

    print('{} \n{}'.format((a//b), (a/b)))



# Loops
if __name__ == '__main__':
    n = int(input())
    print(*[i**2 for i in range(n)], sep="\n")



# Write a function
def is_leap(year):
    leap = False
    if year % 400 == 0:
        leap = True
    elif year % 4 == 0 and not year % 100 == 0:
        leap = True
    return leap
year = int(input())
print(is_leap(year))


def is_leap(year):
    return year % 4 == 0 and (year % 400 == 0 or year % 100 != 0)

year = int(input())
print(is_leap(year))



# Print Function
if __name__ == '__main__':
    n = int(input())
    print(*[i for i in range(1, n + 1)], sep="")

if __name__ == '__main__':
    n = int(input())
    print(*list(range(1, n + 1)), sep="")



# List Comprehensions
if __name__ == '__main__':
    x, y, z, n = (int(input()) for _ in range(4))
    print([[i, j, k] for i in range(x + 1) for j in range(y + 1) for k in range(z + 1) if i + j + k != n])



# Find the Runner-Up Score!
if __name__ == '__main__':
    # n = int(input()) # na co to?
    arr = map(int, input().split())
    arr = list(arr)
    # max_count = arr.count(max(arr))
    [arr.remove(max(arr)) for i in range(arr.count(max(arr)))]
    print(max(arr))

if __name__ == '__main__':
    # n = int(input())
    arr = map(int, input().split())
    arr = sorted(list(set(arr)))[-2]
    print(arr)



# Nested Lists
if __name__ == '__main__':
    nasc = {}
    nasc2 = {}
    names = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        nasc[name] = score
    for key, val in nasc.items():
        if val != min(nasc.values()):
            nasc2[key] = val
    for key, val in nasc2.items():
        if val == min(nasc2.values()):
            names.append(key)
    for name in sorted(names):
        print(name, end="\n")



# Finding the percentage
from statistics import mean as mean
if __name__ == '__main__':
    student_marks = {}
    for _ in range(int(input())):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    # print(round(mean(student_marks[query_name]), 2))
    print('{:.2f}'.format(round(mean(student_marks[query_name]), 2)))



# Lists
if __name__ == '__main__':
    lis1 = []
    for _ in range(int(input())):
        inp = input().split()
        met, arg = inp[0], inp[1:]
        if met == "print":
            print(lis1)
        else:
            arg = ", ".join(arg)
            eval("lis1." + met + "(" + arg + ")")
        '''
        arg = ", ".join(arg)
        print("lis1." + met + "(" + arg + ")")
        '''



# Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    print(hash(tuple(integer_list)))



# sWAP cASE
def swap_case(s):
    new = ""
    for inp in s:
        if inp.isdigit() == False:
            if inp.isupper():
                new += inp.lower()
            else: 
                new += inp.upper()
        else:
            new += inp
    return(new)

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)


def swap_case(s):
    return "".join([i.lower() if i.isupper() else i.upper() for i in s])

def swap_case(s):
    return "".join(map(str.swapcase, s))

def swap_case(s):
    return s.swapcase()



# String Split and Join
def split_and_join(line):
    return "-".join(line.split())

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

def split_and_join(line):
    return line.replace(" ", "-")



# What's Your Name?
def print_full_name(first, last):
    print("Hello {} {}! You just delved into python.".format(first, last))

if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)



#  Mutations
# You are given an immutable string, and you want to make changes to it.
>>> string = "abracadabra"
>>> l = list(string)
>>> l[5] = 'k'
>>> string = ''.join(l)
>>> print string
abrackdabra

>>> string = string[:5] + "k" + string[6:]
>>> print string
abrackdabra

def mutate_string(string, position, character):
    return string[:position] + character + string[position + 1:]
    # li1 = list(string)
    # li1[position] = character
    # return "".join(li1)

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)



# Find a string
def count_substring(string, sub_string):
    counter = 0
    for i in range(len(string)):
        if sub_string == string[i:i + len(sub_string)]:
            counter += 1
    return counter

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)

def count_substring(string, sub_string):
    return len([i for i in range(len(string)) if sub_string == string[i:i + len(sub_string)]])



# String Validators
if __name__ == '__main__':
    s = input()
    print(any(char.isalnum() for char in s))
    print(any(char.isalpha() for char in s))
    print(any(char.isdigit() for char in s))
    print(any(char.islower() for char in s))
    print(any(char.isupper() for char in s))
    


# Text Alignment
print 'HackerRank'.ljust(width,'-')

#Replace all ______ with rjust, ljust or center. 

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))



# Text Wrap
import textwrap
def wrap(string, max_width):
    string2 = ""
    while len(string) > max_width:
        string2 += string[:max_width] + "\n"
        string = string[max_width:]
    string2 += string
    return string2

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)



# Designer Door Mat
inp = int(input("Tylko pierwsza liczba ma znaczenie: ").split()[0])
div = inp//2
for i in range(div):
    print("-"*(div - i)*3 + "." + "|.."*i + "|" + "..|"*i + "." + 3*(div - i)*"-")
print("WELCOME".center(3*inp, "-"))
for i in range(div-1, -1, -1):
    print("-"*(div - i)*3 + "." + "|.."*i + "|" + "..|"*i + "." + 3*(div - i)*"-")

inp = int(input("Tylko pierwsza liczba ma znaczenie: ").split()[0])
div = inp//2
for i in range(div):
    print("-"*(div - i)*3 + ".|."*(2*i + 1) + 3*(div - i)*"-")
print("WELCOME".center(3*inp, "-"))
for i in range(div-1, -1, -1):
    print("-"*(div - i)*3 + ".|."*(2*i + 1) + 3*(div - i)*"-")

inp = int(input("Tylko pierwsza liczba ma znaczenie: ").split()[0])
print("\n".join([(".|."*(2*i + 1)).center(3*inp, "-") for i in range(inp//2)]))
print("WELCOME".center(3*inp, "-"))
print("\n".join([(".|."*(2*i + 1)).center(3*inp, "-") for i in range(inp//2-1, -1, -1)]))

inp = int(input("Tylko pierwsza liczba ma znaczenie: ").split()[0])
up = [(".|."*(2*i + 1)).center(3*inp, "-") for i in range(inp//2)]
middle = ["WELCOME".center(3*inp, "-")]
print("\n".join(up + middle + up[::-1]))

n, m = map(int, input().split())
pattern = [('.|.'*(2*i + 1)).center(m, '-') for i in range(n//2)]
print('\n'.join(pattern + ['WELCOME'.center(m, '-')] + pattern[::-1]))

names = ["Rick Sanchez", "Morty Smith", "Summer Smith", "Jerry Smith", "Beth Smith"]
names[::-1]



# String Formatting
def print_formatted(number):
    str_len = len(bin(number)[2:])
    for i in range(1, number + 1):
        print("{}".format(i).rjust(str_len), end=' ')
        print("{}".format(oct(i)[2:]).rjust(str_len), end=' ')
        print("{}".format(hex(i)[2:]).upper().rjust(str_len), end=' ')
        print("{}".format(bin(i)[2:]).rjust(str_len))

if __name__ == '__main__':
    n = int(input())
    print_formatted(n)


# String Formatting
def print_formatted(n):
    width = len("{:b}".format(n))
    for i in range(1, n + 1):
        print("{0:{width1}d} {0:{width1}o} {0:{width1}X} {0:{width1}b}".format(i, width1=width))

 

# Alphabet Rangoli
n = 5
char_list = [chr(c) for c in range(ord("a"), 97 + n)]
char_list_r = char_list[::-1]


for i in range(0, n):
    char_list_r[0:1 + i] + char_list[n-i:n+1])).center(((n - 1)*2 + 1) + ((n - 1)*" ".join(2), "-")
for i in range(n-2, -1, -1):
    char_list_r[0:1 + i] + char_list[n-i:n+1])).center(((n - 1)*2 + 1) + ((n - 1)*" ".join(2), "-")

char_list[2:-4:-1] + char_list[1:]


n = 5
char_list = [chr(c) for c in range(ord("a"), 97 + n)]
char_list_r = char_list[::-1]

for i in range(0, n):
    char_list[-1:n-2-i:-1] + char_list[n-i:n+1])).center(((n - 1)*2 + 1) + ((n - 1)*" ".join(2), "-")
#for i in range(n-2, -1, -1):
 char_list_r[0:1 + i] + char_list[n-i:n+1])).center(((n - 1)*2 + 1) + ((n - 1)*" ".join(2), "-")

# char_list[2:-4:-1] + char_list[1:]
#[liczy od końca indeks początku inclusive:liczy od końca indeks końca listy exclusive:-1]
char_list_r[::-1]



# Capitalize!
n = "sdfs sadf"
[i.capitalize() for i in n." ".join(split()]

s = input()
for x in s[:].split():
    s = s.replace(x, x.capitalize())
print(s)


# itertools.product()
from itertools import product
A = map(int, input().split())
B = map(int, input().split())
for i in product(A, B):
    print(i, end=' ') 

from itertools import product
A = map(int, input().split())
B = map(int, input().split())
print(*product(A, B)) 



# itertools.permutations()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from itertools import permutations
inp, num = input().split()
inp = [i for i in inp]
num = int(num)
li1 = ["".join(i) for i in list(permutations(inp, num))]
li1.sort()
for i in range(len(li1)):
    print(li1[i])


from itertools import permutations
inp, num = input().split()
print(*["".join(i) for i in permutations(sorted(inp), int(num))], sep="\n")


# itertools.combinations()
from itertools import combinations
inp, num = input().split()
for j in range(1, int(num) + 1):
    for i in combinations(sorted(inp), j):
        print("".join(i))



# itertools.combinations_with_replacement()
from itertools import combinations_with_replacement
inp, num = input().split()
print(*["".join(i) for i in list(combinations_with_replacement(sorted(inp), int(num)))], sep="\n")



# Compress the String!
from itertools import groupby
data = "1222311"
for key, group in groupby(data, lambda x: x[0]):
    print("({1}, {0})".format(key, len(list(group))), end=" ")

print(*["({}, {})".format(len(list(group)), key) for key, group in groupby(data, lambda x: x[0])])

print(*[(len(list(group)), int(key)) for key, group in groupby(data)])



# collections.Counter()
from collections Counter
myList = [1,1,2,3,4,5,3,2,3,4,2,1,2,3]
list(Counter(myList).items())[0]


# Importing defaultdict
from collections import defaultdict
lst = [('Geeks', 1), ('For', 2), ('Geeks', 3)]
orDict = defaultdict(list)
# iterating over list of tuples
for key, val in lst:
    orDict[key].append(val)
print(orDict)



from collections import Counter, defaultdict

# n = int(input())
shoe_sizes = [2, 3, 4, 5, 6, 8, 7, 6, 5, 18]
# shoe_sizes = map(int, input().split())
shoe_size_count = Counter(shoe_sizes)
print(shoe_size_count)


order_list = [[6, 55], [6, 45], [6, 55], [4, 40], [18, 60], [10, 50]]
# order_list = [list(map(int, input().split())) for _ in range(int(input()))]
order_dict = defaultdict(list)
[order_dict[key].append(val) for key, val in order_list]
print(order_dict)


money = 0
for key, val in order_list:
    # if key in shoe_size_count.keys():
    if shoe_size_count[key]:
        money += order_dict[key][0]
        del order_dict[key][0]
        shoe_size_count[key] -= 1
        if shoe_size_count[key] == 0:
            del shoe_size_count[key]
        # shoe_size_count = {x: y for x, y in shoe_size_count.items() if y!=0}

print(money)

'''
# numCust = int(input())
numCust = 6
money = 0
for i in range(numCust):
    size, price = map(int, input().split())
    if shoe_size_count[size]:
        money += price
        shoe_size_count[size] -= 1
print(money)
'''



# Introduction to Sets
def average(array):
    return sum(set(array))/len(set(array))

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)



# DefaultDict Tutorial
from collections import defaultdict
a = ['a', 'a', 'b', 'a', 'b']
b = ['a', 'b']
d = defaultdict(list)
for i in range(len(a)):
    d[a[i]].append(i)
for i in range(len(b)):
    if b[i] in d.keys():
        print(" ".join(d[b[i]])
    else:
        print(-1)


from collections import defaultdict
a = ['a', 'a', 'b', 'a', 'b']
b = ['a', 'b']
#m, n = list(map(int, input().split()))
#a = [input() for _ in range(m)]
#b = [input() for _ in range(n)]

d = defaultdict(list)
[d[a[i]].append(i + 1) for i in range(len(a))]

for i in b:
    if i in d.keys():
        print(*d[i])
    else:
        print(-1)



# Polar Coordinates
from math import sqrt
from cmath import phase
n = complex(input())
print(sqrt(n.real**2 + n.imag**2))
print(phase(n))



# Calendar Module
import calendar
day = calendar.weekday(2015, 8, 5)
days = [i for i in calendar.day_name]
months = [i for i in calendar.month_name]
# print("The day on {} {}th {} was {}.".format(months[8], "5", 2015, days[day]))
print(days[day].upper())

import calendar
list(calendar.day_name)



# Collections.namedtuple()
from collections import namedtuple
from statistics import mean
n = int(input())
nt_templet = namedtuple("some_name", input())
print(mean([int(nt_templet(*input().split()).MARKS) for i in range(n)]))

Input (stdin)
5
ID         MARKS      NAME       CLASS
1          97         Raymond    7
2          50         Steven     4
3          91         Adrian     9
4          72         Stewart    5
5          80         Peter      6



# Exceptions
for i in range(int(input())):
    try:
        m, n = map(int, input().split())
        print(m//n)
    except ZeroDivisionError as e:
        print("Error Code:", e)
    except ValueError as e:
        print("Error Code:", e) 

Input (stdin)
3
1 0
2 $
3 1

for i in range(int(input())):
    try:
        m, n = map(int, input().split())
        print(m//n)
    except BaseException as e: #Exception
        print("Error Code:", e)



# collections.OrderedDict
from collections import OrderedDict
dict1 = OrderedDict()
dict2 = OrderedDict()
for i in range(int(input())):
    inp1 = input().split()
    name, price = " ".join(inp1[:-1]), inp1[-1]
    dict1[name] = dict1.get(name, 0) + 1
    dict2[name] = price

for key, val in dict1.items():
    print(key, int(val) * int(dict2[key]))

Input (stdin)
9
BANANA FRIES 12
POTATO CHIPS 30
APPLE JUICE 10
CANDY 5
APPLE JUICE 10
CANDY 5
CANDY 5
CANDY 5
POTATO CHIPS 30

from collections import OrderedDict
dict1 = OrderedDict()
for _ in range(int(input())):
    name, space, price = input().rpartition(" ")
    dict1[name] = dict1.get(name, 0) + int(price)

[print(key, val) for key, val in dict1.items()]



# Symmetric Difference
n = input()
n = set(map(int, input().split()))
m = input()
m = set(map(int, input().split()))
print(*sorted(m.difference(n).union(n.difference(m))), sep="\n")

Input (stdin)
4
2 4 5 9
4
2 4 11 12

n, n =input(), set(map(int, input().split()))
m, m =input(), set(map(int, input().split()))
print(*sorted(m.difference(n).union(n.difference(m))), sep="\n")

n, n =input(), set(map(int, input().split()))
m, m =input(), set(map(int, input().split()))
print(*sorted(n^m), sep="\n")



# Incorrect Regex
import re
for i in range(int(input())):
    #inp = input()
    #if re.search(r"\*\+", inp):
    #    print(False)
    #else:
    #    print(True)
    ans = True
    try :
        re.compile(input())
    except: #re.error
        ans = False
    print(ans)

Input (stdin)
2
.*\+
.*+



# Set .add()
print(len({input() for i in range(int(input()))}))

Sample Input
7
UK
China
USA
France
New Zealand
UK
France 



# Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))

for _ in range(int(input())):
    inp = input().split()
    met, arg = inp[0], "".join(inp[1:])
    eval("s." + met + "(" + arg + ")")
    
print(sum(s))


Sample Input
9
1 2 3 4 5 6 7 8 9
10
pop
remove 9
discard 9
discard 8
remove 7
pop 
discard 6
remove 5
pop 
discard 5


n = int(input())
s = set(map(int, input().split()))

for _ in range(int(input())):
    eval('s.{0}({1})'.format(*input().split()+[' ']))
print(sum(s))



# Collections.deque()
from collections import deque
s = deque()

for _ in range(int(input())):
    eval('s.{0}({1})'.format(*input().split()+[' ']))
print(*s)



# Set .union() Operation
m, m = input(), set(input().split())
n, n = input(), set(input().split())
print(len(m|n))
#print(len(m.union(n)))

Sample Input
9
1 2 3 4 5 6 7 8 9
9
10 1 2 3 11 21 55 6 8



# Set .intersection() Operation
_, m, _, n = input(), set(input().split()), input(), set(input().split())
print(len(m&n))
# print(len(m.intersection(n)))

Sample Input
9
1 2 3 4 5 6 7 8 9
9
10 1 2 3 11 21 55 6 8



# Mod Divmod
m, n = (int(input()) for _ in range(2))
print(m//n, m%n, (m//n, m%n), sep="\n")

Sample Input
177
10



# Power - Mod Power
a, b, m = (int(input()) for _ in range(3))
print(a**b, pow(a, b, m), sep="\n")

Sample Input
3
4
5



# Set .difference() Operation
_, m, _, n = input(), set(input().split()), input(), set(input().split())
print(len(m-n))
# print(len(m.intersection(n)))

Sample Input
9
1 2 3 4 5 6 7 8 9
9
10 1 2 3 11 21 55 6 8



# Integers Come In All Sizes
a, b, c, d = (int(input()) for _ in range(4))
print(a**b + c**d)



# Set .symmetric_difference() Operation
_, m, _, n = input(), set(input().split()), input(), set(input().split())
print(len(m^n))
# print(len(m.symmetric_difference(n)))

Sample Input
9
1 2 3 4 5 6 7 8 9
9
10 1 2 3 11 21 55 6 8



# Set Mutations
H = set("Hacker")
R = set("Rank")
H|=R
#H.update(R)
print(H)

H&=R
# H.intersection_update(R)
print(H)

H-=R
# H.difference_update(R)
print(H)

H^=R
# H.symmetric_difference_update(R)
print(H)


_, s = input(), set(map(int, input().split()))
for _ in range(int(input())):
    met, _ = input().split()
    n = set(map(int, input().split()))
    eval("s. {}({})".format(met, n))
print(sum(s))

16
1 2 3 4 5 6 7 8 9 10 11 12 13 14 24 52
4
intersection_update 10
2 3 5 6 8 9 1 4 7 11
update 2
55 66
symmetric_difference_update 5
22 7 35 62 58
difference_update 7
11 22 35 55 58 62 66


_, s = input(), set(map(int, input().split()))
for _ in range(int(input())):
    eval("s. {0}({2})".format(*input().split(), set(map(int, input().split()))))
print(sum(s))



# The Captain's Room
from collections import Counter
_ = input()
lis1 = Counter(list(map(int, input().split())))
print(min(lis1, key=lis1.get)) # get min value form list

Sample Input
5
1 2 3 6 5 4 4 2 5 3 6 1 6 5 3 2 4 1 2 5 1 4 3 6 8 4 3 1 5 6 2 

from collections import Counter
_ = input()
num = Counter(map(int, input().split()))
print(min(num, key=num.get))
# działa w 9/10
# print(list(num.keys())[-1])



```
from collections import Counter
_ = input()
num = Counter(map(int, input().split()))
print(num.most_common()[-1][0])
```

another one 
```
print(min(num, key=num.get))
```

this works 9/10
```
print(list(num.keys())[-1])
```

# Check Subset
for _ in range(int(input())):
    _, m, _, n = input(), set(input()), input(), set(input())
    print(m.issubset(n))

Sample Input
3
5
1 2 3 5 6
9
9 8 5 6 3 2 1 4 7
1
2
5
3 6 5 4 1
7
1 2 3 5 6 8 9
3
9 8 2



# Check Strict Superset

{"1", "2", "3"} > {"1", "2"}

m = set(input().split())
i = 0
ran = int(input())
for _ in range(ran):
    n = set(input().split())
    if n.issubset(m) and n != m:
        i += 1
print(i == ran)

Sample Input
1 2 3 4 5 6 7 8 9 10 11 12 23 45 84 78
2
1 2 3 4 5
100 11 12

m = set(input().split())
print(all(list([m > set(input().split()) for _ in range(int(input()))])))



# Zipped!
A = [1,2,3]
B = [6,5,4]
print([A] + [B])
print(list(zip(*([A] + [B]))))


from statistics import mean
_, leng = input().split()
m = zip(*[list(map(float, input().split())) for i in range(int(leng))])
n = [mean(i) for i in m]
print(*n, sep="\n")


Sample Input
5 3
89 90 78 93 80
90 91 85 88 86  
91 92 83 89 90.5


from statistics import mean
_, leng = input().split()
m = [map(float, input().split()) for i in range(int(leng))]
n = [mean(i) for i in zip(*m)]
print(*n, sep="\n")



# Input()
x, y = input().split()
print(eval(input().replace("x", x)) == int(y))

Sample Input
1 4
x**3 + x**2 + x + 1

x, y = map(int, input().split())
print(eval(input()) == y)

str1 = "A**2"
A = int(5)
print(eval(str1))



# Python Evaluation
eval(input())



# Any or All
_ = input()
n = list(map(int, input().split()))
zero = not any([True for i in n if i < 0])
pali = False
if zero:
    pali = any([True for i in n if str(i)[0] == str(i)[-1]])
print(all([zero, pali]))


5
12 9 61 5 14 



_ = input()
n = list(map(int, input().split()))
print(any([str(i)[0] == str(i)[-1] for i in n]) and all([i >= 0 for i in n]))


_ = input()
n = input().split()
print(any([i[0] == i[-1] for i in n]) and all([int(i) >= 0 for i in n]))



# Detect Floating Point Number
import re
print(*[bool(re.match(r"^[+-]?\d*\.\d*$", input())) for _ in range(int(input()))], sep="\n")

4
4.0O0
-1.00
+4.54
SomeRandomStuff



# Map and Lambda Function
cube = lambda x: x**3

def fibonacci(n):
    n1, n2 = 0, 1
    fib_lilst = [0, 1]
    count = 0
    
    while count < n:
       nth = n1 + n2
       fib_lilst.append(nth)
       # update values
       n1 = n2
       n2 = nth
       count += 1
    return fib_lilst[:n]

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))

Sample input
5


cube = lambda x: x**3

def fibonacci(n):
    fib_lilst = [0, 1][:n]
    
    if n > 2:
        for i in range(n-2):
            fib_lilst.append(fib_lilst[i] + fib_lilst[i + 1])
    return fib_lilst

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))



def fibonacci(n):
    a,b = 0,1
    for i in range(n):
        yield a
        a,b = b,a+b



# Re.split()
regex_pattern = r"[,.]"

import re
print("\n".join(re.split(regex_pattern, input())))


Sample Input 0
100,000,000.000

regex_pattern = r"[\,\.]"
regex_pattern = r"[\D+]"



# Group(), Groups() & Groupdict() 
print(re.match(r"\bS\w+", "Spain The rain in Spain"))

import re
print(re.search(r"\b(\d)\1+\b", "12345678910111213141516171820212223")

sdfsdf
12345671189101213141516171820212223

11