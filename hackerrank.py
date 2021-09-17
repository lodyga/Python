#_Introduction
# 1 Say "Hello, World!" With Python
print("Hello, World!")



# 2 Python If-Else
if __name__ == '__main__':
    n = int(input().strip())
    if (n % 2 == 1) or 6 <= n <= 20:
        print("Weird")
    else:
        print("Not Weird")

Input (stdin)
3


# 3 Arithmetic Operators
if __name__ == '__main__':
    a = int(input().strip())
    b = int(input().strip())

    print(a + b, end="\n")
    print(a - b, end="\n")
    print(a * b)

Input (stdin)
4
3



# 4 Divison
if __name__ == '__main__':
    a = int(input().strip())
    b = int(input().strip())

    print('{} \n{:.2f}'.format((a//b), (a/b)))

Input (stdin)
4
3

if __name__ == '__main__':
    a = int(input().strip())
    b = int(input().strip())

    print('%d \n%f' % (a//b, a/b))



# 5 Loops
if __name__=='__main__':
    n = int(input().strip())
    for i in range(n):
        print(i**2)

Input (stdin)
5

if __name__ == '__main__':
    n = int(input().strip())
    print(*[i**2 for i in range(n)], sep='\n')



# 6 Write a function
def is_leap(year):
    leap = False
    if year % 400 == 0:
        leap = True
    elif year % 4 == 0 and not year % 100 == 0:
        leap = True
    return leap
year = int(input().strip())
print(is_leap(year))

Input (stdin)
1990

def is_leap(year):
    return year % 4 == 0 and (year % 400 == 0 or year % 100 != 0)
year = int(input().strip())
print(is_leap(year))



# 7 Print Function
if __name__ == '__main__':
    n = int(input().strip())
    print(*[i for i in range(1, n + 1)], sep="")

if __name__ == '__main__':
    n = int(input().strip())
    print(*list(range(1, n + 1)), sep="")

if __name__== '__main__':
    n = int(input().strip())
    for i in range(1, n + 1):
        print(i, end='')
    print('\n')

Sample Input 0
3

##Introduction






# Basic Data Types
# 8 List Comprehensions
if __name__ == '__main__':
    x, y, z, n = (int(input().strip()) for _ in range(4))
    print([[i, j, k] 
        for i in range(x + 1) 
        for j in range(y + 1)
        for k in range(z + 1) if i + j + k != n])

Sample Input 0
1
1
1
2



# 9 Find the Runner-Up Score!
if __name__ == '__main__':
    # n = int(input())
    arr = map(int, input().split())
    arr = sorted(list(set(arr)), reverse=True)[1]
    print(arr)

Sample Input 0
5
2 3 6 6 5

if __name__ == '__main__':
    # n = int(input().strip()))
    arr = map(int, input().split())
    arr = list(arr)
    # max_count = arr.count(max(arr))
    [arr.remove(max(arr)) for i in range(arr.count(max(arr)))]
    print(max(arr))



# 10 Nested Lists
if __name__ == '__main__':
    nasc = {}
    nasc2 = {}
    names = []

    # zbiera dane
    for _ in range(int(input())):
        name = input()
        score = float(input())
        nasc[name] = score

    # wybiera drugi val
    min_val = sorted(list(set(nasc.values())))[1]

    # bierze imiona po val
    for key, val in nasc.items():
        if val == min_val:
            nasc2[key] = val

    # sort imion
    print(*sorted(nasc2.keys()), sep='\n')

Sample Input 0
5
Harry
37.21
Berry
37.21
Tina
37.2
Akriti
41
Harsh
39

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
    print(*sorted(names), sep='\n')

# to niżej nie działa, ale po coś było
if __name__ == '__main__':
    nasc = {}
    nasc2 = {}
    names = []
    
    for _ in range(int(input())):
        name = input()
        score = float(input())
        nasc[name] = score
    
    # sortuje dict po value
    sort_val = {k: v for k, v in sorted(nasc.items(), key=lambda item: item[1])}



# 11 Finding the percentage
from statistics import mean as mean

from numpy.lib.function_base import append
if __name__ == '__main__':
    # n = int(input().strip())
    student_marks = {}
    for _ in range(int(input().strip())):
        name, *line = input().strip().split()
        student_marks[name] = tuple(map(float, line))
    query_name = input().strip()
    # print('{:.2f}'.format(round(mean(stuldent_marks[query_name]), 2)))
    print('%.2f' % mean(student_marks[query_name]))

Sample Input 0
3
Krishna 67 68 69
Arjun 70 98 63
Malika 52 56 60
Malika

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



# 12 Lists
if __name__ == '__main__':
    lis1 = []
    for _ in range(int(input().strip())):
        #inp = input().split()
        #met, arg = inp[0], inp[1:]
        met, *arg = input().split()
        if met == "print":
            print(lis1)
        else:
            arg = ", ".join(arg)
            eval("lis1." + met + "(" + arg + ")")



# 13 Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    print(hash(tuple(integer_list)))

# Basic Data Types



#!/bin/python3

import math
import os
import random
import re
import sys



class Car:
    max_speed = 0
    speed_unit = 0
    def __init__(self, max_speed, speed_unit):
        self.max_speed=max_speed
        self.speed_unit=speed_unit
        
    def some(self):
        print(self.max_speed)

class Boat:
    max_speed = 0
    def __init__(self, max_speed):
        self.max_speed=max_speed
        
    def some(self):
        return self.max_speed

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    q = int(input())
    queries = []
    for _ in range(q):
        args = input().split()
        vehicle_type, params = args[0], args[1:]
        if vehicle_type == "car":
            max_speed, speed_unit = int(params[0]), params[1]
            vehicle = Car(max_speed, speed_unit)
        elif vehicle_type == "boat":
            max_speed = int(params[0])
            vehicle = Boat(max_speed)
        else:
            raise ValueError("invalid vehicle type")
        fptr.write("%s\n" % vehicle)
    fptr.close()

Input (stdin)
2
car 151 km/h
boat 77



def transformSentence(sentence):
    new = ''
    for i in sentence.split(' '):
        for j in range(len(i)):
            if j == 0:
                new += i[j]
            else:
                if i[j - 1] <= i[j]:
                    new += i[j].upper()
                else:
                    new += i[j].lower()
        new += ' '
    return new[:-1]

transformSentence('a Blue MOON')

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    sentence = input()

    result = transformSentence(sentence)

    fptr.write(result + '\n')

    fptr.close()

'a Blue MOON'

'a' < 'b'



# Strings
# 14 sWAP cASE
def swap_case(s):
    return s.swapcase()

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)

Sample Input 0
HackerRank.com presents "Pythonist 2".

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

def swap_case(s):
    return "".join([i.lower() if i.isupper() else i.upper() for i in s])

def swap_case(s):
    return "".join(map(str.swapcase, s))



# 15 String Split and Join
def split_and_join(line):
    return "-".join(line.split())

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

def split_and_join(line):
    return line.replace(" ", "-")



# 16 What's Your Name?
def print_full_name(first, last):
    print('Hello %s %s! You just delved into python.' % (first, last))
    # print("Hello {} {}! You just delved into python.".format(first, last))

if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)



# 17 Mutations
# You are given an immutable string, and you want to make changes to it.
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

Sample Input
STDIN           Function
-----           --------
abracadabra     s = 'abracadabra'
5 k             position = 5, character = 'k'



# 18 Find a string
def count_substring(string, sub_string):
    counter = 0
    for i in range(len(string) + 1 - len(sub_string)):
        if sub_string == string[i:i + len(sub_string)]:
            counter += 1
    return counter

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)

Sample Input
ABCDCDC
CDC

def count_substring(string, sub_string):
    return len([i for i in range(len(string) + 1 - len(sub_string)) if sub_string == string[i:i + len(sub_string)]])



# 19 String Validators
if __name__ == '__main__':
    s = input()
    print(any(char.isalnum() for char in s))
    print(any(char.isalpha() for char in s))
    print(any(char.isdigit() for char in s))
    print(any(char.islower() for char in s))
    print(any([char.isupper() for char in s]))

Sample Input
qA2



# 20 Text Alignment
print 'HackerRank'.ljust(width,'-')

#Replace all ______ with rjust, ljust or center. 


if __name__ == '__main__':
    thickness = int(input()) #This must be an odd number
    c = 'H'

    #Top Cone
    for i in range(thickness):
        print((i*'A').rjust(thickness-1)+'B'+('C'*i).ljust(thickness-1))

    #Top Pillars
    for i in range(thickness+1):
        print(('D'*thickness).center(thickness*2)+('E'*thickness).center(thickness*6))

    #Middle Belt
    for i in range((thickness+1)//2):
        print(('F'*thickness*5).center(thickness*6))    

    #Bottom Pillars
    for i in range(thickness+1):
        print(('G'*thickness).center(thickness*2)+('H'*thickness).center(thickness*6))    

    #Bottom Cone
    #for i in range(thickness):
    #    print((('I'*(thickness-i-1)).rjust(thickness)+'J'+('K'*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

    for i in reversed(range(thickness)):
        print((('I'*i).rjust(thickness)+'J'+('K'*i).ljust(thickness)).rjust(thickness*6))

Sample Input
5



# 21 Text Wrap
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

Sample Input 0
ABCDEFGHIJKLIMNOQRSTUVWXYZ
4



# 22 Designer Door Mat
inp = int(input("Tylko pierwsza liczba ma znaczenie: ").split()[0])
div = inp//2
for i in range(div):
    print("-"*(div - i)*3 + "." + "|.."*i + "|" + "..|"*i + "." + 3*(div - i)*"-")
print("WELCOME".center(3*inp, "-"))
for i in range(div-1, -1, -1):
    print("-"*(div - i)*3 + "." + "|.."*i + "|" + "..|"*i + "." + 3*(div - i)*"-")

Sample Input
9 27

inp = int(input("Tylko pierwsza liczba ma znaczenie: ").split()[0])
div = inp//2
for i in range(div):
    print("-"*(div - i)*3 + ".|."*(2*i + 1) + 3*(div - i)*"-")
print("WELCOME".center(3*inp, "-"))
for i in range(div-1, -1, -1):
    print("-"*(div - i)*3 + ".|."*(2*i + 1) + 3*(div - i)*"-")

inp = int(input("Tylko pierwsza liczba ma znaczenie: ").split()[0])
print("\n".join((".|."*(2*i + 1)).center(3*inp, "-") for i in range(inp//2)))
print("WELCOME".center(3*inp, "-"))
print("\n".join((".|."*(2*i + 1)).center(3*inp, "-") for i in range(inp//2-1, -1, -1)))

inp = int(input("Tylko pierwsza liczba ma znaczenie: ").split()[0])
up = [(".|."*(2*i + 1)).center(3*inp, "-") for i in range(inp//2)]
middle = ["WELCOME".center(3*inp, "-")]
print("\n".join(up + middle + up[::-1]))

n, m = map(int, input().split())
pattern = [('.|.'*(2*i + 1)).center(m, '-') for i in range(n//2)]
print('\n'.join(pattern + ['WELCOME'.center(m, '-')] + pattern[::-1]))

names = ["Rick Sanchez", "Morty Smith", "Summer Smith", "Jerry Smith", "Beth Smith"]
names[::-1]
names.reverse()



# 23 String Formatting
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

Sample Input
17

def print_formatted(n):
    width = len("{:b}".format(n))
    for i in range(1, n + 1):
        print("{0:{width1}d} {0:{width1}o} {0:{width1}X} {0:{width1}b}".format(i, width1=width))

 

# 24 Alphabet Rangoli
def print_rangoli(size):
    n = size
    char_list = [chr(c) for c in range(ord("a"), 97 + n)]
    char_list_r = char_list[::-1]
    
    for i in range(0, n):
        print("{}".format("-".join(char_list_r[0:1 + i] + char_list[n-i:n+1])).center(((n - 1)*2 + 1) + ((n - 1)*2), "-"))
    for i in range(n-2, -1, -1):
        print("{}".format("-".join(char_list_r[0:1 + i] + char_list[n-i:n+1])).center(((n - 1)*2 + 1) + ((n - 1)*2), "-"))

if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)


def print_rangoli(size):
    n = size
    char_list = [chr(c) for c in range(ord("a"), 97 + n)]
    char_list_r = char_list[::-1]
    # pattern = [("-".join(char_list_r[0:1 + i] + char_list[n-i:n+1])).center(((n - 1)*2 + 1) + ((n - 1)*2), "-") for i in range(0, n)]
    pattern = [("-".join(char_list_r[0:1 + i] + char_list[n-i:n+1])).center((n - 1)*4 + 1, "-") for i in range(0, n)]
    # print('\n'.join(pattern + pattern[-2::-1]))
    print('\n'.join(pattern[:-1] + pattern[::-1]))

Sample Input
5


ord('a')
# char_list[2:-4:-1] + char_list[1:]
#[liczy od końca indeks początku inclusive:liczy od końca indeks końca listy exclusive:-1]
char_list_r[::-1]



# 25 Capitalize!

# pomija spacje dlatego nie przechodzi testów
def solve(s):
    return " ".join([i.capitalize() for i in s.split()])

solve("chirs   alan")

Sample Input
chris alan

def solve(s):
    # for x in s[:].split():
    for x in s.split():
        s = s.replace(x, x.capitalize())
    print(s)



# 26 Merge the Tools!
def merge_the_tools(string, k):
    for i in range(len(string)//k):
        str = string[k*i:k*(i + 1)]
        str2 = ''
        for st in str:
            if st not in str2:
                str2 += st
        print(str2)

merge_the_tools('AABCAAADA', 3)

Sample Input
STDIN       Function
-----       --------
AABCAAADA   s = 'AABCAAADA'
3           k = 3

# kolejność nie liczy się
''.join(set('AACVGVV'))

# String







# Sets
# 27 Introduction to Sets
def average(array):
    return sum(set(array))/len(set(array))

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)

Sample Input
STDIN                                       Function
-----                                       --------
10                                          arr[] size N = 10
161 182 161 154 176 170 167 171 170 174     arr = [161, 181, ..., 174]

from statistics import mean
def average(array):
    return mean(array)



# 28 No Idea!
if __name__ == '__main__':
    _, _ = input().split()
    li1, se1, se2 = input().split(), set(input().split()), set(input().split())
    print(sum(((i in se1) - (i in se2)) for i in li1))
    #print((sum(1 for i in li1 if (i in se1))) + (sum(-1 for i in li1 if (i in se2))))

Sample Input
3 2
1 5 3
3 1
5 7



# 29 Symmetric Difference
if __name__ == '__main__':
    _, n =input(), set(map(int, input().split()))
    _, m =input(), set(map(int, input().split()))
    #print(*sorted(m.difference(n).union(n.difference(m))), sep="\n")
    print(*sorted(n^m), sep="\n")

Input (stdin)
4
2 4 5 9
4
2 4 11 12



# 30 Set .add()
print(len({input().strip() for i in range(int(input().strip()))}))

Sample Input
7
UK
China
USA
France
New Zealand
UK
France 


if __name__ == '__main__':
    set1 = set('')
    for i in range(int(input().strip())):
        set1.add(input().strip())
    print(set1)



# 31 Set .discard(), .remove() & .pop()
if __name__ == '__main__':
    _ = int(input())
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

_ = int(input())
s = set(map(int, input().split()))

for _ in range(int(input())):
    eval('s.{}({})'.format(*input().split()+[' ']))
print(sum(s))



# 32 Set .union() Operation
_, m = input(), set(input().split())
_, n = input(), set(input().split())
print(len(m|n))
#print(len(m.union(n)))

Sample Input
9
1 2 3 4 5 6 7 8 9
9
10 1 2 3 11 21 55 6 8



# 33 Set .intersection() Operation
_, m, _, n = input(), set(input().split()), input(), set(input().split())
print(len(m&n))
# print(len(m.intersection(n)))

Sample Input
9
1 2 3 4 5 6 7 8 9
9
10 1 2 3 11 21 55 6 8



# 34 Set .difference() Operation
_, m, _, n = input(), set(input().split()), input(), set(input().split())
print(len(m-n))
# print(len(m.intersection(n)))

Sample Input
9
1 2 3 4 5 6 7 8 9
9
10 1 2 3 11 21 55 6 8



# 35 Set .symmetric_difference() Operation
_, m, _, n = input(), set(input().split()), input(), set(input().split())
print(len(m^n))
# print(len(m.symmetric_difference(n)))

Sample Input
9
1 2 3 4 5 6 7 8 9
9
10 1 2 3 11 21 55 6 8



# 36 Set Mutations
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



# 37 The Captain's Room

from collections import Counter
if __name__ == '__main__':
    _ = input()
    num = Counter(map(int, input().split()))
    # print(num.most_common()[-1][0])
    # print(list(num.keys())[-1]) 
    print(min(num, key=num.get)) # wibiera minimalny key

Sample Input
5
1 2 3 6 5 4 4 2 5 3 6 1 6 5 3 2 4 1 2 5 1 4 3 6 8 4 3 1 5 6 2 

works 9/10
print(list(num.keys())[-1])



# 38 Check Subset
if __name__ == '__main__':
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



# 39 Check Strict Superset
{"1", "2", "3"} > {"1", "2"}

if __name__ == '__main__':
    m = set(input().split())
    i = 0
    counter = int(input())
    for _ in range(counter):
        n = set(input().split())
        if n.issubset(m) and n != m:
            i += 1
    print(i == counter)

Sample Input
1 2 3 4 5 6 7 8 9 10 11 12 23 45 84 78
2
1 2 3 4 5
100 11 12

m = set(input().split())
print(all([m > set(input().split()) for _ in range(int(input().strip()))]))


# Sets








# Math
# 40 Polar Coordinates
from math import sqrt
from cmath import phase
if __name__ == '__main__':
    n = complex(input())
    print(sqrt(n.real**2 + n.imag**2))
    print(phase(n))
 
Sample Input
1+2j

from cmath import polar
print(*polar(complex(input())), sep="\n")



# 41 Mod Divmod
if __name__ == '__main__':
    m, n = (int(input().strip()) for _ in range(2))
    print(m//n, m%n, divmod(m, n), sep="\n")

Sample Input
177
10



# 42 Power - Mod Power
if __name__ == '__main__':
    a, b, m = (int(input().strip()) for _ in range(3))
    print(a**b, pow(a, b, m), sep="\n")

Sample Input
3
4
5


# 43 Integers Come In All Sizes
if __name__ == '__main__':
    a, b, c, d = (int(input()) for _ in range(4))
    print(a**b + c**d)

Sample Input
9
29
7
27



# 44 Find Angle MBC
import math as m
if __name__ == '__main__':
    a, b = int(input().strip()), int(input().strip())
    print(str(round(m.degrees(m.asin(a/m.sqrt(a**2 + b**2))))) + "\u00b0")

Sample Input
10
10



# 45 Triangle Quest
print(*[(i*10**i)//9 for i in range(1, int(input().strip()))], sep='\n')

Sample Input
5

# to zailcza test
for i in range(1, int(input())):
    print((i*10**i)//9)



# 46 Triangle Quest 2
print(*[((10**i)//9)**2 for i in range(1, int(input().strip()) + 1)], sep='\n')

Sample Input
5

for i in range(1, int(input())+1):
    print(*(range(1, i)), *(range(i, 0, -1)), sep="")

for i in range(1, int(input())+1):
    print("".join([str(j) for j in range(1, i)]) +
          "".join([str(j) for j in range(i, 0, -1)]))

[print("".join([str(j) for j in range(1, i)]) + "".join([str(j)
       for j in range(i, 0, -1)])) for i in range(1, int(input())+1)]


# Math







#Itertools
# 47 itertools.product()
from itertools import product
if __name__ == '__main__':
    A = map(int, input().strip().split())
    B = map(int, input().strip().split())
    print(*product(A, B))

Sample Input
1 2
3 4

from itertools import product
A = map(int, input().strip().split())
B = map(int, input().strip().split())
for i in product(A, B):
    print(i, end=' ') 



# 48 itertools.permutations()
from itertools import permutations
inp, num = input().split()
print(*["".join(i) for i in permutations(sorted(inp), int(num))], sep="\n")

inp, num = input().split()
print(*[''.join(i) for i in sorted(permutations(inp, int(num)))], sep='\n')

Sample Input
HACK 2

from itertools import permutations
inp, num = input().split()
inp = [i for i in inp]
num = int(num)
li1 = ["".join(i) for i in list(permutations(inp, num))]
li1.sort()
for i in range(len(li1)):
    print(li1[i])



# 49 itertools.combinations()
from itertools import combinations
inp, num = input().split()
for j in range(1, int(num) + 1):
    for i in combinations(sorted(inp), j):
        print("".join(i))
    # print(*["".join(i) for i in combinations(sorted(inp), j)], sep="\n")

Sample Input
HACK 2

list_of_list = [[''.join(i) for i in (combinations(sorted(inp), j))] for j in range(1, int(num) + 1)]
print(*[val for sublist in list_of_list for val in sublist], sep='\n')



# 50 itertools.combinations_with_replacement()
from itertools import combinations_with_replacement
inp, num = input().split()
print(*["".join(i) for i in list(combinations_with_replacement(sorted(inp), int(num)))], sep="\n")

Sample Input
HACK 2



# 51 Compress the String!
from itertools import groupby

print(*((len(list(group)), int(key)) for key, group in groupby(str(input().strip()))))

Sample Input
1222311

for key, group in groupby(str(input().strip())):
    print("({1}, {0})".format(key, len(list(group))), end=" ")

print(*["({}, {})".format(len(list(group)), key) for key, group in groupby(str(input().strip()))])



# 52 Iterables and Iterators
from itertools import combinations
from statistics import mean
_, letters, k = int(input()), "".join(list(input().split())), int(input())
print(mean([True if 'a' in i else False for i in  combinations(letters, k)]))

Sample Input
4 
a a c d
2

print(mean([True if 'a' in ''.join(i) else False for i in  combinations(letters, k)]))



# 52 Maximize It!
from itertools import product
K, M = map(int, input().split())
li = [list(map(int, input().split()))[1:] for _ in range(K)]
print(max(map(lambda i: sum(j**2 for j in i)%M, list(product(*li)))))

Sample Input
3 1000
2 5 4
3 7 8 9 
5 5 7 8 9 10 

list(product(*li))
list(combinations(li[1], 2))

#Itertools








# Collections
# 53 DefaultDict Tutorial
from collections import defaultdict
lst = [('Geeks', 1), ('For', 2), ('Geeks', 3)]
dict(lst)
orDict = defaultdict(list)
# iterating over list of tuples
for key, val in lst:
    orDict[key].append(val)
print(orDict)

from collections import Counter
myList = [1, 1, 2, 3, 4, 5, 3, 2, 3, 4, 2, 1, 2, 3]
list(Counter(myList).items())
list(Counter(myList).keys())
list(Counter(myList).values())


from collections import defaultdict
if __name__ == '__main__':
    groupA = ['a', 'a', 'b', 'a', 'b']
    groupB = ['a', 'b']
    d = defaultdict(list)

    [d[groupA[i]].append(i + 1) for i in range(len(groupA))]

    '''starsza wersja
    for i in range(len(groupA)):
        d[groupA[i]].append(i + 1)
    '''

    for i in groupB:
        if i in d.keys():
            print(*d[i])
        else:
            print(-1)
    '''
    to dobrze liczy, ale za dużo nawiasów
    print(*[d[i] if i in d.keys() else -1 for i in groupB], sep='\n')
    '''

    ''' starsza wersja
    for i in range(len(groupB)):
        if groupB[i] in d.keys():
            print(*d[groupB[i]])
        else:
            print(-1)
    '''



# 54 collections.Counter()
from collections import Counter, defaultdict
# n = int(input())
shoe_sizes = [2, 3, 4, 5, 6, 8, 7, 6, 5, 18]
# shoe_sizes = map(int, input().split())
shoe_size_count = Counter(shoe_sizes)

# standardowy dict sypałby się przy zapytaniu o key którego nie ma
shoe_size_count_2 = dict(shoe_size_count)
shoe_size_count_2[10]

# numCust = int(input())
if __name__ == '__main__':
    numCust = 6
    money = 0
    for i in range(numCust):
        size, price = map(int, input().split())
        if shoe_size_count[size]:
            shoe_size_count[size] -= 1
            money += price
    print(money)

Sample Input
10
2 3 4 5 6 8 7 6 5 18
6
6 55
6 45
6 55
4 40
18 60
10 50


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




# 55 Collections.namedtuple()
# tutor
from collections import namedtuple
Car = namedtuple('Car', 'Price Mileage Colour Class')
# xyz = Car(Colour='Cyan', Class='Y', Price=100000, Mileage=30)
xyz = Car(99999, 40, 'Megenta', 'X')
print(xyz)
print(xyz.Colour)


from collections import namedtuple
from statistics import mean
n = int(input())
nt_templet = namedtuple("some_name", input())
print(mean(int(nt_templet(*input().split()).MARKS) for i in range(n)))

Input (stdin)
5
ID         MARKS      NAME       CLASS
1          97         Raymond    7
2          50         Steven     4
3          91         Adrian     9
4          72         Stewart    5
5          80         Peter      6



# 56 collections.OrderedDict
# tutor
from collections import OrderedDict
ordinary_dictionary = {}
ordinary_dictionary['b'] = 2
ordinary_dictionary['a'] = 1
print(ordinary_dictionary)
ordinary_dictionary.get('a')
ordered_dictionary = OrderedDict()
ordered_dictionary['b'] = 2
ordered_dictionary['a'] = 1
print(ordered_dictionary)
ordered_dictionary.get('a')

from collections import OrderedDict
dict1 = OrderedDict()
for _ in range(int(input())):
    name, _, price = input().rpartition(" ")
    # inne sposoby dzielenia stringa
    # *name, price = input().strip().split()
    # name = ' '.join(name)
    #
    # inp1 = input().split()
    # name, price = " ".join(inp1[:-1]), inp1[-1]
    dict1[name] = dict1.get(name, 0) + int(price)

print(*[' '.join([key, str(val)]) for key, val in dict1.items()], sep='\n')
# [print(key, val) for key, val in dict1.items()]

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



# 57 Collections.deque()
from collections import deque

s = deque()
for _ in range(int(input())):
    # eval('s.{0}({1})'.format(*input().split()+[' ']))
    met, *arg = input().split()
    arg = ", ".join(arg)
    eval("s." + met + "(" + arg + ")")
    #print(arg)
print(*s)

Sample Input
6
append 1
append 2
append 3
appendleft 4
pop
popleft



# 58 Word Order
from collections import OrderedDict
dict1 = OrderedDict()
for _ in range(int(input())):
    name = input().strip()
    dict1[name] = dict1.get(name, 0) + 1
print(len(dict1))
#print(*[j for i, j in dict1.items()], sep=' ')
print(*list(dict1.values()), sep=' ')


Sample Input
4
bcdef
abcdefg
bcde
bcdef


from collections import Counter, OrderedDict
class OrderedCounter(Counter, OrderedDict):
    pass
d = OrderedCounter(input() for _ in range(int(input())))
print(len(d))
print(*d.values())



# 59 Company Logo
import math
import os
import random
import re
import sys
from collections import Counter, OrderedDict
import itertools

if __name__ == '__main__':
    dict1 = OrderedDict()
    for i in input():
        dict1[i] = dict1.get(i, 0) + 1
    #sort_val = {k: v for k, v in sorted(dict1.items(), key=lambda item: item[1],reverse=True)}
    #for key, val in dict(itertools.islice(sort_val.items(), 3)).items():
    #    print(key, val)
    # sort OrderedDict
    dict2 = sorted(dict1, key = dict1.get, reverse = True)
    for i in dict2[:3]:
        print(i, dict1[i])
    # print(*[' '.join([i, str(dict1[i])]) for i in dict2[:3]], sep='\n')

Sample Input 0
aabbbccde

if __name__ == '__main__':
    dict1 = OrderedDict()
    for i in input():
        dict1[i] = dict1.get(i, 0) + 1
    #sort_val = {k: v for k, v in sorted(dict1.items(), key=lambda item: item[1],reverse=True)}
    #for key, val in dict(itertools.islice(sort_val.items(), 3)).items():
    #    print(key, val)
    # sort OrderedDict
    dict2 = sorted(dict1, key = dict1.get, reverse = True)
    for i in dict2[:3]:
        print(i, dict1[i])
    # print(*[' '.join([i, str(dict1[i])]) for i in dict2[:3]], sep='\n')
    print(type(dict2))
    print(list(dict2))
    
    # ta metoda faktycznie sortuje OrderedDict, bo poprzednia sortuje OrderedDict jako listę
    # nie przechodzi wszystkich testów
    dict2 = OrderedDict(sorted(dict1.items(), key=lambda t: t[1], reverse = True))
    pr = 0
    for key, val in dict2.items():
        if pr < 3:
            pr += 1
            print(key, val)
        else:
            break
    print(type(dict2))
    


import math
import os
import random
import re
import sys
from collections import Counter, OrderedDict


if __name__ == '__main__':
    class OrderedCounter(Counter, OrderedDict):
        pass

    [print(*i) for i in OrderedCounter(sorted(input())).most_common(3)]

# Collections






# Date and Time
# 60 Calendar Module
import calendar
day = calendar.weekday(2015, 8, 5)
days = [i for i in calendar.day_name]
months = [i for i in calendar.month_name]
# print("The day on {} {}th {} was {}.".format(months[8], "5", 2015, days[day]))
print(days[day].upper())

import calendar
list(calendar.day_name)







# Errors and Exceptions
# 61 Exceptions
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



# 62 Incorrect Regem
import regex
for i in range(int(input())):
    #inp = input()
    #if re.search(r"\*\+", inp):
    #    print(False)
    #else:
    #    print(True)
    ans = True
    try:
        re.compile(input())
    except:  # re.error
        ans = False
    print(ans)

Input (stdin)
2
.*\+
.*+






# Built-Ins
# 62 Zipped!
A = [1, 2, 3]
B = [6, 5, 4]
print([A] + [B])
print(list(zip(*([A] + [B]))))


from statistics import mean
_, leng = input().split()
m = [map(float, input().split()) for i in range(int(leng))]
print(*[mean(i) for i in zip(*m)], sep='\n')



Sample Input
5 3
89 90 78 93 80
90 91 85 88 86  
91 92 83 89 90.5



# 63 Input()
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



# 64 Python Evaluation
eval(input())



# 65 Any or All
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
print(any(i[0] == i[-1] for i in n) and all(int(i) >= 0 for i in n))



# 66 Athlete Sort
import numpy as np
# array sort
a = np.array([[8, 2, -2], [-4, 1, 7], [6, 3, 9]])
print(a)
print(a[a[:, 1].argsort()])
print(a[np.argsort(a[:, 1])])
print(a[:, 1].argsort())
print(np.argsort(a[:, 1]))


https://docs.python.org/3/howto/sorting.html
student_tuples = [
    ('john', 'A', 15),
    ('jane', 'B', 12),
    ('dave', 'B', 10),
]
sorted(student_tuples, key=lambda student: student[2])

n, m = map(int, input().split())
arr = [list(map(int, input().split())) for _ in range(n)]
k = int(input())
for i in sorted(arr, key=lambda student: student[k]):
    print(*i)



# 67 ginortS
# sort string with characters and numbers
str1 = 'Sorting1234'
# w tej lambdzie wybiera x.foo od końca i wstawia w nowego stringa od pdczątku
print(*sorted(str1 , key=lambda x: (x.isdigit() and int(x)%2==0, x.isdigit(), x.isupper(), x.islower(), x)), sep='')
print(*sorted(input(), key=lambda x: (x.isdigit(), x.isdigit() and int(x)%2==0, x.isupper(), x.islower(), x)), sep='')
dir(str)


# Built-Ins







# Python Functionals
# 68 Map and Lambda Function
# The map() function applies a function to every member of an iterable and returns the result.
# Lambda is a single expression anonymous function often used as an inline function. 
# In simple words, it is a function that has only one line in its body.

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










# Regex and Parsing
# 69 Detect Floating Point Number
import re
print(*[bool(re.match(r"^[+-]?\d*\.\d*$", input())) for _ in range(int(input()))], sep="\n")

4
4.0O0
-1.00
+4.54
SomeRandomStuff

print([(re.match(r"^[+-]?\d*\.\d*$", input())) for _ in range(int(input()))])



# 70 Re.split()
regex_pattern = r"[,.]"

import re
print("\n".join(re.split(regex_pattern, input())))


Sample Input 0
100,000,000.000

regex_pattern = r"[\D+]"



# 71 Group(), Groups() & Groupdict() 
import re
m = re.search(r"(\w(?!_))\1+", input())
print(m.group(1) if m else -1)

Input (stdin)
12345678910111213141516171820212223

r"(\d(?!_))\1+"

# _ matches the literal '_' character

m = re.search(r"(\w(?!_))\1+", input())
print(m.group())

m2 = re.match(r"(\w(?!_))\1+", input())
print(m.group(1))



# 72 Re.findall() & Re.finditer() do poprawy
import re
m = re.findall(r'(?<=[bcdfghjklmnpqrstvwxys])[aeuioAEUIO]{2,}(?=[bcdfghjklmnpqrstvwxys])', input(), re.I)
print(*m, sep='\n') if m else print(-1)


Input (stdin)
rabcdeefgyYhFjkIoomnpOeorteeeeet

import re
con = "[bcdfghjklmnpqrstvwxys]"
m = re.findall(r'(?<=' + con + ')[aeuioAEUIO]{2,}(?=' + con + ')', input(), re.I)
print(*m, sep='\n') if m else print(-1)

m = re.search(r'(?<=[bcdfghjklmnpqrstvwxys])[aeuioAEUIO]{2,}(?=[bcdfghjklmnpqrstvwxys])', input())
print(m.group())



# 73 Re.start() & Re.end()
# na hackereanku nie na regexa, tylko re, czyli nie ma overlapped=True
import regex as re
s = "aaadaa"
wor = "aa"
matches = re.finditer(r'(?=(' + wor + '))', s, overlapped=True)
results = [(match.start(), match.start() + len(wor)) for match in matches]
print(*results, sep='\n') if results else print((-1, -1))


Input (stdin)
aaadaa
aa

import re
s, wor = input(), input()
print(*[(i.start(), i.start() + len(wor) - 1) for i in re.finditer('(?='+wor+')',s)], sep='\n')


import re
s, wor = input(), input()
matches = [(m.start(), m.start() + len(wor) - 1) for m in re.finditer('(?='+wor+')',s)]
print(*matches, sep='\n') if matches else print((-1, -1))



# 74 Validating Roman Numerals
import re
regex_pattern = r"^(?=[MDCLXVI])(M{0,3})(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$"
print(str(bool(re.match(regex_pattern, input()))))


Sample Input
CDXXI


^(?=[MDCLXVI])M*C[MD]|D?C{0,3}X[CL]|L?X{0,3}I[XV]|V?I{0,3}$ - bez nawiasów nie łapie całości



# 75 Validating phone numbers
import re
print(*["YES" if bool(re.match(r'^[789]\d{9}$', input())) else "NO" for _ in range(int(input()))], sep='\n')

Sample Input
2
9587456281
1252478965



# 76 Validating and Parsing Email Addresses
import re
for _ in range(int(input())):
    name, email = input().split()
    result = bool(re.match(r'^<[A-Za-z](\w|-|\.|_)+@[A-Za-z]+\.[A-Za-z]{1,3}>$', email))
    if result: print(*(name, email), sep=' ')

Sample Input
2  
DEXTER <dexter@hotmail.com>
VIRUS <virus!@variable.:p>

result = bool(re.match(r'^<[a-z](\w|-|\.|_)+@[a-z]+\.[a-z]{1,3}>$', email, re.I))



# 77 Hex Color Code
import re
for _ in range(int(input())):
    result = re.findall(r'#[\d|a-f]{3,6}(?=\)|,|;)', input(), re.I)
    if result: print(*result, sep='\n')

Sample Input
11
#BED
{
    color: #FfFdF8; background-color:#aef;
    font-size: 123px;
    background: -webkit-linear-gradient(top, #f9f9f9, #fff);
}
#Cab
{
    background-color: #ABC;
    border: 2px dashed #fff;
}   

r'(?<!^)#[\d|a-f]{3,6}' # negative lookbehind
r'[\s:](#[a-f0-9]{6}|#[a-f0-9]{3})'
r'(?<!^)(#(?:[\da-f]{3}){1,2})' # nie wiem jak to działa

# Regex and Parsing







# Closures and Decorators
# 78 Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        n = [" ".join((i[-10:-5], i[-5:])) for i in l]
        n.sort()
        print(*["+91 " + i for i in n], sep="\n")
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l)

Sample Input
3
07895462130
919875641230
9195969878


def wrapper(f):
    def fun(l):
        f(["+91 "+c[-10:-5]+" "+c[-5:] for c in l])
    return fun













# Numpy
# 79 Arrays
import numpy
if __name__ == '__main__':
    def arrays(arr):
        return numpy.array(arr[::-1], float)

    arr = input().strip().split(' ')
    result = arrays(arr)
    print(result)


Sample  Input
1 2 3 4 -8 -10


def arrays(arr):
    return numpy.flipud(numpy.array(arr, float))



# 80 Shape and Reshape
import numpy
ar = numpy.array(input().strip().split(' '), int)
print(numpy.reshape(ar,(3, 3)))


Sample Input
1 2 3 4 5 6 7 8 9

import numpy
ar = numpy.array(input().strip().split(' '), int)
ar.shape = (3, 3)
print(ar)


import numpy
print(numpy.reshape(numpy.array(input().strip().split(' '), int), (3, 3)))


import numpy
print(numpy.array(input().strip().split(' '), int).reshape(3, 3))


import numpy
print(numpy.array(list(map(int, input().split()))).reshape(3, 3))



# 81 Transpose and Flatten
import numpy
rows, cols = map(int, input().strip().split())
# n = numpy.array([list(map(int, input().strip().split(' '))) for _ in range(rows)])
n = numpy.array([input().strip().split(' ') for _ in range(rows)], int)
print(n.transpose())
print(n.flatten())


Sample Input
2 2
1 2
3 4



# 82 Concatenate
import numpy
n, m, _ = map(int, input().strip().split())
a1 = numpy.array([input().strip().split(' ') for _ in range(n)], int)
a2 = numpy.array([input().strip().split(' ') for _ in range(m)], int)
print(numpy.concatenate((a1, a2)))


Sample Input
4 3 2
1 2
1 2 
1 2
1 2
3 4
3 4
3 4 



# 83 Zeros and Ones
import numpy
n = list(map(int, input().strip().split()))
print(numpy.zeros(n, int))
print(numpy.ones(n, int))

Sample Input 0
3 3 3


import numpy
n = tuple(map(int, input().strip().split()))
print(numpy.zeros(n, int))
print(numpy.ones(n, int))



# 84 Eye and Identity
import numpy
numpy.set_printoptions(legacy='1.13')
n = tuple(map(int, input().strip().split()))
print(numpy.eye(n[0], n[1]))


Sample Input
3 3


import numpy
numpy.set_printoptions(legacy='1.13')
#n = tuple(map(int, input().strip().split()))
print(numpy.eye(*map(int, input().strip().split())))



# 85 Array Mathematics
import numpy as np
n, _ = map(int, input().split())
a, b = (np.array([input().split() for _ in range(n)], dtype=int) for _ in range(2))
print(a+b, a-b, a*b, a//b, a%b, a**b, sep='\n')

Sample Input
1 4
1 2 3 4
5 6 7 8


# ten kod nie działa, bo za mało nawiasów generuje
import numpy
_ = input()
a = numpy.array(input().strip().split(' '), int)
b = numpy.array(input().strip().split(' '), int)
# a, b = [numpy.array(input().strip().split(' '), int) for _ in range(2)]
print(numpy.array(a+b), numpy.array(a-b), sep='\n')
print(a+b, a-b, a*b, a//b, a%b, a**b, sep='\n')



# 86 Floor, Ceil and Rint
import numpy
numpy.set_printoptions(legacy='1.13')
arr = numpy.array(input().strip().split(), float)
print(numpy.floor(arr), numpy.ceil(arr), numpy.rint(arr), sep='\n')


Sample Input
1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8 9.9



# 87 Sum and Prod
import numpy
n, _ = input().strip().split()
print(numpy.prod(numpy.sum(numpy.array([input().strip().split() for _ in range(int(n))], int), axis=0)))


Sample Input
2 2
1 2
3 4



# 88 Min and Max
import numpy
n, _ = input().strip().split()
ar = numpy.array([input().strip().split() for _ in range(int(n))], int)
print(max(numpy.min(ar, axis=1)))


Sample Input
4 2
2 5
3 7
1 3
4 0



# 89 Mean, Var, and Std
import numpy
n, _ = input().strip().split()
ar = numpy.array([input().strip().split() for _ in range(int(n))], int)
print(numpy.mean(ar, axis=1), numpy.var(ar, axis=0), round(numpy.std(ar), 11), sep='\n')

Sample Input
2 2
1 2
3 4



# 90 Dot and Cross
import numpy
n = int(input().strip())
ar1, ar2 = [numpy.array([input().strip().split() for _ in range(n)], int) for _ in range(2)]
print(numpy.dot(ar1, ar2))

Sample Input
2
1 2
3 4
1 2
3 4



# 91 Inner and Outer
import numpy
a, b = numpy.array([input().strip().split() for _ in range(2)], int)
print(numpy.dot(a, b))
print(numpy.outer(a, b))

Sample Input
0 1
2 3


# inner product = dot 

# 92 Polynomials
import numpy
print(numpy.polyval(list(map(float, input().split())), float(input())))

Sample Input
1.1 2 3
0



# 93 Linear Algebra
import numpy
ar = numpy.array([input().strip().split() for _ in range(int(input()))], float)
print(round(numpy.linalg.det(ar), 2))


Sample Input
2
1.1 1.1
1.1 1.1

# Numpy










# XML 1 - Find the Score

import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
     return sum([len(elem.items()) for elem in tree.iter()])

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print get_attr_number(root)






# The HackerRank Interview Preparation Kit

# Warm-up Challenges

#1 Sales by Match
#!/bin/python3

import os
from collections import Counter

#
# Complete the 'sockMerchant' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY ar
#

def sockMerchant(n, ar):
    return sum([i//2 for i in list(Counter(ar).values())])

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    ar = list(map(int, input().rstrip().split()))

    result = sockMerchant(n, ar)
    #print(list(result))

    fptr.write(str(result) + '\n')

    fptr.close()




#2 Counting Valleys
path = list('UDDDUDUU')
path = list('DDUUDDUDUUUD')
lvl = 0
vall = 0
for i in path:
    if i == "U":
        lvl+=1
    if i == "D":
        lvl-=1
        if lvl == -1:
            vall+=1
print(vall)

'''
_/\      _
   \    /
    \/\/

_          /\_
 \  /\    /
  \/  \/\/
'''




#3 Jumping on the Clouds
#c = [0, 0, 0, 0, 1, 0]
c = [0, 0, 1, 0, 0, 1, 0]
#c = [0, 0, 0, 1, 0, 0]
print(len(c)-3)
i = 0
count = 0
while i <= len(c)-3:
    if c[i+1] == 0 and c[i+2] == 0:
        i+=2
        count+=1
    elif c[i+1] == 1 and c[i+2] == 0:
        i+=2
        count+=1
    elif c[i+1] == 0:
        i+=1
        count+=1
if i != len(c) - 1: count+=1
print(count)
print(i)

# 6-3=3 
# i0 i2 c1
# i2 i3 c2
# i3 i5 c3
#
# 7-3=4
# i0 i1 c1
# i1 i3 c2
# i3 i4 c3
# i4 i6 c4
#
# 6-3=3
# i0 i2 c1
# i2 i4 c2
# i4

i = 0
count = 0
while i < len(c)-2:
    if c[i+2] == 0:
        i+=2
        count+=1
    else:
        i+=1
        count+=1
if i != len(c) - 1: count+=1        
print(count)
print(i)




# Repeated String
s = 'ab'
n = 9

count2 = s.count('a') * (n//len(s)) + s[:n%len(s)].count('a')

print(count2)





# Arrays
# 2D Array - DS

arr = [[1, 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 2, 4, 4, 0], [0, 0, 0, 2, 0, 0], [0, 0, 1, 2, 4, 0]]
print(max([arr[i][j]+arr[i][j+1]+arr[i][j+2]+arr[i+1][j+1]+arr[i+2][j]+arr[i+2][j+1]+arr[i+2][j+2] for i in range(4) for j in range(4)]))

import numpy as np
print(max([sum(sum(arr[i+0:i+3, j+0:j+3]) - arr[1+i, 0+j] - arr[1+i, 2+j]) for i in range(4) for j in range(4)]))




# Arrays: Left Rotation
d = 4
a = [1, 2, 3, 4, 5]

print(a[d:] + a[:d])




# New Year Chaos

q = [2, 1, 5, 3, 4]
q = [2, 3, 5, 1, 4]
q = [5, 1, 2, 3, 7, 8, 6, 4]
#q = [1, 2, 5, 3, 7, 8, 6, 4]
#print(q)

def minimumBribes(q):
    swap = 0
    # for i in range(len(q)-1, -1, -1):
    for i in range(len(q)):
        # print(q[i], (i + 1))
        if q[i] > (i + 1) + 2:
            swap = 'Too chaotic'
            #print('Too chaotic')
            return print('Too chaotic')
        for j in range(max(0, q[i] - 2), i):
            if q[j] > q[i]:
                swap += 1
    return print(swap)

#print(minimumBribes(q))
minimumBribes(q)

        
        

# Minimum Swaps 2
q = [2, 1, 5, 3, 4]
#q = [2, 3, 5, 1, 4]
q = [5, 1, 2, 3, 7, 8, 6, 4]
#q = [1, 2, 5, 3, 7, 8, 6, 4]
q = [7, 1, 3, 2, 4, 5, 6]
q = [4, 3, 1, 2]


print(q)
def minimumSwaps(q):
    swap = 0
    i = 0
    while i < len(q):
        if q[i] != i + 1:
            tmp = q[i]
            print(tmp)
            print(q[q[i] - 1])
            q[i], q[tmp - 1] = q[tmp - 1], tmp
            print(q)
            swap += 1
        else:
            i += 1
    return print(swap)

#print(minimumBribes(q))
minimumSwaps(q)
        



# Array Manipulation
import collections
from typing import Collection


from collections import Counter
a = [[1, 2, 100], [2, 5, 100], [3, 4, 100]]

def arrayManipulation(n, queries):
    arr = [0] * n
    for i in range(len(queries)):
        for j in range(queries[i][0] - 1 , queries[i][1]):
            arr[j] += queries[i][2]
    return print(max(arr))


arrayManipulation(5, a)




# Dictionaries and Hashmaps
# Hash Tables: Ransom Note
from collections import Counter
def checkMagazine(magazine, note):
    note_counter = Counter(note)
    mag_counter = Counter(magazine)
    # print("Yes" if all([mag_counter[key] >= note_counter[key] for key, val in note_counter.items()]) else "No")
    print("Yes" if Counter(note) - Counter(magazine) == {} else "No")

checkMagazine('give me one grand today night'.rstrip().split(),'dup give one grand today'.rstrip().split())




# Two Strings
from collections import Counter
s1 = 'hello'
s2 = 'world'

def twoStrings(s1, s2):
    return print("YES" if Counter(s1) & Counter(s2) != {} else "NO")

twoStrings(s1, s2)




# Sherlock and Anagrams
# moszna
from collections import Counter
s = 'abba'
s = 'cdcd'
s = 'ifailuhkqq'
#s = 'kkkk'

'''
for i in range(1, 4):
    for j in range(i):
        print(i, j)
'''

def sherlockAndAnagrams(s):
    count = 0
    for i in range(1, len(s)):
        for j in range(i-1, -1, -1):
            print(i, j)
            print(s[j:i], s[j+1:])
            #for k in range(len(s[j+1:]) - len(s[j:i]) + 1):
            #   print(k, s[k:k+i])
            for k in range(len(s[j+1:])):
                if Counter(s[j:i]) == Counter(s[j+1:][k:k + len(s[j:i])]):
                    count += 1
    return count

print(sherlockAndAnagrams(s))
#sherlockAndAnagrams(s)


# Ten jest szybszy i penie lepiej działa
def sherlockAndAnagrams(s):
    count = Counter(("".join(sorted(s[j:j+i])) for i in range(1, len(s)) for j in range(len(s)-i+1)))
    return sum(sum(range(i)) for i in count.values())




# Count Triplets 
# moszna
from itertools import combinations
from collections import Counter
arr = [1, 2, 2, 4]
arr = [1, 5, 5, 25, 125]
r = 5

def countTriplets(arr, r):
    counter = 0
    for i in combinations(arr, 3):
        if len(set(i)) == 3 or set((1, 1)) == {1}:
            #print(list(map(lambda x: x/i[0], i)), end='')
            if list(map(lambda x: x/i[0], i)) == [1, r, r*r]:
                counter += 1
            #print(list(map(lambda x: x/i[0], i)))
    return counter

print(countTriplets(arr, r))

for i in combinations(arr, 3):
    print(set(i))

print(set((1, 1)) == {1})


# zajebane z hackerranka
def countTriplets(arr, r):
    a = Counter(arr)
    b = Counter()
    c = Counter()
    s = 0
    for i in arr:
        print(i)
        j = i//r
        print(j)
        k = i*r
        a[i]-=1
        print(b[j])
        if b[j] and a[k] and not i%r:
            s+=b[j]*a[k]
        b[i]+=1
        print()
    return b

print(countTriplets(arr, r))




# Sorting
# Sorting: Bubble Sort
a = [3, 2, 1]
a = [1, 2, 3]

def countSwaps(a):
    counter = 0
    for i in range(1, len(a)):
        for j in range(i):
            if a[i] < a[j]:
                print(a)
                a[j], a[i] = a[i], a[j]
                counter += 1
    print('Array is sorted in {} swaps.\nFirst Element: {}\nLast Element: {}'.format(counter, a[0], a[-1]))
    return

print(countSwaps(a))




# Mark and Toys
prices = [1, 12, 5, 111, 200, 1000, 10]
k = 50

def maximumToys(prices, k):
    items = 0
    ind = 0
    prices = sorted(prices)
    while items + prices[ind] <= k:
        items += prices[ind]
        ind += 1
    return ind

print(maximumToys(prices, k))




# Sorting: Comparator

