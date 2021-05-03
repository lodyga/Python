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



