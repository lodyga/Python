# codewars kata

from xml import dom
from nbformat import read
from solution import arithmetic
from gettext import find
from regex import D
from solution import enough
import enum
from itertools import reduce
import itertools
from operator import index
from hel import add





# Bit Counting
# https://www.codewars.com/kata/526571aae218b8ee490006f4
"""Write a function that takes an integer as input, and returns the number of bits that are equal
to one in the binary representation of that number. You can guarantee that input is non-negative.
Example: The binary representation of 1234 is 10011010010, so the function should return 5 in this case"""

def count_bits(n):
    return bin(n)[2:].count('1')
    # return bin(n).count('1')
    # return sum(True if int(i) else False for i in bin(n)[2:])
    # return '{:b}'.format(n).count('1')
    # return n.bit_count()
count_bits(1234)





# Sum of Digits / Digital Root
# https://www.codewars.com/kata/541c8630095125aba6000c00/train/python
"""Digital root is the recursive sum of all the digits in a number.
Given n, take the sum of the digits of n. If that value has more than one digit, 
continue reducing in this way until a single-digit number is produced. The input will be a non-negative integer.
"""

from more_itertools import iterate
import numpy as np

def digital_root(n):
    while n > 9:
        n = sum(map(int, str(n)))
    return n

def digital_root(n):
    return n if n < 10 else digital_root(sum(map(int, str(n))))

def digital_root(n):
    while len(str(n)) != 1:
        n = np.sum((int(digit) for digit in str(n)))
    return n

def digital_root(n):
    while True:
        n = sum(int(digit) for digit in str(n))
        if len(str(n)) == 1:
            return n
digital_root(169)
digital_root(942)
digital_root(132189)


def digital_root(n):
    while True:
        if n > 9:
            n = sum([int(digit) for digit in str(n)])
        else:
            return n

import numpy as np
def digital_root(n):
    while True:
        if len(str(n)) != 1:
            n = np.sum((int(digit) for digit in str(n)))
        else:
            return n

test.assert_equals(digital_root(16), 7)
test.assert_equals(digital_root(942), 6)
test.assert_equals(digital_root(132189), 6)
test.assert_equals(digital_root(493193), 2)





# Detect Pangram
# https://www.codewars.com/kata/545cedaa9943f7fe7b000048/train/
"""A pangram is a sentence that contains every single letter of the alphabet at least once.
For example, the sentence "The quick brown fox jumps over the lazy dog" is a pangram, because it uses the letters A-Z at least once (case is irrelevant).
Given a string, detect whether or not it is a pangram. Return True if it is, False if not. Ignore numbers and punctuation."""

import string

def is_pangram(s):
    return all(True if letter in s.lower() else False for letter in string.ascii_lowercase)

pangram = "The quick, brown fox jumps over the lazy dog!"
is_pangram(pangram)

def is_pangram(s):
    return set(string.ascii_lowercase) <= set(s.lower())

def is_pangram(s):
    return set(s.lower()) >= set(string.ascii_lowercase)

def is_pangram(s):
    return set(string.ascii_lowercase).issubset(set(s.lower()))

def is_pangram(s):
    return set(s.lower()).issuperset(string.ascii_lowercase)

def is_pangram(s):
    return set(string.ascii_lowercase) & set(s.lower()) == set(string.ascii_lowercase)

def is_pangram(s):
    return not sum(0 if letter in s.lower() else 1 for letter in string.ascii_lowercase)

test.assert_equals(is_pangram(pangram), True)





# Multiples of 3 or 5
# https://www.codewars.com/kata/514b92a657cdc65150000006/train/python
"""If we list all the natural numbers below 10 that are multiples of 3 or 5, we get 3, 5, 6 and 9.
The sum of these multiples is 23.
Finish the solution so that it returns the sum of all the multiples of 3 or 5 below the number passed in.
Additionally, if the number is negative, return 0 (for languages that do have them).
Note: If the number is a multiple of both 3 and 5, only count it once."""

def solution(numbers):
    # return sum(digit for digit in range(3, numbers) if not digit % 3 or not digit % 5)
    return sum(digit for digit in range(3, numbers) if not (digit % 3 and digit % 5))
solution(4)
solution(6)
solution(16)


import numpy as np
def solution(numbers):
    return np.sum([number for number in range(numbers) if not number % 3 or not number % 5])


test.assert_equals(solution(4), 3)    
test.assert_equals(solution(6), 8)
test.assert_equals(solution(16), 60)
test.assert_equals(solution(3), 0)
test.assert_equals(solution(5), 3)
test.assert_equals(solution(15), 45)
test.assert_equals(solution(0), 0)
test.assert_equals(solution(-1), 0)
test.assert_equals(solution(10), 23)
test.assert_equals(solution(20), 78)
test.assert_equals(solution(200), 9168)





# The Hashtag Generator
# https://www.codewars.com/kata/52449b062fb80683ec000024
"""The marketing team is spending way too much time typing in hashtags.
Let's help them with our own Hashtag Generator!

Here's the deal:

It must start with a hashtag (#).
All words must have their first letter capitalized.
If the final result is longer than 140 chars it must return false.
If the input or the result is an empty string it must return false.
Examples
" Hello there thanks for trying my Kata"  =>  "#HelloThereThanksForTryingMyKata"
"    Hello     World   "                  =>  "#HelloWorld"
""                                        =>  false"""

def generate_hashtag(s):
    solution = "#" + "".join(word.capitalize() for word in s.strip().split())
    # solution = "#" + s.title().replace(" ", "")
    return solution if (s and len(solution) <= 140) else False
generate_hashtag('codewars is nice')
generate_hashtag('CodeWars  is   nice')
generate_hashtag('')

def generate_hashtag(s):
    solution = "#" + "".join(word.capitalize() for word in s.strip().split())
    return False if not s or len(solution) > 140 else solution


# final result must be > than 140 not input
def generate_hashtag(s):
    if (len(s) > 140 or
        not s):
        return False
    return '#' + ''.join(i.capitalize() for i in s.strip().split(' '))

def generate_hashtag(s):
    if (len(s) > 140 ore
            not s):
        return False
    return '#' + ''.join(map(lambda x: x.strip().capitalize(), s.split()))

def generate_hashtag(s):
    if len(s) > 140 or s == '':
        return False
    else:
        return '#' + ''.join([string.strip().capitalize() for string in s.split()])

def generate_hashtag(s):
    if len(s) > 140 or s == '':
        formatedstring = False
    else:
        formatedstring = '#'
        for string in s.split():
            formatedstring += string.strip().capitalize()
    return formatedstring


test.describe("Basic tests")
test.assert_equals(generate_hashtag(''), False, 'Expected an empty string to return False')
test.assert_equals(generate_hashtag('Do We have A Hashtag')[0], '#', 'Expeted a Hashtag (#) at the beginning.')
test.assert_equals(generate_hashtag('Codewars'), '#Codewars', 'Should handle a single word.')
test.assert_equals(generate_hashtag('Codewars      '), '#Codewars', 'Should handle trailing whitespace.')
test.assert_equals(generate_hashtag('Codewars Is Nice'), '#CodewarsIsNice', 'Should remove spaces.')
test.assert_equals(generate_hashtag('codewars is nice'), '#CodewarsIsNice', 'Should capitalize first letters of words.')
test.assert_equals(generate_hashtag('CodeWars is nice'), '#CodewarsIsNice', 'Should capitalize all letters of words - all lower case but the first.')
test.assert_equals(generate_hashtag('c i n'), '#CIN', 'Should capitalize first letters of words even when single letters.')
test.assert_equals(generate_hashtag('codewars  is  nice'), '#CodewarsIsNice', 'Should deal with unnecessary middle spaces.')
test.assert_equals(generate_hashtag('Looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong Cat'), False, 'Should return False if the final word is longer than 140 chars.')





# Product of consecutive Fib numbers
# https://www.codewars.com/kata/5541f58a944b85ce6d00006a
"""The Fibonacci numbers are the numbers in the following integer sequence (Fn):
0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, ...
such as
F(n) = F(n-1) + F(n-2) with F(0) = 0 and F(1) = 1.
Given a number, say prod (for product), we search two Fibonacci numbers F(n) and F(n+1) verifying
F(n) * F(n+1) = prod.
Your function productFib takes an integer (prod) and returns an array:
[F(n), F(n+1), true] or {F(n), F(n+1), 1} or (F(n), F(n+1), True)
depending on the language if F(n) * F(n+1) = prod.
If you don't find two consecutive F(n) verifying F(n) * F(n+1) = prodyou will return
[F(n), F(n+1), false] or {F(n), F(n+1), 0} or (F(n), F(n+1), False)
F(n) being the smallest one such as F(n) * F(n+1) > prod.
Some Examples of Return:
(depend on the language)
productFib(714) # should return (21, 34, true), 
                # since F(8) = 21, F(9) = 34 and 714 = 21 * 34
productFib(800) # should return (34, 55, false), 
                # since F(8) = 21, F(9) = 34, F(10) = 55 and 21 * 34 < 800 < 34 * 55"""

def productFib(prod):
    a, b = 0, 1
    while a * b < prod:
        a, b = b, a + b
    return [a, b, a * b == prod]
productFib(714) # should return (21, 34, true) 
productFib(800) # should return (34, 55, false)
productFib(4895) # [55, 89, True] 
productFib(5895) # [89, 144, False]

def productFib(prod):
    a, b = 0, 1
    while a * b <= prod:
        if prod == a * b:
            return [a, b, True]
        else:
            a, b = b, a + b
    return [a, b, False]

def productFib(prod):
    a, b = 0, 1
    while a * b < prod:
        a, b = b, a + b
    return [a, b, a * b == prod]


def productFib(prod):
    if prod == 0: return [0, 1, True]
    Fib = [0, 1]
    i = 2
    while True:
        Fib.append(Fib[i-1] + Fib[i-2])
        if Fib[i] * Fib[i-1] >= prod:
            break
        i += 1
    return [Fib[i-1], Fib[i], Fib[i] * Fib[i-1] == prod]




test.assert_equals(productFib(4895), [55, 89, True])
test.assert_equals(productFib(5895), [89, 144, False])





# Write Number in Expanded Form
# https://www.codewars.com/kata/5842df8ccbd22792a4000245
"""Write Number in Expanded Form
You will be given a number and you will need to return it as a string in Expanded Form. For example:

expanded_form(12) # Should return '10 + 2'
expanded_form(42) # Should return '40 + 2'
expanded_form(70304) # Should return '70000 + 300 + 4'
NOTE: All numbers will be whole numbers greater than 0."""

def expanded_form(num):
    return " + ".join(str(int(digit) * 10**(len(str(num)) - index)) for index, digit in enumerate(str(num), 1) if int(digit))
expanded_form(70304)

def expanded_form(num):
    return ' + '.join(str(int(j) * 10**(len(str(num)) - i)) for i, j in enumerate(str(num), 1) if j != '0')

def expanded_form(num):
    filter_with_None = map(lambda i, j: str(int(j) * 10**(len(str(num)) - i)) if j != '0' else None, range(1, len(str(num)) + 1), iter(str(num)))
    return ' + '.join(filter(bool, filter_with_None))

def expanded_form(num):
    divided_number = []
    for index, digit in enumerate(str(num), 1):
        if int(digit):
            divided_number.append(str(int(digit) * 10**(len(str(num)) - index)))
    return " + ".join(divided_number)


test.assert_equals(expanded_form(12), '10 + 2');
test.assert_equals(expanded_form(42), '40 + 2');
test.assert_equals(expanded_form(70304), '70000 + 300 + 4');





# RGB To Hex Conversion
# https://www.codewars.com/kata/513e08acc600c94f01000001
"""The rgb function is incomplete. Complete it so that passing in RGB decimal values will result in a hexadecimal representation being returned. Valid decimal values for RGB are 0 - 255. Any values that fall out of that range must be rounded to the closest valid value.

Note: Your answer should always be 6 characters long, the shorthand with 3 will not work here.

The following are examples of expected output values:

rgb(255, 255, 255) # returns FFFFFF
rgb(255, 255, 300) # returns FFFFFF
rgb(0,0,0) # returns 000000
rgb(148, 0, 211) # returns 9400D3"""


def rgb(r, g, b):
    rgb_round = lambda x: min(255, max(0, x))
    return ('{:02X}'*3).format(rgb_round(r), rgb_round(g), rgb_round(b))
    # return ('{:02X}'*3).format(*map(lambda x: rgb_round(x), (r, g, b)))
rgb(1, 2, 3)
rgb(255, 255, 255)
rgb(254, 253, 252)

def rgb(r, g, b):
    r, g, b = min(255, max(r, 0)), min(255, max(g, 0)), min(255, max(b, 0))
    red = "{:02X}".format(r)
    # red = "{:X}".format(r) if len("{:X}".format(r)) == 2 else "0{:X}".format(r)
    green = "{:X}".format(g) if g > 15 else "0{:X}".format(g)
    blue = "{:X}".format(b) if b > 15 else "0{:X}".format(b)
    # return "{:X}{:X}{:X}".format(r, g, b)
    return red+green+blue

def rgb(r, g, b):
    # r, g, b = list(map(lambda x: 0 if x < 0 else 255 if x > 255 else x, (r, g, b)))
    # return ('{:02X}'*3).format(r, g, b)
    return ('{:02X}'*3).format(*map(lambda x: 0 if x < 0 else 255 if x > 255 else x, (r, g, b)))


test.assert_equals(rgb(0,0,0),"000000", "testing zero values")
test.assert_equals(rgb(1,2,3),"010203", "testing near zero values")
test.assert_equals(rgb(255,255,255), "FFFFFF", "testing max values")
test.assert_equals(rgb(254,253,252), "FEFDFC", "testing near max values")
test.assert_equals(rgb(-20,275,125), "00FF7D", "testing out of range values")





# The observed PIN
# https://www.codewars.com/kata/5263c6999e0f40dee200059d
"""Alright, detective, one of our colleagues successfully observed our target person, Robby the robber. We followed him to a secret warehouse, where we assume to find all the stolen stuff. The door to this warehouse is secured by an electronic combination lock. Unfortunately our spy isn't sure about the PIN he saw, when Robby entered it.

The keypad has the following layout:

┌───┬───┬───┐
│ 1 │ 2 │ 3 │
├───┼───┼───┤
│ 4 │ 5 │ 6 │
├───┼───┼───┤
│ 7 │ 8 │ 9 │
└───┼───┼───┘
    │ 0 │
    └───┘
He noted the PIN 1357, but he also said, it is possible that each of the digits he saw could actually be another adjacent digit (horizontally or vertically, but not diagonally). E.g. instead of the 1 it could also be the 2 or 4. And instead of the 5 it could also be the 2, 4, 6 or 8.

He also mentioned, he knows this kind of locks. You can enter an unlimited amount of wrong PINs, they never finally lock the system or sound the alarm. That's why we can try out all possible (*) variations.

* possible in sense of: the observed PIN itself and all variations considering the adjacent digits

Can you help us to find all those variations? It would be nice to have a function, that returns an array (or a list in Java/Kotlin and C#) of all variations for an observed PIN with a length of 1 to 8 digits. We could name the function getPINs (get_pins in python, GetPINs in C#). But please note that all PINs, the observed one and also the results, must be strings, because of potentially leading '0's. We already prepared some test cases for you.

Detective, we are counting on you!"""


around = {'1': ['1', '2', '4'],
          '2': ['2', '1', '3', '5'],
          '3': ['2', '6', '3'],
          '4': ['1', '4', '7', '5'],
          '5': ['5', '2', '4', '6', '8'],
          '6': ['6', '3', '5', '9'],
          '7': ['7', '4', '8'],
          '8': ['8', '5', '7', '9', '0'],
          '9': ['9', '6', '8'],
          '0': ['0', '8']
          }

# both are equal to itertools.product
around = {'1': '124',
          '2': '2135',
          '3': '263',
          '4': '1475',
          '5': '52468',
          '6': '6359',
          '7': '748',
          '8': '85790',
          '9': '968',
          '0': '08'
          }

from itertools import product
def get_pins(observed):
    # to_join = product(*[around[char] for char in observed])
    # return list(''.join(i) for i in to_join)
    return [''.join(i) for i in product(*(around[char] for char in observed))]
get_pins('369')

def get_pins(observed):
    pin_dict =  {
                "0": "08",
                "1": "124",
                "2": "1235",
                "3": "326",
                "4": "1457",
                "5": "45682",
                "6": "6593",
                "7": "748",
                "8": "57890",
                "9": "986"
                }
    adjency_list = [pin_dict[i] for i in str(observed)]
    codes_to_concat = product(*adjency_list)
    return ["".join(pin) for pin in codes_to_concat]
get_pins(369)

def get_pins(observed):
    button_adjancency =  ("08", "124", "1235", "326", "1457", "45682", "6593", "748", "57890", "986")
    return ["".join(pin) for pin in product(*(button_adjancency[int(i)] for i in str(observed)))]
    # return list(map("".join, product(*(button_adjancency[int(i)] for i in str(observed)))))
get_pins(369)


list(product(*[['1', '2', '4'], ['8', '5', '7', '9', '0'], ['0', '1']]))
list(product(['1', '2', '4'], ['8', '5', '7', '9', '0'], ['0', '1']))
list(product(*('124', '85790', '01')))
list(product('124', '85790', '01'))

test.describe('example tests')
expectations = [('8', ['5','7','8','9','0']),
                ('11',["11", "22", "44", "12", "21", "14", "41", "24", "42"]),
                ('369', ["339","366","399","658","636","258","268","669","668","266","369","398","256","296","259","368","638","396","238","356","659","639","666","359","336","299","338","696","269","358","656","698","699","298","236","239"])]





# Maximum subarray sum
# https://www.codewars.com/kata/54521e9ec8e60bc4de000d6c
# too slow
"""The maximum sum subarray problem consists in finding the maximum sum of a contiguous subsequence in an array or list of integers:

max_sequence([-2, 1, -3, 4, -1, 2, 1, -5, 4])
# should be 6: [4, -1, 2, 1]
Easy case is when the list is made up of only positive numbers and the maximum sum is the sum of the whole array. 
If the list is made up of only negative numbers, return 0 instead.

Empty list is considered to have zero greatest sum. Note that the empty list or array is also a valid sublist/subarray."""


def max_sequence(arr):
    # if (not arr or
    #     not list(filter(lambda x: x >= 0, arr))):
    if not (arr and
        list(filter(lambda x: x >= 0, arr))):
        return 0
    return max(0, max(map(sum, (arr[i:j] for i in range(len(arr)) for j in range(1, len(arr) + 1) if j > i))))
max_sequence([-2, 1, -3, 4, -1, 2, 1, -5, 4]) # sum([4, -1, 2, 1]) = 6
max_sequence([-2]) # 0
max_sequence([]) #
max_sequence([-2, 1])

def max_sequence(arr):
    array_max = 0
    for i in range(len(arr)):
        for j in range(1, len(arr) + 1):
            if (i <= j and
                array_max < sum(arr[i:j])):
                array_max = sum(arr[i:j])
    return array_max

# doesn't work on empty arrays, must have at least 2 elements
# in first if loop of '+0' - array must have at least 1 element
# in first if loop of '+1' - empty array possible

def max_sequence(arr):
    return max([sum(arr[i:j]) for i in range(len(arr)) for j in range(1, len(arr) + 1) if i <= j])

def max_sequence(arr):
    return max([sum(arr[i:j]) for i in range(len(arr)+1) for j in range(len(arr)+1)])

def max_sequence(arr):
    if sum(1 if digit < 0 else 0 for digit in arr) == len(arr):
        return 0
    max_sum = 0
    # i_max, j_max = 0, 0
    for i in range(len(arr)):
        for j in range(len(arr) - i):
            print(arr[i:i+j+1])
            # print(sum(arr[i:i+j+1]))
            if sum(arr[i:i+j+1]) > max_sum:
                max_sum = sum(arr[i:i+j+1])
                # i_max, j_max = i, j
    # return sum(arr[i_max:i_max+j_max+1])
    return max_sum
                
def max_sequence(arr):
    return max(max(sum(arr[i:i+j+1]) for i in range(len(arr)) for j in range(len(arr) - i)), 0) if arr else 0

max_sequence([-2, 1, -3, 4, -1, 2, 1, -5, 4])
max_sequence([])
max_sequence([-1, -2])

test.assert_equals(max_sequence([-2, 1, -3, 4, -1, 2, 1, -5, 4]), 6)





# Weight for weight
# https://www.codewars.com/kata/55c6126177c9441a570000cc/train/python
"""My friend John and I are members of the "Fat to Fit Club (FFC)". John is worried because each month a list with the weights of members is published and each month he is the last on the list which means he is the heaviest.

I am the one who establishes the list so I told him: "Don't worry any more, I will modify the order of the list". It was decided to attribute a "weight" to numbers. The weight of a number will be from now on the sum of its digits.

For example 99 will have "weight" 18, 100 will have "weight" 1 so in the list 100 will come before 99.

Given a string with the weights of FFC members in normal order can you give this string ordered by "weights" of these numbers?

Example:
"56 65 74 100 99 68 86 180 90" ordered by numbers weights becomes: 

"100 180 90 56 65 74 68 86 99"
When two numbers have the same "weight", let us class them as if they were strings (alphabetical ordering) and not numbers:

180 is before 90 since, having the same "weight" (9), it comes before as a string.

All numbers in the list are positive numbers and the list can be empty."""


def order_weight(strng):
    # return ' '.join(sorted(strng.split(), key=lambda x: sum(int(d) for d in x))) # without second parameter `x` it sorts but `2000` is before `11`
    return " ".join(sorted(strng.split(), key=lambda x: (sum(int(digit) for digit in x), x)))
order_weight("2000 10003 1234000 44444444 9999 11 11 22 123")
order_weight("103 123 4444 99 2000")

def order_weight(strng):
    if not strng:
        return ''
    splitted = [char.strip() for char in strng.split()]
    weight_list = [(sum(int(digit) for digit in number), number) for number in splitted]
    # return ' '.join(np.array(sorted(weight_list))[:, 1])
    return weight_list


test.assert_equals(order_weight("103 123 4444 99 2000"), "2000 103 123 4444 99")
test.assert_equals(order_weight("2000 10003 1234000 44444444 9999 11 11 22 123"), "11 11 2000 10003 22 123 1234000 44444444 9999")
test.assert_equals(order_weight(""), "")





# Strip Comments
# https://www.codewars.com/kata/51c8e37cee245da6b40000bd/train/python
"""
Complete the solution so that it strips all text that follows any of a set of comment markers passed in. Any whitespace at the end of the line should also be stripped out.
solution("apples, pears # and bananas\ngrapes\nbananas !apples", ["#", "!"])
# result should == "apples, pears\ngrapes\nbananas"
"""


import re
def strip_comments(string, markers):
    for i in markers:
        string = string.replace(i, '!')
    # return '\n'.join((i[:i.find('!')]).rstrip() for i in string.split('\n'))
    return '\n'.join(i.split('!')[0].rstrip() for i in string.split('\n'))
strip_comments('apples, pears # and bananas\ngrapes\nbananas !apples', ['#', '!'])
strip_comments(' a #b\nc\nd $e f g', ['#', '$'])
strip_comments("' = lemons\napples pears\n. avocados\n= ! bananas strawberries avocados !\n= oranges", ['!', '^', '#', '@', '='])
strip_comments('cherries bananas lemons\n, apples watermelons\nwatermelons cherries apples cherries strawberries oranges\ncherries ! apples\ncherries avocados pears 

# this is faster
def strip_comments(string, markers):
    splited_list = string.split('\n')
    for marker in markers:
        splited_list = [elem.split(marker)[0].rstrip() for elem in splited_list]
        # splited_list = [elem if elem.find(marker) == -1 else elem[:elem.find(marker)].rstrip() for elem in splited_list]
        # print([elem if elem.find(marker) == -1 else elem[:elem.find(marker)].rstrip() for elem in splited_list])
    return '\n'.join(splited_list)

def strip_comments(strng, markers):
    if not markers:
        return strng
    cut_indexes = [min(line.find(char) if line.find(char) != -1 else len(line) for char in markers) for line in strng.split("\n")]
    return "\n".join((strng.split("\n")[line_ind][:cut_index]).rstrip() for line_ind, cut_index in enumerate(cut_indexes))

def strip_comments(string, markers):
    splitted_list = string.split('\n')
    for char in markers:
        # splitted_list = [i if i.find(char) == -1 else i[:i.find(char)].rstrip() for i in splitted_list]
        splitted_list = [i.split(char)[0].rstrip() for i in splitted_list]
    return '\n'.join(splitted_list)

def strip_comments(string, markers):
    for marker in markers:
        # the space before works as ' '.join()
        # string = re.sub(r' *?\{}.*$'.format(marker), '', string, flags=re.M)
        string = re.sub(r' *?\{}.*$'.format(marker), '', string, flags=re.M)
    return string


test.assert_equals(solution('apples, pears # and bananas\ngrapes\nbananas !apples', ['#', '!']), 'apples, pears\ngrapes\nbananas')
test.assert_equals(solution('a #b\nc\nd $e f g', ['#', '$']), 'a\nc\nd')
test.assert_equals(solution(' a #b\nc\nd $e f g', ['#', '$']), ' a\nc\nd')


# Some coding
'a{}b'.format(['#', '!'])
r' *?{}.*'.format(['#', '!'])

regex = r' *?[#].*'
re.sub(r' *?[#!].*', '', 'June #24\nAugust 9\nDe#c 12')





# Human readable duration format
# https://www.codewars.com/kata/52742f58faf5485cae000b9a/train/python
# works here but codewars is complaining
"""Your task in order to complete this Kata is to write a function which formats a duration, given as a number of seconds, in a human-friendly way.

The function must accept a non-negative integer. If it is zero, it just returns "now". Otherwise, the duration is expressed as a combination of years, days, hours, minutes and seconds.

It is much easier to understand with an example:

* For seconds = 62, your function should return 
    "1 minute and 2 seconds"
* For seconds = 3662, your function should return
    "1 hour, 1 minute and 2 seconds"
For the purpose of this Kata, a year is 365 days and a day is 24 hours.

Note that spaces are important.

Detailed rules
The resulting expression is made of components like 4 seconds, 1 year, etc. In general, a positive integer and one of the valid units of time, separated by a space. The unit of time is used in plural if the integer is greater than 1.

The components are separated by a comma and a space (", "). Except the last component, which is separated by " and ", just like it would be written in English.

A more significant units of time will occur before than a least significant one. Therefore, 1 second and 1 year is not correct, but 1 year and 1 second is.

Different components have different unit of times. So there is not repeated units like in 5 seconds and 1 second.

A component will not appear at all if its value happens to be zero. Hence, 1 minute and 0 seconds is not valid, but it should be just 1 minute.

A unit of time must be used "as much as possible". It means that the function should not return 61 seconds, but 1 minute and 1 second instead. Formally, the duration specified by of a component must not be greater than any valid more significant unit of time."""


def format_duration(seconds):
    if not seconds:
        return 'now'
    time_digits = (60 * 60 * 24 * 365, 60 * 60 * 24, 60 * 60, 60, 1)
    time_name = ('year', 'day', 'hour', 'minute', 'second')
    time_list = []
    for i in time_digits:
        if seconds // i:
            time_list.append(seconds // i)
            seconds %= i
    result = (i for i in zip(time_list, time_name[-len(time_list):]))
    result2 = [str(i[0]) + ' ' + i[1] + 's' if i[0] > 1 else str(i[0]) + ' ' + i[1] for i in result]
    return result2[0] if len(result2) == 1 else ', '.join(result2[:-1]) + ' and ' + result2[-1]
format_duration(3662)

# from codewars
times = [("year", 365 * 24 * 60 * 60),
         ("day", 24 * 60 * 60),
         ("hour", 60 * 60),
         ("minute", 60),
         ("second", 1)]

def format_duration(seconds):
    if not seconds:
        return "now"
    
    chunks = []
    for name, secs in times:
        qty = seconds // secs
        if qty:
            if qty > 1:
                name += "s"
            chunks.append(str(qty) + " " + name)

        seconds = seconds % secs

    return ', '.join(chunks[:-1]) + ' and ' + chunks[-1] if len(chunks) > 1 else chunks[0]

import datetime
def format_duration(seconds):
    if not seconds:
        return 'now'
    dict_time = {4: 'year', 3: 'day', 2: 'hour', 1: 'minute', 0: 'second', 
                 14: 'years', 13: 'days', 12: 'hours', 11: 'minutes', 10: 'seconds'}
    # str_time = time.strftime('%H:%M:%S', time.gmtime(seconds))
    # transforms seconds to 'days, minuses:hours:seconds'
    dirty_days = str(datetime.timedelta(seconds=seconds)) # days + '%H:%M:%S'
    str_time = dirty_days[-8:].lstrip() # only '%H:%M:%S'
    if 'day' in dirty_days:
        days = int(dirty_days.split()[0])
        if days > 365:
            years = days // 365
            days = days % 365
            # years, days = divmod(days, 365)
            str_time = str(years)+':'+str(days)+':'+str_time
        else:
            str_time = str(days)+':'+str_time
    
    sol = []
    for enum, i in enumerate(reversed(str_time.split(':'))):
        if int(i) != 0:
            if int(i) == 1:
                sol.append(str(int(i))+' '+dict_time[enum])
            else:
                sol.append(str(int(i))+' '+dict_time[enum + 10])
    # sol = list(reversed(sol))
    sol.reverse()
    if len(sol) == 1:
        return sol[0]
    else:
        return ', '.join(sol[:-1]) + ' and ' + sol[-1]

format_duration(6300000000)


test.assert_equals(format_duration(1), "1 second")
test.assert_equals(format_duration(62), "1 minute and 2 seconds")
test.assert_equals(format_duration(120), "2 minutes")
test.assert_equals(format_duration(3600), "1 hour")
test.assert_equals(format_duration(3662), "1 hour, 1 minute and 2 seconds")





# Persistent Bugger.
# https://www.codewars.com/kata/55bf01e5a717a0d57e0000ec
"""Write a function, persistence, that takes in a positive parameter num and returns its multiplicative persistence, which is the number of times you must multiply the digits in num until you reach a single digit.

For example (Input --> Output):

39 --> 3 (because 3*9 = 27, 2*7 = 14, 1*4 = 4 and 4 has only one digit)
999 --> 4 (because 9*9*9 = 729, 7*2*9 = 126, 1*2*6 = 12, and finally 1*2 = 2)
4 --> 0 (because 4 is already a one-digit number)
"""


def persistence(n):
    count = 0
    while len(str(n)) > 1:
        multi = 1
        for digit in str(n):
            multi *= int(digit)
        n = multi
        count += 1
    return count
persistence(39)
persistence(999)

import numpy as np
def persistence(n):
    ind = 0
    while len(str(n)) != 1:
        n = np.product([int(i) for i in str(n)])
        ind += 1
    return ind

from itertools import reduce # stopped working
from functools import reduce
def persistence(n):
    count = 0
    while len(str(n)) > 1:
        count += 1
        n = reduce(lambda x, y: x * y, (int(i) for i in str(n)))
    return count


@test.describe("Persistent Bugger.")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(persistence(39), 3)
        test.assert_equals(persistence(4), 0)
        test.assert_equals(persistence(25), 2)
        test.assert_equals(persistence(999), 4)





# Number of People in the Bus
# https://www.codewars.com/kata/5648b12ce68d9daa6b000099
"""There is a bus moving in the city, and it takes and drop some people in each bus stop.

You are provided with a list (or array) of integer pairs. Elements of each pair represent number of people get into bus (The first item) and number of people get off the bus (The second item) in a bus stop.

Your task is to return number of people who are still in the bus after the last bus station (after the last array). Even though it is the last bus stop, the bus is not empty and some people are still in the bus, and they are probably sleeping there :D

Take a look on the test cases.

Please keep in mind that the test cases ensure that the number of people in the bus is always >= 0. So the return integer can't be negative.

The second value in the first integer array is 0, since the bus is empty in the first bus stop."""


def number(bus_stops):
    return sum(enter - exit for enter, exit in bus_stops)
number([[10, 0], [3, 5], [5, 8]])

def number(bus_stops):
    return sum(enter for enter, _ in bus_stops) - sum(exit for _, exit in bus_stops)
number([[10, 0], [3, 5], [5, 8]])

def number(bus_stops):
    return sum(map(lambda x: x[0] - x[1], bus_stops))
number([[10, 0], [3, 5], [5, 8]])

import numpy as np
def number(bus_stops):
    bus_stops = np.array(bus_stops)
    return np.sum(bus_stops[:, 0]) - np.sum(bus_stops[:, 1])
number([[10, 0], [3, 5], [5, 8]])

import numpy as np
def number(bus_stops):
    return np.subtract.reduce(np.sum((bus_stops), axis=0))
number([[10, 0], [3, 5], [5, 8]])


@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(number([[10,0],[3,5],[5,8]]),5)
        test.assert_equals(number([[3,0],[9,1],[4,10],[12,2],[6,1],[7,10]]),17)
        test.assert_equals(number([[3,0],[9,1],[4,8],[12,2],[6,1],[7,8]]),21)





# Count of positives / sum of negatives
# https://www.codewars.com/kata/576bb71bbbcf0951d5000044/train/python
"""Given an array of integers.

Return an array, where the first element is the count of positives numbers and the second element is sum of negative numbers. 0 is neither positive nor negative.

If the input is an empty array or is null, return an empty array.

Example
For input [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -11, -12, -13, -14, -15], you should return [10, -65]."""


def count_positives_sum_negatives(arr):
    if not arr:
        return [] 
    return [len(list(filter(lambda x: x > 0, arr))), sum(filter(lambda x: x < 0, arr))] if arr else []
count_positives_sum_negatives([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -11, -12, -13, -14, -15])

def count_positives_sum_negatives(arr):
    if not arr:
        return []
    # return [sum(True for i in arr if i > 0), sum(i for i in arr if i < 0)]
    return [sum(i > 0 for i in arr), sum(i for i in arr if i < 0)]

def count_positives_sum_negatives(arr):
    if not arr:
        return []
    pos, neg = 0, 0
    for i in arr:
        if i > 0:
            pos += 1
        else:
            neg += i # can add 0 to negatives
    return [pos, neg]

# positives are always first
def count_positives_sum_negatives(arr):
    if not arr:
        return []
    pos = 0
    for i in arr:
        if i > 0:
            pos += 1
        else:
            break
    return [pos, sum(arr[pos:])]


import codewars_test as test
from solution import count_positives_sum_negatives

@test.describe("Fixed Tests")
def basic_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(count_positives_sum_negatives([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -11, -12, -13, -14, -15]),[10,-65])
        test.assert_equals(count_positives_sum_negatives([0, 2, 3, 0, 5, 6, 7, 8, 9, 10, -11, -12, -13, -14]),[8,-50])
        test.assert_equals(count_positives_sum_negatives([1]),[1,0])
        test.assert_equals(count_positives_sum_negatives([-1]),[0,-1])
        test.assert_equals(count_positives_sum_negatives([0,0,0,0,0,0,0,0,0]),[0,0])
        test.assert_equals(count_positives_sum_negatives([]),[])





# String repeat
# https://www.codewars.com/kata/54c27a33fb7da0db0100040e/solutions/python
"""A square of squares
You like building blocks. You especially like building blocks that are squares. And what you even like more, is to arrange them into a square of square building blocks!

However, sometimes, you can't arrange them into a square. Instead, you end up with an ordinary rectangle! Those blasted things! If you just had a way to know, whether you're currently working in vain… Wait! That's it! You just have to check if your number of building blocks is a perfect square.

Task
Given an integral number, determine if it's a square number:

In mathematics, a square number or perfect square is an integer that is the square of an integer; in other words, it is the product of some integer with itself.

The tests will always use some integral number, so don't worry about that in dynamic typed languages.

Examples
-1  =>  false
 0  =>  true
 3  =>  false
 4  =>  true
25  =>  true
26  =>  false"""


import numpy as np
def is_square(n): 
    return np.sqrt(n).is_integer()
is_square(25)

import numpy as np
def is_square(n):
    if n < 0:
        return False
    return np.sqrt(n).is_integer()
    # return np.power(n, .5).is_integer()

def is_square(n):
    return (n ** 0.5) % 1 == 0

def is_square(n):
    return (n ** 0.5) ** 2 == n


@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(is_square(-1), False, "-1: Negative numbers cannot be square numbers")
        test.assert_equals(is_square( 0), True, "0 is a square number (0 * 0)")
        test.assert_equals(is_square( 3), False, "3 is not a square number")
        test.assert_equals(is_square( 4), True, "4 is a square number (2 * 2)")
        test.assert_equals(is_square(25), True, "25 is a square number (5 * 5)")
        test.assert_equals(is_square(26), False, "26 is not a square number")





# Build a pile of Cubes
# https://www.codewars.com/kata/5592e3bd57b64d00f3000047/train/python
"""Your task is to construct a building which will be a pile of n cubes. The cube at the bottom will have a volume of n^3, the cube above will have volume of (n-1)^3 and so on until the top which will have a volume of 1^3.

You are given the total volume m of the building. Being given m can you find the number n of cubes you will have to build?

The parameter of the function findNb (find_nb, find-nb, findNb, ...) will be an integer m and you have to return the integer n such as n^3 + (n-1)^3 + ... + 1^3 = m if such a n exists or -1 if there is no such n.

Examples:
findNb(1071225) --> 45

findNb(91716553919377) --> -1"""


def find_nb(m):
    cubed, ind = 1, 1
    while cubed < m:
        ind += 1
        cubed += ind ** 3
    return ind if cubed == m else -1
find_nb(1071225)


test.assert_equals(find_nb(4183059834009), 2022)
test.assert_equals(find_nb(24723578342962), -1)
test.assert_equals(find_nb(135440716410000), 4824)
test.assert_equals(find_nb(40539911473216), 3568)
test.assert_equals(find_nb(26825883955641), 3218)





# Beginner Series #2 Clock
# https://www.codewars.com/kata/55f9bca8ecaa9eac7100004a/train/python
"""Clock shows h hours, m minutes and s seconds after midnight.

Your task is to write a function which returns the time since midnight in milliseconds.

Example:
h = 0
m = 1
s = 1

result = 61000
Input constraints:

0 <= h <= 23
0 <= m <= 59
0 <= s <= 59"""


def past(h, m, s):
    return ((((h * 60) + m) * 60) + s) * 1000
past(0, 1, 1)

def past(h, m, s):
    return (s + 60*m + 3600*h) * 1000


@test.describe("Fixed Tests")
def basic_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(past(0,1,1),61000)
        test.assert_equals(past(1,1,1),3661000)
        test.assert_equals(past(0,0,0),0)
        test.assert_equals(past(1,0,1),3601000)
        test.assert_equals(past(1,0,0),3600000)





# Find the next perfect square!
# https://www.codewars.com/kata/56269eb78ad2e4ced1000013/solutions/python        
"""You might know some pretty large perfect squares. But what about the NEXT one?

Complete the findNextSquare method that finds the next integral perfect square after the one passed as a parameter. Recall that an integral perfect square is an integer n such that sqrt(n) is also an integer.

If the parameter is itself not a perfect square then -1 should be returned. You may assume the parameter is non-negative.

Examples:(Input --> Output)

121 --> 144
625 --> 676
114 --> -1 since 114 is not a perfect square"""


def find_next_square(sq):
    return int(np.square(np.sqrt(sq) + 1)) if np.sqrt(sq).is_integer() else -1
find_next_square(121)

def find_next_square(sq):
    root_test = (sq ** .5) % 1 == 0
    return ((sq ** .5) + 1) ** 2 if root_test else -1


@test.describe("Fixed Tests")
def fixed_tests():
    @test.it("should return the next square for perfect squares")
    def basic_test_cases():
        test.assert_equals(find_next_square(121), 144, "Wrong output for 121")
        test.assert_equals(find_next_square(625), 676, "Wrong output for 625")
        test.assert_equals(find_next_square(319225), 320356, "Wrong output for 319225")
        test.assert_equals(find_next_square(15241383936), 15241630849, "Wrong output for 15241383936")

    @test.it("should return -1 for numbers which aren't perfect squares")
    def _():
        test.assert_equals(find_next_square(155), -1, "Wrong output for 155")
        test.assert_equals(find_next_square(342786627), -1, "Wrong output for 342786627")





# Find the unique number
# https://www.codewars.com/kata/585d7d5adb20cf33cb000235/train/python
"""There is an array with some numbers. All numbers are equal except for one. Try to find it!

find_uniq([ 1, 1, 1, 2, 1, 1 ]) == 2
find_uniq([ 0, 0, 0.55, 0, 0 ]) == 0.55
It's guaranteed that array contains at least 3 numbers.

The tests contain some very huge arrays, so think about performance."""


def find_uniq(arr):
    return min(set(arr), key=arr.count)  # with 'set' computes much faster
find_uniq([1, 1, 1, 2, 1, 1])
find_uniq([3, 10, 3, 3, 3])

from collections import Counter
def find_uniq(arr):
    return min(set(arr), key=Counter(arr).get)

from collections import Counter
def find_uniq(arr):
    return Counter(arr).most_common()[-1][0]

from collections import Counter
def find_uniq(arr):
    count_dict = dict(Counter(arr))
    return sorted(count_dict, key=count_dict.get)[0]


@test.describe("Basic Tests")
def f():
    @test.it("Simple tests")
    def _():
        test.assert_equals(find_uniq([ 1, 1, 1, 2, 1, 1 ]), 2)
        test.assert_equals(find_uniq([ 0, 0, 0.55, 0, 0 ]), 0.55)
        test.assert_equals(find_uniq([ 3, 10, 3, 3, 3 ]), 10)





# Descending Order
# https://www.codewars.com/kata/5467e4d82edf8bbf40000155/train/python
"""Your task is to make a function that can take any non-negative integer as an argument and return it with its digits in descending order. Essentially, rearrange the digits to create the highest possible number.

Examples:
Input: 42145 Output: 54421

Input: 145263 Output: 654321

Input: 123456789 Output: 987654321"""


def descending_order(num):
    return int(''.join(sorted(str(num), reverse=True)))
descending_order(123456789)


@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(descending_order(0), 0)
        test.assert_equals(descending_order(15), 51)
        test.assert_equals(descending_order(123456789), 987654321)





# Duplicate Encoder
# https://www.codewars.com/kata/54b42f9314d9229fd6000d9c/train/python
"""The goal of this exercise is to convert a string to a new string where each character in the new string is "(" if that character appears only once in the original string, or ")" if that character appears more than once in the original string. Ignore capitalization when determining if a character is a duplicate.

Examples
"din"      =>  "((("
"recede"   =>  "()()()"
"Success"  =>  ")())())"
"(( @"     =>  "))((" """


def duplicate_encode(word):
    return ''.join(['(' if word.lower().count(char) == 1 else ')' for char in word.lower()])
duplicate_encode('Success')

def duplicate_encode(word):
    return "".join(")" if word.lower().count(letter) > 1 else "(" for letter in word.lower())

def duplicate_encode(word):
    return ''.join(map(lambda x: '(' if Counter(word.lower())[x] == 1 else ')', word.lower()))
duplicate_encode("din")

def duplicate_encode(word):
    multiple_appear = {letter: (word.lower().count(letter) > 1) for letter in set(word.lower())}
    return "".join(")" if multiple_appear[letter] else "(" for letter in word.lower())

from collections import Counter
def duplicate_encode(word):
    return ''.join('(' if Counter(word.lower())[char] == 1 else ')' for char in word.lower())
duplicate_encode("din")
duplicate_encode("recede")
duplicate_encode("Success")
duplicate_encode("(( @")


@test.describe("Duplicate Encoder")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(duplicate_encode("din"),"(((")
        test.assert_equals(duplicate_encode("recede"),"()()()")
        test.assert_equals(duplicate_encode("Success"),")())())","should ignore case")
        test.assert_equals(duplicate_encode("(( @"),"))((")





# Counting sheep...
# https://www.codewars.com/kata/54edbc7200b811e956000556/train/python
"""Consider an array/list of sheep where some sheep may be missing from their place. We need a function that counts the number of sheep present in the array (true means present).

For example,

[True,  True,  True,  False,
  True,  True,  True,  True ,
  True,  False, True,  False,
  True,  False, False, True ,
  True,  True,  True,  True ,
  False, False, True,  True]
The correct answer would be 17.

Hint: Don't forget to check for bad values like null/undefined"""


import numpy as np
# changed first two True's to something more instructive
array1 = [None,  np.NaN,  True,  False,
          True,  True,  True,  True ,
          True,  False, True,  False,
          True,  False, False, True ,
          True,  True,  True,  True ,
          False, False, True,  True ]

def count_sheeps(sheep):
    return sheep.count(True)
    # return sum(i == True for i in sheep)
    # return len([i for i in sheep if i == True])
    # return sheep.count(1)
    # return [i for i in sheep if i == True].count(True)
    # return list(map(lambda x: True if x == True else False, sheep)).count(True)
count_sheeps(array1)

import numpy as np
def count_sheeps(sheep):
    None_list = np.array(sheep)
    Not_None_list = None_list[None_list != None]
    return np.nansum(Not_None_list)

def count_sheeps(sheep):
    return np.nansum(list(filter(bool, sheep)))
    return np.nansum(list(filter(None, sheep)))

              
test.assert_equals(result := count_sheeps(array1), 17, "There are 17 sheeps in total, not %s" % result)





# Highest and Lowest
# https://www.codewars.com/kata/554b4ac871d6813a03000035
"""In this little assignment you are given a string of space separated numbers, and have to return the highest and lowest number.

Examples
high_and_low("1 2 3 4 5")  # return "5 1"
high_and_low("1 2 -3 4 5") # return "5 -3"
high_and_low("1 9 3 4 -5") # return "9 -5"
Notes
All numbers are valid Int32, no need to validate them.
There will always be at least one number in the input string.
Output string must be two numbers separated by a single space, and highest number is first."""


def high_and_low(numbers):
    num_list = [int(number) for number in numbers.split(' ')]
    # return ' '.join([str(max(num_list)), str(min(num_list))])
    return '{} {}'.format(max(num_list), min(num_list))
high_and_low('8 3 -5 42 -1 0 0 -9 4 7 4 -4')

def high_and_low(numbers):
    return ' '.join((str(max(int(i) for i in numbers.split(' '))), str(min(int(i) for i in numbers.split(' ')))))

def high_and_low(numbers):
    list_of_digits = list(int(digit) for digit in numbers.strip().split())
    return str(max(list_of_digits)) + " " + str(min(list_of_digits))

sorted(('8 3 -5 42 -1 0 0 -9 4 7 4 -4').split(), key=int)


@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(high_and_low("8 3 -5 42 -1 0 0 -9 4 7 4 -4"), "42 -9");
        test.assert_equals(high_and_low("1 2 3"), "3 1");





# Your order, please
# https://www.codewars.com/kata/55c45be3b2079eccff00010f/solutions/python
"""Your task is to sort a given string. Each word in the string will contain a single number. This number is the position the word should have in the result.

Note: Numbers can be from 1 to 9. So 1 will be the first word (not 0).

If the input string is empty, return an empty string. The words in the input String will only contain valid consecutive numbers.

Examples
"is2 Thi1s T4est 3a"  -->  "Thi1s is2 3a T4est"
"4of Fo1r pe6ople g3ood th5e the2"  -->  "Fo1r the2 g3ood 4of th5e pe6ople"
""  -->  """""


def order(sentence):
    return " ".join(sorted(sentence.split(), key=min))
order("is2 Thi1s T4est 3a")

def order(sentence):
    return " ".join(sorted(sentence.split(), key=sorted))

def order(sentence):
    return " ".join(sorted(sentence.split(), key=lambda x: sorted(x)))

def order(sentence):
    return " ".join(sorted(sentence.split(), key=lambda x: "".join(filter(str.isdigit, x))))

def order(sentence):
    return " ".join(sorted(sentence.split(), key=lambda x: "".join(i for i in x if i.isdigit())))

from string import digits
def order(sentence):
    return " ".join(sorted(sentence.split(), key=lambda x: ''.join(filter(lambda y: y in digits, x))))

import re
def order(sentence):
    return sorted(sentence.split(), key=lambda x: ''.join(re.findall(r'\d+', x)))
order("is2 Thi1s T4est 3a")

def order(sentence):
    sentence_dict = {int(re.search(r'\d+', i).group()): i for i in sentence.split()}
    return ' '.join(sentence_dict[i] for i in range(1, len(sentence_dict) + 1))
order('is2 Thi1s T4est 3a')

def order(sentence):
    sentence_dict = {int(''.join(filter(str.isdigit, i))): i for i in sentence.split()}
    return ' '.join([sentence_dict[i] for i in range(1, len(sentence_dict) + 1)])


test.assert_equals(order("is2 Thi1s T4est 3a"), "Thi1s is2 3a T4est")
test.assert_equals(order("4of Fo1r pe6ople g3ood th5e the2"), "Fo1r the2 g3ood 4of th5e pe6ople")
test.assert_equals(order(""), "")





# Century From Year
# https://www.codewars.com/kata/5a3fe3dde1ce0e8ed6000097/train/python
"""Introduction
The first century spans from the year 1 up to and including the year 100, the second century - from the year 101 up to and including the year 200, etc.

Task
Given a year, return the century it is in.

Examples
1705 --> 18
1900 --> 19
1601 --> 17
2000 --> 20"""


def century(year):
    ending = int(str(year)[-2:])
    core = int(str(year)[-4:-2]) if str(year)[-4:-2] else 0
    return core + 1 if ending else core
century(1705)
century(1900)
century(1601)
century(2000)
century(356)
century(89)
century(0)

def century(year):
    return year // 100 + 1 if year % 100 else year // 100

import numpy as np
def century(year):
    return int(np.ceil(year / 100))

def century(year):
    return (year + 99) // 100

def century(year):
    if not year:
        return 0
    if (len(str(year)) == 2 or
        len(str(year)) == 1):
        return 1
    return int(str(year)[:-2]) if int(str(year)[-2:]) == 0 else int(str(year)[:-2])+1 
    # return int(str(year)[:-2])
century(1705)


@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(century(1705), 18, 'Testing for year 1705')
        test.assert_equals(century(1900), 19, 'Testing for year 1900')
        test.assert_equals(century(1601), 17, 'Testing for year 1601')
        test.assert_equals(century(2000), 20, 'Testing for year 2000')
        test.assert_equals(century(356), 4, 'Testing for year 356')
        test.assert_equals(century(89), 1, 'Testing for year 89')





# Reversed Strings
# https://www.codewars.com/kata/5168bb5dfe9a00b126000018/solutions/python
"""Complete the solution so that it reverses the string passed into it.

'world'  =>  'dlrow'
'word'   =>  'drow'"""


def solution(string):
    return ''.join(reversed(string))
solution('world')

def solution(string):
    return string[::-1]


@test.describe("Fixed Tests")
def basic_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(solution('world'), 'dlrow')
        test.assert_equals(solution('hello'), 'olleh')
        test.assert_equals(solution(''), '')
        test.assert_equals(solution('h'), 'h')
    




# Calculating with Functions
# https://www.codewars.com/kata/525f3eda17c7cd9f9e000b39/python
# Fist fail, WTF
"""This time we want to write calculations using functions and get the results. Let's have a look at some examples:

seven(times(five())) # must return 35
four(plus(nine())) # must return 13
eight(minus(three())) # must return 5
six(divided_by(two())) # must return 3
Requirements:

There must be a function for each number from 0 ("zero") to 9 ("nine")
There must be a function for each of the following mathematical operations: plus, minus, times, divided_by
Each calculation consist of exactly one operation and two numbers
The most outer function represents the left operand, the most inner function represents the right operand
Division should be integer division. For example, this should return 2, not 2.666666...:
eight(divided_by(three()))"""


def zero(func=None):
    if func:
        return func(0)
    return 0
def one(func=None):
    if func:
        return func(1)
    return 1
def two(func=None):
    if func:
        return func(2)
    return 2
def three(func=None):
    if func:
        return func(3)
    return 3
def four(func=None):
    if func:
        return func(4)
    return 4
def five(func=None):
    if func:
        return func(5)
    return 5
def six(func=None):
    if func:
        return func(6)
    return 6
def seven(func=None):
    if func:
        return func(7)
    return 7
def eight(func=None):
    if func:
        return func(8)
    return 8
def nine(func=None):
    if func:
        return func(9)
    return 9

def plus(x):
    return lambda y: x + y
def minus(x):
    return lambda y: y - x
def times(x):
    return lambda y: x * y    
def divided_by(x):
    return lambda y: y // x

def zero(f=None): 
    return 0 if not f else f(0)
def one(f=None): 
    return 1 if not f else f(1)
def two(f=None): 
    return 2 if not f else f(2)
def three(f=None): 
    return 3 if not f else f(3)
def four(f=None): 
    return 4 if not f else f(4)
def five(f=None): 
    return 5 if not f else f(5)
def six(f=None): 
    return 6 if not f else f(6)
def seven(f=None): 
    return 7 if not f else f(7)
def eight(f=None): 
    return 8 if not f else f(8)
def nine(f=None): 
    return 9 if not f else f(9)


def plus(y):
    return lambda x: x+y
def minus(y):
    return lambda x: x-y
def times(y):
    return lambda x: x*y
def divided_by(y):
    return lambda x: x/y

seven(times(five()))
one(plus(seven(times(five()))))


id_ = lambda x: x
number = lambda x: lambda f=id_: f(x)
zero, one, two, three, four, five, six, seven, eight, nine = map(number, range(10))
plus = lambda x: lambda y: y + x
minus = lambda x: lambda y: y - x
times = lambda x: lambda y: y * x
divided_by = lambda x: lambda y: y / x


test.describe('Basic Tests')
test.assert_equals(seven(times(five())), 35)
test.assert_equals(four(plus(nine())), 13)
test.assert_equals(eight(minus(three())), 5)
test.assert_equals(six(divided_by(two())), 3)





# You only need one - Beginner
# https://www.codewars.com/kata/57cc975ed542d3148f00015b
"""You will be given an array a and a value x. All you need to do is check whether the provided array contains the value.

Array can contain numbers or strings. X can be either.

Return true if the array contains the value, false if not."""


def check(seq, elem):
    return elem in seq
check([66, 101], 66)
check(['t', 'e', 's', 't'], 'e')

def check(seq, elem):
    # return any([i == elem for i in seq])
    return any([i for i in seq if i == elem])


@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        tests = [
            (True, ([66, 101], 66)),
            (False, ([78, 117, 110, 99, 104, 117, 107, 115], 8)),
            (True, ([101, 45, 75, 105, 99, 107], 107)),
            (True, ([80, 117, 115, 104, 45, 85, 112, 115], 45)),
            (True, (['t', 'e', 's', 't'], 'e')),
            (False, (["what", "a", "great", "kata"], "kat")),
            (True, ([66, "codewars", 11, "alex loves pushups"], "alex loves pushups")),
            (False, (["come", "on", 110, "2500", 10, '!', 7, 15], "Come")),
            (True, (["when's", "the", "next", "Katathon?", 9, 7], "Katathon?")),
            (False, ([8, 7, 5, "bored", "of", "writing", "tests", 115], 45)),
            (True, (["anyone", "want", "to", "hire", "me?"], "me?")),
        ]





# Count the smiley faces!
# https://www.codewars.com/kata/583203e6eb35d7980400002a
"""Given an array (arr) as an argument complete the function countSmileys that should return the total number of smiling faces.

Rules for a smiling face:

Each smiley face must contain a valid pair of eyes. Eyes can be marked as : or ;
A smiley face can have a nose but it does not have to. Valid characters for a nose are - or ~
Every smiling face must have a smiling mouth that should be marked with either ) or D
No additional characters are allowed except for those mentioned.

Valid smiley face examples: :) :D ;-D :~)
Invalid smiley faces: ;( :> :} :]

Example
countSmileys([':)', ';(', ';}', ':-D']);       // should return 2;
countSmileys([';D', ':-(', ':-)', ';~)']);     // should return 3;
countSmileys([';]', ':[', ';*', ':$', ';-D']); // should return 1;
Note
In case of an empty array return 0. You will not be tested with invalid input (input will always be an array). Order of the face (eyes, nose, mouth) elements will always be the same.

"""


import re
def count_smileys(arr):
    return sum(True for elem in arr if re.match(r"[:;][-~]?[\)D]", elem))
    # return sum(True for i in arr if re.match(r'^[;:][~-]?[\)D]$', i))
    # return sum(True for i in arr if re.search(r'^[;:][~-]?[\)D]$', i))
    # return sum(True for i in arr if re.findall(r'^[;:][~-]?[\)D]$', i))
    # return sum(True if re.match(r"[:;][-~]?[\)D]", elem) else False for elem in arr)
count_smileys([':)', ';(', ';}', ':-D'])
count_smileys([';D', ':-(', ':-)', ';~)'])
count_smileys([';]', ':[', ';*', ':$', ';-D'])
count_smileys([':D',':~)',';~D',':)'])

import re
def count_smileys(arr):
    return len(re.findall(r'[;:][~-]?[\)D]', ' '.join(arr)))

def count_smileys(arr):
    if not arr:
        return 0
    count = 0
    for i in arr:
        if i[0] in [':', ';']:
            if i[-1] in [')', 'D']:
                if len(i) == 2:
                    count += 1
                elif i[1] in ['-', '~']:
                    count += 1
                else:
                    continue
    return count

test.describe("Basic tests")
test.assert_equals(count_smileys([]), 0)
test.assert_equals(count_smileys([':D',':~)',';~D',':)']), 4)
test.assert_equals(count_smileys([':)',':(',':D',':O',':;']), 2)
test.assert_equals(count_smileys([';]', ':[', ';*', ':$', ';-D']), 1)





# Consecutive strings
# https://www.codewars.com/kata/56a5d994ac971f1ac500003e
"""You are given an array(list) strarr of strings and an integer k. Your task is to return the first longest string consisting of k consecutive strings taken in the array.

Examples:
strarr = ["tree", "foling", "trashy", "blue", "abcdef", "uvwxyz"], k = 2

Concatenate the consecutive strings of strarr by 2, we get:

treefoling   (length 10)  concatenation of strarr[0] and strarr[1]
folingtrashy ("      12)  concatenation of strarr[1] and strarr[2]
trashyblue   ("      10)  concatenation of strarr[2] and strarr[3]
blueabcdef   ("      10)  concatenation of strarr[3] and strarr[4]
abcdefuvwxyz ("      12)  concatenation of strarr[4] and strarr[5]

Two strings are the longest: "folingtrashy" and "abcdefuvwxyz".
The first that came is "folingtrashy" so 
longest_consec(strarr, 2) should return "folingtrashy".

In the same way:
longest_consec(["zone", "abigail", "theta", "form", "libe", "zas", "theta", "abigail"], 2) --> "abigailtheta"
n being the length of the string array, if n = 0 or k > n or k <= 0 return "" (return Nothing in Elm)."""


def longest_consec(strarr, k):
    # if not (k in range(1, len(strarr) + 1) and strarr):
    if not (0 < k <= len(strarr) and strarr):
        return ""
    return max(["".join(strarr[i:i+k]) for i in range(len(strarr) - k + 1)], key=len)
    # return max(["".join(strarr[i:i+k]) for i in range(len(strarr) - k + 1)], key=len) if k in range(1, len(strarr) + 1) and strarr else ""
longest_consec(["tree", "foling", "trashy", "blue", "abcdef", "uvwxyz"], 2)
longest_consec(["zone", "abigail", "theta", "form", "libe", "zas"], 2)

def longest_consec(strarr, k):
    if (not strarr or
        k > len(strarr) or
        k <= 0):
        return ''
    return max((''.join(strarr[i : i+k]) for i in range(len(strarr) - k + 1)), key=len)

# change next to list for all strings
def longest_consec(strarr, k):
    if (not strarr or
        k > len(strarr) or
        k <= 0):
        return ''
    concat_list = [''.join(strarr[i : i+k]) for i in range(len(strarr) - k + 1)]
    return next(filter(lambda x: len(x) == len(max(concat_list, key=len)), concat_list))


test.describe("longest_consec")
test.it("Basic tests")
testing(longest_consec(["zone", "abigail", "theta", "form", "libe", "zas"], 2), "abigailtheta")
testing(longest_consec(["ejjjjmmtthh", "zxxuueeg", "aanlljrrrxx", "dqqqaaabbb", "oocccffuucccjjjkkkjyyyeehh"], 1), "oocccffuucccjjjkkkjyyyeehh")
testing(longest_consec([], 3), "")
testing(longest_consec(["itvayloxrp","wkppqsztdkmvcuwvereiupccauycnjutlv","vweqilsfytihvrzlaodfixoyxvyuyvgpck"], 2), "wkppqsztdkmvcuwvereiupccauycnjutlvvweqilsfytihvrzlaodfixoyxvyuyvgpck")
testing(longest_consec(["wlwsasphmxx","owiaxujylentrklctozmymu","wpgozvxxiu"], 2), "wlwsasphmxxowiaxujylentrklctozmymu")
testing(longest_consec(["zone", "abigail", "theta", "form", "libe", "zas"], -2), "")
testing(longest_consec(["it","wkppv","ixoyx", "3452", "zzzzzzzzzzzz"], 3), "ixoyx3452zzzzzzzzzzzz")
testing(longest_consec(["it","wkppv","ixoyx", "3452", "zzzzzzzzzzzz"], 15), "")
testing(longest_consec(["it","wkppv","ixoyx", "3452", "zzzzzzzzzzzz"], 0), "")





# Build Tower
# https://www.codewars.com/kata/576757b1df89ecf5bd00073b/
"""Build Tower
Build a pyramid-shaped tower given a positive integer number of floors. A tower block is represented with "*" character.

For example, a tower with 3 floors looks like this:

[
  "  *  ",
  " *** ", 
  "*****"
]"""


def tower_builder(n):
    return [("*" * (2*i + 1)).center(2*n - 1) for i in range(n)]
tower_builder(3)

def tower_builder(n):
    return [" " * (n - i - 1) + "*" * (2*i + 1) + " " * (n - i - 1) for i in range(n)]

def tower_builder(n):
    return ["{:^{}}".format("*" * (2*i + 1), (2*n - 1)) for i in range(n)]


@test.describe("Build Tower")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(tower_builder(1), ['*', ])
        test.assert_equals(tower_builder(2), [' * ', '***'])
        test.assert_equals(tower_builder(3), ['  *  ', ' *** ', '*****'])





# Sum without highest and lowest number
# https://www.codewars.com/kata/576b93db1129fcf2200001e6
"""Task
Sum all the numbers of a given array ( cq. list ), except the highest and the lowest element ( by value, not by index! ).

The highest or lowest element respectively is a single element at each edge, even if there are more than one with the same value.

Mind the input validation.

Example
{ 6, 2, 1, 8, 10 } => 16
{ 1, 1, 11, 2, 3 } => 6
Input validation
If an empty value ( null, None, Nothing etc. ) is given instead of an array, or the given array is an empty list or a list with only 1 element, return 0."""


def sum_array(arr):
    if (not arr or 
        len(arr) == 1 or 
        len(arr) == 2):
        return 0
    return sum(arr) - min(arr) - max(arr)
sum_array([3, 5])
sum_array([-3, -5])
sum_array([6, 2, 1, 8, 10])
sum_array([1, 1, 11, 2, 3])
sum_array([-6, 20, -1, 10, -12])

def sum_array(arr):
    if (not arr or
        len(arr) == 1):
        return 0
    return sum(sorted(arr)[1:-1])


@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('None or Empty')
    def basic_test_cases():
        test.assert_equals(sum_array(None), 0)
        test.assert_equals(sum_array([]), 0)

    @test.it("Only one Element")
    def one_test_cases():
        test.assert_equals(sum_array([3]), 0)
        test.assert_equals(sum_array([-3]), 0)
        
    @test.it("Only two Element")
    def two_test_cases():
        test.assert_equals(sum_array([ 3, 5]), 0)
        test.assert_equals(sum_array([-3, -5]), 0)

    @test.it("Real Tests")
    def real_test_cases():
        test.assert_equals(sum_array([6, 2, 1, 8, 10]), 16)
        test.assert_equals(sum_array([6, 0, 1, 10, 10]), 17)
        test.assert_equals(sum_array([-6, -20, -1, -10, -12]), -28)
        test.assert_equals(sum_array([-6, 20, -1, 10, -12]), 3)




# Are You Playing Banjo?
# https://www.codewars.com/kata/53af2b8861023f1d88000832
"""Create a function which answers the question "Are you playing banjo?".
If your name starts with the letter "R" or lower case "r", you are playing banjo!

The function takes a name as its only argument, and returns one of the following strings:

name + " plays banjo" 
name + " does not play banjo"
"""


def are_you_playing_banjo(name):
    return name + " plays banjo" if name[0].lower() == "r" else name + " does not play banjo"

def are_you_playing_banjo(name):
    return '{} plays banjo'.format(name) if name[0].lower() == 'r' else '{} does not play banjo'.format(name)
are_you_playing_banjo('rartin')


@test.describe("Fixed Tests")
def basic_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(are_you_playing_banjo("martin"), "martin does not play banjo");
        test.assert_equals(are_you_playing_banjo("Rikke"), "Rikke plays banjo");
        test.assert_equals(are_you_playing_banjo("bravo"), "bravo does not play banjo")
        test.assert_equals(are_you_playing_banjo("rolf"), "rolf plays banjo")





# Remove the minimum
# https://www.codewars.com/kata/563cf89eb4747c5fb100001b
"""Task
Given an array of integers, remove the smallest value. Do not mutate the original array/list. If there are multiple elements with the same value, remove the one with a lower index. If you get an empty array/list, return an empty array/list.

Don't change the order of the elements that are left.

Examples
* Input: [1,2,3,4,5], output= [2,3,4,5]
* Input: [5,3,2,1,4], output = [5,3,2,4]
* Input: [2,2,1,2,1], output = [2,2,2,1]"""


def remove_smallest(numbers):
    return numbers[:numbers.index(min(numbers))] + numbers[numbers.index(min(numbers)) + 1 :] if numbers else []

def remove_smallest(numbers):
    if not numbers:
        return []
    num = numbers[:]
    num.remove(min(numbers[:]))
    return num
remove_smallest([1, 2, 3, 4, 5])
remove_smallest([5, 3, 2, 1, 4])
remove_smallest([2, 2, 1, 2, 1])
remove_smallest([])


def remove_smallest(numbers):
    new_numbers = numbers[:]
    if numbers:
        new_numbers.remove(min(new_numbers))
    return new_numbers


test.it("works for the examples")
test.assert_equals(remove_smallest([1, 2, 3, 4, 5]), [
                   2, 3, 4, 5], "Wrong result for [1, 2, 3, 4, 5]")
test.assert_equals(remove_smallest([5, 3, 2, 1, 4]), [
                   5, 3, 2, 4], "Wrong result for [5, 3, 2, 1, 4]")
test.assert_equals(remove_smallest([1, 2, 3, 1, 1]), [
                   2, 3, 1, 1], "Wrong result for [1, 2, 3, 1, 1]")
test.assert_equals(remove_smallest([]), [], "Wrong result for []")


def randlist():
    return list(randint(400, size=randint(1, 10)))


test.it("returns [] if list has only one element")
for i in range(10):
    x = randint(1, 400)
    test.assert_equals(remove_smallest(
        [x]), [], "Wrong result for [{}]".format(x))

test.it("returns a list that misses only one element")
for i in range(10):
    arr = randlist()
    test.assert_equals(len(remove_smallest(arr[:])), len(
        arr) - 1, "Wrong sized result for {}".format(arr))





# Two to One
# https://www.codewars.com/kata/5656b6906de340bd1b0000ac
"""Take 2 strings s1 and s2 including only letters from ato z. Return a new sorted string, the longest possible, containing distinct letters - each taken only once - coming from s1 or s2.

Examples:
a = "xyaabbbccccdefww"
b = "xxxxyyyyabklmopq"
longest(a, b) -> "abcdefklmopqwxy"

a = "abcdefghijklmnopqrstuvwxyz"
longest(a, a) -> "abcdefghijklmnopqrstuvwxyz""""


def longest(a1, a2):
    return ''.join(sorted(set(a1) | set(a2)))
longest('xyaabbbccccdefww', 'xxxxyyyyabklmopq')
longest('abcdefghijklmnopqrstuvwxyz', 'abcdefghijklmnopqrstuvwxyz')
longest("aretheyhere", "yestheyarehere")

def longest(a1, a2):
    return ''.join(sorted(set(a1 + a2)))


@test.describe("longest")
def tests():
    @test.it("basic tests")
    def basics():
        test.assert_equals(longest("aretheyhere", "yestheyarehere"), "aehrsty")
        test.assert_equals(longest("loopingisfunbutdangerous", "lessdangerousthancoding"), "abcdefghilnoprstu")
        test.assert_equals(longest("inmanylanguages", "theresapairoffunctions"), "acefghilmnoprstuy")





# Unique In Order
# https://www.codewars.com/kata/54e6533c92449cc251001667
"""Implement the function unique_in_order which takes as argument a sequence and returns a list of items without any elements with the same value next to each other and preserving the original order of elements.

For example:

unique_in_order('AAAABBBCCDAABBB') == ['A', 'B', 'C', 'D', 'A', 'B']
unique_in_order('ABBCcAD')         == ['A', 'B', 'C', 'c', 'A', 'D']
unique_in_order([1,2,2,3,3])       == [1,2,3]"""

def unique_in_order(iterable):
    if not iterable:
        return []
    unique_list = [iterable[i] for i in range(len(iterable) - 1) if iterable[i] != iterable[i+1]]
    unique_list.append(iterable[-1])
    return unique_list
unique_in_order('AAAABBBCCDAABBB')
unique_in_order('ABBCcAD')
unique_in_order([1, 2, 2, 3, 3])

def unique_in_order(iterable):
    if not iterable:
        return []
    unique_list = [iterable[i] for i in range(1, len(iterable)) if iterable[i] != iterable[i - 1]]
    unique_list.insert(0, iterable[0])
    return unique_list

from itertools import groupby
def unique_in_order(iterable):
    return [k for k, _ in groupby(iterable)]

def unique_in_order(iterable):
    if not iterable:
        return []
    taboo = []
    taboo.append(iterable[0])
    for i in iterable[1:]:
        if i != taboo[-1]:
            taboo.append(i)
    return taboo


test.assert_equals(unique_in_order('AAAABBBCCDAABBB'), ['A','B','C','D','A','B'])





# Bouncing Balls
# https://www.codewars.com/kata/5544c7a5cb454edb3c000047
"""A child is playing with a ball on the nth floor of a tall building. The height of this floor, h, is known.

He drops the ball out of the window. The ball bounces (for example), to two-thirds of its height (a bounce of 0.66).

His mother looks out of a window 1.5 meters from the ground.

How many times will the mother see the ball pass in front of her window (including when it's falling and bouncing?

Three conditions must be met for a valid experiment:
Float parameter "h" in meters must be greater than 0
Float parameter "bounce" must be greater than 0 and less than 1
Float parameter "window" must be less than h.
If all three conditions above are fulfilled, return a positive integer, otherwise return -1.

Note:
The ball can only be seen if the height of the rebounding ball is strictly greater than the window parameter.

Examples:
- h = 3, bounce = 0.66, window = 1.5, result is 3

- h = 3, bounce = 1, window = 1.5, result is -1 

(Condition 2) not fulfilled)."""


def bouncing_ball(h, bounce, window):
    # if (h * bounce < window or
    #     h <= 0 or
    #     bounce <= 0 or
    #     bounce >= 1 or
    #     window >= h):
    #         return -1
    if not (h > 0 
            and 0 < bounce < 1 
            and window < h):
        return -1
    counter = 1
    while h * bounce > window:
        counter += 2
        h *= bounce
    return counter
bouncing_ball(3, 0.66, 1.5)
bouncing_ball(2, 0.5, 1)
bouncing_ball(2, 1, 1)

def bouncing_ball(h, bounce, window):
    if not (h > 0 
            and 0 < bounce < 1 
            and window < h):
        return -1
    return 2 + bouncing_ball(bounce * h, bounce, window)



@test.it('Fixed Tests')
def tests():
    testing(2, 0.5, 1, 1)
    testing(3, 0.66, 1.5, 3)
    testing(30, 0.66, 1.5, 15)
    testing(30, 0.75, 1.5, 21)





# Sum of positive
# https://www.codewars.com/kata/5715eaedb436cf5606000381
"""You get an array of numbers, return the sum of all of the positives ones.

Example [1,-4,7,12] => 1 + 7 + 12 = 20

Note: if there is nothing to sum, the sum is default to 0."""


def positive_sum(arr):
    return sum(filter(lambda x: x > 0, arr))
positive_sum([1, -4, 7, 12])

def positive_sum(arr):
    return sum(i for i in arr if i > 0)


@test.describe("positive_sum")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(positive_sum([1,2,3,4,5]),15)
        test.assert_equals(positive_sum([1,-2,3,4,5]),13)
        test.assert_equals(positive_sum([-1,2,3,4,-5]),9)
        
    @test.it("returns 0 when array is empty")
    def empty_case():
        test.assert_equals(positive_sum([]),0)  





# Beginner - Lost Without a Map
# https://www.codewars.com/kata/57f781872e3d8ca2a000007e
"""Given an array of integers, return a new array with each value doubled.

For example:

[1, 2, 3] --> [2, 4, 6]"""


def maps(a):
    return [i * 2 for i in a]
maps([1, 2, 3])

def maps(a):
    return list(map(lambda x: x * 2, a))

import numpy as np
def maps(a):
    return list(2 * np.array(a))


@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(maps([1, 2, 3]), [2, 4, 6])
        test.assert_equals(maps([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), [0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
        test.assert_equals(maps([]), [])





# String ends with?
# https://www.codewars.com/kata/51f2d1cafc9c0f745c00037d/
"""Complete the solution so that it returns true if the first argument(string) passed in ends with the 2nd argument (also a string).

Examples:

solution('abc', 'bc') # returns true
solution('abc', 'd') # returns false"""


def solution(string, ending):
    return string.endswith(ending)
solution('abcde', 'cde')
solution('sumo', 'omo')
solution('samurai', 'ai')
solution('samurai', '')

def solution(text, ending):
    return ending == text[-len(ending):]tex

def solution(string, ending):
    if not ending:
        return True
    return ending == string[-len(ending):]

def solution(string, ending):
    if not ending:
        return True
    return True if ''.join(reversed(string)).find(''.join(reversed(ending))) == 0 and string.find(ending) != -1 else False
solution('abcde', 'cde')
solution('sumo', 'omo')
solution('samurai', 'ai')


test.assert_equals(solution('abcde', 'cde'), True)
test.assert_equals(solution('abcde', 'abc'), False)
test.assert_equals(solution('abcde', ''), True)





# Is this a triangle?
# https://www.codewars.com/kata/56606694ec01347ce800001b
"""Implement a function that accepts 3 integer values a, b, c. The function should return true if a triangle can be built with the sides of given length and false in any other case.

(In this case, all triangles must have surface greater than 0 to be accepted).

Examples:

Input -> Output
1,2,2 -> true
4,2,3 -> true
2,2,2 -> true
1,2,3 -> false
-5,1,3 -> false
0,2,3 -> false
1,2,9 -> false """


def is_triangle(a, b, c):
    a, b, c = sorted((a, b, c))
    return a + b > c
is_triangle(1, 2, 2)
is_triangle(7, 2, 2)
is_triangle(1, 2, 3)
is_triangle(1, 3, 2)
is_triangle(3, 1, 2)
is_triangle(5, 1, 2)
is_triangle(1, 2, 5)
is_triangle(2, 5, 1)
is_triangle(4, 2, 3)
is_triangle(5, 1, 5)
is_triangle(2, 2, 2)

def is_triangle(a, b, c):
    return 2 * max((a, b, c)) < sum((a, b, c))

def is_triangle(a, b, c):
    return (a < b+c) and (b < a+c) and (c < a+b)





# Replace With Alphabet Position
# https://www.codewars.com/kata/546f922b54af40e1e90001da/train/python
"""In this kata you are required to, given a string, replace every letter with its position in the alphabet.

If anything in the text isn't a letter, ignore it and don't return it.

"a" = 1, "b" = 2, etc.

Example
alphabet_position("The sunset sets at twelve o' clock.")
Should return "20 8 5 19 21 14 19 5 20 19 5 20 19 1 20 20 23 5 12 22 5 15 3 12 15 3 11" ( as a string )"""


def alphabet_position(text):
    # text = text.replace(' ', '').lower()
    # return ' '.join(str(ord(i) - 96) for i in text.lower() if i.isalpha())
    return ' '.join(str(ord(i) - ord("a") + 1) for i in text.lower() if i.isalpha())
alphabet_position("The sunset sets at twelve o' clock.")

import string
def alphabet_position(text):
    alph = string.ascii_lowercase
    return " ".join(str(alph.find(i) + 1) for i in text.lower() if i in alph)

import string
def alphabet_position(text):
    text = text.replace(' ', '').lower()
    return ' '.join(str(ord(i) - 96) for i in text if i in string.ascii_letters)

def alphabet_position(text):
    filtered_text = filter(str.isalpha, text)
    return ' '.join(str(ord(i.lower()) - 96) for i in filtered_text)


from random import randint
test.assert_equals(alphabet_position("The sunset sets at twelve o' clock."), "20 8 5 19 21 14 19 5 20 19 5 20 19 1 20 20 23 5 12 22 5 15 3 12 15 3 11")
test.assert_equals(alphabet_position("The narwhal bacons at midnight."), "20 8 5 14 1 18 23 8 1 12 2 1 3 15 14 19 1 20 13 9 4 14 9 7 8 20")

number_test = ""
for item in range(10):
    number_test += str(randint(1, 9))
test.assert_equals(alphabet_position(number_test), "")





# Mexican Wave
# https://www.codewars.com/kata/58f5c63f1e26ecda7e000029
"""Task
In this simple Kata your task is to create a function that turns a string into a Mexican Wave. You will be passed a string and you must return that string in an array where an uppercase letter is a person standing up. 
Rules
 1.  The input string will always be lower case but maybe empty.

 2.  If the character in the string is whitespace then pass over it as if it was an empty seat
Example
wave("hello") => ["Hello", "hEllo", "heLlo", "helLo", "hellO"]"""


def wave(people):
    return [people[:i] + people[i].upper() + people[i+1:] for i in range(len(people)) if (people[:i] + people[i].upper() + people[i+1:]) != people]
wave('hello')
wave('')

def wave(people):
    return [people[:i] + people[i].upper() + people[i+1:] for i in range(len(people)) if people[i].isalpha()]


result = ["Codewars", "cOdewars", "coDewars", "codEwars", "codeWars", "codewArs", "codewaRs", "codewarS"]





# Find the first non-consecutive number
# https://www.codewars.com/kata/58f8a3a27a5c28d92e000144/
"""Your task is to find the first element of an array that is not consecutive.

By not consecutive we mean not exactly 1 larger than the previous element of the array.

E.g. If we have an array [1,2,3,4,6,7,8] then 1 then 2 then 3 then 4 are all consecutive but 6 is not, so that's the first non-consecutive number.

If the whole array is consecutive then return null2.

The array will always have at least 2 elements1 and all elements will be numbers. The numbers will also all be unique and in ascending order. The numbers could be positive or negative and the first non-consecutive could be either too!

"""


def first_non_consecutive(arr):
    for i in range(len(arr) - 1):
        if arr[i+1] - arr[i] != 1:
            return arr[i+1]
    return None
first_non_consecutive([1, 2, 3, 4, 6, 7, 8])
first_non_consecutive([1, 2, 3, 4, 5, 6, 7, 9])
first_non_consecutive([1, 2, 3, 4, 5, 6, 7, 8])
first_non_consecutive([31, 32])

def first_non_consecutive(arr):
    for i, v in enumerate(arr, arr[0]):
        if v != i:
            return v

import numpy as np
def first_non_consecutive(arr):
    index = np.array(arr[1:])[list(map(lambda x: x > 1, (arr[i+1] - arr[i] for i in range(len(arr) - 1))))]
    index = np.array(arr[1:])[list(True if arr[i+1] - arr[i] > 1 else False for i in range(len(arr) - 1))]
    # return index
    return None if not index else arr[index[0] - 2]

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(first_non_consecutive([1,2,3,4,6,7,8]), 6)
        test.assert_equals(first_non_consecutive([1,2,3,4,5,6,7,8]), None)
        test.assert_equals(first_non_consecutive([4,6,7,8,9,11]), 6)
        test.assert_equals(first_non_consecutive([4,5,6,7,8,9,11]), 11)
        test.assert_equals(first_non_consecutive([31,32]), None)
        test.assert_equals(first_non_consecutive([-3,-2,0,1]), 0)
        test.assert_equals(first_non_consecutive([-5,-4,-3,-1]), -1)





# Jaden Casing Strings
# https://www.codewars.com/kata/5390bac347d09b7da40006f6
"""Your task is to convert strings to how they would be written by Jaden Smith. The strings are actual quotes from Jaden Smith, but they are not capitalized in the same way he originally typed them.

Example:

Not Jaden-Cased: "How can mirrors be real if our eyes aren't real"
Jaden-Cased:     "How Can Mirrors Be Real If Our Eyes Aren't Real"
"""

def to_jaden_case(string_text):
    return ' '.join(i.capitalize() for i in string_text.split())
to_jaden_case("How can mirrors be real if our eyes aren't real")

def to_jaden_case(string_text):
    return ' '.join(map(str.capitalize, string_text.split()))
    # return ' '.join(map(lambda x: x.title(), string_text.split()))
to_jaden_case("How can mirrors be real if our eyes aren't real")

import string
def to_jaden_case(string_text):
    return string.capwords(string_text)

quote = "How can mirrors be real if our eyes aren't real"
test.assert_equals(to_jaden_case(quote), "How Can Mirrors Be Real If Our Eyes Aren't Real")





# Find the odd int
# https://www.codewars.com/kata/54da5a58ea159efa38000836
"""Given an array of integers, find the one that appears an odd number of times.

There will always be only one integer that appears an odd number of times.

Examples
[7] should return 7, because it occurs 1 time (which is odd).
[0] should return 0, because it occurs 1 time (which is odd).
[1,1,2] should return 2, because it occurs 1 time (which is odd).
[0,1,0,1,0] should return 0, because it occurs 3 times (which is odd).
[1,2,2,3,3,3,4,3,3,3,2,2,1] should return 4, because it appears 1 time (which is odd)."""


def find_it(seq):
    return [i for i in set(seq) if seq.count(i) % 2][0]
find_it([20, 1, -1, 2, -2, 3, 3, 5, 5, 1, 2, 4, 20, 4, -1, -2, 5])

from collections import Counter
def find_it(seq):
    return [k for k, v in Counter(seq).items() if v % 2][0]

from collections import Counter
def find_it(seq):
    for k, v in Counter(seq).items():
        if v % 2:
            return k
    return None


@test.describe("Sample tests")
def sample_tests():
    
    @test.it("find_it([20,1,-1,2,-2,3,3,5,5,1,2,4,20,4,-1,-2,5]) should return 5 (because it appears 3 times)")
    def _():
        test.assert_equals(find_it([20,1,-1,2,-2,3,3,5,5,1,2,4,20,4,-1,-2,5]), 5)
        
    @test.it("find_it([1,1,2,-2,5,2,4,4,-1,-2,5]) should return -1 (because it appears 1 time)")
    def _():
        test.assert_equals(find_it([1,1,2,-2,5,2,4,4,-1,-2,5]), -1);
        
    @test.it("find_it([20,1,1,2,2,3,3,5,5,4,20,4,5]) should return 5 (because it appears 3 times)")
    def _():
        test.assert_equals(find_it([20,1,1,2,2,3,3,5,5,4,20,4,5]), 5);
        
    @test.it("find_it([10]) should return 10 (because it appears 1 time)")
    def _():
        test.assert_equals(find_it([10]), 10);

    @test.it("find_it([10, 10, 10]) should return 10 (because it appears 3 times)")
    def _():
        test.assert_equals(find_it([10, 10, 10]), 10);        
        
    @test.it("find_it([1,1,1,1,1,1,10,1,1,1,1]) should return 10 (because it appears 1 time)")
    def _():
        test.assert_equals(find_it([1,1,1,1,1,1,10,1,1,1,1]), 10);

    @test.it("find_it([5,4,3,2,1,5,4,3,2,10,10]) should return 1 (because it appears 1 time)")
    def _():
        test.assert_equals(find_it([5,4,3,2,1,5,4,3,2,10,10]), 1);





# Highest Scoring Word
# https://www.codewars.com/kata/57eb8fcdf670e99d9b000272
"""Given a string of words, you need to find the highest scoring word.

Each letter of a word scores points according to its position in the alphabet: a = 1, b = 2, c = 3 etc.

You need to return the highest scoring word as a string.

If two words score the same, return the word that appears earliest in the original string.

All letters will be lowercase and all inputs will be valid."""


def high(x):
    word_list = [sum(ord(j) - 96 for j in i) for i in x.split()]
    return x.split()[word_list.index(max(word_list))]
high('man i need a taxi up to ubud')

def high(x):
    word_list = [sum(ord(letter) - ord("a") + 1 for letter in word) for word in x.split()]
    return x.split()[word_list.index(max(word_list))]

def high(x):
    return max(x.split(), key=lambda x: sum(ord(i) - ord("a") + 1 for i in x))


@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(high('man i need a taxi up to ubud'), 'taxi')
        test.assert_equals(high('what time are we climbing up the volcano'), 'volcano')
        test.assert_equals(high('take me to semynak'), 'semynak')
        test.assert_equals(high('aa b'), 'aa')
        test.assert_equals(high('b aa'), 'b')
        test.assert_equals(high('bb d'), 'bb')
        test.assert_equals(high('d bb'), 'd')
        test.assert_equals(high("aaa b"), "aaa")





# 
"""Complete the function that accepts a string parameter, and reverses each word in the string. All spaces in the string should be retained.

Examples
"This is an example!" ==> "sihT si na !elpmaxe"
"double  spaces"      ==> "elbuod  secaps""""


# string.split(' ') works
def reverse_words(text):
    return ' '.join(i[::-1] for i in text.split(' '))
reverse_words('double   spaced  words')
reverse_words('The quick brown fox jumps over the lazy dog.')

def reverse_words(text):
    return ' '.join(''.join(reversed(i)) for i in text.split(' '))

def reverse_words(text):
    word = ''
    sentence = ''
    for i in text:
        if i != ' ':
            word = i + word
        else:
            sentence += word
            sentence += ' '
            word = ''
    return sentence + word


@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(reverse_words('The quick brown fox jumps over the lazy dog.'), 'ehT kciuq nworb xof spmuj revo eht yzal .god')
        test.assert_equals(reverse_words('apple'), 'elppa')
        test.assert_equals(reverse_words('a b c d'), 'a b c d')
        test.assert_equals(reverse_words('double  spaced  words'), 'elbuod  decaps  sdrow')





# Vowel Count
# https://www.codewars.com/kata/54ff3102c1bad923760001f3
"""Return the number (count) of vowels in the given string.

We will consider a, e, i, o, u as vowels for this Kata (but not y).

The input string will only consist of lower case letters and/or spaces."""


def get_count(sentence):
    return sum(i in 'aeiou' for i in sentence)
get_count('abracadabra')

def get_count(sentence):
    return sum(True for i in sentence if i in 'aeiou')

import re
def get_count(sentence):
    return len(re.findall(r"[aeiou]", sentence))

def get_count(sentence):
    return sum(map(lambda x: True if x in 'aeiou' else False, sentence))


@test.it("Should return 5 for \"abracadabra\"")
def test_abracadabra():    
    test.assert_equals(get_count("abracadabra"), 5, f"Incorrect answer for \"abracadabra\"")





# Can we divide it?
# https://www.codewars.com/kata/5a2b703dc5e2845c0900005a
"""Your task is to create the functionisDivideBy (or is_divide_by) to check if an integer number is divisible by both integers a and b.

A few cases:


(-12, 2, -6)  ->  true
(-12, 2, -5)  ->  false

(45, 1, 6)    ->  false
(45, 5, 15)   ->  true

(4, 1, 4)     ->  true
(15, -5, 3)   ->  true"""

def is_divide_by(number, a, b):
    return not (number % a or number % b)
is_divide_by(8, 2, 4)
is_divide_by(-12, 2, -5)
is_divide_by(45, 1, 6)
is_divide_by(45, 5, 15)
is_divide_by(4, 1, 4)
is_divide_by(15, -5, 3) 

def is_divide_by(number, a, b):
    return not number % a and not number % b


@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(is_divide_by(8, 2, 4), True)
        test.assert_equals(is_divide_by(12, -3, 4), True)
        test.assert_equals(is_divide_by(8, 3, 4), False)
        test.assert_equals(is_divide_by(48, 2, -5), False)
        test.assert_equals(is_divide_by(-100, -25, 10), True)
        test.assert_equals(is_divide_by(10000, 5, -3), False)
        test.assert_equals(is_divide_by(4, 4, 2), True)
        test.assert_equals(is_divide_by(5, 2, 3), False)
        test.assert_equals(is_divide_by(-96, 25, 17), False)
        test.assert_equals(is_divide_by(33, 1, 33), True)





# Poker cards encoder/decoder
# https://www.codewars.com/kata/52ebe4608567ade7d700044a/
"""Consider a deck of 52 cards, which are represented by a string containing their suit and face value.

Your task is to write two functions encode and decode that translate an array of cards to/from an array of integer codes.

function encode :

input : array of strings (symbols)

output : array of integers (codes) sorted in ascending order

function decode :

input : array of integers (codes)

output : array of strings (symbols) sorted by code values

['Ac', 'Ks', '5h', 'Td', '3c'] -> [0, 2 ,22, 30, 51] //encoding
[0, 51, 30, 22, 2] -> ['Ac', '3c', 'Td', '5h', 'Ks'] //decoding
The returned array must be sorted from lowest to highest priority (value or precedence order, see below).

Card suits:
name    |  symbol   |  precedence
---------------------------------
club          c            0
diamond       d            1
heart         h            2
spade         s            3
52-card deck:
  c     |     d     |    h     |    s
----------------------------------------
 0: A      13: A      26: A      39: A
 1: 2      14: 2      27: 2      40: 2
 2: 3      15: 3      28: 3      41: 3
 3: 4      16: 4      29: 4      42: 4
 4: 5      17: 5      30: 5      43: 5
 5: 6      18: 6      31: 6      44: 6
 6: 7      19: 7      32: 7      45: 7
 7: 8      20: 8      33: 8      46: 8
 8: 9      21: 9      34: 9      47: 9
 9: T      22: T      35: T      48: T
10: J      23: J      36: J      49: J
11: Q      24: Q      37: Q      50: Q
12: K      25: K      38: K      51: K"""


def encode(cards):
    figures = "A23456789TJQK"
    symbols = "cdhs"
    return sorted(figures.find(i[0]) + 13 * symbols.find(i[1]) for i in cards)
encode(['Ac', 'Ks', '5h', 'Td', '3c'])
encode(["Td", "8c", "Ks"])

def decode(cards):
    figures = "A23456789TJQK"
    symbols = "cdhs"
    card_list = [figures[i % 13] + symbols[i // 13] for i in cards]
    return sorted(card_list, key=lambda x: figures.find(x[0]) + 13 * symbols.find(x[1]))
decode([0, 51, 30, 22, 2])
decode([7, 22, 51])    

def encode(cards):
    figures = 'A23456789TJQK'
    symbols = 'cdhs'
    return sorted(figures.index(i[0]) + symbols.find(i[1]) * 13 for i in cards)

def decode(cards):
    figures = 'A23456789TJQK'
    symbols = 'cdhs'
    return [figures[i % 13] + symbols[i // 13] for i in sorted(cards)]

encoding_dict = {'Ac': 0, 'Ad': 13, 'Ah': 26, 'As': 39,
                 '2c': 1, '2d': 14, '2h': 27, '2s': 40,
                 '3c': 2, '3d': 15, '3h': 28, '3s': 41,
                 '4c': 3, '4d': 16, '4h': 29, '4s': 42,
                 '5c': 4, '5d': 17, '5h': 30, '5s': 43,
                 '6c': 5, '6d': 18, '6h': 31, '6s': 44,
                 '7c': 6, '7d': 19, '7h': 32, '7s': 45,
                 '8c': 7, '8d': 20, '8h': 33, '8s': 46,
                 '9c': 8, '9d': 21, '9h': 34, '9s': 47,
                 'Tc': 9, 'Td': 22, 'Th': 35, 'Ts': 48,
                'Jc': 10, 'Jd': 23, 'Jh': 36, 'Js': 49,
                'Qc': 11, 'Qd': 24, 'Qh': 37, 'Qs': 50,
                'Kc': 12, 'Kd': 25, 'Kh': 38, 'Ks': 51}

decoding_dict = {v: k for k, v in encoding_dict.items()}
decoding_dict = {0: 'Ac', 13: 'Ad', 26: 'Ah', 39: 'As',
                 1: '2c', 14: '2d', 27: '2h', 40: '2s',
                 2: '3c', 15: '3d', 28: '3h', 41: '3s',
                 3: '4c', 16: '4d', 29: '4h', 42: '4s',
                 4: '5c', 17: '5d', 30: '5h', 43: '5s',
                 5: '6c', 18: '6d', 31: '6h', 44: '6s',
                 6: '7c', 19: '7d', 32: '7h', 45: '7s',
                 7: '8c', 20: '8d', 33: '8h', 46: '8s',
                 8: '9c', 21: '9d', 34: '9h', 47: '9s',
                 9: 'Tc', 22: 'Td', 35: 'Th', 48: 'Ts',
                10: 'Jc', 23: 'Jd', 36: 'Jh', 49: 'Js',
                11: 'Qc', 24: 'Qd', 37: 'Qh', 50: 'Qs',
                12: 'Kc', 25: 'Kd', 38: 'Kh', 51: 'Ks'}

def encode(cards):
    return sorted(encoding_dict[i] for i in cards)
encode(["Td", "8c", "Ks"])

def decode(cards):
    return [decoding_dict[i] for i in sorted(cards)]
decode([7, 22, 51])    




@test.describe("Sample tests")
def sample_tests():
    @test.it("Encode")
    def it_1():
        test.assert_equals(encode(["Td", "8c", "Ks"]), [7, 22, 51])
        test.assert_equals(encode(["Qh", "5h", "Ad"]), [13, 30, 37])
        test.assert_equals(encode(["8c", "Ks", "Td"]), [7, 22, 51])
        test.assert_equals(encode(["Qh", "Ad", "5h"]), [13, 30, 37])

    @test.it("Decode")
    def it_2():
        test.assert_equals(decode([7, 22, 51]), ["8c", "Td", "Ks"])
        test.assert_equals(decode([13, 30, 37]), ["Ad", "5h", "Qh"])
        test.assert_equals(decode([7, 51, 22]), ["8c", "Td", "Ks"])
        test.assert_equals(decode([13, 37, 30]), ["Ad", "5h", "Qh"])





# Simple Events !!!
# https://www.codewars.com/kata/52d3b68215be7c2d5300022f/
"""Your goal is to write an Event constructor function, which can be used to make event objects.

An event object should work like this:

it has a .subscribe() method, which takes a function and stores it as its handler
it has an .unsubscribe() method, which takes a function and removes it from its handlers
Eeeit has an .emit() method, which takes an arbitrary number of arguments and calls all the stored functions with these arguments
As this is an elementary example of events, there are some simplifications:

all functions are called with correct arguments (e.g. only functions will be passed to unsubscribe)
you should not worry about the order of handlers' execution
the handlers will not attempt to modify an event object (e.g. add or remove handlers)
the context of handlers' execution is not important
each handler will be subscribed at most once at any given moment of time. It can still be unsubscribed and then subscribed again
Also see an example test fixture for suggested usage"""


class Event():
    def __init__(self):
        self.handler = set()
    def subscribe(self, fun):
        self.handler.add(fun)
    def unsubscribe(self, fun):
        self.handler.discard(fun)
    def emit(self, *args):
        for f in self.handler:
            f(*args)
    # def emit(self, *args):
    #     map(lambda x: x(*args), self.handler)


# Test
event = Event()
    
class Testf():
    def __init__(self):
        self.calls = 0
        self.args = []
    def __call__(self, *args):
        self.calls += 1
        self.args += args

f = Testf()

event.subscribe(f)
event.emit(1, 'foo', True)
event.emit(2, 'bar', False)
    
test.assert_equals(f.calls, 1) #calls a handler
test.assert_equals(f.args, [1, 'foo', True]) #passes arguments
    
event.unsubscribe(f)
event.emit(2)
    
test.assert_equals(f.calls, 1) #no second call





# Mongodb ObjectID
# https://www.codewars.com/kata/52fefe6cb0091856db00030e/
"""MongoDB is a noSQL database which uses the concept of a document, rather than a table as in SQL. Its popularity is growing.

As in any database, you can create, read, update, and delete elements. But in constrast to SQL, when you create an element, a new field _id is created. This field is unique and contains some information about the time the element was created, id of the process that created it, and so on. More information can be found in the MongoDB documentation (which you have to read in order to implement this Kata).

So let us implement the following helper which will have 2 methods:

one which verifies that the string is a valid Mongo ID string, and
one which finds the timestamp from a MongoID string
Note:

If you take a close look at a Codewars URL, you will notice each kata's id (the XXX in http://www.codewars.com/dojo/katas/XXX/train/javascript) is really similar to MongoDB's ids, which brings us to the conjecture that this is the database powering Codewars.

Examples
The verification method will return true if an element provided is a valid Mongo string and false otherwise:

Mongo.is_valid('507f1f77bcf86cd799439011')  # True
Mongo.is_valid('507f1f77bcf86cz799439011')  # False
Mongo.is_valid('507f1f77bcf86cd79943901')   # False
Mongo.is_valid('111111111111111111111111')  # True
Mongo.is_valid(111111111111111111111111)    # False
Mongo.is_valid('507f1f77bcf86cD799439011')  # False (we arbitrarily only allow lowercase letters)
The timestamp method will return a date/time object from the timestamp of the Mongo string and false otherwise:

# Mongo.get_timestamp should return a datetime object

Mongo.get_timestamp('507f1f77bcf86cd799439011')  # Wed Oct 17 2012 21:13:27 GMT-0700 (Pacific Daylight Time)
Mongo.get_timestamp('507f1f77bcf86cz799439011')  # False
Mongo.get_timestamp('507f1f77bcf86cd79943901')   # False
Mongo.get_timestamp('111111111111111111111111')  # Sun Jan 28 1979 00:25:53 GMT-0800 (Pacific Standard Time)
Mongo.get_timestamp(111111111111111111111111)    # False
When you will implement this correctly, you will not only get some points, but also would be able to check creation time of all the kata here :-)"""


import string
def is_valid(s):
    return all(i in string.hexdigits[:16] for i in s) and len(s) == 24
is_valid('507f1f77bcf86cd799439011')
is_valid('507f1f77bcf86cz799439011')
is_valid('507f1f77bcf86cd79943901')
is_valid('111111111111111111111111')

from datetime import datetime
def get_timestamp(s):
    return datetime.fromtimestamp(int(s[:8], 16)) if is_valid(s)
get_timestamp('507f1f77bcf86cd799439011')
get_timestamp('507f1f77bcf86cz799439011')
get_timestamp('507f1f77bcf86cd79943901')
get_timestamp('111111111111111111111111')


from datetime import datetime
import string
class Mongo(object):

    @classmethod
    def is_valid(cls, s):
        """returns True if s is a valid MongoID; otherwise False"""
        if isinstance(s, str):
            return all(i in string.hexdigits[:16] for i in s) and len(s) == 24
        else:
            return False

    @classmethod
    def get_timestamp(cls, s):
        """if s is a MongoID, returns a datetime object for the timestamp; otherwise False"""
        if isinstance(s, str):
            if all(i in string.hexdigits[:16] for i in s) and len(s) == 24:
                return datetime.fromtimestamp(int(s[:8], 16))
            else:
                return False
        else:
            return False
        

from datetime import datetime
import re
class Mongo(object):

    @classmethod
    def is_valid(cls, s):
        """returns True if s is a valid MongoID; otherwise False"""
        # return isinstance(s, str) and all(i in string.hexdigits[:16] for i in s) and len(s) == 24
        return isinstance(s, str) and bool(re.match(r'[0-9a-f]{24}$', s))

    @classmethod
    def get_timestamp(cls, s):
        """if s is a MongoID, returns a datetime object for the timestamp; otherwise False"""
        return cls.is_valid(s) and datetime.fromtimestamp(int(s[:8], 16))


Mongo.is_valid('507f1f77bcf86cd799439016')
Mongo.get_timestamp('507f1f77bcf86cd799439016')

'{:d}'.format(0x507f1f77bcf86cd799439011)
int("507f1f77bcf86cd799439011", 16)
datetime.fromtimestamp(int("507F1f77bcf86cd799439011"[:8], 16))


from datetime import datetime

test.assert_equals(Mongo.is_valid(False), False)
test.assert_equals(Mongo.is_valid([]), False)
test.assert_equals(Mongo.is_valid(1234), False)
test.assert_equals(Mongo.is_valid('123476sd'), False)
test.assert_equals(Mongo.is_valid('507f1f77bcf86cd79943901'), False)
test.assert_equals(Mongo.is_valid('507f1f77bcf86cd799439016'), True)

test.assert_equals(Mongo.get_timestamp(False), False)
test.assert_equals(Mongo.get_timestamp([]), False)
test.assert_equals(Mongo.get_timestamp(1234), False)
test.assert_equals(Mongo.get_timestamp('123476sd'), False)
test.assert_equals(Mongo.get_timestamp('507f1f77bcf86cd79943901'), False)
test.assert_equals(Mongo.get_timestamp('507f1f77bcf86cd799439016'), datetime(2012, 10, 17, 21, 13, 27))





# Prime number decompositions
# https://www.codewars.com/kata/53c93982689f84e321000d62
"""You have to code a function getAllPrimeFactors, which takes an integer as parameter and returns an array containing its prime decomposition by ascending factors. If a factor appears multiple times in the decomposition, it should appear as many times in the array.

exemple: getAllPrimeFactors(100) returns [2,2,5,5] in this order.

This decomposition may not be the most practical.

You should also write getUniquePrimeFactorsWithCount, a function which will return an array containing two arrays: one with prime numbers appearing in the decomposition and the other containing their respective power.

exemple: getUniquePrimeFactorsWithCount(100) returns [[2,5],[2,2]]

You should also write getUniquePrimeFactorsWithProducts, which returns an array containing the prime factors to their respective powers.

exemple: getUniquePrimeFactorsWithProducts(100) returns [4,25]

Errors, if:

n is not a number
n not an integer
n is negative or 0
The three functions should respectively return [], [[],[]] and [].

Edge cases:

if n=0, the function should respectively return [], [[],[]] and [].
if n=1, the function should respectively return [1], [[1],[1]], [1].
if n=2, the function should respectively return [2], [[2],[1]], [2].
The result for n=2 is normal. The result for n=1 is arbitrary and has been chosen to return a usefull result. The result for n=0 is also arbitrary but can not be chosen to be both usefull and intuitive. ([[0],[0]] would be meaningfull but wont work for general use of decomposition, [[0],[1]] would work but is not intuitive.)"""


def getAllPrimeFactors(n):
    if (not (type(n) == int) or
        n < 1):
        return []
    if n == 1:
        return [1]
    
    sol = []
    divider = 2
    # while True:
    while n >= divider:
        if not n % divider:
            sol.append(divider)
            n //= divider
            divider = 2
        else:
            divider += 1
        # if n == 1:
            # break
    return sol
getAllPrimeFactors(100)

def getAllPrimeFactors(n):
    decomp = []
    while n != 1:
        for i in range(2, n + 1):
            if not n % i:
                # decomp.append(i)
                decomp += [i]
                n //= i
                break
    return decomp
getAllPrimeFactors(100)


from collections import Counter
def getUniquePrimeFactorsWithCount(n):
    factors = Counter(getAllPrimeFactors(n))
    return [list(factors.keys()), list(factors.values())]
getUniquePrimeFactorsWithCount(100)
getUniquePrimeFactorsWithCount(1)
getUniquePrimeFactorsWithCount(0)
getUniquePrimeFactorsWithCount(-1)

def getUniquePrimeFactorsWithCount(n):
    prime_set = set(getAllPrimeFactors(n))
    values = [getAllPrimeFactors(n).count(i) for i in prime_set]
    return [sorted(list(prime_set)), values]

def getUniquePrimeFactorsWithProducts(n):
    factors = getUniquePrimeFactorsWithCount(n)
    return [a**b for a, b in zip(*factors)]
    return [a**b for a, b in zip(factors[0], factors[1])]
getUniquePrimeFactorsWithProducts(100)
getUniquePrimeFactorsWithProducts(1)
getUniquePrimeFactorsWithProducts(0)
getUniquePrimeFactorsWithProducts(-1)

def getUniquePrimeFactorsWithProducts(n):
    return [k ** v for k, v in Counter(getAllPrimeFactors(n)).items()]

def getUniquePrimeFactorsWithProducts(n):
    power1 = getUniquePrimeFactorsWithCount(n)
    # return list(map(lambda x, y: x**y, power1[0], power1[1]))
    return list(map(pow, power1[0], power1[1]))

import numpy as np
def getUniquePrimeFactorsWithProducts(n):
    return list(np.power(list(Counter(getAllPrimeFactors(n)).keys()), 
                         list(Counter(getAllPrimeFactors(n)).values())))

def getUniquePrimeFactorsWithProducts(n):
    return list(np.power(*getUniquePrimeFactorsWithCount(n)))


test.assert_equals(getAllPrimeFactors(100), [2,2,5,5])
test.assert_equals(getUniquePrimeFactorsWithCount(100), [[2,5],[2,2]])
test.assert_equals(getUniquePrimeFactorsWithProducts(100), [4,25])





# Strings Mix
# https://www.codewars.com/kata/5629db57620258aa9d000014
"""Given two strings s1 and s2, we want to visualize how different the two strings are. 
We will only take into account the lowercase letters (a to z). 
First let us count the frequency of each lowercase letters in s1 and s2.

s1 = "A aaaa bb c"

s2 = "& aaa bbb c d"

s1 has 4 'a', 2 'b', 1 'c'

s2 has 3 'a', 3 'b', 1 'c', 1 'd'

So the maximum for 'a' in s1 and s2 is 4 from s1; the maximum for 'b' is 3 from s2. 
In the following we will not consider letters when the maximum of their occurrences is less than or equal to 1.

We can resume the differences between s1 and s2 in the following string: 
"1:aaaa/2:bbb" where 1 in 1:aaaa stands for string s1 and aaaa because the maximum for a is 4. 
In the same manner 2:bbb stands for string s2 and bbb because the maximum for b is 3.

The task is to produce a string in which each lowercase letters of s1 or s2 appears as 
many times as its maximum if this maximum is strictly greater than 1; these letters will 
be prefixed by the number of the string where they appear with their maximum value and :. 
If the maximum is in s1 as well as in s2 the prefix is =:.

In the result, substrings (a substring is for example 2:nnnnn or 1:hhh; it contains the prefix) 
will be in decreasing order of their length and when they have the same length sorted in ascending 
lexicographic order (letters and digits - more precisely sorted by codepoint); the different groups 
will be separated by '/'. See examples and "Example Tests".

Hopefully other examples can make this clearer.

s1 = "my&friend&Paul has heavy hats! &"
s2 = "my friend John has many many friends &"
mix(s1, s2) --> "2:nnnnn/1:aaaa/1:hhh/2:mmm/2:yyy/2:dd/2:ff/2:ii/2:rr/=:ee/=:ss"

s1 = "mmmmm m nnnnn y&friend&Paul has heavy hats! &"
s2 = "my frie n d Joh n has ma n y ma n y frie n ds n&"
mix(s1, s2) --> "1:mmmmmm/=:nnnnnn/1:aaaa/1:hhh/2:yyy/2:dd/2:ff/2:ii/2:rr/=:ee/=:ss"

s1="Are the kids at home? aaaaa fffff"
s2="Yes they are here! aaaaa fffff"
mix(s1, s2) --> "=:aaaaaa/2:eeeee/=:fffff/1:tt/2:rr/=:hh"
"""


import string
from collections import Counter
def mix(s1, s2):
    s1_dict = Counter(filter(str.islower, s1))
    # s1_dict = Counter(char for char in s1 if char.islower())
    # s1 = dict(Counter(i for i in filter(lambda x: x in string.ascii_lowercase, s1)))
    s1_dict = {k: v for k, v in s1_dict.items() if v > 1}
    # s2_dict = Counter(filter(str.islower, s2))
    # s2_dict = {k: v for k, v in s2_dict.items() if v > 1}
    s2_dict = {k: v for k, v in Counter(s2).items() if k.islower() and v > 1}
    s3_dict = Counter(s1_dict) | Counter(s2_dict)
    # s3_dict = s1_dict | s2_dict # (n: 3 | n: 2) = n: 2, if two are present the second one is chosed, so use Counter
    to_sort = (('1:' + k*v) if (s1_dict.get(k, -1) == v and s2_dict.get(k, -1) != v) else 
               ('2:' + k*v) if (s2_dict.get(k, -1) == v and s1_dict.get(k, -1) != v) else 
               ('=:' + k*v) for k, v in s3_dict.items())
    # return '/'.join(sorted(sorted(sorted(to_sort, key=lambda x: x[2]), key=lambda x: x), key=lambda x: -len(x)))
    return '/'.join(sorted(to_sort, key=lambda x: (-len(x), x, x[2])))
mix(s1, s2)
mix("Are they here", "yes, they are here")
mix(s1, s2) --> "2:eeeee/2:yy/=:hh/=:rr"
mix("my&friend&Paul has heavy hats! &", "my friend John has many many friends &")
mix(s1, s2) --> "2:nnnnn/1:aaaa/1:hhh/2:mmm/2:yyy/2:dd/2:ff/2:ii/2:rr/=:ee/=:ss"

from collections import Counter

def mix(s1, s2):
    dict1 = {k: v for k, v in Counter(s1).items() if k.islower() and v > 1}
    dict2 = {k: v for k, v in Counter(s2).items() if k.islower() and v > 1}
    dict_sum = Counter(dict1) | Counter(dict2)
    unsorted_sol = []
    for letter in dict_sum.keys():
        unsorted_sol += ["1:" + letter * dict_sum[letter] if dict1.get(letter, -1) == dict_sum[letter] and dict2.get(letter, -1) != dict_sum[letter] else
                         "2:" + letter * dict_sum[letter] if dict1.get(letter, -1) != dict_sum[letter] and dict2.get(letter, -1) == dict_sum[letter] else
                         "=:" + letter * dict_sum[letter]]
    return "/".join(sorted(unsorted_sol, key=lambda x: (-len(x), x[0], x[2])))

# from codewars
def mix(s1, s2):
    c1 = Counter(filter(str.islower, s1))
    c2 = Counter(filter(str.islower, s2))
    res = []
    for c in set(c1.keys() + c2.keys()):
        n1, n2 = c1.get(c, 0), c2.get(c, 0)
        if n1 > 1 or n2 > 1:
            res.append(('1', c, n1) if n1 > n2 else
                ('2', c, n2) if n2 > n1 else ('=', c, n1))
    res = ['{}:{}'.format(i, c * n) for i, c, n in res]
    return '/'.join(sorted(res, key=lambda s: (-len(s), s)))

# from codewars
def mix(s1, s2):
    hist = {}
    for ch in string.ascii_lowercase:
        val1, val2 = s1.count(ch), s2.count(ch)
        if max(val1, val2) > 1:
            which = "1" if val1 > val2 else "2" if val2 > val1 else "="
            hist[ch] = (-max(val1, val2), which + ":" + ch * max(val1, val2))
    return "/".join(hist[ch][1] for ch in sorted(hist, key=lambda x: hist[x]))


test.describe("Mix")
test.it("Basic Tests")
test.assert_equals(mix("Are they here", "yes, they are here"), "2:eeeee/2:yy/=:hh/=:rr")
test.assert_equals(mix("Sadus:cpms>orqn3zecwGvnznSgacs","MynwdKizfd$lvse+gnbaGydxyXzayp"), '2:yyyy/1:ccc/1:nnn/1:sss/2:ddd/=:aa/=:zz')
test.assert_equals(mix("looping is fun but dangerous", "less dangerous than coding"), "1:ooo/1:uuu/2:sss/=:nnn/1:ii/2:aa/2:dd/2:ee/=:gg")
test.assert_equals(mix(" In many languages", " there's a pair of functions"), "1:aaa/1:nnn/1:gg/2:ee/2:ff/2:ii/2:oo/2:rr/2:ss/2:tt")
test.assert_equals(mix("Lords of the Fallen", "gamekult"), "1:ee/1:ll/1:oo")
test.assert_equals(mix("codewars", "codewars"), "")
test.assert_equals(mix("A generation must confront the looming ", "codewarrs"), "1:nnnnn/1:ooooo/1:tttt/1:eee/1:gg/1:ii/1:mm/=:rr")





# Tribonacci Sequence
# https://www.codewars.com/kata/556deca17c58da83c00002db
"""Well met with Fibonacci bigger brother, AKA Tribonacci.

As the name may already reveal, it works basically like a Fibonacci, but summing the last 3 (instead of 2) numbers of the sequence to generate the next. And, worse part of it, regrettably I won't get to hear non-native Italian speakers trying to pronounce it :(

So, if we are to start our Tribonacci sequence with [1, 1, 1] as a starting input (AKA signature), we have this sequence:

[1, 1 ,1, 3, 5, 9, 17, 31, ...]
But what if we started with [0, 0, 1] as a signature? As starting with [0, 1] instead of [1, 1] basically shifts the common Fibonacci sequence by once place, you may be tempted to think that we would get the same sequence shifted by 2 places, but that is not the case and we would get:

[0, 0, 1, 1, 2, 4, 7, 13, 24, ...]
Well, you may have guessed it by now, but to be clear: you need to create a fibonacci function that given a signature array/list, returns the first n elements - signature included of the so seeded sequence."""


def tribonacci(signature, n):
    a, b, c = signature
    """if not n:
        return []
    if n == 1:
        return [a]
    """
    trib_lst = [a, b]
    for _ in range(n - 2):
        trib_lst += [c]
        a, b, c = b, c, a + b + c
    return trib_lst[:n]
tribonacci([1, 1, 1], 10)
tribonacci([1, 1, 1], 0)
tribonacci([1, 1, 1], 1)
tribonacci([1, 1, 1], 2)
tribonacci([0, 0, 1], 6)

def gen_tri(signature):
    a, b, c = signature
    yield a
    yield b
    while True:
        yield c
        a, b, c = b, c, a + b + c

tri_gen = gen_tri((1, 1, 1))
next(tri_gen)

def tribonacci(signature, n):
    if n < len(signature) + 1:
        return signature[:n]
    # if n == 0:
    #     return []
    # if n == 1:
    #     return [signature[0]]
    # if n == 2:
    #     return signature[0:2]
    tri_gen = gen_tri(signature)
    return signature[0:2] + [next(tri_gen) for _ in range(n - 2)]

# from codewars
def tribonacci(signature, n):
    res = signature[:n]
    for _ in range(n - 3): 
        # res.append(sum(res[-3:]))
        res += [sum(res[-3:])]
    return res

# from codewars
def tribonacci(signature,n):
    while len(signature) < n:
        signature.append(sum(signature[-3:]))    
    return signature[:n]


test.describe("Basic tests")
test.assert_equals(tribonacci([1, 1, 1], 10), [1, 1, 1, 3, 5, 9, 17, 31, 57, 105])
test.assert_equals(tribonacci([0, 0, 1], 10), [0, 0, 1, 1, 2, 4, 7, 13, 24, 44])
test.assert_equals(tribonacci([0, 1, 1], 10), [0, 1, 1, 2, 4, 7, 13, 24, 44, 81])
test.assert_equals(tribonacci([1, 0, 0], 10), [1, 0, 0, 1, 1, 2, 4, 7, 13, 24])
test.assert_equals(tribonacci([0, 0, 0], 10), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
test.assert_equals(tribonacci([1, 2, 3], 10), [1, 2, 3, 6, 11, 20, 37, 68, 125, 230])
test.assert_equals(tribonacci([3, 2, 1], 10), [3, 2, 1, 6, 9, 16, 31, 56, 103, 190])
test.assert_equals(tribonacci([1, 1, 1], 1), [1])
test.assert_equals(tribonacci([300, 200, 100], 0), [])
test.assert_equals(tribonacci([0.5, 0.5, 0.5], 30), [0.5, 0.5, 0.5, 1.5, 2.5, 4.5, 8.5, 15.5, 28.5, 52.5, 96.5, 177.5, 326.5, 600.5, 1104.5, 2031.5, 3736.5, 6872.5, 12640.5, 23249.5, 42762.5, 78652.5, 144664.5, 266079.5, 489396.5, 900140.5, 1655616.5, 3045153.5, 5600910.5, 10301680.5])





# Simple Fun #166: Best Match
# https://www.codewars.com/kata/58b38256e51f1c2af0000081
""""AL-AHLY" and "Zamalek" are the best teams in Egypt, but "AL-AHLY" always wins the matches between them. "Zamalek" managers want to know what is the best match they've played so far.

The best match is the match they lost with the minimum goal difference. If there is more than one match with the same difference, choose the one "Zamalek" scored more goals in.

Given the information about all matches they played, return the index of the best match (0-based). If more than one valid result, return the smallest index.

Example
For ALAHLYGoals = [6,4] and zamalekGoals = [1,2], the output should be 1 (2 in COBOL).

Because 4 - 2 is less than 6 - 1

For ALAHLYGoals = [1,2,3,4,5] and zamalekGoals = [0,1,2,3,4], the output should be 4.

The goal difference of all matches are 1, but at 4th match "Zamalek" scored more goals in. So the result is 4 (5 in COBOL).

Input/Output
[input] integer array ALAHLYGoals
The number of goals "AL-AHLY" scored in each match.

[input] integer array zamalekGoals
The number of goals "Zamalek" scored in each match. It is guaranteed that zamalekGoals[i] < ALAHLYGoals[i] for each element.

[output] an integer
Index of the best match."""


def best_match(goals1, goals2):
    to_sort = [(i, a - b, b) for i, (a, b) in enumerate(list(zip(goals1, goals2)))]
    return sorted(to_sort, key=lambda x: (x[1], -x[2]))[0][0]
    # return min(to_sort, key=lambda x: (x[1], -x[2]))[0]

# from codewars
def best_match(goals1, goals2):
    return min((a-b, -b, i) for i, (a, b) in enumerate(zip(goals1, goals2)))[-1]
best_match([6, 4],[1, 2])
best_match([1, 2, 3, 4, 5], [0, 1, 2, 3, 4])

# from codewars
def best_match(goals1, goals2):
    return sorted((a-b, -b, i) for i, (a, b) in enumerate(zip(goals1, goals2)))[0][-1]

import numpy as np
def best_match(goals1, goals2):
    goals = list(zip(np.array(goals1) - np.array(goals2), goals2))
    return sorted(range(len(goals)), key=lambda x: (goals[x][0], -goals[x][1]))[0]


best_match([6, 4], [1, 2])
best_match([1], [0])
best_match([1, 2, 3, 4, 5], [0, 1, 2, 3, 4])
best_match([3, 4, 3], [1, 1, 2])
best_match([4, 3, 4], [1, 1, 1])

test.it("Basic Tests")
test.assert_equals(best_match([6, 4],[1, 2]),1)
test.assert_equals(best_match([1],[0]),0)
test.assert_equals(best_match([1, 2, 3, 4, 5],[0, 1, 2, 3, 4]),4)
test.assert_equals(best_match([3, 4, 3],[1, 1, 2]),2)
test.assert_equals(best_match([4, 3, 4],[1, 1, 1]),1)





# Roman Numerals Decoder
# https://www.codewars.com/kata/51b6249c4612257ac0000005
"""Create a function that takes a Roman numeral as its argument and returns its value as a numeric decimal integer. You don't need to validate the form of the Roman numeral.

Modern Roman numerals are written by expressing each decimal digit of the number to be encoded separately, starting with the leftmost digit and skipping any 0s. So 1990 is rendered "MCMXC" (1000 = M, 900 = CM, 90 = XC) and 2008 is rendered "MMVIII" (2000 = MM, 8 = VIII). The Roman numeral for 1666, "MDCLXVI", uses each letter in descending order.

Example:

solution('XXI'); // should return 21
Help:

Symbol    Value
I          1
V          5
X          10
L          50
C          100
D          500
M          1,000"""


# only check if number is legit
import re
regex_pattern = r"^(?=[MDCLXVI])(M{0,3})(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$"
print(str(bool(re.match(regex_pattern, input()))))

roman_dict = {'I': 1,
              'V': 5,
              'X': 10,
              'L': 50,
              'C': 100,
              'D': 500,
              'M': 1000}

bool(re.search(r"DM", "MDM"))

import re
def solution(roman):
    result = 0
    # r_find = re.match(r'.*?CM', roman)
    # r_find = re.search(r"CM", roman)
    if re.search(r"CM", roman):
    # if r_find:
        result += 900
        roman = roman.replace('CM', '')
    # r_find = re.match(r'.*?CD', roman)
    if re.search(r"CD", roman):
        result += 400
        roman = roman.replace('CD', '')
    r_find = re.match(r'.*?XC', roman)
    if r_find:
        result += 90
        roman = roman.replace('XC', '')
    r_find = re.match(r'.*?XL', roman)
    if r_find:
        result += 40
        roman = roman.replace('XL', '')
    r_find = re.match(r'.*?IX', roman)
    if r_find:
        result += 9
        roman = roman.replace('IX', '')
    r_find = re.match(r'.*?IV', roman)
    if r_find:
        result += 4
        roman = roman.replace('IV', '')
    return result + sum(roman_dict[i] for i in roman)
solution("MMMCMXCIX")
solution('XXI')
solution('IV')
solution('IX')
solution('MDCLXVI')
solution('MMDCLXVI')
solution('XIV')

# playground
solution("MMM")
solution("MMMCM")

def solution(roman):
    value = 0
    
    value += 1000 * roman.count("M")
    roman = roman.replace("M", "")
    
    if not len(roman):
        return value

    if roman[0] == "C":
        value -= 100
        roman = roman[1:]
    return value

    value += 500 * roman.count("D")
    roman = roman.replace("D", "")

    if roman[0] == "C":
        value -= 100
        roman = roman[1:]

    value += 100 * roman.count("C")
    roman = roman.replace("C", "")

    if roman[0] == "X":
        value -= 10
        roman = roman[1:]

    value += 50 * roman.count("L")
    roman = roman.replace("L", "")

    if roman[0] == "X":
        value -= 10
        roman = roman[1:]

    value += 10 * roman.count("X")
    roman = roman.replace("X", "")

    if roman[0] == "I":
        value -= 1
        roman = roman[1:]

    value += 5 * roman.count("V")
    roman = roman.replace("V", "")

    if roman[0] == "I":
        value -= 1
        roman = roman[1:]

    value += 1 * roman.count("I")
    roman = roman.replace("I", "")

    # value += sum(decoder[i] for i in roman)
    return value

# from codewars, modded
def solution(roman):
    dict = {
        'M': 1000,
        'D': 500,
        'C': 100,
        'L': 50,
        'X': 10,
        'V': 5,
        'I': 1
    }

    last, total = dict[roman[-1]], dict[roman[-1]]
    for c in roman[-2::-1]:
        if last > dict[c]:
            total -= dict[c]
        else:
            total += dict[c]
        last = dict[c]
    return total


SYMBOLS = {
    'M': 1000,
    'CM': 900,
    'D': 500,
    'CD': 400,
    'C': 100,
    'XC': 90,
    'L': 50,
    'XL': 40,
    'X': 10,
    'IX': 9,
    'V': 5,
    'IV': 4,
    'I': 1,
}

def solution(roman):
    if not roman:
        return 0
    for k, v in SYMBOLS.items():
        if roman.startswith(k):
            return v + solution(roman[len(k):])
solution('IV')

values = [('M', 1000), ('CM', -200), ('D', 500), ('CD', -200),
          ('C', 100), ('XC', -20), ('L', 50), ('XL', -20),
          ('X', 10), ('IX', -2), ('V', 5), ('IV', -2),
          ('I', 1)]
def solution(roman):
    return sum(roman.count(s)*v for s, v in values)

test.describe("Example Tests")
test.assert_equals(solution('XXI'), 21, 'XXI should == 21')
test.assert_equals(solution('I'), 1, 'I should == 1')
test.assert_equals(solution('IV'), 4, 'IV should == 4')
test.assert_equals(solution('MMVIII'), 2008, 'MMVIII should == 2008')
test.assert_equals(solution('MDCLXVI'), 1666, 'MDCLXVI should == 1666')





# Friend or Foe?
# https://www.codewars.com/kata/55b42574ff091733d900002f
"""Make a program that filters a list of strings and returns a list with only your friends name in it.

If a name has exactly 4 letters in it, you can be sure that it has to be a friend of yours! Otherwise, you can be sure he's not...

Ex: Input = ["Ryan", "Kieran", "Jason", "Yous"], Output = ["Ryan", "Yous"]

i.e.

friend ["Ryan", "Kieran", "Mark"] `shouldBe` ["Ryan", "Mark"]
Note: keep the original order of the names in the output."""


def friend(x):
    return [i for i in x if len(i) == 4]
friend(["Ryan", "Kieran", "Mark"])

def friend(x):
    return list(filter(lambda y: len(y) == 4, x))


import codewars_test as test
from solution import friend

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Sample Test Cases')
    def sample_test_cases():
        test.assert_equals(friend(["Ryan", "Kieran", "Mark",]), ["Ryan", "Mark"])
        test.assert_equals(friend(["Ryan", "Jimmy", "123", "4", "Cool Man"]), ["Ryan"])
        test.assert_equals(friend(["Jimm", "Cari", "aret", "truehdnviegkwgvke", "sixtyiscooooool"]), ["Jimm", "Cari", "aret"])





# A Needle in the Haystack
# https://www.codewars.com/kata/56676e8fabd2d1ff3000000c
"""Can you find the needle in the haystack?

Write a function findNeedle() that takes an array full of junk but containing one "needle"

After your function finds the needle it should return a message (as a string) that says:

"found the needle at position " plus the index it found the needle, so:

find_needle(['hay', 'junk', 'hay', 'hay', 'moreJunk', 'needle', 'randomJunk'])
should return "found the needle at position 5" (in COBOL "found the needle at position 6")"""


def find_needle(haystack):
    return 'found the needle at position {}'.format(haystack.index('needle'))
    return 'found the needle at position %d' % haystack.index('needle')
    return f'found the needle at position {haystack.index("needle")}'
    return "found the needle at position " + str(haystack.index("needle"))
find_needle(['3', '123124234', None, 'needle', 'world', 'hay', 2, '3', True, False])


import codewars_test as test
from solution import find_needle

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(find_needle(['3', '123124234', None, 'needle', 'world', 'hay', 2, '3', True, False]), 'found the needle at position 3')
        test.assert_equals(find_needle(['283497238987234', 'a dog', 'a cat', 'some random junk', 'a piece of hay', 'needle', 'something somebody lost a while ago']), 'found the needle at position 5')
        test.assert_equals(find_needle([1,2,3,4,5,6,7,8,8,7,5,4,3,4,5,6,67,5,5,3,3,4,2,34,234,23,4,234,324,324,'needle',1,2,3,4,5,5,6,5,4,32,3,45,54]), 'found the needle at position 30')





# Grasshopper - Personalized Message
# https://www.codewars.com/kata/5772da22b89313a4d50012f7
"""Create a function that gives a personalized greeting. This function takes two parameters: name and owner.

Use conditionals to return the proper message:

case	return
name equals owner	'Hello boss'
otherwise	'Hello guest'"""


def greet(name, owner):
    return 'Hello boss' if name == owner else 'Hello guest'


import codewars_test as test
from solution import greet

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(greet('Daniel', 'Daniel'), 'Hello boss')
        test.assert_equals(greet('Greg', 'Daniel'), 'Hello guest')





# The Feast of Many Beasts
# https://www.codewars.com/kata/5aa736a455f906981800360d/
"""All of the animals are having a feast! Each animal is bringing one dish. There is just one rule: the dish must start and end with the same letters as the animal's name. For example, the great blue heron is bringing garlic naan and the chickadee is bringing chocolate cake.

Write a function feast that takes the animal's name and dish as arguments and returns true or false to indicate whether the beast is allowed to bring the dish to the feast.

Assume that beast and dish are always lowercase strings, and that each has at least two letters. beast and dish may contain hyphens and spaces, but these will not appear at the beginning or end of the string. They will not contain numerals.

"""


def feast(beast, dish):
    return beast[0] == dish[0] and beast[-1] == dish[-1]
feast("great blue heron", "garlic naan")

def feast(beast, dish):
    return beast.startswith(dish[0]) and beast.endswith(dish[-1])


import codewars_test as test
from solution import feast

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(feast("great blue heron", "garlic naan"), True)
        test.assert_equals(feast("chickadee", "chocolate cake"), True)
        test.assert_equals(feast("brown bear", "bear claw"), False)





# Break camelCase
# https://www.codewars.com/kata/5208f99aee097e6552000148/
"""Complete the solution so that the function will break up camel casing, using a space between words.

Example
"camelCasing"  =>  "camel Casing"
"identifier"   =>  "identifier"
""             =>  """""


def solution(s):
    return ''.join(' ' + i if i.isupper() else i for i in s)
solution("helloWorld")
solution("camelCase")
solution("breakCamelCase")

import string
def solution(s):
    for i in string.ascii_uppercase:
        s = s.replace(i, ' ' + i)
    return s

import re
def solution(s):
    return re.sub(r'([A-Z])', r' \1', s)

test.assert_equals(solution("helloWorld"), "hello World")
test.assert_equals(solution("camelCase"), "camel Case")
test.assert_equals(solution("breakCamelCase"), "break Camel Case")





# Printer Errors
# https://www.codewars.com/kata/56541980fa08ab47a0000040
"""In a factory a printer prints labels for boxes. For one kind of boxes the printer has to use colors which, for the sake of simplicity, are named with letters from a to m.

The colors used by the printer are recorded in a control string. For example a "good" control string would be aaabbbbhaijjjm meaning that the printer used three times color a, four times color b, one time color h then one time color a...

Sometimes there are problems: lack of colors, technical malfunction and a "bad" control string is produced e.g. aaaxbbbbyyhwawiwjjjwwm with letters not from a to m.

You have to write a function printer_error which given a string will return the error rate of the printer as a string representing a rational whose numerator is the number of errors and the denominator the length of the control string. Don't reduce this fraction to a simpler expression.

The string has a length greater or equal to one and contains only letters from ato z.

Examples:
s="aaabbbbhaijjjm"
printer_error(s) => "0/14"

s="aaaxbbbbyyhwawiwjjjwwm"
printer_error(s) => "8/22""""


def printer_error(s):
    return '{}/{}'.format(sum(i > 'm' for i in s), len(s))
    return '{}/{}'.format(sum(ord(i) > 109 for i in s), len(s))
    return '%d/%d' % (sum(i > 'm' for i in s), len(s))
    return f"{sum(i > 'm' for i in s)}/{len(s)}"
printer_error("aaaxbbbbyyhwawiwjjjwwm")

import string
def printer_error(s):
    nominator = str(sum(i in string.ascii_lowercase[14:] for i in s))
    denominator = str(len(s))
    return  nominator + "/" + denominator

import re
def printer_error(s):
    return '{}/{}'.format(len(re.sub('[a-m]', '', s)), len(s))

import re
def printer_error(s):
    return '{}/{}'.format(len(re.findall('[n-z]', s)), len(s))


import codewars_test as test

@test.describe("printer_error")
def basic_tests():
    @test.it('Example Test Cases')
    def example_test_cases():
        s="aaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbmmmmmmmmmmmmmmmmmmmxyz"
        test.assert_equals(printer_error(s), "3/56")
        s = "kkkwwwaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbmmmmmmmmmmmmmmmmmmmxyz"
        test.assert_equals(printer_error(s), "6/60")
        s = "kkkwwwaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbmmmmmmmmmmmmmmmmmmmxyzuuuuu"
        test.assert_equals(printer_error(s) , "11/65")





# Rock Paper Scissors!
# https://www.codewars.com/kata/5672a98bdbdd995fad00000f
"""Rock Paper Scissors
Let's play! You have to return which player won! In case of a draw return Draw!.

Examples:

rps('scissors','paper') // Player 1 won!
rps('scissors','rock') // Player 2 won!
rps('paper','paper') // Draw!"""


def rps(p1, p2):
    if p1 == p2:
        return 'Draw!'
    winner = {'rock': 'paper', 'scissors': 'rock', 'paper': 'scissors'}
    if winner[p2] == p1:
        return 'Player 1 won!'
    else:
        return 'Player 2 won!'
rps('scissors','paper')
rps('scissors','rock')
rps('paper','paper')
rps('rock','paper')
rps('paper','rock')

def rps(p1, p2):
    items = ["rock", "paper", "scissors"]
    items2 = ["paper", "scissors", "rock"]
    if p1 == p2:
        return "Draw!"
    if items2.index(p1) == items.index(p2):
        return "Player 1 won!"
    return "Player 2 won!"


import codewars_test as test
from solution import rps

@test.describe("rock paper scissors")
def basic_tests():
    @test.it("Player 1 wins")
    def player_1():
        test.assert_equals(rps('rock', 'scissors'), "Player 1 won!")
    @test.it("Player 1 wins")
    def player_1():
        test.assert_equals(rps('scissors', 'rock'), "Player 2 won!")
    @test.it("Draw")
    def draw():
        test.assert_equals(rps('rock', 'rock'), 'Draw!')





# Basic Mathematical Operations
# https://www.codewars.com/kata/57356c55867b9b7a60000bd7/
"""Your task is to create a function that does four basic mathematical operations.

The function should take three arguments - operation(string/char), value1(number), value2(number).
The function should return result of numbers after applying the chosen operation.

Examples(Operator, value1, value2) --> output
('+', 4, 7) --> 11
('-', 15, 18) --> -3
('*', 5, 5) --> 25
('/', 49, 7) --> 7"""


def basic_op(operator, value1, value2):
    return eval(str(value1) + operator + str(value2))
    return eval(f'{value1}{operator}{value2}')
    return eval("{}{}{}".format(value1, operator, value2))
    return eval('value1' + operator + 'value2')
    return {'+':a+b,'-':a-b,'*':a*b,'/':a/b}[o]
basic_op('+', 4, 7)


import codewars_test as test
from solution import basic_op

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(basic_op('+', 4, 7), 11)
        test.assert_equals(basic_op('-', 15, 18), -3)
        test.assert_equals(basic_op('*', 5, 5), 25)
        test.assert_equals(basic_op('/', 49, 7), 7)





# The Supermarket Queue
# https://www.codewars.com/kata/57b06f90e298a7b53d000a86
"""There is a queue for the self-checkout tills at the supermarket. Your task is write a function to calculate the total time required for all the customers to check out!

input
customers: an array of positive integers representing the queue. Each integer represents a customer, and its value is the amount of time they require to check out.
n: a positive integer, the number of checkout tills.
output
The function should return an integer, the total time required.

Important
Please look at the examples and clarifications below, to ensure you understand the task correctly :)

Examples
queue_time([5,3,4], 1)
# should return 12
# because when n=1, the total time is just the sum of the times

queue_time([10,2,3,3], 2)
# should return 10
# because here n=2 and the 2nd, 3rd, and 4th people in the 
# queue finish before the 1st person has finished.

queue_time([2,3,10], 2)
# should return 12
Clarifications
There is only ONE queue serving many tills, and
The order of the queue NEVER changes, and
The front person in the queue (i.e. the first element in the array/list) proceeds to a till as soon as it becomes free.
N.B. You should assume that all the test input will be valid, as specified above.

P.S. The situation in this kata can be likened to the more-computer-science-related idea of a thread pool, with relation to running multiple processes at the same time: https://en.wikipedia.org/wiki/Thread_pool"""


# from codewars
def queue_time(customers, n):
    cashiers = [0] * n
    for customer in customers:
        min_ind = cashiers.index(min(cashiers))
        cashiers[min_ind] += customer
    return max(cashiers)
queue_time([2, 2, 3, 3, 4, 4], 2)
queue_time([1,2,3,4,5], 1)
queue_time([1,2,3,4,5], 100)

import numpy as np
def queue_time(customers, n):
    nparray = np.zeros(n, dtype=int)
    for i in customers:
        nparray[np.where(nparray == min(nparray))[0][0]] += i
    return np.max(nparray)

# from codewars
class MarketQueue():
    
    def __init__(self,customers,n):
        self.customers = customers
        self.n=n
        self.timer = 0
        self.active_checkouts = []
        
    def calculate_total_time(self):
        while self.customers:
            self.process_queue()   
        return self.timer

    def process_queue(self):
        if len(self.active_checkouts) < self.n:
            queue_index = self.n - len(self.active_checkouts)
            self.active_checkouts.extend(self.customers[:queue_index])
            self.customers[:queue_index] = []
        while self.active_checkouts and (len(self.active_checkouts) == self.n or not self.customers) :
            self.timer += 1
            self.process_active_checkouts()
    
    def process_active_checkouts(self):
        finished_customers = []
        for index,customer in enumerate(self.active_checkouts):
            if customer > 1:
                self.active_checkouts[index] = int(customer-1)
            else:
                finished_customers.append(customer)
        
        for finished in finished_customers:
            self.active_checkouts.remove(finished)

# implementing requirements
def queue_time(customers,n):
    return MarketQueue(customers,n).calculate_total_time()


test.assert_equals(queue_time([], 1), 0, "wrong answer for case with an empty queue")
test.assert_equals(queue_time([5], 1), 5, "wrong answer for a single person in the queue")
test.assert_equals(queue_time([2], 5), 2, "wrong answer for a single person in the queue")
test.assert_equals(queue_time([1,2,3,4,5], 1), 15, "wrong answer for a single till")
test.assert_equals(queue_time([1,2,3,4,5], 100), 5, "wrong answer for a case with a large number of tills")
test.assert_equals(queue_time([2,2,3,3,4,4], 2), 9, "wrong answer for a case with two tills")





# Are they the "same"?
# https://www.codewars.com/kata/550498447451fbbd7600041c
"""Given two arrays a and b write a function comp(a, b) (orcompSame(a, b)) that checks whether the two arrays have the "same" elements, with the same multiplicities (the multiplicity of a member is the number of times it appears). "Same" means, here, that the elements in b are the elements in a squared, regardless of the order.

Examples
Valid arrays
a = [121, 144, 19, 161, 19, 144, 19, 11]  
b = [121, 14641, 20736, 361, 25921, 361, 20736, 361]
comp(a, b) returns true because in b 121 is the square of 11, 14641 is the square of 121, 20736 the square of 144, 361 the square of 19, 25921 the square of 161, and so on. It gets obvious if we write b's elements in terms of squares:

a = [121, 144, 19, 161, 19, 144, 19, 11] 
b = [11*11, 121*121, 144*144, 19*19, 161*161, 19*19, 144*144, 19*19]
Invalid arrays
If, for example, we change the first number to something else, comp is not returning true anymore:

a = [121, 144, 19, 161, 19, 144, 19, 11]  
b = [132, 14641, 20736, 361, 25921, 361, 20736, 361]
comp(a,b) returns false because in b 132 is not the square of any number of a.

a = [121, 144, 19, 161, 19, 144, 19, 11]  
b = [121, 14641, 20736, 36100, 25921, 361, 20736, 361]
comp(a,b) returns false because in b 36100 is not the square of any number of a.

Remarks
a or b might be [] or {} (all languages except R, Shell).
a or b might be nil or null or None or nothing (except in C++, COBOL, Crystal, D, Dart, Elixir, Fortran, F#, Haskell, Nim, OCaml, Pascal, Perl, PowerShell, Prolog, PureScript, R, Racket, Rust, Shell, Swift).
If a or b are nil (or null or None, depending on the language), the problem doesn't make sense so return false.

Note for C
The two arrays have the same size (> 0) given as parameter in function comp."""


def comp(array1, array2):
    if array1 == [] and array2 == []:
        return True
    if not (array1 and array2) :
        return False
    return sorted(i ** 2 for i in array1) == sorted(array2)
comp([121, 144, 19, 161, 19, 144, 19, 11], [11*11, 121*121, 144*144, 19*19, 161*161, 19*19, 144*144, 19*19])
comp([], [])


import codewars_test as test
from solution import comp

@test.describe("Same")
def tests():
    @test.it("Fixed tests")
    def basics():
        a1 = [121, 144, 19, 161, 19, 144, 19, 11]
        a2 = [11*11, 121*121, 144*144, 19*19, 161*161, 19*19, 144*144, 19*19]
        test.assert_equals(comp(a1, a2), True)
        a1 = [121, 144, 19, 161, 19, 144, 19, 11]
        a2 = [11*21, 121*121, 144*144, 19*19, 161*161, 19*19, 144*144, 19*19]
        test.assert_equals(comp(a1, a2), False)
        a1 = [121, 144, 19, 161, 19, 144, 19, 11]
        a2 = [11*11, 121*121, 144*144, 190*190, 161*161, 19*19, 144*144, 19*19]
        test.assert_equals(comp(a1, a2), False)





# Directions Reduction
# https://www.codewars.com/kata/550f22f4d758534c1100025a
"""Once upon a time, on a way through the old wild mountainous west,…
… a man was given directions to go from one point to another. The directions were "NORTH", "SOUTH", "WEST", "EAST". Clearly "NORTH" and "SOUTH" are opposite, "WEST" and "EAST" too.

Going to one direction and coming back the opposite direction right away is a needless effort. Since this is the wild west, with dreadfull weather and not much water, it's important to save yourself some energy, otherwise you might die of thirst!

How I crossed a mountainous desert the smart way.
The directions given to the man are, for example, the following (depending on the language):

["NORTH", "SOUTH", "SOUTH", "EAST", "WEST", "NORTH", "WEST"].
or
{ "NORTH", "SOUTH", "SOUTH", "EAST", "WEST", "NORTH", "WEST" };
or
[North, South, South, East, West, North, West]
You can immediatly see that going "NORTH" and immediately "SOUTH" is not reasonable, better stay to the same place! So the task is to give to the man a simplified version of the plan. A better plan in this case is simply:

["WEST"]
or
{ "WEST" }
or
[West]
Other examples:
In ["NORTH", "SOUTH", "EAST", "WEST"], the direction "NORTH" + "SOUTH" is going north and coming back right away.

The path becomes ["EAST", "WEST"], now "EAST" and "WEST" annihilate each other, therefore, the final result is [] (nil in Clojure).

In ["NORTH", "EAST", "WEST", "SOUTH", "WEST", "WEST"], "NORTH" and "SOUTH" are not directly opposite but they become directly opposite after the reduction of "EAST" and "WEST" so the whole path is reducible to ["WEST", "WEST"].

Task
Write a function dirReduc which will take an array of strings and returns an array of strings with the needless directions removed (W<->E or S<->N side by side).

The Haskell version takes a list of directions with data Direction = North | East | West | South.
The Clojure version returns nil when the path is reduced to nothing.
The Rust version takes a slice of enum Direction {North, East, West, South}.
See more examples in "Sample Tests:"
Notes
Not all paths can be made simpler. The path ["NORTH", "WEST", "SOUTH", "EAST"] is not reducible. "NORTH" and "WEST", "WEST" and "SOUTH", "SOUTH" and "EAST" are not directly opposite of each other and can't become such. Hence the result path is itself : ["NORTH", "WEST", "SOUTH", "EAST"].
if you want to translate, please ask before translating."""


def dir_reduc(arr):
    opos = {"NORTH": "SOUTH", "SOUTH": "NORTH", "EAST": "WEST", "WEST": "EAST"}
    directions = []

    for direct in arr:
        if directions and directions[-1] == opos[direct]:
            directions.pop()
        else:
            directions += [direct]
            # directions.append(direct)
    
    return directions
dir_reduc(["NORTH", "SOUTH", "SOUTH", "EAST", "WEST", "NORTH", "WEST"])
dir_reduc(["NORTH", "WEST", "SOUTH", "EAST"])

# from codewars
def dir_reduc(arr):
    arr_mod = " ".join(arr).replace("NORTH SOUTH", "").replace("SOUTH NORTH", "").replace("EAST WEST", "").replace("WEST EAST", "")
    arr_mod = arr_mod.split()
    return arr_mod if len(arr) == len(arr_mod) else dir_reduc(arr_mod)

import re
def dir_reduc(arr):
    arr2 = ' '.join(arr)
    arr2 = re.sub(r'NORTH SOUTH\s?', '', arr2)
    arr2 = re.sub(r'SOUTH NORTH\s?', '', arr2)
    arr2 = re.sub(r'EAST WEST\s?', '', arr2)
    arr2 = re.sub(r'WEST EAST\s?', '', arr2)
    arr2 = arr2.split()
    return arr2 if len(arr) == len(arr2) else dir_reduc(arr2)

import re
def dir_reduc(arr):
    arr2 = ' '.join(arr)
    rep = True
    while rep:
        test_len = len(arr2)
        arr2 = re.sub(r'NORTH SOUTH\s?', '', arr2)
        arr2 = re.sub(r'SOUTH NORTH\s?', '', arr2)
        arr2 = re.sub(r'EAST WEST\s?', '', arr2)
        arr2 = re.sub(r'WEST EAST\s?', '', arr2)
        if test_len == len(arr2):
            rep = False
    return arr2.split()


a = ["NORTH", "SOUTH", "SOUTH", "EAST", "WEST", "NORTH", "WEST"]
test.assert_equals(dir_reduc(a), ['WEST'])
u=["NORTH", "WEST", "SOUTH", "EAST"]
test.assert_equals(dir_reduc(u), ["NORTH", "WEST", "SOUTH", "EAST"])





# Exes and Ohs
# https://www.codewars.com/kata/55908aad6620c066bc00002a
"""Check to see if a string has the same amount of 'x's and 'o's. The method must return a boolean and be case insensitive. The string can contain any char.

Examples input/output:

XO("ooxx") => true
XO("xooxx") => false
XO("ooxXm") => true
XO("zpzpzpp") => true // when no 'x' and 'o' is present should return true
XO("zzoo") => false"""


def xo(s):
    return s.lower().count('x') == s.lower().count('o')
xo('xo')


test.expect(xo('xo'), 'True expected')
test.expect(xo('xo0'), 'True expected')
test.expect(not xo('xxxoo'), 'False expected')





# Take a Number And Sum Its Digits Raised To The Consecutive Powers And ....¡Eureka!!
# https://www.codewars.com/kata/5626b561280a42ecc50000d1
"""The number 89 is the first integer with more than one digit that fulfills the property partially introduced in the title of this kata. What's the use of saying "Eureka"? Because this sum gives the same number.

In effect: 89 = 8^1 + 9^2

The next number in having this property is 135.

See this property again: 135 = 1^1 + 3^2 + 5^3

We need a function to collect these numbers, that may receive two integers a, b that defines the range [a, b] (inclusive) and outputs a list of the sorted numbers in the range that fulfills the property described above.

Let's see some cases:

sum_dig_pow(1, 10) == [1, 2, 3, 4, 5, 6, 7, 8, 9]

sum_dig_pow(1, 100) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 89]
If there are no numbers of this kind in the range [a, b] the function should output an empty list.

sum_dig_pow(90, 100) == []"""


# from codewars
def is_power_sum(number):
    return number == sum(int(digit) ** i for i, digit in enumerate(str(number), 1))

def sum_dig_pow(a, b):
    return list(filter(is_power_sum, range(a, b + 1)))
sum_dig_pow(1, 100)
sum_dig_pow(89, 135)

def sum_dig_pow(a, b):
    return [number for number in range(a, b + 1) if is_power_sum(number)]

def sum_dig_pow(a, b):  # range(a, b + 1) will be studied by the function
    return [number for number in range(a, b + 1) if number == sum(int(digit) ** (i + 1) for i, digit in enumerate(str(number)))]


test.describe("Example Tests")
test.assert_equals(sum_dig_pow(1, 10), [1, 2, 3, 4, 5, 6, 7, 8, 9])
test.assert_equals(sum_dig_pow(1, 100), [1, 2, 3, 4, 5, 6, 7, 8, 9, 89])
test.assert_equals(sum_dig_pow(10, 89),  [89])
test.assert_equals(sum_dig_pow(10, 100),  [89])
test.assert_equals(sum_dig_pow(90, 100), [])
test.assert_equals(sum_dig_pow(89, 135), [89, 135])





# Categorize New Member
# https://www.codewars.com/kata/5502c9e7b3216ec63c0001aa
"""The Western Suburbs Croquet Club has two categories of membership, Senior and Open. They would like your help with an application form that will tell prospective members which category they will be placed.

To be a senior, a member must be at least 55 years old and have a handicap greater than 7. In this croquet club, handicaps range from -2 to +26; the better the player the lower the handicap.

Input
Input will consist of a list of pairs. Each pair contains information for a single potential member. Information consists of an integer for the person's age and an integer for the person's handicap.

Output
Output will consist of a list of string values (in Haskell and C: Open or Senior) stating whether the respective member is to be placed in the senior or open category.

Example
input =  [[18, 20], [45, 2], [61, 12], [37, 6], [21, 21], [78, 9]]
output = ["Open", "Open", "Senior", "Open", "Open", "Senior"]
"""


def open_or_senior(data):
    return ['Senior' if age > 54 and handicap > 7 else 'Open' for age, handicap in data]
open_or_senior([(45, 12),(55,21),(19, -2),(104, 20)])


import codewars_test as test
from solution import open_or_senior

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():    
        test.assert_equals(open_or_senior([(45, 12),(55,21),(19, -2),(104, 20)]),['Open', 'Senior', 'Open', 'Senior'])
        test.assert_equals(open_or_senior([(16, 23),(73,1),(56, 20),(1, -1)]),['Open', 'Open', 'Senior', 'Open'])





# Opposites Attract
# https://www.codewars.com/kata/555086d53eac039a2a000083
"""Timmy & Sarah think they are in love, but around where they live, they will only know once they pick a flower each. If one of the flowers has an even number of petals and the other has an odd number of petals it means they are in love.

Write a function that will take the number of petals of each flower and return true if they are in love and false if they aren't.

"""


def lovefunc(flower1, flower2):
    return bool((flower1 + flower2) % 2)
lovefunc(1,4)
lovefunc(2,2)
lovefunc(0,1)
lovefunc(0,0)

def lovefunc(flower1, flower2):
    return (flower1 + flower2) % 2

def lovefunc( flower1, flower2 ):
    return (flower1 + flower2) % 2 == 1

def lovefunc(flower1, flower2):
    return bool(flower1 % 2 ^ flower2 % 2)


import codewars_test as test
from solution import lovefunc

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(lovefunc(1,4), True)
        test.assert_equals(lovefunc(2,2), False)
        test.assert_equals(lovefunc(0,1), True)
        test.assert_equals(lovefunc(0,0), False)





# Playing with digits
# https://www.codewars.com/kata/5552101f47fc5178b1000050
"""Some numbers have funny properties. For example:

89 --> 8¹ + 9² = 89 * 1

695 --> 6² + 9³ + 5⁴= 1390 = 695 * 2

46288 --> 4³ + 6⁴+ 2⁵ + 8⁶ + 8⁷ = 2360688 = 46288 * 51

Given a positive integer n written as abcd... (a, b, c, d... being digits) and a positive integer p

we want to find a positive integer k, if it exists, such that the sum of the digits of n taken to the successive powers of p is equal to k * n.
In other words:

Is there an integer k such as : (a ^ p + b ^ (p+1) + c ^(p+2) + d ^ (p+3) + ...) = n * k

If it is the case we will return k, if not return -1.

Note: n and p will always be given as strictly positive integers.

dig_pow(89, 1) should return 1 since 8¹ + 9² = 89 = 89 * 1
dig_pow(92, 1) should return -1 since there is no k such as 9¹ + 2² equals 92 * k
dig_pow(695, 2) should return 2 since 6² + 9³ + 5⁴= 1390 = 695 * 2
dig_pow(46288, 3) should return 51 since 4³ + 6⁴+ 2⁵ + 8⁶ + 8⁷ = 2360688 = 46288 * 51"""


def dig_pow(n, p):
    sum_of_pow = sum(pow(int(digit), ind) for ind, digit in enumerate(str(n), p)) / n
    return sum_of_pow if not sum_of_pow % 1 else -1
dig_pow(89, 1)
dig_pow(92, 1)
dig_pow(46288, 3)
dig_pow(695, 2)

def dig_pow(n, p):
    sum_of_pow = sum(int(digit) ** ind for ind, digit in enumerate(str(n), p)) / n
    return sum_of_pow if not sum_of_pow % 1 else -1

def dig_pow(n, p):
    sum_of_pow = sum(int(digit) ** ind for ind, digit in enumerate(str(n), p))
    return sum_of_pow // n if not sum_of_pow % n else -1

def dig_pow(n, p):
    div, mod = divmod(sum(int(digit) ** ind for ind, digit in enumerate(str(n), p)), n)
    return div if not mod else -1

import numpy as np
def dig_pow(n, p):
    the_sum = np.sum(np.power([int(i) for i in str(n)], range(p, p + len(str(n)))))
    return the_sum // n if not the_sum % n else -1


test.assert_equals(dig_pow(89, 1), 1)
test.assert_equals(dig_pow(92, 1), -1)
test.assert_equals(dig_pow(46288, 3), 51)




# L1: Set Alarm
# https://www.codewars.com/kata/568dcc3c7f12767a62000038
"""Write a function named setAlarm which receives two parameters. The first parameter, employed, is true whenever you are employed and the second parameter, vacation is true whenever you are on vacation.

The function should return true if you are employed and not on vacation (because these are the circumstances under which you need to set an alarm). It should return false otherwise. Examples:

setAlarm(true, true) -> false
setAlarm(false, true) -> false
setAlarm(false, false) -> false
setAlarm(true, false) -> true"""


def set_alarm(employed, vacation):
    return employed and not vacation


import codewars_test as test
from solution import set_alarm

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(set_alarm(True, True), False, "Fails when input is True, True")
        test.assert_equals(set_alarm(False, True), False, "Fails when input is False, True")
        test.assert_equals(set_alarm(False, False), False, "Fails when input is False, False")
        test.assert_equals(set_alarm(True, False), True, "Fails when input is True, False")





# Will there be enough space?
# https://www.codewars.com/kata/5875b200d520904a04000003
"""The Story:
Bob is working as a bus driver. However, he has become extremely popular amongst the city's residents. With so many passengers wanting to get aboard his bus, he sometimes has to face the problem of not enough space left on the bus! He wants you to write a simple program telling him if he will be able to fit all the passengers.

Task Overview:
You have to write a function that accepts three parameters:

cap is the amount of people the bus can hold excluding the driver.
on is the number of people on the bus excluding the driver.
wait is the number of people waiting to get on to the bus excluding the driver.
If there is enough space, return 0, and if there isn't, return the number of passengers he can't take.

Usage Examples:
cap = 10, on = 5, wait = 5 --> 0 # He can fit all 5 passengers
cap = 100, on = 60, wait = 50 --> 10 # He can't fit 10 of the 50 waiting"""


def enough(cap, on, wait):
    return -min(cap - on - wait, 0)
enough(10, 5, 5)
enough(100, 60, 5)
enough(100, 60, 50)
enough(20, 5, 5)

def enough(cap, on, wait):
    return max(0, wait - cap + on)


@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(enough(10, 5, 5), 0)
        test.assert_equals(enough(100, 60, 50), 10)
        test.assert_equals(enough(20, 5, 5), 0)





# Sum of the first nth term of Series
# https://www.codewars.com/kata/555eded1ad94b00403000071
"""Task:
Your task is to write a function which returns the sum of following series upto nth term(parameter).

Series: 1 + 1/4 + 1/7 + 1/10 + 1/13 + 1/16 +...
Rules:
You need to round the answer to 2 decimal places and return it as String.

If the given value is 0 then it should return 0.00

You will only be given Natural Numbers as arguments.

Examples:(Input --> Output)
1 --> 1 --> "1.00"
2 --> 1 + 1/4 --> "1.25"
5 --> 1 + 1/4 + 1/7 + 1/10 + 1/13 --> "1.57"
"""


def series_sum(n):
    sequence_sum = sum(1/(3*i + 1) for i in range(n))
    return '{:.2f}'.format(sequence_sum)
    return "%.2f" % sequence_sum
    return f"{sequence_sumr:.2f}"
series_sum(1)
series_sum(2)
series_sum(3)
series_sum(5)

def series_sum(n):
    if n == 1:
        return "1.00"
    if n == 0:
        return "0.00"
    the_sum = sum(1/((3*ind) + 1) for ind in range(n))
    sol = str(round(the_sum, 2)) 
    return sol + "0" if len(sol) == 3 else sol

# prime numbers generator
def series_sum(n):
    prime_list = []
    i = 1
    while True:
        i += 1
        ifin = []
        for j in range(2, i + 1):
            ifin.append(not i % j)
        if sum(ifin) == 1:
            prime_list.append(i)
        if len(prime_list) == n:
            break
    return prime_list


import codewars_test as test
from solution import series_sum

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(series_sum(1), "1.00")
        test.assert_equals(series_sum(2), "1.25")
        test.assert_equals(series_sum(3), "1.39")





# Convert number to reversed array of digits
# https://www.codewars.com/kata/5583090cbe83f4fd8c000051
"""Convert number to reversed array of digits
Given a random non-negative number, you have to return the digits of this number within an array in reverse order.

Example:
348597 => [7,9,5,8,4,3]
0 => [0]
"""


def digitize(n):
    return list(reversed([int(i) for i in str(n)]))
    return [int(i) for i in str(n)][::-1]
    return [int(i) for i in reversed(str(n))]
    return list(map(int, str(n)[::-1]))
    return list(map(int, str(n)))[::-1]
digitize(35231)


import codewars_test as test
from solution import digitize

@test.describe("Fixed Tests")
def basic_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(digitize(35231),[1,3,2,5,3])
        test.assert_equals(digitize(0),[0])
        test.assert_equals(digitize(23582357),[7,5,3,2,8,5,3,2])
        test.assert_equals(digitize(984764738),[8,3,7,4,6,7,4,8,9])
        test.assert_equals(digitize(45762893920),[0,2,9,3,9,8,2,6,7,5,4])
        test.assert_equals(digitize(548702838394),[4,9,3,8,3,8,2,0,7,8,4,5])





# Get the Middle Character
# https://www.codewars.com/kata/56747fd5cb988479af000028
"""est") should return "es"

Kata.getMiddle("testing") should return "t"

Kata.getMiddle("middle") should return "dd"

Kata.getMiddle("A") should return "A"
#Input

A word (string) of length 0 < str < 1000 (In javascript you may get slightly more than 1000 in some test cases due to an error in the test cases). You do not need to test for this. This is only here to tell you that you do not need to worry about your solution timing out.

#Output

The middle character(s) of the word represented as a string."""


def get_middle(s):
    len_s = len(s)
    if len_s % 2:
        return s[(len_s-1) // 2]
    else:
        return s[len_s//2 - 1 : len_s//2 + 1]
get_middle("testing")
get_middle("test")
get_middle('')

def get_middle(s):
    return get_middle(s[1:-1]) if len(s) > 2 else s

def get_middle(s):
    return s[(len(s)-1) // 2 : len(s)//2 + 1]


test.assert_equals(get_middle("test"),"es")
test.assert_equals(get_middle("testing"),"t")
test.assert_equals(get_middle("middle"),"dd")
test.assert_equals(get_middle("A"),"A")
test.assert_equals(get_middle("of"),"of")




# Transportation on vacation
# https://www.codewars.com/kata/568d0dd208ee69389d000016
"""After a hard quarter in the office you decide to get some rest on a vacation. So you will book a flight for you and your girlfriend and try to leave all the mess behind you.

You will need a rental car in order for you to get around in your vacation. The manager of the car rental makes you some good offers.

Every day you rent the car costs $40. If you rent the car for 7 or more days, you get $50 off your total. Alternatively, if you rent the car for 3 or more days, you get $20 off your total.

Write a code that gives out the total amount for different days(d).

"""


def rental_car_cost(d):
    if d >= 7:
        return d * 40 - 50
    if d >= 3:
        return d * 40 - 20
    else:
        return d * 40
    
def rental_car_cost(d):
    return 40*d if d < 3 else 40*d - 20 if d < 7 else 40*d - 50
rental_car_cost(1)
rental_car_cost(4)
rental_car_cost(7)
rental_car_cost(8)

# from codewars
def rental_car_cost(d):
    return 40*d - (d > 2)*20 - (d > 6)*30


import codewars_test as test
from solution import rental_car_cost

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(rental_car_cost(1),40)
        test.assert_equals(rental_car_cost(4),140)
        test.assert_equals(rental_car_cost(7),230)
        test.assert_equals(rental_car_cost(8),270)





# Abbreviate a Two Word Name
# https://www.codewars.com/kata/57eadb7ecd143f4c9c0000a3
"""Write a function to convert a name into initials. This kata strictly takes two words with one space in between them.

The output should be two capital letters with a dot separating them.

It should look like this:

Sam Harris => S.H

patrick feeney => P.F"""


def abbrev_name(name):
    return ".".join(i[0].upper() for i in name.split())
    return '.'.join(i[0] for i in name.split()).upper()
    return ".".join(map(lambda x: x[0].upper(), name.split()))
abbrev_name("Sam Harris")


import codewars_test as test

try:
    from solution import abbrevName as abbrev_name
except ImportError:
    from solution import abbrev_name

@test.describe("Fixed Tests")
def basic_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(abbrev_name("Sam Harris"), "S.H")
        test.assert_equals(abbrev_name("patrick feenan"), "P.F")
        test.assert_equals(abbrev_name("Evan C"), "E.C")
        test.assert_equals(abbrev_name("P Favuzzi"), "P.F")
        test.assert_equals(abbrev_name("David Mendieta"), "D.M")





# Count by X
# https://www.codewars.com/kata/5513795bd3fafb56c200049e
"""Create a function with two arguments that will return an array of the first (n) multiples of (x).

Assume both the given number and the number of times to count will be positive numbers greater than 0.

Return the results as an array (or list in Python, Haskell or Elixir).

Examples:

count_by(1,10) #should return [1,2,3,4,5,6,7,8,9,10]
count_by(2,5) #should return [2,4,6,8,10]"""


def count_by(x, n):
    return list(range(x, n*x + 1, x))
    return [x * i for i in range(1, n + 1)]
    return list(map(lambda y: x*y, range(1, n+1)))
count_by(2, 5)

import numpy as np
def count_by(x, n):
    return list(np.array(range(1, n + 1)) * x)


import codewars_test as test
from solution import count_by

@test.describe("Fixed Tests")
def basic_tests():
    @test.it("Fixed tests")
    def fixed_tests():   
        test.assert_equals(count_by(1, 5), [1, 2, 3, 4, 5])
        test.assert_equals(count_by(2, 5), [2, 4, 6, 8, 10])
        test.assert_equals(count_by(3, 5), [3, 6, 9, 12, 15])
        test.assert_equals(count_by(50, 5), [50, 100, 150, 200, 250])
        test.assert_equals(count_by(100, 5), [100, 200, 300, 400, 500])




# Beginner Series #1 School Paperwork
# https://www.codewars.com/kata/55f9b48403f6b87a7c0000bd
"""Your classmates asked you to copy some paperwork for them. You know that there are 'n' classmates and the paperwork has 'm' pages.

Your task is to calculate how many blank pages do you need. If n < 0 or m < 0 return 0.

Example:
n= 5, m=5: 25
n=-5, m=5:  0"""


def paperwork(n, m):
    return n * m if n > 0 and m > 0 else 0
    return max(n, 0) * max(m, 0)
paperwork(5, 5)
paperwork(-5, 5)


import codewars_test as test
from solution import paperwork

@test.describe("Fixed Tests")
def basic_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(paperwork(5,5), 25, "Failed at Paperwork(5,5)")
        test.assert_equals(paperwork(-5,5), 0, "Failed at Paperwork(-5,5)")
        test.assert_equals(paperwork(5,-5), 0, "Failed at Paperwork(5,-5)")
        test.assert_equals(paperwork(-5,-5), 0, "Failed at Paperwork(-5,-5)")
        test.assert_equals(paperwork(5,0), 0, "Failed at Paperwork(5,0)")





# Ones and Zeros
# https://www.codewars.com/kata/578553c3a1b8d5c40300037c
"""Given an array of ones and zeroes, convert the equivalent binary value to an integer.

Eg: [0, 0, 0, 1] is treated as 0001 which is the binary representation of 1.

Examples:

Testing: [0, 0, 0, 1] ==> 1
Testing: [0, 0, 1, 0] ==> 2
Testing: [0, 1, 0, 1] ==> 5
Testing: [1, 0, 0, 1] ==> 9
Testing: [0, 0, 1, 0] ==> 2
Testing: [0, 1, 1, 0] ==> 6
Testing: [1, 1, 1, 1] ==> 15
Testing: [1, 0, 1, 1] ==> 11
However, the arrays can have varying lengths, not just limited to 4."""


def binary_array_to_number(arr):
    return int(''.join(map(str, arr)), 2)
binary_array_to_number([0, 0, 0, 1])
binary_array_to_number([1, 1, 1, 1])

def binary_array_to_number(arr):
    return int(''.join(str(i) for i in arr), 2)


import codewars_test as test
from solution import binary_array_to_number

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(binary_array_to_number([0,0,0,1]), 1)
        test.assert_equals(binary_array_to_number([0,0,1,0]), 2)
        test.assert_equals(binary_array_to_number([1,1,1,1]), 15)
        test.assert_equals(binary_array_to_number([0,1,1,0]), 6)





# Shortest Word
# https://www.codewars.com/kata/57cebe1dc6fdc20c57000ac9r
"""Simple, given a string of words, return the length of the shortest word(s).

String will never be empty and you do not need to account for different data types."""


def find_short(s):
    return len(min(s.split(), key=len))
    return min(map(len, s.split()))
    return min(len(i) for i in s.split())


import codewars_test as test
from solution import find_short

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(find_short("bitcoin take over the world maybe who knows perhaps"), 3)
        test.assert_equals(find_short("turns out random test cases are easier than writing out basic ones"), 3)
        test.assert_equals(find_short("lets talk about javascript the best language"), 3)
        test.assert_equals(find_short("i want to travel the world writing code one day"), 1)
        test.assert_equals(find_short("Lets all go on holiday somewhere very cold"), 2)   
        test.assert_equals(find_short("Let's travel abroad shall we"), 2)





# Fake Binary
# https://www.codewars.com/kata/57eae65a4321032ce000002d
"""Given a string of digits, you should replace any digit below 5 with '0' and any digit 5 and above with '1'. Return the resulting string.

Note: input will never be an empty string"""


def fake_bin(x):
    return ''.join('0' if int(i) < 5 else '1' for i in x)
    return ''.join('0' if i < '5' else '1' for i in x)
    return ''.join(map(lambda x: '0' if int(x) < 5 else '1', x))
fake_bin("45385593107843568")


import codewars_test as test
from solution import fake_bin

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        tests = [
            # [expected, input]
            ["01011110001100111", "45385593107843568"],
            ["101000111101101", "509321967506747"],
            ["011011110000101010000011011", "366058562030849490134388085"],
            ["01111100", "15889923"],
            ["100111001111", "800857237867"],
        ]
        
        for exp, inp in tests:
            test.assert_equals(fake_bin(inp), exp)





# Take a Ten Minutes Walk
# https://www.codewars.com/kata/54da539698b8a2ad76000228
"""You live in the city of Cartesia where all roads are laid out in a perfect grid. You arrived ten minutes too early to an appointment, so you decided to take the opportunity to go for a short walk. The city provides its citizens with a Walk Generating App on their phones -- everytime you press the button it sends you an array of one-letter strings representing directions to walk (eg. ['n', 's', 'w', 'e']). You always walk only a single block for each letter (direction) and you know it takes you one minute to traverse one city block, so create a function that will return true if the walk the app gives you will take you exactly ten minutes (you don't want to be early or late!) and will, of course, return you to your starting point. Return false otherwise.

Note: you will always receive a valid array (string in COBOL) containing a random assortment of direction letters ('n', 's', 'e', or 'w' only). It will never give you an empty array (that's not a walk, that's standing still!)."""


def is_valid_walk(walk):
    return walk.count('n') == walk.count('s') and walk.count('e') == walk.count('w') and len(walk) == 10
is_valid_walk(['n','s','n','s','n','s','n','s','n','s'])


#some test cases for you...
test.expect(is_valid_walk(['n','s','n','s','n','s','n','s','n','s']), 'should return True');
test.expect(not is_valid_walk(['w','e','w','e','w','e','w','e','w','e','w','e']), 'should return False');
test.expect(not is_valid_walk(['w']), 'should return False');
test.expect(not is_valid_walk(['n','n','n','s','n','s','n','s','n','s']), 'should return False');





# Calculate average
# https://www.codewars.com/kata/57a2013acf1fa5bfc4000921
"""Write a function which calculates the average of the numbers in a given list.

Note: Empty arrays should return 0."""


def find_average(numbers):
    return sum(numbers)/len(numbers) if numbers else 0
find_average([1, 2, 3])
find_average([])

import numpy as np

def find_average(numbers):
    return np.average(numbers)

def find_average(numbers):
    try:
        return sum(numbers) / len(numbers) 
    except ZeroDivisionError:
        return 0


import codewars_test as test
from solution import find_average

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(find_average([1, 2, 3]), 2)





# Sum Mixed Array
# https://www.codewars.com/kata/57eaeb9578748ff92a000009
"""Given an array of integers as strings and numbers, return the sum of the array values as if all were numbers.

Return your answer as a number."""


def sum_mix(arr):
    return sum(map(int, arr))
sum_mix([9, 3, '7', '3'])


import codewars_test as test
from solution import sum_mix

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(sum_mix([9, 3, '7', '3']), 22)
        test.assert_equals(sum_mix(['5', '0', 9, 3, 2, 1, '9', 6, 7]), 42)
        test.assert_equals(sum_mix(['3', 6, 6, 0, '5', 8, 5, '6', 2,'0']), 41)
        test.assert_equals(sum_mix(['1', '5', '8', 8, 9, 9, 2, '3']), 45)
        test.assert_equals(sum_mix([8, 0, 0, 8, 5, 7, 2, 3, 7, 8, 6, 7]), 61)





# Switch it Up!
# https://www.codewars.com/kata/5808dcb8f0ed42ae34000031
"""When provided with a number between 0-9, return it in words.

Input :: 1

Output :: "One".

If your language supports it, try using a switch statement."""


def switch_it_up(number):
    number_dict = {0: 'Zero',
                   1: 'One',
                   2: 'Two',
                   3: 'Three',
                   4: 'Four',
                   5: 'Five',
                   6: 'Six',
                   7: 'Seven',
                   8: 'Eight',
                   9: 'Nine'}
    return number_dict[number]
switch_it_up(0)
switch_it_up(9)

def switch_it_up(number):
    return 'Zero One Two Three Four Five Six Seven Eight Nine'.split()[number]


import codewars_test as test
from solution import switch_it_up

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(switch_it_up(0), "Zero")
        test.assert_equals(switch_it_up(9), "Nine")





# Array.diff
# https://www.codewars.com/kata/523f5d21c841566fde000009
"""Your goal in this kata is to implement a difference function, which subtracts one list from another and returns the result.

It should remove all values from list a, which are present in list b keeping their order.

array_diff([1,2],[1]) == [2]
If a value is present in b, all of its occurrences must be removed from the other:

array_diff([1,2,2,2,3],[2]) == [1,3]"""


# from codewars
def array_diff(a, b):
    return [i for i in a if i not in set(b)]
array_diff([1, 2], [1])
array_diff([1, 2, 2, 3], [2])

# from codewars
def array_diff(a, b):
    return list(filter(lambda x: x not in b, a))

def array_diff(a, b):
    for i in b:
        for _ in range(a.count(i)):
            a.remove(i)
    return a

def array_diff(a, b):
    a = ''.join(map(str, a))
    for i in b:
        a = a.replace(str(i), '')
    return [int(i) for i in a]


import codewars_test as test
from solution import array_diff

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(array_diff([1,2], [1]), [2], "a was [1,2], b was [1], expected [2]")
        test.assert_equals(array_diff([1,2,2], [1]), [2,2], "a was [1,2,2], b was [1], expected [2,2]")
        test.assert_equals(array_diff([1,2,2], [2]), [1], "a was [1,2,2], b was [2], expected [1]")
        test.assert_equals(array_diff([1,2,2], []), [1,2,2], "a was [1,2,2], b was [], expected [1,2,2]")
        test.assert_equals(array_diff([], [1,2]), [], "a was [], b was [1,2], expected []")
        test.assert_equals(array_diff([1,2,3], [1, 2]), [3], "a was [1,2,3], b was [1, 2], expected [3]")





# Welcome!
# https://www.codewars.com/kata/577ff15ad648a14b780000e7
"""Your start-up's BA has told marketing that your website has a large audience in Scandinavia and surrounding countries. Marketing thinks it would be great to welcome visitors to the site in their own language. Luckily you already use an API that detects the user's location, so this is an easy win.

The Task
Think of a way to store the languages as a database (eg an object). The languages are listed below so you can copy and paste!
Write a 'welcome' function that takes a parameter 'language' (always a string), and returns a greeting - if you have it in your database. It should default to English if the language is not in the database, or in the event of an invalid input.
The Database
'english': 'Welcome',
'czech': 'Vitejte',
'danish': 'Velkomst',
'dutch': 'Welkom',
'estonian': 'Tere tulemast',
'finnish': 'Tervetuloa',
'flemish': 'Welgekomen',
'french': 'Bienvenue',
'german': 'Willkommen',
'irish': 'Failte',
'italian': 'Benvenuto',
'latvian': 'Gaidits',
'lithuanian': 'Laukiamas',
'polish': 'Witamy',
'spanish': 'Bienvenido',
'swedish': 'Valkommen',
'welsh': 'Croeso'
Possible invalid inputs include:

IP_ADDRESS_INVALID - not a valid ipv4 or ipv6 ip address
IP_ADDRESS_NOT_FOUND - ip address not in the database
IP_ADDRESS_REQUIRED - no ip address was supplied"""


def greet(language):
    lan_dict = [("english", "Welcome"), ("czech", "Vitejte"), ("danish", "Velkomst"), ("dutch", "Welkom"), ("estonian", "Tere tulemast"), ("finnish", "Tervetuloa"), ("flemish", "Welgekomen"), ("french", "Bienvenue"), ("german", "Willkommen"), ("irish", "Failte"), ("italian", "Benvenuto"), ("latvian", "Gaidits"), ("lithuanian", "Laukiamas"), ("polish", "Witamy"), ("spanish", "Bienvenido"), ("swedish", "Valkommen"), ("welsh", "Croeso")                ]
    real_dict = {k: v for k, v in lan_dict}
    # return real_dict[language] if language in list(zip(*lan_dict))[0] else "Welcome"
    # return real_dict[language] if language in real_dict.keys() else "Welcome"
    return real_dict.get(language, real_dict["english"])
greet('dutch')
greet('IP_ADDRESS_INVALID')
greet('')
greet(2)

# from codewars
def greet(language):
    lang_db = {'english': 'Welcome',
               'czech': 'Vitejte',
               'danish': 'Velkomst',
               'dutch': 'Welkom',
               'estonian': 'Tere tulemast',
               'finnish': 'Tervetuloa',
               'flemish': 'Welgekomen',
               'french': 'Bienvenue',
               'german': 'Willkommen',
               'irish': 'Failte',
               'italian': 'Benvenuto',
               'latvian': 'Gaidits',
               'lithuanian': 'Laukiamas',
               'polish': 'Witamy',
               'spanish': 'Bienvenido',
               'swedish': 'Valkommen',
               'welsh': 'Croeso'
               }
    return lang_db.get(language, lang_db['english'])

def greet(language):
    lang_db = {'english': 'Welcome',
               'czech': 'Vitejte',
               'danish': 'Velkomst',
               'dutch': 'Welkom',
               'estonian': 'Tere tulemast',
               'finnish': 'Tervetuloa',
               'flemish': 'Welgekomen',
               'french': 'Bienvenue',
               'german': 'Willkommen',
               'irish': 'Failte',
               'italian': 'Benvenuto',
               'latvian': 'Gaidits',
               'lithuanian': 'Laukiamas',
               'polish': 'Witamy',
               'spanish': 'Bienvenido',
               'swedish': 'Valkommen',
               'welsh': 'Croeso'
               }
    if language == 'IP_ADDRESS_INVALID'\
        or language == 'IP_ADDRESS_NOT_FOUND'\
        or language == 'IP_ADDRESS_REQUIRED'\
        or language == ''\
        or type(language) != str:
        # or isinstance(language, str) == False:
        return lang_db['english']
    return lang_db.get(language)


test.describe("Basic tests")
test.assert_equals(greet('english'), 'Welcome')
test.assert_equals(greet('dutch'), 'Welkom')
test.assert_equals(greet('IP_ADDRESS_INVALID'), 'Welcome')
test.assert_equals(greet(''), 'Welcome')
test.assert_equals(greet(2), 'Welcome')





# Area or Perimeter
# https://www.codewars.com/kata/5ab6538b379d20ad880000ab
"""You are given the length and width of a 4-sided polygon. The polygon can either be a rectangle or a square.
If it is a square, return its area. If it is a rectangle, return its perimeter.

area_or_perimeter(6, 10) --> 32
area_or_perimeter(3, 3) --> 9
Note: for the purposes of this kata you will assume that it is a square if its length and width are equal, otherwise it is a rectangle."""


def area_or_perimeter(l, w):
    return l ** 2 if l == w else (l+w) << 1
area_or_perimeter(4, 4)
area_or_perimeter(6, 10)

area_or_perimeter = lambda l, w: l ** 2 if l == w else (l+w) << 1


import codewars_test as test
from solution import area_or_perimeter

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(area_or_perimeter(4, 4), 16)
        test.assert_equals(area_or_perimeter(6, 10), 32)





# Total amount of points
# https://www.codewars.com/kata/5bb904724c47249b10000131
"""Our football team finished the championship. The result of each match look like "x:y". Results of all matches are recorded in the collection.

For example: ["3:1", "2:2", "0:1", ...]

Write a function that takes such collection and counts the points of our team in the championship. Rules for counting points for each match:

if x > y: 3 points
if x < y: 0 point
if x = y: 1 point
Notes:

there are 10 matches in the championship
0 <= x <= 4
0 <= y <= 4"""


def points(games):
    return sum(3 if x > y else 1 for x, _, y in games if x >= y)
    return sum(3 if x[0] > x[-1] else 1 for x in games if x[0] >= x[-1])
points(['1:0','2:0','3:0','4:0','2:1','3:1','4:1','3:2','4:2','4:3'])

# from codewars
def points(games):
    return sum((x > y)*3 or x == y for x, _, y in games)

# from codewars
def points(games):
    return sum((x >= y) + 2 * (x > y) for x, y in (s.split(":") for s in games))

def points(games):
	return sum(3 if i[0] > i[-1] else 1 if i[0] == i[-1] else 0 for i in games)

def points(games):
	return sum(map(lambda x: 3 if x[0] > x[-1] else 1 if x[0] == x[-1] else 0, games))


1 > 0 * 3

import codewars_test as test
from solution import points

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(points(['1:0','2:0','3:0','4:0','2:1','3:1','4:1','3:2','4:2','4:3']), 30)
        test.assert_equals(points(['1:1','2:2','3:3','4:4','2:2','3:3','4:4','3:3','4:4','4:4']), 10)
        test.assert_equals(points(['0:1','0:2','0:3','0:4','1:2','1:3','1:4','2:3','2:4','3:4']), 0)
        test.assert_equals(points(['1:0','2:0','3:0','4:0','2:1','1:3','1:4','2:3','2:4','3:4']), 15)
        test.assert_equals(points(['1:0','2:0','3:0','4:4','2:2','3:3','1:4','2:3','2:4','3:4']), 12)





# Two Sum
# https://www.codewars.com/kata/52c31f8e6605bcc646000082
"""Write a function that takes an array of numbers (integers for the tests) and a target number. It should find two different items in the array that, when added together, give the target value. The indices of these items should then be returned in a tuple / list (depending on your language) like so: (index1, index2).

For the purposes of this kata, some tests may have multiple answers; any valid solutions will be accepted.

The input will always be valid (numbers will be an array of length 2 or greater, and all of the items will be numbers; target will always be the sum of two different items from that array).

Based on: http://oj.leetcode.com/problems/two-sum/

twoSum [1, 2, 3] 4 === (0, 2)"""


def two_sum(numbers, target):
    for i in range(len(numbers)):
        for j in range(i):
            if numbers[i] + numbers[j] == target:
                return (i, j)
two_sum([1, 2, 3], 4)

def two_sum(numbers, target):
    for i in range(len(numbers) - 1):
        for j in range(i + 1, len(numbers)):
            if numbers[i] + numbers[j] == target:
                return (i, j)

def two_sum(numbers, target):
    for i, x in enumerate(numbers):
        for j, y in enumerate(numbers):
            if i != j and x + y == target:
                return (i, j)


test.assert_equals(sorted(two_sum([1,2,3], 4)), [0,2])
test.assert_equals(sorted(two_sum([1234,5678,9012], 14690)), [1,2])
test.assert_equals(sorted(two_sum([2,2,3], 4)), [0,1])





# How good are you really?
# https://www.codewars.com/kata/5601409514fc93442500010b
"""There was a test in your class and you passed it. Congratulations!
But you're an ambitious person. You want to know if you're better than the average student in your class.

You receive an array with your peers' test scores. Now calculate the average and compare your score!

Return True if you're better, else False!

Note:
Your points are not included in the array of your class's points. For calculating the average point you may add your point to the given array!"""


def better_than_average(class_points, your_points):
    return your_points > ((sum(class_points) + your_points)/(len(class_points) + 1)) # including your_points in the sum
    return your_points > (sum(class_points)/len(class_points))
better_than_average([100, 40, 34, 57, 29, 72, 57, 88], 75)

import numpy as np
def better_than_average(class_points, your_points):
    return your_points > np.mean(class_points + [your_points])

better_than_average = lambda class_points, your_points: your_points > np.mean(class_points + [your_points])


test.describe("Basic Tests")

test.it("better_than_average([2, 3], 5) should return True")
test.assert_equals(better_than_average([2, 3], 5), True)

test.it("better_than_average([100, 40, 34, 57, 29, 72, 57, 88], 75) should return True")
test.assert_equals(better_than_average([100, 40, 34, 57, 29, 72, 57, 88], 75), True)

test.it("better_than_average([12, 23, 34, 45, 56, 67, 78, 89, 90], 69) should return True")
test.assert_equals(better_than_average([12, 23, 34, 45, 56, 67, 78, 89, 90], 69), True)

test.it("better_than_average([41, 75, 72, 56, 80, 82, 81, 33], 50) should return False")
test.assert_equals(better_than_average([41, 75, 72, 56, 80, 82, 81, 33], 50), False)

test.it("better_than_average([29, 55, 74, 60, 11, 90, 67, 28], 21) should return False")
test.assert_equals(better_than_average([29, 55, 74, 60, 11, 90, 67, 28], 21), False)





"""altERnaTIng cAsE <=> ALTerNAtiNG CaSe
altERnaTIng cAsE <=> ALTerNAtiNG CaSe
Define String.prototype.toAlternatingCase (or a similar function/method such as to_alternating_case/toAlternatingCase/ToAlternatingCase in your selected language; see the initial solution for details) such that each lowercase letter becomes uppercase and each uppercase letter becomes lowercase. For example:

"hello world".toAlternatingCase() === "HELLO WORLD"
"HELLO WORLD".toAlternatingCase() === "hello world"
"hello WORLD".toAlternatingCase() === "HELLO world"
"HeLLo WoRLD".toAlternatingCase() === "hEllO wOrld"
"12345".toAlternatingCase() === "12345" // Non-alphabetical characters are unaffected
"1a2b3c4d5e".toAlternatingCase() === "1A2B3C4D5E"
"String.prototype.toAlternatingCase".toAlternatingCase() === "sTRING.PROTOTYPE.TOaLTERNATINGcASE"
As usual, your function/method should be pure, i.e. it should not mutate the original string."""


def to_alternating_case(string):
    return string.swapcase()
    return "".join(i.upper() if i.islower() else i.lower() for i in string)

to_alternating_case = lambda string: string.swapcase()
to_alternating_case = str.swapcase()
to_alternating_case("hello1 WORLD")


test.describe("Basic tests")
test.it("should work for fixed tests (provided in the description)")
test.assert_equals(to_alternating_case("hello world"), "HELLO WORLD")
test.assert_equals(to_alternating_case("HELLO WORLD"), "hello world")
test.assert_equals(to_alternating_case("hello WORLD"), "HELLO world")
test.assert_equals(to_alternating_case("HeLLo WoRLD"), "hEllO wOrld")
test.assert_equals(to_alternating_case("12345"), "12345")
test.assert_equals(to_alternating_case("1a2b3c4d5e"), "1A2B3C4D5E")
test.assert_equals(to_alternating_case("String.prototype.toAlternatingCase"), "sTRING.PROTOTYPE.TOaLTERNATINGcASE")
test.assert_equals(to_alternating_case(to_alternating_case("Hello World")), "Hello World")
test.it("should work for the title of this Kata")
title = "altERnaTIng cAsE"
title = to_alternating_case(title)
test.assert_equals(title, "ALTerNAtiNG CaSe")
title = to_alternating_case(title)
test.assert_equals(title, "altERnaTIng cAsE")
title = to_alternating_case(title)
test.assert_equals(title, "ALTerNAtiNG CaSe")
title = to_alternating_case(title)
test.assert_equals(title, "altERnaTIng cAsE")
title = "altERnaTIng cAsE <=> ALTerNAtiNG CaSe"
title = to_alternating_case(title)
test.assert_equals(title, "ALTerNAtiNG CaSe <=> altERnaTIng cAsE")
title = to_alternating_case(title)
test.assert_equals(title, "altERnaTIng cAsE <=> ALTerNAtiNG CaSe")
title = to_alternating_case(title)
test.assert_equals(title, "ALTerNAtiNG CaSe <=> altERnaTIng cAsE")
title = to_alternating_case(title)
test.assert_equals(title, "altERnaTIng cAsE <=> ALTerNAtiNG CaSe")





# Sort the odd
# https://www.codewars.com/kata/578aa45ee9fd15ff4600090d
"""You will be given an array of numbers. You have to sort the odd numbers in ascending order while leaving the even numbers at their original positions.

Examples
[7, 1]  =>  [1, 7]
[5, 8, 6, 3, 4]  =>  [3, 8, 6, 5, 4]
[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]  =>  [1, 8, 3, 6, 5, 4, 7, 2, 9, 0]"""


# partially from codewars
def sort_array(source_array):
    # odds = list(reversed(sorted(filter(lambda x: x % 2, source_array))))
    odds = sorted(filter(lambda x: x % 2, source_array), reverse=True)
    return [odds.pop() if i % 2 else i for i in source_array]
sort_array([5, 3, 2, 8, 1, 4])

# partially from codewars
def sort_array(source_array):
    odds = iter(sorted(filter(lambda x: x % 2, source_array)))
    return [next(odds) if i % 2 else i for i in source_array]

def sort_array(source_array):
    odds = sorted(filter(lambda x: x % 2, source_array))
    index = 0
    result_list = []
    for i in source_array:
        if i % 2:
            result_list.append(odds[index])
            index += 1
        else:
            result_list.append(i)
    return result_list


test.assert_equals(sort_array([5, 3, 2, 8, 1, 4]), [1, 3, 2, 8, 5, 4])
test.assert_equals(sort_array([5, 3, 1, 8, 0]), [1, 3, 5, 8, 0])
test.assert_equals(sort_array([]),[])





# Double Char
# https://www.codewars.com/kata/56b1f01c247c01db92000076
"""Given a string, you have to return a string in which each character (case-sensitive) is repeated once.

Examples (Input -> Output):
* "String"      -> "SSttrriinngg"
* "Hello World" -> "HHeelllloo  WWoorrlldd"
* "1234!_ "     -> "11223344!!__  """"


def double_char(s):
    return "".join(map(lambda x: 2*x, s))

double_char = lambda s: ''.join(i * 2 for i in s)
double_char("String")


test.assert_equals(double_char("String"),"SSttrriinngg")
test.assert_equals(double_char("Hello World"),"HHeelllloo  WWoorrlldd")
test.assert_equals(double_char("1234!_ "),"11223344!!__  ")





# Calculate BMI
# https://www.codewars.com/kata/57a429e253ba3381850000fb
"""Write function bmi that calculates body mass index (bmi = weight / height2).

if bmi <= 18.5 return "Underweight"

if bmi <= 25.0 return "Normal"

if bmi <= 30.0 return "Overweight"

if bmi > 30 return "Obese""""


def bmi(weight, height):
    b = weight / height ** 2
    if b > 30:
        return 'Obese'
    elif b > 25:
        return 'Overweight'
    elif b > 18.5:
        return 'Normal'
    else:
        return 'Underweight'
bmi(50, 1.80)
bmi(80, 1.80)
bmi(90, 1.80)


# from codewars
def bmi(weight, height):
    b = weight / height ** 2
    return ['Underweight', 'Normal', 'Overweight', 'Obese'][(b > 30) + (b > 25) + (b > 18.5)]

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(bmi(50, 1.80), "Underweight")
        test.assert_equals(bmi(80, 1.80), "Normal")
        test.assert_equals(bmi(90, 1.80), "Overweight")
        test.assert_equals(bmi(110, 1.80), "Obese")
        test.assert_equals(bmi(50, 1.50), "Normal")





# Sort array by string length
# https://www.codewars.com/kata/57ea5b0b75ae11d1e800006c
"""Write a function that takes an array of strings as an argument and returns a sorted array containing the same strings, ordered from shortest to longest.

For example, if this array were passed as an argument:

["Telescopes", "Glasses", "Eyes", "Monocles"]

Your function would return the following array:

["Eyes", "Glasses", "Monocles", "Telescopes"]

All of the strings in the array passed to your function will be different lengths, so you will not have to decide how to order multiple strings of the same length."""


sort_by_length = lambda arr: sorted(arr, key=len)
sort_by_length(["Telescopes", "Glasses", "Eyes", "Monocles"])

def sort_by_length(arr):
    # return sorted(arr, key=len)
    # return sorted(arr, key=lambda x: len(x))

def sort_by_length(arr):
    arr.sort(key=len)
    return arr


import codewars_test as test
from solution import sort_by_length

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():

        tests = [
            [["i", "to", "beg", "life"], ["beg", "life", "i", "to"]],
            [["", "pizza", "brains", "moderately"], ["", "moderately", "brains", "pizza"]],
            [["short", "longer", "longest"], ["longer", "longest", "short"]],
            [["a", "of", "dog", "food"], ["dog", "food", "a", "of"]],
            [["", "bees", "eloquent", "dictionary"], ["", "dictionary", "eloquent", "bees"]],
            [["a short sentence", "a longer sentence", "the longest sentence"], ["a longer sentence", "the longest sentence", "a short sentence"]],
        ]
        
        for exp, inp in tests:
            test.assert_equals(sort_by_length(inp), exp)





# Make a function that does arithmetic!
# https://www.codewars.com/kata/583f158ea20cfcbeb400000a
"""Given two numbers and an arithmetic operator (the name of it, as a string), return the result of the two numbers having that operator used on them.

a and b will both be positive integers, and a will always be the first number in the operation, and b always the second.

The four operators are "add", "subtract", "divide", "multiply".

A few examples:(Input1, Input2, Input3 --> Output)

5, 2, "add"      --> 7
5, 2, "subtract" --> 3
5, 2, "multiply" --> 10
5, 2, "divide"   --> 2.5
Try to do it without using if statements!"""


def arithmetic(a, b, operator):
    ar_dict = {'add': '+', 'subtract': '-', 'multiply': '*', 'divide': '/'}
    return eval(f"{a}{ar_dict[operator]}{b}")
    # return eval('(' + 'a' + ar_dict[operator] + 'b' + ')')
arithmetic(1, 2, "add")

# from codewars
def arithmetic(a, b, operator):
    return {'add': a + b, 'subtract': a - b, 'multiply': a * b, 'divide': a / b}[operator]

from operator import add, sub, mul, truediv
def arithmetic(a, b, operator):
    ops = {'add': add, 'subtract': sub, 'multiply': mul, 'divide': truediv}
    return ops[operator](a, b)


@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(arithmetic(1, 2, "add"), 3,
                           "'add' should return the two numbers added together!")
        test.assert_equals(arithmetic(8, 2, "subtract"), 6,
                           "'subtract' should return a minus b!")
        test.assert_equals(arithmetic(5, 2, "multiply"), 10,
                           "'multiply' should return a multiplied by b!")
        test.assert_equals(arithmetic(8, 2, "divide"), 4,
                           "'divide' should return a divided by b!")





# Beginner Series #3 Sum of Numbers
# https://www.codewars.com/kata/55f2b110f61eb01779000053
"""Given two integers a and b, which can be positive or negative, find the sum of all the integers between and including them and return it. If the two numbers are equal return a or b.

Note: a and b are not ordered!

Examples (a, b) --> output (explanation)
(1, 0) --> 1 (1 + 0 = 1)
(1, 2) --> 3 (1 + 2 = 3)
(0, 1) --> 1 (0 + 1 = 1)
(1, 1) --> 1 (1 since both are same)
(-1, 0) --> -1 (-1 + 0 = -1)
(-1, 2) --> 2 (-1 + 0 + 1 + 2 = 2)"""


def get_sum(a, b):
    a, b = sorted((a, b))
    return sum(range(a, b + 1))
get_sum(0, 1)

get_sum = lambda a, b: sum(range(min(a, b), max(a, b) + 1))

test.assert_equals(get_sum(0,1),1)
test.assert_equals(get_sum(0,-1),-1)





# Equal Sides Of An Array
# https://www.codewars.com/kata/5679aa472b8f57fb8c000047
"""You are going to be given an array of integers. Your job is to take that array and find an index N where the sum of the integers to the left of N is equal to the sum of the integers to the right of N. If there is no index that would make this happen, return -1.

For example:

Let's say you are given the array {1,2,3,4,3,2,1}:
Your function will return the index 3, because at the 3rd position of the array, the sum of left side of the index ({1,2,3}) and the sum of the right side of the index ({3,2,1}) both equal 6.

Let's look at another one.
You are given the array {1,100,50,-51,1,1}:
Your function will return the index 1, because at the 1st position of the array, the sum of left side of the index ({1}) and the sum of the right side of the index ({50,-51,1,1}) both equal 1.

Last one:
You are given the array {20,10,-80,10,10,15,35}
At index 0 the left side is {}
The right side is {10,-80,10,10,15,35}
They both are equal to 0 when added. (Empty arrays are equal to 0 in this problem)
Index 0 is the place where the left side and right side are equal.

Note: Please remember that in most programming/scripting languages the index of an array starts at 0.

Input:
An integer array of length 0 < arr < 1000. The numbers in the array can be any integer positive or negative.

Output:
The lowest index N where the side to the left of N is equal to the side to the right of N. If you do not find an index that fits these rules, then you will return -1.

Note:
If you are given an array with multiple answers, return the lowest correct index."""


def find_even_index(arr):
    r = [i for i in range(len(arr)) if (sum(arr[:i]) == sum(arr[i + 1:]))]
    return r[0] if r else -1
find_even_index([1, 2, 3, 4, 3, 2, 1])
find_even_index([1, 2, 3, 4, 5, 6])

def find_even_index(arr):
    try:
        return [i for i in range(len(arr)) if (sum(arr[:i]) == sum(arr[i + 1:]))][0]
    except:
        return -1

def find_even_index(arr):
    for i in range(len(arr)):
        if sum(arr[:i]) == sum(arr[i+1:]):
            return i
    return -1


test.assert_equals(find_even_index([1,2,3,4,3,2,1]),3)
test.assert_equals(find_even_index([1,100,50,-51,1,1]),1,)
test.assert_equals(find_even_index([1,2,3,4,5,6]),-1)
test.assert_equals(find_even_index([20,10,30,10,10,15,35]),3)
test.assert_equals(find_even_index([20,10,-80,10,10,15,35]),0)
test.assert_equals(find_even_index([10,-80,10,10,15,35,20]),6)
test.assert_equals(find_even_index(list(range(1,100))),-1)
test.assert_equals(find_even_index([0,0,0,0,0]),0,"Should pick the first index if more cases are valid")
test.assert_equals(find_even_index([-1,-2,-3,-4,-3,-2,-1]),3)
test.assert_equals(find_even_index(list(range(-100,-1))),-1)





# Simple multiplication
# https://www.codewars.com/kata/583710ccaa6717322c000105
"""This kata is about multiplying a given number by eight if it is an even number and by nine otherwise."""


def simple_multiplication(number):
    return number * 9 if number % 2 else number * 8
    return number * 8 if not number % 2 else number * 9
    return number * (8 + number % 2)
simple_multiplication(5)
simple_multiplication(6)

simple_multiplication = lambda number: number * (8 + (number&1))

4 & 1  # 0
100 & 1  # 0
5 & 1  # 1  
101 & 1  # 1

0b101 & 0b101  # 5
0b100 | 0b101  # 5
0b100 ^ 0b101  # 1

import codewars_test as test
from solution import simple_multiplication

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(simple_multiplication(2), 16)
        test.assert_equals(simple_multiplication(1), 9)
        test.assert_equals(simple_multiplication(8), 64)
        test.assert_equals(simple_multiplication(4), 32)
        test.assert_equals(simple_multiplication(5), 45)





# Sum of a sequence
# https://www.codewars.com/kata/586f6741c66d18c22800010a
"""Your task is to make function, which returns the sum of a sequence of integers.

The sequence is defined by 3 non-negative values: begin, end, step (inclusive).

If begin value is greater than the end, function should returns 0

Examples

2,2,2 --> 2
2,6,2 --> 12 (2 + 4 + 6)
1,5,1 --> 15 (1 + 2 + 3 + 4 + 5)
1,5,3  --> 5 (1 + 4)"""


def sequence_sum(begin_number, end_number, step):
    return sum(range(begin_number, end_number + 1, step))
sequence_sum(2, 6, 2)


import codewars_test as test
from solution import sequence_sum

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(sequence_sum(2, 6, 2), 12)
        test.assert_equals(sequence_sum(1, 5, 1), 15)
        test.assert_equals(sequence_sum(1, 5, 3), 5)
        test.assert_equals(sequence_sum(0, 15, 3), 45)
        test.assert_equals(sequence_sum(16, 15, 3), 0)
        test.assert_equals(sequence_sum(2, 24, 22), 26)
        test.assert_equals(sequence_sum(2, 2, 2), 2)
        test.assert_equals(sequence_sum(2, 2, 1), 2)
        test.assert_equals(sequence_sum(1, 15, 3), 35)
        test.assert_equals(sequence_sum(15, 1, 3), 0)





# Extract the domain name from a URL
# https://www.codewars.com/kata/514a024011ea4fb54200004b
"""Write a function that when given a URL as a string, parses out just the domain name and returns it as a string. For example:

* url = "http://github.com/carbonfive/raygun" -> domain name = "github"
* url = "http://www.zombie-bites.com"         -> domain name = "zombie-bites"
* url = "https://www.cnet.com"                -> domain name = cnet"
"""


import re
def domain_name(url):
    return re.search(r"(:\/\/www\.|:\/\/|www\.|^)([\w-]+)\.", url).group(2)

(:\/\/www\.|:\/\/|www\.)  # captures ://www.
(:\/\/|:\/\/www\.|www\.)  # captures :// and www.

def domain_name(url):
    return re.sub(r'^(https?://)?(www\.)?((\w|-)+)\..*$', r'\3', url)
domain_name('http://google.com')
domain_name('https://google.com')
domain_name('https://www.codewars.com')
domain_name('https://google.co.jp')
domain_name('www.xakep.ru')
domain_name('https://hyphen-site.org')
domain_name('icann.org')

# from codewars
import re
def domain_name(url):
    return re.match(r'(https?://)?(www\.)?(?P<domain>[\w-]+)\..*$', url).group('domain')

# from codewars
def domain_name(url):
    return url.split('//')[-1].split('www.')[-1].split('.')[0]
domain_name('https://en.wikipedia.org')
domain_name('http://www.googlewww.com')




test.assert_equals(domain_name("http://google.com"), "google")
test.assert_equals(domain_name("http://google.co.jp"), "google")
test.assert_equals(domain_name("www.xakep.ru"), "xakep")
test.assert_equals(domain_name("https://youtube.com"), "youtube")





# Find the middle element
# https://www.codewars.com/kata/545a4c5a61aa4c6916000755
"""As a part of this Kata, you need to create a function that when provided with a triplet, returns the index of the numerical element that lies between the other two elements.

The input to the function will be an array of three distinct numbers (Haskell: a tuple).

For example:

gimme([2, 3, 1]) => 0
2 is the number that fits between 1 and 3 and the index of 2 in the input array is 0.

Another example (just to make sure it is clear):

gimme([5, 10, 14]) => 1
10 is the number that fits between 5 and 14 and the index of 10 in the input array is 1."""


def gimme(input_array):
    return input_array.index(sorted(input_array)[1])
    

import codewars_test as test
from solution import gimme

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(gimme([2, 3, 1]), 0, 'Finds the index of middle element')
        test.assert_equals(gimme([5, 10, 14]), 1, 'Finds the index of middle element')





# Grasshopper - Messi goals function
# https://www.codewars.com/kata/55f73be6e12baaa5900000d4
"""Messi goals function
Messi is a soccer player with goals in three leagues:

LaLiga
Copa del Rey
Champions
Complete the function to return his total number of goals in all three leagues.

Note: the input will always be valid.

For example:

5, 10, 2  -->  17"""


goals = lambda *x: sum(x)
goals(1, 2, 3)  # 6

def goals(*x):
    return sum(x)


import codewars_test as test
from solution import goals

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(goals(0, 0, 0), 0)
        test.assert_equals(goals(5, 10, 2), 17)





# Beginner Series #4 Cockroach
# https://www.codewars.com/kata/55fab1ffda3e2e44f00000c6
"""The cockroach is one of the fastest insects. Write a function which takes its speed in km per hour and returns it in cm per second, rounded down to the integer (= floored).

For example:

1.08 --> 30"""


import numpy as np
def cockroach_speed(s):
    return np.floor(s * 100_000 / 3600)
cockroach_speed(1.08)


def cockroach_speed(s):
    return int(s * 100000 / 3600)

cockroach_speed = lambda s: s * 100000 // 3600

import codewars_test as test
from solution import cockroach_speed

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(cockroach_speed(1.08),30)
        test.assert_equals(cockroach_speed(1.09),30)
        test.assert_equals(cockroach_speed(0),0)





# Two fighters, one winner.
# https://www.codewars.com/kata/577bd8d4ae2807c64b00045b
"""Create a function that returns the name of the winner in a fight between two fighters.

Each fighter takes turns attacking the other and whoever kills the other first is victorious. Death is defined as having health <= 0.

Each fighter will be a Fighter object/instance. See the Fighter class below in your chosen language.

Both health and damagePerAttack (damage_per_attack for python) will be integers larger than 0. You can mutate the Fighter objects.

Example:
  declare_winner(Fighter("Lew", 10, 2), Fighter("Harry", 5, 4), "Lew") => "Lew"
  
  Lew attacks Harry; Harry now has 3 health.
  Harry attacks Lew; Lew now has 6 health.
  Lew attacks Harry; Harry now has 1 health.
  Harry attacks Lew; Lew now has 2 health.
  Lew attacks Harry: Harry now has -1 health and is dead. Lew wins.
"""

class Fighter(object):
    def __init__(self, name, health, damage_per_attack):
        self.name = name
        self.health = health
        self.damage_per_attack = damage_per_attack

    def __str__(self): return "Fighter({}, {}, {})".format(self.name, self.health, self.damage_per_attack)
    __repr__ = __str__


def declare_winner(fighter1, fighter2, first_attacker):
    while True:
        if fighter1.name == first_attacker:
            fighter2.health -= fighter1.damage_per_attack
            if fighter2.health <= 0:
                return fighter1.name
            fighter1.health -= fighter2.damage_per_attack
            if fighter1.health <= 0:
                return fighter2.name
            
        else:
            fighter1.health -= fighter2.damage_per_attack
            if fighter1.health <= 0:
                return fighter2.name
            fighter2.health -= fighter1.damage_per_attack
            if fighter2.health <= 0:
                return fighter1.name
            
declare_winner(Fighter("Lew", 10, 2), Fighter("Harry", 5, 4), "Lew")
declare_winner(Fighter("Lew", 10, 2), Fighter("Harry", 5, 4), "Harry")
declare_winner(Fighter("Jerry", 30, 3), Fighter("Harald", 20, 5), "Jerry")
declare_winner(Fighter("Harald", 20, 5), Fighter("Harry", 5, 4), "Harry")

import numpy as np
def declare_winner(fighter1, fighter2, first_attacker):
    f1_stamina = np.ceil(fighter1.health / fighter2.damage_per_attack)
    f2_stamina = np.ceil(fighter2.health / fighter1.damage_per_attack)
    if f1_stamina > f2_stamina:
        return fighter1.name
    elif f1_stamina < f2_stamina:
        return fighter2.name
    else:
        return first_attacker

def declare_winner(fighter1, fighter2, first_attacker):
    while fighter1.health > 0 and fighter2.health > 0:
        if fighter1.name == first_attacker:
            fighter2.health -= fighter1.damage_per_attack
            first_attacker = fighter2.name
        else:
            fighter1.health -= fighter2.damage_per_attack
            first_attacker = fighter1.name
    return fighter1.name if fighter2.health < 1 else fighter2.name


import codewars_test as test
from preloaded import Fighter
from solution import declare_winner

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(declare_winner(Fighter("Lew", 10, 2),Fighter("Harry", 5, 4), "Lew"), "Lew")
        
        test.assert_equals(declare_winner(Fighter("Lew", 10, 2),Fighter("Harry", 5, 4), "Harry"),"Harry")
        
        test.assert_equals(declare_winner(Fighter("Harald", 20, 5), Fighter("Harry", 5, 4), "Harry"),"Harald")
        
        test.assert_equals(declare_winner(Fighter("Harald", 20, 5), Fighter("Harry", 5, 4), "Harald"),"Harald")
        
        test.assert_equals(declare_winner(Fighter("Jerry", 30, 3), Fighter("Harald", 20, 5), "Jerry"), "Harald")
            
        test.assert_equals(declare_winner(Fighter("Jerry", 30, 3), Fighter("Harald", 20, 5), "Harald"),"Harald")





# Primes in numbers
# https://www.codewars.com/kata/54d512e62a5e54c96200019e/
"""Given a positive number n > 1 find the prime factor decomposition of n. The result will be a string with the following form :

 "(p1**n1)(p2**n2)...(pk**nk)"
with the p(i) in increasing order and n(i) empty if n(i) is 1.

Example: n = 86240 should return "(2**5)(5)(7**2)(11)""""


from collections import Counter

def prime_factors(n):
    dividers = []
    while True:
        for i in range(2, n + 1):
            if not n % i:
                n //= i
                dividers.append(i)
                break
        if n == 1:
            break
    return "".join(f"({k})" if v == 1 else f"({k}**{v})" for k, v in Counter(dividers).items())

prime_factors(86240), "(2**5)(5)(7**2)(11)"
prime_factors(7775460), "(2**2)(3**3)(5)(7)(11**2)(17)"
prime_factors(7919), "(7919)"






# Count the divisors of a number
# https://www.codewars.com/kata/542c0f198e077084c0000c2e


"""Count the number of divisors of a positive integer n.

Random tests go up to n = 500000.

Examples (input --> output)
4 --> 3 // we have 3 divisors - 1, 2 and 4
5 --> 2 // we have 2 divisors - 1 and 5
12 --> 6 // we have 6 divisors - 1, 2, 3, 4, 6 and 12
30 --> 8 // we have 8 divisors - 1, 2, 3, 5, 6, 10, 15 and 30
Note you should only return a number, the count of divisors. The numbers between parentheses are shown only for you to see which numbers are counted in each case."""


def divisors(n):
    return sum(True for i in range(1, n + 1) if not n % i)  # 3.687169998000172
    return len([i for i in range(1, n + 1) if not n % i])  # 3.5938232469998184
    return len([True for i in range(1, n + 1) if not n % i])  # 3.6050773840001966
    return sum([True for i in range(1, n + 1) if not n % i])  # 3.6067721780000284
    return len([not n % i for i in range(1, n + 1)])  # 4.191705105999972
    return sum(not n % i for i in range(1, n + 1))  # 5.578456058000029
    return sum(True if not n % i else False for i in range(1, n + 1))  # 5.622829734999868
    return len(list(filter(lambda i: not n % i, range(1, n + 1))))  # 6.283454425999935

divisors(1), 1
divisors(4), 3
divisors(5), 2
divisors(12), 6
divisors(30), 8
divisors(4096), 13





# Square Every Digit
# https://www.codewars.com/kata/546e2562b03326a88e000020/train/python
"""Welcome. In this kata, you are asked to square every digit of a number and concatenate them.

For example, if we run 9119 through the function, 811181 will come out, because 92 is 81 and 12 is 1. (81-1-1-81)

Example #2: An input of 765 will/should return 493625 because 72 is 49, 62 is 36, and 52 is 25. (49-36-25)

Note: The function accepts an integer and returns an integer.

Happy Coding!"""


def square_digits(num):
    return int("".join(str(int(i)**2) for i in str(num)))
square_digits(9119)





# Powers of 2
# https://www.codewars.com/kata/57a083a57cb1f31db7000028
"""Complete the function that takes a non-negative integer n as input, and returns a list of all the powers of 2 with the exponent ranging from 0 to n ( inclusive ).

Examples
n = 0  ==> [1]        # [2^0]
n = 1  ==> [1, 2]     # [2^0, 2^1]
n = 2  ==> [1, 2, 4]  # [2^0, 2^1, 2^2]"""


def powers_of_two(n):
    return [2**i for i in range(n + 1)]
    return list(map(lambda x: 2**x, range(n + 1)))
powers_of_two(4), [1, 2, 4, 8, 16]





# Thinkful - Logic Drills: Traffic light
# https://www.codewars.com/kata/58649884a1659ed6cb000072
"""You're writing code to control your town's traffic lights. You need a function to handle each change from green, to yellow, to red, and then to green again.

Complete the function that takes a string as an argument representing the current state of the light and returns a string representing the state the light should change to.

For example, when the input is green, output should be yellow."""


def update_light(current):
    lights = ("green", "yellow", "red", "green")
    return lights[lights.index(current) + 1]
update_light('green'), 'yellow'
update_light('yellow'), 'red'
update_light('red'), 'green'

def update_light(current):
    lights = ("green", "yellow", "red")
    return lights[(lights.index(current)+1) % 3]

def update_lights(current):
    next_light = {"green": "yellow", "yellow": "red", "red": "green"}
    return next_light.get(current)



# Thinkful - Logic Drills: Traffic light
# https://www.codewars.com/kata/58649884a1659ed6cb000072
"""Your function takes two arguments:

current father's age (years)
current age of his son (years)
Сalculate how many years ago the father was twice as old as his son (or in how many years he will be twice as old). The answer is always greater or equal to 0, no matter if it was in the past or it is in the future."""


def twice_as_old(dad_years_old, son_years_old):
    return abs(dad_years_old - 2 * son_years_old)
(twice_as_old(36,7) , 22)
(twice_as_old(55,30) , 5)
(twice_as_old(42,21) , 0)
(twice_as_old(22,1) , 20)
(twice_as_old(29,0) , 29)





# Disemvowel Trolls
# https://www.codewars.com/kata/52fba66badcd10859f00097e
"""Trolls are attacking your comment section!

A common way to deal with this situation is to remove all of the vowels from the trolls' comments, neutralizing the threat.

Your task is to write a function that takes a string and return a new string with all vowels removed.

For example, the string "This website is for losers LOL!" would become "Ths wbst s fr lsrs LL!".

Note: for this kata y isn't considered a vowel."""


def disemvowel(string_):
    for i in "aeoiuAEOIU":
        string_ = string_.replace(i, "")
    return string_
(disemvowel("This website is for losers LOL!"), "Ths wbst s fr lsrs LL!")
(disemvowel("No offense but,\nYour writing is among the worst I've ever read"), "N ffns bt,\nYr wrtng s mng th wrst 'v vr rd")
(disemvowel("What are you, a communist?"), "Wht r y,  cmmnst?")

def disemvowel(string_):
    return "".join(i for i in string_ if i.lower() not in "aeoiu")

import re
def disemvowel(string_):
    return re.sub(r"[aeoiu]", "", string_, re.IGNORECASE)





# Isograms
# https://www.codewars.com/kata/54ba84be607a92aa900000f1
"""An isogram is a word that has no repeating letters, consecutive or non-consecutive. Implement a function that determines whether a string that contains only letters is an isogram. Assume the empty string is an isogram. Ignore letter case.

Example: (Input --> Output)

"Dermatoglyphics" --> true "aba" --> false "moOse" --> false (ignore letter case)

isIsogram "Dermatoglyphics" = true
isIsogram "moose" = false
isIsogram "aba" = false"""


import string as string_imp

def is_isogram(string):
    letters = string_imp.ascii_lowercase
    for i in string.lower():
        if i in letters:
            letters = letters.replace(i, "")
        else:
            return False
    return True
(is_isogram("Dermatoglyphics"), True )
(is_isogram("isogram"), True )
(is_isogram("aba"), False, "same chars may not be adjacent" )
(is_isogram("moOse"), False, "same chars may not be same case" )
(is_isogram("isIsogram"), False )
(is_isogram(""), True, "an empty string is a valid isogram" )

def is_isogram(string):
    return len(string) == len(set(string.lower()))

def is_isogram(string):
    for i in string.lower():
        if string.lower().count(i) > 1:
            return False
    return True
    




# I love you, a little , a lot, passionately ... not at all
# https://www.codewars.com/kata/57f24e6a18e9fad8eb000296
"""Who remembers back to their time in the schoolyard, when girls would take a flower and tear its petals, saying each of the following phrases each time a petal was torn:

"I love you"
"a little"
"a lot"
"passionately"
"madly"
"not at all"
If there are more than 6 petals, you start over with "I love you" for 7 petals, "a little" for 8 petals and so on.

When the last petal was torn there were cries of excitement, dreams, surging thoughts and emotions.

Your goal in this kata is to determine which phrase the girls would say at the last petal for a flower of a given number of petals. The number of petals is always greater than 0."""


def how_much_i_love_you(nb_petals):
    song = ("I love you", "a little", "a lot", "passionately", "madly", "not at all")
    return song[(nb_petals - 1) % 6]
(how_much_i_love_you(7),"I love you")
(how_much_i_love_you(3),"a lot")
(how_much_i_love_you(6),"not at all")

def how_much_i_love_you(nb_petals):
    song = ("not at all", "I love you", "a little", "a lot", "passionately", "madly")
    return song[nb_petals % 6]





# Who likes it?
# https://www.codewars.com/kata/5266876b8f4bf2da9b000362
"""You probably know the "like" system from Facebook and other pages. People can "like" blog posts, pictures or other items. We want to create the text that should be displayed next to such an item.

Implement the function which takes an array containing the names of people that like an item. It must return the display text as shown in the examples:

[]                                -->  "no one likes this"
["Peter"]                         -->  "Peter likes this"
["Jacob", "Alex"]                 -->  "Jacob and Alex like this"
["Max", "John", "Mark"]           -->  "Max, John and Mark like this"
["Alex", "Jacob", "Mark", "Max"]  -->  "Alex, Jacob and 2 others like this"
Note: For 4 or more names, the number in "and 2 others" simply increases."""


def likes(names):
    if not names:
        return "no one likes this"
    elif len(names) == 1:
        return f"{names[0]} likes this"
    elif len(names) == 2:
        return f"{names[0]} and {names[1]} like this"
    elif len(names) == 3:
        return f"{names[0]}, {names[1]} and {names[2]} like this"
    else:
        return f"{names[0]}, {names[1]} and {len(names) - 2} others like this"
(likes([]), 'no one likes this')
(likes(['Peter']), 'Peter likes this')
(likes(['Jacob', 'Alex']), 'Jacob and Alex like this')
(likes(['Max', 'John', 'Mark']), 'Max, John and Mark like this')
(likes(['Alex', 'Jacob', 'Mark', 'Max']), 'Alex, Jacob and 2 others like this')





# Count the Digit
# https://www.codewars.com/kata/566fc12495810954b1000030/train/python
"""Take an integer n (n >= 0) and a digit d (0 <= d <= 9) as an integer.

Square all numbers k (0 <= k <= n) between 0 and n.

Count the numbers of digits d used in the writing of all the k**2.

Implement the function taking n and d as parameters and returning this count.

Examples:
n = 10, d = 1 
the k*k are 0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100
We are using the digit 1 in: 1, 16, 81, 100. The total count is then 4.

The function, when given n = 25 and d = 1 as argument, should return 11 since
the k*k that contain the digit 1 are:
1, 16, 81, 100, 121, 144, 169, 196, 361, 441.
So there are 11 digits 1 for the squares of numbers between 0 and 25.
Note that 121 has twice the digit 1."""


def nb_dig(n, d):
    return "".join(str(i**2) for i in range(n + 1)).count(str(d))
    return "".join(map(lambda x: str(x**2), range(n + 1))).count(str(d))
    return sum(str(i**2).count(str(d)) for i in range(n + 1))
    return sum(map(lambda x: str(x**2).count(str(d)), range(n + 1)))
(nb_dig(5750, 0), 4700)
(nb_dig(11011, 2), 9481)
(nb_dig(12224, 8), 7733)
(nb_dig(11549, 1), 11905)
(nb_dig(14550, 7), 8014)
(nb_dig(8304, 7), 3927)
(nb_dig(10576, 9), 7860)
(nb_dig(12526, 1), 13558)
(nb_dig(7856, 4), 7132)
(nb_dig(14956, 1), 17267)

n = 14656
d = 1
timeit.timeit(lambda: "".join(str(i**2) for i in range(n + 1)).count(str(d)), number=1_000)  # 8.781008230999987
timeit.timeit(lambda: "".join(map(lambda x: str(x**2), range(n + 1))).count(str(d)), number=1_000)  # 10.25051729400002
timeit.timeit(lambda: sum(str(i**2).count(str(d)) for i in range(n + 1)), number=1_000)  # 12.9422026430002
timeit.timeit(lambda: sum(map(lambda x: str(x**2).count(str(d)), range(n + 1))), number=1_000)  # 12.971102688000201





# Sum The Strings
# https://www.codewars.com/kata/5966e33c4e686b508700002d/train/python
"""Create a function that takes 2 integers in form of a string as an input, and outputs the sum (also as a string):

Example: (Input1, Input2 -->Output)

"4",  "5" --> "9"
"34", "5" --> "39"
"", "" --> "0"
"2", "" --> "2"
"-5", "3" --> "-2"
Notes:

If either input is an empty string, consider it as zero.

Inputs and the expected output will never exceed the signed 32-bit integer limit (2^31 - 1)"""


def sum_str(a, b):
    if len(a + b) == 0:
        return "0"
    elif not (a and b):
        return str(int(a + b))
    else:
        return str(int(a) + int(b))
(sum_str("4","5"), "9")
(sum_str("34","5"), "39")
(sum_str("9",""), "9", "x + empty = x")
(sum_str("","9"), "9", "empty + x = x")
(sum_str("","") , "0", "empty + empty = 0")

def sum_str(a, b):
    return str(int(a or 0) + int(b or 0))





# If you can't sleep, just count sheep!!
# https://www.codewars.com/kata/5b077ebdaf15be5c7f000077/train/python
"""If you can't sleep, just count sheep!!

Task:
Given a non-negative integer, 3 for example, return a string with a murmur: "1 sheep...2 sheep...3 sheep...". Input will always be valid, i.e. no negative integers."""


def count_sheep(n):
    return "".join(f"{i+1} sheep..." for i in range(n))
(count_sheep(0), "");
(count_sheep(1), "1 sheep...");
(count_sheep(2), "1 sheep...2 sheep...")
(count_sheep(3), "1 sheep...2 sheep...3 sheep...")





# Anagram Detection
# https://www.codewars.com/kata/529eef7a9194e0cbc1000255/train/python
"""An anagram is the result of rearranging the letters of a word to produce a new word (see wikipedia).

Note: anagrams are case insensitive

Complete the function to return true if the two arguments given are anagrams of each other; return false otherwise.

Examples
"foefet" is an anagram of "toffee"

"Buckethead" is an anagram of "DeathCubeK""""

def is_anagram(test, original):
    return sorted(test.lower()) == sorted(original.lower())
(is_anagram("foefet", "toffee"), True, 'The word foefet is an anagram of toffee')
(is_anagram("Buckethead", "DeathCubeK"), True, 'The word Buckethead is an anagram of DeathCubeK')
(is_anagram("Twoo", "WooT"), True, 'The word Twoo is an anagram of WooT')
(is_anagram("dumble", "bumble"), False, 'Characters do not match for test case dumble, bumble')
(is_anagram("ound", "round"), False, 'Missing characters for test case ound, round')
(is_anagram("apple", "pale"), False, 'Same letters, but different count')





# Testing 1-2-3
# https://www.codewars.com/kata/54bf85e3d5b56c7a05000cf9/train/python
"""Your team is writing a fancy new text editor and you've been tasked with implementing the line numbering.

Write a function which takes a list of strings and returns each line prepended by the correct number.

The numbering starts at 1. The format is n: string. Notice the colon and space in between.

Examples: (Input --> Output)

[] --> []
["a", "b", "c"] --> ["1: a", "2: b", "3: c"]"""


def number(lines):
    return [f"{i}: {line}" for i, line in enumerate(lines, start=1)]
    return ["{}: {}".format(i, line) for i, line in enumerate(lines, start=1)]
    return ["{}: {}".format(*line) for line in enumerate(lines, start=1)]
    return ["%d: %s" % (i, line) for i, line in enumerate(lines, start=1)]
    return [ str(i) + ": " + line for i, line in enumerate(lines, start=1)]
(number([]), [])
(number(["a", "b", "c"]), ["1: a", "2: b", "3: c"])





# Delete occurrences of an element if it occurs more than n times
# https://www.codewars.com/kata/554ca54ffa7d91b236000023/train/python
"""Enough is enough!
Alice and Bob were on a holiday. Both of them took many pictures of the places they've been, and now they want to show Charlie their entire collection. However, Charlie doesn't like these sessions, since the motif usually repeats. He isn't fond of seeing the Eiffel tower 40 times.
He tells them that he will only sit for the session if they show the same motif at most N times. Luckily, Alice and Bob are able to encode the motif as a number. Can you help them to remove numbers such that their list contains each number only up to N times, without changing the order?

Task
Given a list and a number, create a new list that contains each number of list at most N times, without reordering.
For example if the input number is 2, and the input list is [1,2,3,1,2,1,2,3], you take [1,2,3,1,2], drop the next [1,2] since this would lead to 1 and 2 being in the result 3 times, and then take 3, which leads to [1,2,3,1,2,3].
With list [20,37,20,21] and number 1, the result would be [20,37,21]."""


from collections import Counter

def delete_nth(order,max_e):
    to_remove = {k: v-max_e for k, v in Counter(order).items() if v > max_e}
    order = list(reversed(order))
    
    for k, v in to_remove.items():
        for _ in range(v):
            order.remove(k)
            
    order = list(reversed(order))
    return order
(delete_nth([20, 37, 20, 21], 1), [20, 37, 21])

def delete_nth(order,max_e):
    sol = []
    for elem in order:
        if sol.count(elem) < max_e:
            sol.append(elem)
    return sol    





# Sort and Star
# https://www.codewars.com/kata/57cfdf34902f6ba3d300001e/train/python
"""You will be given a list of strings. You must sort it alphabetically (case-sensitive, and based on the ASCII values of the chars) and then return the first value.

The returned value must be a string, and have "***" between each of its letters.

You should not remove or add elements from/to the array."""


def two_sort(array):
    return "***".join(elem for elem in sorted(array)[0])
(two_sort(["bitcoin", "take", "over", "the", "world", "maybe", "who", "knows", "perhaps"]), 'b***i***t***c***o***i***n' )
(two_sort(["turns", "out", "random", "test", "cases", "are", "easier", "than", "writing", "out", "basic", "ones"]), 'a***r***e')
(two_sort(["lets", "talk", "about", "javascript", "the", "best", "language"]), 'a***b***o***u***t')
(two_sort(["i", "want", "to", "travel", "the", "world", "writing", "code", "one", "day"]), 'c***o***d***e')
(two_sort(["Lets", "all", "go", "on", "holiday", "somewhere", "very", "cold"]), 'L***e***t***s')

def two_sort(array):
    return "***".join(min(array))




# Find the capitals
# https://www.codewars.com/kata/539ee3b6757843632d00026b/train/python
"""Instructions
Write a function that takes a single non-empty string of only lowercase and uppercase ascii letters (word) as its argument, and returns an ordered list containing the indices of all capital (uppercase) letters in the string.

Example (Input --> Output)
"CodEWaRs" --> [0,3,4,6]"""


def capitals(word):
    return [i for i, letter in enumerate(word) if letter.isupper()]
(capitals('CodEWaRs'), [0,3,4,6])





# List Filtering
# https://www.codewars.com/kata/53dbd5315a3c69eed20002dd/train/python
"""In this kata you will create a function that takes a list of non-negative integers and strings and returns a new list with the strings filtered out.

Example
filter_list([1,2,'a','b']) == [1,2]
filter_list([1,'a','b',0,15]) == [1,0,15]
filter_list([1,2,'aasf','1','123',123]) == [1,2,123]"""


def filter_list(l):
    # return list(filter(lambda x: isinstance(x, int) , l))
    return list(filter(lambda x: type(x) == int , l))
(filter_list([1, 2, 'a', 'b']), [1, 2], 'For input [1, 2, "a", "b"]')
(filter_list([1, 'a', 'b', 0, 15]), [1, 0, 15], 'Fot input [1, "a", "b", 0, 15]')
(filter_list([1, 2, 'aasf', '1', '123', 123]), [1, 2, 123], 'For input [1, 2, "aasf", "1", "123", 123]')





# Give me a Diamond
# https://www.codewars.com/kata/5503013e34137eeeaa001648/train/python
"""Jamie is a programmer, and James' girlfriend. She likes diamonds, and wants a diamond string from James. Since James doesn't know how to make this happen, he needs your help.

Task
You need to return a string that looks like a diamond shape when printed on the screen, using asterisk (*) characters. Trailing spaces should be removed, and every line must be terminated with a newline character (\n).

Return null/nil/None/... if the input is an even number or negative, as it is not possible to print a diamond of even or negative size.

Examples
A size 3 diamond:

 *
***
 *
...which would appear as a string of " *\n***\n *\n"

A size 5 diamond:

  *
 ***
*****
 ***
  *
...that is:

"  *\n ***\n*****\n ***\n  *\n""""


def diamond(n):
    if n <= 0:
        return None
    elif (n - 1) % 2:
        return None
    else:
        rows = n // 2
        upper = [" " * (rows - i) + "*" * (2*i + 1) for i in range(rows + 1)]
        lower = [" " * (rows - i) + "*" * (2*i + 1) for i in reversed(range(rows))]
        return "\n".join(upper + lower) + "\n"
(diamond(1), "*\n")
(diamond(2), None)
(diamond(3), " *\n***\n *\n")
(diamond(5), "  *\n ***\n*****\n ***\n  *\n")
(diamond(0), None)
(diamond(-3), None)





# Round up to the next multiple of 5
# https://www.codewars.com/kata/55d1d6d5955ec6365400006d/train/python
"""Given an integer as input, can you round it to the next (meaning, "greater than or equal") multiple of 5?

Examples:

input:    output:
0    ->   0
2    ->   5
3    ->   5
12   ->   15
21   ->   25
30   ->   30
-2   ->   0
-5   ->   -5
etc.
Input may be any positive or negative integer (including 0).

You can assume that all inputs are valid integers."""


def round_to_next5(n):
    return n if not n % 5 else (n // 5) * 5 + 5
(round_to_next5(0), 0)
(round_to_next5(2), 5)
(round_to_next5(3), 5)
(round_to_next5(12), 15)
(round_to_next5(21), 25)
(round_to_next5(30), 30)
(round_to_next5(-2), 0)
(round_to_next5(-5), -5)

import numpy as np
def round_to_next5(n):
    return int(np.ceil(n/5) * 5)

def round_to_next5(n):
    return n + (5 - n) % 5





# Quarter of the year
# https://www.codewars.com/kata/5ce9c1000bab0b001134f5af/train/python
"""Given a month as an integer from 1 to 12, return to which quarter of the year it belongs as an integer number.

For example: month 2 (February), is part of the first quarter; month 6 (June), is part of the second quarter; and month 11 (November), is part of the fourth quarter.

Constraint:

1 <= month <= 12"""


def quarter_of(month):
    return month // 3 if not month % 3 else (month//3) + 1
(quarter_of(3), 1)
(quarter_of(8), 3)
(quarter_of(11), 4)

def quarter_of(month):
    return (month + 2) // 3





# Sum of two lowest positive integers
# https://www.codewars.com/kata/558fc85d8fd1938afb000014/train/python
"""Create a function that returns the sum of the two lowest positive numbers given an array of minimum 4 positive integers. No floats or non-positive integers will be passed.

For example, when an array is passed like [19, 5, 42, 2, 77], the output should be 7.

[10, 343445353, 3453445, 3453545353453] should return 3453455."""


def sum_two_smallest_numbers(numbers):
    min1 = min(numbers)
    numbers.remove(min1)
    return min1 + min(numbers)
(sum_two_smallest_numbers([5, 8, 12, 18, 22]), 13)
(sum_two_smallest_numbers([7, 15, 12, 18, 22]), 19)
(sum_two_smallest_numbers([25, 42, 12, 18, 22]), 30)





# Simple Encryption #1 - Alternating Split
# https://www.codewars.com/kata/57814d79a56c88e3e0000786/solutions
"""Implement a pseudo-encryption algorithm which given a string S and an integer N concatenates all the odd-indexed characters of S with all the even-indexed characters of S, this process should be repeated N times.

Examples:

encrypt("012345", 1)  =>  "135024"
encrypt("012345", 2)  =>  "135024"  ->  "304152"
encrypt("012345", 3)  =>  "135024"  ->  "304152"  ->  "012345"

encrypt("01234", 1)  =>  "13024"
encrypt("01234", 2)  =>  "13024"  ->  "32104"
encrypt("01234", 3)  =>  "13024"  ->  "32104"  ->  "20314"
Together with the encryption function, you should also implement a decryption function which reverses the process.

If the string S is an empty value or the integer N is not positive, return the first argument without changes."""


def decrypt(text, n):
    if text == "":
        return ""
    if text == None:
        return None
    
    if n <= 0:
        return text
    else:
        t_odd = text[:len(text)//2]
        t_even = text[len(text)//2:]

        t = ""
        for i in range(len(t_even)):
        #     t += t_even[i]
        #     try:
        #         t += t_odd[i]
        #     except:
        #         t += ""
            t += t_even[i:i+1]
            t += t_odd[i:i+1]

        return decrypt(t, n-1)

    
(decrypt("This is a test!", 0), "This is a test!")
(decrypt("hsi  etTi sats!", 1), "This is a test!")
(decrypt("s eT ashi tist!", 2), "This is a test!")
(decrypt(" Tah itse sits!", 3), "This is a test!")
(decrypt("This is a test!", 4), "This is a test!")
(decrypt("This is a test!", -1), "This is a test!")
(decrypt("hskt svr neetn!Ti aai eyitrsig", 1), "This kata is very interesting!")    


def encrypt(text, n):
    if n <= 0:
        return text
    else:
        return encrypt("".join(elem for i, elem in enumerate(text) if i % 2) + "".join(elem for i, elem in enumerate(text) if not i % 2), n - 1)

(encrypt("This is a test!", 0), "This is a test!")
(encrypt("This is a test!", 1), "hsi  etTi sats!")
(encrypt("This is a test!", 2), "s eT ashi tist!")
(encrypt("This is a test!", 3), " Tah itse sits!")
(encrypt("This is a test!", 4), "This is a test!")
(encrypt("This is a test!", -1), "This is a test!")
(encrypt("This kata is very interesting!", 1), "hskt svr neetn!Ti aai eyitrsig")

def decrypt(text, n):
    if text == "":
        return ""
    if text == None:
        return None
    if n <= 0:
        return text    
    else:        
        o, l = len(text) // 2, list(text)
        l[1::2], l[::2] = l[:o], l[o:]
        return decrypt("".join(l), n-1)

def encrypt(text, n):
    for _ in range(n):
        text = text[1::2] + text[::2]
    return text

def encrypt(text, n):
    if n <= 0:
        return text
    return encrypt(text[1::2] + text[::2], n-1)





# Complementary DNA
# https://www.codewars.com/kata/554e4a2f232cdd87d9000038/train/python
"""Deoxyribonucleic acid (DNA) is a chemical found in the nucleus of cells and carries the "instructions" for the development and functioning of living organisms.

If you want to know more: http://en.wikipedia.org/wiki/DNA

In DNA strings, symbols "A" and "T" are complements of each other, as "C" and "G". Your function receives one side of the DNA (string, except for Haskell); you need to return the other complementary side. DNA strand is never empty or there is no DNA at all (again, except for Haskell).

More similar exercise are found here: http://rosalind.info/problems/list-view/ (source)

Example: (input --> output)

"ATTGC" --> "TAACG"
"GTAT" --> "CATA""""


def DNA_strand(dna):
    return dna.replace("A", "Ą").replace("T", "A").replace("Ą", "T").replace("C", "Ą").replace("G", "C").replace("Ą", "G")
(DNA_strand("AAAA"),"TTTT")
(DNA_strand("ATTGC"),"TAACG")
(DNA_strand("GTAT"),"CATA")

def DNA_strand(dna):
    return dna.translate(str.maketrans("ATCG","TAGC"))

def DNA_strand(dna):
    reference = { "A":"T",
                  "T":"A",
                  "C":"G",
                  "G":"C"
                  }
    return "".join(reference[i] for i in dna)





# Super Duper Easy
# https://www.codewars.com/kata/55a5bfaa756cfede78000026/solutions
"""Make a function that returns the value multiplied by 50 and increased by 6. If the value entered is a string it should return "Error"."""


def problem(a):
    return 50 * a + 6 if type(a) in [float, int] else "Error"
(problem("hello"), "Error")
(problem(1), 56)

def problem(a):
    return "Error" if type(a) == str else 50 * a + 6





# 














 
    
    
    
    


















# Sum of Pairs
# https://www.codewars.com/kata/54d81488b981293527000c8f
"""
Given a list of integers and a single sum value, return the first two values (parse from the left please) in order of appearance that add up to form the sum.

sum_pairs([11, 3, 7, 5],         10)
#              ^--^      3 + 7 = 10
== [3, 7]

sum_pairs([4, 3, 2, 3, 4],         6)
#          ^-----^         4 + 2 = 6, indices: 0, 2 *
#             ^-----^      3 + 3 = 6, indices: 1, 3
#                ^-----^   2 + 4 = 6, indices: 2, 4
#  * entire pair is earlier, and therefore is the correct answer
== [4, 2]

sum_pairs([0, 0, -2, 3], 2)
#  there are no pairs of values that can be added to produce 2.
== None/nil/undefined (Based on the language)

sum_pairs([10, 5, 2, 3, 7, 5],         10)
#              ^-----------^   5 + 5 = 10, indices: 1, 5
#                    ^--^      3 + 7 = 10, indices: 3, 4 *
#  * entire pair is earlier, and therefore is the correct answer
== [3, 7]
Negative numbers and duplicate numbers can and will appear.

NOTE: There will also be lists tested of lengths upwards of 10,000,000 elements. Be sure your code doesn't time out."""


# still to slow
def sum_pairs(ints, s):
    for i in range(1, len(ints)):
        for j in range(i):
            if ints[i] + ints[j] == s:
                return [ints[j], ints[i]]

# too slow, uses dict to caputre all solutions
def sum_pairs(ints, s):
    sol = dict()
    for i in range(len(ints)):
        for j in range(i + 1, len(ints)):
            if ints[i] + ints[j] == s:
                sol[j] = [ints[i], ints[j]]
                # return [ints[i], ints[j]]
    return sol[min(sol)] if sol else None

# caputures only ones form the beginning
def sum_pairs(ints, s):
    for i, val in enumerate(ints):
        try:
            ints[:i].index(s - val)
            return [s - val, val]
            # return i
        except:
            pass
sum_pairs([10, 5, 2, 3, 7, 5], 10)
sum_pairs([1, 4, 8, 7, 3, 15], 8)
sum_pairs([1, -2, 3, 0, -6, 1], -6)

# caputures only ones form the beginning
def sum_pairs(ints, s):
    for val in ints:
        try:
            ints.index(s - val)
            return [val, s - val]
        except:
            pass





l1 = [1, 4, 8, 7, 3, 15]
l2 = [1, -2, 3, 0, -6, 1]
l3 = [20, -13, 40]
l4 = [1, 2, 3, 4, 1, 0]
l5 = [10, 5, 2, 3, 7, 5]
l6 = [4, -2, 3, 3, 4]
l7 = [0, 2, 0]
l8 = [5, 9, 13, -3]

test.describe("Testing For Sum of Pairs")
test.expect(sum_pairs(l1, 8) == [1, 7], "Basic: %s should return [1, 7] for sum = 8" % l1)
test.expect(sum_pairs(l2, -6) == [0, -6], "Negatives: %s should return [0, -6] for sum = -6" % l2)
test.expect(sum_pairs(l3, -7) == None, "No Match: %s should return None for sum = -7" % l3)
test.expect(sum_pairs(l4, 2) == [1, 1], "First Match From Left: %s should return [1, 1] for sum = 2 " % l4)
test.expect(sum_pairs(l5, 10) == [3, 7], "First Match From Left REDUX!: %s should return [3, 7] for sum = 10 " % l5)
test.expect(sum_pairs(l6, 8) == [4, 4], "Duplicates: %s should return [4, 4] for sum = 8" % l6)
test.expect(sum_pairs(l7, 0) == [0, 0], "Zeroes: %s should return [0, 0] for sum = 0" % l7)
test.expect(sum_pairs(l8, 10) == [13, -3], "Subtraction: %s should return [13, -3] for sum = 10" % l8)