# codewars kata


"""Write a function that takes an integer as input, and returns the number of bits that are equal
to one in the binary representation of that number. You can guarantee that input is non-negative.
Example: The binary representation of 1234 is 10011010010, so the function should return 5 in this case"""


def count_bits(n):
    # return bin(n)[2:].count('1')
    return bin(n).count('1')

count_bits(1234)





"""Digital root is the recursive sum of all the digits in a number.
Given n, take the sum of the digits of n. If that value has more than one digit, 
continue reducing in this way until a single-digit number is produced. The input will be a non-negative integer.
"""

def digital_root(n):
    while True:
        if n > 9:
            n = sum([int(digit) for digit in str(n)])
        else:
            return n


import numpy as np
def digital_root(n):
    while len(str(n)) != 1:
        n = np.sum([int(i) for i in str(n)])
    return n


import numpy as np
def digital_root(n):
    while True:
        if len(str(n)) != 1:
            n = np.sum([int(i) for i in str(n)])
        else:
            return n

test.assert_equals(digital_root(16), 7)
test.assert_equals(digital_root(942), 6)
test.assert_equals(digital_root(132189), 6)
test.assert_equals(digital_root(493193), 2)





"""A pangram is a sentence that contains every single letter of the alphabet at least once.
For example, the sentence "The quick brown fox jumps over the lazy dog" is a pangram, because it uses the letters A-Z at least once (case is irrelevant).
Given a string, detect whether or not it is a pangram. Return True if it is, False if not. Ignore numbers and punctuation."""

import string

def is_pangram(s):
    return all([True if letter in s.lower() else False for letter in string.ascii_lowercase])


def is_pangram(s):
    return set(string.ascii_lowercase) <= set(s.lower())


def is_pangram(s):
    return set(string.ascii_lowercase).issubset(set(s.lower()))


def is_pangram(s):
    return (set(string.ascii_lowercase) & set(s.lower())) == set(string.ascii_lowercase)


pangram = "The quick, brown fox jumps over the lazy dog!"
is_pangram(pangram)
test.assert_equals(is_pangram(pangram), True)





"""If we list all the natural numbers below 10 that are multiples of 3 or 5, we get 3, 5, 6 and 9.
The sum of these multiples is 23.
Finish the solution so that it returns the sum of all the multiples of 3 or 5 below the number passed in.
Additionally, if the number is negative, return 0 (for languages that do have them).
Note: If the number is a multiple of both 3 and 5, only count it once."""

import numpy as np

def solution(numbers):
    if numbers < 0:
        return 0
    else:
        return np.sum([number for number in range(numbers) if number%3 == 0 or number%5 == 0])


def solution(numbers):
    return np.sum([number for number in range(numbers) if number%3 == 0 or number%5 == 0])


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
    if len(s) > 140 or s == '':
        formatedstring = False
    else:
        formatedstring = '#'
        for string in s.split():
            formatedstring += string.strip().capitalize()
    return formatedstring


def generate_hashtag(s):
    if len(s) > 140 or s == '':
        return False
    else:
        return '#'+''.join([string.strip().capitalize() for string in s.split()])


if not s

generate_hashtag('CodeWars  is   nice')
generate_hashtag('')

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
    if prod == 0: return [0, 1, True]
    Fib = [0, 1]
    i = 2
    while True:
        Fib.append(Fib[i-1] + Fib[i-2])
        if Fib[i] * Fib[i-1] >= prod:
            break
        i += 1
    return [Fib[i-1], Fib[i], Fib[i] * Fib[i-1] == prod]


def productFib(prod):
    a, b = 0, 1
    while prod > a * b:
        a, b = b, a + b
    return [a, b, a * b == prod]


productFib(4895)


test.assert_equals(productFib(4895), [55, 89, True])
test.assert_equals(productFib(5895), [89, 144, False])





"""Write Number in Expanded Form
You will be given a number and you will need to return it as a string in Expanded Form. For example:

expanded_form(12) # Should return '10 + 2'
expanded_form(42) # Should return '40 + 2'
expanded_form(70304) # Should return '70000 + 300 + 4'
NOTE: All numbers will be whole numbers greater than 0."""


def expanded_form(num):
    number_list = []
    for i, number in enumerate([int(number) for number in str(num)]):
        if number != 0:
            number_list.append(str((number) * 10**(len(str(num)) - i - 1)))
    return ' + '.join(number_list)


def expanded_form(num):
    number_list = []
    for i, number in enumerate(str(num)):
        if number != '0':
            number_list.append(str(int(number) * 10**(len(str(num)) - i - 1)))
    return ' + '.join(number_list)


def expanded_form(num):
    return ' + '.join([str(int(number) * 10**(len(str(num)) - i - 1)) for i, number in enumerate(str(num)) if number != '0'])


test.assert_equals(expanded_form(12), '10 + 2');
test.assert_equals(expanded_form(42), '40 + 2');
test.assert_equals(expanded_form(70304), '70000 + 300 + 4');





"""The rgb function is incomplete. Complete it so that passing in RGB decimal values will result in a hexadecimal representation being returned. Valid decimal values for RGB are 0 - 255. Any values that fall out of that range must be rounded to the closest valid value.

Note: Your answer should always be 6 characters long, the shorthand with 3 will not work here.

The following are examples of expected output values:

rgb(255, 255, 255) # returns FFFFFF
rgb(255, 255, 300) # returns FFFFFF
rgb(0,0,0) # returns 000000
rgb(148, 0, 211) # returns 9400D3"""


def rgb(r, g, b):
    if r <= 0: r = 0
    if g <= 0: g = 0
    if b <= 0: b = 0
    if r > 255: r = 255
    if g > 255: g = 255
    if b > 255: b = 255
    return '{:02X}{:02X}{:02X}'.format(r, g, b)
rgb(254,253,252)


def rgb(r, g, b):
    my_round = lambda x: min(255, max(x, 0))
    return '{:02X}{:02X}{:02X}'.format(my_round(r), my_round(g), my_round(b))


def rgb(r, g, b):
    my_round = lambda x: min(255, max(x, 0))
    # return '{:02X}{:02X}{:02X}'.format(my_round(r), my_round(g), my_round(b))
    return ('{:02X}' * 3).format(my_round(r), my_round(g), my_round(b))


test.assert_equals(rgb(0,0,0),"000000", "testing zero values")
test.assert_equals(rgb(1,2,3),"010203", "testing near zero values")
test.assert_equals(rgb(255,255,255), "FFFFFF", "testing max values")
test.assert_equals(rgb(254,253,252), "FEFDFC", "testing near max values")
test.assert_equals(rgb(-20,275,125), "00FF7D", "testing out of range values")





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


import itertools

def around(char):
    if char == '1': 
        return ['1', '2', '4']
    elif char == '2': 
        return ['2', '1', '3', '5']
    elif char == '3': 
        return ['2', '6', '3']
    elif char == '4': 
        return ['1', '4', '7', '5']
    elif char == '5': 
        return ['5', '2', '4', '6', '8']
    elif char == '6': 
        return ['6', '3', '5', '9']
    elif char == '7': 
        return ['7', '4', '8']
    elif char == '8': 
        return ['8', '5', '7', '9', '0']
    elif char == '9': 
        return ['9', '6', '8']
    elif char == '0': 
        return ['0', '8']
    else:
        return None

def get_pins(observed):
    return [''.join(i) for i in list(itertools.product(*[around(char) for char in observed]))]


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

def get_pins(observed):
    return [''.join(i) for i in list(itertools.product(*[around[char] for char in observed]))]


get_pins('369').sort() == ["339","366","399","658","636","258","268","669","668","266","369","398","256","296","259","368","638","396","238","356","659","639","666","359","336","299","338","696","269","358","656","698","699","298","236","239"].sort()

list(itertools.product(*[['1', '2', '4'], ['8', '5', '7', '9', '0'], ['0', '1']]))
list(itertools.product('124', '85790', '01'))

test.describe('example tests')
expectations = [('8', ['5','7','8','9','0']),
                ('11',["11", "22", "44", "12", "21", "14", "41", "24", "42"]),
                ('369', ["339","366","399","658","636","258","268","669","668","266","369","398","256","296","259","368","638","396","238","356","659","639","666","359","336","299","338","696","269","358","656","698","699","298","236","239"])]

for tup in expectations:
    test.assert_equals(sorted(get_pins(tup[0])), sorted(tup[1]), 'PIN: ' + tup[0])





"""The maximum sum subarray problem consists in finding the maximum sum of a contiguous subsequence in an array or list of integers:

max_sequence([-2, 1, -3, 4, -1, 2, 1, -5, 4])
# should be 6: [4, -1, 2, 1]
Easy case is when the list is made up of only positive numbers and the maximum sum is the sum of the whole array. 
If the list is made up of only negative numbers, return 0 instead.

Empty list is considered to have zero greatest sum. Note that the empty list or array is also a valid sublist/subarray."""


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

import numpy as np
def max_sequence(arr):
    return max([sum(arr[i:j]) for i in range(len(arr) + 0) for j in range(1, len(arr) + 1) if i <= j])

def max_sequence(arr):
    return max([sum(arr[i:j]) for i in range(len(arr)+1) for j in range(len(arr)+1)])


# show arrays
def max_sequence(arr):
    array = []
    for i in range(len(arr)):
        for j in range(1, len(arr) + 1):
            if i < j:
                array.append(arr[i:j])
    return array


max_sequence([-2, 1, -3, 4, -1, 2, 1, -5, 4])
max_sequence([-2])
max_sequence([])


test.describe("Tests")
test.it('should work on an empty array')   
test.assert_equals(max_sequence([]), 0)
test.it('should work on the example')
test.assert_equals(max_sequence([-2, 1, -3, 4, -1, 2, 1, -5, 4]), 6)





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


import numpy as np

def order_weight(strng):
    if not strng:
        return ''
    splitted = [char.strip() for char in strng.split()]
    weight_list = [(sum(int(digit) for digit in number), number) for number in splitted]
    return ' '.join(np.array(sorted(weight_list))[:, 1])


def order_weight(strng):
    return ' '.join(sorted(strng.split(), key=lambda x: (sum(int(d) for d in x), x)))

order_weight("2000 10003 1234000 44444444 9999 11 11 22 123")



test.assert_equals(order_weight("103 123 4444 99 2000"), "2000 103 123 4444 99")
test.assert_equals(order_weight("2000 10003 1234000 44444444 9999 11 11 22 123"), "11 11 2000 10003 22 123 1234000 44444444 9999")
test.assert_equals(order_weight(""), "")


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


from collections import Counter

def mix(s1, s2):
    s1_dict = Counter(char for char in s1 if char.islower())
    s2_dict = Counter(char for char in s2 if char.islower())
    s3_dict = s1_dict | s2_dict
    result = []
    for char, val in sorted(s3_dict.items()):
        if val > 1:
            if s1_dict[char] > s2_dict[char]:
                result.append('1:' + char*val)
            elif s1_dict[char] < s2_dict[char]: 
                result.append('2:' + char*val)
            elif (s1_dict[char] == s2_dict[char]):
                result.append('=:' + char*val)

    # return '/'.join(result)
    return result

def mix(s1, s2):
    s1_dict = dict(Counter(char for char in s1 if char.islower()))
    s1_dict = {key: val for key, val in s1_dict.items() if val > 1}
    s1_dict = dict(sorted(s1_dict.items(), key=lambda x: (x[0], x[1])))
    s2_dict = dict(Counter(char for char in s2 if char.islower()))
    s2_dict = {key: val for key, val in s2_dict.items() if val > 1}
    s2_dict = dict(sorted(s2_dict.items(), key=lambda x: (x[0], x[1])))
    s3_dict = dict(s1_dict.items()|s2_dict.items())

    result = []
    for char, val in s3_dict.items():
        if s1_dict[char] > s2_dict[char]:
            result.append('1:' + char*val)
        elif s1_dict[char] < s2_dict[char]: 
            result.append('2:' + char*val)
        elif (s1_dict[char] == s2_dict[char]):
            result.append('=:' + char*val)

    return s3_dict[e]

mix("Are they here", "yes, they are here")
mix(s1, s2) --> "2:eeeee/2:yy/=:hh/=:rr"
mix("my&friend&Paul has heavy hats! &", "my friend John has many many friends &")
mix(s1, s2) --> "2:nnnnn/1:aaaa/1:hhh/2:mmm/2:yyy/2:dd/2:ff/2:ii/2:rr/=:ee/=:ss"

for char, val in {'e': 5, 'r': 2, 'y': 2, 'h': 2}.items():
    print(char, val)


test.describe("Mix")
test.it("Basic Tests")
test.assert_equals(mix("Are they here", "yes, they are here"), "2:eeeee/2:yy/=:hh/=:rr")
test.assert_equals(mix("Sadus:cpms>orqn3zecwGvnznSgacs","MynwdKizfd$lvse+gnbaGydxyXzayp"), '2:yyyy/1:ccc/1:nnn/1:sss/2:ddd/=:aa/=:zz')
test.assert_equals(mix("looping is fun but dangerous", "less dangerous than coding"), "1:ooo/1:uuu/2:sss/=:nnn/1:ii/2:aa/2:dd/2:ee/=:gg")
test.assert_equals(mix(" In many languages", " there's a pair of functions"), "1:aaa/1:nnn/1:gg/2:ee/2:ff/2:ii/2:oo/2:rr/2:ss/2:tt")
test.assert_equals(mix("Lords of the Fallen", "gamekult"), "1:ee/1:ll/1:oo")
test.assert_equals(mix("codewars", "codewars"), "")
test.assert_equals(mix("A generation must confront the looming ", "codewarrs"), "1:nnnnn/1:ooooo/1:tttt/1:eee/1:gg/1:ii/1:mm/=:rr")





"""Complete the solution so that it strips all text that follows any of a set of comment markers passed in. Any whitespace at the end of the line should also be stripped out.

Example:

Given an input string of:

apples, pears # and bananas
grapes
bananas !apples
The output expected would be:

apples, pears
grapes
bananas
The code would be called like so:

result = solution("apples, pears # and bananas\ngrapes\nbananas !apples", ["#", "!"])
# result should == "apples, pears\ngrapes\nbananas""""



def solution(string, markers):
    for char in markers:
        string = string.replace(char, '')
    return string

solution("apples, pears # and bananas\ngrapes\nbananas !apples", ["#", "!"])


# -*- coding: utf-8 -*-
test.assert_equals(solution("apples, pears # and bananas\ngrapes\nbananas !apples", ["#", "!"]), "apples, pears\ngrapes\nbananas")
test.assert_equals(solution("a #b\nc\nd $e f g", ["#", "$"]), "a\nc\nd")