# codewars kata


"""Write a function that takes an integer as input, and returns the number of bits that are equal
to one in the binary representation of that number. You can guarantee that input is non-negative.
Example: The binary representation of 1234 is 10011010010, so the function should return 5 in this case"""


from numpy.random import randint
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

from audioop import reverse
import numpy as np
import soupsieve

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






# partially solved
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
# result should == "apples, pears\ngrapes\nbananas"
"""


def solution(string, markers):
    splitted_list = string.split('\n')
    for char in markers:
        # splitted_list = [elem if elem.find(char) == -1 else elem[:elem.find(char)].rstrip() for elem in splitted_list]
        splitted_list = [elem.split(char)[0].rstrip() for elem in splitted_list]
    return '\n'.join(splitted_list)


def solution(string, markers):
    for marker in markers:
        # string = re.sub(r' *?\{}.*$'.format(marker), '', string, flags=re.M)
        string = re.sub(r' *?{}.*'.format(marker), '', string, flags=re.M)
    return string

solution('apples, pears # and bananas\ngrapes\nbananas !apples', ['#', '!'])
solution("' = lemons\napples pears\n. avocados\n= ! bananas strawberries avocados !\n= oranges", ['!', '^', '#', '@', '='])
solution('cherries bananas lemons\n, apples watermelons\nwatermelons cherries apples cherries strawberries oranges\ncherries ! apples\ncherries avocados pears apples', [
         '-', '#', ',', "'", '@', '?', '.', '^'])

# -*- coding: utf-8 -*-
test.assert_equals(solution('apples, pears # and bananas\ngrapes\nbananas !apples', ['#', '!']), 'apples, pears\ngrapes\nbananas')
test.assert_equals(solution('a #b\nc\nd $e f g', ['#', '$']), 'a\nc\nd')

Testing for solution('cherries bananas lemons\n, apples watermelons\nwatermelons cherries apples cherries strawberries oranges\ncherries ! apples\ncherries avocados pears apples', ['-', '#', ',', "'", '@', '?', '.', '^'])
It should work with random inputs too: 'cherries bananas lemons\n, apples watermelon\nwatermelons cherries apples cherries strawberries oranges\ncherries ! apples\ncherries avocados pears apples' 
should equal 'cherries bananas lemons\n\nwatermelons cherries apples cherries strawberries oranges\ncherries ! apples\ncherries avocados pears apples'

# Some coding
marker = ['#', '!']
r' *?\{}.*'.format(marker)

regex = r' *?[#].*'
print(re.sub(r' *?[#!].*', '', 'June #24\nAugust 9\nDe#c 12'))





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

format_duration(63)


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

test.assert_equals(format_duration(1), "1 second")
test.assert_equals(format_duration(62), "1 minute and 2 seconds")
test.assert_equals(format_duration(120), "2 minutes")
test.assert_equals(format_duration(3600), "1 hour")
test.assert_equals(format_duration(3662), "1 hour, 1 minute and 2 seconds")





"""Write a function, persistence, that takes in a positive parameter num and returns its multiplicative persistence, which is the number of times you must multiply the digits in num until you reach a single digit.

For example (Input --> Output):

39 --> 3 (because 3*9 = 27, 2*7 = 14, 1*4 = 4 and 4 has only one digit)
999 --> 4 (because 9*9*9 = 729, 7*2*9 = 126, 1*2*6 = 12, and finally 1*2 = 2)
4 --> 0 (because 4 is already a one-digit number)
"""


import numpy as np

def persistence(n):
    ind = 0
    while len(str(n)) != 1:
        n = np.product([int(i) for i in str(n)])
        ind += 1
    return ind

persistence(999)


@test.describe("Persistent Bugger.")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(persistence(39), 3)
        test.assert_equals(persistence(4), 0)
        test.assert_equals(persistence(25), 2)
        test.assert_equals(persistence(999), 4)





"""There is a bus moving in the city, and it takes and drop some people in each bus stop.

You are provided with a list (or array) of integer pairs. Elements of each pair represent number of people get into bus (The first item) and number of people get off the bus (The second item) in a bus stop.

Your task is to return number of people who are still in the bus after the last bus station (after the last array). Even though it is the last bus stop, the bus is not empty and some people are still in the bus, and they are probably sleeping there :D

Take a look on the test cases.

Please keep in mind that the test cases ensure that the number of people in the bus is always >= 0. So the return integer can't be negative.

The second value in the first integer array is 0, since the bus is empty in the first bus stop."""


import numpy as np

def number(bus_stops):
    return np.subtract.accumulate(np.sum((bus_stops), axis=0))[-1]
    # return -np.diff((np.sum((bus_stops), axis=0)))

number([[10, 0], [3, 5], [5, 8]])


import codewars_test as test
from solution import number

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(number([[10,0],[3,5],[5,8]]),5)
        test.assert_equals(number([[3,0],[9,1],[4,10],[12,2],[6,1],[7,10]]),17)
        test.assert_equals(number([[3,0],[9,1],[4,8],[12,2],[6,1],[7,8]]),21)





"""Given an array of integers.

Return an array, where the first element is the count of positives numbers and the second element is sum of negative numbers. 0 is neither positive nor negative.

If the input is an empty array or is null, return an empty array.

Example
For input [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -11, -12, -13, -14, -15], you should return [10, -65]."""


def count_positives_sum_negatives(arr):
    if not arr:
        return []
    pos, neg = 0, 0
    for i in arr:
        if i > 0:
            pos += 1
        else:
            neg += i
    return [pos, neg]

count_positives_sum_negatives(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -11, -12, -13, -14, -15])

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



[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -11, -12, -13, -14, -15].index(-11)



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
    return np.power(n, .5).is_integer()  # it's int, but object is float
    # return isinstance(np.power(n, .5), int) 
is_square(25)


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
    return (s + (m*60) + (h*60*60)) * 1000
past(0,1,1)


@test.describe("Fixed Tests")
def basic_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(past(0,1,1),61000)
        test.assert_equals(past(1,1,1),3661000)
        test.assert_equals(past(0,0,0),0)
        test.assert_equals(past(1,0,1),3601000)
        test.assert_equals(past(1,0,0),3600000)





"""You might know some pretty large perfect squares. But what about the NEXT one?

Complete the findNextSquare method that finds the next integral perfect square after the one passed as a parameter. Recall that an integral perfect square is an integer n such that sqrt(n) is also an integer.

If the parameter is itself not a perfect square then -1 should be returned. You may assume the parameter is non-negative.

Examples:(Input --> Output)

121 --> 144
625 --> 676
114 --> -1 since 114 is not a perfect square"""

import numpy as np

def find_next_square(sq):
    return np.square(np.sqrt(sq) + 1) if np.sqrt(sq).is_integer() else -1
find_next_square(121)


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





"""There is an array with some numbers. All numbers are equal except for one. Try to find it!

find_uniq([ 1, 1, 1, 2, 1, 1 ]) == 2
find_uniq([ 0, 0, 0.55, 0, 0 ]) == 0.55
It's guaranteed that array contains at least 3 numbers.

The tests contain some very huge arrays, so think about performance."""

from collections import Counter

def find_uniq(arr):
    count_dict = dict(Counter(arr))
    return sorted(count_dict, key=count_dict.get)[0]

find_uniq([3, 10, 3, 3, 3 ])

def find_uniq(arr):
    return Counter(arr).most_common()[-1][0]

def find_uniq(arr):
    return min(set(arr), key=arr.count)


@test.describe("Basic Tests")
def f():
    @test.it("Simple tests")
    def _():
        test.assert_equals(find_uniq([ 1, 1, 1, 2, 1, 1 ]), 2)
        test.assert_equals(find_uniq([ 0, 0, 0.55, 0, 0 ]), 0.55)
        test.assert_equals(find_uniq([ 3, 10, 3, 3, 3 ]), 10)





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





"""The goal of this exercise is to convert a string to a new string where each character in the new string is "(" if that character appears only once in the original string, or ")" if that character appears more than once in the original string. Ignore capitalization when determining if a character is a duplicate.

Examples
"din"      =>  "((("
"recede"   =>  "()()()"
"Success"  =>  ")())())"
"(( @"     =>  "))((" """

from collections import Counter

def duplicate_encode(word):
    return ''.join(['(' if dict(Counter(word.lower()))[char] == 1 else ')' for char in word.lower()])
duplicate_encode('Success')

def duplicate_encode(word):
    return ''.join(['(' if word.lower().count(char) == 1 else ')' for char in word.lower()])
duplicate_encode('Success')


@test.describe("Duplicate Encoder")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(duplicate_encode("din"),"(((")
        test.assert_equals(duplicate_encode("recede"),"()()()")
        test.assert_equals(duplicate_encode("Success"),")())())","should ignore case")
        test.assert_equals(duplicate_encode("(( @"),"))((")





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

def count_sheeps(sheep):
    None_list = np.array(sheep)
    Not_None_list = None_list[None_list != None]
    return np.nansum(Not_None_list)

count_sheeps(array1)

def count_sheeps(sheep):
    Not_None_list = list(filter(None, sheep))
    return np.nansum(Not_None_list)

def count_sheeps(sheep):
    return sheep.count(True)


# changed first two True's to something more instructive
array1 = [None,  np.NaN,  True,  False,
          True,  True,  True,  True ,
          True,  False, True,  False,
          True,  False, False, True ,
          True,  True,  True,  True ,
          False, False, True,  True ];
              
test.assert_equals(result := count_sheeps(array1), 17, "There are 17 sheeps in total, not %s" % result)





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
    num_list = [int(number) for number in numbers.split()]
    # return ' '.join([str(max(num_list)), str(min(num_list))])
    return '{} {}'.format(max(num_list), min(num_list))
high_and_low('8 3 -5 42 -1 0 0 -9 4 7 4 -4')


@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(high_and_low("8 3 -5 42 -1 0 0 -9 4 7 4 -4"), "42 -9");
        test.assert_equals(high_and_low("1 2 3"), "3 1");





"""Your task is to sort a given string. Each word in the string will contain a single number. This number is the position the word should have in the result.

Note: Numbers can be from 1 to 9. So 1 will be the first word (not 0).

If the input string is empty, return an empty string. The words in the input String will only contain valid consecutive numbers.

Examples
"is2 Thi1s T4est 3a"  -->  "Thi1s is2 3a T4est"
"4of Fo1r pe6ople g3ood th5e the2"  -->  "Fo1r the2 g3ood 4of th5e pe6ople"
""  -->  """""

import regex as re
int(re.search(r'\d+', 'is2').group())


def order(sentence):
    sentence_dict = {int(re.search(r'\d+', i).group()): i for i in sentence.split()}
    return ' '.join([sentence_dict[i] for i in range(1, len(sentence_dict) + 1)])
order('is2 Thi1s T4est 3a')

# int only [0-9]
def order(sentence):
    sentence_dict = {int(list(filter(str.isdigit, i))[0]): i for i in sentence.split()}
    return ' '.join([sentence_dict[i] for i in range(1, len(sentence_dict) + 1)])

def order(sentence):
    return ' '.join(sorted(sentence.split(), key=lambda x: int(list(filter(str.isdigit, x))[0])))

# the second sorted moves digit in the front, so then first sorted can sort strings them by
def order(sentence):
    return ' '.join(sorted(sentence.split(), key=lambda x: sorted(x)))
order('is2 Thi1s T4est 3a')


test.assert_equals(order("is2 Thi1s T4est 3a"), "Thi1s is2 3a T4est")
test.assert_equals(order("4of Fo1r pe6ople g3ood th5e the2"), "Fo1r the2 g3ood 4of th5e pe6ople")
test.assert_equals(order(""), "")




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
    if not year:
        return 0
    if (len(str(year)) == 2 or
        len(str(year)) == 1):
        return 1
    return int(str(year)[:-2]) if int(str(year)[-2:]) == 0 else int(str(year)[:-2])+1 
    # return int(str(year)[:-2])
century(1705)

def century(year):
    return (year + 99) // 100

import numpy as np

def century(year):
    return np.ceil(year / 100)


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
# Fist fail


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





"""You will be given an array a and a value x. All you need to do is check whether the provided array contains the value.

Array can contain numbers or strings. X can be either.

Return true if the array contains the value, false if not."""


def check(seq, elem):
    return elem in seq
check([66, 101], 66)



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

# tried to cheat with ) and D
def count_smileys(arr):
    if not arr:
        return 0
    count = 0
    for i in arr:
        if ')' in i:
            count += 1
        if 'D' in i:
            count += 1
    return count
count_smileys([':D',':~)',';~D',':)'])

import re
def count_smileys(arr):
    if not arr:
        return 0
    count = 0 
    for elem in arr:
        z = re.match(r'[;:][~-]?[)D]', elem)
        if z:
            count += 1
    return count
count_smileys([':D',':~)',';~D',':)'])

import re
def count_smileys(arr):
    return sum(True for elem in arr if re.match(r'[;:][~-]?[)D]', elem))

import re
def count_smileys(arr):
    return len(re.findall(r'[;:][~-]?[)D]', ' '.join(arr)))


test.describe("Basic tests")
test.assert_equals(count_smileys([]), 0)
test.assert_equals(count_smileys([':D',':~)',';~D',':)']), 4)
test.assert_equals(count_smileys([':)',':(',':D',':O',':;']), 2)
test.assert_equals(count_smileys([';]', ':[', ';*', ':$', ';-D']), 1)





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
    if (not strarr or
        k > len(strarr) or
        k <= 0):
        return ''
    concat_list = [''.join(strarr[i : i+k]) for i in range(len(strarr) - k + 1)]
    return next(filter(lambda x: len(x) == len(max(concat_list, key=len)), concat_list))
longest_consec(["zone", "abigail", "theta", "form", "libe", "zas"], 2)

def longest_consec(strarr, k):
    if (not strarr or
        k > len(strarr) or
        k <= 0):
        return ''
    return max([''.join(strarr[i : i+k]) for i in range(len(strarr) - k + 1)], key=len)


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





"""Build Tower
Build a pyramid-shaped tower given a positive integer number of floors. A tower block is represented with "*" character.

For example, a tower with 3 floors looks like this:

[
  "  *  ",
  " *** ", 
  "*****"
]"""


def tower_builder(n_floors):
    return [('*'*(2*i - 1)).center(2*n_floors - 1) for i in range(1, n_floors + 1)]
tower_builder(3)


@test.describe("Build Tower")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(tower_builder(1), ['*', ])
        test.assert_equals(tower_builder(2), [' * ', '***'])
        test.assert_equals(tower_builder(3), ['  *  ', ' *** ', '*****'])





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
        len(arr) == 1):
        return 0
    return sum(sorted(arr)[1:-1])
sum_array([-6, 20, -1, 10, -12])


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





"""Create a function which answers the question "Are you playing banjo?".
If your name starts with the letter "R" or lower case "r", you are playing banjo!

The function takes a name as its only argument, and returns one of the following strings:

name + " plays banjo" 
name + " does not play banjo""""


def are_you_playing_banjo(name):
    return name + ' plays banjo' if name[0].upper() == 'R' else name + ' does not play banjo'
are_you_playing_banjo('rartin')


@test.describe("Fixed Tests")
def basic_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(are_you_playing_banjo("martin"), "martin does not play banjo");
        test.assert_equals(are_you_playing_banjo("Rikke"), "Rikke plays banjo");
        test.assert_equals(are_you_playing_banjo("bravo"), "bravo does not play banjo")
        test.assert_equals(are_you_playing_banjo("rolf"), "rolf plays banjo")





"""Task
Given an array of integers, remove the smallest value. Do not mutate the original array/list. If there are multiple elements with the same value, remove the one with a lower index. If you get an empty array/list, return an empty array/list.

Don't change the order of the elements that are left.

Examples
* Input: [1,2,3,4,5], output= [2,3,4,5]
* Input: [5,3,2,1,4], output = [5,3,2,4]
* Input: [2,2,1,2,1], output = [2,2,2,1]"""


def remove_smallest(numbers):
    new_numbers = numbers[:]
    if numbers:
        new_numbers.remove(min(new_numbers))
    return new_numbers
remove_smallest([1, 2, 3, 4, 5])
remove_smallest([])


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





"""Take 2 strings s1 and s2 including only letters from ato z. Return a new sorted string, the longest possible, containing distinct letters - each taken only once - coming from s1 or s2.

Examples:
a = "xyaabbbccccdefww"
b = "xxxxyyyyabklmopq"
longest(a, b) -> "abcdefklmopqwxy"

a = "abcdefghijklmnopqrstuvwxyz"
longest(a, a) -> "abcdefghijklmnopqrstuvwxyz""""


def longest(a1, a2):
    return ''.join(sorted(list(set(a1 + a2))))
longest("aretheyhere", "yestheyarehere")

def longest(a1, a2):
    return ''.join(sorted(list(set(a1) | set(a2))))
longest("aretheyhere", "yestheyarehere")


@test.describe("longest")
def tests():
    @test.it("basic tests")
    def basics():
        test.assert_equals(longest("aretheyhere", "yestheyarehere"), "aehrsty")
        test.assert_equals(longest("loopingisfunbutdangerous", "lessdangerousthancoding"), "abcdefghilnoprstu")
        test.assert_equals(longest("inmanylanguages", "theresapairoffunctions"), "acefghilmnoprstuy")





"""Implement the function unique_in_order which takes as argument a sequence and returns a list of items without any elements with the same value next to each other and preserving the original order of elements.

For example:

unique_in_order('AAAABBBCCDAABBB') == ['A', 'B', 'C', 'D', 'A', 'B']
unique_in_order('ABBCcAD')         == ['A', 'B', 'C', 'c', 'A', 'D']
unique_in_order([1,2,2,3,3])       == [1,2,3]"""


def unique_in_order(iterable):
    if not iterable:
        return []
    it = [iterable[i] for i in range(1, len(iterable)) if iterable[i] != iterable[i - 1]]
    it.insert(0, iterable[0])
    return it
unique_in_order('AAAABBBCCDAABBB')

import numpy as np
def unique_in_order(iterable):
    return ([] if not iterable else
            np.concatenate((
                [iterable[0]],
                [iterable[i] for i in range(1, len(iterable)) if iterable[i] != iterable[i - 1]
                ]
            )))

unique_in_order('AAAABBBCCDAABBB')

from itertools import groupby
def unique_in_order(iterable):
    return [k for k, _ in groupby(iterable)]

test.assert_equals(unique_in_order('AAAABBBCCDAABBB'), ['A','B','C','D','A','B'])





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
    if not (h > 0 and 
            0 < bounce < 1 and 
            window < h):
        return -1
    counter = 1
    while window < h * bounce:
        counter += 2
        h *= bounce
    return counter
bouncing_ball(3, 0.66, 1.5)
bouncing_ball(2, 0.5, 1)


@test.it('Fixed Tests')
def tests():
    testing(2, 0.5, 1, 1)
    testing(3, 0.66, 1.5, 3)
    testing(30, 0.66, 1.5, 15)
    testing(30, 0.75, 1.5, 21)





"""You get an array of numbers, return the sum of all of the positives ones.

Example [1,-4,7,12] => 1 + 7 + 12 = 20

Note: if there is nothing to sum, the sum is default to 0."""


def positive_sum(arr):
    return sum(filter(lambda x: x > 0, arr))


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





"""Given an array of integers, return a new array with each value doubled.

For example:

[1, 2, 3] --> [2, 4, 6]"""

import numpy as np
def maps(a):
    return list(2 * np.array(a))
maps([1, 2, 3])

def maps(a):
    return list(map(lambda x: x * 2, a))


@test.describe("Fixed Tests")
def fixed_tests():
    @test.it('Basic Test Cases')
    def basic_test_cases():
        test.assert_equals(maps([1, 2, 3]), [2, 4, 6])
        test.assert_equals(maps([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), [0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
        test.assert_equals(maps([]), [])





"""Complete the solution so that it returns true if the first argument(string) passed in ends with the 2nd argument (also a string).

Examples:

solution('abc', 'bc') # returns true
solution('abc', 'd') # returns false"""


def solution(string, ending):
    if not ending:
        return True
    return True if ''.join(reversed(string)).find(''.join(reversed(ending))) == 0 and string.find(ending) != -1 else False
solution('abcde', 'cde')
solution('sumo', 'omo')
solution('samurai', 'ai')

def solution(string, ending):
    if not ending:
        return True
    return ending == string[-len(ending):]

def solution(string, ending):
    return string.endswith(ending)


test.assert_equals(solution('abcde', 'cde'), True)
test.assert_equals(solution('abcde', 'abc'), False)
test.assert_equals(solution('abcde', ''), True)





"""In this kata you are required to, given a string, replace every letter with its position in the alphabet.

If anything in the text isn't a letter, ignore it and don't return it.

"a" = 1, "b" = 2, etc.

Example
alphabet_position("The sunset sets at twelve o' clock.")
Should return "20 8 5 19 21 14 19 5 20 19 5 20 19 1 20 20 23 5 12 22 5 15 3 12 15 3 11" ( as a string )"""


import string
def alphabet_position(text):
    return ' '.join(str(ord(char.lower()) - 96) for char in text if char in string.ascii_letters)
alphabet_position("The sunset sets at twelve o' clock.")

def alphabet_position(text):
    return ' '.join(str(ord(char.lower()) - 96) for char in text if char.isalpha())
alphabet_position("The sunset sets at twelve o' clock.")



from random import randint
test.assert_equals(alphabet_position("The sunset sets at twelve o' clock."), "20 8 5 19 21 14 19 5 20 19 5 20 19 1 20 20 23 5 12 22 5 15 3 12 15 3 11")
test.assert_equals(alphabet_position("The narwhal bacons at midnight."), "20 8 5 14 1 18 23 8 1 12 2 1 3 15 14 19 1 20 13 9 4 14 9 7 8 20")

number_test = ""
for item in range(10):
    number_test += str(randint(1, 9))
test.assert_equals(alphabet_position(number_test), "")





"""Task
In this simple Kata your task is to create a function that turns a string into a Mexican Wave. You will be passed a string and you must return that string in an array where an uppercase letter is a person standing up. 
Rules
 1.  The input string will always be lower case but maybe empty.

 2.  If the character in the string is whitespace then pass over it as if it was an empty seat
Example
wave("hello") => ["Hello", "hEllo", "heLlo", "helLo", "hellO"]"""


def wave(people):
    return [people[:i] + people[i].upper() + people[i+1:] for i in range(len(people)) if people[i].isalpha()]
wave('hello')


result = ["Codewars", "cOdewars", "coDewars", "codEwars", "codeWars", "codewArs", "codewaRs", "codewarS"]





"""Your task is to find the first element of an array that is not consecutive.

By not consecutive we mean not exactly 1 larger than the previous element of the array.

E.g. If we have an array [1,2,3,4,6,7,8] then 1 then 2 then 3 then 4 are all consecutive but 6 is not, so that's the first non-consecutive number.

If the whole array is consecutive then return null2.

The array will always have at least 2 elements1 and all elements will be numbers. The numbers will also all be unique and in ascending order. The numbers could be positive or negative and the first non-consecutive could be either too!

"""


def first_non_consecutive(arr):
    result = None
    for i in range(len(arr) - 1):
        if arr[i+1] - arr[i] != 1:
            return arr[i+1]
first_non_consecutive([1, 2, 3, 4, 5, 6, 7, 9])


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





"""Your task is to convert strings to how they would be written by Jaden Smith. The strings are actual quotes from Jaden Smith, but they are not capitalized in the same way he originally typed them.

Example:

Not Jaden-Cased: "How can mirrors be real if our eyes aren't real"
Jaden-Cased:     "How Can Mirrors Be Real If Our Eyes Aren't Real""""


def to_jaden_case(string_text):
    return ' '.join(i.capitalize() for i in string_text.split())
to_jaden_case("How can mirrors be real if our eyes aren't real")

import string
def to_jaden_case(string_text):
    return string.capwords(string_text)

quote = "How can mirrors be real if our eyes aren't real"
test.assert_equals(to_jaden_case(quote), "How Can Mirrors Be Real If Our Eyes Aren't Real")





"""Given an array of integers, find the one that appears an odd number of times.

There will always be only one integer that appears an odd number of times.

Examples
[7] should return 7, because it occurs 1 time (which is odd).
[0] should return 0, because it occurs 1 time (which is odd).
[1,1,2] should return 2, because it occurs 1 time (which is odd).
[0,1,0,1,0] should return 0, because it occurs 3 times (which is odd).
[1,2,2,3,3,3,4,3,3,3,2,2,1] should return 4, because it appears 1 time (which is odd)."""


from collections import Counter
def find_it(seq):
    return [k for k, v in dict(Counter(seq)).items() if v % 2][0]
find_it([20, 1, -1, 2, -2, 3, 3, 5, 5, 1, 2, 4, 20, 4, -1, -2, 5])


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





"""Given a string of words, you need to find the highest scoring word.

Each letter of a word scores points according to its position in the alphabet: a = 1, b = 2, c = 3 etc.

You need to return the highest scoring word as a string.

If two words score the same, return the word that appears earliest in the original string.

All letters will be lowercase and all inputs will be valid."""


def high(x):
    word_list = [sum(ord(char.lower()) - ord('a') + 1 for char in word) for word in x.split()]
    return x.split()[word_list.index(max(word_list))]
high('man i need a taxi up to ubud')


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