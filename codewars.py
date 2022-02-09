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

def solution(number):
    if number < 0:
        return 0
    else:
        return np.sum([i for i in range(number) if i%3 == 0 or i%5 == 0])


def solution(number):
    return np.sum([i for i in range(number) if i%3 == 0 or i%5 == 0])


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
    



















