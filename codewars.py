def count_bits(n):
    return bin(n).count('1')


import codewars_test as test
from solution import count_bits

@test.describe("Fixed Tests")
def fixed_tests():
    @test.it("Basic Tests")
    def basic_tests():
        test.assert_equals(count_bits(0), 0)
        test.assert_equals(count_bits(4), 1)
        test.assert_equals(count_bits(7), 3)
        test.assert_equals(count_bits(9), 2)
        test.assert_equals(count_bits(10), 2)




def digital_root(n):
    while True:
        if n > 9:
            n = sum([int(digit) for digit in str(n)])
        else:
            return n

digital_root(16)
digital_root(942)
test.assert_equals(digital_root(16), 7)
test.assert_equals(digital_root(942), 6)
test.assert_equals(digital_root(132189), 6)
test.assert_equals(digital_root(493193), 2)




# A pangram is a sentence that contains every single letter of the alphabet at least once. For example, the sentence "The quick brown fox jumps over the lazy dog" is a pangram, because it uses the letters A-Z at least once (case is irrelevant).
# Given a string, detect whether or not it is a pangram. Return True if it is, False if not. Ignore numbers and punctuation.

import string

def is_pangram(s):
    return all([True if letter in s.lower() else False for letter in string.ascii_lowercase])


def is_pangram(s):
    return set(string.ascii_lowercase) <= set(s.lower())

def is_pangram(s):
    return set(string.ascii_lowercase).issubset(set(s.lower()))


pangram = "The quick, brown fox jumps over the lazy dog!"
is_pangram(pangram)
test.assert_equals(is_pangram(pangram), True)
