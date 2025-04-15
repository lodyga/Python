# codewars kata

# from xml import dom
# from nbformat import read
# from solution import arithmetic
# from gettext import find
# from regex import D
# from solution import enough
# import enum
# from itertools import reduce
# import itertools
# from operator import index
# from hel import add





# Bit Counting
# https://www.codewars.com/kata/526571aae218b8ee490006f4
"""
Write a function that takes an integer as input, and returns the number of bits that are equal
to one in the binary representation of that number. You can guarantee that input is non-negative.
Example: The binary representation of 1234 is 10011010010, so the function should return 5 in this case

(count_bits(1234), 5)
"""
(count_bits(0), 0)
(count_bits(4), 1)
(count_bits(7), 3)
(count_bits(9), 2)
(count_bits(10), 2)
(count_bits(1234), 5)


def count_bits(n):
    return bin(n)[2:].count('1')
    # return f"{number:>b}".count("1")
    # return '{:b}'.format(n).count('1')
    # return sum(True if int(i) else False for i in bin(n)[2:])
    # return n.bit_count()


def count_bits(n):
    num = ""
    counter = 0

    while True:
        mod = n % 2
        n = n // 2
        num = str(mod) + num

        if mod:
            counter += 1
        
        if not n:
            break

    return num




# Sum of Digits / Digital Root
# https://www.codewars.com/kata/541c8630095125aba6000c00/train/python
"""
Digital root is the recursive sum of all the digits in a number.
Given n, take the sum of the digits of n. If that value has more than one digit, 
continue reducing in this way until a single-digit number is produced. The input will be a non-negative integer.

(digital_root(169), 7)
"""
(digital_root(16), 7)
(digital_root(169), 7)
(digital_root(942), 6)
(digital_root(132189), 6)
(digital_root(493193), 2)

def digital_root(number):
    while len(str(number)) != 1:
        number = sum(int(digit) for digit in str(number))
    return number

def digital_root(number):
    while number > 9:
        number = sum(map(int, str(number)))
    return number

def digital_root(n):
    return n if n < 10 else digital_root(sum(map(int, str(n))))

def digital_root(n):
    while len(str(n)) != 1:
        n = sum((int(digit) for digit in str(n)))
    return n





# Detect Pangram
# https://www.codewars.com/kata/545cedaa9943f7fe7b000048/train/
"""
A pangram is a sentence that contains every single letter of the alphabet at least once.
For example, the sentence "The quick brown fox jumps over the lazy dog" is a pangram, because it uses the letters A-Z at least once (case is irrelevant).
Given a string, detect whether or not it is a pangram. Return True if it is, False if not. Ignore numbers and punctuation.

(is_pangram("The quick, brown fox jumps over the lazy dog!"), True)
"""
(is_pangram("The quick, brown fox jumps over the lazy dog!"), True)
(is_pangram("ABCD45EFGH,IJK,LMNOPQR56STUVW3XYZ"), True)


def is_pangram(sentence):
    letters = set()

    for letter in sentence.lower():
        if letter.isalpha():
            letters.add(letter)

    return len(letters) == 26

def is_pangram(sentence):
    return len({letter for letter in sentence.lower() if letter.isalpha()}) == 26


import string

def is_pangram(s):
    return set(string.ascii_lowercase) <= set(s.lower())


def is_pangram(s):
    return set(s.lower()) >= set(string.ascii_lowercase)

def is_pangram(s):
    return all(True if letter in s.lower() else False for letter in string.ascii_lowercase)

def is_pangram(s):
    return set(string.ascii_lowercase).issubset(set(s.lower()))

def is_pangram(s):
    return set(s.lower()).issuperset(string.ascii_lowercase)

def is_pangram(s):
    return set(string.ascii_lowercase) & set(s.lower()) == set(string.ascii_lowercase)





# Multiples of 3 or 5
# https://www.codewars.com/kata/514b92a657cdc65150000006/train/python
"""
If we list all the natural numbers below 10 that are multiples of 3 or 5, we get 3, 5, 6 and 9.
The sum of these multiples is 23.
Finish the solution so that it returns the sum of all the multiples of 3 or 5 below the number passed in.
Additionally, if the number is negative, return 0 (for languages that do have them).
Note: If the number is a multiple of both 3 and 5, only count it once.

(solution(4), 3)
"""
(solution(4), 3)
(solution(6), 8)
(solution(16), 60)
(solution(3), 0)
(solution(5), 3)
(solution(15), 45)
(solution(0), 0)
(solution(-1), 0)
(solution(10), 23)
(solution(20), 78)
(solution(200), 9168)


def solution(nums):
    div_sum = 0
    
    for num in range(nums):
        if (not num % 3 or
            not num % 5):
            div_sum += num
    
    return div_sum

def solution(numbers):
    return sum(digit for digit in range(3, numbers) if not digit % 3 or not digit % 5)





# The Hashtag Generator
# https://www.codewars.com/kata/52449b062fb80683ec000024
"""
The marketing team is spending way too much time typing in hashtags.
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
(generate_hashtag(" Hello there thanks for trying my Kata"), "#HelloThereThanksForTryingMyKata")
(generate_hashtag("    Hello     World   " ), "#HelloWorld")
(generate_hashtag(''), False)  # 'Expected an empty string to return False'
(generate_hashtag('Do We have A Hashtag')[0], '#')  # 'Expeted a Hashtag (#) at the beginning.'
(generate_hashtag('Codewars'), '#Codewars')  # 'Should handle a single word.'
(generate_hashtag('Codewars      '), '#Codewars')  # 'Should handle trailing whitespace.'
(generate_hashtag('Codewars Is Nice'), '#CodewarsIsNice')  # 'Should remove spaces.'
(generate_hashtag('codewars is nice'), '#CodewarsIsNice')  # 'Should capitalize first letters of words.'
(generate_hashtag('CodeWars is nice'), '#CodewarsIsNice')  # 'Should capitalize all letters of words - all lower case but the first.'
(generate_hashtag('c i n'), '#CIN')  # 'Should capitalize first letters of words even when single letters.'
(generate_hashtag('codewars  is  nice'), '#CodewarsIsNice')  # 'Should deal with unnecessary middle spaces.'
(generate_hashtag('Looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong Cat'), False)  # 'Should return False if the final word is longer than 140 chars.'
(generate_hashtag('    '), False)


def generate_hashtag(sentence):
    if not sentence.strip():
        return False
    
    hashtag = "#"
    for word in sentence.split():
        hashtag += word.capitalize()
        
        if len(hashtag) > 140:
            return False
    
    return hashtag


def generate_hashtag(s):
    if not s.strip():
        return False
    
    solution = "#" + "".join(word.capitalize() for word in s.strip().split())
    return False if len(s) > 140 else solution
    # return "#" + s.title().replace(" ", "")


def generate_hashtag(s):
    if not s.strip():
        return False
    
    solution = '#' + ''.join(map(lambda word: word.capitalize(), s.split()))
    return False if len(s) > 140 else solution









# Product of consecutive Fib numbers
# https://www.codewars.com/kata/5541f58a944b85ce6d00006a
"""
The Fibonacci numbers are the numbers in the following integer sequence (Fn):
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
                # since F(8) = 21, F(9) = 34, F(10) = 55 and 21 * 34 < 800 < 34 * 55
                
(fib_prod(4895), [55, 89, True])
"""
(product_fib(714), (21, 34, True) )
(product_fib(800), (34, 55, False))
(product_fib(4895), (55, 89, True))
(product_fib(5895), (89, 144, False))
(product_fib(0), (0, 1, True))


def product_fib(num):
    a = 0
    b = 1

    while a * b < num:
        a, b = b, a + b
        # b = a + b
        # a = b - a

    return (a, b, a * b == num)


# fib_prod(0) should be True
class Fib(object):

    """docstring for Fib"""

    def __init__(self):
        super(Fib, self).__init__()
        self.a = 0
        self.b = 1

    def __call__(self):
        self.a, self.b = self.b, self.a + self.b
        return self.a, self.b


def fib_prod(prod):
    fib = Fib()
    a, b = fib()
    while (a * b) < prod:
        a, b = fib()
    return [a, b, (a * b) == prod]





# Write Number in Expanded Form
# https://www.codewars.com/kata/5842df8ccbd22792a4000245
"""
Write Number in Expanded Form
You will be given a number and you will need to return it as a string in Expanded Form. For example:

expanded_form(12) # Should return '10 + 2'
expanded_form(42) # Should return '40 + 2'
expanded_form(70304) # Should return '70000 + 300 + 4'
NOTE: All numbers will be whole numbers greater than 0.

(expanded_form(42), '40 + 2');
"""
(expanded_form(12), '10 + 2')
(expanded_form(42), '40 + 2')
(expanded_form(70304), '70000 + 300 + 4')


def expanded_form(num):
    expanded = []
    num_lenght = len(str(num))

    for index, digit in enumerate(str(num), 1):
        digit = int(digit)
        power = num_lenght - index

        if digit:
            expanded.append(str(digit * 10 ** power))
    
    return " + ".join(expanded)


def expanded_form(num):
    return " + ".join(str(int(digit) * 10**(len(str(num)) - index)) for index, digit in enumerate(str(num), 1) if int(digit))





# RGB To Hex Conversion
# https://www.codewars.com/kata/513e08acc600c94f01000001
"""
The rgb function is incomplete. Complete it so that passing in RGB decimal values will result in a hexadecimal representation being returned. Valid decimal values for RGB are 0 - 255. Any values that fall out of that range must be rounded to the closest valid value.

Note: Your answer should always be 6 characters long, the shorthand with 3 will not work here.

The following are examples of expected output values:

rgb(255, 255, 255) # returns FFFFFF
rgb(255, 255, 300) # returns FFFFFF
rgb(0,0,0) # returns 000000
rgb(148, 0, 211) # returns 9400D3

(rgb(-20,275,125), "00FF7D", "testing out of range values")
"""
(rgb(0, 0, 0), "000000"),  # "testing zero values"
(rgb(1, 2, 3), "010203"),  # "testing near zero values"
(rgb(255, 255, 255), "FFFFFF"),  # "testing max values"
(rgb(254, 253, 252), "FEFDFC"),  # "testing near max values"
(rgb(-20, 275, 125), "00FF7D"),  # "testing out of range values"


def rgb(r, g, b):
    def digit_to_hex(color):
        color = max(min(color, 255), 0)
        color = hex(color)
        # _, color = color.split("x")
        color = color[2:]
        color = color.upper()
        if len(color) == 1:
            color = "0" + color
    
        return color
    
    # return digit_to_hex(r) + digit_to_hex(g) + digit_to_hex(b)
    return "".join(digit_to_hex(color) for color in (r, g, b))


def rgb(r, g, b):
    rgb_round = lambda x: min(255, max(0, x))
    return ('{:02X}'*3).format(rgb_round(r), rgb_round(g), rgb_round(b))
    # return ('{:02X}'*3).format(*map(lambda x: rgb_round(x), (r, g, b)))


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





# The observed PIN
# https://www.codewars.com/kata/5263c6999e0f40dee200059d
"""
Alright, detective, one of our colleagues successfully observed our target person, Robby the robber. We followed him to a secret warehouse, where we assume to find all the stolen stuff. The door to this warehouse is secured by an electronic combination lock. Unfortunately our spy isn't sure about the PIN he saw, when Robby entered it.

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

Detective, we are counting on you!
"""
(get_pins('8'), ['5', '7', '8', '9', '0'])
(get_pins('11'), ["11", "22", "44", "12", "21", "14", "41", "24", "42"])


# passing index to dfs, combination as a shared variable (list)
def get_pins(observed):
    pin = []
    pins = []
    adjacent = {"0": "08",
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

    def dfs(index):
        if index == len(observed):
            pins.append("".join(pin))
            return

        for number in adjacent[observed[index]]:
            pin.append(number)
            dfs(index + 1)
            pin.pop()  # backtrack

    dfs(0)

    return pins


# pin as an argument (string) to the dfs function, slower than the list approach
def get_pins(observed):
    pins = []
    adjacent = {"0": "08",
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

    def dfs(index, pin):
        if index == len(observed):
            pins.append(pin)
            return

        for number in adjacent[observed[index]]:
            pin += number
            dfs(index + 1, pin)
            pin = pin[:-1]

    dfs(0, "")

    return pins


# oldies
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





# Maximum subarray sum
# https://www.codewars.com/kata/54521e9ec8e60bc4de000d6c
"""
The maximum sum subarray problem consists in finding the maximum sum of a contiguous subsequence in an array or list of integers:

max_sequence([-2, 1, -3, 4, -1, 2, 1, -5, 4])
# should be 6: [4, -1, 2, 1]
Easy case is when the list is made up of only positive numbers and the maximum sum is the sum of the whole array. 
If the list is made up of only negative numbers, return 0 instead.

Empty list is considered to have zero greatest sum. Note that the empty list or array is also a valid sublist/subarray.
"""
(max_sequence([-2, 1, -3, 4, -1, 2]), 5)
(max_sequence([-2, 1, -3, 4, -1, 2, 1, -5, 4]), 6)  # sum([4, -1, 2, 1]) = 6
(max_sequence([-2]), 0)
(max_sequence([]), 0)
(max_sequence([-2, 1]), 1)
(max_sequence([7, 4, 11, -11, 39, 36, 10, -6, 37, -10, -32, 44, -26, -34, 43, 43]), 155)


# dp, tabulation
# O(n), O(n)
def max_sequence(nums):  
    if not nums:  
        return 0  # Return 0 if the input list is empty

    tabul = [0] * len(nums)  # Initialize the array
    tabul[0] = max(nums[0], 0)  # Initialize the first element (either nums[0] or 0)

    for index, num in enumerate(nums[1:], 1):  
        tabul[index] = max(tabul[index - 1] + num, 0)  # Update current element with max of (previous sum + current num) or 0

    return max(tabul)  # Return the maximum value from the tabul list


# dp, tabulation
# O(n), O(1)
def max_sequence(nums):  
    if not nums:  
        return 0  # Return 0 if the input list is empty
    
    max_sum = 0  # Initialize max_sum to track the maximum sum encountered
    current = max(nums[0], 0)  # Initialize current with either nums[0] or 0
    
    for num in nums[1:]:  
        next = max(current + num, 0)  # Calculate the next potential sum (current + num or 0)
        max_sum = max(max_sum, next)  # Update max_sum if next is greater
        current = next  # Move to the next number by updating current
    
    return max_sum  # Return the largest sum found



# oldies
def max_sequence(arr):
    result, current = 0, 0
    for number in arr:
        if current < 0: 
            current = 0
        current += number
        result = max(result, current)
    return result


def max_sequence(arr):
    result, current = 0, 0
    for number in arr:
        if current <= 0: 
            current = number
        else:
            current += number
        result = max(result, current)
    return result



# O(n2) 
def max_sequence(arr):
    max_reselt = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr) + 1):
            max_reselt = max(max_reselt, sum(arr[i:j]))
    return max_reselt





# Weight for weight
# https://www.codewars.com/kata/55c6126177c9441a570000cc/train/python
"""
My friend John and I are members of the "Fat to Fit Club (FFC)". John is worried because each month a list with the weights of members is published and each month he is the last on the list which means he is the heaviest.

I am the one who establishes the list so I told him: "Don't worry any more, I will modify the order of the list". It was decided to attribute a "weight" to numbers. The weight of a number will be from now on the sum of its digits.

For example 99 will have "weight" 18, 100 will have "weight" 1 so in the list 100 will come before 99.

Given a string with the weights of FFC members in normal order can you give this string ordered by "weights" of these numbers?

Example:
"56 65 74 100 99 68 86 180 90" ordered by numbers weights becomes: 

"100 180 90 56 65 74 68 86 99"
When two numbers have the same "weight", let us class them as if they were strings (alphabetical ordering) and not numbers:

180 is before 90 since, having the same "weight" (9), it comes before as a string.

All numbers in the list are positive numbers and the list can be empty."""
(order_weight("103 123 4444 99 2000"), "2000 103 123 4444 99")
(order_weight("2000 10003 1234000 44444444 9999 11 11 22 123"), "11 11 2000 10003 22 123 1234000 44444444 9999")
(order_weight(""), "")


# sorting key as a lambda function
sorting_key = lambda number: (sum(int(digit) for digit in number), number)

def order_weight(nums):
    nums = nums.split()
    nums.sort(key=lambda number: sorting_key(number))
    return " ".join(nums)


# sum of digits as a function
def sum_digits(number):
    return sum(int(digit) for digit in number)

def order_weight(nums):
    nums = nums.split()
    nums.sort(key=lambda number: (sum_digits(number), number))
    return " ".join(nums)





# Strip Comments
# https://www.codewars.com/kata/51c8e37cee245da6b40000bd/train/python
"""
Complete the solution so that it strips all text that follows any of a set of comment markers passed in. Any whitespace at the end of the line should also be stripped out.
solution("apples, pears # and bananas\ngrapes\nbananas !apples", ["#", "!"])
# result should == "apples, pears\ngrapes\nbananas"
"""
(strip_comments('apples, pears # and bananas\ngrapes\nbananas !apples', ['#', '!']), 'apples, pears\ngrapes\nbananas')
(strip_comments(' a #b\nc\nd $e f g', ['#', '$']), ' a\nc\nd')
(strip_comments('a #b\nc\nd $e f g', ['#', '$']), 'a\nc\nd')
(strip_comments("' = lemons\napples pears\n. avocados\n= ! bananas strawberries avocados !\n= oranges", ['!', '^', '#', '@', '=']), "'\napples pears\n. avocados\n\n")
(strip_comments('aa bb cc', []), 'aa bb cc')
(strip_comments('aa bb\n#cc dd', ['#']), 'aa bb\n')

def strip_comments(paragraph, markers):
    sentences = paragraph.split("\n")
    
    for marker in markers:
        clean_sentences = []
        
        for sentence in sentences:
            if marker in sentence:
                sentence, *_ =sentence.split(marker)
            sentence = sentence.rstrip()
            clean_sentences.append(sentence)
        
        sentences = clean_sentences

    return "\n".join(sentences)


# check for all markers in one pass, slightly faster 
def strip_comments(paragraph, markers):
    clean_sentences = []
    
    for sentence in paragraph.split("\n"):
        no_mark = True
        
        for index, marker in enumerate(sentence):
            if marker in markers:
                clean_sentences.append(sentence[: index].rstrip())
                no_mark = False
                break

        if no_mark:
            clean_sentences.append(sentence)

    return "\n".join(clean_sentences)


# oldie
import re
def strip_comments(string, markers):
    for marker in markers:
        string = re.sub(r' *?{}.*$'.format(marker), '', string, flags=re.M)
    return string







# Human readable duration format
# https://www.codewars.com/kata/52742f58faf5485cae000b9a/train/python
"""
Your task in order to complete this Kata is to write a function which formats a duration, given as a number of seconds, in a human-friendly way.

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

A unit of time must be used "as much as possible". It means that the function should not return 61 seconds, but 1 minute and 1 second instead. Formally, the duration specified by of a component must not be greater than any valid more significant unit of time.
"""
(format_duration(0), "now")
(format_duration(1), "1 second")
(format_duration(2), "2 seconds")
(format_duration(62), "1 minute and 2 seconds")
(format_duration(120), "2 minutes")
(format_duration(3600), "1 hour")
(format_duration(3662), "1 hour, 1 minute and 2 seconds")
(format_duration(8237710), "95 days, 8 hours, 15 minutes and 10 seconds")


def format_duration(num):
    if not num:
        return "now"

    how_many_seconds = (
        ("year", 60 * 60 * 24 * 365),
        ("day", 60 * 60 * 24),
        ("hour", 60 * 60),
        ("minute", 60),
        ("second", 1)
    )

    duration_nums = []  # how many years, days, hours, minutes, seconds int list
    for seconds in how_many_seconds:
        duration_nums.append(num // seconds[1])
        num = num % seconds[1]
    
    duration_strs = []  # formated to string date chunks
    for index, dur_num in enumerate(duration_nums):
        duration = ""
        
        if dur_num:  # if chunk is not 0
            duration += f"{dur_num} {how_many_seconds[index][0]}"
            if dur_num > 1:  # if plural
                duration += "s"

            duration_strs.append(duration)

    if len(duration_strs) == 1:  # in only one chunk
        return duration_strs[0]
    else:
        # join chunks all exept the last one with ", " and the last one with " and "
        return ", ".join(duration_strs[:-1]) + " and " + duration_strs[-1]





# Persistent Bugger.
# https://www.codewars.com/kata/55bf01e5a717a0d57e0000ec
"""
Write a function, persistence, that takes in a positive parameter num and returns its multiplicative persistence, which is the number of times you must multiply the digits in num until you reach a single digit.

For example (Input --> Output):

39 --> 3 (because 3*9 = 27, 2*7 = 14, 1*4 = 4 and 4 has only one digit)
999 --> 4 (because 9*9*9 = 729, 7*2*9 = 126, 1*2*6 = 12, and finally 1*2 = 2)
4 --> 0 (because 4 is already a one-digit number)
"""
(persistence(39), 3)
(persistence(4), 0)
(persistence(25), 2)
(persistence(999), 4)


def persistence(num):
    counter = 0

    while num > 9:
        counter += 1
        current_prod = 1
        
        for digit in str(num):
            current_prod *= int(digit)

        num = current_prod

    return counter


import numpy as np

def persistence(num):
    counter = 0

    while num > 9:
        counter += 1
        num = np.prod([int(digit) for digit in str(num)])
    
    return counter






# Number of People in the Bus
# https://www.codewars.com/kata/5648b12ce68d9daa6b000099
"""
There is a bus moving in the city, and it takes and drop some people in each bus stop.

You are provided with a list (or array) of integer pairs. Elements of each pair represent number of people get into bus (The first item) and number of people get off the bus (The second item) in a bus stop.

Your task is to return number of people who are still in the bus after the last bus station (after the last array). Even though it is the last bus stop, the bus is not empty and some people are still in the bus, and they are probably sleeping there :D

Take a look on the test cases.

Please keep in mind that the test cases ensure that the number of people in the bus is always >= 0. So the return integer can't be negative.

The second value in the first integer array is 0, since the bus is empty in the first bus stop."""
(number([[10, 0], [3, 5], [5, 8]]), 5)
(number([[3, 0], [9, 1], [4, 10], [12, 2], [6, 1], [7, 10]]), 17)
(number([[3, 0], [9, 1], [4, 8], [12, 2], [6, 1], [7, 8]]), 21)


def number(bus_stops):
    number_of_people = 0

    for enter_people, exit_people in bus_stops:
        number_of_people += enter_people - exit_people

        if number_of_people < 0:
            return False

    return number_of_people


def number(bus_stops):
    return sum(enter - exit for enter, exit in bus_stops)


def number(bus_stops):
    return sum(map(lambda bus_stop: bus_stop[0] - bus_stop[1], bus_stops))


import numpy as np
def number(bus_stops):
    bus_stops = np.array(bus_stops)
    return np.sum(bus_stops[:, 0]) - np.sum(bus_stops[:, 1])


import numpy as np
def number(bus_stops):
    return np.subtract.reduce(np.sum((bus_stops), axis=0))





# Count of positives / sum of negatives
# https://www.codewars.com/kata/576bb71bbbcf0951d5000044/train/python
"""
Given an array of integers.

Return an array, where the first element is the count of positives numbers and the second element is sum of negative numbers. 0 is neither positive nor negative.

If the input is an empty array or is null, return an empty array.

Example
For input [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -11, -12, -13, -14, -15], you should return [10, -65]."""
(count_positives_sum_negatives([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -11, -12, -13, -14, -15]), [10, -65])
(count_positives_sum_negatives([0, 2, 3, 0, 5, 6, 7, 8, 9, 10, -11, -12, -13, -14]), [8, -50])
(count_positives_sum_negatives([1]), [1, 0])
(count_positives_sum_negatives([-1]), [0, -1])
(count_positives_sum_negatives([0, 0, 0, 0, 0, 0, 0, 0, 0]), [0, 0])
(count_positives_sum_negatives([]), [])


def count_positives_sum_negatives(numbers: list) -> list[int]:
    if not numbers:
        return []
    
    positive_counter = 0
    negative_sum = 0

    for number in numbers:
        if number > 0:
            positive_counter += 1
        if number < 0:
            negative_sum += number
    
    return [positive_counter, negative_sum]


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





# String repeat
# https://www.codewars.com/kata/54c27a33fb7da0db0100040e/solutions/python
"""
A square of squares
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
26  =>  false
"""
(is_square(-1), False,) #  "-1: Negative numbers cannot be square numbers"
(is_square( 0), True,) #  "0 is a square number (0 * 0)"
(is_square( 3), False,) #  "3 is not a square number"
(is_square( 4), True,) #  "4 is a square number (2 * 2)"
(is_square(25), True,) #  "25 is a square number (5 * 5)"
(is_square(26), False,) #  "26 is not a square number"


def is_square(number):
    return (
        number >= 0 and 
        not(number ** 0.5 % 1))


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





# Build a pile of Cubes
# https://www.codewars.com/kata/5592e3bd57b64d00f3000047/train/python
"""
Your task is to construct a building which will be a pile of n cubes. The cube at the bottom will have a volume of n^3, the cube above will have volume of (n-1)^3 and so on until the top which will have a volume of 1^3.

You are given the total volume m of the building. Being given m can you find the number n of cubes you will have to build?

The parameter of the function findNb (find_nb, find-nb, findNb, ...) will be an integer m and you have to return the integer n such as n^3 + (n-1)^3 + ... + 1^3 = m if such a n exists or -1 if there is no such n.

Examples:
find_nb(1071225) --> 45
find_nb(91716553919377) --> -1
"""
(find_nb(4183059834009), 2022)
(find_nb(24723578342962), -1)
(find_nb(135440716410000), 4824)
(find_nb(40539911473216), 3568)
(find_nb(26825883955641), 3218)


def find_nb(number):
    cumuliative_sum = 0  # cumulative sum
    index = 0

    while cumuliative_sum < number:  # while cumulative sum is lower than target number
        index += 1
        cumuliative_sum += index ** 3  # cumulative sum at ith position
    
    return index if cumuliative_sum == number else -1  # if there is match reutrn number of cubes





# Beginner Series #2 Clock
# https://www.codewars.com/kata/55f9bca8ecaa9eac7100004a/train/python
"""
Clock shows h hours, m minutes and s seconds after midnight.

Your task is to write a function which returns the time since midnight in milliseconds.

Example:
h = 0
m = 1
s = 1

result = 61000
Input constraints:

0 <= h <= 23
0 <= m <= 59
0 <= s <= 59
"""


(past(0,1,1), 61000)
(past(1,1,1), 3661000)
(past(0,0,0), 0)
(past(1,0,1), 3601000)
(past(1,0,0), 3600000)


def past(h, m, s):
    return ((((h * 60) + m) * 60) + s) * 1000





# Find the next perfect square!
# https://www.codewars.com/kata/56269eb78ad2e4ced1000013/solutions/python
"""
You might know some pretty large perfect squares. But what about the NEXT one?

Complete the findNextSquare method that finds the next integral perfect square after the one passed as a parameter. Recall that an integral perfect square is an integer n such that sqrt(n) is also an integer.

If the parameter is itself not a perfect square then -1 should be returned. You may assume the parameter is non-negative.

Examples:(Input --> Output)

121 --> 144
625 --> 676
114 --> -1 since 114 is not a perfect square
"""
(find_next_square(121), 144)
(find_next_square(625), 676)
(find_next_square(319225), 320356)
(find_next_square(15241383936), 15241630849)
(find_next_square(155), -1)
(find_next_square(342786627), -1)


def find_next_square(number):
    result = number ** .5 % 1
    return -1 if result else int((number ** .5 + 1) ** 2)
find_next_square(121)


# oldies
def find_next_square(sq):
    return int(np.square(np.sqrt(sq) + 1)) if np.sqrt(sq).is_integer() else -1
find_next_square(121)

def find_next_square(sq):
    root_test = (sq ** .5) % 1 == 0
    return ((sq ** .5) + 1) ** 2 if root_test else -1





# Find the unique number
# https://www.codewars.com/kata/585d7d5adb20cf33cb000235/train/python
"""
There is an array with some numbers. All numbers are equal except for one. Try to find it!

find_uniq([1, 1, 1, 2, 1, 1]) == 2
find_uniq([0, 0, 0.55, 0, 0]) == 0.55
It's guaranteed that array contains at least 3 numbers.

The tests contain some very huge arrays, so think about performance.
"""
(find_uniq([1, 1, 1, 2, 1, 1]), 2)
(find_uniq([0, 0, 0.55, 0, 0]), 0.55)
(find_uniq([3, 10, 3, 3, 3]), 10)


def find_uniq(nums):
    if nums[0] == nums[1]:
        for num in nums[2:]:
            if not num == nums[0]:
                return num
    else:
        return nums[1] if nums[0] == nums[2] else nums[0]


# oldies
def find_uniq(arr):
    a, b = set(arr)
    return a if arr.count(a) == 1 else b

def find_uniq(arr):
    return min(set(arr), key=arr.count)  # with 'set' computes much faster

def find_uniq(arr):
    return min(set(arr), key=lambda x: arr.count(x))

from collections import Counter
def find_uniq(arr):
    return min(set(arr), key=Counter(arr).get)

from collections import Counter
def find_uniq(arr):
    return Counter(arr).most_common()[-1][0]

from collections import Counter
def find_uniq(arr):
    # count_dict = dict(Counter(arr))
    count_dict = Counter(arr)
    return sorted(set(arr), key=count_dict.get)[0]





# Descending Order
# https://www.codewars.com/kata/5467e4d82edf8bbf40000155/train/python
"""
Your task is to make a function that can take any non-negative integer as an argument and return it with its digits in descending order. Essentially, rearrange the digits to create the highest possible number.

Examples:
Input: 42145 Output: 54421
Input: 145263 Output: 654321
Input: 123456789 Output: 987654321
"""
(descending_order(15), 51)
(descending_order(123456789), 987654321)
(descending_order(0), 0)


def descending_order(number):
    return int("".join(sorted(str(number), reverse=True)))





# Duplicate Encoder
# https://www.codewars.com/kata/54b42f9314d9229fd6000d9c/train/python
"""
The goal of this exercise is to convert a string to a new string where each character in the new string is "(" if that character appears only once in the original string, or ")" if that character appears more than once in the original string. Ignore capitalization when determining if a character is a duplicate.

Examples
"din"      =>  "((("
"recede"   =>  "()()()"
"Success"  =>  ")())())"
"(( @"     =>  "))((" 
"""
(duplicate_encode("din"),"(((")
(duplicate_encode("recede"),"()()()")
(duplicate_encode("Success"),")())())")
(duplicate_encode("(( @"),"))((")


def duplicate_encode(word):
    counter = {}
    
    for letter in word.lower():
        counter[letter] = counter.get(letter, 0) + 1

    parentheses = ""

    for letter in word.lower():
        if counter[letter] == 1:
            parentheses += "("
        else:
            parentheses += ")"
    
    return parentheses


# oldies
def duplicate_encode(word):
    word = word.lower()
    return "".join("(" if word.count(char) == 1 else ")" for char in word)


def duplicate_encode(word):
    return ''.join(map(lambda x: '(' if Counter(word.lower())[x] == 1 else ')', word.lower()))
duplicate_encode("din")


def duplicate_encode(word):
    multiple_appear = {letter: (word.lower().count(letter) > 1) for letter in set(word.lower())}
    return "".join(")" if multiple_appear[letter] else "(" for letter in word.lower())

from collections import Counter
def duplicate_encode(word):
    return ''.join('(' if Counter(word.lower())[char] == 1 else ')' for char in word.lower())





# Counting sheep...
# https://www.codewars.com/kata/54edbc7200b811e956000556/train/python
"""
Consider an array/list of sheep where some sheep may be missing from their place. We need a function that counts the number of sheep present in the array (true means present).

For example,

[True,  True,  True,  False,
  True,  True,  True,  True ,
  True,  False, True,  False,
  True,  False, False, True ,
  True,  True,  True,  True ,
  False, False, True,  True]
The correct answer would be 17.

Hint: Don't forget to check for bad values like null/undefined
"""
(count_sheeps([None,  True,  True,  False]), 2)
(count_sheeps([]), 0)


def count_sheeps(sheep):
    return sum(filter(bool, sheep))


# oldies
import numpy as np
# changed first two True's to something more instructive
array1 = [None,  np.NaN,  True,  False]

def count_sheeps(sheep):
    return sum(filter(lambda x: x, sheep))
    # return sum(filter(lambda x: x == True, sheep))
    # return sheep.count(True)
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





# Highest and Lowest
# https://www.codewars.com/kata/554b4ac871d6813a03000035
"""
In this little assignment you are given a string of space separated numbers, and have to return the highest and lowest number.

Examples
high_and_low("1 2 3 4 5")  # return "5 1"
high_and_low("1 2 -3 4 5") # return "5 -3"
high_and_low("1 9 3 4 -5") # return "9 -5"
Notes
All numbers are valid Int32, no need to validate them.
There will always be at least one number in the input string.
Output string must be two numbers separated by a single space, and highest number is first."""
(high_and_low("8 3 -5 42 -1 0 0 -9 4 7 4 -4"), "42 -9")
(high_and_low("1 2 3"), "3 1")


def high_and_low(squence):
    highest = float("-inf")
    lowest = float("inf")

    for element in squence.split():
        number = int(element)
        
        if number > highest:
            highest = number

        if number < lowest:
            lowest = number
    
    return f"{highest} {lowest}"


def high_and_low(numbers):
    num_list = [int(number) for number in numbers.split()]
    return f"{max(num_list)} {min(num_list)}"


# oldies
def high_and_low(numbers):
    return max(numbers.split(), key=int) + " " + min(numbers.split(), key=int)


sorted(('8 3 -5 42 -1 0 0 -9 4 7 4 -4').split(), key=int)
max(('8 3 -5 42 -1 0 0 -9 4 7 4 -4').split(), key=int)





# Your order, please
# https://www.codewars.com/kata/55c45be3b2079eccff00010f/solutions/python
"""
Your task is to sort a given string. Each word in the string will contain a single number. This number is the position the word should have in the result.

Note: Numbers can be from 1 to 9. So 1 will be the first word (not 0).

If the input string is empty, return an empty string. The words in the input String will only contain valid consecutive numbers.

Examples
"is2 Thi1s T4est 3a"  -->  "Thi1s is2 3a T4est"
"4of Fo1r pe6ople g3ood th5e the2"  -->  "Fo1r the2 g3ood 4of th5e pe6ople"
""  -->  ""
"""
(order("is2 Thi1s T4est 3a"), "Thi1s is2 3a T4est")
(order("4of Fo1r pe6ople g3ood th5e the2"), "Fo1r the2 g3ood 4of th5e pe6ople")
(order(""), "")


def extract_digits(word):
    return "".join(filter(str.isdigit, word))
    # return "".join(char for char in word if char.isdigit())

def order(sentence):
    words = sentence.split()
    words.sort(key=extract_digits)
    return " ".join(words)


# oldies
def order(sentence):
    return " ".join(sorted(sentence.split(), key=min))

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





# Century From Year
# https://www.codewars.com/kata/5a3fe3dde1ce0e8ed6000097/train/python
"""
Introduction
The first century spans from the year 1 up to and including the year 100, the second century - from the year 101 up to and including the year 200, etc.

Task
Given a year, return the century it is in.

Examples
1705 --> 18
1900 --> 19
1601 --> 17
2000 --> 20
"""
(century(1705), 18)  # 'Testing for year 1705'
(century(1900), 19)  # 'Testing for year 1900'
(century(1601), 17)  # 'Testing for year 1601'
(century(2000), 20)  # 'Testing for year 2000'
(century(356), 4)  # 'Testing for year 356'
(century(89), 1)  # 'Testing for year 89'


def century(year):
    return year // 100 + bool(year % 100)

def century(year):
    return ((year - 1) // 100) + 1


# oldies
import numpy as np
def century(year):
    return int(np.ceil(year / 100))

def century(year):
    ending = int(str(year)[-2:])
    core = int(str(year)[-4:-2]) if str(year)[-4:-2] else 0
    return core + 1 if ending else core

def century(year):
    return year // 100 + 1 if year % 100 else year // 100

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





# Reversed Strings
# https://www.codewars.com/kata/5168bb5dfe9a00b126000018/solutions/python
"""
Complete the solution so that it reverses the string passed into it.

'world'  =>  'dlrow'
'word'   =>  'drow'
"""
(solution('world'), 'dlrow')
(solution('hello'), 'olleh')
(solution(''), '')
(solution('h'), 'h')


def solution(string):
    return string[::-1]

def solution(string):
    return "".join(reversed(string))





# Calculating with Functions
# https://www.codewars.com/kata/525f3eda17c7cd9f9e000b39/python
# Fist fail, WTF
"""
This time we want to write calculations using functions and get the results. Let's have a look at some examples:

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
eight(divided_by(three()))
"""

(seven(times(five())), 35)
(one(plus(seven(times(five())))), 36)
(four(plus(nine())), 13)
(eight(minus(three())), 5)
(six(divided_by(two())), 3)


nine(minus(four()))
minus(four())(nine())
minus(four())(9)
minus(4)(9)

def zero(f=None): 
    return f(0) if f else 0
def one(f=None): 
    return f(1) if f else 1
def two(f=None): 
    return f(2) if f else 2
def three(f=None): 
    return f(3) if f else 3
def four(f=None): 
    return f(4) if f else 4
def five(f=None): 
    return f(5) if f else 5
def six(f=None): 
    return f(6) if f else 6
def seven(f=None): 
    return f(7) if f else 7
def eight(f=None): 
    return f(8) if f else 8
def nine(f=None): 
    return f(9) if f else 9


def plus(y):
    return lambda x: x + y
def minus(y):
    return lambda x: x - y
def times(y):
    return lambda x: x * y
def divided_by(y):
    return lambda x: x // y

def minus(y):
    def subtract(x):
        return x - y
    return subtract





id_ = lambda x: x
number = lambda x: lambda f=id_: f(x)
zero, one, two, three, four, five, six, seven, eight, nine = map(number, range(10))
plus = lambda x: lambda y: y + x
minus = lambda x: lambda y: y - x
times = lambda x: lambda y: y * x
divided_by = lambda x: lambda y: y // x





# You only need one - Beginner
# https://www.codewars.com/kata/57cc975ed542d3148f00015b
"""
You will be given an array a and a value x. All you need to do is check whether the provided array contains the value.

Array can contain numbers or strings. X can be either.

Return true if the array contains the value, false if not.
"""
(check([66, 101], 66), True)
(check([101, 45, 75, 105, 99, 107], 107), True)
(check(['t', 'e', 's', 't'], 'e'), True)
(check(['what', 'a', 'great', 'kata'], 'kat'), False)


def check(sequence, element):
    return element in sequence

def check(seq, elem):
    # return any([i == elem for i in seq])
    return any([i for i in seq if i == elem])





# Count the smiley faces!
# https://www.codewars.com/kata/583203e6eb35d7980400002a
"""
Given an array (arr) as an argument complete the function countSmileys that should return the total number of smiling faces.

Rules for a smiling face:

Each smiley face must contain a valid pair of eyes. Eyes can be marked as : or ;
A smiley face can have a nose but it does not have to. Valid characters for a nose are - or ~
Every smiling face must have a smiling mouth that should be marked with either ) or D
No additional characters are allowed except for those mentioned.

Valid smiley face examples: :) :D ;-D :~)
Invalid smiley faces: ;( :> :} :]

Example
count_smileys([':)', ';(', ';}', ':-D']);       // should return 2;
count_smileys([';D', ':-(', ':-)', ';~)']);     // should return 3;
count_smileys([';]', ':[', ';*', ':$', ';-D']); // should return 1;
Note
In case of an empty array return 0. You will not be tested with invalid input (input will always be an array). Order of the face (eyes, nose, mouth) elements will always be the same.
"""
(count_smileys([':)', ';(', ';}', ':-D']), 2)
(count_smileys([';D', ':-(', ':-)', ';~)']), 3)
(count_smileys([';]', ':[', ';*', ':$', ';-D']), 1)
(count_smileys([':D', ':~)', ';~D', ':)']), 4)


import regex as re
def count_smileys(smiley_list):
    pattern = r"[:;][-~]?[\)D]"
    return sum(True for smiley in smiley_list if re.match(pattern, smiley))

import regex as re
def count_smileys(smiley_list):
    pattern = r"^[:;][-~]?[\)D]$"
    return sum(True for smiley in smiley_list if re.search(pattern, smiley))

import re
def count_smileys(arr):
    return len(re.findall(r'[;:][~-]?[\)D]', ' '.join(arr)))


# oldies
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





# Consecutive strings
# https://www.codewars.com/kata/56a5d994ac971f1ac500003e
"""
You are given an array(list) strarr of strings and an integer k. Your task is to return the first longest string consisting of k consecutive strings taken in the array.

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
n being the length of the string array, if n = 0 or k > n or k <= 0 return "" (return Nothing in Elm).
"""


def longest_consec(strarr, k):
    if not strarr or k <= 0:
        return ""
    longest_word = ""
    for i in range(len(strarr) + 1 - k):
        longest_word = max(longest_word, "".join(strarr[i:i+k]), key=len)
        # if len("".join(strarr[i:i+k])) > len(longest_word):
            # longest_word = "".join(strarr[i:i+k])
    return longest_word
(longest_consec(["zone", "abigail", "theta", "form", "libe", "zas"], 2), "abigailtheta")
(longest_consec(["ejjjjmmtthh", "zxxuueeg", "aanlljrrrxx", "dqqqaaabbb", "oocccffuucccjjjkkkjyyyeehh"], 1), "oocccffuucccjjjkkkjyyyeehh")
(longest_consec([], 3), "")
(longest_consec(["itvayloxrp","wkppqsztdkmvcuwvereiupccauycnjutlv","vweqilsfytihvrzlaodfixoyxvyuyvgpck"], 2), "wkppqsztdkmvcuwvereiupccauycnjutlvvweqilsfytihvrzlaodfixoyxvyuyvgpck")
(longest_consec(["wlwsasphmxx","owiaxujylentrklctozmymu","wpgozvxxiu"], 2), "wlwsasphmxxowiaxujylentrklctozmymu")
(longest_consec(["zone", "abigail", "theta", "form", "libe", "zas"], -2), "")
(longest_consec(["it","wkppv","ixoyx", "3452", "zzzzzzzzzzzz"], 3), "ixoyx3452zzzzzzzzzzzz")
(longest_consec(["it","wkppv","ixoyx", "3452", "zzzzzzzzzzzz"], 15), "")
(longest_consec(["it","wkppv","ixoyx", "3452", "zzzzzzzzzzzz"], 0), "")


def longest_consec(word_list, k):
    if k < 1:
        return ""
    
    longest_concat = ""

    for index in range(len(word_list) - k + 1):
        longest_concat = max(longest_concat, 
                             "".join(word_list[index: index + k]), 
                             key=len)
    
    return longest_concat


# oldies
def longest_consec(strarr, k):
    if (not strarr or
        k > len(strarr) or
        k <= 0):
        return ""
    return max(("".join(strarr[i : i+k]) for i in range(len(strarr) - k + 1)), key=len)

# change next to list for all strings
def longest_consec(strarr, k):
    if (not strarr or
        k > len(strarr) or
        k <= 0):
        return ''
    concat_list = [''.join(strarr[i : i+k]) for i in range(len(strarr) - k + 1)]
    return next(filter(lambda x: len(x) == len(max(concat_list, key=len)), concat_list))


# solufion for k top length words in order
def find_k_longest_words(length_list, counter, k):
    right_word_list = []

    for key in length_list:
        for word in counter[key]:
            right_word_list.append(word)
            k -= 1

            if k == 0:
                return right_word_list


def longest_consec(word_list, k):
    counter = {}
    concat_word = ""

    for word in word_list:
        if not len(word) in counter:
            counter[len(word)] = []

        counter[len(word)].append(word)

    length_list = sorted(counter.keys(), reverse=True)

    right_word_list = find_k_longest_words(length_list, counter, k)

    for word in word_list:
        if word in right_word_list:
            concat_word += word

    return concat_word





# Build Tower
# https://www.codewars.com/kata/576757b1df89ecf5bd00073b/
"""
Build Tower
Build a pyramid-shaped tower given a positive integer number of floors. A tower block is represented with "*" character.

For example, a tower with 3 floors looks like this:

[
  "  *  ",
  " *** ", 
  "*****"
]
"""
(tower_builder(1), ['*'])
(tower_builder(2), [' * ', '***'])
(tower_builder(3), ['  *  ', ' *** ', '*****'])
"\n".join(tower_builder(3))


def tower_builder(floors):
    tower = []

    for floor in range(floors):  # 0, 1, 2 => 1, 3, 5; 2, 1, 0
        blank_prefix = (" ") * (floors - floor - 1)
        tower_floor = blank_prefix
        tower_floor += ("*") * (2*floor + 1)
        tower_floor += blank_prefix
        tower.append(tower_floor)

    return tower


# oldies
def tower_builder(n):
    return [("*" * (2*i + 1)).center(2*n - 1) for i in range(n)]

f"{'*':^5}"
def tower_builder(n):
    return [f"{('*' * (2*i + 1)):^{2*n - 1}}" for i in range(n)]

def tower_builder(n):
    return [" " * (n - i - 1) + "*" * (2*i + 1) + " " * (n - i - 1) for i in range(n)]

def tower_builder(n):
    return ["{:^{}}".format("*" * (2*i + 1), (2*n - 1)) for i in range(n)]





# Sum without highest and lowest number
# https://www.codewars.com/kata/576b93db1129fcf2200001e6
"""
Task
Sum all the numbers of a given array ( cq. list ), except the highest and the lowest element ( by value, not by index! ).

The highest or lowest element respectively is a single element at each edge, even if there are more than one with the same value.

Mind the input validation.

Example
{ 6, 2, 1, 8, 10 } => 16
{ 1, 1, 11, 2, 3 } => 6
Input validation
If an empty value ( null, None, Nothing etc. ) is given instead of an array, or the given array is an empty list or a list with only 1 element, return 0.
"""
(sum_array(None), 0)
(sum_array([]), 0)
(sum_array([3]), 0)
(sum_array([-3]), 0)
(sum_array([ 3, 5]), 0)
(sum_array([-3, -5]), 0)
(sum_array([6, 2, 1, 8, 10]), 16)
(sum_array([6, 0, 1, 10, 10]), 17)
(sum_array([-6, -20, -1, -10, -12]), -28)
(sum_array([-6, 20, -1, 10, -12]), 3)


def sum_array(nums):
    if (not nums or
        len(nums) < 3
    ):
        return 0

    min_num = nums[0]
    max_num = nums[0]
    num_sum = 0

    for num in nums:
        num_sum += num
        
        if num < min_num:
            min_num = num
        elif num > max_num:
            max_num = num

    return num_sum - max_num - min_num


def sum_array(arr):
    if not arr or len(arr) <= 2:
        return 0
    arr.remove(max(arr))
    arr.remove(min(arr))
    return sum(arr)
    # return sum(arr) - min(arr) - max(arr)





# Are You Playing Banjo?
# https://www.codewars.com/kata/53af2b8861023f1d88000832
"""
Create a function which answers the question "Are you playing banjo?".
If your name starts with the letter "R" or lower case "r", you are playing banjo!

The function takes a name as its only argument, and returns one of the following strings:

name + " plays banjo" 
name + " does not play banjo"
"""
(are_you_playing_banjo("martin"), "martin does not play banjo")
(are_you_playing_banjo("Rikke"), "Rikke plays banjo")
(are_you_playing_banjo("bravo"), "bravo does not play banjo")
(are_you_playing_banjo("rolf"), "rolf plays banjo")


def are_you_playing_banjo(name):
    if name[0].upper() == "R":
        return f"{name} plays banjo"
    else:
        return f"{name} does not play banjo"

def are_you_playing_banjo(name):
    return name + " plays banjo" if name[0].lower() == "r" else name + " does not play banjo"

def are_you_playing_banjo(name):
    return '{} plays banjo'.format(name) if name[0].lower() == 'r' else '{} does not play banjo'.format(name)
are_you_playing_banjo('rartin')





# Remove the minimum
# https://www.codewars.com/kata/563cf89eb4747c5fb100001b
"""
Task
Given an array of integers, remove the smallest value. Do not mutate the original array/list. If there are multiple elements with the same value, remove the one with a lower index. If you get an empty array/list, return an empty array/list.

Don't change the order of the elements that are left.

Examples
* Input: [1,2,3,4,5], output= [2,3,4,5]
* Input: [5,3,2,1,4], output = [5,3,2,4]
* Input: [2,2,1,2,1], output = [2,2,2,1]
"""
(remove_smallest([1, 2, 3, 4, 5]), [2, 3, 4, 5])
(remove_smallest([5, 3, 2, 1, 4]), [5, 3, 2, 4])
(remove_smallest([1, 2, 3, 1, 1]), [2, 3, 1, 1])
(remove_smallest([]), [])


def remove_smallest(nums):
    if (not nums or
        len(nums) == 1
    ):
        return []
    
    nums_copy = nums.copy()
    min_num = min(nums)
    nums_copy.remove(min_num)

    return nums_copy





# Two to One
# https://www.codewars.com/kata/5656b6906de340bd1b0000ac
"""
Take 2 strings s1 and s2 including only letters from ato z. Return a new sorted string, the longest possible, containing distinct letters - each taken only once - coming from s1 or s2.

Examples:
a = "xyaabbbccccdefww"
b = "xxxxyyyyabklmopq"
longest(a, b) -> "abcdefklmopqwxy"

a = "abcdefghijklmnopqrstuvwxyz"
longest(a, a) -> "abcdefghijklmnopqrstuvwxyz"
"""
(longest("aretheyhere", "yestheyarehere"), "aehrsty")
(longest("loopingisfunbutdangerous", "lessdangerousthancoding"), "abcdefghilnoprstu")
(longest("inmanylanguages", "theresapairoffunctions"), "acefghilmnoprstuy")


def longest(word1, word2):
    return "".join(sorted(set(word1) | set(word2)))

def longest(a1, a2):
    return ''.join(sorted(set(a1 + a2)))





# Unique In Order
# https://www.codewars.com/kata/54e6533c92449cc251001667
"""
Implement the function unique_in_order which takes as argument a sequence and returns a list of items without any elements with the same value next to each other and preserving the original order of elements.

For example:

unique_in_order('AAAABBBCCDAABBB') == ['A', 'B', 'C', 'D', 'A', 'B']
unique_in_order('ABBCcAD')         == ['A', 'B', 'C', 'c', 'A', 'D']
unique_in_order([1,2,2,3,3])       == [1,2,3]
"""
(unique_in_order('AAAABBBCCDAABBB'), ['A','B','C','D','A','B'])
(unique_in_order('ABBCcAD'), ['A', 'B', 'C', 'c', 'A', 'D'])
(unique_in_order([1, 2, 2, 3, 3]), [1, 2, 3])
(unique_in_order([]), [])


def unique_in_order(iterable):
    if not iterable:
        return []

    unique_in_order_list = [iterable[0]]
    
    for element in iterable[1:]:
        if element != unique_in_order_list[-1]:
            unique_in_order_list.append(element)

    return unique_in_order_list


# oldies
def unique_in_order(iterable):
    if not iterable:
        return []
    unique_list = [iterable[i] for i in range(len(iterable) - 1) if iterable[i] != iterable[i+1]]
    unique_list.append(iterable[-1])
    return unique_list

def unique_in_order(iterable):
    if not iterable:
        return []
    unique_list = [iterable[i] for i in range(1, len(iterable)) if iterable[i] != iterable[i - 1]]
    unique_list.insert(0, iterable[0])
    return unique_list

from itertools import groupby
def unique_in_order(iterable):
    return [k for k, _ in groupby(iterable)]





# Bouncing Balls
# https://www.codewars.com/kata/5544c7a5cb454edb3c000047
"""
A child is playing with a ball on the nth floor of a tall building. The height of this floor, h, is known.

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

(Condition 2) not fulfilled).
"""
(bouncing_ball(2, 1, 1), -1)
(bouncing_ball(2, 0.5, 1), 1)
(bouncing_ball(3, 0.66, 1.5), 3)
(bouncing_ball(30, 0.66, 1.5), 15)
(bouncing_ball(30, 0.75, 1.5), 21)


def bouncing_ball(height, bounce, window):
    if (height < 0 or
        bounce <= 0 or
        bounce >= 1 or
        window >= height
    ):
        return -1
    
    counter = 1

    while height * bounce > window:
        counter += 2
        height *= bounce

    return counter





# Sum of positive
# https://www.codewars.com/kata/5715eaedb436cf5606000381
"""
You get an array of numbers, return the sum of all of the positives ones.

Example [1,-4,7,12] => 1 + 7 + 12 = 20

Note: if there is nothing to sum, the sum is default to 0.
"""
(positive_sum([1, -4, 7, 12]), 20)
(positive_sum([1, 2, 3, 4, 5]), 15)
(positive_sum([1, -2, 3, 4, 5]), 13)
(positive_sum([-1, 2, 3, 4, -5]), 9)
(positive_sum([]), 0)


def positive_sum(nums):
    positive_nums = filter(lambda num: num > 0, nums)

    return sum(positive_nums)


def positive_sum(nums):
    return sum(filter(lambda num: num > 0, nums))


def positive_sum(nums):
    return sum(num for num in nums if num > 0)





# Beginner - Lost Without a Map
# https://www.codewars.com/kata/57f781872e3d8ca2a000007e
"""
Given an array of integers, return a new array with each value doubled.

For example:

[1, 2, 3] --> [2, 4, 6]
"""
(maps([1, 2, 3]), [2, 4, 6])
(maps([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), [0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
(maps([]), [])


def maps(nums):
    return [num * 2 for num in nums]

def maps(nums):
    return list(map(lambda num: num * 2, nums))

import numpy as np
def maps(nums):
    return list(2 * np.array(nums))





# String ends with?
# https://www.codewars.com/kata/51f2d1cafc9c0f745c00037d/
"""
Complete the solution so that it returns true if the first argument(string) passed in ends with the 2nd argument (also a string).

Examples:

solution('abc', 'bc') # returns true
solution('abc', 'd') # returns false
"""
(solution('abcde', 'cde'), True)
(solution('abcde', 'abc'), False)
(solution('abcde', ''), True)


def solution(text, ending):
    return text.endswith(ending)

def solution(text, ending):
    return (
        not ending or
        text[-len(ending):] == ending
    )

def solution(text, ending):
    if not ending:
        return True
    return ending == text[-len(ending):]





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
1,2,9 -> false 
"""
(is_triangle(1, 2, 2), True)
(is_triangle(7, 2, 2), False)
(is_triangle(1, 2, 3), False)
(is_triangle(1, 3, 2), False)
(is_triangle(3, 1, 2), False)
(is_triangle(5, 1, 2), False)
(is_triangle(1, 2, 5), False)
(is_triangle(2, 5, 1), False)
(is_triangle(4, 2, 3), True)
(is_triangle(5, 1, 5), True)
(is_triangle(2, 2, 2), True)


def is_triangle(a, b, c):
    return (
        a < b + c and
        b < a + c and
        c < a + b
    )


# oldies
def is_triangle(a, b, c):
    a, b, c = sorted((a, b, c))
    return a + b > c

def is_triangle(a, b, c):
    return 2 * max((a, b, c)) < sum((a, b, c))

def is_triangle(a, b, c):
    return (a < b + c) and (b < a + c) and (c < a + b)





# Replace With Alphabet Position
# https://www.codewars.com/kata/546f922b54af40e1e90001da/train/python
"""
In this kata you are required to, given a string, replace every letter with its position in the alphabet.

If anything in the text isn't a letter, ignore it and don't return it.

"a" = 1, "b" = 2, etc.

Example
alphabet_position("The sunset sets at twelve o' clock.")
Should return "20 8 5 19 21 14 19 5 20 19 5 20 19 1 20 20 23 5 12 22 5 15 3 12 15 3 11" ( as a string )
"""
(alphabet_position("The sunset sets at twelve o' clock."), "20 8 5 19 21 14 19 5 20 19 5 20 19 1 20 20 23 5 12 22 5 15 3 12 15 3 11")
(alphabet_position("The narwhal bacons at midnight."), "20 8 5 14 1 18 23 8 1 12 2 1 3 15 14 19 1 20 13 9 4 14 9 7 8 20")
(alphabet_position(""), "")
(alphabet_position(',6?->(?)'), "")


def alphabet_position(text):
    filtered_text = filter(lambda letter: letter.isalpha(), text)
    position_list = [str(ord(letter.lower()) - ord("a") + 1) for letter in filtered_text]
    return " ".join(position_list)


# oldies
def alphabet_position(text):
    position_list = [ord(letter.lower()) - ord("a") + 1 for letter in text if letter.isalpha()]
    return position_list

def alphabet_position(text):
    # return ' '.join(str(ord(i) - 96) for i in text.lower() if i.isalpha())
    return ' '.join(str(ord(i) - ord("a") + 1) for i in text.lower() if i.isalpha())

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





# Mexican Wave
# https://www.codewars.com/kata/58f5c63f1e26ecda7e000029
"""
Task
In this simple Kata your task is to create a function that turns a string into a Mexican Wave. You will be passed a string and you must return that string in an array where an uppercase letter is a person standing up. 
Rules
 1.  The input string will always be lower case but maybe empty.
 2.  If the character in the string is whitespace then pass over it as if it was an empty seat
Example
wave("hello") => ["Hello", "hEllo", "heLlo", "helLo", "hellO"]
"""
(wave('hello'), ["Hello", "hEllo", "heLlo", "helLo", "hellO"])
(wave('codewars'), ["Codewars", "cOdewars", "coDewars", "codEwars", "codeWars", "codewArs", "codewaRs", "codewarS"])
(wave(' gap'), [' Gap', ' gAp', ' gaP'])


def wave(word):
    mexican_wave = []

    for index in range(len(word)):
        if word[index].isalpha():
            mexican_wave.append(
                word[:index] + 
                word[index].upper() + 
                word[index + 1:]
            )

    return mexican_wave


# oldies
def wave(people):
    return [people[:i] + people[i].upper() + people[i+1:] for i in range(len(people)) if people[i].isalpha()]





# Find the first non-consecutive number
# https://www.codewars.com/kata/58f8a3a27a5c28d92e000144/
"""Your task is to find the first element of an array that is not consecutive.

By not consecutive we mean not exactly 1 larger than the previous element of the array.

E.g. If we have an array [1,2,3,4,6,7,8] then 1 then 2 then 3 then 4 are all consecutive but 6 is not, so that's the first non-consecutive number.

If the whole array is consecutive then return null2.

The array will always have at least 2 elements1 and all elements will be numbers. The numbers will also all be unique and in ascending order. The numbers could be positive or negative and the first non-consecutive could be either too!

"""


(first_non_consecutive([1, 2, 3, 4, 6, 7, 8]), 6)
(first_non_consecutive([1, 2, 3, 4, 5, 6, 7, 8]), None)
(first_non_consecutive([4, 6, 7, 8, 9, 11]), 6)
(first_non_consecutive([4, 5, 6, 7, 8, 9, 11]), 11)
(first_non_consecutive([31, 32]), None)
(first_non_consecutive([-3, -2, 0, 1]), 0)
(first_non_consecutive([-5, -4, -3, -1]), -1)


def first_non_consecutive(nums):
    for index in range(1, len(nums)):
        if nums[index] != nums[index - 1] + 1:
            return nums[index]

    return None




# Jaden Casing Strings
# https://www.codewars.com/kata/5390bac347d09b7da40006f6
"""
Your task is to convert strings to how they would be written by Jaden Smith. The strings are actual quotes from Jaden Smith, but they are not capitalized in the same way he originally typed them.

Example:

Not Jaden-Cased: "How can mirrors be real if our eyes aren't real"
Jaden-Cased:     "How Can Mirrors Be Real If Our Eyes Aren't Real"
"""
(to_jaden_case("How can mirrors be real if our eyes aren't real"), "How Can Mirrors Be Real If Our Eyes Aren't Real")


def to_jaden_case(sentence):
    word_list = sentence.split()
    cap_word_list = [word.capitalize() for word in word_list]
        
    return " ".join(cap_word_list)


# oldies
def to_jaden_case(string_text):
    return ' '.join(i.capitalize() for i in string_text.split())

def to_jaden_case(string_text):
    return ' '.join(map(str.capitalize, string_text.split()))
    # return ' '.join(map(lambda x: x.title(), string_text.split()))
to_jaden_case("How can mirrors be real if our eyes aren't real")

import string
def to_jaden_case(string_text):
    return string.capwords(string_text)





# Find the odd int
# https://www.codewars.com/kata/54da5a58ea159efa38000836
"""
Given an array of integers, find the one that appears an odd number of times.

There will always be only one integer that appears an odd number of times.

Examples
[7] should return 7, because it occurs 1 time (which is odd).
[0] should return 0, because it occurs 1 time (which is odd).
[1,1,2] should return 2, because it occurs 1 time (which is odd).
[0,1,0,1,0] should return 0, because it occurs 3 times (which is odd).
[1,2,2,3,3,3,4,3,3,3,2,2,1] should return 4, because it appears 1 time (which is odd).
"""
(find_it([1, 2, 2, 3, 3, 3, 4, 3, 3, 3, 2, 2, 1]), 4)
(find_it([20, 1, -1, 2, -2, 3, 3, 5, 5, 1, 2, 4, 20, 4, -1, -2, 5]), 5)
(find_it([1, 1, 2, -2, 5, 2, 4, 4, -1, -2, 5]), -1)
(find_it([20, 1, 1, 2, 2, 3, 3, 5, 5, 4, 20, 4, 5]), 5)
(find_it([10]), 10)
(find_it([10, 10, 10]), 10)
(find_it([1, 1, 1, 1, 1, 1, 10, 1, 1, 1, 1]), 10)
(find_it([5, 4, 3, 2, 1, 5, 4, 3, 2, 10, 10]), 1)


def find_it(nums):
    counter = {}

    for num in nums:
        counter[num] = counter.get(num, 0) + 1

    for key in counter:
        if counter[key] % 2:
            return key
    
    return None


# oldies
def find_it(seq):
    return [number for number in set(seq) if seq.count(number) % 2][0]


def find_it(seq):
    for number in set(seq):
        if seq.count(number) % 2:
            return number
    return None

from collections import Counter
def find_it(seq):
    return [k for k, v in Counter(seq).items() if v % 2][0]

from collections import Counter
def find_it(seq):
    for k, v in Counter(seq).items():
        if v % 2:
            return k
    return None





# Highest Scoring Word
# https://www.codewars.com/kata/57eb8fcdf670e99d9b000272
"""
Given a string of words, you need to find the highest scoring word.
Each letter of a word scores points according to its position in the alphabet: a = 1, b = 2, c = 3 etc.
You need to return the highest scoring word as a string.
If two words score the same, return the word that appears earliest in the original string.
All letters will be lowercase and all inputs will be valid.

(high('take me to semynak'), 'semynak')
"""
(high('man i need a taxi up to ubud'), 'taxi')
(high('what time are we climbing up the volcano'), 'volcano')
(high('take me to semynak'), 'semynak')
(high('aa b'), 'aa')
(high('b aa'), 'b')
(high('bb d'), 'bb')
(high('d bb'), 'd')
(high("aaa b"), "aaa")


def count_word_score(word):
    return sum(ord(letter) - ord("a") + 1 for letter in word)

def high(text):
    return max(text.split(), key=count_word_score)


def high(text):
    top_word = ""
    top_word_score = 0

    for word in text.split():
        if count_word_score(word) > top_word_score:
            top_word = word
            top_word_score = count_word_score(word)

    return top_word


# oldies
def high(x):
    word_list = [sum(ord(j) - 96 for j in i) for i in x.split()]
    return x.split()[word_list.index(max(word_list))]
high('man i need a taxi up to ubud')

def high(x):
    word_list = [sum(ord(letter) - ord("a") + 1 for letter in word) for word in x.split()]
    return x.split()[word_list.index(max(word_list))]

# slick
def high(x):
    return max(x.split(), key=lambda word: sum(ord(letter) - 96 for letter in word))






# Reverse words
# https://www.codewars.com/kata/5259b20d6021e9e14c0010d4
"""
Complete the function that accepts a string parameter, and reverses each word in the string. All spaces in the string should be retained.

Examples
"This is an example!" ==> "sihT si na !elpmaxe"
"double  spaces"      ==> "elbuod  secaps"
"""
(reverse_words('The quick brown fox jumps over the lazy dog.'), 'ehT kciuq nworb xof spmuj revo eht yzal .god')
(reverse_words('apple'), 'elppa')
(reverse_words('a b c d'), 'a b c d')
(reverse_words('double  spaced  words'), 'elbuod  decaps  sdrow')


def reverse_words(text):
    return " ".join(word[::-1] for word in text.split(" "))


# oldies
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





# Vowel Count
# https://www.codewars.com/kata/54ff3102c1bad923760001f3
"""
Return the number (count) of vowels in the given string.

We will consider a, e, i, o, u as vowels for this Kata (but not y).

The input string will only consist of lower case letters and/or spaces.
"""
(get_count("abracadabra"), 5)


def is_vowel(letter):
    return letter in "aeoiu"

def get_count(word):
    return sum(map(is_vowel, word))
    # return sum(True for letter in word if is_vowel(letter))
    # return sum(is_vowel(letter) for letter in word)
    # return len(list(filter(is_vowel, word)))
    # return sum(map(lambda letter: is_vowel(letter), word))

import re
def get_count(word):
    return len(re.findall(r"[aeiou]", word))





# Is n divisible by x and y?
# https://www.codewars.com/kata/5545f109004975ea66000086/train
"""
Create a function that checks if a number n is divisible by two numbers x AND y. All inputs are positive, non-zero numbers.

Examples:
1) n =   3, x = 1, y = 3 =>  true because   3 is divisible by 1 and 3
2) n =  12, x = 2, y = 6 =>  true because  12 is divisible by 2 and 6
3) n = 100, x = 5, y = 3 => false because 100 is not divisible by 3
4) n =  12, x = 7, y = 5 => false because  12 is neither divisible by 7 nor 5
"""
(is_divisible(8, 2, 4), True)
(is_divisible(12, -3, 4), True)
(is_divisible(8, 3, 4), False)
(is_divisible(48, 2, -5), False)
(is_divisible(-100, -25, 10), True)
(is_divisible(10000, 5, -3), False)
(is_divisible(4, 4, 2), True)
(is_divisible(5, 2, 3), False)
(is_divisible(-96, 25, 17), False)
(is_divisible(33, 1, 33), True)

def is_divisible(num, divider_1, divider_2):
    return not num % divider_1 and not num % divider_2
    # return not (num % divider_1 or num % divider_2)





# Poker cards encoder/decoder
# https://www.codewars.com/kata/52ebe4608567ade7d700044a/
"""
Consider a deck of 52 cards, which are represented by a string containing their suit and face value.

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
12: K      25: K      38: K      51: K
"""
(encode(['Ac', 'Ks', '5h', 'Td', '3c']), [0, 2, 22, 30, 51])
(encode(["Td", "8c", "Ks"]), [7, 22, 51])
(encode(["Qh", "5h", "Ad"]), [13, 30, 37])
(encode(["8c", "Ks", "Td"]), [7, 22, 51])
(encode(["Qh", "Ad", "5h"]), [13, 30, 37])
(encode(["5h", "7c", "Qh", "Qs", "Ad"]), [6, 13, 30, 37, 50])

(decode([0, 51, 30, 22, 2]), ['Ac', '3c', 'Td', '5h', 'Ks'])
(decode([7, 22, 51]), ["8c", "Td", "Ks"])
(decode([13, 30, 37]), ["Ad", "5h", "Qh"])
(decode([7, 51, 22]), ["8c", "Td", "Ks"])
(decode([13, 37, 30]), ["Ad", "5h", "Qh"])


suits = "cdhs"
figures = "A23456789TJQK"

def encode(cards):
    nums = [suits.index(card[1]) * 13 + figures.index(card[0])
            for card in cards]
    nums.sort()

    return nums

def decode(cards):
    cards.sort()

    return [figures[card % 13] + suits[card // 13]
            for card in cards]


suits = "cdhs"
figures = "A23456789TJQK"

def encode(cards):
    num_map =  map(lambda card: suits.index(card[1]) * 13 + figures.index(card[0]), cards)
    num_list = list(num_map)
    num_list.sort()
    return num_list

def decode(cards):
    cards.sort()
    return list(map(lambda card: figures[card % 13] + suits[card // 13], cards))


# oldies
encoding_dict = {
    'Ac': 0, 'Ad': 13, 'Ah': 26, 'As': 39,
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
    'Kc': 12, 'Kd': 25, 'Kh': 38, 'Ks': 51
}

decoding_dict = {v: k for k, v in encoding_dict.items()}
decoding_dict = {
    0: 'Ac', 13: 'Ad', 26: 'Ah', 39: 'As',
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
    12: 'Kc', 25: 'Kd', 38: 'Kh', 51: 'Ks'
}

def encode(cards):
    return sorted(encoding_dict[i] for i in cards)
encode(["Td", "8c", "Ks"])

def decode(cards):
    return [decoding_dict[i] for i in sorted(cards)]
decode([7, 22, 51])    





# Simple Events !!!
# https://www.codewars.com/kata/52d3b68215be7c2d5300022f/
"""
Your goal is to write an Event constructor function, which can be used to make event objects.

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
Also see an example test fixture for suggested usage
"""
event = Event()    
event.subscribe(f)
event.emit(1, 'foo', True)
event.emit(2, 'bar', False)
event.unsubscribe(f)
event.emit(2)


class Event():
    def __init__(self):
        self.handlers = set()
        
    def subscribe(self, func):
        self.handlers.add(func)
    
    def unsubscribe(self, func):
        self.handlers.discard(func)
        
    def emit(self, *args, **kwargs):
        for handler in self.handlers:
            handler(*args, **kwargs)



    

# Mongodb ObjectID
# https://www.codewars.com/kata/52fefe6cb0091856db00030e/
"""
MongoDB is a noSQL database which uses the concept of a document, rather than a table as in SQL. Its popularity is growing.

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
When you will implement this correctly, you will not only get some points, but also would be able to check creation time of all the kata here :-)
"""
(is_valid('507f1f77bcf86cd799439011'), True)
(is_valid('507f1f77bcf86cz799439011'), False)
(is_valid('111111111111111111111111'), True)
(is_valid(111111111111111111111111), False)
(is_valid('507f1f77bcf86cD799439011'), False)
(is_valid(False), False)
(is_valid([]), False)
(is_valid(1234), False)
(is_valid('123476sd'), False)
(is_valid('507f1f77bcf86cd79943901'), False)
(is_valid('507f1f77bcf86cd799439016'), True)


def is_valid(id):
    return (
        type(id) == str and
        len(id) == 24 and
        len(list(filter(lambda char: char in "0123456789abcdef", id))) == 24
    )


from datetime import datetime
(get_timestamp('507f1f77bcf86cd799439011'), datetime(2012, 10, 17, 23, 13, 27))  # Wed Oct 17 2012 21:13:27 GMT-0700 (Pacific Daylight Time)
(get_timestamp('507f1f77bcf86cz799439011'), False)
(get_timestamp('507f1f77bcf86cd79943901'), False)
(get_timestamp('111111111111111111111111'), datetime(1979, 1, 28, 1, 25, 53))  # Sun Jan 28 1979 00:25:53 GMT-0800 (Pacific Standard Time)
(get_timestamp(111111111111111111111111), False)
(get_timestamp(False), False)
(get_timestamp([]), False)
(get_timestamp(1234), False)
(get_timestamp('123476sd'), False)
(get_timestamp('507f1f77bcf86cd79943901'), False)
(get_timestamp('507f1f77bcf86cd799439016'), datetime(2012, 10, 17, 21, 13, 27))


def get_timestamp(id):
    return (
        is_valid(id) and 
        datetime.fromtimestamp(int(id[:8], 16))
    )


from datetime import datetime

class Mongo(object):
    @classmethod
    def is_valid(cls, id):
        return (
            type(id) == str and
            len(id) == 24 and
            len(list(filter(lambda char: char in "0123456789abcdef", id))) == 24
        )
    
    @classmethod
    def get_timestamp(cls, id):
        return (
            cls.is_valid(id) and
            datetime.fromtimestamp(int(id[:8], 16))
        )
        
        
(Mongo.is_valid(False), False)
(Mongo.is_valid([]), False)
(Mongo.is_valid(1234), False)
(Mongo.is_valid('123476sd'), False)
(Mongo.is_valid('507f1f77bcf86cd79943901'), False)
(Mongo.is_valid('507f1f77bcf86cd799439016'), True)

(Mongo.get_timestamp(False), False)
(Mongo.get_timestamp([]), False)
(Mongo.get_timestamp(1234), False)
(Mongo.get_timestamp('123476sd'), False)
(Mongo.get_timestamp('507f1f77bcf86cd79943901'), False)
(Mongo.get_timestamp('507f1f77bcf86cd799439016'), datetime(2012, 10, 17, 21, 13, 27))


# oldies
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





# Prime number decompositions
# https://www.codewars.com/kata/53c93982689f84e321000d62
"""
You have to code a function getAllPrimeFactors, which takes an integer as parameter and returns an array containing its prime decomposition by ascending factors. If a factor appears multiple times in the decomposition, it should appear as many times in the array.

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
The result for n=2 is normal. The result for n=1 is arbitrary and has been chosen to return a usefull result. The result for n=0 is also arbitrary but can not be chosen to be both usefull and intuitive. ([[0],[0]] would be meaningfull but wont work for general use of decomposition, [[0],[1]] would work but is not intuitive.)
"""
(getAllPrimeFactors(100), [2, 2, 5, 5])
(getUniquePrimeFactorsWithCount(100), [[2, 5], [2, 2]])
(getUniquePrimeFactorsWithProducts(100), [4, 25])


def getAllPrimeFactors(num):
    if (not (type(num) == int) or
        num < 1):
        return []
    if num == 1:
        return [1]
    
    dividers = []

    while num != 1:
        for divider in range(2, num + 1):
            if not num % divider:
                dividers.append(divider)
                num //= divider
                break

    return dividers


def getUniquePrimeFactorsWithCount(nums):
    all_prime_factors = getAllPrimeFactors(nums)
    counter = {}

    for prime in all_prime_factors:
        counter[prime] = counter.get(prime, 0) + 1

    return [list(counter.keys()), list(counter.values())]


def getUniquePrimeFactorsWithProducts(nums):
    unique_prime_factors = getUniquePrimeFactorsWithCount(nums)
    
    return [nums[0] ** nums[1] for nums in zip(*unique_prime_factors)]


# oldies
from collections import Counter
def getUniquePrimeFactorsWithCount(n):
    factors = Counter(getAllPrimeFactors(n))
    return list(zip(*factors.items()))
(getUniquePrimeFactorsWithCount(100), [[2, 5], [2, 2]])
getUniquePrimeFactorsWithCount(1)
getUniquePrimeFactorsWithCount(0)
getUniquePrimeFactorsWithCount(-1)

def getUniquePrimeFactorsWithCount(n):
    prime_set = getAllPrimeFactors(n)
    factors = {number: prime_set.count(number) for number in set(prime_set)}
    return list(zip(*factors.items()))
    # return [list(factors.keys()), list(factors.values())]



def getUniquePrimeFactorsWithProducts(n):
    getAll = getAllPrimeFactors(n)
    return [number ** getAll.count(number) for number in set(getAll)]
    # return [k ** v for k, v in Counter(getAllPrimeFactors(n)).items()]
(getUniquePrimeFactorsWithProducts(100), [4, 25])
getUniquePrimeFactorsWithProducts(1)
getUniquePrimeFactorsWithProducts(0)
getUniquePrimeFactorsWithProducts(-1)

def getUniquePrimeFactorsWithProducts(n):
    factors = getUniquePrimeFactorsWithCount(n)
    return [a**b for a, b in zip(*factors)]
    return [a**b for a, b in zip(factors[0], factors[1])]

def getUniquePrimeFactorsWithProducts(n):
    power1 = getUniquePrimeFactorsWithCount(n)
    # return list(map(lambda x, y: x**y, power1[0], power1[1]))
    return list(map(pow, power1[0], power1[1]))

import numpy as np
def getUniquePrimeFactorsWithProducts(n):
    return list(np.power(*getUniquePrimeFactorsWithCount(n)))





# Strings Mix
# https://www.codewars.com/kata/5629db57620258aa9d000014
"""
Given two strings s1 and s2, we want to visualize how different the two strings are. 
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
u
s1="Are the kids at home? aaaaa fffff"
s2="Yes they are here! aaaaa fffff"
mix(s1, s2) --> "=:aaaaaa/2:eeeee/=:fffff/1:tt/2:rr/=:hh"
"""
(mix("my&friend&Paul has heavy hats! &", "my friend John has many many friends &"), "2:nnnnn/1:aaaa/1:hhh/2:mmm/2:yyy/2:dd/2:ff/2:ii/2:rr/=:ee/=:ss")
(mix("Are they here", "yes, they are here"), "2:eeeee/2:yy/=:hh/=:rr")
(mix("Sadus:cpms>orqn3zecwGvnznSgacs", "MynwdKizfd$lvse+gnbaGydxyXzayp"), '2:yyyy/1:ccc/1:nnn/1:sss/2:ddd/=:aa/=:zz')
(mix("looping is fun but dangerous", "less dangerous than coding"), "1:ooo/1:uuu/2:sss/=:nnn/1:ii/2:aa/2:dd/2:ee/=:gg")
(mix(" In many languages", " there's a pair of functions"), "1:aaa/1:nnn/1:gg/2:ee/2:ff/2:ii/2:oo/2:rr/2:ss/2:tt")
(mix("Lords of the Fallen", "gamekult"), "1:ee/1:ll/1:oo")
(mix("codewars", "codewars"), "")
(mix("A generation must confront the looming ", "codewarrs"), "1:nnnnn/1:ooooo/1:tttt/1:eee/1:gg/1:ii/1:mm/=:rr")


def mix(s1, s2):
    counter1 = {}
    counter2 = {}
    max_counter = {}
    mix_list = []

    for char in s1:
        if char.islower():
        # if re.match(r"[a-z]", char):
            counter1[char] = counter1.get(char, 0) + 1

    for char in s2:
        if char.islower():
            counter2[char] = counter2.get(char, 0) + 1

    for key, val in counter1.items():
        max_counter[key] = max(max_counter.get(key, 0), val)

    for key, val in counter2.items():
        max_counter[key] = max(max_counter.get(key, 0), val)

    for key, val in max_counter.items():
        if val > 1:
            if counter1.get(key, 0) == counter2.get(key, 0):
                mix_list.append(f"=:{key*val}") 
            elif counter1.get(key, 0) > counter2.get(key, 0):
                mix_list.append(f"1:{key*val}") 
            else:
                mix_list.append(f"2:{key*val}") 

    mix_list.sort(key=lambda x: (-len(x), x[0], x[2]))

    return "/".join(mix_list)


# oldies
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





# Tribonacci Sequence
# https://www.codewars.com/kata/556deca17c58da83c00002db
"""
Well met with Fibonacci bigger brother, AKA Tribonacci.

As the name may already reveal, it works basically like a Fibonacci, but summing the last 3 (instead of 2) numbers of the sequence to generate the next. And, worse part of it, regrettably I won't get to hear non-native Italian speakers trying to pronounce it :(

So, if we are to start our Tribonacci sequence with [1, 1, 1] as a starting input (AKA signature), we have this sequence:

[1, 1 ,1, 3, 5, 9, 17, 31, ...]
But what if we started with [0, 0, 1] as a signature? As starting with [0, 1] instead of [1, 1] basically shifts the common Fibonacci sequence by once place, you may be tempted to think that we would get the same sequence shifted by 2 places, but that is not the case and we would get:

[0, 0, 1, 1, 2, 4, 7, 13, 24, ...]
Well, you may have guessed it by now, but to be clear: you need to create a fibonacci function that given a signature array/list, returns the first n elements - signature included of the so seeded sequence.
"""
(tribonacci([1, 1, 1], 10), [1, 1, 1, 3, 5, 9, 17, 31, 57, 105])
(tribonacci([0, 0, 1], 10), [0, 0, 1, 1, 2, 4, 7, 13, 24, 44])
(tribonacci([0, 1, 1], 10), [0, 1, 1, 2, 4, 7, 13, 24, 44, 81])
(tribonacci([1, 0, 0], 10), [1, 0, 0, 1, 1, 2, 4, 7, 13, 24])
(tribonacci([0, 0, 0], 10), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
(tribonacci([1, 2, 3], 10), [1, 2, 3, 6, 11, 20, 37, 68, 125, 230])
(tribonacci([3, 2, 1], 10), [3, 2, 1, 6, 9, 16, 31, 56, 103, 190])
(tribonacci([1, 1, 1], 1), [1])
(tribonacci([300, 200, 100], 0), [])
(tribonacci([0.5, 0.5, 0.5], 30), [0.5, 0.5, 0.5, 1.5, 2.5, 4.5, 8.5, 15.5, 28.5, 52.5, 96.5, 177.5, 326.5, 600.5, 1104.5, 2031.5, 3736.5, 6872.5, 12640.5, 23249.5, 42762.5, 78652.5, 144664.5, 266079.5, 489396.5, 900140.5, 1655616.5, 3045153.5, 5600910.5, 10301680.5])
(tribonacci([5], 1), [])


# dp, bottom-up, tabulation
def tribonacci(nums, counter):
    if counter < 3:
        return nums[:counter]
    
    tab = [0] * counter
    tab[0], tab[1], tab[2] = nums

    for index in range(3, counter):
        tab[index] = tab[index - 1] + tab[index - 2] + tab[index - 3]
    
    return tab

# dp, top-bottom, memoization
def tribonacci(nums, counter):
    if counter < 3:
        return nums[:counter]
    
    memo = [0] * counter
    memo[0], memo[1], memo[2] = nums

    def dfs(counter):
        if counter == 2:
            return memo[2]
        elif counter == 1:
            return memo[1]
        elif counter == 0:
            return memo[0]
        elif memo[counter]:
            return memo[counter]

        memo[counter] = dfs(counter - 1) + inner(counter -2) + inner(counter - 3)

        return memo[counter]
    
    inner(counter - 1)

    return memo


# oldies
def tribonacci(signature, n):
    a, b, c = signature
    """if not n:
        return []
    if n == 1:
        return [a]
    """
    trib_lst = []
    for _ in range(n):
        trib_lst.append(a)
        a, b, c = b, c, a + b + c 
    return trib_lst

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
def tribonacci(signature,n):
    signature = list(signature)
    while len(signature) < n:
        signature.append(sum(signature[-3:]))    
    return signature[:n]





# Simple Fun #166: Best Match
# https://www.codewars.com/kata/58b38256e51f1c2af0000081
"""
"AL-AHLY" and "Zamalek" are the best teams in Egypt, but "AL-AHLY" always wins the matches between them. "Zamalek" managers want to know what is the best match they've played so far.

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
Index of the best match.
"""
(best_match([6, 4],[1, 2]), 1)
(best_match([1],[0]), 0)
(best_match([1, 2, 3, 4, 5],[0, 1, 2, 3, 4]), 4)
(best_match([3, 4, 3],[1, 1, 2]), 2)
(best_match([4, 3, 4],[1, 1, 1]), 1)


def best_match_key(x):
    return (x[0] - x[1], -x[1])

def best_match(goals1, goals2):
    match_index = range(len(goals1))

    return min(zip(goals1, goals2, match_index), key=best_match_key)[2]


def best_match_key(goals1, goals2):
    return (goals1- goals2, -goals2)

def best_match(goals1, goals2):
    match_index = range(len(goals1))

    return min(zip(goals1, goals2, match_index), key=lambda x: best_match_key(x[0], x[1]))[2]


# oldies
def best_match(goals1, goals2):
    ran = range(len(goals1))
    diff = [goals1[i] - goals2[i] for i in ran]
    return min(zip(*(ran, diff, goals2)), key=lambda x: (x[1], -x[2]))[0]
    # return max(zip(*(ran, diff, goals2)), key=lambda x: (-x[1], x[2]))[0]

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





# Roman Numerals Decoder
# https://www.codewars.com/kata/51b6249c4612257ac0000005
"""
Create a function that takes a Roman numeral as its argument and returns its value as a numeric decimal integer. You don't need to validate the form of the Roman numeral.

Modern Roman numerals are written by expressing each decimal digit of the number to be encoded separately, starting with the leftmost digit and skipping any 0s. So 1990 is rendered "MCMXC" (1000 = M, 900 = CM, 90 = XC) and 2008 is rendered "MMVIII" (2000 = MM, 8 = VIII). The Roman numeral for 1666, "MDCLXVI", uses each letter in descending order.

Example:
e
solution('XXI'); // should return 21
Help:

Symbol    Value
I          1
V          5
X          10
L          50
C          100
D          500
M          1,000
"""
(solution('XXI'), 21)
(solution('I'), 1)
(solution('IV'), 4)
(solution('MMVIII'), 2008)
(solution('MDCLXVI'), 1666)


to_roman = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}

roman_subtract = {
    "IV": -2,  # 6 => 4
    "IX": -2,
    "XL": -20,
    "XC": -20,
    "CD": -200,
    "CM": -200
}

import re

def solution(roman):
    raw_roman = sum(to_roman[letter] for letter in roman)
    subtract = sum(roman_subtract[rs] for rs in roman_subtract if re.search(rf"{rs}", roman))
    # subtract = sum(roman_subtract[rs] for rs in roman_subtract if roman.find(rs) != -1)

    return raw_roman + subtract


roman_dict = {'I': 1,
              'V': 5,
              'X': 10,
              'L': 50,
              'C': 100,
              'D': 500,
              'M': 1000}

bool(re.search(r"DM", "MDM"))

def solution(roman):
    result = 0
    if roman.find("CM") != -1:
        result += 900
        roman = roman.replace('CM', '')
    if roman.find("CD") != -1:
        result += 400
        roman = roman.replace('CD', '')
    if roman.find("XC") != -1:
        result += 90
        roman = roman.replace('XC', '')
    if roman.find("XL") != -1:
        result += 40
        roman = roman.replace('XL', '')
    if roman.find("IX") != -1:
        result += 9
        roman = roman.replace('IX', '')
    if roman.find("IV") != -1:
        result += 4
        roman = roman.replace('IV', '')
    return result + sum(roman_dict[i] for i in roman)

(solution('XXI'), 21, 'XXI should == 21')
(solution('I'), 1, 'I should == 1')
(solution('IV'), 4, 'IV should == 4')
(solution('MMVIII'), 2008, 'MMVIII should == 2008')
(solution('MDCLXVI'), 1666, 'MDCLXVI should == 1666')


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
    'M': 1000, 'CM': -200,
    'D': 500, 'CD': -200,
    'C': 100, 'XC': -20,
    'L': 50, 'XL': -20,
    'X': 10, 'IX': -2,
    'V': 5, 'IV': -2,
    'I': 1,
}

def solution(roman):
    if not roman:
        return 0
    return sum(roman.count(k) * v for k, v in SYMBOLS.items())
solution('IV')

values = [  ('M', 1000), ('CM', -200),
            ('D', 500), ('CD', -200),
            ('C', 100), ('XC', -20),
            ('L', 50), ('XL', -20),
            ('X', 10), ('IX', -2),
            ('V', 5), ('IV', -2),
            ('I', 1)]
def solution(roman):
    return sum(roman.count(s)*v for s, v in values)



# only check if number is legit
import re
regex_pattern = r"^(?=[MDCLXVI])(M{0,3})(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$"
print(str(bool(re.match(regex_pattern, input()))))



# Friend or Foe?
# https://www.codewars.com/kata/55b42574ff091733d900002f
"""
Make a program that filters a list of strings and returns a list with only your friends name in it.

If a name has exactly 4 letters in it, you can be sure that it has to be a friend of yours! Otherwise, you can be sure he's not...

Ex: Input = ["Ryan", "Kieran", "Jason", "Yous"], Output = ["Ryan", "Yous"]

i.e.

friend ["Ryan", "Kieran", "Mark"] `shouldBe` ["Ryan", "Mark"]
Note: keep the original order of the names in the output.
"""
(friend(["Ryan", "Kieran", "Mark",]), ["Ryan", "Mark"])
(friend(["Ryan", "Jimmy", "123", "4", "Cool Man"]), ["Ryan"])
(friend(["Jimm", "Cari", "aret", "truehdnviegkwgvke", "sixtyiscooooool"]), ["Jimm", "Cari", "aret"])


def friend(name_list):
    return [name for name in name_list if len(name) == 4]
    # return list(filter(lambda name: len(name) == 4, name_list))





# A Needle in the Haystack
# https://www.codewars.com/kata/56676e8fabd2d1ff3000000c
"""
Can you find the needle in the haystack?

Write a function findNeedle() that takes an array full of junk but containing one "needle"

After your function finds the needle it should return a message (as a string) that says:

"found the needle at position " plus the index it found the needle, so:

find_needle(['hay', 'junk', 'hay', 'hay', 'moreJunk', 'needle', 'randomJunk'])
should return "found the needle at position 5" (in COBOL "found the needle at position 6")
"""
(find_needle(['3', '123124234', None, 'needle', 'world', 'hay', 2, '3', True, False]), 'found the needle at position 3')
(find_needle(['283497238987234', 'a dog', 'a cat', 'some random junk', 'a piece of hay', 'needle', 'something somebody lost a while ago']), 'found the needle at position 5')
(find_needle([1,2,3,4,5,6,7,8,8,7,5,4,3,4,5,6,67,5,5,3,3,4,2,34,234,23,4,234,324,324,'needle',1,2,3,4,5,5,6,5,4,32,3,45,54]), 'found the needle at position 30')
(find_needle(['3', '123124234', True, False]), "'needle' is not in list")


def find_needle(haystack):
    try:
        index = haystack.index("needle")
        
        return f"found the needle at position {index}"
    except ValueError as err:  #  Exception
        return f"{err}"


# oldies
def find_needle(haystack):
    # return f'found the needle at position {haystack.index("needle")}'
    # return 'found the needle at position {}'.format(haystack.index('needle'))
    # return 'found the needle at position %d' % haystack.index('needle')
    # return "found the needle at position " + str(haystack.index("needle"))





# Grasshopper - Personalized Message
# https://www.codewars.com/kata/5772da22b89313a4d50012f7
"""
Create a function that gives a personalized greeting. This function takes two parameters: name and owner.

Use conditionals to return the proper message:

case	return
name equals owner	'Hello boss'
otherwise	'Hello guest'
"""


def greet(name, owner):
    return 'Hello boss' if name == owner else 'Hello guest'
(greet('Daniel', 'Daniel'), 'Hello boss')
(greet('Greg', 'Daniel'), 'Hello guest')





# The Feast of Many Beasts
# https://www.codewars.com/kata/5aa736a455f906981800360d/
"""All of the animals are having a feast! Each animal is bringing one dish. There is just one rule: the dish must start and end with the same letters as the animal's name. For example, the great blue heron is bringing garlic naan and the chickadee is bringing chocolate cake.

Write a function feast that takes the animal's name and dish as arguments and returns true or false to indicate whether the beast is allowed to bring the dish to the feast.

Assume that beast and dish are always lowercase strings, and that each has at least two letters. beast and dish may contain hyphens and spaces, but these will not appear at the beginning or end of the string. They will not contain numerals.

"""
(feast("great blue heron", "garlic naan"), True)
(feast("chickadee", "chocolate cake"), True)
(feast("brown bear", "bear claw"), False)


def feast(beast, dish):
    return (
        beast[0] == dish[0] and
        beast[-1] == dish[-1]
    )

def feast(beast, dish):
    return beast.startswith(dish[0]) and beast.endswith(dish[-1])





# Break camelCase
# https://www.codewars.com/kata/5208f99aee097e6552000148/
"""
Complete the solution so that the function will break up camel casing, using a space between words.

Example
"camelCasing"  =>  "camel Casing"
"identifier"   =>  "identifier"
""             =>  ""
"""
(solution("helloWorld"), "hello World")
(solution("camelCase"), "camel Case")
(solution("breakCamelCase"), "break Camel Case")


def solution(text):
    camel_case = ""

    for letter in text:
        if letter.isupper():
            camel_case += " "

        camel_case += letter

    return camel_case


# oldies
def solution(s):
    return re.sub(r"([A-Z])", r" \1", s)  # import re
    # return "".join([" " + letter if letter.isupper() else letter for letter in s])
    # return "".join([" " + letter if letter in string.ascii_uppercase else letter for letter in s])  # import string

import string
def solution(s):
    for i in string.ascii_uppercase:
        s = s.replace(i, ' ' + i)
    return s





# Printer Errors
# https://www.codewars.com/kata/56541980fa08ab47a0000040
"""
In a factory a printer prints labels for boxes. For one kind of boxes the printer has to use colors which, for the sake of simplicity, are named with letters from a to m.

The colors used by the printer are recorded in a control string. For example a "good" control string would be aaabbbbhaijjjm meaning that the printer used three times color a, four times color b, one time color h then one time color a...

Sometimes there are problems: lack of colors, technical malfunction and a "bad" control string is produced e.g. aaaxbbbbyyhwawiwjjjwwm with letters not from a to m.

You have to write a function printer_error which given a string will return the error rate of the printer as a string representing a rational whose numerator is the number of errors and the denominator the length of the control string. Don't reduce this fraction to a simpler expression.

The string has a length greater or equal to one and contains only letters from ato z.

Examples:
s="aaabbbbhaijjjm"
printer_error(s) => "0/14"

s="aaaxbbbbyyhwawiwjjjwwm"
printer_error(s) => "8/22"
"""


def printer_error(s):
    return f'{sum(letter > "m" for letter in s)}/{len(s)}'
    # return f'{sum(True for letter in s if letter > "m" )}/{len(s)}'
    # return '{}/{}'.format(sum(ord(i) > 109 for i in s), len(s))
    # return '%d/%d' % (sum(i > 'm' for i in s), len(s))
(printer_error("aaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbmmmmmmmmmmmmmmmmmmmxyz"), "3/56")
(printer_error("kkkwwwaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbmmmmmmmmmmmmmmmmmmmxyz"), "6/60")
(printer_error("kkkwwwaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbmmmmmmmmmmmmmmmmmmmxyzuuuuu") , "11/65")

import re
def printer_error(s):
    return f'{len(re.findall(r"[o-z]", s))}/{len(s)}'
    return f'{len(re.sub(r"[a-m]", "", s))}/{len(s)}'    

import string
def printer_error(s):
    nominator = str(sum(i in string.ascii_lowercase[14:] for i in s))
    denominator = str(len(s))
    return  nominator + "/" + denominator





# Rock Paper Scissors!
# https://www.codewars.com/kata/5672a98bdbdd995fad00000f
"""
Rock Paper Scissors
Let's play! You have to return which player won! In case of a draw return Draw!.

Examples:

rps('scissors','paper') // Player 1 won!
rps('scissors','rock') // Player 2 won!
rps('paper','paper') // Draw!
"""
(rps('rock', 'scissors'), "Player 1 won!")
(rps('scissors', 'rock'), "Player 2 won!")
(rps('rock', 'rock'), 'Draw!')

def rps(player1, player2):
    next_winner = {"rock": "paper",
                   "paper": "scissors",
                   "scissors": "rock"
                   }
    
    if player1 == player2:
        return "Draw!"
    elif next_winner[player1] == player2:
        return "Player 2 won!"
    else:
        return "Player 1 won!"


def rps(player1, player2):
    next_winner = ("rock", "paper", "scissors")

    if player1 == player2:
        return "Draw!"
    elif (next_winner.index(player1) + 1) % 3 == next_winner.index(player2):
    # elif next_winner[(next_winner.index(player1) + 1) % 3] == player2:
        return "Player 2 won!"
    else:
        return "Player 1 won!"





# Basic Mathematical Operations
# https://www.codewars.com/kata/57356c55867b9b7a60000bd7/
"""
Your task is to create a function that does four basic mathematical operations.

The function should take three arguments - operation(string/char), value1(number), value2(number).
The function should return result of numbers after applying the chosen operation.

Examples(Operator, value1, value2) --> output
('+', 4, 7) --> 11
('-', 15, 18) --> -3
('*', 5, 5) --> 25
('/', 49, 7) --> 7
"""


def basic_op(operator, value1, value2):
    if operator == "+":
        return value1 + value2
    elif operator == "-":
        return value1 - value2
    elif operator == "*":
        return value1 * value2
    elif operator == "/":
        return value1 // value2


def basic_op(operator, value1, value2):
    return eval(f"{value1}{operator}{value2}")
    # return eval(str(value1) + operator + str(value2))
    # return eval("{}{}{}".format(value1, operator, value2))
    # return eval('value1' + operator + 'value2')
    # return {'+':a+b,'-':a-b,'*':a*b,'/':a/b}[o]
(basic_op('+', 4, 7), 11)
(basic_op('-', 15, 18), -3)
(basic_op('*', 5, 5), 25)
(basic_op('/', 49, 7), 7)





# The Supermarket Queue
# https://www.codewars.com/kata/57b06f90e298a7b53d000a86
"""
There is a queue for the self-checkout tills at the supermarket. Your task is write a function to calculate the total time required for all the customers to check out!

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

P.S. The situation in this kata can be likened to the more-computer-science-related idea of a thread pool, with relation to running multiple processes at the same time: https://en.wikipedia.org/wiki/Thread_pool
"""
(queue_time([5], 1), 5)
(queue_time([2], 5), 2)
(queue_time([1, 2, 3, 4, 5], 1), 15)
(queue_time([1, 2, 3, 4, 5], 100), 5)
(queue_time([2, 2, 3, 3, 4, 4], 2), 9)
(queue_time([], 1), 0)


def queue_time(customer_list, cashiers):
    cashier_list = [0] * cashiers

    for customer in customer_list:
        min_index = cashier_list.index(min(cashier_list))
        cashier_list[min_index] += customer

    return max(cashier_list)


# oldies
# from codewars
def queue_time(customers, n):
    chechout = [0] * n
    for customer in customers:
        min_ind = chechout.index(min(chechout))
        chechout[min_ind] += customer
    return max(chechout)

import numpy as np
def queue_time(customers, n):
    nparray = np.zeros(n, dtype=int)
    for customer in customers:
        nparray[np.where(nparray == min(nparray))[0][0]] += customer
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






# Are they the "same"?
# https://www.codewars.com/kata/550498447451fbbd7600041c
"""
Given two arrays a and b write a function comp(a, b) (orcompSame(a, b)) that checks whether the two arrays have the "same" elements, with the same multiplicities (the multiplicity of a member is the number of times it appears). "Same" means, here, that the elements in b are the elements in a squared, regardless of the order.

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
The two arrays have the same size (> 0) given as parameter in function comp.
"""
(comp([121, 144, 19, 161, 19, 144, 19, 11], [11*11, 121*121, 144*144, 19*19, 161*161, 19*19, 144*144, 19*19]), True)
(comp([121, 144, 19, 161, 19, 144, 19, 11], [11*21, 121*121, 144*144, 19*19, 161*161, 19*19, 144*144, 19*19]), False)
(comp([121, 144, 19, 161, 19, 144, 19, 11], [11*11, 121*121, 144*144, 190*190, 161*161, 19*19, 144*144, 19*19]), False)
(comp(None, []), False)


def comp(nums1, nums2):
    if (nums1 == None or nums2 == None):
        return False
    
    return {num ** 2 for num in nums1} == set(nums2)
    # return set(map(lambda num: num ** 2, nums1)) == set(nums2)





# Exes and Ohs
# https://www.codewars.com/kata/55908aad6620c066bc00002a
"""
Check to see if a string has the same amount of 'x's and 'o's. The method must return a boolean and be case insensitive. The string can contain any char.

Examples input/output:

XO("ooxx") => true
XO("xooxx") => false
XO("ooxXm") => true
XO("zpzpzpp") => true // when no 'x' and 'o' is present should return true
XO("zzoo") => false
"""
(xo("xo"), True)
(xo("xo0"), True)
(xo("ooxx"), True)
(xo("xooxx"), False)
(xo("ooxXm"), True)
(xo("zpzpzpp"), True)
(xo("zzoo"), False)


def xo(text):
    return (text.lower().count("o") == text.lower().count("x"))


import re
def xo(text):
    return len(re.findall(r"[xX]", text)) == len(re.findall(r"[oO]", text))





# Take a Number And Sum Its Digits Raised To The Consecutive Powers And ....¡Eureka!!
# https://www.codewars.com/kata/5626b561280a42ecc50000d1
"""
The number 89 is the first integer with more than one digit that fulfills the property partially introduced in the title of this kata. What's the use of saying "Eureka"? Because this sum gives the same number.

In effect: 89 = 8^1 + 9^2

The next number in having this property is 135.

See this property again: 135 = 1^1 + 3^2 + 5^3

We need a function to collect these numbers, that may receive two integers a, b that defines the range [a, b] (inclusive) and outputs a list of the sorted numbers in the range that fulfills the property described above.

Let's see some cases:

is_power_sum(1, 10) == [1, 2, 3, 4, 5, 6, 7, 8, 9]

is_power_sum(1, 100) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 89]
If there are no numbers of this kind in the range [a, b] the function should output an empty list.

is_power_sum(90, 100) == []
"""
(sum_dig_pow(1, 10), [1, 2, 3, 4, 5, 6, 7, 8, 9])
(sum_dig_pow(1, 100), [1, 2, 3, 4, 5, 6, 7, 8, 9, 89])
(sum_dig_pow(10, 89),  [89])
(sum_dig_pow(10, 100),  [89])
(sum_dig_pow(90, 100), [])
(sum_dig_pow(89, 135), [89, 135])


def sum_dig_pow(a, b):
    nums = []

    for number in range(a, b + 1):
        summed_num = sum(int(digit) ** index
                         for index, digit in enumerate(str(number), 1))

        if summed_num == number:
            nums.append(number)

    return nums


# from codewars
def is_power_sum_equal(num):
    return num == sum(int(digit) ** index
                         for index, digit in enumerate(str(num), 1))

def sum_dig_pow(a, b):
    return [num for num in range(a, b + 1) if is_power_sum_equal(num)]

def sum_dig_pow(a, b):
    return list(filter(is_power_sum_equal, range(a, b + 1)))





# Categorize New Member
# https://www.codewars.com/kata/5502c9e7b3216ec63c0001aa
"""
The Western Suburbs Croquet Club has two categories of membership, Senior and Open. They would like your help with an application form that will tell prospective members which category they will be placed.

To be a senior, a member must be at least 55 years old and have a handicap greater than 7. In this croquet club, handicaps range from -2 to +26; the better the player the lower the handicap.

Input
Input will consist of a list of pairs. Each pair contains information for a single potential member. Information consists of an integer for the person's age and an integer for the person's handicap.

Output
Output will consist of a list of string values (in Haskell and C: Open or Senior) stating whether the respective member is to be placed in the senior or open category.

Example
input =  [[18, 20], [45, 2], [61, 12], [37, 6], [21, 21], [78, 9]]
output = ["Open", "Open", "Senior", "Open", "Open", "Senior"]
"""
(open_or_senior([(45, 12),(55,21),(19, -2),(104, 20)]),['Open', 'Senior', 'Open', 'Senior'])
(open_or_senior([(16, 23),(73,1),(56, 20),(1, -1)]),['Open', 'Open', 'Senior', 'Open'])


def open_or_senior(data):
    return ['Senior' if age > 54 and handicap > 7 else 'Open'
            for age, handicap in data]





# Opposites Attract
# https://www.codewars.com/kata/555086d53eac039a2a000083
"""
Timmy & Sarah think they are in love, but around where they live, they will only know once they pick a flower each. If one of the flowers has an even number of petals and the other has an odd number of petals it means they are in love.

Write a function that will take the number of petals of each flower and return true if they are in love and false if they aren't.

"""
(lovefunc(1,4), True)
(lovefunc(2,2), False)
(lovefunc(0,1), True)
(lovefunc(0,0), False)


def lovefunc(flower1, flower2):
    return bool((flower1 + flower2) % 2)


def lovefunc(flower1, flower2):
    return bool(flower1 % 2 ^ flower2 % 2)





# Playing with digits
# https://www.codewars.com/kata/5552101f47fc5178b1000050
"""
Some numbers have funny properties. For example:

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
dig_pow(46288, 3) should return 51 since 4³ + 6⁴+ 2⁵ + 8⁶ + 8⁷ = 2360688 = 46288 * 51
"""
(dig_pow(695, 2), 2)
(dig_pow(89, 1), 1)
(dig_pow(92, 1), -1)
(dig_pow(46288, 3), 51)


def dig_pow(number, power):
    total = sum(int(digit) ** power 
               for power, digit in enumerate(str(number), power))
    
    return total // number if not total % number else -1


# oldies
def dig_pow(n, p):
    sum_of_pow = sum(int(digit) ** index for index, digit in enumerate(str(n), p)) / n
    # sum_of_pow = sum(pow(int(digit), ind) for ind, digit in enumerate(str(n), p)) / n
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






# L1: Set Alarm
# https://www.codewars.com/kata/568dcc3c7f12767a62000038
"""
Write a function named setAlarm which receives two parameters. The first parameter, employed, is true whenever you are employed and the second parameter, vacation is true whenever you are on vacation.

The function should return true if you are employed and not on vacation (because these are the circumstances under which you need to set an alarm). It should return false otherwise. Examples:

setAlarm(true, true) -> false
setAlarm(false, true) -> false
setAlarm(false, false) -> false
setAlarm(true, false) -> true
"""
(set_alarm(True, True), False, "Fails when input is True, True")
(set_alarm(False, True), False, "Fails when input is False, True")
(set_alarm(False, False), False, "Fails when input is False, False")
(set_alarm(True, False), True, "Fails when input is True, False")


def set_alarm(employed, vacation):
    return employed and not vacation





# Will there be enough space?
# https://www.codewars.com/kata/5875b200d520904a04000003
"""
The Story:
Bob is working as a bus driver. However, he has become extremely popular amongst the city's residents. With so many passengers wanting to get aboard his bus, he sometimes has to face the problem of not enough space left on the bus! He wants you to write a simple program telling him if he will be able to fit all the passengers.

Task Overview:
You have to write a function that accepts three parameters:

cap is the amount of people the bus can hold excluding the driver.
on is the number of people on the bus excluding the driver.
wait is the number of people waiting to get on to the bus excluding the driver.
If there is enough space, return 0, and if there isn't, return the number of passengers he can't take.

Usage Examples:
cap = 10, on = 5, wait = 5 --> 0 # He can fit all 5 passengers
cap = 100, on = 60, wait = 50 --> 10 # He can't fit 10 of the 50 waiting
"""
(enough(10, 5, 5), 0)
(enough(100, 60, 50), 10)
(enough(20, 5, 5), 0)


def enough(cap, wait, on):
    return max(wait - (cap - on), 0)





# Sum of the first nth term of Series
# https://www.codewars.com/kata/555eded1ad94b00403000071
"""
Task:
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
(series_sum(0), "0.00")
(series_sum(1), "1.00")
(series_sum(2), "1.25")
(series_sum(3), "1.39")


def series_sum(number):
    sequence_sum =  sum([1 / ((index * 3 + 1)) for index in range(number)])
    # sequence_sum = sum([1 / index for index in range(1, number * 3 + 1, 3)])
    return f"{sequence_sum:.2f}"
    return "{:.2f}".format(sequence_sum)
    return "%.2f" % sequence_sum





# Convert number to reversed array of digits
# https://www.codewars.com/kata/5583090cbe83f4fd8c000051
"""
Convert number to reversed array of digits
Given a random non-negative number, you have to return the digits of this number within an array in reverse order.

Example:
348597 => [7,9,5,8,4,3]
0 => [0]
"""
(digitize(348597), [7, 9, 5, 8, 4, 3])
(digitize(35231), [1, 3, 2, 5, 3])
(digitize(0), [0])
(digitize(23582357), [7, 5, 3, 2, 8, 5, 3, 2])
(digitize(984764738), [8, 3, 7, 4, 6, 7, 4, 8, 9])
(digitize(45762893920), [0, 2, 9, 3, 9, 8, 2, 6, 7, 5, 4])
(digitize(548702838394), [4, 9, 3, 8, 3, 8, 2, 0, 7, 8, 4, 5])


def digitize(number):
    return [int(digit) for digit in str(number)[::-1]]
    return list(map(int, str(n)[::-1]))

# oldies
def digitize(n):
    return [int(digit) for digit in reversed(str(n))]
    return list(reversed([int(digit) for digit in str(n)]))
    return list(map(int, str(n)))[::-1]





# Get the Middle Character
# https://www.codewars.com/kata/56747fd5cb988479af000028
"""
est") should return "es"

Kata.getMiddle("testing") should return "t"

Kata.getMiddle("middle") should return "dd"

Kata.getMiddle("A") should return "A"
#Input

A word (string) of length 0 < str < 1000 (In javascript you may get slightly more than 1000 in some test cases due to an error in the test cases). You do not need to test for this. This is only here to tell you that you do not need to worry about your solution timing out.

#Output

The middle character(s) of the word represented as a string.
"""
(get_middle("test"), "es")
(get_middle("testing"), "t")
(get_middle("middle"), "dd")
(get_middle("A"), "A")
(get_middle("of"), "of")


def get_middle(text):
    text_length = len(text)
    
    if text_length % 2:
        return text[text_length//2]
    else:
        return text[text_length//2 - 1: text_length//2 + 1]


def get_middle(s):
    return s[(len(s)-1) // 2 : len(s)//2 + 1]

def get_middle(s):
    return get_middle(s[1:-1]) if len(s) > 2 else s




# Transportation on vacation
# https://www.codewars.com/kata/568d0dd208ee69389d000016
"""
After a hard quarter in the office you decide to get some rest on a vacation. So you will book a flight for you and your girlfriend and try to leave all the mess behind you.

You will need a rental car in order for you to get around in your vacation. The manager of the car rental makes you some good offers.

Every day you rent the car costs $40. If you rent the car for 7 or more days, you get $50 off your total. Alternatively, if you rent the car for 3 or more days, you get $20 off your total.

Write a code that gives out the total amount for different days(d).

"""
(rental_car_cost(1), 40)
(rental_car_cost(4), 140)
(rental_car_cost(7), 230)
(rental_car_cost(8), 270)


def rental_car_cost(days):
    if days >= 7:
        return days * 40 - 50
    elif days >= 3:
        return days * 40 - 20
    else:
        return days * 40


# oldies
def rental_car_cost(d):
    return 40*d if d < 3 else 40*d - 20 if d < 7 else 40*d - 50

# from codewars
def rental_car_cost(d):
    return 40*d - (d > 2)*20 - (d > 6)*30





# Abbreviate a Two Word Name
# https://www.codewars.com/kata/57eadb7ecd143f4c9c0000a3
"""
Write a function to convert a name into initials. This kata strictly takes two words with one space in between them.

The output should be two capital letters with a dot separating them.

It should look like this:

Sam Harris => S.H

patrick feeney => P.F
"""
(abbrev_name("Sam Harris"), "S.H")
(abbrev_name("patrick feenan"), "P.F")
(abbrev_name("Evan C"), "E.C")
(abbrev_name("P Favuzzi"), "P.F")
(abbrev_name("David Mendieta"), "D.M")


def abbrev_name(name):
    return ".".join(word[0].upper() 
                    for word in name.split())

def abbrev_name(name):
    return ".".join(map(lambda word: word[0].upper(), name.split()))





# Count by X
# https://www.codewars.com/kata/5513795bd3fafb56c200049e
"""
Create a function with two arguments that will return an array of the first (n) multiples of (x).

Assume both the given number and the number of times to count will be positive numbers greater than 0.

Return the results as an array (or list in Python, Haskell or Elixir).

Examples:

count_by(1,10) #should return [1,2,3,4,5,6,7,8,9,10]
count_by(2,5) #should return [2,4,6,8,10]
"""
(count_by(1, 5), [1, 2, 3, 4, 5])
(count_by(2, 5), [2, 4, 6, 8, 10])
(count_by(3, 5), [3, 6, 9, 12, 15])
(count_by(50, 5), [50, 100, 150, 200, 250])
(count_by(100, 5), [100, 200, 300, 400, 500])


def count_by(number, times):
    return list(range(number, number * times + 1, number))


# oldies
def count_by(x, n):
    return [x * i for i in range(1, n + 1)]
    return list(map(lambda y: x*y, range(1, n+1)))

import numpy as np
def count_by(x, n):
    return list(np.array(range(1, n + 1)) * x)





# Beginner Series #1 School Paperwork
# https://www.codewars.com/kata/55f9b48403f6b87a7c0000bd
"""
Your classmates asked you to copy some paperwork for them. You know that there are 'n' classmates and the paperwork has 'm' pages.

Your task is to calculate how many blank pages do you need. If n < 0 or m < 0 return 0.

Example:
n= 5, m=5: 25
n=-5, m=5:  0
"""
(paperwork(5, 5), 25)
(paperwork(-5, 5), 0)
(paperwork(5, -5), 0)
(paperwork(-5, -5), 0)
(paperwork(5, 0), 0)


def paperwork(classmate, page):
    return max(classmate, 0) * max(page, 0)
    return classmate * page if classmate > 0 and page > 0 else 0





# Ones and Zeros
# https://www.codewars.com/kata/578553c3a1b8d5c40300037c
"""
Given an array of ones and zeroes, convert the equivalent binary value to an integer.

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
However, the arrays can have varying lengths, not just limited to 4.
"""
(binary_array_to_number([0, 0, 0, 1]), 1)
(binary_array_to_number([0, 0, 1, 0]), 2)
(binary_array_to_number([1, 1, 1, 1]), 15)
(binary_array_to_number([0, 1, 1, 0]), 6)


def binary_array_to_number(numbers):
    binary_str = "".join(map(str, numbers))
    
    return int(binary_str, 2)

def binary_array_to_number(numbers):
    binary_str = "".join(str(digit) for digit in numbers)
    
    return int(binary_str, 2)





# Shortest Word
# https://www.codewars.com/kata/57cebe1dc6fdc20c57000ac9
"""
Simple, given a string of words, return the length of the shortest word(s).

String will never be empty and you do not need to account for different data types.
"""


def find_short(text):
    return len(min(text.split(), key=len))
    # return len(min(text.split(), key=lambda word: len(word)))
    # return min(map(len, text.split()))
    # return min(len(word) for word in text.split())
(find_short("bitcoin take over the world maybe who knows perhaps"), 3)
(find_short("turns out random test cases are easier than writing out basic ones"), 3)
(find_short("lets talk about javascript the best language"), 3)
(find_short("i want to travel the world writing code one day"), 1)
(find_short("Lets all go on holiday somewhere very cold"), 2)   
(find_short("Let's travel abroad shall we"), 2)





# Fake Binary
# https://www.codewars.com/kata/57eae65a4321032ce000002d
"""
Given a string of digits, you should replace any digit below 5 with '0' and any digit 5 and above with '1'. Return the resulting string.

Note: input will never be an empty string
"""
(fake_bin("45385593107843568"), "01011110001100111")
(fake_bin("509321967506747"), "101000111101101")
(fake_bin("366058562030849490134388085"), "011011110000101010000011011")
(fake_bin("15889923"), "01111100")
(fake_bin("800857237867"), "100111001111")


def fake_bin(number):
    return "".join("0" if digit < "5" else "1" 
                   for digit in str(number))

def fake_bin(number):
    return "".join(map(lambda digit: "0" if digit < "5" else "1", str(number)))





# Take a Ten Minutes Walk
# https://www.codewars.com/kata/54da539698b8a2ad76000228
"""
You live in the city of Cartesia where all roads are laid out in a perfect grid. You arrived ten minutes too early to an appointment, so you decided to take the opportunity to go for a short walk. The city provides its citizens with a Walk Generating App on their phones -- everytime you press the button it sends you an array of one-letter strings representing directions to walk (eg. ['n', 's', 'w', 'e']). You always walk only a single block for each letter (direction) and you know it takes you one minute to traverse one city block, so create a function that will return true if the walk the app gives you will take you exactly ten minutes (you don't want to be early or late!) and will, of course, return you to your starting point. Return false otherwise.

Note: you will always receive a valid array (string in COBOL) containing a random assortment of direction letters ('n', 's', 'e', or 'w' only). It will never give you an empty array (that's not a walk, that's standing still!).
"""
(is_valid_walk(['n','s','n','s','n','s','n','s','n','s']), True)
(is_valid_walk(['w','e','w','e','w','e','w','e','w','e','w','e']), False)
(is_valid_walk(['w']), False)
(is_valid_walk(['n','n','n','s','n','s','n','s','n','s']), False)


def is_valid_walk(directions):
    return (
        len(directions) == 10 and
        directions.count("n") == directions.count("s") and
        directions.count("e") == directions.count("w")
    )





# Calculate average
# https://www.codewars.com/kata/57a2013acf1fa5bfc4000921
"""
Write a function which calculates the average of the numbers in a given list.

Note: Empty arrays should return 0.
"""
(find_average([1, 2, 3]), 2)
(find_average([]), 0)


def find_average(numbers):
    return sum(numbers) / len(numbers) if numbers else 0

import numpy as np

def find_average(numbers):
    return np.average(numbers)

def find_average(numbers):
    try:
        return sum(numbers) / len(numbers)
    except ZeroDivisionError as err:
        return f"Error!!!: {err}"





# Sum Mixed Array
# https://www.codewars.com/kata/57eaeb9578748ff92a000009
"""
Given an array of integers as strings and numbers, return the sum of the array values as if all were numbers.

Return your answer as a number.
"""
(sum_mix([9, 3, '7', '3']), 22)
(sum_mix(['5', '0', 9, 3, 2, 1, '9', 6, 7]), 42)
(sum_mix(['3', 6, 6, 0, '5', 8, 5, '6', 2,'0']), 41)
(sum_mix(['1', '5', '8', 8, 9, 9, 2, '3']), 45)
(sum_mix([8, 0, 0, 8, 5, 7, 2, 3, 7, 8, 6, 7]), 61)


def sum_mix(arr):
    return sum(map(int, arr))





# Switch it Up!
# https://www.codewars.com/kata/5808dcb8f0ed42ae34000031
"""
When provided with a number between 0-9, return it in words.

Input :: 1

Output :: "One".

If your language supports it, try using a switch statement.
"""
(switch_it_up(0), "Zero")
(switch_it_up(9), "Nine")


def switch_it_up(digit):
    return ("Zero One Two Three Four Five Six Seven Eight Nine"
            .split()[digit])


def switch_it_up(digit):
    return ("Zero", "One", "Two", "Three", "Four", "Five", 
            "Six", "Seven", "Eight", "Nine")[digit]


def switch_it_up(digit):
    digit_to_sring = {
        0: 'Zero',
        1: 'One',
        2: 'Two',
        3: 'Three',
        4: 'Four',
        5: 'Five',
        6: 'Six',
        7: 'Seven',
        8: 'Eight',
        9: 'Nine'
    }
    return digit_to_sring[digit]


def switch_it_up(digit):
    match digit:
        case 0: return "Zero"
        case 1: return "One"
        case 2: return "Two"
        case 3: return "Three"
        case 4: return "Four"
        case 5: return "Five"
        case 6: return "Six"
        case 7: return "Seven"
        case 8: return "Eight"
        case 9: return "Nine"





# Array.diff
# https://www.codewars.com/kata/523f5d21c841566fde000009
"""
Your goal in this kata is to implement a difference function, which subtracts one list from another and returns the result.

It should remove all values from list a, which are present in list b keeping their order.

array_diff([1,2],[1]) == [2]
If a value is present in b, all of its occurrences must be removed from the other:

array_diff([1,2,2,2,3],[2]) == [1,3]
"""
(array_diff([1, 2], [1]), [2])
(array_diff([1, 2, 2], [1]), [2, 2])
(array_diff([1, 2, 2], [2]), [1])
(array_diff([1, 2, 2], []), [1, 2, 2])
(array_diff([], [1, 2]), [])
(array_diff([1, 2, 3], [1, 2]), [3])


def array_diff(numbers, tabu_list):
    tabu_set = set(tabu_list)

    return list(filter(lambda digit: digit not in tabu_set, numbers))


def array_diff(numbers, tabu_list):
    tabu_set = set(tabu_list)

    return [number for number in numbers if number not in tabu_set]





# Welcome!
# https://www.codewars.com/kata/577ff15ad648a14b780000e7
"""
Your start-up's BA has told marketing that your website has a large audience in Scandinavia and surrounding countries. Marketing thinks it would be great to welcome visitors to the site in their own language. Luckily you already use an API that detects the user's location, so this is an easy win.

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
IP_ADDRESS_REQUIRED - no ip address was supplied
"""
(greet('english'), 'Welcome')
(greet('dutch'), 'Welkom')
(greet('IP_ADDRESS_INVALID'), 'Welcome')
(greet(''), 'Welcome')
(greet(2), 'Welcome')


language_to_greet = {
    "english": "Welcome",
    "czech": "Vitejte",
    "danish": "Velkomst",
    "dutch": "Welkom",
    "estonian": "Tere tulemast",
    "finnish": "Tervetuloa",
    "flemish": "Welgekomen",
    "french": "Bienvenue",
    "german": "Willkommen",
    "irish": "Failte",
    "italian": "Benvenuto",
    "latvian": "Gaidits",
    "lithuanian": "Laukiamas",
    "polish": "Witamy",
    "spanish": "Bienvenido",
    "swedish": "Valkommen",
    "welsh": "Croeso"
}

def greet(language):
    return language_to_greet.get(language, language_to_greet['english'])





# Area or Perimeter
# https://www.codewars.com/kata/5ab6538b379d20ad880000ab
"""
You are given the length and width of a 4-sided polygon. The polygon can either be a rectangle or a square.
If it is a square, return its area. If it is a rectangle, return its perimeter.

area_or_perimeter(6, 10) --> 32
area_or_perimeter(3, 3) --> 9
Note: for the purposes of this kata you will assume that it is a square if its length and width are equal, otherwise it is a rectangle.
"""
(area_or_perimeter(4, 4), 16)
(area_or_perimeter(6, 10), 32)


def area_or_perimeter(width, height):
    if width == height:
        return width ** 2
    else:
        return (width + height) << 1





# Total amount of points
# https://www.codewars.com/kata/5bb904724c47249b10000131
"""
Our football team finished the championship. The result of each match look like "x:y". Results of all matches are recorded in the collection.

For example: ["3:1", "2:2", "0:1", ...]

Write a function that takes such collection and counts the points of our team in the championship. Rules for counting points for each match:

if x > y: 3 points
if x < y: 0 point
if x = y: 1 point
Notes:

there are 10 matches in the championship
0 <= x <= 4
0 <= y <= 4
"""
(points(['1:0','2:0','3:0','4:0','2:1','3:1','4:1','3:2','4:2','4:3']), 30)
(points(['1:1','2:2','3:3','4:4','2:2','3:3','4:4','3:3','4:4','4:4']), 10)
(points(['0:1','0:2','0:3','0:4','1:2','1:3','1:4','2:3','2:4','3:4']), 0)
(points(['1:0','2:0','3:0','4:0','2:1','1:3','1:4','2:3','2:4','3:4']), 15)
(points(['1:0','2:0','3:0','4:4','2:2','3:3','1:4','2:3','2:4','3:4']), 12)


def points(games):
    points = 0

    for game in games:
        team_a, team_b = game.split(":")
        team_a, team_b = int(team_a), int(team_b)
        
        if team_a > team_b:
            points += 3
        elif team_a == team_b:
            points += 1

    return points


import re
def points(games):
    points = 0
    for game in games:
        a, b = re.findall(r"\d+", game)  # re.split(r":", game)
        if a == b:
            points += 1
        elif a > b:
            points += 3
    return points

def points(games):
    return sum(3 if x > y else 1 for x, _, y in games if x >= y)
    return sum(3 if x[0] > x[-1] else 1 for x in games if x[0] >= x[-1])





# Two Sum
# https://www.codewars.com/kata/52c31f8e6605bcc646000082
"""
Write a function that takes an array of numbers (integers for the tests) and a target number. It should find two different items in the array that, when added together, give the target value. The indices of these items should then be returned in a tuple / list (depending on your language) like so: (index1, index2).

For the purposes of this kata, some tests may have multiple answers; any valid solutions will be accepted.

The input will always be valid (numbers will be an array of length 2 or greater, and all of the items will be numbers; target will always be the sum of two different items from that array).

Based on: http://leetcode.com/problems/two-sum/

twoSum [1, 2, 3] 4 === (0, 2)
"""
(two_sum([1, 2, 3], 4), [0, 2])
(two_sum([1234, 5678, 9012], 14690), [1, 2])
(two_sum([2, 2, 3], 4), [0, 1])


def two_sum(numbers, target):
    seen_numbers = {}

    for index, number in enumerate(numbers):
        diff = target - number

        if diff in seen_numbers:
            return [seen_numbers[diff], index]
        else:
            seen_numbers[number] = index  # seen_numbers.update({number: index})


# O(n2)
def two_sum(numbers, target):
    for i in range(len(numbers) - 1):
        for j in range(i + 1, len(numbers)):
            if numbers[i] + numbers[j] == target:
                return (i, j)





# How good are you really?
# https://www.codewars.com/kata/5601409514fc93442500010b
"""
There was a test in your class and you passed it. Congratulations!
But you're an ambitious person. You want to know if you're better than the average student in your class.

You receive an array with your peers' test scores. Now calculate the average and compare your score!

Return True if you're better, else False!

Note:
Your points are not included in the array of your class's points. For calculating the average point you may add your point to the given array!
"""
(better_than_average([2, 3], 5), True)
(better_than_average([100, 40, 34, 57, 29, 72, 57, 88], 75), True)
(better_than_average([12, 23, 34, 45, 56, 67, 78, 89, 90], 69), True)
(better_than_average([41, 75, 72, 56, 80, 82, 81, 33], 50), False)
(better_than_average([29, 55, 74, 60, 11, 90, 67, 28], 21), False)


def better_than_average(class_points, your_points):
    class_points.append(your_points)

    return your_points > (sum(class_points) / len(class_points))


# oldies
import numpy as np
def better_than_average(class_points, your_points):
    return your_points > np.mean(class_points + [your_points])

better_than_average = lambda class_points, your_points: your_points > np.mean(class_points + [your_points])





# altERnaTIng cAsE <=> ALTerNAtiNG CaSe
# https://www.codewars.com/kata/56efc695740d30f963000557
"""
altERnaTIng cAsE <=> ALTerNAtiNG CaSe
altERnaTIng cAsE <=> ALTerNAtiNG CaSe
Define String.prototype.toAlternatingCase (or a similar function/method such as to_alternating_case/toAlternatingCase/ToAlternatingCase in your selected language; see the initial solution for details) such that each lowercase letter becomes uppercase and each uppercase letter becomes lowercase. For example:

"hello world".to_alternating_case() === "HELLO WORLD"
"HELLO WORLD".to_alternating_case() === "hello world"
"hello WORLD".to_alternating_case() === "HELLO world"
"HeLLo WoRLD".to_alternating_case() === "hEllO wOrld"
"12345".to_alternating_case() === "12345" // Non-alphabetical characters are unaffected
"1a2b3c4d5e".to_alternating_case() === "1A2B3C4D5E"
"String.prototype.toAlternatingCase".toAlternatingCase() === "sTRING.PROTOTYPE.TOaLTERNATINGcASE"
As usual, your function/method should be pure, i.e. it should not mutate the original string.
"""
(to_alternating_case("hello world"), "HELLO WORLD")
(to_alternating_case("HELLO WORLD"), "hello world")
(to_alternating_case("hello WORLD"), "HELLO world")
(to_alternating_case("HeLLo WoRLD"), "hEllO wOrld")
(to_alternating_case("12345"), "12345")
(to_alternating_case("1a2b3c4d5e"), "1A2B3C4D5E")
(to_alternating_case("String.prototype.toAlternatingCase"), "sTRING.PROTOTYPE.TOaLTERNATINGcASE")
(to_alternating_case(to_alternating_case("Hello World")), "Hello World")


def reverse_case(letter):
    return letter.lower() if letter.isupper() else letter.upper()

def to_alternating_case(text):
    return "".join(map(reverse_case, text))
    # return "".join(reverse_case(letter) for letter in text)


def to_alternating_case(text):
    return text.swapcase()

to_alternating_case = lambda text: text.swapcase()
to_alternating_case = str.swapcase





# Sort the odd
# https://www.codewars.com/kata/578aa45ee9fd15ff4600090d
"""
You will be given an array of numbers. You have to sort the odd numbers in ascending order while leaving the even numbers at their original positions.

Examples
[7, 1]  =>  [1, 7]
[5, 8, 6, 3, 4]  =>  [3, 8, 6, 5, 4]
[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]  =>  [1, 8, 3, 6, 5, 4, 7, 2, 9, 0]
"""
(sort_array([5, 3, 2, 8, 1, 4]), [1, 3, 2, 8, 5, 4])
(sort_array([5, 3, 1, 8, 0]), [1, 3, 5, 8, 0])
(sort_array([]), [])
(sort_array([1, 11, 2, 8, 3, 4, 5]), [1, 3, 2, 8, 5, 4, 11])


def sort_array(numbers):
    odd_numbers = []

    # exteract odd numbers
    for number in numbers:
        if number % 2:
            odd_numbers.append(number)

    # sort in reverced order to pop from the end
    odd_numbers.sort(reverse=True)

    # insert odd numbers
    for index, number in enumerate(numbers):
        if number % 2:
            numbers[index] = odd_numbers.pop()

    return numbers


def sort_array(numbers):
    # exteract odd numbers
    odd_numbers = list(filter(lambda number: number % 2, numbers))

    # sort in reverced order to pop from the end
    odd_numbers.sort(reverse=True)

    # insert odd numbers
    return [
        odd_numbers.pop() if number % 2 else number
        for number in numbers]





# Double Char
# https://www.codewars.com/kata/56b1f01c247c01db92000076
"""
Given a string, you have to return a string in which each character (case-sensitive) is repeated once.

Examples (Input -> Output):
* "String"      -> "SSttrriinngg"
* "Hello World" -> "HHeelllloo  WWoorrlldd"
* "1234!_ "     -> "11223344!!__  "
"""
(double_char("String"),"SSttrriinngg")
(double_char("Hello World"),"HHeelllloo  WWoorrlldd")
(double_char("1234!_ "),"11223344!!__  ")

def double_char(word):
    return "".join(letter * 2 for letter in word)
    # return "".join(map(lambda letter: letter * 2, word))





# Calculate BMI
# https://www.codewars.com/kata/57a429e253ba3381850000fb
"""
Write function bmi that calculates body mass index (bmi = weight / height2).

if bmi <= 18.5 return "Underweight"

if bmi <= 25.0 return "Normal"

if bmi <= 30.0 return "Overweight"

if bmi > 30 return "Obese"
"""
(bmi(50, 1.80), "Underweight")
(bmi(80, 1.80), "Normal")
(bmi(90, 1.80), "Overweight")
(bmi(110, 1.80), "Obese")
(bmi(50, 1.50), "Normal")   
(bmi(100, 2.00), "Normal")


def bmi(weight, height):
    bmi = weight / height ** 2
    if bmi > 30:
        return "Obese"
    elif bmi > 25:
        return "Overweight"
    elif bmi > 18.5:
        return "Normal"
    else:
        return "Underweight"


# from codewars
def bmi(weight, height):
    b = weight / height ** 2
    return ['Underweight', 'Normal', 'Overweight', 'Obese'][(b > 30) + (b > 25) + (b > 18.5)]





# Sort array by string length
# https://www.codewars.com/kata/57ea5b0b75ae11d1e800006c
"""
Write a function that takes an array of strings as an argument and returns a sorted array containing the same strings, ordered from shortest to longest.

For example, if this array were passed as an argument:

["Telescopes", "Glasses", "Eyes", "Monocles"]

Your function would return the following array:

["Eyes", "Glasses", "Monocles", "Telescopes"]

All of the strings in the array passed to your function will be different lengths, so you will not have to decide how to order multiple strings of the same length.
"""
(sort_by_length(["beg", "life", "i", "to"]), ["i", "to", "beg", "life"])
(sort_by_length(["", "moderately", "brains", "pizza"]), ["", "pizza", "brains", "moderately"])
(sort_by_length(["longer", "longest", "short"]), ["short", "longer", "longest"])
(sort_by_length(["dog", "food", "a", "of"]), ["a", "of", "dog", "food"])
(sort_by_length(["", "dictionary", "eloquent", "bees"]), ["", "bees", "eloquent", "dictionary"])
(sort_by_length(["a longer sentence", "the longest sentence", "a short sentence"]), ["a short sentence", "a longer sentence", "the longest sentence"])


def sort_by_length(item_list):
    item_list.sort(key=len)
    # item_list.sort(key=lambda word: len(word))
    return item_list


sort_by_length = lambda arr: sorted(arr, key=len)

def sort_by_length(arr):
    return sorted(arr, key=len)
    # return sorted(arr, key=lambda x: len(x))





# Make a function that does arithmetic!
# https://www.codewars.com/kata/583f158ea20cfcbeb400000a
"""
Given two numbers and an arithmetic operator (the name of it, as a string), return the result of the two numbers having that operator used on them.

a and b will both be positive integers, and a will always be the first number in the operation, and b always the second.

The four operators are "add", "subtract", "divide", "multiply".

A few examples:(Input1, Input2, Input3 --> Output)

5, 2, "add"      --> 7
5, 2, "subtract" --> 3
5, 2, "multiply" --> 10
5, 2, "divide"   --> 2.5
Try to do it without using if statements!
"""
(arithmetic(1, 2, "add"), 3)
(arithmetic(8, 2, "subtract"), 6)
(arithmetic(5, 2, "multiply"), 10)
(arithmetic(8, 2, "divide"), 4)


def arithmetic(a, b, operator):
    match operator:
        case "add":
            return a + b
        case "subtract":
            return a - b
        case "multiply":
            return a * b
        case "divide":
            return a / b


def arithmetic(a, b, operator):
    ar_dict = {'add': '+', 'subtract': '-', 'multiply': '*', 'divide': '/'}
    return eval(f"{a}{ar_dict[operator]}{b}")
    # return eval('(' + 'a' + ar_dict[operator] + 'b' + ')')


# from codewars
def arithmetic(a, b, operator):
    return {'add': a + b, 'subtract': a - b, 'multiply': a * b, 'divide': a / b}[operator]

from operator import add, sub, mul, truediv
def arithmetic(a, b, operator):
    ops = {'add': add, 'subtract': sub, 'multiply': mul, 'divide': truediv}
    return ops[operator](a, b)





# Beginner Series #3 Sum of Numbers
# https://www.codewars.com/kata/55f2b110f61eb01779000053
"""
Given two integers a and b, which can be positive or negative, find the sum of all the integers between and including them and return it. If the two numbers are equal return a or b.

Note: a and b are not ordered!

Examples (a, b) --> output (explanation)
(1, 0) --> 1 (1 + 0 = 1)
(1, 2) --> 3 (1 + 2 = 3)
(0, 1) --> 1 (0 + 1 = 1)
(1, 1) --> 1 (1 since both are same)
(-1, 0) --> -1 (-1 + 0 = -1)
(-1, 2) --> 2 (-1 + 0 + 1 + 2 = 2)
"""
(get_sum(1, 0), 1)
(get_sum(1, 2), 3)
(get_sum(0, 1), 1)
(get_sum(1, 1), 1)
(get_sum(-1, 0), -1)
(get_sum(-1, 2), 2 )


def get_sum(start, stop):
    if stop < start:
        start, stop = stop, start

    return sum(range(start, stop + 1))


get_sum = lambda a, b: sum(range(min(a, b), max(a, b) + 1))





# Equal Sides Of An Array
# https://www.codewars.com/kata/5679aa472b8f57fb8c000047
"""
You are going to be given an array of integers. Your job is to take that array and find an index N where the sum of the integers to the left of N is equal to the sum of the integers to the right of N. If there is no index that would make this happen, return -1.

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
If you are given an array with multiple answers, return the lowest correct index.
"""
(find_even_index([1, 2, 3, 4, 3, 2, 1]), 3)
(find_even_index([1, 100, 50, -51, 1, 1]), 1,)
(find_even_index([1, 2, 3, 4, 5, 6]), -1)
(find_even_index([20, 10, 30, 10, 10, 15, 35]), 3)
(find_even_index([20, 10, -80, 10, 10, 15, 35]), 0)
(find_even_index([10, -80, 10, 10, 15, 35, 20]), 6)
(find_even_index(list(range(1, 100))), -1)
(find_even_index([0, 0, 0, 0, 0]), 0, "Should pick the first index if more cases are valid")
(find_even_index([-1, -2, -3, -4, -3, -2, -1]), 3)
(find_even_index(list(range(-100, -1))), -1)


def find_even_index(numbers):
    left = 0
    right = sum(numbers)

    for index, number in enumerate(numbers):
        right -= number
        
        if left == right:
            return index
        
        left += number
    
    return -1


def find_even_index(numbers):
    for index in range(len(numbers)):
        if sum(numbers[:index]) == sum(numbers[index + 1:]):
            return index
    
    return -1





# Simple multiplication
# https://www.codewars.com/kata/583710ccaa6717322c000105
"""
This kata is about multiplying a given number by eight if it is an even number and by nine otherwise.
"""
(simple_multiplication(2), 16)
(simple_multiplication(1), 9)
(simple_multiplication(8), 64)
(simple_multiplication(4), 32)
(simple_multiplication(5), 45)


def simple_multiplication(number):
    return number * 9 if number % 2 else number * 8


def simple_multiplication(number):
    return (9 if number % 2 else 8) * number


simple_multiplication = lambda number: number * (8 + (number&1))





# Sum of a sequence
# https://www.codewars.com/kata/586f6741c66d18c22800010a
"""
Your task is to make function, which returns the sum of a sequence of integers.

The sequence is defined by 3 non-negative values: begin, end, step (inclusive).

If begin value is greater than the end, function should returns 0

Examples

2,2,2 --> 2
2,6,2 --> 12 (2 + 4 + 6)
1,5,1 --> 15 (1 + 2 + 3 + 4 + 5)
1,5,3  --> 5 (1 + 4)
"""
(sequence_sum(2, 6, 2), 12)
(sequence_sum(1, 5, 1), 15)
(sequence_sum(1, 5, 3), 5)
(sequence_sum(0, 15, 3), 45)
(sequence_sum(16, 15, 3), 0)
(sequence_sum(2, 24, 22), 26)
(sequence_sum(2, 2, 2), 2)
(sequence_sum(2, 2, 1), 2)
(sequence_sum(1, 15, 3), 35)
(sequence_sum(15, 1, 3), 0)


def sequence_sum(begin, end, step):
    return sum(range(begin, end + 1, step))





# Extract the domain name from a URL
# https://www.codewars.com/kata/514a024011ea4fb54200004b
"""
Write a function that when given a URL as a string, parses out just the domain name and returns it as a string. For example:

* url = "http://github.com/carbonfive/raygun" -> domain name = "github"
* url = "http://www.zombie-bites.com"         -> domain name = "zombie-bites"
* url = "https://www.cnet.com"                -> domain name = cnet"
"""
domain_name('http://google.com')
domain_name('https://google.com')
domain_name('https://www.codewars.com')
domain_name('https://google.co.jp')
domain_name('www.xakep.ru')
domain_name('https://hyphen-site.org')
domain_name('icann.org')


import re

def domain_name(url):
    match = re.search(r"(https?://)?(www\.)?([\w-]+)\..*", url)
    return match.group(3) if match else None

def domain_name(url):
    return re.sub(r"(https?://)?(www\.)?([\w-]+)\..*", r"\3", url)


# from codewars
import re
def domain_name(url):
    return re.match(r'(https?://)?(www\.)?(?P<domain>[\w-]+)\..*$', url).group('domain')

# from codewars
def domain_name(url):
    return url.split('//')[-1].split('www.')[-1].split('.')[0]





# Find the middle element
# https://www.codewars.com/kata/545a4c5a61aa4c6916000755
"""
As a part of this Kata, you need to create a function that when provided with a triplet, returns the index of the numerical element that lies between the other two elements.

The input to the function will be an array of three distinct numbers (Haskell: a tuple).

For example:

gimme([2, 3, 1]) => 0
2 is the number that fits between 1 and 3 and the index of 2 in the input array is 0.

Another example (just to make sure it is clear):

gimme([5, 10, 14]) => 1
10 is the number that fits between 5 and 14 and the index of 10 in the input array is 1.
"""
(gimme([2, 3, 1]), 0)
(gimme([5, 10, 14]), 1)


def gimme(nums):
    _, middle, _ = sorted(nums)
    return nums.index(middle)





# Grasshopper - Messi goals function
# https://www.codewars.com/kata/55f73be6e12baaa5900000d4
"""
Messi goals function
Messi is a soccer player with goals in three leagues:

LaLiga
Copa del Rey
Champions
Complete the function to return his total number of goals in all three leagues.

Note: the input will always be valid.

For example:

5, 10, 2  -->  17
"""
(goals(0, 0, 0), 0)
(goals(5, 10, 2), 17)


def goals(*args):
    return sum(args)

goals = lambda *args: sum(args)





# Beginner Series #4 Cockroach
# https://www.codewars.com/kata/55fab1ffda3e2e44f00000c6
"""
The cockroach is one of the fastest insects. Write a function which takes its speed in km per hour and returns it in cm per second, rounded down to the integer (= floored).

For example:

1.08 --> 30
"""
(cockroach_speed(1.08), 30)


def cockroach_speed(speed):
    return speed * 100_000 / 3600





# Two fighters, one winner.
# https://www.codewars.com/kata/577bd8d4ae2807c64b00045b
"""
Create a function that returns the name of the winner in a fight between two fighters.

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
(declare_winner(Fighter("Lew", 10, 2), Fighter("Harry", 5, 4), "Lew"), "Lew")
(declare_winner(Fighter("Lew", 10, 2), Fighter("Harry", 5, 4), "Harry"),"Harry")
(declare_winner(Fighter("Harald", 20, 5), Fighter("Harry", 5, 4), "Harry"),"Harald")
(declare_winner(Fighter("Harald", 20, 5), Fighter("Harry", 5, 4), "Harald"),"Harald")
(declare_winner(Fighter("Jerry", 30, 3), Fighter("Harald", 20, 5), "Jerry"), "Harald")
(declare_winner(Fighter("Jerry", 30, 3), Fighter("Harald", 20, 5), "Harald"),"Harald")


class Fighter(object):
    def __init__(self, name, health, damage_per_attack):
        self.name = name
        self.health = health
        self.damage_per_attack = damage_per_attack

    def __str__(self):
        return f"Fighter({self.name}, {self.health}, {self.damage_per_attack})"
    __repr__ = __str__


def attack_order(fighter1, fighter2, first_attacker):
    if fighter1.name == first_attacker:
        return (fighter1, fighter2)
    else:
        return (fighter2, fighter1)
    

def declare_winner(fighter1, fighter2, first_attacker):
    first, second = attack_order(fighter1, fighter2, first_attacker)
    rounds = ((first, second), (second, first))
    
    while True:
        for (attacker, defender) in rounds:
            defender.health -= attacker.damage_per_attack

            if defender.health <= 0:
                return attacker.name


def declare_winner(fighter1, fighter2, first_attacker):
    if fighter1.name == first_attacker:
        attacker, defender = fighter1.name, fighter2.name
    else:
        attacker, defender = fighter2.name, fighter1.name
    
    while True:
        if attacker.name == first_attacker:
            defender.health -= attacker.damage_per_attack

            if defender.health <= 0:
                return attacker.name

            attacker.health -= defender.damage_per_attack

            if attacker.health <= 0:
                return defender.name            


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





# Primes in numbers
# https://www.codewars.com/kata/54d512e62a5e54c96200019e/
"""
Given a positive number n > 1 find the prime factor decomposition of n. The result will be a string with the following form :

 "(p1**n1)(p2**n2)...(pk**nk)"
with the p(i) in increasing order and n(i) empty if n(i) is 1.

Example: n = 86240 should return "(2**5)(5)(7**2)(11)"
"""
(prime_factors(86240), "(2**5)(5)(7**2)(11)")
(prime_factors(7775460), "(2**2)(3**3)(5)(7)(11**2)(17)")
(prime_factors(7919), "(7919)")


def prime_factors(number):
    # get dividers
    dividers = []
    
    while number > 1:
        for divider in range(2, number + 1):
            if not number % divider:
                number //= divider
                dividers.append(divider)
                break

    # count dividers
    counter = {}

    for divider in dividers:
            counter[divider] = counter.get(divider, 0) + 1


    # set answer string
    divider_string = ""

    for key, val in counter.items():
        if val == 1:
            divider_string += f"({key})"
        else:
            divider_string += f"({key}**{val})"
             
    return divider_string


from collections import Counter
def prime_factors(n):
    factors = []
    while n > 1:
        for i in range(2, n + 1):
            if not n % i:
                factors.append(i)
                n //= i
                break
    return "".join(f"({k})" if v == 1 else f"({k}**{v})" for k, v in Counter(factors).items())






# Count the divisors of a number
# https://www.codewars.com/kata/542c0f198e077084c0000c2e
"""
Count the number of divisors of a positive integer n.

Random tests go up to n = 500000.

Examples (input --> output)
4 --> 3 // we have 3 divisors - 1, 2 and 4
5 --> 2 // we have 2 divisors - 1 and 5
12 --> 6 // we have 6 divisors - 1, 2, 3, 4, 6 and 12
30 --> 8 // we have 8 divisors - 1, 2, 3, 5, 6, 10, 15 and 30
Note you should only return a number, the count of divisors. The numbers between parentheses are shown only for you to see which numbers are counted in each case.
"""
(divisors(1), 1)
(divisors(4), 3)
(divisors(5), 2)
(divisors(12), 6)
(divisors(30), 8)
(divisors(4096), 13)

# O(sqrtn)
import numpy as np

def divisors(number):
    counter = 0

    for divider in range(1, int(np.sqrt(number) + 1)):
        if not number % divider:
            counter += 2

    if not np.sqrt(number) % 1:
        counter -= 1

    return counter


# O(n)
def divisors(number):
    return sum(True for divider in range(1, number + 1)
               if not number % divider)


def divisors(n):
    return sum(True for i in range(1, n + 1) if not n % i)  # 3.687169998000172
    return len([i for i in range(1, n + 1) if not n % i])  # 3.5938232469998184
    return len([True for i in range(1, n + 1) if not n % i])  # 3.6050773840001966
    return sum([True for i in range(1, n + 1) if not n % i])  # 3.6067721780000284
    return len([not n % i for i in range(1, n + 1)])  # 4.191705105999972
    return sum(not n % i for i in range(1, n + 1))  # 5.578456058000029
    return sum(True if not n % i else False for i in range(1, n + 1))  # 5.622829734999868
    return len(list(filter(lambda i: not n % i, range(1, n + 1))))  # 6.283454425999935
    return sum(map(lambda x: not n % x, range(1, n + 1)))





# Square Every Digit
# https://www.codewars.com/kata/546e2562b03326a88e000020/train/python
"""
Welcome. In this kata, you are asked to square every digit of a number and concatenate them.

For example, if we run 9119 through the function, 811181 will come out, because 92 is 81 and 12 is 1. (81-1-1-81)

Example #2: An input of 765 will/should return 493625 because 72 is 49, 62 is 36, and 52 is 25. (49-36-25)

Note: The function accepts an integer and returns an integer.

Happy Coding!
"""
(square_digits(9119), 811181)
(square_digits(0), 0)


def square_digits(number):
    return int("".join(map(lambda digit: str(int(digit) ** 2), str(number))))
    # return int("".join(str(int(digit) ** 2) for digit in str(number)))





# Powers of 2
# https://www.codewars.com/kata/57a083a57cb1f31db7000028
"""
Complete the function that takes a non-negative integer n as input, and returns a list of all the powers of 2 with the exponent ranging from 0 to n ( inclusive ).

Examples
n = 0  ==> [1]        # [2^0]
n = 1  ==> [1, 2]     # [2^0, 2^1]
n = 2  ==> [1, 2, 4]  # [2^0, 2^1, 2^2]
"""
(powers_of_two(0), [1])
(powers_of_two(1), [1, 2])
(powers_of_two(4), [1, 2, 4, 8, 16])


def powers_of_two(number):
    return [2 ** power for power in range(number + 1)]
    return list(map(lambda power: 2 ** power, range(number + 1)))





# Thinkful - Logic Drills: Traffic light
# https://www.codewars.com/kata/58649884a1659ed6cb000072
"""
You're writing code to control your town's traffic lights. You need a function to handle each change from green, to yellow, to red, and then to green again.

Complete the function that takes a string as an argument representing the current state of the light and returns a string representing the state the light should change to.

For example, when the input is green, output should be yellow.
"""
(update_light('green'), 'yellow')
(update_light('yellow'), 'red')
(update_light('red'), 'green')


def update_light(color):
    colors = ("green", "yellow", "red")
    index_of_color = colors.index(color)

    return colors[(index_of_color + 1) % 3]

def update_light(color):
    next_color = {
        "green": "yellow",
        "yellow": "red",
        "red": "green"
    }

    return next_color[color]





# Twice as old
# https://www.codewars.com/kata/5b853229cfde412a470000d0/train/
"""
Your function takes two arguments:

current father's age (years)
current age of his son (years)
Calculate how many years ago the father was twice as old as his son (or in how many years he will be twice as old). The answer is always greater or equal to 0, no matter if it was in the past or it is in the future.
"""


def twice_as_old(dad_years_old, son_years_old):
    return abs(dad_years_old - 2 * son_years_old)
(twice_as_old(36,7) , 22)
(twice_as_old(55,30) , 5)
(twice_as_old(42,21) , 0)
(twice_as_old(22,1) , 20)
(twice_as_old(29,0) , 29)





# Disemvowel Trolls
# https://www.codewars.com/kata/52fba66badcd10859f00097e
"""
Trolls are attacking your comment section!

A common way to deal with this situation is to remove all of the vowels from the trolls' comments, neutralizing the threat.

Your task is to write a function that takes a string and return a new string with all vowels removed.

For example, the string "This website is for losers LOL!" would become "Ths wbst s fr lsrs LL!".

Note: for this kata y isn't considered a vowel.
"""
(disemvowel("This website is for losers LOL!"), "Ths wbst s fr lsrs LL!")
(disemvowel("No offense but,\nYour writing is among the worst I've ever read"), "N ffns bt,\nYr wrtng s mng th wrst 'v vr rd")
(disemvowel("What are you, a communist?"), "Wht r y,  cmmnst?")


def disemvowel(text):
    vovels = "aeoiuAEOIU"

    return "".join(letter for letter in text if letter not in vovels)


def is_not_vovel(letter):
    vovels = "aeoiuAEOIU"

    return letter not in vovels

def disemvowel(text):

    return "".join(filter(is_not_vovel, text))


import re

def disemvowel(text):
    return re.sub(r"[aeoiu]", "", text, flags=re.I)


# oldies
def disemvowel(text):
    for i in "aeoiuAEOIU":
        text = text.replace(i, "")
    return text

def disemvowel(text):
    return "".join(i for i in text if i.lower() not in "aeoiu")






# Isograms
# https://www.codewars.com/kata/54ba84be607a92aa900000f1
"""
An isogram is a word that has no repeating letters, consecutive or non-consecutive. Implement a function that determines whether a string that contains only letters is an isogram. Assume the empty string is an isogram. Ignore letter case.

Example: (Input --> Output)

"Dermatoglyphics" --> true "aba" --> false "moOse" --> false (ignore letter case)

isIsogram "Dermatoglyphics" = true
isIsogram "moose" = false
isIsogram "aba" = false
"""
(is_isogram("Dermatoglyphics"), True )
(is_isogram("isogram"), True )
(is_isogram("aba"), False)
(is_isogram("moOse"), False)
(is_isogram("isIsogram"), False)
(is_isogram(""), True)


def is_isogram(text):
    return len(text) == len(set(text.lower()))


# oldies
def is_isogram(string):
    for i in string.lower():
        if string.lower().count(i) > 1:
            return False
    return True
    




# I love you, a little , a lot, passionately ... not at all
# https://www.codewars.com/kata/57f24e6a18e9fad8eb000296
"""
Who remembers back to their time in the schoolyard, when girls would take a flower and tear its petals, saying each of the following phrases each time a petal was torn:

"I love you"
"a little"
"a lot"
"passionately"
"madly"
"not at all"
If there are more than 6 petals, you start over with "I love you" for 7 petals, "a little" for 8 petals and so on.

When the last petal was torn there were cries of excitement, dreams, surging thoughts and emotions.

Your goal in this kata is to determine which phrase the girls would say at the last petal for a flower of a given number of petals. The number of petals is always greater than 0.
"""
(how_much_i_love_you(7), "I love you")
(how_much_i_love_you(3), "a lot")
(how_much_i_love_you(6), "not at all")


def how_much_i_love_you(number):
    number_to_phrase = (
        "I love you", "a little", "a lot",
        "passionately", "madly", "not at all"
    )

    return number_to_phrase[(number - 1) % 6]


def how_much_i_love_you(number):
    number_to_phrase = {
        0: "I love you",
        1: "a little",
        2: "a lot",
        3: "passionately",
        4: "madly",
        5: "not at all",
    }

    return number_to_phrase[(number - 1) % 6]





# Who likes it?
# https://www.codewars.com/kata/5266876b8f4bf2da9b000362
"""
You probably know the "like" system from Facebook and other pages. People can "like" blog posts, pictures or other items. We want to create the text that should be displayed next to such an item.

Implement the function which takes an array containing the names of people that like an item. It must return the display text as shown in the examples:

[]                                -->  "no one likes this"
["Peter"]                         -->  "Peter likes this"
["Jacob", "Alex"]                 -->  "Jacob and Alex like this"
["Max", "John", "Mark"]           -->  "Max, John and Mark like this"
["Alex", "Jacob", "Mark", "Max"]  -->  "Alex, Jacob and 2 others like this"
Note: For 4 or more names, the number in "and 2 others" simply increases.
"""
(likes([]), "no one likes this")
(likes(["Peter"]), "Peter likes this")
(likes(["Jacob", "Alex"]), "Jacob and Alex like this")
(likes(["Max", "John", "Mark"]), "Max, John and Mark like this")
(likes(["Alex", "Jacob", "Mark", "Max"]), "Alex, Jacob and 2 others like this")


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
"""
Take an integer n (n >= 0) and a digit d (0 <= d <= 9) as an integer.

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
Note that 121 has twice the digit 1.
"""
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


def nb_dig(number, digit_to_search):
    squared_numbers = (str(digit ** 2) for digit in range(number + 1))
    # squared_numbers = (map(lambda digit: str(digit ** 2), range(number + 1)))
    return "".join(squared_numbers).count(str(digit_to_search))


# oldies
def nb_dig(n, d):
    return sum(str(number**2).count(str(d)) for number in range(n + 1))
    return sum(map(lambda x: str(x**2).count(str(d)), range(n + 1)))

"""
n = 14656
d = 1
timeit.timeit(lambda: "".join(str(number**2) for number in range(n + 1)).count(str(d)), number=1_000)  # 8.781008230999987
timeit.timeit(lambda: "".join(map(lambda x: str(x**2), range(n + 1))).count(str(d)), number=1_000)  # 10.25051729400002
timeit.timeit(lambda: sum(str(number**2).count(str(d)) for number in range(n + 1)), number=1_000)  # 12.9422026430002
timeit.timeit(lambda: sum(map(lambda x: str(x**2).count(str(d)), range(n + 1))), number=1_000)  # 12.971102688000201
"""





# Sum The Strings
# https://www.codewars.com/kata/5966e33c4e686b508700002d/train/python
"""
Create a function that takes 2 integers in form of a string as an input, and outputs the sum (also as a string):

Example: (Input1, Input2 -->Output)

"4",  "5" --> "9"
"34", "5" --> "39"
"", "" --> "0"
"2", "" --> "2"
"-5", "3" --> "-2"
Notes:

If either input is an empty string, consider it as zero.

Inputs and the expected output will never exceed the signed 32-bit integer limit (2^31 - 1)
"""
(sum_str("4", "5"), "9")
(sum_str("34", "5"), "39")
(sum_str("9", ""), "9")
(sum_str("", "9"), "9")
(sum_str("", ""), "0")


def to_number(number):
    if not number:
        return 0
    else:
        return int(number)

def sum_str(number1, number2):
    return str(to_number(number1) + to_number(number2))


def sum_str(number1, number2):
    return str(int(number1 or 0) + int(number2 or 0))





# If you can't sleep, just count sheep!!
# https://www.codewars.com/kata/5b077ebdaf15be5c7f000077/train/python
"""
If you can't sleep, just count sheep!!

Task:
Given a non-negative integer, 3 for example, return a string with a murmur: "1 sheep...2 sheep...3 sheep...". Input will always be valid, i.e. no negative integers.
"""
(count_sheep(0), "")
(count_sheep(1), "1 sheep...")
(count_sheep(2), "1 sheep...2 sheep...")
(count_sheep(3), "1 sheep...2 sheep...3 sheep...")


def count_sheep(number):
    return "".join(f"{index + 1} sheep..." 
                   for index in range(number))





# Anagram Detection
# https://www.codewars.com/kata/529eef7a9194e0cbc1000255/train/python
"""
An anagram is the result of rearranging the letters of a word to produce a new word (see wikipedia).

Note: anagrams are case insensitive

Complete the function to return true if the two arguments given are anagrams of each other; return false otherwise.

Examples
"foefet" is an anagram of "toffee"

"Buckethead" is an anagram of "DeathCubeK"
"""
(is_anagram("foefet", "toffee"), True,)
(is_anagram("Buckethead", "DeathCubeK"), True)
(is_anagram("Twoo", "WooT"), True)
(is_anagram("dumble", "bumble"), False)
(is_anagram("ound", "round"), False)
(is_anagram("apple", "pale"), False)


# compare dicts
def is_anagram(word1, word2):
    if len(word1) != len(word2):
        return False
    
    counter1 = {}
    counter2 = {}

    for letter in word1.lower():
        counter1[letter] = counter1.get(letter, 0) + 1

    for letter in word2.lower():
        counter2[letter] = counter2.get(letter, 0) + 1
    
    return counter1 == counter2


# without comparing dicts, early exit
def is_anagram(word1, word2):
    if len(word1) != len(word2):
        return False
    
    counter1 = {}

    for letter in word1.lower():
        counter1[letter] = counter1.get(letter, 0) + 1

    # subtract occurences in word2 from word1
    for letter in word2.lower():
        if letter not in counter1:
            return False
        
        counter1[letter] -= 1

        if counter1[letter] == -1:
            return False

    # if anagram evely value should be 0
    for val in counter1.values():
        if val:
            return False

    return True


# O(nlogn)
def is_anagram(test, original):
    return sorted(test.lower()) == sorted(original.lower())





# Testing 1-2-3
# https://www.codewars.com/kata/54bf85e3d5b56c7a05000cf9/train/python
"""
Your team is writing a fancy new text editor and you've been tasked with implementing the line numbering.

Write a function which takes a list of strings and returns each line prepended by the correct number.

The numbering starts at 1. The format is n: string. Notice the colon and space in between.

Examples: (Input --> Output)

[] --> []
["a", "b", "c"] --> ["1: a", "2: b", "3: c"]
"""
(number(["a", "b", "c"]), ["1: a", "2: b", "3: c"])
(number([]), [])


format_letters = lambda x: f"{x[0]}: {x[1]}"

def number(letters):
    return list(map(format_letters, enumerate(letters, 1)))


def number(letters):
    return [f"{index}: {letter}" 
            for index, letter in enumerate(letters, 1)]


# oldies
def number(lines):
    return [f"{ind}: {line}" for ind, line in enumerate(lines, start=1)]
    return ["{}: {}".format(i, line) for i, line in enumerate(lines, start=1)]
    return ["{}: {}".format(*line) for line in enumerate(lines, start=1)]
    return ["%d: %s" % (i, line) for i, line in enumerate(lines, start=1)]
    return [ str(i) + ": " + line for i, line in enumerate(lines, start=1)]





# Delete occurrences of an element if it occurs more than n times
# https://www.codewars.com/kata/554ca54ffa7d91b236000023/train/python
"""
Enough is enough!
Alice and Bob were on a holiday. Both of them took many pictures of the places they've been, and now they want to show Charlie their entire collection. However, Charlie doesn't like these sessions, since the motif usually repeats. He isn't fond of seeing the Eiffel tower 40 times.
He tells them that he will only sit for the session if they show the same motif at most N times. Luckily, Alice and Bob are able to encode the motif as a number. Can you help them to remove numbers such that their list contains each number only up to N times, without changing the order?

Task
Given a list and a number, create a new list that contains each number of list at most N times, without reordering.
For example if the input number is 2, and the input list is [1,2,3,1,2,1,2,3], you take [1,2,3,1,2], drop the next [1,2] since this would lead to 1 and 2 being in the result 3 times, and then take 3, which leads to [1,2,3,1,2,3].
With list [20,37,20,21] and number 1, the result would be [20,37,21].
"""
(delete_nth([20, 37, 20, 21], 1), [20, 37, 21])


def delete_nth(numbers, occurence):
    seen_numbers = {}
    numbers_limited_by_occurence = []

    for number in numbers:
        if seen_numbers.get(number, 0) != occurence:
            seen_numbers[number] = seen_numbers.get(number, 0) + 1
            numbers_limited_by_occurence.append(number)

    return numbers_limited_by_occurence





# Sort and Star
# https://www.codewars.com/kata/57cfdf34902f6ba3d300001e/train/python
"""
You will be given a list of strings. You must sort it alphabetically (case-sensitive, and based on the ASCII values of the chars) and then return the first value.

The returned value must be a string, and have "***" between each of its letters.

You should not remove or add elements from/to the array.
"""
(two_sort(["bitcoin", "take", "over", "the", "world", "maybe", "who", "knows", "perhaps"]), 'b***i***t***c***o***i***n' )
(two_sort(["turns", "out", "random", "test", "cases", "are", "easier", "than", "writing", "out", "basic", "ones"]), 'a***r***e')
(two_sort(["lets", "talk", "about", "javascript", "the", "best", "language"]), 'a***b***o***u***t')
(two_sort(["i", "want", "to", "travel", "the", "world", "writing", "code", "one", "day"]), 'c***o***d***e')
(two_sort(["Lets", "all", "go", "on", "holiday", "somewhere", "very", "cold"]), 'L***e***t***s')


def two_sort(array):
    return "***".join(min(array))





# Find the capitals
# https://www.codewars.com/kata/539ee3b6757843632d00026b/train/python
"""
Instructions
Write a function that takes a single non-empty string of only lowercase and uppercase ascii letters (word) as its argument, and returns an ordered list containing the indices of all capital (uppercase) letters in the string.

Example (Input --> Output)
"CodEWaRs" --> [0,3,4,6]
"""
(capitals('CodEWaRs'), [0, 3, 4, 6])


def capitals(word):
    return [index 
            for index, letter in enumerate(word) 
            if letter.isupper()]


def capitals(word):
    upper_tuple = filter(lambda x: x[1].isupper(), enumerate(word))

    return [index for index, _ in upper_tuple]


# oldies
def capitals(word):
    return [ind for ind, letter in enumerate(word) if letter.isupper()]
    # return list(zip(*filter(lambda x: x[1].isupper(), enumerate(word))))[0]





# List Filtering
# https://www.codewars.com/kata/53dbd5315a3c69eed20002dd/train/python
"""
In this kata you will create a function that takes a list of non-negative integers and strings and returns a new list with the strings filtered out.

Example
filter_list([1,2,'a','b']) == [1,2]
filter_list([1,'a','b',0,15]) == [1,0,15]
filter_list([1,2,'aasf','1','123',123]) == [1,2,123]
"""
(filter_list([1, 2, 'a', 'b']), [1, 2])
(filter_list([1, 'a', 'b', 0, 15]), [1, 0, 15])
(filter_list([1, 2, 'aasf', '1', '123', 123]), [1, 2, 123])


def filter_list(items):
    return list(filter(lambda item: type(item) == int, items))


def filter_list(items):
    return [item for item in items if type(item) == int]


# oldies
def filter_list(l):
    return list(filter(lambda x: isinstance(x, int) , l))
    # return [alnum for alnum in l if isinstance(alnum, int)]





# Give me a Diamond
# https://www.codewars.com/kata/5503013e34137eeeaa001648/train/python
"""
Jamie is a programmer, and James' girlfriend. She likes diamonds, and wants a diamond string from James. Since James doesn't know how to make this happen, he needs your help.

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

"  *\n ***\n*****\n ***\n  *\n"
"""
(diamond(5), "  *\n ***\n*****\n ***\n  *\n")
(diamond(1), "*\n")
(diamond(3), " *\n***\n *\n")
(diamond(0), None)
(diamond(2), None)
(diamond(-3), None)


def diamond(number):
    if (not number % 2 or
            number < 0):
        return None

    half = number // 2

    upper_diamond = [" " * (half - index) + ("*") * ((index * 2) + 1)
                     for index in range(half + 1)]
    
    lower_diamond = [" " * (half - index) + ("*") * ((index * 2) + 1)
                     for index in range(half)[::-1]]

    return "\n".join(upper_diamond + lower_diamond) + "\n"


# oldies
# Computes '  *  \n *** \n*****\n *** \n  *  ' instead of "  *\n ***\n*****\n ***\n  *\n"
def diamond(n):
    # upper = "\n".join([("*"*(2*ind + 1)).center(n) for ind in range((n // 2) + 1)])
    lower = "\n".join([("*"*(2*ind + 1)).center(n) for ind in reversed(range((n // 2) ))])
    upper = "\n".join([f"{'*'*(2*ind + 1):^{n}}" for ind in range((n // 2) + 1)])
    # lower = "\n".join([f"{'*'*(2*ind + 1):^{n}}" for ind in reversed(range((n // 2) ))])
    return upper + "\n" + lower





# Round up to the next multiple of 5
# https://www.codewars.com/kata/55d1d6d5955ec6365400006d/train/python
"""
Given an integer as input, can you round it to the next (meaning, "greater than or equal") multiple of 5?

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

You can assume that all inputs are valid integers.
"""
(round_to_next5(0), 0)
(round_to_next5(2), 5)
(round_to_next5(3), 5)
(round_to_next5(12), 15)
(round_to_next5(21), 25)
(round_to_next5(30), 30)
(round_to_next5(-2), 0)
(round_to_next5(-5), -5)


def round_to_next5(number):
    return (number // 5 * 5) + (5 if number % 5 else 0)


def round_to_next5(number):
    return ((number - 1) // 5 + 1) * 5


# oldies
def round_to_next5(n):
    return n if not n % 5 else (n // 5 + 1) * 5

import numpy as np
def round_to_next5(n):
    return int(np.ceil(n/5) * 5)

def round_to_next5(n):
    return n + (5 - n) % 5





# Quarter of the year
# https://www.codewars.com/kata/5ce9c1000bab0b001134f5af/train/python
"""
Given a month as an integer from 1 to 12, return to which quarter of the year it belongs as an integer number.

For example: month 2 (February), is part of the first quarter; month 6 (June), is part of the second quarter; and month 11 (November), is part of the fourth quarter.

Constraint:

1 <= month <= 12
"""
(quarter_of(3), 1)
(quarter_of(8), 3)
(quarter_of(11), 4)


def quarter_of(month):
    return ((month - 1) // 3) + 1





# Sum of two lowest positive integers
# https://www.codewars.com/kata/558fc85d8fd1938afb000014/train/python
# return
"""
Create a function that returns the sum of the two lowest positive numbers given an array of minimum 4 positive integers. No floats or non-positive integers will be passed.

For example, when an array is passed like [19, 5, 42, 2, 77], the output should be 7.

[10, 343445353, 3453445, 3453545353453] should return 3453455.
"""
(sum_two_smallest_numbers([5, 8, 12, 18, 22]), 13)
(sum_two_smallest_numbers([7, 15, 12, 18, 22]), 19)
(sum_two_smallest_numbers([25, 42, 12, 18, 22]), 30)


def sum_two_smallest_numbers(numbers):
    min_number = min(numbers)
    numbers.remove(min_number)

    return min_number + min(numbers)





# Simple Encryption #1 - Alternating Split
# https://www.codewars.com/kata/57814d79a56c88e3e0000786/solutions
"""
Implement a pseudo-encryption algorithm which given a string S and an integer N concatenates all the odd-indexed characters of S with all the even-indexed characters of S, this process should be repeated N times.

Examples:

encrypt("012345", 1)  =>  "135024"
encrypt("012345", 2)  =>  "135024"  ->  "304152"
encrypt("012345", 3)  =>  "135024"  ->  "304152"  ->  "012345"

encrypt("01234", 1)  =>  "13024"
encrypt("01234", 2)  =>  "13024"  ->  "32104"
encrypt("01234", 3)  =>  "13024"  ->  "32104"  ->  "20314"
Together with the encryption function, you should also implement a decryption function which reverses the process.

If the string S is an empty value or the integer N is not positive, return the first argument without changes.
"""
(encrypt("012345", 1), "135024")
(encrypt("01234", 1), "13024")
(encrypt("This is a test!", 0), "This is a test!")
(encrypt("This is a test!", 1), "hsi  etTi sats!")
(encrypt("This is a test!", 2), "s eT ashi tist!")
(encrypt("This is a test!", 3), " Tah itse sits!")
(encrypt("This is a test!", 4), "This is a test!")
(encrypt("This is a test!", -1), "This is a test!")
(encrypt("This kata is very interesting!", 1), "hskt svr neetn!Ti aai eyitrsig")
(encrypt("", 0), "")
(encrypt(None, 0), None)


def encrypt(text, number):
    for _ in range(number):
        text = text[1::2] + text[::2]

    return text


def encrypt(text, number):
    for _ in range(number):
        even_text = ""
        odd_text = ""

        for index, letter in enumerate(text):
            if index % 2:
                odd_text += letter
            else:
                even_text += letter

            text = odd_text + even_text

    return text


def encrypt(text, n):
    if n <= 0:
        return text
    return encrypt(text[1::2] + text[::2], n-1)


(decrypt("This is a test!", 0), "This is a test!")
(decrypt("hsi  etTi sats!", 1), "This is a test!")
(decrypt("s eT ashi tist!", 2), "This is a test!")
(decrypt(" Tah itse sits!", 3), "This is a test!")
(decrypt("This is a test!", 4), "This is a test!")
(decrypt("This is a test!", -1), "This is a test!")
(decrypt("hskt svr neetn!Ti aai eyitrsig", 1), "This kata is very interesting!")
(decrypt("135024", 1), "012345")
(decrypt("13024", 1), "01234")
(decrypt("304152", 2), "012345")
(decrypt("32104", 2), "01234")
(decrypt("", 0), "")
(decrypt(None, 0), None)


def decrypt(text, number):
    if text == "":
        return ""
    if text == None:
        return None
    
    middle = len(text) // 2
    text = list(text)
        
    for _ in range(number):
        text[::2], text[1::2] = text[middle:], text[:middle]

    return "".join(text)


def decrypt(text, number):
    for _ in range(number):
        odd_text = text[: len(text) // 2]
        even_text = text[len(text) // 2:]
        new_text = ""

        for index in range(len(even_text)):
            new_text += even_text[index]

            if index < len(odd_text):
                new_text += odd_text[index]

        text = new_text

    return text


def decrypt(text, n):
    for _ in range(n):
        odd = text[:len(text)//2]
        even = text[len(text)//2:]
        solution = ""
        for i in range(len(even)):
            solution += even[i:i+1]
            solution += odd[i:i+1]
        text = solution
    return text


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





# Complementary DNA
# https://www.codewars.com/kata/554e4a2f232cdd87d9000038/train/python
"""
Deoxyribonucleic acid (DNA) is a chemical found in the nucleus of cells and carries the "instructions" for the development and functioning of living organisms.

If you want to know more: http://en.wikipedia.org/wiki/DNA

In DNA strings, symbols "A" and "T" are complements of each other, as "C" and "G". Your function receives one side of the DNA (string, except for Haskell); you need to return the other complementary side. DNA strand is never empty or there is no DNA at all (again, except for Haskell).

More similar exercise are found here: http://rosalind.info/problems/list-view/ (source)

Example: (input --> output)

"ATTGC" --> "TAACG"
"GTAT" --> "CATA"
"""
(DNA_strand("AAAA"),"TTTT")
(DNA_strand("ATTGC"),"TAACG")
(DNA_strand("GTAT"),"CATA")


def DNA_strand(dna):
    to_complement = {
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C"
    }

    # return "".join(map(lambda letter: to_complement[letter], dna))
    return "".join(to_complement[letter]
                   for letter in dna)


# oldies
def DNA_strand(dna):
    return dna.replace("A", "Ą").replace("T", "A").replace("Ą", "T").replace("C", "Ą").replace("G", "C").replace("Ą", "G")

def DNA_strand(dna):
    return dna.translate(str.maketrans("ATCG","TAGC"))





# Super Duper Easy
# https://www.codewars.com/kata/55a5bfaa756cfede78000026/solutions
"""
Make a function that returns the value multiplied by 50 and increased by 6. If the value entered is a string it should return "Error".
"""
(problem(1), 56)
(problem("hello"), "Error")


def problem(element):
    if type(element) in (int, float):
        return 50 * element + 6
    else:
        return "Error"





# Add Length
# https://www.codewars.com/kata/559d2284b5bb6799e9000047/train/python
"""
What if we need the length of the words separated by a space to be added at the end of that same word and have it returned as an array?

Example(Input --> Output)

"apple ban" --> ["apple 5", "ban 3"]
"you will win" -->["you 3", "will 4", "win 3"]
Your task is to write a function that takes a String and returns an Array/list with the length of each word added to each element .

Note: String will have at least one element; words will always be separated by a space."""
(add_length('apple ban'), ["apple 5", "ban 3"])
(add_length('you will win'), ["you 3", "will 4", "win 3"])
(add_length('you'), ["you 3"])
(add_length('y'), ["y 1"])


def add_length(text):
    return [f"{word} {len(word)}"
            for word in text.split()]


def add_length(text):
    return list(map(lambda word: f"{word} {len(word)}", text.split()))





# Regex validate PIN code
# https://www.codewars.com/kata/55f8a9c06c018a0d6e000132/train/python
"""
ATM machines allow 4 or 6 digit PIN codes and PIN codes cannot contain anything but exactly 4 digits or exactly 6 digits.

If the function is passed a valid PIN string, return true, else return false.

Examples (Input --> Output)
"1234"   -->  true
"12345"  -->  false
"a234"   -->  false
"""
(validate_pin("1234"), True)
(validate_pin("1"), False)
(validate_pin("12"), False)
(validate_pin("123"), False)
(validate_pin("12345"), False)
(validate_pin("1234567"), False)
(validate_pin("-1234"), False)
(validate_pin("-12345"), False)
(validate_pin("1.234"), False)
(validate_pin("00000000"), False)
(validate_pin("0000"), True)
(validate_pin("12.0"), False)


def validate_pin(pin):
    return (
        len(pin) in (4, 6) and
        pin.isdigit()
    )





# Rot13
# https://www.codewars.com/kata/530e15517bc88ac656000716/train/python
"""
ROT13 is a simple letter substitution cipher that replaces a letter with the letter 13 letters after it in the alphabet. ROT13 is an example of the Caesar cipher.

Create a function that takes a string and returns the string ciphered with Rot13. If there are numbers or special characters included in the string, they should be returned as they are. Only letters from the latin/english alphabet should be shifted, like in the original Rot13 "implementation".

Please note that using encode is considered cheating.
"""
(rot13("test"), "grfg")
(rot13("Test"), "Grfg")
(rot13("aA bB zZ 1234 *!?%"), "nN oO mM 1234 *!?%")


def encode_letter(letter):
    if not letter.isalpha():
        return letter
    
    # if lowercase
    if letter.islower():
        next_index = ord(letter) + 13
        
        return chr(next_index) if next_index <= ord("z") else chr(next_index - 26)
    
    # if uppercase
    if letter.isupper():
        next_index = ord(letter) + 13
        
        return chr(next_index) if next_index <= ord("Z") else chr(next_index - 26)


def rot13(message):
    return "".join(map(encode_letter, message))



import string

def rot13(message):
    letters_l = string.ascii_lowercase
    letters_u = string.ascii_uppercase
    return "".join(letters_l[(letters_l.find(i) + 13) % 26] 
                   if i.islower() else letters_u[(letters_u.find(i) + 13) % 26] 
                   if i.isupper() else i for i in message)


def rot13(message):
    letters = 2 * string.ascii_lowercase + 2 * string.ascii_uppercase
    return "".join(letters[(letters.find(i) + 13)] if i in letters else i for i in message)





# Difference of Volumes of Cuboids
# https://www.codewars.com/kata/58cb43f4256836ed95000f97/train/python
"""
In this simple exercise, you will create a program that will take two lists of integers, a and b. Each list will consist of 3 positive integers above 0, representing the dimensions of cuboids a and b. You must find the difference of the cuboids' volumes regardless of which is bigger.

For example, if the parameters passed are ([2, 2, 3], [5, 4, 1]), the volume of a is 12 and the volume of b is 20. Therefore, the function should return 8.

Your function will be tested with pre-made examples as well as random ones.

If you can, try writing it in one line of code.
"""
(find_difference([3, 2, 5], [1, 4, 4]), 14)
(find_difference([9, 7, 2], [5, 2, 2]), 106)


def find_difference(a, b):
    return abs(a[0]*a[1]*a[2] - b[0]*b[1]*b[2])


import numpy as np
def find_difference(a, b):
    return abs(np.prod(a) - np.prod(b))





# Multiplication table for number
# https://www.codewars.com/kata/5a2fd38b55519ed98f0000ce/train/python
"""
Your goal is to return multiplication table for number that is always an integer from 1 to 10.

For example, a multiplication table (string) for number == 5 looks like below:

1 * 5 = 5
2 * 5 = 10
3 * 5 = 15
4 * 5 = 20
5 * 5 = 25
6 * 5 = 30
7 * 5 = 35
8 * 5 = 40
9 * 5 = 45
10 * 5 = 50
P. S. You can use \n in string to jump to the next line.
"""
(multi_table(5), '1 * 5 = 5\n2 * 5 = 10\n3 * 5 = 15\n4 * 5 = 20\n5 * 5 = 25\n6 * 5 = 30\n7 * 5 = 35\n8 * 5 = 40\n9 * 5 = 45\n10 * 5 = 50')
(multi_table(1), '1 * 1 = 1\n2 * 1 = 2\n3 * 1 = 3\n4 * 1 = 4\n5 * 1 = 5\n6 * 1 = 6\n7 * 1 = 7\n8 * 1 = 8\n9 * 1 = 9\n10 * 1 = 10')


def multi_table(number):
    return "\n".join(f"{index} * {number} = {index * number}" 
            for index in range(1, 11))


def product_line(index, number):
    return  f"{index} * {number} = {index * number}"

def multi_table(number):
    return "\n".join(map(lambda index: product_line(index, number), range(1, 11)))





# Beginner - Reduce but Grow
# https://www.codewars.com/kata/57f780909f7e8e3183000078/train/python
"""
Given a non-empty array of integers, return the result of multiplying the values together in order. Example:

[1, 2, 3, 4] => 1 * 2 * 3 * 4 = 24
"""
(grow([596, 321, 384, 132, 649, 224, 157, 729, 531, 115, 555, 540, 838, 243, 168, 538, 667, 53, 613, 10, 486, 437, 143, 167, 722, 422, 866, 534, 261, 3, 505, 881, 228, 203, 102, 237, 152, 32, 320, 731, 943, 723, 557, 730, 525, 353, 606, 274, 613, 582, 96, 585, 120, 426, 513, 232, 52, 584, 591, 806, 417, 70, 96, 902, 538, 681, 341, 927, 301, 97, 812, 30, 32, 405, 923, 730, 118, 64, 795, 226, 840, 653, 388, 784, 651, 332, 418, 682, 241, 781, 592, 150, 23, 134, 144, 626, 536, 482, 195, 811, 208, 707, 344, 735, 535, 945, 277, 725, 627, 135, 304, 585, 580, 409, 556, 839, 659, 1000, 861, 416, 914, 571, 691, 554, 502, 399, 111, 885, 142, 445, 880, 592, 255, 662, 423, 682, 384, 103, 722, 847, 320, 189, 693, 653, 230, 411, 204, 608, 481, 814, 297, 394, 864, 594, 549, 197, 628, 40, 185, 485, 162, 671, 395, 585, 428, 733, 811, 854, 344, 660, 475, 936, 807, 879, 404, 704, 515, 919, 549, 178, 942, 307, 553, 840, 451, 335, 44, 531, 726, 547, 238, 549, 987, 345, 368, 549, 107, 932, 521, 23, 95, 35, 488, 513, 573, 667, 388, 951, 654, 685, 974, 142, 533, 370, 847, 574, 848, 729, 237, 968, 185, 738, 660, 827, 110, 930, 497, 390, 542, 535, 223, 366, 98, 194, 771, 948, 864, 200, 801, 588, 468, 672, 61, 689, 893, 980, 525, 365, 593, 948, 920, 593, 939, 288, 424, 886, 203, 229, 911, 814, 157, 112, 390, 88, 312, 477, 35, 795, 959, 370, 950, 824, 456, 648, 184, 106, 896]), 39029291620533010931129397295638983676725789177704602999079246351960844311593368912916110759003546377363578700370528243472804365949406949497517632959300394381672629864499924571880170912950325917874291622475321852787750467982683462492413674077024767456900833305256057070814787993483437568900168885202988121573996032301947160895766083716920940392997066971600219779272071440586637340222479790337403296194462108557626985880202413695626127589511963083206906054187744691157794555256034617365518450400009739705212922378004697474916382004021566893720617727317115823822122740759685929320959643767556306049064966612902298901059029531313109353576527822848000000000000000000000000000000000000000000000000000000000000000000000000)


def grow(numbers):
    multi = 1

    for number in numbers:
        multi *= number

    return multi


import math  # np calculates wrong prodct
def grow(numbers):
    return math.prod(numbers)

from functools import reduce
def grow(numbers):
    return reduce((lambda x, y: x * y), numbers, 1)





# Exclamation marks series #11: Replace all vowel to exclamation mark in the sentence
# https://www.codewars.com/kata/57fb09ef2b5314a8a90001ed/train/python
"""
Description:
Replace all vowel to exclamation mark in the sentence. aeiouAEIOU is vowel.

Examples
replace("Hi!") === "H!!"
replace("!Hi! Hi!") === "!H!! H!!"
replace("aeiou") === "!!!!!"
replace("ABCDE") === "!BCD!"
"""
(replace_exclamation("Hi!") , "H!!")
(replace_exclamation("!Hi! Hi!") , "!H!! H!!")
(replace_exclamation("aeiou") , "!!!!!")
(replace_exclamation("ABCDE") , "!BCD!")


import re
def replace_exclamation(text):
    return re.sub(r"[aeoiu]", "!", text, flags=re.I)


def vovel_to_exclamation(letter):
    if letter in "aeoiuAEOIU":
        return "!"
    else:
        return letter

def replace_exclamation(text):
    return "".join(vovel_to_exclamation(letter) for letter in text)


def vovel_to_exclamation(letter):
    if letter in "aeoiuAEOIU":
        return "!"
    else:
        return letter

def replace_exclamation(text):
    return "".join(map(vovel_to_exclamation, text))





# Fix string case
# https://www.codewars.com/kata/5b180e9fedaa564a7000009a/train/python
"""
In this Kata, you will be given a string that may have mixed uppercase and lowercase letters and your task is to convert that string to either lowercase only or uppercase only based on:

make as few changes as possible.
if the string contains equal number of uppercase and lowercase letters, convert the string to lowercase.
For example:

solve("coDe") = "code". Lowercase characters > uppercase. Change only the "D" to lowercase.
solve("CODe") = "CODE". Uppercase characters > lowecase. Change only the "e" to uppercase.
solve("coDE") = "code". Upper == lowercase. Change all to lowercase.
"""
(solve("code"), "code")
(solve("CODe"), "CODE")
(solve("COde"), "code")
(solve("Code"), "code")


def solve(word):
    is_lower_list = [letter.islower() for letter in word]
    lower_percentage = sum(is_lower_list) / len(is_lower_list)

    if lower_percentage >= 0.5:
        return word.lower()
    else:
        return word.upper()


def solve(s):
    upper_count = len(list(filter(str.isupper, s)))
    lower_count = len(s) - upper_count
    return s.upper() if upper_count > lower_count else s.lower()





# Welcome to the City
# https://www.codewars.com/kata/5302d846be2a9189af0001e4/train/python
"""
Create a method that takes as input a name, city, and state to welcome a person. Note that name will be an array consisting of one or more values that should be joined together with one space between each, and the length of the name array in test cases will vary.

Example:

['John', 'Smith'], 'Phoenix', 'Arizona'
This example will return the string Hello, John Smith! Welcome to Phoenix, Arizona!

(say_hello(["John", "Smith"], "Phoenix", "Arizona"), "Hello, John Smith! Welcome to Phoenix, Arizona!")
(say_hello(["Franklin", "Delano", "Roosevelt"], "Chicago", "Illinois"), "Hello, Franklin Delano Roosevelt! Welcome to Chicago, Illinois!")
(say_hello(["Wallace", "Russel", "Osbourne"], "Albany", "New York"), "Hello, Wallace Russel Osbourne! Welcome to Albany, New York!")
(say_hello(["Lupin", "the", "Third"], "Los Angeles", "California"), "Hello, Lupin the Third! Welcome to Los Angeles, California!")
(say_hello(["Marlo", "Stanfield"], "Baltimore", "Maryland"), "Hello, Marlo Stanfield! Welcome to Baltimore, Maryland!")
"""
(say_hello(["John", "Smith"], "Phoenix", "Arizona"), "Hello, John Smith! Welcome to Phoenix, Arizona!")
(say_hello(["Franklin", "Delano", "Roosevelt"], "Chicago", "Illinois"), "Hello, Franklin Delano Roosevelt! Welcome to Chicago, Illinois!")
(say_hello(["Wallace", "Russel", "Osbourne"], "Albany", "New York"), "Hello, Wallace Russel Osbourne! Welcome to Albany, New York!")
(say_hello(["Lupin", "the", "Third"], "Los Angeles", "California"), "Hello, Lupin the Third! Welcome to Los Angeles, California!")
(say_hello(["Marlo", "Stanfield"], "Baltimore", "Maryland"), "Hello, Marlo Stanfield! Welcome to Baltimore, Maryland!")


def say_hello(name, city, state):
    return f"Hello, {' '.join(name)}! Welcome to {city}, {state}!"


# oldies
def say_hello(name, city, state):
    return "Hello, {}! Welcome to {}, {}!".format(' '.join(i for i in name), city, state)
    # return "Hello, %s! Welcome to %s, %s!" % (' '.join(i for i in name), city, state)





# Reversing Words in a String
# https://www.codewars.com/kata/57a55c8b72292d057b000594/train/python
"""
You need to write a function that reverses the words in a given string. A word can also fit an empty string. If this is not clear enough, here are some examples:

As the input may have trailing spaces, you will also need to ignore unneccesary whitespace.

Example (Input --> Output)

"Hello World" --> "World Hello"
"Hi There." --> "There. Hi"
"""
(reverse('Hello World'), 'World Hello')
(reverse('Hi There.'), 'There. Hi')


def reverse(text):
    return " ".join(text.split()[::-1])


def reverse(text):
    return " ".join(reversed(text.split()))





# Alternate capitalization
# https://www.codewars.com/kata/59cfc000aeb2844d16000075/train/python
"""
Given a string, capitalize the letters that occupy even indexes and odd indexes separately, and return as shown below. Index 0 will be considered even.

For example, capitalize("abcdef") = ['AbCdEf', 'aBcDeF']. See test cases for more examples.

The input will be a lowercase string with no spaces.

Good luck!

If you like this Kata, please try:
"""
(capitalize("abcdef"), ['AbCdEf', 'aBcDeF'])
(capitalize("codewars"), ['CoDeWaRs', 'cOdEwArS'])
(capitalize("abracadabra"), ['AbRaCaDaBrA', 'aBrAcAdAbRa'])
(capitalize("codewarriors"), ['CoDeWaRrIoRs', 'cOdEwArRiOrS'])
(capitalize("indexinglessons"), ['InDeXiNgLeSsOnS', 'iNdExInGlEsSoNs'])
(capitalize("codingisafunactivity"), ['CoDiNgIsAfUnAcTiViTy', 'cOdInGiSaFuNaCtIvItY'])


def capitalize(word):
    odd_word = ""
    even_word = ""

    for index, letter in enumerate(word):
        if index % 2:
            odd_word += letter.upper()
            even_word += letter
        else:
            odd_word += letter
            even_word += letter.upper()
    
    return [even_word, odd_word]


# oldies
def capitalize(s):
    first_element = "".join(elem.upper() if not i % 2 else elem for i, elem in enumerate(s))
    second_element = "".join(elem.upper() if i % 2 else elem for i, elem in enumerate(s))
    return [first_element, second_element]


def capitalize(s):
    one_elem = ((char.upper(), char) if not ind % 2 else (char, char.upper()) for ind, char in enumerate(s))
    return ["".join(i) for i in zip(*one_elem)]


def capitalize(s):
    first_element = "".join(elem.upper() if not i % 2 else elem for i, elem in enumerate(s))
    return [first_element, first_element.swapcase()]





# Tortoise racing
# https://www.codewars.com/kata/55e2adece53b4cdcb900006c/train/python
"""
Two tortoises named A and B must run a race. A starts with an average speed of 720 feet per hour. Young B knows she runs faster than A, and furthermore has not finished her cabbage.

When she starts, at last, she can see that A has a 70 feet lead but B's speed is 850 feet per hour. How long will it take B to catch A?

More generally: given two speeds v1 (A's speed, integer > 0) and v2 (B's speed, integer > 0) and a lead g (integer > 0) how long will it take B to catch A?

The result will be an array [hour, min, sec] which is the time needed in hours, minutes and seconds (round down to the nearest second) or a string in some languages.

If v1 >= v2 then return nil, nothing, null, None or {-1, -1, -1} for C++, C, Go, Nim, Pascal, COBOL, Erlang, [-1, -1, -1] for Perl,[] for Kotlin or "-1 -1 -1" for others.

Examples:
(form of the result depends on the language)

race(720, 850, 70) => [0, 32, 18] or "0 32 18"
race(80, 91, 37)   => [3, 21, 49] or "3 21 49"
"""
(race(720, 850, 70), [0, 32, 18])
(race(80, 91, 37), [3, 21, 49])
(race(820, 81, 550), None)
 

import numpy as np

def race(v1, v2, g):
    if v1 >= v2:
        return None
    
    x = (v1*g) / (v2 - v1)
    t = x / v1
    
    hours, rest = t // 1, t % 1
    minutes, rest = rest * 60 // 1, rest * 60 % 1
    seconds = np.floor(rest * 60 + 0.0000000001)  # problems with rounding
    return [hours, minutes, seconds]





# How many lightsabers do you own?
# https://www.codewars.com/kata/51f9d93b4095e0a7200001b8/train/python
"""
Inspired by the development team at Vooza, write the function that

accepts the name of a programmer, and
returns the number of lightsabers owned by that person.
The only person who owns lightsabers is Zach, by the way. He owns 18, which is an awesome number of lightsabers. Anyone else owns 0.

Note: your function should have a default parameter.

For example(Input --> Output):

"anyone else" --> 0
"Zach" --> 18
"""
(how_many_light_sabers_do_you_own("Zach"), 18)
(how_many_light_sabers_do_you_own(), 0)
(how_many_light_sabers_do_you_own("zach"), 0)


def how_many_light_sabers_do_you_own(name=""):
    return 18 if name == "Zach" else 0

how_many_light_sabers_do_you_own = lambda x="": 18 if x == "Zach" else 0





# Enumerable Magic - Does My List Include This?
# https://www.codewars.com/kata/545991b4cbae2a5fda000158/train/python
"""
Create a method that accepts a list and an item, and returns true if the item belongs to the list, otherwise false.
"""
numebers = [0, 1, 2, 3, 5, 8, 13, 2, 2, 2, 11]
(include(numebers, 100), False)
(include(numebers, 2), True)
(include(numebers, 11), True)
(include(numebers, "2"), False)
(include(numebers, 0), True)
(include([], 0), False)


def include(item_list, item):
    return item in item_list


include = list.__contains__
[1, 2, 0].__contains__(0)





# Is the string uppercase?
# https://www.codewars.com/kata/56cd44e1aa4ac7879200010b/train/python
"""
Create a method to see whether the string is ALL CAPS.

Examples (input -> output)
"c" -> False
"C" -> True
"hello I AM DONALD" -> False
"HELLO I AM DONALD" -> True
"ACSKLDFJSgSKLDFJSKLDFJ" -> False
"ACSKLDFJSGSKLDFJSKLDFJ" -> True
In this Kata, a string is said to be in ALL CAPS whenever it does not contain any lowercase letter so any string containing no letters at all is trivially considered to be in ALL CAPS.
"""
(is_uppercase("c"), False)
(is_uppercase("C"), True)
(is_uppercase("hello I AM DONALD"), False)
(is_uppercase("HELLO I AM DONALD"), True)
(is_uppercase("$%&"), True)


def is_uppercase(text):
    return text.upper() == text


def is_uppercase(text):
    return all(letter.upper() == letter 
               for letter in text)


def is_uppercase(text):
    return not any(letter.islower() 
                   for letter in text)


def is_uppercase(text):
    return not any(map(str.islower, text))


def is_uppercase(text):
    return all(map(lambda letter: letter == letter.upper(), text))





# Remove anchor from URL
# https://www.codewars.com/kata/51f2b4448cadf20ed0000386/train/python
"""
Complete the function/method so that it returns the url with anything after the anchor (#) removed.

Examples
"www.codewars.com#about" --> "www.codewars.com"
"www.codewars.com?page=1" -->"www.codewars.com?page=1"
"""
(remove_url_anchor("www.codewars.com#about"), "www.codewars.com")
(remove_url_anchor("www.codewars.com/katas/?page=1#about"), "www.codewars.com/katas/?page=1")
(remove_url_anchor("www.codewars.com/katas/"), "www.codewars.com/katas/")


def remove_url_anchor(url):
    anchor_index = url.find("#")

    if anchor_index == -1:
        return url
    else:
        return url[:anchor_index]


def remove_url_anchor(url):
    return url.split("#")[0]


def remove_url_anchor(url): 
    return url.partition("#")[0]


import re

def remove_url_anchor(url):
    return re.sub(r"#.*", "", url)
    # return re.search(r"[^#]+", url).group(0)


import re

def remove_url_anchor(url):
    match = re.search(r"(.*)#.*", url)

    if match:
        return match.group(1)
    else:
        return url





# Find the Remainder
# https://www.codewars.com/kata/524f5125ad9c12894e00003f/train/python
"""
Task:
Write a function that accepts two integers and returns the remainder of dividing the larger value by the smaller value.

Division by zero should return an empty value.

Examples:
n = 17
m = 5
result = 2 (remainder of `17 / 5`)

n = 13
m = 72
result = 7 (remainder of `72 / 13`)

n = 0
m = -1
result = 0 (remainder of `0 / -1`)

n = 0
m = 1
result - division by zero (refer to the specifications on how to handle this in your language)
"""
(remainder(17, 5), 2)
(remainder(13, 72), 7)  # "The order the arguments are passed should not matter."
(remainder(1, 0), None)  # "Divide by zero should return None"
(remainder(0, 0), None)  # "Divide by zero should return None"
(remainder(0, 1), None)  # "Divide by zero should return None"
(remainder(-1, 0), 0)  # "Divide by zero should only be checked for the lowest number"
(remainder(0, -1), 0)  # "Divide by zero should only be checked for the lowest number"
(remainder(-13, -13), 0)  # "Should handle negative numbers"
(remainder(-60, 340), -20)  # "Should handle negative numbers", in JS == 40
(remainder(60, -40), -20)  # "Should handle negative numbers", in JS == 20


def remainder(number1, number2):
    if number2 > number1:
        number1, number2 = number2, number1

    if not number2 :
        return None

    return number1 % number2


def remainder(number1, number2):
    if number2 > number1:
        number1, number2 = number2, number1

    try: 
        return number1 % number2
    except:
        None





# Regular Ball Super Ball
# https://www.codewars.com/kata/53f0f358b9cb376eca001079/train/python
"""
Create a class Ball. Ball objects should accept one argument for "ball type" when instantiated.

If no arguments are given, ball objects should instantiate with a "ball type" of "regular."

ball1 = Ball()
ball2 = Ball("super")
ball1.ball_type  #=> "regular"
ball2.ball_type  #=> "super"
"""
ball1 = Ball()
(ball1.ball_type, "regular")

ball2 = Ball("super")
(ball2.ball_type, "super")


class Ball(object):
    def __init__(self, ball_type="regular"):
        self.ball_type = ball_type


Ball().ball_type  # "regular"
Ball("super").ball_type  # "super"












# Find the stray number
# https://www.codewars.com/kata/57f609022f4d534f05000024/train/python
"""
You are given an odd-length array of integers, in which all of them are the same, except for one single number.

Complete the method which accepts such an array, and returns that single different number.

The input array will always be valid! (odd-length >= 3)

Examples
[1, 1, 2] ==> 2
[17, 17, 3, 17, 17, 17, 17] ==> 3
"""
(stray([1, 1, 1, 1, 1, 1, 2]), 2)
(stray([2, 3, 2, 2, 2]), 3)
(stray([3, 2, 2, 2, 2]), 3)


# O(n), early exit
def stray(numbers):
    seen_numbers = {}

    for number in numbers:
        seen_numbers[number] = seen_numbers.get(number, 0) + 1

        if len(seen_numbers) == 2:
            if sum(seen_numbers.values()) != 2:
                for key, val in seen_numbers.items():
                    if val == 1:
                        return key


def stray(numbers):
    return min(numbers, key=numbers.count)


def key(numbers, number):
    return numbers.count(number)

def stray(numbers):
    return min(numbers, key=lambda number: key(numbers, number))


def stray(numbers):
    for number in set(numbers):
        if numbers.count(number) == 1:
            return number





# Summing a number's digits
# https://www.codewars.com/kata/52f3149496de55aded000410/train/python
"""
Write a function named sum_digits which takes a number as input and returns the sum of the absolute value of each of the number's decimal digits.

For example: (Input --> Output)

10 --> 1
99 --> 18
-32 --> 5
"""
(sum_digits(10), 1)
(sum_digits(99), 18)
(sum_digits(-32), 5)


def sum_digits(number):
    return sum(int(digit) 
               for digit in str(abs(number)))


def sum_digits(number):
    return sum(map(int, str(abs(number))))


def sum_digits(number):
    return sum(int(digit) 
               for digit in str(number) 
               if digit.isdigit())


def sum_digits(number):
    return sum(map(lambda digit: int(digit), filter(str.isdigit, str(number))))




# Sum of odd numbers
# https://www.codewars.com/kata/55fd2d567d94ac3bc9000064/train/python
"""
Given the triangle of consecutive odd numbers:

             1
          3     5
       7     9    11
   13    15    17    19
21    23    25    27    29
...
Calculate the sum of the numbers in the nth row of this triangle (starting at index 1) e.g.: (Input --> Output)

1 -->  1
2 --> 3 + 5 = 8
3 --> 7 + 9 + 11 = 27
"""
(row_sum_odd_numbers(1), 1)
(row_sum_odd_numbers(2), 8)
(row_sum_odd_numbers(3), 27)
(row_sum_odd_numbers(13), 2197)
(row_sum_odd_numbers(19), 6859)
(row_sum_odd_numbers(41), 68921)


def row_sum_odd_numbers(number):
    return number ** 3


def row_sum_odd_numbers(number):
    start = sum(range(1, number))
    end = sum(range(1, number + 1))

    return sum(range(start * 2 + 1, end * 2, 2))





# Mumbling
# https://www.codewars.com/kata/5667e8f4e3f572a8f2000039/train/python
"""
This time no story, no theory. The examples below show you how to write function accum:

Examples:
accum("abcd") -> "A-Bb-Ccc-Dddd"
accum("RqaEzty") -> "R-Qq-Aaa-Eeee-Zzzzz-Tttttt-Yyyyyyy"
accum("cwAt") -> "C-Ww-Aaa-Tttt"
The parameter of accum is a string which includes only letters from a..z and A..Z.
"""
(accum("ZpglnRxqenU"), "Z-Pp-Ggg-Llll-Nnnnn-Rrrrrr-Xxxxxxx-Qqqqqqqq-Eeeeeeeee-Nnnnnnnnnn-Uuuuuuuuuuu")
(accum("NyffsGeyylB"), "N-Yy-Fff-Ffff-Sssss-Gggggg-Eeeeeee-Yyyyyyyy-Yyyyyyyyy-Llllllllll-Bbbbbbbbbbb")
(accum("MjtkuBovqrU"), "M-Jj-Ttt-Kkkk-Uuuuu-Bbbbbb-Ooooooo-Vvvvvvvv-Qqqqqqqqq-Rrrrrrrrrr-Uuuuuuuuuuu")
(accum("EvidjUnokmM"), "E-Vv-Iii-Dddd-Jjjjj-Uuuuuu-Nnnnnnn-Oooooooo-Kkkkkkkkk-Mmmmmmmmmm-Mmmmmmmmmmm")
(accum("HbideVbxncC"), "H-Bb-Iii-Dddd-Eeeee-Vvvvvv-Bbbbbbb-Xxxxxxxx-Nnnnnnnnn-Cccccccccc-Ccccccccccc")

def accum(text):
    return "-".join((letter * index).capitalize()
                    for index, letter in enumerate(text, 1))





# Remove First and Last Character Part Two
# https://www.codewars.com/kata/570597e258b58f6edc00230d/train/python
"""
This is a spin off of my first kata.

You are given a string containing a sequence of character sequences separated by commas.

Write a function which returns a new string containing the same character sequences except the first and the last ones but this time separated by spaces.

If the input string is empty or the removal of the first and last items would cause the resulting string to be empty, return an empty value (represented as a generic value NULL in the examples below).

Examples
"1,2,3"      =>  "2"
"1,2,3,4"    =>  "2 3"
"1,2,3,4,5"  =>  "2 3 4"

""     =>  NULL
"1"    =>  NULL
"1,2"  =>  NULL
"""
(array("1,2,3"), "2")
(array("1,2,3,4"), "2 3")
(array("1, 2, 3, 4"), "2 3")
(array("1,2,3,4,5"), "2 3 4")
(array(""), None)
(array("1"), None)
(array("1,2"), None)


def array(text):
    if text.count(",") < 2:
        return None
    else:
        return " ".join([digit.strip() 
                         for digit in text.split(',')][1:-1])


def array(text):
    return " ".join([digit.strip() 
                    for digit in text.split(',')][1:-1]) or None





# Sum of Multiples
# https://www.codewars.com/kata/57241e0f440cd279b5000829/train/python
"""
Your Job
Find the sum of all multiples of n below m

Keep in Mind
n and m are natural numbers (positive integers)
m is excluded from the multiples
Examples
sum_mul(2, 9)   ==> 2 + 4 + 6 + 8 = 20
sum_mul(3, 13)  ==> 3 + 6 + 9 + 12 = 30
sum_mul(4, 123) ==> 4 + 8 + 12 + ... = 1860
sum_mul(4, -7)  ==> "INVALID"
"""
(sum_mul(2, 9), 20)
(sum_mul(3, 13), 30)
(sum_mul(4, 123), 1860)
(sum_mul(123, 4567), 86469)
(sum_mul(0, 0), "INVALID")
(sum_mul(4, -7), "INVALID")
(sum_mul(11, 1111), 55550)



def sum_mul(start, end):
    if (start <= 0 or
            end <= 0):
        return "INVALID"
    
    return sum(range(start, end, start))





# Sorted? yes? no? how?
# https://www.codewars.com/kata/580a4734d6df7480600000r45/train/python
"""
Complete the method which accepts an array of integers, and returns one of the following:

"yes, ascending" - if the numbers in the array are sorted in an ascending order
"yes, descending" - if the numbers in the array are sorted in a descending order
"no" - otherwise
You can assume the array will always be valid, and there will always be one correct answer.
"""
(is_sorted_and_how([1, 2]), "yes, ascending")
(is_sorted_and_how([15, 7, 3, -8]), "yes, descending")
(is_sorted_and_how([4, 2, 30]), "no")


def is_sorted_and_how(numbers):
    if numbers == sorted(numbers):
        return "yes, ascending"
    elif numbers == sorted(numbers, reverse=True):
        return "yes, descending"
    else:
        return "no"





# Find Multiples of a Number
# https://www.codewars.com/kata/58ca658cc0d6401f2700045f/train/python
"""
In this simple exercise, you will build a program that takes a value, integer , and returns a list of its multiples up to another value, limit . If limit is a multiple of integer, it should be included as well. There will only ever be positive integers passed into the function, not consisting of 0. The limit will always be higher than the base.
For example, if the parameters passed are (2, 6), the function should return [2, 4, 6] as 2, 4, and 6 are the multiples of 2 up to 6.
"""
(find_multiples(5, 25), [5, 10, 15, 20, 25])
(find_multiples(1, 2), [1, 2])


def find_multiples(start, end):
    return list(range(start, end + 1, start))





# Regex Password Validation
# https://www.codewars.com/kata/52e1476c8147a7547a000811/train/python
"""
You need to write regex that will validate a password to make sure it meets the following criteria:

At least six characters long
contains a lowercase letter
contains an uppercase letter
contains a digit
only contains alphanumeric characters (note that '_' is not alphanumeric)
"""
data_table = (
    ("fjd3IR9", True),
    ("ghdfj32", False),
    ("DSJKHD23", False),
    ("dsF43", False),
    ("4fdg5Fj3", True),
    ("DHSJdhjsU", False),
    ("fjd3IR9.;", False),
    ("fjd3  IR9", False),
    ("djI38D55", True),
    ("a2.d412", False),
    ("JHD5FJ53", False),
    ("!fdjn345", False),
    ("jfkdfj3j", False),
    ("123", False),
    ("abc", False),
    ("123abcABC", True),
    ("ABC123abc", True),
    ("Password123", True),
)

import re

for word, is_true in data_table:
    try:
        sol = re.fullmatch(regex, word).group(0)
        print(True, is_true)
    except:
        print(False, is_true)


regex="^(?=.*?[a-z])(?=.*?[A-Z])(?=.*?\d+)[A-Za-z\d]{6,}$"





# Help the bookseller !
# https://www.codewars.com/kata/54dc6f5a224c26032800005c/train/python
"""
A bookseller has lots of books classified in 26 categories labeled A, B, ... Z. Each book has a code c of 3, 4, 5 or more characters. The 1st character of a code is a capital letter which defines the book category.

In the bookseller's stocklist each code c is followed by a space and by a positive integer n (int n >= 0) which indicates the quantity of books of this code in stock.

For example an extract of a stocklist could be:

L = {"ABART 20", "CDXEF 50", "BKWRK 25", "BTSQZ 89", "DRTYM 60"}.
or
L = ["ABART 20", "CDXEF 50", "BKWRK 25", "BTSQZ 89", "DRTYM 60"] or ....
You will be given a stocklist (e.g. : L) and a list of categories in capital letters e.g :

M = {"A", "B", "C", "W"} 
or
M = ["A", "B", "C", "W"] or ...
and your task is to find all the books of L with codes belonging to each category of M and to sum their quantity according to each category.

For the lists L and M of example you have to return the string (in Haskell/Clojure/Racket/Prolog a list of pairs):

(A : 20) - (B : 114) - (C : 50) - (W : 0)
"""
(stock_list(["BBAR 150", "CDXE 515", "BKWR 250", "BTSQ 890", "DRTY 600"], ["A", "B", "C", "D"]), "(A : 0) - (B : 1290) - (C : 515) - (D : 600)")
(stock_list(["ABAR 200", "CDXE 500", "BKWR 250", "BTSQ 890", "DRTY 600"], ["A", "B"]), "(A : 200) - (B : 1140)")
(stock_list([""], ["B", "R", "D", "X"]), "")

def stock_list(book_quantity_list, category_list):
    if (not book_quantity_list or 
            not category_list or
            not book_quantity_list[0]):
        return ""
    
    category_quantities = {}

    # group book quantities by category
    for book_quantity in book_quantity_list:
        book, quantity = book_quantity.split(" ")
        category = book[0]
        category_quantities[category] = category_quantities.get(category, 0) + int(quantity)

    # Format the result with only the requested categories
    return " - ".join(f"({category} : {category_quantities.get(category, 0)})" 
                      for category in category_list)





# Factorial
# https://www.codewars.com/kata/57a049e253ba33ac5e000212/train/python
"""Your task is to write function factorial.

https://en.wikipedia.org/wiki/Factorial

"""
tests = (
    (0, 1),
    (1, 1),
    (2, 2),
    (3, 6),
    (4, 24),
    (5, 120),
    (6, 720),
    (7, 5040),
)

for t in tests:
    inp, exp = t
    print(factorial(inp), exp)


# recursion
def factorial(number):
    if number == 0:
        return 1
    
    return number * factorial(number - 1)


# iteration
def factorial(number):
    total = 1

    for index in range(1, number + 1):
        total *= index
    
    return total


import numpy as np
factorial = np.math.factorial





# Remove duplicates from list
# https://www.codewars.com/kata/57a5b0dfcf1fa526bb000118/train/python
"""
Define a function that removes duplicates from an array of non negative numbers and returns it as a result.

The order of the sequence has to stay the same.

Examples:

Input -> Output
[1, 1, 2] -> [1, 2]
[1, 2, 1, 1, 3, 2] -> [1, 2, 3]
"""
(distinct([1]), [1])
(distinct([1, 2]), [1, 2])
(distinct([1, 1, 2]), [1, 2])
(distinct([1, 1, 1, 2, 3, 4, 5]), [1, 2, 3, 4, 5])
(distinct([1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7, 7]), [1, 2, 3, 4, 5, 6, 7])


def distinct(numbers):
    seen_numbers_set = set()
    seen_numbers_list = []

    for number in numbers:
        if number not in seen_numbers_set:
            seen_numbers_set.add(number)
            seen_numbers_list.append(number)
    
    return seen_numbers_list


def distinct(numbers):
    seen_numbers = set()

    return [number 
            for number in numbers 
            if not (number in seen_numbers or 
                    seen_numbers.add(number))]  # seen_numbers.add(number)) => None


# When converting a set back to a list, the order of elements may change.
def distinct(numbers):
    return list(set(numbers))





# Convert to Binary
# https://www.codewars.com/kata/59fca81a5712f9fa4700159a/train/python
"""
Task Overview
Given a non-negative integer n, write a function to_binary/ToBinary which returns that number in a binary format.

to_binary(1)  # should return 1 
to_binary(5)  # should return 101
to_binary(11) # should return 1011
"""
(to_binary(1), 1)
(to_binary(2), 10)
(to_binary(3), 11)
(to_binary(5), 101)


def to_binary(number):
    return int(bin(number)[2:])


def to_binary(number):
    return int(f"{number:>b}")





# Factorial
# https://www.codewars.com/kata/54ff0d1f355cfd20e60001fc/train/python
"""
In mathematics, the factorial of a non-negative integer n, denoted by n!, is the product of all positive integers less than or equal to n. For example: 5! = 5 * 4 * 3 * 2 * 1 = 120. By convention the value of 0! is 1. 
Write a function to calculate factorial for a given input. If input is below 0 or above 12 throw an exception of type ArgumentOutOfRangeException (C#) or IllegalArgumentException (Java) or RangeException (PHP) or throw a RangeError (JavaScript) or ValueError (Python) or return -1 (C).
"""
(factorial(0), 1)
(factorial(1), 1)
(factorial(2), 2)
(factorial(3), 6)
(factorial(13), "<class 'ValueError'>")


def factorial(number):
    if (number < 0 or
        number > 12):
        raise ValueError
    
    total = 1

    for index in range(2, number + 1):
        total *= index

    return total





# Simple Fun #176: Reverse Letter
# https://www.codewars.com/kata/58b8c94b7df3f116eb00005b/train/python
"""
Task
Given a string str, reverse it and omit all non-alphabetic characters.

Example
For str = "krishan", the output should be "nahsirk".

For str = "ultr53o?n", the output should be "nortlu".

(reverse_letter("krishan"),"nahsirk")
"""
(reverse_letter("krishan"),"nahsirk")
(reverse_letter("ultr53o?n"),"nortlu")
(reverse_letter("ab23c"),"cba")
(reverse_letter("krish21an"),"nahsirk")


def reverse_letter(word):
    return "".join(letter 
                   for letter in word[::-1] 
                   if letter.isalpha())


def reverse_letter(word):
    return "".join(filter(str.isalpha, word[::-1]))





# Alphabet war
# https://www.codewars.com/kata/59377c53e66267c8f6000027/train/python
"""
The left side letters and their power:

 w - 4
 p - 3
 b - 2
 s - 1
The right side letters and their power:

 m - 4
 q - 3
 d - 2
 z - 1
The other letters don't have power and are only victims.

Example
AlphabetWar("z");        //=> Right side wins!
AlphabetWar("zdqmwpbs"); //=> Let's fight again!
AlphabetWar("zzzzs");    //=> Right side wins!
AlphabetWar("wwwwwwz");  //=> Left side wins!
"""
(alphabet_war("z"), "Right side wins!")
(alphabet_war("zdqmwpbs"), "Let's fight again!")
(alphabet_war("wq"), "Left side wins!")
(alphabet_war("zzzzs"), "Right side wins!")
(alphabet_war("wwwwww"), "Left side wins!")
    

def alphabet_war(word):
    left_letters = "sbpw"
    right_letters = "zdqm"
    score = 0
    
    for letter in word:
        if letter in right_letters:
            score += right_letters.index(letter) + 1
        elif letter in left_letters:
            score -= left_letters.index(letter) + 1
    
    if score > 0:
        return "Right side wins!"
    elif score < 0:
        return "Left side wins!"
    else:
        return "Let's fight again!"


def alphabet_war(word):
    letter_score = {
        "w": -4,
        "p": -3,
        "b": -2,
        "s": -1,
        "m": 4,
        "q": 3,
        "d": 2,
        "z": 1
    }
    score = 0

    for letter in word:
        if letter in letter_score:
            score += letter_score[letter]

    if score < 0:
        return "Left side wins!"
    elif score > 0:
        return "Right side wins!"
    else:
        return "Let's fight again!"





# Predict your age!
# https://www.codewars.com/kata/5aff237c578a14752d0035ae/train/python
"""
In honor of my grandfather's memory we will write a function using his formula!

Take a list of ages when each of your great-grandparent died.
Multiply each number by itself.
Add them all together.
Take the square root of the result.
Divide by two.
Example
predict_age(65, 60, 75, 55, 60, 63, 64, 45) == 86
Note: the result should be rounded down to the nearest integer.
"""
(predict_age(65, 60, 75, 55, 60, 63, 64, 45), 86)


def predict_age(*args):
    return (sum(number ** 2 for number in args) ** 0.5) // 2


def predict_age(*args):
    return (sum(map(lambda number: number ** 2, args)) ** 0.5) // 2





# Multiplication table
# https://www.codewars.com/kata/534d2f5b5371ecf8d2000a08/train/python
"""
Your task, is to create N×N multiplication table, of size provided in parameter.

For example, when given size is 3:

1 2 3
2 4 6
3 6 9
For the given example, the return value should be:

[[1,2,3],[2,4,6],[3,6,9]]
"""
(multiplication_table(1), [[1]])
(multiplication_table(2), [[1, 2], [2, 4]])
(multiplication_table(3), [[1, 2, 3], [2, 4, 6], [3, 6, 9]])
(multiplication_table(4), [[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12], [4, 8, 12, 16]])
(multiplication_table(5), [[1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [3, 6, 9, 12, 15], [4, 8, 12, 16, 20], [5, 10, 15, 20, 25]])


def multiplication_table(size):
    row = range(1, size + 1)
    matrix = []

    for row_index in row:
        matrix.append([index * row_index 
                       for index in row])
    
    return matrix


def multiplication_table(size):
    row = range(1, size + 1)
    
    return [[index * row_index 
             for index in row] 
            for row_index in row]




e
# Integers: Recreation One
# https://www.codewars.com/kata/55aa075506463dac6600010d/train/python
"""
1, 246, 2, 123, 3, 82, 6, 41 are the divisors of number 246. Squaring these divisors we get: 1, 60516, 4, 15129, 9, 6724, 36, 1681. The sum of these squares is 84100 which is 290 * 290.

Task
Find all integers between m and n (m and n integers with 1 <= m <= n) such that the sum of their squared divisors is itself a square.

We will return an array of subarrays or of tuples (in C an array of Pair) or a string. The subarrays (or tuples or Pairs) will have two elements: first the number the squared divisors of which is a square and then the sum of the squared divisors.

Example:
list_squared(1, 250) --> [[1, 1], [42, 2500], [246, 84100]]
list_squared(42, 250) --> [[42, 2500], [246, 84100]]
"""
(list_squared(1, 1), [1, 1])
(list_squared(246, 246), [246, 84100])
(list_squared(1, 250), [[1, 1], [42, 2500], [246, 84100]])
(list_squared(42, 250), [[42, 2500], [246, 84100]])
(list_squared(250, 500), [[287, 84100]])


# O(nsqrt(n))
def get_divisors(number):
    divisors = set([1])

    for index in range(1, int(number ** 0.5) + 1):
        if not number % index:
            divisors.add(index)
            divisors.add(number // index)
    
    return divisors

get_divisors(246)

def get_squared_divisors_sum(number):
    return sum((num ** 2 for num in get_divisors(number)))

get_squared_divisors_sum(246)

def is_divisors_sum_square(number):
    return not get_squared_divisors_sum(number) ** 0.5 % 1

is_divisors_sum_square(246)

def list_squared(start, end):
    return [[number, get_squared_divisors_sum(number)] 
            for number in range(start, end + 1) 
            if is_divisors_sum_square(number)]


# Brute Force, O(nsqrt(n)), O(1)
def list_squared(m, n):
    sol = []
    
    for number in range(m, n + 1):
        divs_sum = 0
        
        for num in range(1, int(number**.5) + 1):
            if not number % num:
                divs_sum += num**2
                if num**2 != number:
                    divs_sum += (number//num)**2

        if not divs_sum ** .5 % 1:
            sol.append([number, divs_sum])
    
    return sol


# Brute Force, O(n2), O(1)
def list_squared(m, n):
    sol = []
    
    for number in range(m, n + 1):
        divs_sum = number**2
        
        for i in range(1, number//2 + 1):
            if not number % i:
                divs_sum += i**2

        if not divs_sum ** .5 % 1:
            sol.append([number, divs_sum])
    
    return sol





# Money, Money, Money
# https://www.codewars.com/kata/563f037412e5ada593000114/train/python
"""
Mr. Scrooge has a sum of money 'P' that he wants to invest. Before he does, he wants to know how many years 'Y' this sum 'P' has to be kept in the bank in order for it to amount to a desired sum of money 'D'.

The sum is kept for 'Y' years in the bank where interest 'I' is paid yearly. After paying taxes 'T' for the year the new sum is re-invested.

Note to Tax: not the invested principal is taxed, but only the year's accrued interest

Example:

  Let P be the Principal = 1000.00      
  Let I be the Interest Rate = 0.05      
  Let T be the Tax Rate = 0.18      
  Let D be the Desired Sum = 1100.00


After 1st Year -->
  P = 1041.00
After 2nd Year -->
  P = 1083.86
After 3rd Year -->
  P = 1128.30
"""
(calculate_years(1000, 0.05, 0.18, 1100), 3)
(calculate_years(1000, 0.01625, 0.18, 1200), 14)
(calculate_years(1000, 0.05, 0.18, 1000), 0)


def calculate_years(principal, interest, tax, desired):
    years = 0

    while  principal < desired:
        principal += principal * interest - principal * interest * tax
        years += 1
    
    return years





# Printing Array elements with Comma delimiters
# https://www.codewars.com/kata/56e2f59fb2ed128081001328/train/python
"""
Input: Array of elements

["h","o","l","a"]

Output: String with comma delimited elements of the array in th same order.

"h,o,l,a"

Note: if this seems too simple for you try the next level

Note2: the input data can be: boolean array, array of objects, array of string arrays, array of number arrays... 😕
"""
(print_array([2, 4, 5, 2]), "2,4,5,2")


def print_array(numbers):
    return ",".join(map(str, numbers))

def print_array(numbers):
    return ",".join(str(number) for number in numbers)





# Find the vowels
# https://www.codewars.com/kata/5680781b6b7c2be860000036/train/python
"""
We want to know the index of the vowels in a given word, for example, there are two vowels in the word super (the second and fourth letters).

So given a string "super", we should return a list of [2, 4].

Some examples:
Mmmm  => []
Super => [2,4]
Apple => [1,5]
YoMama -> [1,2,4,6]
"""
(vowel_indices("mmm"), [])
(vowel_indices("apple"), [1, 5])
(vowel_indices("123456"), [])
(vowel_indices("UNDISARMED"), [1, 4, 6, 9])


def vowel_indices(word):
    return [index + 1 
            for index, letter in enumerate(word) 
            if letter in "aeoiuAEOIU"]





# Sort Numbers
# https://www.codewars.com/kata/5174a4c0f2769dd8b1000003/train/python
"""
Finish the solution so that it sorts the passed in array of numbers. If the function passes in an empty array or null/nil value then it should return an empty array.

For example:

solution([1,2,3,10,5]) # should return [1,2,3,5,10]
solution(None) # should return []
"""
(solution([1, 2, 3, 10, 5]), [1, 2, 3, 5, 10])
(solution(None), [])
(solution([]), [])
(solution([20, 2, 10]), [2, 10, 20])
(solution([2, 20, 10]), [2, 10, 20])



def solution(numbers):
    if not numbers:
        return []

    numbers.sort()

    return numbers





# A Rule of Divisibility by 13
# https://www.codewars.com/kata/564057bc348c7200bd0000ff/train/python
"""
From Wikipedia:

"A divisibility rule is a shorthand way of determining whether a given integer is divisible by a fixed divisor without performing the division, usually by examining its digits."

When you divide the successive powers of 10 by 13 you get the following remainders of the integer divisions:

1, 10, 9, 12, 3, 4 because:

10 ^ 0 ->  1 (mod 13)
10 ^ 1 -> 10 (mod 13)
10 ^ 2 ->  9 (mod 13)
10 ^ 3 -> 12 (mod 13)
10 ^ 4 ->  3 (mod 13)
10 ^ 5 ->  4 (mod 13)
(For "mod" you can see: https://en.wikipedia.org/wiki/Modulo_operation)

Then the whole pattern repeats. Hence the following method:

Multiply

the right most digit of the number with the left most number in the sequence shown above,
the second right most digit with the second left most digit of the number in the sequence.
The cycle goes on and you sum all these products. Repeat this process until the sequence of sums is stationary.

Example:
What is the remainder when 1234567 is divided by 13?

7      6     5      4     3     2     1  (digits of 1234567 from the right)
×      ×     ×      ×     ×     ×     ×  (multiplication)
1     10     9     12     3     4     1  (the repeating sequence)
Therefore following the method we get:

 7×1 + 6×10 + 5×9 + 4×12 + 3×3 + 2×4 + 1×1 = 178

We repeat the process with the number 178:

8x1 + 7x10 + 1x9 = 87

and again with 87:

7x1 + 8x10 = 87

From now on the sequence is stationary (we always get 87) and the remainder of 1234567 by 13 is the same as the remainder of 87 by 13 ( i.e 9).

Task:
Call thirt the function which processes this sequence of operations on an integer n (>=0). thirt will return the stationary number.

thirt(1234567) calculates 178, then 87, then 87 and returns 87.

thirt(321) calculates 48, 48 and returns 48
"""
(thirt(1234567), 87)
(thirt(8529), 79)
(thirt(85299258), 31)
(thirt(5634), 57)
(thirt(1111111111), 71)
(thirt(987654321), 30)

def thirt(number):
    factors = (1, 10, 9, 12, 3, 4, 1)

    while len(str(number)) > 2:
        number = sum(int(digit) * factors[index % 6]
                     for index, digit in enumerate(str(number)[::-1]))

    return number
    




# Odd or Even?
# https://www.codewars.com/kata/5949481f86420f59480000e7/train/python
"""
Given a list of integers, determine whether the sum of its elements is odd or even.

Give your answer as a string matching "odd" or "even".

If the input array is empty consider it as: [0] (array with a zero).

Examples:
Input: [0]
Output: "even"

Input: [0, 1, 4]
Output: "odd"

Input: [0, -1, -5]
Output: "even"
"""
(odd_or_even([0, 1, 2]), "odd")
(odd_or_even([0, 1, 3]), "even")
(odd_or_even([1023, 1, 2]), "even")
(odd_or_even([]), "even")


def odd_or_even(numbers):
    return ("even", "odd")[sum(numbers) % 2]





# Check the exam
# https://www.codewars.com/kata/5a3dd29055519e23ec000074/train/python
"""
The first input array is the key to the correct answers to an exam, like ["a", "a", "b", "d"]. The second one contains a student's submitted answers.

The two arrays are not empty and are the same length. Return the score for this array of answers, giving +4 for each correct answer, -1 for each incorrect answer, and +0 for each blank answer, represented as an empty string (in C the space character is used).

If the score < 0, return 0.

For example:

    Correct answer    |    Student's answer   |   Result         
 ---------------------|-----------------------|-----------
 ["a", "a", "b", "b"]   ["a", "c", "b", "d"]  →     6
 ["a", "a", "c", "b"]   ["a", "a", "b", "" ]  →     7
 ["a", "a", "b", "c"]   ["a", "a", "b", "c"]  →     16
 ["b", "c", "b", "a"]   ["" , "a", "a", "c"]  →     0
"""
(check_exam(["a", "a", "b", "b"], ["a", "c", "b", "d"]), 6)
(check_exam(["a", "a", "c", "b"], ["a", "a", "b",  ""]), 7)
(check_exam(["a", "a", "b", "c"], ["a", "a", "b", "c"]), 16)
(check_exam(["b", "c", "b", "a"], ["",  "a", "a", "c"]), 0)


def check_exam(correct_answers, student_answers):
    score = 0

    for index in range(len(correct_answers)):
        if correct_answers[index] == student_answers[index]:
            score += 4
        elif not student_answers[index]:
            pass
        else:
            score -= 1
    
    return score if score > 0 else 0
    




# Reversing a Process
# https://www.codewars.com/kata/5dad6e5264e25a001918a1fc/python
"""
Suppose we know the process by which a string s was encoded to a string r (see explanation below). The aim of the kata is to decode this string r to get back the original string s.

Explanation of the encoding process:
input: a string s composed of lowercase letters from "a" to "z", and a positive integer num
we know there is a correspondence between abcde...uvwxyzand 0, 1, 2 ..., 23, 24, 25 : 0 <-> a, 1 <-> b ...
if c is a character of s whose corresponding number is x, apply to x the function f: x-> f(x) = num * x % 26, then find ch the corresponding character of f(x)
Accumulate all these ch in a string r
concatenate num and r and return the result
For example:

encode("mer", 6015)  -->  "6015ekx"

m --> 12,   12 * 6015 % 26 = 4,    4  --> e
e --> 4,     4 * 6015 % 26 = 10,   10 --> k
r --> 17,   17 * 6015 % 26 = 23,   23 --> x

So we get "ekx", hence the output is "6015ekx"
Task
A string s was encoded to string r by the above process. complete the function to get back s whenever it is possible.

Indeed it can happen that the decoding is impossible for strings composed of whatever letters from "a" to "z" when positive integer num has not been correctly chosen. In that case return "Impossible to decode".

Examples
decode "6015ekx" -> "mer"
decode "5057aan" -> "Impossible to decode"
"""
(encode("mer", 6015), "6015ekx")
(decode("6015ekx"), "mer")
(decode("761328qockcouoqmoayqwmkkic"), "Impossible to decode")
(decode("1273409kuqhkoynvvknsdwljantzkpnmfgf"), "uogbucwnddunktsjfanzlurnyxmx")
(decode("1544749cdcizljymhdmvvypyjamowl"), "mfmwhbpoudfujjozopaugcb")
(decode("105860ymmgegeeiwaigsqkcaeguicc"), "Impossible to decode")
(decode("1122305vvkhrrcsyfkvejxjfvafzwpsdqgp"), "rrsxppowmjsrclfljrajtybwviqb")


import re
import string

letters = string.ascii_lowercase

def encode(text, number):
    code_part = [letters[(letters.index(letter) * number) % 26]
                 for letter in text]

    return str(number) + "".join(code_part)

def decode(text):
    # parse number
    number = int(re.search(r"\d+", text).group())

    # check if 26 and number are coprime
    is_not_coprime = any(True
                         for divider in range(2, 27)
                         if not (26 % divider) and
                         not (number % divider))

    # 26 and nubmer are not coprime
    if is_not_coprime:
        return "Impossible to decode"

    # parse word
    word = re.search(r"[a-z]+", text).group()
    decoded_letters = ""

    for letter in word:
        index_of_letter = letters.index(letter)

        # for every letter in alphabet check if it encodes to current letter
        for alpha in letters:
            if letters.index(alpha) * number % 26 == index_of_letter:
                decoded_letters += alpha
                break

    return decoded_letters





# How many stairs will Suzuki climb in 20 years?
# https://www.codewars.com/kata/56fc55cd1f5a93d68a001d4e/train/python
"""
Suzuki is a monk who climbs a large staircase to the monastery as part of a ritual. Some days he climbs more stairs than others depending on the number of students he must train in the morning. He is curious how many stairs might be climbed over the next 20 years and has spent a year marking down his daily progress.

The sum of all the stairs logged in a year will be used for estimating the number he might climb in 20.

20_year_estimate = one_year_total * 20

You will receive the following data structure representing the stairs Suzuki logged in a year. You will have all data for the entire year so regardless of how it is logged the problem should be simple to solve.

stairs = [sunday,monday,tuesday,wednesday,thursday,friday,saturday]
Make sure your solution takes into account all of the nesting within the stairs array.

Each weekday in the stairs array is an array.

sunday = [6737, 7244, 5776, 9826, 7057, 9247, 5842, 5484, 6543, 5153, 6832, 8274, 7148, 6152, 5940, 8040, 9174, 7555, 7682, 5252, 8793, 8837, 7320, 8478, 6063, 5751, 9716, 5085, 7315, 7859, 6628, 5425, 6331, 7097, 6249, 8381, 5936, 8496, 6934, 8347, 7036, 6421, 6510, 5821, 8602, 5312, 7836, 8032, 9871, 5990, 6309, 7825]
Your function should return the 20 year estimate of the stairs climbed using the formula above.
"""
sunday = [6737, 7244, 5776, 9826, 7057, 9247, 5842, 5484, 6543, 5153, 6832, 8274,
          7148, 6152, 5940, 8040, 9174, 7555, 7682, 5252, 8793, 8837, 7320, 8478, 6063, 
          5751, 9716, 5085, 7315, 7859, 6628, 5425, 6331, 7097, 6249, 8381, 5936, 8496, 
          6934, 8347, 7036, 6421, 6510, 5821, 8602, 5312, 7836, 8032, 9871, 5990, 6309, 7825]

monday = [9175, 7883, 7596, 8635, 9274, 9675, 5603, 6863, 6442, 9500, 7468, 9719,
          6648, 8180, 7944, 5190, 6209, 7175, 5984, 9737, 5548, 6803, 9254, 5932, 7360, 9221, 
          5702, 5252, 7041, 7287, 5185, 9139, 7187, 8855, 9310, 9105, 9769, 9679, 7842,
          7466, 7321, 6785, 8770, 8108, 7985, 5186, 9021, 9098, 6099, 5828, 7217, 9387]

tuesday = [8646, 6945, 6364, 9563, 5627, 5068, 9157, 9439, 5681, 8674, 6379, 8292,
           7552, 5370, 7579, 9851, 8520, 5881, 7138, 7890, 6016, 5630, 5985, 9758, 8415, 7313,
           7761, 9853, 7937, 9268, 7888, 6589, 9366, 9867, 5093, 6684, 8793, 8116, 8493, 
           5265, 5815, 7191, 9515, 7825, 9508, 6878, 7180, 8756, 5717, 7555, 9447, 7703]

wednesday = [6353, 9605, 5464, 9752, 9915, 7446, 9419, 6520, 7438, 6512, 7102, 
             5047, 6601, 8303, 9118, 5093, 8463, 7116, 7378, 9738, 9998, 7125, 6445, 6031, 8710,
             5182, 9142, 9415, 9710, 7342, 9425, 7927, 9030, 7742, 8394, 9652, 5783, 7698, 
             9492, 6973, 6531, 7698, 8994, 8058, 6406, 5738, 7500, 8357, 7378, 9598, 5405, 9493]

thursday = [6149, 6439, 9899, 5897, 8589, 7627, 6348, 9625, 9490, 5502, 5723, 8197,
            9866, 6609, 6308, 7163, 9726, 7222, 7549, 6203, 5876, 8836, 6442, 6752, 8695, 8402,
            9638, 9925, 5508, 8636, 5226, 9941, 8936, 5047, 6445, 8063, 6083, 7383, 7548, 5066, 
            7107, 6911, 9302, 5202, 7487, 5593, 8620, 8858, 5360, 6638, 8012, 8701]

friday = [5000, 5642, 9143, 7731, 8477, 8000, 7411, 8813, 8288, 5637, 6244, 6589, 6362, 
         6200, 6781, 8371, 7082, 5348, 8842, 9513, 5896, 6628, 8164, 8473, 5663, 9501, 
         9177, 8384, 8229, 8781, 9160, 6955, 9407, 7443, 8934, 8072, 8942, 6859, 5617,
         5078, 8910, 6732, 9848, 8951, 9407, 6699, 9842, 7455, 8720, 5725, 6960, 5127]

saturday = [5448, 8041, 6573, 8104, 6208, 5912, 7927, 8909, 7000, 5059, 6412, 6354, 8943, 
            5460, 9979, 5379, 8501, 6831, 7022, 7575, 5828, 5354, 5115, 9625, 7795, 7003, 
            5524, 9870, 6591, 8616, 5163, 6656, 8150, 8826, 6875, 5242, 9585, 9649, 9838, 
            7150, 6567, 8524, 7613, 7809, 5562, 7799, 7179, 5184, 7960, 9455, 5633, 9085]

stairs = [sunday,monday,tuesday,wednesday,thursday,friday,saturday]

(stairs_in_20(stairs), 54636040)


def stairs_in_20(stairs):
    stairs_in_one_year = sum(sum(inner) for inner in stairs)
    return stairs_in_one_year * 20





# Vowel remover
# https://www.codewars.com/kata/5547929140907378f9000039/train/python
"""
Create a function called shortcut to remove the lowercase vowels (a, e, i, o, u ) in a given string.

Examples
"hello"     -->  "hll"
"codewars"  -->  "cdwrs"
"goodbye"   -->  "gdby"
"HELLO"     -->  "HELLO"
"""
(shortcut("hello"), "hll")
(shortcut("hellooooo"), "hll")
(shortcut("how are you today?"), "hw r y tdy?")
(shortcut("complain"), "cmpln")
(shortcut("never"), "nvr")


import re

def shortcut(word):
    return re.sub(r"[aeoiu]", "", word)

def shortcut(word):
    return "".join(letter 
                   for letter in word 
                   if letter not in "aeoiu")

def is_lower_vovel(vovel):
    return vovel in "aeoiu"

def shortcut(word):
    return "".join(filter(not is_lower_vovel, word))


