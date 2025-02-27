# 1221. Split a String in Balanced Strings
# ["String", "Greedy", "Counting"], Easy
# https://leetcode.com/problems/split-a-string-in-balanced-strings/description/
"""
Balanced strings are those that have an equal quantity of 'L' and 'R' characters.

Given a balanced string s, split it into some number of substrings such that:

Each substring is balanced.
Return the maximum number of balanced strings you can obtain.

Example 1:

Input: s = "RLRRLLRLRL"
Output: 4
Explanation: s can be split into "RL", "RRLL", "RL", "RL", each substring contains same number of 'L' and 'R'.
Example 2:

Input: s = "RLRRRLLRLL"
Output: 2
Explanation: s can be split into "RL", "RRRLLRLL", each substring contains same number of 'L' and 'R'.
Note that s cannot be split into "RL", "RR", "RL", "LR", "LL", because the 2nd and 5th substrings are not balanced.
Example 3:

Input: s = "LLLLRRRR"
Output: 1
Explanation: s can be split into "LLLLRRRR".
"""


class Solution:
    def balancedStringSplit(self, s: str) -> int:
        seen = []
        oppos = {"L": "R", "R": "L"}
        counter = 0

        for direct in s:
            if seen and oppos[direct] == seen[-1]:
                seen.pop()
            else:
                seen.append(direct)
            if not seen:
                counter += 1

        return counter
(Solution().balancedStringSplit("RLRRLLRLRL"), 4)
(Solution().balancedStringSplit("RLRRRLLRLL"), 2)
(Solution().balancedStringSplit("LLLLRRRR"), 1)


class Solution:
    def balancedStringSplit(self, s: str) -> bool:
        seen = []
        oppos = {")": "(", "}": "{", "]": "["}

        for bracket in s:
            if bracket in oppos:
                if seen and oppos[bracket] == seen[-1]:
                    seen.pop()
                else:
                    return False
                    
            else:
                seen.append(bracket)

        return True
(Solution().balancedStringSplit("]()"), False)
(Solution().balancedStringSplit("[]]()"), False)
(Solution().balancedStringSplit("[]()"), True)





# 1323. Maximum 69 Number
# 
# https://leetcode.com/problems/maximum-69-number/description/
"""
You are given a positive integer num consisting only of digits 6 and 9.

Return the maximum number you can get by changing at most one digit (6 becomes 9, and 9 becomes 6).

 

Example 1:

Input: num = 9669
Output: 9969
Explanation: 
Changing the first digit results in 6669.
Changing the second digit results in 9969.
Changing the third digit results in 9699.
Changing the fourth digit results in 9666.
The maximum number is 9969.
Example 2:

Input: num = 9996
Output: 9999
Explanation: Changing the last digit 6 to 9 results in the maximum number.
Example 3:

Input: num = 9999
Output: 9999
Explanation: It is better not to apply any change.
"""


class Solution:
    def maximum69Number (self, num: int) -> int:
        six_find = str(num).find("6")
        if six_find == -1:
            return num
        else:
            return int(str(num)[:six_find] + "9" + str(num)[six_find + 1:])
(Solution().maximum69Number(9669), 9969)
(Solution().maximum69Number(9996), 9999)
(Solution().maximum69Number(9999), 9999)


class Solution:
    def maximum69Number (self, num: int) -> int:
        return int(str(num).replace("6", "9", 1))





# 507. Perfect Number
# https://leetcode.com/problems/perfect-number/description/
"""
A perfect number is a positive integer that is equal to the sum of its positive divisors, excluding the number itself. A divisor of an integer x is an integer that can divide x evenly.

Given an integer n, return true if n is a perfect number, otherwise return false.

Example 1:

Input: num = 28
Output: true
Explanation: 28 = 1 + 2 + 4 + 7 + 14
1, 2, 4, 7, and 14 are all divisors of 28.
Example 2:

Input: num = 7
Output: false
"""


# O(sqrt(n)), O(1)
class Solution:
    def checkPerfectNumber(self, nums: int) -> bool:
        sum_divisors = 0
        for num in range(1, int(nums**0.5) + 1):
            if not nums % num:
                sum_divisors += num
                if num ** 2 != nums:
                    sum_divisors += nums // num
        return sum_divisors - nums == nums
Solution().checkPerfectNumber(28)
Solution().checkPerfectNumber(7)


# O(n), O(1)
class Solution:
    def checkPerfectNumber(self, nums: int) -> bool:
        # return sum(n for n in range(1, nums//2 + 1) if not nums % n) == nums
        sum_divisors = 0
        for num in range(1, nums//2 + 1):
            if not nums % num:   
                sum_divisors += num
        return sum_divisors == nums





# Count Pairs Whose Sum is Less than Target
# https://leetcode.com/problems/count-pairs-whose-sum-is-less-than-target/
"""
Given a 0-indexed integer array nums of length n and an integer target, return the number of pairs (i, j) where 0 <= i < j < n and nums[i] + nums[j] < target.
 
Example 1:

Input: nums = [-1,1,2,3,1], target = 2
Output: 3
Explanation: There are 3 pairs of indices that satisfy the conditions in the statement:
- (0, 1) since 0 < 1 and nums[0] + nums[1] = 0 < target
- (0, 2) since 0 < 2 and nums[0] + nums[2] = 1 < target 
- (0, 4) since 0 < 4 and nums[0] + nums[4] = 0 < target
Note that (0, 3) is not counted since nums[0] + nums[3] is not strictly less than the target.

Example 2:

Input: nums = [-6,2,5,-2,-7,-1,3], target = -2
Output: 10
Explanation: There are 10 pairs of indices that satisfy the conditions in the statement:
- (0, 1) since 0 < 1 and nums[0] + nums[1] = -4 < target
- (0, 3) since 0 < 3 and nums[0] + nums[3] = -8 < target
- (0, 4) since 0 < 4 and nums[0] + nums[4] = -13 < target
- (0, 5) since 0 < 5 and nums[0] + nums[5] = -7 < target
- (0, 6) since 0 < 6 and nums[0] + nums[6] = -3 < target
- (1, 4) since 1 < 4 and nums[1] + nums[4] = -5 < target
- (3, 4) since 3 < 4 and nums[3] + nums[4] = -9 < target
- (3, 5) since 3 < 5 and nums[3] + nums[5] = -3 < target
- (4, 5) since 4 < 5 and nums[4] + nums[5] = -8 < target
- (4, 6) since 4 < 6 and nums[4] + nums[6] = -4 < target
"""


# O(nlogn), O(1)
class Solution:
    def countPairs(self, nums: list[int], target: int) -> int:
        counter = 0
        nums.sort()
        l = 0
        r = len(nums) -1

        while l < r:
            if nums[l] + nums[r] < target:
                # [-1, 1, 1, 2, 3] (r - l) because left = -1, rigth = 2, so add every combination (1, 1, 2) before incresing the left pointer
                counter += r - l
                l += 1
            else:
                r -= 1

        return counter
(Solution().countPairs([-1, 1, 2, 3, 1], 2), 3)
(Solution().countPairs([-6, 2, 5, -2, -7, -1, 3], -2), 10)


# O(n2), O(1)
class Solution:
    def countPairs(self, nums: list[int], target: int) -> int:
        counter = 0

        for i_ind, i in enumerate(nums):
            for j_ind, j in enumerate(nums):
                if i_ind < j_ind and i + j < target:
                    counter += 1
        
        return counter
(Solution().countPairs([-1, 1, 2, 3, 1], 2), 3)
(Solution().countPairs([-6, 2, 5, -2, -7, -1, 3], -2), 10)





# Reverse Prefix of Word
# https://leetcode.com/problems/reverse-prefix-of-word/
"""
Given a 0-indexed string word and a character ch, reverse the segment of word that starts at index 0 and ends at the index of the first occurrence of ch (inclusive). If the character ch does not exist in word, do nothing.

For example, if word = "abcdefd" and ch = "d", then you should reverse the segment that starts at 0 and ends at 3 (inclusive). The resulting string will be "dcbaefd".
Return the resulting string.

Example 1:

Input: word = "abcdefd", ch = "d"
Output: "dcbaefd"
Explanation: The first occurrence of "d" is at index 3. 
Reverse the part of word from 0 to 3 (inclusive), the resulting string is "dcbaefd".
Example 2:

Input: word = "xyxzxe", ch = "z"
Output: "zxyxxe"
Explanation: The first and only occurrence of "z" is at index 3.
Reverse the part of word from 0 to 3 (inclusive), the resulting string is "zxyxxe".
Example 3:

Input: word = "abcd", ch = "z"
Output: "abcd"
Explanation: "z" does not exist in word.
You should not do any reverse operation, the resulting string is "abcd".
"""


# O(n), O(n)
class Solution:
    def reversePrefix(self, word: str, ch: str) -> str:
        seen = ""

        for letter in word:
            if letter != ch:
                seen = letter + seen
            else:
                seen = letter + seen
                return seen + word[len(seen):]
        
        return word
(Solution().reversePrefix("abcdefd", "d"), "dcbaefd")
(Solution().reversePrefix("xyxzxe", "z"), "zxyxxe")
(Solution().reversePrefix("abcd", "z"), "abcd")


# O(n), O(n)
class Solution:
    def reversePrefix(self, word: str, ch: str) -> str:
        letter_pos = word.find(ch)

        if letter_pos == -1:
            return word
        else:
            return word[:letter_pos + 1][::-1] + word[letter_pos + 1:]





# Find First Palindromic String in the Array
# https://leetcode.com/problems/find-first-palindromic-string-in-the-array/
"""
Given an array of strings words, return the first palindromic string in the array. If there is no such string, return an empty string "".

A string is palindromic if it reads the same forward and backward.

Example 1:

Input: words = ["abc","car","ada","racecar","cool"]
Output: "ada"
Explanation: The first string that is palindromic is "ada".
Note that "racecar" is also palindromic, but it is not the first.
Example 2:

Input: words = ["notapalindrome","racecar"]
Output: "racecar"
Explanation: The first and only string that is palindromic is "racecar".
Example 3:

Input: words = ["def","ghi"]
Output: ""
Explanation: There are no palindromic strings, so the empty string is returned.
"""


# O(n), O(1)
class Solution:
    def firstPalindrome(self, words: list[str]) -> str:
        for word in words:
            l = 0
            r = len(word) - 1
            return_word = True

            # while l < r:
            #     if word[l] != word[r]:
            #         return_word = False
            #         break
            #     else:
            #         l += 1
            #         r -= 1
            # 
            # if return_word:
            #     return word

            if word == word[::-1]:
                return word
        
        return ""


(Solution().firstPalindrome(["abc","car","ada","racecar","cool"]), "ada")
(Solution().firstPalindrome(["notapalindrome","racecar"]), "racecar")
(Solution().firstPalindrome(["def","ghi"]), "")





# Number of Arithmetic Triplets
# https://leetcode.com/problems/number-of-arithmetic-triplets/
"""
You are given a 0-indexed, strictly increasing integer array nums and a positive integer diff. A triplet (i, j, k) is an arithmetic triplet if the following conditions are met:

i < j < k,
nums[j] - nums[i] == diff, and
nums[k] - nums[j] == diff.
Return the number of unique arithmetic triplets.

Example 1:

Input: nums = [0,1,4,6,7,10], diff = 3
Output: 2
Explanation:
(1, 2, 4) is an arithmetic triplet because both 7 - 4 == 3 and 4 - 1 == 3.
(2, 4, 5) is an arithmetic triplet because both 10 - 7 == 3 and 7 - 4 == 3. 

Example 2:

Input: nums = [4,5,6,7,8,9], diff = 2
Output: 2
Explanation:
(0, 2, 4) is an arithmetic triplet because both 8 - 6 == 2 and 6 - 4 == 2.
(1, 3, 5) is an arithmetic triplet because both 9 - 7 == 2 and 7 - 5 == 2.
"""


# O(n),O(n)
class Solution:
    def arithmeticTriplets(self, nums: list[int], diff: int) -> int:
        seen = set()
        counter = 0

        for num in nums:
            seen.add(num)
            if num - diff in seen and num - 2*diff in seen:
                counter += 1

        return counter
(Solution().arithmeticTriplets([0, 1, 4, 6, 7, 10], 3), 2)
(Solution().arithmeticTriplets([4, 5, 6, 7, 8, 9], 2), 2)





# Substrings of Size Three with Distinct Characters
# https://leetcode.com/problems/substrings-of-size-three-with-distinct-characters/
"""
A string is good if there are no repeated characters.

Given a string s​​​​​, return the number of good substrings of length three in s​​​​​​.

Note that if there are multiple occurrences of the same substring, every occurrence should be counted.

A substring is a contiguous sequence of characters in a string.

Example 1:

Input: s = "xyzzaz"
Output: 1
Explanation: There are 4 substrings of size 3: "xyz", "yzz", "zza", and "zaz". 
The only good substring of length 3 is "xyz".
Example 2:

Input: s = "aababcabc"
Output: 4
Explanation: There are 7 substrings of size 3: "aab", "aba", "bab", "abc", "bca", "cab", and "abc".
The good substrings are "abc", "bca", "cab", and "abc".
"""


# O(n), O(1)
class Solution:
    def countGoodSubstrings(self, s: str) -> int:
        counter = 0

        for i in range(len(s) - 2):
            if len(set(s[i: i + 3])) == 3:
                counter += 1

        return counter
(Solution().countGoodSubstrings("xyzzaz"), 1)
(Solution().countGoodSubstrings("aababcabc"), 4)





# Trapping Rain Water
# https://leetcode.com/problems/trapping-rain-water/
"""
Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

Example 1:

Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.
Example 2:

Input: height = [4,2,0,3,2,5]
Output: 9

# ____
#   |    _____
#   |____|   |
# ____________|
# (3, 1, 2)
"""


class Solution:
    def trap(self, land: list[int]) -> int:
        l = 0
        r = len(land) - 1
        flood_depth = 0
        flood_sum = 0
        max_land_l = 0
        max_land_r = 0

        while l < r:
            if land[l] < land[r]:
                max_land_l = max(max_land_l, land[l])
                # flood_depth = max(flood_depth, max_land_l - land[l])
                flood_depth = max_land_l - land[l]
                flood_sum += flood_depth
                l += 1
            else:
                max_land_r = max(max_land_r, land[r])
                # flood_depth = max(flood_depth, max_land_r - land[r])
                flood_depth = max_land_r - land[r]
                flood_sum += flood_depth
                r -= 1

        return flood_sum
(Solution().trap([1, 3, 2, 1, 2, 1, 5, 3, 3, 4, 2]), 8)
(Solution().trap([5, 8]), 0)
(Solution().trap([3, 1, 2]), 1)
(Solution().trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]), 6)
(Solution().trap([4, 2, 0, 3, 2, 5]), 9)





# Design Compressed String Iterator
# https://leetcode.com/problems/design-compressed-string-iterator
"""
Design and implement a data structure for a compressed string iterator. It should support the following operations: next and hasNext.

The given compressed string will be in the form of each letter followed by a positive integer representing the number of this letter existing in the original uncompressed string.

next() - if the original string still has uncompressed characters, return the next letter; Otherwise return a white space.
hasNext() - Judge whether there is any letter needs to be uncompressed.

Note:
Please remember to RESET your class variables declared in StringIterator, as static/class variables are persisted across multiple test cases. Please see here for more details.

Example:

StringIterator iterator = new StringIterator("L1e2t1C1o1d1e1");

iterator.next(); // return 'L'
iterator.next(); // return 'e'
iterator.next(); // return 'e'
iterator.next(); // return 't'
iterator.next(); // return 'C'
iterator.next(); // return 'o'
iterator.next(); // return 'd'
iterator.hasNext(); // return true
iterator.next(); // return 'e'
iterator.hasNext(); // return false
iterator.next(); // return ' '
"""


# O(1), O(1)
# Design, Array, String, Iterator
# generator
class StringIterator:
    def __init__(self, compressed_string: str):
        self.generator = self._generate(compressed_string)
        self.letter = next(self.generator, " ")  # Preload the first letter

    def _generate(self, compressed_string):
        index = 0
        
        while index < len(compressed_string):
            letter = compressed_string[index]
            index += 1
            frequency = 0
            
            # Parse the number representing the letter frequency
            while (index < len(compressed_string) and 
                   compressed_string[index].isdigit()):
                frequency = 10 * frequency + int(compressed_string[index])
                index += 1
            
            # Yield the letter `frequency` times
            for _ in range(frequency):
                yield letter
        
    def next(self) -> str:
        current_letter = self.letter
        self.letter = next(self.generator, " ")
        return current_letter

    def hasNext(self) -> bool:
        return self.letter != " "


# O(n): constructor; O(1): next, hasNext; aux space O(n)
# Design, Array, String, Iterator
# compressed string coppied to self.text in constructor
class StringIterator:
    def __init__(self, text: str):
        self.text = text
        self.index = 0
        self.letter = ""
        self.letter_frequency = []
        
    def _get_next_letter(self):
        """
        Method to get next letter and its frequency. Its called when current
        letter frequency reaches zero.
        """
        if self.index == len(self.text):
            self.letter = " "
            self.letter_frequency = 0
            return
        
        self.letter = self.text[self.index]
        self.index += 1
        number = 0

        while (self.index < len(self.text) and
               self.text[self.index].isdigit()):
            number = 10 * number + int(self.text[self.index])
            self.index += 1
        
        self.letter_frequency = number

    def next(self):
        if not self.letter_frequency:
            self._get_next_letter()
        
        if self.letter_frequency:
            self.letter_frequency -= 1
            return self.letter
        
        return " "

        # if self.index == len(self.text):
        #     return " "
        # elif self.letter_frequency:
        #     self.letter_frequency -= 1
        #     return self.letter
        # else:
        #     self._get_next_letter()
        #     self.letter_frequency -= 1
        #     return self.letter

    def hasNext(self):
        return (self.index < len(self.text) or
                self.letter_frequency > 0)


# O(n): constructor; O(1): next, hasNext; aux space O(n)
# Design, Array, String, Iterator
# compressed string processed in constructor
class StringIterator:
    def __init__(self, text: str):
        self.index = 0
        self.text = []
        self.str_to_list(text)
        
    def str_to_list(self, text):
        frequency = 0
        for char in text:
            if char.isalpha():
                if frequency:
                    self.text.append(frequency)
                    frequency = 0
                self.text.append(char)
            elif char.isnumeric():
                frequency = frequency * 10 + int(char)
        self.text.append(frequency)

    def next(self):
        if not self.hasNext():
            return " "
        
        letter = self.text[self.index]  # a letter to return
        self.text[self.index + 1] -= 1  # decrease a letter frequency
        if not self.text[self.index + 1]:  # if all letter occurences used
            self.index += 2  # update index (list pointer)        
        return letter

    def hasNext(self):
        return self.index < len(self.text)


iterator = StringIterator("L1e2t1C1o1d1e1")
print(iterator.next())  # return "L"
print(iterator.next())  # return "e"
print(iterator.next())  # return "e"
print(iterator.next())  # return "t"
print(iterator.next())  # return "C"
print(iterator.next())  # return "o"
print(iterator.next())  # return "d"
print(iterator.hasNext())  # return True
print(iterator.next())  # return "e"
print(iterator.hasNext())  # return False
print(iterator.next())  # return " "





# Binary Search Tree Iterator
# https://leetcode.com/problems/binary-search-tree-iterator/
"""
Implement the BSTIterator class that represents an iterator over the in-order traversal of a binary search tree (BST):

BSTIterator(TreeNode root) Initializes an object of the BSTIterator class. The root of the BST is given as part of the constructor. The pointer should be initialized to a non-existent number smaller than any element in the BST.
boolean hasNext() Returns true if there exists a number in the traversal to the right of the pointer, otherwise returns false.
int next() Moves the pointer to the right, then returns the number at the pointer.
Notice that by initializing the pointer to a non-existent smallest number, the first call to next() will return the smallest element in the BST.

You may assume that next() calls will always be valid. That is, there will be at least a next number in the in-order traversal when next() is called.

 

Example 1:
  7__
 /   \
3     15
     /  \
    9    20

Input
["BSTIterator", "next", "next", "hasNext", "next", "hasNext", "next", "hasNext", "next", "hasNext"]
[[[7, 3, 15, null, null, 9, 20]], [], [], [], [], [], [], [], [], []]
Output
[null, 3, 7, true, 9, true, 15, true, 20, false]

Explanation
BSTIterator bSTIterator = new BSTIterator([7, 3, 15, null, null, 9, 20]);
bSTIterator.next();    // return 3
bSTIterator.next();    // return 7
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 9
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 15
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 20
bSTIterator.hasNext(); // return False
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


# Your BSTIterator object will be instantiated and called as such:
# obj = BSTIterator(root)
# param_1 = obj.next()
# param_2 = obj.hasNext()


# O(n): constructor; O(1): next, hasNext; aux space O(n)
class BSTIterator:
    def __init__(self, root: Optional[TreeNode]):
        self.node_list = []
        self.tree_to_queue(root)
        self.index = 0
    
    def tree_to_queue(self, node):
        if not node:
            return
        
        self.tree_to_queue(node.left)
        self.node_list.append(node.val)
        self.tree_to_queue(node.right)
            
    def next(self) -> int:
        self.index += 1
        return self.node_list[self.index - 1]

    def hasNext(self) -> bool:
        return self.index < len(self.node_list)


# O(n): constructor; O(1): next, hasNext; aux space O(n)
class BSTIterator:
    def __init__(self, root: Optional[TreeNode]):
        self.node_list = self.tree_to_queue(root)
        self.index = 0
    
    def tree_to_queue(self, node):
        if not node:
            return []

        return (self.tree_to_queue(node.left) + 
                [node.val] + 
                self.tree_to_queue(node.right))
            
    def next(self) -> int:
        self.index += 1
        return self.node_list[self.index - 1]

    def hasNext(self) -> bool:
        return self.index < len(self.node_list)




