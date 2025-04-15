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




