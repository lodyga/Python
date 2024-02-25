https://neetcode.io/practice

# 217
# https://leetcode.com/problems/contains-duplicate/
"""Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.

Input: nums = [1,2,3,1]
Output: true
"""

class Solution:
    def containsDuplicate(self, nums: list[int]) -> bool:
        return not (len(set(nums)) == len(nums))  # O(n), O(n)
    
        # alt solution
        # nums_set = set()
        # for number in nums:
        #     if number in nums_set:
        #         return True
        #     else:
        #         nums_set.add(number)
        # return False

        # alt solution
        # import numpy as np
        # return len(np.unique(nums)) == len(nums)


Solution().containsDuplicate([1, 2, 3])
Solution().containsDuplicate([1, 2, 3, 4])
Solution().containsDuplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2])

sol = Solution()
sol.containsDuplicate([1, 2, 3])





# 242
# https://leetcode.com/problems/valid-anagram/
"""Given two strings s and t, return true if t is an anagram of s, and false otherwise.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

Example 1:

Input: s = "anagram", t = "nagaram"
Output: true"""


class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return {i: s.count(i) for i in set(s)} == {i: t.count(i) for i in set(t)}  # 47, 17; O(n), O(n)

        # alt solution
        # from collections import Counter
        # return Counter(s) == Counter(t)  # 35, 17
    
        # alt solution
        # return sorted(s) == sorted(t)  # 53, 18


Solution().isAnagram("anagram", "nagaram")
Solution().isAnagram("rat", "car")





# 1
# https://leetcode.com/problems/two-sum/description/
"""Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1]."""
# https://www.codewars.com/kata/54d81488b981293527000c8f


class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        nums_dict = {}  # 55, 18; O(n), O(n)
        for ind, num in enumerate(nums):
            diff = target - num
            if diff in nums_dict:
                return [nums_dict[diff], ind]
            nums_dict[num] = ind
        return None

    # alt solution
    #     for i in range(len(nums) - 1):  # 1600, 17; O(n2), O(1)
    #         for j in range(i + 1, len(nums)):
    #             if nums[i] + nums[j] == target:
    #                 return [i, j]


Solution().twoSum([2, 7, 11, 15], 9)
Solution().twoSum([3, 2, 4], 6)
Solution().twoSum([3, 3], 6)





# 

