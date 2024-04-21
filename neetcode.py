https://neetcode.io/practice

# 217
# https://leetcode.com/problems/contains-duplicate/
"""Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.

Input: nums = [1,2,3,1]
Output: true
"""

class Solution:
    def containsDuplicate(self, nums: list[int]) -> bool:
    
        nums_set = set()
        for number in nums:
            if number in nums_set:
                return True
            else:
                nums_set.add(number)
        return False
        
        # alt solution 
        # return not (len(set(nums)) == len(nums))  # O(n), O(n)

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





# 1. Two Sum
# https://leetcode.com/problems/two-sum/description/
# ["Array", "Hash Table"]
"""
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
"""
# https://www.codewars.com/kata/54d81488b981293527000c8f


# O(n), O(n)
class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        nums_dict = {}
        for ind, num in enumerate(nums):
            diff = target - num
            if diff in nums_dict:
                return [nums_dict[diff], ind]
            nums_dict[num] = ind
        return None
Solution().twoSum([2, 7, 11, 15], 9)
Solution().twoSum([3, 2, 4], 6)
Solution().twoSum([3, 3], 6)

    # alt solution
    #     for i in range(len(nums) - 1):  # 1600, 17; O(n2), O(1)
    #         for j in range(i + 1, len(nums)):
    #             if nums[i] + nums[j] == target:
    #                 return [i, j]





# 49. Group Anagrams
# https://leetcode.com/problems/group-anagrams/description/
"""Given an array of strings strs, group the anagrams together. You can return the answer in any order.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

 

Example 1:

Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]"""



class Solution:
    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:

        grouped_anagrams = dict() # O(m*n)
        # from collections import defaultdict
        # grouped_anagrams = defaultdict(list)
        
        strs.reverse()
        for word in strs:
            key = [0] * 26
            for letter in word:
                key[ord(letter) - ord("a")] += 1
            key = tuple(key)
            
            if not key in grouped_anagrams:
                grouped_anagrams[key] = []
            grouped_anagrams[key].append(word)
        
            # grouped_anagrams[key].append(word)
        
        return grouped_anagrams.values()
Solution().groupAnagrams(["eat","tea","tan","ate","nat","bat"])

        # alt solutions
        # strs.reverse()  # 83, 19 O(m*n*logn) m - list cout, n - avg word len
        # result = {}
        # for word in strs:
        #     key = "".join(sorted(word))
        #     if key in result:
        #         result[key].append(word)
        #     else:
        #         result[key] = [word]
        # return list(result.values())


        # alt solutions
        # strs.reverse()  # 80-100, 20
        # anagrams = defaultdict(list)
        # for word in strs:
        #     key = "".join(sorted(word))
        #     anagrams[key].append(word)
        # return anagrams.values()

        
        # alt solutions
        # strs.reverse()  # 91, 201
        # result = defaultdict(list)
        # {result["".join(sorted(word))].append(word) for word in strs}
        # return(result.values())





# 347. Top K Frequent Elements
# https://leetcode.com/problems/top-k-frequent-elements/description/
"""Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.

Example 1:

Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]"""


from collections import Counter

class Solution:
    def topKFrequent(self, nums: list[int], k: int) -> list[int]:

        return [elem[0] for elem in Counter(nums).most_common(k)]

        # alt solutions
        # freq_dict = {digit: nums.count(digit) for digit in set(nums)}
        # bucket = {}
        # for number, occurrence in freq_dict.items():
        #     if occurrence in bucket:
        #         bucket[occurrence].append(number)
        #     else:
        #         bucket[occurrence] = [number]
        # 
        # top_elements = []
        # for _ in range(k):
        #     for i in bucket[max(bucket)]:
        #         top_elements.append(i)
        #         if len(top_elements) == k:
        #             return top_elements
        #     bucket.pop(max(bucket))


        # alt solutions
        # count = {}
        # freq = [[] for i in range(len(nums) + 1)]
        # 
        # for n in nums:
        #     count[n] = 1 + count.get(n, 0)
        # for n, c in count.items():
        #     freq[c].append(n)
        # 
        # return freq
        # res = []
        # for i in range(len(freq) - 1, 0, -1):
        #     for n in freq[i]:
        #         res.append(n)
        #         if len(res) == k:
        #             return res

Solution().topKFrequent([1, 1, 1, 2, 2, 3], 2)
# Solution().topKFrequent([1, 2], 2)





# 238. Product of Array Except Self
# https://leetcode.com/problems/product-of-array-except-self/description/
"""Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operation.

Example 1:

Input: nums = [1,2,3,4]
Output: [24,12,8,6]"""


import numpy as np

class Solution:
    def productExceptSelf(self, nums: list[int]) -> list[int]:
        return [np.prod(nums) // i for i in nums]  # uses "//" operator, if it works its works

Solution().productExceptSelf([1, 2, 3, 4])





# https://www.lintcode.com/problem/659/
import re

class Solution:

    def encode(self, strs: list[str]) -> str:
        result = ""
        for word in strs:
            result += str(len(word)) + "Τ" + word
        return result

    def decode(self, s: str) -> list[str]:
        # result = []
        # while True:
        #     i = re.match(r"(\d+)*.", s).group(1)
        #     len_i = len(str(i))
        #     result.append(s[len_i+1:int(i)+len_i+1])
        #     s = s[int(i)+len_i+1:]
        #     if not s:
        #         break

        result = re.findall(r"\d+Τ(\D+)", s)
        
        return result

Solution().decode(Solution().encode(["neet","code","love","you"]))

# Solution().encode(["neet","code","love","you"])
# Solution().encode(["neet","code","love","youuuuuuuuuuuuuuuu"])
# Solution().encode([])
# Solution().encode([""])
Solution().decode(Solution().encode(["neet","code","love","youuuuuuuuuuuuuuuu"]))
# Solution().decode(Solution().encode([]))
# Solution().decode(Solution().encode([""]))





# 128. Longest Consecutive Sequence
# https://leetcode.com/problems/longest-consecutive-sequence/description/
"""Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.

You must write an algorithm that runs in O(n) time.

Example 1:

Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4."""


class Solution:
    def longestConsecutive(self, nums: list[int]) -> int:
        
        n_set = set(nums)
        longest = 0
        for num in n_set:    
            if not (num - 1) in n_set:
                curr_len = 1
                while (num + curr_len) in n_set:
                    curr_len += 1
                longest = max(longest, curr_len)
        
        return longest
    
Solution().longestConsecutive([100, 4, 200, 1, 3, 2])





# 125. Valid Palindrome
# https://leetcode.com/problems/valid-palindrome/description/
"""
A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

Given a string s, return true if it is a palindrome, or false otherwise.

Example 1:

Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.
"""


import string
class Solution:
    def isPalindrome(self, s: str) -> bool:
        for char in string.punctuation + " ":
            s = s.replace(char, "")
        return s.lower() == s[::-1].lower()
Solution().isPalindrome("A man, a plan, a canal: Panama")

import re
class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = re.sub(r"[\W_]", "", s).lower()
        return s == s[::-1]
Solution().isPalindrome("A man, a plan, a canal: Panama")

class Solution:
    def isPalindrome(self, s: str) -> bool:
        cleaned_s = ""
        for i in s:
            if i.isalpha():
                cleaned_s += i.lower()
        return cleaned_s == cleaned_s[::-1]
Solution().isPalindrome("A man, a plan, a canal: Panama")





# 167. Two Sum II - Input Array Is Sorted
# https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
"""
Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number. Let these two numbers be numbers[index1] and numbers[index2] where 1 <= index1 < index2 <= numbers.length.

Return the indices of the two numbers, index1 and index2, added by one as an integer array [index1, index2] of length 2.

The tests are generated such that there is exactly one solution. You may not use the same element twice.

Your solution must use only constant extra space.

Example 1:

Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore, index1 = 1, index2 = 2. We return [1, 2].
"""


class Solution:
    def twoSum(self, numbers: list[int], target: int) -> list[int]:
        
        l, r = 0, len(numbers) - 1

        while l < r:
            two = numbers[l] + numbers[r]
            if two > target:
                r -= 1
            elif two < target:
                l += 1
            else:
                return [l + 1, r + 1]
        
        return None


Solution().twoSum([2, 7, 11, 15], 9)
Solution().twoSum([2, 3, 4], 6)





# 15. 3Sum
# https://leetcode.com/problems/3sum/
"""
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

 

Example 1:

Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Explanation: 
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
The distinct triplets are [-1,0,1] and [-1,-1,2].
Notice that the order of the output and the order of the triplets does not matter.
Example 2:

Input: nums = [0,1,1]
Output: []
Explanation: The only possible triplet does not sum up to 0.
Example 3:

Input: nums = [0,0,0]
Output: [[0,0,0]]
Explanation: The only possible triplet sums up to 0."""


class Solution:
    def threeSum(self, nums: list[int]) -> list[list[int]]:
        seen = []
        nums.sort()

        for ind, num in enumerate(nums[:-2]):
            if ind > 0 and num == nums[ind-1]:
                continue

            l = ind + 1
            r = len(nums) - 1

            while l < r:
                triplet = num + nums[l] + nums[r]
                # if triplet > 0 or nums[r] == nums[r-1]:  # breaks on [0, 0, 0]
                if triplet > 0:
                    r -= 1
                # elif triplet < 0 or nums[l] == nums[l+1]:
                elif triplet < 0:
                    l += 1
                else:
                    seen.append([num, nums[l], nums[r]])
                    # r -= 1
                    l += 1
                    while nums[l] == nums[l-1] and l < r:
                        l += 1                
        return seen
Solution().threeSum([-1, 0, 1, 2, -1, -4])
Solution().threeSum([-1, 1, 1])
Solution().threeSum([0, 0, 0])


# 0(n3)
class Solution:
    def threeSum(self, nums: list[int]) -> list[list[int]]:
        triplets = []
        for i in range(len(nums) - 2):
            for j in range(i + 1, len(nums) - 1):
                for k in range(j + 1, len(nums)):
                    if nums[i] + nums[j] + nums[k] == 0:
                        triplet = sorted([nums[i], nums[j], nums[k]])
                        if not triplet in triplets:
                            triplets.append(triplet)
        return triplets





# 11. Container With Most Water
# https://leetcode.com/problems/container-with-most-water/description/
"""You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

Notice that you may not slant the container."""


class Solution:
    def maxArea(self, height: list[int]) -> int:
        l, r = 0, len(height) - 1
        max_pool_size = 0

        while l < r:
            pool_size = min(height[l], height[r]) * (r - l)
            if pool_size > max_pool_size:
                max_pool_size = pool_size

            if height[l] < height[r]:
                l += 1
            else:
                r -= 1

        return max_pool_size

Solution().maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7])
Solution().maxArea([1, 1])


# O(n2)
class Solution:
    def maxArea(self, height: list[int]) -> int:
        max_pool_size = 0

        for i, h1 in enumerate(height[:-1]):
            for j, h2 in enumerate(height[i+1:]):
                pool_size = min(h1, h2) * (j + 1)
                if  pool_size > max_pool_size:
                    max_pool_size = pool_size

        return max_pool_size





# 121. Best Time to Buy and Sell Stock
# https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/
"""You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

 

Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
Example 2:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.
"""


class Solution:
    def maxProfit(self, prices: list[int]):
        l, r = 0, 1
        deal = 0
        while r < len(prices):
            # if price is lower - buy
            if prices[r] < prices[l]:
                l = r
            # if price is higher - calculate revenue
            else:
                deal = max(deal, prices[r] - prices[l])
            r += 1
        return deal

Solution().maxProfit([2, 4, 1])
Solution().maxProfit([7, 1, 5, 3, 6, 4])
Solution().maxProfit([7, 6, 4, 3, 1])
Solution().maxProfit([2, 1, 2, 1, 0, 1, 2])
Solution().maxProfit([1, 2])





# 3. Longest Substring Without Repeating Characters
# https://leetcode.com/problems/longest-substring-without-repeating-characters/description/
"""Given a string s, find the length of the longest 
substring without repeating characters.

Example 1:

Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
Example 2:

Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
Example 3:

Input: s = "pwwkew"
Output: 3"""


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        unique_elements = set()
        l = 0
        max_len = 0

        for r in range(len(s)):
            while s[r] in unique_elements:
                unique_elements.discard(s[l])
                l += 1

            unique_elements.add(s[r])
            max_len = max(max_len, len(unique_elements))

        return max_len

Solution().lengthOfLongestSubstring("aabaab!bb")
Solution().lengthOfLongestSubstring("aab")
Solution().lengthOfLongestSubstring("abcabcbb")
Solution().lengthOfLongestSubstring("bbbbb")
Solution().lengthOfLongestSubstring("pwwkew")

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:

        seen = []
        slider = ""
        max_len = 0
        for char in s:
            if char in seen:
                slider = slider[slider.index(char) + 1:]
                while char in seen:
                    seen.pop(0)
            seen.append(char)
            slider += char
            max_len = max(max_len, len(slider))
        return max_len






# 424. Longest Repeating Character Replacement
# https://leetcode.com/problems/longest-repeating-character-replacement/description/
"""You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most k times.

Return the length of the longest substring containing the same letter you can get after performing the above operations.

 

Example 1:

Input: s = "ABAB", k = 2
Output: 4
Explanation: Replace the two 'A's with two 'B's or vice versa.
Example 2:

Input: s = "AABABBA", k = 1
Output: 4
Explanation: Replace the one 'A' in the middle with 'B' and form "AABBBBA".
The substring "BBBB" has the longest repeating letters, which is 4.
There may exists other ways to achieve this answer too."""


# in progress
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        
        seen = []
        seen_dict = {}
        slider = ""
        max_len = 0
        for char in s:
            if char in seen:
                slider = slider[slider.index(char) + 1:]
                while char in seen:
                    seen.pop(0)
            seen.append(char)
            seen_dict[char] = seen_dict.get(char, 0) + 1
            slider += char
            max_len = max(max_len, len(slider) - max(seen_dict.values()))
        return max_len

Solution().characterReplacement("ABAB", 2)
Solution().characterReplacement("AABABBA", 1)





# 20. Valid Parentheses
# https://leetcode.com/problems/valid-parentheses/description/
"""
Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Every close bracket has a corresponding open bracket of the same type.
 

Example 1:

Input: s = "()"
Output: true
Example 2:

Input: s = "()[]{}"
Output: true
Example 3:

Input: s = "(]"
Output: false
"""


class Solution:
    def isValid(self, s: str) -> bool:
        seen = []
        oppos_bracket = {")": "(", "]": "[", "}": "{"}

        for bracket in s:
            if bracket in oppos_bracket:
                if seen and oppos_bracket[bracket] == seen[-1]:
                    seen.pop()
                else:
                    return False
            else:
                seen.append(bracket)
        return not seen
(Solution().isValid("()"), True)
(Solution().isValid("({})"), True)
(Solution().isValid("(})"), False)
(Solution().isValid("([)"), False)





# 153. Find Minimum in Rotated Sorted Array
# https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/description/
"""Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:

[4,5,6,7,0,1,2] if it was rotated 4 times.
[0,1,2,4,5,6,7] if it was rotated 7 times.
Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

Given the sorted rotated array nums of unique elements, return the minimum element of this array.

You must write an algorithm that runs in O(log n) time.

 

Example 1:

Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.
Example 2:

Input: nums = [4,5,6,7,0,1,2]
Output: 0
Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.
Example 3:

Input: nums = [11,13,15,17]
Output: 11
Explanation: The original array was [11,13,15,17] and it was rotated 4 times. 
"""


class Solution:
    def findMin(self, nums: list[int]) -> int:
        start, stop = 0, len(nums) - 1
        min_num = float("inf")

        while start <= stop:
            mid = (start + stop) // 2 # start + (stop - start) // 2
            min_num = min(min_num, nums[mid])

            if min_num > nums[stop]:
                start = mid + 1
            else:
                stop = mid - 1

        return min_num

(Solution().findMin([3, 4, 5, 1, 2]), 1)
Solution().findMin([2, 3, 4, 5, 1])
Solution().findMin([1, 2, 3, 4, 5])
(Solution().findMin([4, 5, 6, 7, 0, 1, 2]), 0)
(Solution().findMin([11, 13, 15, 17]), 11)





# 33. Search in Rotated Sorted Array
# https://leetcode.com/problems/search-in-rotated-sorted-array/description/
"""There is an integer array nums sorted in ascending order (with distinct values).

Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

You must write an algorithm with O(log n) runtime complexity.

 

Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
Example 2:

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
Example 3:
"""





# 39. Combination Sum
# https://leetcode.com/problems/combination-sum/description/
"""Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.

The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the 
frequency
 of at least one of the chosen numbers is different.

The test cases are generated such that the number of unique combinations that sum up to target is less than 150 combinations for the given input.

 

Example 1:

Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7.
These are the only two combinations.
Example 2:

Input: candidates = [2,3,5], target = 8
Output: [[2,2,2,2],[2,3,3],[3,5]]
Example 3:

Input: candidates = [2], target = 1
Output: []
"""

class Solution:
    def combinationSum(self, candidates: list[int], target: int) -> list[list[int]]:
        


Solution().combinationSum([2,3,6,7])






# 70. Climbing Stairs
# https://leetcode.com/problems/climbing-stairs/description/
"""You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

 

Example 1:

Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
Example 2:

Input: n = 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step"""


# Fibonnacci problem
class Solution:
    def climbStairs(self, n: int) -> int:
        a, b = 1, 1
        for _ in range(n):
            a, b = b, a + b
        return a
Solution().climbStairs(2)
Solution().climbStairs(3)
Solution().climbStairs(4)


def Fib_gen(n):
    a, b = 1, 1
    for _ in range(n):
        yield a
        a, b = b, a + b
fib5 = Fib_gen(5)

for i in fib5:
    print(i)
next(fib5)





# 198. House Robber
# https://leetcode.com/problems/house-robber/description/
"""You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

 

Example 1:

Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.
Example 2:

Input: nums = [2,7,9,3,1]
Output: 12
Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
Total amount you can rob = 2 + 9 + 1 = 12.
"""


class Solution:
    def rob(self, nums: list[int]) -> int:
        rob1, rob2 = 0, 0

        for num in nums:
            # [rob1, rob2, rob1 + num]
            tmp = max(rob1 + num, rob2)
            rob1 = rob2
            rob2 = tmp

        return rob2
(Solution().rob([1, 2, 3, 1]), 4)
(Solution().rob([2, 7, 9, 3, 1]), 12)





# 213. House Robber II
# https://leetcode.com/problems/house-robber-ii/description/
"""You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

 

Example 1:

Input: nums = [2,3,2]
Output: 3
Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.
Example 2:

Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.
Example 3:

Input: nums = [1,2,3]
Output: 3
"""


class Solution:
    def rob_list(self, nums: list[int]) -> int:
        rob1, rob2 = 0, 0

        for num in nums:
            tmp = max(rob1 + num, rob2)
            rob1 = rob2
            rob2 = tmp

        return rob2
    
    def rob(self, nums: list[int]) -> int:
        return max(nums[0], self.rob_list(nums[1:]), self.rob_list(nums[:-1]))
(Solution().rob([2, 3, 2]), 3)
(Solution().rob([1, 2, 3, 1]), 4)
(Solution().rob([1, 2, 3]), 3)
(Solution().rob([1]), 1)





# 5. Longest Palindromic Substring
# https://leetcode.com/problems/longest-palindromic-substring/description/
"""Given a string s, return the longest 
palindromic 
substring
 in s.

Example 1:

Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.
Example 2:

Input: s = "cbbd"
Output: "bb"
"""


# O(n2)
class Solution:
    def __init__(self) -> None:
        self.longest = ""

    def longest_pal(self, left: int, right: int, s: str) -> str:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            if (right - left + 1) > len(self.longest):
                self.longest = s[left:right + 1]
            left -= 1
            right += 1

    def longestPalindrome(self, s: str) -> str:
        for i in range(len(s)):
            self.longest_pal(i, i, s)
            self.longest_pal(i, i + 1, s)
        return self.longest

(Solution().longestPalindrome("babad"), "bab")
(Solution().longestPalindrome("cbbd"), "bb")
(Solution().longestPalindrome("a"), "a")
(Solution().longestPalindrome(""), "")
(Solution().longestPalindrome("bb"), "bb")
(Solution().longestPalindrome("ab"), "a")
(Solution().longestPalindrome("aacabdkacaa"), "aca")
(Solution().longestPalindrome("abdka"), "a")

class Solution:
    def longest_pal(self, left: int, right: int, s: str, longest: str) -> str:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            if (right - left + 1) > len(longest):
                longest = s[left:right + 1]
            left -= 1
            right += 1
        return longest
        

    def longestPalindrome(self, s: str) -> str:
        longest = ""
        for i in range(len(s)):
            longest = self.longest_pal(i, i, s, longest)
            longest = self.longest_pal(i, i + 1, s, longest)
        return longest



# O(n3)
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s:
            return None
        longest = s[0]
        for left in range(len(s) - 1):
            for right in range(left + 1, len(s)):
                is_longest = s[left:right + 1]
                if is_longest == (is_longest)[::-1] and len(is_longest) > len(longest):
                    longest = is_longest
        return longest





# 647. Palindromic Substrings
# https://leetcode.com/problems/palindromic-substrings/
"""Given a string s, return the number of palindromic substrings in it.

A string is a palindrome when it reads the same backward as forward.

A substring is a contiguous sequence of characters within the string.

 

Example 1:

Input: s = "abc"
Output: 3
Explanation: Three palindromic strings: "a", "b", "c".
Example 2:

Input: s = "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
"""


class Solution:
    def __init__(self) -> None:
        self.substrings_count = 0

    def count_pal(self, left: int, right: int, s: str) -> None:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            self.substrings_count += 1
            left -= 1
            right += 1
        
    def countSubstrings(self, s: str) -> int:
        for i in range(len(s)):
            self.count_pal(i, i, s)
            self.count_pal(i, i + 1, s)
        return self.substrings_count
(Solution().countSubstrings("abc"), 3)
(Solution().countSubstrings("aaa"), 6)





# 322. Coin Change
# https://leetcode.com/problems/coin-change/description/
"""You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.

 

Example 1:

Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1
Example 2:

Input: coins = [2], amount = 3
Output: -1
Example 3:

Input: coins = [1], amount = 0
Output: 0
"""

# bottom up dynamic programming

class Solution:
    def coinChange(self, coins: list[int], amount: int) -> int:
        coin_count = [amount + 1] * (amount + 1)
        coin_count[0] = 0

        for amount_iter in range(1, amount + 1):
            for coin in coins:
                if amount_iter - coin >= 0:
                    coin_count[amount_iter] = min(coin_count[amount_iter], 1 + coin_count[amount_iter - coin])

        return coin_count[amount] if coin_count[amount] != amount + 1 else -1

(Solution().coinChange([1, 2, 5], 11), 3)
(Solution().coinChange([2], 3), -1)
(Solution().coinChange([1], 0), 0)
(Solution().coinChange([2, 5, 10, 1], 27), 4)
(Solution().coinChange([186, 419, 83, 408], 6249), 20)

# greedy
class Solution:
    def coinChange(self, coins: list[int], amount: int) -> int:
        count = 0
        test = []

        for coin in sorted(coins, reverse=True):
            div = amount // coin
            mod = amount % coin

            count += div
            amount = mod

            test.append(div)

        return count if not amount else -1
        # return amount





# 152. Maximum Product Subarray
# https://leetcode.com/problems/maximum-product-subarray/description/
"""
Given an integer array nums, find a 
subarray
 that has the largest product, and return the product.

The test cases are generated so that the answer will fit in a 32-bit integer.

 

Example 1:

Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
Example 2:

Input: nums = [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
"""


# {Time: O(n), Memory: O(1)}
class Solution:
    def maxProduct(self, nums: list[int]) -> int:
        res = nums[0]
        curMin, curMax = 1, 1

        for n in nums:
            # if n == 0:
            #     curMin, curMax = 1, 1
            #     continue
            # curMin, _, curMax = sorted((n * curMax, n * curMin, n))
            # curMin, curMax = min(n * curMax, n * curMin), max(n * curMax, n * curMin)
            curMin, curMax = min(n * curMax, n * curMin, n), max(n * curMax, n * curMin, n)
            # tmp = curMax * n
            # curMax = max(n * curMax, n * curMin)
            # curMin = min(tmp, n * curMin)
            res = max(res, curMax)
        return res
(Solution().maxProduct([2, 3, -2, 4]), 6)
(Solution().maxProduct([0, 2]), 2)
(Solution().maxProduct([-2]), -2)
(Solution().maxProduct([-4, -3]), 12)


# O(n2)
import numpy as np


class Solution:
    def maxProduct(self, nums: list[int]) -> int:
        max_prod = -float("inf")
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                max_prod = max(max_prod, np.prod(nums[i:j + 1]))
                # print(nums[i:j + 1])
        return max_prod





# 152. Maximum Product Subarray
# https://leetcode.com/problems/maximum-product-subarray/description/
"""
Given an integer array nums, find a 
subarray
 that has the largest product, and return the product.

The test cases are generated so that the answer will fit in a 32-bit integer.

Example 1:

Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
Example 2:

Input: nums = [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
"""


class Solution:
    def maxProduct(self, nums: list[int]) -> int:
        max_prod = nums[0]
        min_num, max_num = 1, 1

        for num in nums:
            min_temp = min_num
            min_num = min(min_num*num, max_num*num, num)
            max_num = max(min_temp*num, max_num*num, num)

            max_prod = max(max_prod, max_num)
        return max_prod
(Solution().maxProduct([2, 3, -2, 4]), 6)
(Solution().maxProduct([-2, 0, -1]), 0)





# 139. Word Break
# https://leetcode.com/problems/word-break/description/
"""
Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.

Note that the same word in the dictionary may be reused multiple times in the segmentation.

 

Example 1:

Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
Example 2:

Input: s = "applepenapple", wordDict = ["apple","pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
Note that you are allowed to reuse a dictionary word.
Example 3:

Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: false
"""

class Solution:
    def wordBreak(self, s: str, wordDict: list[str]) -> bool:
        dp = [False] * len(s)
        dp.append(True)

        for ind in range(len(s) - 1, -1, -1):
            for word in wordDict:
                if word == s[ind: ind + len(word)]:
                    if dp[ind + len(word)]:
                        dp[ind] = True
                        break

        return dp[0]

(Solution().wordBreak("leetcode", ["leet", "code"]), True)
(Solution().wordBreak("cars", ["car", "ca", "rs"]), True)





# 300. Longest Increasing Subsequence
# https://leetcode.com/problems/longest-increasing-subsequence/description/
"""
Given an integer array nums, return the length of the longest strictly increasing 
subsequence

Example 1:

Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.
Example 2:

Input: nums = [0,1,0,3,2,3]
Output: 4
Example 3:

Input: nums = [7,7,7,7,7,7,7]
Output: 1
"""


class Solution:
    def lengthOfLIS(self, nums: list[int]) -> int:
        dp = [1] * len(nums)

        for i in range(len(nums) - 1, -1, -1):
            for j in range(i + 1, len(nums)):
                if nums[i] < nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)
(Solution().lengthOfLIS([10, 9, 2, 5, 3, 7, 101, 18]), 4)
(Solution().lengthOfLIS([0, 1, 0, 3, 2, 3]), 4)
(Solution().lengthOfLIS([7, 7, 7, 7, 7, 7, 7]), 1)





# 62. Unique Paths
# https://leetcode.com/problems/unique-paths/description/
"""
There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.

Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.

The test cases are generated so that the answer will be less than or equal to 2 * 109.

Example 1:


Input: m = 3, n = 7
Output: 28
Example 2:

Input: m = 3, n = 2
Output: 3
Explanation: From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Down -> Down
2. Down -> Down -> Right
3. Down -> Right -> Down
"""


class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        bottom_row = [1] * n

        for _ in range(m - 2, -1, -1):
            curr_row = [1] * n
            for i in range(n - 2, -1, -1):
                curr_row[i] = curr_row[i + 1] + bottom_row[i]
            bottom_row = curr_row

        return curr_row[0]
(Solution().uniquePaths(3, 7), 28)
(Solution().uniquePaths(3, 2), 3)






# 1143. Longest Common Subsequence
# https://leetcode.com/problems/longest-common-subsequence/description/
"""
Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.

A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

For example, "ace" is a subsequence of "abcde".
A common subsequence of two strings is a subsequence that is common to both strings.

Example 1:

Input: text1 = "abcde", text2 = "ace" 
Output: 3  
Explanation: The longest common subsequence is "ace" and its length is 3.
Example 2:

Input: text1 = "abc", text2 = "abc"
Output: 3
Explanation: The longest common subsequence is "abc" and its length is 3.
Example 3:

Input: text1 = "abc", text2 = "def"
Output: 0
Explanation: There is no such common subsequence, so the result is 0.
"""


class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        dp = [[0 for i in range(len(text2) + 1)] for j in range(len(text1) + 1)]
        
        for i in range(len(text1) - 1, -1, -1):
            for j in range(len(text2) -1, -1, -1):
                if text1[i] == text2[j]:
                    dp[i][j] = dp[i + 1][j + 1] + 1
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

        return dp[0][0]
(Solution().longestCommonSubsequence("abcde", "ace"), 3)
(Solution().longestCommonSubsequence("abc", "abc"), 3)
(Solution().longestCommonSubsequence("abc", "dew"), 0)





# 53. Maximum Subarray
# https://leetcode.com/problems/maximum-subarray/description/
"""
Given an integer array nums, find the subarray with the largest sum, and return its sum.

Example 1:

Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: The subarray [4,-1,2,1] has the largest sum 6.
Example 2:

Input: nums = [1]
Output: 1
Explanation: The subarray [1] has the largest sum 1.
Example 3:

Input: nums = [5,4,-1,7,8]
Output: 23
Explanation: The subarray [5,4,-1,7,8] has the largest sum 23.
"""


class Solution:
    def maxSubArray(self, nums: list[int]) -> int:
        result = nums[0]
        current = 0

        for num in nums:
            if current < 0:
                current = 0
            current += num
            result = max(result, current)
        return result
(Solution().maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4]), 6)
(Solution().maxSubArray([1]), 1)
(Solution().maxSubArray([5, 4, -1, 7, 8]), 23)





# 55. Jump Game
# https://leetcode.com/problems/jump-game/description/
"""
You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.

Return true if you can reach the last index, or false otherwise.

Example 1:

Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
Example 2:

Input: nums = [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.
"""


class Solution:
    def canJump(self, nums: list[int]) -> bool:
        curr = 0

        for num in nums[:-1]: # no need to check the last element
            curr = max(num, curr - 1)
            if curr < 1:
                return False

        return True
(Solution().canJump([2, 3, 1, 1, 4]), True)
(Solution().canJump([3, 2, 1, 0, 4]), False)


class Solution:
    def canJump(self, nums: list[int]) -> bool:
        stop = len(nums) - 1

        for num in range(len(nums) - 1)[::-1]:
            if num + nums[num] >= stop:
                stop = num

        return not stop 





# 252. Meeting Rooms
"""
Question
Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), determine if a person could attend all meetings.

Example 1:

Input: [[0,30],[5,10],[15,20]]
Output: false
Example 2:

Input: [[7,10],[2,4]]
Output: true
"""


class Solution:
    # @param A : list of list of integers
    # @return an integer
    def solve(self, intervals):
        intervals.sort(key=lambda x: x[0])
        # return all([intervals[ind + 1][0] >= intervals[ind][1] for ind in range(len(intervals) - 1) ])
        for ind in range(len(intervals) - 1):
            if intervals[ind][1] > intervals[ind + 1][0]:
                return False
        return True


(Solution().solve([[0, 30], [5, 10], [15, 20]]), False)
(Solution().solve([[5, 10], [15, 20]]), True)





# 253. Meeting Rooms II
"""
Question
Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), find the minimum number of conference rooms required.

For example, Given [[0, 30],[5, 10],[15, 20]], return 2.
"""


# O(n2)
import numpy as np
class Solution:
    def solve(self, intervals):
        time = [0] * np.max(intervals)
        for interval in intervals:
            for i in range(*interval):
                time[i] +=1
        return time
(Solution().solve([[0, 30], [5, 10], [15, 20]]), 2)
(Solution().solve([[5, 10], [15, 20]]), 1)





# 57. Insert Interval
# https://leetcode.com/problems/insert-interval/description/
"""
You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] represent the start and the end of the ith interval and intervals is sorted in ascending order by starti. You are also given an interval newInterval = [start, end] that represents the start and end of another interval.

Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).

Return intervals after the insertion.

Note that you don't need to modify intervals in-place. You can make a new array and return it.

 

Example 1:

Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]
Example 2:

Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].
"""


class Solution:
    def insert(self, intervals: list[list[int]], newInterval: list[int]) -> list[list[int]]:
        sol = []

        for ind, interval in enumerate(intervals):
            if newInterval[1] < interval[0]:
                sol.append(newInterval)
                return sol + intervals[ind:]
            elif newInterval[0] > interval[1]:
                sol.append(interval)
            else:
                newInterval[0] = min(newInterval[0], interval[0])
                newInterval[1] = max(newInterval[1], interval[1])
        
        sol.append(newInterval)
        return sol


(Solution().insert([[1, 3], [6, 9]], [2, 5]), [[1, 5], [6, 9]])
(Solution().insert([[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], [4, 8]), [[1, 2], [3, 10], [12, 16]])
(Solution().insert([], [5, 7]), [[5, 7]])
(Solution().insert([[1, 5]], [2, 3]), [[1, 5]])





# 56. Merge Intervals
# https://leetcode.com/problems/merge-intervals/description/
"""
Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

Example 1:

Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].
Example 2:

Input: intervals = [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considered overlapping.
"""


# O(nlogn)
class Solution:
    def merge(self, intervals: list[list[int]]) -> list[list[int]]:
        sol = []
        intervals.sort(key=lambda x: x[0])

        for ind, interval in enumerate(intervals[:-1]):
            if intervals[ind][1] >= intervals[ind + 1][0]:
                intervals[ind + 1][0] = min(interval[0], intervals[ind + 1][0])
                intervals[ind + 1][1] = max(interval[1], intervals[ind + 1][1])
            else:
                sol.append(interval)
        
        sol.append(intervals[-1])
        return sol


(Solution().merge([[1, 3], [2, 6], [8, 10], [15, 18]]), [[1, 6], [8, 10], [15, 18]])
(Solution().merge([[1, 4], [4, 5]]), [[1, 5]])
(Solution().merge([[1, 4], [0, 0]]), [[0, 0], [1, 4]])


class Solution:
    def merge(self, intervals: list[list[int]]) -> list[list[int]]:
        intervals.sort(key=lambda pair: pair[0])
        sol = [intervals[0]]

        for interval in intervals[1:]:
            if interval[0] <= sol[-1][1]:
                (sol[-1][1]) = max(sol[-1][1], interval[1])
            else:
                sol.append(interval)

        return sol





# 435. Non-overlapping Intervals
# https://leetcode.com/problems/non-overlapping-intervals/description/
"""
Given an array of intervals intervals where intervals[i] = [starti, endi], return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

 

Example 1:

Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
Output: 1
Explanation: [1,3] can be removed and the rest of the intervals are non-overlapping.
Example 2:

Input: intervals = [[1,2],[1,2],[1,2]]
Output: 2
Explanation: You need to remove two [1,2] to make the rest of the intervals non-overlapping.
Example 3:

Input: intervals = [[1,2],[2,3]]
Output: 0
Explanation: You don't need to remove any of the intervals since they're already non-overlapping.
"""


class Solution:
    def eraseOverlapIntervals(self, intervals: list[list[int]]) -> int:
        intervals.sort(key=lambda x: (x[0], x[1]))
        curr_stop = intervals[0][1]
        counter = 0

        for start, stop in intervals[1:]:
            if start < curr_stop:
                counter += 1
                curr_stop = min(curr_stop, stop)
            else:
                curr_stop = stop

        return counter


(Solution().eraseOverlapIntervals([[1, 2], [2, 3], [3, 4], [1, 3]]), 1)
(Solution().eraseOverlapIntervals([[1, 2], [1, 2], [1, 2]]), 2)
(Solution().eraseOverlapIntervals([[1, 2], [2, 3]]), 0)
(Solution().eraseOverlapIntervals([[1, 100], [11, 22], [1, 11], [2, 12]]), 2)
(Solution().eraseOverlapIntervals([[-52, 31], [-73, -26], [82, 97], [-65, -11], [-62, -49], [95, 99], [58, 95], [-31, 49], [66, 98], [-63, 2], [30, 47], [-40, -26]]), 7)





# 48. Rotate Image
# https://leetcode.com/problems/rotate-image/description/
"""
You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

Example 1:

Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]
Example 2:

Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
"""


class Solution:
    def rotate(self, matrix: list[list[int]]) -> None:
        left, right = 0, len(matrix) - 1

        while left < right:

            top = left
            down = right

            for i in range(right - left):
                temp = matrix[top][left + i]
                matrix[top][left + i] = matrix[down - i][left]
                matrix[down - i][left] = matrix[down][right - i]
                matrix[down][right - i] = matrix[top + i][right]
                matrix[top + i][right] = temp

            left += 1
            right -= 1
        return matrix

(Solution().rotate([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), [[7, 4, 1], [8, 5, 2], [9, 6, 3]])
(Solution().rotate([[5, 1, 9, 11], [2, 4, 8, 10], [13, 3, 6, 7], [15, 14, 12, 16]]), [[15, 13, 2, 5], [14, 3, 4, 1], [12, 6, 8, 9], [16, 7, 10, 11]])


import numpy as np
class Solution:
    def rotate(self, matrix: list[list[int]]) -> None:
        matrix = np.rot90(matrix, -1, (0, 1))
        return matrix





# 54. Spiral Matrix
# https://leetcode.com/problems/spiral-matrix/description/
"""
Given an m x n matrix, return all elements of the matrix in spiral order.

Example 1:

Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,6,9,8,7,4,5]
Example 2:

Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]
"""


# O(n*m), O(1)
class Solution:
    def spiralOrder(self, matrix: list[list[int]]) -> list[int]:
        sol = []

        while matrix:
            for ind in range(len(matrix[0])):
                sol.append(matrix[0][ind])
            matrix.pop(0)
            if not matrix or not matrix[0]:
                return sol

            right = len(matrix[0]) - 1
            for ind in range(len(matrix)):
                sol.append(matrix[ind][right])
                matrix[ind].pop(right)
            if not matrix or not matrix[0]:
                return sol

            down = len(matrix) - 1
            for ind in range(len(matrix[0]))[::-1]:
                sol.append(matrix[down][ind])
            matrix.pop(-1)
            if not matrix or not matrix[0]:
                return sol

            for ind in range(len(matrix))[::-1]:
                sol.append(matrix[ind][0])
                matrix[ind].pop(0)
            if not matrix or not matrix[0]:
                return sol

        return sol


(Solution().spiralOrder([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), [1, 2, 3, 6, 9, 8, 7, 4, 5])
(Solution().spiralOrder([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]), [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7])
(Solution().spiralOrder([[7], [9], [6]]), [7, 9, 6])





# 73. Set Matrix Zeroes
# https://leetcode.com/problems/set-matrix-zeroes/description/
"""
Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.

You must do it in place.

Example 1:

Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]
Example 2:

Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]
"""


# O(n*m), O(n+m)
class Solution:
    def setZeroes(self, matrix: list[list[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        zeros_r = set()
        zeros_c = set()
        rows = len(matrix)
        cols = len(matrix[0])

        for r in range(rows):
            for c in range(cols):
                if matrix[r][c] == 0:
                    zeros_r.add(r)
                    zeros_c.add(c)

        for r in range(rows):
            for c in range(cols):
                if r in zeros_r or c in zeros_c:
                    matrix[r][c] = 0

        return matrix
(Solution().setZeroes([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), [[1, 0, 1], [0, 0, 0], [1, 0, 1]])
(Solution().setZeroes([[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]), [[0, 0, 0, 0], [0, 4, 5, 0], [0, 3, 1, 0]])
(Solution().setZeroes([[1, 2, 3, 4], [5, 0, 7, 8], [0, 10, 11, 12], [13, 14, 15, 0]]), [[0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])





# 191. Number of 1 Bits
# https://leetcode.com/problems/number-of-1-bits/description/
"""
Write a function that takes the binary representation of a positive integer and returns the number of 
set bits
 it has (also known as the Hamming weight).

 

Example 1:

Input: n = 11

Output: 3

Explanation:

The input binary string 1011 has a total of three set bits.

Example 2:

Input: n = 128

Output: 1

Explanation:

The input binary string 10000000 has a total of one set bit.

Example 3:

Input: n = 2147483645

Output: 30

Explanation:

The input binary string 1111111111111111111111111111101 has a total of thirty set bits.
"""

class Solution:
    def hammingWeight(self, n: int) -> int:
        # return bin(n).count("1")
        counter = 0
        
        while n:
            counter += n % 2
            n = n >> 1

        return counter    
Solution().hammingWeight(11)
Solution().hammingWeight(128)





# 338. Counting Bits
# https://leetcode.com/problems/counting-bits/description/
"""
Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.

Example 1:

Input: n = 2
Output: [0,1,1]
Explanation:
0 --> 0
1 --> 1
2 --> 10
Example 2:

Input: n = 5
Output: [0,1,1,2,1,2]
Explanation:
0 --> 0
1 --> 1
2 --> 10
3 --> 11
4 --> 100
5 --> 101
 
"""



class Solution:
    def countBits(self, n: int) -> list[int]:
        sol = []
        for i in range(n + 1):
            counter = 0
            while i:
                counter += i % 2
                i = i >> 1
            sol.append(counter)
        return sol
    
class Solution:
    def countBits(self, n: int) -> list[int]:
        return[bin(i).count("1") for i in range(n + 1)]

class Solution:
    def countBits(self, n: int) -> list[int]:
        dp = [0] * (n + 1)
        offset = 1

        for i in range(1, n + 1):
            if offset * 2 == i:
                offset = i
            dp[i] = 1 + dp[i - offset]
        return dp




 
# 


















 







 











