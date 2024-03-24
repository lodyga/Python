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
"""A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

Given a string s, return true if it is a palindrome, or false otherwise.

 

Example 1:

Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome."""


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
"""Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number. Let these two numbers be numbers[index1] and numbers[index2] where 1 <= index1 < index2 <= numbers.length.

Return the indices of the two numbers, index1 and index2, added by one as an integer array [index1, index2] of length 2.

The tests are generated such that there is exactly one solution. You may not use the same element twice.

Your solution must use only constant extra space.

Example 1:

Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore, index1 = 1, index2 = 2. We return [1, 2]."""


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





class Solution:
    def isValid(self, s: str) -> bool:
        seen = []
        oppos_bracket = {")": "(", "]": "[", "}": "{"}
        for bracket in s:
            # if seen and oppos_bracket.get(bracket, "foo") == seen[-1]:
            if seen and seen[-1] == oppos_bracket[bracket]:
                seen.pop()
            else:
                seen.append(bracket)
        return not bool(seen)

Solution().isValid("[(])")
Solution().isValid("()")
Solution().isValid("([])")


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
        return not bool(seen)







