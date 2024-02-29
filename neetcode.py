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





# 




