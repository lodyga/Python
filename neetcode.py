https://neetcode.io/practice

# 217. Contains Duplicate
# https://leetcode.com/problems/contains-duplicate/
"""
Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.

Example 1:

Input: nums = [1, 2, 3, 1]
Output: true

Example 2:

Input: nums = [1, 2, 3, 4]
Output: false

Example 3:

Input: nums = [1, 1, 1, 3, 3, 4, 3, 2, 4, 2]
Output: true
"""

# O(n), O(n)
class Solution:
    def containsDuplicate(self, nums: list[int]) -> bool:
        seen = set()
    
        for num in nums:
            if num in seen:
                return True
            else:
                seen.add(num)
        return False
        
        # alt solution 
        # return not (len(set(nums)) == len(nums))

        # alt solution
        # import numpy as np
        # return len(np.unique(nums)) == len(nums)


(Solution().containsDuplicate([1, 2, 3]), False)
(Solution().containsDuplicate([1, 2, 3, 4]), False)
(Solution().containsDuplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]), True)

sol = Solution()
sol.containsDuplicate([1, 2, 3])





# Valid Anagram
# https://leetcode.com/problems/valid-anagram/
"""
Given two strings s and t, return true if t is an anagram of s, and false otherwise.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

Example 1:

Input: s = "anagram", t = "nagaram"
Output: true
Example 2:

Input: s = "rat", t = "car"
Output: false
"""


# O(n), O(n)
class Solution:
    def counter(self, word):
        counter = {}

        for letter in word:
            counter[letter] = counter.get(letter, 0) + 1

        return counter

    def isAnagram(self, word1, word2):
        return self.counter(word1) == self.counter(word2)
(Solution().isAnagram("anagram", "nagaram"), True)
(Solution().isAnagram("rat", "car"), False)


from collections import Counter

class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return Counter(s) == Counter(t)
    




# Two Sum
# https://leetcode.com/problems/two-sum/description/
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
        seen = {}

        for ind, num in enumerate(nums):
            diff = target - num
        
            if diff in seen: # seen.get(diff, False)
                return [seen[diff], ind] # [seen.get(diff), ind]
            seen[num] = ind # seen.update({num: ind})
        
        return None
(Solution().twoSum([2, 7, 11, 15], 9), [0, 1])
(Solution().twoSum([3, 2, 4], 6), [1, 2])
(Solution().twoSum([3, 3], 6), [0, 1])
(Solution().twoSum([3, 3], 7), None)

    # alt solution
    #     for i in range(len(nums) - 1):  # 1600, 17; O(n2), O(1)
    #         for j in range(i + 1, len(nums)):
    #             if nums[i] + nums[j] == target:
    #                 return [i, j]





# Group Anagrams
# https://leetcode.com/problems/group-anagrams/description/
"""
Given an array of strings strs, group the anagrams together. You can return the answer in any order.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

Example 1:

Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
"""


# O(m*n), O(m*n)
class Solution:
    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
        grouped_anagrams = dict()
        # from collections import defaultdict
        # grouped_anagrams = defaultdict(list)

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
(Solution().groupAnagrams(["eat","tea","tan","ate","nat","bat"]), [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']])
(Solution().groupAnagrams([""]), [[""]])
(Solution().groupAnagrams(["a"]), [["a"]])
(Solution().groupAnagrams(["tin","ram","zip","cry","pus","jon","zip","pyx"]), [['tin'], ['ram'], ['zip', 'zip'], ['cry'], ['pus'], ['jon'], ['pyx']])

# alt solutions # 83, 19 O(m*n*logn) m - list cout, n - avg word len
class Solution:
    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
        grouped_anagrams = {}
        
        for word in strs:
            key = "".join(sorted(word))
            if key in grouped_anagrams:
                grouped_anagrams[key].append(word)
            else:
                grouped_anagrams[key] = [word]
        return list(grouped_anagrams.values())


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





# Top K Frequent Elements
# https://leetcode.com/problems/top-k-frequent-elements/description/
"""
Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.

Example 1:

Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
Example 2:

Input: nums = [1], k = 1
Output: [1]
"""


# bucket as dict
# It's actually faster to use a dict and sort its keys than to traverse through a list.
class Solution:
    def topKFrequent(self, nums, solution_len):
        solution = []
        counts = {}
        bucket = {}

        # counts values
        for num in nums:
            counts[num] = counts.get(num, 0) + 1

        # reverse counts {key: val} pairs as bucket
        for key, val in counts.items():
            if not val in bucket:
                bucket[val] = []
            bucket[val].append(key)

        # sort frequencies descending
        keys = sorted(bucket.keys(), reverse=True)

        # get top solution_len values
        for key in keys:
            for number in bucket[key]:
                solution.append(number)
                if len(solution) == solution_len:
                    return solution

        return -1
(Solution().topKFrequent([1, 1, 1, 2, 2, 3], 2), [1, 2])
(Solution().topKFrequent([1], 1), [1])


# O(n), O(n)
# bucket as list of lists
# It's actually faster to use a dict and sort its keys than to traverse through a list.
class Solution:
    def topKFrequent(self, nums, solution_len):
        solution = []
        counts = {}
        bucket = [[] for _ in range(len(nums) + 1)]
        
        # counts values
        for num in nums:
            counts[num] = counts.get(num, 0) + 1

        # bucket as a list of lists
        # [[], [3], [2], [1], [], [], []]
        for key, val in counts.items():
            bucket[val].append(key)
        
        # get top solution_len values
        for numbers in bucket[::-1]:
            for number in numbers:
                solution.append(number)
                if len(solution) == solution_len:
                    return solution

        return -1


from collections import Counter

# use Counter from collections
class Solution:
    def topKFrequent(self, nums: list[int], k: int) -> list[int]:
        return [key for key, _ in Counter(nums).most_common(k)]





# Product of Array Except Self
# https://leetcode.com/problems/product-of-array-except-self/
"""
Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operation.

Example 1:

Input: nums = [1,2,3,4]
Output: [24,12,8,6]
"""


# O(n), O(n)
# using 3 loops
class Solution:
    def productExceptSelf(self, nums: list[int]) -> list[int]:
        left_prod_cum = [1] * len(nums)  # prefix
        right_prod_cum = [1] * len(nums)  # postfix

        # cummulative prod from left
        for ind, num in enumerate(nums[:-1], 1):
            left_prod_cum[ind] = left_prod_cum[ind - 1] * num

        # cummulative prod from right
        for ind, num in enumerate(nums[1:][::-1], 1):
            right_prod_cum[len(nums) - 1 - ind] = right_prod_cum[len(nums) - ind] * num

        return [left_prod_cum[i] * right_prod_cum[i] for i in range(len(nums))]
(Solution().productExceptSelf([2, 3, 4, 5]), [60, 40, 30, 24])
(Solution().productExceptSelf([1, 2, 3, 4]), [24, 12, 8, 6])


"""
solution draft
[(1) * 3*4*5=60, 2 * 4*5=40, 2*3 * 5=30, 2*3*4 * (1)=24]
cummulative prod
from left
all elements
[2, 2*3=6, 2*3*4=24, 2*3*4*5=120]
without the last element
[1, 2, 6, 24]
from right
all elements
[2*3*4*5=120, 3*4*5=60, 4*5=20, 5]
without the first element
[60, 20, 5, 1]
"""


# O(n), O(n)
# using 2 loops, memory: one list
class Solution:
    def productExceptSelf(self, nums: list[int]) -> list[int]:
        left_prod_cum = 1
        right_prod_cum = [1] * len(nums)

        # cummulative prod from right
        for ind, num in enumerate(nums[1:][::-1], 1):
            right_prod_cum[len(nums) - 1 - ind] = right_prod_cum[len(nums) - ind] * num

        # cummulative prod from left
        for ind, num in enumerate(nums[:-1]):
            # append cum prod from right in this loop
            right_prod_cum[ind] *= left_prod_cum
            left_prod_cum *= num

        # append the last left_prod_cum
        right_prod_cum[-1] = left_prod_cum

        return right_prod_cum


# O(n2), O(n)
import numpy as np

class Solution:
    def productExceptSelf(self, nums: list[int]) -> list[int]:
        return [np.prod(nums[:i]) * np.prod(nums[i + 1:]) for i in range(len(nums))]





# Encode and Decode Strings
# https://www.lintcode.com/problem/659/
"""
Description
Design an algorithm to encode a list of strings to a string. The encoded string is then sent over the network and is decoded back to the original list of strings.

Please implement encode and decode

Because the string may contain any of the 256 legal ASCII characters, your algorithm must be able to handle any character that may appear

Do not rely on any libraries, the purpose of this problem is to implement the "encode" and "decode" algorithms on your own
"""


class Solution:
    """
    @param: strs: a list of strings
    @return: encodes a list of strings to a single string.
    """

    def encode(self, words):
        return "".join(map(lambda word: f"Τ{len(word)}{word}", words))

    """
    @param: str: A string
    @return: decodes a single string to a list of strings
    """

    def decode(self, word):
        decoded = []
        
        while word:
            if word[0] == "Τ":
                word = word[1:]
                length = ""
                
                while word and word[0].isdigit():
                    length += word[0]
                    word = word[1:]

                length = int(length)
                decoded.append(word[:length])
                word = word[length:]
        
        return decoded
(Solution().encode(["code", "site", "love", "you"]), "Τ4codeΤ4siteΤ4loveΤ3you")
(Solution().decode(Solution().encode(["code", "site", "love", "you"])), ["code", "site", "love", "you"])
(Solution().decode(Solution().encode([])), [])
(Solution().decode(Solution().encode([""])), [""])


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





# Longest Consecutive Sequence
# https://leetcode.com/problems/longest-consecutive-sequence/description/
"""
Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.

You must write an algorithm that runs in O(n) time.

Example 1:

Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.
Example 2:

Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9
"""


# O(n), O(n)
class Solution:
    def longestConsecutive(self, nums: list[int]) -> int:
        nums_set = set(nums)
        longest_concec = 0

        for num in nums_set:
            if not num - 1 in nums_set:
                curr_len = 1
                
                while num + curr_len in nums_set:
                    curr_len += 1
                longest_concec = max(longest_concec, curr_len)

        return longest_concec
(Solution().longestConsecutive([100, 4, 200, 1, 3, 2]), 4)
(Solution().longestConsecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]), 9)





# 125. Valid Palindrome
# https://leetcode.com/problems/valid-palindrome/description/
"""
A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

Given a string s, return true if it is a palindrome, or false otherwise.

Example 1:

Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.
Example 2:

Input: s = "race a car"
Output: false
Explanation: "raceacar" is not a palindrome.
Example 3:

Input: s = " "
Output: true
Explanation: s is an empty string "" after removing non-alphanumeric characters.
Since an empty string reads the same forward and backward, it is a palindrome.
"""


# Two Pointers
class Solution:
    def isPalindrome(self, sentence):
        l = 0
        r = len(sentence) - 1

        while l < r:
            while l < r and not sentence[l].isalnum():
                l += 1
            
            while l < r and not sentence[r].isalnum():
                r -= 1

            if sentence[l].lower() != sentence[r].lower():
                return False
            else:
                l += 1
                r -= 1

        return True
(Solution().isPalindrome("A man, a plan, a canal: Panama"), True)
(Solution().isPalindrome("race a car"), False)
(Solution().isPalindrome(" "), True)
(Solution().isPalindrome("0P"), False)


# Two Pointers, import string
import string
class Solution:
    def isPalindrome(self, s: str) -> bool:
        l = 0
        r = len(s) - 1

        while l < r:
            while s[l] in string.punctuation + " " and l < r:
                l += 1

            while s[r] in string.punctuation + " "  and l < r:
                r -= 1
            
            if s[l].lower() != s[r].lower():
                return False

            l += 1
            r -= 1
        
        return True


# replace
import string
class Solution:
    def isPalindrome(self, s: str) -> bool:
        for char in string.punctuation + " ":
            s = s.replace(char, "")
        return s.lower() == s[::-1].lower()
Solution().isPalindrome("A man, a plan, a canal: Panama")


# regex
import re
class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = re.sub(r"[\W_]", "", s).lower()
        return s == s[::-1]
Solution().isPalindrome("A man, a plan, a canal: Panama")


# comprehension list, isalpha()
class Solution:
    def isPalindrome(self, s: str) -> bool:
        cleaned_s = [alph.lower() for alph in s if alph.isalnum()]
        return cleaned_s == cleaned_s[::-1]
Solution().isPalindrome("A man, a plan, a canal: Panama")


# comprehension list
import string
class Solution:
    def isPalindrome(self, s: str) -> bool:
        # filtered_s = list(filter(lambda x: not x in string.punctuation + " ", s))
        filtered_s = list(alph.lower() for alph in s if not alph in string.punctuation + " ")
        # return "".join(filtered_s).lower() == "".join(filtered_s[::-1]).lower()
        return filtered_s == filtered_s[::-1]
Solution().isPalindrome("A man, a plan, a canal: Panama")





# Two Sum II - Input Array Is Sorted
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
Example 2:

Input: numbers = [2,3,4], target = 6
Output: [1,3]
Explanation: The sum of 2 and 4 is 6. Therefore index1 = 1, index2 = 3. We return [1, 3].
Example 3:

Input: numbers = [-1,0], target = -1
Output: [1,2]
Explanation: The sum of -1 and 0 is -1. Therefore index1 = 1, index2 = 2. We return [1, 2].
"""


# O(n), O(1)
class Solution:
    def twoSum(self, numbers: list[int], target: int) -> list[int]:
        left = 0
        right = len(numbers) - 1

        while left < right:
            curr_sum = numbers[left] + numbers[right]
            if curr_sum > target:
                right -= 1
            elif curr_sum < target:
                left += 1
            else:
                return [left + 1, right + 1]
        
        return None
(Solution().twoSum([2, 7, 11, 15], 9), [1, 2])
(Solution().twoSum([2, 3, 4], 6), [1, 3])
(Solution().twoSum([-1, 0], -1), [1, 2])





# 3Sum
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
Explanation: The only possible triplet sums up to 0.
"""


class Solution:
    def threeSum(self, nums: list[int]) -> list[list[int]]:
        solution = []
        nums.sort()

        for index, num in enumerate(nums[:-2]):
            # Skip positive nums
            if num > 0:
                break
            
            # Skip same num values
            if index and nums[index] == nums[index - 1]:
                continue

            left = index + 1
            right = len(nums) - 1

            # two pointers
            while left < right:
                curr_sum = num + nums[left] + nums[right]
                
                if curr_sum < 0:  # if sum is less than 0
                    left += 1
                elif curr_sum > 0:  # if sum is geater than 0
                    right -= 1
                else:  # if sum is equal to 0
                    solution.append([num, nums[left], nums[right]])
                    left += 1
                    right -= 1
                    
                    # skip same left pointer values
                    while left < right and nums[left] == nums[left - 1]:
                        left += 1                
        
        return solution
(Solution().threeSum([-1, 0, 1, 2, -1, -4]), [[-1, -1, 2], [-1, 0, 1]])
(Solution().threeSum([3, 0, -2, -1, 1, 2]), [[-2, -1, 3], [-2, 0, 2], [-1, 0, 1]])
(Solution().threeSum([1, 1, -2]), [[-2, 1, 1]])
(Solution().threeSum([-1, 1, 1]), [])
(Solution().threeSum([0, 0, 0]), [[0, 0, 0]])
(Solution().threeSum([-2, 0, 0, 2, 2]), [[-2, 0, 2]])


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





# Container With Most Water
# https://leetcode.com/problems/container-with-most-water/description/
"""
You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

Notice that you may not slant the container.

Example 1:

Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.
Example 2:

Input: height = [1,1]
Output: 1
"""


# O(n), O(1)
class Solution:
    def maxArea(self, heights: list[int]) -> int:
        max_area = 0
        left = 0
        right = len(heights) - 1

        while left < right:
            curr_area = (right - left) * min(heights[left], heights[right])
            max_area = max(max_area, curr_area)

            if heights[left] < heights[right]:
                left += 1
            else:
                right -= 1

        return max_area
(Solution().maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]), 49)
(Solution().maxArea([1, 1]), 1)
(Solution().maxArea([2, 3, 4, 5, 18, 17, 6]), 17)


# O(n2)
class Solution:
    def maxArea(self, height: list[int]) -> int:
        max_area = 0

        for i, h1 in enumerate(height[:-1]):
            for j, h2 in enumerate(height[i+1:]):
                curr_area = min(h1, h2) * (j + 1)
                if  curr_area > max_area:
                    max_area = curr_area

        return max_area





# Trapping Rain Water
# https://leetcode.com/problems/trapping-rain-water/
class Solution:
    def trap(self, heights):
        left = 0
        right = len(heights) - 1
        left_max_height = 0
        right_max_height = 0
        trapped_water = 0

        # two pointers
        while left < right:
            if heights[left] < heights[right]:  # choose the lower height because it gets to higher eventually
                left_max_height = max(left_max_height, heights[left])
                curr_water = left_max_height - heights[left]
                trapped_water += curr_water
                left += 1
            else:
                right_max_height = max(right_max_height, heights[right])
                curr_water = right_max_height - heights[right]
                trapped_water += curr_water
                right -= 1

        return trapped_water
(Solution().trap([1, 3, 2, 1, 2, 1, 5, 3, 3, 4, 2]), 8)
(Solution().trap([5, 8]), 0)
(Solution().trap([3, 1, 2]), 1)
(Solution().trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]), 6)
(Solution().trap([4, 2, 0, 3, 2, 5]), 9)





# Best Time to Buy and Sell Stock
# https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/
"""
You are given an array prices where prices[i] is the price of a given stock on the ith day.

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


# pointers as values
class Solution:
    def maxProfit(self, prices):
        left = prices[0]  # the left pointer as a value
        max_profit = 0

        for right in prices[1:]:  # the right pointer as a value
            if left > right:  # if price is lower buy
                left = right
            else:  # if price is higher calculate revenue
                current_profit = right - left
                max_profit = max(max_profit, current_profit)

        return max_profit
(Solution().maxProfit([7, 1, 5, 3, 6, 4]), 5)
(Solution().maxProfit([7, 6, 4, 3, 1]), 0)
(Solution().maxProfit([2, 4, 1]), 2)
(Solution().maxProfit([2, 1, 2, 1, 0, 1, 2]), 2)
(Solution().maxProfit([1, 2]), 1)


# pointers as indexes
class Solution:
    def maxProfit(self, prices):
        max_profit = 0
        left = 0  # the left pointer
        right = 1  # the right pointer

        while right < len(prices):  # bound the right pointer
            if prices[left] > prices[right]:  # if price is lower buy
                left = right
            else:  # if price is higher calculate revenue
                current_profit = prices[right] - prices[left]
                max_profit = max(max_profit, current_profit)

            right += 1

        return max_profit





# Longest Substring Without Repeating Characters
# https://leetcode.com/problems/longest-substring-without-repeating-characters/
"""
Given a string s, find the length of the longest 
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
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
"""


# O(n), O(n)
# window as set()
# both pointers as indexes
class Solution:
    def lengthOfLongestSubstring(self, word):
        window = set()  # slidiong window without repeating characters
        longest_substring = 0
        left = 0  # left pointer

        for right in range(len(word)):  # right pointer
            while word[right] in window:  # if duplicate found
                window.discard(word[left])  # remove (discard) that charactr with every preceding character
                left += 1  # increase the left pointer

            window.add(word[right])  # add an unique letter
            longest_substring = max(longest_substring, right - left + 1)  # update the length of the longest substring

        return longest_substring
(Solution().lengthOfLongestSubstring("abcabcbb"), 3)
(Solution().lengthOfLongestSubstring("bbbbb"), 1)
(Solution().lengthOfLongestSubstring("pwwkew"), 3)
(Solution().lengthOfLongestSubstring("aabaab!bb"), 3)
(Solution().lengthOfLongestSubstring("aab"), 2)


# window as set()
# right pointer as a value (letter), left as index
class Solution:
    def lengthOfLongestSubstring(self, word):
        window = set()  # slidiong window without repeating characters
        longest_substring = 0
        index = 0  # index of the first 'letter' in the 'word'

        for letter in word:  # right pointer as a value
            while letter in window:  # if duplicate found
                window.remove(word[index])  # discard (remove) that charactr with every preceding character
                index += 1  # increase the index

            window.add(letter)  # add an unique letter
            longest_substring = max(longest_substring, len(window))  # update the length of the longest substring

        return longest_substring





# Longest Repeating Character Replacement
# https://leetcode.com/problems/longest-repeating-character-replacement/
"""
You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character.
You can perform this operation at most k times.

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
There may exists other ways to achieve this answer too.
"""


class Solution:
    def characterReplacement(self, word, joker):
        window = {}  # window as dict
        left = 0  # left pointer
        longest = 0

        for right in range(len(word)):  # right pointer
            window[word[right]] = window.get(word[right], 0) + 1  # add new letter to window
            
            while (right - left + 1) - max(window.values()) > joker:  # check ("sum - max") how many character are replaced, if exeded
                window[word[left]] -= 1  # remove "left" character from the window
                left += 1

            longest = max(longest, (right - left + 1))  # update the length of the longest replacement
            # (right - left + 1) is much faster than sum(window.values())

        return longest
(Solution().characterReplacement("ABAB", 2), 4)
(Solution().characterReplacement("AABABBA", 1), 4)





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
(Solution().isValid("(]"), False)
(Solution().isValid(""), True)
(Solution().isValid("["), False)





# Find Minimum in Rotated Sorted Array
# https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
"""
Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:

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
    def findMin(self, nums):
        left = 0
        right = len(nums) - 1
        curr_min = nums[0]  # assign some value

        while left <= right:  # two pointers 
            mid = (left + right) // 2  # get the middle index
            curr_min = min(curr_min, nums[mid])  # check if the value in the middle it lower than currten min

            if nums[mid] < nums[right]: # if the middle value is lower than the most right value
                right = mid - 1  # then the left part should be searched
            else:
                left = mid + 1  # else the right part should be searched

        return curr_min
(Solution().findMin([1, 2, 3, 4]), 1)
(Solution().findMin([4, 1, 2, 3]), 1)
(Solution().findMin([2, 3, 4, 1]), 1)
(Solution().findMin([3, 4, 1, 2]), 1)
(Solution().findMin([4, 5, 1, 2, 3]), 1)
(Solution().findMin([5, 1, 2, 3, 4]), 1)
(Solution().findMin([1, 2, 3, 4, 5]), 1)
(Solution().findMin([2, 3, 4, 5, 1]), 1)
(Solution().findMin([3, 4, 5, 1, 2]), 1)
(Solution().findMin([4, 5, 6, 7, 0, 1, 2]), 0)
(Solution().findMin([11, 13, 15, 17]), 11)
(Solution().findMin([1]), 1)
(Solution().findMin([3, 1, 2]), 1)


class Solution:
    def findMin(self, nums: list[int]) -> int:
        l = 0
        r = len(nums) - 1
        
        while True:
            if r - l < 2:
                return min(nums[l], nums[r])

            mid = (l + r)//2
            
            if nums[mid] < nums[r]:
                r = mid
            else:
                l = mid





# Search in Rotated Sorted Array
# https://leetcode.com/problems/search-in-rotated-sorted-array/
"""
There is an integer array nums sorted in ascending order (with distinct values).

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


# 4, 5, 1, 2, 3     2
# 4, 5, 1, 2, 3     4
# 5, 1, 2, 3, 4     5

# 3, 4, 5, 1, 2     2
# 2, 3, 4, 5, 1     5
# 3, 4, 5, 1, 2     4
# 2, 3, 4, 5, 1

class Solution:
    def esearch(self, nums, target):
        left = 0
        right = len(nums) - 1

        while left <= right:  # two pointers
            mid = (left + right) // 2  # get middle index

            if target == nums[mid]:  # if target found
                return mid
            elif nums[mid] < nums[right]:  # [5, 1, 2, 3, 4] the right chunk [3, 4] is ascending the other has a pivot
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
            else:  # [2, 3, 4, 5, 1] the left chunk [2, 3] is ascending the other has a pivot
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1

        return -1
(Solution().search([4, 5, 6, 7, 8, 1, 2, 3], 8), 4)
(Solution().search([1, 3, 5], 5), 2)
(Solution().search([3, 5, 1], 3), 0)
(Solution().search([3, 5, 1], 1), 2)
(Solution().search([5, 1, 3], 3), 2)
(Solution().search([4, 5, 6, 7, 0, 1, 2], 0), 4)
(Solution().search([4, 5, 6, 7, 0, 1, 2], 3), -1)
(Solution().search([1], 0), -1)
(Solution().search([5, 1, 3], 4), -1)
(Solution().search([4, 5, 6, 7, 8, 1, 2, 3], 8), 4)


# check if target is in pivoted part
class Solution:
    def search(self, nums: list[int], target: int) -> int:
        l = 0
        r = len(nums) - 1

        while l <= r:
            mid = (l + r) // 2

            if nums[mid] == target:
                return mid
            elif nums[mid] >= nums[r]:  # [2, 3, 4, 5, 1] the left chunk [2, 3] is ascending the other has a pivot
                # if in [5, 1]
                if target > nums[mid] or target < nums[l]: # target <= nums[r]
                    l = mid + 1
                else:
                    r = mid - 1    
            else:  # nums[mid] < nums[r] [5, 1, 2, 3, 4] the right chunk [3, 4] is ascending the other has a pivot
                if target > nums[r] or target < nums[mid]: # target <= nums[r] is wrong because (Solution().search([1, 3, 5], 5), 2)
                    r = mid - 1
                else:
                    l = mid + 1

        return -1





# Climbing Stairs
# https://leetcode.com/problems/climbing-stairs/
"""
You are climbing a staircase. It takes n steps to reach the top.

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
3. 2 steps + 1 step
"""


# Fibonnacci problem
class Solution:
    def climbStairs(self, n: int) -> int:
        a = 1
        b = 1

        for _ in range(n):
            a, b = b, a + b
            # print(a, b)

        return a
(Solution().climbStairs(2), 2)
(Solution().climbStairs(3), 3)
(Solution().climbStairs(4), 5)


def Fib_gen(n):
    a = 1
    b = 1

    for _ in range(n):
        yield a
        a, b = b, a + b

fib5 = Fib_gen(5)

for i in fib5:
    print(i)
next(fib5)





# House Robber
# https://leetcode.com/problems/house-robber/
"""
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

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


# O(n), O(1)
class Solution:
    def rob(self, nums: list[int]):
        house_1 = 0
        house_2 = 0

        for house_3 in nums:
            temp = max(house_1 + house_3, house_2)
            house_1 = house_2
            house_2 = temp

        return house_2
(Solution().rob([1, 2, 3, 1]), 4)
(Solution().rob([2, 7, 9, 3, 1]), 12)
(Solution().rob([2, 100, 9, 3, 100]), 200)


# O(n), O(n)
class Solution:
    def rob(self, nums: list[int]):
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])

        for ind in range(2, len(nums)):
            dp[ind] = max(dp[ind - 2] + nums[ind], dp[ind - 1])

        return dp[-1]





# House Robber II
# https://leetcode.com/problems/house-robber-ii/
"""
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, and it will automatically contact the police if two adjacent houses were broken into on the same night.

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
    def rob(self, nums: list[int]):
        house_1 = 0
        house_2 = 0

        for house_3 in nums:
            temp = max(house_1 + house_3, house_2)
            house_1 = house_2
            house_2 = temp

        return house_2
    
    def rob_list(self, nums: list[int]) -> int:
        return max(self.rob(nums[:-1]), self.rob(nums[1:]), nums[0])(Solution().rob([2, 3, 2]), 3)
(Solution().rob_list([2, 3, 2]), 3)
(Solution().rob_list([1, 2, 3, 1]), 4)
(Solution().rob_list([1, 2, 3]), 3)
(Solution().rob_list([1]), 1)





# Longest Palindromic Substring
# https://leetcode.com/problems/longest-palindromic-substring/
"""
Given a string s, return the longest 
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


# O(n2) # No words so it could be "bbbbb"
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s:
            return ""
        
        longest = s[0]

        # odd length
        for i in range(len(s)):
            edge = 1
            while (i - edge) >= 0 and (i + edge) < len(s) and (s[i - edge] == s[i + edge]):
                if  2*edge + 1 > len(longest):
                    longest = s[i - edge:i + edge + 1]
                edge += 1

        # even length
        for i in range(len(s) - 1):
            edge = 0
            while (i - edge) >= 0 and (i + 1 + edge) < len(s) and (s[i - edge] == s[i + 1 + edge]):
                if  2*edge + 2 > len(longest):
                    longest = s[i - edge:i + 1 + edge + 1]
                edge += 1

        return longest
(Solution().longestPalindrome("babad"), "bab")
(Solution().longestPalindrome("cbbd"), "bb")
(Solution().longestPalindrome("a"), "a")
(Solution().longestPalindrome(""), "")
(Solution().longestPalindrome("bb"), "bb")
(Solution().longestPalindrome("ab"), "a")
(Solution().longestPalindrome("aacabdkacaa"), "aca")
(Solution().longestPalindrome("abdka"), "a")


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





# Palindromic Substrings
# https://leetcode.com/problems/palindromic-substrings/
"""
Given a string s, return the number of palindromic substrings in it.

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
    
    def countSubstrings(self, s):
        len_s = len(s)

        # odd length palindroms
        for ind in range(len_s):
            edge = 0
            while ind - edge >=0 and ind + edge < len_s and s[ind - edge] == s[ind + edge]:
                self.substrings_count += 1
                edge += 1

        # even length palindroms
        for ind in range(len_s - 1):
            edge = 0
            while ind - edge >=0 and ind + edge + 1 < len_s and s[ind - edge] == s[ind + edge + 1]:
                self.substrings_count += 1
                edge += 1

        return self.substrings_count
(Solution().countSubstrings("abc"), 3)
(Solution().countSubstrings("aaa"), 6)


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





# Coin Change
# https://leetcode.com/problems/coin-change/
"""
You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

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
        # Initialize min_coins array with amount + 1 which is an impossibly high number
        # Index is ammount, value is the min number of coions to sum to that value
        min_coins = [amount + 1] * (amount + 1)
        # Base case: 0 amount requires 0 coins
        min_coins[0] = 0

        for curr_ammount in range(1, amount + 1):
            for coin in coins:
                # If the current coin can be used (i.e., doesn't make the amount negative)
                if curr_ammount - coin >= 0:
                    # Update the minimum coins needed for the current amount
                    min_coins[curr_ammount] = min(min_coins[curr_ammount], 1 + min_coins[curr_ammount - coin])
        
        # If the last element is unchanged there is no single money combination to sum up.
        if min_coins[amount] == amount + 1:
            return -1
        else:
            return min_coins[amount]
        # return min_coins[amount] if min_coins[amount] != amount + 1 else -1
(Solution().coinChange([1, 2, 5], 11), 3)
(Solution().coinChange([2], 3), -1)
(Solution().coinChange([1], 0), 0)
(Solution().coinChange([2, 5, 10, 1], 27), 4)
(Solution().coinChange([186, 419, 83, 408], 6249), 20)

# greedy no good for (Solution().coinChange([186, 419, 83, 408], 6249), 20)
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





# Maximum Product Subarray
# https://leetcode.com/problems/maximum-product-subarray/
"""
Given an integer array nums, find a subarray that has the largest product, and return the product.

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
        curMin = 1
        curMax = 1
        res = nums[0]

        for n in nums:
            curMin, curMax = min(n * curMax, n * curMin, n), max(n * curMax, n * curMin, n)
            
            # curMin, _, curMax = sorted((n * curMax, n * curMin, n))
            # tmp = curMax * n
            # curMax = max(n * curMax, n * curMin)
            # curMin = min(tmp, n * curMin)
            res = max(res, curMax)
        return res
(Solution().maxProduct([2, 3, -2, 4]), 6)
(Solution().maxProduct([0, 2]), 2)
(Solution().maxProduct([-2]), -2)
(Solution().maxProduct([-4, -3]), 12)
(Solution().maxProduct([-2, 0, -1]), 0)
(Solution().maxProduct([-2, -3, 7]), 42)
(Solution().maxProduct([2, -5, -2, -4, 3]), 24)





# O(n2)
import numpy as np


class Solution:
    def maxProduct(self, nums: list[int]) -> int:
        max_prod = nums[0]

        for i in range(len(nums)):
            for j in range(i, len(nums)):
                print(i, j)
                max_prod = max(max_prod, np.prod(nums[i:j + 1]))
                # print(nums[i:j + 1])
        return max_prod





# Word Break
# https://leetcode.com/problems/word-break/
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
        can_segment = [False] * len(s)
        can_segment.append(True)
        wordSet = set(wordDict)

        for ind in range(len(s))[::-1]:
            for word in wordSet:
                if word == s[ind: ind + len(word)]:
                    if can_segment[ind + len(word)]:
                        can_segment[ind] = True
                        break

        return can_segment[0]
(Solution().wordBreak("leetcode", ["leet", "code"]), True)
(Solution().wordBreak("cars", ["car", "ca", "rs"]), True)





# Longest Increasing Subsequence
# https://leetcode.com/problems/longest-increasing-subsequence/
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
        lis_lengths = [1] * len(nums)
        
        for i in range(len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    lis_lengths[i] = max(lis_lengths[i], lis_lengths[j] + 1)

        return max(lis_lengths)
(Solution().lengthOfLIS([10, 9, 2, 5, 3, 7, 101, 18]), 4)
(Solution().lengthOfLIS([0, 1, 0, 3, 2, 3]), 4)
(Solution().lengthOfLIS([7, 7, 7, 7, 7, 7, 7]), 1)


# Filling dp list from the end
class Solution:
    def lengthOfLIS(self, nums: list[int]) -> int:
        dp = [1] * len(nums)

        for i in reversed(range(len(nums))):
            for j in range(i + 1, len(nums)):
                if nums[i] < nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)





# Partition Equal Subset Sum
# https://leetcode.com/problems/partition-equal-subset-sum
"""
Given an integer array nums, return true if you can partition the array into two subsets such that the sum of the elements in both subsets is equal or false otherwise.

Example 1:

Input: nums = [1,5,11,5]
Output: true
Explanation: The array can be partitioned as [1, 5, 5] and [11].

Example 2:

Input: nums = [1,2,3,5]
Output: false
Explanation: The array cannot be partitioned into equal sum subsets.
"""


class Solution:
    def canPartition(self, nums: list[int]) -> bool:
        if sum(nums) % 2:
            return False
        
        target = sum(nums) // 2
        possible_sums = {0}

        for num in nums:
            if target in possible_sums:
                return True
            
            seen_chunk = set()
            for s in possible_sums:
                seen_chunk.add(s + num)
            possible_sums.update(seen_chunk)
            
            # update possible_sums in one line
            # possible_sums.update({s + num for s in possible_sums})
            
        return False
(Solution().canPartition([1, 5, 11, 5]), True)
(Solution().canPartition([3, 3, 3, 4, 5]), True)
(Solution().canPartition([1, 2, 5]), False)
(Solution().canPartition([1, 2, 3, 5]), False)
(Solution().canPartition([1]), False)





# Unique Paths
# https://leetcode.com/problems/unique-paths/
"""
There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.

Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.

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


# O(n * m) O(n)
class Solution():
    def uniquePaths(self, m: int, n: int) -> int:
        bottom_row = [1] * n

        for _ in range(m - 1):
            curr_row = [1] * n
        
            for i in range(n - 1)[::-1]:
                curr_row[i] = curr_row[i + 1] + bottom_row[i]

            bottom_row = curr_row

        return bottom_row[0]
(Solution().uniquePaths(3, 7), 28)
(Solution().uniquePaths(3, 2), 3)
(Solution().uniquePaths(1, 2), 1)






# Longest Common Subsequence
# https://leetcode.com/problems/longest-common-subsequence/
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


# O(n2), O(n2), dp
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # dp = [[0] * (len(text1) + 1)] * (len(text2) + 1)  creates a list of lists where each sublist is a reference to the same list. This means that updating one element in any sublist will affect all sublists.
        dp = [[0] * (len(text1) + 1) for _ in range(len(text2) + 1)]

        for i in range(len(text2))[::-1]:
            for j in range(len(text1))[::-1]:
                if text2[i] == text1[j]:
                    dp[i][j] = dp[i + 1][j + 1] + 1
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

        return dp[0][0]
(Solution().longestCommonSubsequence("abcde", "ace"), 3)
(Solution().longestCommonSubsequence("abc", "abc"), 3)
(Solution().longestCommonSubsequence("abc", "dew"), 0)
(Solution().longestCommonSubsequence("bsbininm", "jmjkbkjkv"), 1)


#  O(n2), O(n), dp
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        bottom_row = [0] * (len(text1) + 1) 

        for i in range(len(text2))[::-1]:
            curr_row = [0] * (len(text1) + 1) 
            
            for j in range(len(text1))[::-1]:
                if text2[i] == text1[j]:
                    curr_row[j] = bottom_row[j + 1] + 1
                else:
                    curr_row[j] = max(bottom_row[j], curr_row[j + 1])
            
            bottom_row = curr_row

        return bottom_row[0]


# Top-down with cache; O(n2), O(n2), recursion function inside longestCommonSubsequence, but much slower than dp
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        cache = {}

        def rec(i, j):
            if i == len(text2) or j == len(text1):
                return 0
 
            if (i, j) in cache:
                return cache[(i, j)]

            if text2[i] == text1[j]:
                curr_longest = rec(i + 1, j + 1) + 1
            else:
                curr_longest = max(rec(i + 1, j), rec(i, j + 1))
            
            cache[(i, j)] = curr_longest
            
            return curr_longest

        return rec(0, 0)


# Top-down with cache; O(n2), O(n2), recursion function outside longestCommonSubsequence, but much slower than dp
class Solution:
    def __init__(self):
        self.text1 = ''
        self.text2 = ''
        self.cache = {}

    def rec(self, i, j):
        if i == len(self.text2) or j == len(self.text1):
            return 0
        
        if (i, j) in self.cache:
            return self.cache[(i, j)]
        
        if self.text2[i] == self.text1[j]:
            curr_longest = self.rec(i + 1, j + 1) + 1
        else:
            curr_longest = max(self.rec(i + 1, j), self.rec(i, j + 1))
        
        self.cache[(i, j)] = curr_longest
        return curr_longest

    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        self.text1 = text1
        self.text2 = text2
        lcs_length = self.rec(0, 0)
        return lcs_length





# Maximum Subarray
# https://leetcode.com/problems/maximum-subarray/
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
        max_sum = nums[0]
        curr_sum = nums[0]

        for num in nums[1:]:
            curr_sum = max(curr_sum + num, num)
            max_sum = max(max_sum, curr_sum)
    
        return max_sum
(Solution().maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4]), 6)
(Solution().maxSubArray([1]), 1)
(Solution().maxSubArray([5, 4, -1, 7, 8]), 23)
(Solution().maxSubArray([-4, -2, -1, -3]), -1)


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




 
# Binary Search
# https://leetcode.com/problems/binary-search/
"""
Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

You must write an algorithm with O(log n) runtime complexity.

Example 1:

Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4

Example 2:

Input: nums = [-1,0,3,5,9,12], target = 2
Output: -1
Explanation: 2 does not exist in nums so return -1
"""


class Solution:
    def search(self, nums: list[int], target: int):
        left = 0
        right = len(nums) - 1

        while left <= right:  # two poionters
            mid = (left + right) // 2  # find mid index
            
            if target == nums[mid]:  # if target found
                return mid
            elif target < nums[mid]:  # if target is less than middle, choose left chunk
                right = mid - 1
            else:  # if target is greater than middle, choose rigth chunk
                left = mid + 1
        
        return -1
(Solution().search([-1, 0, 3, 5, 9, 12], -1), 0)
(Solution().search([-1, 0, 3, 5, 9, 12], 0), 1)
(Solution().search([-1, 0, 3, 5, 9, 12], 3), 2)
(Solution().search([-1, 0, 3, 5, 9, 12], 5), 3)
(Solution().search([-1, 0, 3, 5, 9, 12], 9), 4)
(Solution().search([-1, 0, 3, 5, 9, 12], 12), 5)
(Solution().search([-1, 0, 3, 5, 9, 12], 2), -1)





# Search a 2D Matrix
# https://leetcode.com/problems/search-a-2d-matrix/
"""
You are given an m x n integer matrix matrix with the following two properties:

Each row is sorted in non-decreasing order.
The first integer of each row is greater than the last integer of the previous row.
Given an integer target, return true if target is in matrix or false otherwise.

You must write a solution in O(log(m * n)) time complexity.

Example 1:

Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
Output: true

Example 2:


Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
Output: false
"""


class Solution:
    def searchMatrix(self, matrix, target):
        top = 0  # boundries
        bottom = len(matrix) - 1
        left = 0
        right = len(matrix[0]) - 1

        while top <= bottom:  # two poionters to find the right row
            mid_row = (top + bottom) // 2  # find middle row index

            if (target >= matrix[mid_row][left] and
                    target <= matrix[mid_row][right]):  # if target row found
                break
            elif target < matrix[mid_row][left]:  # if target is less than the most left, choose top chunk
                bottom = mid_row - 1
            else:  # if target is grater than the most right, choose bottom chunk
                top = mid_row + 1

        while left <= right:  # two poionters to find the right column
            mid_col = (left + right) // 2  # find middle column index

            if target == matrix[mid_row][mid_col]:  # if target column found
                return True
            elif target < matrix[mid_row][mid_col]:  # if target is less than middle colum, choose left chunk
                right = mid_col - 1
            else:  # if target is greater than middle colum, choose rigth chunk
                left = mid_col + 1

        return False
(Solution().searchMatrix([[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 3), True)
(Solution().searchMatrix([[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 13), False)





# Koko Eating Bananas
# https://leetcode.com/problems/koko-eating-bananas/
"""
Koko loves to eat bananas. There are n piles of bananas, the ith pile has piles[i] bananas. The guards have gone and will come back in h hours.

Koko can decide her bananas-per-hour eating speed of k. Each hour, she chooses some pile of bananas and eats k bananas from that pile. If the pile has less than k bananas, she eats all of them instead and will not eat any more bananas during this hour.

Koko likes to eat slowly but still wants to finish eating all the bananas before the guards return.

Return the minimum integer k such that she can eat all the bananas within h hours.

Example 1:

Input: piles = [3,6,7,11], h = 8
Output: 4

Example 2:

Input: piles = [30,11,23,4,20], h = 5
Output: 30

Example 3:

Input: piles = [30,11,23,4,20], h = 6
Output: 23
"""



# Binary search
import numpy as np

class Solution:
    def minEatingSpeed(self, piles: list[int], hours: int) -> int:
        left = 1
        right = max(piles)

        while left <= right:
            mid = (left + right) // 2

            # time to guard to come back < time to eat all piles
            if hours < sum(np.ceil(pile / mid) for pile in piles):  # not enough time to eat all bananas need to increase mid
            # if hours < sum(((pile - 1) // mid) + 1 for pile in piles):  # way to celi wihout numpy
                left = mid + 1
            else:  # enought time to eat all bananas, this might be a solution but there might be a better one
                solution = mid
                right = mid - 1
        
        return solution
(Solution().minEatingSpeed([3, 6, 7, 11], 8), 4)
(Solution().minEatingSpeed([30, 11, 23, 4, 20], 5), 30)
(Solution().minEatingSpeed([30, 11, 23, 4, 20], 6), 23)
(Solution().minEatingSpeed([312884470], 312884469), 2)
(Solution().minEatingSpeed([3], 2), 2)


# Brute force
# Time Limit Exceeded
class Solution:
    def minEatingSpeed(self, piles: list[int], h: int) -> int:
        k = 1

        while sum(number // k if not number % k else (number // k) + 1 for number in piles) > h:
            # if sum(((number - 1) // k) + 1 for number in piles) > h:
            k += 1
        
        return k





# Valid Sudoku
# https://leetcode.com/problems/valid-sudoku/
"""
Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

Each row must contain the digits 1-9 without repetition.
Each column must contain the digits 1-9 without repetition.
Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.
Note:

A Sudoku board (partially filled) could be valid but is not necessarily solvable.
Only the filled cells need to be validated according to the mentioned rules.
 

Example 1:

Input: board = 
[["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: true

Example 2:

Input: board = 
[["8","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: false
Explanation: Same as Example 1, except with the 5 in the top left corner being modified to 8. Since there are two 8's in the top left 3x3 sub-box, it is invalid.
"""


# using set
class Solution:
    # validating sub-box
    def is_subbox_valid(self, row, col):
        seen = set()

        for r in range(3):
            for c in range(3):
                digit = self.board[r + row][c + col]

                if digit != ".":
                    if digit in seen:
                        return False
                    else:
                        seen.add(digit)
        return True

    def isValidSudoku(self, board):
        self.board = board

        rows = len(board)
        cols = len(board[0])

        # validating rows
        for row in board:
            seen = set()

            for digit in row:
                if digit != ".":
                    if digit in seen:
                        return False
                    else:
                        seen.add(digit)

        # validating columns
        for col in range(cols):
            seen = set()

            for row in range(rows):
                digit = board[row][col]

                if digit != ".":
                    if digit in seen:
                        return False
                    else:
                        seen.add(digit)

        # validating sub-boxes
        for row in range(0, 3*3, 3):
            for col in range(0, 3*3, 3):
                if not self.is_subbox_valid(row, col):
                    return False

        return True
(Solution().isValidSudoku([["5", "3", ".", ".", "7", ".", ".", ".", "."], ["6", ".", ".", "1", "9", "5", ".", ".", "."], [".", "9", "8", ".", ".", ".", ".", "6", "."], ["8", ".", ".", ".", "6", ".", ".", ".", "3"], ["4", ".", ".", "8", ".", "3", ".", ".", "1"], ["7", ".", ".", ".", "2", ".", ".", ".", "6"], [".", "6", ".", ".", ".", ".", "2", "8", "."], [".", ".", ".", "4", "1", "9", ".", ".", "5"], [".", ".", ".", ".", "8", ".", ".", "7", "9"]], ), True)
(Solution().isValidSudoku([["8", "3", ".", ".", "7", ".", ".", ".", "."], ["6", ".", ".", "1", "9", "5", ".", ".", "."], [".", "9", "8", ".", ".", ".", ".", "6", "."], ["8", ".", ".", ".", "6", ".", ".", ".", "3"], ["4", ".", ".", "8", ".", "3", ".", ".", "1"], ["7", ".", ".", ".", "2", ".", ".", ".", "6"], [".", "6", ".", ".", ".", ".", "2", "8", "."], [".", ".", ".", "4", "1", "9", ".", ".", "5"], [".", ".", ".", ".", "8", ".", ".", "7", "9"]]), False)
(Solution().isValidSudoku([[".", ".", ".", ".", "5", ".", ".", "1", "."], [".", "4", ".", "3", ".", ".", ".", ".", "."], [".", ".", ".", ".", ".", "3", ".", ".", "1"],  ["8", ".", ".", ".", ".", ".", ".", "2", "."],  [".", ".", "2", ".", "7", ".", ".", ".", "."],  [".", "1", "5", ".", ".", ".", ".", ".", "."],  [".", ".", ".", ".", ".", "2", ".", ".", "."],  [".", "2", ".", "9", ".", ".", ".", ".", "."],  [".", ".", "4", ".", ".", ".", ".", ".", "."]]), False)

# using defaultdict(set) form collections
from collections import defaultdict
class Solution:
    def isValidSudoku(self, board: list[list[str]]) -> bool:
        rows = len(board)
        cols = len(board[0])

        row_uniq = defaultdict(set)
        col_uniq = defaultdict(set)
        box_uniq = defaultdict(set)

        for row in range(rows):
            for col in range(cols):
                element = board[row][col]

                if element != ".":
                    if (element in row_uniq[row]
                        or element in col_uniq[col]
                        or element in box_uniq[(row//3, col//3)]
                    ):
                        return False
                    else:
                        row_uniq[row].add(element)
                        col_uniq[col].add(element)
                        box_uniq[((row//3, col//3))].add(element)

        return True





# Permutation in String
# https://leetcode.com/problems/permutation-in-string/
"""
Given two strings s1 and s2, return true if s2 contains a permutation of s1, or false otherwise.

In other words, return true if one of s1's permutations is the substring of s2.

 

Example 1:

Input: s1 = "ab", s2 = "eidbaooo"
Output: true
Explanation: s2 contains one permutation of s1 ("ba").

Example 2:

Input: s1 = "ab", s2 = "eidboaoo"
Output: false
 

Constraints:

1 <= s1.length, s2.length <= 104
s1 and s2 consist of lowercase English letters.
"""

 
# compare only letters that are at current pointer, previous matches are storred in the 'counter'. Suprisingly timing it the same when comparing the whole dictionary.
class Solution:
    def checkInclusion(self, s1, s2):
        s1_count = {}  # dict from s1
        window = {}  # window as dict
        counter = 0  # counts how many keys (letters) in 'window' have exactly match in 's1_count'
        left = 0  # left pointer
        
        for letter in s1:  # dict from s1
            s1_count[letter] = s1_count.get(letter, 0) + 1
        
        s1_count_len = len(s1_count)

        for right, letter in enumerate(s2):  # right pointer
            window[letter] = window.get(letter, 0) + 1  # add a letter to the window
            
            if (letter in s1_count and  # if letter is significant and
                s1_count[letter] == window[letter]):  # if letter occurences match
                counter += 1
                if counter == s1_count_len:  # if cunter is equal to s1_count that means all letter occurences are matching
                    return True

            if len(s1) > right - left + 1:  # if window is not long enough
                continue

            # if the letter at left pointer that's going to be removed is significant then need to lower the 'counter'
            if (s2[left] in s1_count and # if letter is significant and
                s1_count[s2[left]] == window[s2[left]]): # if letter occurences match
                counter -= 1
            
            window[s2[left]] -= 1  # remove a letter at the left pointer from the window
            left += 1

        return False
(Solution().checkInclusion("ab", "eidbaooo"), True)
(Solution().checkInclusion("ab", "eidboaoo"), False)
(Solution().checkInclusion("ccc", "cbac"), False)
(Solution().checkInclusion("ab", "a"), False)
(Solution().checkInclusion("abcdxabcde", "abcdeabcdx"), True)
(Solution().checkInclusion("adc", "dcda"), True)
(Solution().checkInclusion("hello", "ooolleoooleh"), False)
(Solution().checkInclusion("mart", "karma"), False)
(Solution().checkInclusion("abc", "ccccbbbbaaaa"), False)


# compare whole dicts when checking for inclusion
class Solution:
    def checkInclusion(self, s1, s2):
        s1_count = {}  # dict from s1
        window = {}  # window as dict
        left = 0  # left pointer
        
        for letter in s1:  # dict from s1
            s1_count[letter] = s1_count.get(letter, 0) + 1
        
        for right, letter in enumerate(s2):  # right pointer
            window[letter] = window.get(letter, 0) + 1  # add a letter to the window
            
            if len(s1) > right - left + 1:  # if window is not long enough
                continue

            if s1_count == window:  # comapre dicts
                return True

            window[s2[left]] -= 1

            if not window[s2[left]]:  # if a key with no value exists
                del window[s2[left]]  # remove that key
            left += 1

        return False





# Minimum Window Substring
# https://leetcode.com/problems/minimum-window-substring/
"""
Given two strings s and t of lengths m and n respectively, return the minimum window 
substring
 of s such that every character in t (including duplicates) is included in the window. If there is no such substring, return the empty string "".

The testcases will be generated such that the answer is unique.

Example 1:

Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.

Example 2:

Input: s = "a", t = "a"
Output: "a"
Explanation: The entire string s is the minimum window.

Example 3:

Input: s = "a", t = "aa"
Output: ""
Explanation: Both 'a's from t must be included in the window.
Since the largest window of s only has one 'a', return empty string.
"""


# overcomplicated but fast algorythm
class Solution:
    def minWindow(self, full, pattern):
        if len(pattern) > len(full):  # when pattern in longer than searched string
            return ""

        window = {}  # window as dictionary
        pattern_count = {}  # pattern dictionary
        left = 0  # left pointer
        # counts how many keys (letters) in 'window' have exactly match in searched string
        counter = 0
        min_window = full + "Τ"  # holds solution string

        for letter in pattern:  # pattern dictionary
            pattern_count[letter] = pattern_count.get(letter, 0) + 1

        for right, letter in enumerate(full):  # right pointer
            # add letter to the window
            window[letter] = window.get(letter, 0) + 1

            if (letter in pattern_count and  # if letter is significant
                    window[letter] == pattern_count[letter]):  # if letter occurences match
                counter += 1

            if (right + 1 < len(full) and  # if right is in bounds
                    counter != len(pattern_count)):  # and not all letter occurences are matching
                continue  # go to add anoter letter

            if counter == len(pattern_count):  # if all letter occurences are matching
                # even when all letter occurences are matching need to discard unnesesery letters from the beginning of the match
                while ((not full[left] in pattern_count) or  # if there is not significant letter at the begining or
                       (window[full[left]] > pattern_count[full[left]])):  # there is significant letteres but 'window' have enough copies of it
                    window[full[left]] -= 1
                    left += 1

                if (right - left + 1) < len(min_window):  # if shorter window found
                    min_window = full[left: right + 1]  # update min window
                    if right - left + 1 == len(pattern):  # if the length of the min window is equal to the length of the pattern, that meant the shortest window has been fonud
                        return min_window

                # if window[full[left]] == pattern_count[full[left]]:  # the first letter is always significant so no need to check
                counter -= 1  # lower the counter
                window[full[left]] -= 1  # discard significant letter
                left += 1  # update the left counter

                while (left < len(full) - 1 and  # if left in bounds and
                       not full[left] in pattern_count):  # letter is not significant
                    window[full[left]] -= 1  # discard it
                    left += 1  # update the left counter

        return min_window if min_window != (full + "Τ") else ""
(Solution().minWindow("ADOBECODEBANC", "ABC"), "BANC")
(Solution().minWindow("a", "a"), "a")
(Solution().minWindow("a", "aa"), "")
(Solution().minWindow("a", "b"), "")
(Solution().minWindow("ab", "b"), "b")
(Solution().minWindow("bba", "ab"), "ba")
(Solution().minWindow("abc", "a"), "a")
(Solution().minWindow("jwsdrkqzrthzysvqqazpfnulprulwmwhiqlbcdduxktxepnurpmxegktzegxscfbusexvhruumrydhvvyjucpeyrkeodjvgdnkalutfoizoliliguckmikdtpryanedcqgpkzxahwrvgcdoxiylqjolahjawpfbilqunnvwlmtrqxfphgaxroltyuvumavuomodblbnzvequmfbuganwliwidrudoqtcgyeuattxxlrlruhzuwxieuqhnmndciikhoehhaeglxuerbfnafvebnmozbtdjdo", "qruzywfhkcbovewle"), "vequmfbuganwliwidrudoqtcgyeuattxxlrlruhzuwxieuqhnmndciik")


# simplified but slower
class Solution:
    def minWindow(self, outer_string, substring):
        if len(substring) > len(outer_string):  # when substring in longer than searched string
            return ""

        window = {}  # window as dictionary
        pattern_count = {}  # substring dictionary
        left = 0  # left pointer
        counter = 0  # counts how many keys (letters) in 'window' have exactly match in searched string
        min_window = outer_string + "Τ"  # holds solution string

        for letter in substring:  # substring dictionary
            pattern_count[letter] = pattern_count.get(letter, 0) + 1

        for right, letter in enumerate(outer_string):  # right pointer
            if letter in pattern_count:  # if letter is significant
                window[letter] = window.get(letter, 0) + 1  # add letter to the window
                if window[letter] == pattern_count[letter]:  # if letter occurences match
                    counter += 1

            while counter == len(pattern_count):  # if all letter occurences are matching
                if (right - left + 1) < len(min_window):  # if shorter window found
                    min_window = outer_string[left: right + 1]  # update min window
                    if (right - left + 1) == len(substring):  # if the length of min_window is as short as it can be => equal to the length of the substring
                        return min_window  # fast exit
                
                left_letter = outer_string[left]  # letter at left poionter

                if left_letter in pattern_count:  # if letter is significant
                    if window[left_letter] == pattern_count[left_letter]:  # if left_letter occurences match
                        counter -= 1  # lower the counter
                    
                    window[left_letter] -= 1  # lower significant letter occurence
                
                left += 1  # update the left counter

        return min_window if min_window != (outer_string + "Τ") else ""


# using have, need for comparison
class Solution:
    def counter(self, word):
        d = dict()

        for letter in word:
            d[letter] = d.get(letter, 0) + 1

        return d
    
    def minWindow(self, s: str, pattern: str) -> str:
        if len(pattern) > len(s):
            return ""
        
        l = 0
        r = 0
        pattern_count = self.counter(pattern)
        window_count = {}
        min_window = s + "Τ"
        have = 0
        # number of elements in pattern, count even same ones
        need = len(pattern)

        while r < len(s):
            # update curr window dict if element is significant
            if s[r] in pattern_count:
                # if pattern_count[s[r]] > window_count[s[r]]:
                if window_count.get(s[r], 0) < pattern_count[s[r]]:
                    have += 1
                window_count[s[r]] = window_count.get(s[r], 0) + 1

            # while window and pattern are equal
            # Case where need to trim window from left (Solution().minWindow("bba", "ab"), "ba") is "bba"
            while need == have:
                # update minimum window
                min_window = min(min_window, s[l: r + 1], key=len)
        
                # update current window dict
                if s[l] in window_count:
                    window_count[s[l]] -= 1

                    # removed element affect 'have'.
                    if window_count[s[l]] < pattern_count[s[l]]:
                        have -= 1

                l += 1

            r += 1

        return "" if min_window == s + "Τ" else min_window


# compare dictionaries directly
class Solution:
    def counter(self, word):
        d = dict()

        for letter in word:
            d[letter] = d.get(letter, 0) + 1

        return d
    
    # if window valid subdict of pattern, check only key from pattern
    def is_sub(self, window, pattern):
        for key in pattern:
            if not key in window or window[key] < pattern[key]:
                return False

        return True

    def minWindow(self, s: str, pattern: str) -> str:
        if len(pattern) > len(s):
            return ""
        
        l = 0
        r = 0
        pattern_count = self.counter(pattern)
        window_count = {}
        min_window = s + "Τ"

        while r < len(s):
            # update curr window dict if element is significant
            if s[r] in pattern_count.keys():
                window_count[s[r]] = window_count.get(s[r], 0) + 1

            # while all elements from window are found in pattern
            # Case where need to trim window from left (Solution().minWindow("bba", "ab"), "ba") is "bba"
            while self.is_sub(window_count, pattern_count):
                # update minimum window
                min_window = min(min_window, s[l: r + 1], key=len)
        
                # if l is a significant index
                if s[l] in pattern_count:
                    # update current window dict
                    window_count[s[l]] -= 1

                l += 1

            r += 1

        return "" if min_window == s + "Τ" else min_window

 
# O(n2)
class Solution:
    def counter(self, word):
        d = dict()

        for letter in word:
            d[letter] = d.get(letter, 0) + 1

        return d

    def dict_compare(self, s_dict, t_dict):
        for key in t_dict:
            if not key in s_dict or s_dict[key] < t_dict[key]:
                return False

        return True

    def minWindow(self, s: str, t: str) -> str:
        if len(t) > len(s):
            return ""
        
        t_dict = self.counter(t)
        min_wind = s + "Τ"

        for l in range(len(s)):
            for r in range(len(s)):
                s_dict = self.counter(s[l:r + 1])

                if self.dict_compare(s_dict, t_dict):
                    min_wind = min(min_wind, s[l: r + 1], key=len)

        return "" if min_wind == s + "Τ" else min_wind





# Sliding Window Maximum
# https://leetcode.com/problems/sliding-window-maximum/
"""
You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position.

Return the max sliding window.

Example 1:

Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
Explanation: 
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7

Example 2:

Input: nums = [1], k = 1
Output: [1]
"""


# Monotonic Decreasing Queue

# window as a list of indexes
class Solution:
    def maxSlidingWindow(self, nums, window_size):
        window = []  # slidiong window as a list
        left = 0  # left pointer
        current_max = []  # array with max value from each sliding window

        for right in range(len(nums)):  # right pointer
            num = nums[right]

            while window and window[0] < left:  # remove left out of bounds indexes
                window.pop(0)

            while window and nums[window[-1]] <= num:  # remove right indexes with nums less than current num
                    window.pop()

            if window and num > nums[window[0]]:  # if num is greater than the left most num
                window.insert(0, right) # append it left
            else:  
                window.append(right)  # append it right

            if right - left + 1 == window_size:  # if the window is the right size
                current_max.append(nums[window[0]])  # get the left (max) value
                left += 1  # update left pointer

        return current_max
(Solution().maxSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3), [3, 3, 5, 5, 6, 7])
(Solution().maxSlidingWindow([1], 1), [1])
(Solution().maxSlidingWindow([7, 2, 4], 2), [7, 4])
(Solution().maxSlidingWindow([1, 3, 1, 2, 0, 5], 3), [3, 3, 2, 5])


# sliding window as deque of indexes
from collections import deque
class Solution:
    def maxSlidingWindow(self, nums, window_size):
        window = deque()  # slidiong window as a deque
        left = 0  # left pointer
        current_max = []  # array with max value from each sliding window

        for right in range(len(nums)):  # right pointer
            num = nums[right]  # current num

            while window and window[0] < left:  # remove left out of bounds indexes
                window.popleft()

            while window and nums[window[-1]] <= num:  # remove right indexes with nums less than current num
                    window.pop()

            if window and num > nums[window[0]]:  # if num is greater than the left most num append it left
                window.appendleft(right)
            else:  # append it right
                window.append(right)

            if right - left + 1 == window_size:  # if the window is the right size
                current_max.append(nums[window[0]])  # get the left (max) value
                left += 1  # update left pointer
        return current_max


# O(n2)
class Solution:
    def maxSlidingWindow(self, nums: list[int], k: int) -> list[int]:
        l = 0
        sol = []

        for r in range(k - 1, len(nums)):
            sol.append(max(nums[l : r + 1]))

            l += 1
        return sol





# Min Stack
# https://leetcode.com/problems/min-stack/
"""
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:

MinStack() initializes the stack object.
void push(int val) pushes the element val onto the stack.
void pop() removes the element on the top of the stack.
int top() gets the top element of the stack.
int getMin() retrieves the minimum element in the stack.
You must implement a solution with O(1) time complexity for each function.

Example 1:

Input
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

Output
[null,null,null,null,-3,null,0,-2]

Explanation
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop();
minStack.top();    // return 0
minStack.getMin(); // return -2
"""


class MinStack:
    def __init__(self):
        self.stack = []
        self.stack_min = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        
        if self.stack_min:
            val = min(val, self.stack_min[-1])
        
        self.stack_min.append(val)

    def pop(self) -> None:
        self.stack.pop()
        self.stack_min.pop()
        
    def top(self) -> int:
        return self.stack[-1]
        
    def getMin(self) -> int:
        return self.stack_min[-1]


# Explanation
minStack = MinStack()
minStack.push(-2)
minStack.push(0)
minStack.push(-3)
minStack.getMin() #  return -3
minStack.pop()
minStack.top() #  return 0
minStack.getMin() #  return -2





# Evaluate Reverse Polish Notation
# https://leetcode.com/problems/evaluate-reverse-polish-notation/
"""
You are given an array of strings tokens that represents an arithmetic expression in a Reverse Polish Notation.

Evaluate the expression. Return an integer that represents the value of the expression.

Note that:

The valid operators are '+', '-', '*', and '/'.
Each operand may be an integer or another expression.
The division between two integers always truncates toward zero.
There will not be any division by zero.
The input represents a valid arithmetic expression in a reverse polish notation.
The answer and all the intermediate calculations can be represented in a 32-bit integer.

Example 1:

Input: tokens = ["2","1","+","3","*"]
Output: 9
Explanation: ((2 + 1) * 3) = 9
Example 2:

Input: tokens = ["4","13","5","/","+"]
Output: 6
Explanation: (4 + (13 / 5)) = 6
Example 3:

Input: tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
Output: 22
Explanation: ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22
"""


class Solution:
    def evalRPN(self, tokens: list[str]) -> int:
        stack = []

        for token in tokens:
            if token == "+":
                stack.append(stack.pop() + stack.pop())
            elif token == "*":
                stack.append(stack.pop() * stack.pop())
            elif token == "-":
                b = stack.pop()
                a = stack.pop()
                stack.append(a - b)
            elif token == "/":
                b = stack.pop()
                a = stack.pop()
                stack.append(int(a / b))
                print(stack)
            else:
                stack.append(int(token))

        return stack[0]
(Solution().evalRPN(["2", "1", "+", "3", "*"]), 9)
(Solution().evalRPN(["4", "13", "5", "/", "+"]), 6)
(Solution().evalRPN(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]), 22)
(Solution().evalRPN(["18"]), 18)





# Daily Temperatures
# https://leetcode.com/problems/daily-temperatures/
"""
Given an array of integers temperatures represents the daily temperatures, return an array answer such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature. If there is no future day for which this is possible, keep answer[i] == 0 instead.

Example 1:

Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]

Example 2:

Input: temperatures = [30,40,50,60]
Output: [1,1,1,0]

Example 3:

Input: temperatures = [30,60,90]
Output: [1,1,0]
"""


# O(n)
class Solution:
    def dailyTemperatures(self, temps):
        days_delta = [0 for _ in range(len(temps))]
        stack = []

        for index, temp in enumerate(temps):
            while stack and stack[-1][1] < temp:  # check for lower temps
                days_delta[stack[-1][0]] = index - stack[-1][0]  # update delta_days
                stack.pop()  # pop lower temp
            
            stack.append((index, temp))  # apped new temp

        return days_delta
(Solution().dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73]), [1, 1, 4, 2, 1, 1, 0, 0])
(Solution().dailyTemperatures([30, 40, 50, 60]), [1, 1, 1, 0])
(Solution().dailyTemperatures([30, 60, 90]), [1, 1, 0])


# O(n2)
class Solution:
    def dailyTemperatures(self, temps):
        days_delta = [0 for _ in range(len(temps))]

        for ind_l in range(len(temps) - 1):
            for ind_p in range(ind_l + 1, len(temps)):
                if temps[ind_l] < temps[ind_p]:
                    days_delta[ind_l] = ind_p - ind_l            
                    break

        return days_delta





# Car Fleet
# https://leetcode.com/problems/car-fleet/
"""
There are n cars at given miles away from the starting mile 0, traveling to reach the mile target.

You are given two integer array position and speed, both of length n, where position[i] is the starting mile of the ith car and speed[i] is the speed of the ith car in miles per hour.

A car cannot pass another car, but it can catch up and then travel next to it at the speed of the slower car.

A car fleet is a car or cars driving next to each other. The speed of the car fleet is the minimum speed of any car in the fleet.

If a car catches up to a car fleet at the mile target, it will still be considered as part of the car fleet.

Return the number of car fleets that will arrive at the destination.

Example 1:

Input: target = 12, position = [10,8,0,5,3], speed = [2,4,1,1,3]

Output: 3

Explanation:

The cars starting at 10 (speed 2) and 8 (speed 4) become a fleet, meeting each other at 12. The fleet forms at target.
The car starting at 0 (speed 1) does not catch up to any other car, so it is a fleet by itself.
The cars starting at 5 (speed 1) and 3 (speed 3) become a fleet, meeting each other at 6. The fleet moves at speed 1 until it reaches target.

Example 2:

Input: target = 10, position = [3], speed = [3]

Output: 1

Explanation:

There is only one car, hence there is only one fleet.

Example 3:

Input: target = 100, position = [0,2,4], speed = [4,2,1]

Output: 1

Explanation:

The cars starting at 0 (speed 4) and 2 (speed 2) become a fleet, meeting each other at 4. The car starting at 4 (speed 1) travels to 5.
Then, the fleet at 4 (speed 2) and the car at position 5 (speed 1) become one fleet, meeting each other at 6. The fleet moves at speed 1 until it reaches target.
"""


# (12 - 10) / 2 = 1;
# (12 - 8) / 4 = 1;

# (12 - 8) / 1 = 4;
# (12 - 3) / 3 = 3;

# (12 - 0) / 1 = 12;


class Solution:
    def carFleet(self, target, positions, speeds):
        cars = list(zip(positions, speeds))
        cars.sort(reverse=True)  # sort the cars so to start with the one closest to the target
        fleet_stack = []

        for position, speed in cars:
            curr_dist = (target - position) / speed  # distance to the target
            
            if fleet_stack and fleet_stack[-1] >= curr_dist:  # if the car behind cought up next car
                continue

            fleet_stack.append(curr_dist)  # append a car to a stack

        return len(fleet_stack)
(Solution().carFleet(12, [10, 8, 0, 5, 3], [2, 4, 1, 1, 3]), 3)
(Solution().carFleet(10, [3], [3]), 1)
(Solution().carFleet(100, [0, 2, 4], [4, 2, 1]), 1)
(Solution().carFleet(10, [0, 4, 2], [2, 1, 3]), 1)





# Subsets
# https://leetcode.com/problems/subsets/
"""
Given an integer array nums of unique elements, return all possible 
subsets
 (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.

Example 1:

Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
Example 2:

Input: nums = [0]
Output: [[],[0]]
"""
(Solution().subsets([1, 2, 3]), [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]])
(Solution().subsets([0]), [[], [0]])
 

# "dfs" method inside "subsets" method
class Solution:
    def subsets(self, nums):
        subset = []  # current subcet
        subset_list = []  # solution

        def dfs(level):
            if level == len(nums):  # target level reached
                subset_list.append(subset.copy())  # push subset to subset_list
                return

            subset.append(nums[level])
            dfs(level + 1)  # (left) decision to append current num
            subset.pop()
            dfs(level + 1)  # (right) decision to not append current num

        dfs(0)  # start dfs with level = 0
    
        return subset_list


# "dfs" and "subset" methods inside a class
class Solution:
    def __init__(self) -> None:
        self.subset = []
        self.subset_list = []

    def subsets(self, nums: list[int]) -> list[list[int]]:
        self.nums = nums
        return self.dfs(0)

    def dfs(self, level):
        if level == len(self.nums):
            self.subset_list.append(self.subset.copy())
            return
        
        self.subset.append(self.nums[level])
        self.dfs(level + 1)            
        self.subset.pop()
        self.dfs(level + 1)
        
        return self.subset_list





# Subsets II
# https://leetcode.com/problems/subsets-ii/
"""
Given an integer array nums that may contain duplicates, return all possible 
subsets
 (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.

Example 1:

Input: nums = [1,2,2]
Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]

Example 2:

Input: nums = [0]
Output: [[],[0]]
"""
(Solution().subsetsWithDup([1, 2, 2]), [[], [1], [1, 2], [1, 2, 2], [2], [2, 2]])
(Solution().subsetsWithDup([0]), [[], [0]])
(Solution().subsetsWithDup([4, 4, 4, 1, 4]), [[], [1], [1, 4], [1, 4, 4], [1, 4, 4, 4], [1, 4, 4, 4, 4], [4], [4, 4], [4, 4, 4], [4, 4, 4, 4]])

class Solution:
    def subsetsWithDup(self, nums: list[int]) -> list[list[int]]:
        nums.sort()
        subset = []  # current subcet
        subset_list = []  # solution

        def dfs(index):
            if index == len(nums):  # target level reached
                subset_list.append(subset.copy())  # push subset to subset_list
                return

            subset.append(nums[index])
            dfs(index + 1)  # (left) decision to append current num
            subset.pop()

            # If num at the current index (that was poped previously) is the same as
            # the num at next index skip it.
            while index + 1 < len(nums) and nums[index + 1] == nums[index]:
                index += 1

            dfs(index + 1)  # (right) decision to not append current num

        dfs(0)  # start dfs with level = 0

        return subset_list


# old soluction, use 'sort' to compare and remove duplicates
class Solution:
    def subsetsWithDup(self, nums: list[int]) -> list[list[int]]:
        solution = []
        subset = []

        def dfs(level):
            if level == len(nums):
                if not sorted(subset) in solution:
                    solution.append(sorted(subset.copy()))
                return

            subset.append(nums[level])
            dfs(level + 1)

            subset.pop()
            dfs(level + 1)

        dfs(0)
        return solution


# old solutoin, use dict to compare and remove duplicates
class Solution:
    def subsetsWithDup(self, nums: list[int]) -> list[list[int]]:
        solution = []
        subset = []
        sol_list = []

        def dfs(level):
            if level == len(nums):
                num_count = {}
                for num in subset:
                    num_count[num] = num_count.get(num, 0) + 1

                if not num_count in solution:
                    solution.append(num_count)
                return

            subset.append(nums[level])
            dfs(level + 1)

            subset.pop()
            dfs(level + 1)

        dfs(0)

        sol = []
        for sol_elem in solution:
            so = []
            for k, v in sol_elem.items():
                for _ in range(v):
                    so.append(k)
            sol.append(so)

        return sol





# Combination Sum
# https://leetcode.com/problems/combination-sum/
"""
Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.

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
(Solution().combinationSum([2, 3, 6, 7], 7), [[2, 2, 3], [7]])
(Solution().combinationSum([2, 3, 5], 8), [[2, 2, 2, 2], [2, 3, 3], [3, 5]])
(Solution().combinationSum([2], 1), [])


class Solution:
    def combinationSum(self, candidates: list[int], target: int) -> list[list[int]]:
        # sort candidates to ensure that candidate with index + 1 is greater than previous to not reapat solution
        candidates.sort()
        combination = []  # current combination
        combination_list = []  # solution


        def dfs(value, start_index):
            if value == target:  # target sum reached
                combination_list.append(combination.copy())  # push subset to subset_list
                return
            elif value > target:  # if value is too large
                return
            
            for index in range(start_index, len(candidates)):  # check only equal or higher candidates
                candidate = candidates[index]
                combination.append(candidate)
                dfs(value + candidate, index)
                combination.pop()
               
        dfs(0, 0)

        return combination_list


class Solution:
    def combinationSum(self, candidates: list[int], target: int) -> list[list[int]]:
        combination_list = []  # solution
        # sort candidates to ensure that candidate with index + 1 is greater than previous to not reapat combination_list
        candidates = list(sorted(set(candidates)))

        def dfs(index, subset):
            if sum(subset) == target:  # target sum reached
                combination_list.append(subset.copy())  # push subset to subset_list
                return
            
            # index out of obunds or sum too large
            if index == len(candidates) or sum(subset) > target:
                return
            
            subset.append(candidates[index])  # (left) decision to append current candidate
            dfs(index, subset)

            # (right) decision to not append current candidate
            subset.pop()
            dfs(index + 1, subset)

        dfs(0, [])

        return combination_list





# Combination Sum II
# https://leetcode.com/problems/combination-sum-ii/
"""
Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.

Each number in candidates may only be used once in the combination.

Note: The solution set must not contain duplicate combinations.

Example 1:

Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: 
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]

Example 2:

Input: candidates = [2,5,2,1,2], target = 5
Output: 
[
[1,2,2],
[5]
]
"""


class Solution:
    def combinationSum2(self, nums, target):
        nums.sort()
        combination = []
        combination_list = []

        def dfs(index, value):
            if value == target:
                combination_list.append(combination.copy())
                return
            elif value > target or index == len(nums):
                return

            combination.append(nums[index])
            dfs(index + 1, value + nums[index])
            combination.pop()
            
            while index + 1 < len(nums) and nums[index + 1] == nums[index]:
                index += 1
            
            dfs(index + 1, value)

        dfs(0, 0)

        return combination_list
(Solution().combinationSum2([10, 1, 2, 7, 6, 1, 5], 8), [[1, 1, 6], [1, 2, 5], [1, 7], [2, 6]])
(Solution().combinationSum2([2, 5, 2, 1, 2], 5), [[1, 2, 2], [5]])
(Solution().combinationSum2([6], 6), [])



class Solution:
    def combinationSum2(self, candidates: list[int], target: int) -> list[list[int]]:
        solution = []
        # sort needed to skip same values
        candidates.sort()

        def dfs(index, subset):
            # if sum is achieved
            if sum(subset) == target:
                solution.append(subset.copy())
                return

            # index boundry and sum boundry            
            if index == len(candidates) or sum(subset) > target:
                return

            # (left node) append candidate
            subset.append(candidates[index])
            dfs(index + 1, subset)

            # (right node) skip candidate
            subset.pop()
            # skip same values
            while index + 1 < len(candidates) and candidates[index] == candidates[index + 1]:
                index += 1
            dfs(index + 1, subset)

        dfs(0, [])

        return solution
(Solution().combinationSum2([10, 1, 2, 7, 6, 1, 5], 8), [[1, 1, 6], [1, 2, 5], [1, 7], [2, 6]])
(Solution().combinationSum2([2, 5, 2, 1, 2], 5), [[1, 2, 2], [5]])





# Permutations
# https://leetcode.com/problems/permutations/
"""
Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.

Example 1:

Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

Example 2:

Input: nums = [0,1]
Output: [[0,1],[1,0]]

Example 3:

Input: nums = [1]
Output: [[1]]
"""


class Solution:
    def permute(self, nums):
        per_list = []

        def dfs(prefix, postfix):
            if len(prefix) == len(nums):
                per_list.append(prefix)
                return

            for index in range(len(postfix)):
                dfs(prefix + [postfix[index]], postfix[:index] + postfix[index + 1:])
        
        dfs([], nums)

        return per_list
(Solution().permute([1, 2, 3]), [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]])
(Solution().permute([0, 1]), [[0, 1], [1, 0]])
(Solution().permute([1]), [[1]])



class Solution:
    def permute(self, nums: list[int]) -> list[list[int]]:
        # edge case with one item list
        # if len(nums) == 1:
        #     return [nums]
        
        solution = []

        def dfs(prefix, sublist):
            # when only one (two) element(s) there is 1 (are 2) permutation(s)
            if len(sublist) == 1:
                solution.append(prefix + sublist)
                # solution.append(prefix + sublist[::-1])
                return 

            # for every element in list pop it as prefix and sublist all other elements and run dfs
            for ind in range(len(sublist)):
                dfs(prefix + [sublist[ind]], sublist[:ind] + sublist[ind + 1 :])
        
        dfs([], nums)

        return solution
(Solution().permute([1, 2, 3]), [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]])
(Solution().permute([0, 1]), [[0, 1], [1, 0]])
(Solution().permute([1]), [[1]])





# Word Search
# https://leetcode.com/problems/word-search/
"""
Given an m x n grid of characters board and a string word, return true if word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

Example 1:

Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true

Example 2:

Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
Output: true

Example 3:

Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
Output: false
"""


class Solution:
    def __init__(self) -> None:
        self.end = False

    def exist(self, board: list[list[str]], word: str) -> bool:
        rows = len(board)
        cols = len(board[0])
        index = 0

        def dfs(snake, index):
            # if 'word' it empty all letters matched
            if index == len(word):
                self.end = True
                return
            
            # make 'self.end' True if there's a match
            # if self.end:
            #     return
            
            # current row and coll index
            row, col = snake[-1]

            # check up, down, left, right for neighbouns
            # check if out of bounds and for down match
            # restrict snake from taking the same path (letter) twice
            if (row + 1 < rows and board[row + 1][col] == word[index]
                and (len(snake) == 1 or not (row + 1, col) in snake)):
                dfs(snake + [(row + 1, col)], index + 1)
            
            # check if out of bounds and for up match
            if (row - 1 > -1 and board[row - 1][col] == word[index]
                and (len(snake) == 1 or not (row - 1, col) in snake)):
                dfs(snake + [(row - 1, col)], index + 1)
            
            # check if out of bounds and for right match
            if (col + 1 < cols and board[row][col + 1] == word[index]
                and (len(snake) == 1 or not (row, col + 1) in snake)):
                    dfs(snake + [(row, col + 1)], index + 1)
            
            # check if out of bounds and for left match
            if (col - 1 > -1 and board[row][col - 1] == word[index] 
                and (len(snake) == 1 or not (row, col - 1) in snake)):
                dfs(snake + [(row, col - 1)], index + 1)
        
        for row in range(rows):
            for col in range(cols):
                # if the matching word is found algorithm stops
                if self.end:
                    return True
                # check for the first word letter in all board letters
                if word[index] == board[row][col]:
                    # if so provde 'snake' patch and the word without matched letter
                    dfs([(row, col)], index + 1)

        return self.end
(Solution().exist([["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]], "ABCCED"), True)
(Solution().exist([["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]], "SEE"), True)
(Solution().exist([["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]], "ABCB"), False)





# Palindrome Partitioning
# https://leetcode.com/problems/palindrome-partitioning/
"""
Given a string s, partition s such that every 
substring
 of the partition is a 
palindrome
. Return all possible palindrome partitioning of s.

Example 1:

Input: s = "aab"
Output: [["a","a","b"],["aa","b"]]

Example 2:

Input: s = "a"
Output: [["a"]]
"""


# prefix is passed as an agrument in dfs
class Solution:
    def partition(self, word: str) -> list[list[str]]:
        solution = []

        def dfs(prefix, word):
            # if word is empty that means all letters folded into palindrom
            if not word:
                solution.append(prefix)
                return

            # index starts from '1' instead of '0' because word[:i + 1]
            for index in range(1, len(word) + 1):
                # if prefix is a palindrme
                if word[: index] == word[: index][::-1]:
                    # recurent dfs with
                    # previous prefix + current prefix + rest
                    dfs(prefix + [word[:index]], word[index:])
                        
        dfs([], word)
        return solution

(Solution().partition("aab"), [["a", "a", "b"], ["aa", "b"]])
(Solution().partition("a"), [["a"]])


# one global prefix
class Solution:
    def partition(self, word: str) -> list[list[str]]:
        solution = []
        prefix = []

        def dfs(word):
            # if word is empty that means all letters folded into palindrom
            if not word:
                solution.append(prefix.copy())
                return 

            # index starts from '1' instead of '0' because word[:i + 1]
            for index in range(1, len(word) + 1):
                # if prefix is a palindrme
                if word[: index] == word[: index][::-1]:
                    # recurent dfs with
                    # previous prefix + current prefix + rest
                    prefix.append(word[:index])
                    dfs(word[index:])
                    prefix.pop()
                        
        dfs(word)
        return solution





# Letter Combinations of a Phone Number
# https://leetcode.com/problems/letter-combinations-of-a-phone-number/
"""
Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

A mapping of digits to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

Example 1:

Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
Example 2:

Input: digits = ""
Output: []
Example 3:

Input: digits = "2"
Output: ["a","b","c"]
"""


# passing index to dfs
class Solution:
    def letterCombinations(self, digits: str) -> list[str]:
        if not digits:
            return []
            
        to_letters = {"2": "abc", 
                      "3": "def", 
                      "4": "ghi", 
                      "5": "jkl", 
                      "6": "mno",
                      "7": "pqrs",
                      "8": "tuv",
                      "9": "wxyz"}
        solution = []
        part = []

        def dfs(index):
            if index == len(digits):
                solution.append("".join(part))
                return

            for letter in to_letters[digits[index]]:
                part.append(letter)
                dfs(index + 1)
                part.pop()

        dfs(0)

        return solution
(Solution().letterCombinations("2"), ["a", "b", "c"])
(Solution().letterCombinations("23"), ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"])
(Solution().letterCombinations(""), [])


# passing digits to dfs
class Solution:
    def letterCombinations(self, digits: str) -> list[str]:
        if not digits:
            return []
            
        to_letters = {"2": "abc", 
                      "3": "def", 
                      "4": "ghi", 
                      "5": "jkl", 
                      "6": "mno",
                      "7": "pqrs",
                      "8": "tuv",
                      "9": "wxyz"}
        solution = []
        part = []

        def dfs(digits):
            if not digits:
                solution.append("".join(part))
                return

            for letter in to_letters[digits[0]]:
                part.append(letter)
                dfs(digits[1:])
                part.pop()

        dfs(digits)

        return solution





# N-Queens
# https://leetcode.com/problems/n-queens/
"""
The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.

Given an integer n, return all distinct solutions to the n-queens puzzle. You may return the answer in any order.

Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space, respectively.

Example 1:

Input: n = 4
Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
Explanation: There exist two distinct solutions to the 4-queens puzzle as shown above

Example 2:

Input: n = 1
Output: [["Q"]]
"""


class Solution:
    def solveNQueens(self, n: int) -> list[list[str]]:
        solution = []
        # initiate board
        board = [["."]*n for _ in range(n)]
        tabu_col = set()
        # for each diagonal (col_ind - row_ind) = const
        tabu_diag = set()
        # for each aiti-diagonal (con_ind + row_ind) = const
        tabu_adiag = set()

        def dfs(row):
            # when all row are filled with Queens
            if row == n:
                joined_board = ["".join(row) for row in board]
                solution.append(joined_board)
                return

            for col in range(n):
                # if there is another Queen in the same diagonal or the same col
                if (row - col) in tabu_diag or (row + col) in tabu_adiag or col in tabu_col:
                    continue
                
                # update tabu and board
                tabu_col.add(col)
                tabu_diag.add(row - col)
                tabu_adiag.add(row + col)
                board[row][col] = "Q"

                # check another row
                dfs(row + 1)

                # discard update
                tabu_col.remove(col)
                tabu_diag.remove(row - col)
                tabu_adiag.remove(row + col)
                board[row][col] = "."

        dfs(0)
        return solution
(Solution().solveNQueens(4), [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]])
(Solution().solveNQueens(1), [["Q"]])





# Time Based Key-Value Store
# https://leetcode.com/problems/time-based-key-value-store/
"""
Design a time-based key-value data structure that can store multiple values for the same key at different time stamps and retrieve the key's value at a certain timestamp.

Implement the TimeMap class:

TimeMap() Initializes the object of the data structure.
void set(String key, String value, int timestamp) Stores the key key with the value value at the given time timestamp.
String get(String key, int timestamp) Returns a value such that set was called previously, with timestamp_prev <= timestamp. If there are multiple such values, it returns the value associated with the largest timestamp_prev. If there are no values, it returns "".

Example 1:

Input
["TimeMap", "set", "get", "get", "set", "get", "get"]
[[], ["foo", "bar", 1], ["foo", 1], ["foo", 3], ["foo", "bar2", 4], ["foo", 4], ["foo", 5]]
Output
[null, null, "bar", "bar", null, "bar2", "bar2"]

Explanation
TimeMap timeMap = new TimeMap();
timeMap.set("foo", "bar", 1);  // store the key "foo" and value "bar" along with timestamp = 1.
timeMap.get("foo", 1);         // return "bar"
timeMap.get("foo", 3);         // return "bar", since there is no value corresponding to foo at timestamp 3 and timestamp 2, then the only value is at timestamp 1 is "bar".
timeMap.set("foo", "bar2", 4); // store the key "foo" and value "bar2" along with timestamp = 4.
timeMap.get("foo", 4);         // return "bar2"
timeMap.get("foo", 5);         // return "bar2"
"""


class TimeMap:
    def __init__(self):
        self.store = {}

    def set(self, key: str, value: str, timestamp: int) -> None:
        if not key in self.store:  # if no key at the store add this key with list as value
            self.store[key] = []
        
        self.store[key].append([value, timestamp])  # push value, timestamp pair to the store

    def get(self, key: str, timestamp: int) -> str:
        if not key in self.store:  # when searching for a key that's not in the store
            return ""
        
        values = self.store[key]

        left = 0
        right = len(values) - 1
        solution = ""  # if there is on solution returns ""

        while left <= right:  # two pointers
            mid = (left + right) // 2
            mid_value = values[mid]

            if timestamp == mid_value[1]:
                return values[mid][0]
            elif timestamp < mid_value[1]:  # if timestamp is lower than middle, that means the middle and all to the right is too high, so check left chunk
                right = mid - 1
            else:  # if temistamp is higher than middle, that means the middle and all to the left is too low, so check right chunk
                left = mid + 1
                solution = mid_value[0]  # save the middle because the higher values migght be too high

        return solution

# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)

# Example 1
timeMap = TimeMap()
timeMap.set("foo", "bar", 1)
timeMap.get("foo", 1)
timeMap.get("foo", 3)
timeMap.set("foo", "bar2", 4)
timeMap.get("foo", 4)
timeMap.get("foo", 5)
# Output: [null,null,"bar","bar",null,"bar2","bar2"]

# Example 2
timeMap = TimeMap()
timeMap.set("love","high",10)
timeMap.set("love","low",20)
timeMap.get("love",5)
timeMap.get("love",10)
timeMap.get("love",15)
timeMap.get("love",20)
timeMap.get("love",25)
# Ouput: [null,null,null,"","high","high","low","low"]

# Example 3
timeMap = TimeMap()
timeMap.set("a", "bar", 1)
timeMap.set("x", "b", 3)
timeMap.get("b", 3)
timeMap.set("foo", "bar2", 4)
timeMap.get("foo", 4)
timeMap.get("foo", 5)
# Output: [null,null,null,"",null,"bar2","bar2"]





# 
