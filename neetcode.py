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
(Solution().containsDuplicate([1, 2, 3]), False)
(Solution().containsDuplicate([1, 2, 3, 4]), False)
(Solution().containsDuplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]), True)


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
(Solution().isAnagram("anagram", "nagaram"), True)
(Solution().isAnagram("rat", "car"), False)


# O(n), O(n)
class Solution:
    def counter(self, word):
        counter = {}

        for letter in word:
            counter[letter] = counter.get(letter, 0) + 1

        return counter

    def isAnagram(self, word1, word2):
        return self.counter(word1) == self.counter(word2)


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
(Solution().twoSum([2, 7, 11, 15], 9), [0, 1])
(Solution().twoSum([3, 2, 4], 6), [1, 2])
(Solution().twoSum([3, 3], 6), [0, 1])
(Solution().twoSum([3, 3], 7), None)


# O(n), O(n)
class Solution:
    def twoSum(self, numbers: list[int], target: int) -> list[int]:
        seen = {}

        for index, number in enumerate(numbers):
            diff = target - number
        
            if diff in seen: # seen.get(diff, False)
                return [seen[diff], index] # [seen.get(diff), index]
            seen[number] = index # seen.update({number: ind})
        
        return None

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
(Solution().groupAnagrams(["eat","tea","tan","ate","nat","bat"]), [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']])
(Solution().groupAnagrams([""]), [[""]])
(Solution().groupAnagrams(["a"]), [["a"]])
(Solution().groupAnagrams(["tin","ram","zip","cry","pus","jon","zip","pyx"]), [['tin'], ['ram'], ['zip', 'zip'], ['cry'], ['pus'], ['jon'], ['pyx']])


# O(m*n), O(m*n)
class Solution:
    def groupAnagrams(self, words: list[str]) -> list[list[str]]:
        grouped_anagrams = {}
        # from collections import defaultdict
        # grouped_anagrams = defaultdict(list)

        for word in words:
            key = [0] * 26

            for letter in word:
                key[ord(letter) - ord("a")] += 1

            key = tuple(key)

            if not key in grouped_anagrams:
                grouped_anagrams[key] = []

            grouped_anagrams[key].append(word)

        return list(grouped_anagrams.values())


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
(Solution().topKFrequent([1, 1, 1, 2, 2, 3], 2), [1, 2])
(Solution().topKFrequent([1], 1), [1])


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

        return [left_prod_cum[i] * right_prod_cum[i] 
                for i in range(len(nums))]




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


# 2*3*4  1* 3*4  1*2* 4  1*2*3
# looks good, but too slow for leetcode
class Solution:
    def productExceptSelf(self, nums: list[int]) -> list[int]:
        left_product = [1] * len(nums)  # prefix
        right_product = [1] * len(nums)  # postfix

        for index in range(len(nums) - 1):
            # cummulative prod from left
            left_product[index + 1] = left_product[index] * nums[index]
            # cummulative prod from right
            # right_product[index + 1] = right_product[index] * nums[::-1][index]
            right_product[index + 1] = right_product[index] * nums[len(nums) - 1 - index]
            
        # right_product.reverse()

        return [left_product[index] * right_product[::-1][index] 
                for index in range(len(left_product))]
        # return left_product
        # [1,1,2,6]
        # return right_product
        # [24,12,4,1]





# Encode and Decode Strings
# https://www.lintcode.com/problem/659/
"""
Description
Design an algorithm to encode a list of strings to a string. The encoded string is then sent over the network and is decoded back to the original list of strings.

Please implement encode and decode

Because the string may contain any of the 256 legal ASCII characters, your algorithm must be able to handle any character that may appear

Do not rely on any libraries, the purpose of this problem is to implement the "encode" and "decode" algorithms on your own
"""
(Solution().encode(["code", "site", "love", "you"]), "#4code#4site#4love#3you")
(Solution().decode(Solution().encode(["code", "site", "love", "you"])), ["code", "site", "love", "you"])
(Solution().decode(Solution().encode([])), [])
(Solution().decode(Solution().encode([""])), [""])


class Solution:
    def encode_word(self, word):
        return f"{len(word)}#{word}"

    def encode(self, words):
        return "".join(map(self.encode_word, words))


    def decode(self, word):
        decoded = []
        
        while word:
            word_length = ""

            while word[0].isdigit():
                word_length += word[0]
                word = word[1:]
            
            word = word[1:]
            word_length = int(word_length)
            decoded.append(word[:word_length])
            word = word[word_length:]
        
        return decoded


import re

class Solution:

    def encode(self, strs: list[str]) -> str:
        result = ""
        for word in strs:
            result += str(len(word)) + "#" + word
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

        result = re.findall(r"\d+#(\D+)", s)
        
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
(Solution().longestConsecutive([100, 4, 200, 1, 3, 2]), 4)
(Solution().longestConsecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]), 9)


# O(n), O(n)
class Solution:
    def longestConsecutive(self, nums: list[int]) -> int:
        num_set = set(nums)
        longest_concec = 0

        for num in num_set:
            if not num - 1 in num_set:
                curr_len = 1
                
                while num + curr_len in num_set:
                    curr_len += 1
                longest_concec = max(longest_concec, curr_len)

        return longest_concec





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
(Solution().isPalindrome("A man, a plan, a canal: Panama"), True)
(Solution().isPalindrome("race a car"), False)
(Solution().isPalindrome(" "), True)
(Solution().isPalindrome("0P"), False)


# O(n), O(n)
# Two Pointers
class Solution:
    def isPalindrome(self, text: str) -> bool:
        left = 0
        right = len(text) - 1

        while left < right:
            while (not text[left].isalnum() and 
                   left < right):
                left += 1

            while (not text[right].isalnum() and 
                   left < right):
                right -= 1
            
            if text[left].lower() != text[right].lower():
                return False

            left += 1
            right -= 1

        return True


# O(n), O(n)
# reverse str
class Solution:
    def isPalindrome(self, text: str) -> bool:
        joined = "".join(filter(str.isalnum, text)).lower()
        return joined == joined[::-1]


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
(Solution().twoSum([2, 7, 11, 15], 9), [1, 2])
(Solution().twoSum([2, 3, 4], 6), [1, 3])
(Solution().twoSum([-1, 0], -1), [1, 2])


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
(Solution().threeSum([-1, 0, 1, 2, -1, -4]), [[-1, -1, 2], [-1, 0, 1]])
(Solution().threeSum([3, 0, -2, -1, 1, 2]), [[-2, -1, 3], [-2, 0, 2], [-1, 0, 1]])
(Solution().threeSum([1, 1, -2]), [[-2, 1, 1]])
(Solution().threeSum([-1, 1, 1]), [])
(Solution().threeSum([0, 0, 0]), [[0, 0, 0]])
(Solution().threeSum([-2, 0, 0, 2, 2]), [[-2, 0, 2]])


class Solution:
    def threeSum(self, numbers: list[int]) -> list[list[int]]:
        triplets = []
        numbers.sort()

        for index, num in enumerate(numbers[:-2]):
            # Skip positive numbers
            if num > 0:
                break
            
            # Skip same num values
            if (index and 
                numbers[index] == numbers[index - 1]):
                continue

            left = index + 1  # left pointer
            right = len(numbers) - 1  # right pointer

            # two pointers
            while left < right:
                curr_sum = num + numbers[left] + numbers[right]
                
                if curr_sum < 0:  # if sum is less than 0
                    left += 1
                elif curr_sum > 0:  # if sum is geater than 0
                    right -= 1
                else:  # if sum is equal to 0
                    triplets.append([num, numbers[left], numbers[right]])
                    left += 1
                    right -= 1
                    
                    # skip same left pointer values
                    while (left < right and 
                           numbers[left] == numbers[left - 1]):
                        left += 1                
        
        return triplets


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
"""
Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

Example 1:

Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
              _
      _      | |_   _
  _  | |~ ~ ~|   |~| |_
_| |~|   |~|           |
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.
Example 2:

Input: height = [4,2,0,3,2,5]
Output: 9
"""
(Solution().trap([1, 3, 2, 1, 2, 1, 5, 3, 3, 4, 2]), 8)
(Solution().trap([5, 8]), 0)
(Solution().trap([3, 1, 2]), 1)
(Solution().trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]), 6)
(Solution().trap([4, 2, 0, 3, 2, 5]), 9)


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
(Solution().maxProfit([7, 1, 5, 3, 6, 4]), 5)
(Solution().maxProfit([7, 6, 4, 3, 1]), 0)
(Solution().maxProfit([2, 4, 1]), 2)
(Solution().maxProfit([2, 1, 2, 1, 0, 1, 2]), 2)
(Solution().maxProfit([1, 2]), 1)


# pointers as values
class Solution:
    def maxProfit(self, prices):
        left_price = prices[0]  # the left_price pointer as a value
        max_profit = 0

        for right_price in prices[1:]:  # the right_price pointer as a value
            if left_price > right_price:  # if price is lower buy
                left_price = right_price
            else:  # if price is higher calculate revenue
                profit = right_price - left_price
                max_profit = max(max_profit, profit)

        return max_profit


# pointers as indexes
class Solution:
    def maxProfit(self, prices):
        max_profit = 0
        left = 0  # the left pointer

        for right in range(1, len(prices)):  # the right pointer
            if prices[left] > prices[right]:  # if price is lower buy
                left = right
            else:  # if price is higher calculate revenue
                profit = prices[right] - prices[left]
                max_profit = max(max_profit, profit)

        return max_profit


# O(n), O(n)
# two pass
class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        min_price = [0] * len(prices)  # cache

        for index, price in enumerate(prices):
            if index:
                min_price[index] = min(min_price[index - 1], price)
            else:
                min_price[index] = price
        
        for index, price in enumerate(prices):
            min_price[index] = price - min_price[index]
        
        return max(min_price)
    

# O(n), O(n)
# one pass
class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        min_price = [0] * len(prices)  # cache
        max_profit = 0

        for index, price in enumerate(prices):
            if index:
                min_price[index] = min(min_price[index - 1], price)
            else:
                min_price[index] = price

            max_profit = max(max_profit, price - min_price[index])
        
        return max_profit
    

# O(n), O(1)
# one pass
class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        max_profit = 0

        for index, price in enumerate(prices):
            if index:
                min_price = min(min_price, price)
            else:
                min_price = price

            max_profit = max(max_profit, price - min_price)
        
        return max_profit


# O(n), O(1)
# one pass, basic functions
class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        max_profit = 0

        for index, price in enumerate(prices):
            if index:
                if price < min_price:
                    min_price = price
            else:
                min_price = price

            profit = price - min_price
        
            if profit > max_profit:
                max_profit = profit
        
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
(Solution().lengthOfLongestSubstring("abcabcbb"), 3)
(Solution().lengthOfLongestSubstring("bbbbb"), 1)
(Solution().lengthOfLongestSubstring("pwwkew"), 3)
(Solution().lengthOfLongestSubstring("aabaab!bb"), 3)
(Solution().lengthOfLongestSubstring("aab"), 2)


# O(n), O(n)
# window as set()
# two pointers
class Solution:
    def lengthOfLongestSubstring(self, word):
        window = set()  # slidiong window without repeating characters
        longest_substring = 0
        left = 0  # left pointer

        for right, letter in enumerate(word):  # right pointer
            while letter in window:  # if duplicate found
                window.discard(word[left])  # remove (discard) that charactr with every preceding character
                left += 1  # increase the left pointer

            window.add(letter)  # add an unique letter
            longest_substring = max(longest_substring, right - left + 1)  # update the length of the longest substring

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
(Solution().characterReplacement("ABAB", 2), 4)
(Solution().characterReplacement("AABABBA", 1), 4)


# O(n), O(n)
# sliding window
class Solution:
    def characterReplacement(self, word, joker):
        window = {}  # window as dict
        left = 0  # left pointer
        longest = 0

        for right, letter in enumerate(word):  # right pointer
            window[letter] = window.get(letter, 0) + 1  # add new letter to window
            
            while (right - left + 1) - max(window.values()) > joker:  # check ("sum - max") how many character are replaced, if exeded
                window[word[left]] -= 1  # remove "left" character from the window
                left += 1

            longest = max(longest, (right - left + 1))  # update the length of the longest replacement
            # (right - left + 1) is much faster than sum(window.values())

        return longest

# O(n), O(n)
# sliding window, max_requency
class Solution:
    def characterReplacement(self, word: str, joker: int) -> int:
        window = {}
        longest = 0
        left = 0
        max_frequency = 0
        
        for right, letter in enumerate(word):
            window[letter] = window.get(letter, 0) + 1
            max_frequency = max(max_frequency, window[letter])

            while (right - left + 1) - max_frequency > joker:
                window[word[left]] -= 1
                left += 1
         
            longest = max(longest, right - left + 1)

        return longest





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
(Solution().isValid("()"), True)
(Solution().isValid("({})"), True)
(Solution().isValid("(]"), False)
(Solution().isValid("(})"), False)
(Solution().isValid("([)"), False)
(Solution().isValid(""), True)
(Solution().isValid("["), False)


class Solution:
    def isValid(self, brackets: str) -> bool:
        seen_bracket = []
        oppos_bracket = {
            ")": "(",
            "]": "[",
            "}": "{"
        }

        for bracket in brackets:
            if bracket in oppos_bracket:
                if (seen_bracket and 
                    oppos_bracket[bracket] == seen_bracket[-1]):
                    seen_bracket.pop()
                else:
                    return False
            else:
                seen_bracket.append(bracket)
        
        # if not empty then unmatched brackets left
        return not seen_bracket





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

# if middle > right: search right
# 2, 3, 4, 5, 1
# 3, 4, 5, 1, 2

# 1, 2, 3, 4, 5
# 4, 5, 1, 2, 3
# 5, 1, 2, 3, 4


# O(logn), O(1)
# binary search
class Solution:
    def findMin(self, numbers):
        left = 0
        right = len(numbers) - 1
        curr_min = numbers[0]  # assign some value

        while left <= right:  # two pointers 
            middle = (left + right) // 2  # get the middle index
            curr_min = min(curr_min, numbers[middle])  # check if the value in the middle it lower than currten min

            if numbers[middle] < numbers[right]: # if the middle value is lower than the most right value
                right = middle - 1  # then the left part should be searched
            else:
                left = middle + 1  # else the right part should be searched

        return curr_min


# O(logn), O(1)
# binary search
class Solution:
    def findMin(self, numbers: list[int]) -> int:
        left = 0
        right = len(numbers) - 1

        while left <= right:
            if right - left <= 1:
                return min(numbers[left], numbers[right])
            
            middle = (left + right) // 2

            if numbers[middle] < numbers[right]:  # minimum is in the left portion
                right = middle
            else:
                left = middle + 1





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


# if middle < right: right portion is contiguous
# 1, 2, 3, 4, 5
# 4, 5, 1, 2, 3
# 5, 1, 2, 3, 4

# 2, 3, 4, 5, 1
# 3, 4, 5, 1, 2


# O(logn), O(1)
# binary search
class Solution:
    def search(self, numbers: list[int], target: int) -> int:
        left = 0
        right = len(numbers) - 1

        while left <= right:
            middle = (left + right) // 2
            middle_number = numbers[middle]

            if middle_number == target:
                return middle
            elif middle_number < numbers[right]:  # contiguous right side portion
                if target > middle_number and target <= numbers[right]:  # target in the right side
                    left = middle + 1
                else:
                    right = middle - 1  # tagget in the left side
            else:  # contiguous left side portion
                if target >= numbers[0] and target < middle_number:
                    right = middle - 1  # tagget in the left side
                else:
                    left = middle + 1  # target in the right side

        return -1


# O(logn), O(1)
# binary search
class Solution:
    def search(self, numbers: list[int], target: int) -> int:
        left = 0
        right = len(numbers) - 1

        while left <= right:
            middle = (left + right) // 2
            middle_number = numbers[middle]

            if middle_number == target:
                return middle
            elif middle_number > numbers[right]:  # [2, 3, 4, 5, 1] the left chunk [2, 3] is ascending the other has a pivot
                # if in [5, 1]
                if target > middle_number or target < numbers[left]: # target <= numbers[r]
                    left = middle + 1
                else:
                    right = middle - 1    
            else:  # middle_number < numbers[r] [5, 1, 2, 3, 4] the right chunk [3, 4] is ascending the other has a pivot
                if target > numbers[right] or target < middle_number: # target <= nums[r] is wrong because (Solution().search([1, 3, 5], 5), 2)
                    right = middle - 1
                else:
                    left = middle + 1

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
(Solution().climbStairs(0), 0)
(Solution().climbStairs(1), 1)
(Solution().climbStairs(2), 2)
(Solution().climbStairs(3), 3)
(Solution().climbStairs(4), 5)
(Solution().climbStairs(5), 8)


# Fibonnacci problem
# dp, bottom-up with no auxiliary memory space
# O(n), O(1)
class Solution:
    def climbStairs(self, num: int) -> int:
        if num < 4:
            return num

        a = 0
        b = 1

        for _ in range(num):
            a, b = b, a + b

        return b


# dp, bottom-up
# O(n), O(n)
class Solution:
    def climbStairs(self, num: int) -> int:
        if num < 4:
            return num
        
        dp = [0 for _ in range(num + 2)]
        dp[1] = 1

        for index in range(2, num + 2):
            dp[index] = dp[index - 1] + dp[index - 2]

        return dp[-1]


# dp, top-down with memoization
# O(n), O(n)
class Solution:
    def __init__(self) -> None:
        self.memo = {}

    def climbStairs(self, num: int) -> int:
        if num < 4:
            return num
        
        if num in self.memo:
            return self.memo[num]
        else:
            self.memo[num] = self.climbStairs(num - 1) + self.climbStairs(num - 2)
            return self.memo[num]


# dfs, unefficient
# O(2^n), O(n)
# counter as shared variable (list)
class Solution:
    def climbStairs(self, num: int, index = 0) -> int:
        counter = []

        def dfs(index):
            if index > num:
                return
            
            if index == num:
                counter.append(1)
                return

            dfs(index + 1)
            dfs(index + 2)
            
        dfs(index)
        return len(counter)


# dfs, unefficient
# O(2^n), O(n)
# "counter" as a return statement from dfs
class Solution:
    def climbStairs(self, num: int, index = 0) -> int:
        def dfs(num, index):
            if index > num:
                return 0
            
            if index == num:
                return 1

            return dfs(num, index + 1) + dfs(num, index + 2)
            
        return dfs(num, index)


# generator
def Fib_gen(n):
    a = 1
    b = 1

    for _ in range(n):
        yield b
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
(Solution().rob([2, 100, 9, 3, 100]), 200)
(Solution().rob([100, 9, 3, 100, 2]), 200)
(Solution().rob([1, 2, 3, 1]), 4)
(Solution().rob([2, 7, 9, 3, 1]), 12)
(Solution().rob([0]), 0)
(Solution().rob([2, 1]), 2)

# draft
# [2, 100, 9, 3, 100] data
# [2, 100, 100, 100, 200] dp solution
# [(2), (2, 100), (100, 100) ....]

# [100, 9, 3, 100, 2] data
# [100, 100, 100, 200, 200] dp solution


# dp, bottom-up
# O(n), O(1), cache only two last elements
class Solution:
    def rob(self, nums: list[int]):
        if len(nums) == 1:
            return nums[0]
        
        house_1 = nums[0]
        house_2 = max(nums[0], nums[1])

        for index in range(2, len(nums)):
            house_1, house_2 = house_2, max(nums[index] + house_1, house_2)
        
        return house_2


# dp, bottom-up
# O(n), O(n), cache every robbery
class Solution:
    def rob(self, nums: list[int]):
        if len(nums) == 1:
            return nums[0]
        
        dp = [0 for _ in range(len(nums))]
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])

        for index in range(2, len(nums)):
            dp[index] = max(nums[index] + dp[index - 2], dp[index - 1])
        
        return dp[-1]


# dp, top-down, with memoization
# memo in init
# O(n), O(n)
class Solution:
    def __init__(self):
        self.memo = {}  # memoization cache

    def rob(self, nums: list[int], index=1):
        if len(nums) == 1:  # if one element in nums
            return nums[0]

        if len(nums) - index == 1:  # if index is 1
            return max(nums[0], nums[1])

        if len(nums) - index == 0:  # if index is 0
            return nums[0]

        if not index + 1 in self.memo:  # if "index + 1" is not in the memo
            self.memo[index + 1] = self.rob(nums, index + 1)  # calculate it
        prev = self.memo[index + 1]  # take it

        if not index + 2 in self.memo:
            self.memo[index + 2] = self.rob(nums, index + 2)
        prev_prev = self.memo[index + 2]

        return max(nums[len(nums) - index] + prev_prev, prev)


# dp, top-down, with memoization
# memo as and argumen with default value
# In Python, default arguments are evaluated once when the function is defined, not each time the function is called. So, when you use a mutable object (like a dictionary) as a default argument, it persists between function calls, causing unexpected behavior if the function is called multiple times.
# O(n), O(n)
class Solution:
    def rob(self, nums: list[int], index=1, memo=None):
        if not memo:
            memo = {}

        if len(nums) == 1:  # if one element in nums
            return nums[0]

        if len(nums) - index == 1:  # if index is 1
            return max(nums[0], nums[1])

        if len(nums) - index == 0:  # if index is 0
            return nums[0]

        if not index + 1 in memo:  # if "index + 1" is not in the memo
            memo[index + 1] = self.rob(nums, index + 1, memo)  # calculate it
        prev = memo[index + 1]  # take it

        if not index + 2 in memo:
            memo[index + 2] = self.rob(nums, index + 2, memo)
        prev_prev = memo[index + 2]

        return max(nums[len(nums) - index] + prev_prev, prev)





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
(Solution().rob([2, 3, 2]), 3)
(Solution().rob([1, 2, 3, 1]), 4)
(Solution().rob([1, 2, 3]), 3)
(Solution().rob([1]), 1)
(Solution().rob([0, 0]), 0)
(Solution().rob([1, 3, 1, 3, 100]), 103)


# O(n), 0(1)
# function in function
class Solution:
    def rob(self, nums: list[int]) -> int:  # rob in circle
        if len(nums) < 3:
            return max(nums)
        
        def rob_straight(nums):  # rob in stight line
            house1 = nums[0]
            house2 = max(nums[0], nums[1])

            for num in nums[2:]:
                house1, house2 = house2, max(num + house1, house2)

            return house2

        return max(
            rob_straight(nums[:-1]), 
            rob_straight(nums[1:]))


# O(n), 0(1)
# both functions directly under class
class Solution:
    def rob_straight(self, nums: list[int]) -> int:  # rob in stight line
        house1 = nums[0]
        house2 = max(nums[0], nums[1])

        for num in nums[2:]:
            house1, house2 = house2, max(num + house1, house2)

        return house2

    def rob(self, nums):  # rob in circle
        if len(nums) < 3:
            return max(nums)

        return max(
            self.rob_straight(nums[:-1]), 
            self.rob_straight(nums[1:]))





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
(Solution().longestPalindrome("babad"), "bab")
(Solution().longestPalindrome("cbbd"), "bb")
(Solution().longestPalindrome("a"), "a")
(Solution().longestPalindrome(""), "")
(Solution().longestPalindrome("bb"), "bb")
(Solution().longestPalindrome("ab"), "a")
(Solution().longestPalindrome("aacabdkacaa"), "aca")
(Solution().longestPalindrome("abdka"), "a")
(Solution().longestPalindrome("aaaa"), "aaaa")


# O(n2), O(n)
class Solution:
    def longestPalindrome(self, word):
        longest_palindrome = ""
        max_word_len = 0

        for index in range(len(word)):
            # odd length palindrome
            edge = 1

            while (index - edge >= 0 and  # check if not out of bounds left
                   index + edge < len(word) and  # check if not out of bounds right
                   word[index - edge] == word[index + edge]):  # if letter match
                edge += 1  # 1 -> 3, 2i + 1 increase palindrome length

            if 2 * edge - 1 > max_word_len:  # if longer palindrome found
                max_word_len = 2 * edge - 1
                longest_palindrome = word[index - edge + 1: index + edge]

            # even lenght palindrome
            edge = 0

            while (index - edge >= 0 and  # check if not out of bounds left
                   index + edge + 1 < len(word) and  # check if not out of bounds right
                   word[index - edge] == word[index + 1 + edge]):  # if letter match
                edge += 1  # 2 -> 4, 2i increase palindrome length

            if 2 * edge > max_word_len:  # if longer palindrome found
                max_word_len = 2 * edge
                longest_palindrome = word[index - edge + 1: index + edge + 1]

        return longest_palindrome



# oldies
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
(Solution().countSubstrings("abc"), 3)
(Solution().countSubstrings("aaa"), 6)


# draft
# "aaa"
# a aa a aa aaa a 

class Solution:
    def countSubstrings(self, word):
        counter_sum = 0
        
        for index in range(len(word)):
            # odd length palindrome
            counter = 1

            while (index - counter >=0 and  # check if not out of bounds left
                   index + counter < len(word) and  # check if not out of bounds right
                   word[index - counter] == word[index + counter]):  # if letter match
                counter += 1

            counter_sum += counter  # update counter_sum
    
            # even length palindrome
            counter = 0

            while (index - counter >=0 and
                   index + 1 + counter < len(word) and
                   word[index - counter] == word[index + 1 + counter]):
                counter += 1

            counter_sum += counter

        return counter_sum


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
(Solution().coinChange([1, 2, 5], 11), 3)
(Solution().coinChange([2, 5, 10, 1], 27), 4)
(Solution().coinChange([186, 419, 83, 408], 6249), 20)
(Solution().coinChange([2], 3), -1)
(Solution().coinChange([2], 1), -1)
(Solution().coinChange([1], 0), 0)


# dp, bottom-up, iteration, tabulation (with list)
class Solution:
    def coinChange(self, coins, amount):
        # min number of coins needed to get target amount (equal to the index)
        # "anmount + 1" an imposbile value stays when the last element of min_coins was not modified
        min_coins = [amount + 1 for _ in range(amount + 1)]
        min_coins[0] = 0  # no coins needed to get 0

        for index in range(1, amount + 1):  # check each 'min_coins' index
            for coin in coins:  # check every coin
                # choose current amount of coins or get ammount without current coin and add 1
                if index - coin >= 0:
                    min_coins[index] = min(min_coins[index], min_coins[index - coin] + 1)
        
        if min_coins[amount] == amount + 1:  # if the last value was not modified so there is no valid combination
            return -1
        else:
            return min_coins[amount]  # valid combination
        # return -1 if tabulation[amount] == amount + 1 else tabulation[amount]


# dp, bottom-up, iteration, tabulation (with dict)
# cache as dict and is slower than cache as list
class Solution:
    def coinChange(self, coins, amount):
        # min number of coins needed to get target amount (equal to the index)
        # "anmount + 1" an imposbile value stays when the last element of min_coins was not modified
        min_coins = {k: amount + 1 for k in range(amount + 1)}
        min_coins[0] = 0  # no coins needed to get 0

        for coin in coins:  # check every coin
            for index in range(coin, amount + 1):  # check each 'min_coins' index
                # choose current amount of coins or get ammount without current coin and add 1
                min_coins[index] = min(min_coins[index], min_coins[index - coin] + 1)
        
        if (amount in min_coins and  # if amount exists and
            min_coins[amount] < amount + 1):  # the value is less than impossible value
            return min_coins[amount]  # valid combination
        else:
            return -1


# dp, dfs, top-down, recursion, memoization (with dict)
class Solution:
    def coinChange(self, coins, amount):
        # Create a memoization dictionary
        memo = {}
        memo[0] = 0  # Base case: 0 coins needed to make amount 0

        def dfs(index):
            if index < 0:  # If the remainder is negative
                return amount + 1  # return an impossible value
            elif index in memo:  # Already computed this value
                return memo[index]

            memo[index] = amount + 1

            # Calculate the minimum number of coins needed for this amount
            for coin in coins:
                memo[index] = min(memo[index], dfs(index - coin) + 1)

            return memo[index]
        
        solution = dfs(amount)

        return -1 if solution == amount + 1 else solution


# dp, dfs, top-down, recursion, memoization (with dict)
# list is not working for
# (Solution().coinChange([186, 419, 83, 408], 6249), 20)
class Solution:
    def coinChange(self, coins, amount):
        # Create a memoization array with initial high values
        memo = [amount + 1] * (amount + 1)
        memo[0] = 0  # Base case: 0 coins needed to make amount 0

        def dfs(index):
            if index < 0:  # If the remainder is negative
                return amount + 1  # return an impossible value
            elif memo[index] != amount + 1:  # Already computed this value
                return memo[index]

            # Calculate the minimum number of coins needed for this amount
            for coin in coins:
                memo[index] = min(memo[index], dfs(index - coin) + 1)

            return memo[index]
        
        solution = dfs(amount)

        return -1 if solution == amount + 1 else solution





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
(Solution().maxProduct([-4, -3]), 12)
(Solution().maxProduct([2, 3, -2, 4]), 6)
(Solution().maxProduct([-2]), -2)
(Solution().maxProduct([-4, -3]), 12)
(Solution().maxProduct([-2, 0, -1]), 0)
(Solution().maxProduct([-2, -3, 7]), 42)
(Solution().maxProduct([2, -5, -2, -4, 3]), 24)
(Solution().maxProduct([-2]), -2)
(Solution().maxProduct([0]), 0)
(Solution().maxProduct([-2, 0]), 0)
(Solution().maxProduct([0, 2]), 2)

# [2, 3, -2, 4]
# 2, 2*3 3 = 6, 6*-2 -2=-2

# dp, bottom-up
# O(n), O(1)
class Solution:
    def maxProduct(self, nums: list[int]) -> int:
        dp_0 = (nums[0], nums[0])  # O(1) cache
        max_product = nums[0]  # track max product with default value

        for index in range(1, len(nums)):  # check all nums indexes
            # multiply prefix values with current value to get min, max or
            # current value only when prefix is (0, 0)
            triplet = (dp_0[0] * nums[index], 
                       dp_0[1] * nums[index], 
                       nums[index])
        
            dp_0 = (max(triplet), min(triplet))  # append min, max pair
            max_product = max(max_product, max(triplet))  # update max product

        return max_product


# dp, bottom-up
# O(n), O(n)
class Solution:
    def maxProduct(self, nums: list[int]) -> int:
        dp = [0 for _ in range(len(nums))]  # min, max pair list for tabulation
        dp[0] = (nums[0], nums[0])  # insert the first element
        max_product = nums[0]  # track max product with default value

        for index in range(1, len(nums)):  # check all nums indexes
            # multiply prefix values with current value to get min, max or
            # current value only when prefix is (0, 0)
            triplet = (dp[index - 1][0] * nums[index],
                       dp[index - 1][1] * nums[index], 
                       nums[index])
        
            dp[index] = (max(triplet), min(triplet))  # append min, max pair
            max_product = max(max_product, max(triplet))  # update max product

        return max_product


# dp, bottom-up
# reversed search
# O(n), O(n)
class Solution:
    def maxProduct(self, nums: list[int]) -> int:
        dp = [(1, 1) for _ in range(len(nums) + 1)]  # min, max pair list for tabulation
        max_product = nums[0]  # track max product with default value

        for index in range(len(nums))[::-1]:  # check all nums indexes
            # multiply prefix values with current value to get min, max or
            # current value only when prefix is (0, 0)
            triplet = (dp[index + 1][0] * nums[index],
                       dp[index + 1][1] * nums[index], 
                       nums[index])
        
            dp[index] = (max(triplet), min(triplet))  # append min, max pair
            max_product = max(max_product, max(triplet))  # update max product

        return max_product


# oldies
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
(Solution().wordBreak("leetcode", ["leet", "code"]), True)
(Solution().wordBreak("applepenapple", ["apple","pen"]), True)
(Solution().wordBreak("catsandog", ["cats","dog","sand","and","cat"]), False)
(Solution().wordBreak("cars", ["car", "ca", "rs"]), True)


# dp, bottom-up
# O(n2), O(n)
class Solution:
    def wordBreak(self, sentence: str, word_list: list[str]) -> bool:
        # cache where each elemet tells if sentece can be fold from this index to the right
        can_fold = [False for _ in range(len(sentence) + 1)]
        can_fold[-1] = True  # dummy element tells that everything after "sentence can be folded"
        word_set = set(word_list)

        for index in range(len(sentence))[::-1]:  # go through every index reversed
            for word in word_set:  # go through every word
                if sentence[index : index + len(word)] == word:  # if found the word
                    can_fold[index] = can_fold[index] or can_fold[index + len(word)]  # update can fold

        return can_fold[0]





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
(Solution().lengthOfLIS([10, 9, 2, 5, 3, 7, 101, 18]), 4)
(Solution().lengthOfLIS([0, 1, 0, 3, 2, 3]), 4)
(Solution().lengthOfLIS([7, 7, 7, 7, 7, 7, 7]), 1)


# draft
# [10, 9, 2, 5, 3, 7, 101, 18]
# [1, 1, 1, 2, 2, max(2,2)+1=3, 4, 1]

# dp, bottom-up, tabulation with list
# O(n2), O(n)
class Solution:
    def lengthOfLIS(self, nums: list[int]) -> int:
        dp = [1 for _ in range(len(nums))]  # LIS lengths

        for right in range(len(nums)):  # check every right (index)
            for left in range(right):  # check every left (index) lower than right
                if nums[left] < nums[right]:  # if right num is greater
                    dp[right] = max(dp[right], dp[left] + 1)  # update LIS lengths 

        return max(dp)


# dp, bottom-up, tabulation with dict
# O(n2), O(n)
class Solution:
    def lengthOfLIS(self, nums: list[int]) -> int:
        dp = {}  # LIS lengths

        for right in range(len(nums)):  # check every right (index)
            dp[nums[right]] = 1  # add current element to dp
            
            for left in dp.keys():  # check every left (index) of dp
                if left < nums[right]:  # if right num is greater
                    dp[nums[right]] = max(dp[nums[right]], dp[left] + 1)  # update LIS lengths 

        return max(dp.values())





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
(Solution().canPartition([14, 9, 8, 4, 3, 2]), True)
(Solution().canPartition([1, 2, 5]), False)
(Solution().canPartition([1, 5, 11, 5]), True)
(Solution().canPartition([3, 3, 3, 4, 5]), True)
(Solution().canPartition([1, 2, 3, 5]), False)
(Solution().canPartition([1]), False)
(Solution().canPartition([2, 2, 1, 1]), True)


class Solution:
    def canPartition(self, nums: list[int]):
        if sum(nums) % 2:  # if odd sum (cannot be split in half)
            return False

        half = sum(nums) // 2  # half of the sum
        seen_numbers = set()  # numbers seen in previous loop

        for num in nums:  # for every number
            new_numbers = seen_numbers.copy()  # copy of seen numbers
            new_numbers = {new_number + num for new_number in new_numbers}  # new numbers in current loop
            seen_numbers.update(new_numbers)  # seen_numbers |= new_numbers  update seen numbers
            seen_numbers.add(num)  # add current num

            # update seen_numbers in one line, has to initiate seen_numbers = {0} to have seen_numbers.add(num)
            # seen_numbers.update({new_number + num for new_number in seen_numbers})

            if half in seen_numbers:  # check if half is in seen numbers
                return True

        return False





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
(Solution().maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4]), 6)
(Solution().maxSubArray([1]), 1)
(Solution().maxSubArray([5, 4, -1, 7, 8]), 23)
(Solution().maxSubArray([-4, -2, -1, -3]), -1)

# for [-2, 1, -3, 4, -1, 2, 1, -5, 4]
# cummulative sum [-2, 1, -2, 4, 3, 5, 6, 1, 4]

# dp, bottom-up, dp list => 2 variables: cumulative, maxSum
# O(n), O(1)
class Solution:
    def maxSubArray(self, nums: list[int]) -> int:
        nums_len = len(nums)
        cumulative = nums[0]  # in case of all negative values cannot take "0" as a base
        max_value = nums[0]  # in case of all negative values cannot take "0" as a base

        for index in range(1, nums_len):
            cumulative = max(cumulative, 0) + nums[index]  # keep track of the cumulative sum or start a new one if below zero
            max_value = max(cumulative, max_value)  # keep track of the highest value

        return max_value


# dp, bottom-up, use nums as dp
# O(n), O(n)
class Solution:
    def maxSubArray(self, nums: list[int]) -> int:
        for index in range(1, len(nums)):
            nums[index] += max(nums[index - 1], 0)  # add another element to sum or start a new one

        return max(nums)


# dp, bottom-up
# O(n), O(n)
class Solution:
    def maxSubArray(self, nums: list[int]) -> int:
        nums_len = len(nums)
        dp = [0 for _ in range(nums_len)]  # for cumulative sum
        dp[0] = nums[0]  # in case of all negative values cannot take "0" as a base

        for index in range(1, nums_len):
            dp[index] = max(dp[index - 1], 0) + nums[index]  # add another element to sum or start a new one
        
        return max(dp)





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
(Solution().search([-1, 0, 3, 5, 9, 12], -1), 0)
(Solution().search([-1, 0, 3, 5, 9, 12], 0), 1)
(Solution().search([-1, 0, 3, 5, 9, 12], 3), 2)
(Solution().search([-1, 0, 3, 5, 9, 12], 5), 3)
(Solution().search([-1, 0, 3, 5, 9, 12], 9), 4)
(Solution().search([-1, 0, 3, 5, 9, 12], 12), 5)
(Solution().search([-1, 0, 3, 5, 9, 12], 2), -1)


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

        while top <= bottom-  # two poionters to find the right row
            mid_row = (top + bottom) // 2  # find middle row index

            if (target >= matrix[mid_row][left] and
                    target <= matrix[mid_row][right]):  # if target row found
                break
            elif target < matrix[mid_row][left]:  # if target is less than the most left, choose top chunk
                bottom-= mid_row - 1
            else:  # if target is grater than the most right, choose bottom-chunk
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
(Solution().minEatingSpeed([3, 6, 7, 11], 8), 4)
(Solution().minEatingSpeed([30, 11, 23, 4, 20], 5), 30)
(Solution().minEatingSpeed([30, 11, 23, 4, 20], 6), 23)
(Solution().minEatingSpeed([312884470], 312884469), 2)
(Solution().minEatingSpeed([3], 2), 2)


# O(nlogn), O(1)
# Binary search
import numpy as np

class Solution:
    def minEatingSpeed(self, piles: list[int], hours: int) -> int:
        left = 1  # eat one banana per hour
        right = max(piles)  # eat `tallest pile` per hour

        while left <= right:
            middle = (left + right) // 2

            eat_time = sum(int(np.ceil(number / middle)) for number in piles)  # time to eat all bananas
            # eat_time = sum(int((number - 1) // middle) + 1 for number in piles)  # way to celi wihout numpy
            
            if eat_time <= hours:  # enought time to eat all bananas, check for a better solution
                current_eat_time = middle  # save current banana per hour time
                right = middle - 1  # eat slower
            else:  # not enough time to eat all bananas need to increase middle
                left = middle + 1  # eat faster
        
        return current_eat_time


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
(Solution().isValidSudoku([["5", "3", ".", ".", "7", ".", ".", ".", "."], ["6", ".", ".", "1", "9", "5", ".", ".", "."], [".", "9", "8", ".", ".", ".", ".", "6", "."], ["8", ".", ".", ".", "6", ".", ".", ".", "3"], ["4", ".", ".", "8", ".", "3", ".", ".", "1"], ["7", ".", ".", ".", "2", ".", ".", ".", "6"], [".", "6", ".", ".", ".", ".", "2", "8", "."], [".", ".", ".", "4", "1", "9", ".", ".", "5"], [".", ".", ".", ".", "8", ".", ".", "7", "9"]], ), True)
(Solution().isValidSudoku([["8", "3", ".", ".", "7", ".", ".", ".", "."], ["6", ".", ".", "1", "9", "5", ".", ".", "."], [".", "9", "8", ".", ".", ".", ".", "6", "."], ["8", ".", ".", ".", "6", ".", ".", ".", "3"], ["4", ".", ".", "8", ".", "3", ".", ".", "1"], ["7", ".", ".", ".", "2", ".", ".", ".", "6"], [".", "6", ".", ".", ".", ".", "2", "8", "."], [".", ".", ".", "4", "1", "9", ".", ".", "5"], [".", ".", ".", ".", "8", ".", ".", "7", "9"]]), False)
(Solution().isValidSudoku([[".", ".", ".", ".", "5", ".", ".", "1", "."], [".", "4", ".", "3", ".", ".", ".", ".", "."], [".", ".", ".", ".", ".", "3", ".", ".", "1"],  ["8", ".", ".", ".", ".", ".", ".", "2", "."],  [".", ".", "2", ".", "7", ".", ".", ".", "."],  [".", "1", "5", ".", ".", ".", ".", ".", "."],  [".", ".", ".", ".", ".", "2", ".", ".", "."],  [".", "2", ".", "9", ".", ".", ".", ".", "."],  [".", ".", "4", ".", ".", ".", ".", ".", "."]]), False)


# O(n2), O(n)
# using set, separate methods
class Solution:
    # validating sub-box
    def is_subbox_valid(self, row, col):
        seen = set()

        for r in range(3):
            for c in range(3):
                digit = self.board[row * 3 + r][col * 3 + c]

                if digit != ".":
                    if digit in seen:
                        return False
                    else:
                        seen.add(digit)
        return True

    # validating row
    def is_row_valid(self, row):
        seen = set()

        for digit in self.board[row]:
            if digit != ".":
                if digit in seen:
                    return False
                else:
                    seen.add(digit)
        
        return True

    # validating col
    def is_col_valid(self, col):
        seen = set()

        for row in range(self.rows):
            digit = self.board[row][col]

            if digit != ".":
                if digit in seen:
                    return False
                else:
                    seen.add(digit)
        
        return True

    def isValidSudoku(self, board):
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])

        # validating rows
        for row in range(self.rows):
            if not self.is_row_valid(row):
                return False

        # validating columns
        for col in range(self.cols):
            if not self.is_col_valid(col):
                return False

        # validating sub-boxes
        for row in range(0, 3):
            for col in range(0, 3):
                if not self.is_subbox_valid(row, col):
                    return False

        return True


# O(n2), O(n)
# using set
class Solution:
    def are_unique(self, numbers):
        seen = set()

        for char in numbers:
            if char != ".":
                if char in seen:
                    return False
                else:
                    seen.add(char)
        
        return True
        

    def isValidSudoku(self, board: list[list[str]]) -> bool:
        rows = range(len(board))
        cols = range(len(board[0]))

        # check if rows are legit
        for row in rows:
            if not self.are_unique(board[row]):
                return False
        
        # check if cols are legit
        for col in cols:
            current_col = []

            for row in rows:
                current_col.append(board[row][col])
            
            if not self.are_unique(current_col):
                return False

        # check if boxes are legit
        for row in range(3):
            for col in range(3):
                box = []

                for i in range(3):
                    for j in range(3):
                        box.append(board[row * 3 + i][col * 3 + j]) # 1 => 3, 4, 5 # 2 => 6, 7, 8
                
                if not self.are_unique(box):
                    return False

        return True


# O(n2), O(n2)
# built-in data structure - defaultdict(set) form collections
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
                        box_uniq[(row//3, col//3)].add(element)

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
"""
(Solution().checkInclusion("ab", "eidbaooo"), True)
(Solution().checkInclusion("ab", "eidboaoo"), False)
(Solution().checkInclusion("ccc", "cbac"), False)
(Solution().checkInclusion("ab", "a"), False)
(Solution().checkInclusion("abcdxabcde", "abcdeabcdx"), True)
(Solution().checkInclusion("adc", "dcda"), True)
(Solution().checkInclusion("hello", "ooolleoooleh"), False)
(Solution().checkInclusion("mart", "karma"), False)
(Solution().checkInclusion("abc", "ccccbbbbaaaa"), False)
 

# O(n), O(n)
# sliding window
class Solution:
    def checkInclusion(self, word1, word2):
        pattern = {}  # `pattern` stores the frequency of characters in `word1`
        window = {}  # `window` keeps track of character frequencies in the current window of `word2`
        matched = 0   # `matched` keeps track of how many characters in `pattern` have matching frequencies in `window`
        left = 0  # `left` marks the start of the sliding window

        for letter in word1:
            pattern[letter] = pattern.get(letter, 0) + 1

         # Iterate over `word2` with a sliding window approach
        for right, letter in enumerate(word2):
            window[letter] = window.get(letter, 0) + 1

            # If the current character is in the `pattern` and its frequency matches
            if (letter in pattern and
                    window[letter] == pattern[letter]):
                matched += 1

            # If all characters in the `pattern` are matched, return True
            if matched == len(pattern):
                return True

            # Check if the current window size is less than the size of `word1`
            if right - left + 1 != len(word1):
                continue  # Skip the rest of the loop until the window size matches `word1`

            # get left letter
            left_letter = word2[left]

            # If the character being removed is in the `pattern` and its frequency matches
            if (left_letter in pattern and
                    window[left_letter] == pattern[left_letter]):
                matched -= 1  # Decrement the matched count

            window[left_letter] -= 1  # remove a letter at the left pointer from the window
            left += 1

        return False


# O(n), O(n)
# sliding window, compare whole dicts when checking for inclusion
class Solution:
    def checkInclusion(self, word1, word2):
        counter1 = {}  # dict from word1
        window = {}  # window as dict
        left = 0  # left pointer
        
        for letter in word1:  # dict from word1
            counter1[letter] = counter1.get(letter, 0) + 1
        
        for right, letter in enumerate(word2):  # right pointer and right letter
            window[letter] = window.get(letter, 0) + 1  # add a letter to the window
            
            if len(word1) != right - left + 1:  # if window is not long enough
                continue
            elif counter1 == window:  # comapre dicts
                return True
            else:
                left_letter = word2[left]  # get left letter
                window[left_letter] -= 1  # remove it from the window
                left += 1

                if not window[left_letter]:  # if a key with no value exists, important for compraing dicts
                    window.pop(left_letter)  # remove that key

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
(Solution().minWindow("ADOBECODEBANC", "ABC"), "BANC")
(Solution().minWindow("a", "a"), "a")
(Solution().minWindow("a", "aa"), "")
(Solution().minWindow("a", "b"), "")
(Solution().minWindow("ab", "b"), "b")
(Solution().minWindow("bba", "ab"), "ba")
(Solution().minWindow("abc", "a"), "a")
(Solution().minWindow("jwsdrkqzrthzysvqqazpfnulprulwmwhiqlbcdduxktxepnurpmxegktzegxscfbusexvhruumrydhvvyjucpeyrkeodjvgdnkalutfoizoliliguckmikdtpryanedcqgpkzxahwrvgcdoxiylqjolahjawpfbilqunnvwlmtrqxfphgaxroltyuvumavuomodblbnzvequmfbuganwliwidrudoqtcgyeuattxxlrlruhzuwxieuqhnmndciikhoehhaeglxuerbfnafvebnmozbtdjdo", "qruzywfhkcbovewle"), "vequmfbuganwliwidrudoqtcgyeuattxxlrlruhzuwxieuqhnmndciik")


# O(n), O(n)
# sliding window
class Solution:
    def minWindow(self, word1, word2):
        # `pattern` stores the frequency of characters in `word2` (target)
        pattern = {}
        # `window` tracks character frequencies in the current sliding window of `word1`
        window = {}
        # `matched` counts how many characters in `pattern` have matching frequencies in `window`
        matched = 0
        # `left` is the start pointer for the sliding window
        left = 0
        # `shortest` will store the shortest valid substring that contains all characters of `word2`
        shortest = ""

        # Populate the `pattern` dictionary with character frequencies from `word2`
        for letter in word2:
            pattern[letter] = pattern.get(letter, 0) + 1

        # Iterate over `word1` to expand the sliding window
        for right, letter in enumerate(word1):
            # Add the current character to the `window` dictionary
            window[letter] = window.get(letter, 0) + 1

            # If the current character is in `pattern` and its frequency matches
            if (letter in pattern and
                    window[letter] == pattern[letter]):
                matched += 1  # Increment the matched count

            # Only proceed to shrinking the window if all characters in `pattern` are matched
            if matched != len(pattern):
                continue

            # Shrink the window from the left while keeping it valid
            left_letter = word1[left]
            while (left_letter not in pattern or  # Ignore characters not in the pattern
                   window[left_letter] > pattern[left_letter]):  # Ignore extra occurrences of a valid character
                window[left_letter] -= 1  # Remove the leftmost character from the window
                left += 1  # Move the left pointer forward
                left_letter = word1[left]  # Update the left character

            # Check if the current window is the smallest valid substring so far
            if (not shortest or  # If `shortest` is empty
                    right - left + 1 < len(shortest)):  # If the current window is smaller
                shortest = word1[left: right + 1]  # Update the shortest substring

            # Prepare for the next iteration by removing the leftmost character
            # (even if it is significant) and reducing the `matched` count
            window[left_letter] -= 1
            matched -= 1
            left += 1

        # Return the shortest valid substring or an empty string if no such substring exists
        return shortest


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
        pattern_counter = self.counter(pattern)
        window_count = {}
        min_window = s + ":"

        while r < len(s):
            # update curr window dict if element is significant
            if s[r] in pattern_counter.keys():
                window_count[s[r]] = window_count.get(s[r], 0) + 1

            # while all elements from window are found in pattern
            # Case where need to trim window from left (Solution().minWindow("bba", "ab"), "ba") is "bba"
            while self.is_sub(window_count, pattern_counter):
                # update minimum window
                min_window = min(min_window, s[l: r + 1], key=len)
        
                # if l is a significant index
                if s[l] in pattern_counter:
                    # update current window dict
                    window_count[s[l]] -= 1

                l += 1

            r += 1

        return "" if min_window == s + ":" else min_window

 
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
        min_wind = s + ":"

        for l in range(len(s)):
            for r in range(len(s)):
                s_dict = self.counter(s[l:r + 1])

                if self.dict_compare(s_dict, t_dict):
                    min_wind = min(min_wind, s[l: r + 1], key=len)

        return "" if min_wind == s + ":" else min_wind





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
(Solution().maxSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3), [3, 3, 5, 5, 6, 7])
(Solution().maxSlidingWindow([1], 1), [1])
(Solution().maxSlidingWindow([7, 2, 4], 2), [7, 4])
(Solution().maxSlidingWindow([1, 3, 1, 2, 0, 5], 3), [3, 3, 2, 5])


# O(n), O(n)
# sliding window as deque of indexes, Monotonic Decreasing Queue
from collections import deque

class Solution:
    def maxSlidingWindow(self, numbers, window_size):
        window = deque()  # sliding window as a deque
        left = 0  # left pointer
        max_list = []  # stores the maximum values for each sliding window.

        for right, number in enumerate(numbers):
            # Remove elements from the back of the deque if they are less than or equal to
            # the current number, as they cannot be the maximum in the current or any future window.
            while (window and
                   numbers[window[-1]] <= number):
                window.pop()

            # Add the index of the current number to the deque.
            window.append(right)

            # Remove elements from the front of the deque if they are outside the current window.
            while (window and
                   window[0] < left):
                window.popleft()

            # When the sliding window reaches the required size, record the maximum value.
            if right - left + 1 == window_size:
                # The maximum value in the current window is at the index `window[0]`.
                max_list.append(numbers[window[0]])
                # Slide the window forward by incrementing `left`.
                left += 1

        return max_list


# window as a list of indexes
class Solution:
    def maxSlidingWindow(self, numbers, window_size):
        window = []  # sliding window as a list
        left = 0  # left pointer
        current_max = []  # array with max value from each sliding window

        for right in range(len(numbers)):  # right pointer
            number = numbers[right]

            while window and window[0] < left:  # remove left out of bounds indexes
                window.pop(0)

            while window and numbers[window[-1]] <= number:  # remove right indexes with numbers less than current number
                    window.pop()

            if window and number > numbers[window[0]]:  # if number is greater than the left most number
                window.insert(0, right) # append it left
            else:  
                window.append(right)  # append it right

            if right - left + 1 == window_size:  # if the window is the right size
                current_max.append(numbers[window[0]])  # get the left (max) value
                left += 1  # update left pointer

        return current_max


# O(n2)
class Solution:
    def maxSlidingWindow(self, numbers: list[int], k: int) -> list[int]:
        left = 0
        sol = []

        for right in range(k - 1, len(numbers)):
            sol.append(max(numbers[left: right + 1]))

            left += 1
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
(Solution().evalRPN(["2", "1", "+", "3", "*"]), 9)
(Solution().evalRPN(["4", "13", "5", "/", "+"]), 6)
(Solution().evalRPN(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]), 22)
(Solution().evalRPN(["18"]), 18)


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
(Solution().dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73]), [1, 1, 4, 2, 1, 1, 0, 0])
(Solution().dailyTemperatures([30, 40, 50, 60]), [1, 1, 1, 0])
(Solution().dailyTemperatures([30, 60, 90]), [1, 1, 0])


# O(n), O(n)
class Solution:
    def dailyTemperatures(self, temps: list[int]) -> list[int]:
        stack = []
        days_delta = [0] * len(temps)

        for current_day, temp in enumerate(temps):
            while (stack and 
                   temp > stack[-1][1]):  # check if temp is lower
                
                previous_day, _ = stack.pop()  # index of the last day in stack
                days_delta[previous_day] = current_day - previous_day

            stack.append((current_day, temp))  # apped new temp

        return days_delta


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
(Solution().carFleet(12, [10, 8, 0, 5, 3], [2, 4, 1, 1, 3]), 3)
(Solution().carFleet(10, [3], [3]), 1)
(Solution().carFleet(100, [0, 2, 4], [4, 2, 1]), 1)
(Solution().carFleet(10, [0, 4, 2], [2, 1, 3]), 1)

# (12 - 10) / 2 = 1;
# (12 - 8) / 4 = 1;

# (12 - 8) / 1 = 4;
# (12 - 3) / 3 = 3;

# (12 - 0) / 1 = 12;


# ((0, 2), (2, 3), (4, 1)) 10
#     5   <=?  2.66
#              2.66  <=? 6
#     5         6       6


class Solution:
    def carFleet(self, target: int, position: list[int], speed: list[int]) -> int:
        cars = sorted(zip(position, speed))
        fleets = len(position)
        distances = []
        
        for car in cars[::-1]:
            pos, speed = car
            distance = (target - pos) / speed

            if (distances and 
                distance <= distances[-1]):
                fleets -= 1
                # if the time to tartet is lower than the top stack time to target
                # then the highter time should remain to possibly catch next cars
            else:
                distances.append(distance)

        return fleets


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
        subset = []  # current subset
        subset_list = []  # solution

        def dfs(level):
            if level == len(nums):  # target level reached
                subset_list.append(subset.copy())  # push subset to subset_list
                return

            subset.append(nums[level])  # Include the current element in the subset
            dfs(level + 1)  # Explore the path with the current element
            subset.pop()  # Backtrack by removing the current element from the subset
            dfs(level + 1)  # Explore the path without including the current element

        dfs(0)  # start dfs with level = 0
    
        return subset_list


# "dfs" and "subset" methods inside a class
class Solution:
    def __init__(self) -> None:
        self.subset = []
        self.subset_list = []

    def subsets(self, nums: list[int]) -> list[list[int]]:
        self.nums = nums
        self.dfs(0)
        
        return self.subset_list

    def dfs(self, level):
        if level == len(self.nums):
            self.subset_list.append(self.subset.copy())
            return
        
        self.subset.append(self.nums[level])
        self.dfs(level + 1)            
        self.subset.pop()
        self.dfs(level + 1)
        




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

            subset.append(nums[index])  # Include the current element in the subset
            dfs(index + 1)  # Explore the path with the current element
            subset.pop()  # Backtrack by removing the current element from the subset

            # If num at the current index (that was poped previously) is the same as
            # the num at next index skip it.
            while (index + 1 < len(nums) and
                    nums[index] == nums[index + 1]):
                index += 1

            dfs(index + 1)  # Explore the path without including the current element

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
        combination = []
        combination_list = []

        def dfs(index):
            combination_sum = sum(combination)

            if combination_sum == target:  # target sum reached
                combination_list.append(combination.copy())  # push combination to combination list
                return
            elif (combination_sum > target or # if value is too large or
                  index == len(candidates)):  # index out of bounds
                return
                        
            combination.append(candidates[index])  # Include the current element in the combinatioin
            dfs(index)  # Explore the path with the current element
            combination.pop()  # Backtrack by removing the current element from the subset
            dfs(index + 1)  # Explore the path without including the current element

        dfs(0)

        return combination_list


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
(Solution().combinationSum2([10, 1, 2, 7, 6, 1, 5], 8), [[1, 1, 6], [1, 2, 5], [1, 7], [2, 6]])
(Solution().combinationSum2([2, 5, 2, 1, 2], 5), [[1, 2, 2], [5]])
(Solution().combinationSum2([6], 6), [[6]])


class Solution:
    def combinationSum2(self, candidates: list[int], target: int) -> list[list[int]]:
        candidates.sort()  # sort needed to skip same values
        combination = []
        combination_list = []

        def dfs(index):
            combination_sum = sum(combination)

            if sum(combination) == target:  # if sum is achieved
                combination_list.append(combination.copy())
                return
            elif (combination_sum > target or  # index boundry and sum boundry
                    index == len(candidates)):
                return

            combination.append(candidates[index])  # (left node) append candidate
            dfs(index + 1)

            while (index + 1 < len(candidates) and  # skip same values
                   candidates[index] == candidates[index + 1]):
                index += 1

            combination.pop()  # (right node) skip candidate
            dfs(index + 1)

        dfs(0)

        return combination_list


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
            
            while (index + 1 < len(nums) and 
                   nums[index + 1] == nums[index]):
                index += 1
            
            dfs(index + 1, value)

        dfs(0, 0)

        return combination_list





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
(Solution().permute([1, 2, 3]), [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]])
(Solution().permute([0, 1]), [[0, 1], [1, 0]])
(Solution().permute([1]), [[1]])


class Solution:
    def permute(self, nums: list[int]) -> list[list[int]]:
        pernutation_list = []

        def dfs(permutatioin, nums):
            if not nums:  # in no nums left
                pernutation_list.append(permutatioin)
                return
            
            for index in range(len(nums)):  # for every number in numbers
                # include this number to 'permutations' and exclude it from numbers
                dfs(permutatioin + [nums[index]], 
                    nums[:index] + nums[index + 1 :])

        dfs([], nums)
        return pernutation_list





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
(Solution().exist([["C", "A", "A"], ["A", "A", "A"], ["B", "C", "D"]], "AAB"), True)
(Solution().exist([["C", "A", "A"], ["A", "A", "A"], ["B", "C", "D"]], "AACA"), True)
(Solution().exist([["A", "A"]], "AAA"), False)
(Solution().exist([["A", "B", "C", "E"], ["S", "F", "E", "S"], ["A", "D", "E", "E"]], "ABCEFSADEESE"), True)
(Solution().exist([["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]], "AB"), True)
(Solution().exist([["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]], "AZ"), False)
(Solution().exist([["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]], "ABFS"), True)
(Solution().exist([["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]], "ABCCED"), True)
(Solution().exist([["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]], "SEE"), True)
(Solution().exist([["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]], "ABCB"), False)

# ["A", "B", "C", "E"], 
# ["S", "F", "E", "S"], 
# ["A", "D", "E", "E"]


class Solution:
    def exist(self, board, word):
        rows = len(board)
        cols = len(board[0])
        tabu = set()  # set of visited (forbidden) cells
        
        def dfs(row, col, index):
            if index == len(word):
                return True
            
            tabu.add((row, col))  # add cell pair to tabu set

            # check up, down, left, right for neighbouns
            if (col + 1 < cols and  # check if out of bounds and
                not (row, col + 1) in tabu and  # if cell is not in tabo set
                word[index] == board[row][col + 1] and  # Check if the current position matches the word's character
                dfs(row, col + 1, index + 1)):  # Switch to that letter and check its neighbors
                return True

            if (row + 1 < rows and
                not (row + 1, col) in tabu and
                word[index] == board[row + 1][col] and
                dfs(row + 1, col, index + 1)):
                return True
        
            if (col - 1 >= 0 and
                not (row, col - 1) in tabu and
                word[index] == board[row][col - 1] and
                dfs(row, col - 1, index + 1)):
                return True

            if (row - 1 >= 0 and
                not (row - 1, col) in tabu and
                word[index] == board[row - 1][col] and
                dfs(row - 1, col, index + 1)):
                return True
            
            # Backtrack: remove from tabu
            tabu.remove((row, col))
            
        for row in range(rows):
            for col in range(cols):
                if word[0] == board[row][col]:  # if first letter matches
                    if dfs(row, col, 1):  # check its heighbors
                        return True

        return False  # if word was not found return False




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
(Solution().partition("a"), [["a"]])
(Solution().partition("aa"), [['a', 'a'], ['aa']])
(Solution().partition("ab"), [["a", "b"]])
(Solution().partition("aaa"), [['a', 'a', 'a'], ['a', 'aa'], ['aa', 'a'], ['aaa']])
(Solution().partition("aab"), [["a", "a", "b"], ["aa", "b"]])
(Solution().partition("aba"), [["a", "b", "a"], ["aba"]])


# passing "word" in dfs(), "palindrome" as a side effect
class Solution:
    def partition(self, word):
        palindrome = []  # This will track the current partition
        palindrome_list = []  # This will store all valid palindrome partitions

        def dfs(word):
            if not word:  # if word is empty that means all letters folded into palindrom
                palindrome_list.append(palindrome.copy())
                return

            for index in range(len(word)):  # for every index in "word"
                substring = word[ : index + 1] # Current substring to check

                if substring == substring[::-1]:  # if substring is a palindrme
                    palindrome.append(substring)  # Add it to the current partition
                    dfs(word[index + 1 :])  # Explore the path with the current palindrome and look for the palindrome in the next part of the "word"
                    palindrome.pop()  # Backtrack by removing the last added palindrome

        dfs(word)  # Start DFS with "word"
        return palindrome_list


# passing current partition and "word" in dfs()
class Solution:
    def partition(self, word):
        palindrome_list = []

        def dfs(palindrome, word):
            if not word:
                palindrome_list.append(palindrome.copy())
                return
            
            for index in range(len(word)):
                substring = word[: index + 1]

                if substring == substring[::-1]:
                    dfs(palindrome + [substring], word[index + 1 :])

        dfs([], word)
        
        return palindrome_list


# passing indexes (instead of the "word") in dfs()
class Solution:
    def partition(self, word):
        palindrome = []  # This will track the current partition
        palindrome_list = []  # This will store all valid palindrome partitions

        def dfs(start):
            if start == len(word):  # If we reach the end of the word
                palindrome_list.append(palindrome.copy())  # Add the current partition to the list
                return

            for end in range(start + 1, len(word) + 1):
                substring = word[start:end]  # Current substring to check
                
                if substring == substring[::-1]:  # Check if the substring is a palindrome
                    palindrome.append(substring)  # Add it to the current partition
                    dfs(end)  # Recur for the next part of the word
                    palindrome.pop()  # Backtrack by removing the last added palindrome

        dfs(0)  # Start DFS from index 0
        return palindrome_list







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
(Solution().letterCombinations("2"), ["a", "b", "c"])
(Solution().letterCombinations("23"), ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"])
(Solution().letterCombinations(""), [])


# passing index to dfs, combination as a shared variable (side effect)
class Solution:
    def letterCombinations(self, digits: str) -> list[str]:
        if not digits:
            return []
        
        combination = []
        combination_list = []
        digit_to_letter = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz"
        }

        def dfs(index):
            if index == len(digits):
                combination_list.append("".join(combination))
                return

            for letter in digit_to_letter[digits[index]]:
                combination.append(letter)
                dfs(index + 1)
                combination.pop()

        dfs(0)

        return combination_list



# passing index and combination to dfs
class Solution:
    def letterCombinations(self, digits):
        if not digits:
            return []

        combination_list = []
        digit_to_letter = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz"
        }

        def dfs(index, combination):
            if index == len(digits):
                combination_list.append(combination)
                return

            for letter in digit_to_letter[digits[index]]:
                dfs(index + 1, combination + letter)

        dfs(0, "")

        return combination_list





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
# elements with the same num are on the same diagonal
#       0   1   2
#   0   0   1   2
#   1   -1  0   1
#   2   -2  -1  0

# elements with the same num are on the same anti-diagonal
#       0   1   2
#   0   0   1   2
#   1   1   2   3
#   2   2   3   4
(Solution().solveNQueens(4), [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]])
(Solution().solveNQueens(1), [["Q"]])


class Solution:
    def solveNQueens(self, n: int) -> list[list[str]]:
        board_list = []
        board = [["." for _ in range(n)] for _ in range(n)]  # initiate a board
        tabu_col = set()
        tabu_diag = set()  # for each diagonal (col_ind - row_ind) = const
        tabu_adiag = set()  # for each aiti-diagonal (con_ind + row_ind) = const

        def dfs(row):
            if row == n:  # if all rows are filled with Queens
                joined_board = ["".join(row) for row in board]  # ['.', 'Q', '.', '.'] => ['.Q..']
                board_list.append(joined_board)
                return

            for col in range(n):
                # if there is another Queen in the same diagonal or the same col
                if ((row - col) in tabu_diag or 
                    (row + col) in tabu_adiag or 
                    col in tabu_col):
                    continue
                
                # update tabu and board
                tabu_col.add(col)
                tabu_diag.add(row - col)
                tabu_adiag.add(row + col)
                board[row][col] = "Q"

                # check another row
                dfs(row + 1)

                # backtrack
                tabu_col.remove(col)
                tabu_diag.remove(row - col)
                tabu_adiag.remove(row + col)
                board[row][col] = "."

        dfs(0)
        return board_list





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





# Generate Parentheses
# https://leetcode.com/problems/generate-parentheses/description/
"""
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

Example 1:

Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]

Example 2:

Input: n = 1
Output: ["()"]
"""
(Solution().generateParenthesis(1), ["()"])
(Solution().generateParenthesis(2), ["(())", "()()"])
(Solution().generateParenthesis(3), ["((()))", "(()())", "(())()", "()(())", "()()()"])


class Solution:
    def generateParenthesis(self, num: int) -> list[str]:
        parenthesis = []  # current parenthesis sequence
        parenthesis_list = []  # list of parenthesis sequences

        def dfs(open, close):
            if open + close == 2 * num:  # if all opening and closing parenthesis are used
                parenthesis_list.append("".join(parenthesis))  # append current sequence
                return

            if open < num:  # not all "(" have been used
                parenthesis.append("(")
                dfs(open + 1, close)  # check this branch
                parenthesis.pop()  # backtrack

            if close < open:  # the number of ")" must not be greater than "("
                parenthesis.append(")")
                dfs(open, close + 1)  # check this branch
                parenthesis.pop()  # backtrack

        dfs(0, 0)  # start with no parenthesis

        return parenthesis_list





# Min Cost Climbing Stairs
# https://leetcode.com/problems/min-cost-climbing-stairs/
"""
You are given an integer array cost where cost[i] is the cost of ith step on a staircase. Once you pay the cost, you can either climb one or two steps.

You can either start from the step with index 0, or the step with index 1.

Return the minimum cost to reach the top of the floor.

Example 1:

Input: cost = [10,15,20]
Output: 15
Explanation: You will start at index 1.
- Pay 15 and climb two steps to reach the top.
The total cost is 15.

Example 2:

Input: cost = [1,100,1,1,1,100,1,1,100,1]
Output: 6
Explanation: You will start at index 0.
- Pay 1 and climb two steps to reach index 2.
- Pay 1 and climb two steps to reach index 4.
- Pay 1 and climb two steps to reach index 6.
- Pay 1 and climb one step to reach index 7.
- Pay 1 and climb two steps to reach index 9.
- Pay 1 and climb one step to reach the top.
The total cost is 6.
"""
(Solution().minCostClimbingStairs([10, 15, 20]), 15)
(Solution().minCostClimbingStairs([1, 100, 1, 1, 1, 100, 1, 1, 100, 1]), 6)


# draft
# 1, 100, 1+1=2, 2+1=3, 2+1=3, 103, 3+1=4, 4+1=5, 104, 6

# dp, bottom-up
# O(n), O(n)
class Solution:
    def minCostClimbingStairs(self, cost: list[int]) -> int:
        for index in range(2, len(cost)):
            cost[index] = min(cost[index - 1], cost[index - 2]) + cost[index]  # min

        return min(cost[-2:])


# dp, bottom-up
# O(n), O(1)
class Solution:
    def minCostClimbingStairs(self, cost: list[int]) -> int:
        a = cost[0]
        b = cost[1]

        for index in range(2, len(cost)):
            a, b = b, min(a, b) + cost[index]

        return min(a, b)





# Decode Ways
# https://leetcode.com/problems/decode-ways/description/
"""
You have intercepted a secret message encoded as a string of numbers. The message is decoded via the following mapping:

"1" -> 'A'

"2" -> 'B'

...

"25" -> 'Y'

"26" -> 'Z'

However, while decoding the message, you realize that there are many different ways you can decode the message because some codes are contained in other codes ("2" and "5" vs "25").

For example, "11106" can be decoded into:

"AAJF" with the grouping (1, 1, 10, 6)
"KJF" with the grouping (11, 10, 6)
The grouping (1, 11, 06) is invalid because "06" is not a valid code (only "6" is valid).
Note: there may be strings that are impossible to decode.

Given a string s containing only digits, return the number of ways to decode it. If the entire string cannot be decoded in any valid way, return 0.

Example 1:

Input: s = "12"

Output: 2

Explanation:

"12" could be decoded as "AB" (1 2) or "L" (12).


Example 2:

Input: s = "226"

Output: 3

Explanation:

"226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).


Example 3:

Input: s = "06"

Output: 0

Explanation:

"06" cannot be mapped to "F" because of the leading zero ("6" is different from "06"). In this case, the string is not a valid encoding, so return 0.
"""
(Solution().numDecodings("226"), 3)
(Solution().numDecodings("12"), 2)
(Solution().numDecodings("06"), 0)
(Solution().numDecodings("0"), 0)
(Solution().numDecodings(""), 0)
(Solution().numDecodings("2101"), 1)
(Solution().numDecodings("111111111111111111111111111111111111111111111"), 1836311903)


# draft
# "226"
# 2 2 6, 22 6, 2, 26

# 2101
# 2 10 1

# Bottom-up
# O(n), O(n)
class Solution:
    def numDecodings(self, code: str) -> int:
        if not code:  # ifcode is empty
            return 0

        dp = {len(code): 1}  # assume "code" is a prefix and everything after would be foleded into "1" possible path

        for index in range(len(code))[::-1]:  # check every number in reversed order
            if int(code[index]):  # check if number is not statring with 0
                # one digit number case
                dp[index] = dp.get(index, 0) + dp.get(index + 1, 0)  # continue legit one digit number path

                # two digits number case
                if (index + 1 < len(code) and  # if index in bounds
                    int(code[index : index + 2]) <= 26):  # if two digit number between <10, 27)
                    dp[index] = dp.get(index, 0) + dp.get(index + 2, 0)  # continue legit two digit number path

        return dp.get(0, 0)  # get first value from the dictionary or if code is not legit return 0


# Top-down with memoization
# O(n), O(n)
class Solution:
    def numDecodings(self, code: str) -> int:
        if not code:  # if "code" is empty
            return 0
        
        dp = {len(code): 1}  # assume "code" is a prefix and everything after would be foleded into "1" possible path

        def dfs(index):
            if index in dp:  # if memoized
                return dp[index]  # Return memoized result if already computed.
            
            if code[index] == "0":  # check if number is statring with 0
                return 0  # inwalid number
            
            # one digit number case
            dp[index] = dp.get(index, 0) + dfs(index + 1)  # Proceed to decode the next number.
        
            # two digits number case
            if (index + 1 < len(code) and  # check if second digit within bounds
                int(code[index : index + 2]) <= 26):  # if two digit number between <10, 27)
                dp[index] = dp.get(index, 0) + dfs(index + 2)  # Add the result of two-digit decoding.

            return dp[index]  # Return the result for this index.

        return dfs(0)  # Start decoding from the first index.


# DFS, slow, not for numDecodings("111111111111111111111111111111111111111111111")
# O(2^n) O(n2)
class Solution:
    def numDecodings(self, word: str) -> int:
        if not word or not int(word[0]):
            return 0

        decoded = []
        decoded_list = []

        def dfs(index):
            if index == len(word):
                decoded_list.append(decoded.copy())
                return

            if (index + 1 < len(word)
                and int(word[index + 1])
                    or index == len(word) - 1):

                if int(word[index]) != 0:
                    decoded.append(int(word[index]))
                    dfs(index + 1)
                    decoded.pop()

            if (index + 2 < len(word)
                and int(word[index + 2])
                    or index == len(word) - 2):

                if (index + 1 < len(word) and
                        int(word[index: index + 2]) <= 26):
                    decoded.append(int(word[index: index + 2]))
                    dfs(index + 2)
                    decoded.pop()

        dfs(0)

        return len(decoded_list)





# Invert Binary Tree
# https://leetcode.com/problems/invert-binary-tree/description/
"""
Given the root of a binary tree, invert the tree, and return its root.

Example 1:

Input: root = [4,2,7,1,3,6,9]
    __4__
   /     \
  2       7
 / \     / \
1   3   6   9

Output: [4,7,2,9,6,3,1]
    __4__
   /     \
  7       2
 / \     / \
9   6   3   1

Example 2:

Input: root = [2,1,3]
  2
 / \
1   3

Output: [2,3,1]
  2
 / \
3   1

Example 3:

Input: root = []
Output: []
"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


from collections import deque

# Function to create a binary tree from a list (level-order traversal)
def build_tree_from_list(node_list, node_type=TreeNode):
    if not node_list:
        return

    if type(node_list) == int:  # case when the node list is a single value
        node_list = [node_list]

    root = node_type(node_list[0])  # Create the root node
    queue = deque([root])
    index = 1

    # Process the list and construct the tree
    while index < len(node_list):
        node = queue.popleft()

        # Assign the left child if available
        if index < len(node_list) and node_list[index] != None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] != None:
            node.right = node_type(node_list[index])
            queue.append(node.right)
        index += 1

    return root


from binarytree import Node

tree_from_list = build_tree_from_list([4, 2, 7, 1, 3, 6, 9], TreeNode)  # use TreeNode or Node
print(tree_from_list)
#     __4__
#    /     \
#   2       7
#  / \     / \
# 1   3   6   9


tree_from_list2 = build_tree_from_list([3, 9, 20, None, None, 15, 7], Node)
print(tree_from_list2)
#   3___
#  /    \
# 9     _20
#      /   \
#     15    7


from collections import deque

# Function create a list from a binary tree in level-order (breadth-first traversal)
# def level_order_traversal(root: TreeNode | Node | None):
def level_order_traversal(root):
    if not root:
        return
    
    node_list = []
    queue = deque([root])  # Initialize the queue with the root node

    while queue:
        node = queue.popleft()  # Pop the current node
        node_list.append(node.val)  # add its value to node list
        
        if node.left:  # Add left child to the queue if it exists
            queue.append(node.left)

        if node.right:  # Add right child to the queue if it exists
            queue.append(node.right)

    return node_list


level_order_traversal(tree_from_list)
# [4, 2, 7, 1, 3, 6, 9]

level_order_traversal(tree_from_list2)
# [3, 9, 20, 15, 7]




from binarytree import Node  # or use TreeNode

(level_order_traversal(
    Solution().invertTree(
        build_tree_from_list(
            [4, 2, 7, 1, 3, 6, 9], TreeNode))), [4, 7, 2, 9, 6, 3, 1])
# don't mix Node or TreeNode in one Solution

(Solution().invertTree([4, 2, 7, 1, 3, 6, 9]), [4, 7, 2, 9, 6, 3, 1])
(Solution().invertTree([2, 1, 3]), [2, 3, 1])
(Solution().invertTree([]), [])


# O(n), O(n)
# dfs, recursion
class Solution:
    def invertTree(self, root: TreeNode | None) -> TreeNode | None:
        if not root:
            return

        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        
        return root


# O(n), O(n)
# dfs, recursion
class Solution:
    def invertTree(self, root: TreeNode | None) -> TreeNode | None:
        if root:
            root.left, root.right = root.right, root.left
            self.invertTree(root.left) 
            self.invertTree(root.right)
        
        return root


# O(n), O(n)
# dfs, recursion
class Solution:
    def invertTree(self, root: TreeNode | None) -> TreeNode | None:
        if root:
            root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        
            return root


# O(n), O(n)
# dfs, stack, iteration
class Solution:
    def invertTree(self, root: TreeNode | None) -> TreeNode | None:
        if not root:
            return None
        
        stack = [root]

        while stack:
            node = stack.pop()
            node.left, node.right = node.right, node.left
            
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        
        return root


from collections import deque
# O(n), O(n)
# bfs, deque, iteration
class Solution:
    def invertTree(self, root: TreeNode | None) -> TreeNode | None:
        if not root:
            return None
        
        queue = deque([root])

        while queue:
            node = queue.popleft()
            node.left, node.right = node.right, node.left
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        return root


# Maximum Depth of Binary Tree
# https://leetcode.com/problems/maximum-depth-of-binary-tree/description/
"""
Given the root of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

Example 1:

Input: root = [3,9,20,null,null,15,7]
Output: 3

Example 2:

Input: root = [1,null,2]
Output: 2
"""
(Solution().maxDepth(build_tree_from_list([3, 9, 20, None, None, 15, 7], Node)), 3)
(Solution().maxDepth(build_tree_from_list([1, None, 2], Node)), 2)


from binarytree import Node
from collections import deque


class TreeNode:
    def __init__(self, val=1, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Function to create a binary tree from a list (level-order traversal)
def build_tree_from_list(node_list, node_type=TreeNode):
    if not node_list:
        return

    if type(node_list) == int:  # case when the node list is a single value
        node_list = [node_list]

    root = node_type(node_list[0])  # Create the root node
    queue = deque([root])
    index = 1

    # Process the list and construct the tree
    while index < len(node_list):
        node = queue.popleft()

        # Assign the left child if available
        if index < len(node_list) and node_list[index] != None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] != None:
            node.right = node_type(node_list[index])
            queue.append(node.right)
        index += 1

    return root


tree_from_list = build_tree_from_list(
    [3, 9, 20, None, None, 15, 7], Node)  # use TreeNode or Node
print(tree_from_list)

 
# O(n), O(n)
# dfs, recursion
class Solution:
    def maxDepth(self, root: TreeNode | None) -> int:
        if not root:
            return 0
        
        return max(
            self.maxDepth(root.left),  # left branch depth
            self.maxDepth(root.right)  # right branch depth
            ) + 1


# O(n), O(n)
# dfs, recursion, explict dfs function
class Solution:
    def maxDepth(self, root: TreeNode | None) -> int:
        def dfs(node):
            if not node:
                return 0
        
            left = dfs(node.left)  # left branch depth
            right = dfs(node.right)  # right branch depth
            
            return max(left, right) + 1

        return dfs(root)


# O(n), O(n)
# bfs, iteration, deque, level order traversal
from collections import deque

class Solution:
    def maxDepth(self, root: TreeNode | None) -> int:
        if not root:
            return 0
        
        depth = 0
        queue = deque([root])

        while queue:
            depth += 1
            current_queue_len = len(queue)

            for _ in range(current_queue_len):
                node = queue.popleft()

                if node.left:
                    queue.append(node.left)
                
                if node.right:
                    queue.append(node.right)
        
        return depth


# O(n), O(n)
# bfs, deque, iteration
from collections import deque

class Solution:
    def maxDepth(self, root: TreeNode | None) -> int:
        if not root:
            return 0
        
        depth = 1
        deq = deque([(1, root)])

        while deq:
            level, node = deq.popleft()
            depth = max(depth, level)
            
            if node.left:
                deq.append((level + 1, node.left))
            if node.right:
                deq.append((level + 1, node.right))
        
        return depth
    

# O(n), O(n)
# dfs, iteration, stack, pre-order traversal
class Solution:
    def maxDepth(self, root: TreeNode | None) -> int:
        if not root:
            return 0

        stack = [(root, 1)]
        max_depth = 1

        while stack:
            node, depth = stack.pop()
            max_depth = max(max_depth, depth)

            if node.left:
                stack.append((node.left, depth + 1))
            
            if node.right:
                stack.append((node.right, depth + 1))

        return max_depth

print(Solution().maxDepth(
    build_tree_from_list(
        [3, 9, 20, None, None, 15, 7], Node)), 3)





# Diameter of Binary Tree
# https://leetcode.com/problems/diameter-of-binary-tree/description/
"""
Given the root of a binary tree, return the length of the diameter of the tree.

The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.

The length of a path between two nodes is represented by the number of edges between them.

Example 1:

Input: root = [1,2,3,4,5]

    __1
   /   \
  2     3
 / \
4   5

Output: 3
Explanation: 3 is the length of the path [4,2,1,3] or [5,2,1,3].

Example 2:

Input: root = [1,2]
Output: 1
"""
(Solution().diameterOfBinaryTree(build_tree_from_list([1, 2, 3, 4, 5], Node)), 3)


from binarytree import Node
from collections import deque


class TreeNode:
    def __init__(self, val=1, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Function to create a binary tree from a list (level-order traversal)
def build_tree_from_list(node_list, node_type=TreeNode):
    if not node_list:
        return

    if type(node_list) == int:  # case when the node list is a single value
        node_list = [node_list]

    root = node_type(node_list[0])  # Create the root node
    queue = deque([root])
    index = 1

    # Process the list and construct the tree
    while index < len(node_list):
        node = queue.popleft()

        # Assign the left child if available
        if index < len(node_list) and node_list[index] != None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] != None:
            node.right = node_type(node_list[index])
            queue.append(node.right)
        index += 1

    return root


build_tree_from_list([1, 2, 3, 4, 5], Node)


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# O(n), O(n)
# dfs, recursion
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode | None) -> int:
        self.diameter = 0
        
        def dfs(node):
            if not node:
                return 0
        
            left = dfs(node.left)  # left branch depth
            right = dfs(node.right)  # right branch depth
            self.diameter = max(self.diameter, left + right)  # updates diameter with current node as a root
            
            return max(left, right) + 1  # (height)  current node max depth

        dfs(root)

        return self.diameter





# Balanced Binary Tree
# https://leetcode.com/problems/balanced-binary-tree/description/
"""
Given a binary tree, determine if it is 
height-balanced

Example 1:

Input: root = [3,9,20,null,null,15,7]

  3___
 /    \
9     _20
     /   \
    15    7

Output: true

Example 2:

Input: root = [1,2,2,3,3,null,null,4,4]

        __1
       /   \
    __2     2
   /   \
  3     3
 / \
4   4

Output: false

Example 3:

Input: root = []
Output: true
"""
(Solution().isBalanced(build_tree_from_list([3, 9, 20, None, None, 15, 7], Node)), True)
(Solution().isBalanced(build_tree_from_list([1, 2, 2, 3, 3, None, None, 4, 4], Node)), False)
(Solution().isBalanced(build_tree_from_list([1, 2, 2, 3, None, None, 3, 4, None, None, 4], Node)), False)


from binarytree import Node
from collections import deque


class TreeNode:
    def __init__(self, val=1, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Function to create a binary tree from a list (level-order traversal)
def build_tree_from_list(node_list, node_type=TreeNode):
    if not node_list:
        return

    if type(node_list) == int:  # case when the node list is a single value
        node_list = [node_list]

    root = node_type(node_list[0])  # Create the root node
    queue = deque([root])
    index = 1

    # Process the list and construct the tree
    while index < len(node_list):
        node = queue.popleft()

        # Assign the left child if available
        if index < len(node_list) and node_list[index] != None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] != None:
            node.right = node_type(node_list[index])
            queue.append(node.right)
        index += 1

    return root

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# dfs, recursive
# O(n), O(n)
class Solution:
    def isBalanced(self, root: TreeNode | None) -> bool:
        self.is_balanced = True  # default value for balanced tree
        
        def dfs(node):
            if not node:
                return 0

            left = dfs(node.left)  # left branch depth
            right = dfs(node.right)  # right branch depth

            if abs(left - right) > 1:  # if deep of the two subtrees differs more than by 1
                self.is_balanced = False  # then tree in not balanced
                return -1  # early return

            return max(left, right) + 1  # the depth of the current node

        dfs(root)  # run dfs

        return self.is_balanced





# Same Tree
# https://leetcode.com/problems/same-tree/description/
"""
Given the roots of two binary trees p and q, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

Example 1:

Input: p = [1,2,3], q = [1,2,3]
Output: true

Example 2:

Input: p = [1,2], q = [1,null,2]
Output: false
Example 3:

Input: p = [1,2,1], q = [1,1,2]
Output: false
"""
(Solution().isSameTree(build_tree_from_list([1, 2, 3], TreeNode), build_tree_from_list([1, 2, 3], TreeNode)), True)
(Solution().isSameTree(build_tree_from_list([1, 2], TreeNode), build_tree_from_list([1, None, 2], TreeNode)), False)
(Solution().isSameTree(build_tree_from_list([1, 2, 1], TreeNode), build_tree_from_list([1, 1, 2], TreeNode)), False)
(Solution().isSameTree(build_tree_from_list([10, 5, 15], TreeNode), build_tree_from_list([10, 5, None, None, 15], TreeNode)), False)
(Solution().isSameTree(build_tree_from_list([1, None, 2, 3], TreeNode), build_tree_from_list([1, None, 2, None, 3], TreeNode)), False)


from binarytree import Node
from collections import deque


class TreeNode:
    def __init__(self, val=1, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Function to create a binary tree from a list (level-order traversal)
def build_tree_from_list(node_list, node_type=TreeNode):
    if not node_list:
        return

    if type(node_list) == int:  # case when the node list is a single value
        node_list = [node_list]

    root = node_type(node_list[0])  # Create the root node
    queue = deque([root])
    index = 1

    # Process the list and construct the tree
    while index < len(node_list):
        node = queue.popleft()

        # Assign the left child if available
        if index < len(node_list) and node_list[index] != None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] != None:
            node.right = node_type(node_list[index])
            queue.append(node.right)
        index += 1

    return root


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


# O(n), O(n)
# dfs, recursion
class Solution:
    def isSameTree(self, p: TreeNode | None, q: TreeNode | None) -> bool:
        def dfs(node1, node2):
            if not node1 and not node2:  # if both nodes are None
                return True
            elif not node1 or not node2:  # if one node is None and the other is not None
                return False
            elif node1.val != node2.val:  # if both node values are not equal
                return False

            return (
                dfs(node1.left, node2.left) and  # left subtree is the same and
                dfs(node1.right, node2.right))  # right subtree is the same

        return dfs(p, q)


# O(n), O(n)
# dfs, recursion
class Solution:
    def isSameTree(self, p: TreeNode | None, q: TreeNode | None) -> bool:
        if not p and not q:  # if both nodes are None
            return True
        elif not p or not q:  # if one node is None and the other is not None
            return False
        elif p.val != q.val:  # if both node values are not equal
            return False

        return (
            self.isSameTree(p.left, q.left) and  # left subtree is the same and
            self.isSameTree(p.right, q.right))  # right subtree is the same


# O(n), O(n)
# dfs, recursion
class Solution:
    def isSameTree(self, p: TreeNode | None, q: TreeNode | None) -> bool:
        if not p and not q:  # if both are None
            return True
        
        if (p and q and p.val == q.val):  # if both nodes exist and have equal values
            return (self.isSameTree(p.left, q.left) and  # left subtree is the same
                    self.isSameTree(p.right, q.right))  # right subtree is the same
        else:
            return False


# O(n), O(n)
# bfs, iteration, queue
class Solution:
    def isSameTree(self, p: TreeNode | None, q: TreeNode | None) -> bool:
        p_queue = deque([p])  # initiate dequeues
        q_queue = deque([q])

        while p_queue or q_queue:  # while one queue is not empty
            if len(p_queue) != len(q_queue):  # different queue lengths
                return False
            
            for _ in range(len(p_queue)):  # for every node in queue
                p_node = p_queue.popleft()  # pop a node
                q_node = q_queue.popleft()

                if not p_node and not q_node:  # p and q are empty
                    continue
                elif not p_node or not q_node:  # p or q is empty
                    return False
                elif p_node.val != q_node.val:  # different p and q values
                    return False

                if p_node.left or q_node.left:  # if both nodes are not None add them to their queues
                    p_queue.append(p_node.left)
                    q_queue.append(q_node.left)
            
                if p_node.right or q_node.right:
                    p_queue.append(p_node.right)
                    q_queue.append(q_node.right)

        return True





# Subtree of Another Tree
# https://leetcode.com/problems/subtree-of-another-tree/description/
"""
Given the roots of two binary trees root and subRoot, return true if there is a subtree of root with the same structure and node values of subRoot and false otherwise.

A subtree of a binary tree tree is a tree that consists of a node in tree and all of this node's descendants. The tree tree could also be considered as a subtree of itself.

Example 1:

Input: root = [3,4,5,1,2], 

    __3
   /   \
  4     5
 / \
1   2

subRoot = [4,1,2]
  4
 / \
1   2

Output: true

Example 2:

Input: root = [3,4,5,1,2,null,null,null,null,0],

    ____3
   /     \
  4__     5
 /   \
1     2
     /
    0

subRoot = [4,1,2]

  4
 / \
1   2

Output: false
"""
(Solution().isSameTree(build_tree_from_list([4, 1, 2]), (build_tree_from_list([4, 1, 2]))), True)
(Solution().isSameTree(build_tree_from_list([3, 4, 5, 1, 2]), build_tree_from_list([4, 1, 2])), False)
(Solution().isSubtree(build_tree_from_list([3, 4, 5, 1, 2]), build_tree_from_list([4, 1, 2])), True)
(Solution().isSubtree(build_tree_from_list([3, 4, 5, 1, 2, None, None, None, None, 0]), build_tree_from_list([4, 1, 2])), False)


from binarytree import Node
from collections import deque


class TreeNode:
    def __init__(self, val=1, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Function to create a binary tree from a list (level-order traversal)
def build_tree_from_list(node_list, node_type=TreeNode):
    if not node_list:
        return

    if type(node_list) == int:  # case when the node list is a single value
        node_list = [node_list]

    root = node_type(node_list[0])  # Create the root node
    queue = deque([root])
    index = 1

    # Process the list and construct the tree
    while index < len(node_list):
        node = queue.popleft()

        # Assign the left child if available
        if index < len(node_list) and node_list[index] != None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] != None:
            node.right = node_type(node_list[index])
            queue.append(node.right)
        index += 1

    return root


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# dfs, recursive
# O(n2), O(n)
class Solution:
    def isSameTree(self, p: TreeNode | None, q: TreeNode | None) -> bool:
        if not p and not q:  # if both nodes are None
            return True
        elif not p or not q:  # if one node is None and the other is not None
            return False
        elif p.val != q.val:  # if both node values are not equal
            return False

        return (
            self.isSameTree(p.left, q.left) and  # left subtree is the same and
            self.isSameTree(p.right, q.right))  # right subtree is the same    
    
    
    def isSubtree(self, root: TreeNode | None, subRoot: TreeNode | None) -> bool:
        if not subRoot:  # if no subRoot then always True
            return True
        elif not root:  # if no root then no match found
            return False
        elif self.isSameTree(root, subRoot):  # if tres are equal
            return True
        
        return (
            self.isSubtree(root.left, subRoot) or  # check if subtree if in left tree branch
            self.isSubtree(root.right, subRoot))  # check if subtree if in right tree branch





# Lowest Common Ancestor of a Binary Search Tree
# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/description/
"""
Given a binary search tree (BST), find the lowest common ancestor (LCA) node of two given nodes in the BST.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

Example 1:

Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8

    ______6__
   /         \
  2__         8
 /   \       / \
0     4     7   9
     / \
    3   5

Output: 6
Explanation: The LCA of nodes 2 and 8 is 6.

Example 2:

Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4

    ______6__
   /         \
  2__         8
 /   \       / \
0     4     7   9
     / \
    3   5

Output: 2
Explanation: The LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.

Example 3:

Input: root = [2,1], p = 2, q = 1

  2
 /
1

Output: 2
"""
((Solution().
  lowestCommonAncestor(
    build_tree_from_list([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5]),
    build_tree_from_list(2),
    build_tree_from_list(8))).val,
 6)
((Solution().
  lowestCommonAncestor(
    build_tree_from_list([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5]),
    build_tree_from_list(2),
    build_tree_from_list(4))).val,
 2)
((Solution().
  lowestCommonAncestor(
    build_tree_from_list([2, 1]),
    build_tree_from_list(2),
    build_tree_from_list(1))).val,
 2)
((Solution().
  lowestCommonAncestor(
    build_tree_from_list([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5]),
    build_tree_from_list(3),
    build_tree_from_list(5))).val,
 4)


from binarytree import Node
from collections import deque


class TreeNode:
    def __init__(self, val=1, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Function to create a binary tree from a list (level-order traversal)
def build_tree_from_list(node_list, node_type=TreeNode):
    if not node_list:
        return

    if type(node_list) == int:  # case when the node list is a single value
        node_list = [node_list]

    root = node_type(node_list[0])  # Create the root node
    queue = deque([root])
    index = 1

    # Process the list and construct the tree
    while index < len(node_list):
        node = queue.popleft()

        # Assign the left child if available
        if index < len(node_list) and node_list[index] != None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] != None:
            node.right = node_type(node_list[index])
            queue.append(node.right)
        index += 1

    return root


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# dp, dfs, iteration
# O(logn), O(1)
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        while True:
            if (  # if p and q are lower than the current value
                p.val < root.val and
                q.val < root.val
            ):
                root = root.left  # lower common ancestor node is in the left branch
            elif (  # if p and q are highter than the current value
                p.val > root.val and
                q.val > root.val
            ):
                root = root.right  # lower common ancestor node is in the right branch
            else:  # if one is lower and the other one is higher, THIS is the LCA
                return root


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# dp, dfs, recursive
# O(logn), O(h) # O(h) for recursion stack height
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if (  # if p and q are lower than the current value
            p.val < root.val and
            q.val < root.val
        ):
            return self.lowestCommonAncestor(root.left, p, q)  # lower common ancestor node is in the left branch
        elif (  # if p and q are highter than the current value
            p.val > root.val and
            q.val > root.val
        ):
            return self.lowestCommonAncestor(root.right, p, q)  # lower common ancestor node is in the right branch
        else:
            return root





# Binary Tree Level Order Traversal
# https://leetcode.com/problems/binary-tree-level-order-traversal/description/
"""
Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

Example 1:

Input: root = [3,9,20,null,null,15,7]

  3___
 /    \
9     _20
     /   \
    15    7

Output: [[3],[9,20],[15,7]]
Example 2:

Input: root = [1]
Output: [[1]]
Example 3:

Input: root = []
Output: []
"""
(Solution().levelOrder(build_tree_from_list([3, 9, 20, None, None, 15, 7])), [[3], [9, 20], [15, 7]])
(Solution().levelOrder(build_tree_from_list([1])), [[1]])
(Solution().levelOrder(build_tree_from_list([])), [])


from binarytree import Node
from collections import deque


class TreeNode:
    def __init__(self, val=1, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Function to create a binary tree from a list (level-order traversal)
def build_tree_from_list(node_list, node_type=TreeNode):
    if not node_list:
        return

    if type(node_list) == int:  # case when the node list is a single value
        node_list = [node_list]

    root = node_type(node_list[0])  # Create the root node
    queue = deque([root])
    index = 1

    # Process the list and construct the tree
    while index < len(node_list):
        node = queue.popleft()

        # Assign the left child if available
        if index < len(node_list) and node_list[index] != None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] != None:
            node.right = node_type(node_list[index])
            queue.append(node.right)
        index += 1

    return root


from collections import deque

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# dp, bfs, iteration, dequeue, level order traversal
# O(n), O(n)
class Solution:
    def levelOrder(self, root: TreeNode | None) -> list[list[int]]:
        if not root:
            return []
        
        queue = deque([root])  # Create the root node
        level_order_list = [[root.val]]  # solution

        while queue:  # while queue is not empty
            current_level_list = []  # current level soultion
            
            for _ in range(len(queue)):  # for every node
                node = queue.popleft()  # take that node

                if node.left != None:  # if left subnode is not empty
                    queue.append(node.left)  # append it to queue
                    current_level_list.append(node.left.val)  # append its value to current level solution
                
                if node.right != None:  # if right subnode is not empty
                    queue.append(node.right)  # append it to queue
                    current_level_list.append(node.right.val)  # append its value to current level solution
            
            if current_level_list:  # if current level list has any elements
                level_order_list.append(current_level_list)  # add them to the solution

        return level_order_list





# Binary Tree Right Side View
# https://leetcode.com/problems/binary-tree-right-side-view/description/
"""
Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom-

Example 1:

Input: root = [1,2,3,null,5,null,4]

  __1
 /   \
2     3
 \     \
  5     4

Output: [1,3,4]

Example 2:

Input: root = [1,null,3]

1
 \
  3

Output: [1,3]

Example 3:

Input: root = []
Output: []
"""
(Solution().rightSideView(build_tree_from_list([1, 2, 3, None, 5, None, 4])), [1, 3, 4])
(Solution().rightSideView(build_tree_from_list([1, None, 3])), [1, 3])
(Solution().rightSideView(build_tree_from_list([])), [])


from binarytree import Node
from collections import deque


class TreeNode:
    def __init__(self, val=1, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Function to create a binary tree from a list (level-order traversal)
def build_tree_from_list(node_list, node_type=TreeNode):
    if not node_list:
        return

    if type(node_list) == int:  # case when the node list is a single value
        node_list = [node_list]

    root = node_type(node_list[0])  # Create the root node
    queue = deque([root])
    index = 1

    # Process the list and construct the tree
    while index < len(node_list):
        node = queue.popleft()

        # Assign the left child if available
        if index < len(node_list) and node_list[index] != None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] != None:
            node.right = node_type(node_list[index])
            queue.append(node.right)
        index += 1

    return root


from collections import deque

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# dp, bfs, iteration, dequeue, level order traversal
# O(n), O(n)
class Solution:
    def rightSideView(self, root: TreeNode | None) -> list[int]:
        if not root:
            return []
        
        queue = deque([root])  # Create the root node
        right_side_list = [root.val]  # solution

        while queue:  # while queue is not empty
            for _ in range(len(queue)):  # for every node
                node = queue.popleft()  # take that node

                if node.left:  # if left subnode is not empty
                    queue.append(node.left)  # append it to queue

                if node.right:  # if right subnode is not empty
                    queue.append(node.right)  # append it to queue

            if queue:  # if queue is not empty
                right_side_list.append(queue[-1].val)  # append the most right value

        return right_side_list





# Count Good Nodes in Binary Tree
# https://leetcode.com/problems/count-good-nodes-in-binary-tree/description/
"""
Given a binary tree root, a node X in the tree is named good if in the path from root to X there are no nodes with a value greater than X.

Return the number of good nodes in the binary tree.

Example 1:

Input: root = [3,1,4,3,null,1,5]

    3__
   /   \
  1     4
 /     / \
3     1   5

Output: 4
Explanation: Nodes in blue are good.
Root Node (3) is always a good node.
Node 4 -> (3,4) is the maximum value in the path starting from the root.
Node 5 -> (3,4,5) is the maximum value in the path
Node 3 -> (3,1,3) is the maximum value in the path.

Example 2:

Input: root = [3,3,null,4,2]

    __3
   /
  3
 / \
4   2

Output: 3
Explanation: Node 2 -> (3, 3, 2) is not good, because "3" is higher than it.

Example 3:

Input: root = [1]
Output: 1
Explanation: Root is considered as good.
"""
(Solution().goodNodes(build_tree_from_list([3, 1, 4, 3, None, 1, 5])), 4)
(Solution().goodNodes(build_tree_from_list([3, 3, None, 4, 2])), 3)
(Solution().goodNodes(build_tree_from_list([1])), 1)


from binarytree import Node
from collections import deque


class TreeNode:
    def __init__(self, val=1, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Function to create a binary tree from a list (level-order traversal)
def build_tree_from_list(node_list, node_type=TreeNode):
    if not node_list:
        return

    if type(node_list) == int:  # case when the node list is a single value
        node_list = [node_list]

    root = node_type(node_list[0])  # Create the root node
    queue = deque([root])
    index = 1

    # Process the list and construct the tree
    while index < len(node_list):
        node = queue.popleft()

        # Assign the left child if available
        if index < len(node_list) and node_list[index] != None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] != None:
            node.right = node_type(node_list[index])
            queue.append(node.right)
        index += 1

    return root


from collections import deque

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# dp, bfs, iteration, queue
# O(n), O(n)
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        if not root:
            return None

        queue = deque([(root, root.val)])  # Create the root node
        good_nodes_counter = 1  # solution

        while queue:  # while queue is not empty
            for _ in range(len(queue)):  # for every node
                node, max_value = queue.popleft()  # take that node

                if node.left:  # if left subnode is not empty
                    queue.append((node.left, max(max_value, node.left.val)))  # append it to queue
                    if max_value <= node.left.val:  # if max value from root to current node is less or equal to current node left value
                        good_nodes_counter += 1  # increase counter

                if node.right:  # if right subnode is not empty
                    queue.append((node.right, max(max_value, node.right.val)))  # append it to queue
                    if max_value <= node.right.val:  # if max value from root to current node is less or equal to current node right value
                        good_nodes_counter += 1  # increase counter

        return good_nodes_counter


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# dp, dfs, iteration, stack, pre-order traversal
# O(n), O(n)
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        if not root:
            return None

        stack = [(root, root.val)]  # create the stack with root node
        good_nodes_counter = 1  # solution

        while stack:  # while stack is not empty
            node, max_value = stack.pop()  # take that node

            if node.left:  # if left subnode is not empty
                stack.append((node.left, max(max_value, node.left.val)))  # append it to stack
                if max_value <= node.left.val:  # if max value from root to current node is less or equal to current node left value
                    good_nodes_counter += 1  # increase counter
            
            if node.right:  # if right subnode is not empty
                stack.append((node.right, max(max_value, node.right.val)))  # append it to stack
                if max_value <= node.right.val:  # if max value from root to current node is less or equal to current node right value
                    good_nodes_counter += 1  # increase counter

        return good_nodes_counter


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# dp, dfs, recursion, in-order traversal
# purely recursive (functional recursion)
# Functional recursion: Accumulates and passes results back through return values. This is a common approach in functional programming, where you avoid side effects and handle all state within the function's return values.
# O(n), O(n)
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        
        def dfs(node, max_till_root):
            if not node:  # if None node (nothing to add)
                return 0
            
            node_val = 1 if max_till_root <= node.val else 0  # if there are no nodes with a value greater than max till root value.
            node_val += dfs(node.left, max(max_till_root, node.val))  # calculate left subnode
            node_val += dfs(node.right, max(max_till_root, node.val))  # calculate right subnode

            return node_val  # return current level sum
        
        return dfs(root, root.val)


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# dp, dfs, recursion, in-order traversal
# stateful recursion (side-effect recursion")
# Stateful recursion: Uses external state (such as class variables) to track information as a side effect. This is more common in imperative or object-oriented programming, where functions may modify shared state.
# O(n), O(n)
class Solution:
    def __init__(self):
        self.good_nodes_counter = 0

    def goodNodes(self, root: TreeNode) -> int:        
        def dfs(node, max_till_root):
            if not node:  # if None node (nothing to add)
                return
            
            self.good_nodes_counter += 1 if max_till_root <= node.val else 0  # if there are no nodes with a value greater than max till root value.
            dfs(node.left, max(max_till_root, node.val))  # calculate left subnode
            dfs(node.right, max(max_till_root, node.val))  # calculate right subnode

        dfs(root, root.val)

        return self.good_nodes_counter


# Functional recursion
def goodNodes(root: TreeNode) -> int:        
    def dfs(node, max_till_root, good_nodes_counter):
        if not node:  # if None node (nothing to add)
            return 0
        
        good_nodes_counter = 1 if max_till_root <= node.val else 0  # if there are no nodes with a value greater than max till root value.
        good_nodes_counter += dfs(node.left, max(max_till_root, node.val), good_nodes_counter)  # calculate left subnode
        good_nodes_counter += dfs(node.right, max(max_till_root, node.val), good_nodes_counter)  # calculate right subnode

        return good_nodes_counter

    return dfs(root, root.val, 0)

(goodNodes(build_tree_from_list([3, 1, 4, 3, None, 1, 5])), 4)


# Stateful recursion
def goodNodes(root: TreeNode, good_nodes_counter=0) -> int:
    # good_nodes_counter = 0

    def dfs(node, max_till_root):
        # nonlocal good_nodes_counter

        if not node:  # if None node (nothing to add)
            return 0
            
        good_nodes_counter += 1 if max_till_root <= node.val else 0  # if there are no nodes with a value greater than max till root value.
        dfs(node.left, max(max_till_root, node.val))  # calculate left subnode
        dfs(node.right, max(max_till_root, node.val))  # calculate right subnode

    dfs(root, root.val)

    return good_nodes_counter

(goodNodes(build_tree_from_list([3, 1, 4, 3, None, 1, 5])), 4)





# Validate Binary Search Tree
# https://leetcode.com/problems/validate-binary-search-tree/description/
"""
Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:

The left 
subtree
 of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.
 

Example 1:

Input: root = [2,1,3]

  2
 / \
1   3

Output: true

Example 2:

Input: root = [5,1,4,null,null,3,6]

  5__
 /   \
1     4
     / \
    3   6

Output: false
Explanation: The root node's value is 5 but its right child's value is 4.
"""
(Solution().isValidBST(build_tree_from_list([2, 1, 3])), True)
(Solution().isValidBST(build_tree_from_list([5, 1, 4, None, None, 3, 6])), False)
(Solution().isValidBST(build_tree_from_list([2, 2, 2])), False)


from binarytree import Node
from collections import deque


class TreeNode:
    def __init__(self, val=1, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Function to create a binary tree from a list (level-order traversal)
def build_tree_from_list(node_list, node_type=TreeNode):
    if not node_list:
        return

    if type(node_list) == int:  # case when the node list is a single value
        node_list = [node_list]

    root = node_type(node_list[0])  # Create the root node
    queue = deque([root])
    index = 1

    # Process the list and construct the tree
    while index < len(node_list):
        node = queue.popleft()

        # Assign the left child if available
        if index < len(node_list) and node_list[index] != None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] != None:
            node.right = node_type(node_list[index])
            queue.append(node.right)
        index += 1

    return root


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# dp, dfs, recursion, in-order traversal
# O(n), O(n)
class Solution:
    def isValidBST(self, root: TreeNode | None) -> bool:
        def dfs(node, left_min,  right_max):
            if not node:  # if node is None then the current branch i legit
                return True  
            
            if not left_min < node.val < right_max:  # if value not in bounds
                return False
            
            return (
                dfs(node.left, left_min, node.val) and  # branch left 
                dfs(node.right, node.val, right_max)  # branch right
                )

        return dfs(root, float("-inf"), float("inf"))
        

from collections import deque

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# dp, bfs, iteration, level order traversal
# O(n), O(n)
class Solution:
    def isValidBST(self, root: TreeNode | None) -> bool:
        if not root:
            return None
        
        queue = deque([(root, float("-Inf"), float("Inf"))])  # Create the root node

        while queue:  # while queue is not empty
            for _ in range(len(queue)):
                node, min_val, max_val = queue.pop()  # for every node

                if not min_val < node.val < max_val:  # if node value not in bounds return with False
                    return False
                
                if node.left:  # if left subnode is not empty
                    queue.append((node.left, min_val, min(max_val, node.val)))  # append it to queue
                
                if node.right:  # if right subnode is not empty
                    queue.append((node.right, max(min_val, node.val), max_val))  # append it to queue

        return True  # every node is legit, so return True





# Kth Smallest Element in a BST
# https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/
"""
Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.

Example 1:

Input: root = [3,1,4,null,2], k = 1

  __3
 /   \
1     4
 \
  2

Output: 1

Example 2:

Input: root = [5,3,6,2,4,null,null,1], k = 3

      __5
     /   \
    3     6
   / \
  2   4
 /
1

Output: 3
"""
(Solution().kthSmallest(build_tree_from_list([5, 3, 7, 2, 4, None, 8]), 3), 4)
(Solution().kthSmallest(build_tree_from_list([3, 1, 4, None, 2]), 1), 1)
(Solution().kthSmallest(build_tree_from_list([5, 3, 6, 2, 4, None, None, 1]), 3), 3)
(Solution().kthSmallest(build_tree_from_list([41,37,44,24,39,42,48,1,35,38,40,None,43,46,49,0,2,30,36,None,None,None,None,None,None,45,47,None,None,None,None,None,4,29,32,None,None,None,None,None,None,3,9,26,None,31,34,None,None,7,11,25,27,None,None,33,None,6,8,10,16,None,None,None,28,None,None,5,None,None,None,None,None,15,19,None,None,None,None,12,None,18,20,None,13,17,None,None,22,None,14,None,None,21,23]), 25), 24)


from binarytree import Node
from collections import deque


class TreeNode:
    def __init__(self, val=1, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Function to create a binary tree from a list (level-order traversal)
def build_tree_from_list(node_list, node_type=TreeNode):
    if not node_list:
        return

    if type(node_list) == int:  # case when the node list is a single value
        node_list = [node_list]

    root = node_type(node_list[0])  # Create the root node
    queue = deque([root])
    index = 1

    # Process the list and construct the tree
    while index < len(node_list):
        node = queue.popleft()

        # Assign the left child if available
        if index < len(node_list) and node_list[index] != None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] != None:
            node.right = node_type(node_list[index])
            queue.append(node.right)
        index += 1

    return root


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# dp, dfs, recursion, in-order traversal
# O(n), O(n)
class Solution:
    def kthSmallest(self, root: TreeNode | None, k: int) -> int:
        value_list = []

        def dfs(node):
            if node == None:  # if no Node
                return

            dfs(node.left)  # traverse left

            if len(value_list) == k:  # early exit
                return

            value_list.append(node.val)  # append curren node value
            dfs(node.right)  # traverse right

        dfs(root)

        return value_list[-1]  # return kth element


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# dp, dfs, iteration, in-order traversal
# O(n), O(n)
class Solution:
    def kthSmallest(self, root: TreeNode | None, k: int) -> int:
        stack = []
        node = root
        # value_list = []

        while stack or node:  # while stack and node are not empty
            while node:  # while node is not empty
                stack.append(node)  # append node to stack
                node = node.left  # branch left

            node = stack.pop()  # take a node
            # value_list.append(node.val)  # append node.val to value list
            k -= 1  #  decrement counter

            if not k:  # if counter = 0
                return node.val  # current node value is kth element

            node = node.right  # branch right

        return False




# Construct Binary Tree from Preorder and Inorder Traversal
# https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/
"""
Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.

Example 1:

Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]

  3___
 /    \
9     _20
     /   \
    15    7

Example 2:

Input: preorder = [-1], inorder = [-1]
Output: [-1]
"""
(level_order_traversal(Solution().buildTree([3, 9, 20, 15, 7], [9, 3, 15, 20, 7])), [3, 9, 20, None, None, 15, 7])
(level_order_traversal(Solution().buildTree([-1], [-1])), [-1])


class TreeNode:
    def __init__(self, val=1, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


from binarytree import Node
from collections import deque

# Function create a list from a binary tree in level-order (breadth-first traversal)
def level_order_traversal(root):
    if not root:
        return []
    
    node_list = []
    queue = deque([root])  # Initialize the queue with the root node

    while queue:
        node = queue.popleft()  # Pop the current node

        if node:
            node_list.append(node.val)  # Add the value if node is not None
            queue.append(node.left)  # Append left child (could be None)
            queue.append(node.right)  # Append right child (could be None)
        else:
            node_list.append(None)  # Append None when the node is absent
    
    # Remove trailing None values
    while node_list and node_list[-1] is None:
        node_list.pop()

    return node_list


level_order_traversal(Solution().buildTree([3, 9, 20, 15, 7], [9, 3, 15, 20, 7]))


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# dp, dfs, recursion, in-order traversal, pre-order traversal
# O(n), O(n)
class Solution:
    def buildTree(self, preorder: list[int], inorder: list[int]) -> TreeNode | None:
        if not preorder or not inorder:
            return
        
        node = Node(preorder[0])
        split_index = inorder.index(preorder[0])  # 
        node.left = self.buildTree(preorder[1 : split_index + 1], inorder[: split_index])
        node.right = self.buildTree(preorder[split_index + 1 : ], inorder[split_index + 1 :])
        return node


(level_order_traversal(Solution().buildTree([3, 9, 1, 20, 15, 7], [1, 9, 3, 15, 20, 7])))





# Combination Sum III
# https://leetcode.com/problems/combination-sum-iii/description/
"""
Find all valid combinations of k numbers that sum up to n such that the following conditions are true:

Only numbers 1 through 9 are used.
Each number is used at most once.
Return a list of all possible valid combinations. The list must not contain the same combination twice, and the combinations may be returned in any order.

Example 1:

Input: k = 3, n = 7
Output: [[1,2,4]]
Explanation:
1 + 2 + 4 = 7
There are no other valid combinations.
Example 2:

Input: k = 3, n = 9
Output: [[1,2,6],[1,3,5],[2,3,4]]
Explanation:
1 + 2 + 6 = 9
1 + 3 + 5 = 9
2 + 3 + 4 = 9
There are no other valid combinations.
Example 3:

Input: k = 4, n = 1
Output: []
Explanation: There are no valid combinations.
Using 4 different numbers in the range [1,9], the smallest sum we can get is 1+2+3+4 = 10 and since 10 > 1, there are no valid combination.
"""
(Solution().combinationSum3(3, 7), [1, 2, 4])
(Solution().combinationSum3(3, 9), [[1, 2, 6], [1, 3, 5], [2, 3, 4]])
(Solution().combinationSum3(4, 1), [])


class Solution:
    def combinationSum3(self, k: int, n: int) -> list[list[int]]:
        combination = []
        combination_list = []

        def dfs(index):
            if (sum(combination) == n and  # if target sum reached and
                len(combination) == k):  # target length reached
                combination_list.append(combination.copy())  # add current solution
                return
            elif (index == 10 or  # if digit out of bounds
                  sum(combination) > n):  # and target sum exceeded
                return
            
            combination.append(index)  # include digit in solution
            dfs(index + 1)  # explore path with current digit
            combination.pop()  # exclude digit from soultion
            dfs(index + 1)  # explore path withou curren digit

        dfs(1)

        return combination_list





# Combination Sum IV
# https://leetcode.com/problems/combination-sum-iv/description/
"""
Given an array of distinct integers nums and a target integer target, return the number of possible combinations that add up to target.

The test cases are generated so that the answer can fit in a 32-bit integer.

Example 1:

Input: nums = [1,2,3], target = 4
Output: 7
Explanation:
The possible combination ways are:
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)
Note that different sequences are counted as different combinations.

Example 2:

Input: nums = [9], target = 3
Output: 0
"""
(Solution().combinationSum4([1, 2, 3], 4), 7)
(Solution().combinationSum4([9], 3), 0)
(Solution().combinationSum4([4, 2, 1], 32), 39882198)
(Solution().combinationSum4([2, 3], 7), 3)


# dp, bottom-up, iteration, tabulation (with List)
class Solution:
    def combinationSum4(self, nums: list[int], target: int) -> int:
        # Initialize a list of zeros for tabulation, where tab[i] is the number of ways to make sum i
        tab = [0 for _ in range(target + 1)]
        # Base case: 1 way to make target 0 (empty combination)
        tab[0] = 1

        # Iterate through all indices from 1 to target
        for index in range(1, target + 1):
            # For each number in nums, check if it can contribute to the current target (index)
            for num in nums:
                # If num can be subtracted from index, add the number of ways to make (index - num)
                if index - num >= 0:
                    tab[index] += tab[index - num]
        
        # Return the result for the target, which is stored in the last element of the list
        return tab[-1]


# dp, bottom-up, iteration, tabulation (with Dictionary)
class Solution:
    def combinationSum4(self, nums: list[int], target: int) -> int:
        # Tabulation dictionary, storing base case: 1 way to make target 0 (empty combination)
        tab = {0: 1}

        # Iterate through every index from 1 to target
        for index in range(1, target + 1):
            # For each number in nums, check if it can be part of the combination
            for num in nums:
                # If the number can be used (valid combination), update the tabulation table
                if index - num >= 0:
                    tab[index] = tab.get(index, 0) + tab.get(index - num, 0)
        
        # Return the result for the target value, default to 0 if no combinations
        return tab.get(target, 0)


# dp, dfs, top-down, recursion, memoization (with Dictionary)
class Solution:
    def combinationSum4(self, nums: list[int], target: int) -> int:
        # Memoization dictionary, storing base case: 1 way to make target 0 (empty combination)
        memo = {0: 1}

        # Helper function that performs depth-first search (DFS)
        def dfs(index):
            # If the index is negative, no valid combination can be made, return 0
            if index < 0:
                return 0
            # If the value has already been computed, return it (memoization check)
            elif index in memo:
                return memo[index]

            # Iterate over each number in the list
            for num in nums:
                # Recursively compute number of combinations by reducing the target (index - num)
                memo[index] = memo.get(index, 0) + dfs(index - num)

            # Return the computed value for the current target (index)
            return memo[index]

        # Start the recursion with the target value
        return dfs(target)


# dp,  dfs, top-down, recursion, memoization (with List)
class Solution:
    def combinationSum4(self, nums: list[int], target: int) -> int:
        # Memoization dictionary, storing base case: 1 way to make target 0 (empty combination)
        memo = [-1 for _ in range(target + 1)]
        memo[0] = 1

        # Helper function that performs depth-first search (DFS)
        def dfs(index):
            # If the index is negative, no valid combination can be made, return 0
            if index < 0:
                return 0
            # If the value has already been computed, return it (memoization check)
            elif memo[index] != -1:
                return memo[index]

            # Initialize the number of ways to make the current index
            memo[index] = 0

            # Iterate over each number in the list
            for num in nums:
                # Recursively compute number of combinations by reducing the target (index - num)
                memo[index] += dfs(index - num)

            # Return the computed value for the current target (index)
            return memo[index]

        # Start the recursion with the target value
        return dfs(target)





# returns all possible permutations, not only the number of them
class Solution:
    def combinationSum4(self, nums: list[int], target: int) -> int:
        combination = []
        combination_list = []

        def dfs():
            combination_sum = sum(combination)

            if combination_sum == target:
                combination_list.append(combination.copy())
                # combination_list.append(True)
                return
            elif (combination_sum > target):
                return

            for num in nums:
                combination.append(num)
                dfs()
                combination.pop()

        dfs()

        return combination_list





# Number of Islands
# https://leetcode.com/problems/number-of-islands/description/
"""
Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

Example 1:

Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
Example 2:

Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3
"""
(Solution().numIslands([["1", "1", "1", "1", "0"], ["1", "1", "0", "1", "0"], ["1", "1", "0", "0", "0"], ["0", "0", "0", "0", "0"]]), 1)
(Solution().numIslands([["1", "1", "0", "0", "0"], ["1", "1", "0", "0", "0"], ["0", "0", "1", "0", "0"], ["0", "0", "0", "1", "1"]]), 3)


# dfs, recursion
# Boundary checks: It ensures that neighbors are within grid bounds and haven't been visited before exploring them.
class Solution:
    def numIslands(self, grid: list[list[str]]) -> int:
        rows = len(grid)  # Number of rows in the grid.
        cols = len(grid[0])  # Number of columns in the grid.
        visited_land = set()  # Set to keep track of visited land cells.
        island_counter = 0  # Counter for the number of islands found.

        def dfs(row, col):
            visited_land.add((row, col))  # Mark the current cell as visited.

            # Iterate over the possible directions (right, left, down, up).
            for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                i = row + di
                j = col + dj
                
                # If the neighbor is within bounds, not visited, and is land, explore it.
                if (0 <= i < rows and
                    0 <= j < cols and
                    not (i, j) in visited_land and
                    grid[i][j] == "1"):
                    dfs(i, j)

        # Check each cell in the grid.
        for row in range(rows):
            for col in range(cols):
                # Start a new DFS for every unvisited land cell, indicating a new island.
                if (grid[row][col] == "1" and
                    not (row, col) in visited_land):
                    island_counter += 1  # Increment the island counter.
                    dfs(row, col)  # Perform DFS to mark the entire island.

        return island_counter  # Return the total number of islands found.


# dfs, recursion
# Boundary checks: It includes all the boundary and visited checks directly in the base case of the recursion.
class Solution:
    def numIslands(self, grid: list[list[str]]) -> int:
        rows = len(grid)  # Number of rows in the grid.
        cols = len(grid[0])  # Number of columns in the grid.
        visited_land = set()  # Set to keep track of visited land cells.
        island_counter = 0  # Counter for the number of islands found.

        def dfs(row, col):
            # If the cell is out of bounds, water, or already visited, stop exploring.
            if (row < 0 or
                col < 0 or
                row == rows or
                col == cols or
                (row, col) in visited_land or
                grid[row][col] == "0"):
                return
            
            visited_land.add((row, col))  # Mark the current cell as visited.

            # Explore all four possible directions: right, left, down, up.
            for i, j in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                dfs(row + i, col + j)  # Recursively visit neighboring cells.

        # Iterate over each cell in the grid.
        for row in range(rows):
            for col in range(cols):
                # Start a new DFS when an unvisited land cell ("1") is found.
                if (grid[row][col] == "1" and
                    (row, col) not in visited_land):
                    island_counter += 1  # Increment the island counter.
                    dfs(row, col)  # Perform DFS to mark the entire island.

        return island_counter  # Return the total number of islands found.


from collections import deque

# bfs, iterative
class Solution:
    def numIslands(self, grid: list[list[str]]) -> int:
        rows = len(grid)  # Number of rows in the grid.
        cols = len(grid[0])  # Number of columns in the grid.
        visited_land = set()  # Set to keep track of visited land cells.
        island_counter = 0  # Counter for the number of islands found.

        # BFS function to explore the island starting from the given cell.
        def bfs(row, col):
            visited_land.add((row, col))  # Mark the current cell as visited.
            queue = deque()  # Initialize a queue for BFS.
            queue.append((row, col))  # Start with the current land cell.

            while queue:
                row, col = queue.popleft()  # Dequeue the next cell to explore.

                # Iterate over the possible directions (right, left, down, up).
                for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    i = row + di
                    j = col + dj

                    # Check if the neighbor is within bounds, unvisited, and land.
                    if (0 <= i < rows and
                        0 <= j < cols and
                        (i, j) not in visited_land and
                        grid[i][j] == "1"):
                        queue.append((i, j))  # Add the land cell to the queue.
                        visited_land.add((i, j))  # Mark it as visited.

        # Iterate over each cell in the grid.
        for row in range(rows):
            for col in range(cols):
                # If the cell is land and hasn't been visited, start a new BFS.
                if grid[row][col] == "1" and (row, col) not in visited_land:
                    island_counter += 1  # Increment the island counter.
                    bfs(row, col)  # Perform BFS to mark the entire island.

        return island_counter  # Return the total number of islands found.


from collections import deque

# dfs, iterative
class Solution:
    def numIslands(self, grid: list[list[str]]) -> int:
        rows = len(grid)  # Number of rows in the grid.
        cols = len(grid[0])  # Number of columns in the grid.
        visited_land = set()  # Set to keep track of visited land cells.
        island_counter = 0  # Counter for the number of islands found.

        # BFS function to explore the island starting from the given cell.
        def bfs(row, col):
            visited_land.add((row, col))  # Mark the current cell as visited.
            queue = deque()  # Initialize a queue for BFS.
            queue.append((row, col))  # Start with the current land cell.

            while queue:
                row, col = queue.pop()  # Dequeue the next cell to explore.

                # Iterate over the possible directions (right, left, down, up).
                for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    i = row + di
                    j = col + dj

                    # Check if the neighbor is within bounds, unvisited, and land.
                    if (0 <= i < rows and
                        0 <= j < cols and
                        (i, j) not in visited_land and
                        grid[i][j] == "1"
                    ):
                        queue.append((i, j))  # Add the land cell to the queue.
                        visited_land.add((i, j))  # Mark it as visited.

        # Iterate over each cell in the grid.
        for row in range(rows):
            for col in range(cols):
                # If the cell is land and hasn't been visited, start a new BFS.
                if grid[row][col] == "1" and (row, col) not in visited_land:
                    island_counter += 1  # Increment the island counter.
                    bfs(row, col)  # Perform BFS to mark the entire island.

        return island_counter  # Return the total number of islands found.
    




# Max Area of Island
# https://leetcode.com/problems/max-area-of-island/description/
"""
You are given an m x n binary matrix grid. An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.

The area of an island is the number of cells with a value 1 in the island.

Return the maximum area of an island in grid. If there is no island, return 0.

Example 1:

Input: grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
Output: 6
Explanation: The answer is not 11, because the island must be connected 4-directionally.
Example 2:

Input: grid = [[0,0,0,0,0,0,0,0]]
Output: 0
"""
(Solution().maxAreaOfIsland([[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]]), 6)
(Solution().maxAreaOfIsland([[0, 0, 0, 0, 0, 0, 0, 0]]), 0)


# dfs, recursion, check boundary first
class Solution:
    def maxAreaOfIsland(self, grid: list[list[int]]) -> int:
        rows = len(grid)  # Get the number of rows and columns in the grid
        cols = len(grid[0])
        visited_land = set()  # Set to keep track of the visited land cells
        max_island_area = 0  # Variable to store the maximum area of an island found so far
        
        def dfs(row, col):  # Depth-First Search (DFS) function to explore an island
            visited_land.add((row, col))  # Mark the current cell as visited
            adjecent_area = 0  # Variable to track the area of the current island's adjacent cells
            directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]  # Possible directions to move: right, down, up, and left
            
            for di, dj in directions:  # Explore the neighboring cells
                i = row + di  # Row in the new direction
                j = col + dj  # Column in the new direction
                
                if (0 <= i < rows and  # Check if the new cell is within bounds, hasn't been visited, and is land (grid[i][j] == 1)
                    0 <= j < cols and
                    not (i, j) in visited_land and
                    grid[i][j] == 1
                ):
                    adjecent_area += dfs(i, j)  # Recursively explore the neighboring land cell and add its area
            
            return adjecent_area + 1  # Return the area of the current cell (1) plus the adjacent area found
                
        for row in range(rows):  # Iterate through each cell in the grid
            for col in range(cols):
                if (grid[row][col] == 1 and  # If the current cell is land and hasn't been visited yet
                    not (row, col) in visited_land
                ):
                    max_island_area = max(max_island_area, dfs(row, col))  # Perform DFS from this cell and update the maximum island area
        
        return max_island_area  # Return the largest island area found


# # dfs, recursion, check boundary in recursion
class Solution:
    def maxAreaOfIsland(self, grid: list[list[int]]) -> int:
        # Get the number of rows and columns in the grid
        rows = len(grid)
        cols = len(grid[0])
        visited_land = set()  # Set to keep track of the visited land cells
        max_island_area = 0  # Variable to store the maximum area of an island found so far
        
        def dfs(row, col):  # Depth-First Search (DFS) function to explore an island
            if (not 0 <= row < rows or  # Check if the new cell is within bounds, hasn't been visited, and is land (grid[i][j] == 1)
                not 0 <= col < cols or
                (row, col) in visited_land or
                grid[row][col] == 0
            ):
                return 0
                     
            visited_land.add((row, col))  # Mark the current cell as visited
            adjecent_area = 0  # Variable to track the area of the current island's adjacent cells
            directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]  # Possible directions to move: right, down, up, and left
            
            for di, dj in directions:  # Explore the neighboring cells
                i = row + di  # Row in the new direction
                j = col + dj  # Column in the new direction
                adjecent_area += dfs(i, j)  # Recursively explore the neighboring land cell and add its area
            
            return adjecent_area + 1  # Return the area of the current cell (1) plus the adjacent area found
                
        for row in range(rows):  # Iterate through each cell in the grid
            for col in range(cols):
                max_island_area = max(max_island_area, dfs(row, col))  # Perform DFS from this cell and update the maximum island area
        
        return max_island_area  # Return the largest island area found





# Clone Graph
# https://leetcode.com/problems/clone-graph/description/
"""
Given a reference of a node in a connected undirected graph.

Return a deep copy (clone) of the graph.

Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.

class Node {
    public int val;
    public List<Node> neighbors;
}

Test case format:

For simplicity, each node's value is the same as the node's index (1-indexed). For example, the first node with val == 1, the second node with val == 2, and so on. The graph is represented in the test case using an adjacency list.

An adjacency list is a collection of unordered lists used to represent a finite graph. Each list describes the set of neighbors of a node in the graph.

The given node will always be the first node with val = 1. You must return the copy of the given node as a reference to the cloned graph.

Example 1:

Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]
Explanation: There are 4 nodes in the graph.
1st node (val = 1)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
2nd node (val = 2)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
3rd node (val = 3)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
4th node (val = 4)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
Example 2:


Input: adjList = [[]]
Output: [[]]
Explanation: Note that the input contains one empty list. The graph consists of only one node with val = 1 and it does not have any neighbors.
Example 3:

Input: adjList = []
Output: []
Explanation: This an empty graph, it does not have any nodes.
"""


# Definition for a Node.
class Node:
    def __init__(self, val=0, neighbors=[]):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


# from typing import Optional
# def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
class Solution:
    def cloneGraph(self, node: Node | None) -> Node | None:
        org_to_copy = {}  # Dictionary to map original nodes to their clones

        def clone(node):
            if node in org_to_copy:  # If the node is already cloned, return the clone
                return org_to_copy[node]

            new_node = Node(node.val)  # Create a new node with the same value
            org_to_copy[node] = new_node  # Map the original node to the new clone

            for neighbor in node.neighbors:  # Iterate through all neighbors
                new_node.neighbors.append(clone(neighbor))  # Recursively clone neighbors and add to the clone's neighbor list
            
            return new_node  # Return the cloned node

        return clone(node) if node else None  # Return cloned graph or None if input node is None


# [[2, 4], [1, 3], [2, 4], [1, 3]]

node1 = Node(1)
node2 = Node(2)
node3 = Node(3)
node4 = Node(4)

node1.neighbors = [node2, node4]
node2.neighbors = [node1, node3]
node3.neighbors = [node2, node4]
node4.neighbors = [node1, node3]
node4.neighbors
node4.neighbors[0].val

new_node = Solution().cloneGraph(node1)
new_node.val





# Islands and Treasure (Walls and Gates)
# https://neetcode.io/problems/islands-and-treasure
"""
You are given a m×n m×n 2D grid initialized with these three possible values:
-1 - A water cell that can not be traversed.
0 - A treasure chest.
INF - A land cell that can be traversed. We use the integer 2^31 - 1 = 2147483647 to represent INF.
Fill each land cell with the distance to its nearest treasure chest. If a land cell cannot reach a treasure chest than the value should remain INF.

Assume the grid can only be traversed up, down, left, or right.

Example 1:

Input: [
  [2147483647,-1,0,2147483647],
  [2147483647,2147483647,2147483647,-1],
  [2147483647,-1,2147483647,-1],
  [0,-1,2147483647,2147483647]
]

Output: [
  [3,-1,0,1],
  [2,2,1,-1],
  [1,-1,2,-1],
  [0,-1,3,4]
]
Example 2:

Input: [
  [0,-1],
  [2147483647,2147483647]
]

Output: [
  [0,-1],
  [1,2]
]
"""
(Solution().islandsAndTreasure([[0, -1], [2147483647, 2147483647]]), [[0, -1], [1, 2]])
(Solution().islandsAndTreasure([[2147483647, 2147483647, 2147483647], [2147483647, -1, 2147483647], [0, 2147483647, 2147483647]]), [[2, 3, 4], [1, -1, 3], [0, 1, 2]])
(Solution().islandsAndTreasure([[2147483647, -1, 0, 2147483647], [2147483647, 2147483647, 2147483647, -1], [2147483647, -1, 2147483647, -1], [0, -1, 2147483647, 2147483647]]), [[3, -1, 0, 1], [2, 2, 1, -1], [1, -1, 2, -1], [0, -1, 3, 4]])

# dfs, recursion
# O(n4), O(n2)  - O(n4) = O(n2)2 - may vistit the same land more than once
class Solution:
    def islandsAndTreasure(self, grid: list[list[int]]) -> None:
        rows = len(grid)  # Get the number of rows
        cols = len(grid[0])  # Get the number of columns

        def dfs(row, col, distance):
            grid[row][col] = distance  # Mark the current cell with the distance
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Possible 4 directions

            for di, dj in directions:
                i = row + di  # Calculate the next row
                j = col + dj  # Calculate the next column

                if (0 <= i < rows and  # Check if row is within bounds
                    0 <= j < cols and  # Check if column is within bounds
                    not grid[i][j] in (0, 1) and  # Skip water and land cells
                    grid[i][j] > distance + 1  # Check if the current distance is smaller
                ):
                    dfs(i, j, distance + 1)  # Perform DFS on the next cell

        for row in range(rows):  # Iterate over each row
            for col in range(cols):  # Iterate over each column
                if grid[row][col] == 0:  # If the cell is water, start DFS
                    dfs(row, col, 0)  # Start DFS with distance 0

        return grid  # Return the modified grid

# bfs, iteration, dequeue
# O(n2), O(n2)
from collections import deque  # Import deque for efficient queue operations

class Solution:
    def islandsAndTreasure(self, grid: list[list[int]]) -> None:
        rows = len(grid)  # Get the number of rows
        cols = len(grid[0])  # Get the number of columns
        queue = deque()  # Initialize a queue for BFS
        visited_land = set()  # Set to keep track of visited land

        def bfs(row, col):
            # Check if the cell is out of bounds, water (-1), or already visited
            if (not 0 <= row < rows or
                not 0 <= col < cols or
                grid[row][col] == -1 or
                (row, col) in visited_land
            ):
                return  # Stop if any condition is met
            
            queue.append((row, col))  # Add the cell to the queue
            visited_land.add((row, col))  # Mark the cell as visited

        for row in range(rows):  # Iterate over each row
            for col in range(cols):  # Iterate over each column
                if grid[row][col] == 0:  # If the cell is land (0)
                    queue.append((row, col))  # Add the land cell to the queue
                    visited_land.add((row, col))  # Mark it as visited

        distance = 0  # Initialize distance from the treasure

        while queue:  # While there are cells to process
            for _ in range(len(queue)):  # Process each cell in the current layer
                row, col = queue.popleft()  # Get the next cell from the queue
                grid[row][col] = distance  # Set the current distance in the grid
                bfs(row + 1, col)  # Explore the cell below
                bfs(row, col + 1)  # Explore the cell to the right
                bfs(row - 1, col)  # Explore the cell above
                bfs(row, col - 1)  # Explore the cell to the left

            distance += 1  # Increment distance for the next layer

        return grid  # Return the modified grid





# Rotting Oranges
# https://leetcode.com/problems/rotting-oranges/description/
"""
You are given an m x n grid where each cell can have one of three values:

0 representing an empty cell,
1 representing a fresh orange, or
2 representing a rotten orange.
Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.

Return the minimum number of minutes that must elapse until no cell has a fresh orange. If this is impossible, return -1.

Example 1:

Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
Output: 4
Example 2:

Input: grid = [[2,1,1],[0,1,1],[1,0,1]]
Output: -1
Explanation: The orange in the bottom left corner (row 2, column 0) is never rotten, because rotting only happens 4-directionally.
Example 3:

Input: grid = [[0,2]]
Output: 0
Explanation: Since there are already no fresh oranges at minute 0, the answer is just 0.
"""
(Solution().orangesRotting([[2, 1, 1], [1, 1, 0], [0, 1, 1]]), 4)
(Solution().orangesRotting([[2, 1, 1], [0, 1, 1], [1, 0, 1]]), -1)
(Solution().orangesRotting([[0, 2]]), 0)
(Solution().orangesRotting([[0]]), 0)


from collections import deque

class Solution:
    def orangesRotting(self, grid: list[list[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        queue = deque()
        visited_cell = set()
        fresh_orange = set()

        def bfs(row, col):
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Define the four possible directions (right, down, left, up)
            
            for di, dj in directions:  # Explore each direction
                i = row + di
                j = col + dj

                if (0 <= i < rows and  # Check if within row bounds
                    0 <= j < cols and  # Check if within column bounds
                    (i, j) not in visited_cell and  # Ensure the cell has not been visited
                    grid[i][j] == 1  # Ensure it's a fresh orange
                ):
                    queue.append((i, j))  # Add fresh orange to the queue to rot
                    visited_cell.add((i, j))  # Mark it as visited
                    fresh_orange.discard((i, j))  # Remove it from the fresh oranges set

        # Initialize BFS with all rotting oranges and record fresh oranges
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == 2:  # Rotting orange
                    queue.append((row, col))  # Add to the queue
                    visited_cell.add((row, col))  # Mark it as visited
                elif grid[row][col] == 1:  # Fresh orange
                    fresh_orange.add((row, col))  # Add to the fresh orange set
    
        if not fresh_orange:
            return 0

        counter = -1  # Initialize counter for time

        # BFS to rot all fresh oranges
        while queue:
            counter += 1  # Increment time for each level/layer of BFS (each minute)

            for _ in range(len(queue)):
                row, col = queue.popleft()  # Process the current rotten orange
                bfs(row, col)  # Try to rot neighboring fresh oranges

        # If there are still fresh oranges left, return -1, otherwise return the time taken
        return -1 if fresh_orange else counter






(Solution().pacificAtlantic([[1]]), [[0, 0]])
(sorted((Solution().pacificAtlantic([[1, 2, 2, 3, 5], [3, 2, 3, 4, 4], [2, 4, 5, 3, 1], [6, 7, 1, 4, 5], [5, 1, 1, 2, 4]]))), sorted([[0, 4], [1, 3], [1, 4], [2, 2], [3, 0], [3, 1], [4, 0]]))





# Reverse Linked List
# https://leetcode.com/problems/reverse-linked-list/description/
"""
Given the head of a singly linked list, reverse the list, and return the reversed list.

Example 1:

Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]

Example 2:

Input: head = [1,2]
Output: [2,1]

Example 3:

Input: head = []
Output: []
"""
Solution().reverseList(node0)


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Define 3 nodes
node2 = ListNode(2)
node2.next
node1 = ListNode(1, node2)
node1.next
node0 = ListNode(0, node1)
node0.next
node0.next.next

# original Linked List
# Node0(0, Node1) => Node1(1, Node2) => Node2(2, None) => None

# reversed Linked List
# None <= Node0(0, None) <= Node1(1, Node0) <= Node2(2, Node1)

# previous   current            next
# None <= Node0(0, None) <= Node1(1, Node2)


# O(n), O(1)
# Iteration, two pointers
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        node = head  # assign head to node
        previous = None

        while node:
            next_node = node.next  # save next node pointer
            node.next = previous  # reverse pointer
            
            previous = node  # shift pointer
            node = next_node  # shift pointer
        
        return previous


# O(n), O(1)
# Iteration, two pointers
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        node = head
        previous = None

        while node:
            node.next, previous, node = previous, node, node.next  # oneliner
        
        return previous





# Merge Two Sorted Lists
# https://leetcode.com/problems/merge-two-sorted-lists/description/
"""
You are given the heads of two sorted linked lists list1 and list2.

Merge the two lists into one sorted list. The list should be made by splicing together the nodes of the first two lists.

Return the head of the merged linked list.

Example 1:


Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]
Example 2:

Input: list1 = [], list2 = []
Output: []
Example 3:

Input: list1 = [], list2 = [0]
Output: [0]
"""
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeTwoLists(self, list1: ListNode, list2: ListNode) -> ListNode:
        anchor = node = ListNode()

        while list1 and list2:
            if list1.val < list2.val:
                node.next = list1
                list1 = list1.next
            else:
                node.next = list2
                list2 = list2.next
            
            node = node.next
        
        node.next = list1 or list2

        return anchor.next

node_a2 = ListNode(4, None)
node_a1 = ListNode(2, node_a2)
node_a0 = ListNode(1, node_a1)

node_b2 = ListNode(4, None)
node_b1 = ListNode(3, node_b2)
node_b0 = ListNode(1, node_b1)

Solution().mergeTwoLists(node_a0, node_b0)





# Linked List Cycle
# https://leetcode.com/problems/linked-list-cycle/description/
"""
Given head, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.

Return true if there is a cycle in the linked list. Otherwise, return false.

Example 1:

Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).
Example 2:

Input: head = [1,2], pos = 0
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 0th node.
Example 3:

Input: head = [1], pos = -1
Output: false
Explanation: There is no cycle in the linked list.
"""


# O(n), O(1)
# Floyd's tortoise and hare
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, node: ListNode) -> bool:
        slow = node
        fast = node

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

            if slow == fast:
                return True
        
        return False


# O(n), O(n)
# set for node store
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, node: ListNode) -> bool:
        seen = set()

        while node:
            if node in seen:
                return True
            else:
                seen.add(node)
                node = node.next
        
        return False
    




# Reorder List
# https://leetcode.com/problems/reorder-list/description/
"""
You are given the head of a singly linked-list. The list can be represented as:

L0 → L1 → … → Ln - 1 → Ln
Reorder the list to be on the following form:

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
You may not modify the values in the list's nodes. Only nodes themselves may be changed.

Example 1:


Input: head = [1,2,3,4]
Output: [1,4,2,3]
Example 2:


Input: head = [1,2,3,4,5]
Output: [1,5,2,4,3]
"""
# 1, 2, 3, 4, 5, None
# s  f
#    s     f
#       s        f
# 1, 2, 3, 4
# s  f
#    s     f  


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# O(n), O(1)
# linked list (Reverse And Merge)
class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        # find middle node
        slow = head
        fast = head.next

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        # slow.next is an end
        previous = None
        node = slow.next
        slow.next = None  # Cut the list into two halves

        # reverse second part (always even size)
        while node:
            next_node = node.next
            node.next = previous
            previous = node
            node = next_node
        
        # reorder
        left = head
        right = previous

        while right:
            next_left = left.next
            next_right = right.next
            left.next = right
            right.next = next_left
            left = next_left
            right = next_right


# node4 = ListNode(5, None)
# node3 = ListNode(4, node4)
node3 = ListNode(4, None)
node2 = ListNode(3, node3)
node1 = ListNode(2, node2)
node0 = ListNode(1, node1)

print(node0.next)
reordered = Solution().reorderList(node0)
print(node0.next)



# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# O(n), O(n)
# linked list to list of nodes
class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        nodes = []
        node = head

        while node:
            nodes.append(node)
            node = node.next

        # return nodes

        left = 0
        right = len(nodes) -1

        while left < right:
            nodes[left].next = nodes[right]
            print(nodes[left].val)
            left += 1

            if left == right:
                break

            nodes[right].next = nodes[left]
            print(nodes[right].val)
            right -= 1
        
        nodes[left].next = None

        return node

node4 = ListNode(5, None)
node3 = ListNode(4, node4)
# node3 = ListNode(4, None)
node2 = ListNode(3, node3)
node1 = ListNode(2, node2)
node0 = ListNode(1, node1)

print(node1.next)
reordered = Solution().reorderList(node0)
print(node1.next)





# Remove Nth Node From End of List
# https://leetcode.com/problems/remove-nth-node-from-end-of-list/
"""
Given the head of a linked list, remove the nth node from the end of the list and return its head.

Example 1:


Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]
Example 2:

Input: head = [1], n = 1
Output: []
Example 3:

Input: head = [1,2], n = 1
Output: [1]
"""


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# O(n), O(1)
# two pointers
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        anchor = ListNode(0, head)  # add and anchor
        left = anchor  # left pointer
        right = anchor  # right pointer

        # move the right pointer `n` nodes ahead
        while n:
            right = right.next
            n -= 1
        
        # when right.next points to None, left.next points to node to remove
        while right.next:
            left = left.next
            right = right.next
        
        left.next = left.next.next  # skip nth node

        return anchor.next


# O(n), O(n)
# List
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        nodes = []
        node = head

        while node:
            nodes.append(node)
            node = node.next

        if n == len(nodes):
            return head.next
        else:
            node_index = len(nodes) - n
            nodes[node_index - 1].next = nodes[node_index].next
            return head





# Copy List with Random Pointer
# https://leetcode.com/problems/copy-list-with-random-pointer/description/
"""
A linked list of length n is given such that each node contains an additional random pointer, which could point to any node in the list, or null.

Construct a deep copy of the list. The deep copy should consist of exactly n brand new nodes, where each new node has its value set to the value of its corresponding original node. Both the next and random pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state. None of the pointers in the new list should point to nodes in the original list.

For example, if there are two nodes X and Y in the original list, where X.random --> Y, then for the corresponding two nodes x and y in the copied list, x.random --> y.

Return the head of the copied linked list.

The linked list is represented in the input/output as a list of n nodes. Each node is represented as a pair of [val, random_index] where:

val: an integer representing Node.val
random_index: the index of the node (range from 0 to n-1) that the random pointer points to, or null if it does not point to any node.
Your code will only be given the head of the original linked list.

Example 1:


Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]
Example 2:


Input: head = [[1,1],[2,1]]
Output: [[1,1],[2,1]]
Example 3:



Input: head = [[3,null],[3,0],[3,null]]
Output: [[3,null],[3,0],[3,null]]
"""


# Definition for a Node.
class ListNode:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

# O(n), O(n):
# dictionary (two pass)
class Solution:
    def copyRandomList(self, head: ListNode) -> ListNode:
        node_copy = {None: None}
        node = head

        while node:
            node_copy[node] = Node(node.val)
            node = node.next

        node = head

        while node:
            node_copy[node].next = node_copy[node.next]
            node_copy[node].random = node_copy[node.random]
            node = node.next

        return node_copy[head]





# Add Two Numbers
# https://leetcode.com/problems/add-two-numbers/description/
"""
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.
 
Example 1:

Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.
Example 2:

Input: l1 = [0], l2 = [0]
Output: [0]
Example 3:

Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
Output: [8,9,9,9,0,0,0,1]
"""


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# O(n), O(1)
# Iteration
class Solution:
    def addTwoNumbers(self, head1: ListNode, head2: ListNode) -> ListNode:
        anchor = node = ListNode()
        carry = 0

        while head1 or head2:
            first_value = head1.val if head1 else 0
            second_value = head2.val if head2 else 0
            value = first_value + second_value + carry
            
            carry = value // 10
            value = value % 10
                
            node.next = ListNode(value)
            node = node.next
            head1 = head1.next if head1 else None
            head2 = head2.next if head2 else None

        # case where there'a a carry after while loop finishes
        if carry:
            node.next = ListNode(carry)
            node = node.next
        
        node.next = None  # end linked list with None

        return anchor.next


nodeA2 = ListNode(3, None)
nodeA1 = ListNode(4, nodeA2)
nodeA0 = ListNode(2, nodeA1)

nodeB2 = ListNode(4, None)
nodeB1 = ListNode(6, nodeB2)
nodeB0 = ListNode(5, nodeB1)

print(nodeA0.val)
summed = Solution().addTwoNumbers(nodeA0, nodeB0)
print(summed.next.next.val)





# LRU Cache
# https://leetcode.com/problems/lru-cache/description/
"""
Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the LRUCache class:

LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
int get(int key) Return the value of the key if the key exists, otherwise return -1.
void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache. If the number of keys exceeds the capacity from this operation, evict the least recently used key.
The functions get and put must each run in O(1) average time complexity.
Input
["LRUCache","put","put","get","put","get","put","get","get","get"]
[[2],       [1,1],[2,2],[1],  [3,3],[2],  [4,4], [1]  ,[3],  [4]]
Expected
[null,       null,null,  1,    null,-1,    null,  -1,   3,    4]

Your LRUCache object will be instantiated and called as such:
obj = LRUCache(capacity)
param_1 = obj.get(key)
obj.put(key,value)
"""
# lRUCache = LRUCache(2)
# lRUCache.put(1, 1)  # cache is {1=1}
# lRUCache.put(2, 2)  # cache is {1=1, 2=2}
# lRUCache.get(1)    # return 1
# lRUCache.put(3, 3)  # LRU key was 2, evicts key 2, cache is {1=1, 3=3}
# lRUCache.get(2)    # returns -1 (not found)
# lRUCache.put(4, 4)  # LRU key was 1, evicts key 1, cache is {4=4, 3=3}
# lRUCache.get(1)    # return -1 (not found)
# lRUCache.get(3)    # return 3
# lRUCache.get(4)    # return 4

# lRUCache = LRUCache(2)
# lRUCache.get(2)  # -1
# lRUCache.put(2, 6)
# lRUCache.get(1)  # -1
# lRUCache.put(1, 5)
# lRUCache.put(1, 2)
# lRUCache.get(1)  # 2
# lRUCache.get(2)  # 6

lRUCache = LRUCache(2)
lRUCache.put(2, 1)
lRUCache.put(1, 1)
lRUCache.put(2, 3)
lRUCache.put(4, 1)
lRUCache.get(1)  # -1
lRUCache.get(2)  # 3


# O(1), O(n)
# Built-In Data Structure
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key in self.cache:
            # Move the accessed key to the end to mark it as recently used
            self.cache.move_to_end(key)
            
            return self.cache[key]
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Update the value and mark as recently used
            self.cache.move_to_end(key)
        elif self.capacity:
            # reduce capacity
            self.capacity -= 1
        else:
            # Pop the least recently used item (first item)
            self.cache.popitem(last=False)

        # Insert or update the key-value pair
        self.cache[key] = value


# O(1), O(n)
# Double Linked List
class Node:
    def __init__(self, key=0, value=0, next=None, prev=None) -> None:
        self.key = key
        self.value = value
        self.next = next
        self.prev = prev


class LRUCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cache = {}
        self.first = Node()
        self.last = Node()
        self.first.next = self.last
        self.last.prev = self.first


    def push_node(self, node: Node) -> None:
        prev = self.last.prev
        next = self.last
        prev.next = node
        next.prev = node
        node.prev = prev
        node.next = next


    def pop_node(self, node: Node) -> None:
        node.prev.next, node.next.prev = node.next, node.prev


    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self.pop_node(node)
            self.push_node(node)

            return node.value
        else:
            return -1
        

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.pop_node(self.cache[key])
        elif self.capacity:
            self.capacity -= 1
        else:
            lru = self.first.next
            self.pop_node(lru)
            self.cache.pop(lru.key)

        node = Node(key, value)
        self.cache[key] = node
        self.push_node(node)


# O(1), O(n)
# three dicts (cache, {key: index}, {index: key})
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # {key: value}
        self.index = 0  # in which iteration key has been modified/added
        self.key_index = {}  # {key: index}
        self.index_key = {}  # {index: key}

    def get(self, key: int) -> int:
        if key in self.cache:
            self.index_key.pop(self.key_index[key])  # pop least recent index_key element
            self.key_index[key] = self.index  # update key_index with a new index
            self.index_key[self.index] = key  # update new index_key with a key
            self.index += 1  # increase index
            return self.cache[key]
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:  # if key in cache
            self.cache[key] = value  # add new key, value pair to cache
            
            # when updating the key with max index no need to update index
            if self.key_index[key] != self.index:
                self.index_key.pop(self.key_index[key])  # pop least recent index_key element
                self.key_index[key] = self.index  # update key_index with a new index
                self.index_key[self.index] = key  # update new index_key with a key
                self.index += 1  # increase index
        elif self.capacity:  # there it capacity for another key
            self.capacity -= 1  # decrease capacity
            self.cache[key] = value  # add new key, value pair to cache
            self.key_index[key] = self.index  # update key_index with a new index
            self.index_key[self.index] = key  # update new index_key with a key
            self.index += 1  # increase index
        else:  # no capacity for new key
            min_index = next(iter(self.index_key))  # indexes are in ascending order but are not contiguous
            min_key = self.index_key[min_index]  # key to the lowest index
            self.cache.pop(min_key)  # pop key with the lowest index
            self.key_index.pop(min_key)  # pop key with the lowest index
            self.index_key.pop(min_index)  # pop lowest index from index_key

            self.cache[key] = value  # add new key, value pair to cache
            self.key_index[key] = self.index  # update key_index with a new index
            self.index_key[self.index] = key  # update new index_key with a key
            self.index += 1  # increase index





# Largest Rectangle in Histogram
# https://leetcode.com/problems/largest-rectangle-in-histogram/description/
"""
Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.

Example 1:

Input: heights = [2,1,5,6,2,3]
      _
     _x|
    |xx|
    |xx|  _
 _  |xx|_| |
| |_|xx    |
|    xx    |

Output: 10
Explanation: The above is a histogram where width of each bar is 1.
The largest rectangle is shown in the red area, which has an area = 10 units.

Example 2:

Input: heights = [2,4]
Output: 4
"""
print(Solution().largestRectangleArea([2, 1, 5, 6, 2, 3]), 10)
print(Solution().largestRectangleArea([2, 4]), 4)
print(Solution().largestRectangleArea([2, 1, 2]), 3)


class Solution:
    def largestRectangleArea(self, heights: list[int]) -> int:
        # Stack to keep track of indices and heights of rectangles
        stack = []
        # Variable to store the maximum area found so far
        max_area = 0

        # Iterate through the heights array
        for index, height in enumerate(heights):
            # Store the starting index for the current rectangle
            prev_index = index

            # If the current height is less than the height of the rectangle
            # on top of the stack, calculate the area of the rectangle
            # with the height on top of the stack as the smallest height
            while (stack and
                   height < stack[-1][1]):
                prev_index, prev_height = stack.pop()
                # Update max_area with the area of the rectangle
                # formed by the popped height
                max_area = max(max_area, prev_height * (index - prev_index))

            # Push the current index and height onto the stack
            stack.append((prev_index, height))
        
        # Process any remaining rectangles in the stack
        for index, height in stack:
            # Calculate the area for rectangles extending to the end of the array
            max_area = max(max_area, height * (len(heights) - index))
        
        # Return the largest rectangle area found
        return max_area





# Concatenation of Array
# https://leetcode.com/problems/concatenation-of-array/description/
"""
Given an integer array nums of length n, you want to create an array ans of length 2n where ans[i] == nums[i] and ans[i + n] == nums[i] for 0 <= i < n (0-indexed).

Specifically, ans is the concatenation of two nums arrays.

Return the array ans.

Example 1:

Input: nums = [1,2,1]
Output: [1,2,1,1,2,1]
Explanation: The array ans is formed as follows:
- ans = [nums[0],nums[1],nums[2],nums[0],nums[1],nums[2]]
- ans = [1,2,1,1,2,1]
Example 2:

Input: nums = [1,3,2,1]
Output: [1,3,2,1,1,3,2,1]
Explanation: The array ans is formed as follows:
- ans = [nums[0],nums[1],nums[2],nums[3],nums[0],nums[1],nums[2],nums[3]]
- ans = [1,3,2,1,1,3,2,1]
"""
print(Solution().getConcatenation([1, 2, 1]), [1, 2, 1, 1, 2, 1])
print(Solution().getConcatenation([1, 3, 2, 1]), [1, 3, 2, 1, 1, 3, 2, 1])


class Solution:
    def getConcatenation(self, numbers: list[int]) -> list[int]:
        concat_array = [0] * len(numbers) * 2

        for index, number in enumerate(numbers):
            concat_array[index] = number
            concat_array[index + len(numbers)] = number
        
        return concat_array





# Valid Palindrome II
# https://leetcode.com/problems/valid-palindrome-ii/description/
"""
Given a string s, return true if the s can be palindrome after deleting at most one character from it.

Example 1:

Input: s = "aba"
Output: true
Example 2:

Input: s = "abca"
Output: true
Explanation: You could delete the character 'c'.
Example 3:

Input: s = "abc"
Output: false
"""
print(Solution().validPalindrome("aba"), True)
print(Solution().validPalindrome("abca"), True)
print(Solution().validPalindrome("abc"), False)
print(Solution().validPalindrome("eeccccbebaeeabebccceea"), False)
print(Solution().validPalindrome("aguokepatgbnvfqmgmlcupuufxoohdfpgjdmysgvhmvffcnqxjjxqncffvmhvgsymdjgpfdhooxfuupuculmgmqfvnbgtapekouga"), True)


# O(n), O(n)
class Solution:
    def validPalindrome(self, word: str) -> bool:
        left = 0
        right = len(word) - 1

        while left < right:
            if word[left] == word[right]:
                left += 1
                right -= 1
            else:
                # check if palindrome is valid after skipping one character
                # from the left or from the right
                word_skip_left = word[left + 1: right + 1]
                word_skip_right = word[left:right]

                return (word_skip_left == word_skip_left[::-1] or
                        word_skip_right == word_skip_right[::-1])

        return True


# O(n), O(n)
# recurrence
class Solution:
    def validPalindrome(self, word: str, joker: bool = True) -> bool:
        left = 0
        right = len(word) - 1

        while left < right:
            if word[left] == word[right]:
                left += 1
                right -= 1
            elif joker:
                return (self.validPalindrome(word[left + 1: right + 1], joker=False) or
                        self.validPalindrome(word[left: right], joker=False))
            else:
                return False

        return True



# Baseball Game
# https://leetcode.com/problems/baseball-game/description/
"""
You are keeping the scores for a baseball game with strange rules. At the beginning of the game, you start with an empty record.

You are given a list of strings operations, where operations[i] is the ith operation you must apply to the record and is one of the following:

An integer x.
Record a new score of x.
'+'.
Record a new score that is the sum of the previous two scores.
'D'.
Record a new score that is the double of the previous score.
'C'.
Invalidate the previous score, removing it from the record.
Return the sum of all the scores on the record after applying all the operations.

The test cases are generated such that the answer and all intermediate calculations fit in a 32-bit integer and that all operations are valid.

 

Example 1:

Input: ops = ["5","2","C","D","+"]
Output: 30
Explanation:
"5" - Add 5 to the record, record is now [5].
"2" - Add 2 to the record, record is now [5, 2].
"C" - Invalidate and remove the previous score, record is now [5].
"D" - Add 2 * 5 = 10 to the record, record is now [5, 10].
"+" - Add 5 + 10 = 15 to the record, record is now [5, 10, 15].
The total sum is 5 + 10 + 15 = 30.
Example 2:

Input: ops = ["5","-2","4","C","D","9","+","+"]
Output: 27
Explanation:
"5" - Add 5 to the record, record is now [5].
"-2" - Add -2 to the record, record is now [5, -2].
"4" - Add 4 to the record, record is now [5, -2, 4].
"C" - Invalidate and remove the previous score, record is now [5, -2].
"D" - Add 2 * -2 = -4 to the record, record is now [5, -2, -4].
"9" - Add 9 to the record, record is now [5, -2, -4, 9].
"+" - Add -4 + 9 = 5 to the record, record is now [5, -2, -4, 9, 5].
"+" - Add 9 + 5 = 14 to the record, record is now [5, -2, -4, 9, 5, 14].
The total sum is 5 + -2 + -4 + 9 + 5 + 14 = 27.
Example 3:

Input: ops = ["1","C"]
Output: 0
Explanation:
"1" - Add 1 to the record, record is now [1].
"C" - Invalidate and remove the previous score, record is now [].
Since the record is empty, the total sum is 0.
"""
print(Solution().calPoints(["5","2","C","D","+"]), 30)
print(Solution().calPoints(["5","-2","4","C","D","9","+","+"]), 27)
print(Solution().calPoints(["1","C"]), 0)


class Solution:
    def calPoints(self, operations: list[str]) -> int:
        stack = []

        for operation in operations:
            if operation == "D":
                stack.append(stack[-1] * 2)
            elif operation == "C":
                stack.pop()
            elif operation == "+":
                x = stack.pop()
                y = stack.pop()
                stack.append(y)
                stack.append(x)
                stack.append(x + y)
            else:
                stack.append(int(operation))
        
        return sum(stack)





# Replace Elements with Greatest Element on Right Side
# https://leetcode.com/problems/replace-elements-with-greatest-element-on-right-side/description/
"""
Given an array arr, replace every element in that array with the greatest element among the elements to its right, and replace the last element with -1.

After doing so, return the array.

Example 1:

Input: arr = [17,18,5,4,6,1]
Output: [18,6,6,6,1,-1]
Explanation: 
- index 0 --> the greatest element to the right of index 0 is index 1 (18).
- index 1 --> the greatest element to the right of index 1 is index 4 (6).
- index 2 --> the greatest element to the right of index 2 is index 4 (6).
- index 3 --> the greatest element to the right of index 3 is index 4 (6).
- index 4 --> the greatest element to the right of index 4 is index 5 (1).
- index 5 --> there are no elements to the right of index 5, so we put -1.
Example 2:

Input: arr = [400]
Output: [-1]
Explanation: There are no elements to the right of index 0.
"""
print(Solution().replaceElements([17, 18, 5, 4, 6, 1]), [18, 6, 6, 6, 1, -1])
print(Solution().replaceElements([400]), [-1])


# O(n), O(n)
class Solution:
    def replaceElements(self, numbers: list[int]) -> list[int]:
        greatest_right = [-1] * len(numbers)

        for index, number in enumerate(numbers[::-1][:-1], 1):
            greatest_right[index] = max(number, greatest_right[index - 1])
        
        greatest_right.reverse()
        
        return greatest_right


# O(n), O(n)
class Solution:
    def replaceElements(self, numbers: list[int]) -> list[int]:
        numbers.pop(0)
        numbers.append(-1)

        for index in range(len(numbers) - 1)[::-1]:
            numbers[index] = max(numbers[index], numbers[index + 1])
        
        return numbers





# Minimum Difference Between Highest and Lowest of K Scores
# https://leetcode.com/problems/minimum-difference-between-highest-and-lowest-of-k-scores/description/
"""
You are given a 0-indexed integer array nums, where nums[i] represents the score of the ith student. You are also given an integer k.

Pick the scores of any k students from the array so that the difference between the highest and the lowest of the k scores is minimized.

Return the minimum possible difference.

Example 1:

Input: nums = [90], k = 1
Output: 0
Explanation: There is one way to pick score(s) of one student:
- [90]. The difference between the highest and lowest score is 90 - 90 = 0.
The minimum possible difference is 0.
Example 2:

Input: nums = [9,4,1,7], k = 2
Output: 2
Explanation: There are six ways to pick score(s) of two students:
- [9,4,1,7]. The difference between the highest and lowest score is 9 - 4 = 5.
- [9,4,1,7]. The difference between the highest and lowest score is 9 - 1 = 8.
- [9,4,1,7]. The difference between the highest and lowest score is 9 - 7 = 2.
- [9,4,1,7]. The difference between the highest and lowest score is 4 - 1 = 3.
- [9,4,1,7]. The difference between the highest and lowest score is 7 - 4 = 3.
- [9,4,1,7]. The difference between the highest and lowest score is 7 - 1 = 6.
The minimum possible difference is 2.
"""
print(Solution().minimumDifference([90], 1), 0)
print(Solution().minimumDifference([9, 4, 1, 7], 2), 2)
print(Solution().minimumDifference([87063, 61094, 44530, 21297, 95857, 93551, 9918], 6), 74560)


class Solution:
    def minimumDifference(self, numbers: list[int], k: int) -> int:
        numbers.sort()

        return min(numbers[index + k - 1] - numbers[index]
                   for index in range(len(numbers) - k + 1))





# Merge Strings Alternately
# https://leetcode.com/problems/merge-strings-alternately/description/
"""
You are given two strings word1 and word2. Merge the strings by adding letters in alternating order, starting with word1. If a string is longer than the other, append the additional letters onto the end of the merged string.

Return the merged string.

Example 1:

Input: word1 = "abc", word2 = "pqr"
Output: "apbqcr"
Explanation: The merged string will be merged as so:
word1:  a   b   c
word2:    p   q   r
merged: a p b q c r
Example 2:

Input: word1 = "ab", word2 = "pqrs"
Output: "apbqrs"
Explanation: Notice that as word2 is longer, "rs" is appended to the end.
word1:  a   b 
word2:    p   q   r   s
merged: a p b q   r   s
Example 3:

Input: word1 = "abcd", word2 = "pq"
Output: "apbqcd"
Explanation: Notice that as word1 is longer, "cd" is appended to the end.
word1:  a   b   c   d
word2:    p   q 
merged: a p b q c   d
"""
print(Solution().mergeAlternately("abc", "pqr"), "apbqcr")
print(Solution().mergeAlternately("ab", "pqrs"), "apbqrs")
print(Solution().mergeAlternately("abcd", "pq"), "apbqcd")


class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        concat = ""

        for index in range(max(len(word1), len(word2))):
            letter = word1[index] if index < len(word1) else ""
            concat += letter
            letter = word2[index] if index < len(word2) else ""
            concat += letter

        return concat





# Implement Stack using Queues
# https://leetcode.com/problems/implement-stack-using-queues/description/
"""
Implement a last-in-first-out (LIFO) stack using only two queues. The implemented stack should support all the functions of a normal stack (push, top, pop, and empty).

Implement the MyStack class:

void push(int x) Pushes element x to the top of the stack.
int pop() Removes the element on the top of the stack and returns it.
int top() Returns the element on the top of the stack.
boolean empty() Returns true if the stack is empty, false otherwise.
Notes:

You must use only standard operations of a queue, which means that only push to back, peek/pop from front, size and is empty operations are valid.
Depending on your language, the queue may not be supported natively. You may simulate a queue using a list or deque (double-ended queue) as long as you use only a queue's standard operations.
 

Example 1:

Input
["MyStack", "push", "push", "top", "pop", "empty"]
[[], [1], [2], [], [], []]
Output
[null, null, null, 2, 2, false]

Explanation
MyStack myStack = new MyStack();
myStack.push(1);
myStack.push(2);
myStack.top(); // return 2
myStack.pop(); // return 2
myStack.empty(); // return False
"""


from collections import deque

class MyStack:

    def __init__(self):
        self.queue = deque()

    def push(self, x: int) -> None:
        self.queue.append(x)

    def pop(self) -> int:
        for _ in range(len(self.queue) - 1):
            self.queue.append(self.queue.popleft())
        return self.queue.popleft()

    def top(self) -> int:
        for _ in range(len(self.queue) - 1):
            self.queue.append(self.queue.popleft())
        popped = self.queue.popleft()
        self.queue.append(popped)
        return popped

    def empty(self) -> bool:
        return not self.queue


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()





# Is Subsequence
# https://leetcode.com/problems/is-subsequence/description/
"""
Given two strings s and t, return true if s is a subsequence of t, or false otherwise.

A subsequence of a string is a new string that is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (i.e., "ace" is a subsequence of "abcde" while "aec" is not).

Example 1:

Input: s = "abc", t = "ahbgdc"
Output: true
Example 2:

Input: s = "axc", t = "ahbgdc"
Output: false
"""
print(Solution().isSubsequence("abc", "ahbgdc"), True)
print(Solution().isSubsequence("axc", "ahbgdc"), False)
print(Solution().isSubsequence("", "ahbgdc"), True)
print(Solution().isSubsequence("", ""), True)


class Solution:
    def isSubsequence(self, word1: str, word2: str) -> bool:
        if not word1: 
            return True
        
        left = 0

        for letter in word2:
            if letter == word1[left]:
                left += 1

            if left == len(word1):
                return True
        
        return False





# Length of Last Word
# https://leetcode.com/problems/length-of-last-word/description/
"""
Given a string s consisting of words and spaces, return the length of the last word in the string.

A word is a maximal 
substring
 consisting of non-space characters only.

Example 1:

Input: s = "Hello World"
Output: 5
Explanation: The last word is "World" with length 5.
Example 2:

Input: s = "   fly me   to   the moon  "
Output: 4
Explanation: The last word is "moon" with length 4.
Example 3:

Input: s = "luffy is still joyboy"
Output: 6
Explanation: The last word is "joyboy" with length 6.
"""
print(Solution().lengthOfLastWord("Hello World"), 5)
print(Solution().lengthOfLastWord("   fly me   to   the moon  "), 4)
print(Solution().lengthOfLastWord("luffy is still joyboy"), 6)


class Solution:
    def lengthOfLastWord(self, text: str) -> int:
        return len(text.split()[-1])





# Reverse String
# https://leetcode.com/problems/reverse-string/description/
"""
Write a function that reverses a string. The input string is given as an array of characters s.

You must do this by modifying the input array in-place with O(1) extra memory.

Example 1:

Input: s = ["h","e","l","l","o"]
Output: ["o","l","l","e","h"]
Example 2:

Input: s = ["H","a","n","n","a","h"]
Output: ["h","a","n","n","a","H"]
"""
(Solution().reverseString(["h", "e", "l", "l", "o"]), ["o", "l", "l", "e", "h"])
(Solution().reverseString(["H", "a", "n", "n", "a", "h"]), ["h", "a", "n", "n", "a", "H"])


# O(n), O(1)
# built-in function
class Solution:
    def reverseString(self, s: list[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        s.reverse()


# O(n), O(1)
# two pointers
class Solution:
    def reverseString(self, s: list[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        
        left = 0
        right = len(s) - 1

        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1


# O(n), O(n)
# stack
class Solution:
    def reverseString(self, s: list[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        
        stack = []

        for letter in s:
            stack.append(letter)

        for index in range(len(s)):
            s[index] = stack.pop()


# O(n), O(n)
# recursion
class Solution:
    def reverseString(self, s: list[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        
        def local_reverse(left, right):
            if left < right:
                s[left], s[right] = s[right], s[left]
                local_reverse(left + 1, right - 1)

        local_reverse(0, len(s) - 1)





# Contains Duplicate II
# https://leetcode.com/problems/contains-duplicate-ii/description/
"""
Given an integer array nums and an integer k, return true if there are two distinct indices i and j in the array such that nums[i] == nums[j] and abs(i - j) <= k.

Example 1:

Input: nums = [1,2,3,1], k = 3
Output: true
Example 2:

Input: nums = [1,0,1,1], k = 1
Output: true
Example 3:

Input: nums = [1,2,3,1,2,3], k = 2
Output: false
"""
print(Solution().containsNearbyDuplicate([1, 2, 3, 1], 3), True)
print(Solution().containsNearbyDuplicate([1, 0, 1, 1], 1), True)
print(Solution().containsNearbyDuplicate([1, 2, 3, 1, 2, 3], 2), False)


# O(n), O(n)
# sliding window
class Solution:
    def containsNearbyDuplicate(self, numbers: list[int], window_length: int) -> bool:
        left = 0
        window = set()

        for right, number in enumerate(numbers):
            if number in window:
                return True
            else:
                window.add(number)

            if right - left + 1 != window_length + 1:
                continue
            
            window.remove(numbers[left])  # discard
            left += 1

        return False





# Implement Queue using Stacks
# https://leetcode.com/problems/implement-queue-using-stacks/description/
"""
Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (push, peek, pop, and empty).

Implement the MyQueue class:

void push(int x) Pushes element x to the back of the queue.
int pop() Removes the element from the front of the queue and returns it.
int peek() Returns the element at the front of the queue.
boolean empty() Returns true if the queue is empty, false otherwise.
Notes:

You must use only standard operations of a stack, which means only push to top, peek/pop from top, size, and is empty operations are valid.
Depending on your language, the stack may not be supported natively. You may simulate a stack using a list or deque (double-ended queue) as long as you use only a stack's standard operations.

Example 1:

Input
["MyQueue", "push", "push", "peek", "pop", "empty"]
[[], [1], [2], [], [], []]
Output
[null, null, null, 1, 1, false]

Explanation
MyQueue myQueue = new MyQueue();
myQueue.push(1); // queue is: [1]
myQueue.push(2); // queue is: [1, 2] (leftmost is front of the queue)
myQueue.peek(); // return 1
myQueue.pop(); // return 1, queue is [2]
myQueue.empty(); // return false
"""
print(Solution().containsNearbyDuplicate([1, 2, 3, 1], 3), True)
print(Solution().containsNearbyDuplicate([1, 0, 1, 1], 1), True)
print(Solution().containsNearbyDuplicate([1, 2, 3, 1, 2, 3], 2), False)


# O(1), O(n)
# sliding window
class MyQueue:
    def __init__(self):
        self.stack = []
        self.stack_reversed = []


    def push(self, x: int) -> None:
        self.stack.append(x)


    def pop(self) -> int:
        if self.stack_reversed:
            return self.stack_reversed.pop()

        while self.stack:
            self.stack_reversed.append(self.stack.pop())
        
        return self.stack_reversed.pop()
        

    def peek(self) -> int:
        if self.stack_reversed:
            return self.stack_reversed[-1]

        while self.stack:
            self.stack_reversed.append(self.stack.pop())
        
        return self.stack_reversed[-1]


    def empty(self) -> bool:
        return (not self.stack and 
                not self.stack_reversed)


# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()





# Longest Common Prefix
# https://leetcode.com/problems/longest-common-prefix/description/
"""
Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".

Example 1:

Input: strs = ["flower","flow","flight"]
Output: "fl"
Example 2:

Input: strs = ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.
"""
print(Solution().longestCommonPrefix(["flower", "flow", "flight"]), "fl")
print(Solution().longestCommonPrefix(["dog", "racecar", "car"]), "")


class Solution:
    def longestCommonPrefix(self, words: list[str]) -> str:
        prefix = words[0]

        for word in words[1:]:
            for index, letter in enumerate(prefix):
                if (index < len(word) and
                        letter == word[index]):
                    continue
                else:
                    prefix = prefix[:index]
                    break

        return prefix


class Solution:
    def longestCommonPrefix(self, words: list[str]) -> str:
        prefix = words[0]

        for word in words[1:]:
            for index, letter in enumerate(prefix):
                if (index == len(word) or
                        letter != word[index]):
                    prefix = prefix[:index]
                    break

        return prefix


class Solution:
    def longestCommonPrefix(self, words: list[str]) -> str:
        prefix = ""

        for index, letter in enumerate(words[0]):
            for word in words:
                if (index == len(word) or
                        letter != word[index]):
                    return prefix
            
            prefix += letter





# Merge Sorted Array
# https://leetcode.com/problems/merge-sorted-array/description/
"""
You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of elements in nums1 and nums2 respectively.

Merge nums1 and nums2 into a single array sorted in non-decreasing order.

The final sorted array should not be returned by the function, but instead be stored inside the array nums1. To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, and the last n elements are set to 0 and should be ignored. nums2 has a length of n.

Example 1:

Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]
Explanation: The arrays we are merging are [1,2,3] and [2,5,6].
The result of the merge is [1,2,2,3,5,6] with the underlined elements coming from nums1.
Example 2:

Input: nums1 = [1], m = 1, nums2 = [], n = 0
Output: [1]
Explanation: The arrays we are merging are [1] and [].
The result of the merge is [1].
Example 3:

Input: nums1 = [0], m = 0, nums2 = [1], n = 1
Output: [1]
Explanation: The arrays we are merging are [] and [1].
The result of the merge is [1].
Note that because m = 0, there are no elements in nums1. The 0 is only there to ensure the merge result can fit in nums1.
"""
print(Solution().merge([1, 2, 3, 0, 0, 0], 3, [2, 5, 6], 3), [1, 2, 2, 3, 5, 6])
print(Solution().merge([1], 1, [], 0), [1])
print(Solution().merge([0], 0, [1], 1), [1])
print(Solution().merge([2, 0], 1, [1], 1), [1, 2])
print(Solution().merge([-1, 0, 0, 3, 3, 3, 0, 0, 0], 6, [1, 2, 2], 3), [-1, 0, 0, 1, 2, 2, 3, 3, 3])
print(Solution().merge([4, 5, 6, 0, 0, 0], 3, [1, 2, 3], 3), [1, 2, 3, 4, 5, 6])



# O(n), O(1)
# in-place method, start from the end of the list
class Solution:
    def merge(self, nums1: list[int], m: int, nums2: list[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        m -= 1
        n -= 1
        
        for index in reversed(range(m + n + 2)):
            # if all nums from nums2 are used, all nums from nums2 are already in the correct place
            if n == -1:
                return nums1
            
            # if all nums from nums1 are used, the rest of the nums from nums2 need to 
            # replace the nums in nums1 from the start
            if m == -1:
                break
            
            if nums1[m] > nums2[n]:
                nums1[index] = nums1[m]
                m -= 1
            else:
                nums1[index] = nums2[n]
                n -= 1
            
        # reaplacing nums1 with nums2 from the start
        if n != -1:
            for index in range(n + 1):
                nums1[index] = nums2[index]

        return nums1


# O(n), O(n)
# temp list, start from the beginning of the list
class Solution:
    def merge(self, nums1: list[int], m: int, nums2: list[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        l1 = 0
        l2 = 0
        index = 0
        # sum_numbers = [0] * (m + n)
        sum_numbers = []

        while (l1 < m and l2 < n):
            if nums1[l1] < nums2[l2]:
                # sum_numbers[index] = nums1[l1]
                sum_numbers.append(nums1[l1])
                l1 += 1
            else:
                # sum_numbers[index] = nums2[l2]
                sum_numbers.append(nums2[l2])
                l2 += 1

            index += 1

        if l2 != n:
            sum_numbers.extend(nums2[l2:])
        elif l1 != m:
            sum_numbers.extend(nums1[l1: m])

        for index in range(len(nums1)):
            nums1[index] = sum_numbers[index]

        return sum_numbers


# O(n), O(1)
# in-place method, deque, start from the beginning of the list
from collections import deque

class Solution:
    def merge(self, nums1: list[int], m: int, nums2: list[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        if not nums2:
            return

        l2 = 0
        index = 0
        temp = deque()

        for index in range(m + n):
            if temp:
                if index < m:
                    temp.append(nums1[index])

                if l2 < n and temp[0] < nums2[l2]:
                    nums1[index] = temp.popleft()
                elif l2 == n:
                    nums1[index] = temp.popleft()
                else:
                    nums1[index] = nums2[l2]
                    l2 += 1
            else:
                if (index >= m and nums1[index] == 0):
                    nums1[index] = nums2[l2]
                    l2 += 1
                elif nums1[index] > nums2[l2]:
                    temp.append(nums1[index])
                    nums1[index] = nums2[l2]
                    l2 += 1

        return nums1





# Number of Sub-arrays of Size K and Average Greater than or Equal to Threshold
# https://leetcode.com/problems/number-of-sub-arrays-of-size-k-and-average-greater-than-or-equal-to-threshold/description/
"""
Given an array of integers arr and two integers k and threshold, return the number of sub-arrays of size k and average greater than or equal to threshold.

Example 1:

Input: arr = [2,2,2,2,5,5,5,8], k = 3, threshold = 4
Output: 3
Explanation: Sub-arrays [2,5,5],[5,5,5] and [5,5,8] have averages 4, 5 and 6 respectively. All other sub-arrays of size 3 have averages less than 4 (the threshold).
Example 2:

Input: arr = [11,13,17,23,29,31,7,5,2,3], k = 3, threshold = 5
Output: 6
Explanation: The first 6 sub-arrays of size 3 have averages greater than 5. Note that averages are not integers.
"""
print(Solution().numOfSubarrays([2, 2, 2, 2, 5, 5, 5, 8], 3, 4), 3)
print(Solution().numOfSubarrays([11, 13, 17, 23, 29, 31, 7, 5, 2, 3], 3, 5), 6)
print(Solution().numOfSubarrays([8246,1867,193,4539,2650,4721,2944,5777,8055,7502,4334,2137,3658,4156,4628,1139,7963,8035,6008,8427,1841,9169,1059,6158,9116,8052,7074,7866,584,666,192,8081,8273,2809,3017,7852,1869,3395,4649,5366,8834,9100,1643,9511,4136,3897,7193,2500,2721,8477,2887,8300,3922,579,4228,7983,4247,5362,5581,9270,8602,1944,240,6044,6036,1219,6901,2007,2123,9699,3388,390,9144,7697,5160,6442,7078,9758,8841,2064,4096,146,7362,3952,2346,4171,7598,1201,1860,9101,8979,8437,1989,5349,5148,9422,7217,1406,8414,3586,5935,7395,2257,7802,9449,3824,6874,3684,4252,3947,8985,1052,7295,2976,2045,2315,4887,307,8784,988,942,7960,747,1593,1112,7874], 1, 307), 122)


class Solution:
    def numOfSubarrays(self, numbers: list[int], size: int, threshold: int) -> int:
        counter = 0

        for index in range(len(numbers) - size + 1):
            # calculate the first sum
            if not index:
                subarray_sum = sum(numbers[index: index + size])
            # calculate every other sum
            else:
                subarray_sum += - numbers[index - 1] + numbers[index + size - 1]

            # if average is greather than threshold
            if subarray_sum / threshold >= threshold:
                counter += 1

        return counter





# Make The String Great
# https://leetcode.com/problems/make-the-string-great/description/
"""
Given a string s of lower and upper case English letters.

A good string is a string which doesn't have two adjacent characters s[i] and s[i + 1] where:

0 <= i <= s.length - 2
s[i] is a lower-case letter and s[i + 1] is the same letter but in upper-case or vice-versa.
To make the string good, you can choose two adjacent characters that make the string bad and remove them. You can keep doing this until the string becomes good.

Return the string after making it good. The answer is guaranteed to be unique under the given constraints.

Notice that an empty string is also good.

Example 1:

Input: s = "leEeetcode"
Output: "leetcode"
Explanation: In the first step, either you choose i = 1 or i = 2, both will result "leEeetcode" to be reduced to "leetcode".
Example 2:

Input: s = "abBAcC"
Output: ""
Explanation: We have many possible scenarios, and all lead to the same answer. For example:
"abBAcC" --> "aAcC" --> "cC" --> ""
"abBAcC" --> "abBA" --> "aA" --> ""
Example 3:

Input: s = "s"
Output: "s"
"""
print(Solution().makeGood("leEeetcode"), "leetcode")
print(Solution().makeGood("abBAcC"), "")
print(Solution().makeGood("s"), "s")
print(Solution().makeGood("Mc"), "Mc")


class Solution:
    def makeGood(self, word: str) -> str:
        stack = []

        for letter in word:
            if (stack and  # Check if the stack is not empty
                    letter != stack[-1]  and # and if the top of the stack differs the current letter
                    letter.lower() == stack[-1].lower()):  # and both lowercase are the same
                stack.pop()
            else:
                stack.append(letter)

        return "".join(stack)





# Search Insert Position
# https://leetcode.com/problems/search-insert-position/description/
"""
Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You must write an algorithm with O(log n) runtime complexity.

Example 1:

Input: nums = [1,3,5,6], target = 5
Output: 2
Example 2:

Input: nums = [1,3,5,6], target = 2
Output: 1
Example 3:

Input: nums = [1,3,5,6], target = 7
Output: 4
"""
print(Solution().searchInsert([1, 3, 5, 6], 0), 0)
print(Solution().searchInsert([1, 3, 5, 6], 1), 0)
print(Solution().searchInsert([1, 3, 5, 6], 2), 1)
print(Solution().searchInsert([1, 3, 5, 6], 3), 1)
print(Solution().searchInsert([1, 3, 5, 6], 4), 2)
print(Solution().searchInsert([1, 3, 5, 6], 5), 2)
print(Solution().searchInsert([1, 3, 5, 6], 6), 3)
print(Solution().searchInsert([1, 3, 5, 6], 7), 4)


class Solution:
    def searchInsert(self, numbers: list[int], target: int) -> int:
        left = 0
        right = len(numbers) - 1

        while left <= right:
            middle = (left + right) // 2
            middle_number = numbers[middle]

            if target == middle_number:
                return middle
            elif target < middle_number:
                right = middle - 1
            else:
                left = middle + 1

        return left





# Pascal's Triangle
# https://leetcode.com/problems/pascals-triangle/description/
"""
Given an integer numRows, return the first numRows of Pascal's triangle.

In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:

Example 1:

Input: numRows = 5
Output: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
Example 2:

Input: numRows = 1
Output: [[1]]
"""
print(Solution().generate(5), [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]])
print(Solution().generate(1), [[1]])


class Solution:
    def generate(self, numRows: int) -> list[list[int]]:
        triangle = []

        for index in range(numRows):
            if not index:  # first row in Pascal's Triangle
                triangle.append([1])
            else:  # not first row in Pascal's Triangle
                row = [1]  # every row starts with `1``

                for number in range(index - 1):  # for at least 3 elemenst in the row
                    row.append(sum(triangle[-1][number: number + 2]))

                row.append(1)  # every row ends with `1``
                triangle.append(row)  # push row to Pascal's Triangle

        return triangle





# Move Zeroes
# https://leetcode.com/problems/move-zeroes/description/
"""
Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.

Note that you must do this in-place without making a copy of the array.

Example 1:

Input: nums = [0,1,0,3,12]
Output: [1,3,12,0,0]
Example 2:

Input: nums = [0]
Output: [0]
"""
print(Solution().moveZeroes([0, 1, 0, 3, 12]), [1, 3, 12, 0, 0])
print(Solution().moveZeroes([0]), [0])
print(Solution().moveZeroes([1]), [1])


class Solution:
    def moveZeroes(self, nums: list[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        left = 0

        for right, number in enumerate(nums):
            if number:
                nums[left], nums[right] = number, nums[left]
                left += 1

        return nums



class Solution:
    def moveZeroes(self, nums: list[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        zero_counter = 0
        left = 0

        for right, number in enumerate(nums):
            if number:
                nums[left] = number
                left += 1
            else:
                zero_counter += 1

        for right in range(len(nums) - zero_counter, len(nums)):
            nums[right] = 0

        return nums





# Frequency of the Most Frequent Element
# https://leetcode.com/problems/frequency-of-the-most-frequent-element/description/
"""
The frequency of an element is the number of times it occurs in an array.

You are given an integer array nums and an integer k. In one operation, you can choose an index of nums and increment the element at that index by 1.

Return the maximum possible frequency of an element after performing at most k operations.

Example 1:

Input: nums = [1,2,4], k = 5
Output: 3
Explanation: Increment the first element three times and the second element two times to make nums = [4,4,4].
4 has a frequency of 3.
Example 2:

Input: nums = [1,4,8,13], k = 5
Output: 2
Explanation: There are multiple optimal solutions:
- Increment the first element three times to make nums = [4,4,8,13]. 4 has a frequency of 2.
- Increment the second element four times to make nums = [1,8,8,13]. 8 has a frequency of 2.
- Increment the third element five times to make nums = [1,4,13,13]. 13 has a frequency of 2.
Example 3:

Input: nums = [3,9,6], k = 2
Output: 1
"""
print(Solution().maxFrequency([1, 2, 4], 5), 3)
print(Solution().maxFrequency([1, 4, 8, 13], 5), 2)
print(Solution().maxFrequency([3, 9, 6], 2), 1)
print(Solution().maxFrequency([9930, 9923, 9983, 9997, 9934, 9952, 9945, 9914, 9985, 9982, 9970, 9932, 9985, 9902, 9975, 9990, 9922, 9990, 9994, 9937, 9996, 9964, 9943, 9963, 9911, 9925, 9935, 9945, 9933, 9916, 9930, 9938, 10000, 9916, 9911, 9959, 9957, 9907, 9913, 9916, 9993, 9930, 9975, 9924, 9988, 9923, 9910, 9925, 9977, 9981, 9927, 9930, 9927, 9925, 9923, 9904, 9928, 9928, 9986, 9903, 9985, 9954, 9938, 9911, 9952, 9974, 9926, 9920, 9972, 9983, 9973, 9917, 9995, 9973, 9977, 9947, 9936, 9975, 9954, 9932, 9964, 9972, 9935, 9946, 9966], 3056), 73)


# O(nlogn), O(1)
# sliding window
class Solution:
    def maxFrequency(self, numbers: list[int], k: int) -> int:
        numbers.sort()
        left = 0
        total = 0
        frequency = 1

        for right, number in enumerate(numbers):
            total += number  # Add the current number to the total sum of the window.

            # Check if the current window is valid:
            # `(number * (right - left + 1))` represents the total sum needed
            # to make all elements in the window equal to `number`.
            # If the required total exceeds the available increment `k`, shrink the window.
            while (number * (right - left + 1) - total > k):
                total -= numbers[left]  # Remove the element at the left pointer from the total sum.
                left += 1  # Move the left pointer to the right, shrinking the window.
        
            frequency = max(frequency, right - left + 1)  # Update the maximum frequency found so far.

        return frequency





# Removing Stars From a String
# https://leetcode.com/problems/removing-stars-from-a-string/description/
"""
You are given a string s, which contains stars *.

In one operation, you can:

Choose a star in s.
Remove the closest non-star character to its left, as well as remove the star itself.
Return the string after all stars have been removed.

Note:

The input will be generated such that the operation is always possible.
It can be shown that the resulting string will always be unique.

Example 1:

Input: s = "leet**cod*e"
Output: "lecoe"
Explanation: Performing the removals from left to right:
- The closest character to the 1st star is 't' in "leet**cod*e". s becomes "lee*cod*e".
- The closest character to the 2nd star is 'e' in "lee*cod*e". s becomes "lecod*e".
- The closest character to the 3rd star is 'd' in "lecod*e". s becomes "lecoe".
There are no more stars, so we return "lecoe".
Example 2:

Input: s = "erase*****"
Output: ""
Explanation: The entire string is removed, so we return an empty string.
"""
print(Solution().removeStars("leet**cod*e"), "lecoe")
print(Solution().removeStars("erase*****"), "")


class Solution:
    def removeStars(self, word: str) -> str:
        stack = []

        for letter in word:
            if stack and letter == "*":
                stack.pop()
            else:
                stack.append(letter)
        
        return "".join(stack)





# Guess Number Higher or Lower
# https://leetcode.com/problems/guess-number-higher-or-lower/description/
"""
We are playing the Guess Game. The game is as follows:

I pick a number from 1 to n. You have to guess which number I picked.

Every time you guess wrong, I will tell you whether the number I picked is higher or lower than your guess.

You call a pre-defined API int guess(int num), which returns three possible results:

-1: Your guess is higher than the number I picked (i.e. num > pick).
1: Your guess is lower than the number I picked (i.e. num < pick).
0: your guess is equal to the number I picked (i.e. num == pick).
Return the number that I picked.

Example 1:

Input: n = 10, pick = 6
Output: 6
Example 2:

Input: n = 1, pick = 1
Output: 1
Example 3:

Input: n = 2, pick = 1
Output: 1
"""
print(Solution().guessNumber(10), 6)


# The guess API is already defined for you.
# @param num, your guess
# @return -1 if num is higher than the picked number
#          1 if num is lower than the picked number
#          otherwise return 0
# def guess(num: int) -> int:

# O(logn), O(1)
# binary search
class Solution:
    def guess(self, number):
        if number == 6:
            return 0
        elif number < 6:
            return 1
        else:
            return -1

    def guessNumber(self, n: int) -> int:
        left = 1
        right = n

        while left <= right:
            middle = (left + right) // 2
            current_guess = self.guess(middle)
        
            if current_guess == 0:
                return middle
            elif current_guess == 1:
                left = middle + 1
            else:
                right = middle - 1





# Remove Element
# https://leetcode.com/problems/remove-element/description/
"""
Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. The order of the elements may be changed. Then return the number of elements in nums which are not equal to val.

Consider the number of elements in nums which are not equal to val be k, to get accepted, you need to do the following things:

Change the array nums such that the first k elements of nums contain the elements which are not equal to val. The remaining elements of nums are not important as well as the size of nums.
Return k.
Custom Judge:

The judge will test your solution with the following code:

int[] nums = [...]; // Input array
int val = ...; // Value to remove
int[] expectedNums = [...]; // The expected answer with correct length.
                            // It is sorted with no values equaling val.

int k = removeElement(nums, val); // Calls your implementation

assert k == expectedNums.length;
sort(nums, 0, k); // Sort the first k elements of nums
for (int i = 0; i < actualLength; i++) {
    assert nums[i] == expectedNums[i];
}
If all assertions pass, then your solution will be accepted.

Example 1:

Input: nums = [3,2,2,3], val = 3
Output: 2, nums = [2,2,_,_]
Explanation: Your function should return k = 2, with the first two elements of nums being 2.
It does not matter what you leave beyond the returned k (hence they are underscores).
Example 2:

Input: nums = [0,1,2,2,3,0,4,2], val = 2
Output: 5, nums = [0,1,4,0,3,_,_,_]
Explanation: Your function should return k = 5, with the first five elements of nums containing 0, 0, 1, 3, and 4.
Note that the five elements can be returned in any order.
It does not matter what you leave beyond the returned k (hence they are underscores).
"""
print(Solution().removeElement([3, 2, 2, 3], 3), [2, 2])
print(Solution().removeElement([0, 1, 2, 2, 3, 0, 4, 2], 2), [0, 1, 3, 4, 2])


class Solution:
    def removeElement(self, numbers: list[int], value: int) -> int:
        left = 0

        for number in numbers:
            if number != value:
                numbers[left] = number
                left += 1
                
        return left





# Palindrome Linked List
# https://leetcode.com/problems/palindrome-linked-list/description/
"""
Given the head of a singly linked list, return true if it is a 
palindrome
 or false otherwise.

Example 1:


Input: head = [1,2,2,1]
Output: true
Example 2:


Input: head = [1,2]
Output: false
"""


# Find the middle of the list draft
# [1, 2, 2L, 1 NoneP]
# [1, 2, 3L, 2, 1P, None]


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# O(n), O(1)
# linked list
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        if not head or not head.next:
            return True  # Single node or empty list is always a palindrome
        
        # Step 1: Find the middle of the list
        slow = head  # slow pointer
        fast = head  # fast pointer

        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        
        # Step 2: Reverse the second half of the list
        previous = None

        while slow:
            next_node = slow.next
            slow.next = previous
            previous = slow
            slow = next_node

        # Step 3: Compare the two halves
        left = head
        right = previous
        
        while right:
            if left.val != right.val:
                return False
            
            left = left.next
            right = right.next

        return True

# initiate [1, 2, 2, 1] linked list
node4 = ListNode(1)
node3 = ListNode(2, node4)
node2 = ListNode(2, node3)
node1 = ListNode(1, node2)

print(Solution().isPalindrome(node1), True)


# O(n), O(n)
# convert to list
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        numbers = []

        while head:
            numbers.append(head.val)
            head = head.next
        
        return numbers == numbers[::-1]





# Remove Linked List Elements
# https://leetcode.com/problems/remove-linked-list-elements/description/
"""
Given the head of a linked list and an integer val, remove all the nodes of the linked list that has Node.val == val, and return the new head.

Example 1:

Input: head = [1,2,6,3,4,5,6], val = 6
Output: [1,2,3,4,5]
Example 2:

Input: head = [], val = 1
Output: []
Example 3:

Input: head = [7,7,7,7], val = 7
Output: []
"""
print(Solution().removeElements(node1, 7), None)


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        anchor = node = ListNode()
        node.next = head

        while node:
            if node.next and node.next.val == val:
                node.next = node.next.next
            else:
                node = node.next

        return anchor.next


# initiate [(0), 7, 7, 7, 7, (None)]
node4 = ListNode(7)
node3 = ListNode(7, node4)
node2 = ListNode(7, node3)
node1 = ListNode(7, node2)





# Unique Email Addresses
# https://leetcode.com/problems/unique-email-addresses/description/
"""
Every valid email consists of a local name and a domain name, separated by the '@' sign. Besides lowercase letters, the email may contain one or more '.' or '+'.

For example, in "alice@leetcode.com", "alice" is the local name, and "leetcode.com" is the domain name.
If you add periods '.' between some characters in the local name part of an email address, mail sent there will be forwarded to the same address without dots in the local name. Note that this rule does not apply to domain names.

For example, "alice.z@leetcode.com" and "alicez@leetcode.com" forward to the same email address.
If you add a plus '+' in the local name, everything after the first plus sign will be ignored. This allows certain emails to be filtered. Note that this rule does not apply to domain names.

For example, "m.y+name@email.com" will be forwarded to "my@email.com".
It is possible to use both of these rules at the same time.

Given an array of strings emails where we send one email to each emails[i], return the number of different addresses that actually receive mails.

Example 1:

Input: emails = ["test.email+alex@leetcode.com","test.e.mail+bob.cathy@leetcode.com","testemail+david@lee.tcode.com"]
Output: 2
Explanation: "testemail@leetcode.com" and "testemail@lee.tcode.com" actually receive mails.
Example 2:

Input: emails = ["a@leetcode.com","b@leetcode.com","c@leetcode.com"]
Output: 3
"""
print(Solution().numUniqueEmails(["test.email+alex@leetcode.com", "test.e.mail+bob.cathy@leetcode.com", "testemail+david@lee.tcode.com"]), 2)
print(Solution().numUniqueEmails(["a@leetcode.com", "b@leetcode.com", "c@leetcode.com"]), 3)


# O(n2), O(n)
# built-in function
class Solution:
    def numUniqueEmails(self, emails: list[str]) -> int:
        clean_emails = set()

        for emial in emails:
            name, domain = emial.split("@")

            # remove all after "+" in the name
            clean_name = name.split("+")[0]

            # remove "." from the name
            clean_name = clean_name.replace(".", "")
            
            clean_emails.add(clean_name + "@" + domain)

        return len(clean_emails)


# O(n2), O(n)
# regex
import re

class Solution:
    def numUniqueEmails(self, emails: list[str]) -> int:
        clean_emails = set()

        for emial in emails:
            name, domain = emial.split("@")

            # remove "." from the name
            clean_name = re.sub(r"\.", "", name)

            # remove all after "+" in the name
            clean_name = re.search(r"(\w+)(\+?)", clean_name).group(1)

            clean_emails.add(clean_name + "@" + domain)

        return len(clean_emails)





# Remove Duplicates from Sorted Array
# https://leetcode.com/problems/remove-duplicates-from-sorted-array/description/
"""
Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once. The relative order of the elements should be kept the same. Then return the number of unique elements in nums.

Consider the number of unique elements of nums to be k, to get accepted, you need to do the following things:

Change the array nums such that the first k elements of nums contain the unique elements in the order they were present in nums initially. The remaining elements of nums are not important as well as the size of nums.
Return k.
Custom Judge:

The judge will test your solution with the following code:

int[] nums = [...]; // Input array
int[] expectedNums = [...]; // The expected answer with correct length

int k = removeDuplicates(nums); // Calls your implementation

assert k == expectedNums.length;
for (int i = 0; i < k; i++) {
    assert nums[i] == expectedNums[i];
}
If all assertions pass, then your solution will be accepted.

Example 1:

Input: nums = [1,1,2]
Output: 2, nums = [1,2,_]
Explanation: Your function should return k = 2, with the first two elements of nums being 1 and 2 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
Example 2:

Input: nums = [0,0,1,1,1,2,2,3,3,4]
Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
Explanation: Your function should return k = 5, with the first five elements of nums being 0, 1, 2, 3, and 4 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
"""
print(Solution().removeDuplicates([1, 1, 2]), 2)
print(Solution().removeDuplicates([0, 0, 1, 1, 1, 2, 2, 3, 3, 4]), 5)


# O(n), O(1)
# two pointers
class Solution:
    def removeDuplicates(self, numbers: list[int]) -> int:
        left = 0
        
        for number in numbers[1:]:
            if number > numbers[left]:
                left += 1
                numbers[left] = number

        return left + 1





# Fruit Into Baskets
# https://leetcode.com/problems/fruit-into-baskets/description/
"""
You are visiting a farm that has a single row of fruit trees arranged from left to right. The trees are represented by an integer array fruits where fruits[i] is the type of fruit the ith tree produces.

You want to collect as much fruit as possible. However, the owner has some strict rules that you must follow:

You only have two baskets, and each basket can only hold a single type of fruit. There is no limit on the amount of fruit each basket can hold.
Starting from any tree of your choice, you must pick exactly one fruit from every tree (including the start tree) while moving to the right. The picked fruits must fit in one of your baskets.
Once you reach a tree with fruit that cannot fit in your baskets, you must stop.
Given the integer array fruits, return the maximum number of fruits you can pick.

Example 1:

Input: fruits = [1,2,1]
Output: 3
Explanation: We can pick from all 3 trees.
Example 2:

Input: fruits = [0,1,2,2]
Output: 3
Explanation: We can pick from trees [1,2,2].
If we had started at the first tree, we would only pick from trees [0,1].
Example 3:

Input: fruits = [1,2,3,2,2]
Output: 4
Explanation: We can pick from trees [2,3,2,2].
If we had started at the first tree, we would only pick from trees [1,2].
"""
print(Solution().totalFruit([1, 2, 1]), 3)
print(Solution().totalFruit([0, 1, 2, 2]), 3)
print(Solution().totalFruit([1, 2, 3, 2, 2]), 4)
print(Solution().totalFruit([3, 3, 3, 1, 2, 1, 1, 2, 3, 3, 4]), 5)
print(Solution().totalFruit([1, 0, 1, 4, 1, 4, 1, 2, 3]), 5)


# O(n), O(1)
# sliding window
class Solution:
    def totalFruit(self, fruits: list[int]) -> int:
        left = 0
        basket = {}
        max_fruit = 0

        for right, fruit in enumerate(fruits):
            basket[fruit] = basket.get(fruit, 0) + 1  # add a fruit to the basket

            while len(basket) > 2:  # while too many fruit types
                left_fruit = fruits[left]  # `left` fruit type
                basket[left_fruit] -= 1  # remove one `left` fruit from basket
                left += 1  # increase the left pointer

                if not basket[left_fruit]:  # if no `left` fruit
                    basket.pop(left_fruit)  # pop that fruit type

            if right - left + 1 > max_fruit:  # update max fruit counter
                max_fruit = right - left + 1

        return max_fruit





# Validate Stack Sequences
# https://leetcode.com/problems/validate-stack-sequences/description/
"""
Given two integer arrays pushed and popped each with distinct values, return true if this could have been the result of a sequence of push and pop operations on an initially empty stack, or false otherwise.

Example 1:

Input: pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
Output: true
Explanation: We might do the following sequence:
push(1), push(2), push(3), push(4),
pop() -> 4,
push(5),
pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
Example 2:

Input: pushed = [1,2,3,4,5], popped = [4,3,5,1,2]
Output: false
Explanation: 1 cannot be popped before 2.
"""
print(Solution().validateStackSequences([1, 2, 3, 4, 5], [4, 5, 3, 2, 1]), True)
print(Solution().validateStackSequences([1, 2, 3, 4, 5], [4, 3, 5, 1, 2]), False)
print(Solution().validateStackSequences([2, 1, 0], [1, 2, 0]), True)


# O(n), O(n)
# stack, deque
from collections import deque

class Solution:
    def validateStackSequences(self, pushed: list[int], popped: list[int]) -> bool:
        queue = deque(popped)
        stack = []

        for number in pushed:
            stack.append(number)

            while stack and stack[-1] == queue[0]:
                stack.pop()
                queue.popleft()

        if stack or queue:
            return False
        else:
            return True





# Arranging Coins
# https://leetcode.com/problems/arranging-coins/description/
"""
You have n coins and you want to build a staircase with these coins. The staircase consists of k rows where the ith row has exactly i coins. The last row of the staircase may be incomplete.

Given the integer n, return the number of complete rows of the staircase you will build.

Example 1:


Input: n = 5
Output: 2
Explanation: Because the 3rd row is incomplete, we return 2.
Example 2:


Input: n = 8
Output: 3
Explanation: Because the 4th row is incomplete, we return 3.
"""


print(Solution().arrangeCoins(5), 2)
print(Solution().arrangeCoins(8), 3)
print(Solution().arrangeCoins(2), 1)

# O(logn), O(1)
# arithmetic sequence
class Solution:
    def arrangeCoins(self, coins: int) -> int:
        left = 1  # minimum number of coins
        right = coins  # max upper boundry eg. for 10 coins there will be less than 10 rows

        while left <= right:
            middle = (left + right) // 2  # row number
            coin_stack = (1 + middle) / 2 * middle  # number of coins in coin stack

            if coin_stack == coins:  # if the number of provided coins is equal to the number of coins in the stack
                return middle
            elif coin_stack > coins:  # to much coins in the stack or the last row is not full
                right = middle - 1
            else:  # too little coins in the stack
                left = middle + 1

        return right

# O(n), O(n)
class Solution:
    def arrangeCoins(self, coins: int) -> int:
        total = 0
        index = 0

        while total < coins:
            index += 1
            total += index

        # if totlal == coins then the last row if full of coins
        return index if total == coins else index - 1





# Remove Duplicates from Sorted List
# https://leetcode.com/problems/remove-duplicates-from-sorted-list/description/
"""
Given the head of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.

Example 1:

Input: head = [1,1,2]
Output: [1,2]
Example 2:


Input: head = [1,1,2,3,3]
Output: [1,2,3]
"""


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head

        node = head

        while node.next:
            if node.val == node.next.val:
                node.next = node.next.next
            else:
                node = node.next
        
        return head


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head

        node = head

        while node:
            while node.next and node.val == node.next.val:
                node.next = node.next.next
            
            node = node.next
        
        return head





# Isomorphic Strings
# https://leetcode.com/problems/isomorphic-strings/description/
"""
Given two strings s and t, determine if they are isomorphic.

Two strings s and t are isomorphic if the characters in s can be replaced to get t.

All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character, but a character may map to itself.

Example 1:

Input: s = "egg", t = "add"

Output: true

Explanation:

The strings s and t can be made identical by:

Mapping 'e' to 'a'.
Mapping 'g' to 'd'.
Example 2:

Input: s = "foo", t = "bar"

Output: false

Explanation:

The strings s and t can not be made identical as 'o' needs to be mapped to both 'a' and 'r'.

Example 3:

Input: s = "paper", t = "title"

Output: true
"""
print(Solution().isIsomorphic("egg", "add"), True)
print(Solution().isIsomorphic("foo", "bar"), False)
print(Solution().isIsomorphic("paper", "title"), True)
print(Solution().isIsomorphic("badc", "baba"), False)


class Solution:
    def isIsomorphic(self, word1: str, word2: str) -> bool:
        iso_map = {}  # letter from word1 => letter in word2

        for index, letter in enumerate(word1):
            if letter in iso_map:  # if the letter appeared before
                if iso_map[letter] != word2[index]:  # if no match in same indexes
                    return False
            else:  # if the letter is seen for the first time
                if word2[index] in iso_map.values():  # if there's already another key pointing to that letter
                    return False

                iso_map[letter] = word2[index]  # add letter to iso dict

        return True





# Assign Cookies
# https://leetcode.com/problems/assign-cookies/description/
"""
Assume you are an awesome parent and want to give your children some cookies. But, you should give each child at most one cookie.

Each child i has a greed factor g[i], which is the minimum size of a cookie that the child will be content with; and each cookie j has a size s[j]. If s[j] >= g[i], we can assign the cookie j to the child i, and the child i will be content. Your goal is to maximize the number of your content children and output the maximum number.

Example 1:

Input: g = [1,2,3], s = [1,1]
Output: 1
Explanation: You have 3 children and 2 cookies. The greed factors of 3 children are 1, 2, 3. 
And even though you have 2 cookies, since their size is both 1, you could only make the child whose greed factor is 1 content.
You need to output 1.
Example 2:

Input: g = [1,2], s = [1,2,3]
Output: 2
Explanation: You have 2 children and 3 cookies. The greed factors of 2 children are 1, 2. 
You have 3 cookies and their sizes are big enough to gratify all of the children, 
You need to output 2.
"""
print(Solution().findContentChildren([1, 2, 3], [1, 1]), 1)
print(Solution().findContentChildren([1, 2], [1, 2, 3]), 2)
print(Solution().findContentChildren([10, 9, 8, 7], [5, 6, 7, 8]), 2)


class Solution:
    def findContentChildren(self, children: list[int], cookies: list[int]) -> int:
        children.sort()
        cookies.sort()
        left = 0

        for cookie in cookies:
            if (left < len(children) and  # if in bounds of children and
                    children[left] <= cookie):  # child is content
                left += 1  # move to the next child

        return left



# Maximum Number of Vowels in a Substring of Given Length
# https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/description/
"""
Given a string s and an integer k, return the maximum number of vowel letters in any substring of s with length k.

Vowel letters in English are 'a', 'e', 'i', 'o', and 'u'.

Example 1:

Input: s = "abciiidef", k = 3
Output: 3
Explanation: The substring "iii" contains 3 vowel letters.
Example 2:

Input: s = "aeiou", k = 2
Output: 2
Explanation: Any substring of length 2 contains 2 vowels.
Example 3:

Input: s = "leetcode", k = 3
Output: 2
Explanation: "lee", "eet" and "ode" contain 2 vowels.
"""
print(Solution().maxVowels("abciiidef", 3), 3)
print(Solution().maxVowels("aeiou", 2), 2)
print(Solution().maxVowels("leetcode", 3), 2)


class Solution:
    def maxVowels(self, word: str, substring_length: int) -> int:
        left = 0
        vovels = "aeoiu"
        vovel_count = 0
        max_vovels = 0

        for right, letter in enumerate(word):
            if letter in vovels:  # if right letter is a vovel increase the counter
                vovel_count += 1

            if right - left + 1 == substring_length + 1:  # if substring is longer than its upper limit
                if word[left] in vovels:  # if left letter is a vovel decrease the counter
                    vovel_count -= 1
                
                left += 1  # maintain the window length
            
            max_vovels = max(max_vovels, vovel_count)

        return max_vovels





# Asteroid Collision
# https://leetcode.com/problems/asteroid-collision/description/
"""
We are given an array asteroids of integers representing asteroids in a row.

For each asteroid, the absolute value represents its size, and the sign represents its direction (positive meaning right, negative meaning left). Each asteroid moves at the same speed.

Find out the state of the asteroids after all collisions. If two asteroids meet, the smaller one will explode. If both are the same size, both will explode. Two asteroids moving in the same direction will never meet.

Example 1:

Input: asteroids = [5,10,-5]
Output: [5,10]
Explanation: The 10 and -5 collide resulting in 10. The 5 and 10 never collide.
Example 2:

Input: asteroids = [8,-8]
Output: []
Explanation: The 8 and -8 collide exploding each other.
Example 3:

Input: asteroids = [10,2,-5]
Output: [10]
Explanation: The 2 and -5 collide resulting in -5. The 10 and -5 collide resulting in 10.
"""
print(Solution().asteroidCollision([5, 10, -5]), [5, 10])
print(Solution().asteroidCollision([8, -8]), [])
print(Solution().asteroidCollision([10, 2, -5]), [10])
print(Solution().asteroidCollision([-2, -1, 1, 2]), [-2, -1, 1, 2])
print(Solution().asteroidCollision([1, 2, -5]), [-5])
print(Solution().asteroidCollision([-2, -2, 1, -2]), [-2, -2, -2])
print(Solution().asteroidCollision([-2, -1, 1, -2]), [-2, -1, -2])
print(Solution().asteroidCollision([-2, 2, 1, -2]), [-2])


class Solution:
    def asteroidCollision(self, asteroids: list[int]) -> list[int]:
        stack = []

        for asteroid in asteroids:
            add_astr = True

            # pop all asteroids from the end of the stack that collide with current asteroid
            while (add_astr and  # can add asteroid 
                   stack and  # stack is not empty
                   asteroid < 0 and  # curren asteroid moves left
                   stack[-1] > 0):  # asteroid from stack movet right
                diff = asteroid + stack[-1]  

                if not diff:  # both asteroids are the same size
                    add_astr = False  # stop the loop and don't add the curren asteroid to the stack
                    stack.pop()
                elif diff < 0:  # current asteroid is bigger
                    stack.pop()
                else:  # asteroid in the stack is bigger
                    add_astr = False  # stop the loop and don't add the curren asteroid to the stack

            if add_astr:
                stack.append(asteroid)

        return stack





# Squares of a Sorted Array
# https://leetcode.com/problems/squares-of-a-sorted-array/description/
"""
Given an integer array nums sorted in non-decreasing order, return an array of the squares of each number sorted in non-decreasing order.

Example 1:

Input: nums = [-4,-1,0,3,10]
Output: [0,1,9,16,100]
Explanation: After squaring, the array becomes [16,1,0,9,100].
After sorting, it becomes [0,1,9,16,100].
Example 2:

Input: nums = [-7,-3,2,3,11]
Output: [4,9,9,49,121]
"""
print(Solution().sortedSquares([-4, -1, 0, 3, 10]), [0, 1, 9, 16, 100])
print(Solution().sortedSquares([-7, -3, 2, 3, 11]), [4, 9, 9, 49, 121])
print(Solution().sortedSquares([1, 2, 3]), [1, 4, 9])
print(Solution().sortedSquares([-3, -2, -1]), [1, 4, 9])
print(Solution().sortedSquares([0]), [0])
print(Solution().sortedSquares([0, 1]), [0, 1])
print(Solution().sortedSquares([-10000, -9999, -7, -5, 0, 0, 10000]), [0, 0, 25, 49, 99980001, 100000000, 100000000])
print(Solution().sortedSquares([-1, 1]), [1, 1])
print(Solution().sortedSquares([-1, 1, 1]), [1, 1, 1])
print(Solution().sortedSquares([-3, -3, -2, 1]), [1, 4, 9, 9])


class Solution:
    def sortedSquares(self, numbers: list[int]) -> list[int]:
        left = 0
        right = len(numbers) - 1
        sorted_squared = []

        while left <= right:
            if numbers[left] ** 2 > numbers[right] ** 2:
                sorted_squared.append(numbers[left] ** 2)
                left += 1
            else:
                sorted_squared.append(numbers[right] ** 2)
                right -= 1
        
        sorted_squared.reverse()
        
        return sorted_squared





# Middle of the Linked List
# https://leetcode.com/problems/middle-of-the-linked-list/description/
"""
Given the head of a singly linked list, return the middle node of the linked list.

If there are two middle nodes, return the second middle node.

Example 1:

Input: head = [1,2,3,4,5]
Output: [3,4,5]
Explanation: The middle node of the list is node 3.
Example 2:

Input: head = [1,2,3,4,5,6]
Output: [4,5,6]
Explanation: Since the list has two middle nodes with values 3 and 4, we return the second one.
"""

# [1, 2, 3_, 4, 5_]
# [1, 2, 3, 4_, 5, 6, None_]

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        slow = head
        fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        return slow





# Can Place Flowers
# https://leetcode.com/problems/can-place-flowers/description/
"""
You have a long flowerbed in which some of the plots are planted, and some are not. However, flowers cannot be planted in adjacent plots.

Given an integer array flowerbed containing 0's and 1's, where 0 means empty and 1 means not empty, and an integer n, return true if n new flowers can be planted in the flowerbed without violating the no-adjacent-flowers rule and false otherwise.

Example 1:

Input: flowerbed = [1,0,0,0,1], n = 1
Output: true
Example 2:

Input: flowerbed = [1,0,0,0,1], n = 2
Output: false
"""
print(Solution().canPlaceFlowers([1, 0, 0, 0, 1], 1), True)
print(Solution().canPlaceFlowers([1, 0, 0, 0, 1], 2), False)
print(Solution().canPlaceFlowers([1, 0, 0, 0], 1), True)
print(Solution().canPlaceFlowers([1, 0, 1, 0, 1, 0, 1], 0), True)
print(Solution().canPlaceFlowers([0, 0, 1, 0, 1], 1), True)
print(Solution().canPlaceFlowers([1, 0, 0, 0, 1, 0, 0], 2), True)
print(Solution().canPlaceFlowers([0, 0, 0], 2), True)


# O(n), O(1)
# Iteration
class Solution:
    def canPlaceFlowers(self, flowerbed: list[int], n: int) -> bool:
        if not n:
            return True
        
        counter = 0
        contiguous_zeros = 1  # count current contiguous zeros length, initialize with 1 because start with  [0, 0, 1, ...] is legit place for a flower

        for place in flowerbed:
            if place:  # if occupied by a flower
                if contiguous_zeros >= 3:
                    counter += (contiguous_zeros - 1) // 2  # 3, 4 => 1; 5, 6 => 2

                    # early exit
                    if counter >= n:
                        return True

                contiguous_zeros = 0  # reset contiguous zeros length
            else:  # if free place for a flower
                contiguous_zeros += 1

        # if free places at the end of the flowerbed
        counter += contiguous_zeros // 2

        return counter >= n





# Find First Palindromic String in the Array
# https://leetcode.com/problems/find-first-palindromic-string-in-the-array/description/
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
print(Solution().firstPalindrome(["abc", "car", "ada", "racecar", "cool"]), "ada")
print(Solution().firstPalindrome(["notapalindrome", "racecar"]), "racecar")
print(Solution().firstPalindrome(["def", "ghi"]), "")

# O(n), O(1)
# built-in function
class Solution:
    def firstPalindrome(self, words: list[str]) -> str:
        for word in words:
            if word == word[::-1]:
                return word

        return ""


# O(n), O(1)
# two pointers
class Solution:
    def firstPalindrome(self, words: list[str]) -> str:
        for word in words:
            left = 0
            right = len(word) - 1

            while left < right:
                if word[left] != word[right]:
                    break
                
                left += 1
                right -= 1
            
            else:
                return word

        return ""





# Sort Array By Parity
# https://leetcode.com/problems/sort-array-by-parity/description/
"""
Given an integer array nums, move all the even integers at the beginning of the array followed by all the odd integers.

Return any array that satisfies this condition.

Example 1:

Input: nums = [3,1,2,4]
Output: [2,4,3,1]
Explanation: The outputs [4,2,3,1], [2,4,1,3], and [4,2,1,3] would also be accepted.
Example 2:

Input: nums = [0]
Output: [0]
"""


# O(n), O(1)
# two pointers
class Solution:
    def sortArrayByParity(self, numbers: list[int]) -> list[int]:
        left = 0

        for right, number in enumerate(numbers):
            if not number % 2:
                numbers[left], numbers[right] = numbers[right], numbers[left]
                left += 1

        return numbers


# O(n), O(n)
# deque
from collections import deque

class Solution:
    def sortArrayByParity(self, numbers: list[int]) -> list[int]:
        queue = deque()

        for number in numbers:
            if number % 2:
                queue.append(number)
            else:
                queue.appendleft(number)

        return list(queue)





# Intersection of Two Linked Lists
# https://leetcode.com/problems/intersection-of-two-linked-lists/description/
"""
Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect. If the two linked lists have no intersection at all, return null.

For example, the following two linked lists begin to intersect at node c1:


The test cases are generated such that there are no cycles anywhere in the entire linked structure.

Note that the linked lists must retain their original structure after the function returns.

Example 1:


Input: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
Output: Intersected at '8'
Explanation: The intersected node's value is 8 (note that this must not be 0 if the two lists intersect).
From the head of A, it reads as [4,1,8,4,5]. From the head of B, it reads as [5,6,1,8,4,5]. There are 2 nodes before the intersected node in A; There are 3 nodes before the intersected node in B.
- Note that the intersected node's value is not 1 because the nodes with value 1 in A and B (2nd node in A and 3rd node in B) are different node references. In other words, they point to two different locations in memory, while the nodes with value 8 in A and B (3rd node in A and 4th node in B) point to the same location in memory.
Example 2:


Input: intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
Output: Intersected at '2'
Explanation: The intersected node's value is 2 (note that this must not be 0 if the two lists intersect).
From the head of A, it reads as [1,9,1,2,4]. From the head of B, it reads as [3,2,4]. There are 3 nodes before the intersected node in A; There are 1 node before the intersected node in B.
Example 3:


Input: intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
Output: No intersection
Explanation: From the head of A, it reads as [2,6,4]. From the head of B, it reads as [1,5]. Since the two lists do not intersect, intersectVal must be 0, while skipA and skipB can be arbitrary values.
Explanation: The two lists do not intersect, so return null.
"""


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # get the length of a linked list
    def getLength(self, node):
        list_length = 0
        
        while node:
            list_length += 1
            node = node.next
        
        return list_length

    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:        
        a_length = self.getLength(headA)
        b_length = self.getLength(headB)
        diff_length = abs(a_length - b_length)
        
        if a_length > b_length:
            for _ in range(diff_length):
                headA = headA.next
        else:
            for _ in range(diff_length):
                headB = headB.next

        while headA and headB:
            if headA == headB:
                return headA
            else:
                headA = headA.next
                headB = headB.next

        return None


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:        
        nodeA = headA
        nodeB = headB

        while nodeA != nodeB:
            nodeA = nodeA.next if nodeA else headB
            nodeB = nodeB.next if nodeB else headA
        else:
            return nodeA





# Majority Element
# https://leetcode.com/problems/majority-element/description/
"""
Given an array nums of size n, return the majority element.

The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.

Example 1:

Input: nums = [3,2,3]
Output: 3
Example 2:

Input: nums = [2,2,1,1,1,2,2]
Output: 2
"""
print(Solution().majorityElement([3, 2, 3]), 3)
print(Solution().majorityElement([2, 2, 1, 1, 1, 2, 2]), 2)


# O(n), O(n)
# dict, two pass
class Solution:
    def majorityElement(self, numbers: list[int]) -> int:
        counter = {}

        for number in numbers:
            counter[number] = counter.get(number, 0) + 1

        max_value = 0
        max_key = 0
        
        for key, value in counter.items():            
            if value > max_value:
                max_value = max(max_value, value)
                max_key = key

        return max_key


# O(n), O(n)
# dict, one pass
class Solution:
    def majorityElement(self, numbers: list[int]) -> int:
        counter = {}
        max_value = 0
        max_key = 0

        for number in numbers:
            counter[number] = counter.get(number, 0) + 1

            if counter[number] > max_value:
                max_key = number
                max_value = max(max_value, counter[number])

        return max_key


# O(n), O(1)
# Boyer-Moore
class Solution:
    def majorityElement(self, numbers: list[int]) -> int:
        key = 0
        value = 0
        
        for number in numbers:
            if not value:
                key = number
                value += 1
            else:
                if key == number:
                    value += 1
                else:
                    value -= 1

        return key

# O(n), O(1)
# Boyer-Moore
class Solution:
    def majorityElement(self, numbers: list[int]) -> int:
        key = 0
        value = 0
        
        for number in numbers:
            if not value:
                key = number
            
            value += 1 if key == number else -1

        return key





# Reverse Words in a String III
# https://leetcode.com/problems/reverse-words-in-a-string-iii/
"""
Given a string s, reverse the order of characters in each word within a sentence while still preserving whitespace and initial word order.

Example 1:

Input: s = "Let's take LeetCode contest"
Output: "s'teL ekat edoCteeL tsetnoc"
Example 2:

Input: s = "Mr Ding"
Output: "rM gniD"
"""
print(Solution().reverseWords("Let's take LeetCode contest"), "s'teL ekat edoCteeL tsetnoc")
print(Solution().reverseWords("Mr Ding"), "rM gniD")
print(Solution().reverseWords("hehhhhhhe"), "ehhhhhheh")


# O(n), O(n)
# built-in function
class Solution:
    def reverseWords(self, text: str) -> str:
        return " ".join(word[::-1] 
                        for word in text.split())


# O(n), O(n)
# two pionters
class Solution:
    def reverseWords(self, text: str) -> str:
        text = list(text)
        text.append(" ")  # add space to catch the last word
        left = 0  # the start of the word pointer

        for index, letter in enumerate(text):
            if letter == " ":
                right = index - 1  # the end of the word pointer

                while left < right:
                    text[left], text[right] = text[right], text[left]  # swap letters
                    left += 1
                    right -= 1
                
                left = index + 1  # move the left pointer to the beginning of the next word

        return "".join(text[:-1])


class Solution:
    def reverseWords(self, text: str) -> str:
        rev_str = ""

        for word in text.split():
            rev_str = rev_str + word[::-1] + " "

        return rev_str[:-1]


class Solution:
    def reverseWords(self, text: str) -> str:
        rev_word = ""
        rev_text = ""

        for letter in text:
            if letter != " ":
                rev_word = letter + rev_word
            else:
                rev_text += rev_word + " "
                rev_word = ""
        
        return rev_text + rev_word




# Minimum Number of Flips to Make the Binary String Alternating
# https://leetcode.com/problems/minimum-number-of-flips-to-make-the-binary-string-alternating/description/
"""
You are given a binary string s. You are allowed to perform two types of operations on the string in any sequence:

Type-1: Remove the character at the start of the string s and append it to the end of the string.
Type-2: Pick any character in s and flip its value, i.e., if its value is '0' it becomes '1' and vice-versa.
Return the minimum number of type-2 operations you need to perform such that s becomes alternating.

The string is called alternating if no two adjacent characters are equal.

For example, the strings "010" and "1010" are alternating, while the string "0100" is not.
 

Example 1:

Input: s = "111000"
Output: 2
Explanation: Use the first operation two times to make s = "100011".
Then, use the second operation on the third and sixth elements to make s = "101010".
Example 2:

Input: s = "010"
Output: 0
Explanation: The string is already alternating.
Example 3:

Input: s = "1110"
Output: 1
Explanation: Use the second operation on the second element to make s = "1010".
"""
print(Solution().minFlips("111000"), 2)
print(Solution().minFlips("010"), 0)
print(Solution().minFlips("1110"), 1)


# blueprint
# "111000" numbers
# "010101" target A
# "101010" target B


# O(n), O(n)
# sliding window
class Solution:
    def minFlips(self, numbers: str) -> int:
        left = 0
        target_a = "01" * len(numbers)  # target string A "0101..."
        target_b = "10" * len(numbers)  # target string B "1010..."
        numbers = numbers * 2  # double the numbers length
        flip_a = 0  # count numbers to flip to get current tagret A in a loop
        flip_b = 0  # count numbers to flip to get current tagret B in a loop
        min_flip_a = len(numbers)  # minimum flips to get the best matching target A
        min_flip_b = len(numbers)  # minimum flips to get the best matching target B


        for right, number in enumerate(numbers):
            # if no match increase the flip counter
            if number != target_a[right]:
                flip_a += 1
            if number != target_b[right]:
                flip_b += 1

            # if the window is the right length
            if right - left + 1 == len(numbers) / 2:
                min_flip_a = min(min_flip_a, flip_a)
                min_flip_b = min(min_flip_b, flip_b)

                # early exit when any of min flips is zero
                if not min_flip_a or not min_flip_b:
                    return 0

                # if no match decrease the flip counter
                if numbers[left] != target_a[left]:
                    flip_a -= 1
                if numbers[left] != target_b[left]:
                    flip_b -= 1

                left += 1

        return min(min_flip_a, min_flip_b)


# O(n2), O(n)
# brute force
class Solution:
    def minFlips(self, numbers: str) -> int:
        min_flip_a = len(numbers)
        min_flip_b = len(numbers)
        
        if len(numbers) % 2:
            target_a = "01" * (len(numbers) // 2) + "0"
            target_b = "10" * (len(numbers) // 2) + "1"
        else:
            target_a = "01" * (len(numbers) // 2)
            target_b = "10" * (len(numbers) // 2)

        for _ in range(len(numbers)):
            numbers = numbers[1:] + numbers[0]
            flip_a = 0
            flip_b = 0

            for index, number in enumerate(numbers):
                if number != target_a[index]:
                    flip_a += 1

                if number != target_b[index]:
                    flip_b += 1

            min_flip_a = min(min_flip_a, flip_a)
            min_flip_b = min(min_flip_b, flip_b)
        
        return min(min_flip_a, min_flip_b)





# Online Stock Span
# https://leetcode.com/problems/online-stock-span/description/
"""
Design an algorithm that collects daily price quotes for some stock and returns the span of that stock's price for the current day.

The span of the stock's price in one day is the maximum number of consecutive days (starting from that day and going backward) for which the stock price was less than or equal to the price of that day.

For example, if the prices of the stock in the last four days is [7,2,1,2] and the price of the stock today is 2, then the span of today is 4 because starting from today, the price of the stock was less than or equal 2 for 4 consecutive days.
Also, if the prices of the stock in the last four days is [7,34,1,2] and the price of the stock today is 8, then the span of today is 3 because starting from today, the price of the stock was less than or equal 8 for 3 consecutive days.
Implement the StockSpanner class:

StockSpanner() Initializes the object of the class.
int next(int price) Returns the span of the stock's price given that today's price is price.
 

Example 1:

Input
["StockSpanner", "next", "next", "next", "next", "next", "next", "next"]
[[], [100], [80], [60], [70], [60], [75], [85]]
Output
[null, 1, 1, 1, 2, 1, 4, 6]

Explanation
StockSpanner stockSpanner = new StockSpanner();
stockSpanner.next(100); // return 1
stockSpanner.next(80);  // return 1
stockSpanner.next(60);  // return 1
stockSpanner.next(70);  // return 2
stockSpanner.next(60);  // return 1
stockSpanner.next(75);  // return 4, because the last 4 prices (including today's price of 75) were less than or equal to today's price.
stockSpanner.next(85);  // return 6
"""


# 100
# 100, 80
# 100, 80, 60
# 100, 80, 70
# 100, 80, 70, 60
# 100, 80, 70, 60


# O(n), O(n)
# stack
class StockSpanner:
    def __init__(self):
        self.prices = []  # (price, counter)

    def next(self, price: int) -> int:
        counter = 1  # counts the same or lower prices as the price
        right = len(self.prices) - 1  # the top of the stack index

        while (right >= 0 and  # while index in bounds and
               price >= self.prices[-1][0]):  # price higher of equal to the price of the top of the stack
            _, current_counter = self.prices.pop()  # pop that price and get counter
            counter += current_counter  # increase the counter
            right -= 1  # decrease index

        self.prices.append((price, counter))  # add (price, counter)

        return counter


# Your StockSpanner object will be instantiated and called as such:
# obj = StockSpanner()
# param_1 = obj.next(price)


# O(n2), O(n)
# brute force
class StockSpanner:
    def __init__(self):
        self.prices = []

    def next(self, price: int) -> int:
        self.prices.append(price)
        counter = 0

        for current_price in reversed(self.prices):
            if price < current_price:
                break

            counter += 1

        return counter


# O(n2), O(n)
# brute force
class StockSpanner:
    def __init__(self):
        self.prices = []

    def next(self, price: int) -> int:
        self.prices.append(price)
        counter = 0
        right = len(self.prices) - 1

        while (right >= 0 and
               price >= self.prices[right]):
            counter += 1
            right -= 1

        return counter





# Valid Perfect Square
# https://leetcode.com/problems/valid-perfect-square/description/
"""
Given a positive integer num, return true if num is a perfect square or false otherwise.

A perfect square is an integer that is the square of an integer. In other words, it is the product of some integer with itself.

You must not use any built-in library function, such as sqrt.

Example 1:

Input: num = 16
Output: true
Explanation: We return true because 4 * 4 = 16 and 4 is an integer.
Example 2:

Input: num = 14
Output: false
Explanation: We return false because 3.742 * 3.742 = 14 and 3.742 is not an integer.
"""
print(Solution().isPerfectSquare(16), True)
print(Solution().isPerfectSquare(14), False)


# O(logn), O(1)
# binary search
class Solution:
    def isPerfectSquare(self, number: int) -> bool:
        left = 0
        right = number

        while left <= right:
            middle = (left + right) // 2
            square = middle ** 2

            if square == number:
                return True
            elif square > number:
                right = middle - 1
            else:
                left = middle + 1

        return False





# Merge In Between Linked Lists
# https://leetcode.com/problems/merge-in-between-linked-lists/description/
"""
You are given two linked lists: list1 and list2 of sizes n and m respectively.

Remove list1's nodes from the ath node to the bth node, and put list2 in their place.

The blue edges and nodes in the following figure indicate the result:

Build the result list and return its head.

Example 1:

Input: list1 = [10,1,13,6,9,5], a = 3, b = 4, list2 = [1000000,1000001,1000002]
Output: [10,1,13,1000000,1000001,1000002,5]
Explanation: We remove the nodes 3 and 4 and put the entire list2 in their place. The blue edges and nodes in the above figure indicate the result.
Example 2:


Input: list1 = [0,1,2,3,4,5,6], a = 2, b = 5, list2 = [1000000,1000001,1000002,1000003,1000004]
Output: [0,1,1000000,1000001,1000002,1000003,1000004,6]
Explanation: The blue edges and nodes in the above figure indicate the result.
"""


# O(n), O(1)
# linked list
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeInBetween(self, head1: ListNode, a: int, b: int, head2: ListNode) -> ListNode:
        node = ListNode(0, head1)
        counter = 0

        while counter - 1 != b: # 0!=5,
            if counter == a: # 0==3,
                start = node  # node before start # start = 6

            counter += 1  # 1, 2, 3, 4
            node = node.next  # 10, 1, 13, 6
        
        end = node  # node before end  # 9

        start.next = head2  # inject head2 after start

        while head2.next:  # find the end of the inserted list
            head2 = head2.next  # node points to the element before None # 1000002

        head2.next = end.next  # switch end in list2

        return head1

nodeA5 = ListNode(5)
nodeA4 = ListNode(9, nodeA5)
nodeA3 = ListNode(6, nodeA4)
nodeA2 = ListNode(13, nodeA3)
nodeA1 = ListNode(1, nodeA2)
nodeA0 = ListNode(10, nodeA1)

nodeB2 = ListNode(1000002)
nodeB1 = ListNode(1000001, nodeB2)
nodeB0 = ListNode(1000000, nodeB1)

print(Solution().mergeInBetween(nodeA0, 3, 4, nodeB0))





# Next Greater Element I
# https://leetcode.com/problems/next-greater-element-i/description/
"""
The next greater element of some element x in an array is the first greater element that is to the right of x in the same array.

You are given two distinct 0-indexed integer arrays nums1 and nums2, where nums1 is a subset of nums2.

For each 0 <= i < nums1.length, find the index j such that nums1[i] == nums2[j] and determine the next greater element of nums2[j] in nums2. If there is no next greater element, then the answer for this query is -1.

Return an array ans of length nums1.length such that ans[i] is the next greater element as described above.
 

Example 1:

Input: nums1 = [4,1,2], nums2 = [1,3,4,2]
Output: [-1,3,-1]
Explanation: The next greater element for each value of nums1 is as follows:
- 4 is underlined in nums2 = [1,3,4,2]. There is no next greater element, so the answer is -1.
- 1 is underlined in nums2 = [1,3,4,2]. The next greater element is 3.
- 2 is underlined in nums2 = [1,3,4,2]. There is no next greater element, so the answer is -1.
Example 2:

Input: nums1 = [2,4], nums2 = [1,2,3,4]
Output: [3,-1]
Explanation: The next greater element for each value of nums1 is as follows:
- 2 is underlined in nums2 = [1,2,3,4]. The next greater element is 3.
- 4 is underlined in nums2 = [1,2,3,4]. There is no next greater element, so the answer is -1.
"""
print(Solution().nextGreaterElement([4, 1, 2], [1, 3, 4, 2]), [-1, 3, -1])
print(Solution().nextGreaterElement([2, 4], [1, 2, 3, 4]), [3, -1])
print(Solution().nextGreaterElement([1, 3, 5, 2, 4], [6, 5, 4, 3, 2, 1, 7]), [7, 7, 7, 7, 7])


# O(n), O(1)
# stack
class Solution:
    def nextGreaterElement(self, numbers1: list[int], numbers2: list[int]) -> list[int]:
        next_greater = [-1] * len(numbers1)
        stack = []
        numbers1_index = {number: index
                          for index, number in enumerate(numbers1)}

        for number in numbers2:
            while stack and number > stack[-1]:
                value = stack.pop()
                index = numbers1_index[value]
                next_greater[index] = number

            if number in numbers1_index:
                stack.append(number)

        return next_greater


# O(n2), O(1)
# brute force
class Solution:
    def nextGreaterElement(self, numbers1: list[int], numbers2: list[int]) -> list[int]:
        next_greater = [-1] * len(numbers1)

        for index, number in enumerate(numbers1):
            next_right = numbers2.index(number) + 1

            for index_right in range(next_right, len(numbers2)):
                if numbers2[index_right] > number:
                    next_greater[index] = numbers2[index_right]
                    break

        return next_greater





# Minimum Size Subarray Sum
# https://leetcode.com/problems/minimum-size-subarray-sum/description/
"""
Given an array of positive integers nums and a positive integer target, return the minimal length of a 
subarray
 whose sum is greater than or equal to target. If there is no such subarray, return 0 instead.

Example 1:

Input: target = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: The subarray [4,3] has the minimal length under the problem constraint.
Example 2:

Input: target = 4, nums = [1,4,4]
Output: 1
Example 3:

Input: target = 11, nums = [1,1,1,1,1,1,1,1]
Output: 0
"""
print(Solution().minSubArrayLen(7, [2, 3, 1, 2, 4, 3]), 2)
print(Solution().minSubArrayLen(4, [1, 4, 4]), 1)
print(Solution().minSubArrayLen(11, [1, 1, 1, 1, 1, 1, 1, 1]), 0)


# O(n), O(1)
# sliding window
class Solution:
    def minSubArrayLen(self, target: int, numbers: list[int]) -> int:
        left = 0
        window_sum = 0
        window_size = len(numbers) + 1

        for right, number in enumerate(numbers):
            window_sum += number

            while window_sum >= target:
                window_size = min(window_size, right - left + 1)
                window_sum -= numbers[left]
                left += 1

        return window_size if window_size != len(numbers) + 1 else 0





# Backspace String Compare
# https://leetcode.com/problems/backspace-string-compare/description/
"""
Given two strings s and t, return true if they are equal when both are typed into empty text editors. '#' means a backspace character.

Note that after backspacing an empty text, the text will continue empty.

Example 1:

Input: s = "ab#c", t = "ad#c"
Output: true
Explanation: Both s and t become "ac".
Example 2:

Input: s = "ab##", t = "c#d#"
Output: true
Explanation: Both s and t become "".
Example 3:

Input: s = "a#c", t = "b"
Output: false
Explanation: s becomes "c" while t becomes "b".
"""
print(Solution().backspaceCompare("ab#c", "ad#c"), True)
print(Solution().backspaceCompare("ab##", "c#d#"), True)
print(Solution().backspaceCompare("a#c", "b"), False)
print(Solution().backspaceCompare("xywrrmp", "xywrrmu#p"), True)
print(Solution().backspaceCompare("nzp#o#g", "b#nzp#o#g"), True)


# O(n), O(1)
# two pointers
class Solution:
    """
    Find the first valid character left to `index`.
    """
    def next_valid_char(self, index, text):
        joker = 0

        while (index >= 0 and 
               (text[index] == "#" or joker)):
            joker += 1 if text[index] == "#" else -1            
            index -= 1
        
        return index

    def backspaceCompare(self, text1: str, text2: str) -> bool:
        index1 = len(text1) - 1
        index2 = len(text2) - 1
        
        while index1 >= 0 and index2 >= 0:
            # find the next valid character from the end
            index1 = self.next_valid_char(index1, text1)
            index2 = self.next_valid_char(index2, text2)

            # if characters doesn't match
            if text1[index1] != text2[index2]:
                return False

            index1 -= 1
            index2 -= 1
            
            # both texts folded to empty string
            if index1 == 0 and index2 == 0:
                return True

        # when one index is -1 (string folded to empty string) and the other is not, the other may still fold to empty string
        return (self.next_valid_char(index1, text1) == 
                self.next_valid_char(index2, text2))


# O(n), O(n)
# stack
class Solution:
    def clean_text(self, text):
        stack = []
        
        for letter in text:
            if letter == "#":
                stack and stack.pop()
            else:
                stack.append(letter)
        
        return "".join(stack)

    def backspaceCompare(self, text1: str, text2: str) -> bool:
        text1_clean = self.clean_text(text1)
        text2_clean = self.clean_text(text2)

        return text1_clean == text2_clean





# Find Pivot Index
# https://leetcode.com/problems/find-pivot-index/description/
"""
Given an array of integers nums, calculate the pivot index of this array.

The pivot index is the index where the sum of all the numbers strictly to the left of the index is equal to the sum of all the numbers strictly to the index's right.

If the index is on the left edge of the array, then the left sum is 0 because there are no elements to the left. This also applies to the right edge of the array.

Return the leftmost pivot index. If no such index exists, return -1.

 

Example 1:

Input: nums = [1,7,3,6,5,6]
Output: 3
Explanation:
The pivot index is 3.
Left sum = nums[0] + nums[1] + nums[2] = 1 + 7 + 3 = 11
Right sum = nums[4] + nums[5] = 5 + 6 = 11
Example 2:

Input: nums = [1,2,3]
Output: -1
Explanation:
There is no index that satisfies the conditions in the problem statement.
Example 3:

Input: nums = [2,1,-1]
Output: 0
Explanation:
The pivot index is 0.
Left sum = 0 (no elements to the left of index 0)
Right sum = nums[1] + nums[2] = 1 + -1 = 0
"""
print(Solution().pivotIndex([1, 7, 3, 6, 5, 6]), 3)
print(Solution().pivotIndex([1, 2, 3]), -1)
print(Solution().pivotIndex([2, 1, -1]), 0)


# O(n), O(1)
# prefix sum
class Solution:
    def pivotIndex(self, numbers: list[int]) -> int:
        right_sum = sum(numbers)
        left_sum = 0

        for index, number in enumerate(numbers):
            right_sum -= number
            left_sum += numbers[index - 1] if index else 0

            if left_sum == right_sum:
                return index

        return -1





# Find K Closest Elements
# https://leetcode.com/problems/find-k-closest-elements/description/
"""
Given a sorted integer array arr, two integers k and x, return the k closest integers to x in the array. The result should also be sorted in ascending order.

An integer a is closer to x than an integer b if:

|a - x| < |b - x|, or
|a - x| == |b - x| and a < b
 

Example 1:

Input: arr = [1,2,3,4,5], k = 4, x = 3

Output: [1,2,3,4]

Example 2:

Input: arr = [1,1,2,3,4,5], k = 4, x = -1

Output: [1,1,2,3]
"""
print(Solution().findClosestElements([1, 1, 2, 3, 4, 5], 4, -1), [1, 1, 2, 3])
print(Solution().findClosestElements([1, 2, 3, 4, 5], 4, 3), [1, 2, 3, 4])

# [2, 1, 0, 1, 2]
# [0, 0, 1, 2, 3, 4]


# O(n), O(n)
# brute force
from collections import deque
class Solution:
    def findClosestElements(self, numbers: list[int], k: int, x: int) -> list[int]:
        numbers = deque(numbers)

        while len(numbers) > k:
            if x - numbers[0] > numbers[-1] - x:
                numbers.popleft()
            else:
                numbers.pop()

        return list(numbers)


# O(n), O(1)
# sliding window
class Solution:
    def findClosestElements(self, numbers: list[int], k: int, x: int) -> list[int]:
        left = 0
        right = len(numbers) - 1

        while right - left + 1 > k:
            if x - numbers[0] > numbers[-1] - x:
                left += 1
            else:
                right -= 1

        return numbers[left: right + 1]


# O(log(n-k)+k), O(1)
# sliding window
class Solution:
    def findClosestElements(self, numbers: list[int], k: int, x: int) -> list[int]:
        left = 0
        right = len(numbers) - k  # 2

        while left < right:  # O(log(n-k))
            middle = (left + right) // 2  # 1, 0
            if numbers[middle + k] - x < x - numbers[middle]:  # 5--1=6<-1-1=-2; 4--1=5<-1-1=-2
                left = middle + 1
            else:
                right = middle  # 1, 0

        return numbers[left: left + k]  # O(k))




class Solution:
    def findClosestElements(self, numbers: list[int], k: int, x: int) -> list[int]:
        left = 0
        right = len(numbers) - 1

        while right - left + 1 != k:
            if ((x - numbers[left]) < (numbers[right] - x) or
                    (x - numbers[left]) == (numbers[right] - x) and
                    numbers[left] < numbers[right]):
                right -= 1
            else:
                left += 1

        return numbers[left: right + 1]





# Range Sum Query - Immutable
# https://leetcode.com/problems/range-sum-query-immutable/description/
"""
Given an integer array nums, handle multiple queries of the following type:

Calculate the sum of the elements of nums between indices left and right inclusive where left <= right.
Implement the NumArray class:

NumArray(int[] nums) Initializes the object with the integer array nums.
int sumRange(int left, int right) Returns the sum of the elements of nums between indices left and right inclusive (i.e. nums[left] + nums[left + 1] + ... + nums[right]).
 

Example 1:

Input
["NumArray", "sumRange", "sumRange", "sumRange"]
[[[-2, 0, 3, -5, 2, -1]], [0, 2], [2, 5], [0, 5]]
Output
[null, 1, -1, -3]

Explanation
NumArray numArray = new NumArray([-2, 0, 3, -5, 2, -1]);
numArray.sumRange(0, 2); // return (-2) + 0 + 3 = 1
numArray.sumRange(2, 5); // return 3 + (-5) + 2 + (-1) = -1
numArray.sumRange(0, 5); // return (-2) + 0 + 3 + (-5) + 2 + (-1) = -3
"""


# O(1), O(n)
# prefix sum
class NumArray:
    def __init__(self, numbers: list[int]):
        prefix = 0
        self.numbers = []

        for number in numbers:
            prefix += number
            self.prefixes.append(prefix)

    def sumRange(self, left: int, right: int) -> int:
        if not left:
            return self.prefixes[right]
        else:
            return self.prefixes[right] - self.prefixes[left - 1]


# O(n), O(1)
class NumArray:
    def __init__(self, numbers: list[int]):
        self.numbers = numbers

    def sumRange(self, left: int, right: int) -> int:
        return sum(self.numbers[left: right + 1])

# Your NumArray object will be instantiated and called as such:
# obj = NumArray(numbers)
# param_1 = obj.sumRange(left,right)





# Check If Two String Arrays are Equivalent
# https://leetcode.com/problems/check-if-two-string-arrays-are-equivalent/description/
"""
Given two string arrays word1 and word2, return true if the two arrays represent the same string, and false otherwise.

A string is represented by an array if the array elements concatenated in order forms the string.

 

Example 1:

Input: word1 = ["ab", "c"], word2 = ["a", "bc"]
Output: true
Explanation:
word1 represents string "ab" + "c" -> "abc"
word2 represents string "a" + "bc" -> "abc"
The strings are the same, so return true.
Example 2:

Input: word1 = ["a", "cb"], word2 = ["ab", "c"]
Output: false
Example 3:

Input: word1  = ["abc", "d", "defg"], word2 = ["abcddefg"]
Output: true
"""
print(Solution().arrayStringsAreEqual(["ab", "c"], ["a", "bc"]), True)
print(Solution().arrayStringsAreEqual(["a", "cb"], ["ab", "c"]), False)
print(Solution().arrayStringsAreEqual(["abc", "d", "defg"], ["abcddefg"]), True)
print(Solution().arrayStringsAreEqual(["abc", "d", "defg"], ["abcddef"]), False)


# O(n), O(n)
# build-in function
class Solution:
    def arrayStringsAreEqual(self, words1: list[str], words2: list[str]) -> bool:
        return "".join(words1) == "".join(words2)


# O(n), O(1)
# two pointers
class Solution:
    def arrayStringsAreEqual(self, words1: list[str], words2: list[str]) -> bool:
        index_word1 = 0  # index for a word
        index_word2 = 0
        index_letter1 = 0  # index for a letter in the word
        index_letter2 = 0

        while (index_word1 < len(words1) and 
               index_word2 < len(words2)):
            if words1[index_word1][index_letter1] != words2[index_word2][index_letter2]:
                return False

            index_letter1 += 1  # get next letter index
            index_letter2 += 1

            # if end of the word reached
            if index_letter1 == len(words1[index_word1]):
                index_letter1 = 0  # zero the letter index
                index_word1 += 1  # get next word index

            if index_letter2 == len(words2[index_word2]):
                index_letter2 = 0
                index_word2 += 1

        # both index_word have to get out of bounds to strings to be equal
        return (index_word1 == len(words1) and
                index_word2 == len(words2))






# Minimum Operations to Reduce X to Zero
# https://leetcode.com/problems/minimum-operations-to-reduce-x-to-zero/description/
"""
You are given an integer array nums and an integer x. In one operation, you can either remove the leftmost or the rightmost element from the array nums and subtract its value from x. Note that this modifies the array for future operations.

Return the minimum number of operations to reduce x to exactly 0 if it is possible, otherwise, return -1.

 

Example 1:

Input: nums = [1,1,4,2,3], x = 5
Output: 2
Explanation: The optimal solution is to remove the last two elements to reduce x to zero.
Example 2:

Input: nums = [5,6,7,8,9], x = 4
Output: -1
Example 3:

Input: nums = [3,2,20,1,1,3], x = 10
Output: 5
Explanation: The optimal solution is to remove the last three elements and the first two elements (5 operations in total) to reduce x to zero.
"""
print(Solution().minOperations([1, 1, 4, 2, 3], 5), 2)
print(Solution().minOperations([5, 6, 7, 8, 9], 4), -1)
print(Solution().minOperations([3, 2, 20, 1, 1, 3], 10), 5)
print(Solution().minOperations([5, 2, 3, 1, 1], 5), 1)
print(Solution().minOperations([8828, 9581, 49, 9818, 9974, 9869, 9991, 10000, 10000, 10000, 9999, 9993, 9904, 8819, 1231, 6309], 134365), 16)
print(Solution().minOperations([1, 1], 3), -1)


# O(n), O(1)
# sliding window
class Solution:
    def minOperations(self, numbers: list[int], x: int) -> int:
        left = 0
        window_sum = 0
        window_min_length = len(numbers) + 1
        target = sum(numbers) - x

        for right, number in enumerate(numbers):
            window_sum += number

            while (left <= right and  # target might be less than zero and left gets out of bounds
                   window_sum > target):
                window_sum -= numbers[left]
                left += 1

            if window_sum == target:
                window_min_length = min(window_min_length, len(numbers) - (right - left + 1))

        return window_min_length if window_min_length != len(numbers) + 1 else - 1



# Simplify Path
# https://leetcode.com/problems/simplify-path/description/
"""
You are given an absolute path for a Unix-style file system, which always begins with a slash '/'. Your task is to transform this absolute path into its simplified canonical path.

The rules of a Unix-style file system are as follows:

A single period '.' represents the current directory.
A double period '..' represents the previous/parent directory.
Multiple consecutive slashes such as '//' and '///' are treated as a single slash '/'.
Any sequence of periods that does not match the rules above should be treated as a valid directory or file name. For example, '...' and '....' are valid directory or file names.
The simplified canonical path should follow these rules:

The path must start with a single slash '/'.
Directories within the path must be separated by exactly one slash '/'.
The path must not end with a slash '/', unless it is the root directory.
The path must not have any single or double periods ('.' and '..') used to denote current or parent directories.
Return the simplified canonical path.

 

Example 1:

Input: path = "/home/"

Output: "/home"

Explanation:

The trailing slash should be removed.

Example 2:

Input: path = "/home//foo/"

Output: "/home/foo"

Explanation:

Multiple consecutive slashes are replaced by a single one.

Example 3:

Input: path = "/home/user/Documents/../Pictures"

Output: "/home/user/Pictures"

Explanation:

A double period ".." refers to the directory up a level (the parent directory).

Example 4:

Input: path = "/../"

Output: "/"

Explanation:

Going one level up from the root directory is not possible.

Example 5:

Input: path = "/.../a/../b/c/../d/./"

Output: "/.../b/d"

Explanation:

"..." is a valid name for a directory in this problem.
"""
print(Solution().simplifyPath("/home/"), "/home")
print(Solution().simplifyPath("/home//foo/"), "/home/foo")
print(Solution().simplifyPath("/home/user/Documents/../Pictures"), "/home/user/Pictures")
print(Solution().simplifyPath("/../"), "/")
print(Solution().simplifyPath("/.../a/../b/c/../d/./"), "/.../b/d")
print(Solution().simplifyPath("/a/../../b/../c//.//"), "/c")
print(Solution().simplifyPath("/."), "/")
print(Solution().simplifyPath("/..hidden"), "/..hidden")


class Solution:
    def simplifyPath(self, path: str) -> str:
        stack = []
        cache = ""

        for char in (path + "/"):
            if char == "/":
                if cache == "..":
                    if stack:
                        stack.pop()
                elif cache and cache != ".":
                    stack.append(cache)
                    
                cache = ""
            else:
                cache += char
            
        return "/" + "/".join(stack)
            




# Sqrt(x)
# https://leetcode.com/problems/sqrtx/description/
"""
Given a non-negative integer x, return the square root of x rounded down to the nearest integer. The returned integer should be non-negative as well.

You must not use any built-in exponent function or operator.

For example, do not use pow(x, 0.5) in c++ or x ** 0.5 in python.
 

Example 1:

Input: x = 4
Output: 2
Explanation: The square root of 4 is 2, so we return 2.
Example 2:

Input: x = 8
Output: 2
Explanation: The square root of 8 is 2.82842..., and since we round it down to the nearest integer, 2 is returned.
"""
print(Solution().mySqrt(4), 2)
print(Solution().mySqrt(8), 2)

class Solution:
    def mySqrt(self, x: int) -> int:
        left = 1
        right = x

        while left <= right:
            middle = (left + right) // 2

            if middle ** 2 == x:
                return middle
            elif middle ** 2 > x:
                right = middle - 1
            else:
                left = middle + 1

        return right




# Find All Numbers Disappeared in an Array
# https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/description/
"""
Given an array nums of n integers where nums[i] is in the range [1, n], return an array of all the integers in the range [1, n] that do not appear in nums.

Example 1:

Input: nums = [4,3,2,7,8,2,3,1]
Output: [5,6]
Example 2:

Input: nums = [1,1]
Output: [2]
"""
print(Solution().findDisappearedNumbers([4, 3, 2, 7, 8, 2, 3, 1]), [5, 6])
print(Solution().findDisappearedNumbers([1, 1]), [2])


# O(n), O(n)
# dict
class Solution:
    def findDisappearedNumbers(self, numbers: list[int]) -> list[int]:
        number_set = set(range(1, len(numbers) + 1))

        for num in set(numbers):
            if num in number_set:
                number_set.discard(num)

        return list(number_set)


# O(n), O(1) # You may assume the returned list does not count as extra space.
# mutate input array
class Solution:
    def findDisappearedNumbers(self, numbers: list[int]) -> list[int]:
        for number in numbers:
            numbers[abs(number) - 1] = -abs(numbers[abs(number) - 1])

        return [index for
                index, number in enumerate(numbers, 1)
                if number > 0]





# Binary Tree Inorder Traversal
# https://leetcode.com/problems/binary-tree-inorder-traversal/description/
"""
Given the root of a binary tree, return the inorder traversal of its nodes' values.

Example 1:

Input: root = [1,null,2,3]

Output: [1,3,2]

Explanation:
1__
   \
    2
   /
  3


Example 2:

Input: root = [1,2,3,4,5,null,8,null,null,6,7,9]

Output: [4,2,6,5,7,1,3,9,8]

Explanation:
    ______1
   /       \
  2__       3__
 /   \         \
4     5         8
     / \       /
    6   7     9


Example 3:

Input: root = []

Output: []

Example 4:

Input: root = [1]

Output: [1]
"""
print(Solution().inorderTraversal(build_tree_from_list([1,None,2,3], Node)), [1, 3, 2])
print(Solution().inorderTraversal(build_tree_from_list([1,2,3,4,5,None,8,None,None,6,7,9])), [4,2,6,5,7,1,3,9,8])
print(Solution().inorderTraversal(build_tree_from_list([], Node)), [])
print(Solution().inorderTraversal(build_tree_from_list([1], Node)), [1])


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# O(n), O(n)
# dfs, recursion
class Solution:
    def inorderTraversal(self, root: TreeNode | None) -> list[int]:
        node_list = []

        def dfs(node):
            if not node:
                return

            dfs(node.left)  # traverse left
            node_list.append(node.val)
            dfs(node.right)  # traverse right
            
        dfs(root)

        return node_list
    

# O(n), O(n)
# dfs, iterative, stack
class Solution:
    def inorderTraversal(self, root: TreeNode | None) -> list[int]:
        node_list = []
        stack = []
        node = root

        while node or stack:
            while node:
                stack.append(node)
                node = node.left
            
            node = stack.pop()
            node_list.append(node.val)
            node = node.right

        return node_list





# Maximum Number of Balloons
# https://leetcode.com/problems/maximum-number-of-balloons/
"""
Given a string text, you want to use the characters of text to form as many instances of the word "balloon" as possible.

You can use each character in text at most once. Return the maximum number of instances that can be formed.

Example 1:

Input: text = "nlaebolko"
Output: 1
Example 2:

Input: text = "loonbalxballpoon"
Output: 2
Example 3:

Input: text = "leetcode"
Output: 0
"""
print(Solution().maxNumberOfBalloons("nlaebolko"), 1)
print(Solution().maxNumberOfBalloons("loonbalxballpoon"), 2)
print(Solution().maxNumberOfBalloons("leetcode"), 0)
print(Solution().maxNumberOfBalloons("balon"), 0)


class Solution:
    def maxNumberOfBalloons(self, text: str) -> int:
        balloon = {"b": 1, "a": 1, "l": 2, "o": 2, "n": 1}
        text_map = {}
        max_balloons = len(text)

        for letter in text:
            text_map[letter] = text_map.get(letter, 0) + 1
        
        for letter in balloon:
            if letter not in text_map:
                return 0

            max_balloons = min(max_balloons, text_map[letter] // balloon[letter])

        return max_balloons




# Binary Tree Preorder Traversal
# https://leetcode.com/problems/binary-tree-preorder-traversal/description/
"""
Given the root of a binary tree, return the preorder traversal of its nodes' values.

 

Example 1:

Input: root = [1,null,2,3]

Output: [1,2,3]

Explanation:
1__
   \
    2
   /
  3


Example 2:

Input: root = [1,2,3,4,5,null,8,null,null,6,7,9]

Output: [1,2,4,5,6,7,3,8,9]

Explanation:
    ______1
   /       \
  2__       3__
 /   \         \
4     5         8
     / \       /
    6   7     9


Example 3:

Input: root = []

Output: []

Example 4:

Input: root = [1]

Output: [1]
"""
print(Solution().preorderTraversal(build_tree_from_list([1, 2, 3, 4, 5, None, 8, None, None, 6, 7, 9], TreeNode)), [1, 2, 4, 5, 6, 7, 3, 8, 9])
print(Solution().preorderTraversal(build_tree_from_list([1, None, 2, 3], TreeNode)), [1, 2, 3])
print(Solution().preorderTraversal(build_tree_from_list([], TreeNode)), [])
print(Solution().preorderTraversal(build_tree_from_list([1], TreeNode)), [1])


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# 
# O(n), O(n)
# dfs, recursion
class Solution:
    def preorderTraversal(self, root: TreeNode) -> list[int]:
        node_list = []  # initialize the preorder traversal value list

        def dfs(node):
            if not node:  # if node is None return
                return

            node_list.append(node.val)  # add its value to node list
            dfs(node.left)  # traverse left
            dfs(node.right)  # traverse right

        dfs(root)

        return node_list
    

# O(n), O(n)
# dfs, iteration, stack
class Solution:
    def preorderTraversal(self, root: TreeNode) -> list[int]:
        if not root:
            return []

        node_list = []  # initialize the preorder traversal value list
        stack = [root]  # Initialize the stack with the root node

        while stack:
            for _ in stack:
                node = stack.pop()  # Pop the current node
                node_list.append(node.val)  # add its value to node list

                if node.right:  # Add left child to the stack if it exists
                    stack.append(node.right)
                if node.left:  # Add right child to the stack if it exists
                    stack.append(node.left)

        return node_list


# O(n), O(n)
# dfs, iteration, stack
class Solution:
    def preorderTraversal(self, root: TreeNode) -> list[int]:
        node_list = []  # initialize the preorder traversal value list
        stack = []  # Initialize an empty stack
        node = root  # Start traversal from the root node

        while stack or node:
            if node:
                node_list.append(node.val)  # Add current node's value to the result
                stack.append(node.right)  # Add right child to the stack even if it's null
                node = node.left  # Pop the current node
            else:
                node = stack.pop()  # Backtrack to the last right child

        return node_list





# Binary Tree Postorder Traversal
# https://leetcode.com/problems/binary-tree-postorder-traversal/description/
"""
Given the root of a binary tree, return the postorder traversal of its nodes' values.

 

Example 1:

Input: root = [1,null,2,3]

Output: [3,2,1]

Explanation:
1__
   \
    2
   /
  3


Example 2:

Input: root = [1,2,3,4,5,null,8,null,null,6,7,9]

Output: [4,6,7,5,2,9,8,3,1]

Explanation:
    ______1
   /       \
  2__       3__
 /   \         \
4     5         8
     / \       /
    6   7     9


Example 3:

Input: root = []

Output: []

Example 4:

Input: root = [1]

Output: [1]
"""
print(Solution().postorderTraversal(build_tree_from_list([1, 2, 3, 4, 5, None, 8, None, None, 6, 7, 9], TreeNode)), [4, 6, 7, 5, 2, 9, 8, 3, 1])
print(Solution().postorderTraversal(build_tree_from_list([1, None, 2, 3], TreeNode)), [3, 2, 1])
print(Solution().postorderTraversal(build_tree_from_list([], TreeNode)), [])
print(Solution().postorderTraversal(build_tree_from_list([1], TreeNode)), [1])


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# O(n), O(n)
# dfs, recursion
class Solution:
    def postorderTraversal(self, root: TreeNode) -> list[int]:
        node_list = []  # initialize the preorder traversal value list

        def dfs(node):
            if not node:  # if node is None return
                return

            dfs(node.left)  # traverse left
            dfs(node.right)  # traverse right
            node_list.append(node.val)  # add its value to node list

        dfs(root)

        return node_list


# O(n), O(n)
# dfs, iteration, stack
class Solution:
    def postorderTraversal(self, root: TreeNode) -> list[int]:
        node_list = []  # initialize the preorder traversal value list
        stack = [root]  # Initialize the stack with the root node
        visited = [False]

        while stack:
            node = stack.pop()  # Pop the current node
            visit = visited.pop()

            if node:
                if visit:
                    node_list.append(node.val)  # add its value to node list
                else:
                    stack.append(node)  # Add node to the stack
                    visited.append(True)
                    stack.append(node.right)  # Add left child to the stack
                    visited.append(False)
                    stack.append(node.left)  # Add right child to the stack
                    visited.append(False)

        return node_list





# Word Pattern
# https://leetcode.com/problems/word-pattern/description/
"""
Given a pattern and a string s, find if s follows the same pattern.

Here follow means a full match, such that there is a bijection between a letter in pattern and a non-empty word in s. Specifically:

Each letter in pattern maps to exactly one unique word in s.
Each unique word in s maps to exactly one letter in pattern.
No two letters map to the same word, and no two words map to the same letter.
 

Example 1:

Input: pattern = "abba", s = "dog cat cat dog"

Output: true

Explanation:

The bijection can be established as:

'a' maps to "dog".
'b' maps to "cat".
Example 2:

Input: pattern = "abba", s = "dog cat cat fish"

Output: false

Example 3:

Input: pattern = "aaaa", s = "dog cat cat dog"

Output: false
"""
print(Solution().wordPattern("abba", "dog cat cat dog"), True)
print(Solution().wordPattern("abba", "dog cat cat fish"), False)
print(Solution().wordPattern("aaaa", "dog cat cat dog"), False)

# O(n), O(n)
# dictionary, set
class Solution:
    def wordPattern(self, pattern: str, text: str) -> bool:
        text_list = text.split()  # split text into words
        text_set = set()  # unique word set
        pattern_text = {}  # letter to word map
        
        if len(pattern) != len(text_list):  # if text list length and pattern length are not the same
            return False

        for index, letter in enumerate(pattern):
            if letter in pattern_text:
                if pattern_text[letter] != text_list[index]:  # if letter is already mapped to another word
                    return False
            else:
                word = text_list[index]  # current word
                
                if word in text_set:  # if another letter is mapped to current word
                    return False
                
                pattern_text[letter] = word  # update `letter to word` map
                text_set.add(word)  # add word to unique set
        
        return True





# Remove Duplicates from Sorted Array II
# https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/description/
"""
Given an integer array nums sorted in non-decreasing order, remove some duplicates in-place such that each unique element appears at most twice. The relative order of the elements should be kept the same.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.


Example 1:

Input: nums = [1,1,1,2,2,3]
Output: 5, nums = [1,1,2,2,3,_]
Explanation: Your function should return k = 5, with the first five elements of nums being 1, 1, 2, 2 and 3 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
Example 2:

Input: nums = [0,0,1,1,1,1,2,3,3]
Output: 7, nums = [0,0,1,1,2,3,3,_,_]
Explanation: Your function should return k = 7, with the first seven elements of nums being 0, 0, 1, 1, 2, 3 and 3 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
"""
print(Solution().removeDuplicates([1, 1, 1, 2, 2, 3]), 5)
print(Solution().removeDuplicates([0, 0, 1, 1, 1, 1, 2, 3, 3]), 7)


# O(n), O(1)
# two pointers
class Solution:
    def removeDuplicates(self, numbers: list[int]) -> int:
        left = 0

        for right, number in enumerate(numbers):
            if (right < 2 or  # the first two numbers or
                    number > numbers[left - 2]):  # number is greater than the first of the two last values
                numbers[left] = number  # move the number to the left pointer position
                left += 1

        return left


# O(n), O(1)
# two pointers
class Solution:
    def removeDuplicates(self, numbers: list[int]) -> int:
        left = 0
        twice = False

        for number in numbers[1:]:
            if number == numbers[left] and not twice:
                left += 1
                numbers[left] = number
                twice = True
            elif number > numbers[left]:
                left += 1
                numbers[left] = number
                twice = False

        return left + 1





# Get Equal Substrings Within Budget
# https://leetcode.com/problems/get-equal-substrings-within-budget/description/
"""
You are given two strings s and t of the same length and an integer maxCost.

You want to change s to t. Changing the ith character of s to ith character of t costs |s[i] - t[i]| (i.e., the absolute difference between the ASCII values of the characters).

Return the maximum length of a substring of s that can be changed to be the same as the corresponding substring of t with a cost less than or equal to maxCost. If there is no substring from s that can be changed to its corresponding substring from t, return 0.

 

Example 1:

Input: s = "abcd", t = "bcdf", maxCost = 3
Output: 3
Explanation: "abc" of s can change to "bcd".
That costs 3, so the maximum length is 3.
Example 2:

Input: s = "abcd", t = "cdef", maxCost = 3
Output: 1
Explanation: Each character in s costs 2 to change to character in t,  so the maximum length is 1.
Example 3:

Input: s = "abcd", t = "acde", maxCost = 0
Output: 1
Explanation: You cannot make any change, so the maximum length is 1.
"""
print(Solution().equalSubstring("abcd", "bcdf", 3), 3)
print(Solution().equalSubstring("abcd", "cdef", 3), 1)
print(Solution().equalSubstring("abcd", "acde", 0), 1)


# O(n), O(1)
# sliding window
class Solution:
    def equalSubstring(self, text1: str, text2: str, max_cost: int) -> int:
        left = 0
        window_length = 0

        for right, (letter1, letter2) in enumerate(zip(text1, text2)):
            max_cost -= abs(ord(letter1) - ord(letter2))

            while max_cost < 0:
                max_cost += abs(ord(text1[left]) - ord(text2[left]))
                left += 1

            window_length = max(window_length, right - left + 1)

        return window_length


