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
        numbers.sort()  # sort list to be able to ignore duplicate numbers
        triplets = []

        for index, number in enumerate(numbers[:-2]):
            # Skip positive numbers
            if number > 0:
                break
            
            # Skip same number values
            if (index and 
                numbers[index] == numbers[index - 1]):
                continue

            left = index + 1  # left pointer
            right = len(numbers) - 1  # right pointer

            while left < right:  # two pointers
                triplet = number + numbers[left] + numbers[right]
                
                if triplet < 0:  # if sum is less than 0
                    left += 1
                elif triplet > 0:  # if sum is geater than 0
                    right -= 1
                else:  # if sum is equal to 0
                    triplets.append([number, numbers[left], numbers[right]])
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
print(Solution().lengthOfLongestSubstring("abcabcbb"), 3)
print(Solution().lengthOfLongestSubstring("bbbbb"), 1)
print(Solution().lengthOfLongestSubstring("pwwkew"), 3)
print(Solution().lengthOfLongestSubstring("aabaab!bb"), 3)
print(Solution().lengthOfLongestSubstring("aab"), 2)


# O(n), O(n)
# sliding window (as hash set)
class Solution:
    def lengthOfLongestSubstring(self, word: str) -> int:
        left = 0
        window = set()  # slidiong window without repeating characters
        longest = 0

        for right, letter in enumerate(word):
            while letter in window:  # while duplicate found
                window.remove(word[left])  # remove (discard) that charactr with every preceding character
                left += 1  # increase the left pointer

            window.add(letter)  # add an unique letter
            longest = max(longest, right - left + 1)  # update the length of the longest substring
        return longest


# O(n), O(n)
# sliding window (as hash map)
class Solution:
    def lengthOfLongestSubstring(self, word: str) -> int:
        left = 0
        window = {}  # sliding window as hash map
        longest = 0

        for right, letter in enumerate(word):
            window[letter] = window.get(letter, 0) + 1  # update window

            while window[letter] == 2:  # if duplicate found
                window[word[left]] -= 1  #  pop window left
                if window[word[left]] == 0:  # if no key
                    window.pop(word[left])  # remove a key
                left += 1  # increase the left pointer

            longest = max(longest, right - left + 1)  # update the length of the longest substring
        return longest


# O(n3), O(1)
# brute force
class Solution:
    def lengthOfLongestSubstring(self, word: str) -> int:
            longest = 0

            for left in range(len(word)):
                for right in range(left, len(word)):
                    substring = word[left: right + 1]

                    if len(substring) == len(set(substring)):
                        longest = max(longest, right - left + 1)

            return longest





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
print(Solution().characterReplacement("ABAB", 2), 4)
print(Solution().characterReplacement("AABABBA", 1), 4)


# O(n), O(n)
# sliding window
class Solution:
    def characterReplacement(self, word: str, joker: int) -> int:
        left = 0  # left pointer
        window = {}  # sliding window as hash map
        longest = 0  # length of the longest substring meeting conditions
        max_count = 0  # most frequent character counter

        for right, letter in enumerate(word):  # right pointer
            window[letter] = window.get(letter, 0) + 1  # update window
            max_count = max(max_count, window[letter])  # add new letter to window
            
            while (right - left + 1) - max_count > joker:  # while too many characters to change
                window[word[left]] -= 1  # remove "left" character from the window
                left += 1  # update left pointer

            longest = max(longest, right - left + 1)  # update the length of the longest substring
        return longest


# O(n2), O(n)
# brute force
class Solution:
    def characterReplacement(self, word: str, joker: int) -> int:
            longest = 0

            for left in range(len(word)):
                counter = {}
                max_count = 0
                
                for right in range(left, len(word)):    
                    counter[word[right]] = counter.get(word[right], 0) + 1
                    max_count = max(max_count, counter[word[right]])
                    if (right - left + 1) - max_count > joker:
                        break
                    
                longest = max(longest, min(max_count + joker, right - left + 1))

            return longest


# oldies
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
print(Solution().isValid("()"), True)
print(Solution().isValid("({})"), True)
print(Solution().isValid("(]"), False)
print(Solution().isValid("(})"), False)
print(Solution().isValid("([)"), False)
print(Solution().isValid(""), True)
print(Solution().isValid("["), False)


# O(n), O(n)
# stack
class Solution:
    def isValid(self, brackets: str) -> bool:
        stacked_brackets = []
        opposing_bracket = {
            ")": "(",
            "]": "[",
            "}": "{"
        }

        for bracket in brackets:
            if bracket in opposing_bracket:
                if (stacked_brackets and 
                        opposing_bracket[bracket] == stacked_brackets[-1]):
                    stacked_brackets.pop()
                else:
                    return False
            else:
                stacked_brackets.append(bracket)
        
        return not stacked_brackets  # if not empty then unmatched brackets left





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
print(Solution().search([4, 5, 6, 7, 8, 1, 2, 3], 8), 4)
print(Solution().search([1, 3, 5], 5), 2)
print(Solution().search([3, 5, 1], 3), 0)
print(Solution().search([3, 5, 1], 1), 2)
print(Solution().search([5, 1, 3], 3), 2)
print(Solution().search([4, 5, 6, 7, 0, 1, 2], 0), 4)
print(Solution().search([4, 5, 6, 7, 0, 1, 2], 3), -1)
print(Solution().search([1], 0), -1)
print(Solution().search([5, 1, 3], 4), -1)
print(Solution().search([4, 5, 6, 7, 8, 1, 2, 3], 8), 4)


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
                if middle_number < target <= numbers[right]:  # target in the right side # numbers[right] or numbers[len(numbers) - 1]
                    left = middle + 1
                else:
                    right = middle - 1  # target in the left side
            else:  # contiguous left side portion
                if numbers[left] <= target < middle_number:  # numbers[left] or numebrs[0]
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
print(Solution().climbStairs(0), 0)
print(Solution().climbStairs(1), 1)
print(Solution().climbStairs(2), 2)
print(Solution().climbStairs(3), 3)
print(Solution().climbStairs(4), 5)
print(Solution().climbStairs(5), 8)


"""
blueprint
             ___5
           _| 4
         _| 3
       _| 2
     _| 1
____|

           ___4
         _|  3
       _|  2
     _|  1
____|

         ___3
       _|  2
     _|  1
____|


Adding steps
                   0
             /1          \2
            1             2
        /1     \2      /1   \2
       2        3     3      4
     /1  \2    /1    /1
   3      4   4     4
  /1
 4

Subtracting steps
                   4
             /1          \2
            3             2
        /1     \2      /1   \2
       2        1     1      0
     /1  \2    /1    /1
   1      0   0     0
  /1
 0


5 -> 1 + 4 + 3 = 8
4 -> 5
3 -> 3
2 -> 2
1 -> 1
0 -> 0

Fibonnacci problem
"""


class Solution:
    def climbStairs(self, number: int) -> int:
        """
        O(n), O(1)
        dp, bottom-up
        """
        if number == 0:
            return 0
        elif number == 1:
            return 1
        
        cache = [1, 1]

        for _ in range(2, number + 1):
            cache = [cache[1], cache[0] + cache[1]]

        return cache[-1]


class Solution:
    def climbStairs(self, number: int) -> int:
        """
        O(n), O(n)
        dp, bottom-up
        """
        if number == 0:
            return 0
        elif number == 1:
            return 1
        
        cache = [1] * (number + 1)

        for index in range(2, number + 1):
            cache[index] = cache[index - 1] + cache[index - 2]

        return cache[number]


class Solution:
    def climbStairs(self, number: int) -> int:
        """
        O(n), O(n)
        dp, top-down with memoization as hash map
        """
        if number == 0:
            return 0
        
        memo = {0: 1}

        def dfs(index):
            if index < 0:
                return 0
            elif index in memo:
                return memo[index]

            memo[index] = dfs(index - 1) + dfs(index - 2)
            return memo[index]

        return dfs(number)


class Solution:
    def climbStairs(self, number: int) -> int:
        """
        O(n), O(n)
        dp, top-down with memoization as list
        """
        if number == 0:
            return 0
        
        memo = [None] * (number + 1)
        memo[0] = 1

        def dfs(index):
            if index < 0:
                return 0
            elif memo[index] is not None:
                return memo[index]

            memo[index] = dfs(index - 1) + dfs(index - 2)
            return memo[index]

        return dfs(number)


class Solution:
    def climbStairs(self, number: int) -> int:
        """
        O(2^n), O(n)
        brute force, pure recursion, tle
        converts to top-down
        """
        if number == 0:
            return 0
        
        def dfs(index):
            if index < 0:
                return 0
            elif index == 0:
                return 1

            return dfs(index - 1) + dfs(index - 2)

        return dfs(number)


class Solution:
    def climbStairs(self, number: int) -> int:
        """
        O(2^n), O(n)
        brute force, shared state, tle
        """
        if number == 0:
            return 0
        
        self.counter = 0

        def dfs(index):
            if index == 0:
                self.counter += 1
                return
            elif index < 0:
                return

            dfs(index - 1)
            dfs(index - 2)

        dfs(number)
        return self.counter


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
print(Solution().rob([2]), 2)
print(Solution().rob([0]), 0)
print(Solution().rob([2, 1]), 2)
print(Solution().rob([2, 100, 9, 3, 100]), 200)
print(Solution().rob([100, 9, 3, 100, 2]), 200)
print(Solution().rob([1, 2, 3, 1]), 4)
print(Solution().rob([2, 7, 9, 3, 1]), 12)
print(Solution().rob([183,219,57,193,94,233,202,154,65,240,97,234,100,249,186,66,90,238,168,128,177,235,50,81,185,165,217,207,88,80,112,78,135,62,228,247,211]), 3365)
print(Solution().rob([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), 0)

# draft
# values     [2, 7, 9,  3,  1]
# cumulative [2, 7, 11,11. 12]

# [2, 100, 9, 3, 100] data
# [2, 100, 100, 100, 200] dp solution
# [(2), (2, 100), (100, 100) ....]

# [100, 9, 3, 100, 2] data
# [100, 100, 100, 200, 200] dp solution


class Solution:
    def rob(self, numbers: list[int]) -> int:
        """
        O(n), O(1)
        dp, bottom-up
        """
        if len(numbers) < 3:
            return max(numbers)

        cache = [numbers[0], max(numbers[:2])]

        for index in range(2, len(numbers)):
            cache = [cache[1], max(numbers[index] + cache[0], cache[1])]

        return cache[1]


class Solution:
    def rob(self, numbers: list[int]) -> int:
        """
        O(n), O(1)
        dp, bottom-up
        mutate input list
        """
        if len(numbers) < 3:
            return max(numbers)

        numbers[1] = max(numbers[:2])

        for index in range(2, len(numbers)):
            numbers[index] = max(numbers[index] + numbers[index - 2], 
                                 numbers[index - 1])

        return numbers[-1]


class Solution:
    def rob(self, numbers: list[int]) -> int:
        """
        O(n), O(n)
        dp, bottom-up
        """
        if len(numbers) < 3:
            return max(numbers)

        cache = [0] * len(numbers)  # stores max money we can rob considering only first i houses.
        cache[0] = numbers[0]
        cache[1] = max(numbers[:2])

        for index in range(2, len(numbers)):
            cache[index] = max(numbers[index] + cache[index - 2], 
                               cache[index - 1])

        return cache[-1]


class Solution:
    def rob(self, numbers: list[int]) -> int:
        """
        O(n), O(n)
        dp, top-down with memoization as hash map
        """
        memo = {}  # caches max money we can rob from house index to end.

        def dfs(index):
            if index >= len(numbers):
                return 0
            elif index in memo:
                return memo[index]

            memo[index] = max(numbers[index] + dfs(index + 2), 
                              dfs(index + 1))
            return memo[index]

        return dfs(0)


class Solution:
    def rob(self, numbers: list[int]) -> int:
        """
        O(n), O(n)
        dp, top-down with memoization as list
        """
        memo = [None] * (len(numbers) + 1)  # caches max money we can rob from house index to end.

        def dfs(index):
            if index >= len(numbers):
                return 0
            elif memo[index] is not None:
                return memo[index]

            memo[index] = max(numbers[index] + dfs(index + 2), 
                              dfs(index + 1))
            return memo[index]

        return dfs(0)


class Solution:
    def rob(self, numbers: list[int]) -> int:
        """
        O(2^n), O(n)
        brute force, tle
        """
        def dfs(index):
            if index >= len(numbers):
                return 0

            return max(numbers[index] + dfs(index + 2), 
                       dfs(index + 1))

        return dfs(0)





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
print(Solution().rob([2, 3, 2]), 3)
print(Solution().rob([1, 2, 3, 1]), 4)
print(Solution().rob([1, 2, 3]), 3)
print(Solution().rob([1]), 1)
print(Solution().rob([0, 0]), 0)
print(Solution().rob([1, 3, 1, 3, 100]), 103)


class Solution:
    def rob(self, numbers: list[int]) -> int:
        """
        O(n), O(1)
        dp, bottom-up
        """
        if len(numbers) < 3:
            return max(numbers)
        
        def rob2(numbers):
            if len(numbers) < 3:
                return max(numbers)
            
            cache = [numbers[0], max(numbers[:2])]

            for index in range(2, len(numbers)):
                cache = [cache[1], max(numbers[index] + cache[0], cache[1])]

            return cache[1]

        return max(rob2(numbers[1:]), 
                   rob2(numbers[:-1]))


class Solution:
    def rob(self, numbers: list[int]) -> int:
        """
        O(n), O(n)
        dp, bottom-up
        """
        if len(numbers) < 3:
            return max(numbers)
        
        def rob2(numbers):
            if len(numbers) < 3:
                return max(numbers)
            
            cache = [0] * len(numbers)  # stores max money we can rob considering only first i houses.
            cache[:2] = [numbers[0], max(numbers[:2])]

            for index in range(2, len(numbers)):
                cache[index] = max(numbers[index] + cache[index - 2], 
                                   cache[index - 1])

            return cache[-1]

        return max(rob2(numbers[1:]), 
                   rob2(numbers[:-1]))


# O(n), 0(n)
# both functions directly under class
class Solution:
    def inner_rob(self, nums: list[int]) -> int:  # rob in stight line
        house1 = nums[0]
        house2 = max(nums[0], nums[1])

        for num in nums[2:]:
            house1, house2 = house2, max(num + house1, house2)

        return house2

    def rob(self, nums):  # rob in circle
        if len(nums) < 3:
            return max(nums)

        return max(
            self.inner_rob(nums[:-1]), 
            self.inner_rob(nums[1:]))





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
print(Solution().longestPalindrome("babad"), "bab")
print(Solution().longestPalindrome("cbbd"), "bb")
print(Solution().longestPalindrome("a"), "a")
print(Solution().longestPalindrome(""), "")
print(Solution().longestPalindrome("bb"), "bb")
print(Solution().longestPalindrome("ab"), "a")
print(Solution().longestPalindrome("aacabdkacaa"), "aca")
print(Solution().longestPalindrome("abdka"), "a")
print(Solution().longestPalindrome("aaaa"), "aaaa")


# O(n2), O(1)
# two pointers
class Solution:
    def longestPalindrome(self, word: str) -> str:
        start = 0
        palindrome_length = 0

        for index in range(len(word)):
            # odd length palindrome
            left = index
            right = index
            while (left >= 0 and  # check if not out of bounds left
                   right < len(word) and  # check if not out of bounds right
                   word[left] == word[right]):  # if letter match
                if right - left + 1 > palindrome_length:  # if longer palindrome found
                    palindrome_length = right - left + 1
                    start = left
                
                left -= 1
                right += 1

            # even lenght palindrome
            left = index
            right = index + 1
            while (left >= 0 and  # check if not out of bounds left
                   right < len(word) and  # check if not out of bounds right
                   word[left] == word[right]):  # if letter match
                if right - left + 1 > palindrome_length:  # if longer palindrome found
                    palindrome_length = right - left + 1
                    start = left

                left -= 1
                right += 1

        return word[start: start + palindrome_length]


# O(n2), O(1)
# two pointers
class Solution:
    def longestPalindrome(self, word: str) -> str:
        left = 0
        palindrome_length = 0

        for index in range(len(word)):
            # odd length palindrome
            edge = 1
            while (index - edge >= 0 and  # check if not out of bounds left
                   index + edge < len(word) and  # check if not out of bounds right
                   word[index - edge] == word[index + edge]):  # if letter match
                edge += 1  # 1 -> 3, 2i + 1 increase palindrome length

            edge -= 1
            if 2 * edge + 1 > palindrome_length:  # if longer palindrome found
                palindrome_length = 2 * edge + 1
                left = index - edge

            # even lenght palindrome
            edge = 0
            while (index - edge >= 0 and  # check if not out of bounds left
                   index + edge + 1 < len(word) and  # check if not out of bounds right
                   word[index - edge] == word[index + 1 + edge]):  # if letter match
                edge += 1  # 2 -> 4, 2i increase palindrome length

            edge -= 1
            if 2 * (edge + 1) > palindrome_length:  # if longer palindrome found
                palindrome_length = 2 * (edge + 1)
                left = index - edge

        return word[left: left + palindrome_length]


# O(n3), O(n)
# brute force
class Solution:
    def is_palindrome(self, word: str) -> bool:
        return word == word[::-1]
    
    def longestPalindrome(self, word: str) -> str:
        palindrome = ""
        palindrome_length = 0

        for left in range(len(word)):
            for right in range(left, len(word)):
                if self.is_palindrome(word[left: right + 1]):
                    if right - left + 1 > palindrome_length:
                        palindrome = word[left: right + 1]
                        palindrome_length = right - left + 1

        return palindrome


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
print(Solution().countSubstrings("abc"), 3)
print(Solution().countSubstrings("aaa"), 6)


# draft
# "aaa"
# a aa a aa aaa a 

# O(n2), O(1)
# two pointers
class Solution:
    def __init__(self) -> None:
        self.counter = 0  # count palindroms

    def count_palindroms(self, left: int, right: int) -> None:
        while (left >= 0 and  # check if not out of bounds left
               right < len(self.word) and  # check if not out of bounds right
               self.word[left] == self.word[right]):  # if letter match
            self.counter += 1  # increase counter
            left -= 1  # move left pointer
            right += 1  # move right pointer

    def countSubstrings(self, word: str) -> int:
        self.word = word

        for index in range(len(word)):
            self.count_palindroms(index, index)  # odd length palindromes
            self.count_palindroms(index, index + 1)  # even length palindromes

        return self.counter





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
print(Solution().coinChange([5], 5), 1)
print(Solution().coinChange([1, 2], 3), 2)
print(Solution().coinChange([1, 2, 5], 11), 3)
print(Solution().coinChange([2, 5, 10, 1], 27), 4)
print(Solution().coinChange([186, 419, 83, 408], 6249), 20)
print(Solution().coinChange([2], 3), -1)
print(Solution().coinChange([2], 1), -1)
print(Solution().coinChange([1], 0), 0)


class Solution:
    def coinChange(self, coins: list[int], amount: int) -> int:
        """
        O(n2), O(n)
        dp, bottom-up with tabulation as list
        """
        # min number of coins needed to get target amount (equal to the index)
        # "anmount + 1" an imposbile value stays when the last element of min_coins was not modified
        cache = [amount + 1] * (amount + 1)  # cache
        cache[0] = 0  # no coins needed to get 0

        for index in range(1, amount + 1):  # check each 'cache' index
            for coin in coins:  # check every coin
                if index - coin >= 0:  # target ammount must be greater equal to current coin
                    # choose current amount of coins or get ammount without current coin and add 1
                    cache[index] = min(cache[index], cache[index - coin] + 1)
        
        # if the last value was not modified, there is no valid combination
        return (cache[amount] 
                if cache[amount] != amount + 1
                else -1)


class Solution:
    def coinChange(self, coins: list[int], amount: int) -> int:
        """
        O(n2), O(n)
        dp, bottom-up with tabulation as hash map
        """
        # min number of coins needed to get target amount (equal to the index)
        cache = {0: 0}  # {ammont: min coins to get that amount}

        for index in range(1, amount + 1):
            for coin in coins:
                if index - coin >= 0:
                    cache[index] = min(cache.get(index, amount + 1), # "anmount + 1" an imposbile value stays when the last element of min_coins was not founded
                                       cache.get(index - coin, amount + 1) + 1)
        
        return (cache[amount] 
                if (amount in cache and  # if amount exists and
                    cache[amount] != amount + 1)  # the value is not impossible value
                else -1)


class Solution:
    def coinChange(self, coins: list[int], amount: int) -> int:
        """
        O(n2), O(n)
        dp, top-down with memoization as hash map
        """
        base_amount = amount
        memo = {0: 0}  # {ammont: min coins to get that amount}
        
        def dfs(amount):
            if amount < 0:
                return base_amount + 1  # return impossible value
            elif amount in memo:
                return memo[amount]

            # Calculate the minimum number of coins needed for this amount
            memo[amount] = min(1 + dfs(amount - coin) 
                               for coin in coins)

            return memo[amount]

        number_of_coins = dfs(amount)
        return (number_of_coins 
                if number_of_coins <= amount
                else -1)


class Solution:
    def coinChange(self, coins: list[int], amount: int) -> int:
        """
        O(n2), O(n)
        dp, top-down with memoization as list
        """
        memo = [None] * (amount + 1)  # [min coins to get index amount]
        memo[0] = 0
        base_amount = amount
        
        def dfs(amount):
            if amount < 0:
                return base_amount + 1
            elif memo[amount] is not None:
                return memo[amount]

            memo[amount] = min(1 + dfs(amount - coin) 
                               for coin in coins)

            return memo[amount]

        number_of_coins = dfs(amount)
        return (number_of_coins 
                if number_of_coins <= amount
                else -1)


class Solution:
    def coinChange(self, coins: list[int], amount: int) -> int:
        """
        O(2^n), O(n)
        brute force, tle
        O(coins^amount)
        """
        base_amount = amount
        
        def dfs(amount):
            if amount == 0:
                return 0
            elif amount < 0:
                return base_amount + 1

            return min(1 + dfs(amount - coin) 
                       for coin in coins)

        number_of_coins = dfs(amount)
        return (number_of_coins 
                if number_of_coins <= amount
                else -1)





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
print(Solution().maxProduct([-2]), -2)
print(Solution().maxProduct([-4, -3]), 12)
print(Solution().maxProduct([2, 3, -2, 4]), 6)
print(Solution().maxProduct([-2, 0, -1]), 0)
print(Solution().maxProduct([-2, -3, 7]), 42)
print(Solution().maxProduct([2, -5, -2, -4, 3]), 24)
print(Solution().maxProduct([0]), 0)
print(Solution().maxProduct([-2, 0]), 0)
print(Solution().maxProduct([0, 2]), 2)


# draft
# [2, 3, -2, 4]
# (2, 2), (3, !2*3!), (-2*6, -2), (-48, -2)


class Solution:
    def maxProduct(self, numbers: list[int]) -> int:
        """
        O(n), O(1)
        dp, bottom-up
        """
        cache = (1, 1)
        max_product = numbers[0]

        for number in numbers:
            triplet = (cache[0] * number, 
                       cache[1] * number, 
                       number)
            cache = (max(triplet), min(triplet))
            max_product = max(max_product, cache[0])

        return max_product


class Solution:
    def maxProduct(self, numbers: list[int]) -> int:
        """
        O(n), O(1)
        dp, bottom-up
        mutate input array
        """
        max_product = numbers[0]
        
        for index, number in enumerate(numbers):
            if index == 0:
                numbers[0] = (numbers[0], numbers[0])
                continue

            triplet = (numbers[index - 1][0] * number, 
                       numbers[index - 1][1] * number, 
                       number)
            numbers[index] = (max(triplet), min(triplet))
            max_product = max(max_product, numbers[index][0])

        return max_product


class Solution:
    def maxProduct(self, numbers: list[int]) -> int:
        """
        O(n), O(n)
        dp, top-down
        single recursive call per index
        """
        self.max_product = numbers[0]
        
        def dfs(index):
            if index == len(numbers):
                return (1, 1)

            next_dfs = dfs(index + 1)
            triplet = (next_dfs[0] * numbers[index], 
                       next_dfs[1] * numbers[index], 
                       numbers[index])
            
            current_max = max(triplet)
            self.max_product = max(self.max_product, current_max)

            return (current_max, min(triplet))

        return max(max(dfs(0)), self.max_product)


class Solution:
    def maxProduct(self, numbers: list[int]) -> int:
        """
        O(n2), O(1)
        brute force
        """
        max_product = numbers[0]

        for left in range(len(numbers)):
            val = 1
            for right in range(left, len(numbers)):
                val *= numbers[right]
                max_product = max(max_product, val)
        
        return max_product





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
print(Solution().wordBreak("leetcode", ["leet", "code"]), True)
print(Solution().wordBreak("applepenapple", ["apple","pen"]), True)
print(Solution().wordBreak("catsandog", ["cats","dog","sand","and","cat"]), False)
print(Solution().wordBreak("cars", ["car", "ca", "rs"]), True)


class Solution:
    def wordBreak(self, text: str, word_list: list[str]) -> bool:
        """
        O(n3), O(n)
        dp, bottom-up
        """
        # cache where each elemet tells if sentece can be fold from this index to the right
        cache = [False] * (len(text) + 1)  # can fold
        cache[-1] = True  # dummy element tells that everything after "text can be folded"
        word_set = set(word_list)

        for index in reversed(range(len(text))):  # go through every index reversed
            for word in word_set:  # go through every word
                if text[index: index + len(word)] == word:  # if found the word
                    cache[index] = (cache[index] or  # if already can fold
                                       cache[index + len(word)])  # update can fold
                if cache[index]:  # early exit
                    break
        return cache[0]


class Solution:
    def wordBreak(self, text: str, word_list: list[str]) -> bool:
        """
        O(n3), O(n)
        dp, top-down with memoization as hash map
        """
        memo = {len(text): True}
        word_set = set(word_list)

        def dfs(index):
            if index in memo:
                return memo[index]
            
            for word in word_set:
                if (text[index: index + len(word)] == word and
                        dfs(index + len(word))):
                    memo[index] = True
                    return True
            
            memo[index] = False
            return False

        return dfs(0)


class Solution:
    def wordBreak(self, text: str, word_list: list[str]) -> bool:
        """
        O(2^n), O(n)
        brute force, tle
        """
        word_set = set(word_list)

        def dfs(index):
            if index == len(text):
                return True
            
            for word in word_set:
                if (text[index: index + len(word)] == word and
                        dfs(index + len(word))):
                    return True

            return False

        return dfs(0)


class TrieNode:
    def __init__(self):
        # Each TrieNode contains a dictionary of children and a boolean to indicate if it's the end of a word.
        self.children = {}
        self.is_word = False

class Trie:
    def __init__(self) -> None:
        # Initialize the Trie with a root node.
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        # Inserts a word into the Trie.
        node = self.root
        for letter in word:
            if letter not in node.children:
                # If the letter is not in children, create a new TrieNode.
                node.children[letter] = TrieNode()
            # Move to the child node.
            node = node.children[letter]
        # Mark the end of the word.
        node.is_word = True

    def search(self, text: str, start: int, end: int) -> bool:
        # Checks if the substring `text[start:end+1]` exists in the Trie.
        node = self.root
        for letter in text[start: end + 1]:
            if letter not in node.children:
                # If a letter is not found in the children, return False.
                return False
            # Move to the child node.
            node = node.children[letter]
        # Return True if the substring is a complete word in the Trie.
        return node.is_word

class Solution:
    def wordBreak(self, text: str, word_list: list[str]) -> bool:
        """
        O(n3), O(n2)
        dp, bottom-up, trie

        Determines if the string `text` can be segmented into words from `word_list`.

        Args:
        text (str): The input string.
        word_list (list[str]): List of valid words.

        Returns:
        bool: True if `text` can be segmented, False otherwise.
        """
        # Cache stores whether `text[i:]` can be segmented (True or False).
        cache = [False] * (len(text) + 1)
        cache[-1] = True  # Base case: empty string is always segmentable.
        
        max_word_len = 0  # Track the length of the longest word in `word_list`.
        trie = Trie()
        
        # Insert all words into the Trie and update the max word length.
        for word in word_list:
            trie.insert(word)
            max_word_len = max(max_word_len, len(word))

        # Iterate over the text from right to left.
        for left in reversed(range(len(text))):
            # Check substrings starting from the current position.
            for right in range(left, min(len(text), left + max_word_len)):
                # If a substring exists in the Trie and the rest of the string can be segmented.
                if trie.search(text, left, right):
                    cache[left] = cache[right + 1]
                if cache[left]:
                    # Stop checking further substrings if segmentation is already confirmed.
                    break

        # Return whether the entire string can be segmented.
        return cache[0]





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
print(Solution().lengthOfLIS([5]), 1)
print(Solution().lengthOfLIS([5, 6]), 2)
print(Solution().lengthOfLIS([5, 4]), 1)
print(Solution().lengthOfLIS([10, 9, 2, 5, 3, 7, 101, 18]), 4)
print(Solution().lengthOfLIS([0, 1, 0, 3, 2, 3]), 4)
print(Solution().lengthOfLIS([7, 7, 7, 7, 7, 7, 7]), 1)
print(Solution().lengthOfLIS([4, 10, 4, 3, 8, 9]), 3)


"""
draft
[10, 9, 2, 5, 3, 7, 101, 18]
[1, 1, 1, 2, 2, max(2,2)+1=3, 4, 1]
                 .
        /               \
        2               .
      /   \           /   \
     8    .          8     .
    / \  / \        / \   / \
   4  . 4   .      4  .  4   .
  /\ /\ /\  /\    /\  /\ /\  /\
  5. 5. 5.  5.    5.  5.  5.  5.
"""


class Solution:
    def lengthOfLIS(self, numbers: list[int]) -> int:
        """
        O(n2), O(n)
        dp, bottom-up with cache as list
        """
        cache = [1] * len(numbers)  # LIS lengths

        for right in range(len(numbers)):  # check every right (index)
            for left in range(right):  # check every left (index) lower than right
                if numbers[left] < numbers[right]:  # if right number is greater
                    cache[right] = max(cache[right], cache[left] + 1)  # update LIS lengths 

        return max(cache)


class Solution:
    def lengthOfLIS(self, numbers: list[int]) -> int:
        """
        O(n2), O(n)
        dp, bottom-up with cache as hash map
        """
        cache = {}  # LIS lengths

        for right in range(len(numbers)):  # check every right (index)
            cache[numbers[right]] = 1  # add current element to cache
            
            for left_val in cache.keys():  # check every left_val (index) of cache
                if left_val < numbers[right]:  # if right number is greater
                    cache[numbers[right]] = max(cache[numbers[right]], 
                                                cache[left_val] + 1)  # update LIS lengths 

        return max(cache.values())


class Solution:
    def lengthOfLIS(self, numbers: list[int]) -> int:
        """
        O(n2), O(n)
        dp, top-down with memoization as hash map
        mle
        """
        memo = {}

        def dfs(index, prev_index):
            if index == len(numbers):
                return 0
            elif (index, prev_index) in memo:
                return memo[(index, prev_index)]

            # Try skipping numbers[index]
            longest = dfs(index + 1, prev_index)
            
            # Try including numbers[index] if it maintains an increasing order
            if (prev_index == -1 or 
                    numbers[index] > numbers[prev_index]):
                longest = max(longest, 1 + dfs(index + 1, index))

            memo[(index, prev_index)] = longest
            return longest

        return dfs(0, -1)


class Solution:
    def lengthOfLIS(self, numbers: list[int]) -> int:
        """
        O(2^n), O(n)
        brute force, pure recursion
        tle, mle, converts to top-down
        """

        def dfs(index, prev_index):
            if index == len(numbers):
                return 0

            # Try skipping numbers[index]
            longest = dfs(index + 1, prev_index)
            
            # Try including numbers[index] if it maintains an increasing order
            if (prev_index == -1 or 
                    numbers[index] > numbers[prev_index]):
                longest = max(longest, 1 + dfs(index + 1, index))

            return longest

        return dfs(0, -1)


class Solution:
    def lengthOfLIS(self, numbers: list[int]) -> int:
        """
        O(2^n), O(n)
        brute force, function argument
        tle
        """
        self.longest = 0
        
        def dfs(index, subsequence):
            if index == len(numbers):
                self.longest = max(self.longest, len(subsequence))
                return

            # Try including numbers[index] if it maintains an increasing order
            if not subsequence or subsequence[-1] < numbers[index]:
                dfs(index + 1, subsequence + [numbers[index]])
            
            # Try skipping numbers[index]
            dfs(index + 1, subsequence)

        dfs(0, [])
        return self.longest


class Solution:
    def lengthOfLIS(self, numbers: list[int]) -> int:
        """
        O(n2^n), O(n)
        brute force, backtracking
        tle
        """
        self.counter = 1
        subsequence = []

        def dfs(index):
            if index == len(numbers):
                if len(subsequence) > 1:
                    for ind in range(len(subsequence) - 1):
                        if subsequence[ind] >= subsequence[ind + 1]:
                            return
                    self.counter = max(self.counter, len(subsequence))
                return

            subsequence.append(numbers[index])
            dfs(index + 1)
            subsequence.pop()
            dfs(index + 1)

        dfs(0)
        return self.counter





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
print(Solution().canPartition([2]), False)
print(Solution().canPartition([2, 2]), True)
print(Solution().canPartition([1, 5, 11, 5]), True)
print(Solution().canPartition([14, 9, 8, 4, 3, 2]), True)
print(Solution().canPartition([1, 2, 5]), False)
print(Solution().canPartition([3, 3, 3, 4, 5]), True)
print(Solution().canPartition([1, 2, 3, 5]), False)
print(Solution().canPartition([1]), False)
print(Solution().canPartition([2, 2, 1, 1]), True)


"""
draft
                  .
            /           \
           1            .
         /    \       /    \
        5     .       5     .
       /\     /\     /\    /\
      11 .   11 .   11 .  11 .
    /\
   5 .
"""


class Solution:
    def canPartition(self, numbers: list[int]) -> bool:
        """
        O(n2), O(n)
        dp, bottom-up
        """
        total = sum(numbers)
        if total % 2:  # if odd sum (cannot be split in half)
            return False
        half = total // 2  # half of the sum
        subset = set({0})  # set with addition neutral element

        for number in numbers:
            subset.update({val + number  # update every val in subset with (val + number), but preserve orginal vals
                           for val in subset.copy()})
            subset.add(number)

            if half in subset:
                return True

        return False


class Solution:
    def canPartition(self, numbers: list[int]) -> bool:
        """
        O(n2), O(n)
        dp, top-down with memoization as hash map
        """
        total = sum(numbers)
        if total % 2:  # if odd sum (cannot be split in half)
            return False

        memo = {}  # {(index, target): bool}
        
        def dfs(index: int, target: int) -> bool:
            if (index, target) in memo:
                return memo[(index, target)]
            elif target <= 0:
                return target == 0
            elif index == len(numbers):
                return False

            memo[(index, target)] = (dfs(index + 1, target - numbers[index]) or 
                                     dfs(index + 1, target))
            return memo[(index, target)]

        return dfs(0, total // 2)


class Solution:
    def canPartition(self, numbers: list[int]) -> bool:
        """
        O(2^n), O(n)
        brute force, pure recursion, tle
        converts to top-down
        """
        total = sum(numbers)
        if total % 2:  # if odd sum (cannot be split in half)
            return False

        def dfs(index: int, target: int) -> bool:
            if target <= 0:
                return target == 0
            elif index == len(numbers):
                return False
        
            return (dfs(index + 1, target - numbers[index]) or
                    dfs(index + 1, target))

        return dfs(0, total // 2)


class Solution:
    def canPartition(self, numbers: list[int]) -> bool:
        """
        O(2^n), O(n)
        brute force, backtracking, tle
        """
        total = sum(numbers)
        if total % 2:
            return False
        half = total // 2
        subset = []

        def dfs(index: int) -> bool:
            if sum(subset) == half:
                return True
            elif (index == len(numbers) or
                  sum(subset) > half):
                return

            subset.append(numbers[index])
            if dfs(index + 1):
                return True
            subset.pop()
            if dfs(index + 1):
                return True

            return False

        return dfs(0)





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
print(Solution().maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4]), 6)
print(Solution().maxSubArray([1]), 1)
print(Solution().maxSubArray([5, 4, -1, 7, 8]), 23)
print(Solution().maxSubArray([-4, -2, -1, -3]), -1)


#                      -2     
#                    /    \    
#                  -2+1   1    
#                       /   \ 
#                     1-3    -3
#                     /  \
#                  -2+4   4
#                        /  \
#                      4-1   -1
#                      /  \
#                   3+2    2
#                  /   \
#                5+1    1
# 
# for [-2, 1, -3, 4, -1, 2, 1, -5, 4]
# cummulative sum [-2, 1, -2, 4, 3, 5, 6, 1, 4]


# O(n), O(1)
# dp, bottom-up
class Solution:
    def maxSubArray(self, numbers: list[int]) -> int:
        cumulative = numbers[0]  # take the first value as a default value
        max_value = numbers[0]  # take the first value as a default value

        for number in numbers[1:]:
            cumulative = max(cumulative + number, number)  # keep track of the cumulative sum or start a new one
            max_value = max(cumulative, max_value)  # keep track of the highest value

        return max_value


# O(n), O(n)
# dp, bottom-up
class Solution:
    def maxSubArray(self, numbers: list[int]) -> int:
        dp = [0 for _ in range(len(numbers))]  # for cumulative sum
        dp[0] = numbers[0]  # in case of all negative values cannot take "0" as a base

        for index in range(1, len(numbers)):
            dp[index] = max(dp[index - 1] + numbers[index], numbers[index])   # add another element to sum or start a new one
        
        return max(dp)


# O(n), O(n)
# dp, bottom-up
# use numbers as dp
class Solution:
    def maxSubArray(self, numbers: list[int]) -> int:
        for index in range(1, len(numbers)):
            numbers[index] += max(numbers[index - 1], 0)  # add another element to sum or start a new one

        return max(numbers)





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
# tle
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
uOutput: true
Explanation: s2 contains one permutation of s1 ("ba").

Example 2:

Input: s1 = "ab", s2 = "eidboaoo"
Output: false
"""
print(Solution().checkInclusion("ab", "eidbaooo"), True)
print(Solution().checkInclusion("ab", "eidboaoo"), False)
print(Solution().checkInclusion("ccc", "cbac"), False)
print(Solution().checkInclusion("ab", "a"), False)
print(Solution().checkInclusion("abcdxabcde", "abcdeabcdx"), True)
print(Solution().checkInclusion("adc", "dcda"), True)
print(Solution().checkInclusion("hello", "ooolleoooleh"), False)
print(Solution().checkInclusion("mart", "karma"), False)
print(Solution().checkInclusion("abc", "ccccbbbbaaaa"), False)


# O(n), O(n)
# sliding window
class Solution:
    def checkInclusion(self, word1: str, word2: str) -> bool:
        left = 0  # `left` marks the start of the sliding window
        pattern = {}  # `pattern` stores the frequency of characters in `word1`
        window = {}  # `window` keeps track of character frequencies in the current window of `word2`
        
        for letter in word1:
            pattern[letter] = pattern.get(letter, 0) + 1
        
        needed = len(pattern)  # `needed` keeps track of how many characters in `pattern` are still needed to match frequencies in `window`

        for right, letter in enumerate(word2):
            window[letter] = window.get(letter, 0) + 1  # add `letter` to `window`
            
            if (letter in pattern and  # if `letter` is in `pattern` and
                    window[letter] == pattern[letter]):  # `letter` frequencies match
                needed -= 1  # one less character needed to match strings

                if needed == 0:  # if no `needed` (strings matched)
                    return True
            
            # if `window` is as long as `text1`
            if (right - left + 1) == len(word1):
                left_letter = word2[left]  # get left letter
                
                if (left_letter in pattern and  # if left letter is in pattern and
                        window[left_letter] == pattern[left_letter]):  # frequency of that letter in window is same as in pattern
                    needed += 1  # increase matching character frequences

                window[left_letter] -= 1  # decrease character frequency

                if window[left_letter] == 0:  # if character frequency is zero
                    window.pop(left_letter)  # pop that character

                left += 1
        
        return False


# O(n), O(n)
# sliding window
# This code is wrong somehow. In this version correct window length is checked
# just after adding a new letter. So when its too long it gets shorten.
# In the working version window lenght is checkd as the end of the loop.
# If it's the size of the pattern, it gets trimmed for the next loop.
class Solution:
    def checkInclusion(self, word1: str, word2: str) -> bool:
        left = 0  # `left` marks the start of the sliding window
        pattern = {}  # `pattern` stores the frequency of characters in `word1`
        window = {}  # `window` keeps track of character frequencies in the current window of `word2`
        
        for letter in word1:
            pattern[letter] = pattern.get(letter, 0) + 1
        
        needed = len(pattern)  # `needed` keeps track of how many characters in `pattern` are still needed to match frequencies in `window`

        for right, letter in enumerate(word2):
            window[letter] = window.get(letter, 0) + 1  # add `letter` to `window`
            
            # if `window` is longer than `text1`
            if (right - left + 1) == len(word1) + 1:
                left_letter = word2[left]  # get left letter
                
                if (left_letter in pattern and  # if left letter is in pattern and
                        window[left_letter] == pattern[left_letter]):  # frequency of that letter in window is same as in pattern
                    needed += 1  # increase matching character frequences

                window[left_letter] -= 1  # decrease character frequency

                if window[left_letter] == 0:  # if character frequency is zero
                    window.pop(left_letter)  # pop that character

                left += 1

            if (letter in pattern and  # if `letter` is in `pattern` and
                    window[letter] == pattern[letter]):  # `letter` frequencies match
                needed -= 1  # one less character needed to match strings

                if needed == 0:  # if no `needed` (strings matched)
                    return True
        
        return False


# O(n2), O(n)
# brute force
class Solution:
    def checkInclusion(self, word1: str, word2: str) -> bool:
        counter_word1 = {}
        for letter in word1:
            counter_word1[letter] = counter_word1.get(letter, 0) + 1

        for left in range(len(word2) - len(word1) + 1):
            counter_word1_copy = counter_word1.copy()
            
            for right in range(left, left + len(word1)):
                letter = word2[right]
                if letter not in counter_word1_copy:
                    break
                counter_word1_copy[letter] -= 1
                if counter_word1_copy[letter] == 0:
                    counter_word1_copy.pop(letter)
            
            if not counter_word1_copy:
                return True

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
print(Solution().minWindow("ADOBECODEBANC", "ABC"), "BANC")
print(Solution().minWindow("a", "a"), "a")
print(Solution().minWindow("a", "aa"), "")
print(Solution().minWindow("a", "b"), "")
print(Solution().minWindow("ab", "b"), "b")
print(Solution().minWindow("bba", "ab"), "ba")
print(Solution().minWindow("abc", "a"), "a")
print(Solution().minWindow("jwsdrkqzrthzysvqqazpfnulprulwmwhiqlbcdduxktxepnurpmxegktzegxscfbusexvhruumrydhvvyjucpeyrkeodjvgdnkalutfoizoliliguckmikdtpryanedcqgpkzxahwrvgcdoxiylqjolahjawpfbilqunnvwlmtrqxfphgaxroltyuvumavuomodblbnzvequmfbuganwliwidrudoqtcgyeuattxxlrlruhzuwxieuqhnmndciikhoehhaeglxuerbfnafvebnmozbtdjdo", "qruzywfhkcbovewle"), "vequmfbuganwliwidrudoqtcgyeuattxxlrlruhzuwxieuqhnmndciik")


# O(n), O(n)
# sliding window
class Solution:
    def minWindow(self, word1: str, word2: str) -> str:
        pattern = {}  # stores the frequency of characters in `word2` (target)
        window = {}  # tracks character frequencies in the current sliding window of `word1`
        left = 0  # the start pointer for the sliding window
        start = -1  # the start of the shortest window
        min_length = len(word1) + 1
        
        for letter in word2:
            pattern[letter] = pattern.get(letter, 0) + 1
        
        need = len(pattern)  # the number of unique letters left to match the pattern
        
        for right, letter in enumerate(word1):
            if letter in pattern:  # if pattern contains current letter
                window[letter] = window.get(letter, 0) + 1  # Add the current character to the `window` dictionary
                if window[letter] == pattern[letter]:
                    need -= 1

            while need == 0:  # if pattern is matched
                if right - left + 1 < min_length:  # if current string shorter
                    min_length = right - left + 1  # update shortest window length
                    start = left  # mark the start of the shortest window
        
                # Shrink the window:
                if word1[left] in pattern:
                    if window[word1[left]] == pattern[word1[left]]:
                        need += 1
                    window[word1[left]] -= 1
                left += 1

        return (word1[start: start + min_length] 
                if min_length != len(word1) + 1
                else "")


# O(n), O(n)
# sliding window
class Solution:
    def minWindow(self, word1: str, word2: str) -> str:
        pattern = {}  # stores the frequency of characters in `word2` (target)
        window = {}  # tracks character frequencies in the current sliding window of `word1`
        matched = 0  # counts how many characters in `pattern` have matching frequencies in `window`
        left = 0  # the start pointer for the sliding window
        # shortest = ""  # store the shortest valid substring that contains all characters of `word2`
        start = -1
        min_length = len(word1) + 1

        for letter in word2:
            pattern[letter] = pattern.get(letter, 0) + 1

        for right, letter in enumerate(word1):
            window[letter] = window.get(letter, 0) + 1  # Add the current character to the `window` dictionary

            # If the current character is in `pattern` and its frequency matches
            if (letter in pattern and
                    window[letter] == pattern[letter]):
                matched += 1  # Increment the matched count

            if matched != len(pattern):  # If not all characters in `pattern` are matched `continue`
                continue

            # Shrink the window from the left while keeping it valid
            left_letter = word1[left]
            while (left_letter not in pattern or  # Ignore characters not in the pattern
                   window[left_letter] > pattern[left_letter]):  # Ignore extra occurrences of a valid character
                window[left_letter] -= 1  # Remove the leftmost character from the window
                left += 1  # Move the left pointer forward
                left_letter = word1[left]  # Update the left character

            # Check if the current window is the smallest valid substring so far
            if right - left + 1 < min_length:  # If the current window is smaller
                min_length = right - left + 1  # Update the shortest substring
                start = left

            # Prepare for the next iteration by removing the leftmost character
            # (even if it is significant) and reducing the `matched` count
            window[left_letter] -= 1
            matched -= 1
            left += 1

        # Return the shortest valid substring or an empty string if no such substring exists
        return (word1[start: start + min_length] 
                if min_length != len(word1) + 1
                else "")





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
print(Solution().maxSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3), [3, 3, 5, 5, 6, 7])
print(Solution().maxSlidingWindow([1], 1), [1])
print(Solution().maxSlidingWindow([7, 2, 4], 2), [7, 4])
print(Solution().maxSlidingWindow([1, 3, 1, 2, 0, 5], 3), [3, 3, 2, 5])


# O(n), O(n)
# sliding window, deque, monotonic queue
# monotonically decreasing queue
class Solution:
    def maxSlidingWindow(self, numbers: list[int], window_size: int) -> list[int]:
        window = deque()  # sliding window as a deque
        left = 0  # left pointer
        maxs_list = []  # stores the maximum values for each sliding window.

        for right, number in enumerate(numbers):
            # Remove elements from the back of the deque if they are less than or equal to
            # the current number, as they cannot be the maximum in the current 
            # or any future window. monotonically decreasing queue
            while (window and
                   window[-1][1] <= number):
                window.pop()

            # Remove elements from the front of the deque if they are outside the current window.
            while (window and
                   window[0][0] < left):
                window.popleft()

            window.append((right, number))  # add the index, number pair to the deque.

            if right - left + 1 == window_size:  # When the sliding window reaches the required size, record the maximum value.
                maxs_list.append(window[0][1])  # The maximum value in the current window is at the index `window[0][1]`.
                left += 1  # Slide the window forward by incrementing `left`.

        return maxs_list


# window as a list of indexes
class Solution:
    def maxSlidingWindow(self, numbers: list[int], window_size: int) -> list[int]:
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


# O(n2), O(n)
# brute force
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
print(Solution().evalRPN(["2", "1", "+", "3", "*"]), 9)
print(Solution().evalRPN(["4", "13", "5", "/", "+"]), 6)
print(Solution().evalRPN(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]), 22)
print(Solution().evalRPN(["18"]), 18)


# O(n), O(n)
# stack
class Solution:
    def evalRPN(self, tokens: list[str]) -> int:
        stack = []

        for token in tokens:
            if token == "+":
                stack.append(stack.pop() + stack.pop())
            elif token == "*":
                stack.append(stack.pop() * stack.pop())
            elif token == "-":
                last = stack.pop()
                stack.append(stack.pop() - last)
            elif token == "/":
                last = stack.pop()
                stack.append(int(stack.pop() / last))
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
print(Solution().dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73]), [1, 1, 4, 2, 1, 1, 0, 0])
print(Solution().dailyTemperatures([30, 40, 50, 60]), [1, 1, 1, 0])
print(Solution().dailyTemperatures([30, 60, 90]), [1, 1, 0])


# O(n), O(n)
# stack, monotonic stack
# monotonically decreasing stack
class Solution:
    def dailyTemperatures(self, temps: list[int]) -> list[int]:
        stack = []  # [(day, temperature), ]
        days_to_warmer_day = [0] * len(temps)  # number of days needed to wait after day ith for a warmer day to arrive

        for current_day, temp in enumerate(temps):
            while stack and temp > stack[-1][1]:  # check if temp is lower
                previous_day, _ = stack.pop()  # index of the last day in stack
                days_to_warmer_day[previous_day] = current_day - previous_day  # set days needed to get higher temperature

            stack.append((current_day, temp))  # apped new temp

        return days_to_warmer_day


# O(n2), O(1)
# brute force
class Solution:
    def dailyTemperatures(self, temps: list[int]) -> list[int]:
        days_to_warmer_day = [0] * len(temps)  # number of days needed to wait after day ith for a warmer day to arrive

        for left in range(len(temps)):
            for right in range(left + 1, len(temps)):
                if temps[left] < temps[right]:
                    days_to_warmer_day[left] = right - left
                    break
        
        return days_to_warmer_day





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
print(Solution().carFleet(12, [10, 8, 0, 5, 3], [2, 4, 1, 1, 3]), 3)
print(Solution().carFleet(10, [3], [3]), 1)
print(Solution().carFleet(100, [0, 2, 4], [4, 2, 1]), 1)
print(Solution().carFleet(10, [0, 4, 2], [2, 1, 3]), 1)


# draft
# (12 - 10) / 2 = 1;
# (12 - 8) / 4 = 1;

# (12 - 8) / 1 = 4;
# (12 - 3) / 3 = 3;

# (12 - 0) / 1 = 12;


# ((0, 2), (2, 3), (4, 1)) 10
#     5   <=?  2.66
#              2.66  <=? 6
#     5         6       6


# O(nlogn), O(n)
class Solution:
    def carFleet(self, target: int, positions: list[int], speeds: list[int]) -> int:
        cars = sorted(zip(positions, speeds), reverse=True)  # sort cars statring from ones closest to finish
        previous_ttf = 0  # previous time to finish
        fleets = 0  # count fleets

        for position, speed in cars:
            time_to_finish = (target - position) / speed  # calculate time to finish for curent car
            fleets += time_to_finish > previous_ttf  # if current car time to finish is greater than the next fleet, add another fleet
            previous_ttf = max(previous_ttf, time_to_finish)  # time to finish for the current fleet

        return fleets


# O(nlogn), O(n)
class Solution:
    def carFleet(self, target: int, positions: list[int], speeds: list[int]) -> int:
        cars = sorted(zip(positions, speeds), reverse=True)
        fleets = len(positions)
        times = []
        
        for position, speed in cars:
            time = (target - position) / speed

            if (times and 
                    time <= times[-1]):
                fleets -= 1
                # if the time to tartet is lower than the top stack time to target
                # then the highter time should remain to possibly catch next cars
            else:
                times.append(time)

        return fleets


# O(nlogn), O(n)
class Solution:
    def carFleet(self, target, positions, speeds):
        cars = sorted(zip(positions, speeds), reverse=True)  # sort the cars so to start with the one closest to the target
        fleet_stack = []

        for position, speed in cars:
            time = (target - position) / speed  # time to the target
            
            if fleet_stack and fleet_stack[-1] >= time:  # if the car behind cought up next car
                continue

            fleet_stack.append(time)  # append a car to a stack

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
print(sorted(Solution().subsets([1, 2, 3])), [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]])
print(sorted(Solution().subsets([0])), [[], [0]])
 

# O(n2^n), O(n)
# backtracking
# "dfs" method inside "subsets" method
class Solution:
    def subsets(self, numbers: list[int]) -> list[list[int]]:
        subset = []  # current subset
        subset_list = []  # possible subset list

        def dfs(index):
            if index == len(numbers):  # target index reached
                subset_list.append(subset.copy())  # push subset to subset_list
                return  

            subset.append(numbers[index])  # Include the current element in the subset
            dfs(index + 1)  # Explore the path with the current element
            subset.pop()  # Backtrack by removing the current element from the subset
            dfs(index + 1)  # Explore the path without including the current element

        dfs(0)  # start dfs with level = 0
    
        return subset_list


# O(n2^n), O(n)
# backtracking
# "dfs" and "subset" methods inside a class
class Solution:
    def __init__(self) -> None:
        self.subset = []
        self.subset_list = []

    def subsets(self, numbers: list[int]) -> list[list[int]]:
        self.numbers = numbers
        self.dfs(0)
        
        return self.subset_list

    def dfs(self, level: int) -> None:
        if level == len(self.numbers):
            self.subset_list.append(self.subset.copy())
            return
        
        self.subset.append(self.numbers[level])
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
print(Solution().subsetsWithDup([1, 2, 2]), [[], [1], [1, 2], [1, 2, 2], [2], [2, 2]])
print(Solution().subsetsWithDup([0]), [[], [0]])
print(Solution().subsetsWithDup([4, 4, 4, 1, 4]), [[], [1], [1, 4], [1, 4, 4], [1, 4, 4, 4], [1, 4, 4, 4, 4], [4], [4, 4], [4, 4, 4], [4, 4, 4, 4]])


# O(n2^n), O(n)
# backtracking
class Solution:
    def subsetsWithDup(self, numbers: list[int]) -> list[list[int]]:
        numbers.sort()
        subset = []  # current subset
        subset_list = []  # solution

        def dfs(index):
            if index == len(numbers):  # target level reached
                subset_list.append(subset.copy())  # push subset to subset_list
                return

            subset.append(numbers[index])  # Include the current element in the subset
            dfs(index + 1)  # Explore the path with the current element
            subset.pop()  # Backtrack by removing the current element from the subset

            # Skip over duplicate elements to avoid generating duplicate subsets
            # If number at the `index + 1` (that was poped previously) is the same as
            # the number at current index skip it.
            while (index + 1 < len(numbers) and
                    numbers[index] == numbers[index + 1]):
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
print(Solution().combinationSum([2, 3, 6, 7], 7), [[2, 2, 3], [7]])
print(Solution().combinationSum([2, 3, 5], 8), [[2, 2, 2, 2], [2, 3, 3], [3, 5]])
print(Solution().combinationSum([2], 1), [])


# O(n!), O(n)
# backtracking
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
print(Solution().combinationSum2([10, 1, 2, 7, 6, 1, 5], 8), [[1, 1, 6], [1, 2, 5], [1, 7], [2, 6]])
print(Solution().combinationSum2([2, 5, 2, 1, 2], 5), [[1, 2, 2], [5]])
print(Solution().combinationSum2([6], 6), [[6]])
print(Solution().combinationSum2([2, 2, 2], 2), [[2]])


# O(n2^n), O(n)
# backtracking
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
print(Solution().permute([1, 2, 3]), [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]])
print(Solution().permute([0, 1]), [[0, 1], [1, 0]])
print(Solution().permute([1]), [[1]])


# O(n!), O(n)
# backtracking
class Solution:
    def permute(self, numbers: list[int]) -> list[list[int]]:
        permutation_list = []

        def dfs(left):
            if left == len(numbers):
                permutation_list.append(numbers.copy())
                return
            
            for right in range(left, len(numbers)):
                numbers[left], numbers[right] = numbers[right], numbers[left]
                dfs(left + 1)
                numbers[left], numbers[right] = numbers[right], numbers[left]
        
        dfs(0)
        return permutation_list


# O(n!), O(n!)
# backtracking
class Solution:
    def permute(self, numbers: list[int]) -> list[list[int]]:
        permutation = []  # Current permutation being built
        permutation_list = []  # List to store all permutations
        
        def dfs(numbers):
            if not numbers:  # Base case: when there are no more numbers to permute
                permutation_list.append(permutation.copy())
                return
            
            for index in range(len(numbers)):  # Loop through all numbers to generate permutations
                permutation.append(numbers[index])  # Choose the current number and add it to the permutation
                dfs(numbers[:index] + numbers[index + 1:])  # Recurse with the remaining numbers (excluding the chosen one)
                permutation.pop()  # Backtrack by removing the last added number

        dfs(numbers)  # Start the recursive DFS with the initial list of numbers

        return permutation_list


# O(n!), O(n!)
# backtracking
class Solution:
    def permute(self, numbers: list[int]) -> list[list[int]]:
        permutation_list = []  # List to store all permutations
        
        def dfs(permutation, numbers):
            if not numbers:  # Base case: when there are no more numbers to permute
                permutation_list.append(permutation)
                return
            
            for index in range(len(numbers)):  # Loop through all numbers to generate permutations
                dfs(permutation + [numbers[index]], 
                    numbers[:index] + numbers[index + 1:])

        dfs([], numbers)  # Start the recursive DFS with empy permutation list and the initial list of numbers

        return permutation_list





# Word Search
# https://leetcode.com/problems/word-search/
"""
Given an m x n grid of characters board and a string word, return true if word exists in the grid.
UUu
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
print(Solution().exist([["C", "A", "A"]], "AA"), True)
print(Solution().exist([["C", "A", "A"], ["C", "C", "B"]], "AAB"), True)
print(Solution().exist([["C", "A", "A"], ["A", "A", "A"], ["B", "C", "D"]], "AAB"), True)
print(Solution().exist([["C", "A", "A"], ["A", "A", "A"], ["B", "C", "D"]], "AACA"), True)
print(Solution().exist([["A", "A"]], "AAA"), False)
print(Solution().exist([["A", "B", "C", "E"], ["S", "F", "E", "S"], ["A", "D", "E", "E"]], "ABCEFSADEESE"), True)
print(Solution().exist([["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]], "AB"), True)
print(Solution().exist([["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]], "AZ"), False)
print(Solution().exist([["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]], "ABFS"), True)
print(Solution().exist([["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]], "ABCCED"), True)
print(Solution().exist([["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]], "SEE"), True)
print(Solution().exist([["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]], "ABCB"), False)

# ["A", "B", "C", "E"], 
# ["S", "F", "E", "S"], 
# ["A", "D", "E", "E"]


# O(nm3^k), O(nm)
# backtracking
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
                word[index] == board[row][col + 1] and  # Check if the current position matches the word's character
                (row, col + 1) not in tabu and  # if cell is not in tabo set
                    dfs(row, col + 1, index + 1)):  # Switch to that letter and check its neighbors
                return True

            if (row + 1 < rows and
                word[index] == board[row + 1][col] and
                (row + 1, col) not in tabu and
                    dfs(row + 1, col, index + 1)):
                return True
        
            if (col - 1 >= 0 and
                word[index] == board[row][col - 1] and
                (row, col - 1) not in tabu and
                    dfs(row, col - 1, index + 1)):
                return True

            if (row - 1 >= 0 and
                word[index] == board[row - 1][col] and
                (row - 1, col) not in tabu and
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
print(Solution().partition("a"), [["a"]])
print(Solution().partition("aa"), [['a', 'a'], ['aa']])
print(Solution().partition("ab"), [["a", "b"]])
print(Solution().partition("aaa"), [['a', 'a', 'a'], ['a', 'aa'], ['aa', 'a'], ['aaa']])
print(Solution().partition("aab"), [["a", "a", "b"], ["aa", "b"]])
print(Solution().partition("aba"), [["a", "b", "a"], ["aba"]])


# O(n2^n), O(2^n)
# Space O(2^n) because every dfs call contains new string
# backtracking `palindrome` as a global variable
class Solution:
    def partition(self, word):
        palindrome = []  # This will track the current partition
        palindrome_list = []  # This will store all valid palindrome partitions
    
        def is_palindrome(word):
            return word == word[::-1]

        def dfs(word):
            if not word:  # if word is empty that means all letters folded into palindrom
                palindrome_list.append(palindrome.copy())
                return

            for index in range(len(word)):  # for every index in "word"
                prefix = word[: index + 1]  # Current prefix to check

                if is_palindrome(prefix):  # if prefix is a palindrme
                    palindrome.append(prefix)  # Add it to the current partition
                    dfs(word[index + 1 :])  # Explore the path with the current palindrome and look for the palindrome in the next part of the "word"
                    palindrome.pop()  # Backtrack by removing the last added palindrome

        dfs(word)  # Start DFS with "word"
        return palindrome_list


# O(n2^n), O(2^n)
# Space O(2^n) because every dfs call contains new string
# `palindrome` as a function variable
class Solution:
    def partition(self, word):
        palindrome_list = []

        def dfs(palindrome, word):
            if not word:
                palindrome_list.append(palindrome)
                return
            
            for index in range(len(word)):
                substring = word[: index + 1]

                if substring == substring[::-1]:
                    dfs(palindrome + [substring], word[index + 1 :])

        dfs([], word)
        
        return palindrome_list


# O(n2^n), O(2^n)
# Space O(2^n) because every `substring` contains new string
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
print(Solution().letterCombinations("2"), ["a", "b", "c"])
print(Solution().letterCombinations("23"), ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"])
print(Solution().letterCombinations(""), [])


# O(n2^n), O(2^n)
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
    def letterCombinations(self, digits: str) -> list[str]:
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
print(Solution().generateParenthesis(1), ["()"])
print(Solution().generateParenthesis(2), ["(())", "()()"])
print(Solution().generateParenthesis(3), ["((()))", "(()())", "(())()", "()(())", "()()()"])


# O(n2^n), O(n)
# backtracking
class Solution:
    def generateParenthesis(self, number: int) -> list[str]:
        parenthesis = []  # current parenthesis sequence
        parenthesis_list = []  # list of parenthesis sequences

        def dfs(opened, closed):
            if opened + closed == 2 * number:  # if all opening and closing parenthesis are used
                parenthesis_list.append("".join(parenthesis))  # append current sequence
                return

            if opened < number:  # not all "(" have been used
                parenthesis.append("(")
                dfs(opened + 1, closed)  # check this branch
                parenthesis.pop()  # backtrack

            if closed < opened:  # the number of ")" must not be greater than "("
                parenthesis.append(")")
                dfs(opened, closed + 1)  # check this branch
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
print(Solution().minCostClimbingStairs([10, 15, 20]), 15)
print(Solution().minCostClimbingStairs([1, 100, 1, 1, 1, 100, 1, 1, 100, 1]), 6)


# draft
# cost to move from the i-th step       [1,100,1,1,1,100,1,1,100,1]
# cumulative cost to get to i-th step   [0, 0, 1,1,2, 2, 3,4, 5, 5, 6]
# cumulative cost to get from i-th step [1,100,2,3,3,103,4,5,104,6]


class Solution:
    def minCostClimbingStairs(self, cost: list[int]) -> int:
        """
        O(n), O(1)
        dp, bottom-up
        """
        cache = [cost[0], cost[1]]

        for index in range(2, len(cost)):
            prev = cache[0]
            prev_prev = cache[1]
            cache = [cache[1], cost[index] + min(prev, prev_prev)]

        return min(cache[-1], cache[-2])


class Solution:
    def minCostClimbingStairs(self, cost: list[int]) -> int:
        """
        O(n), O(1)
        dp, bottom-up
        """
        a = cost[0]
        b = cost[1]

        for index in range(2, len(cost)):
            a, b = b, cost[index] + min(a, b)

        return min(a, b)


class Solution:
    def minCostClimbingStairs(self, cost: list[int]) -> int:
        """
        O(n), O(1)
        dp, bottom-up
        mutate input list
        """
        cache = [None] * len(cost)
        cache[0] = cost[0]
        cache[1] = cost[1]

        for index in range(2, len(cost)):
            cost[index] = cost[index] + min(cost[index - 1], cost[index - 2])

        return min(cost[-1], cost[-2])


class Solution:
    def minCostClimbingStairs(self, cost: list[int]) -> int:
        """
        O(n), O(n)
        dp, bottom-up
        """
        cache = [None] * len(cost)
        cache[0] = cost[0]
        cache[1] = cost[1]

        for index in range(2, len(cost)):
            cache[index] = cost[index] + min(cache[index - 1], cache[index - 2])

        return min(cache[-1], cache[-2])


class Solution:
    def minCostClimbingStairs(self, cost: list[int]) -> int:
        """
        O(n), O(n)
        dp, top-down with memoization as hash map
        """
        # Memoization dictionary to store the minimum cost to reach the top from each step
        memo = {}

        def dfs(index):
            # Base case: if we're already at or beyond the top, no cost is needed
            if index >= len(cost):
                # No extra cost beyond the last step
                return 0
            # If the result is already computed, return it
            elif index in memo:
                return memo[index]
            
            # Recursively compute the cost of taking 1 step or 2 steps
            # Store the result in the memo dictionary
            memo[index] = cost[index] + min(dfs(index + 1), dfs(index + 2))
            
            return memo[index]

        # Start from step 0 or step 1, whichever is cheaper
        return min(dfs(0), dfs(1))


class Solution:
    def minCostClimbingStairs(self, cost: list[int]) -> int:
        """
        O(n), O(n)
        dp, top-down with memoization as list
        """
        memo = [None] * len(cost)

        def dfs(index):
            if index >= len(cost):
                return 0  # No extra cost beyond the last step
            elif memo[index] is not None:
                return memo[index]

            memo[index] = cost[index] + min(dfs(index + 1), dfs(index + 2))
            return memo[index]

        return min(dfs(0), dfs(1))


class Solution:
    def minCostClimbingStairs(self, cost: list[int]) -> int:
        """
        O(2^n), O(n)
        brute force, pure recursion, tle
        converts to top-down
        """
        def dfs(index):
            if index >= len(cost):
                return 0
            
            return cost[index] + min(dfs(index + 1), dfs(index + 2))

        return min(dfs(0), dfs(1))


class Solution:
    def minCostClimbingStairs(self, cost: list[int]) -> int:
        """
        O(2^n), O(n)
        brute force, function argument, tle
        """
        self.min_total_cost = sum(cost)

        def dfs(index, total_cost):
            if index >= len(cost):
                self.min_total_cost = min(self.min_total_cost, total_cost)
                return

            dfs(index + 1, total_cost + cost[index])
            dfs(index + 2, total_cost + cost[index])

        dfs(0, 0)
        dfs(1, 0)
        return self.min_total_cost


class Solution:
    def minCostClimbingStairs(self, cost: list[int]) -> int:
        """
        O(2^n), O(n)
        brute force, backtracking, tle
        """
        self.total_cost = 0
        self.min_total_cost = sum(cost)

        def dfs(index):
            if index >= len(cost):
                self.min_total_cost = min(self.min_total_cost, self.total_cost)
                return

            self.total_cost += cost[index]
            dfs(index + 1)
            dfs(index + 2)
            self.total_cost -= cost[index]


        dfs(0)
        dfs(1)
        return self.min_total_cost


class Solution:
    def minCostClimbingStairs(self, cost: list[int]) -> int:
        """
        O(2^n), O(n)
        brute force, function argument, tle
        """
        def dfs(index, total_cost):
            if index >= len(cost):
                return total_cost

            return min(
                dfs(index + 1, total_cost + cost[index]),
                dfs(index + 2, total_cost + cost[index]))

        return min(dfs(0, 0), dfs(1, 0))





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
print(Solution().numDecodings("5"), 1)
print(Solution().numDecodings("226"), 3)
print(Solution().numDecodings("2261"), 3)
print(Solution().numDecodings("12"), 2)
print(Solution().numDecodings("2101"), 1)
print(Solution().numDecodings("06"), 0)
print(Solution().numDecodings("0"), 0)
print(Solution().numDecodings(""), 0)
print(Solution().numDecodings("111111111111111111111111111111111111111111111"), 1836311903)


"""
draft
[2, 2, 6, 1]
 2  2  6  1
 2    26  1
 22    6  1
                  .
            /           \
           2            22
        /     \       /
       2      26     6
     /       /     /
    6       1     1
   /
  1

[3, 2, 1, 1, (1)]
{0: 3, 1: 2, 2: 1, 3: 1, 4: 1}
"""


class Solution:
    def numDecodings(self, code: str) -> int:
        """
        O(n), O(1)
        dp, bottom-up
        """
        if not code:
            return 0
        
        cache = [1, 1]  # `1` when the whole number is partitioned

        for index in reversed(range(len(code))):
            cache_0 = 0

            if code[index] != "0":
                cache_0 = cache[0]

            if (index + 1 < len(code) and
                (code[index] == "1" or 
                 (code[index] == "2" and 
                  code[index + 1] <= "6"))):
                cache_0 += cache[1]

            cache = [cache_0, cache[0]]

        return cache[0]


class Solution:
    def numDecodings(self, code: str) -> int:
        """
        O(n), O(n)
        dp, bottom-up, list as cache
        """
        if not code:
            return 0
        
        cache = [0] * (len(code) + 1)
        cache[len(code)] = 1  # `1` when the whole number is partitioned

        for index in reversed(range(len(code))):
            if code[index] != "0":
                cache[index] += cache[index + 1]

            if (index + 1 < len(code) and
                (code[index] == "1" or 
                 (code[index] == "2" and 
                  code[index + 1] <= "6"))):
                cache[index] += cache[index + 2]

        return cache[0]


class Solution:
    def numDecodings(self, code: str) -> int:
        """
        O(n), O(n)
        dp, bottom-up, hash map as cache
        """
        if not code:
            return 0
        
        cache = {len(code): 1}  # `1` when the whole number is partitioned

        for index in reversed(range(len(code))):
            cache[index] = 0
            
            if code[index] != "0":
                cache[index] += cache[index + 1]

            if (index + 1 < len(code) and
                (code[index] == "1" or 
                 (code[index] == "2" and 
                  code[index + 1] <= "6"))):
                cache[index] += cache[index + 2]

        return cache[0]


class Solution:
    def numDecodings(self, code: str) -> int:
        """
        O(n), O(n)
        dp, top-down with memoization as hash map
        """
        if not code:  # if "code" is empty
            return 0
        
        memo = {len(code): 1}  # `1` when the whole number is partitioned

        def dfs(index):            
            if index in memo:  # if memoized
                return memo[index]  # Return memoized result if already computed.
            elif code[index] == "0":  # check if number is statring with 0
                return 0  # inwalid number
            
            # one digit number case
            memo[index] = dfs(index + 1)  # Proceed to decode the next number.

            # two digits number case
            if (index + 1 < len(code) and  # check if second digit within bounds
                (code[index] == "1" or  # two digit number starts with one or
                 (code[index] == "2" and  # two digit number starts with two and 
                  code[index + 1] <= "6"))):  # ends with less equal to six
                memo[index] += dfs(index + 2)  # Add the result of two-digit decoding.

            return memo[index]  # Return the result for this index.

        return dfs(0)


class Solution:
    def numDecodings(self, code: str) -> int:
        """
        O(n), O(n)
        dp, top-down with memoization as list
        """
        if not code:  # if "code" is empty
            return 0
        
        memo = [None] * (len(code) + 1)
        memo[len(code)] = 1  # `1` when the whole number is partitioned

        def dfs(index):
            if memo[index] is not None:
                return memo[index]
            elif code[index] == "0":  # check if number is statring with 0
                return 0  # inwalid number
            
            # one digit number case
            memo[index] = dfs(index + 1)  # Proceed to decode the next number.

            # two digits number case
            if (index + 1 < len(code) and  # check if second digit within bounds
                (code[index] == "1" or  # two digit number starts with one or
                 (code[index] == "2" and  # two digit number starts with two and 
                  code[index + 1] <= "6"))):  # ends with less equal to six
                memo[index] += dfs(index + 2)  # Add the result of two-digit decoding.

            return memo[index]  # Return the result for this index.

        return dfs(0)


class Solution:
    def numDecodings(self, code: str) -> int:
        """
        O(2^n), O(n)
        brute force, tle
        """
        if not code:  # if "code" is empty
            return 0
        
        def dfs(index):
            if index == len(code):
                return 1  # `1` when the whole number is partitioned
            elif code[index] == "0":  # check if number is statring with 0
                return 0  # inwalid number
            
            # one digit number case
            one_digit_number = dfs(index + 1)  # Proceed to decode the next number.

            # two digits number case
            two_digit_number = 0
            if (index + 1 < len(code) and  # check if second digit within bounds
                (code[index] == "1" or  # two digit number starts with one or
                 (code[index] == "2" and  # two digit number starts with two and 
                  code[index + 1] <= "6"))):  # ends with less equal to six
                two_digit_number = dfs(index + 2)  # Add the result of two-digit decoding.

            return one_digit_number + two_digit_number  # Return the result for this index.

        return dfs(0)


class Solution:
    def numDecodings(self, word: str) -> int:
        """
        O(2^n), O(n)
        brute force, backtracking, tle
        """
        if (not word or word[0] == "0"):
            return 0

        decoded = []
        decoded_list = []

        def dfs(index):
            if index == len(word):
                decoded_list.append(decoded.copy())
                return

            if (index == len(word) - 1 or  # if last digit or
                    index + 1 < len(word) and  # next index in bounds
                    word[index + 1] != "0"):  # next digit is not zero

                if word[index] != "0":
                    decoded.append(word[index])
                    dfs(index + 1)
                    decoded.pop()

            if (index == len(word) - 2 or  # if two last digits or
                    index + 2 < len(word) and  # next index in bounds
                    word[index + 2] != "0"):  # next digit is not zero

                if (index + 1 < len(word) and
                        word[index: index + 2] <= "26"):
                    decoded.append(word[index: index + 2])
                    dfs(index + 2)
                    decoded.pop()

        dfs(0)
        return len(decoded_list)


class Solution:
    def numDecodings(self, code: str) -> int:
        """
        O(2^n), O(n)
        brute force, backtracking, tle
        """
        if not code:
            return 0

        decoded = []
        decoded_list = []

        def dfs(index):
            if index == len(code):
                decoded_list.append(decoded.copy())
                return

            # check if next digit is not zero
            is_not_zero = True
            if (index + 1 < len(code) and
                    code[index + 1] == "0"):
                is_not_zero = False

            if (code[index] != "0" and
                    is_not_zero):
                decoded.append(code[index])
                dfs(index + 1)
                decoded.pop()

            # check if next next digit is not zero
            is_not_zero = True
            if (index + 2 < len(code) and
                    code[index + 2] == "0"):
                is_not_zero = False

            if (code[index] != "0" and
                code[index] <= "2" and
                index + 1 < len(code) and
                code[index + 1] <= "6" and
                    is_not_zero):

                decoded.append(code[index: index + 2])
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
        if index < len(node_list) and node_list[index] is not None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] is not None:
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
        if index < len(node_list) and node_list[index] is not None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] is not None:
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
        if index < len(node_list) and node_list[index] is not None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] is not None:
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
        if index < len(node_list) and node_list[index] is not None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] is not None:
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
        if index < len(node_list) and node_list[index] is not None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] is not None:
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
        if index < len(node_list) and node_list[index] is not None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] is not None:
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
        if index < len(node_list) and node_list[index] is not None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] is not None:
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
        if index < len(node_list) and node_list[index] is not None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] is not None:
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

                if node.left is not None:  # if left subnode is not empty
                    queue.append(node.left)  # append it to queue
                    current_level_list.append(node.left.val)  # append its value to current level solution
                
                if node.right is not None:  # if right subnode is not empty
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
        if index < len(node_list) and node_list[index] is not None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] is not None:
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
        if index < len(node_list) and node_list[index] is not None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] is not None:
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
(Solution().isValidBST(build_tree_from_list([0, -1])), True)
(Solution().isValidBST(build_tree_from_list([5, 4, 6, None, None, 3, 7])), False)


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
        if index < len(node_list) and node_list[index] is not None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] is not None:
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
        def dfs(node, min_val, max_val):
            if (not node.left and 
                    not node.right):
                return True

            is_left = not node.left
            is_right = not node.right
            
            if (node.left and 
                   min_val < node.left.val < node.val):
                is_left = dfs(node.left, min_val, node.val)
            
            if (node.right and 
                   node.val < node.right.val < max_val):
                is_right = dfs(node.right, node.val, max_val)

            return is_left and is_right

        return dfs(root, float("-inf"), float("inf"))


# dp, dfs, recursion, in-order traversal
# O(n), O(n)
class Solution:
    def isValidBST(self, root: TreeNode | None) -> bool:
        def dfs(node, left_min, right_max):
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
        if index < len(node_list) and node_list[index] is not None:
            node.left = node_type(node_list[index])
            queue.append(node.left)
        index += 1

        # Assign the right child if available
        if index < len(node_list) and node_list[index] is not None:
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
print(Solution().combinationSum4([5], 5), 1)
print(Solution().combinationSum4([2, 3], 7), 3)
print(Solution().combinationSum4([1, 2, 3], 4), 7)
print(Solution().combinationSum4([9], 3), 0)
print(Solution().combinationSum4([4, 2, 1], 32), 39882198)


class Solution:
    def combinationSum4(self, numbers: list[int], target: int) -> int:
        """
        O(n2), O(n)
        dp, bottom-up with tabulation as list
        """
        # Initialize a list of zeros for tabulation, where tab[i] is the number of ways to make sum i
        cache = [0] * (target + 1)
        # Base case: 1 way to make target 0 (empty combination)
        cache[0] = 1

        for index in range(1, target + 1):
            for number in numbers:
                # If num can be subtracted from index, add the number of ways to make (index - number)
                if index - number >= 0:
                    cache[index] += cache[index - number]
            
        return cache[-1]


class Solution:
    def combinationSum4(self, numbers: list[int], target: int) -> int:
        """
        O(n2), O(n)
        dp, bottom-up with tabulation as hash map
        """
        # Tabulation dictionary, storing base case: 1 way to make target 0 (empty combination)
        cache = {0: 1}

        for index in range(1, target + 1):
            for number in numbers:
                if index - number >= 0:
                    # If the number can be used (valid combination), update the tabulation table
                    cache[index] = (cache.get(index, 0) + 
                                    cache.get(index - number, 0))
            
        return cache.get(target, 0)


class Solution:
    def combinationSum4(self, numbers: list[int], target: int) -> int:
        """
        O(n2), O(n)
        dp, top-down with memoization as hash map
        """
        # Memoization dictionary, storing base case: 1 way to make target 0 (empty combination)
        memo = {0: 1}

        def dfs(target: int) -> int:
            if target in memo:
                return memo[target]
            elif target < 0:
                return 0

            # Recursively compute number of combinations by reducing the target (target - number)
            memo[target] = sum(dfs(target - number)
                               for number in numbers)
            return memo[target]

        return dfs(target)


class Solution:
    def combinationSum4(self, numbers: list[int], target: int) -> int:
        """
        O(n2), O(n)
        dp, top-down with memoization as list
        """
        # Memoization list, storing base case: 1 way to make target 0 (empty combination)
        memo = [None] * (target + 1)
        memo[0] = 1

        def dfs(target: int) -> int:
            if target < 0:
                return 0
            elif memo[target] is not None:
                return memo[target]

            # Recursively compute number of combinations by reducing the target (target - number)
            memo[target] = sum(dfs(target - number)
                               for number in numbers)
            return memo[target]

        return dfs(target)


class Solution:
    def combinationSum4(self, numbers: list[int], target: int) -> int:
        """
        O(2^n), O(n)
        brute force, tle
        converts to top-down
        """
        def dfs(target: int) -> int:
            if target == 0:
                return 1
            elif target < 0:
                return 0

            return sum(dfs(target - number)
                       for number in numbers)

        return dfs(target)


class Solution:
    def combinationSum4(self, numbers: list[int], target: int) -> int:
        """
        O(2^n), O(n)
        brute force, backtracking, tle
        """
        combination = []
        combination_list = []
        self.combination_counter = 0

        def dfs(target): #
            if target < 0:
                return
            elif target == 0:
                # combination_list.append(combination.copy())
                # combination_list.append(True)
                self.combination_counter += 1
                return

            for number in numbers:
                combination.append(number)
                dfs(target - number)
                combination.pop()

        dfs(target)
        # return combination_list
        # return len(combination_list)
        return self.combination_counter





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
print(Solution().numIslands([["0"]]), 0)
print(Solution().numIslands([["1"]]), 1)
print(Solution().numIslands([["0", "0"], ["0", "1"]]), 1)
print(Solution().numIslands([["1", "0"], ["0", "1"]]), 2)
print(Solution().numIslands([["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]]), 3)
print(Solution().numIslands([["1", "1", "0"], ["0", "1", "0"], ["0", "0", "1"]]), 2)
print(Solution().numIslands([["1", "1", "1", "1", "0"], ["1", "1", "0", "1", "0"], ["1", "1", "0", "0", "0"], ["0", "0", "0", "0", "0"]]), 1)
print(Solution().numIslands([["1", "1", "0", "0", "0"], ["1", "1", "0", "0", "0"], ["0", "0", "1", "0", "0"], ["0", "0", "0", "1", "1"]]), 3)


# O(n2), O(n2)
# dfs, recursion
class Solution:
    """
    Starts from not visited land tile, check recursively all neighbours and 
    mark them as visited.
    Add one to land counter.

    Boundary checks: It includes all the boundary 
    and visited checks directly in the base case of the recursion.
    """

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


# O(n2), O(n2)
# dfs, recursion
class Solution:
    """
    Starts from land tile and check recursively all neighbours.
    Mark every visited cell and return area (True) to be counted as a one island.
    Skip visited lands.
    """
    
    def numIslands(self, grid: list[list[str]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        visited_land = set()
        island_counter = 0

        def dfs(row, col):
            if (row < 0 or 
                row == rows or 
                col < 0 or 
                col == cols or 
                grid[row][col] == "0" or 
                    (row, col) in visited_land):
                return 0
            
            visited_land.add((row, col))

            return (
                1 +
                dfs(row - 1, col) + 
                dfs(row + 1, col) +
                dfs(row, col - 1) +
                dfs(row, col + 1)
            )

        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == "1":
                    if dfs(row, col):
                        island_counter += 1

        return island_counter


# O(n2), O(n2)
# dfs, recursion
class Solution:
    """
    Starts from land tile and check recursively all neighbours and 
    mark them as water tiles. Add one to land counter.
    Skip visited water.
    """
    
    def numIslands(self, grid: list[list[str]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        island_counter = 0

        def dfs(row, col):
            if (row < 0 or 
                row == rows or 
                col < 0 or 
                col == cols or 
                    grid[row][col] == "0"):
                return
            
            grid[row][col] = "0"

            dfs(row - 1, col)
            dfs(row + 1, col)
            dfs(row, col - 1)
            dfs(row, col + 1)

        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == "1":
                    dfs(row, col)
                    island_counter += 1

        return island_counter


from collections import deque

# O(n2), O(n2)
# bfs, iterative
# dfs, iterative with queue.pop()
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
                for i, j in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    r = row + i
                    c = col + j

                    # Check if the neighbor is within bounds, unvisited, and land.
                    if (0 <= r < rows and
                        0 <= c < cols and
                        grid[r][c] == "1" and
                            (r, c) not in visited_land):
                        queue.append((r, c))  # Add the land cell to the queue.
                        visited_land.add((r, c))  # Mark it as visited.

        # Iterate over each cell in the grid.
        for row in range(rows):
            for col in range(cols):
                # If the cell is land and hasn't been visited, start a new BFS.
                if (grid[row][col] == "1" and 
                        (row, col) not in visited_land):
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
print(Solution().maxAreaOfIsland([[0]]), 0)
print(Solution().maxAreaOfIsland([[1]]), 1)
print(Solution().maxAreaOfIsland([[0, 0], [0, 1]]), 1)
print(Solution().maxAreaOfIsland([[1, 0], [0, 1]]), 1)
print(Solution().maxAreaOfIsland([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 1)
print(Solution().maxAreaOfIsland([[1, 1, 0], [0, 1, 0], [0, 0, 1]]), 3)
print(Solution().maxAreaOfIsland([[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]]), 6)
print(Solution().maxAreaOfIsland([[0, 0, 0, 0, 0, 0, 0, 0]]), 0)


# O(n2), O(n2)
# dfs, recursion
class Solution:
    def maxAreaOfIsland(self, grid: list[list[int]]) -> int:
        # Get the number of rows and columns in the grid
        rows = len(grid)
        cols = len(grid[0])
        visited_land = set()  # Set to keep track of the visited land cells
        max_island_area = 0  # Variable to store the maximum area of an island found so far
        
        def dfs(row, col):  # Depth-First Search (DFS) function to explore an island
            # Check if the new cell is within bounds, hasn't been visited, and is land (grid[i][j] == 1)
            if (row < 0 or
                row == rows or
                col < 0 or
                col == cols or
                not grid[row][col] or
                    (row, col) in visited_land):
                return 0
                     
            visited_land.add((row, col))  # Mark the current cell as visited
            
            return (  # Return the area of the current cell (1) plus the adjacent area found
                1 +
                dfs(row - 1, col) +
                dfs(row + 1, col) +
                dfs(row, col - 1) +
                dfs(row, col + 1)
            )
                
        for row in range(rows):  # Iterate through each cell in the grid
            for col in range(cols):
                max_island_area = max(max_island_area, dfs(row, col))  # Perform DFS from this cell and update the maximum island area
        
        return max_island_area  # Return the largest island area found


from collections import deque

# O(n2), O(n2)
# bfs, iteration
# dfs, iteration with queue.pop()
class Solution:
    def maxAreaOfIsland(self, grid: list[list[int]]) -> int:
        rows = len(grid)  # Get the number of rows and columns in the grid
        cols = len(grid[0])
        visited_land = set()  # Set to keep track of the visited land cells
        max_island_area = 0  # Variable to store the maximum area of an island found so far
        
        def bfs(row, col):  # Depth-First Search (DFS) function to explore an island
            visited_land.add((row, col))  # Mark the current cell as visited
            directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]  # Possible directions to move: right, down, up, and left            
            queue = deque()
            queue.append((row, col))
            island_area = 1

            while queue:
                row, col = queue.popleft()  # Dequeue the next cell to explore.

                for i, j in directions:  # Explore the neighboring cells
                    r = row + i  # Row in the new direction
                    c = col + j  # Column in the new direction

                    if (0 <= r < rows and  # Check if the new cell is within bounds, hasn't been visited, and is land (grid[i][j] == 1)
                        0 <= c < cols and
                        grid[r][c] == 1 and
                            (r, c) not in visited_land):
                        queue.append((r, c))  # add land cell to queue explore the neighboring land cell and add its area
                        island_area += 1  # increase island area
                        visited_land.add((r, c))  # Mark it as visited.
            
            return island_area  # Return the area of the current cell (1) plus the adjacent area found
                
        for row in range(rows):  # Iterate through each cell in the grid
            for col in range(cols):
                if (grid[row][col] == 1 and  # If the current cell is land and hasn't been visited yet
                        not (row, col) in visited_land):
                    max_island_area = max(max_island_area, bfs(row, col))  # Perform DFS from this cell and update the maximum island area
        
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
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


# O(n), O(n)
# dfs, recursion
class Solution:
    def cloneGraph(self, node: Node | None) -> Node | None:
        org_to_copy = {}  # {original node: copy node}, Dictionary to map original nodes to their clones

        def dfs(node):
            if node in org_to_copy:  # If the node is already cloned, return the clone
                return org_to_copy[node]

            node_copy = Node(node.val)  # Create a new node with the same value
            org_to_copy[node] = node_copy  # Map the original node to the new clone

            for neighbor in node.neighbors:  # Iterate through all neighbors
                node_copy.neighbors.append(dfs(neighbor))  # Recursively clone neighbors and add to the clone's neighbor list
            
            return node_copy  # Return the cloned node

        return dfs(node) if node else None  # Return cloned graph or None if input node is None


#  Input: [[2,4],[1,3],[2,4],[1,3]]
# Output: [[2,4],[1,3],[2,4],[1,3]]

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
print(Solution().islandsAndTreasure([[0, -1], [2147483647, 2147483647]]), [[0, -1], [1, 2]])
print(Solution().islandsAndTreasure([[2147483647, 2147483647, 2147483647], [2147483647, -1, 2147483647], [0, 2147483647, 2147483647]]), [[2, 3, 4], [1, -1, 3], [0, 1, 2]])
print(Solution().islandsAndTreasure([[2147483647, -1, 0, 2147483647], [2147483647, 2147483647, 2147483647, -1], [2147483647, -1, 2147483647, -1], [0, -1, 2147483647, 2147483647]]), [[3, -1, 0, 1], [2, 2, 1, -1], [1, -1, 2, -1], [0, -1, 3, 4]])


# O(n4), O(n2)
# may vistit the same land more than once
# dfs, recursion
class Solution:
    def islandsAndTreasure(self, grid: list[list[int]]) -> None:
        # def wallsAndGates(self, grid: list[list[int]]) -> None:
        rows = len(grid)  # Get the number of rows
        cols = len(grid[0])  # Get the number of columns
        directions = ((-1, 0), (1, 0), (0, -1), (0, 1))  # Possible 4 directions

        def dfs(row, col, distance):
            if (row < 0 or
                row == rows or  # Check if row out of bounds
                col < 0 or
                col == cols or  # Check if column out of bounds
                grid[row][col] == -1 or  # check if water
                (distance and grid[row][col] == 0) or  # check if starting treasure
                    grid[row][col] < distance):  # check if shorter distance is already found
                return

            grid[row][col] = distance  # Mark the current cell with the distance

            for i, j in directions:
                dfs(row + i, col + j, distance + 1)

        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == 0:  # check if the cell is a treasure
                    dfs(row, col, 0)  # Start DFS with distance 0

        return grid


from collections import deque

# O(n2), O(n2)
# bfs, iteration, deque
class Solution:
    def islandsAndTreasure(self, grid: list[list[int]]) -> None:
    # def wallsAndGates(self, grid: list[list[int]]) -> None:
        rows = len(grid)
        cols = len(grid[0])
        directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
        queue = deque()  # Initialize a queue for BFS
        visited = set()  # Set to keep track of visited land

        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == 0:  # If the cell is land (0)
                    queue.append((row, col))  # Add the land cell to the queue

        distance = 0  # Initialize distance from the treasure
        while queue:  # While there are cells to process
            for _ in range(len(queue)):  # Process each cell in the current layer
                row, col = queue.popleft()  # Get the next cell freom the queue

                # Check if the cell is in bounds, not water (-1), and not visited    
                if (0 <= row < rows and
                    0 <= col < cols and
                    grid[row][col] != -1 and
                        (row, col) not in visited):

                    grid[row][col] = distance  # Set the current distance in the grid
                    visited.add((row, col))  # Mark the cell as visited

                    for i, j in directions:
                        queue.append((row + i, col + j))  # Add the cell to the queue
                    
            distance += 1  # Increment distance for the next layer

        return grid


# O(n2), O(n2)
# bfs, iteration, deque
from collections import deque  # Import deque for efficient queue operations

class Solution:
    def islandsAndTreasure(self, grid: list[list[int]]) -> None:
    # def wallsAndGates(self, grid: list[list[int]]) -> None:
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

# O(n), O(n)
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
# linked list, doubly linked list
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


# O(n), O(n)
# stack
# monotonically increasing stack
class Solution:
    def largestRectangleArea(self, heights: list[int]) -> int:
        stack = []  # [(index, height), ]
        max_area = 0

        for index, height in enumerate(heights):
            width = 0
            while stack and height < stack[-1][1]:
                width = index - stack[-1][0]
                max_area = max(max_area, width * stack[-1][1])
                stack.pop()

            stack.append((index - width, height))  # width: extend the width (index) by width of popped heights
            
        while stack:
            width = len(heights) - stack[-1][0]
            max_area = max(max_area, width * stack[-1][1])
            stack.pop()

        return max_area


# O(n), O(n)
# stack
class Solution:
    def largestRectangleArea(self, heights: list[int]) -> int:
        # Stack to keep track of indices and heights of rectangles
        stack = []
        # Variable to store the maximum area found so far
        max_area = 0

        for index, height in enumerate(heights):
            # Store the starting index for the current rectangle
            prev_index = index

            # If the current height is less than the height of the rectangle
            # on top of the stack, calculate the area of the rectangle
            # with the height on top of the stack as the smallest height
            while stack and height < stack[-1][1]:
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


# O(n2), O(1)
# brute force
class Solution:
    def largestRectangleArea(self, heights: list[int]) -> int:
        max_area = 0

        for left in range(len(heights)):
            min_height = heights[left]
            
            for right in range(left, len(heights)):
                min_height = min(min_height, heights[right])
                current_area = min_height * (right - left + 1)
                max_area = max(max_area, current_area)
        
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


# O(n), O(n)
# stack
class Solution:
    def calPoints(self, operations: list[str]) -> int:
        stack = []

        for operation in operations:
            if operation == "C":
                stack.pop()
            elif operation == "D":
                stack.append(stack[-1] * 2)
            elif operation == "+":
                prev = stack.pop()
                last = stack[-1] + prev
                stack.extend([prev, last])
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

            if right - left + 1 == window_length + 1:
                window.remove(numbers[left])  # discard
                left += 1

        return False


# O(n2), O(1)
# brute force
class Solution:
    def containsNearbyDuplicate(self, numbers: list[int], window_length: int) -> bool:
        for index in range(len(numbers) - window_length):
            subarray = numbers[index: index + window_length + 1]
            if len(subarray) != len(set(subarray)):
                return True
            
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
        self.reversed = []


    def push(self, x: int) -> None:
        self.stack.append(x)


    def pop(self) -> int:
        if self.reversed:
            return self.reversed.pop()

        while self.stack:
            self.reversed.append(self.stack.pop())
        
        return self.reversed.pop()
        

    def peek(self) -> int:
        if self.reversed:
            return self.reversed[-1]

        while self.stack:
            self.reversed.append(self.stack.pop())
        
        return self.reversed[-1]


    def empty(self) -> bool:
        return (not self.stack and 
                not self.reversed)


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


# O(n), O(n)
# sliding window
class Solution:
    def numOfSubarrays(self, numbers: list[int], size: int, threshold: int) -> int:
        left = 0
        window = 0
        counter = 0

        for right, number in enumerate(numbers):
            window += number

            # check threshold
            if right - left + 1 == size:  # if window is the right size
                if window / size >= threshold:  # if average is greather equal than threshold
                    counter += 1

                # remove left
                window -= numbers[left]
                left += 1

        return counter


# O(n2), O(1)
# brute force
class Solution:
    def numOfSubarrays(self, numbers: list[int], size: int, threshold: int) -> int:
            counter = 0

            for index in range(len(numbers) - size + 1):
                subarray = numbers[index: index + size]
                
                if sum(subarray) / size >= threshold:
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
e
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
                letter != stack[-1] and # and if the top of the stack differs the current letter
                    letter == stack[-1].upper()):  # and both lowercase are the same
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


# O(n2), O(n)
class Solution:
    def generate(self, rowIndex: int) -> list[int]:
        triangle = [[1]]  # first row in Pascal's Triangle

        for index1 in range(rowIndex - 1):
            new_row = [1] * (index1 + 2)  # every new row starts and ends with with `1`
            new_row[0] = 1  
            row = triangle[-1]  # previous row

            for index2 in range(index1):
                # new_row[index2 + 1] = sum(row[index2: index2 + 2])
                new_row[index2 + 1] = row[index2] + row[index2 + 1]
            
            triangle.append(new_row)  # push row to Pascal's Triangle

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


# O(nlogn), O(n)
# sliding window
class Solution:
    def maxFrequency(self, numbers: list[int], joker: int) -> int:
        numbers.sort()
        left = 0
        longest = 0

        for right, number in enumerate(numbers):
            joker -= (right - left) * (number - numbers[right - 1])  # decrease joker wirdcard by sum needed to level current window to its maximum value
            
            while joker < 0:  # while joker exceeded
                joker += (number - numbers[left])  # return joker value from left number
                left += 1  # move left pointer
                
            longest = max(longest, right - left + 1)  # update longest frequency

        return longest


# O(nlogn), O(n)
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


# O(n), O(n)
# stack
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
        max_fruits = 0

        for right, fruit in enumerate(fruits):
            basket[fruit] = basket.get(fruit, 0) + 1  # add a fruit to the basket

            while len(basket) > 2:  # while too many fruit types
                left_fruit = fruits[left]  # `left` fruit type
                basket[left_fruit] -= 1  # remove one `left` fruit from basket
                left += 1  # increase the left pointer

                if not basket[left_fruit]:  # if no `left` fruit
                    basket.pop(left_fruit)  # pop that fruit type

            if right - left + 1 > max_fruits:  # update max fruit counter
                max_fruits = right - left + 1

        return max_fruits





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
# stack
class Solution:
    def validateStackSequences(self, pushed: list[int], popped: list[int]) -> bool:
        stack = []
        index = 0

        for number in pushed:
            stack.append(number)

            while stack and stack[-1] == popped[index]:
                stack.pop()
                index += 1

        return not stack


from collections import deque

# O(n), O(n)
# stack, deque
class Solution:
    def validateStackSequences(self, pushed: list[int], popped: list[int]) -> bool:
        queue = deque(popped)
        stack = []

        for number in pushed:
            stack.append(number)

            while stack and stack[-1] == queue[0]:
                stack.pop()
                queue.popleft()

        return not stack





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


# O(n), O(1)
# sliding window
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
            
            max_vovels = max(max_vovels, vovel_count)  # update max vovels

        return max_vovels





# Asteroid Collision
# https://leetcode.com/problems/asteroid-collision/description/
u"""
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


# O(n), O(n)
# stack
class Solution:
    def asteroidCollision(self, asteroids: list[int]) -> list[int]:
        stack = []

        for asteroid in asteroids:
            if (stack and
                stack[-1] > 0 and
                    asteroid < 0):
                while (stack and
                       stack[-1] > 0 and
                       stack[-1] < -asteroid):
                    stack.pop()
                if (stack and 
                        stack[-1] == -asteroid):
                    stack.pop()
                    continue
                elif (stack and 
                    stack[-1] > 0 and 
                        stack[-1] > asteroid):
                    continue

            stack.append(asteroid)
        return stack


# O(n), O(n)
# stack
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
print(Solution().minFlips("01001001101"), 2)


# blueprint
# "111000" numbers
# "010101" target A
# "101010" target B

# "111000111000" numbers
# "010101010101" target A
# "101010101010" target B

# O(n), O(n)
# sliding window
class Solution:
    def minFlips(self, numbers: str) -> int:
        left = 0
        double_numbers = numbers * 2  # double the numbers length
        zero = "01" * len(numbers)  # target string A "0101..."
        one = "10" * len(numbers)  # target string B "1010..."
        flip_a = 0  # count numbers to flip to get current tagret A in a loop
        flip_b = 0  # count numbers to flip to get current tagret B in a loop
        min_flip_a = len(numbers)  # minimum flips to get the best matching target A
        min_flip_b = len(numbers)  # minimum flips to get the best matching target B


        for right, number in enumerate(double_numbers):
            # if no match increase the flip counter
            if number != zero[right]:
                flip_a += 1
            if number != one[right]:
                flip_b += 1

            # if the window is the right length
            if right - left + 1 == len(numbers):
                min_flip_a = min(min_flip_a, flip_a)
                min_flip_b = min(min_flip_b, flip_b)

                # early exit when any of min flips is zero
                if not min_flip_a or not min_flip_b:
                    return 0

                # if no match decrease the flip counter
                if double_numbers[left] != zero[left]:
                    flip_a -= 1
                if double_numbers[left] != one[left]:
                    flip_b -= 1

                left += 1

        return min(min_flip_a, min_flip_b)


# O(n2), O(n)
# brute force
class Solution:
    def minFlips(self, numbers: str) -> int:
        min_flip_a = len(numbers)
        min_flip_b = len(numbers)
        
        target_a = "".join(str(i % 2) 
                           for i in range(len(numbers)))
        target_b = "".join(str((i + 1) % 2) 
                           for i in range(len(numbers)))

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
# Your StockSpanner object will be instantiated and called as such:
# obj = StockSpanner()
# param_1 = obj.next(price)


stockSpanner = StockSpanner()
print(stockSpanner.next(100)) # return 1
print(stockSpanner.next(80))  # return 1
print(stockSpanner.next(60))  # return 1
print(stockSpanner.next(70))  # return 2
print(stockSpanner.next(60))  # return 1
print(stockSpanner.next(75))  # return 4, because the last 4 prices (including today's price of 75) were less than or equal to today's price.
print(stockSpanner.next(85))  # return 6


# draft
# 100
# 100, 80
# 100, 80, 60
# 100(0), 80(1), <- 70(ind = 2)
# 100, 80, 70(2), 60(4)


# O(n), O(n)
# stack, monotonic stack
# monotonically decreasing stack
class StockSpanner:
    def __init__(self):
        self.stack = []
        self.index = 0  # index of the current number

    def next(self, price: int) -> int:
        days_of_lower = 0  # consecutive days for which the stock price was less than or equal to the price of that day
        index = self.index
        
        while self.stack and self.stack[-1][1] <= price:  # check while top stack price is lower equal to curent price
            index, _ = self.stack.pop()  # get index of popped price
            days_of_lower = self.index - index  # update days

        self.stack.append((min(index, self.index), price))  # use current index or lower index for last popped price
        self.index += 1  # update index

        return days_of_lower + 1  # increase `1` to count current day


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

        return (window_size 
                if window_size != len(numbers) + 1 
                else 0)





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
print(Solution().findClosestElements([1, 2, 3, 4, 5], 4, 3), [1, 2, 3, 4])
print(Solution().findClosestElements([1, 1, 2, 3, 4, 5], 4, -1), [1, 1, 2, 3])
print(Solution().findClosestElements([0, 1, 2, 2, 2, 3, 6, 8, 8, 9], 5, 9), [3, 6, 8, 8, 9])


# draft
# [2, 1, 0, 1, 2]
# [0, 0, 1, 2, 3, 4]


from collections import deque

# O(n-k), O(n)
# deque
class Solution:
    def findClosestElements(self, numbers: list[int], k: int, x: int) -> list[int]:
        queue = deque(numbers)

        while len(queue) > k:
            if x - queue[0] > queue[-1] - x:
                queue.popleft()
            else:
                queue.pop()

        return list(queue)


# O(n-k), O(1)
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
# binary search, sliding window
class Solution:
    def findClosestElements(self, numbers: list[int], k: int, x: int) -> list[int]:
        """
        Starts with `sliding window` positioned in the middle of the `numbers`.
        Start binary search. If the first number outside of the `sliding window` 
        on the right minus `x` is less than `x` minus the first character 
        in the `sliding window` then search the right portion of the binary search.
        The solution on the right would be better than current `sliding window`.
        Else search the left portion of the binary search while preserving current 
        `sliding window` (current `sliding window` could be the solution).
        """
        left = 0
        right = len(numbers) - k

        while left < right:  # O(log(n-k))
            middle = (left + right) // 2
            if x - numbers[middle] > numbers[middle + k] - x:
                left = middle + 1
            else:
                right = middle

        return numbers[left: left + k]  # O(k))


# O(n), O(n)
# two pointers
class Solution:
    def findClosestElements(self, numbers: list[int], k: int, x: int) -> list[int]:
        disttances = [abs(number - x)
                      for number in numbers]

        min_number = min(disttances)
        min_index = disttances.index(min_number)

        left = min_index - 1
        right = min_index + 1
        if left < 0:
            left = 0
            right = k - 1
        elif right == len(disttances):
            left = len(numbers) - k
            right = len(numbers) - 1
        else:
            while k - 1:
                if (right < len(disttances) and
                        disttances[left] > disttances[right]):
                    right += 1
                else:
                    left -= 1
                k -= 1
            left += 1
            right -= 1

        return numbers[left: right + 1]


# O(n), O(1)
# two pointers
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
        window_max_length = len(numbers) + 1  # minimum operations to reduce
        target = sum(numbers) - x

        for right, number in enumerate(numbers):
            window_sum += number

            while (left <= right and  # left in bounds and
                   window_sum > target):  # window sum is too large
                window_sum -= numbers[left]
                left += 1

            if window_sum == target:
                window_max_length = min(window_max_length, 
                                        len(numbers) - (right - left + 1))

        return (window_max_length 
                if window_max_length != len(numbers) + 1
                else -1)





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


# O(n), O(n)
# stack
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

            
# O(n), O(n)
# built-in function
class Solution:
    def simplifyPath(self, path: str) -> str:
        clean_dirs = []
        
        for dir in path.split("/"):
            if not dir or dir == ".":
                continue
            elif dir == "..":
                if clean_dirs:
                    clean_dirs.pop()
            else:
                clean_dirs.append(dir)
        
        return "/" + "/".join(clean_dirs)





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





# Decode String
# https://leetcode.com/problems/decode-string/description/
"""
Given an encoded string, return its decoded string.

The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times. Note that k is guaranteed to be a positive integer.

You may assume that the input string is always valid; there are no extra white spaces, square brackets are well-formed, etc. Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, k. For example, there will not be input like 3a or 2[4].

The test cases are generated so that the length of the output will never exceed 105.

 

Example 1:

Input: s = "3[a]2[bc]"
Output: "aaabcbc"
Example 2:

Input: s = "3[a2[c]]"
Output: "accaccacc"
Example 3:

Input: s = "2[abc]3[cd]ef"
Output: "abcabccdcdcdef"
"""
print(Solution().decodeString("3[a]2[bc]"), "aaabcbc")
print(Solution().decodeString("3[a2[c]]"), "accaccacc")
print(Solution().decodeString("2[abc]3[cd]ef"), "abcabccdcdcdef")
print(Solution().decodeString("3[z]2[2[y]pq4[2[jk]e1[f]]]ef"), "zzzyypqjkjkefjkjkefjkjkefjkjkefyypqjkjkefjkjkefjkjkefjkjkefef")
print(Solution().decodeString("10[leetcode]"), "leetcodeleetcodeleetcodeleetcodeleetcodeleetcodeleetcodeleetcodeleetcodeleetcode")


# O(n), O(n)
# recursion
class Solution:
    def decodeString(self, text: str) -> str:
        self.index = 0
        
        def recursion():
            number = 0
            word = ""
            
            while self.index < len(text):
                char = text[self.index]

                if char.isdigit():
                    number += 10 * number + int(char)
                    self.index += 1
                elif char == "[":
                    self.index += 1
                    word += number * recursion()
                    number = 0
                elif char == "]":
                    self.index += 1
                    return word
                else:
                    word += char
                    self.index += 1

            return word
            
        return recursion()


# O(n), O(n)
# stack
class Solution:
    def decodeString(self, text: str) -> str:
        stack = []

        for char in text:
            if char != "]":
                stack.append(char)
            else:
                word = ""
                while stack[-1] != "[":
                    word = stack.pop() + word 
                stack.pop()
                
                number = ""
                while stack and stack[-1].isdigit():
                    number = stack.pop() + number
                
                stack.append(int(number) * word)
        
        return "".join(stack)


# O(n), O(n)
# stack
class Solution:
    def decodeString(self, text: str) -> str:
        stack = []  # (prefix, multiplier)
        word = ""
        number = ""
        decoded = ""
        opened = 0

        for char in text:
            if char not in "[]":
                if char.isdigit():
                    number += char
                else:
                    word += char  # c
            else:
                if char == "[":
                    opened += 1
                    if not number:
                        number = 1
                    stack.append((word, int(number)))  # ("", 3), (a, 2)
                    word, number = "", ""
                else:
                    opened -= 1
                    prefix, multi = stack.pop()  # (a, 2), (, 3)
                    word = prefix + multi * word  # (a+2*c)
                    if opened == 0:
                        decoded += word
                        word = ""

        return decoded + word





# Single Element in a Sorted Array
# https://leetcode.com/problems/single-element-in-a-sorted-array/description/
"""
You are given a sorted array consisting of only integers where every element appears exactly twice, except for one element which appears exactly once.

Return the single element that appears only once.

Your solution must run in O(log n) time and O(1) space.

 

Example 1:

Input: nums = [1,1,2,3,3,4,4,8,8]
Output: 2
Example 2:

Input: nums = [3,3,7,7,10,11,11]
Output: 10
"""
print(Solution().singleNonDuplicate([1, 1, 2, 3, 3, 4, 4, 8, 8]), 2)
print(Solution().singleNonDuplicate([3, 3, 7, 7, 10, 11, 11]), 10)
print(Solution().singleNonDuplicate([1, 2, 2]), 1)
print(Solution().singleNonDuplicate([1]), 1)

# draft
# [1, 1, 2]
# [1, 2, 2]
# [1, 1, 2, 2, 3]
# [1, 2, 2, 3, 3]


# O(logn), O(1)
# binary search
class Solution:
    def singleNonDuplicate(self, numbers: list[int]) -> int:
        left = 0
        right = len(numbers) - 1

        while left <= right:
            middle = left + (right - left) // 2
            middle_number = numbers[middle]

            if (middle - 1 >= 0 and  # if index in bounds
                    numbers[middle - 1] == middle_number):  # middle number same as previous one
                if (middle - 1) % 2:  # if the previous index is odd
                    right = middle - 1  # check the left part
                else:
                    left = middle + 1  # check the right part                
            elif (middle + 1 < len(numbers) and  # if index in bounds
                    numbers[middle + 1] == middle_number):  # middle number same as next one
                if (middle + 1) % 2:  # if the next index is odd
                    left = middle + 1  # check the right part
                else:
                    right = middle - 1  # check the left part
            else:
                return middle_number





# Remove Nodes From Linked List
# https://leetcode.com/problems/remove-nodes-from-linked-list/description/
"""
You are given the head of a linked list.

Remove every node which has a node with a greater value anywhere to the right side of it.

Return the head of the modified linked list.

Example 1:

Input: head = [5,2,13,3,8]
Output: [13,8]
Explanation: The nodes that should be removed are 5, 2 and 3.
- Node 13 is to the right of node 5.
- Node 13 is to the right of node 2.
- Node 8 is to the right of node 3.
Example 2:

Input: head = [1,1,1,1]
Output: [1,1,1,1]
Explanation: Every node has value 1, so no nodes are removed.
"""


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# O(n), O(1)
# linked list
class Solution:
    def removeNodes(self, head: ListNode | None) -> ListNode | None:
        """
        Reverse Linked List and return it's new head.
        """
        def reverse_linkedlist(node: ListNode | None) -> ListNode | None:
            previous = None

            while node:
                next_node = node.next
                node.next = previous
                previous = node
                node = next_node
            
            return previous

        anchor = node = reverse_linkedlist(head)  # reversed link list head
        max_value = node.val  # init maximum value with the last node value

        while node.next:  # while in bounds
            if node.next.val < max_value:  # it's value is less than current max value
                node.next = node.next.next  # pop the next node
            else:
                max_value = max(max_value, node.val)  # update max value
                node = node.next  # point to the next node

        return reverse_linkedlist(anchor)


list_nodes = [5, 2, 13, 3, 8]

def list_to_linkedlist(list_nodes):
    anchor = node = ListNode()

    for static_node in list_nodes:
        node.next = ListNode(static_node)
        node = node.next
    
    node.next = None
    
    return anchor.next

linkedlist_1 = list_to_linkedlist(list_nodes)
print(linkedlist_1.val)


# define LinkedList class to iterate throungh linked list
class LinkedList:
    def __init__(self, head):
        self.head = head
        self.node = None

    # define iterator
    def __iter__(self):
        self.node = self.head
        return self

    # iterate
    def __next__(self):
        if self.node:
            val = self.node.val
            self.node = self.node.next
            return val
        else:
            raise StopIteration

linkedList1 = LinkedList(linkedlist_1)

# iterate through LinkedList
for linkedlist_node in linkedList1:
    print(linkedlist_node)



def linkedlist_to_list(node):
    list_nodes = []

    while node:
        list_nodes.append(node.val)
        node = node.next

    return list_nodes

print(linkedlist_to_list(linkedlist_1))

print(linkedlist_to_list(Solution().removeNodes((list_to_linkedlist([5, 2, 13, 3, 8])))), [13, 8])
print(linkedlist_to_list(Solution().removeNodes((list_to_linkedlist([1, 1, 1, 1])))), [1,1,1,1])

print(
    linkedlist_to_list(
        Solution().removeNodes((
            list_to_linkedlist([5, 2, 13, 3, 8])))), [13, 8])





# Convert Sorted Array to Binary Search Tree
# https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/description/
"""
Given an integer array nums where the elements are sorted in ascending order, convert it to a 
height-balanced
 binary search tree.

Example 1:
____0
  /     \
-10      5
   \      \
    -3     9

Input: nums = [-10,-3,0,5,9]

Output: [0,-3,9,-10,null,5]
Explanation: [0,-10,5,null,-3,null,9] is also accepted:


Example 2:


Input: nums = [1,3]
Output: [3,1]
Explanation: [1,null,3] and [3,1] are both height-balanced BSTs.
"""
print(Solution().sortedArrayToBST([-10, -3, 0, 5, 9]))


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from binarytree import Node

# O(n), O(1)
# binary treee
class Solution:
    def sortedArrayToBST(self, numbers: list[int]) -> Node | None:
        if not numbers:  # if the list of child nodes is empty
            return None

        left = 0  # left pointer
        right = len(numbers) - 1  # right pointer 
        middle = left + (right - left) // 2  # middle pointer
        middle_value = numbers[middle]  # middle value
        left_node = self.sortedArrayToBST(numbers[:middle])
        right_node = self.sortedArrayToBST(numbers[middle + 1:])
        node = Node(middle_value, left_node, right_node)  # create a node

        return node





# Merge Two Binary Trees
# https://leetcode.com/problems/merge-two-binary-trees/description/
"""
You are given two binary trees root1 and root2.

Imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped while the others are not. You need to merge the two trees into a new binary tree. The merge rule is that if two nodes overlap, then sum node values up as the new value of the merged node. Otherwise, the NOT null node will be used as the node of the new tree.

Return the merged tree.

Note: The merging process must start from the root nodes of both trees.

 

Example 1:
    1         __2           __3
   / \       /   \         /   \
  3   2  +  1     3   =   4     5
 /           \     \     / \     \
5             4     7   5   4     7



Input: root1 = [1,3,2,5], root2 = [2,1,3,null,4,null,7]
Output: [3,4,5,5,4,null,7]
Example 2:

Input: root1 = [1], root2 = [1,2]
Output: [2,2]
"""
print(level_order_traversal(Solution().mergeTrees(build_tree_from_list([1, 3, 2, 5]), build_tree_from_list([2, 1, 3, None, 4, None, 7]))))


# O(n), O(n)
# binary search
class Solution:
    def mergeTrees(self, root1: TreeNode | None, root2: TreeNode | None) -> TreeNode | None:
        if not root1 and not root2:
            return None
        
        if not root1:
            root1 = TreeNode()
        if not root2:
            root2 = TreeNode()
        
        root = TreeNode(root1.val + root2.val)
        root.left = self.mergeTrees(root1.left, root2.left)
        root.right = self.mergeTrees(root1.right, root2.right)
        
        return root


# O(n), O(n)
# binary search
class Solution:
    def mergeTrees(self, root1: TreeNode | None, root2: TreeNode | None) -> TreeNode | None:
        if not root1:
            return root2
        if not root2:
            return root1
        
        root = TreeNode(root1.val + root2.val)
        root.left = self.mergeTrees(root1.left, root2.left)
        root.right = self.mergeTrees(root1.right, root2.right)
        
        return root

print(level_order_traversal(
    Solution().mergeTrees(
        build_tree_from_list([1, 3, 2, 5]), 
        build_tree_from_list([2, 1, 3, None, 4, None, 7]))))





# Design HashSet
# https://leetcode.com/problems/design-hashset/description/
"""
Design a HashSet without using any built-in hash table libraries.

Implement MyHashSet class:

void add(key) Inserts the value key into the HashSet.
bool contains(key) Returns whether the value key exists in the HashSet or not.
void remove(key) Removes the value key in the HashSet. If key does not exist in the HashSet, do nothing.
 

Example 1:

Input
["MyHashSet", "add", "add", "contains", "contains", "add", "contains", "remove", "contains"]
[[], [1], [2], [1], [3], [2], [2], [2], [2]]
Output
[null, null, null, true, false, null, true, null, false]

Explanation
MyHashSet myHashSet = new MyHashSet();
myHashSet.add(1);      // set = [1]
myHashSet.add(2);      // set = [1, 2]
myHashSet.contains(1); // return True
myHashSet.contains(3); // return False, (not found)
myHashSet.add(2);      // set = [1, 2]
myHashSet.contains(2); // return True
myHashSet.remove(2);   // set = [1]
myHashSet.contains(2); // return False, (already removed)
"""


class ListNode:
    def __init__(self, val: int = 0):
        self.val = val
        self.next = None


# O(1), O(n)
# linked list
class MyHashSet:
    def __init__(self):
        self.set = [ListNode() for _ in range(10**4)]


    def add(self, key: int) -> None:
        node = self.set[key % len(self.set)]

        while node.next:
            if node.next.val == key:
                return
            node = node.next
        
        node.next = ListNode(key)


    def remove(self, key: int) -> None:
        node = self.set[key % len(self.set)]

        while node.next:
            if node.next.val == key:
                node.next = node.next.next
                return
            node = node.next


    def contains(self, key: int) -> bool:
        node = self.set[key % len(self.set)]

        while node.next:
            if node.next.val == key:
                return True
            node = node.next
        
        return False


# Your MyHashSet object will be instantiated and called as such:
# obj = MyHashSet()
# obj.add(key)
# obj.remove(key)
# param_3 = obj.contains(key)


# O(n), O(n)
# list
class MyHashSet:
    def __init__(self):
        self.hash_set = []

    def add(self, key: int) -> None:
        if key not in self.hash_set:
            self.hash_set.append(key)

    def remove(self, key: int) -> None:
        if key in self.hash_set:
            self.hash_set.remove(key)

    def contains(self, key: int) -> bool:
        return key in self.hash_set


# O(1), O(10**6)
# Memory is O(10**6) since the table size is constant
# boolean list
class MyHashSet:
    def __init__(self):
        self.hash_set = [False] * (10 ** 6 + 1)  # first: 0, last: 10 **6

    def add(self, key: int) -> None:
        self.hash_set[key] = True

    def remove(self, key: int) -> None:
        self.hash_set[key] = False

    def contains(self, key: int) -> bool:
        return self.hash_set[key]





# 4Sum
# https://leetcode.com/problems/4sum/description/
"""
Given an array nums of n integers, return an array of all the unique quadruplets [nums[a], nums[b], nums[c], nums[d]] such that:

0 <= a, b, c, d < n
a, b, c, and d are distinct.
nums[a] + nums[b] + nums[c] + nums[d] == target
You may return the answer in any order.

 

Example 1:

Input: nums = [1,0,-1,0,-2,2], target = 0
Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
Example 2:

Input: nums = [2,2,2,2,2], target = 8
Output: [[2,2,2,2]]
"""
print(Solution().fourSum([1, 0, -1, 0, -2, 2], 0), [[-2, -1, 1, 2], [-2, 0, 0, 2], [-1, 0, 0, 1]])
print(Solution().fourSum([2, 2, 2, 2, 2], 8), [[2, 2, 2, 2]])
print(Solution().fourSum([0, 0, 0, 0], 0), [[0, 0, 0, 0]])


# [-2, -1, 0, 0, 1, 2]

# O(n3), O(1)
# two pointers
class Solution:
    def fourSum(self, numbers: list[int], target: int) -> list[list[int]]:
        numbers.sort()  # sort list to be able to ignore duplicate numbers
        quarduplets = []

        for index1, number1 in enumerate(numbers[:-3]):
            # Skip same number values
            if (index1 and number1 == numbers[index1 - 1]):
                continue

            # for index2, number2 in enumerate(numbers[index1 + 1 :-2], index1 + 1):
            for index2 in range(index1 + 1, len(numbers) - 2):
                number2 = numbers[index2]
                
                # Skip same number values
                if (index2 > index1 + 1 and number2 == numbers[index2 - 1]):
                    continue

                left = index2 + 1  # left pointer
                right = len(numbers) - 1  # right pointer

                while left < right:  # two pointers
                    quarduplet = (number1 +
                                  number2 +
                                  numbers[left] +
                                  numbers[right])

                    if quarduplet == target:  # if sum is equal to the target
                        quarduplets.append([number1,
                                            number2,
                                            numbers[left],
                                            numbers[right]])
                        left += 1
                        right -= 1
                        # skip same left pointer values
                        while (left < right and
                               numbers[left] == numbers[left - 1]):
                            left += 1

                    elif quarduplet < target:  # if sum is less than the target
                        left += 1
                    else:  # if sum is greater than the target
                        right -= 1

        return quarduplets





# Implement Trie (Prefix Tree)
# https://leetcode.com/problems/implement-trie-prefix-tree/description/
"""
A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the Trie class:

Trie() Initializes the trie object.
void insert(String word) Inserts the string word into the trie.
boolean search(String word) Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.
 

Example 1:

Input
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
Output
[null, null, true, false, true, null, true]

Explanation
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // return True
trie.search("app");     // return False
trie.startsWith("app"); // return True
trie.insert("app");
trie.search("app");     // return True
"""


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False


# O(n), O(n)
# dict
class Trie:
    def __init__(self):
        self.root = TrieNode()


    def insert(self, word: str) -> None:
        node = self.root

        for letter in word:
            if letter not in node.children:
                node.children[letter] = TrieNode()
            
            node = node.children[letter]
        
        node.is_word = True


    def search(self, word: str) -> bool:
        node = self.root

        for letter in word:
            if letter not in node.children:
                return False
            
            node = node.children[letter]

        return node.is_word
        

    def startsWith(self, prefix: str) -> bool:
        node = self.root

        for letter in prefix:
            if letter not in node.children:
                return False
            
            node = node.children[letter]

        return True


class TrieNode:
    def __init__(self):
        self.children = [False] * 26
        self.is_word = False


# O(n), O(n)
# list
class Trie:
    def __init__(self):
        self.root = TrieNode()


    def insert(self, word: str) -> None:
        node = self.root

        for letter in word:
            index = ord(letter) - ord("a")
            
            if not node.children[index]:
                node.children[index] = TrieNode()
            
            node = node.children[index]

        node.is_word = True


    def search(self, word: str) -> bool:
        node = self.root

        for letter in word:
            index = ord(letter) - ord("a")
            
            if not node.children[index]:
                return False
            
            node = node.children[index]

        return node.is_word
        

    def startsWith(self, prefix: str) -> bool:
        node = self.root

        for letter in prefix:
            index = ord(letter) - ord("a")
            
            if not node.children[index]:
                return False
            
            node = node.children[index]

        return True

# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)


trie = Trie()
trie.insert("apple")
trie.search("apple")   # return True
trie.search("app")     # return False
trie.startsWith("app") # return True
trie.insert("app")
trie.search("app")     # return True



    
# Design HashMap
# https://leetcode.com/problems/design-hashmap/description/
"""
Design a HashMap without using any built-in hash table libraries.

Implement the MyHashMap class:

MyHashMap() initializes the object with an empty map.
void put(int key, int value) inserts a (key, value) pair into the HashMap. If the key already exists in the map, update the corresponding value.
int get(int key) returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key.
void remove(key) removes the key and its corresponding value if the map contains the mapping for the key.
 

Example 1:

Input
["MyHashMap", "put", "put", "get", "get", "put", "get", "remove", "get"]
[[], [1, 1], [2, 2], [1], [3], [2, 1], [2], [2], [2]]
Output
[null, null, null, 1, -1, null, 1, null, -1]

Explanation
MyHashMap myHashMap = new MyHashMap();
myHashMap.put(1, 1); // The map is now [[1,1]]
myHashMap.put(2, 2); // The map is now [[1,1], [2,2]]
myHashMap.get(1);    // return 1, The map is now [[1,1], [2,2]]
myHashMap.get(3);    // return -1 (i.e., not found), The map is now [[1,1], [2,2]]
myHashMap.put(2, 1); // The map is now [[1,1], [2,1]] (i.e., update the existing value)
myHashMap.get(2);    // return 1, The map is now [[1,1], [2,1]]
myHashMap.remove(2); // remove the mapping for 2, The map is now [[1,1]]
myHashMap.get(2);    // return -1 (i.e., not found), The map is now [[1,1]]
"""


class ListNode:
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.next = None


# O(1), O(n)
# linked list
class MyHashMap:
    def __init__(self):
        self.map = [0] * 10**4

    def put(self, key: int, val: int) -> None:
        index = key % len(self.map)
        
        if not self.map[index]:
            self.map[index] = ListNode()

        node = self.map[index]

        while node.next:
            if node.next.key == key:
                node.next.val = val
                return
            node = node.next

        node.next = ListNode(key, val)
        

    def get(self, key: int) -> int:
        index = key % len(self.map)

        if not self.map[index]:
            return -1
        
        node = self.map[index]

        while node.next:
            if node.next.key == key:
                return node.next.val
            node = node.next
        
        return -1
        

    def remove(self, key: int) -> None:
        index = key % len(self.map)

        if not self.map[index]:
            return
        
        node = self.map[index]

        while node.next:
            if node.next.key == key:
                node.next = node.next.next
                return
            node = node.next


# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)

myHashMap = MyHashMap()
myHashMap.put(1, 1) #  The map is now [[1,1]]
myHashMap.put(2, 2) #  The map is now [[1,1], [2,2]]
myHashMap.get(1)    #  return 1, The map is now [[1,1], [2,2]]
myHashMap.get(3)    #  return -1 (i.e., not found), The map is now [[1,1], [2,2]]
myHashMap.put(2, 1) #  The map is now [[1,1], [2,1]] (i.e., update the existing value)
myHashMap.get(2)    #  return 1, The map is now [[1,1], [2,1]]
myHashMap.remove(2) #  remove the mapping for 2, The map is now [[1,1]]
myHashMap.get(2)    #  return -1 (i.e., not found), The map is now [[1,1]]





# Rotate Array
# https://leetcode.com/problems/rotate-array/description/
"""
Given an integer array nums, rotate the array to the right by k steps, where k is non-negative.

 

Example 1:

Input: nums = [1,2,3,4,5,6,7], k = 3
Output: [5,6,7,1,2,3,4]
Explanation:
rotate 1 steps to the right: [7,1,2,3,4,5,6]
rotate 2 steps to the right: [6,7,1,2,3,4,5]
rotate 3 steps to the right: [5,6,7,1,2,3,4]
Example 2:

Input: nums = [-1,-100,3,99], k = 2
Output: [3,99,-1,-100]
Explanation: 
rotate 1 steps to the right: [99,-1,-100,3]
rotate 2 steps to the right: [3,99,-1,-100]
"""
print(Solution().rotate([1, 2, 3, 4, 5, 6, 7], 3), [5, 6, 7, 1, 2, 3, 4])
print(Solution().rotate([-1, -100, 3, 99], 2), [3, 99, -1, -100])
print(Solution().rotate([-1], 2), [-1])
print(Solution().rotate([1, 2], 3), [2, 1])


# O(n), O(1)
# reverse
class Solution:
    def reverse_str(self, left, right):
        while left < right:
            self.nums[left], self.nums[right] = self.nums[right], self.nums[left]
            left += 1
            right -= 1


    def rotate(self, nums: list[int], k: int) -> None:
        k = k % len(nums)
        self.nums = nums
        self.reverse_str(0, len(nums) - 1)
        self.reverse_str(0, k - 1)
        self.reverse_str(k, len(nums) - 1)
        return self.nums


# O(n), O(n)
# slice
class Solution:
    def rotate(self, nums: list[int], k: int) -> None:
        right = len(nums) - k % len(nums)
        return nums[right:] + nums[:right]


# O(n), O(n)
# iteraton
class Solution:
    def rotate(self, nums: list[int], k: int) -> None:
        nums_copy = [0] * len(nums)
        
        for index in range(len(nums)):
            nums_copy[(index + k) % len(nums)] = nums[index]

        nums[:] = nums_copy
        
        return nums





# Binary Subarrays With Sum
# https://leetcode.com/problems/binary-subarrays-with-sum/description/
"""
Given a binary array nums and an integer goal, return the number of non-empty subarrays with a sum goal.

A subarray is a contiguous part of the array.

 

Example 1:

Input: nums = [1,0,1,0,1], goal = 2
Output: 4
Explanation: The 4 subarrays are bolded and underlined below:
[(1,0,1),0,1]
[(1,0,1,0),1]
[1,(0,1,0,1)]
[1,0,(1,0,1)]
Example 2:

Input: nums = [0,0,0,0,0], goal = 0
Output: 15
"""
print(Solution().numSubarraysWithSum([1, 0, 1, 0, 1], 2), 4)
print(Solution().numSubarraysWithSum([0, 0, 0, 0, 0], 0), 15)


# O(n3), O(n)
# brute force
class Solution:
    def numSubarraysWithSum(self, numbers: list[int], goal: int) -> int:
        counter = 0

        for i in range(len(numbers)):
            for j in range(i, len(numbers)):
                if sum(numbers[i: j + 1]) == goal:
                    counter += 1
                elif sum(numbers[i: j + 1]) > goal:
                    break
            
        return counter


# O(n2), O(1)
# brute force
class Solution:
    def numSubarraysWithSum(self, numbers: list[int], goal: int) -> int:
        counter = 0

        for i in range(len(numbers)):
            sub_goal = goal
            
            for j in range(i, len(numbers)):
                sub_goal -= numbers[j]
                if sub_goal == 0:
                    counter += 1
                elif sub_goal < 0:
                    break
            
        return counter





# Remove K Digits
# https://leetcode.com/problems/remove-k-digits/description/
"""
Given string num representing a non-negative integer num, and an integer k, return the smallest possible integer after removing k digits from num.

 

Example 1:

Input: num = "1432219", k = 3
Output: "1219"
Explanation: Remove the three digits 4, 3, and 2 to form the new number 1219 which is the smallest.
Example 2:

Input: num = "10200", k = 1
Output: "200"
Explanation: Remove the leading 1 and the number is 200. Note that the output must not contain leading zeroes.
Example 3:

Input: num = "10", k = 2
Output: "0"
Explanation: Remove all the digits from the number and it is left with nothing which is 0.
"""
print(Solution().removeKdigits("1432219", 3), "1219")
print(Solution().removeKdigits("10200", 1), "200")
print(Solution().removeKdigits("10", 2), "0")
print(Solution().removeKdigits("9", 1), "0")
print(Solution().removeKdigits("112", 1), "11")
print(Solution().removeKdigits("1173", 2), "11")
print(Solution().removeKdigits("10", 1), "0")


# O(n), O(n)
# stack, monotonic stack
# monotonically increasing stack
class Solution:
    def removeKdigits(self, numbers: str, k: int) -> str:
        stack = []
        
        for number in numbers:
            while (k and stack and
                   number < stack[-1]):
                stack.pop()
                k -= 1  # the number of digits to remove
                
            stack.append(number)
        
        # calculate `start` where leading zeros (if present) stops
        start = 0
        while (start < len(stack) and 
               stack[start] == "0"):
            start += 1
        
        # `start` to remove leading zeros and 
        # `len(stack) - k` to remove exceeded characters if `k` is not zero
        clean_stack = stack[start: len(stack) - k]

        return ("".join(clean_stack) 
                if clean_stack
                else "0")





# Find Peak Element
# https://leetcode.com/problems/find-peak-element/description/
"""
A peak element is an element that is strictly greater than its neighbors.

Given a 0-indexed integer array nums, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks.

You may imagine that nums[-1] = nums[n] = -∞. In other words, an element is always considered to be strictly greater than a neighbor that is outside the array.

You must write an algorithm that runs in O(log n) time.

 

Example 1:

Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.
Example 2:

Input: nums = [1,2,1,3,5,6,4]
Output: 5
Explanation: Your function can return either index number 1 where the peak element is 2, or index number 5 where the peak element is 6.
"""
print(Solution().findPeakElement([1, 2, 3, 1]), 2)
print(Solution().findPeakElement([1, 2, 1, 3, 5, 6, 4]), 5)
print(Solution().findPeakElement([1]), 0)
print(Solution().findPeakElement([1, 2]), 1)


# O(logn), O(1)
# binary search
class Solution:
    def findPeakElement(self, numbers: list[int]) -> int:
        left = 0
        right = len(numbers) - 1

        while left <= right:
            if right - left + 1 == 1:  # if word length is 1
                return left

            middle = left + (right - left) // 2
            middle_number = numbers[middle]
            previous = middle_number - 1 if middle == 0 else numbers[middle - 1]  # if left index out of bounds
            next = middle_number - 1 if middle == len(numbers) - 1 else numbers[middle + 1]  # if right indext out of bounds

            if (previous < middle_number and
                    middle_number > next):
                return middle
            elif next > previous:
                left = middle + 1
            else:
                right = middle - 1





# Maximum Twin Sum of a Linked List
# https://leetcode.com/problems/maximum-twin-sum-of-a-linked-list/description/
"""
In a linked list of size n, where n is even, the ith node (0-indexed) of the linked list is known as the twin of the (n-1-i)th node, if 0 <= i <= (n / 2) - 1.

For example, if n = 4, then node 0 is the twin of node 3, and node 1 is the twin of node 2. These are the only nodes with twins for n = 4.
The twin sum is defined as the sum of a node and its twin.

Given the head of a linked list with even length, return the maximum twin sum of the linked list.

 

Example 1:


Input: head = [5,4,2,1]
Output: 6
Explanation:
Nodes 0 and 1 are the twins of nodes 3 and 2, respectively. All have twin sum = 6.
There are no other nodes with twins in the linked list.
Thus, the maximum twin sum of the linked list is 6. 
Example 2:


Input: head = [4,2,2,3]
Output: 7
Explanation:
The nodes with twins present in this linked list are:
- Node 0 is the twin of node 3 having a twin sum of 4 + 3 = 7.
- Node 1 is the twin of node 2 having a twin sum of 2 + 2 = 4.
Thus, the maximum twin sum of the linked list is max(7, 4) = 7. 
Example 3:


Input: head = [1,100000]
Output: 100001
Explanation:
There is only one node with a twin in the linked list having twin sum of 1 + 100000 = 100001.
"""


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# O(n), O(1)
# linked list, two pass
class Solution:
    def pairSum(self, head: ListNode | None) -> int:
        twin_sum = 0

        # find the last node of the left portion
        slow, fast = head, head
        previous = None
        while fast and fast.next:
            fast = fast.next.next
            nextSlow = slow.next
            # reverse the left portion
            slow.next = previous
            previous = slow
            slow = nextSlow

        # traverse through the list to find the twin sum
        while previous:
            twin_sum = max(twin_sum, previous.val + slow.val)
            previous = previous.next
            slow = slow.next

        return twin_sum


# O(n), O(1)
# linked list, three pass
class Solution:
    def pairSum(self, head: ListNode | None) -> int:
        node = head
        index = 1
        twin_sum = 0

        # find the length of the linked list
        while node and node.next:
            node = node.next
            index += 1
        
        node = head
        middle = index // 2
        # fast forward to the right portion
        for _ in range(middle):
            node = node.next

        # reverse the right portion
        previous = None
        while node:
            next_node = node.next
            node.next = previous
            previous = node
            node = next_node
        
        # traverse through the list to find the twin sum
        right_head = previous
        while right_head:
            twin_sum = max(twin_sum, head.val + right_head.val)
            head = head.next
            right_head = right_head.next

        return twin_sum





# Path Sum
# https://leetcode.com/problems/path-sum/description/
"""
Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum.

A leaf is a node with no children.

 

Example 1:
         5___
        /    \
    ___4     _8
   /        /  \
  11       13   4
 /  \            \
7    2            1

Input: root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
Output: true
Explanation: The root-to-leaf path with the target sum is shown.
Example 2:
  1
 / \
2   3

Input: root = [1,2,3], targetSum = 5
Output: false
Explanation: There are two root-to-leaf paths in the tree:
(1 --> 2): The sum is 3.
(1 --> 3): The sum is 4.
There is no root-to-leaf path with sum = 5.
Example 3:

Input: root = [], targetSum = 0
Output: false
Explanation: Since the tree is empty, there are no root-to-leaf paths.
"""
print(Solution().hasPathSum(build_tree_from_list([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1]), 22), True)
print(Solution().hasPathSum(build_tree_from_list([1, 2, 3]), 5), False)
print(Solution().hasPathSum(build_tree_from_list([]), 0), False)


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
        def dfs(node, targetSum):
            if not node:
                return False
            elif not node.left and not node.right:
                return targetSum - node.val == 0

            return (dfs(node.left, targetSum - node.val) or
                    dfs(node.right, targetSum - node.val))

        return dfs(root, targetSum)





# Sum of All Subset XOR Totals
# https://leetcode.com/problems/sum-of-all-subset-xor-totals/description/
"""
The XOR total of an array is defined as the bitwise XOR of all its elements, or 0 if the array is empty.

For example, the XOR total of the array [2,5,6] is 2 XOR 5 XOR 6 = 1.
Given an array nums, return the sum of all XOR totals for every subset of nums. 

Note: Subsets with the same elements should be counted multiple times.

An array a is a subset of an array b if a can be obtained from b by deleting some (possibly zero) elements of b.

 

Example 1:

Input: nums = [1,3]
Output: 6
Explanation: The 4 subsets of [1,3] are:
- The empty subset has an XOR total of 0.
- [1] has an XOR total of 1.
- [3] has an XOR total of 3.
- [1,3] has an XOR total of 1 XOR 3 = 2.
0 + 1 + 3 + 2 = 6
Example 2:

Input: nums = [5,1,6]
Output: 28
Explanation: The 8 subsets of [5,1,6] are:
- The empty subset has an XOR total of 0.
- [5] has an XOR total of 5.
- [1] has an XOR total of 1.
- [6] has an XOR total of 6.
- [5,1] has an XOR total of 5 XOR 1 = 4.
- [5,6] has an XOR total of 5 XOR 6 = 3.
- [1,6] has an XOR total of 1 XOR 6 = 7.
- [5,1,6] has an XOR total of 5 XOR 1 XOR 6 = 2.
0 + 5 + 1 + 6 + 4 + 3 + 7 + 2 = 28
Example 3:

Input: nums = [3,4,5,6,7,8]
Output: 480
Explanation: The sum of all XOR totals for every subset is 480.
"""
print(Solution().subsetXORSum([1, 3]), 6)
print(Solution().subsetXORSum([5, 1, 6]), 28)
print(Solution().subsetXORSum([3, 4, 5, 6, 7, 8]), 480)


# O(n2^n), O(n)
# backtracking
# subset sum as a list
class Solution:
    def subsetXORSum(self, numbers: list[int]) -> int:
        subset = []
        subset_list = []

        def dfs(index):
            if index == len(numbers):
                current = 0

                for digit in subset:
                    current ^= digit

                subset_list.append(current)
                return

            subset.append(numbers[index])
            dfs(index + 1)
            subset.pop()
            dfs(index + 1)

        dfs(0)
        return sum(subset_list)


# O(n2^n), O(n)
# backtracking
# subset sum as class (immutable) value
class Solution:
    def __init__(self):
        self.subset_sum = 0
        
    def subsetXORSum(self, numbers: list[int]) -> int:
        subset = []

        def dfs(index):
            if index == len(numbers):
                current = 0

                for digit in subset:
                    current ^= digit

                self.subset_sum += current
                return

            subset.append(numbers[index])
            dfs(index + 1)
            subset.pop()
            dfs(index + 1)

        dfs(0)

        return self.subset_sum


# O(n2^n), O(2^n)
# backtracking
# subset sum passed as a funcion parameter
class Solution:
    def subsetXORSum(self, numbers: list[int]) -> int:
        def dfs(index, subset_sum):
            if index == len(numbers):
                return subset_sum

            return (dfs(index + 1, subset_sum ^ numbers[index]) + 
                    dfs(index + 1, subset_sum))

        return dfs(0, 0)


# O(n2^n), O(2^n)
# backtracking
# subset sum as an (immutable) integer
class Solution:
    def subsetXORSum(self, numbers: list[int]) -> int:
        subset = []

        def dfs(index, subset_sum):
            if index == len(numbers):
                current = 0

                for digit in subset:
                    current ^= digit

                subset_sum += current
                return subset_sum

            subset.append(numbers[index])
            sum_with = dfs(index + 1, subset_sum)
            subset.pop()
            sum_without = dfs(index + 1, subset_sum)

            return sum_with + sum_without

        return dfs(0, 0)





# Combinations
# https://leetcode.com/problems/combinations/description/
"""
Given two integers n and k, return all possible combinations of k numbers chosen from the range [1, n].

You may return the answer in any order.

 

Example 1:

Input: n = 4, k = 2
Output: [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
Explanation: There are 4 choose 2 = 6 total combinations.
Note that combinations are unordered, i.e., [1,2] and [2,1] are considered to be the same combination.
Example 2:

Input: n = 1, k = 1
Output: [[1]]
Explanation: There is 1 choose 1 = 1 total combination.
"""
print(Solution().combine(4, 2), [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
print(Solution().combine(1, 1), [[1]])


# O(n^k), O(n)
# backtracking
class Solution:
    def combine(self, n: int, k: int) -> list[list[int]]:
        combination = []
        combination_list = []

        def dfs(index):
            if index == n:
                if len(combination) == k:
                    combination_list.append(combination.copy())
                return

            combination.append(index + 1)
            dfs(index + 1)
            combination.pop()
            dfs(index + 1)

        dfs(0)
        return combination_list


# O(n^k), O(n)
# backtracking
class Solution:
    def combine(self, n: int, k: int) -> list[list[int]]:
        combination_list = []

        def dfs(index, combination):
            if index == n:
                if len(combination) == k:
                    combination_list.append(combination.copy())
                return

            combination.append(index + 1)
            dfs(index + 1, combination)
            combination.pop()
            dfs(index + 1, combination)

        dfs(0, [])
        return combination_list





# Monotonic Array
# https://leetcode.com/problems/monotonic-array/description/
"""
An array is monotonic if it is either monotone increasing or monotone decreasing.

An array nums is monotone increasing if for all i <= j, nums[i] <= nums[j]. An array nums is monotone decreasing if for all i <= j, nums[i] >= nums[j].

Given an integer array nums, return true if the given array is monotonic, or false otherwise.

 

Example 1:

Input: nums = [1,2,2,3]
Output: true
Example 2:

Input: nums = [6,5,4,4]
Output: true
Example 3:

Input: nums = [1,3,2]
Output: false
"""
print(Solution().isMonotonic([1, 2, 2, 3]), True)
print(Solution().isMonotonic([6, 5, 4, 4]), True)
print(Solution().isMonotonic([1, 3, 2]), False)


# O(n), O(1)
class Solution:
    def isMonotonic(self, numbers: list[int]) -> bool:
        increasing = True
        decreasing = True

        for index in range(1, len(numbers)):
            # chieck if the stack is monoconicly increasing
            if numbers[index - 1] > numbers[index]:
                increasing = False
            # chieck if the stack is monoconicly decreasing
            elif numbers[index - 1] < numbers[index]:
                decreasing = False
            # early exit
            if (not increasing and
                not decreasing):
                return False
            
        return (increasing or 
                decreasing)





# Number of Good Pairs
# https://leetcode.com/problems/number-of-good-pairs/description/
"""
Given an array of integers nums, return the number of good pairs.

A pair (i, j) is called good if nums[i] == nums[j] and i < j.

 

Example 1:

Input: nums = [1,2,3,1,1,3]
Output: 4
Explanation: There are 4 good pairs (0,3), (0,4), (3,4), (2,5) 0-indexed.
Example 2:

Input: nums = [1,1,1,1]
Output: 6
Explanation: Each pair in the array are good.
Example 3:

Input: nums = [1,2,3]
Output: 0
"""
print(Solution().numIdenticalPairs([1, 2, 3, 1, 1, 3]), 4)
print(Solution().numIdenticalPairs([1, 1, 1, 1]), 6)
print(Solution().numIdenticalPairs([1, 2, 3]), 0)


# O(n), O(n)
class Solution:
    def numIdenticalPairs(self, numbers: list[int]) -> int:
        counter = {}  # {number: frequency}
        pair_counter = 0

        # count numbers
        for number in numbers:
            counter[number] = counter.get(number, 0) + 1
        
        # count identical pairs
        for frequency in counter.values():
            pair_counter += frequency * (frequency - 1) // 2
        
        return pair_counter





# Number of Subsequences That Satisfy the Given Sum Condition
# https://leetcode.com/problems/number-of-subsequences-that-satisfy-the-given-sum-condition/description/
"""
You are given an array of integers nums and an integer target.

Return the number of non-empty subsequences of nums such that the sum of the minimum and maximum element on it is less or equal to target. Since the answer may be too large, return it modulo 109 + 7.

 

Example 1:

Input: nums = [3,5,6,7], target = 9
Output: 4
Explanation: There are 4 subsequences that satisfy the condition.
[3] -> Min value + max value <= target (3 + 3 <= 9)
[3,5] -> (3 + 5 <= 9)
[3,5,6] -> (3 + 6 <= 9)
[3,6] -> (3 + 6 <= 9)
Example 2:

Input: nums = [3,3,6,8], target = 10
Output: 6
Explanation: There are 6 subsequences that satisfy the condition. (nums can have repeated numbers).
[3] , [3] , [3,3], [3,6] , [3,6] , [3,3,6]
Example 3:

Input: nums = [2,3,3,4,6,7], target = 12
Output: 61
Explanation: There are 63 non-empty subsequences, two of them do not satisfy the condition ([6,7], [7]).
Number of valid subsequences (63 - 2 = 61).
"""
print(Solution().numSubseq([3, 5, 6, 7], 9), 4)
print(Solution().numSubseq([3, 3, 6, 8], 10), 6)
print(Solution().numSubseq([2, 3, 3, 4, 6, 7], 12), 61)
print(Solution().numSubseq([7, 10, 7, 3, 7, 5, 4], 12), 56)
print(Solution().numSubseq([14,4,6,6,20,8,5,6,8,12,6,10,14,9,17,16,9,7,14,11,14,15,13,11,10,18,13,17,17,14,17,7,9,5,10,13,8,5,18,20,7,5,5,15,19,14], 22), 272187084)
print(Solution().numSubseq([9,25,9,28,24,12,17,8,28,7,21,25,10,2,16,19,12,13,15,28,14,12,24,9,6,7,2,15,19,13,30,30,23,19,11,3,17,2,14,20,22,30,12,1,11,2,2,20,20,27,15,9,10,4,12,30,13,5,2,11,29,5,3,13,22,5,16,19,7,19,11,16,11,25,29,21,29,3,2,9,20,15,9], 32), 91931447)


# O(nlogn), O(n)
# two pointers
class Solution:
    def numSubseq(self, numbers: list[int], target: int) -> int:
        numbers.sort()
        right = len(numbers) - 1
        counter = 0

        for left in range(len(numbers)):
            while (left <= right and 
                   numbers[left] + numbers[right] > target):
                right -= 1

            if left <= right:
                counter += 2 ** (right - left)
                counter %= 10**9 + 7
            
            if left == right:
                break

        return counter


# O(n2^n), O(n)
# backtracking
# slow
class Solution:
    def __init__(self):
        self.counter = 0

    def numSubseq(self, numbers: list[int], target: int) -> int:
        numbers.sort()
        subsequence = []

        def dfs(index):
            if index == len(numbers):
                if (subsequence and 
                        subsequence[0] + subsequence[-1] <= target):
                    self.counter += 1
                return
            
            subsequence.append(numbers[index])
            dfs(index + 1)
            subsequence.pop()
            dfs(index + 1)

        dfs(0)
        return self.counter % (10 ** 9 + 7)


# O(n2^n), O(n)
# backtracking
# slow
class Solution:
    def numSubseq(self, numbers: list[int], target: int) -> int:
        numbers.sort()
        subsequence = []

        def dfs(index, counter):
            if index == len(numbers):
                if (subsequence and 
                        subsequence[0] + subsequence[-1] <= target):
                    return 1
                else:
                    return 0
            
            subsequence.append(numbers[index])
            left = dfs(index + 1, counter)
            subsequence.pop()
            right = dfs(index + 1, counter)
            return (left + right) % (10 ** 9 + 7)

        return dfs(0, 0)





# Subarray Product Less Than K
# https://leetcode.com/problems/subarray-product-less-than-k/description/
"""
Given an array of integers nums and an integer k, return the number of contiguous subarrays where the product of all the elements in the subarray is strictly less than k.

 

Example 1:

Input: nums = [10,5,2,6], k = 100
Output: 8
Explanation: The 8 subarrays that have product less than 100 are:
[10], [5], [2], [6], [10, 5], [5, 2], [2, 6], [5, 2, 6]
Note that [10, 5, 2] is not included as the product of 100 is not strictly less than k.
Example 2:

Input: nums = [1,2,3], k = 0
Output: 0
"""
print(Solution().numSubarrayProductLessThanK([10, 5, 2, 6], 100), 8)
print(Solution().numSubarrayProductLessThanK([1, 2, 3], 0), 0)


# O(n), O(1)
# sliding window
class Solution:
    def numSubarrayProductLessThanK(self, numbers: list[int], k: int) -> int:
        left = 0
        window = 1
        counter = 0
        
        for right in range(len(numbers)):
            window *= numbers[right]  # add a number to the window

            while (left <= right and  # while in bounds and
                   window >= k):  # product is too large
                window //= numbers[left]  # subtract a number from the window
                left += 1

            counter += right - left + 1  # add the number of contiguous subarrays
            # every new `number` adds new number to all previous arrays
            # [5, 2] + [6], [2] + [6], [6]
            # eg. add 3 for [5, 2, 6], where in previous loop added 2 for [5, 2]
            
        return counter


# O(n3), O(1)
# brute force
class Solution:
    def numSubarrayProductLessThanK(self, numbers: list[int], k: int) -> int:
        counter = 0
        
        for left in range(len(numbers)):
            for right in range(left, len(numbers)):
                window = 1
        
                for number in numbers[left: right + 1]:
                    window *= number
                
                if window < k:
                    counter += 1
        
        return counter


# O(n2), O(1)
# brute force
class Solution:
    def numSubarrayProductLessThanK(self, numbers: list[int], k: int) -> int:
        counter = 0
        
        for left in range(len(numbers)):
            window = 1
      
            for right in range(left, len(numbers)):
                window *= numbers[right]
                
                if window < k:
                    counter += 1
        
        return counter





# Remove All Adjacent Duplicates in String II
# https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii/description/
"""
You are given a string s and an integer k, a k duplicate removal consists of choosing k adjacent and equal letters from s and removing them, causing the left and the right side of the deleted substring to concatenate together.

We repeatedly make k duplicate removals on s until we no longer can.

Return the final string after all such duplicate removals have been made. It is guaranteed that the answer is unique.

 

Example 1:

Input: s = "abcd", k = 2
Output: "abcd"
Explanation: There's nothing to delete.
Example 2:

Input: s = "deeedbbcccbdaa", k = 3
Output: "aa"
Explanation: 
First delete "eee" and "ccc", get "ddbbbdaa"
Then delete "bbb", get "dddaa"
Finally delete "ddd", get "aa"
Example 3:

Input: s = "pbbcggttciiippooaais", k = 2
Output: "ps"
"""
print(Solution().removeDuplicates("abcd", 2), "abcd")
print(Solution().removeDuplicates("deeedbbcccbdaa", 3), "aa")
print(Solution().removeDuplicates("pbbcggttciiippooaais", 2), "ps")


# O(n), O(n)
# stack
class Solution:
    def removeDuplicates(self, letters: str, k: int) -> str:
        stack = []  # [[letter, frequency]]

        for letter in letters:
            if (stack and 
                    stack[-1][0] == letter):
                stack[-1][1] += 1
                
                if (stack[-1][1] == k):
                    stack.pop()

            else:
                stack.append([letter, 1])

        return "".join(letter * frequency 
                       for letter, frequency in stack)


# O(n), O(n)
# stack
class Solution:
    def removeDuplicates(self, letters: str, k: int) -> str:
        stack = []  # [(letter, frequency)]

        for letter in letters:
            if (stack and 
                stack[-1][0] == letter and
                    stack[-1][1] == k - 1):
                for _ in range(k - 1):
                    stack.pop()
            elif (stack and 
                    stack[-1][0] == letter):
                stack.append((letter, stack[-1][1] + 1))
            else:
                stack.append((letter, 1))

        return "".join(letter for letter, _ in stack)  # "".join(map(lambda x: x[0], stack))


# O(n2), O(n)
class Solution:
    def removeDuplicates(self, letters: str, k: int) -> str:
        stack = []

        for letter in letters:
            is_substring = False

            if (stack and 
                    len(stack) >= k - 1):
                is_substring = True
                
                for index in range(1, k):
                    if stack[-index] != letter:
                        is_substring = False
                        break
                
                if is_substring:
                    for _ in range(k - 1):
                        stack.pop()
            
            if not is_substring:
                stack.append(letter)
        
        return "".join(stack)





# Successful Pairs of Spells and Potions
# https://leetcode.com/problems/successful-pairs-of-spells-and-potions/description/
"""
You are given two positive integer arrays spells and potions, of length n and m respectively, where spells[i] represents the strength of the ith spell and potions[j] represents the strength of the jth potion.

You are also given an integer success. A spell and potion pair is considered successful if the product of their strengths is at least success.

Return an integer array pairs of length n where pairs[i] is the number of potions that will form a successful pair with the ith spell.

 

Example 1:

Input: spells = [5,1,3], potions = [1,2,3,4,5], success = 7
Output: [4,0,3]
Explanation:
- 0th spell: 5 * [1,2,3,4,5] = [5,10,15,20,25]. 4 pairs are successful.
- 1st spell: 1 * [1,2,3,4,5] = [1,2,3,4,5]. 0 pairs are successful.
- 2nd spell: 3 * [1,2,3,4,5] = [3,6,9,12,15]. 3 pairs are successful.
Thus, [4,0,3] is returned.
Example 2:

Input: spells = [3,1,2], potions = [8,5,8], success = 16
Output: [2,0,2]
Explanation:
- 0th spell: 3 * [8,5,8] = [24,15,24]. 2 pairs are successful.
- 1st spell: 1 * [8,5,8] = [8,5,8]. 0 pairs are successful. 
- 2nd spell: 2 * [8,5,8] = [16,10,16]. 2 pairs are successful. 
Thus, [2,0,2] is returned.
"""
print(Solution().successfulPairs([5, 1, 3], [1, 2, 3, 4, 5], 7), [4, 0, 3])
print(Solution().successfulPairs([3, 1, 2], [8, 5, 8], 16), [2, 0, 2])
print(Solution().successfulPairs([39, 34, 6, 35, 18, 24, 40], [27, 37, 33, 34, 14, 7, 23, 12, 22, 37], 43), [10, 10, 9, 10, 10, 10, 10])

# blueprint
# [lm5,r10,15,20,25]
# [1, 2, 3, l4, r5]
# [3, 6, lr9, 12, 15]


# O(nlogn), O(n)
# binary search
class Solution:
    def successfulPairs(self, spells: list[int], potions: list[int], success: int) -> list[int]:
        potions.sort()
        success_list = [0] * len(spells)

        for index, spell in enumerate(spells):
            left = 0
            right = len(potions) - 1

            while left < right:
                middle = left + (right - left) // 2
                middle_potion = potions[middle]

                if middle_potion * spell < success:
                    left = middle + 1
                else:
                    right = middle

            if potions[right] * spell >= success:
                success_list[index] = len(potions) - right
        
        return success_list





# Swapping Nodes in a Linked List
# https://leetcode.com/problems/swapping-nodes-in-a-linked-list/description/
"""
You are given the head of a linked list, and an integer k.

Return the head of the linked list after swapping the values of the kth node from the beginning and the kth node from the end (the list is 1-indexed).

 

Example 1:


Input: head = [1,2,3,4,5], k = 2
Output: [1,4,3,2,5]
Example 2:

Input: head = [7,9,6,6,7,8,3,0,9,5], k = 5
Output: [7,9,6,6,8,7,3,0,9,5]
"""


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

# O(n), O(1)
# linked list, one pass
class Solution:
    def swapNodes(self, head: ListNode, k: int) -> ListNode:
        left = head
        right = head
        node = head

        for _ in range(k - 1):  # move both pointers
            left = left.next
            node = node.next
    
        # when node reaches the last node, right reaches its destination
        while node.next:
            node = node.next
            right = right.next
        
        # swap the node values
        left.val, right.val = right.val, left.val

        return head


# O(n), O(1)
# linked list, two pass
class Solution:
    def swapNodes(self, head: ListNode, k: int) -> ListNode:
        """
        right node index: list_lenght - k + 1 
        nodes distance: list_lenght - 2k + 1 # 5 - 4 + 1 = 2, 10 - 10 + 1 
        """
        # get list length
        list_length = 0
        node = head

        while node:
            list_length += 1
            node = node.next
        
        k = min(k, list_length - k + 1)  # k cannot exceed half of the linked list

        # move pointers to the right positions
        left = head
        right = head

        for _ in range(list_length - 2*k + 1):  # move the right pointer by the distance between those two
            right = right.next
        
        for _ in range(k - 1):  # move both pointers
            left = left.next
            right = right.next
        
        # swap the node values
        left.val, right.val = right.val, left.val

        return head





# Range Sum of BST
# https://leetcode.com/problems/range-sum-of-bst/description/
"""
Given the root node of a binary search tree and two integers low and high, return the sum of values of all nodes with a value in the inclusive range [low, high].

 

Example 1:
    __10
   /    \
  5      15
 / \       \
3   7       18

Input: root = [10,5,15,3,7,null,18], low = 7, high = 15
Output: 32
Explanation: Nodes 7, 10, and 15 are in the range [7, 15]. 7 + 10 + 15 = 32.
Example 2:
      ____10___
     /         \
    5__        _15
   /   \      /   \
  3     7    13    18
 /     /
1     6

Input: root = [10,5,15,3,7,13,18,1,null,6], low = 6, high = 10
Output: 23
Explanation: Nodes 6, 7, and 10 are in the range [6, 10]. 6 + 7 + 10 = 23.
"""


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# O(n), O(n)
# binary tree
class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        def dfs(node):
            if not node:
                return 0
            
            return (
                (node.val if node.val >= low and node.val <= high else 0) +
                (dfs(node.left) if  node.val > low else 0) + 
                (dfs(node.right) if  node.val < high else 0))

        return dfs(root)


# O(n), O(n)
# binary tree
class Solution:
    def __init__(self):
        self.total = 0

    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        def dfs(node):
            if not node:
                return
            
            if (node.val >= low and 
                    node.val <= high):
                self.total += node.val
            
            if node.val > low:
                dfs(node.left)
            if node.val < high:
                dfs(node.right)
        
        dfs(root)
        return self.total





# Permutations II
# https://leetcode.com/problems/permutations-ii/description/
"""
Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order.

 

Example 1:

Input: nums = [1,1,2]
Output:
[[1,1,2],
 [1,2,1],
 [2,1,1]]
Example 2:

Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
"""
print(Solution().permuteUnique([1, 1, 2]), [[1, 2, 1], [2, 1, 1], [1, 1, 2]])
print(Solution().permuteUnique([1, 2, 3]), [[1, 3, 2], [1, 2, 3], [2, 1, 3], [3, 2, 1], [3, 1, 2], [2, 3, 1]])
print(Solution().permuteUnique([1]), [[1]])


# O(n!), O(n!)
# backtracking, set
class Solution:
    def permuteUnique(self, numbers: list[int]) -> list[list[int]]:
        permutation = []
        permutation_set = set()

        def dfs(numbers):
            if not numbers:
                permutation_set.add(tuple(permutation))
                return

            for index, number in enumerate(numbers):
                permutation.append(number)
                dfs(numbers[:index] + numbers[index + 1:])
                permutation.pop()

        dfs(numbers)
        return list(permutation_set)


# O(n!), O(n!)
# backtracking, set
class Solution:
    def permuteUnique(self, numbers: list[int]) -> list[list[int]]:
        permutation_set = set()

        def dfs(left):
            if left == len(numbers):
                permutation_set.add(tuple(numbers))
                return

            for right in range(left, len(numbers)):
                numbers[left], numbers[right] = numbers[right], numbers[left]
                dfs(left + 1)
                numbers[left], numbers[right] = numbers[right], numbers[left]

        dfs(0)
        return list(permutation_set)


# O(n!), O(n!)
# backtracking, hash map
class Solution:
    def permuteUnique(self, numbers: list[int]) -> list[list[int]]:
        permutation = []
        permutation_list = []
        counter = {}

        for number in numbers:
            counter[number] = counter.get(number, 0) + 1

        def dfs():
            if len(permutation) == len(numbers):
                permutation_list.append(permutation[:])
                return

            for number in counter:
                if counter[number] > 0:
                    permutation.append(number)
                    counter[number] -= 1
                    dfs()
                    permutation.pop()
                    counter[number] += 1

        dfs()
        return permutation_list




# Pascal's Triangle II
# https://leetcode.com/problems/pascals-triangle-ii/description/
"""
Given an integer rowIndex, return the rowIndexth (0-indexed) row of the Pascal's triangle.

In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:


Example 1:

Input: rowIndex = 3
Output: [1,3,3,1]
Example 2:

Input: rowIndex = 0
Output: [1]
Example 3:

Input: rowIndex = 1
Output: [1,1]
"""
print(Solution().getRow(3), [1, 3, 3, 1])
print(Solution().getRow(0), [1])
print(Solution().getRow(1), [1, 1])
print(Solution().getRow(2), [1, 2, 1])


# O(n2), O(n)
class Solution:
    def getRow(self, rowIndex: int) -> list[int]:
        row = [1]

        for index1 in range(rowIndex):
            new_row = [1] * (index1 + 2)

            for index2 in range(index1):
                new_row[index2 + 1] = row[index2] + row[index2 + 1]
            
            row = new_row

        return row





# N-th Tribonacci Number
# https://leetcode.com/problems/n-th-tribonacci-number/description/
"""
The Tribonacci sequence Tn is defined as follows: 

T0 = 0, T1 = 1, T2 = 1, and Tn+3 = Tn + Tn+1 + Tn+2 for n >= 0.

Given n, return the value of Tn.

 

Example 1:

Input: n = 4
Output: 4
Explanation:
T_3 = 0 + 1 + 1 = 2
T_4 = 1 + 1 + 2 = 4
Example 2:

Input: n = 25
Output: 1389537
"""
print(Solution().tribonacci(3), 2)
print(Solution().tribonacci(4), 4)
print(Solution().tribonacci(25), 1389537)
print(Solution().tribonacci(0), 0)
print(Solution().tribonacci(31), 53798080)
print(Solution().tribonacci(35), 615693474)


class Solution:
    def tribonacci(self, number: int) -> int:
        """
        O(n), O(1)
        dp, bottom-up
        """
        cache = [0, 1, 1]

        if number < 3:
            return cache[number]
        
        for _ in range(3, number + 1):
            cache = [cache[1], cache[2], cache[2] + cache[1] + cache[0]]

        return cache[2]


class Solution:
    def tribonacci(self, number: int) -> int:
        """
        O(n), O(1)
        dp, bottom-up, fancy
        """
        triplet = [0, 1, 1]
        
        for index in range(3, number + 1):
            triplet[index % 3] = sum(triplet)

        return triplet[number % 3]


class Solution:
    def tribonacci(self, number: int) -> int:
        """
        O(n), O(n)
        dp, bottom-up
        """
        cache = [None] * (number + 1)
        cache[:3] = [0, 1, 1]

        if number < 3:
            return cache[number]
        
        for index in range(3, number + 1):
            cache[index] = cache[index - 1] + cache[index - 2] + cache[index - 3]

        return cache[number]


class Solution:
    def tribonacci(self, number: int) -> int:
        """
        O(n), O(n)
        dp, top-down with memoization as hash map
        """
        memo = {0: 0, 1: 1, 2: 1}
        
        def dfs(index):
            if index < 0:
                return 0
            elif index in memo:
                return memo[index]
            
            memo[index] = dfs(index - 1) + dfs(index - 2) + dfs(index - 3)
            return memo[index]

        return dfs(number)


class Solution:
    def tribonacci(self, number: int) -> int:
        """
        O(n), O(n)
        dp, top-down with memoization as list
        """
        memo = [None] * (number + 1)
        memo[:3] = [0, 1, 1]
        
        def dfs(index):
            if index < 0:
                return 0
            elif memo[index] is not None:
                return memo[index]
            
            memo[index] = dfs(index - 1) + dfs(index - 2) + dfs(index - 3)
            return memo[index]

        return dfs(number)


class Solution:
    def tribonacci(self, number: int) -> int:
        """
        O(2^n), O(n)
        brute force, tle
        """
        memo = {0: 0, 1: 1, 2: 1}
        
        def dfs(index):
            if index < 0:
                return 0
            elif index in memo:
                return memo[index]
            
            return dfs(index - 1) + dfs(index - 2) + dfs(index - 3)

        return dfs(number)


class Solution:
    def tribonacci(self, number: int) -> int:
        """
        O(2^n), O(n)
        brute force, tle
        """
        memo = [None] * (number + 1)
        memo[:3] = [0, 1, 1]
        
        def dfs(index):
            if index < 0:
                return 0
            elif index in memo:
                return memo[index]
            
            return dfs(index - 1) + dfs(index - 2) + dfs(index - 3)

        return dfs(number)





# Find Words That Can Be Formed by Characters
# https://leetcode.com/problems/find-words-that-can-be-formed-by-characters/description/
"""
You are given an array of strings words and a string chars.

A string is good if it can be formed by characters from chars (each character can only be used once).

Return the sum of lengths of all good strings in words.

 

Example 1:

Input: words = ["cat","bt","hat","tree"], chars = "atach"
Output: 6
Explanation: The strings that can be formed are "cat" and "hat" so the answer is 3 + 3 = 6.
Example 2:

Input: words = ["hello","world","leetcode"], chars = "welldonehoneyr"
Output: 10
Explanation: The strings that can be formed are "hello" and "world" so the answer is 5 + 5 = 10.
"""
print(Solution().countCharacters(["cat", "bt", "hat", "tree"], "atach"), 6)
print(Solution().countCharacters(["hello", "world", "leetcode"], "welldonehoneyr"), 10)


# O(n2), O(1)
# hash map
class Solution:
    def is_word_good(self, word: str, counter: dict) -> bool:
        counter_copy = counter.copy()

        for letter in word:
            if letter not in counter_copy:
                return
            else:
                if counter_copy[letter] == 0:
                    return
                counter_copy[letter] -= 1
        
        return True

    def countCharacters(self, words: list[str], chars: str) -> int:
        counter = {}  # {letter: frequency}
        sum_of_lengths = 0

        for char in chars:
            counter[char] = counter.get(char, 0) + 1

        for word in words:
            if self.is_word_good(word, counter):
                sum_of_lengths += len(word)
            
        return sum_of_lengths





# Largest 3-Same-Digit Number in String
# https://leetcode.com/problems/largest-3-same-digit-number-in-string/description/
"""
You are given a string num representing a large integer. An integer is good if it meets the following conditions:

It is a substring of num with length 3.
It consists of only one unique digit.
Return the maximum good integer as a string or an empty string "" if no such integer exists.

Note:

A substring is a contiguous sequence of characters within a string.
There may be leading zeroes in num or a good integer.
 

Example 1:

Input: num = "6777133339"
Output: "777"
Explanation: There are two distinct good integers: "777" and "333".
"777" is the largest, so we return "777".
Example 2:

Input: num = "2300019"
Output: "000"
Explanation: "000" is the only good integer.
Example 3:

Input: num = "42352338"
Output: ""
Explanation: No substring of length 3 consists of only one unique digit. Therefore, there are no good integers.
"""
print(Solution().largestGoodInteger("6777133339"), "777")
print(Solution().largestGoodInteger("2300019"), "000")
print(Solution().largestGoodInteger("42352338"), "")
print(Solution().largestGoodInteger("7678222622241118390785777474281834906756431393782326744172075725179542796491876218340"), "777")


# O(n), O(1)
class Solution:
    def largestGoodInteger(self, number: str) -> str:
        triplet = []
        solution = []
        
        for digit in number:
            if (solution and  # if solution exists
                    digit <= solution[0]):  # digit smaller than solution
                triplet = []  # reset triplet
                continue

            if (triplet and  # if triplet exists
                    digit != triplet[-1]):  # digit is different than those in current triplet
                triplet = []  # reset triplet
            
            triplet.append(digit)  # append ditigt to the triplet

            if len(triplet) == 3:  # if triplet is long enough
                solution = triplet.copy()  # save curren triplet
        
        return "".join(solution)

# O(n), O(1)
class Solution:
    def largestGoodInteger(self, number: str) -> str:
        triplet = ""
        
        for index in range(len(number) - 2):
            if number[index] == number[index + 1] == number[index + 2]:
                triplet = max(triplet, number[index] * 3)
        
        return triplet





# Array With Elements Not Equal to Average of Neighbors
# https://leetcode.com/problems/array-with-elements-not-equal-to-average-of-neighbors/description/
"""
You are given a 0-indexed array nums of distinct integers. You want to rearrange the elements in the array such that every element in the rearranged array is not equal to the average of its neighbors.

More formally, the rearranged array should have the property such that for every i in the range 1 <= i < nums.length - 1, (nums[i-1] + nums[i+1]) / 2 is not equal to nums[i].

Return any rearrangement of nums that meets the requirements.

 

Example 1:

Input: nums = [1,2,3,4,5]
Output: [1,2,4,5,3]
Explanation:
When i=1, nums[i] = 2, and the average of its neighbors is (1+4) / 2 = 2.5.
When i=2, nums[i] = 4, and the average of its neighbors is (2+5) / 2 = 3.5.
When i=3, nums[i] = 5, and the average of its neighbors is (4+3) / 2 = 3.5.
Example 2:

Input: nums = [6,2,0,9,7]
Output: [9,7,6,2,0]
Explanation:
When i=1, nums[i] = 7, and the average of its neighbors is (9+6) / 2 = 7.5.
When i=2, nums[i] = 6, and the average of its neighbors is (7+2) / 2 = 4.5.
When i=3, nums[i] = 2, and the average of its neighbors is (6+0) / 2 = 3.
"""
print(Solution().rearrangeArray([1, 2, 3, 4, 5]), [1, 5, 2, 4, 3])
print(Solution().rearrangeArray([1, 2, 3, 4]), [1, 4, 2, 3])
print(Solution().rearrangeArray([6, 2, 0, 9, 7]), [0, 9, 2, 7, 6])
print(Solution().rearrangeArray([1, 3, 2]), [1, 3, 2])


# O(nlogn), O(n)
# two pointers
class Solution:
    def rearrangeArray(self, numbers: list[int]) -> list[int]:
        numbers.sort()
        new_numbers = [0] * len(numbers)
        index = 0
        left = 0
        right = len(numbers) - 1
        
        while left < right:
            new_numbers[index] = numbers[left]
            index += 1
            new_numbers[index] = numbers[right]
            index += 1
            left += 1
            right -= 1
        
        if len(numbers) % 2:
            new_numbers[-1] = numbers[len(numbers) // 2]
        
        return new_numbers


# O(nlogn), O(n)
# slice
class Solution:
    def rearrangeArray(self, numbers: list[int]) -> list[int]:
        numbers.sort()
        length = len(numbers)
        new_numbers = [0] * length

        new_numbers[: length + 1 :2] = numbers[:(length + 1) // 2]
        new_numbers[1: length + 1 :2] = numbers[(length + 1) // 2 :]

        return new_numbers





# Length of Longest Subarray With at Most K Frequency
# https://leetcode.com/problems/length-of-longest-subarray-with-at-most-k-frequency/description/
"""
You are given an integer array nums and an integer k.

The frequency of an element x is the number of times it occurs in an array.

An array is called good if the frequency of each element in this array is less than or equal to k.

Return the length of the longest good subarray of nums.

A subarray is a contiguous non-empty sequence of elements within an array.

 

Example 1:

Input: nums = [1,2,3,1,2,3,1,2], k = 2
Output: 6
Explanation: The longest possible good subarray is [1,2,3,1,2,3] since the values 1, 2, and 3 occur at most twice in this subarray. Note that the subarrays [2,3,1,2,3,1] and [3,1,2,3,1,2] are also good.
It can be shown that there are no good subarrays with length more than 6.
Example 2:

Input: nums = [1,2,1,2,1,2,1,2], k = 1
Output: 2
Explanation: The longest possible good subarray is [1,2] since the values 1 and 2 occur at most once in this subarray. Note that the subarray [2,1] is also good.
It can be shown that there are no good subarrays with length more than 2.
Example 3:

Input: nums = [5,5,5,5,5,5,5], k = 4
Output: 4
Explanation: The longest possible good subarray is [5,5,5,5] since the value 5 occurs 4 times in this subarray.
It can be shown that there are no good subarrays with length more than 4.
"""
print(Solution().maxSubarrayLength([1, 2, 3, 1, 2, 3, 1, 2], 2), 6)
print(Solution().maxSubarrayLength([1, 2, 1, 2, 1, 2, 1, 2], 1), 2)
print(Solution().maxSubarrayLength([5, 5, 5, 5, 5, 5, 5], 4), 4)
print(Solution().maxSubarrayLength([1, 1, 2], 2), 3)
print(Solution().maxSubarrayLength([1, 4, 4, 3], 1), 2)


# O(n), O(n)
# sliding window
class Solution:
    def maxSubarrayLength(self, numbers: list[int], k: int) -> int:
        window = {}  # {number: frequency}
        left = 0
        subarray_length = 0

        for right, number in enumerate(numbers):
            window[number] = window.get(number, 0) + 1

            while window[number] > k:
                window[numbers[left]] -= 1
                if number not in window:
                    window.pop(number)
                left += 1

            subarray_length = max(subarray_length, 
                                  right - left + 1)

        return subarray_length





# Capacity To Ship Packages Within D Days
# https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/submissions/1505021062/
"""
A conveyor belt has packages that must be shipped from one port to another within days days.

The ith package on the conveyor belt has a weight of weights[i]. Each day, we load the ship with packages on the conveyor belt (in the order given by weights). We may not load more weight than the maximum weight capacity of the ship.

Return the least weight capacity of the ship that will result in all the packages on the conveyor belt being shipped within days days.

 

Example 1:

Input: weights = [1,2,3,4,5,6,7,8,9,10], days = 5
Output: 15
Explanation: A ship capacity of 15 is the minimum to ship all the packages in 5 days like this:
1st day: 1, 2, 3, 4, 5
2nd day: 6, 7
3rd day: 8
4th day: 9
5th day: 10

Note that the cargo must be shipped in the order given, so using a ship of capacity 14 and splitting the packages into parts like (2, 3, 4, 5), (1, 6, 7), (8), (9), (10) is not allowed.
Example 2:

Input: weights = [3,2,2,4,1,4], days = 3
Output: 6
Explanation: A ship capacity of 6 is the minimum to ship all the packages in 3 days like this:
1st day: 3, 2
2nd day: 2, 4
3rd day: 1, 4
Example 3:

Input: weights = [1,2,3,1,1], days = 4
Output: 3
Explanation:
1st day: 1
2nd day: 2
3rd day: 3
4th day: 1, 1
"""
print(Solution().shipWithinDays([1, 2, 3, 1, 1], 4), 3)
print(Solution().shipWithinDays([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5), 15)
print(Solution().shipWithinDays([3, 2, 2, 4, 1, 4], 3), 6)


# O(nlogn), O(1)
# binary search
class Solution:
    def days_to_ship(self, capacity: int) -> int:
        days = 1  # days to ship with current capacity
        current_capacity = capacity  # capacity for current day

        for weight in self.weights:
            if current_capacity - weight < 0:  # if ship capacity in bounds
                days += 1  # increse days to ship
                current_capacity = capacity

            current_capacity -= weight  # add a cargo to the current ship         

        return days  # days to ship with current capacity


    def shipWithinDays(self, weights: list[int], days: int) -> int:
        self.weights = weights
        low_capacity = max(weights)  # min ship cargo
        high_capacity = sum(weights)  # max ship cargo

        while low_capacity < high_capacity:
            capacity = (low_capacity + high_capacity) // 2  # capacity of a cargo

            if (self.days_to_ship(capacity) > days):  # if more days to ship than planned
                low_capacity = capacity + 1  # increase capacity
            else:
                high_capacity = capacity  # decrease capacity

        return high_capacity





# Design Linked List
# https://leetcode.com/problems/design-linked-list/description/
"""
Design your implementation of the linked list. You can choose to use a singly or doubly linked list.
A node in a singly linked list should have two attributes: val and next. val is the value of the current node, and next is a pointer/reference to the next node.
If you want to use the doubly linked list, you will need one more attribute prev to indicate the previous node in the linked list. Assume all nodes in the linked list are 0-indexed.

Implement the MyLinkedList class:

MyLinkedList() Initializes the MyLinkedList object.
int get(int index) Get the value of the indexth node in the linked list. If the index is invalid, return -1.
void addAtHead(int val) Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
void addAtTail(int val) Append a node of value val as the last element of the linked list.
void addAtIndex(int index, int val) Add a node of value val before the indexth node in the linked list. If index equals the length of the linked list, the node will be appended to the end of the linked list. If index is greater than the length, the node will not be inserted.
void deleteAtIndex(int index) Delete the indexth node in the linked list, if the index is valid.
 

Example 1:

Input
["MyLinkedList", "addAtHead", "addAtTail", "addAtIndex", "get", "deleteAtIndex", "get"]
[[], [1], [3], [1, 2], [1], [1], [1]]
Output
[null, null, null, null, 2, null, 3]

Explanation
MyLinkedList myLinkedList = new MyLinkedList();
myLinkedList.addAtHead(1);
myLinkedList.addAtTail(3);
myLinkedList.addAtIndex(1, 2);    // linked list becomes 1->2->3
myLinkedList.get(1);              // return 2
myLinkedList.deleteAtIndex(1);    // now the linked list is 1->3
myLinkedList.get(1);              // return 3
"""


# dummy –> head –> .. -> node -> .. –> tail –> None
# O(n), O(n)
# linked list, singly linked list

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class MyLinkedList:
    def __init__(self):
      self.dummy = ListNode()  

    def get(self, index: int) -> int:
        node = self.dummy.next
        if not node:  # no nodes
            return -1

        while index:  # while not in the right index
            if not node.next:  # if no next node
                return -1
            
            node = node.next  # next node
            index -= 1  # next index
            
        return node.val


    def addAtHead(self, val: int) -> None:
        head = ListNode(val, self.dummy.next)  # create a new head
        self.dummy.next = head  # point dummy to new head
        

    def addAtTail(self, val: int) -> None:
        tail = ListNode(val)  # create a new tail
        node = self.dummy
        
        while node.next:
            node = node.next
        
        node.next = tail


    def addAtIndex(self, index: int, val: int) -> None:
        node = self.dummy

        while index:  # while not in the right index
            if not node.next:  # if no next node
                return
            
            node = node.next  # next node
            index -= 1  # next index

        new_node = ListNode(val, node.next)
        node.next = new_node


    def deleteAtIndex(self, index: int) -> None:
        node = self.dummy

        while index:  # while not in the right index
            if not node.next:  # if no next node
                return
            
            node = node.next  # next node
            index -= 1  # next index

        if not node.next:  # if no next node
            return

        node.next = node.next.next


# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)


# left <–> head <–> ... <–> node <–> ... <–> tail <–> right –> None
# O(n), O(n)
# linked list, doubly linked list
class ListNode:
    def __init__(self, val=0, next=None, prev=None):
        self.val = val
        self.next = next
        self.prev = prev


class MyLinkedList:
    def __init__(self):
        self.left = ListNode()  # left dummy node
        self.right = ListNode()  # right dummy node
        self.left.next = self.right
        self.right.prev = self.left

    def addAtHead(self, val: int) -> None:
        next = self.left.next  # get old head
        prev = self.left  # get left dummy node
        node = ListNode(val, next, prev)  # create a new head
        prev.next = node  # point left dummy to new head
        next.prev = node  # point old head to new head

    def addAtTail(self, val: int) -> None:
        next = self.right  # get right dummy node
        prev = self.right.prev  # get old tail
        node = ListNode(val, next, prev)  # create a new tail
        prev.next = node  # point old tail to new tail
        next.prev = node  # point right dummy to new tail

    def get(self, index: int) -> int:
        node = self.left.next  # get head node

        while index and node != self.right:  # while index > 0 and not on right node
            node = node.next  # next node
            index -= 1  # next index

        if (index == 0 and
                node != self.right):  # not on right node
            return node.val
        else:
            return - 1

    def addAtIndex(self, index: int, val: int) -> None:
        next = self.left.next  # get head node

        while index and next != self.right:  # while index > 0 and next not on None
            next = next.next  # next node
            index -= 1  # next index

        if (index == 0 and next):  # not on right node
            prev = next.prev  # get previous node
            node = ListNode(val, next, prev)  # create a new node between next and previous nodes
            prev.next = node  # point previous node to new node
            next.prev = node  # point next node to new node

    def deleteAtIndex(self, index: int) -> None:
        node = self.left.next  # get head node

        while index and node.next != self.right:  # while index > 0 and next not on None
            node = node.next  # next node
            index -= 1  # next index

        if (node and index == 0 and 
                node != self.right):
            node.prev.next = node.next  # point previous node to next node
            node.next.prev = node.prev  # poion next node to previous node


# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)





# Leaf-Similar Trees
# https://leetcode.com/problems/leaf-similar-trees/description/
"""
Consider all the leaves of a binary tree, from left to right order, the values of those leaves form a leaf value sequence.
    ______3__
   /         \
  5__         1
 /   \       / \
6     2     9   8
     / \
    7   4


For example, in the given tree above, the leaf value sequence is (6, 7, 4, 9, 8).

Two binary trees are considered leaf-similar if their leaf value sequence is the same.

Return true if and only if the two given trees with head nodes root1 and root2 are leaf-similar.

 

Example 1:
    ______3__
   /         \
  5__         1
 /   \       / \
6     2     9   8
     / \
    7   4

    __3__
   /     \
  5       1__
 / \     /   \
6   7   4     2
             / \
            9   8
Input: root1 = [3,5,1,6,2,9,8,null,null,7,4], root2 = [3,5,1,6,7,4,2,null,null,null,null,null,null,9,8]
Output: true
Example 2:
  1
 / \
2   3

  1
 / \
3   2
Input: root1 = [1,2,3], root2 = [1,3,2]
Output: false
"""
print(Solution().leafSimilar(build_tree_from_list([3, 5, 1, 6, 2, 9, 8, None, None, 7, 4]), build_tree_from_list([3, 5, 1, 6, 7, 4, 2, None, None, None, None, None, None, 9, 8])), True)
print(Solution().leafSimilar(build_tree_from_list([1, 2, 3]), build_tree_from_list([1, 3, 2])), False)
print(Solution().leafSimilar(build_tree_from_list([1, 2]), build_tree_from_list([2, 2])), True)
print(Solution().leafSimilar(build_tree_from_list([3, 5, 1, 6, 7, 4, 2, None, None, None, None, None,None, 9, 11, None, None, 8, 10]), build_tree_from_list([3, 5, 1, 6, 2, 9, 8, None, None, 7, 4])), False)
print(Solution().leafSimilar(build_tree_from_list([3, 5, 1, 6, 2, 9, 8, None, None, 7, 4]), build_tree_from_list([3, 5, 1, 6, 7, 4, 2, None, None, None, None, None, None, 9, 11, None, None, 8, 10])), False)


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# O(n), O(n)
# binary tree
class Solution:
    def leafSimilar(self, root1: TreeNode, root2: TreeNode) -> bool:
        leaf_list = []

        def dfs(node):
            if not node:
                return
            elif (not node.left and  # if both child nodes are None (== leaf), 
                    not node.right):
                leaf_list.append(node.val)  # append value to leaf list
                return

            dfs(node.left)
            dfs(node.right)
        
        dfs(root1)
        leaf_list.reverse()

        def dfs2(node):
            if not node:
                return True
            elif (not node.left and 
                    not node.right):
                # early exit
                if (leaf_list and  # if list is empty then there is nothing to compare to
                        node.val == leaf_list[-1]):  # same values
                    leaf_list.pop()
                    return True
                else:
                    return False

            return (dfs2(node.left) and  # left child is leaf-similar
                    dfs2(node.right))  # right child is leaf-similar
        
        return (dfs2(root2) and  # all leafs from root2 are in root1
                not leaf_list)  # leaf list is empty





# Restore IP Addresses
# https://leetcode.com/problems/restore-ip-addresses/description/
"""
A valid IP address consists of exactly four integers separated by single dots. Each integer is between 0 and 255 (inclusive) and cannot have leading zeros.

For example, "0.1.2.201" and "192.168.1.1" are valid IP addresses, but "0.011.255.245", "192.168.1.312" and "192.168@1.1" are invalid IP addresses.
Given a string s containing only digits, return all possible valid IP addresses that can be formed by inserting dots into s. You are not allowed to reorder or remove any digits in s. You may return the valid IP addresses in any order.

 

Example 1:

Input: s = "25525511135"
Output: ["255.255.11.135","255.255.111.35"]
Example 2:

Input: s = "0000"
Output: ["0.0.0.0"]
Example 3:

Input: s = "101023"
Output: ["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]
"""
print(Solution().restoreIpAddresses("25525511135"), ["255.255.11.135", "255.255.111.35"])
print(Solution().restoreIpAddresses("0000"), ["0.0.0.0"])
print(Solution().restoreIpAddresses("101023"), ["1.0.10.23", "1.0.102.3", "10.1.0.23", "10.10.2.3", "101.0.2.3"])
print(Solution().restoreIpAddresses("000256"), [])


# O(1), O(1)
# backtarcking
# the recursion tree is bounded by 3**4 and validation within each call is O(1)
class Solution:
    def restoreIpAddresses(self, text: str) -> list[str]:
        if (len(text) < 4 or
                len(text) > 12):
            return []
        
        ip = []
        ip_list = []

        def dfs(index):
            if index == len(text):
                if len(ip) == 4:
                    ip_list.append(".".join(number[0] for number in ip))
                return
            
            # 0->3, 1->2, 2->1, 3->0
            # if len(text) - (index + 1) >= (3 - index):
            # check for 0:9
            ip.append([text[index]])
            dfs(index + 1)
            ip.pop()
            
            # check for 10:99
            if (index < len(text) - 1 and 
                    text[index] != "0"):
                ip.append([text[index: index + 2]])
                dfs(index + 2)
                ip.pop()

            # check for 100:255
            if (index < len(text) - 2 and 
                text[index] != "0" and
                    text[index: index + 3] <= "255"):
                ip.append([text[index: index + 3]])
                dfs(index + 3)
                ip.pop()

        dfs(0)
        return ip_list





# Design Add and Search Words Data Structure
# https://leetcode.com/problems/design-add-and-search-words-data-structure/description/
"""
Design a data structure that supports adding new words and finding if a string matches any previously added string.

Implement the WordDictionary class:

WordDictionary() Initializes the object.
void addWord(word) Adds word to the data structure, it can be matched later.
bool search(word) Returns true if there is any string in the data structure that matches word or false otherwise. word may contain dots '.' where dots can be matched with any letter.
 

Example:

Input
["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
[[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
Output
[null,null,null,null,false,true,true,true]

Explanation
WordDictionary wordDictionary = new WordDictionary();
wordDictionary.addWord("bad");
wordDictionary.addWord("dad");
wordDictionary.addWord("mad");
wordDictionary.search("pad"); // return False
wordDictionary.search("bad"); // return True
wordDictionary.search(".ad"); // return True
wordDictionary.search("b.."); // return True
"""
word_dictionary = WordDictionary()
word_dictionary.addWord("bad")
word_dictionary.addWord("dad")
word_dictionary.addWord("mad")
print(word_dictionary.search("pad")) # False
print(word_dictionary.search("bad")) # True
print(word_dictionary.search(".ad")) # True
print(word_dictionary.search("b..")) # True


class TrieNode:
    def __init__(self):
        self.letters = {}
        self.is_word = False


class WordDictionary:
    def __init__(self):
        self.root = TrieNode()


    def addWord(self, word: str) -> None:
        node = self.root
        
        for letter in word:
            if letter not in node.letters:
                node.letters[letter] = TrieNode()
            
            node = node.letters[letter]
        
        node.is_word = True


    def search(self, word: str) -> bool:
        def dfs(left, node):
            for right, letter in enumerate(word[left:], left):
                if letter == ".":
                    for value_node in node.letters.values():  # if key is `.` check all value nodes
                        if dfs(right + 1, value_node):
                            return True
                    return False
                else:
                    if letter not in node.letters:
                        return False
                    else:
                        node = node.letters[letter]
    
            return node.is_word
    
        return dfs(0, self.root)


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)





# Triangle
# https://leetcode.com/problems/triangle/description/
"""
Given a triangle array, return the minimum path sum from top to bottom.

For each step, you may move to an adjacent number of the row below. More formally, if you are on index i on the current row, you may move to either index i or index i + 1 on the next row.

 

Example 1:

Input: triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
Output: 11
Explanation: The triangle looks like:
   2
  3 4
 6 5 7
4 1 8 3
The minimum path sum from top to bottom is 2 + 3 + 5 + 1 = 11 (underlined above).
Example 2:

Input: triangle = [[-10]]
Output: -10
"""
print(Solution().minimumTotal([[2]]), 2)
print(Solution().minimumTotal([[2], [3, 4], [6, 5, 7], [4, 1, 8, 3]]), 11)
print(Solution().minimumTotal([[-10]]), -10)


class Solution:
    def minimumTotal(self, triangle: list[list[int]]) -> int:
        """
        O(n), O(n)
        dp, bottom-up
        in-place
        """
        for level in reversed(range(len(triangle) - 1)):
            for index in range(len(triangle[level])):
                triangle[level][index] += (
                    min(triangle[level + 1][index], 
                        triangle[level + 1][index + 1]))
                
        return triangle[0][0]


class Solution:
    def minimumTotal(self, triangle: list[list[int]]) -> int:
        """
        O(n), O(n)
        dp, bottom-up
        in-place
        """
        for level in range(1, len(triangle)):
            triangle[level - 1] = ([triangle[level - 1][0]] + 
                                   triangle[level - 1] + 
                                   [triangle[level - 1][-1]])
            
            for index in range(len(triangle[level])):
                triangle[level][index] += (
                    min(triangle[level - 1][index], 
                        triangle[level - 1][index + 1]))
                
        return min(triangle[-1])


class Solution:
    def minimumTotal(self, triangle: list[list[int]]) -> int:
        """
        O(2^n), O(n)
        brute force, tle
        """
        def dfs(index: int, level: int) -> int:
            if level == len(triangle):
                return 0
            
            return (triangle[level][index] + 
                    min(dfs(index, level + 1), 
                        dfs(index + 1, level + 1)))
        
        return dfs(0, 0)





# Kth Largest Element in a Stream
# https://leetcode.com/problems/kth-largest-element-in-a-stream/description/
"""
You are part of a university admissions office and need to keep track of the kth highest test score from applicants in real-time. This helps to determine cut-off marks for interviews and admissions dynamically as new applicants submit their scores.

You are tasked to implement a class which, for a given integer k, maintains a stream of test scores and continuously returns the kth highest test score after a new score has been submitted. More specifically, we are looking for the kth highest score in the sorted list of all scores.

Implement the KthLargest class:

KthLargest(int k, int[] nums) Initializes the object with the integer k and the stream of test scores nums.
int add(int val) Adds a new test score val to the stream and returns the element representing the kth largest element in the pool of test scores so far.
 

Example 1:

Input:
["KthLargest", "add", "add", "add", "add", "add"]
[[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]]

Output: [null, 4, 5, 5, 8, 8]

Explanation:

KthLargest kthLargest = new KthLargest(3, [4, 5, 8, 2]);
kthLargest.add(3); // return 4
kthLargest.add(5); // return 5
kthLargest.add(10); // return 5
kthLargest.add(9); // return 8
kthLargest.add(4); // return 8

Example 2:

Input:
["KthLargest", "add", "add", "add", "add"]
[[4, [7, 7, 7, 7, 8, 3]], [2], [10], [9], [9]]

Output: [null, 7, 7, 7, 8]

Explanation:

KthLargest kthLargest = new KthLargest(4, [7, 7, 7, 7, 8, 3]);
kthLargest.add(2); // return 7
kthLargest.add(10); // return 7
kthLargest.add(9); // return 7
kthLargest.add(9); // return 8
"""
kthLargest = KthLargest(3, [4, 5, 8, 2])
print(kthLargest.add(3)) #  return 4
print(kthLargest.add(5)) #  return 5
print(kthLargest.add(10)) #  return 5
print(kthLargest.add(9)) #  return 8
print(kthLargest.add(4)) #  return 8


import heapq

# O(nlogn), O(k)
# heap
class KthLargest:
    def __init__(self, k: int, numbers: list[int]):
        self.k = k
        self.numbers = numbers
        heapq.heapify(self.numbers)  # min heap

        while len(self.numbers) > self.k:
            heapq.heappop(self.numbers)

    def add(self, val: int) -> int:
        heapq.heappush(self.numbers, val)

        while len(self.numbers) > self.k:
            heapq.heappop(self.numbers)
        
        return self.numbers[0]


# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)


# O(n^2*logn), O(n)
# sort
class KthLargest:
    def __init__(self, k: int, numbers: list[int]):
        self.numbers = numbers
        self.k = k

    def add(self, val: int) -> int:
        self.numbers.append(val)
        self.numbers.sort(reverse=True)

        return self.numbers[self.k - 1]





# Last Stone Weight
# https://leetcode.com/problems/last-stone-weight/description/
"""
You are given an array of integers stones where stones[i] is the weight of the ith stone.

We are playing a game with the stones. On each turn, we choose the heaviest two stones and smash them together. Suppose the heaviest two stones have weights x and y with x <= y. The result of this smash is:

If x == y, both stones are destroyed, and
If x != y, the stone of weight x is destroyed, and the stone of weight y has new weight y - x.
At the end of the game, there is at most one stone left.

Return the weight of the last remaining stone. If there are no stones left, return 0.

 

Example 1:

Input: stones = [2,7,4,1,8,1]
Output: 1
Explanation: 
We combine 7 and 8 to get 1 so the array converts to [2,4,1,1,1] then,
we combine 2 and 4 to get 2 so the array converts to [2,1,1,1] then,
we combine 2 and 1 to get 1 so the array converts to [1,1,1] then,
we combine 1 and 1 to get 0 so the array converts to [1] then that's the value of the last stone.
Example 2:

Input: stones = [1]
Output: 1
"""
print(Solution().lastStoneWeight([2, 7, 4, 1, 8, 1]), 1)
print(Solution().lastStoneWeight([1]), 1)
print(Solution().lastStoneWeight([1, 1]), 0)


import heapq

# O(nlogn), O(n)
# heap
class Solution:
    def lastStoneWeight(self, stones: list[int]) -> int:
        stones = [-stone for stone in stones]
        heapq.heapify(stones)  # make a heap
        
        while len(stones) > 1:
            # 8 - 7 = 1  => -8 - -7 = - 1
            stone = heapq.heappop(stones) - heapq.heappop(stones)
            
            if stone:
                heapq.heappush(stones, stone)

        return -stones[0] if stones else 0


# O(nlogn), O(n)
# binary search
class Solution:
    def lastStoneWeight(self, stones: list[int]) -> int:
        stones.sort()
        
        while len(stones) > 1:
            stone = stones.pop() - stones.pop()
            
            if stone:
                left = 0
                right = len(stones)

                while left < right:  # binary search to find slot for a stone
                    middle = (right + left) // 2
                    middle_stone = stones[middle]

                    if stone > middle_stone:
                        left = middle + 1
                    else:
                        right = middle

                stones.insert(right, stone)

        return stones[0] if stones else 0





# Sort an Array
# https://leetcode.com/problems/sort-an-array/description/
"""
Given an array of integers nums, sort the array in ascending order and return it.

You must solve the problem without using any built-in functions in O(nlog(n)) time complexity and with the smallest space complexity possible.

 

Example 1:

Input: nums = [5,2,3,1]
Output: [1,2,3,5]
Explanation: After sorting the array, the positions of some numbers are not changed (for example, 2 and 3), while the positions of other numbers are changed (for example, 1 and 5).
Example 2:

Input: nums = [5,1,1,2,0,0]
Output: [0,0,1,1,2,5]
Explanation: Note that the values of nums are not necessairly unique.
"""
print(Solution().sortArray([5, 2, 3, 1]), [1, 2, 3, 5])
print(Solution().sortArray([5, 1, 1, 2, 0, 0]), [0, 0, 1, 1, 2, 5])

# O(nlogn), O(n)
# megret sort
class Solution:
    def sortArray(self, numbers: list[int]) -> list[int]:
        def merge(left, middle, right):
            left_chunk = numbers[left: middle + 1]  # clone left chunk
            right_chunk = numbers[middle + 1: right + 1]  # colone right chunk
            index = left
            left, right = 0, 0

            while (left < len(left_chunk) and  # while numbers in left chunk and
                   right < len(right_chunk)):  # in right chunk
                # if left number is less than right
                if left_chunk[left] <= right_chunk[right]:
                    numbers[index] = left_chunk[left]
                    left += 1
                else:
                    numbers[index] = right_chunk[right]
                    right += 1
                index += 1

            while left < len(left_chunk):  # while still numbers in left chunk
                numbers[index] = left_chunk[left]
                left += 1
                index += 1

            while right < len(right_chunk):  # while still numbers in right chunk
                numbers[index] = right_chunk[right]
                right += 1
                index += 1

        def merge_sort(left, right):
            if left == right:  # if pointers point only one number
                return

            middle = (left + right) // 2
            # divide
            merge_sort(left, middle)
            merge_sort(middle + 1, right)
            # merge
            merge(left, middle, right)

        merge_sort(0, len(numbers) - 1)
        return numbers


# O(nlogn), O(n)
# quick sort, tle
class Solution:
    def partition(self, numbers: list[int], left: int, end: int) -> int:
        pivot = numbers[end]
        left -= 1

        for right in range(left + 1, end):
            if numbers[right] < pivot:
                left += 1
                self.swap(numbers, left, right)

        self.swap(numbers, left + 1, end)
        return left + 1

    def swap(self, numbers: list[int], left: int, right: int) -> None:
        numbers[left], numbers[right] = numbers[right], numbers[left]

    def quick_sort(self, numbers: list[int], left: int, right: int) -> None:
        if left < right:
            pivot = self.partition(numbers, left, right)
            self.quick_sort(numbers, left, pivot - 1)
            self.quick_sort(numbers, pivot + 1, right)

    def quick_sort(self, numbers: list[int], left: int, right: int) -> None:
        if right <= left:
            return 
        pivot = self.partition(numbers, left, right)
        self.quick_sort(numbers, left, pivot - 1)
        self.quick_sort(numbers, pivot + 1, right)

    def sortArray(self, numbers: list[int]) -> list[int]:
        self.quick_sort(numbers, 0, len(numbers) - 1)
        return numbers


# O(n2), O(1)
# insertion sort, tle
class Solution:
    def insertionSort(self, numbers: list[int]) -> None:
        for right in range(1, len(numbers)):
            key = numbers[right]
            left = right - 1

            # Move elements of arr[0..i-1], that are
            # greater than key, to one position ahead
            # of their current position
            while (left >= 0 and 
                   numbers[left] > key):
                numbers[left + 1] = numbers[left]
                left -= 1
            numbers[left + 1] = key

    def sortArray(self, numbers: list[int]) -> list[int]:
        self.insertionSort(numbers)
        return numbers





# Sort Colors
# https://leetcode.com/problems/sort-colors/description/
"""
Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.

We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.

You must solve this problem without using the library's sort function.

 

Example 1:

Input: nums = [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]
Example 2:

Input: nums = [2,0,1]
Output: [0,1,2]
"""
print(Solution().sortColors([2, 0, 2, 1, 1, 0]), [0, 0, 1, 1, 2, 2])
print(Solution().sortColors([2, 0, 1]), [0, 1, 2])


# O(n), O(1)
# two pointers, two pass
class Solution:
    def sortColors(self, nums: list[int]) -> None:
        left = 0

        # move zeros left
        for right, number in enumerate(nums):
            if number == 0:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1

        # move ones left
        for right, number in enumerate(nums[left:], left):
            if number == 1:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1

        return nums


# O(n), O(1)
# bucket sort, tow pass
class Solution:
    def sortColors(self, nums: list[int]) -> None:
        bucket = [0] * 3  # i-th index is a frequency of i-th number
        
        for number in nums:
            bucket[number] += 1
        
        index = 0
        
        for number, frequency in enumerate(bucket):
            for _ in range(frequency):
                nums[index] = number
                index += 1
            
        return nums


# O(n), O(1)
# three pointers, one pass
class Solution:
    def sortColors(self, nums: list[int]) -> None:
        left = 0
        right = len(nums) - 1
        index = 0

        while index <= right:
            number = nums[index]

            if number == 0:
                nums[left], nums[index] = nums[index], nums[left]
                left += 1
            elif number == 2:
                nums[index], nums[right] = nums[right], nums[index]
                right -= 1
                # swapped value should be checked at the next iteration
                index -= 1

            index += 1

        return nums




# Boats to Save People
# https://leetcode.com/problems/boats-to-save-people/description/
"""
You are given an array people where people[i] is the weight of the ith person, and an infinite number of boats where each boat can carry a maximum weight of limit. Each boat carries at most two people at the same time, provided the sum of the weight of those people is at most limit.

Return the minimum number of boats to carry every given person.

 

Example 1:

Input: people = [1,2], limit = 3
Output: 1
Explanation: 1 boat (1, 2)
Example 2:

Input: people = [3,2,2,1], limit = 3
Output: 3
Explanation: 3 boats (1, 2), (2) and (3)
Example 3:

Input: people = [3,5,3,4], limit = 5
Output: 4
Explanation: 4 boats (3), (3), (4), (5)
"""
print(Solution().numRescueBoats([1, 2], 3), 1)
print(Solution().numRescueBoats([3, 2, 2, 1], 3), 3)
print(Solution().numRescueBoats([3, 5, 3, 4], 5), 4)
print(Solution().numRescueBoats([3, 2, 3, 2, 2], 6), 3)


# O(nlogn), O(n)
# two pointers
class Solution:
    def numRescueBoats(self, people: list[int], limit: int) -> int:
        people.sort()
        left = 0
        right = len(people) - 1
        boats = 0

        while left <= right:
            delta = limit - people[right]
            right -= 1
            boats += 1

            if (left <= right and
                   people[left] <= delta):
                left += 1

        return boats


# O(n), O(n)
# two pointers, bucket
class Solution:
    def numRescueBoats(self, people: list[int], limit: int) -> int:
        bucket = [0] * (max(people) + 1)
        for weight in people:
            bucket[weight] += 1

        weight = 1  # current weight
        for index in range(len(people)):
            while bucket[weight] == 0:  # while no people of current weight
                weight += 1  # next weight

            people[index] = weight  # populate people weights
            bucket[weight] -= 1  # decrease current weight in bucket

        left = 0
        right = len(people) - 1
        boats = 0

        while left <= right:
            delta = limit - people[right]
            right -= 1
            boats += 1

            if (left <= right and
                    people[left] <= delta):
                left += 1

        return boats


# slow
class Solution:
    def numRescueBoats(self, people: list[int], limit: int) -> int:
        counter = {}
        weights = list(set(people))
        weights.sort()
        boat_count = 0

        for weight in people:
            counter[weight] = counter.get(weight, 0) + 1

        while weights:
            weight = weights[-1]
            counter[weight] -= 1
            if counter[weight] == 0:
                counter.pop(weight)
                weights.pop()

            if not weights:
                return boat_count + 1

            diff = limit - weight

            for smaller_weight in reversed(range(min(weights), diff + 1)):
                if smaller_weight in counter:
                    counter[smaller_weight] -= 1
                    if counter[smaller_weight] == 0:
                        counter.pop(smaller_weight)
                        weights.remove(smaller_weight)
                    break

            boat_count += 1

        return boat_count


# tle
class Solution:
    def numRescueBoats(self, people: list[int], limit: int) -> int:
        people.sort()
        boat_count = 0

        while people:
            weight = people.pop()
            if not people:
                return boat_count + 1

            for smaller_weight in reversed(people):
                if smaller_weight <= limit - weight:
                    people.remove(smaller_weight)
                    break

            boat_count += 1

        return boat_count





# Count Subarrays Where Max Element Appears at Least K Times
# https://leetcode.com/problems/count-subarrays-where-max-element-appears-at-least-k-times/description/
"""
You are given an integer array nums and a positive integer k.

Return the number of subarrays where the maximum element of nums appears at least k times in that subarray.

A subarray is a contiguous sequence of elements within an array.

 

Example 1:

Input: nums = [1,3,2,3,3], k = 2
Output: 6
Explanation: The subarrays that contain the element 3 at least 2 times are: [1,3,2,3], [1,3,2,3,3], [3,2,3], [3,2,3,3], [2,3,3] and [3,3].
Example 2:

Input: nums = [1,4,2,1], k = 3
Output: 0
Explanation: No subarray contains the element 4 at least 3 times.
"""
print(Solution().countSubarrays([1, 3, 2, 3, 3], 2), 6)
print(Solution().countSubarrays([1, 3, 2, 3, 3, 1], 2), 10)
print(Solution().countSubarrays([1, 3, 2, 3, 1], 2), 4)
print(Solution().countSubarrays([1, 3, 2, 3, 1, 1], 2), 6)
print(Solution().countSubarrays([1, 3, 2, 3, 1, 1, 3], 2), 10)
print(Solution().countSubarrays([1, 4, 2, 1], 3), 0)
print(Solution().countSubarrays([37,20,38,66,34,38,9,41,1,14,25,63,8,12,66,66,60,12,35,27,16,38,12,66,38,36,59,54,66,54,66,48,59,66,34,11,50,66,42,51,53,66,31,24,66,44,66,1,66,66,29,54], 5), 594)


# draft
# [1, 3, 2, 3, 3]
# [1,3,2,3], [3,2,3],        [1,3,2,3,3], [3,2,3,3], [2,3,3] and [3,3]

# [1, 3, 2, 3, 1]
# [1,3,2,3], [3,2,3],        [1,3,2,3,1], [3,2,3,1]


# O(n), O(1)
# sliding window
class Solution:
    def countSubarrays(self, numbers: list[int], k: int) -> int:
        left = 0  # start of the sliding window
        max_number = max(numbers)  # the maximum number from numbers
        max_count = 0  # count occurences of the maximum number
        counter = 0  # count number of subsets

        for right, number in enumerate(numbers):
            if number == max_number:  # if number is the maximum number
                max_count += 1  # increase maximum number counter

            # move the left pointer that in between left and right pointers
            # (inclusive) are exactly k maximun numbers
            while (max_count > k or  # too many maximum numbers or
                   (max_count == k and  # exact number of maximum numbers and
                    numbers[left] != max_number)):  # current numbers is not a maximum number
                if numbers[left] == max_number:  # if current number is a maximum number
                    max_count -= 1  # decrease maximum number counter
                left += 1  # move left pointer
            
            if max_count == k:  # exact number of maximum numbers
                counter += left + 1  # `left + 1` is the current subarray counter to add
        
        return counter


# O(n3), O(1)
# brute force
class Solution:
    def countSubarrays(self, numbers: list[int], k: int) -> int:
        counter = 0

        for i in range(len(numbers)):
            for j in range(i, len(numbers)):
                if numbers[i: j + 1].count(max(numbers)) >= 2:
                    counter += 1

        return counter

# O(n2), O(1)
# brute force
class Solution:
    def countSubarrays(self, numbers: list[int], k: int) -> int:
        counter = 0
        max_number = max(numbers)

        for i in range(len(numbers)):
            max_count = 0
           
            for j in range(i, len(numbers)):
                if numbers[j] == max_number:
                    max_count += 1
                if  max_count >= 2:
                    counter += 1

        return counter


# O(n), O(1)
# sliding window, tle
class Solution:
    def countSubarrays(self, numbers: list[int], k: int) -> int:
        left = 0  # left pointer for sliding window
        counter = 0  # count number of subsets
        numbers_max = max(numbers)  # the maximum number from numbers
        max_num_count = 0  # count occurences of the maximum number

        for number in numbers:
            if number == numbers_max:
                max_num_count += 1
            
            left_back = left  # copy of the left pointer
            max_num_count_back = max_num_count  # copy of occurences of the maximum number starting from left pointer
            
            while max_num_count >= k:  # while maximum occcurs greater equal to k
                counter += 1
                if numbers[left] == numbers_max:
                    max_num_count -= 1
                left += 1

            left = left_back  # restore left
            max_num_count = max_num_count_back  # restore max num count

        return counter





# 132 Pattern
# https://leetcode.com/problems/132-pattern/description/
"""
Given an array of n integers nums, a 132 pattern is a subsequence of three integers nums[i], nums[j] and nums[k] such that i < j < k and nums[i] < nums[k] < nums[j].

Return true if there is a 132 pattern in nums, otherwise, return false.

 

Example 1:

Input: nums = [1,2,3,4]
Output: false
Explanation: There is no 132 pattern in the sequence.
Example 2:

Input: nums = [3,1,4,2]
Output: true
Explanation: There is a 132 pattern in the sequence: [1, 4, 2].
Example 3:

Input: nums = [-1,3,2,0]
Output: true
Explanation: There are three 132 patterns in the sequence: [-1, 3, 2], [-1, 3, 0] and [-1, 2, 0].
"""


print(Solution().find132pattern([3, 1, 4, 2]), True)
print(Solution().find132pattern([1, 2, 3, 4]), False)
print(Solution().find132pattern([-1, 3, 2, 0]), True)
print(Solution().find132pattern([3, 5, 0, 3, 4]), True)
print(Solution().find132pattern([1, 0, 1, -4, -3]), False)
print(Solution().find132pattern([-2, 1, 2, -2, 1, 2]), True)
print(Solution().find132pattern([1, 2, 3, 4, -4, -3, -5, -1]), False)
print(Solution().find132pattern([1, 3, -4, 2]), True)


# O(n), O(n)
# stack
# monotonically decreasing stack
class Solution:
    def find132pattern(self, numbers: list[int]) -> bool:
        stack = []  # (min_left, number)  # monotonic decreasing
        prev_min = numbers[0]  # minimum before `j`

        for number in numbers[1:]:
            # pop all smaller equal values from the stack
            while stack and number >= stack[-1][1]:
                stack.pop()

            # check if `numbers[k]` is: nums[i] < nums[k] < nums[j]
            if stack and stack[-1][0] < number < stack[-1][1]:  # `number < stack[-1][1]` not nesessery because of previous while loop
                return True

            stack.append((prev_min, number))  # if stack is empty or `numbers[k]` not in bounds
            prev_min = min(prev_min, number)  # update previous minimum value

        return False


# O(n2), O(1)
# tle
class Solution:
    def find132pattern(self, numbers: list[int]) -> bool:
        for right, number in enumerate(numbers[2:], 2):
            middle = right - 1

            while middle > 0:
                if numbers[middle] > number:
                    left = middle - 1

                    while left >= 0:
                        if numbers[left] < number:
                            return True

                        left -= 1

                middle -= 1

        return False


# O(n3), O(1)
# brute force
class Solution:
    def find132pattern(self, numbers: list[int]) -> bool:
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                for k in range(j + 1, len(numbers)):
                    if numbers[i] < numbers[k] < numbers[j]:
                        return True
        return False





# Design Browser History
# https://leetcode.com/problems/design-browser-history/description/
"""
You have a browser of one tab where you start on the homepage and you can visit another url, get back in the history number of steps or move forward in the history number of steps.

Implement the BrowserHistory class:

BrowserHistory(string homepage) Initializes the object with the homepage of the browser.
void visit(string url) Visits url from the current page. It clears up all the forward history.
string back(int steps) Move steps back in history. If you can only return x steps in the history and steps > x, you will return only x steps. Return the current url after moving back in history at most steps.
string forward(int steps) Move steps forward in history. If you can only forward x steps in the history and steps > x, you will forward only x steps. Return the current url after forwarding in history at most steps.
 

Example:

Input:
["BrowserHistory","visit","visit","visit","back","back","forward","visit","forward","back","back"]
[["leetcode.com"],["google.com"],["facebook.com"],["youtube.com"],[1],[1],[1],["linkedin.com"],[2],[2],[7]]
Output:
[null,null,null,null,"facebook.com","google.com","facebook.com",null,"linkedin.com","google.com","leetcode.com"]

Explanation:
BrowserHistory browserHistory = new BrowserHistory("leetcode.com");
browserHistory.visit("google.com");       // You are in "leetcode.com". Visit "google.com"
browserHistory.visit("facebook.com");     // You are in "google.com". Visit "facebook.com"
browserHistory.visit("youtube.com");      // You are in "facebook.com". Visit "youtube.com"
browserHistory.back(1);                   // You are in "youtube.com", move back to "facebook.com" return "facebook.com"
browserHistory.back(1);                   // You are in "facebook.com", move back to "google.com" return "google.com"
browserHistory.forward(1);                // You are in "google.com", move forward to "facebook.com" return "facebook.com"
browserHistory.visit("linkedin.com");     // You are in "facebook.com". Visit "linkedin.com"
browserHistory.forward(2);                // You are in "linkedin.com", you cannot move forward any steps.
browserHistory.back(2);                   // You are in "linkedin.com", move back two steps to "facebook.com" then to "google.com". return "google.com"
browserHistory.back(7);                   // You are in "google.com", you can move back only one step to "leetcode.com". return "leetcode.com"
"""


browserHistory = BrowserHistory("leetcode.com")
browserHistory.visit("google.com")  # You are in "leetcode.com". Visit "google.com"
browserHistory.visit("facebook.com")  # You are in "google.com". Visit "facebook.com"
browserHistory.visit("youtube.com")  # You are in "facebook.com". Visit "youtube.com"
print(browserHistory.back(1))  # You are in "youtube.com", move back to "facebook.com" return "facebook.com"
print(browserHistory.back(1))  # You are in "facebook.com", move back to "google.com" return "google.com"
print(browserHistory.forward(1))  # You are in "google.com", move forward to "facebook.com" return "facebook.com"
browserHistory.visit("linkedin.com")  # You are in "facebook.com". Visit "linkedin.com"
print(browserHistory.forward(2))  # You are in "linkedin.com", you cannot move forward any steps.
print(browserHistory.back(2))  # You are in "linkedin.com", move back two steps to "facebook.com" then to "google.com". return "google.com"
print(browserHistory.back(7))  # You are in "google.com", you can move back only one step to "leetcode.com". return "leetcode.com"


# O(1): visit, O(n): back, forward, O(n)
# linked list
class ListNode:
    def __init__(self, val="", next=None, prev=None):
        self.val = val
        self.next = next
        self.prev = prev

class BrowserHistory:
    def __init__(self, homepage: str):
        self.home = ListNode(homepage)
        self.node = self.home

    def visit(self, url: str) -> None:
        self.node.next = ListNode(url, None, self.node)
        self.node = self.node.next

    def back(self, steps: int) -> str:
        while (steps and 
               self.node.prev):
            self.node = self.node.prev
            steps -= 1
        
        return self.node.val

    def forward(self, steps: int) -> str:
        while (steps and 
               self.node.next):
            self.node = self.node.next
            steps -= 1
        
        return self.node.val


# Your BrowserHistory object will be instantiated and called as such:
# obj = BrowserHistory(homepage)
# obj.visit(url)
# param_2 = obj.back(steps)
# param_3 = obj.forward(steps)


# linked list
# first try
class ListNode:
    def __init__(self, val="", next=None, prev=None):
        self.val = val
        self.next = next
        self.prev = prev

class BrowserHistory:
    def __init__(self, homepage: str):
        self.home = ListNode(homepage)
        self.end = ListNode()
        self.home.next = self.end
        self.end.prev = self.home
        self.pointer = 0

    def visit(self, url: str) -> None:
        prev = self.home
        pointer = self.pointer
        
        while pointer > 0:
            prev = prev.next
            pointer -= 1

        self.node = ListNode(url, self.end, prev)
        prev.next = self.node
        self.end.prev = self.node
        self.pointer += 1

    def back(self, steps: int) -> str:
        node = self.home
        pointer = self.pointer
        backing = max(pointer - steps, 0)
        self.pointer = 0
        
        while backing != 0:
            node = node.next
            backing -= 1
            self.pointer += 1
        
        return node.val

    def forward(self, steps: int) -> str:
        node = self.home
        pointer = 0
        
        while pointer != self.pointer:
            node = node.next
            pointer += 1

        while steps and node.next.next:
            node = node.next
            steps -= 1
            self.pointer += 1
        
        return node.val


# O(1), O(n)
# stack
class BrowserHistory:
    def __init__(self, homepage: str):
        self.history = [homepage]  # stack
        self.index = 0  # current page index
        self.len = 1  # true length of the stack

    def visit(self, url: str) -> None:
        self.index += 1
        if len(self.history) == self.index:
            self.history.append(url)
        else:
            self.history[self.index] = url
        # soft delete values after current index by setting length
        self.len = self.index + 1

    def back(self, steps: int) -> str:
        self.index = max(self.index - steps, 0)
        return self.history[self.index]

    def forward(self, steps: int) -> str:
        self.index = min(self.index + steps, self.len - 1)
        return self.history[self.index]


# O(n): visit, O(1): back, forward, O(n)
# stack
class BrowserHistory:
    def __init__(self, homepage: str):
        self.history = [homepage]  # stack
        self.index = 0  # current page index

    def visit(self, url: str) -> None:
        self.index += 1
        # self.history = self.history[:self.index]
        while len(self.history) > self.index:
            self.history.pop()
        self.history.append(url)

    def back(self, steps: int) -> str:
        steps = min(steps, self.index)
        self.index -= steps
        # self.index = max(self.index - steps, 0)
        return self.history[self.index]

    def forward(self, steps: int) -> str:
        steps = min(steps, len(self.history) - 1 - self.index)
        self.index += steps
        # self.index = min(self.index + steps, len(self.history) - 1)
        return self.history[self.index]





# Evaluate Boolean Binary Tree
# https://leetcode.com/problems/evaluate-boolean-binary-tree/description/
"""
You are given the root of a full binary tree with the following properties:

Leaf nodes have either the value 0 or 1, where 0 represents False and 1 represents True.
Non-leaf nodes have either the value 2 or 3, where 2 represents the boolean OR and 3 represents the boolean AND.
The evaluation of a node is as follows:

If the node is a leaf node, the evaluation is the value of the node, i.e. True or False.
Otherwise, evaluate the node's two children and apply the boolean operation of its value with the children's evaluations.
Return the boolean result of evaluating the root node.

A full binary tree is a binary tree where each node has either 0 or 2 children.

A leaf node is a node that has zero children.

 

Example 1:
   __And______                         __And_
  /           \                       /      \
True         __Or_         =>       True     True     =>     True
            /     \       
         False    True    

Input: root = [2,1,3,null,null,0,1]
Output: true
Explanation: The above diagram illustrates the evaluation process.
The AND node evaluates to False AND True = False.
The OR node evaluates to True OR False = True.
The root node evaluates to True, so we return true.
Example 2:

Input: root = [0]
Output: false
Explanation: The root node is a leaf node and it evaluates to false, so we return false.
"""
print(Solution().evaluateTree(build_tree_from_list([2, 1, 3, None, None, 0, 1])), True)
print(Solution().evaluateTree(build_tree_from_list([0])), False)


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def evaluateTree(self, root: TreeNode) -> bool:
        if (not root or
                not root.val):
            return False
        elif root.val == 1:
            return True
        elif root.val == 2:
            return (self.evaluateTree(root.left) or
                    self.evaluateTree(root.right))
        else:  # root.val == 3:
            return (self.evaluateTree(root.left) and
                    self.evaluateTree(root.right))





# K Closest Points to Origin
# https://leetcode.com/problems/k-closest-points-to-origin/description/
"""
Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane and an integer k, return the k closest points to the origin (0, 0).

The distance between two points on the X-Y plane is the Euclidean distance (i.e., √(x1 - x2)2 + (y1 - y2)2).

You may return the answer in any order. The answer is guaranteed to be unique (except for the order that it is in).

 

Example 1:


Input: points = [[1,3],[-2,2]], k = 1
Output: [[-2,2]]
Explanation:
The distance between (1, 3) and the origin is sqrt(10).
The distance between (-2, 2) and the origin is sqrt(8).
Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
We only want the closest k = 1 points from the origin, so the answer is just [[-2,2]].
Example 2:

Input: points = [[3,3],[5,-1],[-2,4]], k = 2
Output: [[3,3],[-2,4]]
Explanation: The answer [[-2,4],[3,3]] would also be accepted.
"""
print(Solution().kClosest([[1, 3], [-2, 2]], 1), [[-2, 2]])
print(Solution().kClosest([[3, 3], [5, -1], [-2, 4]], 2), [[3, 3], [-2, 4]])


import numpy as np
import heapq

# O(klogn), O(n)
# heap, optimal
class Solution:
    def kClosest(self, points: list[list[int]], k: int) -> list[list[int]]:
        distances = []  # [-distance, [x, y]]
        k_points = [0] * k  # solution frame

        for (x, y) in points:
            # distance = np.sqrt(x**2 + y**2)  # calculate distance to origin
            distance = -(x**2 + y**2)  # calculate distance to origin
            if len(distances) < k:
                heapq.heappush(distances, (distance, (x, y)))  # push to heap: [distance, [x, y]]
            else:
                heapq.heappushpop(distances, (distance, (x, y)))  # push then pop largest element
        for index in range(k):  # loop k times
            k_points[index] = heapq.heappop(distances)[1]  # update k closest points

        return k_points


import numpy as np
import heapq

# O(klogn), O(n)
# heap
class Solution:
    def kClosest(self, points: list[list[int]], k: int) -> list[list[int]]:
        distances = []  # [distance, [x, y]]
        k_points = [0] * k  # solution frame

        for (x, y) in points:
            distance = np.sqrt(x**2 + y**2)  # calculate distance to origin
            heapq.heappush(distances, (distance, (x, y)))  # push to heap: [distance, [x, y]]
        for index in range(k):  # loop k times
            k_points[index] = heapq.heappop(distances)[1]  # update k closest points

        return k_points





# Matchsticks to Square
# https://leetcode.com/problems/matchsticks-to-square/description/
"""
You are given an integer array matchsticks where matchsticks[i] is the length of the ith matchstick. You want to use all the matchsticks to make one square. You should not break any stick, but you can link them up, and each matchstick must be used exactly one time.

Return true if you can make this square and false otherwise.

 

Example 1:


Input: matchsticks = [1,1,2,2,2]
Output: true
Explanation: You can form a square with length 2, one side of the square came two sticks with length 1.
Example 2:

Input: matchsticks = [3,3,3,3,4]
Output: false
Explanation: You cannot find a way to form a square with all the matchsticks.
"""
print(Solution().makesquare([1, 1, 2, 2, 2]), True)
print(Solution().makesquare([3, 3, 3, 3, 4]), False)
print(Solution().makesquare([5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3]), True)
print(Solution().makesquare([7215807,6967211,5551998,6632092,2802439,821366,2465584,9415257,8663937,3976802,2850841,803069,2294462,8242205,9922998]), False)


# O(4^n), O(n)
# backtracking
class Solution:
    def makesquare(self, matchsticks: list[int]) -> bool:
        if sum(matchsticks) % 4:
            return False

        matchsticks.sort(reverse=True)
        max_side_len = sum(matchsticks) // 4
        side_len = [0] * 4  # each side length cache

        def dfs(index):
            if index == len(matchsticks):
                return True

            for side in range(4):  # check every square side
                if side_len[side] + matchsticks[index] <= max_side_len:  # if side length in bounds
                    side_len[side] += matchsticks[index]
                    
                    if dfs(index + 1):  # check next match
                        return True
                    
                    side_len[side] -= matchsticks[index]  # backtrack
            
            return False

        return dfs(0)





# Extra Characters in a String
# https://leetcode.com/problems/extra-characters-in-a-string/description/
"""
You are given a 0-indexed string s and a dictionary of words dictionary. You have to break s into one or more non-overlapping substrings such that each substring is present in dictionary. There may be some extra characters in s which are not present in any of the substrings.

Return the minimum number of extra characters left over if you break up s optimally.

 

Example 1:

Input: s = "leetscode", dictionary = ["leet","code","leetcode"]
Output: 1
Explanation: We can break s in two substrings: "leet" from index 0 to 3 and "code" from index 5 to 8. There is only 1 unused character (at index 4), so we return 1.

Example 2:

Input: s = "sayhelloworld", dictionary = ["hello","world"]
Output: 3
Explanation: We can break s in two substrings: "hello" from index 3 to 7 and "world" from index 8 to 12. The characters at indices 0, 1, 2 are not used in any substring and thus are considered as extra characters. Hence, we return 3.
"""
print(Solution().minExtraChar("leetcode", ["leet", "code", "leetcode"]), 0)
print(Solution().minExtraChar("leetscode", ["leet", "code", "leetcode"]), 1)
print(Solution().minExtraChar("sayhelloworld", ["hello", "world"]), 3)


# draft
# "leets43210"


# O(n3), O(n)
# dp, bottom-up
class Solution:
    def minExtraChar(self, text: str, word_list: list[str]) -> int:
        words = set(word_list)
        text_len = len(text)
        cache = [0] * (text_len + 1)

        for index in reversed(range(text_len)):  # check all indexes
            cache[index] = cache[index + 1] + 1  # set default cache as if word is not found
            
            for word in words:  # check every word
                if (index + len(word) <= text_len and  # check if end of the word is in bounds
                    text[index: index + len(word)] in words):  # if word in text
                        cache[index] = min(cache[index], cache[index + len(word)])  # update cache
                       
                        if not cache[index]:  # early exit, if cache is 0 than this is the best case
                            break
        
        return cache[0]


# O(n3), O(n)
# dp, bottom-up, optimal
class Solution:
    def minExtraChar(self, text: str, word_list: list[str]) -> int:
        words = set(word_list)
        text_len = len(text)
        cache = [0] * (text_len + 1)

        # make word lengths to later iterate in word lengths not in words
        # don't check the same word length twice
        # when legit length is found still check in words if the word is legit
        word_lengths = {len(word) for word in word_list}

        for index in reversed(range(text_len)):  # check all indexes
            cache[index] = cache[index + 1] + 1  # set default cache as if word is not found
            
            for word_length in word_lengths:  # check every word length
                if (index + word_length <= text_len and  # check if end of the word is in bounds
                    text[index: index + word_length] in words):  # if word in text
                        cache[index] = min(cache[index], cache[index + word_length])  # update cache
                       
                        if not cache[index]:  # early exit, if cache is 0 than this is the best case
                            break
        
        return cache[0]


# O(n3), O(n2)
# dp, bottom-up, trie, optimal

class TrieNode:
    def __init__(self):
        self.letters = {}
        self.is_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root

        for letter in word:
            if letter not in node.letters:
                node.letters[letter] = TrieNode()
            
            node = node.letters[letter]

        node.is_word = True

    def search(self, word):
        node = self.root

        for letter in word:
            if letter not in node.letters:
                return False
            
            node = node.letters[letter]

        return node.is_word


class Solution:
    def minExtraChar(self, text: str, word_list: list[str]) -> int:
        # initialize trie
        trie = Trie()
        for word in word_list:
            trie.insert(word)
            
        text_len = len(text)
        cache = [0] * (text_len + 1)

        # make word lengths to later iterate in word lengths not in words
        # don't check the same word length twice
        # when legit length is found still check in words if the word is legit
        word_lengths = {len(word) for word in word_list}

        for index in reversed(range(text_len)):  # check all indexes
            cache[index] = cache[index + 1] + 1  # set default cache as if word is not found
            
            for word_length in word_lengths:  # check every word length
                if (index + word_length <= text_len and  # check if end of the word is in bounds
                    trie.search(text[index: index + word_length])):  # if word in trie
                        cache[index] = min(cache[index], cache[index + word_length])  # update cache
                       
                        if not cache[index]:  # early exit, if cache is 0 than this is the best case
                            break
        
        return cache[0]





# Delete and Earn
# https://leetcode.com/problems/delete-and-earn/description/
"""
You are given an integer array nums. You want to maximize the number of points you get by performing the following operation any number of times:

Pick any nums[i] and delete it to earn nums[i] points. Afterwards, you must delete every element equal to nums[i] - 1 and every element equal to nums[i] + 1.
Return the maximum number of points you can earn by applying the above operation some number of times.

 

Example 1:

Input: nums = [3,4,2]
Output: 6
Explanation: You can perform the following operations:
- Delete 4 to earn 4 points. Consequently, 3 is also deleted. nums = [2].
- Delete 2 to earn 2 points. nums = [].
You earn a total of 6 points.
Example 2:

Input: nums = [2,2,3,3,3,4]
Output: 9
Explanation: You can perform the following operations:
- Delete a 3 to earn 3 points. All 2's and 4's are also deleted. nums = [3,3].
- Delete a 3 again to earn 3 points. nums = [3].
- Delete a 3 once more to earn 3 points. nums = [].
You earn a total of 9 points.
"""
print(Solution().deleteAndEarn([1]), 1)
print(Solution().deleteAndEarn([2, 3]), 3)
print(Solution().deleteAndEarn([2, 4]), 6)
print(Solution().deleteAndEarn([3, 4, 2]), 6)
print(Solution().deleteAndEarn([2, 2, 3, 3, 3, 4]), 9)
print(Solution().deleteAndEarn([8, 10, 4, 9, 1, 3, 5, 9, 4, 10]), 37)
print(Solution().deleteAndEarn([1, 1, 1, 2, 4, 5, 5, 5, 6]), 18)
print(Solution().deleteAndEarn([1, 6, 3, 3, 8, 4, 8, 10, 1, 3]), 43)
print(Solution().deleteAndEarn([1, 1, 1]), 3)
print(Solution().deleteAndEarn([12,32,93,17,100,72,40,71,37,92,58,34,29,78,11,84,77,90,92,35,12,5,27,92,91,23,65,91,85,14,42,28,80,85,38,71,62,82,66,3,33,33,55,60,48,78,63,11,20,51,78,42,37,21,100,13,60,57,91,53,49,15,45,19,51,2,96,22,32,2,46,62,58,11,29,6,74,38,70,97,4,22,76,19,1,90,63,55,64,44,90,51,36,16,65,95,64,59,53,93]), 3451)


"""
draft
[2, 3, 4]
2; max(2, 3); max((2 + 4)+ or (3)+)
"""


class Solution:
    def deleteAndEarn(self, numbers: list[int]) -> int:
        """
        O(n), O(n)
        dp, bottom-up
        """
        counter = {}
        for number in numbers:
            counter[number] = counter.get(number, 0) + 1

        sorted_numbers = sorted(set(numbers))
        cache = [0] * len(sorted_numbers)

        for index, number in enumerate(sorted_numbers):
            cache[index] = number * counter[number]  # calculate current value
            if index == 0:
                continue

            if sorted_numbers[index - 1] + 1 != number:  # if previous number was not less by one from current number
                cache[index] += cache[index - 1]  # add previous cache
            else:
                if index != 1:  # exclude the first index
                    cache[index] += cache[index - 2]  # add `before previous` cache
                cache[index] = max(cache[index], cache[index - 1])  # previous number is one less to current number, but cache[index - 1] (previous cache) is better

        return cache[-1]


class Solution:
    def deleteAndEarn(self, numbers: list[int]) -> int:
        """
        O(n), O(n)
        dp, bottom-up
        cache optimized
        """
        counter = {}
        for number in numbers:
            counter[number] = counter.get(number, 0) + 1

        sorted_numbers = sorted(set(numbers))
        cache = [0, 0]

        for index, number in enumerate(sorted_numbers):
            current_cache = number * counter[number]
            if index == 0:
                cache = [0, current_cache]
                continue

            if sorted_numbers[index - 1] + 1 != number:
                current_cache += cache[1]
            else:
                if index != 1:
                    current_cache += cache[0]
                current_cache = max(current_cache, cache[1])

            cache = [cache[1], current_cache]

        return cache[-1]


class Solution:
    def deleteAndEarn(self, numbers: list[int]) -> int:
        """
        O(n), O(n)
        dp, top-down with memoization as list
        """
        counter = {}
        for number in numbers:
            counter[number] = counter.get(number, 0) + 1

        sorted_numbers = sorted(set(numbers))
        memo = [None] * (len(sorted_numbers) + 2)
        memo[-1] = memo[-2] = 0

        def dfs(index: int) -> int:
            if memo[index] is not None:
                return memo[index]
            
            number = sorted_numbers[index]
            val = number * counter[number]

            if (index + 1 < len(sorted_numbers) and 
                    sorted_numbers[index] + 1 != sorted_numbers[index + 1]):
                val += dfs(index + 1)
            else:
                val += dfs(index + 2)
            
            memo[index] = max(val, dfs(index + 1))
            return memo[index]

        return dfs(0)


class Solution:
    def deleteAndEarn(self, numbers: list[int]) -> int:
        """
        O(n), O(n)
        dp, top-down with memoization as hash map
        """
        counter = {}
        for number in numbers:
            counter[number] = counter.get(number, 0) + 1

        sorted_numbers = sorted(set(numbers))
        memo = {
            len(sorted_numbers): 0, 
            len(sorted_numbers) + 1: 0
        }

        def dfs(index: int) -> int:
            if index in memo:
                return memo[index]
            
            number = sorted_numbers[index]
            val = number * counter[number]

            if (index + 1 < len(sorted_numbers) and 
                    sorted_numbers[index] + 1 != sorted_numbers[index + 1]):
                val += dfs(index + 1)
            else:
                val += dfs(index + 2)
            
            memo[index] = max(val, dfs(index + 1))
            return memo[index]

        return dfs(0)


class Solution:
    def deleteAndEarn(self, numbers: list[int]) -> int:
        """
        O(2^n), O(n)
        brute force, tle
        """
        counter = {}
        for number in numbers:
            counter[number] = counter.get(number, 0) + 1

        sorted_numbers = sorted(set(numbers))

        def dfs(index: int) -> int:
            if index >= len(sorted_numbers):
                return 0
            
            number = sorted_numbers[index]
            val = number * counter[number]

            if (index + 1 < len(sorted_numbers) and 
                    sorted_numbers[index] + 1 != sorted_numbers[index + 1]):
                val += dfs(index + 1)
            else:
                val += dfs(index + 2)
            
            return max(val, dfs(index + 1))

        return dfs(0)





# Paint House
# https://leetcode.com/problems/paint-house/
# https://leetcode.ca/all/256.html
"""
There are a row of n houses, each house can be painted with one of the three colors: red, blue or green. The cost of painting each house with a certain color is different. You have to paint all the houses such that no two adjacent houses have the same color, and you need to cost the least. Return the minimum cost.

The cost of painting each house with a certain color is represented by a n x 3 cost matrix. For example, costs[0][0] is the cost of painting house 0 with color red; costs[1][2] is the cost of painting house 1 with color green, and so on... Find the minimum cost to paint all houses.
"""
print(Solution().minCost([[1, 2, 3]]), 1)
print(Solution().minCost([[1, 2, 3], [1, 4, 6]]), 3)
print(Solution().minCost([[17, 2, 17], [16, 16, 5], [14, 3, 19]]), 10)


"""
draft
                                .
                   /            |           \
                  17            2           17
              /       \     /      \      /     \
            16        5    16      5    16      16
          /    \    /  \   /  \   / \  /  \    /   \
        3      19  14  3  3   19 14  3  3  19 14   19
"""


class Solution:
    def minCost(self, houses: list[list[int]]) -> int:
        """
        O(n), O(1)
        dp, bottom-up
        in-place
        """
        for index, house in enumerate(houses[1:], 1):
            prev_house = houses[index - 1]
            houses[index] = [
                house[0] + min(prev_house[1], prev_house[2]),
                house[1] + min(prev_house[2], prev_house[0]),
                house[2] + min(prev_house[0], prev_house[1])
            ]

        return min(houses[-1])


class Solution:
    def minCost(self, houses: list[list[int]]) -> int:
        """
        O(n), O(1)
        dp, bottom-up
        """
        cache = tuple(houses[0])
        
        for house in houses[1:]:
            cache = (
                house[0] + min(cache[1], cache[2]),
                house[1] + min(cache[2], cache[0]),
                house[2] + min(cache[0], cache[1])
            )

        return min(cache)


class Solution:
    def minCost(self, houses: list[list[int]]) -> int:
        """
        O(n), O(n)
        dp, bottom-up
        """
        cache = [()] * len(houses)
        cache[0] = tuple(houses[0])
        
        for index, house in enumerate(houses[1:], 1):
            prev_house = cache[index - 1]
            cache[index] = (
                house[0] + min(prev_house[1], prev_house[2]),
                house[1] + min(prev_house[2], prev_house[0]),
                house[2] + min(prev_house[0], prev_house[1])
            )

        return min(cache[-1])


class Solution:
    def minCost(self, houses: list[list[int]]) -> int:
        """
        O(n), O(n)
        dp, top-down with memoization as hash map
        """
        memo = {len(houses): 0}
        
        def dfs(index: int, house_number: int) -> int:
            if index in memo:
                return memo[index]


            memo[index] = min(house + dfs(index + 1, house_ind) 
                              for house_ind, house in enumerate(houses[index])
                              if house_ind != house_number)
            return memo[index]

        return memo


class Solution:
    def minCost(self, houses: list[list[int]]) -> int:
        """
        O(2^n), O(n)
        brute force, tle
        """
        def dfs(index: int, house_number: int) -> int:
            if index == len(houses):
                return 0

            cost = float("inf")
            for house_ind, house in enumerate(houses[index]):
                if house_ind != house_number:
                    cost = min(cost, house + dfs(index + 1, house_ind))

            return cost

        return dfs(0, -1)


class Solution:
    def minCost(self, houses: list[list[int]]) -> int:
        """
        O(2^n), O(n)
        brute force, tle
        """
        def dfs(index: int, house_number: int) -> int:
            if index == len(houses):
                return 0

            return min(house + dfs(index + 1, house_ind) 
                       for house_ind, house in enumerate(houses[index])
                       if house_ind != house_number)

        return dfs(0, -1)





# Destination City
# https://leetcode.com/problems/destination-city/description/
"""
You are given the array paths, where paths[i] = [cityAi, cityBi] means there exists a direct path going from cityAi to cityBi. Return the destination city, that is, the city without any path outgoing to another city.

It is guaranteed that the graph of paths forms a line without any loop, therefore, there will be exactly one destination city.

 

Example 1:

Input: paths = [["London","New York"],["New York","Lima"],["Lima","Sao Paulo"]]
Output: "Sao Paulo" 
Explanation: Starting at "London" city you will reach "Sao Paulo" city which is the destination city. Your trip consist of: "London" -> "New York" -> "Lima" -> "Sao Paulo".
Example 2:

Input: paths = [["B","C"],["D","B"],["C","A"]]
Output: "A"
Explanation: All possible trips are: 
"D" -> "B" -> "C" -> "A". 
"B" -> "C" -> "A". 
"C" -> "A". 
"A". 
Clearly the destination city is "A".
Example 3:

Input: paths = [["A","Z"]]
Output: "Z"
"""
print(Solution().destCity([["London", "New York"], ["New York", "Lima"], ["Lima", "Sao Paulo"]]), "Sao Paulo")
print(Solution().destCity([["B", "C"], ["D", "B"], ["C", "A"]]), "A")
print(Solution().destCity([["A", "Z"]]), "Z")


# O(n), O(n)
# hash map
class Solution:
    def destCity(self, paths: list[list[str]]) -> str:
        city_map = {start: stop  # {starting city: destination city}
                    for start, stop in paths}

        city = paths[0][0]  # some starting city
        
        while city in city_map:  # while starting city have destination city
            city = city_map[city]
            
        return city


# O(n), O(n)
# hash set
class Solution:
    def destCity(self, paths: list[list[str]]) -> str:
        cities_a = {city for city, _ in paths}  # starting cities
        cities_b = {city for _, city in paths}  # destination cities

        for city in cities_b:  # for every destination city
            if city not in cities_a:  # if its not a starting city
                return city





# Maximum Product Difference Between Two Pairs
# https://leetcode.com/problems/maximum-product-difference-between-two-pairs/description/
"""
The product difference between two pairs (a, b) and (c, d) is defined as (a * b) - (c * d).

For example, the product difference between (5, 6) and (2, 7) is (5 * 6) - (2 * 7) = 16.
Given an integer array nums, choose four distinct indices w, x, y, and z such that the product difference between pairs (nums[w], nums[x]) and (nums[y], nums[z]) is maximized.

Return the maximum such product difference.

 

Example 1:

Input: nums = [5,6,2,7,4]
Output: 34
Explanation: We can choose indices 1 and 3 for the first pair (6, 7) and indices 2 and 4 for the second pair (2, 4).
The product difference is (6 * 7) - (2 * 4) = 34.
Example 2:

Input: nums = [4,2,5,9,7,4,8]
Output: 64
Explanation: We can choose indices 3 and 6 for the first pair (9, 8) and indices 1 and 5 for the second pair (2, 4).
The product difference is (9 * 8) - (2 * 4) = 64.
"""
print(Solution().maxProductDifference([5, 6, 2, 7, 4]), 34)
print(Solution().maxProductDifference([4, 2, 5, 9, 7, 4, 8]), 64)


import heapq

# O(n), O(1)
# heap
class Solution:
    def maxProductDifference(self, numbers: list[int]) -> int:
        heap_min = []
        heap_max = []

        for number in numbers:
            if len(heap_max) < 2:
                heapq.heappush(heap_max, number)
            else:
                heapq.heappushpop(heap_max, number)

            if len(heap_min) < 2:
                heapq.heappush(heap_min, -number)
            else:
                heapq.heappushpop(heap_min, -number)

        return (heap_max[0] * heap_max[1] - heap_min[0] * heap_min[1])


# draft
# 5, 6   4
# 4, 6,  5

# O(n), O(1)
class Solution:
    def maxProductDifference(self, numbers: list[int]) -> int:
        min1 = min2 = max(numbers)
        max1 = max2 = 0

        for number in numbers:
            if number < min1:
                min1, min2 = number, min1
            elif number < min2:
                min2 = number
            
            if number > max1:
                max1, max2 = number, max1
            elif number > max2:
                max2 = number
        
        return max1 * max2 - min1 * min2





# K-th Symbol in Grammar
# https://leetcode.com/problems/k-th-symbol-in-grammar/description/
"""
We build a table of n rows (1-indexed). We start by writing 0 in the 1st row. Now in every subsequent row, we look at the previous row and replace each occurrence of 0 with 01, and each occurrence of 1 with 10.

For example, for n = 3, the 1st row is 0, the 2nd row is 01, and the 3rd row is 0110.
Given two integer n and k, return the kth (1-indexed) symbol in the nth row of a table of n rows.

 

Example 1:

Input: n = 1, k = 1
Output: 0
Explanation: row 1: 0
Example 2:

Input: n = 2, k = 1
Output: 0
Explanation: 
row 1: 0
row 2: 01
Example 3:

Input: n = 2, k = 2
Output: 1
Explanation: 
row 1: 0
row 2: 01
"""
print(Solution().kthGrammar(1, 1), 0)
print(Solution().kthGrammar(2, 1), 0)
print(Solution().kthGrammar(2, 2), 1)
print(Solution().kthGrammar(30, 434991989), 0)


# draft
#         0          1 = 2**0
#        01          2 = 2**1
#       0110         4 = 2**2
#     01101001
# 0110100110010110


# O(n), O(1)
# two pointers, binary search
class Solution:
    def kthGrammar(self, n: int, k: int) -> int:
        left = 1
        right = 2 ** (n - 1)
        value = 0

        while left < right:
            middle = (left + right) // 2

            if k <= middle:
                right = middle
            else:
                left = middle + 1
                value = 0 if value else 1

        return value


# O(n2), O(n)
# mle
class Solution:
    def kthGrammar(self, number: int, k: int) -> int:
        row = [0]
        
        for _ in range(number - 1):
            for index in range(len(row)):
                if row[index]:
                    row.append(0)
                else:
                    row.append(1)
        
        return row[k - 1]





# Subarrays with K Different Integers
# https://leetcode.com/problems/subarrays-with-k-different-integers/description/
"""
Given an integer array nums and an integer k, return the number of good subarrays of nums.

A good array is an array where the number of different integers in that array is exactly k.

For example, [1,2,3,1,2] has 3 different integers: 1, 2, and 3.
A subarray is a contiguous part of an array.

 

Example 1:

Input: nums = [1,2,1,2,3], k = 2
Output: 7
Explanation: Subarrays formed with exactly 2 different integers: [1,2], [2,1], [1,2], [2,3], [1,2,1], [2,1,2], [1,2,1,2]
Example 2:

Input: nums = [1,2,1,3,4], k = 3
Output: 3
Explanation: Subarrays formed with exactly 3 different integers: [1,2,1,3], [2,1,3], [1,3,4].
"""
print(Solution().subarraysWithKDistinct([1, 2, 1, 2, 3], 2), 7)
print(Solution().subarraysWithKDistinct([1, 2, 1, 3, 4], 3), 3)


# draft
# [1, 2, 1, 2, 3]
# [1,2], 
# ([1,2,1], [2,1]), 
# ([1,2,1,2], [2,1,2], [1,2]), 
# [2,3]

# [1,2,1,3], [2,1,3], 
# [1,3,4]


# O(n), O(n)
# sliding window
class Solution:
    def subarraysWithKDistinct(self, numbers: list[int], k: int) -> int:
        left = 0
        middle = 0
        counter = {}  # {number: frequency}
        subarray_count = 0

        for right, number in enumerate(numbers):
            counter[number] = counter.get(number, 0) + 1

            # if more than k unique numbers in subarray
            while len(counter) > k:
                middle_number = numbers[middle]
                counter[middle_number] -= 1

                if not counter[middle_number]:  # if key has no value
                    counter.pop(middle_number)  # remove key
               
                middle += 1  # move pointer to next value
                left = middle  # if middle pointer moves, reset the left pointer

            # if there is more than one copy of middle number
            # than separate middle from left and move if right
            # while there's at least one copy of each unique number
            while counter[numbers[middle]] > 1:
                counter[numbers[middle]] -= 1
                middle += 1

            # if exactly k unique numbers in subarray
            if len(counter) == k:
                subarray_count += (middle - left + 1)
        
        return subarray_count


# O(n2), O(n)
# sliding window, tle
class Solution:
    def subarraysWithKDistinct(self, numbers: list[int], k: int) -> int:
        left = 0
        counter = {}  # {number: frequency}
        subarray_count = 0

        for right, number in enumerate(numbers):
            counter[number] = counter.get(number, 0) + 1

            while len(counter) > k:
               left_number = numbers[left]
               counter[left_number] -= 1

               if not counter[left_number]:
                   counter.pop(left_number)
               left += 1

            left_copy = left
            counter_copy = counter.copy()
            while len(counter) == k:
                subarray_count += 1
                left_number = numbers[left]
                counter[left_number] -= 1

                if not counter[left_number]:
                    counter.pop(left_number)
                left += 1

            left = left_copy
            counter = counter_copy.copy()
        
        return subarray_count


# O(n2), O(n)
# brute force
class Solution:
    def subarraysWithKDistinct(self, numbers: list[int], k: int) -> int:
        good_subarray_count = 0
        
        for i in range(len(numbers)):
            counter = {}
            
            for j in range(i, len(numbers)):
                counter[numbers[j]] = counter.get(numbers[j], 0) + 1
                if len(counter) == k:
                    good_subarray_count += 1
        
        return good_subarray_count





# Search in Rotated Sorted Array II
# https://leetcode.com/problems/search-in-rotated-sorted-array-ii/
"""
There is an integer array nums sorted in non-decreasing order (not necessarily with distinct values).

Before being passed to your function, nums is rotated at an unknown pivot index k (0 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,4,4,5,6,6,7] might be rotated at pivot index 5 and become [4,5,6,6,7,0,1,2,4,4].

Given the array nums after the rotation and an integer target, return true if target is in nums, or false if it is not in nums.

You must decrease the overall operation steps as much as possible.

 

Example 1:

Input: nums = [2,5,6,0,0,1,2], target = 0
Output: true
Example 2:

Input: nums = [2,5,6,0,0,1,2], target = 3
Output: false
"""


print(Solution().search([2, 5, 6, 0, 0, 1, 2], 0), True)
print(Solution().search([2, 5, 6, 0, 0, 1, 2], 3), False)
print(Solution().search([1], 0), False)
print(Solution().search([0, 1], 0), True)
print(Solution().search([1, 0], 0), True)
print(Solution().search([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1], 2), True)
print(Solution().search([1, 0, 1, 1, 1], 0), True)


# draft
# 3, 4, 0, 1, 2
# 4, 0, 1, 2, 3
# 0, 1, 2, 3, 4

# 2, 3, 4, 0, 1
# 1, 2, 3, 4, 0

# O(n), O(1)
# binary search
class Solution:
    def search(self, numbers: list[int], target: int) -> bool:
        left = 0
        right = len(numbers) - 1

        while left <= right:
            middle = (left + right) // 2

            if numbers[middle] == target:
                return True
            elif numbers[middle] < numbers[right]:  # the right side of portion is contiguous
                if numbers[middle] < target <= numbers[right]:  # if target value in the right portion
                    left = middle + 1
                else:
                    right = middle - 1
            elif numbers[middle] > numbers[right]:  # the left side of portion if contiguous
                if numbers[left] <= target < numbers[middle]:  # if target value in the left side portion
                    right = middle - 1
                else:
                    left = middle + 1
            else:  # no way to tell which side of portion is contiguous
                # Comparing middle number with right number, so need to
                # remove the right number. Removing left could remove the target.
                right -= 1

        return False





# Construct String from Binary Tree
# https://leetcode.com/problems/construct-string-from-binary-tree/description/
"""
Given the root node of a binary tree, your task is to create a string representation of the tree following a specific set of formatting rules. The representation should be based on a preorder traversal of the binary tree and must adhere to the following guidelines:

Node Representation: Each node in the tree should be represented by its integer value.

Parentheses for Children: If a node has at least one child (either left or right), its children should be represented inside parentheses. Specifically:

If a node has a left child, the value of the left child should be enclosed in parentheses immediately following the node's value.
If a node has a right child, the value of the right child should also be enclosed in parentheses. The parentheses for the right child should follow those of the left child.
Omitting Empty Parentheses: Any empty parentheses pairs (i.e., ()) should be omitted from the final string representation of the tree, with one specific exception: when a node has a right child but no left child. In such cases, you must include an empty pair of parentheses to indicate the absence of the left child. This ensures that the one-to-one mapping between the string representation and the original binary tree structure is maintained.

In summary, empty parentheses pairs should be omitted when a node has only a left child or no children. However, when a node has a right child but no left child, an empty pair of parentheses must precede the representation of the right child to reflect the tree's structure accurately.

 

Example 1:
    1
   / \
  2   3
 /
4

Input: root = [1,2,3,4]
Output: "1(2(4))(3)"
Explanation: Originally, it needs to be "1(2(4)())(3()())", but you need to omit all the empty parenthesis pairs. And it will be "1(2(4))(3)".
Example 2:
  __1
 /   \
2     3
 \
  4

Input: root = [1,2,3,null,4]
Output: "1(2()(4))(3)"
Explanation: Almost the same as the first example, except the () after 2 is necessary to indicate the absence of a left child for 2 and the presence of a right child.
"""
print(Solution().tree2str(build_tree_from_list([1, 2, 3, 4])), "1(2(4))(3)")
print(Solution().tree2str(build_tree_from_list([1, 2, 3, None, 4])), "1(2()(4))(3)")


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# O(n), O(n)
# binary tree
class Solution:
    def tree2str(self, root: TreeNode) -> str:
        text = []

        def dfs(node):
            if not node:
                return

            text.append(str(node.val))
            if node.left or node.right:
                text.append("(")
            dfs(node.left)
            if node.left or node.right:
                text.append(")")
            if node.right:
                text.append("(")
            dfs(node.right)
            if node.right:
                text.append(")")

        dfs(root)
        return "".join(text)


# O(n), O(n)
# binary tree
class Solution:
    def tree2str(self, root: TreeNode) -> str:
        text = []

        def preorder(node):
            if not node:
                return

            text.append("(")
            text.append(str(node.val))
            if not node.left and node.right:
                text.append("()")
            preorder(node.left)
            preorder(node.right)
            text.append(")")

        preorder(root)
        return "".join(text)[1:-1]





# Kth Largest Element in an Array
# https://leetcode.com/problems/kth-largest-element-in-an-array/description/
"""
Given an integer array nums and an integer k, return the kth largest element in the array.

Note that it is the kth largest element in the sorted order, not the kth distinct element.

Can you solve it without sorting?

 

Example 1:

Input: nums = [3,2,1,5,6,4], k = 2
Output: 5
Example 2:

Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
Output: 4
"""
print(Solution().findKthLargest([3, 2, 1, 5, 6, 4], 2), 5)
print(Solution().findKthLargest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4), 4)


import heapq

# O(nlogk), O(k)
# heap
class Solution:
    def findKthLargest(self, numbers: list[int], k: int) -> int:
        heap_min = []

        for number in numbers:
            if len(heap_min) < k:
                heapq.heappush(heap_min, number)
            else:
                heapq.heappushpop(heap_min, number)

        return heapq.heappop(heap_min)


# O(n + klogn), O(1)
# heap
class Solution:
    def findKthLargest(self, numbers: list[int], k: int) -> int:
        for index in range(len(numbers)):
            numbers[index] = -numbers[index]
        
        heapq.heapify(numbers)

        for _ in range(k - 1):
            heapq.heappop(numbers)
        
        return -heapq.heappop(numbers)


# O(nlogk), O(n)
# heap
class Solution:
    def findKthLargest(self, numbers: list[int], k: int) -> int:
        return heapq.nlargest(k, numbers)[-1]


# O(nlogn), O(n)
# quick select, tle
class Solution:
    def findKthLargest(self, numbers: list[int], k: int) -> int:
        k = len(numbers) - k
        
        def quick_select(left, right):
            pivot = numbers[right]
            pivot_index = left

            for index in range(left, right):
                if numbers[index] < pivot:
                    numbers[index], numbers[pivot_index] = numbers[pivot_index], numbers[index]
                    pivot_index += 1
                
            numbers[pivot_index], numbers[right] = numbers[right], numbers[pivot_index]
            
            if k < pivot_index:
                return quick_select(left, pivot_index - 1)
            elif k > pivot_index:
                return quick_select(pivot_index + 1, right)
            else:
                return numbers[pivot_index]
        
        return quick_select(0, len(numbers) - 1)





# Splitting a String Into Descending Consecutive Values
# https://leetcode.com/problems/splitting-a-string-into-descending-consecutive-values/description/
"""
You are given a string s that consists of only digits.

Check if we can split s into two or more non-empty substrings such that the numerical values of the substrings are in descending order and the difference between numerical values of every two adjacent substrings is equal to 1.

For example, the string s = "0090089" can be split into ["0090", "089"] with numerical values [90,89]. The values are in descending order and adjacent values differ by 1, so this way is valid.
Another example, the string s = "001" can be split into ["0", "01"], ["00", "1"], or ["0", "0", "1"]. However all the ways are invalid because they have numerical values [0,1], [0,1], and [0,0,1] respectively, all of which are not in descending order.
Return true if it is possible to split s​​​​​​ as described above, or false otherwise.

A substring is a contiguous sequence of characters in a string.

 

Example 1:

Input: s = "1234"
Output: false
Explanation: There is no valid way to split s.
Example 2:

Input: s = "050043"
Output: true
Explanation: s can be split into ["05", "004", "3"] with numerical values [5,4,3].
The values are in descending order with adjacent values differing by 1.
Example 3:

Input: s = "9080701"
Output: false
Explanation: There is no valid way to split s.
"""
print(Solution().splitString("1"), False)
print(Solution().splitString("21"), True)
print(Solution().splitString("1234"), False)
print(Solution().splitString("050043"), True)
print(Solution().splitString("9080701"), False)
print(Solution().splitString("0090089"), True)
print(Solution().splitString("001"), False)


# O(n^2*2^n), O(n)
# backtracking
class Solution:
    def splitString(self, text: str) -> bool:
        def dfs(index, prev, parts):
            if index == len(text):
                return parts > 1  # valid solution has at least two parts

            for right in range(index, len(text)):  # check all substrings
                value = int(text[index: right + 1])  # convert to int

                if (prev == float("inf") or  # first numeber or
                        value == prev - 1):  # lower than previous one by one
                    if dfs(right + 1, value, parts + 1):  # check current number
                        return True

            return False

        return dfs(0, float("inf"), 0)




# Island Perimeter
# https://leetcode.com/problems/island-perimeter/description/
"""
You are given row x col grid representing a map where grid[i][j] = 1 represents land and grid[i][j] = 0 represents water.

Grid cells are connected horizontally/vertically (not diagonally). The grid is completely surrounded by water, and there is exactly one island (i.e., one or more connected land cells).

The island doesn't have "lakes", meaning the water inside isn't connected to the water around the island. One cell is a square with side length 1. The grid is rectangular, width and height don't exceed 100. Determine the perimeter of the island.

 

Example 1:


Input: grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
Output: 16
Explanation: The perimeter is the 16 yellow stripes in the image above.
Example 2:

Input: grid = [[1]]
Output: 4
Example 3:

Input: grid = [[1,0]]
Output: 4
"""
print(Solution().islandPerimeter([[0, 1, 0, 0], [1, 1, 1, 0], [0, 1, 0, 0], [1, 1, 0, 0]]), 16)
print(Solution().islandPerimeter([[1]]), 4)
print(Solution().islandPerimeter([[1, 0]]), 4)


# O(n2), O(1)
# matrix
class Solution:
    def islandPerimeter(self, grid: list[list[int]]) -> int:
        def count_borders(row, col, border):
            # if adjecent tile out of bounds or a water tile, add a border
            if (row - 1 < 0 or grid[row - 1][col] == 0):
                border += 1
            if (row + 1 == rows or grid[row + 1][col] == 0):
                border += 1
            if (col - 1 < 0 or grid[row][col - 1] == 0):
                border += 1
            if (col + 1 == cols or grid[row][col + 1] == 0):
                border += 1
            return border

        border = 0
        rows = len(grid)
        cols = len(grid[0])
        
        for row in range(rows):
            for col in range(cols):
                if grid[row][col]:
                    border = count_borders(row, col, border)
        
        return border


# O(n2), O(1)
# matrix
class Solution:
    def __init__(self):
        self.border = 0
    def islandPerimeter(self, grid: list[list[int]]) -> int:
        def count_borders(row, col):
            self.border += 4
            
            # if adjecent tile in bounds and a land tile: subtract a border
            if (row - 1 >= 0 and grid[row - 1][col] == 1):
                self.border -= 1
            if (row + 1 < rows and grid[row + 1][col] == 1):
                self.border -= 1
            if (col - 1 >= 0 and grid[row][col - 1] == 1):
                self.border -= 1
            if (col + 1 < cols and grid[row][col + 1] == 1):
                self.border -= 1

        rows = len(grid)
        cols = len(grid[0])
        
        for row in range(rows):
            for col in range(cols):
                if grid[row][col]:
                    count_borders(row, col)
        
        return self.border


# O(n2), O(n2)
# matrix, dfs
class Solution:
    def islandPerimeter(self, grid: list[list[int]]) -> int:
        tabu = set()
        
        def dfs(row, col):
            if ((row, col) in tabu):
                return 0
            elif (row < 0 or
                row == rows or
                col < 0 or
                col == cols or
                    grid[row][col] == 0):
                return 1
            
            tabu.add((row, col))
            return (
                dfs(row - 1, col) + 
                dfs(row + 1, col) + 
                dfs(row, col - 1) + 
                dfs(row, col + 1)
            )

        rows = len(grid)
        cols = len(grid[0])
        
        for row in range(rows):
            for col in range(cols):
                if grid[row][col]:
                    return dfs(row, col)





# Verifying an Alien Dictionary
# https://leetcode.com/problems/verifying-an-alien-dictionary/description/
"""
In an alien language, surprisingly, they also use English lowercase letters, but possibly in a different order. The order of the alphabet is some permutation of lowercase letters.

Given a sequence of words written in the alien language, and the order of the alphabet, return true if and only if the given words are sorted lexicographically in this alien language.

 

Example 1:

Input: words = ["hello","leetcode"], order = "hlabcdefgijkmnopqrstuvwxyz"
Output: true
Explanation: As 'h' comes before 'l' in this language, then the sequence is sorted.
Example 2:

Input: words = ["word","world","row"], order = "worldabcefghijkmnpqstuvxyz"
Output: false
Explanation: As 'd' comes after 'l' in this language, then words[0] > words[1], hence the sequence is unsorted.
Example 3:

Input: words = ["apple","app"], order = "abcdefghijklmnopqrstuvwxyz"
Output: false
Explanation: The first three characters "app" match, and the second string is shorter (in size.) According to lexicographical rules "apple" > "app", because 'l' > '∅', where '∅' is defined as the blank character which is less than any other character (More info).
"""
print(Solution().isAlienSorted(["hello", "leetcode"], "hlabcdefgijkmnopqrstuvwxyz"), True)
print(Solution().isAlienSorted(["word", "world", "row"], "worldabcefghijkmnpqstuvxyz"), False)
print(Solution().isAlienSorted(["apple", "app"], "abcdefghijklmnopqrstuvwxyz"), False)
print(Solution().isAlienSorted(["ubg", "kwh"], "qcipyamwvdjtesbghlorufnkzx"), True)



# O(n2), O(1)
# hash map
class Solution:
    def isAlienSorted(self, words: list[str], order: str) -> bool:
        order_of_letters = {letter: index  # {letter: index}
                            for index, letter 
                            in enumerate(order)}

        for word_index in range(len(words) - 1):  # for every two words
            for letter_index in range(len(words[word_index])):  # for every letter first word
                if letter_index == len(words[word_index + 1]):  # first word is seconds word prefix
                    return False

                left_letter = words[word_index][letter_index]  # left word letter
                right_letter = words[word_index + 1][letter_index]  # right word letter
                
                if order_of_letters[left_letter] < order_of_letters[right_letter]:  # if left letter is lower, check next pair
                    break
                elif order_of_letters[left_letter] > order_of_letters[right_letter]:  # if right letter is higher, then wrong sort
                    return False
                
        return True





# Find the Town Judge
# https://leetcode.com/problems/find-the-town-judge/description/
"""
In a town, there are n people labeled from 1 to n. There is a rumor that one of these people is secretly the town judge.

If the town judge exists, then:

The town judge trusts nobody.
Everybody (except for the town judge) trusts the town judge.
There is exactly one person that satisfies properties 1 and 2.
You are given an array trust where trust[i] = [ai, bi] representing that the person labeled ai trusts the person labeled bi. If a trust relationship does not exist in trust array, then such a trust relationship does not exist.

Return the label of the town judge if the town judge exists and can be identified, or return -1 otherwise.

 

Example 1:

Input: n = 2, trust = [[1,2]]
Output: 2
Example 2:

Input: n = 3, trust = [[1,3],[2,3]]
Output: 3
Example 3:

Input: n = 3, trust = [[1,3],[2,3],[3,1]]
Output: -1
"""
print(Solution().findJudge(2, [[1, 2]]), 2)
print(Solution().findJudge(3, [[1, 3], [2, 3]]), 3)
print(Solution().findJudge(3, [[1, 3], [2, 3], [3, 1]]), -1)
print(Solution().findJudge(3, [[1, 2], [2, 3]]), -1)
print(Solution().findJudge(1, []), 1)


# O(n), O(n)
# hash map
class Solution:
    def findJudge(self, n: int, trust_list: list[list[int]]) -> int:
        if not trust_list and n == 1:  # one person and no votes
            return 1

        if (not trust_list or  # if no list or
                len(trust_list) < n - 1):  # not enough votes to identify the judge
            return -1

        a_to_b = {}  # {a: [b]}  person a to person b map
        b_to_a = {}  # {b: [a]}  person b to person a map

        for a, b in trust_list:  # traverse trust list
            if a not in a_to_b:
                a_to_b[a] = []
            if b not in b_to_a:
                b_to_a[b] = []

            a_to_b[a].append(b)
            b_to_a[b].append(a)

        for key, val in b_to_a.items():
            if (len(val) == (n - 1) and  # enough votes for the judge and
                    key not in a_to_b):  # judge didn't vote
                return key

        return -1





# Perfect Squares
# https://leetcode.com/problems/perfect-squares/description/
"""
Given an integer n, return the least number of perfect square numbers that sum to n.

A perfect square is an integer that is the square of an integer; in other words, it is the product of some integer with itself. For example, 1, 4, 9, and 16 are perfect squares while 3 and 11 are not.

 

Example 1:

Input: n = 12
Output: 3
Explanation: 12 = 4 + 4 + 4.
Example 2:

Input: n = 13
Output: 2
Explanation: 13 = 4 + 9.
"""
print(Solution().numSquares(1), 1)  # 1
print(Solution().numSquares(9), 1)  # 9
print(Solution().numSquares(16), 1)  # 1
print(Solution().numSquares(2), 2)  # 2 = 1 + 1
print(Solution().numSquares(5), 2)  # 5 = 4 + 1
print(Solution().numSquares(13), 2)  # 13 = 9 + 4
print(Solution().numSquares(12), 3)  # 12 = 4 + 4 + 4
print(Solution().numSquares(7), 4)  # 7 = 4 + 1 + 1 + 1
print(Solution().numSquares(28), 4)  # 28 = 16 + 9 + 1 + 1 + 1 or 28 = 25 + 1 + 1 + 1
print(Solution().numSquares(43), 3)


"""
draft
[1, 4, 9, 16, 25]
12 = 4 + 4 + 4 => 3
13 = 4 + 9 => 2
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
[   1  1  2  1  2  2  3  2  1           3]
[   1, 2,    4              9]
"""


def get_squares(n: int) -> set:
    """
    Get all squared numbers smaller than n.
    """
    squares = set()
    number = 0

    while True:
        number += 1
        if number**2 > n:
            break
        squares.add(number**2)
        
    return squares


class Solution:
    def numSquares(self, n: int) -> int:
        """
        O(2^n), O(n)
        brute force, tle
        O(n^sqrtn)
        """
        self.squares = get_squares(n)

        def dfs(n: int) -> int:
            if n == 0:
                return 0
            # elif n < 0:  # equals to `if n - number >= 0` in return
            #     return 2
            
            return min(1 + dfs(n - number)
                       for number in self.squares
                       if n - number >= 0)
        
        return dfs(n)


class Solution:
    def numSquares(self, n: int) -> int:
        """
        O(n2), O(n)
        dp, top-down with memoization as hash map
        O(nsqrtn)
        """
        self.squares = get_squares(n)
        memo = {0: 0}  # number of ways to get to the target (index)

        def dfs(n: int) -> int:
            if n in memo:
                return memo[n]
            elif n in self.squares:  # Explicitly handle perfect squares
                memo[n] = 1
                return 1
            # elif n < 0:  # equals to `if n - number >= 0` in return
            #     return 2
            
            memo[n] = min(1 + dfs(n - number)
                          for number in self.squares
                          if n - number >= 0)
            return memo[n]
        
        return dfs(n)


class Solution:
    def numSquares(self, n: int) -> int:
        """
        O(n2), O(n)
        dp, top-down with memoization as list
        O(nsqrtn)
        """
        self.squares = get_squares(n)
        memo = [None] * (n + 1)  # number of ways to get to the target (index)
        memo[0] = 0

        def dfs(n: int) -> int:
            if memo[n] is not None:
                return memo[n]
            elif n in self.squares:  # Explicitly handle perfect squares
                memo[n] = 1
                return 1
            # elif n < 0:  # equals to `if n - number >= 0` in return
            #     return 2
            
            memo[n] = min(1 + dfs(n - number)
                          for number in self.squares
                          if n - number >= 0)
            return memo[n]
        return dfs(n)


class Solution:
    def numSquares(self, n: int) -> int:
        """
        O(n2), O(n)
        dp, bottom-up
        O(nsqrtn)
        """
        self.squares = get_squares(n)
        cache = [n + 1] * (n + 1)  # number of ways to get to the target (index)
        cache[0] = 0

        for index in range(1, n + 1):
            if index in self.squares:  # early continue: if target (index) is in squares there is no need to calculate it
                cache[index] = 1
                continue

            for number in self.squares:
                if index - number >= 0:
                    cache[index] = min(cache[index], 
                                       cache[index - number] + 1)

        return cache[-1]





# Buy Two Chocolates
# https://leetcode.com/problems/buy-two-chocolates/description/
"""
You are given an integer array prices representing the prices of various chocolates in a store. You are also given a single integer money, which represents your initial amount of money.

You must buy exactly two chocolates in such a way that you still have some non-negative leftover money. You would like to minimize the sum of the prices of the two chocolates you buy.

Return the amount of money you will have leftover after buying the two chocolates. If there is no way for you to buy two chocolates without ending up in debt, return money. Note that the leftover must be non-negative.

 

Example 1:

Input: prices = [1,2,2], money = 3
Output: 0
Explanation: Purchase the chocolates priced at 1 and 2 units respectively. You will have 3 - 3 = 0 units of money afterwards. Thus, we return 0.
Example 2:

Input: prices = [3,2,3], money = 3
Output: 3
Explanation: You cannot buy 2 chocolates without going in debt, so we return 3.
"""
print(Solution().buyChoco([1, 2, 2], 3), 0)
print(Solution().buyChoco([3, 2, 3], 3), 3)


import heapq

# O(n), O(1)
# heap
class Solution:
    def buyChoco(self, prices: list[int], money: int) -> int:
        if len(prices) < 2:
            return money
        
        heapq.heapify(prices)
        chocolate_cost = heapq.heappop(prices) + heapq.heappop(prices)
        # chocolate_cost = sum(heapq.nsmallest(2, prices))

        if chocolate_cost <= money:
            return money - chocolate_cost
        else:
            return money


# O(n), O(1)
class Solution:
    def buyChoco(self, prices: list[int], money: int) -> int:
        if len(prices) < 2:
            return money
        
        min1 = min2 = max(prices)

        for price in prices:
            if price < min1:
                min1, min2 = price, min1
            elif price < min2:
                min2 = price

        if (min1 + min2) <= money:
            return money - (min1 + min2)
        else:
            return money





# Lemonade Change
# https://leetcode.com/problems/lemonade-change/description/
"""
At a lemonade stand, each lemonade costs $5. Customers are standing in a queue to buy from you and order one at a time (in the order specified by bills). Each customer will only buy one lemonade and pay with either a $5, $10, or $20 bill. You must provide the correct change to each customer so that the net transaction is that the customer pays $5.

Note that you do not have any change in hand at first.

Given an integer array bills where bills[i] is the bill the ith customer pays, return true if you can provide every customer with the correct change, or false otherwise.

 

Example 1:

Input: bills = [5,5,5,10,20]
Output: true
Explanation: 
From the first 3 customers, we collect three $5 bills in order.
From the fourth customer, we collect a $10 bill and give back a $5.
From the fifth customer, we give a $10 bill and a $5 bill.
Since all customers got correct change, we output true.
Example 2:

Input: bills = [5,5,10,10,20]
Output: false
Explanation: 
From the first two customers in order, we collect two $5 bills.
For the next two customers in order, we collect a $10 bill and give back a $5 bill.
For the last customer, we can not give the change of $15 back because we only have two $10 bills.
Since not every customer received the correct change, the answer is false.
"""
print(Solution().lemonadeChange([5, 5, 5, 10, 20]), True)
print(Solution().lemonadeChange([5, 5, 10, 10, 20]), False)
print(Solution().lemonadeChange([5,5,10,20,5,5,5,5,5,5,5,5,5,10,5,5,20,5,20,5]), True)
print(Solution().lemonadeChange([5,5,5,10,5,5,10,20,20,20]), False)


# O(n), O(n)
# hash map
class Solution:
    def lemonadeChange(self, bills: list[int]) -> bool:
        cash = {5: 0, 10: 0, 20: 0}

        for bill in bills:
            if bill == 5:
                cash[5] += 1
            elif bill == 10:
                cash[5] -= 1
                cash[10] += 1
            elif cash[10] > 0:
                cash[5] -= 1
                cash[10] -= 1
            else:
                cash[5] -= 3
                cash[20] += 1
            
            if (cash[5] < 0 or 
                    cash[10] < 0):
                return False
            
        return True





# Maximum Odd Binary Number
# https://leetcode.com/problems/maximum-odd-binary-number/description/
"""
You are given a binary string s that contains at least one '1'.

You have to rearrange the bits in such a way that the resulting binary number is the maximum odd binary number that can be created from this combination.

Return a string representing the maximum odd binary number that can be created from the given combination.

Note that the resulting string can have leading zeros.

 

Example 1:

Input: s = "010"
Output: "001"
Explanation: Because there is just one '1', it must be in the last position. So the answer is "001".
Example 2:

Input: s = "0101"
Output: "1001"
Explanation: One of the '1's must be in the last position. The maximum number that can be made with the remaining digits is "100". So the answer is "1001".
"""
print(Solution().maximumOddBinaryNumber("010"), "001")
print(Solution().maximumOddBinaryNumber("0101"), "1001")
print(Solution().maximumOddBinaryNumber("10"), "01")


# O(n), O(n)
class Solution:
    def maximumOddBinaryNumber(self, text: str) -> str:
        ones = zeros = 0
        
        for digit in text:
            if digit == "1":
                ones += 1
            else:
                zeros += 1

        return "1" * (ones - 1) + "0" * zeros + "1"


# O(n), O(n)
# quick sort
class Solution:
    def maximumOddBinaryNumber(self, text: str) -> str:
        text = [digit for digit in text]
        left = 0

        for index in range(len(text)):
            if text[index] == "1":
                text[index], text[left] = text[left], text[index]
                left += 1
            
        text[left - 1], text[len(text) - 1] = text[len(text) - 1], text[left - 1]
        return "".join(text)





# Maximum Nesting Depth of the Parentheses
# https://leetcode.com/problems/maximum-nesting-depth-of-the-parentheses/description/
"""
Given a valid parentheses string s, return the nesting depth of s. The nesting depth is the maximum number of nested parentheses.

 

Example 1:

Input: s = "(1+(2*3)+((8)/4))+1"

Output: 3

Explanation:

Digit 8 is inside of 3 nested parentheses in the string.

Example 2:

Input: s = "(1)+((2))+(((3)))"

Output: 3

Explanation:

Digit 3 is inside of 3 nested parentheses in the string.

Example 3:

Input: s = "()(())((()()))"

Output: 3
"""
print(Solution().maxDepth("(1+(2*3)+((8)/4))+1"), 3)
print(Solution().maxDepth("(1)+((2))+(((3)))"), 3)
print(Solution().maxDepth("()(())((()()))"), 3)


# O(n), O(1)
class Solution:
    def maxDepth(self, text: str) -> int:
        depth = 0
        opening = 0

        for char in text:
            if char == "(":
                opening += 1
            elif char == ")":
                opening -= 1
            
            depth = max(depth, opening)
        
        return depth





# Maximum Score After Splitting a String
# https://leetcode.com/problems/maximum-score-after-splitting-a-string/description/
"""
Given a string s of zeros and ones, return the maximum score after splitting the string into two non-empty substrings (i.e. left substring and right substring).

The score after splitting a string is the number of zeros in the left substring plus the number of ones in the right substring.

 

Example 1:

Input: s = "011101"
Output: 5 
Explanation: 
All possible ways of splitting s into two non-empty substrings are:
left = "0" and right = "11101", score = 1 + 4 = 5 
left = "01" and right = "1101", score = 1 + 3 = 4 
left = "011" and right = "101", score = 1 + 2 = 3 
left = "0111" and right = "01", score = 1 + 1 = 2 
left = "01110" and right = "1", score = 2 + 1 = 3
Example 2:

Input: s = "00111"
Output: 5
Explanation: When left = "00" and right = "111", we get the maximum score = 2 + 3 = 5
Example 3:

Input: s = "1111"
Output: 3
"""
print(Solution().maxScore("011101"), 5)
print(Solution().maxScore("00111"), 5)
print(Solution().maxScore("1111"), 3)
print(Solution().maxScore("00"), 1)


# O(n), O(1)
class Solution:
    def maxScore(self, text: str) -> int:
        left_score = 0
        right_score = text.count("1")
        max_score = 0
        
        for index in range(len(text) - 1):
            if text[index] == "0":
                left_score += 1
            else:
                right_score -= 1
        
            max_score = max(max_score, left_score + right_score)
        
        return max_score





# Path Crossing
# https://leetcode.com/problems/path-crossing/description/
"""
Given a string path, where path[i] = 'N', 'S', 'E' or 'W', each representing moving one unit north, south, east, or west, respectively. You start at the origin (0, 0) on a 2D plane and walk on the path specified by path.

Return true if the path crosses itself at any point, that is, if at any time you are on a location you have previously visited. Return false otherwise.

 

Example 1:


Input: path = "NES"
Output: false 
Explanation: Notice that the path doesn't cross any point more than once.
Example 2:


Input: path = "NESWW"
Output: true
Explanation: Notice that the path visits the origin twice.
"""
print(Solution().isPathCrossing("NES"), False)
print(Solution().isPathCrossing("NESWW"), True)
print(Solution().isPathCrossing("WNSN"), True)


class Solution:
    def isPathCrossing(self, path: str) -> bool:
        stops = {(0, 0)}  # {(x, y)}
        prev_stop = (0, 0)

        directions = {
            "E": (1, 0),
            "W": (-1, 0),
            "N": (0, 1),
            "S": (0, -1)
        }

        for direction in path:
            next_stop = (prev_stop[0] + directions[direction][0],
                         prev_stop[1] + directions[direction][1])

            if next_stop in stops:
                return True
            else:
                stops.add(next_stop)
                prev_stop = next_stop

        return False





# Minimum Time to Make Rope Colorful
# https://leetcode.com/problems/minimum-time-to-make-rope-colorful/description/
"""
Alice has n balloons arranged on a rope. You are given a 0-indexed string colors where colors[i] is the color of the ith balloon.

Alice wants the rope to be colorful. She does not want two consecutive balloons to be of the same color, so she asks Bob for help. Bob can remove some balloons from the rope to make it colorful. You are given a 0-indexed integer array neededTime where neededTime[i] is the time (in seconds) that Bob needs to remove the ith balloon from the rope.

Return the minimum time Bob needs to make the rope colorful.

 

Example 1:


Input: colors = "abaac", neededTime = [1,2,3,4,5]
Output: 3
Explanation: In the above image, 'a' is blue, 'b' is red, and 'c' is green.
Bob can remove the blue balloon at index 2. This takes 3 seconds.
There are no longer two consecutive balloons of the same color. Total time = 3.
Example 2:


Input: colors = "abc", neededTime = [1,2,3]
Output: 0
Explanation: The rope is already colorful. Bob does not need to remove any balloons from the rope.
Example 3:


Input: colors = "aabaa", neededTime = [1,2,3,4,1]
Output: 2
Explanation: Bob will remove the balloons at indices 0 and 4. Each balloons takes 1 second to remove.
There are no longer two consecutive balloons of the same color. Total time = 1 + 1 = 2.
"""
print(Solution().minCost("abaac", [1, 2, 3, 4, 5]), 3)
print(Solution().minCost("abc", [1, 2, 3]), 0)
print(Solution().minCost("aabaa", [1, 2, 3, 4, 1]), 2)


class Solution:
    def minCost(self, colors: str, neededTime: list[int]) -> int:
        left = 0  # left pointer
        minimum_time = 0  # minnium time to remove consecutive same color baloons

        for right in range(1, len(colors)):
            if colors[left] != colors[right]:  # different color
                left = right  # move left pointer
            else:
                if neededTime[left] <= neededTime[right]:  # left time is less (equal) than right time
                    minimum_time += neededTime[left]  # add time needed to remove the left baloon
                    left = right  # move left pointer
                else:
                    minimum_time += neededTime[right]  # add time needed to remove the right baloon

        return minimum_time





# Find First and Last Position of Element in Sorted Array
# https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/
"""
Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value.

If target is not found in the array, return [-1, -1].

You must write an algorithm with O(log n) runtime complexity.

 

Example 1:

Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
Example 2:

Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]
Example 3:

Input: nums = [], target = 0
Output: [-1,-1]
"""
print(Solution().searchRange([5, 7, 7, 8, 8, 10], 8), [3, 4])
print(Solution().searchRange([5, 7, 7, 8, 8, 10], 6), [-1, -1])
print(Solution().searchRange([], 0), [-1, -1])
print(Solution().searchRange([1], 1), [0, 0])


# O(logn), O(1)
# binary search
class Solution:
    def searchRange(self, numbers: list[int], target: int) -> list[int]:
        left = 0
        right = len(numbers) - 1
        has_target = False

        # find target starting position
        while left <= right:
            middle = (left + right) // 2
            middle_number = numbers[middle]

            if target > middle_number:
                left = middle + 1
            else:
                if target == middle_number:  # check if numbers has tagret numebr
                    has_target = True
                first_position = middle  # first_position == left, after exiting while loop
                right = middle - 1

        # no target in numbers
        if not has_target:
            return [-1, -1]

        left = 0
        right = len(numbers) - 1

        # find target ending position
        while left <= right:
            middle = (left + right) // 2
            middle_number = numbers[middle]

            if target < middle_number:
                right = middle - 1
            else:
                last_position = middle  # last_position == right, after exiting while loop
                left = middle + 1

        return [first_position, last_position]





# Find the Duplicate Number
# https://leetcode.com/problems/find-the-duplicate-number/description/
"""
Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.

There is only one repeated number in nums, return this repeated number.

You must solve the problem without modifying the array nums and using only constant extra space.

 

Example 1:

Input: nums = [1,3,4,2,2]
Output: 2
Example 2:

Input: nums = [3,1,3,4,2]
Output: 3
Example 3:

Input: nums = [3,3,3,3,3]
Output: 3
"""
print(Solution().findDuplicate([1, 3, 4, 2, 2]), 2)
print(Solution().findDuplicate([3, 1, 3, 4, 2]), 3)
print(Solution().findDuplicate([3, 3, 3, 3, 3]), 3)
print(Solution().findDuplicate([2, 5, 9, 6, 9, 3, 8, 9, 7, 1]), 9)


# 1 -> 3 –> 2 <-> 4
# 3 -> 4 -> 2 -> 3 -> 4
# O(n), O(1)
# linked list
class Solution:
    def findDuplicate(self, numbers: list[int]) -> int:
        slow = 0
        fast = 0

        # find the intersection
        while True:
            slow = numbers[slow]
            fast = numbers[numbers[fast]]
            if fast == slow:
                break
        
        # return [slow, numbers[slow]]  # [index, number]

        # find the beginning of the cycle
        # The distance from the intersection to the beginning of the cycle is the same as
        # from the beginning of the graph to the beginning of the cycle.
        slow2 = 0
        while True:
            slow = numbers[slow]
            slow2 = numbers[slow2]
            if slow == slow2:
                return slow





# Insert into a Binary Search Tree
# https://leetcode.com/problems/insert-into-a-binary-search-tree/description/
"""
You are given the root node of a binary search tree (BST) and a value to insert into the tree. Return the root node of the BST after the insertion. It is guaranteed that the new value does not exist in the original BST.

Notice that there may exist multiple valid ways for the insertion, as long as the tree remains a BST after insertion. You can return any of them.

 

Example 1:
    __4
   /   \
  2     7
 / \
1   3

    __4__
   /     \
  2       7
 / \     /
1   3   5

Input: root = [4,2,7,1,3], val = 5
Output: [4,2,7,1,3,5]
Explanation: Another accepted tree is:
    ____5
   /     \
  2       7
 / \
1   3
     \
      4

Example 2:
     ____40___
    /         \
  _20         _60
 /   \       /   \
10    30    50    70

     _______40___
    /            \
  _20___         _60
 /      \       /   \
10      _30    50    70
       /
      25

Input: root = [40,20,60,10,30,50,70], val = 25
Output: [40,20,60,10,30,50,70,null,null,25]

Example 3:
    __4
   /   \
  2     7
 / \
1   3

    __4__
   /     \
  2       7
 / \     /
1   3   5

Input: root = [4,2,7,1,3,null,null,null,null,null,null], val = 5
Output: [4,2,7,1,3,5]
"""
print(level_order_traversal(Solution().insertIntoBST(build_tree_from_list([4, 2, 7, 1, 3]), 5)), [4, 2, 7, 1, 3, 5])
print(level_order_traversal(Solution().insertIntoBST(build_tree_from_list([40, 20, 60, 10, 30, 50, 70]), 25)), [40, 20, 60, 10, 30, 50, 70, None, None, 25])
print(level_order_traversal(Solution().insertIntoBST(build_tree_from_list([4, 2, 7, 1, 3, None, None, None, None, None, None]), 5)), [4, 2, 7, 1, 3, 5])


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# O(n), O(n)
# binary search, recursion
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        def dfs(node):
            if not node:
                return TreeNode(val)
            elif val < node.val:
                node.left = dfs(node.left)
            else:
                node.right = dfs(node.right)
            
            return node
        return dfs(root)


from collections import deque

# O(n), O(1)
# binary search, iteration
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root:
            return TreeNode(val)
        
        queue = deque([root])

        while queue:
            node = queue.popleft()

            if val < node.val:
                if not node.left:
                    node.left = TreeNode(val)
                else:
                    queue.append(node.left)
            else:
                if not node.right:
                    node.right = TreeNode(val)
                else:
                    queue.append(node.right)
            
        return root





# Task Scheduler
# https://leetcode.com/problems/task-scheduler/description/
"""
You are given an array of CPU tasks, each labeled with a letter from A to Z, and a number n. Each CPU interval can be idle or allow the completion of one task. Tasks can be completed in any order, but there's a constraint: there has to be a gap of at least n intervals between two tasks with the same label.

Return the minimum number of CPU intervals required to complete all tasks.

 

Example 1:

Input: tasks = ["A","A","A","B","B","B"], n = 2

Output: 8

Explanation: A possible sequence is: A -> B -> idle -> A -> B -> idle -> A -> B.

After completing task A, you must wait two intervals before doing A again. The same applies to task B. In the 3rd interval, neither A nor B can be done, so you idle. By the 4th interval, you can do A again as 2 intervals have passed.

Example 2:

Input: tasks = ["A","C","A","B","D","B"], n = 1

Output: 6

Explanation: A possible sequence is: A -> B -> C -> D -> A -> B.

With a cooling interval of 1, you can repeat a task after just one other task.

Example 3:

Input: tasks = ["A","A","A", "B","B","B"], n = 3

Output: 10

Explanation: A possible sequence is: A -> B -> idle -> idle -> A -> B -> idle -> idle -> A -> B.

There are only two types of tasks, A and B, which need to be separated by 3 intervals. This leads to idling twice between repetitions of these tasks.
"""
print(Solution().leastInterval(["A", "A", "A", "B", "B", "B"], 2), 8)  # A -> B -> idle -> A -> B -> idle -> A -> B
print(Solution().leastInterval(["A", "C", "A", "B", "D", "B"], 1), 6)  # A -> B -> C -> D -> A -> B
print(Solution().leastInterval(["A", "A", "A", "B", "B", "B"], 3), 10)  # A -> B -> idle -> idle -> A -> B -> idle -> idle -> A -> B


import heapq
from collections import deque

# O(n), O(n)
# heap, deque
# O(nlogn) -> log26 => const ->  O(n)
class Solution:
    def leastInterval(self, tasks: list[str], idle: int) -> int:
        counter = {}
        for task in tasks:
            counter[task] = counter.get(task, 0) + 1

        task_heap = [-value for value in counter.values()]
        queue = deque()
        heapq.heapify(task_heap)

        time = 0
        while task_heap or queue:
            time += 1
            
            # take most frequent task from heap, decrease frequency by one
            # and append it to the queue
            if task_heap:  # []
                task = heapq.heappop(task_heap) + 1
                if task:
                    queue.append((time + idle, task))

            # if task is not idle pop it from the queue and
            # push it to the task heap
            if queue and queue[0][0] == time:
                _, task = queue.popleft()
                heapq.heappush(task_heap, task)

        return time





# Design Twitter
# https://leetcode.com/problems/design-twitter/description/
"""
Design a simplified version of Twitter where users can post tweets, follow/unfollow another user, and is able to see the 10 most recent tweets in the user's news feed.

Implement the Twitter class:

Twitter() Initializes your twitter object.
void postTweet(int userId, int tweetId) Composes a new tweet with ID tweetId by the user userId. Each call to this function will be made with a unique tweetId.
List<Integer> getNewsFeed(int userId) Retrieves the 10 most recent tweet IDs in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user themself. Tweets must be ordered from most recent to least recent.
void follow(int followerId, int followeeId) The user with ID followerId started following the user with ID followeeId.
void unfollow(int followerId, int followeeId) The user with ID followerId started unfollowing the user with ID followeeId.
 

Example 1:

Input
["Twitter", "postTweet", "getNewsFeed", "follow", "postTweet", "getNewsFeed", "unfollow", "getNewsFeed"]
[[], [1, 5], [1], [1, 2], [2, 6], [1], [1, 2], [1]]
Output
[null, null, [5], null, null, [6, 5], null, [5]]

Explanation
Twitter twitter = new Twitter();
twitter.postTweet(1, 5); // User 1 posts a new tweet (id = 5).
twitter.getNewsFeed(1);  // User 1's news feed should return a list with 1 tweet id -> [5]. return [5]
twitter.follow(1, 2);    // User 1 follows user 2.
twitter.postTweet(2, 6); // User 2 posts a new tweet (id = 6).
twitter.getNewsFeed(1);  // User 1's news feed should return a list with 2 tweet ids -> [6, 5]. Tweet id 6 should precede tweet id 5 because it is posted after tweet id 5.
twitter.unfollow(1, 2);  // User 1 unfollows user 2.
twitter.getNewsFeed(1);  // User 1's news feed should return a list with 1 tweet id -> [5], since user 1 is no longer following user 2.
"""


import heapq
from collections import defaultdict


# O(1): postTweet, follow, unfollow, O(n): getNewsFeed, O(n)
# heap
# O(nlog10) => O(n)
class Twitter:

    def __init__(self):
        self.user_follows = defaultdict(set)  # {user: {followee_1, followee_2, ...}}
        self.user_tweets = defaultdict(list)  #  {user: [(time stamp, tweet id), ...]}
        self.time_stamp = 0

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.time_stamp += 1  # update time stamp with higher (newer) value
        self.user_tweets[userId].append((self.time_stamp, tweetId))  # save users tweet

    def getNewsFeed(self, userId: int) -> list[int]:
        def merge_feed(tweets: dict[list]) -> None:
            if news_feed and tweets[-1][0] < news_feed[0][0]:  # if the newest tweet from a user is older than the oldest tweet from the feed
                return
            for tweet in tweets[-10:]:  # Only consider the 10 most recent tweets
                if len(news_feed) < 10:
                    heapq.heappush(news_feed, tweet)
                else:
                    heapq.heappushpop(news_feed, tweet)

        news_feed = []  # [(time stamp, tweet id)], news feed with timestamps
        # Add user's own tweets
        if userId in self.user_tweets:  # if user has tweets
            merge_feed(self.user_tweets[userId])

        # Add followees' tweets
        if userId in self.user_follows:  # if user has followees
            for followee in self.user_follows[userId]:  # check every followee
                merge_feed(self.user_tweets[followee])
        
        clean_news_feed = []  # [tweet id_1, tweet id_2], tweets witwhout timestamps starting from the newest
        for _ in range(len(news_feed)):
            clean_news_feed.append(heapq.heappop(news_feed)[1])  # clean news feed from time stamps
        return (list(reversed(clean_news_feed)))


    def follow(self, followerId: int, followeeId: int) -> None:
        if followerId != followeeId:  # user should not follow itself
            self.user_follows[followerId].add(followeeId)


    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followerId in self.user_follows:  # if user has a following list (set)
            self.user_follows[followerId].discard(followeeId)


# Your Twitter object will be instantiated and called as such:
# obj = Twitter()
# obj.postTweet(userId,tweetId)
# param_2 = obj.getNewsFeed(userId)
# obj.follow(followerId,followeeId)
# obj.unfollow(followerId,followeeId)


def test_input(operations: list[str], arguments: list[list[int | None]]) -> list[list[int] | None]:
    """
    Test imput provided in two separate lists: operations and arguments
    """
    twitter = None
    result = []
    
    for operation, argument in zip(operations, arguments):
        if operation == "Twitter":
            twitter = Twitter()
            result.append(None)
        elif operation == "postTweet":
            result.append(twitter.postTweet(*argument))
        elif operation == "getNewsFeed":
            result.append(twitter.getNewsFeed(*argument))
        elif operation == "follow":
            result.append(twitter.follow(*argument))
        elif operation == "unfollow":
            result.append(twitter.unfollow(*argument))

    return result

# Example Input
operations = ["Twitter", "getNewsFeed"]
arguments = [[], [1]]
expected_output = [None, []]

operations = ["Twitter","postTweet","getNewsFeed","follow","postTweet","getNewsFeed","unfollow","getNewsFeed"]
arguments = [[],[1,5],[1],[1,2],[2,6],[1],[1,2],[1]]
expected_output = [None, None, [5], None, None, [6, 5], None, [5]]

operations = ["Twitter","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","postTweet","getNewsFeed","follow","getNewsFeed","unfollow","getNewsFeed"]
arguments = [[],[1,5],[2,3],[1,101],[2,13],[2,10],[1,2],[1,94],[2,505],[1,333],[2,22],[1,11],[1,205],[2,203],[1,201],[2,213],[1,200],[2,202],[1,204],[2,208],[2,233],[1,222],[2,211],[1],[1,2],[1],[1,2],[1]]
expected_output = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,[222,204,200,201,205,11,333,94,2,101],None,[211,222,233,208,204,202,200,213,201,203],None,[222,204,200,201,205,11,333,94,2,101]]


# Run tests
test_output = test_input(operations, arguments)
print(test_output == expected_output)
print(test_output)





twitter = Twitter()
print(twitter.postTweet(1, 5))  # User 1 posts a new tweet (id = 5).
print(twitter.getNewsFeed(1))  # User 1's news feed should return a list with 1 tweet id -> [5]. return [5]
print(twitter.follow(1, 2))  # User 1 follows user 2.
print(twitter.postTweet(2, 6))  # User 2 posts a new tweet (id = 6).
print(twitter.getNewsFeed(1))  # User 1's news feed should return a list with 2 tweet ids -> [6, 5]. Tweet id 6 should precede tweet id 5 because it is posted after tweet id 5.
print(twitter.unfollow(1, 2))  # User 1 unfollows user 2.
print(twitter.getNewsFeed(1))  # User 1's news feed should return a list with 1 tweet id -> [5], since user 1 is no longer following user 2.
print()
twitter = Twitter()
print(twitter.postTweet(1, 1))
print(twitter.getNewsFeed(1))  # [1]
print(twitter.follow(2, 1))
print(twitter.getNewsFeed(2))  # [1]
print(twitter.unfollow(2, 1))
print(twitter.getNewsFeed(2))  # []
print()
twitter = Twitter()
print(twitter.getNewsFeed(1))  # [0]
print()
twitter = Twitter()
print(twitter.follow(1, 5))
print(twitter.getNewsFeed(1))  # []
print()
twitter = Twitter()
print(twitter.postTweet(1, 5))
print(twitter.postTweet(1, 3))
print(twitter.getNewsFeed(1))  # [3, 5]




import heapq
from collections import defaultdict


class ListNode:
    def __init__(self, val=(0, 0), next=None):  # val=(timestamp, tweetId)
        self.val = val
        self.next = next

# O(1): postTweet, follow, unfollow, O(n): getNewsFeed, O(n)
# heap, linked list
# O(nlog10) => O(n)
class Twitter:
    def __init__(self):
        self.user_follows = defaultdict(set)  # {user: {followee_1, followee_2, ...}}
        self.user_tweets = {}  #  {user: ListNode((time stamp, tweet id), next node)}
        self.time_stamp = 0

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.time_stamp += 1  # update time stamp with higher (newer) value
        if userId not in self.user_tweets:
            self.user_tweets[userId] = ListNode()  # Dummy head node
        
        head = self.user_tweets[userId]
        head.next = ListNode((self.time_stamp, tweetId), head.next)  # Insert new tweet at the front
        

    def getNewsFeed(self, userId: int) -> list[int]:
        def merge_feed(user: int) -> None:
            if (user not in self.user_tweets or # User has no tweets
                    not self.user_tweets[user].next):
                return
            
            # if news_feed and tweets[-1][0] < news_feed[0][0]:  # if the newest tweet from a user is older than the oldest tweet from the feed
                # return
            node = self.user_tweets[user].next
            count = 0

            while (node and count < 10  # take at most 10 tweets
                   ):  #and node.val[0] < news_feed[0][0] take tweets that aren't older than the oldest tweet in news feed
                if len(news_feed) < 10:
                    heapq.heappush(news_feed, node.val)
                else:
                    heapq.heappushpop(news_feed, node.val)
                node = node.next
                count += 1

        news_feed = []  # [(time stamp, tweet id)]  Min-heap for top 10 tweets
        # Add user's own tweets
        merge_feed(userId)

        # Add followees' tweets
        if userId in self.user_follows:  # if user has followees
            for followee in self.user_follows[userId]:  # check every followee
                merge_feed(followee)
        
        clean_news_feed = []  # [tweet id_1, tweet id_2], tweets witwhout timestamps starting from the newest
        for _ in range(len(news_feed)):
            clean_news_feed.append(heapq.heappop(news_feed)[1])  # clean news feed from time stamps
        return clean_news_feed[::-1]


    def follow(self, followerId: int, followeeId: int) -> None:
        if followerId != followeeId:  # user should not follow itself
            self.user_follows[followerId].add(followeeId)


    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followerId in self.user_follows:  # if user has a following list (set)
            self.user_follows[followerId].discard(followeeId)





# Find Unique Binary String
# https://leetcode.com/problems/find-unique-binary-string/description/
"""
Given an array of strings nums containing n unique binary strings each of length n, return a binary string of length n that does not appear in nums. If there are multiple answers, you may return any of them.

 

Example 1:

Input: nums = ["01","10"]
Output: "11"
Explanation: "11" does not appear in nums. "00" would also be correct.
Example 2:

Input: nums = ["00","01"]
Output: "11"
Explanation: "11" does not appear in nums. "10" would also be correct.
Example 3:

Input: nums = ["111","011","001"]
Output: "101"
Explanation: "101" does not appear in nums. "000", "010", "100", and "110" would also be correct.
"""
print(Solution().findDifferentBinaryString(["0"]), "1")
print(Solution().findDifferentBinaryString(["01", "10"]), "11")
print(Solution().findDifferentBinaryString(["00", "01"]), "11")
print(Solution().findDifferentBinaryString(["111", "011", "001"]), "101")


# O(2^n), O(n)
# backtracking
class Solution:
    def findDifferentBinaryString(self, numbers: list[str]) -> str:
        permutation = []
        number_set = set(numbers)

        def dfs(index):
            if index == len(numbers):
                if "".join(permutation) in number_set:
                    return
                else:
                    return "".join(permutation)

            for number in "01":
                permutation.append(number)
                unique = dfs(index + 1)
                if unique:
                    return unique
                permutation.pop()

        return dfs(0)





# Check if There is a Valid Partition For The Array
# https://leetcode.com/problems/check-if-there-is-a-valid-partition-for-the-array/description/
"""
You are given a 0-indexed integer array nums. You have to partition the array into one or more contiguous subarrays.

We call a partition of the array valid if each of the obtained subarrays satisfies one of the following conditions:

The subarray consists of exactly 2, equal elements. For example, the subarray [2,2] is good.
The subarray consists of exactly 3, equal elements. For example, the subarray [4,4,4] is good.
The subarray consists of exactly 3 consecutive increasing elements, that is, the difference between adjacent elements is 1. For example, the subarray [3,4,5] is good, but the subarray [1,3,5] is not.
Return true if the array has at least one valid partition. Otherwise, return false.

 

Example 1:

Input: nums = [4,4,4,5,6]
Output: true
Explanation: The array can be partitioned into the subarrays [4,4] and [4,5,6].
This partition is valid, so we return true.
Example 2:

Input: nums = [1,1,1,2]
Output: false
Explanation: There is no valid partition for this array.
"""
print(Solution().validPartition([4, 4, 4, 5, 6]), True)
print(Solution().validPartition([1, 1, 1, 2]), False)
print(Solution().validPartition([993335, 993336, 993337, 993338, 993339, 993340, 993341]), False)
print(Solution().validPartition([803201, 803201, 803201, 803201, 803202, 803203]), True)
print(Solution().validPartition([67149,67149,67149,67149,67149,136768,136768,136768,136768,136768,136768,136768,136769,136770,136771,136772,136773,136774,136775,136776,136777,136778,136779,136780,136781,136782,136783,136784]), False)


"""
draft
                 .
             /        \
            44        444
          /   \      /    \
        .     456   .     .

[4, 4, 4, 5, 6]
[T, F, T, F, F, T]
[1, 1, 1, 2]
[F, F, F, F, T]
"""


class Solution:
    def validPartition(self, numbers: list[int]) -> bool:
        """
        O(n), O(n)
        dp, bottom-up
        """
        cache = [False] * (len(numbers) + 1)
        cache[-1] = True

        for index in reversed(range(len(numbers))):
            # two digit number 
            if index + 1 < len(numbers):
                if numbers[index] == numbers[index + 1]:
                    cache[index] = cache[index + 2]
                    if cache[index]:  # if partitions here, no need to check three digit number
                        continue  # early continue

            # three digit number
            if index + 2 < len(numbers):
                if (numbers[index] == numbers[index + 1] == numbers[index + 2] or
                        numbers[index] + 2 == numbers[index + 1] + 1 == numbers[index + 2]): # -
                    cache[index] = cache[index + 3]

        return cache[0]


class Solution:
    def validPartition(self, numbers: list[int]) -> bool:
        """
        O(n), O(1)
        dp, bottom-up
        """
        cache = [True, True, True]

        for index in reversed(range(len(numbers))):
            cache_0 = False
            
            # two digit number 
            if index + 1 < len(numbers):
                if numbers[index] == numbers[index + 1]:
                    cache_0 = cache[1]
                    continue

            # three digit number
            if index + 2 < len(numbers):
                if (numbers[index] == numbers[index + 1] == numbers[index + 2] or
                        numbers[index] + 2 == numbers[index + 1] + 1 == numbers[index + 2]): # -
                    cache_0 = cache[2]
            
            cache = [cache_0, cache[0], cache[1]]

        return cache[0]


class Solution:
    def validPartition(self, numbers: list[int]) -> bool:
        """
        O(n), O(n)
        dp, top-down with memoization as hash map
        """
        memo = {len(numbers): True}

        def dfs(index: int) -> bool:
            if index in memo:
                return memo[index]
            
            can_partition = False
    
            # two digit number
            if index + 1 < len(numbers):
                if numbers[index] == numbers[index + 1]:
                    can_partition = dfs(index + 2)
                    if can_partition:  # if partitions here, no need to check three digit number
                        memo[index] = True
                        return True  # early return

            # three digit number
            if index + 2 < len(numbers):
                if (numbers[index] == numbers[index + 1] == numbers[index + 2] or
                        numbers[index] + 2 == numbers[index + 1] + 1 == numbers[index + 2]): # -
                    can_partition = can_partition or dfs(index + 3)

            memo[index] = can_partition
            return memo[index]
        
        return dfs(0)


class Solution:
    def validPartition(self, numbers: list[int]) -> bool:
        """
        O(n), O(n)
        dp, top-down with memoization as list
        """
        memo = [None] * (len(numbers) + 1)
        memo[-1] = True

        def dfs(index: int) -> bool:
            if memo[index]:
                return memo[index]
            elif memo[index] == False:
                return False
            
            can_partition = False
            
            # two digit number
            if index + 1 < len(numbers):
                if numbers[index] == numbers[index + 1]:
                    can_partition = dfs(index + 2)
                    if can_partition:  # if partitions here, no need to check three digit number
                        memo[index] = True
                        return True  # early return

            # three digit number
            if index + 2 < len(numbers):
                if (numbers[index] == numbers[index + 1] == numbers[index + 2] or
                        numbers[index] + 2 == numbers[index + 1] + 1 == numbers[index + 2]): # -
                    can_partition = can_partition or dfs(index + 3)

            memo[index] = can_partition
            return memo[index]
        
        return dfs(0)


class Solution:
    def validPartition(self, numbers: list[int]) -> bool:
        """
        O(2^n), O(n)
        brute force, tle
        """
        def dfs(index: int) -> bool:
            if index >= len(numbers):
                return index == len(numbers)
            
            can_partition = False
    
            # two digit number
            if index + 1 < len(numbers):
                if numbers[index] == numbers[index + 1]:
                    can_partition = dfs(index + 2)
                    if can_partition:  # if partitions here, no need to check three digit number
                        return True  # early return

            # three digit number
            if index + 2 < len(numbers):
                if (numbers[index] == numbers[index + 1] == numbers[index + 2] or
                        numbers[index] + 2 == numbers[index + 1] + 1 == numbers[index + 2]): # -
                    can_partition = can_partition or dfs(index + 3)

            return can_partition
        
        return dfs(0)





# Maximum Subarray Min-Product
# https://leetcode.com/problems/maximum-subarray-min-product/description/
"""
The min-product of an array is equal to the minimum value in the array multiplied by the array's sum.

For example, the array [3,2,5] (minimum value is 2) has a min-product of 2 * (3+2+5) = 2 * 10 = 20.
Given an array of integers nums, return the maximum min-product of any non-empty subarray of nums. Since the answer may be large, return it modulo 109 + 7.

Note that the min-product should be maximized before performing the modulo operation. Testcases are generated such that the maximum min-product without modulo will fit in a 64-bit signed integer.

A subarray is a contiguous part of an array.

 

Example 1:

Input: nums = [1,2,3,2]
Output: 14
Explanation: The maximum min-product is achieved with the subarray [2,3,2] (minimum value is 2).
2 * (2+3+2) = 2 * 7 = 14.
Example 2:

Input: nums = [2,3,3,1,2]
Output: 18
Explanation: The maximum min-product is achieved with the subarray [3,3] (minimum value is 3).
3 * (3+3) = 3 * 6 = 18.
Example 3:

Input: nums = [3,1,5,6,4,2]
Output: 60
Explanation: The maximum min-product is achieved with the subarray [5,6,4] (minimum value is 4).
4 * (5+6+4) = 4 * 15 = 60.
"""
print(Solution().maxSumMinProduct([1]), 1)  # [1] * 1
print(Solution().maxSumMinProduct([1, 2]), 4)  # [2] * 2
print(Solution().maxSumMinProduct([1, 2, 3]), 10)  # [2, 3] * 2
print(Solution().maxSumMinProduct([1, 2, 3, 2]), 14)  # [2, 3, 2] * 2
print(Solution().maxSumMinProduct([2, 3, 3, 1, 2]), 18)  # [3, 3] * 3
print(Solution().maxSumMinProduct([3, 1, 5, 6, 4, 2]), 60)  # [5, 6, 4] * 4


class Solution:
    def maxSumMinProduct(self, numbers: list[int]) -> int:
        """
        O(n2), O(1)
        brute force, tle
        """
        mod = 10 ** 9 + 7
        max_min_prod = 0
        
        for left in range(len(numbers)):
            subarray_sum = 0
            min_number = numbers[left]

            for right in range(left, len(numbers)):
                subarray_sum += numbers[right]
                min_number = min(min_number, numbers[right])
                max_min_prod = max(max_min_prod, 
                                   subarray_sum * min_number % mod)

        return max_min_prod





# Integer Break
# https://leetcode.com/problems/integer-break/description/
"""
Given an integer n, break it into the sum of k positive integers, where k >= 2, and maximize the product of those integers.

Return the maximum product you can get.

 

Example 1:

Input: n = 2
Output: 1
Explanation: 2 = 1 + 1, 1 × 1 = 1.
Example 2:

Input: n = 10
Output: 36
Explanation: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36.
"""
print(Solution().integerBreak(2), 1)  # Explanation: 2 = 1 + 1, 1 × 1 = 1.
print(Solution().integerBreak(3), 2)  # Explanation: 3 = 1 + 2, 1 × 2 = 2.
print(Solution().integerBreak(4), 4)  # Explanation: 4 = 2 + 2, 2 × 2 = 4.
print(Solution().integerBreak(5), 6)  # Explanation: 5 = 2 + 3, 2 × 3 = 6.
print(Solution().integerBreak(6), 9)  # Explanation: 6 = 3 + 3, 3 × 3 = 9.
print(Solution().integerBreak(7), 12)  # Explanation: 7 = 3 + 4, 3 × 4 = 12; 2 = 2 + 2 + 3, 2 x 2 x 3 = 12.
print(Solution().integerBreak(10), 36)  # Explanation: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36.
print(Solution().integerBreak(24), 6561)  # tle testcase


"""
draft
                        4
                  1/            2|           3\  
                  3              2            1
               1/     2\        1/ \2        1|
              2       1        1     0        0
           1/  \2    1|       1|
           1    0     1        0
          1|
           0

"""


class Solution:
    def integerBreak(self, n: int) -> int:
        """
        O(n2), O(n)
        dp, top-down with memoization as hash map
        """
        memo = {0: 1}  # {number: product}  maximum product for current number

        def dfs(index: int, is_first: bool) -> int:
            if index in memo:
                return memo[index]
            
            memo[index] = max(number * dfs(index - number, False) 
                              for number 
                              in range(1, index + bool(not is_first)))  # bool is `1` excep of the first loop
            return memo[index]

        return dfs(n, True)


class Solution:
    def integerBreak(self, n: int) -> int:
        """
        O(n2), O(n)
        dp, top-down with memoization as list
        """
        memo = [None] * (n + 1)  # maximum product for current number (index)
        memo[0] = 1

        def dfs(index: int, is_first: bool) -> int:
            if memo[index] is not None:
                return memo[index]
            
            memo[index] = max(number * dfs(index - number, False) 
                              for number 
                              in range(1, index + bool(not is_first)))  # bool is `1` excep of the first loop
            return memo[index]

        return dfs(n, True)


class Solution:
    def integerBreak(self, n: int) -> int:
        """
        O(2^n), O(n)
        brute force, tle
        """
        def dfs(index: int, is_first: bool) -> int:
            if index == 0:
                return 1
            
            max_number = 0

            for number in range(1, index + bool(not is_first)):  # bool is `1` excep of the first loop
                max_number = max(max_number, 
                                 number * dfs(index - number, False))

            return max_number

        return dfs(n, True)


class Solution:
    def integerBreak(self, n: int) -> int:
        """
        O(2^n), O(n)
        brute force, backtracking, tle
        """
        integers = []
        max_product = 0

        def product():
            p = 1
            for number in integers:
                p *= number
            return p

        def dfs(n, max_product):
            if (n == 0 and
                    len(integers) > 1):
                max_product = max(max_product, product())
                return max_product

            for index in range(1, n + 1):
                integers.append(index)
                max_product = dfs(n - index, max_product)
                integers.pop()
            
            return max_product

        return dfs(n, max_product)


class Solution:
    def integerBreak(self, n: int) -> int:
        """
        O(n2), O(n)
        dp, bottom-up
        """
        cache = [None] * (n + 1)  # maximum product for current number (index)
        cache[0] = 1

        for index in range(2, n + 1):
            # if cache[index] is not None:
            #     return cache[index]
            
            cache[index] = max(number * cache[index - number, False]
                              for number 
                              in range(1, index + bool(not is_first)))  # bool is `1` excep of the first loop
            return cache[index]

        return cache





# Minimum Number of Operations to Make Array Continuous
# https://leetcode.com/problems/minimum-number-of-operations-to-make-array-continuous/description/
"""
You are given an integer array nums. In one operation, you can replace any element in nums with any integer.

nums is considered continuous if both of the following conditions are fulfilled:

All elements in nums are unique.
The difference between the maximum element and the minimum element in nums equals nums.length - 1.
For example, nums = [4, 2, 5, 3] is continuous, but nums = [1, 2, 3, 5, 6] is not continuous.

Return the minimum number of operations to make nums continuous.

 

Example 1:

Input: nums = [4,2,5,3]
Output: 0
Explanation: nums is already continuous.
Example 2:

Input: nums = [1,2,3,5,6]
Output: 1
Explanation: One possible solution is to change the last element to 4.
The resulting array is [1,2,3,5,4], which is continuous.
Example 3:

Input: nums = [1,10,100,1000]
Output: 3
Explanation: One possible solution is to:
- Change the second element to 2.
- Change the third element to 3.
- Change the fourth element to 4.
The resulting array is [1,2,3,4], which is continuous.
"""
print(Solution().minOperations([4, 2, 5, 3]), 0)
print(Solution().minOperations([1, 2, 3, 5, 6]), 1)
print(Solution().minOperations([1, 10, 100, 1000]), 3)


# O(n2), O(n)
# tle
class Solution:
    def minOperations(self, numbers: list[int]) -> int:
        number_set = set(numbers)
        numbers.sort()
        min_difference = len(numbers)


        for number in number_set:  # check every unique number
            difference = 0  # number of dffferences for current number
            
            for j in range(1, len(numbers)):  # check every distance starting from `1`
                difference += number + j not in number_set  # increase difference if no distance from current number found
            
            min_difference = min(min_difference, difference)  # update max distance
        
        return min_difference


# O(nlog), O(n)
# sliding window
class Solution:
    def minOperations(self, numbers: list[int]) -> int:
        unique_numbers = sorted(set(numbers))
        right = 0
        min_difference = len(numbers)

        for left, number in enumerate(unique_numbers):

            while (right < len(numbers) and
                   numbers[right] < number + len(numbers)):
                right += 1

            window_length = right - left
            difference = len(numbers) - window_length
            min_difference = min(min_difference, difference)

        return min_difference





# Maximum Frequency Stack
# https://leetcode.com/problems/maximum-frequency-stack/description/
"""
Design a stack-like data structure to push elements to the stack and pop the most frequent element from the stack.

Implement the FreqStack class:

FreqStack() constructs an empty frequency stack.
void push(int val) pushes an integer val onto the top of the stack.
int pop() removes and returns the most frequent element in the stack.
If there is a tie for the most frequent element, the element closest to the stack's top is removed and returned.
 

Example 1:

Input
["FreqStack", "push", "push", "push", "push", "push", "push", "pop", "pop", "pop", "pop"]
[[], [5], [7], [5], [7], [4], [5], [], [], [], []]
Output
[null, null, null, null, null, null, null, 5, 7, 5, 4]

Explanation
FreqStack freqStack = new FreqStack();
freqStack.push(5); // The stack is [5]
freqStack.push(7); // The stack is [5,7]
freqStack.push(5); // The stack is [5,7,5]
freqStack.push(7); // The stack is [5,7,5,7]
freqStack.push(4); // The stack is [5,7,5,7,4]
freqStack.push(5); // The stack is [5,7,5,7,4,5]
freqStack.pop();   // return 5, as 5 is the most frequent. The stack becomes [5,7,5,7,4].
freqStack.pop();   // return 7, as 5 and 7 is the most frequent, but 7 is closest to the top. The stack becomes [5,7,5,4].
freqStack.pop();   // return 5, as 5 is the most frequent. The stack becomes [5,7,4].
freqStack.pop();   // return 4, as 4, 5 and 7 is the most frequent, but 4 is closest to the top. The stack becomes [5,7].
"""


# Your FreqStack object will be instantiated and called as such:
# obj = FreqStack()
# obj.push(val)
# param_2 = obj.pop()


# O(1): push O(n): pop; aux space O(n)
# brute force
class FreqStack:
    def __init__(self):
        self.stack = []
        self.counter = {}

    def push(self, val: int) -> None:
        self.stack.append(val)
        self.counter[val] = self.counter.get(val, 0) + 1

    def pop(self) -> int:
        max_frequency = max(self.counter.values())
        reversed_stack = []

        while (self.stack and
               self.counter[self.stack[-1]] != max_frequency):
            reversed_stack.append(self.stack.pop())

        most_frequent = self.stack.pop()
        self.counter[most_frequent] -= 1

        while reversed_stack:
            self.stack.append(reversed_stack.pop())

        return most_frequent


import heapq

# O(logn), O(n)
# heap
class FreqStack:
    def __init__(self):
        self.stack = []  # [((-frequency, -index), value), ]
        self.counter = {}
        self.index = 0

    def push(self, val: int) -> None:
        self.counter[val] = self.counter.get(val, 0) + 1
        heapq.heappush(self.stack, ((-self.counter[val], -self.index), val))
        self.index += 1

    def pop(self) -> int:
        val = heapq.heappop(self.stack)[1]
        self.counter[val] -= 1
        return val


# O(n), O(n)
# 2 hash maps
class FreqStack:
    def __init__(self):
        self.counter = {}  # {value: frequency, }
        self.bucket = {}  # {frequency: [value1, value2, ], }
        self.max_frequency = 0

    def push(self, val: int) -> None:
        self.counter[val] = self.counter.get(val, 0) + 1
        self.max_frequency = max(self.max_frequency, self.counter[val])
        if self.max_frequency not in self.bucket:
            self.bucket[self.max_frequency] = []
        self.bucket[self.counter[val]].append(val)

    def pop(self) -> int:
        val = self.bucket[self.max_frequency].pop()
        self.counter[val] -= 1
        if not self.bucket[self.max_frequency]:
            self.max_frequency -= 1
        return val


from collections import deque

# O(n), O(n)
# 2 hash maps, deque
class FreqStack:
    def __init__(self):
        self.counter = {}  # {value: frequency, }
        self.bucket = {}  # {frequency: deque(value1, value2, ), }
        self.max_frequency = 0

    def push(self, val: int) -> None:
        self.counter[val] = self.counter.get(val, 0) + 1
        self.max_frequency = max(self.max_frequency, self.counter[val])
        if self.max_frequency not in self.bucket:
            self.bucket[self.max_frequency] = deque()
        self.bucket[self.counter[val]].append(val)

    def pop(self) -> int:
        val = self.bucket[self.max_frequency].pop()
        self.counter[val] -= 1
        if not self.bucket[self.max_frequency]:
            self.max_frequency -= 1
        return val


from collections import defaultdict

# O(n), O(n)
# 2 default hash maps
class FreqStack:
    def __init__(self):
        self.counter = defaultdict(int)  # {value: frequency, }
        self.bucket = defaultdict(list)  # {frequency: [value1, value2, ], }
        self.max_frequency = 0

    def push(self, val: int) -> None:
        self.counter[val] += 1
        self.max_frequency = max(self.max_frequency, self.counter[val])
        self.bucket[self.counter[val]].append(val)

    def pop(self) -> int:
        val = self.bucket[self.max_frequency].pop()
        self.counter[val] -= 1
        if not self.bucket[self.max_frequency]:
            self.max_frequency -= 1
        return val


from collections import defaultdict, deque

# O(n), O(n)
# 2 default hash maps, deque
class FreqStack:
    def __init__(self):
        self.counter = defaultdict(int)  # {value: frequency, }
        self.bucket = defaultdict(deque)  # {frequency: deque(value1, value2, ), }
        self.max_frequency = 0

    def push(self, val: int) -> None:
        self.counter[val] += 1
        self.max_frequency = max(self.max_frequency, self.counter[val])
        self.bucket[self.counter[val]].append(val)

    def pop(self) -> int:
        val = self.bucket[self.max_frequency].pop()
        self.counter[val] -= 1
        if not self.bucket[self.max_frequency]:
            self.max_frequency -= 1
        return val


freqStack = FreqStack()
print(freqStack.push(5))  # The stack is [5]
print(freqStack.push(7))  # The stack is [5,7]
print(freqStack.push(5))  # The stack is [5,7,5]
print(freqStack.push(7))  # The stack is [5,7,5,7]
print(freqStack.push(4))  # The stack is [5,7,5,7,4]
print(freqStack.push(5))  # The stack is [5,7,5,7,4,5]
print(freqStack.pop())  # return 5, as 5 is the most frequent. The stack becomes [5,7,5,7,4].
print(freqStack.pop())  # return 7, as 5 and 7 is the most frequent, but 7 is closest to the top. The stack becomes [5,7,5,4].
print(freqStack.pop())  # return 5, as 5 is the most frequent. The stack becomes [5,7,4].
print(freqStack.pop())  # return 4, as 4, 5 and 7 is the most frequent, but 4 is closest to the top. The stack becomes [5,7].


def test_input(operations: list[str], arguments: list[list[int | None]]) -> list[list[int] | None]:
    result = []

    for operation, argument in zip(operations, arguments):
        if operation == "FreqStack":
            freqStack = FreqStack()
            result.append(None)
        elif operation == "push":
            freqStack.push(*argument)
            result.append(None)
        elif operation == "pop":
            result.append(freqStack.pop())
    
    return result

# Example Input
operations = ["FreqStack", "push", "push", "push", "push", "push", "push", "pop", "pop", "pop", "pop"]
arguments = [[], [5], [7], [5], [7], [4], [5], [], [], [], []]
expected_output = [None, None, None, None, None, None, None, 5, 7, 5, 4]

operations = ["FreqStack","push","push","push","push","pop", "pop", "push", "push", "push", "pop", "pop", "pop"]
arguments = [[],[1], [1], [1], [2], [], [], [2], [2], [1], [], [], []]
expected_output = [None, None, None, None, None, 1, 1, None, None, None, 2, 1, 2]

# Run tests
test_output = test_input(operations, arguments)
print(test_output == expected_output)
print(test_output)





# Minimum Cost For Tickets
# https://leetcode.com/problems/minimum-cost-for-tickets/description/
"""
You have planned some train traveling one year in advance. The days of the year in which you will travel are given as an integer array days. Each day is an integer from 1 to 365.

Train tickets are sold in three different ways:

a 1-day pass is sold for costs[0] dollars,
a 7-day pass is sold for costs[1] dollars, and
a 30-day pass is sold for costs[2] dollars.
The passes allow that many days of consecutive travel.

For example, if we get a 7-day pass on day 2, then we can travel for 7 days: 2, 3, 4, 5, 6, 7, and 8.
Return the minimum number of dollars you need to travel every day in the given list of days.

 

Example 1:

Input: days = [1,4,6,7,8,20], costs = [2,7,15]
Output: 11
Explanation: For example, here is one way to buy passes that lets you travel your travel plan:
On day 1, you bought a 1-day pass for costs[0] = $2, which covered day 1.
On day 3, you bought a 7-day pass for costs[1] = $7, which covered days 3, 4, ..., 9.
On day 20, you bought a 1-day pass for costs[0] = $2, which covered day 20.
In total, you spent $11 and covered all the days of your travel.
Example 2:

Input: days = [1,2,3,4,5,6,7,8,9,10,30,31], costs = [2,7,15]
Output: 17
Explanation: For example, here is one way to buy passes that lets you travel your travel plan:
On day 1, you bought a 30-day pass for costs[2] = $15 which covered days 1, 2, ..., 30.
On day 31, you bought a 1-day pass for costs[0] = $2 which covered day 31.
In total, you spent $17 and covered all the days of your travel.
"""
print(Solution().mincostTickets([5], [2, 7, 15]), 2)
print(Solution().mincostTickets([1, 4, 6, 7, 8, 20], [2, 7, 15]), 11)
print(Solution().mincostTickets([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 31], [2, 7, 15]), 17)
print(Solution().mincostTickets([1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,24,25,27,28,29,30,31,34,37,38,39,41,43,44,45,47,48,49,54,57,60,62,63,66,69,70,72,74,76,78,80,81,82,83,84,85,88,89,91,93,94,97,99], [9,38,134]), 423)


"""
draft
                                               1
                      (1,2)/               (7,7)|                 (30,15)\
                         2                     8                         31
          /             |   \               /    |     \
         3              9    32            9    15     39
    /    | \        /   | \
   4    10  33     10   16  39

[1,  4, 6, 7, 8, 20]
[12, 10, 8, 6, 4, 2, 0]
[                   ,  0]
"""


class Solution:
    def mincostTickets(self, days: list[int], costs: list[int]) -> int:
        """
        O(3^n), O(n)
        brute force, tle
        """
        def dfs(day_index: int) -> int:
            if day_index >= len(days):
                return 0
            
            min_cost = float("inf")

            for cost, validity in zip(costs, (1, 7, 30)):
                ticket_index = 0
                
                while (day_index + ticket_index < len(days) and
                       days[day_index + ticket_index] < days[day_index] + validity):
                    ticket_index += 1
                
                min_cost = min(min_cost, cost + dfs(day_index + ticket_index))

            return min_cost

        return dfs(0)


class Solution:
    def mincostTickets(self, days: list[int], costs: list[int]) -> int:
        """
        O(n), O(n)
        dp, top-down with memoization as hash map
        """
        memo = {}  # {day_index: cost} minimum cost to travel from day index pointing day onwards

        def dfs(day_index: int) -> int:
            if day_index >= len(days):
                return 0
            elif day_index in memo:
                return memo[day_index]
            
            min_cost = float("inf")

            for cost, validity in zip(costs, (1, 7, 30)):
                ticket_index = 0
                
                while (day_index + ticket_index < len(days) and
                       days[day_index + ticket_index] < days[day_index] + validity):
                    ticket_index += 1
                
                min_cost = min(min_cost, cost + dfs(day_index + ticket_index))

            memo[day_index] = min_cost
            return min_cost

        return dfs(0)


class Solution:
    def mincostTickets(self, days: list[int], costs: list[int]) -> int:
        """
        O(n), O(n)
        dp, top-down with memoization as list
        """
        memo = [None] * len(days)  # {day_index: cost} minimum cost to travel from day index pointing day onwards

        def dfs(day_index: int) -> int:
            if day_index >= len(days):
                return 0
            elif memo[day_index] is not None:
                return memo[day_index]
            
            min_cost = float("inf")

            for cost, validity in zip(costs, (1, 7, 30)):
                ticket_index = 0
                
                while (day_index + ticket_index < len(days) and
                       days[day_index + ticket_index] < days[day_index] + validity):
                    ticket_index += 1
                
                min_cost = min(min_cost, cost + dfs(day_index + ticket_index))

            memo[day_index] = min_cost
            return min_cost

        return dfs(0)


class Solution:
    def mincostTickets(self, days: list[int], costs: list[int]) -> int:
        """
        O(n), O(n)
        dp, bottom-up
        """
        cache = [float("inf")] * (len(days) + 0)  # {day_index: cost} minimum cost to travel from day index pointing day onwards
        # cache = 

        for day_index, day in enumerate(days):
            for cost, validity in zip(costs, (1, 7, 30)):
                ticket_index = 0
                
                while (day_index + ticket_index < len(days) and
                       days[day_index + ticket_index] < days[day_index] + validity):
                    ticket_index += 1
                
                cache[day_index] = min(cache[day_index], 
                                       cost + cache[day_index + ticket_index])

            
            # if day_index >= len(days):
            #     return 0
            # elif cache[day_index] is not None:
            #     return cache[day_index]
            

        return cache


class Solution:
    def mincostTickets(self, days: list[int], costs: list[int]) -> int:
        self.max_cost = float("inf")
        day_set = set(days)
        cost_list = []

        def dfs(day):
            while (day not in day_set and 
                    day < max(day_set)):
                day += 1
            
            if day > max(day_set):
                self.max_cost = min(self.max_cost, sum(cost_list))
                return

            # one day ticket
            cost_list.append(costs[0])
            dfs(day + 1)
            cost_list.pop()

            # seven days ticket
            cost_list.append(costs[1])
            dfs(day + 7)
            cost_list.pop()

            # 30 days ticket
            cost_list.append(costs[2])
            dfs(day + 30)
            cost_list.pop()

        dfs(days[0])
        return self.max_cost





# Number of Longest Increasing Subsequence
# https://leetcode.com/problems/number-of-longest-increasing-subsequence/description/
"""
Given an integer array nums, return the number of longest increasing subsequences.

Notice that the sequence has to be strictly increasing.

 

Example 1:

Input: nums = [1,3,5,4,7]
Output: 2
Explanation: The two longest increasing subsequences are [1, 3, 4, 7] and [1, 3, 5, 7].
Example 2:

Input: nums = [2,2,2,2,2]
Output: 5
Explanation: The length of the longest increasing subsequence is 1, and there are 5 increasing subsequences of length 1, so output 5.
"""
print(Solution().findNumberOfLIS([1, 3, 5, 4]), 2)  # [1, 3, 4] and [1, 3, 5]
print(Solution().findNumberOfLIS([1, 3, 5, 4, 7]), 2)  # [1, 3, 4, 7] and [1, 3, 5, 7]
print(Solution().findNumberOfLIS([2, 2, 2, 2, 2]), 5)  # [2] * 5
print(Solution().findNumberOfLIS([1, 2, 4, 3, 5, 4, 7, 2]), 3)  # [1, 2, 3, 4, 7], [1, 2, 3, 5, 7], [1, 2, 4, 5, 7]
print(Solution().findNumberOfLIS([1, 1, 1, 2, 2, 2, 3, 3, 3]), 27)


"""
draft
[1, 3, 5, 4, 7]
[1, 1]      [[LIS lengths, frequency]]
   [2, 1]
      [3, 1]
         [3, 1]
            [4, 2]
"""


class Solution:
    def findNumberOfLIS(self, numbers: list[int]) -> int:
        """
        O(n2), O(n)
        dp, bottom-up
        """
        cache = [[1, 1] for _ in range(len(numbers))]  # [[LIS lengths, frequency]]

        for right in range(len(numbers)):
            for left in range(right):
                if numbers[left] < numbers[right]:  # if right number is greater
                    if cache[left][0] + 1 > cache[right][0]:  # longer LIS
                        cache[right] = [cache[left][0] + 1,  # update LIS length
                                        cache[left][1]]  # carry frequencies to new LIS
                    elif cache[left][0] + 1 == cache[right][0]:  # same length LIS                   
                        cache[right] = [cache[right][0], 
                                        cache[right][1] + cache[left][1]]  # update LIS frequency

        # Travrse through cache and gather max frequencies.
        max_lis_length = 0
        for lis_length, frequency in cache:
            if lis_length > max_lis_length:
                max_lis_length = lis_length
                max_frequency = frequency
            elif lis_length == max_lis_length:
               max_frequency += frequency

        return max_frequency





# Uncrossed Lines
# https://leetcode.com/problems/uncrossed-lines/description/
"""
You are given two integer arrays nums1 and nums2. We write the integers of nums1 and nums2 (in the order they are given) on two separate horizontal lines.

We may draw connecting lines: a straight line connecting two numbers nums1[i] and nums2[j] such that:

nums1[i] == nums2[j], and
the line we draw does not intersect any other connecting (non-horizontal) line.
Note that a connecting line cannot intersect even at the endpoints (i.e., each number can only belong to one connecting line).

Return the maximum number of connecting lines we can draw in this way.

 

Example 1:


Input: nums1 = [1,4,2], nums2 = [1,2,4]
Output: 2
Explanation: We can draw 2 uncrossed lines as in the diagram.
We cannot draw 3 uncrossed lines, because the line from nums1[1] = 4 to nums2[2] = 4 will intersect the line from nums1[2]=2 to nums2[1]=2.
Example 2:

Input: nums1 = [2,5,1,2,5], nums2 = [10,5,2,1,5,2]
Output: 3
Example 3:

Input: nums1 = [1,3,7,1,7,5], nums2 = [1,9,2,5,1]
Output: 2
"""
print(Solution().maxUncrossedLines([1, 4, 2], [1, 2, 4]), 2)
print(Solution().maxUncrossedLines([2, 5, 1, 2, 5], [10, 5, 2, 1, 5, 2]), 3)
print(Solution().maxUncrossedLines([1, 3, 7, 1, 7, 5], [1, 9, 2, 5, 1]), 2)
print(Solution().maxUncrossedLines([4,1,2,5,1,5,3,4,1,5], [3,1,1,3,2,5,2,4,1,3,2,2,5,5,3,5,5,1,2,1]), 7)
print(Solution().maxUncrossedLines([5,1,2,5,1,2,2,3,1,1,1,1,1,3,1], [2,5,1,3,4,5,5,2,2,4,5,2,2,3,1,4,5,3,2,4,5,2,4,4,2,2,2,1,3,1]), 11)


"""
draft
[1, 4, 2]
 |    \\
[1, 2, 4]
[1, 2], [1, 4]

[2,  5, 1, 2, 5]
     |    \   |
[10, 5, 2, 1, 5, 2]
[2, 5, 2], [5, 1, 5], [5, 1, 2]
"""


class Solution:
    def maxUncrossedLines(self, numbers_1: list[int], numbers_2: list[int]) -> int:
        """
        O(n2), O(n2)
        dp, bottom-up
        reversed iteration
        """
        cache = [[0] * (len(numbers_2) + 1)
                for _ in range((len(numbers_1) + 1))]  # [(index_1, index_2): number of connections]

        for index_1 in reversed(range(len(numbers_1))):
            for index_2 in reversed(range(len(numbers_2))):
                if numbers_1[index_1] == numbers_2[index_2]:
                    cache[index_1][index_2] = cache[index_1 + 1][index_2 + 1] + 1
                else:
                    cache[index_1][index_2] = max(cache[index_1 + 1][index_2], 
                                                  cache[index_1][index_2 + 1])
                
        return cache[0][0]


class Solution:
    def maxUncrossedLines(self, numbers_1: list[int], numbers_2: list[int]) -> int:
        """
        O(n2), O(n2)
        dp, bottom-up
        """
        cache = [[0] * (len(numbers_2) + 1)
                for _ in range((len(numbers_1) + 1))]  # [(index_1, index_2): number of connections]

        for index_1 in range(len(numbers_1)):
            for index_2 in range(len(numbers_2)):
                if numbers_1[index_1] == numbers_2[index_2]:
                    cache[index_1 + 1][index_2 + 1] = cache[index_1][index_2] + 1
                else:
                    cache[index_1 + 1][index_2 + 1] = max(cache[index_1 + 1][index_2], 
                                                          cache[index_1][index_2 + 1])
        
        return cache[len(numbers_1)][len(numbers_2)]


class Solution:
    def maxUncrossedLines(self, numbers_1: list[int], numbers_2: list[int]) -> int:
        """
        O(n2), O(n2)
        dp, top-down with memoization as hash map
        """
        cache = {}  # {(index_1, index_2): number of connections}

        def dfs(index_1, index_2):
            if (index_1 == len(numbers_1) or
                    index_2 == len(numbers_2)):
                return 0
            elif (index_1, index_2) in cache:
                return cache[(index_1, index_2)]

            if numbers_1[index_1] == numbers_2[index_2]:
                cache[(index_1, index_2)] = 1 + dfs(index_1 + 1, index_2 + 1)
            else:
                cache[(index_1, index_2)] = max(dfs(index_1 + 1, index_2), 
                                               dfs(index_1, index_2 + 1))
            
            return cache[(index_1, index_2)]

        return dfs(0, 0)


class Solution:
    def maxUncrossedLines(self, numbers_1: list[int], numbers_2: list[int]) -> int:
        """
        O(n2), O(n2)
        dp, top-down with memoization as list
        """
        memo = [[-1] * len(numbers_2) 
                for _ in range(len(numbers_1))]  # [(index_1, index_2): number of connections]

        def dfs(index_1, index_2):
            if (index_1 == len(numbers_1) or
                    index_2 == len(numbers_2)):
                return 0
            elif memo[index_1][index_2] != -1:
                return memo[index_1][index_2]

            if numbers_1[index_1] == numbers_2[index_2]:
                memo[index_1][index_2] = 1 + dfs(index_1 + 1, index_2 + 1)
            else:
                memo[index_1][index_2] = max(dfs(index_1 + 1, index_2), 
                                               dfs(index_1, index_2 + 1))
            
            return memo[index_1][index_2]

        return dfs(0, 0)


class Solution:
    def maxUncrossedLines(self, numbers_1: list[int], numbers_2: list[int]) -> int:
        """
        O(2^n), O(n)
        brute force, tle
        """
        def dfs(index_1, index_2):
            if (index_1 == len(numbers_1) or
                    index_2 == len(numbers_2)):
                return 0

            if numbers_1[index_1] == numbers_2[index_2]:
                return 1 + dfs(index_1 + 1, index_2 + 1)
            else:
                return max(dfs(index_1 + 1, index_2), 
                           dfs(index_1, index_2 + 1))

        return dfs(0, 0)




# Solving Questions With Brainpower
# https://leetcode.com/problems/solving-questions-with-brainpower/description/
"""
You are given a 0-indexed 2D integer array questions where questions[i] = [pointsi, brainpoweri].

The array describes the questions of an exam, where you have to process the questions in order (i.e., starting from question 0) and make a decision whether to solve or skip each question. Solving question i will earn you pointsi points but you will be unable to solve each of the next brainpoweri questions. If you skip question i, you get to make the decision on the next question.

For example, given questions = [[3, 2], [4, 3], [4, 4], [2, 5]]:
If question 0 is solved, you will earn 3 points but you will be unable to solve questions 1 and 2.
If instead, question 0 is skipped and question 1 is solved, you will earn 4 points but you will be unable to solve questions 2 and 3.
Return the maximum points you can earn for the exam.

 

Example 1:

Input: questions = [[3,2],[4,3],[4,4],[2,5]]
Output: 5
Explanation: The maximum points can be earned by solving questions 0 and 3.
- Solve question 0: Earn 3 points, will be unable to solve the next 2 questions
- Unable to solve questions 1 and 2
- Solve question 3: Earn 2 points
Total points earned: 3 + 2 = 5. There is no other way to earn 5 or more points.
Example 2:

Input: questions = [[1,1],[2,2],[3,3],[4,4],[5,5]]
Output: 7
Explanation: The maximum points can be earned by solving questions 1 and 4.
- Skip question 0
- Solve question 1: Earn 2 points, will be unable to solve the next 2 questions
- Unable to solve questions 2 and 3
- Solve question 4: Earn 5 points
Total points earned: 2 + 5 = 7. There is no other way to earn 7 or more points.
"""
print(Solution().mostPoints([[3, 2], [4, 3], [4, 4], [2, 5]]), 5)
print(Solution().mostPoints([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]), 7)
print(Solution().mostPoints([[72,5],[36,5],[95,5],[50,1],[62,1],[14,3],[72,5],[86,2],[43,3],[51,3],[14,1],[91,5],[47,4],[72,4],[63,5],[40,3],[68,1],[8,3],[84,5],[7,5],[40,1],[35,3],[66,2],[39,5],[40,1],[92,3],[67,5],[34,3],[84,4],[75,5],[6,1],[71,3],[77,3],[25,3],[53,3],[32,3],[88,5],[18,2],[21,3],[78,2],[69,5],[45,4],[94,2],[70,1],[85,2],[7,2],[68,4],[71,4],[57,2],[12,5],[53,5],[51,3],[46,1],[28,3]]), 845)


class Solution:
    def mostPoints(self, questions: list[list[int]]) -> int:
        """
        O(n), O(n)
        dp, bottom-up with tabulation as hash map
        """
        cache = {}  # {question: maximum points cumulated}
    
        for index in range(len(questions) -1, -1, -1):
            cache[index] = max(
                questions[index][0] + cache.get(index + 1 + questions[index][1], 0),  # solve question
                cache.get(index + 1, 0)  # skip question
            )

        return cache.get(0)


class Solution:
    def mostPoints(self, questions: list[list[int]]) -> int:
        """
        O(n), O(n)
        dp, top-down with memoization as hash map
        """
        memo = {len(questions): 0}  # {question: maximum points cumulated}

        def dfs(index):
            if index >= len(questions):
                return 0
            elif index in memo:
                return memo[index]
            

            memo[index] = max(
                questions[index][0] + dfs(index + 1 + questions[index][1]),  # solve question
                dfs(index + 1)  # skip question
            )
            return memo[index]

        return dfs(0)


class Solution:
    def mostPoints(self, questions: list[list[int]]) -> int:
        """
        O(n), O(n)
        dp, top-down with memoization as list
        """
        memo = [-1] * (len(questions) + 1)  # [question: maximum points cumulated]
        memo[len(questions)]

        def dfs(index):
            if index >= len(questions):
                return 0
            elif memo[index] != -1:
                return memo[index]
            

            memo[index] = max(
                questions[index][0] + dfs(index + 1 + questions[index][1]),  # solve question
                dfs(index + 1)  # skip question
            )
            return memo[index]
        
        return dfs(0)


class Solution:
    def mostPoints(self, questions: list[list[int]]) -> int:
        """
        O(2^n), O(n)
        brute force, tle
        """

        def dfs(index):
            if index >= len(questions):
                return 0
            
            return max(
                questions[index][0] + dfs(index + 1 + questions[index][1]),  # solve question
                dfs(index + 1)  # skip question
            )

        return dfs(0)





# Count Ways To Build Good Strings
# https://leetcode.com/problems/count-ways-to-build-good-strings/description/
"""
Given the integers zero, one, low, and high, we can construct a string by starting with an empty string, and then at each step perform either of the following:

Append the character '0' zero times.
Append the character '1' one times.
This can be performed any number of times.

A good string is a string constructed by the above process having a length between low and high (inclusive).

Return the number of different good strings that can be constructed satisfying these properties. Since the answer can be large, return it modulo 109 + 7.

 

Example 1:

Input: low = 3, high = 3, zero = 1, one = 1
Output: 8
Explanation: 
One possible valid good string is "011". 
It can be constructed as follows: "" -> "0" -> "01" -> "011". 
All binary strings from "000" to "111" are good strings in this example.
Example 2:

Input: low = 2, high = 3, zero = 1, one = 2
Output: 5
Explanation: The good strings are "00", "11", "000", "110", and "011".
"""
print(Solution().countGoodStrings(1, 1, 1, 1), 2)
print(Solution().countGoodStrings(1, 2, 1, 1), 6)
print(Solution().countGoodStrings(2, 2, 1, 1), 4)
print(Solution().countGoodStrings(1, 3, 1, 1), 14)
print(Solution().countGoodStrings(3, 3, 1, 1), 8)
print(Solution().countGoodStrings(2, 3, 1, 2), 5)
print(Solution().countGoodStrings(200, 200, 10, 1), 764262396)  # tle
print(Solution().countGoodStrings(1, 100000, 1, 1), 215447031)  # RecursionError: maximum recursion depth exceeded


"""
draft
[1, 1, 1, 1]
"0", "1"

[1, 2, 1, 1]
"0", "1", "00", "01", "10", "11"

[2, 2, 1, 1]
"00", "01", "10", "11"
2**2 = 4

[3, 3, 1, 1]
2**3 + 2**2 + 2**1 = 8 + 4 + 2 = 14

[2, 3, 1, 2]
"00", "11", "000", "110", and "011"
"""


class Solution:
    def countGoodStrings(self, low: int, high: int, zero_times: int, one_times: int) -> int:
        """
        O(n), O(n)
        dp, bottom-up
        """
        mod = 10 ** 9 + 7
        cache = {}  # memo[string length]: store the number of ways to make a good string of length index
        cache[0] = 1  # one way to create zero length (empty) string

        for index in range(1, high + 1):
            cache[index] = (cache.get(index - zero_times, 0) + 
                            cache.get(index - one_times, 0)) % mod
        
        return sum(cache[index]
                   for index in range(low, high + 1)) % mod


class Solution:
    def countGoodStrings(self, low: int, high: int, zero_times: int, one_times: int) -> int:
        """
        O(n), O(n)
        dp, top-down with memoization as hash map
        """
        mod = 10 ** 9 + 7
        memo = {}  # represents the total number of valid good strings that can be constructed starting from length index and going up to high

        def dfs(index: int) -> int:
            if index > high:
                return 0
            elif index in memo:
                return memo[index]
            
            current_node_count = 1 if index >= low else 0
            zero_branch = dfs(index + zero_times)
            one_branch = dfs(index + one_times)
            
            memo[index] = (current_node_count + zero_branch + one_branch) % mod
            return memo[index]

        return dfs(0)


class Solution:
    def countGoodStrings(self, low: int, high: int, zero_times: int, one_times: int) -> int:
        """
        O(2^n), O(n)
        brute force, tle
        """
        mod = 10 ** 9 + 7

        def dfs(index: int) -> int:
            if index > high:
                return 0
            
            current_node_count = 1 if index >= low else 0
            zero_branch = dfs(index + zero_times)
            one_branch = dfs(index + one_times)
            
            return (current_node_count + zero_branch + one_branch) % mod

        return dfs(0)





# New 21 Game
# https://leetcode.com/problems/new-21-game/description/
"""
Alice plays the following game, loosely based on the card game "21".

Alice starts with 0 points and draws numbers while she has less than k points. During each draw, she gains an integer number of points randomly from the range [1, maxPts], where maxPts is an integer. Each draw is independent and the outcomes have equal probabilities.

Alice stops drawing numbers when she gets k or more points.

Return the probability that Alice has n or fewer points.

Answers within 10-5 of the actual answer are considered accepted.

 

Example 1:

Input: n = 10, k = 1, maxPts = 10
Output: 1.00000
Explanation: Alice gets a single card, then stops.
Example 2:

Input: n = 6, k = 1, maxPts = 10
Output: 0.60000
Explanation: Alice gets a single card, then stops.
In 6 out of 10 possibilities, she is at or below 6 points.
Example 3:

Input: n = 21, k = 17, maxPts = 10
Output: 0.73278
"""
print(Solution().new21Game(10, 1, 10), 1)
print(Solution().new21Game(3, 2, 3), 0.88889)
print(Solution().new21Game(6, 1, 10), 0.6)
print(Solution().new21Game(21, 17, 10), 0.73278)
print(Solution().new21Game(421, 400, 47), 0.71188)  # tle
print(Solution().new21Game(9811, 8776, 1096), 0.99670)  # tle


class Solution:
    def new21Game(self, n: int, k: int, maxPts: int) -> float:
        """
        O(2^n), O(n)
        brute force
        tle, O(maxPts^k)
        """
        def dfs(score): 
            if score >= k:
                return score <= n 

            return sum(
                dfs(score + points) 
                for points in range(1, maxPts + 1)
            ) / maxPts
        
        return dfs(0)



class Solution:
    def new21Game(self, n: int, k: int, maxPts: int) -> float:
        """
        O(n2), O(n)
        dp, top-down with memoization as hash map
        O(k*maxPts), tle
        """
        memo = {}  # the probability that Alice has n or fewer score after drawing a card if she had k or less score

        def dfs(score): 
            if score >= k:
                return score <= n
            elif score in memo:
                return memo[score]

            valid_count = 0  # is the number of valid outcomes (paths) from the current state.
            for points in range(1, maxPts + 1):
                valid_count += (dfs(score + points)) 

            memo[score] = valid_count / maxPts  #  is the probability of reaching a valid score from the current state.
            return memo[score]
        
        return dfs(0)


class Solution:
    def new21Game(self, n: int, k: int, maxPts: int) -> float:
        if k == 0:
            return 1.0

        windowSum = sum(
            i <= n 
            for i in range(k, k + maxPts))
        
        dp = {}
        for i in range(k - 1, -1, -1):
            dp[i] = windowSum / maxPts
            remove = 0
            if i + maxPts <= n:
                remove = dp.get(i + maxPts, 1)
            
            windowSum += dp[i] - remove
        
        return dp[0]




# 