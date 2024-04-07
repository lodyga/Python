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



